"""Opt-in Telegram channel selection and completion delivery for native roots.

Installed into an existing NativeConversationIngress, never a second runner.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from gateway.config import Platform
from gateway.platforms.base import SendResult


def channel_key(source):
    identity = [
        source.platform.value,
        source.chat_id,
        source.thread_id,
        source.user_id,
        source.scope_id,
        source.profile,
    ]
    return hashlib.sha256(
        json.dumps(identity, separators=(",", ":")).encode()
    ).hexdigest()


class SilentProgressAdapter:
    """Keep native queue/control state, suppress PWA-origin outbound traffic."""

    def __init__(self, adapter):
        self.adapter = adapter

    def __getattr__(self, name):
        value = getattr(self.adapter, name)
        if name.startswith("send") or name in {
            "edit_message",
            "_send_with_retry",
            "_keep_typing",
        }:

            async def suppressed(*args, **kwargs):
                return SendResult(
                    success=False, error="PWA progress is not mirrored", retryable=False
                )

            return suppressed
        return value


class _FinalAttemptAdapter:
    def __init__(self, adapter, eligible):
        self.adapter, self.eligible = adapter, eligible
        self.started = self.unknown = self.skipped = False

    def __getattr__(self, name):
        value = getattr(self.adapter, name)
        if not name.startswith("send"):
            return value

        async def dispatch(*args, **kwargs):
            # Every text/media dispatch rechecks current selection, including
            # attachments sent after a previous network await.
            if not self.eligible():
                self.skipped = True
                return SendResult(
                    success=False,
                    error="conversation is no longer selected",
                    retryable=False,
                )
            self.started = True
            try:
                result = await value(*args, **kwargs)
            except BaseException:
                self.unknown = True
                raise
            if not getattr(result, "success", False):
                self.unknown = True
            return result

        return dispatch


class TelegramConversationChannel:
    def __init__(self, ingress):
        if getattr(ingress, "telegram_channel", None) is not None:
            raise ValueError("Telegram channel policy already installed")
        self.ingress, self.runner, self.db = ingress, ingress.runner, ingress.db
        self.store = self.runner.session_store
        self._silent_adapters = {}
        # N07 supplies branch-aware resolution through this existing ingress.
        # SessionStore keeps its legacy resolver when no policy is installed.
        self.store._native_conversation_resolver = ingress._resolve
        # Installation marks existing trusted channel entries. No pointer is
        # moved and no transcript is ended or reopened.
        for source in ingress.trusted_channel_sources():
            if source.platform == Platform.TELEGRAM:
                entry = self.store.lookup_by_session_key(
                    self.runner._session_key_for_source(source)
                )
                if entry:
                    self.store.select_channel_conversation(source, entry.session_id)
        ingress.telegram_channel = self

    def _trusted_source(self, source):
        if source.platform != Platform.TELEGRAM:
            return None
        for trusted in self.ingress.trusted_channel_sources():
            if channel_key(trusted) == channel_key(source):
                return replace(trusted, native_conversation_route=None)
        return None

    def _binding(self, source):
        entry = self.store.lookup_by_session_key(
            self.runner._session_key_for_source(source)
        )
        if not entry or not entry.native_binding_version:
            raise LookupError("Telegram binding unavailable")
        projection = self.ingress._resolve(entry.session_id)
        return {
            "conversation_id": projection["conversation_id"],
            "native_session_id": projection["native_session_id"],
            "binding_version": entry.native_binding_version,
        }

    async def inspect(self, principal, conversation_id):
        _, grant = self.ingress._authorize(principal, conversation_id)
        if (
            grant.source.platform != Platform.TELEGRAM
            or not self.runner._is_user_authorized_for_source(grant.source)
        ):
            raise PermissionError("Telegram binding unavailable")
        return self._binding(grant.source)

    async def delivery(self, principal, conversation_id, execution_id):
        projection, grant = self.ingress._authorize(principal, conversation_id)
        if not self.runner._is_user_authorized_for_source(grant.source):
            raise PermissionError("delivery unavailable")
        row = self.db.native_delivery_lookup(execution_id, channel_key(grant.source))
        if row is None or row["conversation_id"] != projection["conversation_id"]:
            raise LookupError("delivery unavailable")
        return {
            **row,
            "state": "unknown" if row["state"] == "attempting" else row["state"],
        }

    async def select(self, principal, conversation_id, expected_version):
        projection, grant = self.ingress._authorize(principal, conversation_id)
        if (
            grant.source.platform != Platform.TELEGRAM
            or not self.runner._is_user_authorized_for_source(grant.source)
        ):
            raise PermissionError("Telegram binding unavailable")
        if (
            not isinstance(expected_version, int)
            or isinstance(expected_version, bool)
            or expected_version < 1
        ):
            raise ValueError("expected binding version required")
        self.store.select_channel_conversation(
            grant.source, projection["native_session_id"], expected_version
        )
        return self._binding(grant.source)

    def route_event(self, event):
        source = event.source
        if (
            event.internal
            and source.native_conversation_route
            and self._trusted_source(source)
        ):
            # Restored native wakes retain routing, not original browser/user
            # provenance. Never treat the Telegram transport as proof of origin.
            if not hasattr(event, "_native_origin"):
                event._native_origin = "system"
            if event._native_origin != "telegram":
                source._native_silent_progress = True
        if source.native_conversation_route or event.internal:
            return
        trusted = self._trusted_source(source)
        if trusted is None or not self.runner._is_user_authorized_for_source(source):
            return
        native_entry = self.store.lookup_by_session_key(
            self.runner._session_key_for_source(trusted)
        )
        if native_entry is None:
            # Persist the first native root before enabling managed routing.
            native_entry = self.store.get_or_create_session(trusted)
        if not native_entry.native_binding_version:
            self.store.select_channel_conversation(trusted, native_entry.session_id)
        binding = self._binding(trusted)
        # /new confirmation belongs to the channel-selection command, not to
        # whichever execution currently occupies its selected root.
        from tools.slash_confirm import get_pending

        native_key = self.runner._session_key_for_source(trusted)
        if get_pending(native_key):
            return
        # Native session-selection commands address the channel pointer. The
        # SessionStore managed branch does not end an independently active root.
        if event.allow_gateway_control and event.get_command() in {
            "new",
            "reset",
            "resume",
        }:
            return
        if not self.ingress._event_authorized(event, binding["conversation_id"]):
            raise PermissionError("Telegram conversation unavailable")
        event.source = replace(
            source, native_conversation_route=binding["conversation_id"]
        )
        alias = self.store.bind_conversation_alias(
            event.source, binding["native_session_id"]
        )
        event.metadata = {
            **(event.metadata or {}),
            "gateway_session_strict": True,
            "gateway_session_key": alias.session_key,
            "gateway_session_id": alias.session_id,
        }
        event._native_origin = "telegram"

    def execution(self, key):
        state = self.runner._peek_session_state(key)
        execution = state.turn.conversation_execution if state else None
        if execution and self._trusted_source(execution.source):
            return execution
        return None

    def progress_adapter(self, source, adapter):
        if (
            not source.native_conversation_route
            or adapter is None
            or self._trusted_source(source) is None
        ):
            return adapter
        owner = self.ingress._owner(source.native_conversation_route)
        if getattr(source, "_native_silent_progress", False) is not True and (
            owner and owner[2].origin == "telegram"
        ):
            return adapter
        key = id(adapter)
        if key not in self._silent_adapters:
            self._silent_adapters[key] = SilentProgressAdapter(adapter)
        return self._silent_adapters[key]

    async def deliver(self, execution, content, *, failed=False):
        source = self._trusted_source(execution.source)
        if source is None:
            return None
        binding = self._binding(source)
        key = channel_key(source)
        row, reserved = self.db.native_delivery_reserve(
            execution.execution_id,
            key,
            execution.conversation_id,
            binding["binding_version"],
        )
        if not reserved:
            return {
                **row,
                "state": "unknown" if row["state"] == "attempting" else row["state"],
            }

        def eligible():
            current = self._binding(source)
            return (
                self.runner._is_user_authorized_for_source(source)
                and current["conversation_id"] == execution.conversation_id
                and current["binding_version"] == binding["binding_version"]
            )

        state = "unknown"
        try:
            if not eligible() or not content:
                state = "skipped"
            else:
                adapter = self.runner._adapter_for_source(source)
                if adapter is not None:
                    attempt = _FinalAttemptAdapter(adapter, eligible)
                    await self.runner._deliver_queued_first_response(
                        content,
                        source,
                        attempt,
                        metadata={
                            **(self.runner._thread_metadata_for_source(source) or {}),
                            "notify": True,
                            "_native_delivery_guard": eligible,
                            "_native_delivery_once": True,
                        },
                        deliver_media=not failed,
                    )
                    state = (
                        "unknown"
                        if attempt.unknown or (attempt.started and attempt.skipped)
                        else "delivered"
                        if attempt.started
                        else "skipped"
                    )
        except Exception:
            # Delivery failure does not change the completed native execution.
            state = "unknown"
        finally:
            # A cancellation/exception or failed ACK cannot become an automatic
            # resend. A failed commit leaves attempting, also projected unknown.
            self.db.native_delivery_finish(execution.execution_id, key, state)
        return self.db.native_delivery_lookup(execution.execution_id, key)
