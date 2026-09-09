"""Durable commands mounted into the existing native ingress and TurnRunner.

The journal is an input mailbox, not an executor. Wakeups only submit native
MessageEvents; cached AIAgent instances and native leases still run every turn.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import re
from dataclasses import replace
from uuid import uuid4

from gateway.platforms.base import MessageEvent
from hermes_state_commands import CommandConflict, canonical, receipt, process_owner

logger = logging.getLogger(__name__)
CONTRACT_PIN = "fba9a1bea7c86834d672f3a1524128f6c61ebd76"


class NativeCommandContext:
    """Per-TurnRunner handle; durable rows carry individual steer identities."""

    def __init__(self, ingress, execution, command, loop):
        self.ingress = ingress
        self.execution = execution
        self.command = command
        self.loop = loop
        self.user_message = None

    def start_input(self, agent, message, clean=None):
        self.user_message = message
        stored = dict(message)
        if clean is not None:
            stored["content"] = clean
            if message.get("content") != clean:
                stored["api_content"] = message.get("content")
        result = self.ingress.db.native_command_apply(
            self.execution.execution_id,
            self.ingress._command_owner,
            agent.session_id,
            getattr(agent, "_active_session_turn_lease_holder", None),
            stored,
            command=self.command,
        )
        if self.command:
            message["_row_id"] = result["_row_id"]
            message["_db_persisted"] = True

    def refresh_input(self, agent, messages):
        if self.command and any(message is self.user_message for message in messages):
            self.ingress.db.native_command_refresh_input(
                self.execution.execution_id,
                self.ingress._command_owner,
                agent.session_id,
                agent._active_session_turn_lease_holder,
                self.command,
                self.user_message,
            )

    def consume(self, agent, messages, num_tool_msgs=None):
        # The native safe boundary owns message mutation; the existing agent
        # lock also excludes legacy steer publication while we commit it.
        if num_tool_msgs is None:
            start = next(
                (i + 1 for i, m in enumerate(messages) if m is self.user_message),
                len(messages),
            )
        else:
            start = max(0, len(messages) - num_tool_msgs)
        target = next(
            (
                m
                for m in reversed(messages[start:])
                if isinstance(m, dict) and m.get("role") == "tool"
            ),
            None,
        )
        if target is None:
            return
        if not self.ingress.db.native_command_rows(
            self.execution.conversation_id, phase="steer"
        ):
            return
        if agent._flush_messages_to_session_db(messages) is False:
            raise RuntimeError("cannot persist native steer target")
        lock = agent._pending_steer_lock
        with lock:
            result = self.ingress.db.native_command_apply(
                self.execution.execution_id,
                self.ingress._command_owner,
                agent.session_id,
                getattr(agent, "_active_session_turn_lease_holder", None),
                target,
            )
            if result is not None:
                target["content"] = result["content"]

    def finish(self, *, failed=False):
        self.ingress.db.native_execution_close(
            self.execution.execution_id, self.ingress._command_owner, crashed=failed
        )
        self.loop.call_soon_threadsafe(
            self.ingress._commands_finished, self.execution.conversation_id
        )


class DurableCommandIngressMixin:
    def _init_commands(self, concierge_id):
        if str(self.db.db_path) == ":memory:":
            raise ValueError("durable native commands require a persistent SessionDB")
        self._concierge_id = concierge_id
        self._command_owner = f"{process_owner()}:native={uuid4()}"
        self._command_recovery_timer = None
        self._command_wakeups = {}  # Input-dispatch tasks only; never execution ownership.

    def _command_scope(self, principal):
        # SessionDB is already profile-local; conversation is deliberately
        # excluded so changing a target under the same ID is a conflict.
        return canonical([self._concierge_id, principal.issuer, principal.subject])

    async def command_receipt(self, principal, command_id):
        row = await asyncio.to_thread(
            self.db.native_command_lookup, self._command_scope(principal), command_id
        )
        if row is None:
            raise LookupError("command unavailable; absence does not authorize resend")
        _, grant = await asyncio.to_thread(
            self._authorize, principal, row["conversation_id"]
        )
        if not self.runner._is_user_authorized_for_source(grant.source):
            raise PermissionError("native source is not authorized")
        return receipt(row)

    async def submit(self, principal, command):
        """Return a persisted receipt; caller cancellation cannot cancel accepted work."""
        allowed = {
            "schema_version",
            "command_id",
            "conversation_id",
            "target_execution_id",
            "type",
            "payload",
            "expected_model_version",
            "expected_binding_version",
        }
        if not isinstance(command, dict) or set(command) - allowed:
            raise ValueError("invalid command fields")
        if command.get("schema_version") != "1.0":
            raise ValueError("unsupported command schema version")
        for field in ("command_id", "conversation_id"):
            if not isinstance(command.get(field), str) or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", command[field]
            ):
                raise ValueError("invalid command identity")
        payload = command.get("payload")
        if (
            not isinstance(payload, dict)
            or set(payload) != {"text"}
            or not isinstance(payload["text"], str)
            or not payload["text"].strip()
            or len(payload["text"]) > 100000
        ):
            raise ValueError("expected nonempty text payload")
        projection, grant = await asyncio.to_thread(
            self._authorize, principal, command["conversation_id"]
        )
        if not self.runner._is_user_authorized_for_source(grant.source):
            raise PermissionError("native source is not authorized")
        body = {**command, "conversation_id": projection["conversation_id"]}
        scope = self._command_scope(principal)
        old = await asyncio.to_thread(
            self.db.native_command_lookup, scope, body["command_id"]
        )
        if old:
            fingerprint = hashlib.sha256(
                ("native-command-v1\n" + canonical(body)).encode()
            ).hexdigest()
            if not hmac.compare_digest(old["fingerprint"], fingerprint):
                raise CommandConflict("command ID reused with a different payload")
            self._arm_command_recovery()
            self._commands_finished(projection["conversation_id"])
            return receipt(old)
        if self.runner._get_proxy_url():
            raise ValueError("durable commands require the local native executor")
        if body.get("type") not in {"send", "steer", "queue"}:
            raise ValueError("unsupported command action")
        if any(
            key in body
            for key in ("expected_model_version", "expected_binding_version")
        ):
            raise ValueError("version preconditions are not available in N02")
        if body["type"] == "steer" and not body.get("target_execution_id"):
            raise ValueError("steer requires target_execution_id")
        if "target_execution_id" in body and (
            not isinstance(body["target_execution_id"], str)
            or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", body["target_execution_id"]
            )
        ):
            raise ValueError("invalid execution target")
        source = replace(
            grant.source, native_conversation_route=projection["conversation_id"]
        )
        # Fail before receipt admission if the trusted native alias cannot be
        # persisted. The native channel's own current binding is never moved.
        await self.runner.async_session_store.bind_conversation_alias(
            source, projection["native_session_id"]
        )
        self.db.native_execution_reconcile(projection["conversation_id"])
        # No await between durable acceptance and registering its native wakeup.
        row, new = self.db.native_command_admit(
            scope, body, effect="start" if body["type"] == "send" else "queue"
        )
        self._arm_command_recovery()
        if new:
            owner = self._owner(projection["conversation_id"])
            if owner:
                self._enqueue_commands(owner)
            else:
                self._wake_commands(projection["conversation_id"])
        self._arm_command_recovery()
        return receipt(row)

    def _arm_command_recovery(self):
        if self._command_recovery_timer is None and not getattr(
            self.runner, "_draining", False
        ):
            self._command_recovery_timer = asyncio.get_running_loop().call_later(
                30, self._command_recovery_tick
            )

    def _command_recovery_tick(self):
        self._command_recovery_timer = None
        if getattr(self.runner, "_draining", False):
            return
        try:
            self.reconcile_commands()
        except Exception:
            logger.exception("Native command reconciliation unavailable")
            self._arm_command_recovery()

    def stop_command_recovery(self):
        if self._command_recovery_timer is not None:
            self._command_recovery_timer.cancel()
            self._command_recovery_timer = None

    def _command_event(self, row, source, *, entry=None):
        event = MessageEvent(
            text=json.loads(row["payload_json"])["payload"]["text"],
            source=replace(source),
            allow_gateway_control=False,
        )
        event._native_browser_ingress = self
        event._native_send_mode = "queue"
        event._native_command_key = (row["scope"], row["command_id"])
        event._native_execution_id = row["resulting_execution_id"] or str(uuid4())
        if entry:
            event.metadata = {
                "gateway_session_strict": True,
                "gateway_session_key": entry.session_key,
                "gateway_session_id": entry.session_id,
            }
        return event

    def _enqueue_commands(self, owner):
        key, state, execution = owner
        adapter = self.runner._adapter_for_source(execution.source)
        if adapter is None:
            return
        events = [
            getattr(adapter, "_pending_messages", {}).get(key),
            *state.conversation.queued_events,
        ]
        present = {
            getattr(event, "_native_command_key", None) for event in events if event
        }
        for row in self.db.native_command_rows(
            execution.conversation_id, phase="queued"
        ):
            if (row["scope"], row["command_id"]) not in present:
                self.runner._enqueue_fifo(
                    key, self._command_event(row, execution.source), adapter
                )

    def _wake_commands(self, root):
        if (
            root in self._command_wakeups
            or self._owner(root)
            or self.db.native_execution(root)
        ):
            return
        rows = self.db.native_command_rows(root, phase="queued")
        if not rows:
            return
        task = asyncio.create_task(self._dispatch_command(rows[0]))
        self._command_wakeups[root] = task
        tasks = getattr(self.runner, "_background_tasks", None)
        if tasks is not None:
            tasks.add(task)

        def done(future):
            self._command_wakeups.pop(root, None)
            if tasks is not None:
                tasks.discard(future)
            if not future.cancelled() and future.exception():
                logger.error(
                    "Native command dispatch failed", exc_info=future.exception()
                )
            # A release may have happened while this native input task was
            # still unwinding. Only retry if this command actually advanced.
            current = self.db.native_command_lookup(
                rows[0]["scope"], rows[0]["command_id"]
            )
            if current and current["phase"] != "queued":
                self._wake_commands(root)

        task.add_done_callback(done)

    async def _dispatch_command(self, row):
        from gateway.conversation_control import Principal

        _, issuer, subject = json.loads(row["scope"])
        projection, grant = await asyncio.to_thread(
            self._authorize, Principal(issuer, subject), row["conversation_id"]
        )
        if not self.runner._is_user_authorized_for_source(grant.source):
            raise PermissionError("native source is not authorized")
        source = replace(grant.source, native_conversation_route=row["conversation_id"])
        entry = await self.runner.async_session_store.bind_conversation_alias(
            source, projection["native_session_id"]
        )
        return await self.runner._handle_message(
            self._command_event(row, source, entry=entry)
        )

    def _commands_finished(self, root):
        owner = self._owner(root)
        if owner:
            self._enqueue_commands(owner)
        else:
            self._wake_commands(root)

    def reconcile_commands(self):
        roots = set()
        provider = getattr(self, "_grant_provider", None)
        grants = (provider.recovery_grants() if provider is not None else
                  (grant for entries in self.grants.values() for grant in entries))
        for grant in grants:
            try:
                roots.add(self._resolve(grant.session_id)["conversation_id"])
            except LookupError:
                continue
        for root in roots:
            self.db.native_execution_reconcile(root)
            self._commands_finished(root)
            if self.db.native_execution(root) or self.db.native_command_rows(
                root, phase="queued"
            ):
                self._arm_command_recovery()

    def command_managed(self, session_id):
        try:
            root = self._resolve(session_id)["conversation_id"]
        except LookupError:
            return False
        return self.db.native_execution_command_managed(root)

    def command_context(self, key, generation, loop):
        state = self.runner._peek_session_state(key)
        execution = state.turn.conversation_execution if state else None
        if not execution or execution.generation != generation:
            return None
        return NativeCommandContext(
            self, execution, getattr(execution, "command", None), loop
        )
