"""Authorized bounded native event subscriptions over the existing SessionDB.

Subscriptions own a cursor only; closing one has no runner/control side effects.
The HTTP mount lives in native_pwa_http, and application metadata lives in S02.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import asdict

from hermes_state_events import timestamp

logger = logging.getLogger(__name__)


def execution_view(row):
    state = row["observed_state"]
    if row["state"] != "open" and state in {"starting", "running"}:
        state = "unknown"
    return dict(
        schema_version="1.0",
        execution_id=row["execution_id"],
        conversation_id=row["conversation_id"],
        origin={"channel": row["origin"]},
        original_model=None,
        original_model_state="unavailable",
        state=state,
        remote_cancellation={
            "state": "unknown"
            if state in {"unknown", "interrupted"}
            else "not_requested"
        },
        created_at=timestamp(row["created_at"])
        if row["created_at"] is not None
        else None,
        updated_at=timestamp(row["updated_at"])
        if row["updated_at"] is not None
        else None,
    )


class NativeEventFeed:
    def __init__(self, ingress, *, conversation_provider=None):
        self.ingress = ingress
        self.conversation_provider = conversation_provider
        self._subscriptions = set()

    async def recover(self, principal, conversation_id, cursor=None):
        ingress = self.ingress
        projection, grant = await asyncio.to_thread(
            ingress._authorize, principal, conversation_id
        )
        if not ingress.runner._is_user_authorized_for_source(grant.source):
            raise PermissionError("native source is not authorized")
        provider = self.conversation_provider or getattr(
            ingress, "native_conversation", None
        )
        if provider is None:
            raise RuntimeError("native conversation projection is unavailable")
        conversation = await provider(principal, projection["conversation_id"])
        state = await asyncio.to_thread(
            ingress.db.native_event_recovery,
            projection["conversation_id"],
            ingress._command_scope(principal),
            cursor,
        )
        # Check again after awaited I/O so removed grants cannot race publication.
        _, grant = await asyncio.to_thread(
            ingress._authorize, principal, conversation_id
        )
        if not ingress.runner._is_user_authorized_for_source(grant.source):
            raise PermissionError("native source is not authorized")
        result = {
            key: state[key] for key in ("schema_version", "status", "cursor", "events")
        }
        with ingress._completion_lock:
            pending_completions = set(ingress._completion_observations)
        for row in state["executions"]:
            if row["state"] == "open" and (
                row["owner"] != ingress._command_owner
                or row["execution_id"] in pending_completions
            ):
                row["observed_state"] = "unknown"
                if row["execution_id"] in pending_completions:
                    result["status"] = "gap"
                    result["detail"] = {"reason": "completion_persistence_pending"}
        active = next(
            (row for row in state["executions"] if row["state"] == "open"), None
        )
        conversation = {
            **conversation,
            "active_execution": (
                dict(
                    execution_id=active["execution_id"],
                    conversation_id=active["conversation_id"],
                    origin=active["origin"],
                    execution_state=active["observed_state"]
                    if active["observed_state"] in {"starting", "running"}
                    else "unknown",
                )
                if active
                else None
            ),
        }
        result["snapshot"] = dict(
            conversation=conversation,
            captured_at=state["captured_at"],
            recovered=cursor is not None,
            recent_executions=[execution_view(row) for row in state["executions"]],
            command_receipts=state["command_receipts"],
            coverage=state["coverage"],
        )
        result["retention"] = asdict(ingress.db._native_events_limits)
        if "detail" in state:
            result["detail"] = state["detail"]
        return result

    def subscribe(self, principal, conversation_id, cursor=None):
        if (
            len(self._subscriptions)
            >= self.ingress.db._native_events_limits.max_subscribers
        ):
            raise RuntimeError("native event subscriber limit reached")
        subscription = NativeSubscription(self, principal, conversation_id, cursor)
        self._subscriptions.add(subscription)
        return subscription

    def close(self):
        for subscription in tuple(self._subscriptions):
            subscription.close()


class NativeSubscription:
    def __init__(self, feed, principal, conversation_id, cursor):
        self.feed, self.principal, self.conversation_id = (
            feed,
            principal,
            conversation_id,
        )
        self.cursor = cursor
        self.closed = False
        self._polling = False

    async def poll(self):
        if self.closed:
            raise RuntimeError("native event subscription closed")
        if self._polling:
            raise RuntimeError("native event subscription already polling")
        self._polling = True
        try:
            result = await self.feed.recover(
                self.principal, self.conversation_id, self.cursor
            )
            if self.closed:
                raise RuntimeError("native event subscription closed")
            self.cursor = result["cursor"]
            return result
        except PermissionError:
            self.close()
            raise
        finally:
            self._polling = False

    def close(self):
        self.closed = True
        self.feed._subscriptions.discard(self)


class NativeActivityObserver:
    """Compose real per-turn callbacks without changing cached agent identity."""

    def __init__(self, context, agent):
        self.context, self.agent = context, agent
        self.original = {}

    def emit(self, kind, payload):
        ctx = self.context
        try:
            ctx.ingress.db.native_activity_event(
                ctx.execution.execution_id, ctx.ingress._command_owner, kind, payload
            )
        except Exception:
            # Losing optional observation must not fail a tool or alter its side
            # effects. Rotate this conversation's epoch so clients see a gap.
            ctx.ingress.db.native_event_mark_gap(ctx.execution.conversation_id)
            logger.warning(
                "Native activity observation unavailable; recovery epoch changed"
            )

    @staticmethod
    def tool_detail(name):
        # Native tool identifiers only: no arbitrary preview, args or result.
        if isinstance(name, str) and re.fullmatch(
            r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}", name
        ):
            return {"tool_name": name}
        return {"tool_name_state": "unavailable"}

    def install(self):
        def observe_start(call_id, name, args):
            identity = str(call_id or "")
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", identity):
                self.emit(
                    "tool_changed",
                    {
                        "activity_id": identity,
                        "state": "running",
                        "detail": self.tool_detail(name),
                    },
                )
            else:
                self.context.ingress.db.native_event_mark_gap(
                    self.context.execution.conversation_id
                )

        def observe_complete(call_id, name, args, result):
            identity = str(call_id or "")
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", identity):
                from agent.display import _detect_tool_failure

                failed, _ = _detect_tool_failure(str(name or "tool"), result)
                self.emit(
                    "tool_changed",
                    {
                        "activity_id": identity,
                        "state": "failed" if failed else "completed",
                        "detail": self.tool_detail(name),
                    },
                )
            else:
                self.context.ingress.db.native_event_mark_gap(
                    self.context.execution.conversation_id
                )

        for name, observer in (
            ("tool_start_callback", observe_start),
            ("tool_complete_callback", observe_complete),
        ):
            original = getattr(self.agent, name, None)
            self.original[name] = original

            def combined(*args, _original=original, _observer=observer, **kwargs):
                try:
                    _observer(*args, **kwargs)
                except Exception:
                    self.context.ingress.db.native_event_mark_gap(
                        self.context.execution.conversation_id
                    )
                if _original is not None:
                    return _original(*args, **kwargs)

            setattr(self.agent, name, combined)
        return self

    def restore(self):
        for name, callback in self.original.items():
            setattr(self.agent, name, callback)
