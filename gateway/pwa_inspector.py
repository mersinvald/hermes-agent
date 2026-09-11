"""Bounded, read-only execution inspection over native authority and retained facts.

The administrative projection is explicit native configuration. It never sends
requests as another principal and never exports foreign text or original IDs.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import re
import secrets
import time
from collections import OrderedDict
from datetime import datetime

from gateway.native_events import execution_view
from gateway.pwa_config import canonical
from hermes_state_events import timestamp

STATES = frozenset({
    "queued",
    "starting",
    "running",
    "input_required",
    "cancellation_requested",
    "cancelled",
    "completed",
    "failed",
    "interrupted",
    "unknown",
})
ACTIVE = frozenset({
    "queued",
    "starting",
    "running",
    "input_required",
    "cancellation_requested",
})
KINDS = frozenset({
    "execution_state_changed",
    "tool_changed",
    "delegation_changed",
    "model_route_observed",
    "command_application_changed",
    "delivery_changed",
    "cancel_state_changed",
    "clarification_state_changed",
    "warning",
    "error",
})
TOOLS = frozenset({
    "terminal",
    "read_file",
    "write_file",
    "patch",
    "search_files",
    "a2a_call",
    "a2a_orchestrate",
    "delegate_task",
    "web_search",
    "web_extract",
})
REMOTE_STATES = {
    "submitted": "observed",
    "working": "running",
    "input-required": "input_required",
    "completed": "completed",
    "failed": "failed",
    "rejected": "failed",
    "canceled": "cancelled",
    "auth-required": "input_required",
}


def summary(rows):
    durations = [r["duration_ms"] for r in rows if r["duration_ms"] is not None]
    return dict(
        scope="returned_page",
        task_count=len(rows),
        active_count=sum(r["state"] in ACTIVE for r in rows),
        failed_count=sum(r["state"] == "failed" for r in rows),
        completed_count=sum(r["state"] == "completed" for r in rows),
        duration_count=len(durations),
        duration_ms_total=sum(durations) if durations else None,
    )


class NativeInspector:
    def __init__(self, http):
        self.http = http
        self.key = secrets.token_bytes(32)
        self.handles = OrderedDict()
        self.ordinals = {}

    def source(self, *reasons):
        limits = self.http.db._native_events_limits
        return dict(
            source="native",
            state="partial",
            reasons=list(reasons),
            retention=dict(
                state="source_limited",
                max_age_seconds=limits.max_age_seconds,
                max_count=limits.max_count,
                max_bytes=limits.max_bytes,
            ),
        )

    def authority(self, principal, scope, config):
        ownership = self.http.ownership
        if ownership.config is not config:
            raise PermissionError("native inspector configuration changed")
        binding = ownership.binding(principal)
        if scope == "all" and not binding.inspector_admin:
            raise PermissionError("native inspector metadata authority unavailable")
        return binding

    def owner(self, principal, root, scope, config):
        viewer = self.authority(principal, scope, config)
        choices = (
            (viewer,) + tuple(b for b in config.bindings if b != viewer)
            if scope == "all"
            else (viewer,)
        )
        for binding in choices:
            try:
                projection, grant = self.http.ownership.authorize(
                    binding.principal, root
                )
                if not self.http.runner._is_user_authorized_for_source(grant.source):
                    continue
                if projection["conversation_id"] == root:
                    return binding
            except (PermissionError, LookupError, ValueError):
                continue
        raise PermissionError("native inspector source unavailable")

    @staticmethod
    def owner_keys(query):
        value = query.get("owner_keys")
        if value is not None and not re.fullmatch(
            r"[a-f0-9]{64}(?:,[a-f0-9]{64}){0,99}", value
        ):
            raise ValueError("invalid inspector owner selection")
        return set(value.split(",")) if value is not None else None

    @staticmethod
    def permitted_owner(owner, config, keys):
        key = hashlib.sha256(
            canonical([
                owner.principal.issuer,
                owner.principal.subject,
                config.concierge_id,
            ]).encode()
        ).hexdigest()
        return keys is None or key in keys

    def ref(self, principal, owner, config, kind, value):
        raw = canonical([
            principal.issuer,
            principal.subject,
            config.fingerprint,
            owner.principal.issuer,
            owner.principal.subject,
            kind,
            value,
        ])
        return "d_" + hmac.digest(self.key, raw.encode(), hashlib.sha256).hex()

    def remember(self, ref, principal, owner, config, root, execution):
        if ref not in self.ordinals:
            self.ordinals[ref] = 1 + max(
                (
                    self.ordinals.get(key, 0)
                    for key, handle in self.handles.items()
                    if handle[1] == principal
                    and handle[2] == owner
                    and handle[3] is config
                ),
                default=0,
            )
        self.handles[ref] = (
            time.monotonic() + self.http.config.cursor_ttl,
            principal,
            owner,
            config,
            root,
            execution,
        )
        self.handles.move_to_end(ref)
        while len(self.handles) > 4096:
            removed, _ = self.handles.popitem(last=False)
            self.ordinals.pop(removed, None)

    def task(self, principal, owner, config, row, remote, tool_events=()):
        ref = lambda kind, value: self.ref(principal, owner, config, kind, value)
        own = owner.principal == principal
        native = execution_view(row)
        state = native["state"] if native["state"] in STATES else "unknown"
        ingress = self.http.ingress
        with ingress._completion_lock:
            pending = row["execution_id"] in ingress._completion_observations
        if row["state"] == "open" and (
            row["owner"] != ingress._command_owner or pending
        ):
            state = "unknown"
        task_ref = ref("execution", row["execution_id"])
        self.remember(
            task_ref,
            principal,
            owner,
            config,
            row["conversation_id"],
            row["execution_id"],
        )
        start, end = row["created_at"], row.get("completed_at")
        duration = (
            (
                datetime.fromisoformat(timestamp(end))
                - datetime.fromisoformat(timestamp(start))
            ).total_seconds()
            * 1000
            if start is not None and end is not None and end >= start
            else None
        )
        participants = []
        for item in remote[:16]:
            # A dispatch attempt is a real invocation, not a successful operation.
            participants.append(
                dict(
                    participant_ref=ref(
                        "participant", item["binding_key"] or item["dispatch_id"]
                    ),
                    invocation_ref=ref("dispatch", item["dispatch_id"]),
                    kind="agent",
                    role="specialist",
                    relationship="delegation",
                    state=(
                        "attempted"
                        if item["phase"] == "attempted"
                        else REMOTE_STATES.get(item["observed_state"], "unknown")
                    ),
                    first_observed_at=timestamp(item["recorded_at"]),
                    last_observed_at=timestamp(item["updated_at"]),
                )
            )
        tools = {}
        for event in reversed(tool_events):
            body = json.loads(event["body"])
            payload = body.get("payload", {})
            activity = payload.get("activity_id")
            if not isinstance(activity, str):
                continue
            invocation = ref("activity", [row["execution_id"], activity])
            name = payload.get("detail", {}).get("tool_name")
            current = tools.get(invocation)
            tools[invocation] = dict(
                invocation_ref=invocation,
                role="tool",
                tool_name=name if own and name in TOOLS else None,
                state=payload.get("state")
                if payload.get("state")
                in {"observed", "running", "completed", "failed"}
                else "unknown",
                first_observed_at=current["first_observed_at"]
                if current
                else body["occurred_at"],
                last_observed_at=body["occurred_at"],
            )
        # Only catalog IDs can become selected-model labels, never arbitrary DB strings.
        catalog = config.models
        model = row.get("model_id")
        selected = model if catalog and catalog.route(model) else None
        return dict(
            task_ref=task_ref,
            conversation_ref=ref("conversation", row["conversation_id"]),
            owner_ref=ref("owner", "owner"),
            owner_scope="self" if own else "other",
            ordinal=self.ordinals[task_ref],
            state=state,
            channel=native["origin"]["channel"],
            created_at=native["created_at"],
            updated_at=native["updated_at"],
            completed_at=timestamp(end) if end is not None else None,
            duration_ms=duration,
            selected_model=selected,
            actual_model=None,
            provider=None,
            input_tokens=None,
            output_tokens=None,
            cost=None,
            participants=participants,
            participants_has_more=len(remote) > 16,
            tools=list(tools.values())[-16:],
            tools_has_more=len(tools) > 16 or len(tool_events) > 64,
            tools_state="partial"
            if tools
            else "expired"
            if row["activity_evicted"]
            else "unavailable",
            payload_state="unavailable" if own else "withheld",
        )

    def page_read(self, upper, after):
        db = self.http.db
        with db._read_ctx() as conn:
            if upper is None:
                upper = conn.execute(
                    "SELECT COALESCE(MAX(created_order),0) FROM native_executions"
                ).fetchone()[0]
            rows = [
                dict(r)
                for r in conn.execute(
                    "SELECT e.*,m.model_id,m.model_version FROM native_executions e "
                    "LEFT JOIN native_execution_models m USING(execution_id) "
                    "WHERE e.created_order<=? AND e.created_order<? ORDER BY e.created_order DESC LIMIT 501",
                    (upper, after if after is not None else upper + 1),
                )
            ]
        return upper, rows

    def remote_read(self, execution):
        with self.http.db._read_ctx() as conn:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM native_remote_dispatches WHERE execution_id=? ORDER BY ordinal LIMIT 17",
                    (execution,),
                )
            ]

    def tool_read(self, root, execution, scope):
        with self.http.db._read_ctx() as conn:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT body FROM native_events WHERE conversation_id=? AND occurred_at>=? "
                    "AND json_extract(body,'$.execution_id')=? AND json_extract(body,'$.type')='tool_changed' "
                    "AND (principal_scope IS NULL OR principal_scope=?) ORDER BY ordinal DESC LIMIT 65",
                    (
                        root,
                        time.time()
                        - self.http.db._native_events_limits.max_age_seconds,
                        execution,
                        scope,
                    ),
                )
            ]

    async def page(self, request, principal):
        query = self.http._query(request, ("cursor", "limit", "scope", "owner_keys"))
        limit, scope = self.selection(query)
        keys = self.owner_keys(query)
        config = self.http.ownership.config
        self.authority(principal, scope, config)
        cursor_scope = self.http._scope(
            principal,
            "inspector-tasks:" + scope + ":" + query.get("owner_keys", ""),
            limit,
        )
        state = (
            self.http._cursor(query["cursor"], cursor_scope)
            if "cursor" in query
            else {}
        )
        upper, candidates = await asyncio.to_thread(
            self.page_read, state.get("upper"), state.get("after")
        )
        self.authority(principal, scope, config)
        output, selected, last = [], [], None
        for row in candidates[:500]:
            last = row["created_order"]
            try:
                owner = self.owner(principal, row["conversation_id"], scope, config)
            except PermissionError:
                continue
            if not self.permitted_owner(owner, config, keys):
                continue
            remote = await asyncio.to_thread(self.remote_read, row["execution_id"])
            tool_events = await asyncio.to_thread(
                self.tool_read,
                row["conversation_id"],
                row["execution_id"],
                self.http.ingress._command_scope(owner.principal),
            )
            if self.owner(principal, row["conversation_id"], scope, config) != owner:
                raise PermissionError("native inspector owner changed")
            output.append(self.task(principal, owner, config, row, remote, tool_events))
            selected.append((row["conversation_id"], owner))
            if len(output) == limit:
                break
        for root, owner in selected:
            if self.owner(principal, root, scope, config) != owner:
                raise PermissionError("native inspector owner changed")
        self.authority(principal, scope, config)
        more = last is not None and any(r["created_order"] < last for r in candidates)
        next_cursor = (
            self.http._new_cursor(dict(upper=upper, after=last), cursor_scope)
            if more
            else None
        )
        return dict(
            schema_version="1.0",
            view="tasks",
            captured_at=timestamp(time.time()),
            source=self.source(
                "bounded_window", "native_activity_instrumentation_partial"
            ),
            rows=output,
            next_cursor=next_cursor,
            has_more=more,
            limit=limit,
            summary=summary(output),
            limitations=[
                "returned_page_only",
                "native_activity_instrumentation_partial",
            ],
        )

    @staticmethod
    def selection(query):
        text = query.get("limit", "50")
        if not text.isascii() or not text.isdecimal() or not 1 <= int(text) <= 100:
            raise ValueError("invalid inspector page limit")
        scope = query.get("scope", "self")
        if scope not in {"self", "all"}:
            raise ValueError("invalid inspector scope")
        return int(text), scope

    def detail_read(self, root, execution, command_scope, after, upper):
        db = self.http.db

        def read(conn):
            db._native_event_prune(conn, time.time())
            row = conn.execute(
                "SELECT e.*,m.model_id,m.model_version FROM native_executions e "
                "LEFT JOIN native_execution_models m USING(execution_id) "
                "WHERE e.conversation_id=? AND e.execution_id=?",
                (root, execution),
            ).fetchone()
            if row is None:
                raise LookupError("execution unavailable")
            high = (
                upper
                if upper is not None
                else conn.execute(
                    "SELECT COALESCE(MAX(ordinal),0) FROM native_events"
                ).fetchone()[0]
            )
            expired = (
                after > 0
                and conn.execute(
                    "SELECT 1 FROM native_events WHERE ordinal=? AND conversation_id=?",
                    (after, root),
                ).fetchone()
                is None
            )
            events = [
                dict(r)
                for r in conn.execute(
                    "SELECT ordinal,body,inspector_payload FROM native_events WHERE conversation_id=? AND ordinal>? AND ordinal<=? "
                    "AND json_extract(body,'$.execution_id')=? AND (principal_scope IS NULL OR principal_scope=?) "
                    "ORDER BY ordinal LIMIT 101",
                    (root, after, high, execution, command_scope),
                )
            ]
            return dict(row), events, high, expired

        return db._execute_write(read)

    def activity(self, principal, owner, config, event):
        ref = lambda kind, value: self.ref(principal, owner, config, kind, value)
        body = json.loads(event["body"])
        payload = body.get("payload", {})
        detail = payload.get("detail", {})
        own = owner.principal == principal
        kind = body["type"] if body["type"] in KINDS else "unknown"
        name = detail.get("tool_name") if isinstance(detail, dict) else None
        state = payload.get("state", "unknown")
        allowed = STATES | {"observed", "configured", "expired"}
        inspection = dict(
            state="unavailable" if own else "withheld",
            text=None,
            reason="not_recorded" if own else "other_owner",
        )
        if own and event.get("inspector_payload") is not None:
            retained = json.loads(event["inspector_payload"])
            if (
                isinstance(retained, dict)
                and set(retained) == {"state", "text", "reason"}
                and retained["state"] in {"available", "redacted", "unavailable"}
                and (
                    retained["text"] is None
                    or isinstance(retained["text"], str)
                    and len(retained["text"].encode()) <= 8192
                )
            ):
                inspection = retained
        return dict(
            event_ref=ref("event", body["event_id"]),
            activity_ref=ref("activity", [body["execution_id"], payload["activity_id"]])
            if payload.get("activity_id")
            else None,
            kind=kind,
            state=state if state in allowed else "unknown",
            occurred_at=body["occurred_at"],
            tool_name=name if own and name in TOOLS else None,
            payload=inspection,
        )

    async def detail(self, request, principal, task_ref):
        query = self.http._query(request, ("cursor", "limit", "scope", "owner_keys"))
        limit, scope = self.selection(query)
        keys = self.owner_keys(query)
        config = self.http.ownership.config
        self.authority(principal, scope, config)
        handle = self.handles.get(task_ref)
        if (
            handle is None
            or handle[0] <= time.monotonic()
            or handle[1] != principal
            or handle[3] is not config
        ):
            raise LookupError("inspector reference expired")
        _, _, owner, _, root, execution = handle
        if not self.permitted_owner(owner, config, keys):
            raise PermissionError("native inspector owner selection changed")
        if self.owner(principal, root, scope, config) != owner:
            raise PermissionError("native inspector owner changed")
        cursor_scope = self.http._scope(
            principal,
            "inspector-detail:"
            + task_ref
            + ":"
            + scope
            + ":"
            + query.get("owner_keys", ""),
            limit,
        )
        state = (
            self.http._cursor(query["cursor"], cursor_scope)
            if "cursor" in query
            else {}
        )
        row, events, high, evicted = await asyncio.to_thread(
            self.detail_read,
            root,
            execution,
            self.http.ingress._command_scope(owner.principal),
            state.get("after", 0),
            state.get("upper"),
        )
        remote = await asyncio.to_thread(self.remote_read, execution)
        tool_events = await asyncio.to_thread(
            self.tool_read,
            root,
            execution,
            self.http.ingress._command_scope(owner.principal),
        )
        if self.owner(principal, root, scope, config) != owner:
            raise PermissionError("native inspector owner changed")
        task = self.task(principal, owner, config, row, remote, tool_events)
        output = [
            self.activity(principal, owner, config, event) for event in events[:limit]
        ]
        if owner.principal == principal and any(
            e["payload"]["text"] is not None for e in output
        ):
            task["payload_state"] = "partial"
        more = len(events) > limit
        next_cursor = (
            self.http._new_cursor(
                dict(upper=high, after=events[limit - 1]["ordinal"]), cursor_scope
            )
            if more
            else None
        )
        self.authority(principal, scope, config)
        return dict(
            schema_version="1.0",
            view="task_detail",
            captured_at=timestamp(time.time()),
            source=self.source(
                "bounded_window", "native_activity_instrumentation_partial"
            ),
            task=task,
            events=output,
            next_cursor=next_cursor,
            has_more=more,
            limit=limit,
            history_state="expired"
            if evicted or row["activity_evicted"] and not events
            else "partial",
        )

    async def handle(self, request, principal, parts):
        if request.method != "GET":
            raise LookupError("inspector method unavailable")
        if parts == ["inspector", "tasks"]:
            return await self.page(request, principal), 200
        if parts == ["inspector", "runtime-events"]:
            return await self.runtime(request, principal), 200
        if len(parts) == 3 and parts[:2] == ["inspector", "tasks"]:
            return await self.detail(request, principal, parts[2]), 200
        raise LookupError("inspector resource unavailable")

    def runtime_read(self, after, upper):
        db = self.http.db

        def read(conn):
            db._native_event_prune(conn, time.time())
            high = (
                upper
                if upper is not None
                else conn.execute(
                    "SELECT COALESCE(MAX(ordinal),0) FROM native_events"
                ).fetchone()[0]
            )
            rows = [
                dict(r)
                for r in conn.execute(
                    "SELECT ordinal,conversation_id,principal_scope,body FROM native_events "
                    "WHERE ordinal<=? AND ordinal<? ORDER BY ordinal DESC LIMIT 501",
                    (high, after if after is not None else high + 1),
                )
            ]
            return high, rows

        return db._execute_write(read)

    async def runtime(self, request, principal):
        query = self.http._query(request, ("cursor", "limit", "scope", "owner_keys"))
        limit, scope = self.selection(query)
        keys = self.owner_keys(query)
        config = self.http.ownership.config
        self.authority(principal, scope, config)
        cursor_scope = self.http._scope(
            principal,
            "inspector-runtime:" + scope + ":" + query.get("owner_keys", ""),
            limit,
        )
        state = (
            self.http._cursor(query["cursor"], cursor_scope)
            if "cursor" in query
            else {}
        )
        high, candidates = await asyncio.to_thread(
            self.runtime_read, state.get("after"), state.get("upper")
        )
        self.authority(principal, scope, config)
        output, selected, last = [], [], None
        for event in candidates[:500]:
            last = event["ordinal"]
            root = event["conversation_id"]
            try:
                owner = self.owner(principal, root, scope, config)
            except PermissionError:
                continue
            if not self.permitted_owner(owner, config, keys):
                continue
            if event["principal_scope"] not in {
                None,
                self.http.ingress._command_scope(owner.principal),
            }:
                continue
            body = json.loads(event["body"])
            if not body.get("execution_id"):
                continue
            projected = self.activity(principal, owner, config, event)
            kind = projected["kind"]
            task_ref = self.ref(
                principal, owner, config, "execution", body["execution_id"]
            )
            self.remember(
                task_ref, principal, owner, config, root, body["execution_id"]
            )
            output.append(
                dict(
                    event_ref=projected["event_ref"],
                    task_ref=task_ref,
                    owner_scope="self" if owner.principal == principal else "other",
                    occurred_at=projected["occurred_at"],
                    kind=kind,
                    state=projected["state"],
                    severity="ERROR"
                    if kind == "error"
                    else "WARN"
                    if kind == "warning"
                    else "INFO"
                    if kind != "unknown"
                    else "unknown",
                )
            )
            selected.append((root, owner))
            if len(output) == limit:
                break
        for root, owner in selected:
            if self.owner(principal, root, scope, config) != owner:
                raise PermissionError("native inspector owner changed")
        self.authority(principal, scope, config)
        more = last is not None and any(r["ordinal"] < last for r in candidates)
        next_cursor = (
            self.http._new_cursor(dict(upper=high, after=last), cursor_scope)
            if more
            else None
        )
        return dict(
            schema_version="1.0",
            view="runtime_events",
            captured_at=timestamp(time.time()),
            source=self.source(
                "bounded_window", "native_activity_instrumentation_partial"
            ),
            rows=output,
            next_cursor=next_cursor,
            has_more=more,
            limit=limit,
            scope="retained_native_journal",
        )
