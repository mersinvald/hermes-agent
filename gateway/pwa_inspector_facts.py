"""Native SQL aggregates and execution-linked metadata projections."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime

from hermes_state_events import timestamp

LIMITATIONS = [
    "logical_api_calls_include_retries",
    "ttft_not_captured",
    "remote_model_usage_not_captured",
]

# SQL groups actual retained observations, not a scan of a paginated task API.
FACTS_CTE = """
WITH executions AS (
 SELECT * FROM native_executions WHERE conversation_id IN (SELECT value FROM json_each(?))
 AND created_at>=? AND created_at<?
), facts AS (
 SELECT f.execution_id,f.kind,f.body FROM native_inspector_facts f JOIN executions e USING(execution_id)
 UNION ALL
 SELECT r.execution_id,'a2a',json_object(
 'state',CASE WHEN r.observed_state='completed' THEN 'completed'
 WHEN r.observed_state IN ('failed','rejected') THEN 'failed'
 WHEN r.observed_state='canceled' THEN 'cancelled'
 WHEN r.observed_state IN ('submitted','working','input-required') THEN 'running' ELSE 'unknown' END,
 'configured_endpoint_fingerprint',r.configured_endpoint_fingerprint,'configured_tenant',r.configured_tenant,
 'agent_ref',r.caller_agent_ref,'parent_agent_ref',r.caller_parent_agent_ref,'agent_role',r.caller_role,
 'role','specialist','peer_name',CASE WHEN r.binding_key IS NOT NULL AND r.configured_endpoint_fingerprint IS NOT NULL THEN r.peer_name END,'duration_ms',MAX(0,(r.updated_at-r.recorded_at)*1000))
 FROM native_remote_dispatches r JOIN executions e USING(execution_id)
), normalized AS (
 SELECT f.*,
 CASE WHEN f.kind='local' THEN CASE WHEN json_extract(body,'$.parent_agent_ref')=e.facts_ref THEN 'primary' ELSE json_extract(body,'$.parent_agent_ref') END
 ELSE CASE WHEN json_extract(body,'$.agent_ref')=e.facts_ref THEN 'primary' ELSE json_extract(body,'$.agent_ref') END END agent_ref,
 CASE WHEN f.kind='a2a' THEN json_extract(body,'$.agent_role') WHEN f.kind='local' THEN json_extract(body,'$.parent_role') ELSE json_extract(body,'$.role') END agent_role,
 CASE WHEN f.kind='local' THEN CASE WHEN json_extract(body,'$.parent_parent_agent_ref')=e.facts_ref THEN 'primary' ELSE json_extract(body,'$.parent_parent_agent_ref') END
 ELSE CASE WHEN json_extract(body,'$.parent_agent_ref')=e.facts_ref THEN 'primary' ELSE json_extract(body,'$.parent_agent_ref') END END parent_agent_ref,
 CASE WHEN json_extract(body,'$.state')='running' AND e.state!='open'
 THEN 'unknown' ELSE json_extract(body,'$.state') END outcome,
 json_extract(body,'$.duration_ms') duration,
 json_extract(body,'$.input_tokens') input_tokens,json_extract(body,'$.output_tokens') output_tokens,
 json_extract(body,'$.cost.amount') cost
 FROM facts f JOIN executions e USING(execution_id)
)
"""
STATS_SQL = """COUNT(*) call_count, SUM(outcome='completed') completed_count,
 SUM(outcome='failed') failed_count,SUM(outcome='running') active_count,
 COUNT(duration) duration_count,SUM(duration) duration_ms_total,
 SUM(input_tokens IS NOT NULL AND output_tokens IS NOT NULL) usage_known_count,
 SUM(input_tokens) input_tokens,SUM(output_tokens) output_tokens,
 COUNT(cost) cost_known_count,SUM(cost) cost_usd,
 SUM(CASE WHEN cost IS NOT NULL AND json_extract(body,'$.cost.basis')='reported' THEN 1 ELSE 0 END) reported_cost_count,
 SUM(CASE WHEN cost IS NOT NULL AND json_extract(body,'$.cost.basis')='estimated' THEN 1 ELSE 0 END) estimated_cost_count,
 SUM(COALESCE(json_extract(body,'$.attempt_count'),0)) attempt_count,
 SUM(COALESCE(json_extract(body,'$.failed_attempt_count'),0)) failed_attempt_count,
 SUM(MAX(0,COALESCE(json_extract(body,'$.attempt_count'),1)-1)) retry_count"""


def peer_alias(row):
    import re

    name = row.get("peer_name")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
        return None
    from plugins.platforms.a2a.tools import _resolve_peer
    from plugins.platforms.a2a.cancellation import endpoint_fingerprint

    configured = _resolve_peer(name)
    if (
        not configured
        or endpoint_fingerprint(configured.get("url", ""))
        != row.get("configured_endpoint_fingerprint")
        or (configured.get("tenant") or "") != row.get("configured_tenant", "")
    ):
        return None
    return name


class InspectorFacts:
    def __init__(self, inspector):
        self.i = inspector
        self.http = inspector.http

    def coverage(self, rows, now, *extra):
        missing = sum(r.get("_missing", not r["facts_started_at"]) for r in rows)
        expired = sum(r.get("_expired", bool(r["facts_evicted"])) for r in rows)
        starts = [r["facts_started_at"] for r in rows if r["facts_started_at"]]
        reasons = list(LIMITATIONS) + list(extra)
        if missing:
            reasons.append("historical_executions_not_recorded")
        if expired:
            reasons.append("retained_facts_expired")
        return dict(
            state="partial" if missing or expired or extra else "complete",
            capture_started_at=timestamp(min(starts)) if starts else None,
            retained_from=timestamp(
                now - self.http.db._native_events_limits.max_age_seconds
            ),
            not_recorded_execution_count=missing,
            expired_execution_count=expired,
            limitations=list(dict.fromkeys(reasons)),
        )

    def read_detail(self, root, execution, after, upper, limit):
        db = self.http.db

        def read(conn):
            db._native_inspector_prune(conn, time.time())
            row = conn.execute(
                "SELECT * FROM native_executions WHERE conversation_id=? AND execution_id=?",
                (root, execution),
            ).fetchone()
            if row is None:
                raise LookupError("native execution unavailable")
            high = (
                upper
                if upper is not None
                else conn.execute(
                    "SELECT COALESCE(MAX(ordinal),0) FROM native_inspector_facts WHERE execution_id=?",
                    (execution,),
                ).fetchone()[0]
            )
            calls = [
                dict(r)
                for r in conn.execute(
                    "SELECT ordinal,body FROM native_inspector_facts WHERE execution_id=? AND kind='model' AND ordinal>? AND ordinal<=? ORDER BY ordinal LIMIT ?",
                    (execution, after, high, limit + 1),
                )
            ]
            local = [
                json.loads(r[0])
                for r in conn.execute(
                    "SELECT body FROM native_inspector_facts WHERE execution_id=? AND kind='local' ORDER BY ordinal LIMIT 101",
                    (execution,),
                )
            ]
            remote = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM native_remote_dispatches WHERE execution_id=? ORDER BY ordinal LIMIT 101",
                    (execution,),
                )
            ]
            tools = [
                json.loads(r[0])
                for r in conn.execute(
                    "SELECT body FROM native_inspector_facts WHERE execution_id=? AND kind='tool' ORDER BY ordinal LIMIT 101",
                    (execution,),
                )
            ]
            return dict(row), calls, local, remote, high, tools

        return db._execute_write(read)

    async def detail(self, request, principal, task_ref):
        i = self.i
        query = self.http._query(request, ("cursor", "limit", "scope", "owner_keys"))
        limit, scope = i.selection(query)
        config = self.http.ownership.config
        i.authority(principal, scope, config)
        handle = i.handles.get(task_ref)
        if (
            not handle
            or handle[0] <= time.monotonic()
            or handle[1] != principal
            or handle[3] is not config
        ):
            raise LookupError("inspector reference expired")
        _, _, owner, _, root, execution = handle
        if (
            not i.permitted_owner(owner, config, i.owner_keys(query))
            or i.owner(principal, root, scope, config) != owner
        ):
            raise PermissionError("native inspector owner unavailable")
        cursor_scope = self.http._scope(
            principal,
            "inspector-facts:"
            + task_ref
            + ":"
            + scope
            + ":"
            + query.get("owner_keys", ""),
            limit,
        )
        cursor = (
            self.http._cursor(query["cursor"], cursor_scope)
            if "cursor" in query
            else {}
        )
        row, calls, local, remote, high, tools = await asyncio.to_thread(
            self.read_detail,
            root,
            execution,
            cursor.get("after", 0),
            cursor.get("upper"),
            limit,
        )
        if i.owner(principal, root, scope, config) != owner:
            raise PermissionError("native inspector owner changed")
        own = owner.principal == principal
        ref = lambda kind, value: i.ref(principal, owner, config, kind, value)
        agent_ref = lambda value: (
            ref("primary-agent", owner.principal.subject)
            if value == row["facts_ref"]
            else ref("agent", value)
        )
        result = []
        for stored in calls[:limit]:
            fact = json.loads(stored["body"])
            state = fact["state"]
            if state == "running" and (
                row["state"] != "open"
                or row["owner"] != self.http.ingress._command_owner
            ):
                state = "unknown"
            result.append(
                dict(
                    call_ref=ref("call", fact["ref"]),
                    execution_ref=task_ref,
                    agent_ref=agent_ref(fact["agent_ref"]),
                    parent_agent_ref=agent_ref(fact["parent_agent_ref"])
                    if fact.get("parent_agent_ref")
                    else None,
                    role=fact["role"],
                    state=state,
                    started_at=timestamp(fact["started_at"]),
                    ended_at=timestamp(fact["ended_at"])
                    if fact.get("ended_at") is not None
                    else None,
                    duration_ms=fact.get("duration_ms"),
                    requested_model=fact.get("requested_model") if own else None,
                    actual_model=fact.get("actual_model") if own else None,
                    provider=fact.get("provider") if own else None,
                    input_tokens=fact.get("input_tokens") if own else None,
                    output_tokens=fact.get("output_tokens") if own else None,
                    cost=fact.get("cost") if own else None,
                    error_code=fact.get("error_code"),
                    status_code=fact.get("status_code"),
                    trace_ref=ref("trace", fact["trace_id"])
                    if own and fact.get("trace_id")
                    else None,
                    observation_ref=ref("observation", fact["observation_id"])
                    if own and fact.get("observation_id")
                    else None,
                    trace_state="recorded"
                    if own and fact.get("trace_id") and fact.get("observation_id")
                    else "not_recorded",
                    native_correlation_ref=fact["ref"] if own else None,
                    ttft_ms=None,
                    payload_state="withheld",
                    attempt_count=fact["attempt_count"],
                    failed_attempt_count=fact.get("failed_attempt_count", 0),
                    last_attempt_error_code=fact.get("last_attempt_error_code"),
                    last_attempt_status_code=fact.get("last_attempt_status_code"),
                )
            )
        participants = []
        for fact in local[:100]:
            state = (
                fact["state"]
                if fact["state"] != "running" or row["state"] == "open"
                else "unknown"
            )
            participants.append(
                dict(
                    participant_ref=ref("agent", fact["ref"]),
                    invocation_ref=ref("delegation", fact["ref"]),
                    parent_agent_ref=agent_ref(fact["parent_agent_ref"]),
                    kind="agent",
                    role=fact["role"],
                    delegation_kind="local",
                    peer_name=None,
                    state=state,
                    first_observed_at=timestamp(fact["started_at"]),
                    last_observed_at=timestamp(
                        fact.get("ended_at") or fact["started_at"]
                    ),
                    duration_ms=fact.get("duration_ms")
                    if fact.get("ended_at") is not None
                    else 0,
                )
            )
        for fact in remote[: max(0, 100 - len(participants))]:
            state = {
                "completed": "completed",
                "failed": "failed",
                "rejected": "failed",
                "canceled": "cancelled",
                "working": "running",
                "submitted": "running",
            }.get(fact["observed_state"], "unknown")
            participants.append(
                dict(
                    participant_ref=ref(
                        "participant", fact["binding_key"] or fact["dispatch_id"]
                    ),
                    invocation_ref=ref("dispatch", fact["dispatch_id"]),
                    parent_agent_ref=agent_ref(fact["caller_agent_ref"])
                    if fact.get("caller_agent_ref") else None,
                    kind="agent",
                    role="specialist",
                    delegation_kind="a2a",
                    peer_name=peer_alias(fact) if own else None,
                    state=state,
                    first_observed_at=timestamp(fact["recorded_at"]),
                    last_observed_at=timestamp(fact["updated_at"]),
                    duration_ms=max(
                        0, (fact["updated_at"] - fact["recorded_at"]) * 1000
                    ),
                )
            )
        tool_calls = []
        for fact in tools[:100]:
            state = (
                fact["state"]
                if fact["state"] != "running" or row["state"] == "open"
                else "unknown"
            )
            tool_calls.append(
                dict(
                    invocation_ref=ref("tool", fact["ref"]),
                    agent_ref=agent_ref(fact["agent_ref"]),
                    parent_agent_ref=agent_ref(fact["parent_agent_ref"])
                    if fact.get("parent_agent_ref")
                    else None,
                    role=fact["role"],
                    tool_name=fact.get("tool_name") if own else None,
                    state=state,
                    started_at=timestamp(fact["started_at"])
                    if fact.get("started_at") is not None
                    else None,
                    ended_at=timestamp(fact["ended_at"])
                    if fact.get("ended_at") is not None
                    else None,
                    duration_ms=fact.get("duration_ms"),
                )
            )
        more = len(calls) > limit
        return dict(
            schema_version="1.0",
            view="task_facts",
            captured_at=timestamp(time.time()),
            source=i.source("bounded_window"),
            task_ref=task_ref,
            owner_ref=ref("owner", owner.principal.subject),
            owner_scope="self" if own else "other",
            payload_state="withheld",
            model_calls=result,
            delegations=participants,
            tool_calls=tool_calls,
            tool_calls_has_more=len(tools) > 100,
            coverage=self.coverage([row], time.time()),
            model_calls_next_cursor=self.http._new_cursor(
                dict(after=calls[limit - 1]["ordinal"], upper=high), cursor_scope
            )
            if more
            else None,
            model_calls_has_more=more,
            delegations_has_more=len(local) + len(remote) > 100,
            limit=limit,
        )

    def roots(self, start, end):
        with self.http.db._read_ctx() as conn:
            return [
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT conversation_id FROM native_executions WHERE created_at>=? AND created_at<? ORDER BY conversation_id LIMIT 5001",
                    (start, end),
                )
            ]

    def aggregate(self, roots, start, end, visible_names=True):
        db = self.http.db
        params = (json.dumps(roots), start, end)

        def read(conn):
            db._native_inspector_prune(conn, time.time())
            executions = [
                dict(r)
                for r in conn.execute(
                    FACTS_CTE
                    + "SELECT MIN(facts_started_at) facts_started_at,0 facts_evicted,COALESCE(SUM(facts_started_at IS NULL),0) _missing,COALESCE(SUM(facts_evicted!=0),0) _expired FROM executions",
                    params,
                )
            ]
            counts = dict(
                conn.execute(
                    FACTS_CTE
                    + "SELECT COUNT(*) execution_count,SUM(observed_state IN ('queued','starting','running','input_required','cancellation_requested') AND state='open') active_count,SUM(observed_state='completed') completed_count,SUM(observed_state='failed') failed_count,COUNT(completed_at) duration_count,SUM((completed_at-created_at)*1000) duration_ms_total,MIN((completed_at-created_at)*1000) duration_ms_min,MAX((completed_at-created_at)*1000) duration_ms_max FROM executions",
                    params,
                ).fetchone()
            )
            kinds = {
                r["kind"]: dict(r)
                for r in conn.execute(
                    FACTS_CTE
                    + "SELECT kind,"
                    + STATS_SQL
                    + " FROM normalized GROUP BY kind",
                    params,
                )
            }
            groups = {}
            for kind, keys in (
                ("model", ("requested_model", "actual_model", "provider")),
                ("tool", ("tool_name",)),
                (
                    "delegation",
                    (
                        "role",
                        "peer_name",
                        "configured_endpoint_fingerprint",
                        "configured_tenant",
                    ),
                ),
            ):
                where = (
                    "kind IN ('local','a2a')"
                    if kind == "delegation"
                    else "kind='" + kind + "'"
                )
                caller_keys = ("agent_ref", "agent_role", "parent_agent_ref")
                keys = caller_keys + keys
                expression = lambda k: (
                    k
                    if k in caller_keys
                    else "NULL"
                    if not visible_names and k != "role"
                    else "json_extract(body,'$." + k + "')"
                )
                fields = [expression(k) + " AS " + k for k in keys]
                if kind == "delegation":
                    fields.insert(0, "kind AS delegation_kind")
                groupkeys = (
                    ["delegation_kind"] if kind == "delegation" else []
                ) + list(keys)
                rows = [
                    dict(r)
                    for r in conn.execute(
                        FACTS_CTE
                        + "SELECT "
                        + ",".join(fields)
                        + ","
                        + STATS_SQL
                        + " FROM normalized WHERE "
                        + where
                        + " GROUP BY "
                        + ",".join(groupkeys)
                        + " ORDER BY call_count DESC,"
                        + ",".join(groupkeys)
                        + " LIMIT 51",
                        params,
                    )
                ]
                for row in rows[:50]:
                    predicates = [expression(k) + " IS ?" for k in keys]
                    selected = [row[k] for k in keys]
                    if kind == "delegation":
                        predicates.append("kind=?")
                        selected.append(row["delegation_kind"])
                    row["executions"] = [
                        tuple(r)
                        for r in conn.execute(
                            FACTS_CTE
                            + "SELECT DISTINCT e.execution_id,e.conversation_id FROM normalized n JOIN executions e USING(execution_id) WHERE "
                            + where.replace("kind", "n.kind")
                            + " AND "
                            + " AND ".join(predicates)
                            + " ORDER BY e.execution_id LIMIT 21",
                            (*params, *selected),
                        )
                    ]
                groups[kind] = rows
            return executions, counts, kinds, groups

        return db._execute_write(read)

    async def metrics(self, request, principal):
        i = self.i
        query = self.http._query(request, ("from", "to", "scope", "owner_keys"))
        _, scope = i.selection(query)
        try:
            dates = [
                datetime.fromisoformat(query[k].replace("Z", "+00:00"))
                for k in ("from", "to")
            ]
            if any(
                d.utcoffset() is None or d.utcoffset().total_seconds() != 0
                for d in dates
            ):
                raise ValueError()
            start, end = [d.timestamp() for d in dates]
            if not 0 < end - start <= 86400 or end > time.time() + 1:
                raise ValueError()
        except (ValueError, KeyError, OverflowError):
            raise ValueError(
                "native inspector requires a bounded explicit UTC window"
            ) from None
        config = self.http.ownership.config
        i.authority(principal, scope, config)
        candidates = await asyncio.to_thread(self.roots, start, end)
        selected = {}
        keys = i.owner_keys(query)
        for root in candidates[:5000]:
            try:
                owner = i.owner(principal, root, scope, config)
            except PermissionError:
                continue
            if i.permitted_owner(owner, config, keys):
                selected.setdefault(owner.principal, (owner, []))[1].append(root)
        result, captured, limitations = [], [], []
        if len(candidates) > 5000:
            limitations.append("authorization_scope_bounded")
        for owner, roots in selected.values():
            executions, row, kinds, groups = await asyncio.to_thread(
                self.aggregate, roots, start, end, owner.principal == principal
            )
            if any(i.owner(principal, root, scope, config) != owner for root in roots):
                raise PermissionError("native inspector owner changed")
            captured.extend(executions)
            own = owner.principal == principal
            ref = lambda kind, value: i.ref(principal, owner, config, kind, value)
            row.update(
                agent_ref=ref("primary-agent", owner.principal.subject),
                owner_ref=ref("owner", owner.principal.subject),
                owner_scope="self" if own else "other",
                role="primary",
            )
            for kind in ("model", "tool"):
                facts = kinds.get(kind, {})
                for source, target in (
                    ("call_count", "call_count"),
                    ("completed_count", "call_completed_count"),
                    ("failed_count", "call_error_count"),
                    ("active_count", "call_active_count"),
                    ("duration_count", "duration_count"),
                    ("duration_ms_total", "duration_ms_total"),
                ):
                    row[kind + "_" + target] = facts.get(
                        source, None if source == "duration_ms_total" else 0
                    )
            model = kinds.get("model", {})
            for field in (
                "input_tokens",
                "output_tokens",
                "usage_known_count",
                "cost_usd",
                "cost_known_count",
                "attempt_count",
                "failed_attempt_count",
                "retry_count",
                "reported_cost_count",
                "estimated_cost_count",
            ):
                row[field] = model.get(field, 0 if field.endswith("count") else None)
            if not own:
                for field in ("input_tokens", "output_tokens", "cost_usd"):
                    row[field] = None
                for field in (
                    "usage_known_count",
                    "cost_known_count",
                    "reported_cost_count",
                    "estimated_cost_count",
                ):
                    row[field] = 0
            row["local_delegation_count"] = kinds.get("local", {}).get("call_count", 0)
            row["a2a_delegation_count"] = kinds.get("a2a", {}).get("call_count", 0)
            for kind, plural in (
                ("model", "models"),
                ("tool", "tools"),
                ("delegation", "delegations"),
            ):
                row[plural + "_has_more"] = len(groups[kind]) > 50
                if row[plural + "_has_more"]:
                    limitations.append("group_limit_reached")
                row[plural] = []
                for group in groups[kind][:50]:
                    tasks = group.pop("executions")
                    if kind == "delegation":
                        group["peer_name"] = peer_alias(group) if own else None
                        group.pop("configured_endpoint_fingerprint")
                        group.pop("configured_tenant")
                    for key in ("agent_ref", "parent_agent_ref"):
                        raw = group[key]
                        group[key] = (
                            (
                                ref("primary-agent", owner.principal.subject)
                                if raw == "primary"
                                else ref("agent", raw)
                            )
                            if raw
                            else None
                        )
                    group["task_refs_has_more"] = len(tasks) > 20
                    group["task_refs"] = []
                    for execution, root in tasks[:20]:
                        task_ref = ref("execution", execution)
                        i.remember(task_ref, principal, owner, config, root, execution)
                        group["task_refs"].append(task_ref)
                    if group["task_refs_has_more"]:
                        limitations.append("task_ref_limit_reached")
                    if kind != "model":
                        for field in (
                            "input_tokens",
                            "output_tokens",
                            "usage_known_count",
                            "cost_usd",
                            "cost_known_count",
                            "attempt_count",
                            "failed_attempt_count",
                            "retry_count",
                            "reported_cost_count",
                            "estimated_cost_count",
                        ):
                            group.pop(field)
                    if not own:
                        if kind == "model":
                            for field in ("input_tokens", "output_tokens", "cost_usd"):
                                group[field] = None
                            for field in (
                                "usage_known_count",
                                "cost_known_count",
                                "reported_cost_count",
                                "estimated_cost_count",
                            ):
                                group[field] = 0
                        for field in (
                            "requested_model",
                            "actual_model",
                            "provider",
                            "tool_name",
                            "peer_name",
                        ):
                            if field in group:
                                group[field] = None
                    row[plural].append(group)
            result.append(row)
        for owner, roots in selected.values():
            if any(i.owner(principal, root, scope, config) != owner for root in roots):
                raise PermissionError("native inspector owner changed")
        i.authority(principal, scope, config)
        return dict(
            schema_version="1.0",
            view="execution_metrics",
            captured_at=timestamp(time.time()),
            source=i.source("bounded_window"),
            requested_from=timestamp(start),
            requested_to=timestamp(end),
            end_exclusive=True,
            cohort="execution_created",
            coverage=self.coverage(captured, time.time(), *limitations),
            rows=result,
            limitations=list(dict.fromkeys(LIMITATIONS + limitations)),
        )
