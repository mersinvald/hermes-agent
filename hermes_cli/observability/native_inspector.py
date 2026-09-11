"""Content-free observation of existing native model/delegation hooks."""

from __future__ import annotations

import math
import re
import time

HOOKS = frozenset({
    "pre_api_request",
    "post_api_request",
    "api_request_error",
    "subagent_start",
    "subagent_stop",
})


def handles_hook(name):
    return name in HOOKS


def finite(value):
    return (
        value
        if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1e15
        else None
    )


def count(value):
    return value if type(value) is int and 0 <= value <= 10**12 else None


def context():
    from agent.native_execution_context import current_native_execution

    current = current_native_execution()
    if not current or current[1] is None:
        return None
    origin, controller = current
    provider = controller.ingress._grant_provider
    if provider is None:
        return None
    # Exact native root authorization must identify one current owner. The
    # process owner in NativeExecutionOrigin is deliberately not a principal.
    matches = []
    for binding in provider.config.bindings:
        try:
            projection, grant = provider.authorize(
                binding.principal, origin.conversation_id
            )
            if (
                projection["conversation_id"] == origin.conversation_id
                and provider.runner._is_user_authorized_for_source(grant.source)
            ):
                matches.append(binding)
        except (PermissionError, LookupError, ValueError):
            continue
    if len(matches) != 1:
        return None
    return origin, controller.db, matches[0], provider


def key(session, request):
    if (
        not isinstance(session, str)
        or not isinstance(request, str)
        or not request
        or len(session) > 512
        or len(request) > 512
    ):
        return None
    return session + "\0" + request


def labels(provider, values):
    catalog = provider.config.models
    models = {entry.model for entry in catalog.entries} if catalog else set()
    providers = {entry.provider for entry in catalog.entries} if catalog else set()
    models.update(provider.config.inspector_model_names)
    providers.update(provider.config.inspector_provider_names)
    return {
        "requested_model": values.get("model")
        if values.get("model") in models
        else None,
        "actual_model": values.get("response_model")
        if values.get("response_model") in models
        else None,
        "provider": values.get("provider")
        if values.get("provider") in providers
        else None,
    }


def observe_lifecycle(name, **values):
    if name not in HOOKS:
        return
    native = context()
    if native is None:
        return
    origin, db, binding, provider = native
    now = time.time()
    if name.startswith("subagent_"):
        child = values.get("child_session_id")
        identity = key(values.get("parent_turn_id", ""), child)
        role = values.get("child_role")
        if identity is None or role not in {"leaf", "orchestrator"}:
            return
        if name == "subagent_start":
            parent = db.native_inspector_agent(origin, values.get("parent_session_id"))
            if parent is None:
                return
            facts = dict(
                child_session_id=child,
                parent_agent_ref=parent[0],
                parent_role=parent[2],
                parent_parent_agent_ref=parent[1],
                role=role,
                started_at=now,
                ended_at=None,
                state="running",
                duration_ms=None,
            )
        else:
            prior = db.native_inspector_fact(origin.execution_id, "local", identity)
            if prior is None:
                return  # A stop without captured start is not invented delegation.
            status = values.get("child_status")
            facts = dict(
                ended_at=now,
                state=status
                if status in {"completed", "failed", "cancelled"}
                else "unknown",
                duration_ms=max(0, (now - prior["started_at"]) * 1000),
            )
        db.native_inspector_record(origin, "local", identity, facts)
        return
    session = values.get("session_id", "")
    identity = key(session, values.get("api_request_id", ""))
    if identity is None:
        return
    agent = db.native_inspector_agent(origin, session)
    if agent is None:
        return
    previous = db.native_inspector_fact(origin.execution_id, "model", identity)
    start = finite(values.get("started_at"))
    if start is None or start > now + 1:
        return
    attempt = count(values.get("retry_count", 0))
    facts = dict(
        agent_ref=agent[0],
        parent_agent_ref=agent[1],
        role=agent[2],
        execution_ref=agent[3],
        started_at=start,
        attempt_count=max((attempt or 0) + 1, (previous or {}).get("attempt_count", 1)),
        failed_attempt_count=(previous or {}).get("failed_attempt_count", 0),
        last_attempt_error_code=(previous or {}).get("last_attempt_error_code"),
        last_attempt_status_code=(previous or {}).get("last_attempt_status_code"),
        **labels(provider, values),
    )
    if name == "pre_api_request":
        facts.update(
            state="running",
            ended_at=None,
            duration_ms=None,
            input_tokens=None,
            output_tokens=None,
            cost=None,
            error_code=None,
            status_code=None,
        )
    else:
        end = finite(values.get("ended_at"))
        if end is None or end < start or end > now + 1:
            return
        facts.update(ended_at=end, duration_ms=(end - start) * 1000)
        if name == "post_api_request":
            usage = values.get("usage") or {}
            cost = values.get("accounting_cost")
            if not (
                isinstance(cost, dict)
                and set(cost) == {"amount", "currency", "basis", "source"}
                and finite(cost.get("amount")) is not None
                and cost.get("currency") == "USD"
                and cost.get("basis") in {"estimated", "reported"}
                and cost.get("source") == "native_accounting"
            ):
                cost = None
            facts.update(
                state="completed",
                input_tokens=count(usage.get("input_tokens")),
                output_tokens=count(usage.get("output_tokens")),
                cost=cost,
                error_code=None,
                status_code=None,
            )
        else:
            status = count(values.get("status_code"))
            status = status if status is not None and 100 <= status <= 599 else None
            error = values.get("error") or {}
            kind = error.get("type", "") if isinstance(error, dict) else ""
            code = (
                "rate_limit"
                if status == 429
                else "authentication"
                if status in {401, 403}
                else "timeout"
                if kind
                in {"TimeoutError", "APITimeoutError", "ReadTimeout", "ConnectTimeout"}
                else "cancelled"
                if kind == "CancelledError"
                else "provider_error"
                if status
                else "unknown"
            )
            facts.update(
                failed_attempt_count=facts["failed_attempt_count"]
                + int(
                    (previous or {}).get("last_error_attempt") != facts["attempt_count"]
                ),
                last_error_attempt=facts["attempt_count"],
                last_attempt_error_code=code,
                last_attempt_status_code=status,
            )
            facts.update(
                state="failed",
                input_tokens=None,
                output_tokens=None,
                cost=None,
                error_code=code,
                status_code=status,
            )
    db.native_inspector_record(origin, "model", identity, facts)


def correlation(session_id="", api_request_id=""):
    """Plugin-only metadata, sourced from current native authority and storage."""
    native = context()
    if native is None:
        return None
    origin, db, binding, provider = native
    if binding.observation_actor is None:
        return None
    agent = db.native_inspector_agent(origin, session_id)
    if agent is None:
        return None
    identity = key(session_id, api_request_id)
    fact = (
        db.native_inspector_fact(origin.execution_id, "model", identity)
        if identity
        else None
    )
    metadata = {
        "hermes_native_execution_ref": agent[3],
        "hermes_native_task_ref": agent[3],
        "hermes_native_agent_ref": agent[0],
    }
    if agent[1]:
        metadata["hermes_native_parent_agent_ref"] = agent[1]
    if fact:
        metadata["hermes_native_correlation_ref"] = fact["ref"]
        metadata["hermes_native_attempt_count"] = str(fact["attempt_count"])
        for label in ("requested_model", "actual_model", "provider"):
            if fact.get(label) is not None:
                metadata[label] = fact[label]
    http = getattr(provider.runner, "_pwa_http", None)
    workload = getattr(http, "workload", None)
    if workload is not None:
        try:
            workload._check(binding.principal)
            from gateway.pwa_workload import digest

            metadata["hermes_native_workload_ref"] = "d_" + digest(
                "native-inspector-workload-v1", workload.identity
            )
        except Exception:
            pass  # No workload attribution when the exact native source is absent.
    return {
        "user_id": binding.observation_actor,
        "session_id": agent[3],
        "metadata": metadata,
    }


def record_trace(session_id, api_request_id, trace_id, observation_id):
    if not isinstance(trace_id, str) or not re.fullmatch(r"[a-f0-9]{32}", trace_id):
        return
    if not isinstance(observation_id, str) or not re.fullmatch(
        r"[a-f0-9]{16}", observation_id
    ):
        return
    native = context()
    identity = key(session_id, api_request_id)
    if native and identity:
        origin, db, _, _ = native
        if db.native_inspector_fact(origin.execution_id, "model", identity):
            db.native_inspector_record(
                origin,
                "model",
                identity,
                dict(trace_id=trace_id, observation_id=observation_id),
            )
