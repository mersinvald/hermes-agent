"""Observe actual native A2A dispatch identity without transcript-based authority."""

import logging
import re

from agent.native_execution_context import current_native_execution
from .cancellation import RecordedTask, _task, endpoint_fingerprint

logger = logging.getLogger(__name__)


def _binding(agent_label, peer, endpoint, version, tenant):
    from .tools import _resolve_peer, _redact_auth, _auth_values, _auth_header

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", agent_label):
        return None  # Ordinary direct-URL dispatch remains uncancellable coverage.
    configured = _resolve_peer(agent_label)
    if not configured or configured.get("url") != peer.get("url"):
        raise ValueError("configured peer changed before dispatch")
    values = dict(
        peer_name=agent_label,
        configured_endpoint_fingerprint=endpoint_fingerprint(peer["url"]),
        rpc_endpoint=endpoint,
        protocol_version=version,
        tenant=tenant,
        configured_tenant=configured.get("tenant") or "",
    )
    # Reject secret echoes in every durable field before persisting metadata.
    auth_values = _auth_values(_auth_header(peer.get("auth", {})))
    if any(
        not isinstance(value, str) or _redact_auth(value, auth_values) != value
        for value in values.values()
    ):
        raise ValueError("peer metadata conflicts with credentials")
    RecordedTask(
        dispatch_id="validation",
        task_id="validation",
        context_id="validation",
        **values,
    ).validate()
    return values


def validate_managed_task(
    agent_label, peer, endpoint, version, tenant, task_id, context_id
):
    context = current_native_execution()
    if not context or not task_id:
        return
    origin, controller = context
    binding = _binding(agent_label, peer, endpoint, version, tenant)
    if not binding or not controller.db.native_remote_task_owned(
        origin.conversation_id, binding, task_id, context_id
    ):
        raise ValueError("remote task ownership unavailable")


class DispatchObservation:
    def __init__(self, origin, controller, dispatch_id, binding, auth_values):
        self.origin, self.controller, self.dispatch_id = origin, controller, dispatch_id
        self.binding, self.auth_values = binding, auth_values

    def observe(self, task_id, context_id, state):
        if not self.binding:
            return  # Unconfigured target identity cannot become cancel authority.
        try:
            target = RecordedTask(
                dispatch_id=self.dispatch_id,
                task_id=task_id,
                context_id=context_id,
                **self.binding,
            )
            target.validate()
            normalized, _ = _task(
                {
                    "jsonrpc": "2.0",
                    "id": "native-observation",
                    "result": {
                        "id": task_id,
                        "contextId": context_id,
                        "status": {"state": state},
                    },
                },
                "native-observation",
                target,
                self.auth_values,
            )
            self.controller.db.native_remote_dispatch_observe(
                self.origin, self.dispatch_id, task_id, context_id, normalized
            )
            self.controller.observed_dispatch(self.origin)
        except Exception:
            # The original write may already have been accepted. Preserve its
            # attempted/unknown row and never replay it because observation failed.
            self.controller.db.native_event_mark_gap(self.origin.conversation_id)
            logger.warning("Native remote dispatch observation unavailable")


def prepare_dispatch(
    agent_label,
    peer,
    endpoint,
    version,
    tenant,
    request_id,
    *,
    task_id=None,
    context_id=None,
    auth_values=(),
):
    context = current_native_execution()
    if not context:
        return None
    origin, controller = context
    binding = _binding(agent_label, peer, endpoint, version, tenant)
    identity = controller.db.native_remote_dispatch_prepare(
        origin,
        binding,
        request_id,
        task_id=task_id or None,
        context_id=context_id or None,
    )
    return DispatchObservation(origin, controller, identity, binding, auth_values)
