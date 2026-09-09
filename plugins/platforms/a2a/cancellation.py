"""Native owner-controller A2A cancellation transport; never a model tool.

Targets come only from execution-scoped dispatch evidence. This module does not
infer ownership from A2A conversation logs, accept caller task URLs, retry a
cancel write, or claim termination of unobserved descendants.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import urllib.parse
from dataclasses import dataclass
from typing import Callable
from uuid import uuid4

import aiohttp

MAX_RESPONSE_BYTES = 256 * 1024
MAX_CONTROL_SECONDS = 30
SUPPORTED_VERSIONS = {"1.0", "1.0.0", "0.3", "0.3.0"}
TASK_STATES = {
    "submitted",
    "working",
    "input-required",
    "auth-required",
    "completed",
    "failed",
    "canceled",
    "rejected",
    "unknown",
}


class ControlProtocolError(ValueError):
    """Fixed diagnostics only; transport and response bodies stay private."""


class ControlRequestRejected(ValueError):
    """The native owner fence rejected this target before cancellation I/O."""


def bounded_identifier(value):
    return (
        isinstance(value, str)
        and 0 < len(value) <= 1024
        and not any(ord(char) < 32 or 0xD800 <= ord(char) <= 0xDFFF for char in value)
    )


def endpoint_fingerprint(url):
    """Exact configured route identity, including path and trailing slash."""
    if not isinstance(url, str) or len(url) > 4096:
        raise ControlProtocolError("invalid_peer_endpoint")
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or any(ord(char) < 33 or ord(char) > 126 for char in url)
    ):
        raise ControlProtocolError("invalid_peer_endpoint")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        raise ControlProtocolError("invalid_peer_endpoint") from None
    return hashlib.sha256(
        json.dumps(
            [parsed.scheme, parsed.hostname.lower(), port, parsed.path],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _origin(url):
    parsed = urllib.parse.urlsplit(url)
    return (
        parsed.scheme,
        parsed.hostname.lower(),
        parsed.port or (443 if parsed.scheme == "https" else 80),
    )


@dataclass(frozen=True)
class RecordedTask:
    """Validated native dispatch row; construction alone is not authorization.

    The owner controller must load this from its execution-scoped journal. It
    must never construct it from browser/model task IDs or global A2A logs.
    """

    dispatch_id: str
    peer_name: str
    configured_endpoint_fingerprint: str
    rpc_endpoint: str
    protocol_version: str
    tenant: str
    task_id: str
    context_id: str
    configured_tenant: str = ""

    def validate(self):
        if (
            not bounded_identifier(self.dispatch_id)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.peer_name)
            or not re.fullmatch(r"[a-f0-9]{64}", self.configured_endpoint_fingerprint)
            or self.protocol_version not in SUPPORTED_VERSIONS
            or not isinstance(self.tenant, str)
            or not isinstance(self.configured_tenant, str)
            or (
                self.configured_tenant
                and not bounded_identifier(self.configured_tenant)
            )
            or (self.tenant and not bounded_identifier(self.tenant))
            or not bounded_identifier(self.task_id)
            or not bounded_identifier(self.context_id)
        ):
            raise ControlProtocolError("invalid_recorded_task")
        endpoint_fingerprint(self.rpc_endpoint)


@dataclass(frozen=True)
class ControlObservation:
    cancel_state: str
    task_state: str
    reason: str
    write_attempted: bool = False


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ControlProtocolError("duplicate_response_member")
        result[key] = value
    return result


def _decode_response(raw):
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ControlProtocolError("response_byte_limit")
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _: (_ for _ in ()).throw(
                ControlProtocolError("non_json_number")
            ),
        )
    except (UnicodeError, ValueError, RecursionError):
        raise ControlProtocolError("invalid_response_json") from None


class _BoundedResolver(aiohttp.abc.AbstractResolver):
    """Deduplicate DNS and retain a finite slot until OS resolution finishes.

    A timed-out HTTP caller cannot later send. OS getaddrinfo may outlive that
    caller; its bounded residual work is not claimed to be physically canceled.
    """

    def __init__(self, limit):
        self._resolver = aiohttp.resolver.ThreadedResolver()
        self._limit = limit
        self._pending = {}

    async def resolve(self, host, port=0, family=0):
        key = (host, port, family)
        task = self._pending.get(key)
        if task is None:
            if len(self._pending) >= self._limit:
                raise OSError("control_dns_capacity")
            task = asyncio.create_task(self._resolver.resolve(host, port, family))
            self._pending[key] = task

            def finished(done):
                self._pending.pop(key, None)
                if not done.cancelled():
                    done.exception()

            task.add_done_callback(finished)
        return await asyncio.shield(task)

    async def close(self):
        await self._resolver.close()


def _envelope(response, request_id):
    if (
        not isinstance(response, dict)
        or response.get("jsonrpc") != "2.0"
        or type(response.get("id")) is not str
        or response["id"] != request_id
        or ("result" in response) == ("error" in response)
        or set(response) - {"jsonrpc", "id", "result", "error"}
    ):
        raise ControlProtocolError("invalid_response_envelope")
    if "error" in response:
        error = response["error"]
        if (
            not isinstance(error, dict)
            or type(error.get("code")) is not int
            or not isinstance(error.get("message"), str)
        ):
            raise ControlProtocolError("invalid_error_envelope")
        return None, error["code"]
    return response["result"], None


def _task(response, request_id, target, auth_values):
    from .tools import _redact_auth

    result, error = _envelope(response, request_id)
    if error is not None:
        return None, error
    # GetTask and CancelTask return a bare Task, in both protocol versions.
    if (
        not isinstance(result, dict)
        or result.get("id") != target.task_id
        or result.get("contextId") != target.context_id
        or not isinstance(result.get("status"), dict)
    ):
        raise ControlProtocolError("task_context_mismatch")
    state = result["status"].get("state")
    if target.protocol_version.startswith("0.3"):
        if state not in TASK_STATES:
            raise ControlProtocolError("invalid_legacy_task_state")
        normalized = state
    else:
        if not isinstance(state, str) or state not in {
            "TASK_STATE_" + s.replace("-", "_").upper() for s in TASK_STATES
        }:
            raise ControlProtocolError("invalid_v1_task_state")
        normalized = state.removeprefix("TASK_STATE_").replace("_", "-").lower()
    if any(
        _redact_auth(value, auth_values) != value
        for value in (target.task_id, target.context_id, target.tenant)
    ):
        raise ControlProtocolError("identifier_conflicts_with_credentials")
    return normalized, None


class TaskControllerClient:
    """One exact configured peer/task read or explicit cancel write at a time."""

    def __init__(self, *, resolve_peer=None, post=None, max_connections=4):
        from .tools import _resolve_peer

        self.resolve_peer = resolve_peer or _resolve_peer
        if type(max_connections) is not int or not 1 <= max_connections <= 64:
            raise ValueError("control connections must be between 1 and 64")
        self.post = post or self._post
        self._max_connections = max_connections
        self._session = None
        self._resolver = None

    async def close(self):
        if self._session is not None:
            await self._session.close()
            self._session = None
        # Keep bounded residual resolver work shared across session reopen.

    async def _post(self, endpoint, body, headers, timeout):
        from .tools import _validate_headers

        headers = {"Content-Type": "application/json", **headers}
        _validate_headers(headers)
        if self._session is None:
            if self._resolver is None:
                self._resolver = _BoundedResolver(self._max_connections)
            connector = aiohttp.TCPConnector(
                limit=self._max_connections,
                limit_per_host=self._max_connections,
                resolver=self._resolver,
                ttl_dns_cache=300,
            )
            self._session = aiohttp.ClientSession(
                connector=connector, trust_env=False, auto_decompress=False
            )
        # Total includes connector/DNS wait, headers and every body byte. No
        # HTTP executor thread survives timeout/cancellation; redirects and
        # compressed response expansion are never enabled.
        async with self._session.post(
            endpoint,
            json=body,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
            allow_redirects=False,
        ) as response:
            if response.status != 200:
                raise ControlProtocolError("unexpected_response_status")
            if response.content_type != "application/json":
                raise ControlProtocolError("unexpected_response_content_type")
            raw = bytearray()
            async for chunk in response.content.iter_chunked(16 * 1024):
                raw.extend(chunk)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise ControlProtocolError("response_byte_limit")
            return _decode_response(raw)

    def _binding(self, target):
        from .tools import _auth_header, _auth_values, _redact_auth

        target.validate()
        peer = self.resolve_peer(target.peer_name)
        if (
            not peer
            or endpoint_fingerprint(peer.get("url"))
            != target.configured_endpoint_fingerprint
        ):
            raise ControlProtocolError("configured_peer_changed")
        if (peer.get("tenant") or "") != target.configured_tenant:
            raise ControlProtocolError("configured_peer_tenant_changed")
        if _origin(peer["url"]) != _origin(target.rpc_endpoint):
            raise ControlProtocolError("configured_peer_origin_changed")
        headers = {
            **_auth_header(peer.get("auth", {})),
            "A2A-Version": target.protocol_version,
        }
        auth_values = _auth_values(headers)
        if any(
            _redact_auth(value, auth_values) != value
            for value in (
                target.task_id,
                target.context_id,
                target.tenant,
                target.rpc_endpoint,
            )
        ):
            raise ControlProtocolError("identifier_conflicts_with_credentials")
        timeout = peer.get("timeout", MAX_CONTROL_SECONDS)
        if type(timeout) is not int or timeout < 1:
            raise ControlProtocolError("invalid_peer_timeout")
        return headers, min(timeout, MAX_CONTROL_SECONDS), auth_values

    async def _rpc(self, target, method, headers, timeout, auth_values):
        request_id = uuid4().hex
        params = {"id": target.task_id}
        if target.tenant:
            params["tenant"] = target.tenant
        body = dict(jsonrpc="2.0", id=request_id, method=method, params=params)
        response = self.post(target.rpc_endpoint, body, headers, timeout)
        if inspect.isawaitable(response):
            response = await response
        return _task(response, request_id, target, auth_values)

    async def observe(self, target, *, cancellation_requested=False):
        try:
            headers, timeout, auth_values = self._binding(target)
            state, error = await self._rpc(
                target,
                "tasks/get" if target.protocol_version.startswith("0.3") else "GetTask",
                headers,
                timeout,
                auth_values,
            )
            if error is not None:
                return ControlObservation("unknown", "unknown", "task_lookup_rejected")
            if state == "canceled":
                return ControlObservation("confirmed", state, "task_reports_canceled")
            if state == "completed":
                return ControlObservation(
                    "already_completed", state, "task_reports_completed"
                )
            return ControlObservation(
                "rejected"
                if cancellation_requested and state in {"failed", "rejected"}
                else "requested"
                if cancellation_requested
                else "not_requested",
                state,
                "task_observed",
            )
        except Exception:
            return ControlObservation(
                "unknown", "unknown", "task_observation_unavailable"
            )

    async def cancel_once(self, target, *, before_send: Callable[[], None]):
        """Read matching task, persist attempt through before_send, write ONCE.

        before_send must durably reserve this dispatch's cancel attempt. Its
        failure prevents the write. A timeout or malformed response is unknown;
        only observe(), never cancel_once(), is safe for automatic reconciliation.
        """
        attempted = False
        try:
            headers, timeout, auth_values = self._binding(target)
            legacy = target.protocol_version.startswith("0.3")
            state, error = await self._rpc(
                target,
                "tasks/get" if legacy else "GetTask",
                headers,
                timeout,
                auth_values,
            )
            if error is not None:
                return ControlObservation("rejected", "unknown", "task_lookup_rejected")
            if state == "canceled":
                return ControlObservation("confirmed", state, "task_already_canceled")
            if state == "completed":
                return ControlObservation(
                    "already_completed", state, "task_already_completed"
                )
            if state in {"failed", "rejected"}:
                return ControlObservation("rejected", state, "task_already_terminal")
            before_send()
            attempted = True
            state, error = await self._rpc(
                target,
                "tasks/cancel" if legacy else "CancelTask",
                headers,
                timeout,
                auth_values,
            )
            if error is not None:
                return ControlObservation(
                    "rejected", "unknown", "cancel_rpc_rejected", True
                )
            # Even a canceled Task response is acknowledgement of this request.
            # A subsequent GetTask may confirm that task's observed state; it
            # cannot establish that unobserved downstream computation stopped.
            return ControlObservation(
                "acknowledged", state, "cancel_rpc_accepted", True
            )
        except ControlRequestRejected:
            return ControlObservation(
                "rejected", "unknown", "cancel_target_unavailable", attempted
            )
        except Exception:
            return ControlObservation(
                "unknown" if attempted else "failed",
                "unknown",
                "cancel_outcome_unavailable" if attempted else "cancel_not_sent",
                attempted,
            )
