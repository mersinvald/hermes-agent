"""
A2A client tools — let the Hermes agent talk to *other* agents as a peer.

Tools (registered in the ``a2a`` toolset):
  - a2a_discover(url)         -> fetch + summarize a peer's Agent Card
  - a2a_call(agent, message)  -> send a task to a peer, return its reply
  - a2a_list()                -> list configured peers + persisted conversations
  - a2a_history(context_id)   -> recall a persisted A2A conversation
  - a2a_orchestrate(...)      -> fan-out task to multiple peers by capability

Peers are resolved from config.yaml under ``a2a_agents``::

    a2a_agents:
      researcher:
        url: "http://localhost:9999"
        auth: { type: bearer, token: "sk-..." }
        timeout: 120
        capabilities: [web_search, research]

Transport is stdlib urllib (no a2a-sdk dependency). The wire format is the A2A
v1.0 JSON-RPC ``message/send`` method; replies from v0.3 peers still parse.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from typing import Any, Optional, TypedDict

from . import protocol, security

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 120
_MAX_AUTH_LENGTH = 8192
_ORCHESTRATE_MAX_WORKERS = 6  # max parallel peers for fan-out


class _PeerError(ValueError):
    """Only locally authored, credential-free diagnostics belong here."""


def _error_text(error: Exception) -> str:
    if isinstance(error, _PeerError):
        return str(error)
    if isinstance(error, urllib.error.HTTPError):
        code = error.code
        if type(code) is int and 100 <= code <= 599:
            return f"Error: peer request failed (HTTP {code})."
    if isinstance(error, TimeoutError):
        return "Error: peer request timed out."
    if isinstance(error, urllib.error.URLError):
        if isinstance(error.reason, TimeoutError):
            return "Error: peer request timed out."
        return "Error: peer transport failed."
    return "Error: A2A request failed (invalid configuration, response, or transport)."


def _auth_values(headers: Optional[dict] = None) -> tuple[str, ...]:
    """Ephemeral snapshot, including the exact Authorization value sent."""
    values = [(entry.get("auth") or {}).get("token")
              for entry in (_load_config().get("a2a_agents") or {}).values()]
    authorization = (headers or {}).get("Authorization", "")
    values.extend((authorization, authorization.partition(" ")[2]))
    return tuple(value for value in values
                 if isinstance(value, str) and 0 < len(value) <= _MAX_AUTH_LENGTH + 7)


def _redact_auth(text: str, values: tuple[str, ...]) -> str:
    """Remove known literal credentials and ASCII case/escaped variants only."""
    for token in values:
        for value in {token, repr(token)[1:-1], json.dumps(token)[1:-1],
                      urllib.parse.quote(token, safe="")}:
            if value:
                text = re.sub(re.escape(value), "[REDACTED]", text, flags=re.IGNORECASE | re.ASCII)
    return text


def _public_tool(handler):
    @wraps(handler)
    def safe_handler(*args, **kwargs):
        try:
            values = _auth_values()
            return _redact_auth(handler(*args, **kwargs), values)
        except Exception as error:
            # Never format transport exceptions, response bodies, URLs, or chains.
            return _error_text(error)
    return safe_handler


def _peer_name(label: str) -> str:
    return label if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", label) else "peer"


def _peer_origin(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    origin = (parsed.scheme, parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    return hashlib.sha256(repr(origin).encode()).hexdigest()


def _recoverable_peer(label: str) -> bool:
    if _peer_name(label) == label:
        return True
    url = urllib.parse.urlsplit(label)
    return (url.scheme in ("http", "https") and bool(url.hostname)
            and not (url.username or url.password or url.query or url.fragment))


# --------------------------------------------------------------------------
# Peer resolution
# --------------------------------------------------------------------------

def _load_config() -> dict:
    try:
        from hermes_cli.config import load_config
        return load_config() or {}
    except Exception:
        return {}


def _resolve_peer(agent: str) -> Optional[dict]:
    """Resolve a peer name to {url, auth, timeout, capabilities}, or treat ``agent`` as a URL."""
    if agent.startswith("http://") or agent.startswith("https://"):
        return {"url": agent, "auth": {}, "timeout": _DEFAULT_TIMEOUT, "capabilities": []}
    cfg = _load_config()
    peers = cfg.get("a2a_agents") or {}
    entry = peers.get(agent)
    if not entry:
        return None
    return {
        "url": entry.get("url", ""),
        "auth": entry.get("auth", {}) or {},
        "timeout": int(entry.get("timeout", _DEFAULT_TIMEOUT)),
        "capabilities": entry.get("capabilities", []) or [],
        "tenant": entry.get("tenant", ""),
        "progress_notify": entry.get("progress_notify") is True,
        "display_name": entry.get("display_name", "Specialist"),
        "progress_messages": entry.get("progress_messages", {}) or {},
        "streaming": entry.get("streaming") is True,
        "expected_permission_actor": entry.get("expected_permission_actor", agent),
    }


def _auth_header(auth: dict) -> dict:
    if auth and auth.get("type") == "bearer" and auth.get("token"):
        token = auth["token"]
        if (not isinstance(token, str) or len(token) > _MAX_AUTH_LENGTH
                or any(ord(c) < 33 or ord(c) > 126 for c in token)):
            raise _PeerError("Error: invalid bearer configuration; token must be bounded printable ASCII without whitespace.")
        return {"Authorization": f"Bearer {token}"}
    return {}


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

class _PeerRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Do not forward peer credentials or continuations across origins."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        old = urllib.parse.urlsplit(req.full_url)
        new = urllib.parse.urlsplit(newurl)
        old_origin = (old.scheme, old.hostname, old.port or (443 if old.scheme == "https" else 80))
        new_origin = (new.scheme, new.hostname, new.port or (443 if new.scheme == "https" else 80))
        if old_origin != new_origin or new.username or new.password:
            raise _PeerError("Error: refusing cross-origin peer redirect.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _http_get_json(url: str, headers: dict, timeout: int) -> dict:
    _validate_headers(headers)
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.build_opener(_PeerRedirectHandler()).open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post_json(url: str, body: dict, headers: dict, timeout: int) -> dict:
    data = json.dumps(body).encode("utf-8")
    hdrs = {"Content-Type": "application/json", "A2A-Version": protocol.PROTOCOL_VERSION, **headers}
    _validate_headers(hdrs)
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
    with urllib.request.build_opener(_PeerRedirectHandler()).open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _validate_headers(headers: dict) -> None:
    # Reject before entering HTTP libraries, whose debug logs may echo headers.
    for key, value in headers.items():
        if (not isinstance(key, str) or not isinstance(value, str)
                or not re.fullmatch(r"[A-Za-z0-9-]+", key)
                or len(value) > _MAX_AUTH_LENGTH + 7
                or any(ord(c) < 32 or ord(c) > 126 for c in value)):
            raise _PeerError("Error: invalid HTTP header configuration; unprintable values are not allowed.")


def _card_url(base_url: str) -> str:
    # A2A v1.0 canonical discovery path. v0.2 used agent.json; servers may
    # still serve that as a legacy alias, but clients should prefer this.
    return base_url.rstrip("/") + "/.well-known/agent-card.json"


def _legacy_card_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/.well-known/agent.json"


def _fetch_card(base_url: str, headers: dict, timeout: int) -> dict:
    try:
        return _http_get_json(_card_url(base_url), headers, timeout)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
    return _http_get_json(_legacy_card_url(base_url), headers, timeout)


def _select_jsonrpc_interface(card: Optional[dict]) -> Optional[dict]:
    if isinstance(card, dict):
        for iface in card.get("supportedInterfaces", []) or []:
            if isinstance(iface, dict) and iface.get("protocolBinding") == "JSONRPC" and iface.get("url"):
                return iface
    return None


def _rpc_url(base_url: str, card: Optional[dict]) -> str:
    """Prefer the card's JSONRPC interface (v1.0 supportedInterfaces), then the
    card's legacy top-level url, then the configured base."""
    iface = _select_jsonrpc_interface(card)
    if iface:
        return str(iface["url"])
    if isinstance(card, dict) and isinstance(card.get("url"), str) and card["url"]:
        return card["url"]
    return base_url.rstrip("/")


def _interface_tenant(card: Optional[dict], peer: dict) -> str:
    iface = _select_jsonrpc_interface(card)
    if iface and iface.get("tenant"):
        return str(iface["tenant"])
    return str(peer.get("tenant") or "")


# --------------------------------------------------------------------------
# Shared send path (used by a2a_call and a2a_orchestrate)
# --------------------------------------------------------------------------

def _short_state(state: str) -> str:
    """TASK_STATE_COMPLETED -> completed (also passes through v0.3 states)."""
    return state.replace("TASK_STATE_", "").replace("_", "-").lower() if state else ""


def _validate_data(data: Any) -> list[dict]:
    """Accept bounded JSON objects, without coercion or implicit decisions."""
    budget = 4096

    def check(value: Any, depth: int = 0) -> None:
        nonlocal budget
        budget -= 1
        if budget < 0 or depth > 16:
            raise ValueError("Error: data exceeds the nesting or item limit.")
        if isinstance(value, str):
            value.encode("utf-8")
        elif value is None or type(value) in (bool, int):
            pass
        elif type(value) is float and math.isfinite(value):
            pass
        elif type(value) is list:
            for item in value:
                check(item, depth + 1)
        elif type(value) is dict and all(type(key) is str for key in value):
            for key, item in value.items():
                check(key, depth + 1)
                check(item, depth + 1)
        else:
            raise ValueError("Error: data must contain only JSON values with string keys.")

    if type(data) is not list or not 1 <= len(data) <= 16 or any(type(item) is not dict or not item for item in data):
        raise ValueError("Error: data must be 1-16 nonempty JSON objects.")
    check(data)
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False)
    if len(encoded.encode("utf-8")) > 65536:
        raise ValueError("Error: data exceeds 65536 UTF-8 bytes.")
    if security.redact_outbound(encoded) != encoded:
        raise ValueError("Error: data contains sensitive content; refusing to alter structured values.")
    return data


def _send_task(agent_label: str, peer: dict, message: str, context_id: str,
               task_id: str = "", data: Optional[list[dict]] = None,
               status_callback=None, follow=False) -> tuple[str, str, str, str]:
    """Send once or explicitly follow existing work. Returns text/context/state/task.

    Raises urllib errors / ValueError for the caller to format. Handles
    outbound redaction, audit, persistence, and metrics.
    """
    base_url = peer.get("url", "")
    headers = _auth_header(peer.get("auth", {}) or {})
    # One auth snapshot covers discovery, GetTask, SendMessage, and persistence.
    # Never reload credentials while processing a response or share this tuple.
    auth_values = _auth_values(headers)
    last_notice = [None]

    def notify(stage):
        if not peer.get("progress_notify") or not callable(status_callback):
            return
        defaults = {"dispatch": "Sending request to {peer}.",
                    "follow": "Following the existing task on {peer}.",
                    "tool": "{peer} is using a tool.",
                    "tool_result": "{peer} received a tool result.",
                    "tool_error": "A tool returned an error to {peer}."}
        catalog = peer.get("progress_messages") or {}
        if not isinstance(catalog, dict):
            return
        kind = "tool" if stage.startswith("tool:") else stage
        text = catalog.get(stage, catalog.get(kind, defaults.get(kind, defaults["tool"])))
        label = peer.get("display_name", "Specialist")
        if not isinstance(text, str) or not isinstance(label, str):
            return
        text = text.replace("{peer}", label)
        if len(text) > 512 or any(ord(c) < 32 for c in text):
            return
        text = _redact_auth(text, auth_values)
        if text == last_notice[0]:
            return
        last_notice[0] = text
        try:
            status_callback("tool_activity", text)
        except Exception:
            pass  # Presentation failure must not replay or fail a peer action.
    timeout = int(peer.get("timeout", _DEFAULT_TIMEOUT))

    # Best-effort card fetch (to learn the rpc URL); non-fatal on failure.
    card = None
    try:
        card = _fetch_card(base_url, headers, min(timeout, 30))
    except Exception:
        if peer.get("streaming") or follow:
            raise _PeerError("Error: streaming discovery failed; no message was sent or replayed.") from None
        pass

    endpoint = _rpc_url(base_url, card)
    def origin(url: str) -> tuple:
        parsed = urllib.parse.urlsplit(url)
        return parsed.scheme, parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)

    if origin(endpoint) != origin(base_url):
        raise _PeerError("Error: Agent Card RPC URL must have the configured peer's origin.")
    iface = _select_jsonrpc_interface(card) or {}
    version = str(iface.get("protocolVersion") or (card or {}).get("protocolVersion") or protocol.PROTOCOL_VERSION)
    legacy = version in ("0.3", "0.3.0")
    if not legacy and version not in ("1.0", "1.0.0"):
        raise _PeerError("Error: unsupported peer protocol version.")
    headers = {**headers, "A2A-Version": version}
    tenant = _interface_tenant(card, peer)
    if task_id:
        records = protocol.load_conversation(context_id, limit=200)
        known = [record["peer_task"] for record in records
                 if (record.get("peer_task") or {}).get("version") == 1
                 and record["peer_task"].get("task_id") == task_id]
        if known and not any(meta.get("peer") == agent_label and meta.get("origin") == _peer_origin(base_url)
                             for meta in known):
            raise _PeerError("Error: task belongs to a different configured peer origin.")
        if follow and not any(meta.get("peer") == agent_label and meta.get("origin") == _peer_origin(base_url)
                              and meta.get("context_id") == context_id for meta in known):
            raise _PeerError("Error: follow requires a persisted task bound to this configured peer/context.")
        if follow and any(_redact_auth(value, auth_values) != value for value in (agent_label, task_id, context_id)):
            raise _PeerError("Error: follow identifiers conflict with configured credentials.")
        # Query the selected peer, not a caller-supplied task URL or local cache.
        params = {"id": task_id}
        if tenant:
            params["tenant"] = tenant
        lookup_id = protocol.new_task_id()
        response = _http_post_json(endpoint, {"jsonrpc": "2.0", "id": lookup_id,
            "method": "tasks/get" if legacy else "GetTask", "params": params}, headers, timeout)
        error = response.get("error") if isinstance(response, dict) else None
        if (not legacy and isinstance(error, dict)
                and type(error.get("code")) is int and error["code"] == -32601):
            if (response.get("jsonrpc") != "2.0" or response.get("id") != lookup_id
                    or "result" in response or not isinstance(error.get("message"), str)):
                raise _PeerError("Error: invalid task lookup error envelope.")
            # Some v1 peers expose task queries only through their legacy binding.
            # Retry this read once, never a send, with the same endpoint/auth/task.
            lookup_id = protocol.new_task_id()
            response = _http_post_json(endpoint, {"jsonrpc": "2.0", "id": lookup_id,
                "method": "tasks/get", "params": params}, {**headers, "A2A-Version": "0.3"}, timeout)
            if (not isinstance(response, dict) or response.get("jsonrpc") != "2.0"
                    or response.get("id") != lookup_id
                    or ("result" in response) == ("error" in response)):
                raise _PeerError("Error: invalid legacy task lookup response envelope.")
        pending = protocol.unwrap_send_message_response(response.get("result", {}))
        if follow:
            if (response.get("jsonrpc") != "2.0" or response.get("id") != lookup_id
                    or "error" in response or not isinstance(pending, dict)
                    or pending.get("id") != task_id or pending.get("contextId") != context_id):
                raise _PeerError("Error: follow task lookup did not match this task/context.")
            permission_hint = ""
            state = _short_state((pending.get("status") or {}).get("state", ""))
            if state in ("submitted", "working", "unknown"):
                if ((card or {}).get("capabilities") or {}).get("streaming") is not True:
                    raise _PeerError("Error: peer does not advertise task streaming; follow unavailable. No new work was sent.")
                from .streaming import send_stream
                notify("follow")
                follow_body = {"jsonrpc": "2.0", "id": protocol.new_task_id(),
                               "method": "tasks/resubscribe" if legacy else "SubscribeToTask", "params": params}
                followed = send_stream(endpoint, follow_body, headers, timeout,
                    context_id=context_id, task_id=task_id, notify=notify,
                    auth_values=auth_values, peer=agent_label, origin=_peer_origin(base_url),
                    initial_task=pending)
                pending = followed["result"]
                permission_hint = followed.get("permission_required", "")
                state = _short_state((pending.get("status") or {}).get("state", ""))
            elif state not in ("input-required", "completed", "canceled", "failed", "rejected", "auth-required"):
                raise _PeerError("Error: follow received an unsupported task state.")
            reply = _redact_auth(_reply_text_from_result(pending), auth_values)
            from gateway.permission_bridge import request_permission
            permission = request_permission(permission_hint or reply, "a2a_agents", agent_label)
            if permission is not None:
                reply = permission
            protocol.persist_message(context_id, "agent", reply, task_id, request_id=lookup_id,
                peer_task={"version": 1, "peer": agent_label, "origin": _peer_origin(base_url),
                           "task_id": task_id, "context_id": context_id, "state": state})
            return reply, context_id, state, task_id
        if ("error" in response or not isinstance(pending, dict) or pending.get("id") != task_id
                or pending.get("contextId") != context_id
                or _short_state((pending.get("status") or {}).get("state", "")) != "input-required"):
            raise _PeerError("Error: task/context is not a matching input-required task on this peer.")

    ctx = context_id or protocol.new_context_id()
    safe_message = _redact_auth(security.redact_outbound(message), auth_values)
    parts = [protocol.text_part(safe_message)] if safe_message else []
    parts.extend(protocol.data_part(item) for item in (data or []))
    outbound = protocol.message_with_parts(protocol.ROLE_USER, parts, context_id=ctx)
    if task_id:
        outbound["taskId"] = task_id
    if legacy:
        outbound["role"] = "user"
        outbound["kind"] = "message"
        for part in outbound["parts"]:
            part["kind"] = "data" if "data" in part else "text"
            part.pop("mediaType", None)
    # v1.0: contextId lives inside the Message, not at the params top level.
    rpc_body = {
        "jsonrpc": "2.0",
        "id": protocol.new_task_id(),
        "method": "message/send" if legacy else "SendMessage",
        "params": {
            "message": outbound,
        },
    }

    if tenant:
        rpc_body["params"]["tenant"] = tenant

    security.audit("outbound", _redact_auth(_peer_name(agent_label), auth_values), rpc_body["id"], safe_message)
    protocol.persist_message(ctx, "user", safe_message, request_id=rpc_body["id"])
    protocol.metrics.outbound_total += 1

    use_stream = peer.get("streaming") and ((card or {}).get("capabilities") or {}).get("streaming") is True
    if use_stream:
        from .streaming import MAX_EVENT_BYTES, send_stream
        rpc_body["method"] = "message/stream" if legacy else "SendStreamingMessage"
        if len(json.dumps(rpc_body).encode("utf-8")) > MAX_EVENT_BYTES:
            raise _PeerError("Error: streaming request exceeds the size limit.")
    _validate_headers(headers)
    notify("dispatch")
    if use_stream:
        resp = send_stream(endpoint, rpc_body, headers, timeout,
                           context_id=context_id, task_id=task_id,
                           notify=notify, auth_values=auth_values,
                           peer=agent_label, origin=_peer_origin(base_url))
    else:
        resp = _http_post_json(endpoint, rpc_body, headers, timeout)
    if "error" in resp:
        raise _PeerError("Error: peer returned a JSON-RPC error.")

    result = resp.get("result", {})
    payload = protocol.unwrap_send_message_response(result)
    reply = _redact_auth(_reply_text_from_result(payload), auth_values)
    reply_ctx, state, reply_task = ctx, "", ""
    if isinstance(payload, dict):
        reply_ctx = payload.get("contextId", "" if "status" in payload else ctx)
        state = (payload.get("status") or {}).get("state", "")
        reply_task = payload.get("id", "") if "status" in payload else ""
    if task_id and (reply_task != task_id or reply_ctx != context_id):
        raise _PeerError("Error: peer did not resume the requested task/context; inspect the peer before retrying.")
    if context_id and reply_ctx != context_id:
        raise _PeerError("Error: peer returned a different context from the requested conversation.")
    if (not isinstance(reply_ctx, str) or not reply_ctx or len(reply_ctx) > 1024
            or any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in reply_ctx)
            or not isinstance(reply_task, str) or len(reply_task) > 1024
            or any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in reply_task)
            or _redact_auth(reply_ctx, auth_values) != reply_ctx or _redact_auth(reply_task, auth_values) != reply_task
            or (isinstance(payload, dict) and "status" in payload and not reply_task)):
        raise _PeerError("Error: peer response lacks valid task/context identifiers.")
    normalized_state = _short_state(state)
    if normalized_state not in ("", "submitted", "working", "input-required", "completed", "canceled", "failed", "rejected", "auth-required", "unknown"):
        raise _PeerError("Error: peer returned an unsupported task state.")
    from gateway.permission_bridge import request_permission
    permission = request_permission(resp.get("permission_required") or reply, "a2a_agents", agent_label)
    if permission is not None:
        reply = permission
    metadata = {"version": 1, "peer": agent_label, "origin": _peer_origin(base_url),
                "task_id": reply_task, "context_id": reply_ctx, "state": normalized_state or "unknown"}
    # Do not persist direct URLs carrying userinfo or query credentials.
    if (not reply_task or not _recoverable_peer(agent_label)
            or _redact_auth(json.dumps(metadata), auth_values) != json.dumps(metadata)):
        metadata = None
    protocol.persist_message(reply_ctx, "agent", reply, reply_task if metadata else "",
                             request_id=rpc_body["id"], peer_task=metadata)
    protocol.metrics.inbound_total += 1
    return reply, reply_ctx, state, reply_task


def _reply_text_from_result(result: Any) -> str:
    result = protocol.unwrap_send_message_response(result)
    if not isinstance(result, dict):
        return str(result)
    status = result.get("status", {}) or {}
    if _short_state(status.get("state", "")) == "input-required":
        message = status.get("message") or {}
        message = {**message, "parts": [p for p in message.get("parts", []) if isinstance(p, dict)
            and not any((p.get("metadata") or {}).get(k) for k in ("adk_thought", "kagent_thought", "thought"))]}
        return protocol.extract_text(message) or "Peer requires input but supplied no question. Stop and inspect the task."
    def final_text(container):
        return protocol.extract_text({"parts": [p for p in container.get("parts", [])
            if isinstance(p, dict) and isinstance(p.get("text"), str)
            and not any((p.get("metadata") or {}).get(k) for k in ("adk_thought", "kagent_thought", "thought"))]})
    # Artifacts are final output only when there is no pending question.
    texts = []
    for artifact in result.get("artifacts", []) or []:
        txt = final_text(artifact)
        if txt:
            texts.append(txt)
    if texts:
        return "\n\n".join(texts)
    status = result.get("status", {}) or {}
    msg = status.get("message")
    if msg:
        return final_text(msg)
    # Bare message result (message/send may return a Message instead of a Task)
    return final_text(result)


# --------------------------------------------------------------------------
# Tool handlers
# --------------------------------------------------------------------------

@_public_tool
def a2a_discover(args: dict, **_: Any) -> str:
    """Fetch and summarize the Agent Card at ``url``."""
    url = str(args.get("url") or "").strip()
    if not url:
        return "Error: 'url' is required (e.g. http://localhost:9999)."
    try:
        card = _fetch_card(url, {}, _DEFAULT_TIMEOUT)
    except Exception as e:
        return _error_text(e)

    name = card.get("name", "?")
    desc = card.get("description", "")
    caps = card.get("capabilities", {}) or {}
    skills = card.get("skills", []) or []
    auth = "yes" if card.get("security") else "no"
    ifaces = card.get("supportedInterfaces", []) or []
    proto = ", ".join(
        f"{i.get('protocolBinding', '?')} v{i.get('protocolVersion', '?')}"
        for i in ifaces if isinstance(i, dict)
    ) or f"v{card.get('protocolVersion', '?')} (pre-1.0 card)"
    lines = [
        f"Agent: {name}",
        f"Description: {desc}",
        f"URL: {_rpc_url(url, card)}",
        f"Protocol: {proto}",
        f"Streaming: {bool(caps.get('streaming'))}  Push: {bool(caps.get('pushNotifications'))}  Auth required: {auth}",
        f"Skills ({len(skills)}):",
    ]
    for s in skills[:20]:
        lines.append(f"  - {s.get('name', s.get('id', '?'))}: {s.get('description', '')}")
    return "\n".join(lines)


@_public_tool
def a2a_call(args: dict, status_callback=None, **_: Any) -> str:
    """Send a task to a peer agent and return its reply.

    ``agent`` is a configured peer name (from ``a2a_agents``) or a direct URL.
    ``context_id`` continues a prior exchange (multi-turn) when provided.
    """
    # Accept common aliases models reach for (observed live: 'agent_name').
    agent = str(args.get("agent") or args.get("agent_name") or args.get("name") or "").strip()
    action = args.get("action", "send")
    if action not in ("send", "follow"):
        return "Error: action must be send or follow."
    message = args.get("message", args.get("text", args.get("task", "")))
    if not isinstance(message, str):
        return "Error: message must be text; use data for structured JSON objects."
    message = message.strip()
    context_id = args.get("context_id", args.get("contextId", ""))
    task_id = args.get("task_id", args.get("taskId", ""))
    data = args.get("data")
    for value in (context_id, task_id):
        if not isinstance(value, str) or len(value) > 1024 or any(ord(char) < 32 or 0xD800 <= ord(char) <= 0xDFFF for char in value):
            return "Error: task_id/context_id must be bounded Unicode strings without control characters."
    context_id = context_id.strip()
    if not isinstance(task_id, str) or (task_id and (not context_id or len(task_id) > 1024)):
        return "Error: task_id must be a string and requires context_id."
    if data is not None:
        try:
            data = _validate_data(data)
        except (ValueError, TypeError, OverflowError, RecursionError) as e:
            return f"Error: invalid structured data ({type(e).__name__})."
        if not task_id:
            return "Error: structured continuation data requires task_id and context_id."
    if action == "follow" and (not task_id or not context_id or message or data is not None
                               or agent.startswith(("http://", "https://"))):
        return "Error: follow requires a configured agent, task_id and context_id, without message or data."
    if not agent or (action == "send" and not (message or data)):
        return "Error: both 'agent' and 'message' are required."

    peer = _resolve_peer(agent)
    if not peer or not peer.get("url"):
        return (
            f"Error: unknown agent '{_peer_name(agent)}'. Configure it under 'a2a_agents' in "
            f"config.yaml or pass a full http(s):// URL."
        )

    try:
        reply, reply_ctx, state, reply_task = _send_task(
            agent, peer, message, context_id, task_id, data,
            **({"follow": True} if action == "follow" else {}),
            **({"status_callback": status_callback} if status_callback else {}))
    except Exception as e:
        return _error_text(e)

    header = f"[{agent} · context {reply_ctx}"
    if reply_task:
        header += f" · task {reply_task}"
    if state:
        header += f" · {_short_state(state)}"
    header += "]"
    body = reply or "(no text reply)"
    if _short_state(state) == "input-required":
        body += (
            "\n\n(The peer needs more input — answer by calling a2a_call again "
            f"with context_id '{reply_ctx}' and task_id '{reply_task}'. "
            "Transmit only explicit user answers in the peer's required format. "
            "Do not invent answers or approval decisions. Context alone starts a new task, not a resume.)"
        )
    return f"{header}\n{body}"


@_public_tool
def a2a_list(args: dict | None = None, **_: Any) -> str:
    """List configured A2A peers and any persisted conversations."""
    cfg = _load_config()
    peers = cfg.get("a2a_agents") or {}
    lines = []
    if peers:
        lines.append(f"Configured peers ({len(peers)}):")
        for name, entry in peers.items():
            auth = (entry.get("auth") or {}).get("type", "none")
            caps = entry.get("capabilities", [])
            cap_str = f" caps: {', '.join(caps)}" if caps else ""
            lines.append(f"  - {name}: {entry.get('url', '?')} (auth: {auth}){cap_str}")
    else:
        lines.append("No peers configured. Add them under 'a2a_agents' in config.yaml.")

    convos = protocol.list_conversations()
    if convos:
        lines.append("")
        lines.append(f"Persisted conversations ({len(convos)}) — recall with a2a_history:")
        for c in convos[:25]:
            lines.append(f"  - {c}")

    # Show metrics snapshot
    m = protocol.metrics.snapshot()
    lines.append("")
    lines.append(f"Metrics: {m['inbound_total']} in / {m['outbound_total']} out, "
                 f"{m['tasks_completed']} completed, {m['tasks_failed']} failed, "
                 f"{m['streams_started']} streams, {m['push_sent']} push sent, "
                 f"{m['anti_loop_triggers']} anti-loop, {m['rate_limit_triggers']} rate-limited, "
                 f"avg {m['avg_latency_ms']}ms")

    return "\n".join(lines)


@_public_tool
def a2a_history(args: dict, **_: Any) -> str:
    """Recall a persisted A2A conversation by context_id.

    This is how prior A2A exchanges survive compaction/restarts: every turn is
    written to ~/.hermes/a2a_conversations/<context>.jsonl and can be reloaded
    here.
    """
    context_id = str(args.get("context_id") or args.get("contextId") or "")
    if not context_id:
        return "Error: 'context_id' is required (see a2a_list for known conversations)."
    try:
        limit = max(1, min(int(args.get("limit") or 50), 200))
    except (ValueError, TypeError):
        limit = 50
    messages = protocol.load_conversation(context_id, limit=limit)
    if not messages:
        return f"No persisted conversation for context '{context_id}'."
    lines = [f"Conversation {context_id} (last {len(messages)} messages):"]
    latest = {}
    for m in messages:
        meta = m.get("peer_task") or {}
        if meta.get("version") == 1:
            latest[(meta["peer"], meta["origin"], meta["task_id"])] = meta
    for meta in latest.values():
        fields = {key: meta[key] for key in ("peer", "task_id", "context_id", "state")}
        lines.append("Task: " + json.dumps(fields, ensure_ascii=False))
    for m in messages:
        role = m.get("role", "?")
        text = (m.get("text") or "").strip()
        if len(text) > 1000:
            text = text[:1000] + " …[truncated]"
        meta = m.get("peer_task") or {}
        association = (f"peer {meta['peer']}; task {meta['task_id']}; recorded state {meta['state']}"
                       if meta.get("version") == 1 else "peer/task/state unknown")
        lines.append(f"[{association}] [{role}] {text}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# a2a_orchestrate: capability-based routing with fan-out
# --------------------------------------------------------------------------

def _match_peers_by_capability(capability: str) -> list[tuple[str, dict]]:
    """Find configured peers that advertise the given capability."""
    cfg = _load_config()
    peers = cfg.get("a2a_agents") or {}
    matches = []
    for name, entry in peers.items():
        caps = entry.get("capabilities", []) or []
        if capability in caps or capability == "*":
            matches.append((name, entry))
    return matches


def _call_peer_sync(agent_name: str, peer_entry: dict, message: str, context_id: str = "") -> tuple[str, str]:
    """Call a single peer synchronously. Returns (agent_name, reply_text)."""
    try:
        peer = {
            "url": peer_entry.get("url", ""),
            "auth": peer_entry.get("auth", {}) or {},
            "timeout": int(peer_entry.get("timeout", _DEFAULT_TIMEOUT)),
        }
        reply, _ctx, _state, _task = _send_task(agent_name, peer, message, context_id)
        if _short_state(_state) == "input-required":
            return (agent_name, f"[input-required; task {_task}; context {_ctx}]\n{reply}\n"
                    "Await explicit user answers; resume with a2a_call task_id and context_id on this peer.")
        return (agent_name, reply or "(no reply)")
    except Exception as e:
        return (_peer_name(agent_name), _error_text(e))


@_public_tool
def a2a_orchestrate(args: dict, **_: Any) -> str:
    """Fan-out a task to multiple peer agents by capability.

    Modes:
      - ``all``: send to all peers matching the capability, return all replies.
      - ``first``: send to all matching peers, return the first successful reply.
      - ``best``: send to all, return the longest successful reply (a coarse
        detail heuristic — use ``all`` when you want to judge yourself).

    Configured peers advertise capabilities in config.yaml::

      a2a_agents:
        researcher:
          url: "http://localhost:9991"
          capabilities: [web_search, research]
        coder:
          url: "http://localhost:9992"
          capabilities: [code, debug]
    """
    capability = str(args.get("capability") or "").strip()
    message = str(args.get("message") or args.get("task") or "").strip()
    mode = str(args.get("mode") or "all").strip().lower()
    context_id = str(args.get("context_id") or "").strip()

    if not message:
        return "Error: 'message' is required."
    if not capability:
        return "Error: 'capability' is required (or use '*' for all peers)."

    matches = _match_peers_by_capability(capability)
    if not matches:
        return "Error: no configured peers advertise the requested capability."

    if mode not in ("all", "first", "best"):
        mode = "all"

    # Fan-out
    results: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=min(len(matches), _ORCHESTRATE_MAX_WORKERS)) as pool:
        futures = {
            pool.submit(_call_peer_sync, name, entry, message, context_id): name
            for name, entry in matches
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                results.append(fut.result())
                if mode == "first" and not results[-1][1].startswith("Error:"):
                    # Got a good reply; cancel peers that haven't started yet.
                    for f in futures:
                        f.cancel()
                    break
            except Exception as e:
                results.append((_peer_name(name), _error_text(e)))

    # Sort results by peer name for deterministic output
    results.sort(key=lambda r: r[0])
    successes = [(name, reply) for name, reply in results if not reply.startswith("Error:")]

    def _all_failed() -> str:
        lines = ["All peers failed:"]
        for name, reply in results:
            lines.append(f"  {name}: {reply}")
        return "\n".join(lines)

    if mode == "best":
        if not successes:
            return _all_failed()
        best = max(successes, key=lambda r: len(r[1]))
        return f"[best: {best[0]}]\n{best[1]}"
    elif mode == "first":
        if not successes:
            return _all_failed()
        name, reply = successes[0]
        return f"[first: {name}]\n{reply}"
    else:  # mode == "all"
        lines = [f"Orchestrated '{capability}' to {len(matches)} peer(s):"]
        for name, reply in results:
            lines.append(f"\n--- {name} ---")
            lines.append(reply)
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Tool schemas + registration
# --------------------------------------------------------------------------

_FunctionSchema = TypedDict("_FunctionSchema", {"name": str, "description": str, "parameters": dict[str, Any]}, total=False)
_ToolSchema = TypedDict("_ToolSchema", {"type": str, "function": _FunctionSchema}, total=False)
_SCHEMAS: dict[str, _ToolSchema] = {
    "a2a_discover": {
        "type": "function",
        "function": {
            "name": "a2a_discover",
            "description": (
                "Fetch and summarize another agent's A2A Agent Card from a URL "
                "(its name, description, capabilities, and skills). Use this to "
                "find out what a remote agent can do before calling it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Base URL of the remote A2A agent, e.g. http://localhost:9999"},
                },
                "required": ["url"],
            },
        },
    },
    "a2a_call": {
        "type": "function",
        "function": {
            "name": "a2a_call",
            "description": (
                "Send a natural-language task to a remote A2A agent and return "
                "its reply. The agent is a peer (any A2A-compliant framework), "
                "not a sub-agent you control. Pass 'context_id' from a previous "
                "reply to continue a multi-turn exchange. To resume input-required, pass both "
                "task_id and context_id from that peer. data carries explicit user responses as "
                "JSON DataParts, never auto-generated approvals. This transport does not grant "
                "permission to execute remote tools; backend authorization must enforce that separately."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent": {"type": "string", "description": "Configured peer name (from a2a_agents) or a full http(s):// URL."},
                    "action": {"type": "string", "enum": ["send", "follow"], "description": "Default send creates/resumes work. Explicit follow only reads/subscribes to an existing persisted task; requires task_id/context_id and no message/data. Never resend after a disconnected stream."},
                    "message": {"type": "string", "description": "The task / message to send the peer, in natural language."},
                    "context_id": {"type": "string", "description": "Optional: context id from a prior reply, to continue the conversation."},
                    "task_id": {"type": "string", "maxLength": 1024, "description": "Existing task ID from the same peer; requires context_id. Sending continues only input-required tasks; action=follow reads/subscribes without sending work."},
                    "data": {"type": "array", "minItems": 1, "maxItems": 16, "items": {"type": "object"}, "description": "Optional structured continuation DataParts, at most 64 KiB and depth 16. Exact peer-defined JSON objects supplied for an explicit user response. Requires task_id/context_id. Never infer approval or fill decision defaults."},
                },
                "required": ["agent"],
            },
        },
    },
    "a2a_list": {
        "type": "function",
        "function": {
            "name": "a2a_list",
            "description": "List configured A2A peer agents, persisted A2A conversations, and metrics.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    "a2a_history": {
        "type": "function",
        "function": {
            "name": "a2a_history",
            "description": (
                "Recall a persisted A2A conversation transcript by context_id "
                "(survives restarts and context compaction). Use a2a_list to "
                "see known context ids."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "context_id": {"type": "string", "description": "Context id of the conversation to recall."},
                    "limit": {"type": "integer", "description": "Max messages to return (default 50, max 200)."},
                },
                "required": ["context_id"],
            },
        },
    },
    "a2a_orchestrate": {
        "type": "function",
        "function": {
            "name": "a2a_orchestrate",
            "description": (
                "Fan-out a task to multiple peer agents by capability. Peers are "
                "matched from config.yaml a2a_agents.*.capabilities. Modes: 'all' "
                "(return all replies), 'first' (first successful), 'best' (longest "
                "successful reply)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "capability": {"type": "string", "description": "Capability to match (e.g. 'research', 'code') or '*' for all peers."},
                    "message": {"type": "string", "description": "The task to send to all matching peers."},
                    "mode": {"type": "string", "enum": ["all", "first", "best"], "description": "How to aggregate results. Default: 'all'."},
                    "context_id": {"type": "string", "description": "Optional: shared context id for all peers."},
                },
                "required": ["capability", "message"],
            },
        },
    },
}

_HANDLERS = {
    "a2a_discover": a2a_discover,
    "a2a_call": a2a_call,
    "a2a_list": a2a_list,
    "a2a_history": a2a_history,
    "a2a_orchestrate": a2a_orchestrate,
}


def _a2a_tools_available() -> bool:
    """check_fn for the outbound client tools: serve them ONLY when the
    operator has opted into A2A somehow — peers configured under
    ``a2a_agents`` in config.yaml, or the inbound platform enabled
    (a peer-reachable Hermes plausibly dials back).

    Maintainer-directed (#95681): these registered unconditionally, so
    every session on every install paid ~561 tok/call for tools whose
    only possible output without config is 'no peers configured'. A2A is
    unrelated to Bot Mode (bots talk over gateway RPCs) — for most
    installs this toolset is foreign-agent plumbing they never enabled.
    Config adds mid-session surface at the next compaction (#97073).
    """
    cfg = {}
    try:
        cfg = _load_config()
        if cfg.get("a2a_agents"):
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        import os as _os

        if _os.getenv("A2A_PORT"):
            return True
        platforms = cfg.get("platforms") or {}
        a2a_cfg = platforms.get("a2a") or {}
        if isinstance(a2a_cfg, dict) and a2a_cfg.get("enabled"):
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def register_tools(ctx) -> None:
    """Register the client tools in the ``a2a`` toolset (config-gated)."""
    for name, schema in _SCHEMAS.items():
        function_schema = schema["function"]
        ctx.register_tool(
            name=name,
            toolset="a2a",
            schema=function_schema,
            handler=_HANDLERS[name],
            description=function_schema["description"],
            emoji="\U0001f9e9",  # puzzle piece
            check_fn=_a2a_tools_available,
        )
