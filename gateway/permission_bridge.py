"""Native human-only bridge to the fixed MCP permissions HTTP service.

Config: mcp_permissions.{enabled,url,owner}; credentials come exclusively from
MCP_PERMISSIONS_HUMAN_TOKEN. Peer/server expected_permission_actor is an
operator mapping to Gateway apiKey.name, never a display name or tool argument.
No execution, automatic continuation, model tool or permission ledger lives here.
"""

import html
import json
import math
import re
import secrets
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from urllib.parse import urlsplit


MARKER = re.compile(r"\[mkl\.hitl\.request:([A-Za-z0-9_-]{32})\]")
REQUIRED = "Permission unavailable. No confirmed authorization is available; do not replay or invent approval data."


class BridgeError(ValueError):
    def __init__(self):
        super().__init__("Permission bridge unavailable or request not applicable.")


class BridgeRejected(BridgeError):
    """Definitive local validation or service refusal, not transport uncertainty."""


def request_reference(text):
    if not isinstance(text, str) or len(text) > 262144:
        return None
    matches = MARKER.findall(text)
    return matches[0] if len(matches) == 1 else None


@dataclass(eq=False, repr=False)
class RemotePermission:
    reference: str
    actor: str
    owner: str
    user: str
    chat: str
    url: str
    nonce: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    digest: str = ""
    choices: tuple = ()
    expires: float = 0
    message_id: str = ""
    prompt_delivery_unknown: bool = False


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise BridgeError()


def _http(remote, user, chat, body=None):
    """Only native UI code calls this, using actual source/callback identity."""
    from agent.secret_scope import get_secret

    try:
        origin = urlsplit(remote.url)
        if (origin.scheme != "https" or not origin.hostname or origin.username
                or origin.password or origin.path or origin.query or origin.fragment):
            raise BridgeError()
        token = get_secret("MCP_PERMISSIONS_HUMAN_TOKEN") or ""
        if not 32 <= len(token) <= 8192 or not all(33 <= ord(c) <= 126 for c in token):
            raise BridgeError()
        for identity in (user, chat):
            if not isinstance(identity, str) or not re.fullmatch(r"-?[0-9]{1,20}", identity):
                raise BridgeError()
        url = remote.url + "/requests/" + remote.reference
        if not MARKER.fullmatch("[mkl.hitl.request:" + remote.reference + "]"):
            raise BridgeError()
        data = None if body is None else json.dumps(body, allow_nan=False).encode()
        request = urllib.request.Request(
            url + ("/decision" if body is not None else ""), data=data,
            headers={"Authorization": "Bearer " + token,
                     "X-Telegram-User-Id": user, "X-Telegram-Chat-Id": chat,
                     "Content-Type": "application/json"},
            method="GET" if body is None else "POST")
        # No environment proxy or redirect may receive the human credential.
        with urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect()).open(request, timeout=5) as response:
            if response.status != 200:
                raise BridgeError()
            raw = response.read(32769)
        if len(raw) > 32768:
            raise BridgeError()
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise BridgeError()
                result[key] = value
            return result
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
        if not isinstance(value, dict):
            raise BridgeError()
        return value
    except urllib.error.HTTPError as error:
        if error.code in (403, 404):
            raise BridgeRejected() from None
        raise BridgeError() from None
    except Exception:
        # Exceptions and HTTP bodies can contain secrets: never format them.
        raise BridgeError() from None


def inspect_state(remote):
    """Read authority, validating identity before exposing even terminal state."""
    details = _http(remote, remote.user, remote.chat)
    action = details.get("action")
    if (details.get("request_id") != remote.reference or not isinstance(action, dict)
            or action.get("actor") != remote.actor or action.get("owner") != remote.owner):
        raise BridgeRejected()
    state = details.get("state")
    if state not in ("pending", "approved", "consumed", "rejected", "revoked"):
        raise BridgeError()
    # Consumption is irreversible, even after the original consent deadline.
    if state in ("pending", "approved"):
        expiry = details.get("expires")
        if type(expiry) not in (int, float) or not math.isfinite(expiry):
            raise BridgeError()
        if expiry <= time.time():
            state = "expired"
    return details, state


def feedback(status="unavailable", remote=None, details=None, lookup_reference=None):
    sent = bool(remote.message_id) if remote else False
    if remote and not sent and remote.prompt_delivery_unknown:
        sent = None
    message = {
        "unavailable": REQUIRED,
        "forbidden": "Permission request forbidden or not applicable.",
        "consumed": "Operation already consumed; replay blocked. No action executed this invocation.",
        "approved": "Permission already approved but not consumed; no action executed this invocation.",
        "expired": "Permission request expired before execution.",
        "rejected": "Permission request rejected.",
        "revoked": "Permission request revoked.",
        "pending": "Permission remains pending; no confirmed decision is available.",
        "decision_recorded": "Human permission decision recorded for the exact displayed action. No tool was executed or A2A task restarted by this bridge.",
    }[status]
    message += (" Do not retry automatically, change the original operation ID, or invent a fresh ID."
                " A new logical operation requires new user intent. Do not claim a button was sent"
                " unless approval_prompt_sent is true; do not offer command-based approval.")
    result = {"error": message, "permission_status": status,
              "code": "permission_replay" if status == "consumed" else "permission_" + status,
              "approval_prompt_sent": sent, "action_executed": False}
    if lookup_reference is not None:
        # Echo only the untrusted lookup hint, never label it verified authority.
        result["lookup_reference"] = lookup_reference
    if remote and status not in ("forbidden", "unavailable"):
        result["request_id"] = remote.reference
    if status == "consumed":
        outcome = (details or {}).get("outcome")
        result["priorOutcome"] = outcome if outcome in ("result_received", "tool_error", "unknown") else "unknown"
        result["error"] += " Prior outcome is backend metadata, not proof of physical success or an effect count."
    return json.dumps(result)


def inspect_prompt(remote):
    details, state = inspect_state(remote)
    action, choices = details.get("action"), details.get("choices")
    expiry, digest = details.get("expires"), details.get("digest")
    if (details.get("request_id") != remote.reference or state != "pending"
            or not isinstance(action, dict) or action.get("actor") != remote.actor
            or action.get("owner") != remote.owner
            or not {"actor", "owner", "backend", "tool", "schema_version", "contract_digest", "arguments", "resources"} <= action.keys()
            or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or type(expiry) not in (int, float) or not math.isfinite(expiry) or expiry <= time.time()
            or not isinstance(choices, list) or not 2 <= len(choices) <= 8):
        raise BridgeError()
    ids = []
    for choice in choices:
        if (not isinstance(choice, dict) or not isinstance(choice.get("id"), str)
                or not 1 <= len(choice["id"]) <= 128
                or not isinstance(choice.get("choice_scope_label"), str)
                or not choice["choice_scope_label"]
                or (choice["id"] not in ("once", "deny") and not isinstance(choice.get("scope"), dict))):
            raise BridgeError()
        if choice["id"] not in ("once", "deny"):
            scope = choice["scope"]
            template = scope.get("template")
            if (any(scope.get(key) != action[key] for key in
                    ("actor", "owner", "backend", "tool", "contract_digest", "resources"))
                    or not isinstance(template, dict) or template.get("id") != choice["id"]
                    or not isinstance(template.get("constraints"), dict)
                    or "ttl_seconds" not in template
                    or (template["ttl_seconds"] is not None and
                        (type(template["ttl_seconds"]) is not int or template["ttl_seconds"] <= 0))):
                raise BridgeError()
        ids.append(choice["id"])
    if len(set(ids)) != len(ids) or not {"once", "deny"} <= set(ids):
        raise BridgeError()
    # Full, ASCII-escaped service data prevents hidden bidi/control text and
    # HTML injection. Never truncate an action or a standing scope for consent.
    text = "Permission request\n" + json.dumps(
        {"request_id": remote.reference, "digest": digest, "expires": expiry,
         "action": action, "choices": [{"button": i + 1, **choice} for i, choice in enumerate(choices)]}, ensure_ascii=True, allow_nan=False,
        separators=(",", ":"))
    if len(text) > 3800:
        raise BridgeError()
    remote.digest, remote.choices, remote.expires = digest, tuple(ids), expiry
    # Rendering permits a send attempt; only the adapter's message ID proves delivery.
    remote.prompt_delivery_unknown = True
    return "<pre>" + html.escape(text) + "</pre>"


def decide(remote, user, chat, index):
    if (user != remote.user or chat != remote.chat or remote.expires <= time.time()
            or not remote.digest or not 0 <= index < len(remote.choices)):
        raise BridgeRejected()
    choice = remote.choices[index]
    result = _http(remote, user, chat, {"digest": remote.digest, "choice": choice})
    if (result.get("idempotent") is True
            or result.get("state") != ("rejected" if choice == "deny" else "approved")):
        # Consumed/unknown/revoked idempotent responses are not new authority.
        raise BridgeError()
    return "deny" if choice == "deny" else "once"


def request_permission(text, source_kind, source_name):
    """Wait on the native queue, never on a model-provided decision or identity."""
    if not isinstance(text, str) or "[mkl.hitl.request:" not in text:
        return None
    reference = request_reference(text)
    if reference is None:
        return feedback("forbidden")
    remote = None
    try:
        from hermes_cli.config import load_config
        from gateway.session_context import get_session_env
        from tools import approval

        cfg = load_config() or {}
        settings = cfg.get("mcp_permissions") or {}
        peers = cfg.get(source_kind) if source_kind in ("a2a_agents", "mcp_servers") else None
        peer = (peers or {}).get(source_name)
        if (settings.get("enabled") is not True or not isinstance(peer, dict)
                or source_name.startswith(("http://", "https://"))):
            return feedback(lookup_reference=reference)
        actor = peer.get("expected_permission_actor", source_name if source_kind == "a2a_agents" else "")
        owner, url = settings.get("owner"), settings.get("url", "")
        parsed = urlsplit(url)
        if (not isinstance(actor, str) or not actor or not isinstance(owner, str) or not owner
                or parsed.scheme != "https" or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ("", "/")):
            return feedback(lookup_reference=reference)
        def current(name):
            return get_session_env("HERMES_SESSION_" + name, allow_env=False)
        user, chat, session = current("USER_ID"), current("CHAT_ID"), current("KEY")
        if current("PLATFORM") != "telegram" or not user or not chat or not session:
            return feedback(lookup_reference=reference)
        if approval._is_cron_approval_context():
            return feedback(lookup_reference=reference)
        with approval._lock:
            notify = approval._gateway_notify_cbs.get(session)
        if notify is None:
            return feedback(lookup_reference=reference)
        remote = RemotePermission(reference, actor, owner, str(user), str(chat), url.rstrip("/"))
        details, state = inspect_state(remote)
        if state != "pending":
            return feedback(state, remote, details)
        result = approval._await_gateway_decision(session, notify, {
            "command": "", "description": "Remote permission required",
            "remote_permission": remote, "allow_session": False, "allow_permanent": False,
        })
        if result.get("resolved") and result.get("choice") == "once":
            return feedback("decision_recorded", remote)
        # One bounded read handles pending -> terminal between inspection and UI.
        # No polling, requeue, decision POST, or execution is performed here.
        details, state = inspect_state(remote)
        return feedback(state, remote, details)
    except BridgeRejected:
        return feedback("forbidden", remote)
    except Exception:
        return feedback("unavailable", remote)
