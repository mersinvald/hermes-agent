"""Small reviewed owner payload projections, never the ordinary activity wire.

Supported: configured a2a_call text arguments/results and read_file path/range
arguments. Terminal commands, arbitrary files/results, continuation DataParts,
direct URL peers and unsupported tools stay omitted. This is not a universal
secret detector; known credentials are removed and suspect text is withheld.
"""

import json
import re
from urllib.parse import quote


def omitted(reason):
    return {"state": "unavailable", "text": None, "reason": reason}


def credentials(agent, ingress, name):
    values = [
        getattr(agent, "api_key", None),
        getattr(getattr(ingress, "_grant_provider", None), "_secret", None),
    ]
    if name == "a2a_call":
        try:
            from plugins.platforms.a2a.tools import _auth_values

            values.extend(_auth_values())
        except Exception:
            return None
    return tuple(v for v in values if isinstance(v, str) and 0 < len(v) <= 8192)


def capture(name, args, result, phase, secrets):
    if name not in {"a2a_call", "read_file"}:
        return omitted("unsupported_tool")
    if secrets is None:
        return omitted("credential_boundary_unproven")
    try:
        if isinstance(args, str) and len(args.encode()) <= 8192:
            args = json.loads(args)
        if not isinstance(args, dict):
            return omitted("credential_boundary_unproven")
        if name == "a2a_call":
            from plugins.platforms.a2a.tools import _resolve_peer

            peer = args.get("agent")
            if (
                not isinstance(peer, str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", peer)
                or not _resolve_peer(peer)
            ):
                return omitted("credential_boundary_unproven")
            if args.get("data") is not None:
                return omitted("credential_boundary_unproven")
            if phase == "arguments":
                message = args.get("message", "")
                action = args.get("action", "send")
                if not isinstance(message, str) or action not in {"send", "follow"}:
                    return omitted("credential_boundary_unproven")
                value = {"agent": peer, "action": action, "message": message}
            else:
                if not isinstance(result, str):
                    return omitted("credential_boundary_unproven")
                value = {"result": result}
        elif phase == "arguments":
            path = args.get("path")
            if not isinstance(path, str) or not 1 <= len(path) <= 2048:
                return omitted("credential_boundary_unproven")
            value = {"path": path}
            for field in ("offset", "limit"):
                if field in args:
                    if type(args[field]) is not int or not 0 <= args[field] <= 1000000:
                        return omitted("credential_boundary_unproven")
                    value[field] = args[field]
        else:
            return omitted("credential_boundary_unproven")
        raw = json.dumps({"tool": name, "phase": phase, **value}, ensure_ascii=False)
        if len(raw.encode()) > 7000:
            return omitted("oversized")
        text = raw
        for secret in secrets:
            for encoded in {
                secret,
                json.dumps(secret)[1:-1],
                repr(secret)[1:-1],
                quote(secret, safe=""),
            }:
                text = re.sub(
                    re.escape(encoded),
                    "[REDACTED]",
                    text,
                    flags=re.IGNORECASE | re.ASCII,
                )
        from agent.redact import redact_sensitive_text

        screened = redact_sensitive_text(text, force=True, redact_url_credentials=True)
        # Generic redaction can retain token prefixes/suffixes. Withhold the whole
        # suspected payload instead of exposing even these diagnostic fragments.
        if screened != text:
            return omitted("credential_boundary_unproven")
        inspection = {
            "state": "redacted" if text != raw else "available",
            "text": text,
            "reason": "redacted" if text != raw else None,
        }
        if len(json.dumps(inspection, ensure_ascii=False).encode()) > 8192:
            return omitted("oversized")
        return inspection
    except Exception:
        return omitted("credential_boundary_unproven")
