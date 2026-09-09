"""Human-only grant inspection using the existing permissions HTTP contract.

The service has no grant-by-ID or version/CAS endpoint. Its cursor, random ID,
scope and expiry are immutable and IDs are never reused; only revoked changes.
We bind those immutable fields as a local display fingerprint, then read the
exact cursor and ID again before revocation. This is not a server CAS token.
"""

import hashlib
import html
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlsplit

from gateway.permission_bridge import BridgeError, BridgeRejected, human_http

PAGE_SIZE = 5
MAX_PAGE_BYTES = 4194304
MAX_CURSOR = 9223372036854775807


@dataclass(frozen=True, repr=False)
class GrantOwner:
    user: str
    chat: str
    owner: str
    url: str


@dataclass(frozen=True, repr=False)
class Grant:
    cursor: int
    grant_id: str
    immutable_json: str
    fingerprint: str
    expires: float | None
    revoked: bool

    @property
    def status(self):
        if self.revoked:
            return "revoked"
        if self.expires is not None and self.expires <= time.time():
            return "expired"
        return "not revoked (subject to current service policy)"


def configured_owner(user, chat):
    from hermes_cli.config import load_config

    settings = (load_config() or {}).get("mcp_permissions") or {}
    if settings.get("enabled") is not True:
        raise BridgeRejected()
    owner, url = settings.get("owner"), settings.get("url")
    if not isinstance(owner, str) or not owner or not isinstance(url, str):
        raise BridgeRejected()
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username
            or parsed.password or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
        raise BridgeRejected()
    if any(not isinstance(x, str) or not re.fullmatch(r"-?[0-9]{1,20}", x)
           for x in (user, chat)):
        raise BridgeRejected()
    return GrantOwner(user, chat, owner, url.rstrip("/"))


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                          separators=(",", ":"))
    except (ValueError, TypeError, RecursionError):
        raise BridgeError() from None


def _grant(row, owner):
    if not isinstance(row, dict) or set(row) != {"cursor", "grant_id", "scope", "expires", "revoked"}:
        raise BridgeError()
    cursor, identity, scope, expiry = (row[k] for k in ("cursor", "grant_id", "scope", "expires"))
    if (type(cursor) is not int or not 1 <= cursor <= MAX_CURSOR
            or not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9_-]{32}", identity)
            or type(row["revoked"]) not in (int, bool) or row["revoked"] not in (0, 1)
            or (expiry is not None and (type(expiry) not in (int, float) or not math.isfinite(expiry)))
            or not isinstance(scope, dict)
            or set(scope) != {"actor", "owner", "backend", "tool", "contract_digest", "resources", "template"}
            or any(not isinstance(scope[k], str) or not scope[k] for k in ("actor", "owner", "backend", "tool"))
            or scope["owner"] != owner.owner
            or not isinstance(scope["contract_digest"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", scope["contract_digest"])):
        raise BridgeError()
    resources, template = scope["resources"], scope["template"]
    if (not isinstance(resources, dict) or not resources
            or any(not isinstance(k, str) or not k.startswith("/")
                   or not isinstance(v, str) or not v for k, v in resources.items())
            or not isinstance(template, dict) or set(template) != {"id", "constraints", "ttl_seconds"}
            or not isinstance(template["id"], str) or not template["id"]
            or not isinstance(template["constraints"], dict)
            or (template["ttl_seconds"] is not None
                and (type(template["ttl_seconds"]) is not int or template["ttl_seconds"] <= 0))):
        raise BridgeError()
    immutable = _json({key: row[key] for key in ("cursor", "grant_id", "scope", "expires")})
    return Grant(cursor, identity, immutable, hashlib.sha256(immutable.encode()).hexdigest(),
                 expiry, bool(row["revoked"]))


def list_grants(owner, after=0):
    if type(after) is not int or not 0 <= after <= MAX_CURSOR:
        raise BridgeRejected()
    result = human_http(owner.url, owner.user, owner.chat, f"/grants?after={after}",
                        max_response=MAX_PAGE_BYTES)
    if not isinstance(result, dict) or set(result) != {"items"}:
        raise BridgeError()
    rows = result["items"]
    # Validate the whole response before exposing even a single displayed row.
    if not isinstance(rows, list) or len(rows) > 100 or len(_json(result).encode()) > MAX_PAGE_BYTES:
        raise BridgeError()
    grants = tuple(_grant(row, owner) for row in rows)
    previous, identities = after, set()
    for grant in grants:
        if grant.cursor <= previous or grant.grant_id in identities:
            raise BridgeError()
        previous = grant.cursor
        identities.add(grant.grant_id)
    return grants


def refresh_grant(owner, displayed):
    grants = list_grants(owner, displayed.cursor - 1)
    if (not grants or grants[0].cursor != displayed.cursor
            or grants[0].grant_id != displayed.grant_id
            or grants[0].fingerprint != displayed.fingerprint):
        raise BridgeRejected()
    return grants[0]


def grant_prompt(grant):
    expiry = ("No time limit recorded" if grant.expires is None else
              datetime.fromtimestamp(grant.expires, timezone.utc).isoformat())
    text = ("Standing grant\n" + grant.immutable_json + "\nDisplay fingerprint: " + grant.fingerprint
            + "\nExpiry (UTC): " + expiry
            + "\nStatus: " + grant.status
            + "\nRevocation blocks future use. It cannot recall execution already authorized.")
    if len(text) > 3800:
        raise BridgeError()
    return "<pre>" + html.escape(text) + "</pre>"


def revoke_grant(owner, displayed):
    """Submit one human revocation. On lost ACK, reconcile by one read only.

    Caller already refreshed the fingerprint and rechecked its native UI lease.
    The service serializes revocation against future execution authorization.
    """
    try:
        result = human_http(owner.url, owner.user, owner.chat,
                            "/grants/" + displayed.grant_id + "/revoke", {})
        if not isinstance(result, dict) or set(result) != {"revoked"} or result["revoked"] is not True:
            raise BridgeError()
        return "revoked"
    except BridgeRejected:
        return "rejected"
    except BridgeError:
        try:
            if refresh_grant(owner, displayed).revoked:
                return "revoked_after_uncertain_response"
        except BridgeError:
            pass
        return "unconfirmed"
