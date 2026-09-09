"""Strict opt-in settings for the private, single-profile native PWA listener."""
from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from gateway.conversation_control import Principal
from gateway.session import SessionSource

IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")


def identifier(value):
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError("invalid native PWA identifier")
    return value


def bounded_text(value, maximum=512):
    if (not isinstance(value, str) or not value or len(value) > maximum
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise ValueError("invalid native PWA text setting")
    return value


def closed(value, allowed, required=()):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise ValueError("invalid native PWA fields")
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def source_identity(source):
    return {"platform": source.platform.value, "chat_id": source.chat_id,
            "chat_type": source.chat_type, "user_id": source.user_id,
            "thread_id": source.thread_id, "scope_id": source.scope_id,
            "profile": source.profile}


def parse_principal(data):
    closed(data, {"issuer", "subject", "concierge_id"}, {"issuer", "subject", "concierge_id"})
    issuer = bounded_text(data["issuer"], 2048)
    parsed = urlsplit(issuer)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("invalid native PWA issuer")
    return Principal(issuer, bounded_text(data["subject"])), identifier(data["concierge_id"])


@dataclass(frozen=True)
class SourceBinding:
    source_id: str
    source: SessionSource
    session_ids: tuple[str, ...]

    @property
    def identity_json(self):
        return canonical(source_identity(self.source))


@dataclass(frozen=True)
class OwnerBinding:
    principal: Principal
    sources: tuple[SourceBinding, ...]
    default_source_id: str

    @property
    def default_source(self):
        return next(source for source in self.sources if source.source_id == self.default_source_id)


@dataclass(frozen=True)
class PwaHttpConfig:
    enabled: bool = False
    concierge_id: str = "default"
    host: str = "127.0.0.1"
    port: int = 8766
    allowed_hosts: tuple[str, ...] = ()
    private_network: bool = False
    tls_cert: str | None = None
    tls_key: str | None = None
    token_env: str = "HERMES_PWA_FACADE_TOKEN"
    max_body_bytes: int = 524288
    max_response_bytes: int = 2097152
    max_requests: int = 32
    request_timeout: int = 15
    cursor_ttl: int = 900
    bindings: tuple[OwnerBinding, ...] = ()

    @classmethod
    def from_dict(cls, raw):
        if raw is None:
            return cls()
        fields = set(cls.__dataclass_fields__)
        closed(raw, fields)
        for key in ("enabled", "private_network"):
            if key in raw and type(raw[key]) is not bool:
                raise ValueError("native PWA flags must be boolean")
        enabled = raw.get("enabled", False)
        if not enabled:
            if set(raw) != {"enabled"} and raw:
                raise ValueError("disabled native PWA settings must contain only enabled")
            return cls()
        required = {"concierge_id", "allowed_hosts", "bindings"}
        closed(raw, fields, required)
        concierge_id = identifier(raw["concierge_id"])
        host = bounded_text(raw.get("host", "127.0.0.1"), 255)
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            raise ValueError("native PWA bind must be an IP literal") from None
        private_network = raw.get("private_network", False)
        cert, key = raw.get("tls_cert"), raw.get("tls_key")
        if bool(cert) != bool(key):
            raise ValueError("native PWA TLS certificate and key must be paired")
        if not address.is_loopback and (not private_network or not cert):
            raise ValueError("non-loopback native PWA requires private network and TLS")
        if cert:
            cert, key = bounded_text(cert, 4096), bounded_text(key, 4096)
        hosts = raw["allowed_hosts"]
        if not isinstance(hosts, list) or not 1 <= len(hosts) <= 16:
            raise ValueError("native PWA requires exact allowed hosts")
        for value in hosts:
            bounded_text(value, 255)
            if value != value.lower() or any(c in value for c in "*/?#@\\ \t\r\n"):
                raise ValueError("native PWA requires exact allowed hosts")
            if not urlsplit("https://" + value).hostname:
                raise ValueError("invalid native PWA host")
        token_env = bounded_text(raw.get("token_env", cls.token_env), 128)
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", token_env):
            raise ValueError("invalid native PWA token environment name")
        limits = {"port": (0, 65535), "max_body_bytes": (1024, 1048576),
                  "max_response_bytes": (262144, 8388608), "max_requests": (1, 128),
                  "request_timeout": (1, 60), "cursor_ttl": (30, 3600)}
        numbers = {}
        for name, (lower, upper) in limits.items():
            value = raw.get(name, getattr(cls, name))
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError("invalid native PWA numeric limit")
            numbers[name] = value
        owners = raw["bindings"]
        if not isinstance(owners, list) or not 1 <= len(owners) <= 100:
            raise ValueError("native PWA requires explicit owner bindings")
        bindings, principals, identities = [], set(), set()
        for owner in owners:
            closed(owner, {"issuer", "subject", "sources", "default_source_id"},
                   {"issuer", "subject", "sources", "default_source_id"})
            principal, _ = parse_principal({"issuer": owner["issuer"], "subject": owner["subject"], "concierge_id": concierge_id})
            if principal in principals:
                raise ValueError("duplicate native PWA principal")
            principals.add(principal)
            sources, source_ids = [], set()
            if not isinstance(owner["sources"], list) or not 1 <= len(owner["sources"]) <= 16:
                raise ValueError("invalid native PWA source bindings")
            for row in owner["sources"]:
                allowed = {"source_id", "platform", "chat_id", "chat_type", "user_id", "thread_id", "scope_id", "session_ids"}
                closed(row, allowed, {"source_id", "platform", "chat_id", "chat_type", "user_id"})
                from gateway.config import Platform
                try:
                    platform = Platform(row["platform"])
                except (TypeError, ValueError):
                    raise ValueError("invalid native PWA platform") from None
                source_id = identifier(row["source_id"])
                if source_id in source_ids or row["chat_type"] not in {"dm", "group", "channel", "thread", "forum"}:
                    raise ValueError("invalid native PWA source identity")
                source_ids.add(source_id)
                source = SessionSource(platform=platform, chat_id=bounded_text(row["chat_id"]),
                    chat_type=row["chat_type"], user_id=bounded_text(row["user_id"]),
                    thread_id=bounded_text(row["thread_id"]) if row.get("thread_id") is not None else None,
                    scope_id=bounded_text(row["scope_id"]) if row.get("scope_id") is not None else None)
                ids = row.get("session_ids", [])
                if not isinstance(ids, list) or len(ids) > 1000:
                    raise ValueError("invalid native PWA explicit sessions")
                entry = SourceBinding(source_id, source, tuple(identifier(i) for i in ids))
                if entry.identity_json in identities:
                    raise ValueError("native source cannot be assigned to multiple owners")
                identities.add(entry.identity_json)
                sources.append(entry)
            default = identifier(owner["default_source_id"])
            if default not in source_ids:
                raise ValueError("native PWA default source is not assigned")
            bindings.append(OwnerBinding(principal, tuple(sources), default))
        return cls(enabled=True, concierge_id=concierge_id, host=host, allowed_hosts=tuple(hosts),
                   private_network=private_network, tls_cert=cert, tls_key=key,
                   token_env=token_env, bindings=tuple(bindings), **numbers)

    @property
    def fingerprint(self):
        data = [[b.principal.issuer, b.principal.subject, b.default_source_id,
                 [[s.source_id, s.identity_json, list(s.session_ids)] for s in b.sources]] for b in self.bindings]
        return hashlib.sha256(canonical([self.concierge_id, data]).encode()).hexdigest()
