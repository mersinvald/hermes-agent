"""Strict opt-in settings for the private, single-profile native PWA listener."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from gateway.conversation_control import Principal
from gateway.session import SessionSource
from hermes_state_events import EventLimits
from gateway.pwa_images import ImageLimits
from gateway.pwa_workload import PwaWorkloadConfig

IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")
MODEL_CAPABILITY_STATES = frozenset({"supported", "unsupported", "unknown"})


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
    inspector_admin: bool = False
    observation_actor: str | None = None

    @property
    def default_source(self):
        return next(source for source in self.sources if source.source_id == self.default_source_id)


@dataclass(frozen=True)
class PwaModelCapabilities:
    text_input: str
    image_input: str
    tools: str
    reasoning_controls: str

    def to_wire(self):
        return {
            "text_input": self.text_input,
            "image_input": self.image_input,
            "tools": self.tools,
            "reasoning_controls": self.reasoning_controls,
        }


@dataclass(frozen=True)
class PwaModelRoute:
    model_id: str
    display_name: str
    provider: str
    model: str
    capabilities: PwaModelCapabilities

    @property
    def identity(self):
        return [
            self.model_id,
            self.display_name,
            self.provider,
            self.model,
            self.capabilities.to_wire(),
        ]


@dataclass(frozen=True)
class PwaModelCatalogConfig:
    default_model_id: str
    entries: tuple[PwaModelRoute, ...]

    @classmethod
    def from_dict(cls, raw):
        closed(raw, {"default_model_id", "entries"}, {"default_model_id", "entries"})
        default_model_id = identifier(raw["default_model_id"])
        rows = raw["entries"]
        if not isinstance(rows, list) or not 1 <= len(rows) <= 100:
            raise ValueError("native PWA model catalog requires one to 100 entries")
        entries, model_ids = [], set()
        capability_keys = {
            "text_input", "image_input", "tools", "reasoning_controls",
        }
        for row in rows:
            closed(
                row,
                {"model_id", "display_name", "provider", "model", "capabilities"},
                {"model_id", "display_name", "provider", "model", "capabilities"},
            )
            model_id = identifier(row["model_id"])
            if len(model_id) > 200:
                raise ValueError("native PWA model id is too long")
            if model_id in model_ids:
                raise ValueError("duplicate native PWA model id")
            model_ids.add(model_id)
            capabilities = closed(
                row["capabilities"], capability_keys, capability_keys,
            )
            if any(
                not isinstance(value, str) or value not in MODEL_CAPABILITY_STATES
                for value in capabilities.values()
            ):
                raise ValueError("invalid native PWA model capability state")
            provider = identifier(row["provider"])
            if provider.lower() == "auto":
                raise ValueError("native PWA models require an explicit provider")
            entries.append(PwaModelRoute(
                model_id=model_id,
                display_name=bounded_text(row["display_name"], 120),
                provider=provider,
                model=bounded_text(row["model"], 512),
                capabilities=PwaModelCapabilities(**capabilities),
            ))
        if default_model_id not in model_ids:
            raise ValueError("native PWA default model is not in the catalog")
        return cls(default_model_id, tuple(entries))

    @property
    def identity(self):
        return [self.default_model_id, [entry.identity for entry in self.entries]]

    def route(self, model_id):
        return next((entry for entry in self.entries if entry.model_id == model_id), None)


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
    event_limits: EventLimits = field(default_factory=EventLimits)
    event_poll_interval: int = 1
    stream_write_timeout: int = 5
    stream_max_seconds: int = 30
    models: PwaModelCatalogConfig | None = None
    images: ImageLimits = field(default_factory=ImageLimits)
    workload_context: PwaWorkloadConfig | None = None
    inspector_model_names: tuple[str, ...] = ()
    inspector_provider_names: tuple[str, ...] = ()

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
                raise ValueError(
                    "disabled native PWA settings must contain only enabled"
                )
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
        limits = {
            "port": (0, 65535),
            "max_body_bytes": (1024, 1048576),
            "max_response_bytes": (262144, 8388608),
            "max_requests": (1, 128),
            "request_timeout": (1, 60),
            "cursor_ttl": (30, 3600),
            "event_poll_interval": (1, 10),
            "stream_write_timeout": (1, 30),
            "stream_max_seconds": (1, 300),
        }
        numbers = {}
        for name, (lower, upper) in limits.items():
            value = raw.get(name, getattr(cls, name))
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError("invalid native PWA numeric limit")
            numbers[name] = value
        event_raw = raw.get("event_limits", {})
        event_caps = {
            "max_count": 65536,
            "max_bytes": 67108864,
            "max_age_seconds": 604800,
            "max_event_bytes": 65536,
            "batch_count": 100,
            "snapshot_count": 100,
            "max_subscribers": 128,
        }
        closed(event_raw, event_caps)
        if any(
            type(v) is not int or not 1 <= v <= event_caps[k]
            for k, v in event_raw.items()
        ):
            raise ValueError("invalid native event limit")
        event_limits = EventLimits(**event_raw)
        models = (
            PwaModelCatalogConfig.from_dict(raw["models"])
            if "models" in raw
            else None
        )
        observed = {}
        for field_name in ("inspector_model_names", "inspector_provider_names"):
            values = raw.get(field_name, [])
            if not isinstance(values, list) or len(values) > 100 or any(
                not isinstance(value, str) or (field_name == "inspector_provider_names" and len(value) > 128) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./:+-]{0,199}", value)
                for value in values
            ) or len(set(values)) != len(values):
                raise ValueError("invalid native inspector observed vocabulary")
            observed[field_name] = tuple(values)
        owners = raw["bindings"]
        if not isinstance(owners, list) or not 1 <= len(owners) <= 100:
            raise ValueError("native PWA requires explicit owner bindings")
        bindings, principals = [], set()
        for owner in owners:
            closed(
                owner,
                {"issuer", "subject", "sources", "default_source_id", "inspector_admin", "observation_actor"},
                {"issuer", "subject", "sources", "default_source_id"},
            )
            principal, _ = parse_principal({
                "issuer": owner["issuer"],
                "subject": owner["subject"],
                "concierge_id": concierge_id,
            })
            if principal in principals:
                raise ValueError("duplicate native PWA principal")
            principals.add(principal)
            sources, source_ids = [], set()
            if (
                not isinstance(owner["sources"], list)
                or not 1 <= len(owner["sources"]) <= 16
            ):
                raise ValueError("invalid native PWA source bindings")
            for row in owner["sources"]:
                allowed = {
                    "source_id",
                    "platform",
                    "chat_id",
                    "chat_type",
                    "user_id",
                    "thread_id",
                    "scope_id",
                    "session_ids",
                }
                closed(
                    row,
                    allowed,
                    {"source_id", "platform", "chat_id", "chat_type", "user_id"},
                )
                from gateway.config import Platform

                try:
                    platform = Platform(row["platform"])
                except (TypeError, ValueError):
                    raise ValueError("invalid native PWA platform") from None
                source_id = identifier(row["source_id"])
                if source_id in source_ids or row["chat_type"] not in {
                    "dm",
                    "group",
                    "channel",
                    "thread",
                    "forum",
                }:
                    raise ValueError("invalid native PWA source identity")
                source_ids.add(source_id)
                source = SessionSource(
                    platform=platform,
                    chat_id=bounded_text(row["chat_id"]),
                    chat_type=row["chat_type"],
                    user_id=bounded_text(row["user_id"]),
                    thread_id=bounded_text(row["thread_id"])
                    if row.get("thread_id") is not None
                    else None,
                    scope_id=bounded_text(row["scope_id"])
                    if row.get("scope_id") is not None
                    else None,
                )
                ids = row.get("session_ids", [])
                if not isinstance(ids, list) or len(ids) > 1000:
                    raise ValueError("invalid native PWA explicit sessions")
                entry = SourceBinding(
                    source_id, source, tuple(identifier(i) for i in ids)
                )
                sources.append(entry)
            default = identifier(owner["default_source_id"])
            if default not in source_ids:
                raise ValueError("native PWA default source is not assigned")
            admin = owner.get("inspector_admin", False)
            if type(admin) is not bool:
                raise ValueError("native inspector authority must be boolean")
            actor = owner.get("observation_actor")
            if actor is not None:
                identifier(actor)
                if len(actor) > 128 or any(b.observation_actor == actor for b in bindings):
                    raise ValueError("native observation actor must identify exactly one owner")
            bindings.append(OwnerBinding(principal, tuple(sources), default, admin, actor))
        return cls(
            enabled=True,
            concierge_id=concierge_id,
            host=host,
            allowed_hosts=tuple(hosts),
            private_network=private_network,
            tls_cert=cert,
            tls_key=key,
            token_env=token_env,
            bindings=tuple(bindings),
            event_limits=event_limits,
            models=models,
            images=ImageLimits.from_dict(raw.get("images", {})),
            workload_context=PwaWorkloadConfig.from_dict(raw.get("workload_context")),
            **observed,
            **numbers,
        )

    @property
    def fingerprint(self):
        data = [[b.principal.issuer, b.principal.subject, b.default_source_id, b.inspector_admin, b.observation_actor,
                 [[s.source_id, s.identity_json, list(s.session_ids)] for s in b.sources]] for b in self.bindings]
        models = self.models.identity if self.models is not None else None
        return hashlib.sha256(
            canonical([self.concierge_id, data, models, self.inspector_model_names, self.inspector_provider_names]).encode()
        ).hexdigest()
