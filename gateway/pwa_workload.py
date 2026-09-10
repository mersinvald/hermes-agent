"""Optional, owner-bound Kubernetes identity; no discovery or home-directory reads."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat


DIRECTORY = Path("/run/hermes-pwa-workload")
FILES = {"namespace": "namespace", "pod_name": "pod-name", "pod_uid": "pod-uid"}
LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
POD = re.compile(rf"{LABEL}(?:\.{LABEL})*\Z")
UID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")


def digest(domain, value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256((domain + "\n" + data).encode("utf-8")).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class PwaWorkloadConfig:
    source: str = "kubernetes_downward_api_v1"

    @classmethod
    def from_dict(cls, raw):
        if raw is None:
            return None
        if type(raw) is not dict or raw != {"source": "kubernetes_downward_api_v1"}:
            raise ValueError("native workload context requires the closed Kubernetes source")
        return cls()


class WorkloadUnavailable(Exception):
    """The startup context is absent, ambiguous, changed, or no longer trustworthy."""


def read_identity():
    result = {}
    for key, filename in FILES.items():
        # Kubernetes projected volumes use symlinks. Follow only these fixed paths;
        # reject devices/FIFOs and bound the read independently of reported size.
        fd = os.open(DIRECTORY / filename, os.O_RDONLY | os.O_NONBLOCK)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > 254:
                raise WorkloadUnavailable()
            data = os.read(fd, 255)
        finally:
            os.close(fd)
        value = data.removesuffix(b"\n").decode("ascii")
        if not 1 <= len(value) <= 253:
            raise WorkloadUnavailable()
        result[key] = value
    if (not re.fullmatch(LABEL, result["namespace"])
            or not POD.fullmatch(result["pod_name"])
            or not UID.fullmatch(result["pod_uid"])):
        raise WorkloadUnavailable()
    return result


class NativeWorkloadContext:
    def __init__(self, ownership, config):
        self.ownership = ownership
        self.invalid = config is None
        self.identity = self.policy = self.owner = self.revision = None
        if not self.invalid:
            try:
                self.identity = read_identity()
                self.policy, self.owner = self._ownership_state()
            except Exception:
                # This optional capability must not prevent native execution.
                self.invalid = True
        # A conservative ownership epoch, never a claim about pod creation time.
        self.started = utc_now()
        if not self.invalid:
            self.revision = digest("native-workload-context-v1", [
                secrets.token_hex(32), self.started, self.identity, self.policy,
            ])
        self.last_observed = self.started

    def _ownership_state(self):
        effective, states = set(), []
        config = self.ownership.config
        for binding in config.bindings:
            for source in binding.sources:
                allowed = self.ownership.runner._is_user_authorized_for_source(source.source)
                if type(allowed) is not bool:
                    raise WorkloadUnavailable()
                states.append([binding.principal.issuer, binding.principal.subject,
                               source.source_id, source.identity_json, allowed])
                if allowed:
                    effective.add(binding.principal)
        if len(effective) != 1:
            raise WorkloadUnavailable()
        return digest("native-workload-ownership-v1", [config.fingerprint, states]), effective.pop()

    def _check(self, principal):
        if self.invalid:
            raise WorkloadUnavailable()
        try:
            policy, owner = self._ownership_state()
            if policy != self.policy or owner != self.owner or read_identity() != self.identity:
                raise WorkloadUnavailable()
        except Exception:
            # Restoring an old binding/identity does not resurrect its old epoch.
            self.invalid = True
            raise WorkloadUnavailable() from None
        if principal != self.owner:
            raise PermissionError("workload owner unavailable")

    def snapshot(self, principal):
        self._check(principal)
        observed = utc_now()
        if datetime.fromisoformat(observed) < datetime.fromisoformat(self.last_observed):
            self.invalid = True
            raise WorkloadUnavailable()
        self.last_observed = observed
        return {
            "schema_version": "1.0", **self.identity,
            "owner_scope_hash": digest("native-workload-owner-v1", [
                self.ownership.config.concierge_id, principal.issuer, principal.subject,
            ]),
            "context_revision": self.revision,
            "epoch_started_at": self.started,
            "observed_at": observed,
        }

    def validate(self, principal, result):
        self._check(principal)
        if result.get("context_revision") != self.revision:
            raise WorkloadUnavailable()
