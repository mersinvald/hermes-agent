"""Native-owned principal assignments and bounded display projection."""
from __future__ import annotations

import asyncio
import json
import math
from dataclasses import replace
from datetime import datetime, timezone
from uuid import uuid4

from gateway.conversation_control import ConversationGrant, Principal
from gateway.pwa_config import canonical, identifier, source_identity


def timestamp(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        return None
    try:
        return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")
    except (OverflowError, ValueError, OSError):
        return None


def object_json(raw):
    if raw is None:
        return {}
    if len(raw) > 16384:
        raise LookupError("native metadata unavailable")
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        raise LookupError("native metadata unavailable") from None
    if not isinstance(value, dict):
        raise LookupError("native metadata unavailable")
    return value


class NativePwaOwnership:
    def __init__(self, runner, db, config, *, secret):
        self.runner, self.db, self.config = runner, db, config
        self._secret = secret

    def binding(self, principal):
        for binding in self.config.bindings:
            if binding.principal == principal:
                return binding
        raise PermissionError("principal unavailable")

    def scope(self, principal):
        self.binding(principal)
        return self.config.concierge_id, principal.issuer, principal.subject

    def trusted_channel_sources(self):
        return tuple(replace(source.source) for binding in self.config.bindings for source in binding.sources)

    def _row(self, session_id):
        identifier(session_id)
        row = self.db.native_pwa_row(session_id)
        if row is None:
            raise LookupError("native conversation unavailable")
        row["config"] = object_json(row["model_config"])
        row["origin"] = object_json(row["origin_json"])
        return row

    @staticmethod
    def _delegate(row):
        return row["source"] == "tool" or row["config"].get("_delegate_from") is not None

    @staticmethod
    def _branch(row):
        marker = row["config"].get("_branched_from")
        return marker is not None and (not row["parent_session_id"] or marker == row["parent_session_id"])

    def resolve_rows(self, session_id):
        row = self._row(session_id)
        if self._delegate(row):
            raise LookupError("subordinate conversation unavailable")
        seen = {row["id"]}
        root = row
        while root["parent_session_id"] and not self._branch(root):
            parent = self._row(root["parent_session_id"])
            if parent["end_reason"] != "compression":
                break
            if parent["id"] in seen or len(seen) >= 512 or self._delegate(parent):
                raise LookupError("native lineage unavailable")
            seen.add(parent["id"])
            root = parent
        rows, seen = [root], {root["id"]}
        while rows[-1]["end_reason"] == "compression":
            children = self.db.native_pwa_children(rows[-1]["id"])
            if len(children) > 512:
                raise LookupError("native lineage unavailable")
            candidates = []
            for child in children:
                child["config"] = object_json(child["model_config"])
                child["origin"] = object_json(child["origin_json"])
                if not self._branch(child) and not self._delegate(child):
                    candidates.append(child)
            if not candidates:
                break
            # Match the existing native first-continuation projection; competing
            # continuation children are ambiguous rather than silently hidden.
            if len(candidates) > 1:
                raise LookupError("native lineage unavailable")
            child = candidates[0]
            if child["id"] in seen or len(rows) >= 512:
                raise LookupError("native lineage unavailable")
            seen.add(child["id"])
            rows.append(child)
        if session_id not in seen:
            raise LookupError("native lineage unavailable")
        return rows

    def resolve(self, session_id):
        rows = self.resolve_rows(session_id)
        return {"conversation_id": rows[0]["id"], "native_root_session_id": rows[0]["id"],
                "native_session_id": rows[-1]["id"], "native_session_ids": [row["id"] for row in rows]}

    def _matches(self, row, source, *, explicit=False):
        values = {"source": source.platform.value, "user_id": source.user_id,
                  "chat_id": source.chat_id, "chat_type": source.chat_type, "thread_id": source.thread_id}
        for key, expected in values.items():
            actual = row[key]
            if actual != expected and not (explicit and actual is None):
                return False
        # Profile and extended source scope must never be borrowed from another
        # native profile, nor inferred from the facade's request context.
        if row["profile_name"] not in (None, self.db._own_profile_name()):
            return False
        origin = row["origin"]
        for key, expected in source_identity(source).items():
            if key in origin and origin[key] != expected:
                return False
        if source.scope_id is not None and origin.get("scope_id", origin.get("guild_id")) != source.scope_id:
            return False
        if origin.get("profile") is not None:
            return False
        return True

    def authorize(self, principal, session_id):
        binding = self.binding(principal)
        try:
            rows = self.resolve_rows(session_id)
            root = rows[0]["id"]
            assignment = self.db.native_pwa_assignment(self.scope(principal), root=root)
            for source in binding.sources:
                if not self.runner._is_user_authorized_for_source(source.source):
                    continue
                explicit = False
                for explicit_id in source.session_ids:
                    try:
                        if self.resolve(explicit_id)["conversation_id"] == root:
                            explicit = True
                            break
                    except (LookupError, ValueError):
                        continue
                created = bool(assignment and assignment["assignment_kind"] == "created"
                               and assignment["source_json"] == source.identity_json)
                if all(self._matches(row, source.source, explicit=explicit) for row in rows):
                    # A persisted explicit grant must still be configured; a
                    # created root's source must still match current config.
                    if assignment and assignment["assignment_kind"] == "created" and not created:
                        continue
                    kind = "created" if created else "explicit" if explicit else "discovered"
                    self.db.native_pwa_assign(self.scope(principal), root, source.identity_json, kind)
                    projection = {"conversation_id": root, "native_root_session_id": root,
                                  "native_session_id": rows[-1]["id"], "native_session_ids": [r["id"] for r in rows]}
                    return projection, ConversationGrant(root, replace(source.source))
        except (LookupError, ValueError):
            pass
        raise PermissionError("native conversation unavailable")

    def event_authorized(self, source, root):
        identity = canonical(source_identity(source))
        for binding in self.config.bindings:
            if any(s.identity_json == identity for s in binding.sources):
                try:
                    self.authorize(binding.principal, root)
                    return True
                except PermissionError:
                    return False
        return False

    def recovery_grants(self):
        scopes = {canonical(list(self.scope(b.principal))): b.principal for b in self.config.bindings}
        for scope, root in self.db.native_pwa_recovery_roots(scopes):
            try:
                yield self.authorize(scopes[scope], root)[1]
            except PermissionError:
                continue

    def discover(self, principal, after, limit):
        binding = self.binding(principal)
        roots, reasons, last, more = [], set(), after, False
        candidates = self.db.native_pwa_candidates(after)
        for index, row in enumerate(candidates):
            last = row["id"]
            potential = any(row["source"] == s.source.platform.value
                and row["user_id"] in (None, s.source.user_id)
                and row["chat_id"] in (None, s.source.chat_id) for s in binding.sources)
            explicit = any(row["id"] in s.session_ids for s in binding.sources)
            if potential or explicit:
                try:
                    projection, _ = self.authorize(principal, row["id"])
                    if projection["conversation_id"] == row["id"]:
                        roots.append(row["id"])
                except PermissionError:
                    try:
                        parsed = self._row(row["id"])
                        if not self._delegate(parsed):
                            reasons.add("legacy_ownership_unproven" if row["user_id"] is None or row["chat_id"] is None else "lineage_unproven")
                    except (LookupError, ValueError):
                        reasons.add("lineage_unproven")
            if len(roots) >= limit:
                more = index + 1 < len(candidates) or len(candidates) == 500
                break
        else:
            more = len(candidates) == 500
            if more:
                reasons.add("scan_limit")
        return roots, last if more else None, {"state": "partial" if reasons else "source_scoped", "reasons": sorted(reasons)}

    def safe_text(self, value, limit):
        if not isinstance(value, str) or len(value) > 65536:
            return None
        try:
            from agent.redact import redact_sensitive_text
            value = value.replace(self._secret, "[redacted]")
            return redact_sensitive_text(value, force=True, redact_url_credentials=True)[:limit]
        except Exception:
            return None

    async def native_conversation(self, ingress, principal, session_id):
        projection, _ = await asyncio.to_thread(self.authorize, principal, session_id)
        first = await asyncio.to_thread(self._row, projection["native_root_session_id"])
        last = await asyncio.to_thread(self._row, projection["native_session_id"])
        active = ingress._execution_view(ingress._owner(projection["conversation_id"]))
        if active is not None and active["conversation_id"] != projection["conversation_id"]:
            raise RuntimeError("native execution projection mismatch")
        title = first["title"] or last["title"]
        safe_title = self.safe_text(title, 300)
        final_projection, _ = await asyncio.to_thread(self.authorize, principal, session_id)
        if final_projection != projection:
            from hermes_state_commands import CommandConflict
            raise CommandConflict("native conversation lineage changed")
        return {"schema_version": "1.0", **projection, "native_title": safe_title,
                "native_title_truncated": safe_title is not None and len(title) > 300,
                "native_created_at": timestamp(first["started_at"]),
                "native_updated_at": timestamp(last["last_activity_at"] or last["started_at"]),
                "active_execution": active}

    async def create(self, ingress, principal, create_id):
        binding = self.binding(principal)
        source = binding.default_source
        if not self.runner._is_user_authorized_for_source(source.source):
            raise PermissionError("native source unavailable")
        old = await asyncio.to_thread(self.db.native_pwa_assignment, self.scope(principal), create_id=create_id)
        if old is not None:
            projection, grant = await asyncio.to_thread(self.authorize, principal, old["conversation_id"])
            root, created = projection["conversation_id"], False
        else:
            root = str(uuid4())
            routed = replace(source.source, native_conversation_route=root)
            key = self.runner.session_store._generate_session_key(routed)
            root, created = await asyncio.to_thread(self.db.native_pwa_create, self.scope(principal), create_id,
                root, source.identity_json, key, canonical(source.source.to_dict()))
            projection, grant = await asyncio.to_thread(self.authorize, principal, root)
        await self.runner.async_session_store.bind_conversation_alias(
            replace(grant.source, native_conversation_route=root), projection["native_session_id"])
        return await self.native_conversation(ingress, principal, root), created

    def message(self, row):
        role = row["role"]
        if role not in {"user", "assistant", "tool", "system"}:
            raise LookupError("unsupported native message role")
        content, reason = row["content"], None
        if role == "system":
            content, reason = None, "system_content"
        elif row["content_length"] > 65536:
            content, reason = None, "oversized"
        else:
            try:
                content = self.db._decode_content(content)
                if role == "user":
                    from agent.context_compressor import split_user_originated_turn
                    handoff, view = split_user_originated_turn({"role": role, "content": content,
                        "display_kind": row["display_kind"], "display_metadata": object_json(row["display_metadata"])})
                    if handoff is not None or row["display_kind"]:
                        content = view.get("content") if view else None
                if not isinstance(content, str):
                    content, reason = None, "unsupported_content"
                else:
                    content = self.safe_text(content, 65536)
                    if content is None:
                        reason = "redaction_unavailable"
            except (TypeError, ValueError, LookupError):
                content, reason = None, "unsupported_content"
        return {"native_session_id": row["session_id"], "message_id": row["id"], "role": role,
                "content": content, "content_state": "omitted" if reason else "available",
                "omission_reason": reason, "tool_name": self.safe_text(row["tool_name"], 256),
                "created_at": timestamp(row["timestamp"])}
