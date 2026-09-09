"""Bounded literal display-history scan primitives for the private PWA service.

Neither a query nor a cursor grants authority. Callers authorize the current
principal and source before reading these handles and before publishing a page.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import secrets
import time
import unicodedata
from collections import OrderedDict

from hermes_state_commands import CommandConflict
from hermes_state_pwa_scan import history_candidates, history_snapshot
from gateway.pwa_config import canonical, identifier


def literal_query(value):
    """Return the canonical literal query; never interpret SQL/FTS operators."""
    if not isinstance(value, str) or len(value) > 1024:
        raise ValueError("invalid history query")
    # Validate before trimming so control-only prefixes cannot disguise input.
    if any(unicodedata.category(char) in {"Cc", "Cs"} for char in value):
        raise ValueError("invalid history query")
    query = value.strip()
    if not query or len(query) > 256 or len(query.encode("utf-8")) > 1024:
        raise ValueError("invalid history query")
    return query


def literal_snippet(text, query, *, limit=500):
    """Find a case-folded literal and return a window of actual source text.

    Case folding can expand a single source character, e.g. ß -> ss. Translating
    the folded match back to original code points avoids invented text/offsets.
    Inputs are already bounded, redacted display text and a validated query.
    """
    if not isinstance(text, str) or len(text) > 65536:
        raise ValueError("invalid history display text")
    if type(limit) is not int or not 256 <= limit <= 500:
        raise ValueError("invalid snippet limit")
    query = literal_query(query)
    folded = text.casefold()
    start = folded.find(query.casefold())
    if start < 0:
        return None
    end = start + len(query.casefold())
    folded_position = 0
    source_start = source_end = None
    partial_start = partial_end = False
    for index, char in enumerate(text):
        next_position = folded_position + len(char.casefold())
        if source_start is None and next_position > start:
            source_start = index
            partial_start = start > folded_position
        if next_position >= end:
            source_end = index + 1
            partial_end = end < next_position
            break
        folded_position = next_position
    if source_start is None or source_end is None:
        raise RuntimeError("history snippet mapping unavailable")
    truncated = source_end - source_start > limit
    if truncated:
        # A folded query can expand: 256 ligatures can match 768 source points.
        # Keep only whole source characters strictly within the match, including
        # when the folded match starts/ends inside a single character expansion.
        left = source_start + int(partial_start)
        right = min(source_end - int(partial_end), left + limit)
    else:
        left = max(0, source_start - (limit - (source_end - source_start)) // 2)
        right = min(len(text), left + limit)
        left = max(0, right - limit)
    return {
        "text": text[left:right],
        "prefix_omitted": left > 0,
        "suffix_omitted": right < len(text),
        "match_truncated": truncated,
    }


def _canonical(value):
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("ascii")


class ScanHandles:
    """Finite process-local immutable state; restart invalidates every handle.

    A keyed state index reuses an identical continuation within the same scope,
    keeping retries stable without storing private response payloads. Reads do
    not extend absolute expiry. Count and encoded-byte budgets bound memory.
    """

    def __init__(
        self,
        *,
        ttl=900,
        max_entries=4096,
        max_bytes=4 * 1024 * 1024,
        max_state_bytes=32768,
        clock=time.monotonic,
    ):
        if (
            type(ttl) not in (int, float)
            or not 1 <= ttl <= 3600
            or type(max_entries) is not int
            or not 1 <= max_entries <= 4096
            or type(max_bytes) is not int
            or not 1024 <= max_bytes <= 16 * 1024 * 1024
            or type(max_state_bytes) is not int
            or not 128 <= max_state_bytes <= max_bytes
        ):
            raise ValueError("invalid history handle limits")
        self.ttl, self.max_entries = ttl, max_entries
        self.max_bytes, self.max_state_bytes = max_bytes, max_state_bytes
        self.clock = clock
        self._key = secrets.token_bytes(32)

        self._entries = OrderedDict()
        self._identities = {}
        self._bytes = 0

    def _remove(self, token):
        entry = self._entries.pop(token)
        del self._identities[entry[3]]
        self._bytes -= len(entry[2])

    def _expire(self):
        now = self.clock()
        for token, entry in tuple(self._entries.items()):
            if entry[0] <= now:
                self._remove(token)

    def put(self, scope, state):
        scope_hash = hmac.digest(self._key, _canonical(scope), "sha256")
        encoded = _canonical(state)
        if len(encoded) > self.max_state_bytes:
            raise ValueError("history continuation unavailable")
        identity = hmac.digest(self._key, scope_hash + encoded, "sha256")
        self._expire()
        if identity in self._identities:
            return self._identities[identity]
        while self._entries and (
            len(self._entries) >= self.max_entries
            or self._bytes + len(encoded) > self.max_bytes
        ):
            self._remove(next(iter(self._entries)))
        token = secrets.token_hex(32)
        self._entries[token] = (self.clock() + self.ttl, scope_hash, encoded, identity)
        self._identities[identity] = token
        self._bytes += len(encoded)
        return token

    def get(self, token, scope):
        if not isinstance(token, str) or len(token) != 64:
            raise CommandConflict("invalid or expired history cursor")
        self._expire()
        entry = self._entries.get(token)
        scope_hash = hmac.digest(self._key, _canonical(scope), "sha256")
        if entry is None or not hmac.compare_digest(entry[1], scope_hash):
            raise CommandConflict("invalid or expired history cursor")
        self._entries.move_to_end(token)
        return json.loads(entry[2])

    def lineage_version(self, lineage, generation):
        return hmac.new(
            self._key, _canonical([lineage, generation]), hashlib.sha256
        ).hexdigest()

    def close(self):
        self._entries.clear()
        self._identities.clear()
        self._bytes = 0
        self._key = secrets.token_bytes(32)


class NativeHistoryScan:
    """A finite authorized sweep over native display history, never an FTS proxy."""

    def __init__(self, http):
        self.http = http
        self.handles = ScanHandles(ttl=http.config.cursor_ttl, max_state_bytes=262144)

    def _scope(self, principal):
        ownership = self.http.ownership
        binding = ownership.binding(principal)
        sources = tuple(
            source
            for source in binding.sources
            if self.http.runner._is_user_authorized_for_source(source.source)
        )
        if not sources:
            raise PermissionError("native sources unavailable")
        return [
            *ownership.scope(principal),
            ownership.config.fingerprint,
            [source.identity_json for source in sources],
        ], sources

    def _load(self, token, scope, kind):
        state = self.handles.get(token, scope)
        if state["kind"] != kind or state["expires"] <= self.handles.clock():
            raise CommandConflict("invalid or expired history operation")
        return state

    async def _authorize(self, principal, root):
        projection, _ = await asyncio.to_thread(
            self.http._authorize_root, principal, root
        )
        return projection

    def _candidate(self, principal, row, sources):
        potential = any(
            row["source"] == source.source.platform.value
            and row["user_id"] in (None, source.source.user_id)
            and row["chat_id"] in (None, source.source.chat_id)
            for source in sources
        )
        explicit = any(row["id"] in source.session_ids for source in sources)
        if not (potential or explicit):
            return None, None
        try:
            identifier(row["id"])
            projection, _ = self.http.ingress._authorize(principal, row["id"])
            if projection["conversation_id"] != row["id"]:
                return None, None
            return projection, None
        except (PermissionError, ValueError):
            try:
                parsed = self.http.ownership._row(row["id"])
                if self.http.ownership._delegate(parsed):
                    return None, None
            except (LookupError, ValueError):
                pass
            reason = (
                "legacy_ownership_unproven"
                if row["user_id"] is None or row["chat_id"] is None
                else "lineage_unproven"
            )
            return None, reason

    async def request(self, request, principal, kind):
        if kind == "sync" and "checkpoint" in request.query:
            query = self.http._query(request, ("checkpoint",))
            scope, _ = self._scope(principal)
            state = self._load(query["checkpoint"], scope, "checkpoint")
            if state["query"][3] is not None:
                await self._authorize(principal, state["query"][3])
            current = await asyncio.to_thread(history_snapshot, self.http.db)
            if self._scope(principal)[0] != scope:
                raise PermissionError("native source binding changed")
            return {
                "schema_version": "1.0",
                "kind": "native_sync_probe",
                "state": "unchanged"
                if current == state["snapshot"]
                else "refresh_required",
            }
        allowed = (
            ("limit", "cursor", "conversation_id", "q")
            if kind == "search"
            else ("limit", "cursor", "conversation_id")
        )
        query = self.http._query(request, allowed)
        text = literal_query(query.get("q")) if kind == "search" else None
        raw_limit = query.get("limit", "20" if kind == "search" else "50")
        if (
            not raw_limit.isascii()
            or not raw_limit.isdigit()
            or str(int(raw_limit)) != raw_limit
        ):
            raise ValueError("invalid history limit")
        limit = int(raw_limit)
        if not 1 <= limit <= (50 if kind == "search" else 100):
            raise ValueError("invalid history limit")
        root = (
            identifier(query["conversation_id"]) if "conversation_id" in query else None
        )
        scope, sources = self._scope(principal)
        expected = [kind, text, limit, root]
        snapshot = await asyncio.to_thread(history_snapshot, self.http.db)
        if "cursor" in query:
            state = self._load(query["cursor"], scope, "cursor")
            if (
                state["query"] != expected
                or snapshot["generation"] != state["snapshot"]["generation"]
            ):
                raise CommandConflict("native history changed")
        else:
            state = {
                "kind": "cursor",
                "query": expected,
                "snapshot": snapshot,
                "scan_id": secrets.token_urlsafe(32),
                "expires": self.handles.clock() + self.http.config.cursor_ttl,
                "after": "",
                "current": None,
                "reasons": [],
                "done": False,
            }
            if root:
                projection = await self._authorize(principal, root)
                state["current"] = self._position(projection)
        return await self._page(principal, scope, sources, state)

    @staticmethod
    def _position(projection):
        return {
            "root": projection["conversation_id"],
            "lineage": projection["native_session_ids"],
            "segment": 0,
            "after": 0,
            "include_title": True,
        }

    async def _page(self, principal, scope, sources, state):
        kind, query, limit, selected_root = state["query"]
        snapshot = state["snapshot"]
        reasons, groups, checked = set(state["reasons"]), [], {}
        total = units = candidates_seen = size = 0
        candidates, exhausted = [], False
        # Reserve framing/cursors and a small metadata expansion margin for the
        # facade. Oversized required headers fail rather than being skipped.
        budget = self.http.config.max_response_bytes - 8192
        while not state["done"] and total < limit and units < 256:
            if state["current"] is None:
                if selected_root:
                    state["done"] = True
                    break
                if len(groups) >= 32 or candidates_seen >= 500:
                    break
                if not candidates:
                    if exhausted:
                        state["done"] = True
                        break
                    candidates = await asyncio.to_thread(
                        history_candidates,
                        self.http.db,
                        state["after"],
                        snapshot["session_upper"],
                        500 - candidates_seen,
                    )
                    exhausted = len(candidates) < 500 - candidates_seen
                    if not candidates:
                        state["done"] = True
                        break
                row = candidates.pop(0)
                candidates_seen += 1
                projection, reason = await asyncio.to_thread(
                    self._candidate, principal, row, sources
                )
                if reason:
                    reasons.add(reason)
                if projection is None:
                    state["after"] = row["id"]
                    continue
                state["current"] = self._position(projection)
                state["after"] = row["id"]
            current = state["current"]
            projection = await self._authorize(principal, current["root"])
            if projection["native_session_ids"] != current["lineage"]:
                raise CommandConflict("native history lineage changed")
            conversation = await self.http.ingress.native_conversation(
                principal, current["root"]
            )
            if conversation["native_session_ids"] != current["lineage"]:
                raise CommandConflict("native history lineage changed")
            field = "matches" if kind == "search" else "messages"
            group = {
                "conversation": conversation,
                "lineage_version": self.handles.lineage_version(
                    current["lineage"], snapshot["generation"]
                ),
                field: [],
            }
            if kind == "search":
                group["include_title"] = current["include_title"]
            header_size = len(canonical(group).encode()) + 2
            if size + header_size + 1024 > budget:
                if not groups:
                    raise RuntimeError("native history header unavailable")
                break
            groups.append(group)
            checked[current["root"]] = current["lineage"]
            size += header_size
            if kind == "search" and current["include_title"]:
                total += 1
            current["include_title"] = False
            paused = False
            while (
                current["segment"] < len(current["lineage"])
                and total < limit
                and units < 256
            ):
                segment = current["lineage"][current["segment"]]
                rows = await asyncio.to_thread(
                    self.http.db.native_pwa_history_rows,
                    segment,
                    current["after"],
                    snapshot["message_upper"],
                    min(32, 256 - units),
                )
                if not rows:
                    units += 1
                    current["segment"] += 1
                    current["after"] = 0
                    continue
                for row in rows:
                    units += 1
                    if row["role"] == "system":
                        current["after"] = row["id"]
                        continue
                    try:
                        message = self.http.ownership.message(row)
                    except LookupError:
                        reasons.add("unsupported_role")
                        current["after"] = row["id"]
                        continue
                    omission = message["omission_reason"]
                    if omission:
                        reasons.add(
                            "oversized_content" if omission == "oversized" else omission
                        )
                    item = message
                    if kind == "search":
                        snippet = (
                            literal_snippet(message["content"], query)
                            if message["content"] is not None
                            else None
                        )
                        if snippet is None:
                            current["after"] = row["id"]
                            continue
                        item = {
                            "kind": "message",
                            "native_session_id": segment,
                            "message_id": row["id"],
                            "role": message["role"],
                            "created_at": message["created_at"],
                            "snippet": snippet,
                        }
                    item_size = len(canonical(item).encode()) + 2
                    if kind == "sync" and item_size + header_size > budget:
                        item = {
                            **message,
                            "content": None,
                            "content_state": "omitted",
                            "omission_reason": "oversized",
                        }
                        reasons.add("oversized_content")
                        item_size = len(canonical(item).encode()) + 2
                    if size + item_size > budget:
                        paused = True
                        break
                    group[field].append(item)
                    size += item_size
                    total += 1
                    current["after"] = row["id"]
                    if total >= limit:
                        break
                if paused:
                    break
            if current["segment"] >= len(current["lineage"]):
                state["current"] = None
            else:
                break
        if selected_root and state["current"] is None:
            state["done"] = True
        for root, lineage in checked.items():
            if (await self._authorize(principal, root))[
                "native_session_ids"
            ] != lineage:
                raise CommandConflict("native history lineage changed")
        final = await asyncio.to_thread(history_snapshot, self.http.db)
        if final["generation"] != snapshot["generation"]:
            raise CommandConflict("native history changed")
        if self._scope(principal)[0] != scope:
            raise PermissionError("native source binding changed")
        if state["expires"] <= self.handles.clock():
            raise CommandConflict("native history operation expired")
        state["reasons"] = sorted(reasons)
        coverage = {
            "state": "partial" if reasons else "complete",
            "scope": "retained_display_text",
            "reasons": sorted(reasons),
        }
        cursor = checkpoint = None
        if state["done"]:
            state["kind"] = "checkpoint"
            checkpoint = self.handles.put(scope, state)
        else:
            coverage["state"] = "in_progress"
            cursor = self.handles.put(scope, state)
        return {
            "schema_version": "1.0",
            "kind": "native_" + kind,
            "scan_id": state["scan_id"],
            "groups": groups,
            "coverage": coverage,
            "next_cursor": cursor,
            "checkpoint": checkpoint,
        }
