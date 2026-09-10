"""Bounded native ownership/history queries on the existing SessionDB handle."""
from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager

from hermes_state_commands import CommandConflict

ROW_COLUMNS = """id, source, user_id, chat_id, chat_type, thread_id, profile_name,
    parent_session_id, end_reason, started_at, last_activity_at,
    substr(title,1,4097) AS title, substr(origin_json,1,16385) AS origin_json,
    substr(model_config,1,16385) AS model_config"""

# A native image caption can contain 100,000 Unicode characters. The native
# JSON encoder escapes each astral character as 12 ASCII bytes. Allow that
# bounded envelope plus the ten frozen descriptors, but extract only public
# display fields inside SQLite; never return private representations/pixels.
MAX_IMAGE_DISPLAY_SOURCE_BYTES = 2 * 1024 * 1024
IMAGE_DISPLAY_SQL = f"""CASE WHEN role='user'
    AND length(CAST(display_metadata AS BLOB))<={MAX_IMAGE_DISPLAY_SOURCE_BYTES}
    AND json_valid(display_metadata) THEN CASE
    WHEN json_type(display_metadata,'$.pwa_images')='object'
        AND json_type(display_metadata,'$.pwa_images.text')='text'
        AND json_type(display_metadata,'$.pwa_images.image_ids')='array'
        AND json_array_length(display_metadata,'$.pwa_images.image_ids') BETWEEN 1 AND 10
        AND length(CAST(json_extract(display_metadata,'$.pwa_images.image_ids') AS BLOB))<=1100
        AND length(CAST(json_extract(display_metadata,'$.pwa_images.text') AS BLOB))<=409600
    THEN CASE WHEN pwa_text_length(json_extract(display_metadata,'$.pwa_images.text'))<=100000
        THEN json_object('image_ids',json_extract(display_metadata,'$.pwa_images.image_ids'),
            'text_length',pwa_text_length(json_extract(display_metadata,'$.pwa_images.text')),
            'text',CASE WHEN pwa_text_length(json_extract(display_metadata,'$.pwa_images.text'))<=65536
                THEN json_extract(display_metadata,'$.pwa_images.text') END) END END END"""


@contextmanager
def bounded_read(db):
    with db._read_ctx() as conn:
        deadline = time.monotonic() + 5
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
        try:
            yield conn
        finally:
            conn.set_progress_handler(None, 0)


def _display_key(content, kind, metadata):
    # SQLite passes only bounded substrings. Large/unknown carrier rows are
    # explicitly omitted by the display projection rather than decoded here.
    if content is None:
        return None
    try:
        from agent.context_compressor import split_user_originated_turn
        decoded = json.loads(content) if content.startswith(("[", '"', "{")) else content
        handoff, view = split_user_originated_turn({"role": "user", "content": decoded,
            "display_kind": kind, "display_metadata": json.loads(metadata) if metadata else None})
        if handoff is not None and view is not None:
            decoded = view.get("content")
        return hashlib.sha256(json.dumps(decoded, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    except (ValueError, TypeError):
        return hashlib.sha256(content.encode()).hexdigest()


class NativePwaStateMixin:
    def native_pwa_row(self, session_id):
        with bounded_read(self) as conn:
            row = conn.execute(f"SELECT {ROW_COLUMNS} FROM sessions WHERE id=?", (session_id,)).fetchone()
            return dict(row) if row else None

    def native_pwa_children(self, parent):
        with bounded_read(self) as conn:
            return [dict(r) for r in conn.execute(
                f"SELECT {ROW_COLUMNS} FROM sessions WHERE parent_session_id=? ORDER BY started_at,id LIMIT 513", (parent,))]

    def native_pwa_candidates(self, after, limit=500):
        with bounded_read(self) as conn:
            return [dict(r) for r in conn.execute(
                f"SELECT {ROW_COLUMNS} FROM sessions WHERE id>? ORDER BY id LIMIT ?", (after, limit))]

    def native_pwa_assignment(self, scope, root=None, create_id=None):
        with bounded_read(self) as conn:
            key, value = ("conversation_id", root) if root is not None else ("create_id", create_id)
            row = conn.execute(f"SELECT * FROM native_pwa_conversations WHERE concierge_id=? AND issuer=? AND subject=? AND {key}=?", (*scope, value)).fetchone()
            return dict(row) if row else None

    def native_pwa_assign(self, scope, root, source_json, kind):
        def write(conn):
            conn.execute("""INSERT INTO native_pwa_conversations
                (concierge_id,issuer,subject,conversation_id,source_json,assignment_kind,created_at)
                VALUES (?,?,?,?,?,?,?) ON CONFLICT(concierge_id,issuer,subject,conversation_id)
                DO UPDATE SET source_json=excluded.source_json,
                assignment_kind=CASE WHEN native_pwa_conversations.create_id IS NULL THEN excluded.assignment_kind ELSE 'created' END""",
                (*scope, root, source_json, kind, time.time()))
        self._execute_write(write)

    def native_pwa_create(self, scope, create_id, root, source_json, session_key, origin_json):
        source = json.loads(source_json)
        def write(conn):
            old = conn.execute("SELECT * FROM native_pwa_conversations WHERE concierge_id=? AND issuer=? AND subject=? AND create_id=?", (*scope, create_id)).fetchone()
            if old is not None:
                if old["source_json"] != source_json:
                    raise CommandConflict("native PWA creation binding changed")
                return old["conversation_id"], False
            conn.execute("""INSERT INTO sessions
                (id,source,user_id,session_key,chat_id,chat_type,thread_id,origin_json,profile_name,started_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)""", (root, source["platform"], source["user_id"], session_key,
                source["chat_id"], source["chat_type"], source["thread_id"], origin_json, self._own_profile_name(), time.time()))
            conn.execute("""INSERT INTO native_pwa_conversations
                (concierge_id,issuer,subject,conversation_id,source_json,assignment_kind,create_id,created_at)
                VALUES (?,?,?,?,?,'created',?,?)""", (*scope, root, source_json, create_id, time.time()))
            return root, True
        return self._execute_write(write)

    def native_pwa_recovery_roots(self, scopes):
        # Recovery enumerates only native journal work, not the complete archive.
        with bounded_read(self) as conn:
            for scope in scopes:
                cursor = conn.execute("SELECT DISTINCT conversation_id FROM native_commands WHERE scope=? AND phase IN ('queued','steer','assigned') ORDER BY conversation_id", (scope,))
                while rows := cursor.fetchmany(128):
                    for row in rows:
                        yield scope, row[0]

    def native_pwa_history_signature(self, lineage):
        with bounded_read(self) as conn:
            fence = conn.execute("SELECT generation FROM native_pwa_history_fence WHERE singleton=1").fetchone()
            if fence is None:
                raise RuntimeError("native history fence unavailable")
            values = [tuple(conn.execute("""SELECT COALESCE(MAX(id),0),COUNT(*),COALESCE(SUM(active),0),COALESCE(SUM(compacted),0)
                FROM messages WHERE session_id=?""", (segment,)).fetchone()) for segment in lineage]
        signature = hashlib.sha256(json.dumps([lineage, values, fence[0]]).encode()).hexdigest()
        return signature, [v[0] for v in values]

    def native_pwa_history_rows(self, segment, after, upper, limit):
        # Deduplication runs inside SQLite (which can spill sort state), never by
        # loading all retained payloads into a Python list as get_messages does.
        # Oversized user carriers retain exact raw-content equality; only small
        # structured carriers enter the native canonical display normalizer.
        # session_meta is a native bookkeeping row appended at turn completion,
        # not a transcript message; never expose it or reject the whole page.
        with bounded_read(self) as conn:
            conn.create_function("pwa_display_key", 3, _display_key, deterministic=True)
            # SQL length(TEXT) stops at NUL. Only the extracted caption, already
            # capped at 409,600 UTF-8 bytes above, enters this exact character
            # count; the arbitrary metadata envelope never enters Python.
            conn.create_function("pwa_text_length", 1, len, deterministic=True)
            try:
                rows = conn.execute(f"""WITH ranked AS (
                    SELECT id,session_id,role,content,tool_name,timestamp,display_kind,display_metadata,
                        ROW_NUMBER() OVER (PARTITION BY role,
                            CASE WHEN role='user' AND length(CAST(content AS BLOB))<=65536
                                AND COALESCE(length(CAST(display_metadata AS BLOB)),0)<=16384
                                THEN pwa_display_key(content,display_kind,display_metadata) ELSE content END,
                            timestamp,tool_call_id,tool_calls,tool_name
                            ORDER BY active DESC,id DESC) AS rank
                    FROM messages WHERE session_id=? AND id<=? AND (active=1 OR compacted=1)
                        AND role!='session_meta'
                ) SELECT id,session_id,role,
                    CASE WHEN length(CAST(content AS BLOB))<=65536 THEN content END AS content,
                    COALESCE(length(CAST(content AS BLOB)),0) AS content_length,
                    substr(tool_name,1,4097) AS tool_name,timestamp,display_kind,
                    CASE WHEN ({IMAGE_DISPLAY_SQL}) IS NULL AND length(CAST(display_metadata AS BLOB))<=16384
                        THEN display_metadata END AS display_metadata,
                    ({IMAGE_DISPLAY_SQL}) AS image_display
                    FROM ranked WHERE rank=1 AND id>? ORDER BY id LIMIT ?""", (segment, upper, after, limit)).fetchall()
                return [dict(row) for row in rows]
            finally:
                conn.create_function("pwa_display_key", 3, None)
                conn.create_function("pwa_text_length", 1, None)
