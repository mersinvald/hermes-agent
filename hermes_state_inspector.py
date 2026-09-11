"""Retained, content-free lifecycle facts, separate from chat/replay payloads.

Logical model calls include retries. Retention is explicitly shared with the
native journal's configured age/count/byte limits in a separate budget, with
durable per-execution gap evidence.
"""

from __future__ import annotations

import json
import secrets
import time


def opaque():
    return "d_" + secrets.token_hex(32)


class NativeInspectorStateMixin:
    def _native_inspector_begin(self, conn, execution_id, now):
        conn.execute(
            "UPDATE native_executions SET facts_started_at=COALESCE(facts_started_at,?), "
            "facts_ref=COALESCE(facts_ref,?) WHERE execution_id=?",
            (now, opaque(), execution_id),
        )

    def _native_inspector_prune(self, conn, now):
        limits = self._native_events_limits
        # SQLite performs the bounded-storage accounting, without loading all
        # observations or any source content into the API process.
        conn.execute(
            "UPDATE native_executions SET facts_evicted=1 WHERE execution_id IN "
            "(SELECT execution_id FROM (SELECT execution_id,updated_at, "
            "ROW_NUMBER() OVER (ORDER BY ordinal DESC) n, "
            "SUM(length(body)) OVER (ORDER BY ordinal DESC) bytes "
            "FROM native_inspector_facts) WHERE n>? OR bytes>? OR updated_at<?)",
            (limits.max_count, limits.max_bytes, now - limits.max_age_seconds),
        )
        conn.execute(
            "DELETE FROM native_inspector_facts WHERE ordinal IN (SELECT ordinal FROM "
            "(SELECT ordinal,updated_at,ROW_NUMBER() OVER (ORDER BY ordinal DESC) n, "
            "SUM(length(body)) OVER (ORDER BY ordinal DESC) bytes FROM native_inspector_facts) "
            "WHERE n>? OR bytes>? OR updated_at<?)",
            (limits.max_count, limits.max_bytes, now - limits.max_age_seconds),
        )

    def _native_inspector_fact(self, conn, execution, kind, identity, values):
        row = conn.execute(
            "SELECT * FROM native_inspector_facts WHERE execution_id=? AND kind=? AND identity=?",
            (execution, kind, identity),
        ).fetchone()
        body = json.loads(row["body"]) if row else {"ref": opaque()}
        # Never regress the first timestamp when callbacks are repeated.
        first = body.get("started_at")
        body.update(values)
        if first is not None:
            body["started_at"] = first
        raw = json.dumps(body, separators=(",", ":"), allow_nan=False)
        if len(raw.encode()) > 8192:
            raise ValueError("native inspector metadata exceeds bound")
        now = time.time()
        conn.execute(
            "INSERT INTO native_inspector_facts(execution_id,kind,identity,body,updated_at) "
            "VALUES(?,?,?,?,?) ON CONFLICT(execution_id,kind,identity) DO UPDATE SET "
            "body=excluded.body,updated_at=excluded.updated_at",
            (execution, kind, identity, raw, now),
        )
        self._native_inspector_prune(conn, now)
        return body

    def native_inspector_record(self, origin, kind, identity, values):
        def write(conn):
            execution = conn.execute(
                "SELECT * FROM native_executions WHERE execution_id=? AND conversation_id=? "
                "AND owner=? AND state='open'",
                (origin.execution_id, origin.conversation_id, origin.owner),
            ).fetchone()
            if not execution or not execution["facts_started_at"]:
                return None
            return self._native_inspector_fact(
                conn, origin.execution_id, kind, identity, values
            )

        return self._execute_write(write)

    def native_inspector_fact(self, execution, kind, identity):
        with self._read_ctx() as conn:
            row = conn.execute(
                "SELECT body FROM native_inspector_facts WHERE execution_id=? AND kind=? AND identity=?",
                (execution, kind, identity),
            ).fetchone()
            return json.loads(row[0]) if row else None

    def native_inspector_agent(self, origin, session):
        with self._read_ctx() as conn:
            child = conn.execute(
                "SELECT body FROM native_inspector_facts WHERE execution_id=? AND kind='local' "
                "AND json_extract(body,'$.child_session_id')=? ORDER BY ordinal DESC LIMIT 1",
                (origin.execution_id, session),
            ).fetchone()
            execution = conn.execute(
                "SELECT facts_ref FROM native_executions WHERE execution_id=? AND owner=?",
                (origin.execution_id, origin.owner),
            ).fetchone()
            if not execution or not execution[0]:
                return None
            if child:
                data = json.loads(child[0])
                return data["ref"], data["parent_agent_ref"], data["role"], execution[0]
            return execution[0], None, "primary", execution[0]
