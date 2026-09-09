"""Native managed-model selection in SessionDB transactions."""

from __future__ import annotations

import hashlib
import hmac
import json
import time

from hermes_state_commands import CommandConflict


MAX_VERSION = 9007199254740991


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def mutation_receipt(row, *, replayed):
    return {
        "schema_version": "1.0",
        "mutation_id": row["mutation_id"],
        "conversation_id": row["conversation_id"],
        "outcome": "replayed" if replayed else "applied",
        "model": {
            "model_id": row["model_id"],
            "model_version": row["resulting_model_version"],
        },
    }


class NativeModelStateMixin:
    def native_model_mutation_replay(self, scope, body):
        """Return an immutable retry receipt before consulting today's catalog."""
        payload = canonical(body)
        fingerprint = hashlib.sha256(
            ("native-model-mutation-v1\n" + payload).encode()
        ).hexdigest()
        with self._read_ctx() as conn:
            row = conn.execute(
                "SELECT * FROM native_model_mutations WHERE scope=? AND mutation_id=?",
                (scope, body["mutation_id"]),
            ).fetchone()
            if row is None:
                return None
            if not hmac.compare_digest(row["fingerprint"], fingerprint):
                raise CommandConflict("model mutation ID reused with a different payload")
            return mutation_receipt(row, replayed=True)

    def native_model_selection(self, root):
        with self._read_ctx() as conn:
            row = conn.execute(
                "SELECT model_id,model_version FROM native_conversation_models "
                "WHERE conversation_id=?",
                (root,),
            ).fetchone()
            return dict(row) if row else None

    def native_model_initialize(self, root, default_model_id):
        """Atomically initialize a canonical root to the configured default."""

        def write(conn):
            if conn.execute("SELECT 1 FROM sessions WHERE id=?", (root,)).fetchone() is None:
                raise LookupError("conversation unavailable")
            conn.execute(
                "INSERT INTO native_conversation_models "
                "(conversation_id,model_id,model_version,updated_at) VALUES (?,?,1,?) "
                "ON CONFLICT(conversation_id) DO NOTHING",
                (root, default_model_id, time.time()),
            )
            return dict(
                conn.execute(
                    "SELECT model_id,model_version FROM native_conversation_models "
                    "WHERE conversation_id=?",
                    (root,),
                ).fetchone()
            )

        return self._execute_write(write)

    def native_model_mutate(self, scope, body):
        payload = canonical(body)
        fingerprint = hashlib.sha256(
            ("native-model-mutation-v1\n" + payload).encode()
        ).hexdigest()

        def write(conn):
            old = conn.execute(
                "SELECT * FROM native_model_mutations WHERE scope=? AND mutation_id=?",
                (scope, body["mutation_id"]),
            ).fetchone()
            if old is not None:
                if not hmac.compare_digest(old["fingerprint"], fingerprint):
                    raise CommandConflict("model mutation ID reused with a different payload")
                return mutation_receipt(old, replayed=True)
            if conn.execute(
                "SELECT 1 FROM sessions WHERE id=?", (body["conversation_id"],)
            ).fetchone() is None:
                raise LookupError("conversation unavailable")
            current = conn.execute(
                "SELECT model_version FROM native_conversation_models "
                "WHERE conversation_id=?",
                (body["conversation_id"],),
            ).fetchone()
            current_version = current["model_version"] if current else 0
            if current_version != body["expected_model_version"]:
                raise CommandConflict("model version changed")
            if current_version >= MAX_VERSION:
                raise CommandConflict("model version exhausted")
            resulting_version = current_version + 1
            now = time.time()
            conn.execute(
                "INSERT INTO native_conversation_models "
                "(conversation_id,model_id,model_version,updated_at) VALUES (?,?,?,?) "
                "ON CONFLICT(conversation_id) DO UPDATE SET "
                "model_id=excluded.model_id,model_version=excluded.model_version,"
                "updated_at=excluded.updated_at",
                (
                    body["conversation_id"],
                    body["model_id"],
                    resulting_version,
                    now,
                ),
            )
            conn.execute(
                "INSERT INTO native_model_mutations "
                "(scope,mutation_id,conversation_id,fingerprint,model_id,"
                "expected_model_version,resulting_model_version,recorded_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    scope,
                    body["mutation_id"],
                    body["conversation_id"],
                    fingerprint,
                    body["model_id"],
                    body["expected_model_version"],
                    resulting_version,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM native_model_mutations WHERE scope=? AND mutation_id=?",
                (scope, body["mutation_id"]),
            ).fetchone()
            return mutation_receipt(row, replayed=False)

        return self._execute_write(write)

    def native_model_capture(
        self,
        execution_id,
        root,
        owner,
        lease_holder,
        default_model_id,
    ):
        """Capture selection after exact execution and live lease validation."""

        def write(conn):
            execution = conn.execute(
                "SELECT * FROM native_executions WHERE execution_id=? "
                "AND conversation_id=? AND owner=? AND state='open'",
                (execution_id, root, owner),
            ).fetchone()
            lease = conn.execute(
                "SELECT holder,expires_at FROM session_turn_leases "
                "WHERE conversation_id=?",
                (root,),
            ).fetchone()
            now = time.time()
            if (
                execution is None
                or not lease_holder
                or lease is None
                or lease["holder"] != lease_holder
                or lease["expires_at"] <= now
            ):
                raise CommandConflict("execution owner or lease changed")
            previous = conn.execute(
                "SELECT * FROM native_execution_models WHERE execution_id=?",
                (execution_id,),
            ).fetchone()
            if previous is not None:
                observed = tuple(
                    previous[key]
                    for key in ("conversation_id", "owner", "lease_holder")
                )
                if observed != (root, owner, lease_holder):
                    raise CommandConflict("execution model capture changed")
                return {
                    "model_id": previous["model_id"],
                    "model_version": previous["model_version"],
                }
            selection = conn.execute(
                "SELECT model_id,model_version FROM native_conversation_models "
                "WHERE conversation_id=?",
                (root,),
            ).fetchone()
            if selection is None:
                conn.execute(
                    "INSERT INTO native_conversation_models "
                    "(conversation_id,model_id,model_version,updated_at) "
                    "VALUES (?,?,1,?)",
                    (root, default_model_id, now),
                )
                selection = conn.execute(
                    "SELECT model_id,model_version FROM native_conversation_models "
                    "WHERE conversation_id=?",
                    (root,),
                ).fetchone()
            expected = (
                root,
                owner,
                lease_holder,
                selection["model_id"],
                selection["model_version"],
            )
            conn.execute(
                "INSERT INTO native_execution_models "
                "(execution_id,conversation_id,owner,lease_holder,model_id,"
                "model_version,captured_at) VALUES (?,?,?,?,?,?,?)",
                (execution_id, *expected, now),
            )
            return dict(selection)

        return self._execute_write(write)

    def native_execution_model(self, execution_id):
        with self._read_ctx() as conn:
            row = conn.execute(
                "SELECT model_id,model_version FROM native_execution_models "
                "WHERE execution_id=?",
                (execution_id,),
            ).fetchone()
            return dict(row) if row else None
