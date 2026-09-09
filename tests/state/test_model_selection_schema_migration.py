"""Schema33 adds inactive native model selection and execution capture storage."""

from __future__ import annotations

import sqlite3

import pytest

import hermes_state_schema as schema
from hermes_state import SessionDB
from hermes_state_common import SCHEMA_SQL
from tests.state.test_native_commands import database, open_input


MODEL_TABLES = {
    "native_conversation_models",
    "native_model_mutations",
    "native_execution_models",
}
RETAINED_TABLES = (
    "sessions",
    "messages",
    "session_turn_leases",
    "native_executions",
    "native_commands",
    "native_cancel_commands",
    "native_remote_dispatches",
    "native_remote_cancel_attempts",
    "native_pwa_history_fence",
    "native_control_commands",
    "native_redirect_state",
    "native_clarifications",
    "native_pwa_conversations",
    "native_channel_deliveries",
)


def capture(conn):
    return {
        table: conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
        for table in RETAINED_TABLES
    }


@pytest.mark.parametrize("existing", [False, True])
def test_schema33_fresh_or_genuine32_upgrade_is_additive(tmp_path, monkeypatch, existing):
    path = tmp_path / "state.db"
    before = None
    if existing:
        start = SCHEMA_SQL.index("-- S03 canonical conversation selection")
        end = SCHEMA_SQL.index("-- Native PWA assignments", start)
        with monkeypatch.context() as old:
            old.setattr(schema, "SCHEMA_SQL", SCHEMA_SQL[:start] + SCHEMA_SQL[end:])
            old.setattr(schema, "SCHEMA_VERSION", 32)
            db = database(tmp_path)
            owner = open_input(db)
            db.native_command_apply(
                "e1",
                owner,
                "root",
                "holder",
                {"role": "user", "content": "retained input"},
                command=("owner", "c1"),
            )
            db.native_pwa_assign(
                ("concierge", "https://issuer.invalid", "subject"),
                "root",
                '{"platform":"telegram"}',
                "discovered",
            )
            db.native_delivery_reserve("e1", "telegram:synthetic", "root", 1)
            assert db.native_delivery_finish("e1", "telegram:synthetic", "unknown")

            def retain_controls(conn):
                conn.execute(
                    "INSERT INTO native_control_commands "
                    "(scope,command_id,kind,conversation_id,execution_id,owner,"
                    "fingerprint,payload_json,recorded_at,updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        "control-scope", "redirect", "redirect", "root", "e1",
                        owner, "a" * 64, "{}", 1.0, 1.0,
                    ),
                )
                conn.execute(
                    "INSERT INTO native_redirect_state "
                    "(scope,command_id,confirmation_required,dispatch_frontier,"
                    "release_basis_json,released_at,updated_at) VALUES (?,?,?,?,?,?,?)",
                    (
                        "control-scope", "redirect", 0, "closed", "{}", 1.0, 1.0,
                    ),
                )
                conn.execute(
                    "INSERT INTO native_control_commands "
                    "(scope,command_id,kind,conversation_id,execution_id,owner,"
                    "fingerprint,payload_json,reference_id,reference_revision,"
                    "answer_state,delivery_evidence,write_reserved,recorded_at,updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        "control-scope", "answer", "clarification_response", "root",
                        "e1", owner, "b" * 64, "{}", "question", 1, "delivered",
                        "native_waiter_handoff", 1, 2.0, 2.0,
                    ),
                )
                conn.execute(
                    "INSERT INTO native_clarifications "
                    "(clarification_id,question_revision,conversation_id,execution_id,"
                    "owner,origin,origin_key,origin_json,question_json,state,answer_scope,"
                    "answer_command_id,claim_token,recorded_at,updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        "question", 1, "root", "e1", owner, "native",
                        "immutable-question", "{}", "{}", "answered",
                        "control-scope", "answer", "claim", 2.0, 2.0,
                    ),
                )

            db._execute_write(retain_controls)
            db.close()
        with sqlite3.connect(path) as conn:
            assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 32
            before = capture(conn)
            old_tables = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            assert not old_tables & MODEL_TABLES
            triggers = conn.execute(
                "SELECT name,sql FROM sqlite_master "
                "WHERE type='trigger' AND name LIKE 'pwa_history_%' ORDER BY name"
            ).fetchall()
            assert len(triggers) == 9

    for _ in range(2):
        db = SessionDB(path)
        db.close()
        with sqlite3.connect(path) as conn:
            assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 33
            tables = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            assert MODEL_TABLES <= tables
            assert all(
                conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
                for table in MODEL_TABLES
            )
            if existing:
                assert capture(conn) == before
                assert (
                    conn.execute(
                        "SELECT name,sql FROM sqlite_master "
                        "WHERE type='trigger' AND name LIKE 'pwa_history_%' ORDER BY name"
                    ).fetchall()
                    == triggers
                )
            assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_schema33_constraints_keep_receipts_and_captures_bounded(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("root", "telegram")
    db.native_execution_open("root", "execution", "owner")

    def write(conn):
        conn.execute(
            "INSERT INTO native_conversation_models VALUES (?,?,?,?)",
            ("root", "daily", 1, 1.0),
        )
        conn.execute(
            "INSERT INTO native_model_mutations "
            "(scope,mutation_id,conversation_id,fingerprint,model_id,"
            "expected_model_version,resulting_model_version,recorded_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            ("scope", "mutation", "root", "a" * 64, "daily", 0, 1, 1.0),
        )
        conn.execute(
            "INSERT INTO native_execution_models VALUES (?,?,?,?,?,?,?)",
            ("execution", "root", "owner", "holder", "daily", 1, 1.0),
        )

    db._execute_write(write)
    with pytest.raises(sqlite3.IntegrityError):
        db._execute_write(
            lambda conn: conn.execute(
                "INSERT INTO native_model_mutations "
                "(scope,mutation_id,conversation_id,fingerprint,model_id,"
                "expected_model_version,resulting_model_version,recorded_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                ("scope", "mutation", "root", "b" * 64, "other", 1, 2, 2.0),
            )
        )
    with pytest.raises(sqlite3.IntegrityError):
        db._execute_write(
            lambda conn: conn.execute(
                "INSERT INTO native_execution_models VALUES (?,?,?,?,?,?,?)",
                ("other", "root", "owner", "holder", "x" * 201, 1, 1.0),
            )
        )
    db.close()
