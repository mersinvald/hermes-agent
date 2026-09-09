"""Additive schema30 prerequisite, independent of cancellation runtime activation."""

import sqlite3

import pytest

import hermes_state_schema as schema
from hermes_state import SessionDB
from hermes_state_common import SCHEMA_SQL


@pytest.mark.parametrize("existing", [False, True])
def test_schema30_additive_upgrade_preserves_native_state_and_reopens(
    tmp_path, monkeypatch, existing
):
    path = tmp_path / "state.db"
    if existing:
        start = SCHEMA_SQL.index("-- Explicit native cancellation")
        end = SCHEMA_SQL.index("-- Native PWA assignments", start)
        old_sql = SCHEMA_SQL[:start] + SCHEMA_SQL[end:]
        with monkeypatch.context() as old:
            old.setattr(schema, "SCHEMA_SQL", old_sql)
            old.setattr(schema, "SCHEMA_VERSION", 29)
            db = SessionDB(path)
            db.create_session("retained", "telegram", model="synthetic")
            db.append_message("retained", "user", "retained native input")
            db._execute_write(
                lambda conn: conn.execute(
                    "INSERT INTO native_executions(execution_id,conversation_id,owner,state,input_started) VALUES('execution','retained','owner','open',1)"
                )
            )
            db.close()
        with sqlite3.connect(path) as conn:
            old_tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            retained = {
                table: list(conn.execute(f"SELECT * FROM {table}"))
                for table in ["sessions", "messages", "native_executions"]
            }
            assert (
                conn.execute("SELECT version FROM schema_version").fetchone()[0] == 29
            )
    for _ in range(2):
        db = SessionDB(path)
        db.close()
        with sqlite3.connect(path) as conn:
            assert (
                conn.execute("SELECT version FROM schema_version").fetchone()[0]
                == schema.SCHEMA_VERSION
            )
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            new_tables = {
                "native_cancel_commands",
                "native_remote_dispatches",
                "native_remote_cancel_attempts",
            }
            assert new_tables <= tables
            if existing:
                assert tables - old_tables == new_tables
                assert retained == {
                    table: list(conn.execute(f"SELECT * FROM {table}"))
                    for table in retained
                }
            columns = {
                r[1]: r
                for r in conn.execute("PRAGMA table_info(native_cancel_commands)")
            }
            assert (
                columns["remote_after"][4] == "0" and columns["next_poll_at"][4] == "0"
            )
            keys = {
                r[1]: r[5]
                for r in conn.execute(
                    "PRAGMA table_info(native_remote_cancel_attempts)"
                )
            }
            assert {
                key: keys[key] for key in ["dispatch_id", "scope", "command_id"]
            } == {"dispatch_id": 1, "scope": 2, "command_id": 3}
            assert "write_reserved" in keys
            assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
