"""Additive schema30 prerequisite, independent of cancellation runtime activation."""

import sqlite3

import pytest

import hermes_state_schema as schema
from hermes_state import SessionDB
from hermes_state_common import SCHEMA_SQL
from agent.native_execution_context import NativeExecutionOrigin


@pytest.mark.parametrize("existing", [False, True])
def test_schema30_additive_upgrade_preserves_native_state_and_reopens(
    tmp_path, monkeypatch, existing
):
    path = tmp_path / "state.db"
    if existing:
        # A schema29 fixture must not inherit later schema31 fence objects.
        start = SCHEMA_SQL.index("-- Native PWA history fence (schema31)")
        end = SCHEMA_SQL.index("-- Native PWA assignments", start)
        old_sql = SCHEMA_SQL[:start] + SCHEMA_SQL[end:]
        start = old_sql.index("-- Explicit native cancellation")
        end = old_sql.index("-- Native PWA assignments", start)
        old_sql = old_sql[:start] + old_sql[end:]
        with monkeypatch.context() as old:
            old.setattr(schema, "SCHEMA_SQL", old_sql)
            old.setattr(schema, "SCHEMA_VERSION", 29)
            old.setattr(schema, "install_history_fence", lambda _: None)
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
                "native_pwa_history_fence",
            }
            assert new_tables <= tables
            if existing:
                assert new_tables <= tables - old_tables
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


def test_dispatch_caller_nullable_upgrade_preserves_existing_rows_and_old_schema_reopen(tmp_path, monkeypatch):
    path = tmp_path / "caller-upgrade.db"
    columns = ("caller_agent_ref", "caller_parent_agent_ref", "caller_role")
    old_sql = SCHEMA_SQL
    for column in columns:
        old_sql = old_sql.replace(f"    {column} TEXT,\n", "")
    origin = NativeExecutionOrigin("root", "execution", "owner")
    with monkeypatch.context() as old:
        old.setattr(schema, "SCHEMA_SQL", old_sql)
        db = SessionDB(path)
        db.create_session("root", "telegram")
        db.append_message("root", "user", "retained history")
        db.native_execution_open("root", "execution", "owner")
        db._execute_write(lambda conn: conn.execute(
            "INSERT INTO native_remote_dispatches(dispatch_id,conversation_id,execution_id,owner,request_id,phase,recorded_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
            ("old-dispatch", "root", "execution", "owner", "old-request", "attempted", 1, 2),
        ))
        db.close()
    with sqlite3.connect(path) as conn:
        old_columns = [item[1] for item in conn.execute("PRAGMA table_info(native_remote_dispatches)")]
        before = conn.execute("SELECT * FROM native_remote_dispatches").fetchall()
        version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
    db = SessionDB(path)
    try:
        assert db.native_remote_targets("execution")[0]["caller_agent_ref"] is None
        with db._read_ctx() as conn:
            assert conn.execute("SELECT " + ",".join(old_columns) + " FROM native_remote_dispatches").fetchall()[0][:] == before[0]
        caller = db.native_inspector_agent(origin, "root")
        assert caller is not None
        new_dispatch = db.native_remote_dispatch_prepare(origin, None, "new-request", caller=caller)
    finally:
        db.close()
    # The pre-change schema reconciler tolerates additive fields on rollback;
    # reads and explicit-column INSERTs still work without erasing caller data.
    with monkeypatch.context() as old:
        old.setattr(schema, "SCHEMA_SQL", old_sql)
        db = SessionDB(path)
        try:
            assert len(db.native_remote_targets("execution")) == 2
            retained = next(row for row in db.native_remote_targets("execution") if row["dispatch_id"] == new_dispatch)
            assert tuple(retained[column] for column in columns) == caller[:3]
            assert db.get_messages("root")[0]["content"] == "retained history"
            with db._read_ctx() as conn:
                assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == version
                assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        finally:
            db.close()
