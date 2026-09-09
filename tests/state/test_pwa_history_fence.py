"""Actual SQLite generation triggers and genuine pre-fence upgrade behavior."""
import json
import sqlite3
from pathlib import Path

import pytest

import hermes_state_schema as schema
from hermes_state import SessionDB
from hermes_state_common import SCHEMA_SQL
from hermes_state_pwa_scan import history_snapshot, install_history_fence
from gateway.pwa_ownership import NativePwaOwnership


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    database.create_session("root", "telegram", user_id="synthetic")
    try:
        yield database
    finally:
        database.close()


def write(db, sql, params=()):
    return db._execute_write(lambda connection: connection.execute(sql, params))


def generation(db):
    return history_snapshot(db)["generation"]


@pytest.mark.parametrize("version", [29, 30])
def test_real_pre_fence_upgrade_preserves_history_and_reopens(tmp_path, monkeypatch, version):
    path = tmp_path / "upgrade.db"
    start, end = SCHEMA_SQL.index("-- Native PWA history fence (schema31)"), SCHEMA_SQL.index("-- Native PWA assignments")
    old_sql = SCHEMA_SQL[:start] + SCHEMA_SQL[end:]
    if version == 29:
        start, end = old_sql.index("-- Explicit native cancellation"), old_sql.index("-- Native PWA assignments")
        old_sql = old_sql[:start] + old_sql[end:]
    with monkeypatch.context() as legacy:
        legacy.setattr(schema, "SCHEMA_SQL", old_sql)
        legacy.setattr(schema, "SCHEMA_VERSION", version)
        legacy.setattr(schema, "install_history_fence", lambda _: None)
        db = SessionDB(path)
        db.create_session("retained", "telegram", user_id="synthetic")
        db.append_message("retained", "user", "retained history")
        db.close()
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == version
        assert conn.execute("SELECT name FROM sqlite_master WHERE name LIKE 'pwa_history_%' OR name='native_pwa_history_fence'").fetchall() == []
        before = {table: conn.execute(f"SELECT * FROM {table}").fetchall() for table in ("sessions", "messages")}
    for _ in range(2):
        db = SessionDB(path)
        assert generation(db) == 0
        db.close()
        with sqlite3.connect(path) as conn:
            assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 31
            assert before == {table: conn.execute(f"SELECT * FROM {table}").fetchall() for table in before}
            assert len(conn.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'pwa_history_%'").fetchall()) == 9
            assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_native_append_setup_and_bookkeeping_do_not_change_generation(db):
    start = history_snapshot(db)
    for index in range(3):
        db.create_session("root", "telegram", user_id="synthetic")
        db.append_message("root", "user", f"new message {index}")
        write(db, "UPDATE sessions SET last_activity_at=?,message_count=message_count+1,model_config=? WHERE id='root'",
              (index, '{"provider":"synthetic","usage":' + str(index) + '}'))
        db.append_message("root", "session_meta", f"bookkeeping {index}")
    assert generation(db) == start["generation"]
    assert history_snapshot(db)["message_upper"] > start["message_upper"]
    before = generation(db)
    write(db, "UPDATE messages SET content='different metadata' WHERE role='session_meta'")
    write(db, "DELETE FROM messages WHERE role='session_meta'")
    assert generation(db) == before


@pytest.mark.parametrize("change", ["content='zzzz'", "active=0", "compacted=1", "display_kind='carrier'", "tool_name='changed'", "timestamp=timestamp+1"])
def test_protected_message_rewrites_change_fence_even_at_equal_size(db, change):
    db.append_message("root", "user", "aaaa")
    old = generation(db)
    write(db, "UPDATE messages SET " + change + " WHERE session_id='root'")
    assert generation(db) > old
    steady = generation(db)
    write(db, "UPDATE messages SET content=content,active=active,compacted=compacted")
    assert generation(db) == steady


@pytest.mark.parametrize("change", ["user_id='foreign'", "parent_session_id='parent'", "end_reason='compression'", "title='Changed'", "model_config='{\"_branched_from\":\"parent\"}'", "model_config='{\"_delegate_from\":\"parent\"}'", "model_config='not json'"])
def test_ownership_lineage_and_title_updates_fence(db, change):
    db.create_session("parent", "telegram", user_id="synthetic")
    old = generation(db)
    write(db, "UPDATE sessions SET " + change + " WHERE id='root'")
    assert generation(db) > old


def test_model_metadata_crossing_authorization_size_bound_fences(db):
    old = generation(db)
    write(db, "UPDATE sessions SET model_config=? WHERE id='root'", ('{"usage":"' + 'x' * 16384 + '"}',))
    assert generation(db) > old
    old = generation(db)
    write(db, "UPDATE sessions SET model_config='{}' WHERE id='root'")
    assert generation(db) > old


def test_multirow_ignore_replace_and_rollback_do_not_starve_or_hide_rewrites(db):
    db.append_message("root", "user", "original")
    old = generation(db)
    write(db, "INSERT OR IGNORE INTO sessions(id,source,started_at) VALUES('root','telegram',0),('next','telegram',0),('root','telegram',0)")
    write(db, "INSERT OR IGNORE INTO messages(id,session_id,role,content,timestamp) VALUES(1,'root','user','ignored',0),(900,'root','user','new',0),(1,'root','user','ignored again',0)")
    assert generation(db) == old
    write(db, "INSERT OR REPLACE INTO sessions(id,source,started_at) VALUES('root','telegram',0),('next','telegram',0)")
    assert generation(db) == old + 2
    before = generation(db)
    write(db, "INSERT OR REPLACE INTO messages(id,session_id,role,content,timestamp) VALUES(1,'root','user','replaced',0),(900,'root','user','changed',0)")
    assert generation(db) == before + 2
    before = history_snapshot(db)
    def rollback(conn):
        conn.execute("INSERT OR REPLACE INTO sessions(id,source,started_at) VALUES('root','telegram',0)")
        conn.execute("UPDATE messages SET content='rollback'")
        raise RuntimeError("synthetic rollback")
    with pytest.raises(RuntimeError, match="synthetic rollback"):
        db._execute_write(rollback)
    assert history_snapshot(db) == before
    write(db, "UPDATE sessions SET title='unique' WHERE id='next'")
    before = generation(db)
    write(db, "INSERT OR REPLACE INTO sessions(id,source,started_at,title) VALUES('replacement','telegram',0,'unique')")
    assert generation(db) > before


def test_delete_and_rowid_reuse_changes_generation(db):
    db.append_message("root", "user", "retained")
    before = generation(db)
    write(db, "DELETE FROM messages")
    assert generation(db) > before
    before = generation(db)
    write(db, "DELETE FROM sessions WHERE id='root'")
    assert generation(db) > before
    db.create_session("replacement", "telegram")
    assert generation(db) > before


@pytest.mark.parametrize("model_config,source,changes", [
    ({}, "telegram", True), ({"_branched_from": "root"}, "telegram", False),
    ({"_delegate_from": "root"}, "telegram", False), ({}, "tool", False),
    ("malformed", "tool", True), ({"provider": "x" * 17000}, "telegram", True),
])
def test_existing_compressed_parent_new_child_matches_resolver_availability(db, model_config, source, changes):
    db.end_session("root", "compression")
    old = generation(db)
    # Raw malformed JSON is deliberate legacy-adversary input. Product writes
    # normally serialize dicts, so ordinary branches use the same actual API.
    if isinstance(model_config, str):
        write(db, "INSERT INTO sessions(id,source,parent_session_id,started_at,model_config) VALUES('child',?,'root',0,?)", (source, model_config))
    else:
        db.create_session("child", source, user_id="synthetic", parent_session_id="root", model_config=model_config)
    assert (generation(db) > old) is changes
    if not changes:
        assert generation(db) == old
    resolver = NativePwaOwnership(None, db, None, secret="synthetic-only")
    if isinstance(model_config, str) or len(str(model_config)) > 16384:
        with pytest.raises(LookupError):
            resolver.resolve("root")
    else:
        expected = ["root", "child"] if changes else ["root"]
        assert resolver.resolve("root")["native_session_ids"] == expected


def test_native_branch_ownership_backfill_is_conservative_invalidation(db):
    db.end_session("root", "compression")
    old = generation(db)
    db.create_session("branch", "telegram", parent_session_id="root",
                      model_config={"_branched_from": "root"})
    assert generation(db) > old
    assert db.get_session("branch")["user_id"] == "synthetic"
    resolver = NativePwaOwnership(None, db, None, secret="synthetic-only")
    assert resolver.resolve("root")["native_session_ids"] == ["root"]


def install_original31(db):
    fixture = json.loads((Path(__file__).parents[1] / "fixtures/pwa_history_fence_31_initial.json").read_text())
    def old(conn):
        for name, statement in fixture["triggers"].items():
            conn.execute(f'DROP TRIGGER "{name}"')
            conn.execute(statement)
    db._execute_write(old)


@pytest.mark.parametrize("marker", ["_branched_from", "_delegate_from"])
def test_original31_duplicate_last_key_transition_is_repaired_on_reopen(tmp_path, marker):
    path = tmp_path / "original31.db"
    db = SessionDB(path)
    db.create_session("root", "telegram", user_id="synthetic")
    db.end_session("root", "compression")
    inactive = '{"' + marker + '":null,"' + marker + '":"root"}'
    active = '{"' + marker + '":null,"' + marker + '":null}'
    write(db, "INSERT INTO sessions(id,source,user_id,parent_session_id,started_at,model_config) VALUES('child','telegram','synthetic','root',0,?)", (inactive,))
    resolver = NativePwaOwnership(None, db, None, secret="synthetic-only")
    assert resolver.resolve("root")["native_session_ids"] == ["root"]
    install_original31(db)
    old_generation = generation(db)
    write(db, "UPDATE sessions SET model_config=? WHERE id='child'", (active,))
    assert generation(db) == old_generation  # Actual original31 first-key gap.
    assert resolver.resolve("root")["native_session_ids"] == ["root", "child"]
    db.close()
    db = SessionDB(path)
    repaired_generation = generation(db)
    assert repaired_generation == old_generation + 1
    write(db, "UPDATE sessions SET model_config=? WHERE id='child'", (inactive,))
    assert generation(db) == repaired_generation + 1
    assert NativePwaOwnership(None, db, None, secret="synthetic-only").resolve("root")["native_session_ids"] == ["root"]
    steady = generation(db)
    db.close()
    db = SessionDB(path)
    assert generation(db) == steady
    db.close()


@pytest.mark.parametrize("value,expected", [
    ('{"_branched_from":"root","_branched_from":null}', ["root", "child"]),
    ('{"_branched_from":null,"_branched_from":"root"}', ["root"]),
])
def test_duplicate_marker_child_insert_is_conservatively_fenced(db, value, expected):
    db.end_session("root", "compression")
    before = generation(db)
    write(db, "INSERT INTO sessions(id,source,user_id,parent_session_id,started_at,model_config) VALUES('child','telegram','synthetic','root',0,?)", (value,))
    assert generation(db) > before
    assert NativePwaOwnership(None, db, None, secret="synthetic-only").resolve("root")["native_session_ids"] == expected


def test_owned_trigger_repair_rolls_back_atomically_and_preserves_other_triggers(db):
    install_original31(db)
    write(db, "CREATE TRIGGER unrelated_synthetic AFTER UPDATE ON sessions BEGIN SELECT 1; END")
    with db._read_ctx() as conn:
        before = conn.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger' ORDER BY name").fetchall()
    old_generation = generation(db)
    class BrokenCursor:
        def __init__(self, conn):
            self.cursor = conn.cursor()
        def execute(self, statement, *args):
            if statement.startswith("CREATE TRIGGER IF NOT EXISTS pwa_history_session_update"):
                raise sqlite3.OperationalError("synthetic DDL failure")
            return self.cursor.execute(statement, *args)
    with pytest.raises(sqlite3.OperationalError, match="synthetic DDL failure"):
        db._execute_write(lambda conn: install_history_fence(BrokenCursor(conn)))
    with db._read_ctx() as conn:
        assert conn.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger' ORDER BY name").fetchall() == before
    assert generation(db) == old_generation
    db._execute_write(lambda conn: install_history_fence(conn.cursor()))
    assert generation(db) == old_generation + 1
    with db._read_ctx() as conn:
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='unrelated_synthetic'").fetchone()
    old_generation = generation(db)
    write(db, "DROP TRIGGER pwa_history_message_update")
    db._execute_write(lambda conn: install_history_fence(conn.cursor()))
    assert generation(db) == old_generation + 1
    db._execute_write(lambda conn: install_history_fence(conn.cursor()))
    assert generation(db) == old_generation + 1
