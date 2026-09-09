"""Genuine schema31 upgrade and additive, inactive N06 control bookkeeping."""

import sqlite3

import pytest

import hermes_state_schema as schema
from agent.native_execution_context import NativeExecutionOrigin
from hermes_state import SessionDB
from hermes_state_common import SCHEMA_SQL
from hermes_state_pwa_scan import history_snapshot
from tests.state.test_native_cancellation import cancel, prepare
from tests.state.test_native_commands import database, open_input


CONTROL_TABLES = {
    "native_control_commands",
    "native_redirect_state",
    "native_clarifications",
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
)


def capture(conn):
    return {
        table: conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
        for table in RETAINED_TABLES
    }


@pytest.mark.parametrize("existing", [False, True])
def test_schema32_fresh_or_genuine31_upgrade_preserves_history_cancel_and_reopens(
    tmp_path, monkeypatch, existing
):
    path = tmp_path / "state.db"
    before = None
    if existing:
        start = SCHEMA_SQL.index("-- N06 durable control identities")
        end = SCHEMA_SQL.index("-- Native PWA assignments", start)
        with monkeypatch.context() as old:
            old.setattr(schema, "SCHEMA_SQL", SCHEMA_SQL[:start] + SCHEMA_SQL[end:])
            old.setattr(schema, "SCHEMA_VERSION", 31)
            db = database(tmp_path)
            owner = open_input(db)
            db.native_command_apply(
                "e1",
                owner,
                "root",
                "holder",
                {"role": "user", "content": "retained native input"},
                command=("owner", "c1"),
            )
            origin = NativeExecutionOrigin("root", "e1", owner)
            dispatch = prepare(db, origin)
            db.native_remote_dispatch_observe(
                origin, dispatch, "task", "ctx", "working"
            )
            db.native_cancel_admit("owner", cancel())
            db.native_remote_cancel_reserve("owner", "cancel-1", dispatch)
            db._execute_write(
                lambda conn: conn.execute(
                    "UPDATE messages SET content='retained corrected input' WHERE role='user'"
                )
            )
            fence = history_snapshot(db)
            assert fence["generation"] > 0
            db.close()
        with sqlite3.connect(path) as conn:
            assert (
                conn.execute("SELECT version FROM schema_version").fetchone()[0] == 31
            )
            old_tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            assert not old_tables & CONTROL_TABLES
            before = capture(conn)
            triggers = conn.execute(
                "SELECT name,sql FROM sqlite_master WHERE type='trigger' AND name LIKE 'pwa_history_%' ORDER BY name"
            ).fetchall()
            assert triggers
    for _ in range(2):
        db = SessionDB(path)
        if existing:
            assert history_snapshot(db) == fence
            assert db.get_messages("root")[0]["content"] == "retained corrected input"
            receipt = db.native_cancel_snapshot("owner", "cancel-1")
            assert receipt["known_count"] == 1
            assert receipt["targets"][0]["write_reserved"] == 1
            assert receipt["targets"][0]["cancel_request_state"] == "unknown"
        db.close()
        with sqlite3.connect(path) as conn:
            assert (
                conn.execute("SELECT version FROM schema_version").fetchone()[0]
                == schema.SCHEMA_VERSION
            )
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            assert CONTROL_TABLES <= tables
            assert all(
                conn.execute(f"SELECT count(*) FROM {name}").fetchone()[0] == 0
                for name in CONTROL_TABLES
            )
            if existing:
                assert CONTROL_TABLES <= tables - old_tables
                assert capture(conn) == before
                assert (
                    conn.execute(
                        "SELECT name,sql FROM sqlite_master WHERE type='trigger' AND name LIKE 'pwa_history_%' ORDER BY name"
                    ).fetchall()
                    == triggers
                )
            assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def insert(conn, table, values):
    columns = ",".join(values)
    placeholders = ",".join("?" for _ in values)
    conn.execute(
        f"INSERT INTO {table}({columns}) VALUES({placeholders})", tuple(values.values())
    )


def command():
    return dict(
        scope="owner",
        command_id="redirect",
        kind="redirect",
        conversation_id="root",
        execution_id="execution",
        owner="native-owner",
        fingerprint="a" * 64,
        payload_json="{}",
        recorded_at=1,
        updated_at=1,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        {"kind": "send"},
        {"kind": "redirect_confirm"},
        {"write_reserved": 2},
        {"write_reserved": 1},
        {"answer_state": "applied"},
        {"answer_state": "delivered"},
        {"payload_json": "\U0001f600" * 131073},
    ],
)
def test_control_schema_rejects_invalid_kind_handoff_and_unbounded_bytes(
    tmp_path, mutation
):
    db = SessionDB(tmp_path / "state.db")
    try:
        with pytest.raises(sqlite3.IntegrityError):
            db._execute_write(
                lambda conn: insert(
                    conn, "native_control_commands", {**command(), **mutation}
                )
            )
    finally:
        db.close()


@pytest.mark.parametrize(
    "mutation",
    [
        {"origin": "unknown"},
        {"question_revision": 0},
        {"question_json": "\U0001f600" * 16385},
        {"origin_json": "x" * 16385},
        {"answer_command_id": "without-owner-or-token"},
    ],
)
def test_question_schema_requires_exact_origin_and_bounded_complete_claim(
    tmp_path, mutation
):
    db = SessionDB(tmp_path / "state.db")
    values = dict(
        clarification_id="question",
        question_revision=1,
        conversation_id="root",
        execution_id="execution",
        owner="native-owner",
        origin="native",
        origin_key="immutable-origin",
        origin_json="{}",
        question_json="{}",
        recorded_at=1,
        updated_at=1,
    )
    try:
        with pytest.raises(sqlite3.IntegrityError):
            db._execute_write(
                lambda conn: insert(
                    conn, "native_clarifications", {**values, **mutation}
                )
            )
    finally:
        db.close()


def test_redirect_confirmation_revision_and_release_evidence_are_paired(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db._execute_write(
            lambda conn: insert(conn, "native_control_commands", command())
        )
        db._execute_write(
            lambda conn: insert(
                conn,
                "native_control_commands",
                {
                    **command(),
                    "command_id": "confirm",
                    "kind": "redirect_confirm",
                    "reference_id": "redirect",
                    "reference_revision": 1,
                },
            )
        )
        valid = dict(
            scope="owner",
            command_id="redirect",
            confirmation_required=1,
            confirmation_revision=1,
            confirmation_command_id="confirm",
            confirmed_revision=1,
            updated_at=1,
        )
        for mutation in (
            {"confirmed_revision": 2},
            {"confirmation_required": 0},
            {"confirmation_revision": 0},
            {"released_at": 1},
            {"release_basis_json": "{}"},
        ):
            with pytest.raises(sqlite3.IntegrityError):
                db._execute_write(
                    lambda conn: insert(
                        conn, "native_redirect_state", {**valid, **mutation}
                    )
                )
        db._execute_write(lambda conn: insert(conn, "native_redirect_state", valid))
    finally:
        db.close()
