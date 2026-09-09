"""Real SQLite retention, isolation, atomicity and recovery behavior."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from hermes_state_events import EventLimits
from tests.state.test_native_commands import database, body, open_input


def event_db(tmp_path, **limits):
    db = database(tmp_path)
    db.native_events_enable(EventLimits(**limits))
    return db


def capture(db, root="root", scope="owner", cursor=None):
    return db.native_event_recovery(root, scope, cursor)


def emit(db, identity="e1"):
    db._execute_write(
        lambda c: db._native_event_append(
            c, "root", identity, "execution_state_changed", {"state": "running"}
        )
    )


def test_replay_duplicate_cursor_order_and_conversation_scope(tmp_path):
    db = event_db(tmp_path)
    first = capture(db)["cursor"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda n: emit(db, f"e{n}"), range(20)))
    one, two = capture(db, cursor=first), capture(db, cursor=first)
    assert one["events"] == two["events"]
    assert [e["sequence"] for e in one["events"]] == list(range(1, 21))
    assert len({e["event_id"] for e in one["events"]}) == 20
    assert capture(db, cursor=one["cursor"])["events"] == []
    foreign = capture(db, root="other", cursor=first)
    assert foreign["status"] == "gap" and foreign["events"] == []
    assert foreign["cursor"]["sequence"] == 0
    db.close()


@pytest.mark.parametrize(
    "cursor",
    [
        {},
        {"epoch": "x", "sequence": True},
        {"epoch": "x", "sequence": -1},
        {"epoch": "x", "sequence": 1, "secret": "bad"},
    ],
)
def test_invalid_cursor_is_explicit_recovery(tmp_path, cursor):
    db = event_db(tmp_path)
    emit(db)
    result = capture(db, cursor=cursor)
    assert result["status"] == "gap"
    assert result["detail"]["reason"] == "invalid_cursor"
    assert result["events"] == []
    db.close()


def test_event_identity_and_future_cursor_are_rejected(tmp_path):
    db = event_db(tmp_path)
    emit(db)
    cursor = capture(db)["cursor"]
    assert capture(db, cursor={**cursor, "event_id": "wrong"})["status"] == "gap"
    assert (
        capture(db, cursor={**cursor, "sequence": cursor["sequence"] + 1})["status"]
        == "gap"
    )
    db.close()


def test_count_eviction_and_slow_reader_are_explicit(tmp_path):
    db = event_db(tmp_path, max_count=2)
    old = capture(db)["cursor"]
    for _ in range(4):
        emit(db)
    result = capture(db, cursor=old)
    assert result["status"] == "expired"
    assert [e["sequence"] for e in result["events"]] == [3, 4]
    assert result["cursor"]["sequence"] == 4
    db.close()


def test_bytes_age_and_event_size_are_bounded(tmp_path, monkeypatch):
    db = event_db(tmp_path, max_bytes=1024, max_event_bytes=1024, max_age_seconds=10)
    now = 1000000
    monkeypatch.setattr("hermes_state_events.time.time", lambda: now)
    old = capture(db)["cursor"]
    for _ in range(8):
        emit(db)
    with db._read_ctx() as conn:
        assert (
            conn.execute("SELECT SUM(byte_count) FROM native_events").fetchone()[0]
            <= 1024
        )
    with pytest.raises(ValueError, match="byte limit"):
        db._execute_write(
            lambda c: db._native_event_append(
                c,
                "root",
                "e1",
                "tool_changed",
                {"activity_id": "a", "state": "running", "text": "x" * 1500},
            )
        )
    now += 11
    result = capture(db, cursor=old)
    assert result["status"] == "expired" and result["events"] == []
    db.close()


def test_restart_keeps_old_detail_but_reports_new_epoch_and_unknown(tmp_path):
    from hermes_state import SessionDB

    db = event_db(tmp_path)
    old = capture(db)["cursor"]
    owner = open_input(db, owner="pid=99999999:birth=0")
    db.native_command_apply(
        "e1",
        owner,
        "root",
        "holder",
        {"role": "user", "content": "hello"},
        command=("owner", "c1"),
    )
    db._execute_write(
        lambda c: c.execute("UPDATE session_turn_leases SET expires_at=0")
    )
    db.close()
    db = SessionDB(tmp_path / "state.db")
    db.native_events_enable()
    assert db.native_execution_reconcile("root")
    result = capture(db, cursor=old)
    assert result["status"] == "gap" and result["cursor"]["epoch"] != old["epoch"]
    assert result["events"] and all(
        e["epoch"] == old["epoch"] for e in result["events"]
    )
    assert result["executions"][0]["observed_state"] == "unknown"
    assert result["command_receipts"][0]["application_state"] == "applied"
    assert len(db.get_messages("root")) == 1
    db.close()


def test_event_failure_rolls_back_command_admission_and_input(tmp_path):
    db = event_db(tmp_path)
    trigger = "CREATE TRIGGER no_event BEFORE INSERT ON native_events BEGIN SELECT RAISE(ABORT,'events unavailable'); END"
    db._execute_write(lambda c: c.execute(trigger))
    with pytest.raises(sqlite3.DatabaseError):
        db.native_command_admit("owner", body(), effect="start")
    assert db.native_command_lookup("owner", "c1") is None
    db._execute_write(lambda c: c.execute("DROP TRIGGER no_event"))
    owner = open_input(db)
    db._execute_write(lambda c: c.execute(trigger))
    with pytest.raises(sqlite3.DatabaseError):
        db.native_command_apply(
            "e1",
            owner,
            "root",
            "holder",
            {"role": "user", "content": "hello"},
            command=("owner", "c1"),
        )
    assert db.get_messages("root") == []
    assert db.native_command_lookup("owner", "c1")["phase"] == "assigned"
    db.close()


def test_private_command_positions_advance_without_gaps_and_bounded_snapshot(tmp_path):
    db = event_db(tmp_path, batch_count=1, snapshot_count=2)
    old = capture(db)["cursor"]
    for n in range(4):
        db.native_command_admit("foreign", body(id=f"f{n}"), effect="queue")
    db.native_command_admit("owner", body(), effect="queue")
    for n in range(4):
        result = capture(db, cursor=old)
        assert result["status"] == "current" and result["events"] == []
        assert result["cursor"]["sequence"] == n + 1
        assert [r["command_id"] for r in result["command_receipts"]] == ["c1"]
        old = result["cursor"]
    result = capture(db, cursor=old)
    assert result["events"][0]["payload"]["command_id"] == "c1"
    assert len(capture(db, scope="foreign")["command_receipts"]) == 2
    assert capture(db, scope="foreign")["coverage"]["commands_has_more"]
    db.close()


def test_snapshot_cursor_and_command_state_are_one_transaction(tmp_path, monkeypatch):
    import threading

    db = event_db(tmp_path)
    inside = threading.Event()
    release = threading.Event()
    writer_started = threading.Event()
    prune = db._native_event_prune
    first = True

    def gated(conn, now):
        nonlocal first
        prune(conn, now)
        if first:
            first = False
            inside.set()
            assert release.wait(10)

    monkeypatch.setattr(db, "_native_event_prune", gated)

    def write():
        writer_started.set()
        return db.native_command_admit("owner", body(), effect="start")

    with ThreadPoolExecutor(max_workers=2) as pool:
        before_future = pool.submit(capture, db)
        assert inside.wait(10)
        after_future = pool.submit(write)
        assert writer_started.wait(10)
        release.set()
        before = before_future.result(timeout=10)
        after_future.result(timeout=10)
    assert before["command_receipts"] == [] and before["cursor"]["sequence"] == 0
    after = capture(db, cursor=before["cursor"])
    event = after["events"][0]
    receipt = after["command_receipts"][0]
    assert event["payload"]["receipt"] == receipt
    assert event["conversation_id"] == receipt["conversation_id"]
    assert event["payload"]["command_id"] == receipt["command_id"]
    assert event["payload"]["application_state"] == receipt["application_state"]
    db.close()


def test_additive_migration_preserves_legacy_execution_and_native_history(tmp_path):
    from hermes_state import SessionDB

    path = tmp_path / "state.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE native_executions (execution_id TEXT PRIMARY KEY,conversation_id TEXT NOT NULL,owner TEXT NOT NULL,state TEXT NOT NULL,input_started INTEGER NOT NULL DEFAULT 0,created_order INTEGER NOT NULL DEFAULT 0,lease_holder TEXT)"
        )
        conn.execute(
            "INSERT INTO native_executions VALUES ('legacy','root','old','closed',1,1,NULL)"
        )
    db = SessionDB(path)
    db.create_session("root", "telegram")
    db.append_message("root", "user", "native retained history")
    db.native_events_enable()
    view = capture(db)
    assert view["executions"][0]["execution_id"] == "legacy"
    assert view["executions"][0]["created_at"] is None
    assert view["executions"][0]["observed_state"] == "unknown"
    assert db.get_messages("root")[0]["content"] == "native retained history"
    db.close()


def test_completed_identity_cannot_generate_false_start_event(tmp_path):
    from hermes_state_commands import CommandConflict

    db = event_db(tmp_path)
    owner = open_input(db)
    db.native_command_apply(
        "e1",
        owner,
        "root",
        "holder",
        {"role": "user", "content": "hello"},
        command=("owner", "c1"),
    )
    db.native_execution_close("e1", owner, outcome="completed")
    cursor = capture(db)["cursor"]
    with pytest.raises(CommandConflict):
        db.native_execution_open("root", "e1", owner)
    result = capture(db, cursor=cursor)
    assert result["events"] == []
    assert result["executions"][0]["observed_state"] == "completed"
    db.close()
