"""SQLite atomicity, migration, restart and fencing for native command inputs."""

import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from hermes_state import SessionDB
from hermes_state_commands import CommandConflict, process_owner, receipt


def body(id="c1", kind="send", text="hello"):
    return {
        "schema_version": "1.0",
        "command_id": id,
        "conversation_id": "root",
        "type": kind,
        "payload": {"text": text},
    }


def database(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("root", "telegram")
    return db


def open_input(db, *, owner=None):
    owner = owner or process_owner()
    db.native_command_admit("owner", body(), effect="start")
    db.native_execution_open("root", "e1", owner, command=("owner", "c1"))
    assert db.try_acquire_session_turn_lease("root", "holder")
    return owner


def test_atomic_user_input_and_receipt_failure_rolls_back(tmp_path):
    db = database(tmp_path)
    owner = open_input(db)
    db._execute_write(
        lambda c: c.execute(
            "CREATE TRIGGER reject_application BEFORE UPDATE OF phase ON native_commands WHEN NEW.phase='applied' BEGIN SELECT RAISE(ABORT,'synthetic storage failure'); END"
        )
    )
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
    db._execute_write(lambda c: c.execute("DROP TRIGGER reject_application"))
    row = db.native_command_apply(
        "e1",
        owner,
        "root",
        "holder",
        {"role": "user", "content": "hello"},
        command=("owner", "c1"),
    )
    assert row["_row_id"] == db.native_command_lookup("owner", "c1")["input_row_id"]
    with pytest.raises(CommandConflict):
        db.native_command_apply(
            "e1",
            owner,
            "root",
            "holder",
            {"role": "user", "content": "hello"},
            command=("owner", "c1"),
        )
    assert len(db.get_messages("root")) == 1
    db.close()


def test_atomic_tool_amendment_failure_preserves_row_and_pending_receipt(tmp_path):
    db = database(tmp_path)
    owner = open_input(db)
    db.native_command_apply(
        "e1",
        owner,
        "root",
        "holder",
        {"role": "user", "content": "hello"},
        command=("owner", "c1"),
    )
    db.native_command_admit("owner", body("s1", "send", "correction"), effect="steer")
    tool_id = db.append_message(
        "root", "tool", "before", tool_call_id="call", turn_lease_holder="holder"
    )
    message = {"role": "tool", "content": "before", "_row_id": tool_id}
    db._execute_write(
        lambda c: c.execute(
            "CREATE TRIGGER reject_application BEFORE UPDATE OF phase ON native_commands WHEN NEW.phase='applied' BEGIN SELECT RAISE(ABORT,'synthetic storage failure'); END"
        )
    )
    with pytest.raises(sqlite3.DatabaseError):
        db.native_command_apply("e1", owner, "root", "holder", message)
    assert message["content"] == "before"
    assert db.get_messages("root")[-1]["content"] == "before"
    assert db.native_command_lookup("owner", "s1")["phase"] == "steer"
    db.close()


@pytest.mark.parametrize("applied", [False, True])
def test_same_pid_replacement_reconciles_only_after_live_lease_releases(
    tmp_path, applied
):
    db = database(tmp_path)
    dead_instance = f"pid={os.getpid()}:birth=0:native=previous-container"
    owner = open_input(db, owner=dead_instance)
    if applied:
        db.native_command_apply(
            "e1",
            owner,
            "root",
            "holder",
            {"role": "user", "content": "hello"},
            command=("owner", "c1"),
        )
    # Same PID is deliberately alive, but its recorded process birth differs.
    assert db.native_execution_reconcile("root") is False
    assert db.native_execution("root") is not None
    db.release_session_turn_lease("root", "holder")
    db.close()
    reopened = SessionDB(tmp_path / "state.db")
    assert reopened.native_execution_reconcile("root")
    assert reopened.native_execution("root") is None
    recovered = reopened.native_command_lookup("owner", "c1")
    assert recovered["phase"] == ("applied" if applied else "queued")
    assert recovered["resulting_execution_id"] == "e1"
    assert len(reopened.get_messages("root")) == int(applied)
    assert reopened.native_execution_reconcile("root") is False
    reopened.close()


def test_live_same_process_instance_and_stale_generation_are_fenced(tmp_path):
    db = database(tmp_path)
    owner = open_input(db)
    db.release_session_turn_lease("root", "holder")
    assert not db.native_execution_reconcile("root")
    with pytest.raises(CommandConflict):
        db.native_execution_open("root", "e2", owner)
    assert not db.native_execution_close("e1", "different-generation")
    assert db.native_execution_close("e1", owner)
    db.native_execution_open("root", "e2", owner)
    assert not db.native_execution_close("e1", owner)
    assert db.native_execution("root")["execution_id"] == "e2"
    db.close()


def test_concurrent_close_and_steer_have_one_known_disposition(tmp_path):
    db = database(tmp_path)
    owner = open_input(db)
    db.native_command_apply(
        "e1",
        owner,
        "root",
        "holder",
        {"role": "user", "content": "hello"},
        command=("owner", "c1"),
    )
    sibling = SessionDB(tmp_path / "state.db")
    with ThreadPoolExecutor(max_workers=2) as pool:
        accept = pool.submit(
            db.native_command_admit, "owner", body("s1", "send", "tail"), effect="queue"
        )
        close = pool.submit(sibling.native_execution_close, "e1", owner)
        accept.result()
        close.result()
    row = db.native_command_lookup("owner", "s1")
    assert row["phase"] == "queued"
    assert row["effective_action"] == "queue"
    assert row["resulting_execution_id"] is None
    assert len(db.native_command_rows("root", phase="queued")) == 1
    assert not db.native_execution_close("e1", owner)
    sibling.close()
    db.close()


def test_concurrent_same_id_and_scope_conflicts(tmp_path):
    db = database(tmp_path)
    sibling = SessionDB(tmp_path / "state.db")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda d: d.native_command_admit("owner", body(), effect="start"),
                [db, sibling],
            )
        )
    assert sum(created for _, created in results) == 1
    with pytest.raises(CommandConflict):
        sibling.native_command_admit("owner", body(text="different"), effect="start")
    row, created = sibling.native_command_admit("foreign", body(), effect="queue")
    assert created and row["scope"] == "foreign"
    sibling.close()
    db.close()


def test_schema_upgrade_preserves_history_and_journal_survives_reopen(tmp_path):
    db = database(tmp_path)
    db.append_message("root", "user", "existing transcript")

    # Simulate an existing v26 store, which had neither journal table.
    def legacy(conn):
        conn.execute("DROP TABLE native_commands")
        conn.execute("DROP TABLE native_executions")
        conn.execute("UPDATE schema_version SET version=26")

    db._execute_write(legacy)
    db.close()
    db = SessionDB(tmp_path / "state.db")
    row, created = db.native_command_admit("owner", body(), effect="start")
    db.close()
    db = SessionDB(tmp_path / "state.db")
    replay, created = db.native_command_admit("owner", body(), effect="queue")
    assert not created and receipt(replay) == receipt(row)
    assert [m["content"] for m in db.get_messages("root")] == ["existing transcript"]
    db.close()


def test_old_receipts_do_not_take_over_later_native_resume(tmp_path):
    db = database(tmp_path)
    owner = open_input(db)
    db.native_command_apply(
        "e1",
        owner,
        "root",
        "holder",
        {"role": "user", "content": "hello"},
        command=("owner", "c1"),
    )
    assert db.native_execution_command_managed("root")
    db.native_execution_close("e1", owner)
    db.native_execution_open("root", "telegram-later", owner)
    db.native_command_apply(
        "telegram-later",
        owner,
        "root",
        "holder",
        {"role": "user", "content": "native next"},
    )
    db.native_execution_close("telegram-later", owner, crashed=True)
    assert db.native_command_lookup("owner", "c1")["phase"] == "applied"
    assert not db.native_execution_command_managed("root")
    db.close()


def test_recovered_same_id_after_interposed_native_execution_is_latest(tmp_path):
    db = database(tmp_path)
    open_input(db, owner=f"pid={os.getpid()}:birth=0:native=old")
    db.release_session_turn_lease("root", "holder")
    assert db.native_execution_reconcile("root")
    live = process_owner()
    db.native_execution_open("root", "telegram-interposed", live)
    db.native_execution_close("telegram-interposed", live)
    assert not db.native_execution_command_managed("root")
    db.native_execution_open("root", "e1", live, command=("owner", "c1"))
    assert db.try_acquire_session_turn_lease("root", "holder")
    db.native_command_apply(
        "e1",
        live,
        "root",
        "holder",
        {"role": "user", "content": "hello"},
        command=("owner", "c1"),
    )
    db.native_execution_close("e1", live, crashed=True)
    assert db.native_execution_command_managed("root")
    assert db.native_command_lookup("owner", "c1")["resulting_execution_id"] == "e1"
    db.close()
