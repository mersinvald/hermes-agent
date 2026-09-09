"""Transactional behavior for native managed-model selection."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from hermes_state import SessionDB
from hermes_state_commands import CommandConflict, process_owner


def database(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("root", "telegram")
    return db


def mutation(id, model, expected):
    return {
        "schema_version": "1.0",
        "mutation_id": id,
        "conversation_id": "root",
        "model_id": model,
        "expected_model_version": expected,
    }


def command(id, expected):
    return {
        "schema_version": "1.0",
        "command_id": id,
        "conversation_id": "root",
        "type": "send",
        "payload": {"text": "hello"},
        "expected_model_version": expected,
    }


def open_execution(db, id="e1", holder="holder"):
    owner = process_owner()
    db.native_execution_open("root", id, owner, origin="pwa")
    assert db.try_acquire_session_turn_lease("root", holder)
    return owner


def test_default_initialization_is_atomic_across_database_handles(tmp_path):
    first = database(tmp_path)
    second = SessionDB(tmp_path / "state.db")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda db: db.native_model_initialize("root", "daily"),
                (first, second),
            )
        )
    assert results == [
        {"model_id": "daily", "model_version": 1},
        {"model_id": "daily", "model_version": 1},
    ]
    second.close()
    first.close()


def test_mutation_cas_has_immutable_durable_replay(tmp_path):
    db = database(tmp_path)
    first = mutation("m1", "daily", 0)
    assert db.native_model_mutate("principal", first) == {
        "schema_version": "1.0",
        "mutation_id": "m1",
        "conversation_id": "root",
        "outcome": "applied",
        "model": {"model_id": "daily", "model_version": 1},
    }
    db.close()

    db = SessionDB(tmp_path / "state.db")
    assert db.native_model_mutate(
        "principal", mutation("m2", "deep", 1)
    )["model"] == {"model_id": "deep", "model_version": 2}
    replay = db.native_model_mutate("principal", first)
    assert replay["outcome"] == "replayed"
    assert replay["model"] == {"model_id": "daily", "model_version": 1}
    assert db.native_model_selection("root") == {
        "model_id": "deep",
        "model_version": 2,
    }
    with pytest.raises(CommandConflict):
        db.native_model_mutate("principal", mutation("m1", "other", 0))
    with pytest.raises(CommandConflict):
        db.native_model_mutate("principal", mutation("stale", "daily", 1))
    db.close()


def test_capture_validates_execution_owner_and_live_lease_atomically(tmp_path):
    db = database(tmp_path)
    owner = open_execution(db)
    for values in (
        ("missing", "root", owner, "holder"),
        ("e1", "different", owner, "holder"),
        ("e1", "root", "other-owner", "holder"),
        ("e1", "root", owner, "other-holder"),
    ):
        with pytest.raises(CommandConflict):
            db.native_model_capture(*values, "daily")
    assert db.native_model_selection("root") is None
    assert db.native_execution_model("e1") is None

    db._execute_write(
        lambda conn: conn.execute(
            "UPDATE session_turn_leases SET expires_at=0 WHERE conversation_id='root'"
        )
    )
    with pytest.raises(CommandConflict):
        db.native_model_capture("e1", "root", owner, "holder", "daily")
    assert db.native_model_selection("root") is None
    db.close()


def test_execution_capture_is_immutable_while_later_execution_uses_latest(tmp_path):
    db = database(tmp_path)
    owner = open_execution(db)
    assert db.native_model_capture(
        "e1", "root", owner, "holder", "daily"
    ) == {"model_id": "daily", "model_version": 1}
    db.native_model_mutate("principal", mutation("m1", "deep", 1))
    assert db.native_model_capture(
        "e1", "root", owner, "holder", "daily"
    ) == {"model_id": "daily", "model_version": 1}
    assert db.native_execution_model("e1") == {
        "model_id": "daily",
        "model_version": 1,
    }

    db.release_session_turn_lease("root", "holder")
    assert db.native_execution_close("e1", owner)
    db.native_execution_open("root", "e2", owner, origin="pwa")
    assert db.try_acquire_session_turn_lease("root", "next-holder")
    assert db.native_model_capture(
        "e2", "root", owner, "next-holder", "daily"
    ) == {"model_id": "deep", "model_version": 2}
    db.close()


def test_command_admission_checks_current_model_version_in_write_transaction(tmp_path):
    db = database(tmp_path)
    sibling = SessionDB(tmp_path / "state.db")
    db.native_model_mutate("principal", mutation("m1", "daily", 0))

    with ThreadPoolExecutor(max_workers=2) as pool:
        admit = pool.submit(
            db.native_command_admit,
            "principal",
            command("c1", 1),
            effect="start",
        )
        change = pool.submit(
            sibling.native_model_mutate,
            "principal",
            mutation("m2", "deep", 1),
        )
        admitted = None
        try:
            admitted = admit.result()
        except CommandConflict:
            pass
        change.result()

    # Either admission serialized first while version 1 was current, or the
    # mutation serialized first and fenced it. A stale command cannot appear
    # after version 2 is committed.
    row = db.native_command_lookup("principal", "c1")
    assert (admitted is not None) == (row is not None)
    with pytest.raises(CommandConflict):
        db.native_command_admit(
            "principal", command("c2", 1), effect="start"
        )
    sibling.close()
    db.close()
