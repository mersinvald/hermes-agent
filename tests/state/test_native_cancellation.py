"""Real SQLite control admission, task provenance and write-attempt fencing."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from agent.native_execution_context import NativeExecutionOrigin
from hermes_state_commands import CommandConflict
from plugins.platforms.a2a.cancellation import endpoint_fingerprint
from tests.state.test_native_commands import database, body, open_input


def cancel(root="root", execution="e1", identity="cancel-1"):
    return dict(
        schema_version="1.0",
        command_id=identity,
        conversation_id=root,
        type="cancel",
        target_execution_id=execution,
        payload={},
    )


def binding(peer="peer"):
    endpoint = f"http://gateway.test/agent/{peer}/"
    return dict(
        peer_name=peer,
        configured_endpoint_fingerprint=endpoint_fingerprint(endpoint),
        rpc_endpoint=endpoint,
        protocol_version="1.0",
        tenant="tenant",
        configured_tenant="",
    )


def prepare(db, origin, *, task=None, context=None, peer="peer"):
    return db.native_remote_dispatch_prepare(
        origin, binding(peer), "rpc-id", task_id=task, context_id=context
    )


def test_cancel_receipt_durable_duplicate_and_all_cross_kind_collisions(tmp_path):
    db = database(tmp_path)
    open_input(db)
    one, created = db.native_cancel_admit("owner", cancel())
    assert created and one["native_request_state"] == "pending"
    assert db.native_cancel_admit("owner", cancel()) == (one, False)
    for changed in (
        {**cancel(), "payload": {"reason": "different"}},
        cancel(execution="e2"),
        cancel(root="other"),
    ):
        with pytest.raises(CommandConflict):
            db.native_cancel_admit("owner", changed)
    with pytest.raises(CommandConflict):
        db.native_command_admit("owner", body(id="cancel-1"), effect="queue")
    with pytest.raises(CommandConflict):
        db.native_cancel_admit("owner", cancel(identity="c1"))
    assert db.native_cancel_admit("other-owner", cancel())[1]
    assert db.native_cancel_lookup("foreign", "cancel-1") is None
    db.close()


def test_simultaneous_cross_kind_admission_has_one_logical_id(tmp_path):
    db = database(tmp_path)
    open_input(db)

    def admit(kind):
        try:
            if kind == "cancel":
                db.native_cancel_admit("owner", cancel(identity="same"))
            else:
                db.native_command_admit("owner", body(id="same"), effect="queue")
            return "accepted"
        except CommandConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(admit, ("send", "cancel"))) == ["accepted", "conflict"]
    assert bool(db.native_cancel_lookup("owner", "same")) != bool(
        db.native_command_lookup("owner", "same")
    )
    db.close()


def test_actual_dispatch_only_lost_id_and_bounded_whole_execution_coverage(tmp_path):
    db = database(tmp_path)
    owner = open_input(db)
    origin = NativeExecutionOrigin("root", "e1", owner)
    identities = [prepare(db, origin) for _ in range(105)]
    db.native_remote_dispatch_observe(origin, identities[0], "task", "ctx", "working")
    db.native_cancel_admit("owner", cancel())
    snapshot = db.native_cancel_snapshot("owner", "cancel-1", limit=100)
    assert snapshot["known_count"] == 105 and snapshot["task_id_unknown_count"] == 104
    assert len(snapshot["targets"]) == 100 and snapshot["has_more"]
    following = db.native_cancel_snapshot(
        "owner", "cancel-1", after=snapshot["targets"][-1]["ordinal"], limit=100
    )
    assert len(following["targets"]) == 5 and not following["has_more"]
    assert db.native_cancel_snapshot("foreign", "cancel-1") is None
    with pytest.raises(CommandConflict, match="cancellation requested"):
        prepare(db, origin)
    with pytest.raises(CommandConflict):
        db.native_remote_cancel_reserve("owner", "cancel-1", identities[1])
    assert not db.get_messages("root")
    db.close()


def test_remote_claim_requires_actual_same_conversation_evidence_and_latest_owner(
    tmp_path,
):
    db = database(tmp_path)
    owner = open_input(db)
    first = NativeExecutionOrigin("root", "e1", owner)
    with pytest.raises(CommandConflict):
        prepare(db, first, task="forged", context="ctx")
    dispatch = prepare(db, first)
    db.native_remote_dispatch_observe(first, dispatch, "task", "ctx", "input-required")
    db.native_execution_close("e1", owner)
    db.native_execution_open("root", "e2", owner)
    second = replace(first, execution_id="e2")
    resumed = prepare(db, second, task="task", context="ctx")
    db.native_cancel_admit("owner", cancel())
    with pytest.raises(CommandConflict, match="superseded"):
        db.native_remote_cancel_reserve("owner", "cancel-1", dispatch)
    db.native_cancel_admit("owner", cancel(execution="e2", identity="cancel-2"))
    db.native_remote_cancel_reserve("owner", "cancel-2", resumed)
    db.create_session("other", "telegram")
    db.native_execution_open("other", "e3", owner)
    foreign = NativeExecutionOrigin("other", "e3", owner)
    with pytest.raises(CommandConflict, match="ownership"):
        prepare(db, foreign, task="task", context="ctx")
    untrusted = prepare(db, foreign)
    with pytest.raises(CommandConflict, match="ownership"):
        db.native_remote_dispatch_observe(foreign, untrusted, "task", "ctx", "working")
    with pytest.raises(CommandConflict):
        db.native_remote_cancel_reserve("owner", "cancel-2", dispatch)
    db.close()


def test_cancel_reservation_persists_uncertainty_across_restart_and_blocks_resume(
    tmp_path,
):
    from hermes_state import SessionDB

    db = database(tmp_path)
    owner = open_input(db)
    origin = NativeExecutionOrigin("root", "e1", owner)
    dispatch = prepare(db, origin)
    db.native_remote_dispatch_observe(origin, dispatch, "task", "ctx", "working")
    db.native_cancel_admit("owner", cancel())
    db.native_remote_cancel_reserve("owner", "cancel-1", dispatch)
    path = db.db_path
    db.close()
    db = SessionDB(path)
    target = db.native_remote_targets("e1")[0]
    assert (
        target["cancel_request_state"] == "unknown"
        and target["cancel_reason"] == "attempt_reserved"
    )
    with pytest.raises(CommandConflict, match="already reserved"):
        db.native_remote_cancel_reserve("owner", "cancel-1", dispatch)
    db.native_execution_close("e1", owner)
    db.native_execution_open("root", "e2", owner)
    with pytest.raises(CommandConflict, match="already requested"):
        prepare(db, replace(origin, execution_id="e2"), task="task", context="ctx")
    db.close()


def test_foreign_execution_owner_peer_context_and_failed_storage_cannot_claim(tmp_path):
    import sqlite3

    db = database(tmp_path)
    owner = open_input(db)
    origin = NativeExecutionOrigin("root", "e1", owner)
    with pytest.raises(CommandConflict):
        prepare(db, replace(origin, owner="foreign"))
    dispatch = prepare(db, origin)
    db.native_remote_dispatch_observe(origin, dispatch, "task", "ctx", "working")
    for args in ({"peer": "foreign"}, {"context": "foreign"}):
        with pytest.raises(CommandConflict):
            prepare(db, origin, task="task", **{"context": "ctx", **args})
    db._execute_write(
        lambda c: c.execute(
            "CREATE TRIGGER no_cancel BEFORE INSERT ON native_cancel_commands BEGIN SELECT RAISE(ABORT,'unavailable'); END"
        )
    )
    with pytest.raises(sqlite3.DatabaseError):
        db.native_cancel_admit("owner", cancel())
    assert db.native_cancel_lookup("owner", "cancel-1") is None
    db._execute_write(lambda c: c.execute("DROP TRIGGER no_cancel"))
    db.native_cancel_admit("owner", cancel())
    db._execute_write(
        lambda c: c.execute(
            "CREATE TRIGGER no_attempt BEFORE INSERT ON native_remote_cancel_attempts BEGIN SELECT RAISE(ABORT,'unavailable'); END"
        )
    )
    with pytest.raises(sqlite3.DatabaseError):
        db.native_remote_cancel_reserve("owner", "cancel-1", dispatch)
    assert db.native_remote_targets("e1")[0]["cancel_request_state"] is None
    db.close()


def test_control_event_storage_failure_rolls_back_admission_and_reservation(tmp_path):
    import sqlite3

    db = database(tmp_path)
    from hermes_state_events import EventLimits

    db.native_events_enable(EventLimits())
    owner = open_input(db)
    origin = NativeExecutionOrigin("root", "e1", owner)
    dispatch = prepare(db, origin)
    db.native_remote_dispatch_observe(origin, dispatch, "task", "ctx", "working")
    db._execute_write(
        lambda c: c.execute(
            "CREATE TRIGGER no_control BEFORE INSERT ON native_events WHEN json_extract(NEW.body,'$.type')='cancel_state_changed' BEGIN SELECT RAISE(ABORT,'control event unavailable'); END"
        )
    )
    with pytest.raises(sqlite3.DatabaseError):
        db.native_cancel_admit("owner", cancel())
    assert db.native_cancel_lookup("owner", "cancel-1") is None
    db._execute_write(lambda c: c.execute("DROP TRIGGER no_control"))
    db.native_cancel_admit("owner", cancel())
    db._execute_write(
        lambda c: c.execute(
            "CREATE TRIGGER no_attempt BEFORE INSERT ON native_events WHEN json_extract(NEW.body,'$.payload.change')='remote_request' BEGIN SELECT RAISE(ABORT,'attempt event unavailable'); END"
        )
    )
    with pytest.raises(sqlite3.DatabaseError):
        db.native_remote_cancel_reserve("owner", "cancel-1", dispatch)
    assert db.native_remote_targets("e1")[0]["cancel_request_state"] is None
    db._execute_write(lambda c: c.execute("DROP TRIGGER no_attempt"))
    db.native_remote_cancel_reserve("owner", "cancel-1", dispatch)
    assert db.native_remote_targets("e1")[0]["cancel_request_state"] == "unknown"
    db.close()


def test_fresh_explicit_retry_preserves_failed_attempt_and_atomic_alias_write_fence(
    tmp_path,
):
    from plugins.platforms.a2a.cancellation import ControlObservation

    db = database(tmp_path)
    owner = open_input(db)
    origin = NativeExecutionOrigin("root", "e1", owner)
    first = prepare(db, origin)
    db.native_remote_dispatch_observe(origin, first, "task", "ctx", "working")
    alias = prepare(db, origin, task="task", context="ctx")
    db.native_cancel_admit("owner", cancel())
    db.native_remote_cancel_no_write(
        "owner",
        "cancel-1",
        first,
        ControlObservation("failed", "unknown", "cancel_not_sent"),
    )
    with pytest.raises(CommandConflict):
        db.native_remote_cancel_reserve("owner", "cancel-1", first)
    db.native_cancel_admit("owner", cancel(identity="manual-2"))
    db.native_cancel_admit("owner", cancel(identity="manual-3"))

    def reserve(item):
        command_id, dispatch_id = item
        try:
            db.native_remote_cancel_reserve("owner", command_id, dispatch_id)
            return "reserved"
        except CommandConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(
            pool.map(reserve, [("manual-2", first), ("manual-3", alias)])
        ) == ["conflict", "reserved"]
    old = db.native_cancel_snapshot("owner", "cancel-1")
    assert old["targets"][0]["cancel_request_state"] == "failed"
    assert old["targets"][0]["cancel_command_id"] == "cancel-1"
    with db._read_ctx() as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM native_remote_cancel_attempts WHERE write_reserved=1"
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM native_remote_cancel_attempts"
            ).fetchone()[0]
            == 2
        )
    db.close()
