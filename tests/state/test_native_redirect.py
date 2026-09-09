"""Redirect gates compose native interrupt and input commits under one identity."""

from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3

import pytest

from hermes_state_commands import CommandConflict
from tests.state.test_native_commands import database, body, open_input
from tests.state.test_native_cancellation import cancel


def redirect(identity="r1"):
    return {**body(identity, "redirect", "new direction"), "target_execution_id": "e1"}


def confirm(revision=1, identity="confirm-1"):
    return {
        **redirect(identity),
        "type": "redirect_confirm",
        "payload": {
            "redirect_command_id": "r1",
            "confirmation_revision": revision,
            "allow_unresolved_remote": True,
        },
    }


def release(db, owner):
    db.native_execution_close("e1", owner, outcome="interrupted")
    db.release_session_turn_lease("root", "holder")


def test_confirmation_never_releases_live_writer_and_fifo_is_retained(tmp_path):
    db = database(tmp_path)
    owner = open_input(db)
    first, created = db.native_redirect_admit("owner", redirect())
    assert created and db.native_redirect_admit("owner", redirect()) == (first, False)
    db.native_redirect_confirm("owner", confirm())
    db.native_command_admit("owner", body("queued", "queue"), effect="queue")
    assert not db.native_redirect_progress("owner", "r1", frontier="uncertain")
    assert db.native_command_lookup("owner", "r1") is None
    release(db, owner)
    assert db.native_redirect_progress("owner", "r1", frontier="uncertain")
    snapshot = db.native_control_snapshot("owner", "r1")
    basis = json.loads(snapshot["redirect"]["release_basis_json"])
    assert basis == dict(
        remote_gate="explicit_confirmation",
        known_count_at_release=0,
        confirmation_revision=1,
    )
    assert (
        snapshot["input"]["queue_order"]
        > db.native_command_lookup("owner", "queued")["queue_order"]
    )
    assert not db.native_redirect_progress("owner", "r1", frontier="closed")
    assert (
        db.native_control_snapshot("owner", "r1")["redirect"]["release_basis_json"]
        == snapshot["redirect"]["release_basis_json"]
    )
    assert not db.get_messages("root")
    db.close()


def test_closed_frontier_releases_without_confirmation_but_stale_token_fenced(tmp_path):
    db = database(tmp_path)
    owner = open_input(db)
    db.native_redirect_admit("owner", redirect())
    db.native_redirect_progress("owner", "r1", frontier="closed", release=False)
    with pytest.raises(CommandConflict):
        db.native_redirect_confirm("owner", confirm())
    db.native_redirect_progress("owner", "r1", frontier="uncertain", release=False)
    with pytest.raises(CommandConflict):
        db.native_redirect_confirm("owner", confirm())
    assert (
        db.native_control_snapshot("owner", "r1")["redirect"]["confirmation_revision"]
        == 2
    )
    release(db, owner)
    assert db.native_redirect_progress("owner", "r1", frontier="closed")
    state = db.native_control_snapshot("owner", "r1")["redirect"]
    assert (
        json.loads(state["release_basis_json"])["remote_gate"]
        == "closed_observed_frontier"
    )
    assert not state["confirmation_required"]
    db.close()


def test_cross_kind_simultaneous_admission_and_private_composition(tmp_path):
    db = database(tmp_path)
    open_input(db)

    def admit(kind):
        try:
            if kind == "redirect":
                db.native_redirect_admit("owner", redirect("same"))
            elif kind == "cancel":
                db.native_cancel_admit("owner", cancel(identity="same"))
            else:
                db.native_command_admit("owner", body("same"), effect="queue")
            return "accepted"
        except CommandConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=3) as pool:
        assert sorted(pool.map(admit, ["redirect", "cancel", "input"])) == [
            "accepted",
            "conflict",
            "conflict",
        ]
    db.close()


def test_failed_release_transaction_is_retryable_without_duplicate_input(tmp_path):
    db = database(tmp_path)
    owner = open_input(db)
    db.native_redirect_admit("owner", redirect())
    release(db, owner)
    db._execute_write(
        lambda c: c.execute(
            "CREATE TRIGGER fail_release BEFORE UPDATE OF release_basis_json ON native_redirect_state WHEN NEW.release_basis_json IS NOT NULL BEGIN SELECT RAISE(ABORT,'storage failure'); END"
        )
    )
    with pytest.raises(sqlite3.DatabaseError):
        db.native_redirect_progress("owner", "r1", frontier="closed")
    assert db.native_command_lookup("owner", "r1") is None
    db._execute_write(lambda c: c.execute("DROP TRIGGER fail_release"))
    assert db.native_redirect_progress("owner", "r1", frontier="closed")
    assert not db.native_redirect_progress("owner", "r1", frontier="closed")
    db.close()
