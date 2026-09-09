"""Actual blocking native entry, durable claim and waiter ACK without a browser lifetime."""

import asyncio
from types import SimpleNamespace

import pytest

from agent.native_execution_context import NativeExecutionOrigin, native_execution_scope
from gateway.native_clarification import NativeClarificationController
from hermes_state_commands import CommandConflict
from tests.state.test_native_commands import database, open_input
from tools import clarify_gateway


def setup(tmp_path):
    db = database(tmp_path)
    owner = open_input(db)
    ingress = SimpleNamespace(
        db=db, _command_owner=owner, cancellations=SimpleNamespace()
    )
    manager = NativeClarificationController(ingress)
    return db, manager, NativeExecutionOrigin("root", "e1", owner)


def answer(identity, command_id="a1", **changes):
    return dict(
        schema_version="1.0",
        command_id=command_id,
        conversation_id="root",
        type="clarification_response",
        target_execution_id="e1",
        payload=dict(
            clarification_id=identity,
            question_revision=1,
            answer=dict(kind="text", text="native answer"),
        ),
        **changes,
    )


def register(manager, origin, choices=None):
    with native_execution_scope(origin, manager.ingress.cancellations):
        return manager.register("Exact question?", choices, session_key="native-route")


def test_claim_recorded_before_actual_waiter_ack_and_duplicate_never_wakes_new_question(
    tmp_path,
):
    db, manager, origin = setup(tmp_path)
    entry = register(manager, origin)
    request = answer(entry.clarify_id)
    with clarify_gateway._lock:
        entry.managed.claim_answer(entry, "owner", request)
    assert db.native_control_lookup("owner", "a1")["answer_state"] == "recorded"
    assert db.native_question_lookup(entry.clarify_id)["state"] == "answer_recorded"
    assert clarify_gateway.wait_for_response(entry.clarify_id, 0.1) == "native answer"
    assert db.native_control_lookup("owner", "a1")["answer_state"] == "delivered"
    assert db.native_question_lookup(entry.clarify_id)["state"] == "answered"
    other = register(manager, origin)
    with pytest.raises(CommandConflict):
        other.managed.claim_answer(other, "owner", answer(other.clarify_id))
    assert not other.event.is_set()
    manager.cancel_execution(origin)
    clarify_gateway.wait_for_response(other.clarify_id, 0.1)
    db.close()


def test_telegram_selection_and_browser_claim_share_first_writer(tmp_path):
    db, manager, origin = setup(tmp_path)
    entry = register(manager, origin, ["One", "Two"])
    assert clarify_gateway.resolve_gateway_clarify(entry.clarify_id, "Two")
    assert not clarify_gateway.resolve_gateway_clarify(entry.clarify_id, "One")
    with pytest.raises(CommandConflict):
        entry.managed.claim_answer(entry, "browser", answer(entry.clarify_id))
    assert clarify_gateway.wait_for_response(entry.clarify_id, 0.1) == "Two"
    assert db.native_question_lookup(entry.clarify_id)["state"] == "answered"
    db.close()


@pytest.mark.asyncio
async def test_lost_waiter_reconciles_unknown_without_replay_and_expiry_is_not_answer(
    tmp_path,
):
    db, manager, origin = setup(tmp_path)
    entry = register(manager, origin)
    entry.managed.claim_answer(entry, "owner", answer(entry.clarify_id))
    with clarify_gateway._lock:
        clarify_gateway._entries.pop(entry.clarify_id)
    await manager.reconcile()
    assert db.native_control_lookup("owner", "a1")["answer_state"] == "unknown"
    assert db.native_question_lookup(entry.clarify_id)["state"] == "unknown"
    second = register(manager, origin)
    assert "expired" in await asyncio.to_thread(
        clarify_gateway.wait_for_response, second.clarify_id, 0.01
    )
    assert db.native_question_lookup(second.clarify_id)["state"] == "expired"
    db.close()


def test_other_revises_exact_native_question_before_text_and_fences_stale_choices(
    tmp_path,
):
    db, manager, origin = setup(tmp_path)
    entry = register(manager, origin, ["One", "Two"])
    stale = answer(entry.clarify_id)
    stale["payload"]["answer"] = dict(kind="selection", option_ids=["o1"])
    assert clarify_gateway.mark_awaiting_text(entry.clarify_id)
    row = db.native_question_lookup(entry.clarify_id)
    assert row["question_revision"] == 2
    assert clarify_gateway.resolve_managed_choice(entry.clarify_id, 0) is False
    with pytest.raises(CommandConflict):
        entry.managed.claim_answer(entry, "browser", stale)
    assert clarify_gateway.resolve_text_response_for_session(
        "native-route", "custom answer"
    )
    assert clarify_gateway.wait_for_response(entry.clarify_id, 0.1) == "custom answer"
    row = db.native_question_lookup(entry.clarify_id)
    assert row["state"] == "answered" and row["question_revision"] == 2
    db.close()


def test_choice_claim_winner_prevents_other_revision_change(tmp_path):
    db, manager, origin = setup(tmp_path)
    entry = register(manager, origin, ["One", "Two"])
    request = answer(entry.clarify_id)
    request["payload"]["answer"] = dict(kind="selection", option_ids=["o1"])
    with clarify_gateway._lock:
        entry.managed.claim_answer(entry, "browser", request)
    assert not clarify_gateway.mark_awaiting_text(entry.clarify_id)
    assert db.native_question_lookup(entry.clarify_id)["question_revision"] == 1
    assert clarify_gateway.wait_for_response(entry.clarify_id, 0.1) == "One"
    db.close()


def test_actual_bound_child_can_ask_after_root_release_but_foreign_handle_cannot(
    tmp_path,
):
    class Child:
        session_id = "actual-native-child"

    db, manager, origin = setup(tmp_path)
    child = Child()
    with native_execution_scope(origin, manager.ingress.cancellations):
        manager.bind_child(child)
    identity = child._native_clarification_binding[1]
    db.native_execution_close("e1", origin.owner, outcome="completed")
    db.release_session_turn_lease("root", "holder")
    with native_execution_scope(origin, manager.ingress.cancellations):
        with pytest.raises(CommandConflict):
            manager.register("forged", None, child_id="foreign-child")
        entry = manager.register("Actual child question", None, child_id=identity)
    assert manager.has_children(origin)
    body = answer(entry.clarify_id)
    with clarify_gateway._lock:
        entry.managed.claim_answer(entry, "browser", body)
    assert clarify_gateway.wait_for_response(entry.clarify_id, 0.1) == "native answer"
    assert db.native_question_lookup(entry.clarify_id)["origin"] == "native_child"
    manager.unbind_child(identity)
    assert not manager.has_children(origin)
    db.close()


@pytest.mark.asyncio
async def test_claim_commit_then_handoff_read_failure_recovers_exact_same_entry(
    tmp_path, monkeypatch
):
    db, manager, origin = setup(tmp_path)
    entry = register(manager, origin)
    original = db.native_question_lookup
    monkeypatch.setattr(
        db,
        "native_question_lookup",
        lambda *_: (_ for _ in ()).throw(OSError("storage read failed")),
    )
    with clarify_gateway._lock, pytest.raises(OSError):
        entry.managed.claim_answer(entry, "owner", answer(entry.clarify_id))
    assert not entry.event.is_set()
    assert db.native_control_lookup("owner", "a1")["answer_state"] == "recorded"
    monkeypatch.setattr(db, "native_question_lookup", original)
    await manager.reconcile()
    assert entry.event.is_set()
    assert clarify_gateway.wait_for_response(entry.clarify_id, 0.1) == "native answer"
    assert db.native_control_lookup("owner", "a1")["answer_state"] == "delivered"
    db.close()


@pytest.mark.asyncio
async def test_waiter_ack_storage_failure_never_returns_unrecorded_answer(
    tmp_path, monkeypatch
):
    db, manager, origin = setup(tmp_path)
    entry = register(manager, origin)
    entry.managed.claim_answer(entry, "owner", answer(entry.clarify_id))
    monkeypatch.setattr(
        db,
        "native_question_ack",
        lambda *_: (_ for _ in ()).throw(OSError("storage write failed")),
    )
    assert (
        clarify_gateway.wait_for_response(entry.clarify_id, 0.1)
        == "[clarification answer delivery is unresolved]"
    )
    await manager.reconcile()
    assert db.native_control_lookup("owner", "a1")["answer_state"] == "unknown"
    db.close()
