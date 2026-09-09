"""Actual native runner controls with synthetic model work, no remote acceptance claim."""

import asyncio

import pytest

from tests.gateway.test_native_cancellation import initialized_interrupt_state, settled
from tests.gateway.test_native_events import event_setup, provider_loop
from tests.gateway.test_native_commands import command, started, finish, JournalAgent
from tests.gateway.test_conversation_control import OWNER
from hermes_state_commands import CommandConflict
from gateway.native_redirect import validate_control


@pytest.mark.parametrize("missing", ["schema_version", "command_id", "conversation_id", "target_execution_id", "payload"])
def test_versioned_redirect_still_requires_every_control_field(missing):
    body = command(
        "root", "new direction", "redirect", kind="redirect",
        target_execution_id="execution", expected_model_version=1,
    )
    del body[missing]
    with pytest.raises(ValueError, match="unsupported control fields"):
        validate_control(body)


@pytest.mark.parametrize("expected", [False, 0, 9007199254740992, "1"])
def test_redirect_rejects_invalid_model_version_precondition(expected):
    body = command(
        "root",
        "new direction",
        "redirect",
        kind="redirect",
        target_execution_id="execution",
        expected_model_version=expected,
    )
    with pytest.raises(ValueError, match="expected model version"):
        validate_control(body)


@pytest.mark.asyncio
async def test_redirect_model_version_fence_replays_before_current_check(
    monkeypatch, tmp_path
):
    runner, ingress, db, store, entry, source, adapter = event_setup(
        monkeypatch, tmp_path
    )
    db.native_model_initialize(entry.session_id, "daily")
    try:
        await ingress.submit(OWNER, command(entry.session_id))
        await started()
        execution = ingress._owner(entry.session_id)[2]
        request = command(
            entry.session_id,
            "new direction",
            "redirect-versioned",
            kind="redirect",
            target_execution_id=execution.execution_id,
            expected_model_version=1,
        )
        accepted = await ingress.submit(OWNER, request)
        db.native_model_mutate(
            "independent-model-scope",
            {
                "schema_version": "1.0",
                "mutation_id": "switch",
                "conversation_id": entry.session_id,
                "model_id": "deep",
                "expected_model_version": 1,
            },
        )
        replay = await ingress.submit(OWNER, request)
        assert replay["command_id"] == accepted["command_id"]
        assert replay["receipt_kind"] == "redirect"

        stale = {**request, "command_id": "redirect-stale"}
        with pytest.raises(CommandConflict, match="model version changed"):
            await ingress.submit(OWNER, stale)
        assert (
            db.native_control_lookup(
                ingress._command_scope(OWNER), "redirect-stale"
            )
            is None
        )
    finally:
        JournalAgent.gates[0].set()
        ingress.controls.stop()
        await finish(ingress)
        db.close()


@pytest.mark.asyncio
async def test_redirect_actual_writer_fifo_and_no_public_internal_receipts(
    monkeypatch, tmp_path
):
    runner, ingress, db, store, entry, source, adapter = event_setup(
        monkeypatch, tmp_path
    )

    def loop(agent, *args, **kwargs):
        result = provider_loop(agent, *args, **kwargs)
        return {
            **result,
            "interrupted": bool(agent._interrupt_requested),
            "completed": not agent._interrupt_requested,
        }

    monkeypatch.setattr("agent.conversation_loop.run_conversation", loop)
    try:
        await ingress.submit(OWNER, command(entry.session_id))
        first = await started()
        execution = ingress._owner(entry.session_id)[2]
        await ingress.submit(
            OWNER, command(entry.session_id, "queued", "q1", kind="queue")
        )
        request = command(
            entry.session_id,
            "new direction",
            "r1",
            kind="redirect",
            target_execution_id=execution.execution_id,
        )
        accepted = await ingress.submit(OWNER, request)
        assert (
            accepted["receipt_kind"] == "redirect"
            and accepted["native_release"] == "pending"
        )
        assert first._interrupt_requested
        assert ingress._owner(entry.session_id)[2] == execution
        assert JournalAgent.effects == ["first"]
        JournalAgent.gates[0].set()
        newer = await started()
        assert not newer._interrupt_requested
        # Reconciliation sees the old lease actually released and appends once
        # after the already retained queued input.
        await ingress.controls._reconcile()
        duplicate = await ingress.submit(OWNER, request)
        assert duplicate["receipt_kind"] == "redirect"
        assert duplicate["release_basis"]["remote_gate"] == "closed_observed_frontier"
        assert duplicate["new_direction"]["application_state"] == "pending"
        JournalAgent.gates[1].set()
        await started()
        receipt = await ingress.command_receipt(OWNER, "r1")
        assert receipt["receipt_kind"] == "redirect"
        assert receipt["new_direction"]["application_state"] == "applied"
        JournalAgent.gates[2].set()
        await finish(ingress)
        assert JournalAgent.effects == ["first", "queued", "new direction"]
        assert (
            len([
                message
                for message in db.get_messages(entry.session_id)
                if message["role"] == "user"
            ])
            == 3
        )
    finally:
        ingress.controls.stop()
        await settled(ingress.cancellations)
        await finish(ingress)
        db.close()
