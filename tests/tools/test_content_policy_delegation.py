"""Native delegate and recovery boundaries with invented tasks and real ledger rows."""

import queue
import time
from dataclasses import replace

import pytest

from agent.content_policy import NativeContentPolicyDenied, content_policy_scope
from gateway.run import GatewayRunner
from tools import async_delegation as background, delegate_tool as delegate
from tests.agent.test_content_policy import enabled_agent, make_context
from tests.gateway.test_native_content_policy import setup_policy


@pytest.mark.parametrize("operation", ["spawn", "construct", "run", "finalize"])
def test_sensitive_delegation_denies_before_child_construction_or_result_processing(operation):
    context, _ = make_context(profiles=("restricted-documents",))
    parent = enabled_agent(context)
    callbacks = {
        "spawn": lambda: delegate.delegate_task(goal="invented", parent_agent=parent),
        "construct": lambda: delegate._build_child_agent(0, "invented", None, None, None, 1, 1, parent),
        "run": lambda: delegate._run_single_child(0, "invented", parent_agent=parent),
        "finalize": lambda: delegate._finalize_child_results([], [], [], parent),
    }
    with pytest.raises(NativeContentPolicyDenied):
        callbacks[operation]()


@pytest.mark.parametrize("batch", [False, True])
def test_sensitive_background_dispatch_never_schedules_runner(batch):
    context, state = make_context()
    calls = []
    callback = background.dispatch_async_delegation_batch if batch else background.dispatch_async_delegation
    kwargs = {"goals": ["invented"]} if batch else {"goal": "invented"}
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(context):
            state["binding"] = replace(state["binding"], profiles=("restricted-documents",))
            callback(**kwargs, context=None, toolsets=None, role="leaf", model=None,
                     session_key="invented", runner=lambda: calls.append("runner") or {})
            pytest.fail("background dispatch returned")
    assert calls == []


@pytest.mark.asyncio
async def test_real_completion_restore_does_not_restore_policy_authority(monkeypatch, tmp_path):
    runner, ingress, db, entry, source = setup_policy(monkeypatch, tmp_path)
    event = {"type": "async_delegation", "delegation_id": "invented-delegation",
             "parent_session_id": entry.session_id, "session_key": entry.session_key,
             "status": "completed", "completed_at": time.time()}
    record = {"delegation_id": event["delegation_id"], "session_key": entry.session_key,
              "parent_session_id": entry.session_id, "dispatched_at": time.time(), "goal": "invented"}
    monkeypatch.setattr(background, "_db_path", lambda: tmp_path / "delegations.db")
    background._persist_dispatch(record)
    background._persist_completion(event, {"summary": "invented result"})
    recovered = queue.Queue()
    try:
        assert background.restore_undelivered_completions(recovered) == 1
        restored = recovered.get_nowait()
        runner._build_process_event_source = lambda _: source
        assert await GatewayRunner._deliver_completion_notification(runner, "invented result", restored) is False
        retained = background.get_durable_delegation(event["delegation_id"])
        assert retained["delivery_state"] == "pending"
        assert retained["result"] == {"summary": "invented result"}
    finally:
        ingress.stop_command_recovery()
        db.close()
