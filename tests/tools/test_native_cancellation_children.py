"""Native copied provenance and warn-only managed children, with real registry."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from agent.native_execution_context import (
    NativeExecutionOrigin,
    native_execution_scope,
    native_origin_record,
)
from tools import async_delegation as ad
from tools.thread_context import propagate_context_to_thread
from tests.tools.test_async_delegation import (
    _clean_state,
    _fast_stale_monitor,
    _drain_for,
)
from tests.tools.test_delegate import _make_mock_parent


@pytest.mark.parametrize("in_tool", [False, True])
def test_managed_async_child_warns_without_interrupt_or_force_finalize(
    monkeypatch, in_tool
):
    _fast_stale_monitor(monkeypatch, idle=0.02, in_tool=0.02, grace=0.02)
    gate = threading.Event()
    warned = threading.Event()
    original_warning = ad.logger.warning

    def warning(message, *args, **kwargs):
        if message.startswith("Managed native child"):
            warned.set()
        return original_warning(message, *args, **kwargs)

    monkeypatch.setattr(ad.logger, "warning", warning)
    origin = NativeExecutionOrigin("root", "e1", "owner")
    seen, interrupts = [], []

    def run():
        seen.append(native_origin_record())
        assert gate.wait(10)
        return {"status": "completed", "summary": "actual worker return"}

    try:
        with native_execution_scope(origin):
            result = ad.dispatch_async_delegation(
                goal="synthetic",
                context=None,
                toolsets=None,
                role="leaf",
                model="synthetic",
                session_key="same-route",
                runner=run,
                interrupt_fn=lambda: interrupts.append(True),
                progress_fn=lambda: ((0, "terminal" if in_tool else None), in_tool),
            )
        assert result["status"] == "dispatched" and warned.wait(5)
        time.sleep(0.1)  # Exceeds both deliberately tiny stale/grace test windows.
        assert interrupts == [] and ad.active_count() == 1
        assert seen == [origin.record()]
        assert ad._records[result["delegation_id"]]["status"] == "running"
        assert "owner" not in json.dumps(ad.list_async_delegations())
        with ad._DB_LOCK, ad._transaction() as conn:
            payload = json.loads(
                conn.execute(
                    "SELECT task_json FROM async_delegations WHERE delegation_id=?",
                    (result["delegation_id"],),
                ).fetchone()[0]
            )
        assert payload["_native_execution_origin"] == origin.record()
        gate.set()
        event = _drain_for(result["delegation_id"])
        assert event["status"] == "completed" and ad.active_count() == 0
    finally:
        gate.set()


def test_execution_interrupt_never_uses_route_alias_or_releases_live_children():
    origin = NativeExecutionOrigin("root", "e1", "owner")
    with ad._records_lock:
        calls = []
        for identity, record_origin in (
            ("own", origin.record()),
            ("newer", {**origin.record(), "execution_id": "e2"}),
            ("foreign", {**origin.record(), "owner": "other"}),
            ("unknown", None),
        ):
            ad._records[identity] = dict(
                delegation_id=identity,
                session_key="same-route",
                status="running",
                _native_execution_origin=record_origin,
                interrupt_fn=lambda key=identity: calls.append(key),
            )
    assert ad.interrupt_for_native_execution(origin.record()) == 1
    assert calls == ["own"] and ad.active_count() == 4
    assert ad.interrupt_for_native_execution(origin.record()) == 0
    assert all(record["status"] == "running" for record in ad._records.values())
    # No fake worker was started by this selection-only test.
    with ad._records_lock:
        ad._records.clear()


def test_context_copy_is_exact_per_thread_and_unmanaged_default_is_empty():
    origin = NativeExecutionOrigin("root", "e1", "owner")
    with ThreadPoolExecutor(max_workers=2) as pool:
        with native_execution_scope(origin):
            captured = pool.submit(propagate_context_to_thread(native_origin_record))
        unmanaged = pool.submit(native_origin_record)
        assert captured.result() == origin.record()
        assert unmanaged.result() is None
    assert native_origin_record() is None


def test_managed_sync_child_keeps_parent_heartbeat_after_stale_threshold(monkeypatch):
    from tools import delegate_tool

    monkeypatch.setattr(delegate_tool, "_HEARTBEAT_INTERVAL", 0.01)
    monkeypatch.setattr(delegate_tool, "_HEARTBEAT_STALE_CYCLES_IDLE", 1)
    parent = _make_mock_parent()
    touched = []
    enough = threading.Event()

    def touch(description):
        touched.append(description)
        if len(touched) >= 5:
            enough.set()

    parent._touch_activity = touch
    child = MagicMock()
    child.get_activity_summary.return_value = dict(
        current_tool=None, api_call_count=1, max_iterations=10, last_activity_ts=1
    )

    def run(**kwargs):
        assert enough.wait(5)
        return {"final_response": "done", "completed": True, "api_calls": 1}

    child.run_conversation.side_effect = run
    with native_execution_scope(NativeExecutionOrigin("root", "e1", "owner")):
        result = delegate_tool._run_single_child(0, "synthetic", child, parent)
    assert result["status"] == "completed" and len(touched) >= 5


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("race", [False, True])
def test_accepted_cancel_prevents_late_async_child_start(monkeypatch, batch, race):
    from types import SimpleNamespace

    calls, queued = [], []
    canceled = [not race]
    controller = SimpleNamespace(
        db=SimpleNamespace(
            native_execution_cancel_requested=lambda execution: canceled[0]
        )
    )
    monkeypatch.setattr(
        ad,
        "_get_executor",
        lambda limit: SimpleNamespace(submit=lambda fn: queued.append(fn)),
    )
    function = (
        ad.dispatch_async_delegation_batch if batch else ad.dispatch_async_delegation
    )
    kwargs = {"goals": ["late"]} if batch else {"goal": "late"}
    with native_execution_scope(
        NativeExecutionOrigin("root", "e1", "owner"), controller
    ):
        result = function(
            **kwargs,
            context=None,
            toolsets=None,
            role="leaf",
            model="synthetic",
            session_key="route",
            runner=lambda: calls.append("started"),
        )
    if race:
        assert result["status"] == "dispatched"
        canceled[0] = True
        queued[0]()  # Actual registry worker sees cancellation after submission.
        event = _drain_for(result["delegation_id"])
        assert event["status"] == "interrupted"
    else:
        assert result["status"] == "rejected" and not queued
    assert not calls and ad.active_count() == 0
