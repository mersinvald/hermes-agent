"""Actual native writer/FIFO and BasePlatformAdapter drain interoperability."""

import asyncio
import threading
from dataclasses import replace

import pytest

from gateway.platforms.base import MessageEvent
from gateway.telegram_conversations import TelegramConversationChannel
from tests.gateway.test_conversation_control import OWNER
from tests.gateway.test_native_commands import JournalAgent, command, finish, started
from tests.gateway.test_native_events import event_setup


async def wait_until(predicate):
    async with asyncio.timeout(8):
        while not predicate():
            await asyncio.sleep(0.01)


def configured(monkeypatch, tmp_path):
    native = event_setup(monkeypatch, tmp_path)
    runner, ingress, db, store, entry, source, adapter = native
    adapter.config.typing_indicator = False
    adapter.set_message_handler(runner._handle_message)
    adapter.set_session_store(store)
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    TelegramConversationChannel(ingress)
    return native


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_before_input", [False, True])
@pytest.mark.parametrize("hold_adapter_cleanup", [False, True])
async def test_actual_pwa_owner_leaves_one_telegram_followup_for_native_handoff(
    monkeypatch, tmp_path, fail_before_input, hold_adapter_cleanup
):
    runner, ingress, db, store, entry, source, adapter = configured(monkeypatch, tmp_path)
    reserved, release = asyncio.Event(), asyncio.Event()
    cleanup_entered, cleanup_release = asyncio.Event(), asyncio.Event()
    handle = runner._handle_message_with_agent
    stop_typing = adapter._stop_typing_refresh

    async def delayed_cleanup(*args, **kwargs):
        await stop_typing(*args, **kwargs)
        if hold_adapter_cleanup and ingress._deferred_adapter_drains and not cleanup_entered.is_set():
            cleanup_entered.set()
            await cleanup_release.wait()

    async def gated(*args, **kwargs):
        if args[0].text == "first":
            reserved.set()
            await release.wait()
            if fail_before_input:
                raise RuntimeError("synthetic pre-input failure")
        return await handle(*args, **kwargs)

    monkeypatch.setattr(runner, "_handle_message_with_agent", gated)
    monkeypatch.setattr(adapter, "_stop_typing_refresh", delayed_cleanup)
    try:
        await ingress.submit(OWNER, command(entry.session_id, "first"))
        await asyncio.wait_for(reserved.wait(), 5)
        owner_key, _, execution = ingress._owner(entry.session_id)
        event = MessageEvent(text="queued telegram", source=replace(source))
        await adapter.handle_message(event)
        await wait_until(lambda: owner_key in ingress._deferred_adapter_drains)
        if hold_adapter_cleanup:
            await asyncio.wait_for(cleanup_entered.wait(), 5)
        else:
            await wait_until(lambda: owner_key not in adapter._session_tasks)
        pending = adapter._pending_messages[owner_key]
        assert pending.text == "queued telegram"
        assert not JournalAgent.effects
        # Several loop passes cannot respawn adapter work behind this writer.
        for _ in range(5):
            await asyncio.sleep(0)
            assert adapter._pending_messages[owner_key] is pending
            assert bool(owner_key in adapter._session_tasks) == hold_adapter_cleanup
        ingress.release(owner_key, execution.generation + 1)
        assert ingress._owner(entry.session_id)[2] is execution
        assert adapter._pending_messages[owner_key] is pending
        release.set()
        if hold_adapter_cleanup and fail_before_input:
            await wait_until(lambda: ingress._owner(entry.session_id) is None and not ingress._command_wakeups)
            assert adapter._pending_messages[owner_key] is pending
            assert not JournalAgent.effects
            assert not adapter._session_tasks[owner_key].done()
        cleanup_release.set()
        await started()
        JournalAgent.gates[0].set()
        if not fail_before_input:
            await started()
            JournalAgent.gates[1].set()
        await finish(ingress)
        await wait_until(lambda: not adapter._session_tasks)
        assert JournalAgent.effects == (
            ["queued telegram"] if fail_before_input else ["first", "queued telegram"]
        )
        assert [m["content"] for m in db.get_messages(entry.session_id) if m["role"] == "user"] == JournalAgent.effects
        assert not adapter._pending_messages
    finally:
        release.set()
        cleanup_release.set()
        await finish(ingress)
        for task in tuple(adapter._background_tasks):
            await asyncio.gather(task, return_exceptions=True)
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("blocker", ["lease", "completion", "scheduling", "task_creation"])
async def test_reconciliation_preserves_pending_until_durable_and_adapter_handoff_safe(
    monkeypatch, tmp_path, blocker
):
    runner, ingress, db, store, entry, source, adapter = configured(monkeypatch, tmp_path)
    key = entry.session_key
    event = MessageEvent(text="retained telegram", source=replace(source))
    ingress.reserve(entry.session_id, source, key, 7, event)
    execution = ingress._owner(entry.session_id)[2]
    adapter._pending_messages[key] = event
    assert adapter._defer_native_pending_drain(key)
    holder = "synthetic-q01-held-turn"
    close = db.native_execution_close
    schedule = adapter._start_session_processing
    failure = [True]
    acquired, worker_release = asyncio.Event(), threading.Event()
    worker_task = None
    loop = asyncio.get_running_loop()

    def held_worker():
        assert db.acquire_session_turn_lease(entry.session_id, holder, wait_seconds=0)
        loop.call_soon_threadsafe(acquired.set)
        try:
            assert worker_release.wait(10)
        finally:
            db.release_session_turn_lease(entry.session_id, holder)

    def fail_close(*args, **kwargs):
        if failure[0]:
            raise RuntimeError("synthetic unknown completion persistence")
        return close(*args, **kwargs)

    def fail_schedule(*args, **kwargs):
        if failure[0]:
            if blocker == "task_creation":
                with monkeypatch.context() as patch:
                    def unavailable(_coroutine):
                        raise RuntimeError("synthetic create_task failure")
                    patch.setattr(asyncio, "create_task", unavailable)
                    return schedule(*args, **kwargs)
            raise RuntimeError("synthetic unavailable task scheduling")
        return schedule(*args, **kwargs)

    try:
        if blocker == "lease":
            worker_task = asyncio.create_task(asyncio.to_thread(held_worker))
            await asyncio.wait_for(acquired.wait(), 5)
            # Model input has started under a real lease. release must not close
            # this live worker just because the async runner is unwinding.
            db.native_command_apply(execution.execution_id, ingress._command_owner,
                                    entry.session_id, holder, {"role": "user", "content": "held"})
        elif blocker == "completion":
            ingress._completion_observations[execution.execution_id] = (
                execution.execution_id, ingress._command_owner, entry.session_id, "completed")
            monkeypatch.setattr(db, "native_execution_close", fail_close)
        else:
            monkeypatch.setattr(adapter, "_start_session_processing", fail_schedule)
        ingress.release(key, 7)
        assert ingress._owner(entry.session_id) is None
        for _ in range(3):
            ingress.reconcile_commands()
            await asyncio.sleep(0)
            assert adapter._pending_messages[key] is event
            assert not adapter._session_tasks
            assert not JournalAgent.effects
        failure[0] = False
        if blocker == "lease":
            worker_release.set()
            await worker_task
            close(execution.execution_id, ingress._command_owner, outcome="completed")
        # Use the actual native periodic recovery entry, including completion
        # retry; no facade or copied queue dispatcher wakes the input.
        ingress.reconcile_commands()
        await started()
        JournalAgent.gates[0].set()
        await finish(ingress)
        await wait_until(lambda: not adapter._session_tasks)
        assert JournalAgent.effects == ["retained telegram"]
        assert not adapter._pending_messages
    finally:
        failure[0] = False
        worker_release.set()
        if worker_task is not None:
            await worker_task
        await finish(ingress)
        for task in tuple(adapter._background_tasks):
            await asyncio.gather(task, return_exceptions=True)
        db.close()
