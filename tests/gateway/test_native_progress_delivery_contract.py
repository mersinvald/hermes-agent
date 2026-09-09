"""Native Telegram send/edit/delete contract, extending the parent regressions.

No manual delete stands in for cleanup: finish_activity owns registration and
the adapter's post-delivery callback owns deletion. Only Bot I/O is mocked.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from tests.gateway.test_telegram_status_update import _install_fake_telegram


@pytest_asyncio.fixture
async def delivery(monkeypatch):
    _install_fake_telegram(monkeypatch)
    from gateway import run
    from gateway.config import Platform, PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter
    clock = SimpleNamespace(value=100.0)
    monkeypatch.setattr(run, "time", SimpleNamespace(monotonic=lambda: clock.value))
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="synthetic-unused"))
    events = []
    async def capture(operation, **kwargs):
        mid = kwargs.get("message_id", 102 if kwargs.get("text") == "Final" else 101)
        events.append((operation, mid, kwargs.get("text", ""), clock.value))
        return SimpleNamespace(message_id=mid)
    async def send(**kw):
        return await capture("send", **kw)
    async def edit(**kw):
        return await capture("edit", **kw)
    async def delete(**kw):
        return await capture("delete", **kw)
    adapter._bot = SimpleNamespace(send_message=AsyncMock(side_effect=send),
        edit_message_text=AsyncMock(side_effect=edit), delete_message=AsyncMock(side_effect=delete),
        send_chat_action=AsyncMock())
    active = SimpleNamespace(value=True)
    ctx = SimpleNamespace(_status_adapter=adapter, _status_chat_id="424242",
        _status_thread_metadata=None, _loop_for_step=asyncio.get_running_loop(),
        _run_still_current=lambda: active.value, _cleanup_progress=True, _cleanup_msg_ids=[],
        session_key="test-dm", run_generation=1, source=SimpleNamespace(platform=Platform.TELEGRAM))
    return SimpleNamespace(turn=run.TurnRunner(None, ctx), adapter=adapter,
        ctx=ctx, clock=clock, events=events, active=active)


async def complete_future(future):
    return await asyncio.wait_for(asyncio.wrap_future(future), 3)


async def final_delivery(d):
    await d.turn.finish_activity()
    await d.adapter.send("424242", "Final", metadata={"notify": True})
    callback = d.adapter.pop_post_delivery_callback("test-dm", generation=1)
    if callback:
        await callback()


@pytest.mark.asyncio
async def test_native_adapter_one_bubble_and_automatic_cleanup(delivery):
    d = delivery
    for instant, label in [(100, "Dispatch"), (102, "Reading clock"), (104, "Clock result received")]:
        d.clock.value = instant
        d.turn._status_callback_sync("tool_activity", label)
        assert (await complete_future(d.turn._activity_future)).success
    await final_delivery(d)
    d.clock.value = 110
    d.turn._status_callback_sync("tool_activity", "Late update")
    assert [e[0] for e in d.events] == ["send", "edit", "edit", "send", "delete"]
    assert d.events[-1][1] == 101 and d.events[-2][1] == 102
    assert d.ctx._cleanup_msg_ids == []


@pytest.mark.asyncio
async def test_latest_tool_status_is_not_lost_behind_pending_delivery(delivery):
    d = delivery
    entered, release = asyncio.Event(), asyncio.Event()
    original = d.adapter._bot.send_message.side_effect
    async def blocked(**kwargs):
        entered.set()
        await release.wait()
        return await original(**kwargs)
    d.adapter._bot.send_message.side_effect = blocked
    d.turn._status_callback_sync("tool_activity", "Dispatch")
    future = d.turn._activity_future
    await asyncio.wait_for(entered.wait(), 2)
    d.clock.value = 103
    d.turn._status_callback_sync("tool_activity", "Intermediate")
    d.turn._status_callback_sync("tool_activity", "Reading clock")
    ending = asyncio.create_task(final_delivery(d))
    await asyncio.sleep(0)
    assert not ending.done() and not d.events
    release.set()
    await complete_future(future)
    await asyncio.wait_for(ending, 3)
    assert [e[2] for e in d.events[:3]] == ["Dispatch", "Reading clock", "Final"]
    assert [e[0] for e in d.events] == ["send", "edit", "send", "delete"]


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["handover", "timeout", "cancel"])
async def test_late_ack_deleted_without_entering_final_cleanup_snapshot(delivery, reason):
    d = delivery
    entered, release, deleted = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_send = d.adapter._bot.send_message.side_effect
    original_delete = d.adapter._bot.delete_message.side_effect
    async def blocked(**kwargs):
        entered.set()
        await release.wait()
        return await original_send(**kwargs)
    async def observe_delete(**kwargs):
        result = await original_delete(**kwargs)
        deleted.set()
        return result
    d.adapter._bot.send_message.side_effect = blocked
    d.adapter._bot.delete_message.side_effect = observe_delete
    d.turn._status_callback_sync("tool_activity", "Dispatch")
    future = d.turn._activity_future
    await asyncio.wait_for(entered.wait(), 2)
    if reason == "handover":
        d.active.value = False
    if reason == "cancel":
        ending = asyncio.create_task(d.turn.finish_activity())
        await asyncio.sleep(0)
        ending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await ending
    else:
        await d.turn.finish_activity(timeout=0.01)
    release.set()
    await complete_future(future)
    await asyncio.wait_for(deleted.wait(), 2)
    assert d.ctx._cleanup_msg_ids == []
    assert [e[0] for e in d.events] == ["send", "delete"]
    d.turn._status_callback_sync("tool_activity", "Late status")
    assert d.turn._activity_pending is None


@pytest.mark.asyncio
async def test_handover_does_not_delete_successor_bubble(delivery):
    from gateway.run import TurnRunner
    d = delivery
    entered, release = asyncio.Event(), asyncio.Event()
    sent = 0
    async def send(**kwargs):
        nonlocal sent
        sent += 1
        mid = 200 + sent
        if mid == 201:
            entered.set()
            await release.wait()
        d.events.append(("send", mid, kwargs["text"], d.clock.value))
        return SimpleNamespace(message_id=mid)
    d.adapter._bot.send_message.side_effect = send
    d.turn._status_callback_sync("tool_activity", "Old")
    old = d.turn._activity_future
    await asyncio.wait_for(entered.wait(), 2)
    d.active.value = False
    next_ctx = SimpleNamespace(**vars(d.ctx))
    next_ctx._run_still_current = lambda: True
    next_ctx._cleanup_msg_ids = []
    next_ctx.run_generation = 2
    successor = TurnRunner(None, next_ctx)
    successor._status_callback_sync("tool_activity", "New")
    await complete_future(successor._activity_future)
    release.set()
    await complete_future(old)
    assert [e[1] for e in d.events if e[0] == "delete"] == [201]
    assert next_ctx._cleanup_msg_ids == ["202"]
    await successor.finish_activity()
