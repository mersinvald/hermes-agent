"""Native Telegram transport boundary with deterministic Bot API doubles."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from telegram.error import NetworkError
from tests.gateway.test_telegram_send_reconnect_wait import (
    _make_adapter,
    _connected_bot,
)


@pytest.mark.asyncio
async def test_binding_is_rechecked_after_native_reconnect_wait():
    adapter = _make_adapter()
    adapter._bot = None
    selected = True
    bot = _connected_bot()

    async def reconnect():
        nonlocal selected
        await asyncio.sleep(0.05)
        selected = False
        adapter._bot = bot

    task = asyncio.create_task(reconnect())
    result = await adapter.send(
        "123", "private final", metadata={"_native_delivery_guard": lambda: selected}
    )
    await task
    assert not result.success
    bot.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_final_does_not_retry_uncertain_network_send():
    adapter = _make_adapter()
    adapter._bot = _connected_bot()
    adapter._bot.send_message = AsyncMock(
        side_effect=NetworkError("synthetic unknown acknowledgement")
    )
    result = await adapter.send(
        "123", "final", metadata={"_native_delivery_once": True, "notify": True}
    )
    assert not result.success and not result.retryable
    assert adapter._bot.send_message.await_count == 1


@pytest.mark.asyncio
async def test_markdown_rejection_rechecks_binding_before_plain_fallback():
    from telegram.error import BadRequest

    adapter = _make_adapter()
    adapter._bot = _connected_bot()
    selected = True

    async def rejected(**kwargs):
        nonlocal selected
        selected = False
        raise BadRequest("Can't parse entities")

    adapter._bot.send_message = AsyncMock(side_effect=rejected)
    result = await adapter.send(
        "123",
        "final",
        metadata={
            "_native_delivery_once": True,
            "_native_delivery_guard": lambda: selected,
        },
    )
    assert not result.success
    assert adapter._bot.send_message.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("uncertain", [False, True])
async def test_rich_failure_cannot_resend_after_switch_or_unknown_ack(uncertain):
    from telegram.error import BadRequest

    adapter = _make_adapter()
    adapter._bot = _connected_bot()
    adapter._should_attempt_rich = lambda *a, **k: True
    selected = True

    async def rejected(*args, **kwargs):
        nonlocal selected
        selected = False
        if uncertain:
            raise NetworkError("synthetic unknown ACK")
        raise BadRequest("Can't parse entities")

    adapter._bot.do_api_request = AsyncMock(side_effect=rejected)
    result = await adapter.send(
        "123",
        "final",
        metadata={
            "_native_delivery_once": True,
            "_native_delivery_guard": lambda: selected,
        },
    )
    assert not result.success
    assert adapter._bot.do_api_request.await_count == 1
    adapter._bot.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_media_anchor_retry_rechecks_binding():
    from telegram.error import BadRequest

    adapter = _make_adapter()
    selected = True

    async def rejected(**kwargs):
        nonlocal selected
        selected = False
        raise BadRequest("reply message not found")

    send = AsyncMock(side_effect=rejected)
    adapter._should_retry_without_dm_topic_reply_anchor = lambda *args: True
    with pytest.raises(RuntimeError, match="binding changed"):
        await adapter._send_with_dm_topic_reply_anchor_retry(
            send,
            {"chat_id": "123"},
            {"_native_delivery_guard": lambda: selected},
            1,
            "photo",
        )
    assert send.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,bot_method",
    [
        ("send_image_file", "send_photo"),
        ("send_document", "send_document"),
        ("send_video", "send_video"),
        ("send_voice", "send_audio"),
    ],
)
async def test_native_media_unknown_ack_never_uses_alternate_fallback(
    tmp_path, method, bot_method
):
    adapter = _make_adapter()
    adapter._bot = _connected_bot()
    path = tmp_path / "synthetic.mp3"
    path.write_bytes(b"synthetic")
    send = AsyncMock(side_effect=NetworkError("synthetic unknown ACK"))
    setattr(adapter._bot, bot_method, send)
    result = await getattr(adapter, method)(
        "123",
        str(path),
        metadata={
            "_native_delivery_once": True,
            "_native_delivery_guard": lambda: True,
        },
    )
    assert not result.success and not result.retryable
    assert send.await_count == 1
    adapter._bot.send_message.assert_not_awaited()
