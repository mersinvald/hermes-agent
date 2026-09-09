"""Real native runner/FIFO and channel selection with synthetic model/transport."""

import asyncio
import json
from dataclasses import replace

import pytest

from gateway.conversation_control import ConversationGrant
from gateway.platforms.base import MessageEvent, SendResult
from gateway.telegram_conversations import TelegramConversationChannel, channel_key
from tests.gateway.test_native_commands import (
    setup,
    command,
    started,
    finish,
    JournalAgent,
)
from tests.gateway.test_conversation_control import OWNER, FOREIGN


def configured(monkeypatch, tmp_path):
    runner, ingress, db, store, entry, source, adapter = setup(monkeypatch, tmp_path)
    constructor = JournalAgent.__init__

    def initialize(self, **kwargs):
        constructor(self, **kwargs)
        self.session_id = kwargs["session_id"]

    monkeypatch.setattr(JournalAgent, "__init__", initialize)
    db.create_session(
        "conversation-b",
        "telegram",
        user_id=source.user_id,
        session_key=entry.session_key,
        chat_id=source.chat_id,
        chat_type=source.chat_type,
        origin_json=json.dumps(source.to_dict()),
    )
    ingress.grants[OWNER] += (ConversationGrant("conversation-b", replace(source)),)
    policy = TelegramConversationChannel(ingress)
    adapter.set_message_handler(runner._handle_message)
    return runner, ingress, db, store, entry, source, adapter, policy


def finals(adapter):
    return [sent for sent in adapter.sent if (sent.get("metadata") or {}).get("notify")]


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["telegram", "pwa"])
async def test_current_binding_at_finish_for_both_origins(
    monkeypatch, tmp_path, origin
):
    runner, ingress, db, store, entry, source, adapter, policy = configured(
        monkeypatch, tmp_path
    )
    native_task = None
    try:
        if origin == "telegram":
            native_task = asyncio.create_task(
                runner._handle_message(MessageEvent(text="A", source=replace(source)))
            )
        else:
            await ingress.submit(OWNER, command(entry.session_id, "A"))
        await started()
        execution_a = policy.execution(ingress._owner(entry.session_id)[0])
        assert execution_a.origin == origin
        binding = await policy.inspect(OWNER, entry.session_id)
        selected = await policy.select(
            OWNER, "conversation-b", binding["binding_version"]
        )
        assert selected["conversation_id"] == "conversation-b"
        # A remains a real writer while the independent B root starts.
        assert db.native_execution(entry.session_id) is not None
        await ingress.submit(OWNER, command("conversation-b", "B", "b1"))
        await started()
        execution_b = policy.execution(ingress._owner("conversation-b")[0])
        JournalAgent.gates[0].set()
        if native_task:
            await native_task
        for _ in range(300):
            row = db.native_delivery_lookup(
                execution_a.execution_id, channel_key(source)
            )
            if row:
                break
            await asyncio.sleep(0.01)
        assert row["state"] == "skipped"
        assert not finals(adapter)
        JournalAgent.gates[1].set()
        await finish(ingress)
        assert len(finals(adapter)) == 1
        assert (
            db.native_delivery_lookup(execution_b.execution_id, channel_key(source))[
                "state"
            ]
            == "delivered"
        )
        # Repeated completion and later switching back never replay skipped A.
        await policy.deliver(execution_b, "duplicate callback")
        await policy.select(OWNER, entry.session_id, selected["binding_version"])
        await policy.deliver(execution_a, "late callback")
        assert len(finals(adapter)) == 1
        assert [
            m["content"]
            for m in db.get_messages(entry.session_id)
            if m["role"] == "user"
        ] == ["A"]
        assert [
            m["content"]
            for m in db.get_messages("conversation-b")
            if m["role"] == "user"
        ] == ["B"]
    finally:
        await finish(ingress)
        if native_task:
            await native_task
        db.close()


@pytest.mark.asyncio
async def test_binding_versions_aliases_auth_and_persistence_failure(
    monkeypatch, tmp_path
):
    runner, ingress, db, store, entry, source, adapter, policy = configured(
        monkeypatch, tmp_path
    )
    try:
        old = await policy.inspect(OWNER, entry.session_id)
        with pytest.raises(PermissionError):
            await policy.select(FOREIGN, "conversation-b", old["binding_version"])
        with pytest.raises(ValueError):
            await policy.select(OWNER, "conversation-b", old["binding_version"] + 1)
        save = store._save
        monkeypatch.setattr(
            store,
            "_save",
            lambda: (_ for _ in ()).throw(OSError("synthetic write failure")),
        )
        with pytest.raises(OSError):
            await policy.select(OWNER, "conversation-b", old["binding_version"])
        assert await policy.inspect(OWNER, entry.session_id) == old
        monkeypatch.setattr(store, "_save", save)
        db.end_session(entry.session_id, "compression")
        db.create_session(
            "compression-tip", "telegram", parent_session_id=entry.session_id
        )
        same = await policy.select(OWNER, "compression-tip", old["binding_version"])
        assert same["binding_version"] == old["binding_version"]
        assert same["conversation_id"] == entry.session_id
        assert same["native_session_id"] == "compression-tip"
    finally:
        await finish(ingress)
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["rejected", "raised"])
async def test_failed_delivery_is_unknown_and_retry_cannot_duplicate(
    monkeypatch, tmp_path, failure
):
    runner, ingress, db, store, entry, source, adapter, policy = configured(
        monkeypatch, tmp_path
    )
    calls = []

    async def send(*args, **kwargs):
        calls.append(args)
        if failure == "raised":
            raise ConnectionError("synthetic lost acknowledgement")
        return SendResult(
            success=False, error="synthetic network failure", retryable=True
        )

    monkeypatch.setattr(adapter, "send", send)
    try:
        await ingress.submit(OWNER, command(entry.session_id))
        await started()
        execution = policy.execution(ingress._owner(entry.session_id)[0])
        await finish(ingress)
        row = db.native_delivery_lookup(execution.execution_id, channel_key(source))
        assert row["state"] == "unknown"
        await policy.deliver(execution, "retry completion")
        assert len(calls) == 1
        assert (await ingress.command_receipt(OWNER, "c1"))[
            "application_state"
        ] == "applied"
    finally:
        await finish(ingress)
        db.close()


@pytest.mark.asyncio
async def test_native_new_and_resume_select_without_ending_active_root(
    monkeypatch, tmp_path
):
    runner, ingress, db, store, entry, source, adapter, policy = configured(
        monkeypatch, tmp_path
    )
    task = asyncio.create_task(
        runner._handle_message(MessageEvent(text="active", source=replace(source)))
    )
    try:
        await started()
        execution = ingress._owner(entry.session_id)[2]
        # Exercise the native SessionStore operations called by /new /resume.
        new_entry = store.reset_session(entry.session_key)
        assert new_entry.native_binding_version == 2
        assert db.get_session(entry.session_id)["ended_at"] is None
        assert (
            db.native_execution(entry.session_id)["execution_id"]
            == execution.execution_id
        )
        selected = store.switch_session(entry.session_key, "conversation-b")
        assert selected.native_binding_version == 3
        assert db.get_session(new_entry.session_id)["ended_at"] is None
        assert db.get_session("conversation-b")["ended_at"] is None
        assert (
            db.native_execution(entry.session_id)["execution_id"]
            == execution.execution_id
        )
        assert len(JournalAgent.instances) == 1
    finally:
        await finish(ingress)
        await task
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["pwa", "telegram", "system", "unknown"])
async def test_progress_policy_preserves_native_status_without_pwa_mirroring(
    monkeypatch, tmp_path, origin
):
    runner, ingress, db, store, entry, source, adapter, policy = configured(
        monkeypatch, tmp_path
    )
    task = None
    try:
        if origin == "pwa":
            await ingress.submit(OWNER, command(entry.session_id))
        else:
            event = MessageEvent(text="native", source=replace(source))
            if origin in {"system", "unknown"}:
                event.internal = True
                event.source = replace(
                    source, native_conversation_route=entry.session_id
                )
                store.bind_conversation_alias(event.source, entry.session_id)
                if origin == "unknown":
                    event._native_origin = "unknown"
            task = asyncio.create_task(runner._handle_message(event))
        agent = await started()
        key, _, execution = ingress._owner(entry.session_id)
        assert execution.origin == origin
        assert agent.stream_delta_callback is None
        assert agent.interim_assistant_callback is None
        progress = runner._adapter_for_source(execution.source)
        await progress.send(source.chat_id, "native status")
        assert any(s["content"] == "native status" for s in adapter.sent) == (
            origin == "telegram"
        )
    finally:
        await finish(ingress)
        if task:
            await task
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", ["clarify", "approval"])
async def test_telegram_prompt_answer_uses_current_root_and_pwa_text_cannot_answer(
    monkeypatch, tmp_path, prompt
):
    from tools import clarify_gateway, approval

    runner, ingress, db, store, entry, source, adapter, policy = configured(
        monkeypatch, tmp_path
    )
    del runner.__dict__["_handle_active_session_busy_message"]
    adapter._busy_session_handler = runner._handle_active_session_busy_message
    adapter._session_store = store
    task = asyncio.create_task(
        runner._handle_message(MessageEvent(text="native", source=replace(source)))
    )
    key = None
    try:
        await started()
        key, state, execution = ingress._owner(entry.session_id)
        if prompt == "clarify":
            pending = clarify_gateway.register("q1", key, "Which option?", None)
            answer = "answer"
        else:
            pending = approval._ApprovalEntry({"command": "synthetic operation"})
            approval._gateway_queues.setdefault(key, []).append(pending)
            answer = "yes"
        # Accepted PWA text is a command on the same execution, not a prompt reply.
        await ingress.submit(OWNER, command(entry.session_id, answer, "pwa-answer"))
        assert not pending.event.is_set()
        adapter._active_sessions[key] = asyncio.Event()
        await adapter.handle_message(MessageEvent(text=answer, source=replace(source)))
        assert pending.event.is_set()
        assert not adapter._pending_messages
    finally:
        if key:
            clarify_gateway.clear_session(key)
            approval._gateway_queues.pop(key, None)
            adapter._active_sessions.pop(key, None)
        await finish(ingress)
        await task
        db.close()


@pytest.mark.asyncio
async def test_real_new_and_resume_handlers_keep_old_execution_and_delegations(
    monkeypatch, tmp_path
):
    from hermes_state import AsyncSessionDB
    from unittest.mock import Mock

    runner, ingress, db, store, entry, source, adapter, policy = configured(
        monkeypatch, tmp_path
    )
    runner._session_db = AsyncSessionDB(db)
    monkeypatch.setattr(
        runner,
        "_read_user_config",
        lambda: {"approvals": {"destructive_slash_confirm": False}},
    )
    interrupted = Mock()
    monkeypatch.setattr("tools.async_delegation.interrupt_for_session", interrupted)
    task = asyncio.create_task(
        runner._handle_message(MessageEvent(text="A", source=replace(source)))
    )
    try:
        await started()
        execution = ingress._owner(entry.session_id)[2]
        response = await runner._handle_message(
            MessageEvent(text="/new named", source=replace(source))
        )
        assert response
        new = store.lookup_by_session_key(entry.session_key)
        assert new.session_id != entry.session_id
        assert new.native_binding_version == 2
        assert db.get_session(new.session_id)["title"] == "named"
        assert db.get_session(entry.session_id)["ended_at"] is None
        interrupted.assert_not_called()
        response = await runner._handle_message(
            MessageEvent(text="/resume conversation-b", source=replace(source))
        )
        assert response
        assert (await policy.inspect(OWNER, "conversation-b"))[
            "conversation_id"
        ] == "conversation-b"
        assert (
            db.native_execution(entry.session_id)["execution_id"]
            == execution.execution_id
        )
        assert len(JournalAgent.instances) == 1
    finally:
        await finish(ingress)
        await task
        db.close()


@pytest.mark.asyncio
async def test_selection_confirmation_stays_on_native_channel_key(
    monkeypatch, tmp_path
):
    from tools import slash_confirm

    runner, ingress, db, store, entry, source, adapter, policy = configured(
        monkeypatch, tmp_path
    )
    choices = []

    async def confirm(choice):
        choices.append(choice)
        return "confirmed"

    slash_confirm.register(entry.session_key, "new-confirm", "new", confirm)
    try:
        await runner._handle_message(
            MessageEvent(text="approve once", source=replace(source))
        )
        assert choices == ["once"]
        assert not JournalAgent.instances
    finally:
        slash_confirm.clear(entry.session_key)
        await finish(ingress)
        db.close()


@pytest.mark.asyncio
async def test_queued_execution_finals_have_distinct_single_delivery(
    monkeypatch, tmp_path
):
    runner, ingress, db, store, entry, source, adapter, policy = configured(
        monkeypatch, tmp_path
    )
    try:
        await ingress.submit(OWNER, command(entry.session_id, "first"))
        await started()
        first = ingress._owner(entry.session_id)[2]
        await ingress.submit(
            OWNER, command(entry.session_id, "next", "q1", kind="queue")
        )
        JournalAgent.gates[0].set()
        await started()
        next_execution = ingress._owner(entry.session_id)[2]
        assert first.execution_id != next_execution.execution_id
        assert len(finals(adapter)) == 1
        await finish(ingress)
        assert len(finals(adapter)) == 2
        for execution in (first, next_execution):
            assert (
                db.native_delivery_lookup(execution.execution_id, channel_key(source))[
                    "state"
                ]
                == "delivered"
            )
            await policy.deliver(execution, "repeated completion")
        assert len(finals(adapter)) == 2
    finally:
        await finish(ingress)
        db.close()


@pytest.mark.asyncio
async def test_adapter_routes_telegram_b_independently_then_switch_back_skips_b(
    monkeypatch, tmp_path
):
    runner, ingress, db, store, entry, source, adapter, policy = configured(
        monkeypatch, tmp_path
    )
    try:
        await ingress.submit(OWNER, command(entry.session_id, "A"))
        await started()
        a = ingress._owner(entry.session_id)[2]
        await policy.select(OWNER, "conversation-b", 1)
        await adapter.handle_message(
            MessageEvent(text="Telegram B", source=replace(source))
        )
        await started()
        b = ingress._owner("conversation-b")[2]
        assert b.origin == "telegram" and len(JournalAgent.instances) == 2
        await policy.select(OWNER, entry.session_id, 2)
        JournalAgent.gates[1].set()
        await asyncio.gather(*list(adapter._background_tasks))
        assert (
            db.native_delivery_lookup(b.execution_id, channel_key(source))["state"]
            == "skipped"
        )
        assert not finals(adapter)
        JournalAgent.gates[0].set()
        await finish(ingress)
        assert len(finals(adapter)) == 1
        assert (
            db.native_delivery_lookup(a.execution_id, channel_key(source))["state"]
            == "delivered"
        )
    finally:
        await finish(ingress)
        await asyncio.gather(*list(adapter._background_tasks), return_exceptions=True)
        db.close()


@pytest.mark.asyncio
async def test_binding_survives_store_reopen_and_compression(monkeypatch, tmp_path):
    from gateway.session import SessionStore

    runner, ingress, db, store, entry, source, adapter, policy = configured(
        monkeypatch, tmp_path
    )
    try:
        await policy.select(OWNER, "conversation-b", 1)
        reopened = SessionStore(store.sessions_dir, runner.config)
        reopened._db = db
        restored = reopened.lookup_by_session_key(entry.session_key)
        assert restored.native_binding_version == 2
        assert restored.session_id == "conversation-b"
        # Native idle/reset policy cannot silently replace explicit selection.
        assert reopened.get_or_create_session(source).session_id == "conversation-b"
    finally:
        await finish(ingress)
        db.close()


@pytest.mark.asyncio
async def test_duplicate_completion_during_send_and_late_switch_are_fenced(
    monkeypatch, tmp_path
):
    runner, ingress, db, store, entry, source, adapter, policy = configured(
        monkeypatch, tmp_path
    )
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def sending(*args, **kwargs):
        calls.append(args)
        entered.set()
        await release.wait()
        return SendResult(success=True, message_id="sent")

    monkeypatch.setattr(adapter, "send", sending)
    try:
        await ingress.submit(OWNER, command(entry.session_id))
        await started()
        execution = ingress._owner(entry.session_id)[2]
        JournalAgent.gates[0].set()
        await asyncio.wait_for(entered.wait(), 10)
        duplicate = await policy.deliver(execution, "duplicate")
        assert duplicate["state"] == "unknown" and len(calls) == 1
        await policy.select(OWNER, "conversation-b", 1)
        release.set()
        await finish(ingress)
        # The already-dispatched message cannot be retracted by a later switch.
        assert (
            db.native_delivery_lookup(execution.execution_id, channel_key(source))[
                "state"
            ]
            == "delivered"
        )
        assert len(calls) == 1
    finally:
        release.set()
        await finish(ingress)
        db.close()


@pytest.mark.asyncio
async def test_delivery_persistence_failure_blocks_send_and_lookup_reauthorizes(
    monkeypatch, tmp_path
):
    runner, ingress, db, store, entry, source, adapter, policy = configured(
        monkeypatch, tmp_path
    )
    reserve = db.native_delivery_reserve
    try:
        await ingress.submit(OWNER, command(entry.session_id))
        await started()
        execution = ingress._owner(entry.session_id)[2]

        def unavailable(*args):
            raise OSError("synthetic storage failure")

        monkeypatch.setattr(db, "native_delivery_reserve", unavailable)
        await finish(ingress)
        assert not finals(adapter)
        assert (await ingress.command_receipt(OWNER, "c1"))[
            "application_state"
        ] == "applied"
        monkeypatch.setattr(db, "native_delivery_reserve", reserve)
        await policy.deliver(execution, "explicit completion retry after no dispatch")
        assert len(finals(adapter)) == 1
        assert (await policy.delivery(OWNER, entry.session_id, execution.execution_id))[
            "state"
        ] == "delivered"
        with pytest.raises(PermissionError):
            await policy.delivery(FOREIGN, entry.session_id, execution.execution_id)
        runner._is_user_authorized_for_source = lambda source: False
        with pytest.raises(PermissionError):
            await policy.delivery(OWNER, entry.session_id, execution.execution_id)
    finally:
        monkeypatch.setattr(db, "native_delivery_reserve", reserve)
        await finish(ingress)
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("revoke", ["native", "browser"])
async def test_final_rechecks_native_authorization_but_not_browser_login(
    monkeypatch, tmp_path, revoke
):
    runner, ingress, db, store, entry, source, adapter, policy = configured(
        monkeypatch, tmp_path
    )
    try:
        await ingress.submit(OWNER, command(entry.session_id))
        await started()
        execution = ingress._owner(entry.session_id)[2]
        if revoke == "native":
            runner._is_user_authorized_for_source = lambda source: False
        else:
            # Browser principal removed after accepted native input. Telegram
            # channel permission remains independently configured and allowed.
            ingress.grants.clear()
            monkeypatch.setattr(ingress, "trusted_channel_sources", lambda: (source,))
        await finish(ingress)
        row = db.native_delivery_lookup(execution.execution_id, channel_key(source))
        assert row["state"] == ("skipped" if revoke == "native" else "delivered")
        assert len(finals(adapter)) == (0 if revoke == "native" else 1)
    finally:
        await finish(ingress)
        db.close()


@pytest.mark.asyncio
async def test_first_message_bootstraps_configured_native_channel_fail_closed(
    monkeypatch, tmp_path
):
    runner, ingress, db, store, entry, source, adapter, policy = configured(
        monkeypatch, tmp_path
    )
    fresh = replace(source, chat_id="fresh-channel")
    monkeypatch.setattr(ingress, "trusted_channel_sources", lambda: (fresh,))

    # Dynamic source-backed grants are N07-owned. This deterministic provider
    # grants only this explicitly configured native channel's new durable root.
    def event_authorized(event, root):
        return (
            event.source.chat_id == fresh.chat_id and db.get_session(root) is not None
        )

    monkeypatch.setattr(ingress, "_event_authorized", event_authorized)
    try:
        event = MessageEvent(text="first", source=fresh)
        save = store._save
        monkeypatch.setattr(
            store, "_save", lambda: (_ for _ in ()).throw(OSError("write failed"))
        )
        with pytest.raises(OSError):
            policy.route_event(event)
        assert not event.source.native_conversation_route
        assert not JournalAgent.instances
        monkeypatch.setattr(store, "_save", save)
        policy.route_event(event)
        binding = policy._binding(fresh)
        assert binding["binding_version"] == 1
        assert event.source.native_conversation_route == binding["conversation_id"]
        assert db.get_session(binding["native_session_id"]) is not None
        assert (
            store.lookup_by_session_key(entry.session_key).session_id
            == entry.session_id
        )
    finally:
        await finish(ingress)
        db.close()


@pytest.mark.asyncio
async def test_channel_selection_uses_installed_branch_aware_resolver(
    monkeypatch, tmp_path
):
    runner, ingress, db, store, entry, source, adapter, policy = configured(
        monkeypatch, tmp_path
    )
    db.create_session("user-branch", "telegram", parent_session_id=entry.session_id)
    resolver = ingress._resolve

    def resolve(session_id):
        if session_id == "user-branch":
            return {
                "conversation_id": session_id,
                "native_root_session_id": session_id,
                "native_session_id": session_id,
                "native_session_ids": [session_id],
            }
        return resolver(session_id)

    monkeypatch.setattr(ingress, "_resolve", resolve)
    store._native_conversation_resolver = resolve
    ingress.grants[OWNER] += (ConversationGrant("user-branch", source),)
    try:
        selected = await policy.select(OWNER, "user-branch", 1)
        assert selected["conversation_id"] == "user-branch"
        assert selected["native_session_id"] == "user-branch"
        assert selected["binding_version"] == 2
        event = MessageEvent(text="branch input", source=replace(source))
        policy.route_event(event)
        assert event.source.native_conversation_route == "user-branch"
        assert event.metadata["gateway_session_id"] == "user-branch"
        assert db.get_session(entry.session_id)["ended_at"] is None
    finally:
        await finish(ingress)
        db.close()


@pytest.mark.asyncio
async def test_failed_runner_final_keeps_media_failure_guard(monkeypatch, tmp_path):
    from tests.gateway.test_native_commands import synthetic_loop
    from unittest.mock import AsyncMock

    runner, ingress, db, store, entry, source, adapter, policy = configured(
        monkeypatch, tmp_path
    )

    def failed(*args, **kwargs):
        result = synthetic_loop(*args, **kwargs)
        result.update(
            failed=True,
            error="synthetic failure",
            final_response="failed MEDIA:/tmp/not-a-real-output.pdf",
        )
        return result

    monkeypatch.setattr("agent.conversation_loop.run_conversation", failed)
    deliver = AsyncMock(wraps=runner._deliver_queued_first_response)
    monkeypatch.setattr(runner, "_deliver_queued_first_response", deliver)
    try:
        await ingress.submit(OWNER, command(entry.session_id))
        await started()
        await finish(ingress)
        assert deliver.await_count == 1
        assert deliver.call_args.kwargs["deliver_media"] is False
    finally:
        await finish(ingress)
        db.close()


@pytest.mark.asyncio
async def test_unrelated_native_telegram_retains_ordinary_runner_reply(
    monkeypatch, tmp_path
):
    from tests.gateway.test_conversation_control import setup_native, BlockingAgent

    runner, ingress, db, store, entry, source, adapter = setup_native(
        monkeypatch, tmp_path
    )
    policy = TelegramConversationChannel(ingress)
    other = replace(source, chat_id="unmanaged")
    event = MessageEvent(text="ordinary", source=other)
    task = asyncio.create_task(runner._handle_message(event))
    try:
        await asyncio.wait_for(BlockingAgent.started.get(), 10)
        assert event.source.native_conversation_route is None
        assert policy.progress_adapter(other, adapter) is adapter
        assert (
            policy.progress_adapter(
                replace(other, native_conversation_route="unmanaged-root"), adapter
            )
            is adapter
        )
        BlockingAgent.gate.set()
        assert await task == "synthetic done"
        assert not finals(adapter)
        with db._read_ctx() as conn:
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM native_channel_deliveries"
                ).fetchone()[0]
                == 0
            )
    finally:
        BlockingAgent.gate.set()
        await task
        db.close()


@pytest.mark.asyncio
async def test_native_authorization_revoked_during_transport_reconnect(
    monkeypatch, tmp_path
):
    from tests.gateway.test_telegram_send_reconnect_wait import (
        _make_adapter,
        _connected_bot,
    )

    runner, ingress, db, store, entry, source, adapter, policy = configured(
        monkeypatch, tmp_path
    )
    try:
        await ingress.submit(OWNER, command(entry.session_id))
        await started()
        execution = ingress._owner(entry.session_id)[2]
        transport = _make_adapter()
        transport._bot = None
        bot = _connected_bot()

        async def reconnect():
            runner._is_user_authorized_for_source = lambda source: False
            transport._bot = bot
            return True

        monkeypatch.setattr(transport, "_wait_for_reconnection", reconnect)
        runner.adapters[source.platform] = transport
        await finish(ingress)
        bot.send_message.assert_not_awaited()
        assert (
            db.native_delivery_lookup(execution.execution_id, channel_key(source))[
                "state"
            ]
            == "unknown"
        )
    finally:
        await finish(ingress)
        db.close()
