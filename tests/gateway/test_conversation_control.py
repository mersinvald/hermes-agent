"""Real native ingress/runner path with isolated SQLite and deterministic agents."""
import asyncio
import sys
import threading
import types
from dataclasses import replace

import pytest

from gateway.conversation_control import ConversationGrant, NativeConversationIngress, Principal
from gateway.platforms.base import MessageEvent
from gateway.session import SessionStore
from hermes_state import SessionDB
from tests.gateway.test_42039_duplicate_user_message import _bootstrap, _source
from tests.gateway.test_queued_native_image_session_key import CaptureAdapter

OWNER = Principal("https://issuer.invalid", "owner")
FOREIGN = Principal("https://issuer.invalid", "foreign")


class BlockingAgent:
    instances = []
    calls = []
    gate = None
    next_gate = None
    started = None
    loop = None
    db = None

    def __init__(self, **kwargs):
        self.tools = []
        self.steers = []
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.model = "synthetic"
        type(self).instances.append(self)

    def steer(self, text):
        self.steers.append(text)
        return True

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        assert type(self).db.get_session(task_id) is not None
        type(self).calls.append((self, message))
        type(self).loop.call_soon_threadsafe(type(self).started.put_nowait, self)
        gate = type(self).gate if len(type(self).calls) == 1 else type(self).next_gate
        assert gate.wait(20), "test did not release model gate"
        # Stand in for the model executor's native persistence, allowing the
        # real gateway's dedup/continuation boundary to be exercised.
        type(self).db.append_message(task_id, "user", message)
        type(self).db.append_message(task_id, "assistant", "synthetic done")
        return {"final_response": "synthetic done", "messages": list(conversation_history or []) + [
            {"role": "user", "content": message}, {"role": "assistant", "content": "synthetic done"}],
            "agent_persisted": True, "api_calls": 1}


def setup_native(monkeypatch, tmp_path):
    runner = _bootstrap(monkeypatch, tmp_path)
    del runner.__dict__["_begin_session_run_generation"]
    del runner.__dict__["_is_session_run_current"]
    source = _source()
    adapter = CaptureAdapter()
    runner.adapters = {source.platform: adapter}
    runner._is_user_authorized_for_source = lambda candidate: candidate.user_id == source.user_id
    runner._session_db = None  # Synthetic model: native gateway flush owns SQLite.
    db = SessionDB(tmp_path / "native.db")
    store = SessionStore(tmp_path / "sessions", runner.config)
    store._db = db
    runner.session_store = store
    entry = store.get_or_create_session(source)
    ingress = NativeConversationIngress(runner, db, {OWNER: (ConversationGrant(entry.session_id, source),)})
    BlockingAgent.db = db
    BlockingAgent.instances = []
    BlockingAgent.calls = []
    BlockingAgent.gate = threading.Event()
    BlockingAgent.next_gate = threading.Event()
    BlockingAgent.started = asyncio.Queue()
    BlockingAgent.loop = asyncio.get_running_loop()
    fake = types.ModuleType("run_agent")
    fake.AIAgent = BlockingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake)
    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "30")
    return runner, ingress, db, store, entry, source, adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["telegram", "pwa"])
async def test_both_origins_control_one_real_owner_and_queue_runs_next(monkeypatch, tmp_path, origin):
    runner, ingress, db, store, entry, source, adapter = setup_native(monkeypatch, tmp_path)
    first = (ingress.send(OWNER, entry.session_id, "first") if origin == "pwa"
             else runner._handle_message(MessageEvent(text="first", source=source)))
    task = asyncio.create_task(first)
    try:
        agent = await asyncio.wait_for(BlockingAgent.started.get(), 10)
        async def wait_promotion():
            while (await ingress.inspect(OWNER, entry.session_id))["active_execution"]["execution_state"] != "running":
                await asyncio.sleep(0.01)
        await asyncio.wait_for(wait_promotion(), 10)
        initial = await ingress.inspect(OWNER, entry.session_id)
        execution_id = initial["active_execution"]["execution_id"]
        assert initial["active_execution"]["origin"] == origin
        if origin == "telegram":
            result = await ingress.send(OWNER, entry.session_id, "browser correction")
            assert result["disposition"] == "steered"
            assert agent.steers == ["browser correction"]
        else:
            assert await runner._handle_message(MessageEvent(text="telegram correction", source=source)) is None
            assert agent.steers == ["telegram correction"]
        result = await ingress.send(OWNER, entry.session_id, "next", mode="queue")
        assert result["disposition"] == "queued"
        assert result["execution"]["execution_id"] == execution_id
        assert len(BlockingAgent.calls) == 1
        owner_key = ingress._owner(entry.session_id)[0]
        assert runner._queue_depth(owner_key, adapter=adapter) == 1
        BlockingAgent.gate.set()
        await asyncio.wait_for(BlockingAgent.started.get(), 10)
        next_execution = (await ingress.inspect(OWNER, entry.session_id))["active_execution"]
        assert next_execution["execution_id"] != execution_id
        assert next_execution["origin"] == "pwa"
        BlockingAgent.next_gate.set()
        await asyncio.wait_for(task, 10)
        assert len(BlockingAgent.calls) == 2
        assert "next" in BlockingAgent.calls[1][1]
        assert (await ingress.inspect(OWNER, entry.session_id))["active_execution"] is None
        messages = db.get_messages(entry.session_id)
        assert [m["role"] for m in messages if m["role"] != "session_meta"] == [
            "user", "assistant", "user", "assistant"]
    finally:
        BlockingAgent.gate.set()
        BlockingAgent.next_gate.set()
        await asyncio.gather(task, return_exceptions=True)
        db.close()


@pytest.mark.asyncio
async def test_authorization_and_native_auth_cannot_be_forged(monkeypatch, tmp_path):
    runner, ingress, db, store, entry, source, _ = setup_native(monkeypatch, tmp_path)
    db.create_session("foreign", "telegram", user_id="foreign")
    for principal, sid in [(FOREIGN, entry.session_id), (OWNER, "foreign"), (OWNER, "forged")]:
        with pytest.raises(PermissionError):
            await ingress.inspect(principal, sid)
        with pytest.raises(PermissionError):
            await ingress.send(principal, sid, "attack")
        with pytest.raises(PermissionError):
            await ingress.history(principal, sid)
    runner._is_user_authorized_for_source = lambda _: False
    with pytest.raises(PermissionError):
        await ingress.send(OWNER, entry.session_id, "attack")
    assert not BlockingAgent.instances
    db.close()


@pytest.mark.asyncio
async def test_pwa_b_runs_independently_without_switching_telegram_a(monkeypatch, tmp_path):
    runner, ingress, db, store, entry, source, _ = setup_native(monkeypatch, tmp_path)
    db.create_session("conversation-b", "telegram", user_id=source.user_id)
    ingress.grants[OWNER] += (ConversationGrant("missing-old-grant", source),
                              ConversationGrant("conversation-b", source))
    task_a = asyncio.create_task(runner._handle_message(MessageEvent(text="A", source=source)))
    task_b = None
    try:
        await asyncio.wait_for(BlockingAgent.started.get(), 10)
        task_b = asyncio.create_task(ingress.send(OWNER, "conversation-b", "B"))
        await asyncio.wait_for(BlockingAgent.started.get(), 10)
        a = (await ingress.inspect(OWNER, entry.session_id))["active_execution"]
        b = (await ingress.inspect(OWNER, "conversation-b"))["active_execution"]
        assert a["execution_id"] != b["execution_id"]
        assert len(BlockingAgent.instances) == 2
        assert store.lookup_by_session_key(entry.session_key).session_id == entry.session_id
        assert db.get_session(entry.session_id)["ended_at"] is None
        # An alias survives native SessionStore reload, including its source
        # route; source transport decoding still cannot forge this discriminator.
        from gateway.session import SessionEntry, SessionSource
        alias_key = ingress._owner("conversation-b")[0]
        alias = store.lookup_by_session_key(alias_key)
        restored = SessionEntry.from_dict(alias.to_dict())
        assert runner._session_key_for_source(restored.origin) == alias_key
        reopened_store = SessionStore(tmp_path / "sessions", runner.config)
        reopened_store._db = db
        runner.session_store = reopened_store
        wake_source = runner._build_process_event_source({"type": "async_delegation", "session_key": alias_key})
        assert runner._session_key_for_source(wake_source) == alias_key
        assert runner._build_process_event_source({"session_key": "agent:main:native:conversation:missing",
                                                   "platform": "telegram", "chat_id": source.chat_id}) is None
        runner.session_store = store
        forged = source.to_dict() | {"native_conversation_route": "conversation-b"}
        assert SessionSource.from_dict(forged).native_conversation_route is None
    finally:
        BlockingAgent.gate.set()
        BlockingAgent.next_gate.set()
        await asyncio.gather(*(t for t in (task_a, task_b) if t), return_exceptions=True)
        db.close()


@pytest.mark.asyncio
async def test_pending_alias_and_compression_target_same_owner(monkeypatch, tmp_path):
    runner, ingress, db, store, entry, source, adapter = setup_native(monkeypatch, tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    original_acquire = runner._turn_leases.acquire

    async def gated_acquire(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original_acquire(*args, **kwargs)

    monkeypatch.setattr(runner._turn_leases, "acquire", gated_acquire)
    task = asyncio.create_task(runner._handle_message(MessageEvent(text="first", source=source)))
    try:
        await asyncio.wait_for(entered.wait(), 10)
        initial = (await ingress.inspect(OWNER, entry.session_id))["active_execution"]
        assert initial["execution_state"] == "starting"
        db.end_session(entry.session_id, "compression")
        db.create_session("compressed-tip", "telegram", parent_session_id=entry.session_id)
        result = await ingress.send(OWNER, "compressed-tip", "pending")
        assert result["disposition"] == "queued"
        assert result["execution"]["execution_id"] == initial["execution_id"]
        assert (await ingress.inspect(OWNER, "compressed-tip"))["conversation_id"] == entry.session_id
        assert not BlockingAgent.instances
        assert runner._queue_depth(entry.session_key, adapter=adapter) == 1
        # Early agent cleanup cannot release native ownership. A stale lease
        # generation cannot release it either.
        runner._release_running_agent_state(entry.session_key)
        runner._release_turn_lease(entry.session_key, 999)
        assert (await ingress.inspect(OWNER, entry.session_id))["active_execution"]["execution_id"] == initial["execution_id"]
        result = await ingress.send(OWNER, entry.session_id, "still queued")
        assert result["disposition"] == "queued"
        assert not BlockingAgent.instances
    finally:
        release.set()
        BlockingAgent.gate.set()
        BlockingAgent.next_gate.set()
        await asyncio.gather(task, return_exceptions=True)
        db.close()


@pytest.mark.asyncio
async def test_alias_binding_failure_does_not_change_telegram(monkeypatch, tmp_path):
    runner, ingress, db, store, entry, source, _ = setup_native(monkeypatch, tmp_path)
    def fail():
        raise OSError("synthetic storage failure")
    monkeypatch.setattr(store, "_save", fail)
    with pytest.raises(OSError):
        await ingress.send(OWNER, entry.session_id, "not accepted")
    assert store.lookup_by_session_key(entry.session_key).session_id == entry.session_id
    assert len(store._entries) == 1
    assert not BlockingAgent.instances
    db.close()


@pytest.mark.asyncio
async def test_missing_native_row_failure_rejects_before_agent(monkeypatch, tmp_path):
    runner, ingress, db, store, entry, source, _ = setup_native(monkeypatch, tmp_path)
    # Simulate SessionStore's legacy best-effort create failing while its
    # routing row was published. The real ingress admission must fail closed.
    monkeypatch.setattr(db, "get_session", lambda _: None)
    def fail(*args, **kwargs):
        raise OSError("synthetic native persistence failure")
    monkeypatch.setattr(db, "record_gateway_session_peer", fail)
    with pytest.raises(OSError):
        await runner._handle_message(MessageEvent(text="not admitted", source=source))
    assert not BlockingAgent.instances
    assert all(state.turn.conversation_execution is None for state in runner._sessions.values())
    db.close()


@pytest.mark.asyncio
async def test_history_preserves_lineage_segments_and_read_errors(monkeypatch, tmp_path):
    runner, ingress, db, store, entry, source, _ = setup_native(monkeypatch, tmp_path)
    db.append_message(entry.session_id, "user", "before compression")
    db.end_session(entry.session_id, "compression")
    db.create_session("tip", "telegram", parent_session_id=entry.session_id)
    db.append_message("tip", "assistant", "retained summary")
    history = await ingress.history(OWNER, "tip")
    assert [s["native_session_id"] for s in history["segments"]] == [entry.session_id, "tip"]
    assert [s["messages"][0]["content"] for s in history["segments"]] == ["before compression", "retained summary"]
    def fail(*args, **kwargs):
        raise OSError("history unavailable")
    monkeypatch.setattr(db, "get_messages", fail)
    with pytest.raises(OSError):
        await ingress.history(OWNER, "tip")
    db.close()


@pytest.mark.asyncio
async def test_racing_browser_admission_creates_only_one_active_agent(monkeypatch, tmp_path):
    runner, ingress, db, store, entry, source, adapter = setup_native(monkeypatch, tmp_path)
    one = asyncio.create_task(ingress.send(OWNER, entry.session_id, "one"))
    two = asyncio.create_task(ingress.send(OWNER, entry.session_id, "two"))
    try:
        await asyncio.wait_for(BlockingAgent.started.get(), 10)
        # Both request coroutines have crossed admission once one acknowledges
        # the pending control. Wait for that fact, never a negative sleep test.
        done, _ = await asyncio.wait({one, two}, timeout=10, return_when=asyncio.FIRST_COMPLETED)
        assert len(done) == 1
        assert len(BlockingAgent.instances) == 1
        assert len(BlockingAgent.calls) == 1
        assert len([s for s in runner._sessions.values() if s.turn.conversation_execution]) == 1
    finally:
        BlockingAgent.gate.set()
        BlockingAgent.next_gate.set()
        await asyncio.gather(one, two)
        db.close()


@pytest.mark.asyncio
async def test_native_row_is_repaired_before_competing_admission(monkeypatch, tmp_path):
    runner, ingress, db, store, entry, source, _ = setup_native(monkeypatch, tmp_path)
    db.delete_session(entry.session_id)
    task = asyncio.create_task(runner._handle_message(MessageEvent(text="create first", source=source)))
    try:
        await asyncio.wait_for(BlockingAgent.started.get(), 10)
        assert db.get_session(entry.session_id) is not None
        assert (await ingress.inspect(OWNER, entry.session_id))["active_execution"] is not None
    finally:
        BlockingAgent.gate.set()
        BlockingAgent.next_gate.set()
        await asyncio.gather(task, return_exceptions=True)
        db.close()


@pytest.mark.asyncio
async def test_binding_change_during_native_preparation_cannot_change_owner(monkeypatch, tmp_path):
    runner, ingress, db, store, entry, source, _ = setup_native(monkeypatch, tmp_path)
    db.create_session("other-conversation", "telegram", user_id=source.user_id)
    get_session = store.get_or_create_session
    calls = 0
    def changing_binding(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = get_session(*args, **kwargs)
        return replace(result, session_id="other-conversation") if calls == 2 else result
    monkeypatch.setattr(store, "get_or_create_session", changing_binding)
    with pytest.raises(RuntimeError, match="conversation changed during admission"):
        await runner._handle_message(MessageEvent(text="must not move", source=source))
    assert not BlockingAgent.instances
    assert (await ingress.inspect(OWNER, entry.session_id))["active_execution"] is None
    db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", ["clarify", "slash_confirm", "update", "tool_approval"])
async def test_native_prompt_answers_remain_native_and_browser_text_is_not_approval(monkeypatch, tmp_path, prompt):
    from tools import clarify_gateway, slash_confirm, approval
    runner, ingress, db, store, entry, source, adapter = setup_native(monkeypatch, tmp_path)
    del runner.__dict__["_handle_active_session_busy_message"]
    adapter._message_handler = runner._handle_message
    adapter._busy_session_handler = runner._handle_active_session_busy_message
    adapter._session_store = store
    task = asyncio.create_task(runner._handle_message(MessageEvent(text="first", source=source)))
    pending, choices = None, []
    try:
        agent = await asyncio.wait_for(BlockingAgent.started.get(), 10)
        async def promoted():
            while runner._session_state(entry.session_key).turn.agent is not agent:
                await asyncio.sleep(0.01)
        await asyncio.wait_for(promoted(), 10)
        if prompt == "clarify":
            pending = clarify_gateway.register("native-question", entry.session_key, "Which?", None)
            answer = "this answer"
        elif prompt == "slash_confirm":
            async def confirm(choice):
                choices.append(choice)
                return "confirmed"
            slash_confirm.register(entry.session_key, "native-confirm", "reload-mcp", confirm)
            answer = "approve once"
        elif prompt == "update":
            runner._session_state(entry.session_key).persistent.update_prompt_pending = True
            answer = "yes"
        else:
            pending = approval._ApprovalEntry({"command": "synthetic harmless operation"})
            approval._gateway_queues.setdefault(entry.session_key, []).append(pending)
            answer = "yes"
        # Browser ordinary text never answers a native prompt on another alias.
        result = await ingress.send(OWNER, entry.session_id, answer)
        assert result["disposition"] == "steered"
        assert agent.steers == [answer]
        if pending is not None:
            assert not pending.event.is_set()
        assert not choices
        if prompt == "tool_approval":
            # Real adapter busy ingress owns bare-word tool approval routing.
            adapter._active_sessions[entry.session_key] = asyncio.Event()
            await adapter.handle_message(MessageEvent(text=answer, source=source))
            assert pending.event.is_set(), (adapter.sent, agent.steers, entry.session_key, adapter._pending_messages)
            assert pending.result == "once"
        else:
            await runner._handle_message(MessageEvent(text=answer, source=source))
            if prompt == "clarify":
                assert pending.event.is_set()
                assert pending.response == answer
            elif prompt == "slash_confirm":
                assert choices == ["once"]
            else:
                assert (tmp_path / ".update_response").read_text() == "yes"
        assert agent.steers == [answer]
        assert runner._queue_depth(entry.session_key, adapter=adapter) == 0
    finally:
        clarify_gateway.clear_session(entry.session_key)
        slash_confirm.clear(entry.session_key)
        approval._gateway_queues.pop(entry.session_key, None)
        adapter._active_sessions.pop(entry.session_key, None)
        BlockingAgent.gate.set()
        BlockingAgent.next_gate.set()
        await asyncio.gather(task, return_exceptions=True)
        db.close()


@pytest.mark.asyncio
async def test_trusted_plugin_rewrite_preserves_pwa_origin_and_explicit_queue(monkeypatch, tmp_path):
    runner, ingress, db, store, entry, source, adapter = setup_native(monkeypatch, tmp_path)
    def rewrite(hook, **kwargs):
        if hook == "pre_gateway_dispatch":
            return [{"action": "rewrite", "text": "rewritten " + kwargs["event"].text}]
        return []
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", rewrite)
    task = asyncio.create_task(ingress.send(OWNER, entry.session_id, "first"))
    try:
        agent = await asyncio.wait_for(BlockingAgent.started.get(), 10)
        initial = (await ingress.inspect(OWNER, entry.session_id))["active_execution"]
        assert initial["origin"] == "pwa"
        result = await ingress.send(OWNER, entry.session_id, "next", mode="queue")
        assert result["disposition"] == "queued"
        assert not agent.steers
        BlockingAgent.gate.set()
        await asyncio.wait_for(BlockingAgent.started.get(), 10)
        followup = (await ingress.inspect(OWNER, entry.session_id))["active_execution"]
        assert followup["origin"] == "pwa"
        assert "rewritten next" in BlockingAgent.calls[1][1]
    finally:
        BlockingAgent.gate.set()
        BlockingAgent.next_gate.set()
        await asyncio.gather(task, return_exceptions=True)
        db.close()
