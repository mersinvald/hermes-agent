"""Durable commands at GatewayRunner -> native AIAgent lease -> safe boundary.

Only model/tool work is synthetic. Journal, leases, runner admission, queue,
continuation, authorization and native steer injection helpers are real.
"""

import asyncio
import copy
import threading

import pytest

import run_agent as real_run_agent
from agent.agent_runtime_helpers import apply_pending_steer_to_tool_results
from agent.tool_dispatch_helpers import project_tool_content_for_storage
from gateway.platforms.base import MessageEvent
from hermes_state_commands import CommandConflict
from tests.gateway.test_conversation_control import OWNER, FOREIGN, setup_native
from tests.run_agent.test_cross_process_turn_lease import _agent_with_db


class JournalAgent(real_run_agent.AIAgent):
    db = None
    root = None
    instances = []
    starts = None
    loop = None
    gates = []
    tool = False
    after_tool = None
    tool_ready = None
    effects = []

    def __init__(self, **kwargs):
        self.__dict__.update(
            _agent_with_db(
                type(self).db, session_id=type(self).root, platform="telegram"
            ).__dict__
        )
        self.tools = []
        self._pending_steer = None
        self._pending_steer_lock = threading.Lock()
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self._last_flushed_db_idx = 0
        type(self).instances.append(self)

    def _flush_messages_to_session_db(self, messages, conversation_history=None):
        rows = []
        live = []
        for msg in messages:
            if not msg.get("_db_persisted") and not msg.get("_row_id"):
                rows.append({
                    **msg,
                    "content": project_tool_content_for_storage(msg.get("content")),
                })
                live.append(msg)
        self._session_db.append_messages_batch(
            self.session_id,
            rows,
            turn_lease_holder=self._active_session_turn_lease_holder,
        )
        for msg, row in zip(live, rows):
            msg.update(_row_id=row["_row_id"], _db_persisted=True)
        return True


def synthetic_loop(agent, text, system, history, *args, **kwargs):
    messages = copy.deepcopy(history or [])
    message = {"role": "user", "content": text}
    messages.append(message)
    agent._native_command_context.start_input(
        agent, message, kwargs.get("persist_user_message")
    )
    agent._flush_messages_to_session_db(messages)
    index = len(JournalAgent.effects)
    JournalAgent.effects.append(text)
    JournalAgent.loop.call_soon_threadsafe(JournalAgent.starts.put_nowait, agent)
    assert JournalAgent.gates[index].wait(15), "synthetic model gate not released"
    if JournalAgent.tool and index == 0:
        tool = {
            "role": "tool",
            "content": copy.deepcopy(JournalAgent.tool),
            "tool_call_id": "call-1",
            "tool_name": "synthetic",
            "effect_disposition": "observed",
        }
        messages.extend([
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "synthetic", "arguments": "{}"},
                    }
                ],
            },
            tool,
        ])
        agent._flush_messages_to_session_db(messages)
        apply_pending_steer_to_tool_results(agent, messages, 1)
        # A repeated pre-API safe-boundary drain cannot duplicate the marker.
        agent._native_command_context.consume(agent, messages)
        JournalAgent.loop.call_soon_threadsafe(JournalAgent.tool_ready.set)
        assert JournalAgent.after_tool.wait(15)
    messages.append({"role": "assistant", "content": "synthetic done"})
    agent._flush_messages_to_session_db(messages)
    return {
        "final_response": "synthetic done",
        "messages": messages,
        "agent_persisted": True,
        "api_calls": 1,
    }


def setup(monkeypatch, tmp_path):
    native = setup_native(monkeypatch, tmp_path)
    runner, ingress, db, store, entry, source, adapter = native
    monkeypatch.setitem(__import__("sys").modules, "run_agent", real_run_agent)
    monkeypatch.setattr(real_run_agent, "AIAgent", JournalAgent)
    monkeypatch.setattr("agent.conversation_loop.run_conversation", synthetic_loop)
    JournalAgent.db = db
    JournalAgent.root = entry.session_id
    JournalAgent.instances = []
    JournalAgent.effects = []
    JournalAgent.starts = asyncio.Queue()
    JournalAgent.loop = asyncio.get_running_loop()
    JournalAgent.gates = [threading.Event() for _ in range(8)]
    JournalAgent.after_tool = threading.Event()
    JournalAgent.tool_ready = asyncio.Event()
    JournalAgent.tool = False
    return native


def command(root, text="first", id="c1", kind="send", **kwargs):
    return {
        "schema_version": "1.0",
        "command_id": id,
        "conversation_id": root,
        "type": kind,
        "payload": {"text": text},
        **kwargs,
    }


async def started():
    return await asyncio.wait_for(JournalAgent.starts.get(), 10)


async def finish(ingress):
    ingress.stop_command_recovery()
    for gate in JournalAgent.gates:
        gate.set()
    JournalAgent.after_tool.set()
    for _ in range(1200):
        if not ingress._command_wakeups and not any(
            s.turn.conversation_execution
            for s in ingress.runner._sessions_map().values()
        ):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(
        f"native execution did not finish: {ingress._command_wakeups!r}; states={[(k, s.turn.conversation_execution) for k, s in ingress.runner._sessions_map().items()]!r}"
    )


@pytest.mark.asyncio
async def test_receipt_retry_conflict_auth_and_atomic_input(monkeypatch, tmp_path):
    runner, ingress, db, store, entry, source, adapter = setup(monkeypatch, tmp_path)
    request = command(entry.session_id)
    try:
        replies = await asyncio.gather(
            *(ingress.submit(OWNER, request) for _ in range(6))
        )
        assert {r["command_id"] for r in replies} == {"c1"}
        assert all(r["durability"] == "durable" for r in replies)
        await started()
        row = await ingress.command_receipt(OWNER, "c1")
        assert row["application_state"] == "applied"
        assert JournalAgent.effects == ["first"]
        assert (
            len([m for m in db.get_messages(entry.session_id) if m["role"] == "user"])
            == 1
        )
        assert await ingress.submit(OWNER, request) == row
        with pytest.raises(CommandConflict):
            await ingress.submit(OWNER, {**request, "payload": {"text": "changed"}})
        with pytest.raises((PermissionError, LookupError)):
            await ingress.command_receipt(FOREIGN, "c1")
        runner._is_user_authorized_for_source = lambda source: False
        with pytest.raises(PermissionError):
            await ingress.submit(OWNER, request)
        with pytest.raises(PermissionError):
            await ingress.command_receipt(OWNER, "c1")
    finally:
        await finish(ingress)
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool",
    [
        "tool text",
        [
            {"type": "text", "text": "tool text"},
            {"type": "image_url", "image_url": {"url": "data:synthetic"}},
        ],
        {
            "_multimodal": True,
            "text_summary": "tool summary",
            "content": [{"type": "image_url", "image_url": {"url": "data:synthetic"}}],
        },
    ],
)
async def test_tracked_steer_applies_once_to_persisted_tool_content(
    monkeypatch, tmp_path, tool
):
    runner, ingress, db, store, entry, source, adapter = setup(monkeypatch, tmp_path)
    JournalAgent.tool = tool
    try:
        await ingress.submit(OWNER, command(entry.session_id))
        await started()
        initial = await ingress.command_receipt(OWNER, "c1")
        request = command(
            entry.session_id,
            "correct this",
            "s1",
            kind="steer",
            target_execution_id=initial["resulting_execution_id"],
        )
        pending = await ingress.submit(OWNER, request)
        assert (
            pending["effective_action"] == "steer"
            and pending["application_state"] == "pending"
        )
        JournalAgent.gates[0].set()
        await asyncio.wait_for(JournalAgent.tool_ready.wait(), 10)
        applied = await ingress.command_receipt(OWNER, "s1")
        assert applied["application_state"] == "applied" and applied["fallback"] is None
        assert applied["resulting_execution_id"] == initial["resulting_execution_id"]
        assert await ingress.submit(OWNER, request) == applied
        persisted = next(
            m for m in db.get_messages(entry.session_id) if m["role"] == "tool"
        )
        assert persisted["content"].count("correct this") == 1
        assert "data:synthetic" not in persisted["content"]
        assert (
            persisted["tool_call_id"] == "call-1"
            and persisted["effect_disposition"] == "observed"
        )
    finally:
        await finish(ingress)
        db.close()


@pytest.mark.asyncio
async def test_unconsumed_steer_survives_explicit_queue_and_has_one_fallback(
    monkeypatch, tmp_path
):
    runner, ingress, db, store, entry, source, adapter = setup(monkeypatch, tmp_path)
    try:
        await ingress.submit(OWNER, command(entry.session_id))
        await started()
        first = await ingress.command_receipt(OWNER, "c1")
        request = command(entry.session_id, "leftover", "s1")
        await ingress.submit(OWNER, request)
        await ingress.submit(
            OWNER, command(entry.session_id, "explicit next", "q1", kind="queue")
        )
        JournalAgent.gates[0].set()
        await started()
        q = await ingress.command_receipt(OWNER, "q1")
        assert q["resulting_execution_id"] != first["resulting_execution_id"]
        JournalAgent.gates[1].set()
        await started()
        fallback = await ingress.command_receipt(OWNER, "s1")
        assert fallback["fallback"] == {"reason": "unconsumed_steer"}
        assert fallback["application_state"] == "applied"
        assert fallback["target_execution_id"] == first["resulting_execution_id"]
        assert fallback["resulting_execution_id"] not in {
            first["resulting_execution_id"],
            q["resulting_execution_id"],
        }
        assert await ingress.submit(OWNER, request) == fallback
        assert JournalAgent.effects == ["first", "explicit next", "leftover"]
        with pytest.raises(CommandConflict):
            await ingress.submit(
                OWNER,
                command(
                    entry.session_id,
                    "late",
                    "late",
                    kind="steer",
                    target_execution_id=first["resulting_execution_id"],
                ),
            )
    finally:
        await finish(ingress)
        db.close()


@pytest.mark.asyncio
async def test_admission_storage_failure_and_unsupported_fields_have_no_effect(
    monkeypatch, tmp_path
):
    runner, ingress, db, store, entry, source, adapter = setup(monkeypatch, tmp_path)
    try:
        for extra in (
            {"expected_model_version": 1},
            {"expected_binding_version": 1},
            {"user_id": "forged"},
            {"type": "redirect"},
        ):
            with pytest.raises(ValueError):
                await ingress.submit(OWNER, {**command(entry.session_id), **extra})
        db._execute_write(
            lambda c: c.execute(
                "CREATE TRIGGER reject_receipt BEFORE INSERT ON native_commands BEGIN SELECT RAISE(ABORT,'synthetic disk failure'); END"
            )
        )
        with pytest.raises(Exception, match="synthetic disk failure"):
            await ingress.submit(OWNER, command(entry.session_id))
        assert not JournalAgent.effects and not JournalAgent.instances
        assert db.native_command_rows(entry.session_id) == []
        assert not ingress._command_wakeups
    finally:
        await finish(ingress)
        db.close()


@pytest.mark.asyncio
async def test_pending_start_queue_and_compression_alias_retry(monkeypatch, tmp_path):
    runner, ingress, db, store, entry, source, adapter = setup(monkeypatch, tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    acquire = runner._turn_leases.acquire

    async def gated(*args, **kwargs):
        entered.set()
        await release.wait()
        return await acquire(*args, **kwargs)

    monkeypatch.setattr(runner._turn_leases, "acquire", gated)
    try:
        await ingress.submit(OWNER, command(entry.session_id))
        await asyncio.wait_for(entered.wait(), 10)
        pending = await ingress.submit(OWNER, command(entry.session_id, "second", "c2"))
        assert pending["effective_action"] == "queue" and pending["fallback"] is None
        assert not JournalAgent.instances
        # An alias resolves to the same canonical payload for idempotency.
        db.end_session(entry.session_id, "compression")
        db.create_session(
            "compressed-tip", "telegram", parent_session_id=entry.session_id
        )
        assert (
            await ingress.submit(OWNER, command("compressed-tip", "second", "c2"))
            == pending
        )
        release.set()
        await started()
        JournalAgent.gates[0].set()
        await started()
        assert JournalAgent.effects == ["first", "second"]
    finally:
        release.set()
        await finish(ingress)
        db.close()


@pytest.mark.asyncio
async def test_pre_agent_failure_releases_own_reservation(monkeypatch, tmp_path):
    runner, ingress, db, store, entry, source, adapter = setup(monkeypatch, tmp_path)
    constructor = JournalAgent.__init__

    def fail(self, **kwargs):
        raise RuntimeError("synthetic constructor failure")

    monkeypatch.setattr(JournalAgent, "__init__", fail)
    try:
        await ingress.submit(OWNER, command(entry.session_id))
        await finish(ingress)
        assert db.native_execution(entry.session_id) is None
        assert (await ingress.command_receipt(OWNER, "c1"))[
            "application_state"
        ] == "not_applied"
        assert JournalAgent.effects == []
        monkeypatch.setattr(JournalAgent, "__init__", constructor)
        await ingress.submit(OWNER, command(entry.session_id, "healthy", "c2"))
        await started()
        assert JournalAgent.effects == ["healthy"]
    finally:
        await finish(ingress)
        db.close()


@pytest.mark.asyncio
async def test_early_control_cleanup_does_not_replace_live_writer(
    monkeypatch, tmp_path
):
    runner, ingress, db, store, entry, source, adapter = setup(monkeypatch, tmp_path)
    try:
        await ingress.submit(OWNER, command(entry.session_id))
        await started()
        key, state, execution = ingress._owner(entry.session_id)
        runner._release_running_agent_state(key)
        runner._release_turn_lease(key, execution.generation)
        reply = await ingress.submit(OWNER, command(entry.session_id, "tail", "s1"))
        assert reply["effective_action"] == "steer"
        assert len(JournalAgent.instances) == 1
        assert (
            db.native_execution(entry.session_id)["execution_id"]
            == execution.execution_id
        )
        JournalAgent.gates[0].set()
        await started()
        assert JournalAgent.effects == ["first", "tail"]
        assert (await ingress.command_receipt(OWNER, "s1"))["fallback"] is not None
    finally:
        await finish(ingress)
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["telegram", "pwa"])
async def test_cross_channel_durable_control_shares_owner(
    monkeypatch, tmp_path, origin
):
    runner, ingress, db, store, entry, source, adapter = setup(monkeypatch, tmp_path)
    task = None
    try:
        if origin == "telegram":
            task = asyncio.create_task(
                runner._handle_message(MessageEvent(text="native first", source=source))
            )
        else:
            await ingress.submit(OWNER, command(entry.session_id, "native first"))
        await started()
        if origin == "telegram":
            reply = await ingress.submit(
                OWNER, command(entry.session_id, "browser steer", "s1")
            )
            assert reply["effective_action"] == "steer"
        else:
            # Existing Telegram plain-text controls retain their native seam.
            await runner._handle_message(
                MessageEvent(text="native correction", source=source)
            )
        assert len(JournalAgent.instances) == 1
    finally:
        await finish(ingress)
        if task:
            await task
        db.close()


@pytest.mark.asyncio
async def test_legacy_spool_and_startup_resume_do_not_replay_command_input(
    monkeypatch, tmp_path
):
    runner, ingress, db, store, entry, source, adapter = setup(monkeypatch, tmp_path)
    from gateway.shutdown_flush import flush_pending_to_file

    try:
        await ingress.submit(OWNER, command(entry.session_id))
        await started()
        await ingress.submit(
            OWNER, command(entry.session_id, "queued", "q1", kind="queue")
        )
        key, _, _ = ingress._owner(entry.session_id)
        assert flush_pending_to_file({key: adapter._pending_messages[key]}) == 0
        assert store.mark_resume_pending(
            entry.session_key, reason="restart_interrupted"
        )
        assert runner._schedule_resume_pending_sessions() == 0
        assert len(JournalAgent.instances) == 1
    finally:
        await finish(ingress)
        db.close()


@pytest.mark.asyncio
async def test_distinct_ids_same_text_keep_fifo_order(monkeypatch, tmp_path):
    runner, ingress, db, store, entry, source, adapter = setup(monkeypatch, tmp_path)
    try:
        await ingress.submit(OWNER, command(entry.session_id, "first"))
        await started()
        for id, text in [("q1", "same"), ("q2", "same"), ("q3", "third")]:
            request = command(entry.session_id, text, id, kind="queue")
            await ingress.submit(OWNER, request)
            await ingress.submit(OWNER, request)
        for index in range(3):
            JournalAgent.gates[index].set()
            await started()
        assert JournalAgent.effects == ["first", "same", "same", "third"]
        receipts = [
            await ingress.command_receipt(OWNER, id) for id in ("q1", "q2", "q3")
        ]
        assert len({r["resulting_execution_id"] for r in receipts}) == 3
    finally:
        await finish(ingress)
        db.close()


@pytest.mark.asyncio
async def test_lost_ack_does_not_cancel_or_resubmit_accepted_work(
    monkeypatch, tmp_path
):
    runner, ingress, db, store, entry, source, adapter = setup(monkeypatch, tmp_path)
    import gateway.native_commands as native_commands

    encode = native_commands.receipt
    request = command(entry.session_id)

    def lose_ack(row):
        raise asyncio.CancelledError("synthetic disconnected caller")

    try:
        monkeypatch.setattr(native_commands, "receipt", lose_ack)
        with pytest.raises(asyncio.CancelledError):
            await ingress.submit(OWNER, request)
        monkeypatch.setattr(native_commands, "receipt", encode)
        await started()
        replay = await ingress.submit(OWNER, request)
        assert replay["application_state"] == "applied"
        assert JournalAgent.effects == ["first"]
    finally:
        await finish(ingress)
        db.close()


@pytest.mark.asyncio
async def test_handoff_preparation_failure_releases_claim_without_duplicate_input(
    monkeypatch, tmp_path
):
    runner, ingress, db, store, entry, source, adapter = setup(monkeypatch, tmp_path)
    prepare = runner._prepare_profile_scoped_inbound_message_text

    async def fail_queue(*args, **kwargs):
        if getattr(kwargs["event"], "_native_command_key", (None, None))[1] == "q1":
            raise RuntimeError("synthetic handoff failure")
        return await prepare(*args, **kwargs)

    try:
        await ingress.submit(OWNER, command(entry.session_id))
        await started()
        await ingress.submit(
            OWNER, command(entry.session_id, "next", "q1", kind="queue")
        )
        monkeypatch.setattr(
            runner, "_prepare_profile_scoped_inbound_message_text", fail_queue
        )
        await finish(ingress)
        q = await ingress.command_receipt(OWNER, "q1")
        assert q["application_state"] == "not_applied"
        assert q["resulting_execution_id"] is not None
        assert JournalAgent.effects == ["first"]
        assert db.native_execution(entry.session_id) is None
        assert [
            m["content"]
            for m in db.get_messages(entry.session_id)
            if m["role"] == "user"
        ] == ["first"]
    finally:
        await finish(ingress)
        db.close()


@pytest.mark.asyncio
async def test_completed_command_does_not_suppress_later_telegram_restart_resume(
    monkeypatch, tmp_path
):
    runner, ingress, db, store, entry, source, adapter = setup(monkeypatch, tmp_path)
    seen = []

    async def capture(adapter, event, key):
        seen.append(event)
        runner._release_running_agent_state(key)

    monkeypatch.setattr(runner, "_run_startup_resume_event", capture)
    monkeypatch.setattr(runner, "_is_user_authorized", lambda source: True)
    monkeypatch.setattr(
        "gateway.restart_loop_guard.check_and_record", lambda *a, **k: False
    )
    try:
        await ingress.submit(OWNER, command(entry.session_id))
        await started()
        await finish(ingress)
        # A later native-only input is the interrupted execution. The retained
        # earlier receipt must not opt that unrelated input into command replay.
        db.native_execution_open(
            entry.session_id, "telegram-later", ingress._command_owner
        )
        db.native_execution_close(
            "telegram-later", ingress._command_owner, crashed=True
        )
        assert store.mark_resume_pending(
            entry.session_key, reason="restart_interrupted"
        )
        assert runner._schedule_resume_pending_sessions() == 1
        await asyncio.gather(*list(runner._background_tasks))
        assert len(seen) == 1 and seen[0].internal
        assert seen[0].source == source
        assert (await ingress.command_receipt(OWNER, "c1"))[
            "application_state"
        ] == "applied"
    finally:
        await finish(ingress)
        db.close()


@pytest.mark.asyncio
async def test_restart_dispatches_pre_input_reservation_once_with_same_identity(
    monkeypatch, tmp_path
):
    runner, ingress, db, store, entry, source, adapter = setup(monkeypatch, tmp_path)
    import os

    request = command(entry.session_id)
    scope = ingress._command_scope(OWNER)
    db.native_command_admit(scope, request, effect="start")
    db.native_execution_open(
        entry.session_id,
        "recovered-execution",
        f"pid={os.getpid()}:birth=0:native=old",
        command=(scope, "c1"),
    )
    try:
        ingress.reconcile_commands()
        ingress.reconcile_commands()
        await started()
        row = await ingress.command_receipt(OWNER, "c1")
        assert row["resulting_execution_id"] == "recovered-execution"
        assert row["application_state"] == "applied"
        assert await ingress.submit(OWNER, request) == row
        assert JournalAgent.effects == ["first"]
    finally:
        await finish(ingress)
        db.close()
