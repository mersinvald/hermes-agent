"""Native runner and journal are real; only provider/tool behavior is synthetic."""

import asyncio
import json
import threading
from dataclasses import replace

import pytest

from gateway.native_events import NativeActivityObserver, execution_view
from tests.gateway.test_native_commands import (
    setup,
    command,
    started,
    finish,
    JournalAgent,
)
from tests.gateway.test_conversation_control import OWNER, FOREIGN


def provider_loop(agent, text, system, history, *args, **kwargs):
    # Use the actual native incremental persistence implementation. The older
    # N02 helper's deep-copy-and-flush stub re-inserts past history on a second
    # turn; it cannot establish transcript deduplication for this failure test.

    messages = list(history or [])
    message = {"role": "user", "content": text}
    messages.append(message)
    agent._native_command_context.start_input(
        agent, message, kwargs.get("persist_user_message")
    )
    assert super(JournalAgent, agent)._flush_messages_to_session_db(messages, history)
    index = len(JournalAgent.effects)
    JournalAgent.effects.append(text)
    JournalAgent.loop.call_soon_threadsafe(JournalAgent.starts.put_nowait, agent)
    assert JournalAgent.gates[index].wait(15)
    messages.append({"role": "assistant", "content": "synthetic done"})
    assert super(JournalAgent, agent)._flush_messages_to_session_db(messages, history)
    return dict(
        final_response="synthetic done",
        messages=messages,
        agent_persisted=True,
        api_calls=1,
    )


def event_setup(monkeypatch, tmp_path):
    native = setup(monkeypatch, tmp_path)
    monkeypatch.setattr("agent.conversation_loop.run_conversation", provider_loop)
    runner, ingress, db, store, entry, source, adapter = native

    async def conversation(principal, root):
        projection, _ = ingress._authorize(principal, root)
        return dict(
            schema_version="1.0",
            **projection,
            native_title=None,
            native_title_truncated=False,
            native_created_at=None,
            native_updated_at=None,
            active_execution=None,
        )

    ingress.events.conversation_provider = conversation
    return native


@pytest.mark.asyncio
async def test_two_subscribers_late_replay_and_disconnect_independent_native_work(
    monkeypatch, tmp_path
):
    runner, ingress, db, store, entry, source, adapter = event_setup(
        monkeypatch, tmp_path
    )
    one = ingress.events.subscribe(OWNER, entry.session_id)
    two = ingress.events.subscribe(OWNER, entry.session_id)
    try:
        initial = await one.poll()
        assert (await two.poll())["cursor"] == initial["cursor"]
        await ingress.submit(OWNER, command(entry.session_id))
        await started()
        a, b = await asyncio.gather(one.poll(), two.poll())
        assert a["events"] == b["events"]
        assert [e["type"] for e in a["events"]] == [
            "command_application_changed",
            "execution_state_changed",
            "command_application_changed",
            "execution_state_changed",
            "command_application_changed",
        ]
        assert a["events"][0]["execution_id"] is None
        one.close()
        assert JournalAgent.effects == ["first"]
        JournalAgent.gates[0].set()
        await finish(ingress)
        final = await two.poll()
        assert final["snapshot"]["recent_executions"][0]["state"] == "completed"
        late = await ingress.events.recover(OWNER, entry.session_id, initial["cursor"])
        assert late["events"][: len(a["events"])] == a["events"]
        assert JournalAgent.effects == ["first"]
        assert (await ingress.command_receipt(OWNER, "c1"))[
            "application_state"
        ] == "applied"
    finally:
        ingress.events.close()
        await finish(ingress)
        db.close()


@pytest.mark.asyncio
async def test_reauthorization_limits_and_cross_conversation_cursor(
    monkeypatch, tmp_path
):
    runner, ingress, db, store, entry, source, adapter = event_setup(
        monkeypatch, tmp_path
    )
    db._native_events_limits = replace(db._native_events_limits, max_subscribers=1)
    sub = ingress.events.subscribe(OWNER, entry.session_id)
    try:
        first = await sub.poll()
        with pytest.raises(RuntimeError, match="limit"):
            ingress.events.subscribe(OWNER, entry.session_id)
        with pytest.raises(PermissionError):
            await ingress.events.recover(FOREIGN, entry.session_id)
        with pytest.raises(PermissionError):
            await ingress.events.recover(OWNER, "missing", first["cursor"])
        runner._is_user_authorized_for_source = lambda source: False
        with pytest.raises(PermissionError):
            await sub.poll()
        assert sub.closed and not ingress.events._subscriptions
    finally:
        await finish(ingress)
        db.close()


@pytest.mark.asyncio
async def test_terminal_event_storage_failure_recovers_same_process_without_replay(
    monkeypatch, tmp_path
):
    runner, ingress, db, store, entry, source, adapter = event_setup(
        monkeypatch, tmp_path
    )
    try:
        await ingress.submit(OWNER, command(entry.session_id))
        first_agent = await started()
        first_context = first_agent._native_command_context
        first = await ingress.command_receipt(OWNER, "c1")
        # Accept a queue before the failure. The worker completes normally but
        # its terminal transaction fails, including the event in that transaction.
        await ingress.submit(
            OWNER, command(entry.session_id, "second", "q1", kind="queue")
        )
        observed_failure = threading.Event()
        original = db._native_event_append

        def append(conn, root, execution, kind, payload, **kwargs):
            if (
                kind == "execution_state_changed"
                and payload["state"] == "completed"
                and execution == first["resulting_execution_id"]
            ):
                observed_failure.set()
            return original(conn, root, execution, kind, payload, **kwargs)

        monkeypatch.setattr(db, "_native_event_append", append)
        db._execute_write(
            lambda c: c.execute(
                "CREATE TRIGGER no_terminal BEFORE INSERT ON native_events WHEN json_extract(NEW.body,'$.payload.state')='completed' BEGIN SELECT RAISE(ABORT,'terminal events unavailable'); END"
            )
        )
        JournalAgent.gates[0].set()
        assert await asyncio.to_thread(observed_failure.wait, 10)
        # The worker has returned, but the same native generation remains
        # reserved until its observed completion can commit. No competing owner
        # is started, and the native FIFO/cache path is retained.
        assert ingress._completion_observations
        assert (
            ingress._owner(entry.session_id)[2].execution_id
            == first["resulting_execution_id"]
        )
        assert (await ingress.command_receipt(OWNER, "c1"))[
            "application_state"
        ] == "applied"
        pending_snapshot = await ingress.events.recover(OWNER, entry.session_id)
        assert pending_snapshot["status"] == "gap"
        assert (
            pending_snapshot["snapshot"]["recent_executions"][0]["state"] == "unknown"
        )
        db._execute_write(lambda c: c.execute("DROP TRIGGER no_terminal"))
        await started()
        assert JournalAgent.effects == ["first", "second"]
        assert not ingress._completion_observations
        newer = ingress._owner(entry.session_id)[2].execution_id
        assert newer != first["resulting_execution_id"]
        first_context.finish()  # Delayed duplicate completion from old worker.
        assert db.native_execution(entry.session_id)["execution_id"] == newer
        assert ingress._owner(entry.session_id)[2].execution_id == newer
        JournalAgent.gates[1].set()
        await finish(ingress)
        result = await ingress.events.recover(OWNER, entry.session_id)
        assert [e["state"] for e in result["snapshot"]["recent_executions"]] == [
            "completed",
            "completed",
        ]
        users = [m for m in db.get_messages(entry.session_id) if m["role"] == "user"]
        assert len(users) == 2, [(m.get("id"), m.get("content")) for m in users]
    finally:
        db._execute_write(lambda c: c.execute("DROP TRIGGER IF EXISTS no_terminal"))
        ingress.reconcile_commands()
        await finish(ingress)
        db.close()


@pytest.mark.asyncio
async def test_tool_callbacks_preserve_identity_thread_and_existing_consumer(
    monkeypatch, tmp_path
):
    runner, ingress, db, store, entry, source, adapter = event_setup(
        monkeypatch, tmp_path
    )
    try:
        await ingress.submit(OWNER, command(entry.session_id))
        agent = await started()
        cursor = (await ingress.events.recover(OWNER, entry.session_id))["cursor"]
        seen = []
        original = lambda *args: seen.append((threading.get_ident(), args))
        # Real TurnRunner already installed observers on this agent. Compose
        # an existing callback as a separate consumer and exercise its IDs.
        context = agent._native_command_context
        from types import SimpleNamespace

        target = SimpleNamespace(
            tool_start_callback=original, tool_complete_callback=original
        )
        observer = NativeActivityObserver(context, target).install()

        def tools():
            tid = threading.get_ident()
            for call in ("call-1", "call-2"):
                target.tool_start_callback(call, "terminal", {"secret": "provider-key"})
                target.tool_complete_callback(
                    call, "terminal", {}, '{"output":"private"}'
                )
            return tid

        thread_id = await asyncio.to_thread(tools)
        observer.restore()
        # Exercise the observer installed by the real TurnRunner as well.
        await asyncio.to_thread(
            agent.tool_start_callback, "installed-call", "terminal", {}
        )
        await asyncio.to_thread(
            agent.tool_complete_callback, "installed-call", "terminal", {}, "{}"
        )
        result = await ingress.events.recover(OWNER, entry.session_id, cursor)
        events = result["events"]
        assert [
            (e["payload"]["activity_id"], e["payload"]["state"]) for e in events
        ] == [
            ("call-1", "running"),
            ("call-1", "completed"),
            ("call-2", "running"),
            ("call-2", "completed"),
            ("installed-call", "running"),
            ("installed-call", "completed"),
        ]
        assert all(e["payload"]["detail"] == {"tool_name": "terminal"} for e in events)
        assert len(seen) == 4 and all(tid == thread_id for tid, _ in seen)
        assert (
            target.tool_start_callback is original
            and target.tool_complete_callback is original
        )
        assert "provider-key" not in json.dumps(
            result
        ) and '"private"' not in json.dumps(result)
        db._execute_write(
            lambda c: c.execute(
                "CREATE TRIGGER no_tools BEFORE INSERT ON native_events BEGIN SELECT RAISE(ABORT,'unavailable'); END"
            )
        )
        previous = result["cursor"]
        observer.emit("tool_changed", {"activity_id": "call-3", "state": "running"})
        db._execute_write(lambda c: c.execute("DROP TRIGGER no_tools"))
        recovered = await ingress.events.recover(OWNER, entry.session_id, previous)
        assert recovered["status"] == "gap"
    finally:
        await finish(ingress)
        db.close()


def test_closed_ownership_and_legacy_timestamps_are_not_success():
    row = dict(
        execution_id="e1",
        conversation_id="c1",
        origin="unknown",
        state="closed",
        observed_state="unknown",
        created_at=None,
        updated_at=None,
    )
    view = execution_view(row)
    assert view["state"] == "unknown" and view["created_at"] is None


@pytest.mark.asyncio
async def test_storage_read_failure_is_not_empty_success(monkeypatch, tmp_path):
    import sqlite3

    runner, ingress, db, store, entry, source, adapter = event_setup(
        monkeypatch, tmp_path
    )
    try:
        db._conn.set_authorizer(
            lambda action, table, *_: (
                sqlite3.SQLITE_DENY
                if action == sqlite3.SQLITE_READ and table == "native_events"
                else sqlite3.SQLITE_OK
            )
        )
        with pytest.raises(sqlite3.DatabaseError):
            await ingress.events.recover(OWNER, entry.session_id)
    finally:
        db._conn.set_authorizer(None)
        await finish(ingress)
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "native_result,expected",
    [
        ({"completed": False, "partial": True, "error": error}, "failed")
        for error in (
            "Reasoning exhausted the output token budget",
            "Response truncated due to output length limit",
            "Incomplete REASONING_SCRATCHPAD after 2 retries",
            "Codex response remained incomplete after 3 continuation attempts",
        )
    ]
    + [
        ({"completed": False, "partial": True}, "unknown"),
        ({"completed": False}, "unknown"),
        ({"failed": True}, "failed"),
        ({"interrupted": True, "partial": True}, "interrupted"),
        ({"completed": True}, "completed"),
    ],
)
async def test_native_incomplete_outcomes_preserve_actual_evidence(
    monkeypatch, tmp_path, native_result, expected
):
    runner, ingress, db, store, entry, source, adapter = event_setup(
        monkeypatch, tmp_path
    )

    def incomplete_loop(*args, **kwargs):
        # Actual AIAgent/TurnRunner and incremental transcript writer; synthetic
        # provider return matches the native incomplete/partial producer shapes.
        return {**provider_loop(*args, **kwargs), **native_result}

    monkeypatch.setattr("agent.conversation_loop.run_conversation", incomplete_loop)
    try:
        await ingress.submit(OWNER, command(entry.session_id))
        await started()
        JournalAgent.gates[0].set()
        await finish(ingress)
        snapshot = await ingress.events.recover(OWNER, entry.session_id)
        assert snapshot["snapshot"]["recent_executions"][0]["state"] == expected
        assert (await ingress.command_receipt(OWNER, "c1"))[
            "application_state"
        ] == "applied"
        assert (
            len([m for m in db.get_messages(entry.session_id) if m["role"] == "user"])
            == 1
        )
    finally:
        await finish(ingress)
        db.close()
