"""Actual native runner/journal/interrupt with synthetic provider and remote IO."""

import asyncio
import threading

import pytest

from gateway.native_cancellation import NativeCancellationController
from hermes_state_commands import CommandConflict
from plugins.platforms.a2a import tools, protocol
from plugins.platforms.a2a.cancellation import TaskControllerClient
from tests.gateway.test_native_events import event_setup, provider_loop
from tests.gateway.test_native_commands import command, started, finish, JournalAgent
from tests.gateway.test_conversation_control import OWNER, FOREIGN
from tests.state.test_native_cancellation import cancel


@pytest.fixture(autouse=True)
def initialized_interrupt_state(monkeypatch):
    original = JournalAgent.__init__

    def initialize(agent, **kwargs):
        original(agent, **kwargs)
        # The older lease/command fixture intentionally skipped AIAgent.__init__.
        # Install native initialization fields needed by actual hard_interrupt.
        agent._active_children_lock = threading.Lock()
        agent._active_children = []
        agent.quiet_mode = True
        agent._hard_interrupt_requested = threading.Event()

    monkeypatch.setattr(JournalAgent, "__init__", initialize)


async def settled(controller):
    await asyncio.sleep(0.02)  # Allow a coalesced durable wake to enter the pump.
    for _ in range(1000):
        if not controller._jobs and not controller._reconciling:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("control worker did not settle")


@pytest.mark.asyncio
async def test_actual_interrupt_retains_writer_until_return_and_keeps_queue_fallback(
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
        await ingress.submit(
            OWNER,
            command(
                entry.session_id,
                "unconsumed",
                "s1",
                kind="steer",
                target_execution_id=execution.execution_id,
            ),
        )
        request = cancel(entry.session_id, execution.execution_id)
        accepted = await ingress.submit(OWNER, request)
        assert (
            accepted["receipt_kind"] == "cancel" and "application_state" not in accepted
        )
        await settled(ingress.cancellations)
        pending = await ingress.command_receipt(OWNER, request["command_id"])
        assert pending["native"] == {
            "request_state": "requested",
            "observed_state": "running",
        }
        assert (
            pending["queue_scope"] == "execution_only"
            and pending["queued_and_unconsumed_steers"] == "retained"
        )
        assert first._interrupt_requested
        assert ingress._owner(entry.session_id)[2] == execution
        assert JournalAgent.effects == ["first"]
        assert (
            db.native_execution(entry.session_id)["execution_id"]
            == execution.execution_id
        )
        assert (await ingress.command_receipt(OWNER, "c1"))[
            "application_state"
        ] == "applied"
        duplicate = await ingress.submit(OWNER, request)
        assert duplicate["payload_fingerprint"] == accepted["payload_fingerprint"]
        with pytest.raises((PermissionError, LookupError)):
            await ingress.submit(FOREIGN, request)
        with pytest.raises(CommandConflict):
            await ingress.submit(OWNER, {**request, "payload": {"reason": "different"}})
        JournalAgent.gates[0].set()
        newer = await started()
        assert (
            ingress._owner(entry.session_id)[2].execution_id != execution.execution_id
        )
        await ingress.submit(OWNER, request)  # Old duplicate cannot target new owner.
        await settled(ingress.cancellations)
        assert not newer._interrupt_requested
        assert (await ingress.command_receipt(OWNER, request["command_id"]))["native"][
            "observed_state"
        ] == "interrupted"
        JournalAgent.gates[1].set()
        await started()
        JournalAgent.gates[2].set()
        await finish(ingress)
        assert JournalAgent.effects == ["first", "queued", "unconsumed"]
        assert (await ingress.command_receipt(OWNER, "s1"))["fallback"] == {
            "reason": "unconsumed_steer"
        }
        assert (
            len([m for m in db.get_messages(entry.session_id) if m["role"] == "user"])
            == 3
        )
    finally:
        await settled(ingress.cancellations)
        await finish(ingress)
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("lost_ack", [False, True])
async def test_actual_dispatch_is_owned_and_cancel_reads_after_response_or_timeout(
    monkeypatch, tmp_path, lost_ack
):
    runner, ingress, db, store, entry, source, adapter = event_setup(
        monkeypatch, tmp_path
    )
    config = {
        "a2a_agents": {
            "peer": {
                "url": "http://peer.test/agent/",
                "auth": {"type": "bearer", "token": "SYNTHETIC_NATIVE_TOKEN"},
            }
        }
    }
    monkeypatch.setattr(tools, "_load_config", lambda: config)
    monkeypatch.setattr(
        tools,
        "_fetch_card",
        lambda *_: protocol.build_agent_card(
            name="peer", url="http://peer.test/agent/", description="synthetic"
        ),
    )
    methods = []
    remote_state = ["TASK_STATE_WORKING"]

    def post(url, body, headers, timeout):
        methods.append(body["method"])
        task = {
            "id": "remote-task",
            "contextId": "remote-context",
            "status": {"state": remote_state[0]},
        }
        if body["method"] == "SendMessage":
            return {"jsonrpc": "2.0", "id": body["id"], "result": {"task": task}}
        if body["method"] == "CancelTask":
            remote_state[0] = task["status"]["state"] = "TASK_STATE_CANCELED"
            if lost_ack:
                raise TimeoutError("response lost after remote acceptance")
        return {"jsonrpc": "2.0", "id": body["id"], "result": task}

    monkeypatch.setattr(tools, "_http_post_json", post)
    ingress.cancellations.client = TaskControllerClient(post=post)

    def loop(agent, *args, **kwargs):
        answer = tools.a2a_call({"agent": "peer", "message": "synthetic delegation"})
        assert "remote-task" in answer, answer
        return provider_loop(agent, *args, **kwargs)

    monkeypatch.setattr("agent.conversation_loop.run_conversation", loop)
    try:
        await ingress.submit(OWNER, command(entry.session_id))
        await started()
        execution = ingress._owner(entry.session_id)[2].execution_id
        rows = db.native_remote_targets(execution)
        assert (
            len(rows) == 1
            and rows[0]["task_id"] == "remote-task"
            and rows[0]["owner"] == ingress._command_owner
        )
        request = cancel(entry.session_id, execution)
        await ingress.submit(OWNER, request)
        await settled(ingress.cancellations)
        receipt = await ingress.command_receipt(OWNER, "cancel-1")
        assert methods == ["SendMessage", "GetTask", "CancelTask", "GetTask"]
        assert receipt["remote_targets"][0]["request_state"] == (
            "unknown" if lost_ack else "acknowledged"
        )
        assert receipt["remote_targets"][0]["observed_task_state"] == "canceled"
        assert receipt["remote_coverage"] == {
            "known_count": 1,
            "task_id_unknown_count": 0,
            "descendants": "unknown",
            "has_more": False,
        }
        serialized = __import__("json").dumps(receipt)
        assert all(
            value not in serialized
            for value in ("remote-task", "peer.test", "SYNTHETIC_NATIVE_TOKEN", "pid=")
        )
        await ingress.submit(OWNER, request)
        await settled(ingress.cancellations)
        assert methods.count("CancelTask") == 1
        # New controller simulates restart of control observation. Persisted
        # unknown ACK remains read-only; no remote or native input is replayed.
        ingress.cancellations.stop_recovery()
        replacement = NativeCancellationController(
            ingress, client=TaskControllerClient(post=post)
        )
        replacement.wake(
            db.native_cancel_lookup(ingress._command_scope(OWNER), "cancel-1")
        )
        await settled(replacement)
        replacement.stop_recovery()
        assert methods.count("CancelTask") == 1 and methods.count("SendMessage") == 1
    finally:
        await settled(ingress.cancellations)
        await finish(ingress)
        db.close()


@pytest.mark.asyncio
async def test_durable_fair_control_jobs_bound_and_process_all_known_chains(tmp_path):
    from collections import Counter
    from types import SimpleNamespace
    from agent.native_execution_context import NativeExecutionOrigin
    from hermes_state_commands import process_owner
    from tests.state.test_native_cancellation import database, prepare

    db = database(tmp_path)
    owner = process_owner()
    commands = []
    # More executions than both the concurrency limit and a journal page. Every
    # first-page remote stays working forever; later executions must still run.
    for index in range(107):
        root, execution = f"root-{index}", f"execution-{index}"
        db.create_session(root, "telegram")
        db.native_execution_open(root, execution, owner)
        origin = NativeExecutionOrigin(root, execution, owner)
        dispatch = prepare(db, origin)
        db.native_remote_dispatch_observe(
            origin, dispatch, f"task-{index}", f"ctx-{index}", "working"
        )
        db.native_execution_close(execution, owner)
        commands.append(
            db.native_cancel_admit("owner", cancel(root, execution, f"cancel-{index}"))[
                0
            ]
        )
    calls = Counter()
    active = peak = 0

    async def post(url, body, headers, timeout):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.001)
            identity = body["params"]["id"]
            calls[(identity, body["method"])] += 1
            return {
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {
                    "id": identity,
                    "contextId": identity.replace("task-", "ctx-"),
                    "status": {"state": "TASK_STATE_WORKING"},
                },
            }
        finally:
            active -= 1

    client = TaskControllerClient(
        resolve_peer=lambda name: {
            "url": "http://gateway.test/agent/peer/",
            "auth": {},
            "timeout": 1,
        },
        post=post,
    )
    ingress = SimpleNamespace(
        db=db, runner=SimpleNamespace(_draining=False), _owner=lambda root: None
    )
    controller = NativeCancellationController(
        ingress, client=client, max_control_jobs=3, poll_interval=1
    )
    try:
        # Simulates many accepted commands and startup recovery without retaining
        # one Python task or an executor future for each durable pending row.
        for row in commands:
            controller.wake(row)
            assert len(controller._jobs) <= 3 and len(controller._dirty) <= 3
        for _ in range(2000):
            assert len(controller._jobs) <= 3
            if sum(method == "CancelTask" for _, method in calls) == 107:
                break
            await asyncio.sleep(0.005)
        else:
            pytest.fail("later durable cancellation starved")
        assert peak <= 3
        assert all(
            count == 1 for (_, method), count in calls.items() if method == "CancelTask"
        )
        await settled(controller)
        controller.stop_recovery()
        await asyncio.sleep(0.01)
        # New process/controller reads stored attempts, never repeats writes.
        db._execute_write(
            lambda conn: conn.execute(
                "UPDATE native_cancel_commands SET next_poll_at=0"
            )
        )
        recovered = NativeCancellationController(
            ingress, client=client, max_control_jobs=3, poll_interval=1
        )
        await recovered.reconcile()
        await settled(recovered)
        recovered.stop_recovery()
        assert all(
            count == 1 for (_, method), count in calls.items() if method == "CancelTask"
        )
    finally:
        controller.stop_recovery()
        await settled(controller)
        await asyncio.sleep(0.01)
        db.close()


@pytest.mark.live_system_guard_bypass  # Signals only the asserted test-owned subprocess below.
@pytest.mark.asyncio
async def test_cancel_reaches_actual_local_tool_process_and_waits_for_native_return(
    monkeypatch, tmp_path
):
    import shlex
    from tools.environments.local import LocalEnvironment
    from tools.interrupt import clear_current_thread_interrupt

    runner, ingress, db, store, entry, source, adapter = event_setup(
        monkeypatch, tmp_path
    )
    marker = tmp_path / "owned-tool-started"
    tool_results = []
    release_return = threading.Event()

    def loop(agent, *args, **kwargs):
        result = provider_loop(agent, *args, **kwargs)
        agent._execution_thread_id = threading.get_ident()
        environment = LocalEnvironment(cwd=str(tmp_path), timeout=20)
        kill_owned = environment._kill_process

        def checked_kill(proc):
            assert str(marker) in " ".join(proc.args)
            return kill_owned(proc)

        environment._kill_process = checked_kill
        try:
            tool_results.append(
                environment.execute(
                    f"printf started > {shlex.quote(str(marker))}; sleep 15", timeout=20
                )
            )
            assert release_return.wait(10)
            return {
                **result,
                "interrupted": bool(agent._interrupt_requested),
                "completed": False,
            }
        finally:
            environment.cleanup()
            clear_current_thread_interrupt()

    monkeypatch.setattr("agent.conversation_loop.run_conversation", loop)
    try:
        await ingress.submit(OWNER, command(entry.session_id))
        await started()
        execution = ingress._owner(entry.session_id)[2]
        JournalAgent.gates[0].set()
        for _ in range(1000):
            if marker.exists():
                break
            await asyncio.sleep(0.01)
        assert marker.exists()
        await ingress.submit(OWNER, cancel(entry.session_id, execution.execution_id))
        for _ in range(1000):
            if tool_results:
                break
            await asyncio.sleep(0.01)
        assert tool_results and tool_results[0]["returncode"] == 130, tool_results
        assert "[Command interrupted]" in tool_results[0]["output"]
        assert ingress._owner(entry.session_id)[2] == execution
        assert (await ingress.command_receipt(OWNER, "cancel-1"))["native"][
            "observed_state"
        ] == "running"
        release_return.set()
        await finish(ingress)
        assert (await ingress.command_receipt(OWNER, "cancel-1"))["native"][
            "observed_state"
        ] == "interrupted"
        assert (
            len([m for m in db.get_messages(entry.session_id) if m["role"] == "user"])
            == 1
        )
    finally:
        release_return.set()
        ingress.stop_command_recovery()
        await finish(ingress)
        await settled(ingress.cancellations)
        db.close()


@pytest.mark.asyncio
async def test_atomic_compact_control_recovery_private_events_and_cursor_pages(
    monkeypatch, tmp_path
):
    import json
    from agent.native_execution_context import NativeExecutionOrigin
    from tests.state.test_native_cancellation import prepare

    runner, ingress, db, store, entry, source, adapter = event_setup(
        monkeypatch, tmp_path
    )
    try:
        await ingress.submit(OWNER, command(entry.session_id))
        await started()
        execution = ingress._owner(entry.session_id)[2]
        origin = NativeExecutionOrigin(
            entry.session_id, execution.execution_id, ingress._command_owner
        )
        dispatches = [prepare(db, origin) for _ in range(103)]
        before = await ingress.events.recover(OWNER, entry.session_id)
        request = cancel(entry.session_id, execution.execution_id)
        accepted = await ingress.submit(OWNER, request)
        await settled(ingress.cancellations)
        recovery = await ingress.events.recover(
            OWNER, entry.session_id, before["cursor"]
        )
        snapshot = recovery["snapshot"]
        compact = snapshot["control_receipts"][0]
        assert compact["command_id"] == accepted["command_id"]
        assert compact["conversation_id"] == entry.session_id
        assert compact["remote_targets"] == []
        assert compact["remote_coverage"] == dict(
            known_count=103,
            task_id_unknown_count=103,
            descendants="unknown",
            has_more=True,
        )
        assert snapshot["control_receipt_coverage"] == dict(
            limit=10, has_more=False, target_details="command_lookup"
        )
        assert (
            snapshot["recent_executions"][0]["remote_cancellation"]["state"]
            == "requested"
        )
        controls = [
            event
            for event in recovery["events"]
            if event["type"] == "cancel_state_changed"
        ]
        assert controls and all(
            event["payload"]["command_id"] == request["command_id"]
            and event["execution_id"] == execution.execution_id
            for event in controls
        )
        foreign = db.native_event_recovery(
            entry.session_id, "other-grantee", before["cursor"]
        )
        assert not foreign["control_receipts"]
        assert not [
            event
            for event in foreign["events"]
            if event["type"] == "cancel_state_changed"
        ]
        assert (
            foreign["cursor"] == recovery["cursor"] and foreign["status"] == "current"
        )
        ids = []
        cursor = compact["next_remote_cursor"]
        while cursor:
            page = await ingress.command_receipt(
                OWNER, "cancel-1", remote_cursor=cursor, remote_limit=31
            )
            assert page["remote_coverage"]["known_count"] == 103
            ids.extend(target["dispatch_id"] for target in page["remote_targets"])
            cursor = page["next_remote_cursor"]
        assert ids == dispatches and len(set(ids)) == 103
        with pytest.raises(ValueError):
            await ingress.command_receipt(
                OWNER, "cancel-1", remote_cursor="1." + "0" * 64
            )
        # A transition immediately after transaction completion must remain
        # absent from that snapshot and present after its exact cursor boundary.
        original = db._execute_write
        armed = [True]

        def after_capture(fn):
            value = original(fn)
            if armed[0] and isinstance(value, dict) and "captured_at" in value:
                armed[0] = False
                db.native_cancel_admit(
                    ingress._command_scope(OWNER),
                    cancel(entry.session_id, execution.execution_id, "after-capture"),
                )
            return value

        monkeypatch.setattr(db, "_execute_write", after_capture)
        atomic = await ingress.events.recover(OWNER, entry.session_id)
        assert "after-capture" not in [
            r["command_id"] for r in atomic["snapshot"]["control_receipts"]
        ]
        following = await ingress.events.recover(
            OWNER, entry.session_id, atomic["cursor"]
        )
        assert any(
            event["payload"].get("command_id") == "after-capture"
            for event in following["events"]
        )
        evidence = {"receipt": accepted, "recovery": recovery, "last_page": page}
        # Saved only when explicitly opted into synthetic review evidence.
        import os

        if os.environ.get("HERMES_N05_EVIDENCE"):
            from pathlib import Path

            Path(os.environ["HERMES_N05_EVIDENCE"]).write_text(
                json.dumps(evidence, indent=2)
            )
    finally:
        ingress.stop_command_recovery()
        await finish(ingress)
        await settled(ingress.cancellations)
        db.close()


@pytest.mark.asyncio
async def test_slow_multi_target_quantum_does_not_delay_later_native_interrupt(
    monkeypatch, tmp_path
):
    from agent.native_execution_context import NativeExecutionOrigin
    from tests.state.test_native_cancellation import prepare

    runner, ingress, db, store, entry, source, adapter = event_setup(
        monkeypatch, tmp_path
    )
    calls = []
    first_slow = asyncio.Event()

    async def post(url, body, headers, timeout):
        task = body["params"]["id"]
        calls.append((task, body["method"]))
        if task in {"slow-0-0", "slow-1-0"}:
            first_slow.set()
            await asyncio.sleep(1)  # Actual coroutine canceled by total quantum.
        state = (
            "TASK_STATE_CANCELED"
            if body["method"] == "CancelTask"
            else "TASK_STATE_WORKING"
        )
        return dict(
            jsonrpc="2.0",
            id=body["id"],
            result=dict(id=task, contextId="ctx-" + task, status=dict(state=state)),
        )

    client = TaskControllerClient(
        resolve_peer=lambda name: {
            "url": "http://gateway.test/agent/peer/",
            "auth": {},
            "timeout": 1,
        },
        post=post,
    )
    controller = NativeCancellationController(
        ingress,
        client=client,
        max_control_jobs=2,
        poll_interval=1,
        control_turn_seconds=0.1,
    )
    ingress.cancellations = controller
    try:
        for index in range(2):
            root, identity = f"slow-root-{index}", f"slow-execution-{index}"
            db.create_session(root, "telegram")
            db.native_execution_open(root, identity, ingress._command_owner)
            origin = NativeExecutionOrigin(root, identity, ingress._command_owner)
            for number in range(4):
                dispatch = prepare(db, origin)
                task = f"slow-{index}-{number}"
                db.native_remote_dispatch_observe(
                    origin, dispatch, task, "ctx-" + task, "working"
                )
            db.native_execution_close(identity, ingress._command_owner)
            controller.wake(
                db.native_cancel_admit(
                    ingress._command_scope(OWNER),
                    cancel(root, identity, f"stop-slow-{index}"),
                )[0]
            )
        await asyncio.wait_for(first_slow.wait(), 1)
        await ingress.submit(OWNER, command(entry.session_id))
        agent = await started()
        execution = ingress._owner(entry.session_id)[2]
        origin = NativeExecutionOrigin(
            entry.session_id, execution.execution_id, ingress._command_owner
        )
        dispatch = prepare(db, origin)
        db.native_remote_dispatch_observe(
            origin, dispatch, "later", "ctx-later", "working"
        )
        await ingress.submit(OWNER, cancel(entry.session_id, execution.execution_id))
        assert (
            agent._interrupt_requested
        )  # Native signal did not wait for an HTTP slot.
        assert ingress._owner(entry.session_id)[2] == execution
        for _ in range(1000):
            if len([c for c in calls if c[1] == "CancelTask"]) == 7:
                break
            await asyncio.sleep(0.01)
        writes = [task for task, method in calls if method == "CancelTask"]
        assert len(writes) == 7 and len(set(writes)) == 7
        assert writes.index("later") < writes.index("slow-0-3")
        assert writes.index("later") < writes.index("slow-1-3")
        for index in range(2):
            first = db.native_cancel_snapshot(
                ingress._command_scope(OWNER), f"stop-slow-{index}"
            )["targets"][0]
            assert (
                first["cancel_request_state"] == "failed"
                and not first["write_reserved"]
            )
    finally:
        ingress.stop_command_recovery()
        await finish(ingress)
        await settled(controller)
        db.close()


@pytest.mark.asyncio
async def test_known_not_sent_preflight_needs_fresh_command_and_preserves_old_receipt(
    tmp_path,
):
    from types import SimpleNamespace
    from agent.native_execution_context import NativeExecutionOrigin
    from tests.state.test_native_cancellation import database, open_input, prepare

    db = database(tmp_path)
    owner = open_input(db)
    dispatch = prepare(db, NativeExecutionOrigin("root", "e1", owner))
    db.native_remote_dispatch_observe(
        NativeExecutionOrigin("root", "e1", owner), dispatch, "task", "ctx", "working"
    )
    db.native_execution_close("e1", owner)
    calls, broken = [], [True]

    async def post(url, body, headers, timeout):
        calls.append(body["method"])
        if broken[0]:
            raise TimeoutError("synthetic preflight")
        return dict(
            jsonrpc="2.0",
            id=body["id"],
            result=dict(
                id="task", contextId="ctx", status=dict(state="TASK_STATE_WORKING")
            ),
        )

    client = TaskControllerClient(
        resolve_peer=lambda _: {
            "url": "http://gateway.test/agent/peer/",
            "auth": {},
            "timeout": 1,
        },
        post=post,
    )
    ingress = SimpleNamespace(
        db=db,
        runner=SimpleNamespace(_draining=False),
        _owner=lambda root: None,
        _completion_lock=threading.Lock(),
        _completion_observations={},
        _command_owner=owner,
    )
    controller = NativeCancellationController(ingress, client=client)
    try:
        first = db.native_cancel_admit("owner", cancel())[0]
        controller.wake(first)
        await settled(controller)
        old = controller.receipt_view(db.native_cancel_snapshot("owner", "cancel-1"))
        assert old["remote_targets"][0]["request_state"] == "failed"
        assert old["remote_targets"][0]["request_origin"] == "this_command"
        broken[0] = False
        controller.wake(first)  # Same stable ID cannot replay even known-not-sent.
        await settled(controller)
        assert calls == ["GetTask"]
        second = db.native_cancel_admit("owner", cancel(identity="manual-2"))[0]
        controller.wake(second)
        await settled(controller)
        assert calls == ["GetTask", "GetTask", "CancelTask", "GetTask"]
        again = controller.receipt_view(db.native_cancel_snapshot("owner", "cancel-1"))
        fresh = controller.receipt_view(db.native_cancel_snapshot("owner", "manual-2"))
        assert again["remote_targets"][0]["request_state"] == "failed"
        assert fresh["remote_targets"][0]["request_state"] == "acknowledged"
        assert fresh["remote_targets"][0]["request_origin"] == "this_command"
        third = db.native_cancel_admit("owner", cancel(identity="manual-3"))[0]
        controller.wake(third)
        await settled(controller)
        shared = controller.receipt_view(db.native_cancel_snapshot("owner", "manual-3"))
        assert shared["remote_targets"][0]["request_origin"] == "prior_command"
        assert calls.count("CancelTask") == 1
    finally:
        controller.stop_recovery()
        await settled(controller)
        db.close()


@pytest.mark.asyncio
async def test_rejected_sent_request_still_reconciles_task_without_retry(tmp_path):
    from types import SimpleNamespace
    from agent.native_execution_context import NativeExecutionOrigin
    from plugins.platforms.a2a.cancellation import ControlObservation
    from tests.state.test_native_cancellation import database, open_input, prepare

    db = database(tmp_path)
    owner = open_input(db)
    origin = NativeExecutionOrigin("root", "e1", owner)
    dispatch = prepare(db, origin)
    db.native_remote_dispatch_observe(origin, dispatch, "task", "ctx", "working")
    command_row = db.native_cancel_admit("owner", cancel())[0]
    db.native_remote_cancel_reserve("owner", "cancel-1", dispatch)
    db.native_remote_cancel_note(
        dispatch, ControlObservation("rejected", "working", "cancel_rpc_rejected", True)
    )
    observed = []

    async def observe(target, **kwargs):
        observed.append(target.task_id)
        return ControlObservation(
            "already_completed", "completed", "task_reports_completed"
        )

    controller = NativeCancellationController(
        SimpleNamespace(db=db), client=SimpleNamespace(observe=observe)
    )
    try:
        assert db.native_cancel_recovery_rows()
        await controller._remote(command_row, db.native_remote_targets("e1")[0])
        result = db.native_cancel_snapshot("owner", "cancel-1")["targets"][0]
        assert result["cancel_request_state"] == "rejected"
        assert result["observed_state"] == "completed" and observed == ["task"]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_continuous_actual_observation_wakes_cannot_reclaim_fair_slots(tmp_path):
    from types import SimpleNamespace
    from agent.native_execution_context import NativeExecutionOrigin
    from hermes_state_commands import process_owner
    from tests.state.test_native_cancellation import database, prepare

    db = database(tmp_path)
    owner = process_owner()
    commands = []
    origins = {}
    for index in range(5):
        root, execution = f"root-{index}", f"e-{index}"
        db.create_session(root, "telegram")
        db.native_execution_open(root, execution, owner)
        origin = NativeExecutionOrigin(root, execution, owner)
        for number in range(3):
            task = f"task-{index}-{number}"
            origins[task] = origin
            dispatch = prepare(db, origin)
            db.native_remote_dispatch_observe(
                origin, dispatch, task, "ctx-" + task, "working"
            )
        db.native_execution_close(execution, owner)
        commands.append(
            db.native_cancel_admit("owner", cancel(root, execution, f"cancel-{index}"))[
                0
            ]
        )
    writes = []

    async def post(url, body, headers, timeout):
        task = body["params"]["id"]
        if task.startswith(("task-0-", "task-1-")):
            for _ in range(6):
                # Same callback seam used by actual late A2A status observations.
                controller.observed_dispatch(origins[task])
                await asyncio.sleep(0.002)
        if body["method"] == "CancelTask":
            writes.append(task)
        return dict(
            jsonrpc="2.0",
            id=body["id"],
            result=dict(
                id=task,
                contextId="ctx-" + task,
                status=dict(state="TASK_STATE_WORKING"),
            ),
        )

    client = TaskControllerClient(
        resolve_peer=lambda _: {
            "url": "http://gateway.test/agent/peer/",
            "auth": {},
            "timeout": 1,
        },
        post=post,
    )
    ingress = SimpleNamespace(
        db=db, runner=SimpleNamespace(_draining=False), _owner=lambda root: None
    )
    controller = NativeCancellationController(
        ingress, client=client, max_control_jobs=2, poll_interval=1
    )
    try:
        for row in commands:
            controller.wake(row)
        for _ in range(1000):
            assert len(controller._jobs) <= 2 and len(controller._dirty) <= 2
            if "task-4-0" in writes:
                break
            await asyncio.sleep(0.005)
        assert "task-4-0" in writes
        assert "task-0-2" not in writes and "task-1-2" not in writes
    finally:
        controller.stop_recovery()
        await settled(controller)
        db.close()
