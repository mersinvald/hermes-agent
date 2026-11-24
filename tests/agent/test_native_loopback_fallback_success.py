"""Check one current owned HTTP fallback after an actual main timeout."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import pytest
from hermes_content_policy import FailedDispatch

from agent import native_main_payload as native
from agent.content_policy import NativeContentPolicyDenied
from tests.agent.owned_loopback_receiver import OwnedReceiver, ResponseStep, response_bytes
from tests.agent.test_native_controlled_conversation import (
    compression_excluded as compression_excluded,
    construct,
    excluded as excluded,
)
from tests.agent.test_native_controlled_fallback import bind_fallback, fixed_turn
from tests.agent.test_native_loopback_fallback_denial import (
    ACK,
    QUALIFICATION,
    MainSendObservation,
    RetainedMainPort,
    assert_plan,
    assert_receipt,
    build_world,
    charges,
    graph,
    ordinary_effects as ordinary_effects,
    qualify,
    record as record,
)
from tests.agent.test_native_main_payload import (
    ControlledPort,
    World,
    caller_boundary as caller_boundary,
    canonical,
    reply,
)


EXPECTED_REPLY = b'{"choices":[{"finish_reason":"stop","index":0,"message":{"content":"The blue folder stays.","role":"assistant"}}],"created":12,"id":"invented_fallback_reply","model":"invented_model_5","object":"chat.completion","usage":{"completion_tokens":2,"prompt_tokens":3,"total_tokens":5}}'


def fallback_reply_bytes():
    value = json.loads(reply("The blue folder stays."))
    value.update(id="invented_fallback_reply", model="invented_model_5")
    return canonical(value)


def observe_failed_main(world, b, record):
    retained = {}

    def trigger():
        adapter = world.adapter
        if retained or adapter is None or adapter._fallback_phase != "authenticate" or adapter._fallback_frame is None:
            return
        failed, = adapter.failed_main_attempts
        frame = adapter._fallback_frame
        assert type(failed) is native.NativeFailedMainAttempt
        assert frame.failed_main is failed and frame.binding is adapter._fallback_binding
        assert frame.activated is False and frame.source_content is frame.invocation is None
        entry = adapter._runtime._failures[id(failed.shared_failure)]
        assert entry.token is failed.shared_failure and entry.facts is failed.failure_facts
        assert entry.request_handle is world.request
        assert failed.send_observation.exception_type is TimeoutError
        assert type(failed.shared_failure) is FailedDispatch and type(frame.token) is native.NativeFallbackInput
        assert frame.token is not failed.shared_failure and world.now == 10
        with b.condition:
            b.check()
            assert b.ready.is_set() and b.thread.is_alive() and not b._connections
            assert len(b.rows) == 1 and b.rows[0]["settled"] and b._mark is None
            prefix = dict(b.rows[0])
            listener, listener_id, bound = b.listener, b._listener_id, b.bound
            assert listener_id == id(listener)
        retained.update(failed=failed, frame=frame, entry=entry, token=frame.token,
            binding=frame.binding, parents=frame.parents, parent_requests=frame.parent_requests,
            prefix=prefix, listener=listener, listener_id=listener_id, bound=bound,
            graph=graph(adapter._runtime), charges=charges(adapter._runtime))
        record("trigger", dict(phase=adapter._fallback_phase, now=world.now,
            frame_id=id(frame), token_id=id(frame.token), failed_id=id(failed), binding_id=id(frame.binding),
            shared_token_id=id(failed.shared_failure), entry_id=id(entry), original_registry_associations=True,
            native_token_distinct=True, activated=False, source_content=None, invocation=None,
            prefix=prefix, listener_id=listener_id, bound=bound, graph=retained["graph"],
            charges=retained["charges"], receiver_window=False))
    return trigger, retained


def test_inert_current_fallback_success_plan(record):
    a, b = OwnedReceiver(observation_seconds=590.0), OwnedReceiver(observation_seconds=590.0)
    try:
        world = build_world(a, b)
        assert_plan(world, a, b)
        hook, observed = observe_failed_main(world, b, record)
        world.fallback_selection_hook = hook
        for _ in range(3):
            world.fallback_selected(world.fallback_request)
        assert not observed and world.now == 10 and b._mark is None
        assert world.adapter is None and not world.qualifications and a.thread is b.thread is None
        record("plan", dict(routes=[asdict(row) for row in world.routes],
            profiles=[asdict(row) for row in world.profiles], endpoints=[asdict(row) for row in world.endpoints],
            genuine_native_history=False, trigger_inert=True))
    finally:
        a.close()
        b.close()


def test_current_fallback_response_recipe(record):
    payload = fallback_reply_bytes()
    assert payload == EXPECTED_REPLY and len(payload) <= 65536
    value = native._decode_response(payload, "invented_model_5")
    response = native._response(value)
    assert type(response) is native.NativeMainResponse and response.model == "invented_model_5"
    assert native._response_bytes(response, "invented_model_5") == payload
    with pytest.raises(NativeContentPolicyDenied) as caught:
        native._decode_response(payload, "invented_model_0")
    assert type(caught.value) is NativeContentPolicyDenied
    record("recipe", dict(payload=payload, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
        native_response_type=type(response).__name__, wrong_model_type=type(caught.value).__name__,
        wrong_model_reason=caught.value.args, genuine_native_history=False, new_qualification=False))


def test_genuine_current_fallback_success(tmp_path, monkeypatch, record, ordinary_effects, caller_boundary):
    started = time.monotonic()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("model: invented_model\nmemory:\n  enabled: false\n")
    assert Path(sys.prefix).resolve() == Path(
        "/home/mersinvald/.local/share/orca/runtime/accountant/hpwa-p10.47/venv").resolve()
    root = Path(__file__).resolve().parents[2]
    expected_hashes = {
        "agent.native_main_payload": "4305a1edb5227bb66acdf66e716a53aebb10b2e8b9f93d3512323083f5e586fe",
        "agent.content_policy": "8db1a9898a2e21407edf87bfe227e65091428daf8e671313f7069faf5f560836",
        "agent.transports.owned_loopback_http": "05ec8c7eede3dc872914ff2a3b011b91d5fa63960be2b7a4fadf76d5dc7e218f",
        "tests.agent.owned_loopback_receiver": "ccc33c07764571d3723e8bcca8b879e9c2dc6bc17c7bb77aae95fa12521ca63d",
        "tests.agent.test_native_loopback_fallback_denial": "02f943767adf01eda785da8e2a8eddf72405fb6e8e89e32132214f728e5ab8a1",
        "hermes_content_policy": "a39f10d0b93e1eb77d36170367cd5880bfedee57497b6eda562fd5582fcc7865",
    }
    loaded = {name: Path(sys.modules[name].__file__).resolve() for name in (*expected_hashes, __name__)}
    for name, path in loaded.items():
        assert path.is_relative_to(root) if name != "hermes_content_policy" else path == Path(sys.prefix) / "lib/python3.11/site-packages/hermes_content_policy/__init__.py"
        if name in expected_hashes:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hashes[name]
    record("source", dict(interpreter=sys.executable, prefix=sys.prefix,
        modules={name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                 for name, path in loaded.items()}))
    a, b = OwnedReceiver(observation_seconds=590.0), OwnedReceiver(observation_seconds=590.0)
    world, result, observed = None, None, {}
    receipts = {}
    try:
        world = build_world(a, b)
        assert_plan(world, a, b)
        hook, observed = observe_failed_main(world, b, record)
        world.fallback_selection_hook = hook
        originals = (world.routes, world.profiles, world.transports, world.selection, world.fallback_selection,
            tuple((port, port.resolve, port.send) for port in world.ports.values()))
        listeners = (a.listener, b.listener)
        a.start((ResponseStep(world.endpoints[0].request_path.encode(), response_bytes(ACK), QUALIFICATION),
                 ResponseStep(world.endpoints[0].request_path.encode(), b"", hold_seconds=6.0, allow_disconnect=True)))
        b.start((ResponseStep(world.endpoints[1].request_path.encode(), response_bytes(ACK), QUALIFICATION),
                 ResponseStep(world.endpoints[1].request_path.encode(), response_bytes(fallback_reply_bytes()))))
        record("phase", dict(name="receivers_ready", seconds=time.monotonic() - started))
        qualifier = qualify(world, record)
        a.wait(1)
        b.wait(1)
        record("qualification_complete", dict(charges=charges(qualifier), graph=graph(qualifier),
            envelopes=[asdict(row.envelope) for row in qualifier._envelopes.values()],
            http=2, in_process=14, rows_a=tuple(a.rows), rows_b=tuple(b.rows), seconds=time.monotonic() - started))
        assert charges(qualifier) == (32, 16, 4229)
        World.bind(world, qualify=False)
        construct(world, model="invented_model_0", max_tokens=None)
        bind_fallback(world)
        original_agent = dict(vars(world.agent))
        assert not observed and world.now == 10
        record("phase", dict(name="native_bound", seconds=time.monotonic() - started))
        result = fixed_turn(world)
        adapter, runtime = world.adapter, world.adapter._runtime
        record("released", dict(result=result, graph=graph(runtime), charges=charges(runtime),
            phase=adapter._fallback_phase, terminal=adapter._fallback_terminal,
            pending_turn=adapter._turn_pending is not None, operation=adapter._operation is not None,
            seconds=time.monotonic() - started))
        a.wait(2)
        b.wait(2)
        mark = b.mark_window()
        receipts["a"] = a.finish()
        record("a_final", asdict(receipts["a"]))
        receipts["b"] = b.finish()
        record("b_final", asdict(receipts["b"]))
        failed, = adapter.failed_main_attempts
        invocation, = adapter.fallback_invocations
        frame = adapter._fallback_frame
        assert type(invocation) is native.NativeFallbackInvocation
        assert failed is observed["failed"] and frame.token is observed["token"]
        assert frame.failed_main is failed and frame.binding is observed["binding"] is adapter._fallback_binding
        assert frame.parents is observed["parents"] and frame.parent_requests is observed["parent_requests"]
        assert runtime._failures[id(failed.shared_failure)] is observed["entry"]
        entry = observed["entry"]
        assert entry.token is failed.shared_failure and entry.facts is failed.failure_facts
        assert entry.request_handle is failed.request_handle is world.request
        assert entry.request is runtime._contents[id(failed.request_content)]
        assert entry.content is runtime._contents[id(failed.failure_content)]
        assert entry.issued is runtime._envelopes[id(failed.send_observation.envelope)]
        assert entry.issued.envelope is failed.send_observation.envelope
        assert charges(runtime)[:2] == (14, 2) and observed["charges"] == (5, 1, 1492)
        assert type(world.ports["main"]) is RetainedMainPort
        qualification, attempt = world.ports["main"].observations
        assert type(attempt) is MainSendObservation
        assert qualification.output == ACK and qualification.error is None
        assert type(attempt.error) is TimeoutError and attempt.output is None
        assert attempt.transport_receipt is attempt.error.owned_loopback_receipt
        diagnostic = attempt.transport_receipt
        assert diagnostic[:2] == ("owned_loopback_http_v1", "response")
        assert diagnostic[2] == diagnostic[3] == a.bound and diagnostic[4] == 1 and diagnostic[6] == b""
        assert attempt.envelope is failed.send_observation.envelope is failed.failure_facts.envelope
        assert failed.send_observation.exception_type is TimeoutError
        assert (failed.send_observation.send_count_before, failed.send_observation.send_count_after) == (0, 1)
        expected_request = dict(messages=[dict(content="Be concise.", role="system"),
            dict(content="Keep the blue folder.", role="user")], model="invented_model_0", stream=False)
        assert failed.request == expected_request and failed.request is caller_boundary.dispatches[-1][2]
        assert failed.turn_frame.request_bytes == canonical(expected_request)
        assert failed.request_bytes is failed.send_observation.operation.payload
        assert failed.request_bytes == attempt.envelope.payload == entry.request.payload
        assert json.loads(failed.request_bytes)["request"] == expected_request
        assert failed.source_bytes == failed.turn_frame.raw_bytes
        assert failed.send_observation.operation.content is failed.request_content
        facts = failed.failure_facts
        assert facts.request_content is failed.request_content and facts.failure_content is failed.failure_content
        assert facts.source is attempt.envelope.source and facts.profiles is attempt.envelope.profiles
        assert (facts.failure_kind, facts.phase, facts.send_count, facts.response_bytes) == ("timeout_error", "send", 1, None)
        assert entry.content.payload == canonical(dict(failure_kind="timeout_error", phase="send", send_count=1, response_bytes=None))
        assert entry.content.parents == (failed.request_content,)
        expected_parents = (failed.turn_frame.admitted, failed.turn_frame.input_content,
            failed.turn_frame.output_content, *failed.turn_frame.request_contents, failed.request_content, failed.failure_content)
        assert frame.parents == failed.parents == expected_parents
        assert frame.parent_requests == failed.parent_requests == tuple(world.request for _ in expected_parents)
        assert invocation.failed_main is failed and invocation.request_handle is world.fallback_request
        assert invocation.request == dict(expected_request, model="invented_model_5")
        assert invocation.request["messages"] is failed.request["messages"]
        assert invocation.request_bytes == canonical(invocation.request)
        assert invocation.selection is world.fallback_selection
        assert invocation.observed_destination == invocation.envelope.destination == world.fallback_selection.destination
        assert world.fallback_ports["fallback"].resolve() is world.fallback_selection.destination
        assert invocation.envelope.path == "fallback" and invocation.envelope.payload == invocation.input_stages[-1]
        assert json.loads(invocation.envelope.payload)["request"] == invocation.request
        assert invocation.response_bytes == fallback_reply_bytes() == EXPECTED_REPLY
        assert type(invocation.native_response) is native.NativeMainResponse
        assert invocation.native_response is adapter._dispatch_response
        assert invocation.native_response.model == "invented_model_5"
        assert native._response_bytes(invocation.native_response, "invented_model_5") == invocation.response_bytes
        assert frame.result is invocation.output is result
        assert frame.result_bytes == invocation.output_bytes == canonical(result)
        assert frame.result_content is invocation.output_content
        assert len(invocation.input_stages) == len(invocation.stage_contents) == 3
        assert len(invocation.output_stages) == len(invocation.output_contents) == 2
        assert runtime._contents[id(frame.source_content)].parents == frame.parents
        assert runtime._contents[id(frame.source_content)].payload == invocation.input_stages[0]
        for index, (payload, content) in enumerate(zip(invocation.input_stages, invocation.stage_contents, strict=True)):
            row = runtime._contents[id(content)]
            assert row.payload == payload and row.parents == (*frame.parents, frame.source_content, *invocation.stage_contents[:index])
        base = (frame.source_content, *frame.stage_contents)
        request_row = runtime._contents[id(invocation.request_content)]
        assert request_row.payload == invocation.envelope.payload and request_row.parents == (*frame.parents, *base)
        raw_result = runtime._contents[id(invocation.result_content)]
        assert raw_result.payload == invocation.response_bytes and raw_result.parents == (invocation.request_content,)
        base += (invocation.request_content, invocation.result_content)
        for index, (payload, content) in enumerate(zip(invocation.output_stages, invocation.output_contents, strict=True)):
            row = runtime._contents[id(content)]
            assert row.payload == payload and row.parents == (*frame.parents, *base, *invocation.output_contents[:index])
        assert runtime._contents[id(invocation.output_content)].parents == (*frame.parents, *base, *invocation.output_contents)
        assert runtime._contents[id(invocation.output_content)].payload == canonical(result)
        assert result == dict(completed=True, failed=False, interrupted=False, partial=False,
            final_response="The blue folder stays.", messages=[*expected_request["messages"],
                dict(role="assistant", content="The blue folder stays.")], fallback=True,
            selected_provider="synthetic_native", selected_model="invented_model_0", provider="synthetic_native",
            model="invented_model_5", base_url="", api_calls=2, main_attempt_usage=None,
            verified_response_usage=dict(prompt_tokens=3, completion_tokens=2, total_tokens=5),
            total_usage_known=False, estimated_cost_usd=None, cost_status="unknown")
        assert adapter._fallback_terminal is True and adapter._fallback_phase == "released"
        assert adapter._turn_pending is adapter._operation is None and not world.context._denied
        assert adapter.send_count == adapter.fallback_send_count == 1 and adapter.auxiliary_send_count == 0
        assert not adapter.invocations and not adapter._completions
        assert adapter._aux_binding is adapter._compression_binding is None
        assert all(vars(world.agent)[key] is value for key, value in original_agent.items())
        assert (world.routes, world.profiles, world.transports, world.selection, world.fallback_selection) == originals[:5]
        for port, resolve, send in originals[5]:
            assert port.resolve == resolve and port.send == send
            if type(port) is ControlledPort:
                assert not port.sent
        assert world.binding is world.initial_originals[2] and world.policy is world.initial_originals[4]
        assert world.fallback_binding is world.initial_originals[3] and world.fallback_policy is world.initial_originals[5]
        assert world.now == 10 and all(world.now < route.expires_at for route in world.routes)
        assert_plan(world, a, b)
        for current in (qualifier, runtime):
            rows = tuple(current._contents.values())
            assert sum(len(row.payload) for row in rows) == current._retained_bytes <= 8 * 1024 * 1024
            assert len(rows) + len(current._envelopes) + len(current._failures) <= 128
            assert all(len(row.payload) <= 1024 * 1024 and len(row.parents) <= 32 for row in rows)
            assert all(row.profiles == ("restricted_documents",) for row in rows)
            assert all(lineage.hops <= 128 for lineage in current._lineages.values())
        assert all(len(payload) <= 262144 for payload in (*invocation.input_stages, *invocation.output_stages,
            failed.request_bytes, invocation.request_bytes))
        assert len(invocation.response_bytes) <= 65536
        for name, receiver in (("a", a), ("b", b)):
            receipt = receipts[name]
            assert_receipt(receipt)
            receiver.check()
            assert len(receipt.connections) == 2
            assert receipt.listener_id == id(listeners[0 if name == "a" else 1])
            for row in receipt.connections:
                assert row.local == row.bound == receipt.bound and row.peer[0] == "127.0.0.1"
                assert row.complete and row.settled
            assert receipt.connections[0].body == QUALIFICATION
            assert receipt.connections[0].response == response_bytes(ACK)
        assert mark.row_count == 2 and mark.active_count == 0 and receipts["b"].mark is mark
        assert receipts["b"].connections[mark.row_count:] == ()
        assert b.listener is observed["listener"] and mark.listener_id == observed["listener_id"]
        assert mark.bound == observed["bound"] and observed["prefix"]["body"] == QUALIFICATION
        assert observed["prefix"]["closed_at"] <= receipts["b"].connections[1].accepted_at
        main, fallback = receipts["a"].connections[1], receipts["b"].connections[1]
        assert main.body == attempt.envelope.payload and main.raw_length == diagnostic[5] > len(main.body)
        assert main.response_accepted == 0 and main.response == b""
        assert main.status == "planned_disconnect" and main.failure == "eof_after_request"
        assert fallback.path == world.endpoints[1].request_path.encode()
        assert fallback.body == invocation.envelope.payload == request_row.payload
        assert fallback.body_sha256 == hashlib.sha256(request_row.payload).hexdigest()
        assert fallback.response == response_bytes(invocation.response_bytes)
        assert fallback.response_accepted == len(fallback.response)
        assert not world.sdk_calls
        adapter.validate_scope()
        world.agent.close()
        world.agent.close()
        assert adapter.failed_main_attempts == (failed,) and adapter.fallback_invocations == (invocation,)
        record("verified", dict(result=result, graph=graph(runtime), charges=charges(runtime),
            qualification_charges=charges(qualifier), released_phase=adapter._fallback_phase,
            original_main_error_id=id(attempt.error), original_main_error_type=type(attempt.error).__name__,
            main_client_receipt=diagnostic, main_envelope=asdict(attempt.envelope),
            fallback_envelope=asdict(invocation.envelope), native_response_id=id(invocation.native_response),
            result_id=id(result), combined_capacity=len(runtime._contents) + len(runtime._envelopes) + len(runtime._failures),
            lineage_hops=[lineage.hops for lineage in runtime._lineages.values()],
            agent_attributes_preserved=True, conversation_exit_verified=True, seconds=time.monotonic() - started))
        with pytest.raises(NativeContentPolicyDenied) as caught:
            fixed_turn(world)
        assert type(caught.value) is NativeContentPolicyDenied and world.context._denied
        assert adapter._turn_pending is None and adapter._operation is None
        assert adapter.failed_main_attempts == (failed,) and adapter.fallback_invocations == (invocation,)
        assert adapter._send_count == adapter._fallback_send_count == 1
        assert len(world.ports["main"].observations) == 2
        record("terminal_refusal", dict(error_type=type(caught.value).__name__, reason=caught.value.args,
            denied=world.context._denied, closed=adapter._constructed.closed, phase=adapter._fallback_phase,
            sends=adapter._send_count, fallback_sends=adapter._fallback_send_count,
            seconds=time.monotonic() - started))
    finally:
        if world is not None and world.adapter is not None:
            adapter = world.adapter
            record("retained_final", dict(graph=graph(adapter._runtime), charges=charges(adapter._runtime),
                failed_count=len(adapter._failed_main_attempts), fallback_count=len(adapter._fallback_records),
                phase=adapter._fallback_phase, result_present=result is not None,
                normalization="verified" if adapter._dispatch_response is not None else "UNKNOWN",
                release="verified" if result is not None else "UNKNOWN"))
            if type(world.ports["main"]) is RetainedMainPort:
                record("main_observations", [dict(envelope=asdict(row.envelope), output=row.output,
                    error_type=None if row.error is None else type(row.error).__name__, error_id=id(row.error),
                    transport_receipt=row.transport_receipt, started_at=row.started_at, finished_at=row.finished_at)
                    for row in world.ports["main"].observations])
        for name, receiver in (("a", a), ("b", b)):
            try:
                if name not in receipts and receiver.thread is not None:
                    receipts[name] = receiver.finish()
                    record(name + "_partial_final", asdict(receipts[name]))
            finally:
                receiver.close()
        record("cleanup", dict(a_closed=a.listener.fileno() == -1, b_closed=b.listener.fileno() == -1,
            a_stopped=a.thread is None or not a.thread.is_alive(), b_stopped=b.thread is None or not b.thread.is_alive(),
            a_errors=[type(error).__name__ for error in a.errors], b_errors=[type(error).__name__ for error in b.errors],
            seconds=time.monotonic() - started))
