"""Check native expiry after shared compression response retention."""

from dataclasses import asdict, replace
from types import SimpleNamespace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import pytest
from hermes_content_policy import PATHS

from agent import native_main_payload as native, context_compressor as compression
from agent.content_policy import NativeContentPolicyDenied, content_policy_scope
from tests.agent.owned_loopback_receiver import OwnedReceiver, ResponseStep, response_bytes
from tests.agent.test_native_controlled_conversation import (
    bind_compression, construct_compressor, summarize, construct, turn,
    compression_excluded as compression_excluded,
    excluded as excluded,
)
from tests.agent.test_native_loopback_fallback_denial import (
    ACK, QUALIFICATION, assert_receipt, charges, graph,
    ordinary_effects as ordinary_effects,
    record as record,
)
from tests.agent.test_native_main_payload import (
    ControlledPort, World, canonical, reply,
    caller_boundary as caller_boundary,
)

SUMMARY_TEXT = "<think>scratch</think>## Goal\nKeep the blue folder."
MAIN_TEXT = "The blue folder stays."
USER_TEXT = "Keep the blue folder."


class _PhaseReceipts:
    def __init__(self, path):
        self.path = path

    def pytest_runtest_logreport(self, report):
        if report.nodeid.split("::", 1)[0] != "tests/agent/test_native_loopback_retained_compression_result_expiry.py":
            return
        row = dict(nodeid=report.nodeid, when=report.when, outcome=report.outcome,
            duration=report.duration, recorded_at=time.time())
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, "a") as output:
            output.write(json.dumps(row) + "\n")
            output.flush()
            os.fsync(output.fileno())


@pytest.fixture(scope="session", autouse=True)
def phase_receipts(pytestconfig):
    path = os.environ.get("HERMES_NATIVE_PHASE_RECEIPTS")
    if not path and pytestconfig.option.xmlpath:
        path = Path(pytestconfig.option.xmlpath).with_suffix(".phases.jsonl")
    if path:
        plugin = _PhaseReceipts(path)
        pytestconfig.pluginmanager.register(plugin)
        pytestconfig.add_cleanup(lambda: pytestconfig.pluginmanager.unregister(plugin))


from tests.agent.test_native_loopback_retained_compression import (
    compression_reply_bytes,
)


from tests.agent.test_native_loopback_retained_compression_expiry import (
    configure_expiry_world, qualify_expiry_world,
)


def expire_after_registered_result(world, record):
    retained = {}
    expected = compression_reply_bytes()
    originals = (world.routes, world.profiles, world.transports, world.selection,
        world.compression_selection, world.admission.current, world.selection_port.current,
        world.title_selection_port.current, world.compression_selection_port.current, world.result_port.current)
    ports = tuple((port, port.resolve, port.send) for port in world.ports.values())
    authorities = (world.request, world.title_request, world.compression_request,
        world.binding, world.title_binding, world.compression_binding,
        world.policy, world.title_policy, world.compression_policy)
    clock = []

    def trigger():
        if retained:
            assert retained["spent"] and world.now == 100
            return
        adapter = world.adapter
        if adapter is None:
            return
        if not clock:
            clock.append(adapter._clock)
        assert adapter._clock is clock[0]
        if adapter._pending is None:
            return
        operation = adapter._operation
        envelope, actual, output = adapter._pending
        if operation is None or operation.path != "compression" or type(output) is not bytes or output != expected:
            return
        sealed = json.loads(operation.payload)
        if sealed.get("task") != "compression":
            return
        result = getattr(adapter, "_dispatch_result", None)
        runtime = adapter._runtime
        row = runtime._contents.get(id(result))
        if result is None or row is None:
            return
        if row.path == "main" and row.binding != tuple(asdict(world.compression_binding).values()):
            return
        assert row.handle is result and row.payload is output and row.path == "compression"
        assert row.parents == (operation.content,) and row.parents[0] is operation.content
        assert row.binding == tuple(asdict(world.compression_binding).values())
        assert row.profiles == world.compression_policy.profiles == world.compression_binding.profiles == ("restricted_documents",)
        request_row = runtime._contents[id(operation.content)]
        assert request_row.handle is operation.content and request_row.payload is operation.payload
        assert request_row.binding == row.binding and request_row.profiles == row.profiles and request_row.path == "compression"
        assert type(operation) is native._PendingOperation and adapter._checked_operation is operation
        assert operation.auxiliary and operation.handle is world.compression_request
        assert operation.source is world.compression_binding and operation.producer is world.compression_policy
        assert operation.selected is world.compression_selection and operation.route_ref == "compression_compression"
        frame = adapter._aux_pending
        assert type(frame) is native._CompressionFrame and frame.record is None
        assert frame.request_bytes is operation.payload and frame.sealed is not None
        assert frame.request is frame.preparation.value["request"]
        assert frame.completion is adapter._completions[-1]
        assert sealed["request"] == frame.request
        assert sealed["path"] == "compression" and sealed["selection"] == ["compression_selection", 6, "chat_completions"]
        assert sealed["destination"] == asdict(world.compression_selection.destination)
        assert envelope.payload is operation.payload and envelope.source == world.compression_binding
        assert envelope.path == "compression" and envelope.destination == actual == world.compression_selection.destination
        port = world.compression_ports["compression"]
        assert len(port.observations) == 2
        observation = port.observations[1]
        assert observation.envelope is envelope and observation.output is output
        assert observation.error is observation.transport_receipt is None
        assert runtime._busy is runtime._poisoned is False and not runtime._failures
        assert adapter._dispatch_response is None and adapter._busy and not adapter._invalid
        assert world.now == 10 and not world.context._denied
        current = (world.routes, world.profiles, world.transports, world.selection,
            world.compression_selection, world.admission.current, world.selection_port.current,
            world.title_selection_port.current, world.compression_selection_port.current, world.result_port.current)
        assert all(a is b if not callable(a) else a == b for a, b in zip(current, originals, strict=True))
        current_authorities = (world.request, world.title_request, world.compression_request,
            world.binding, world.title_binding, world.compression_binding,
            world.policy, world.title_policy, world.compression_policy)
        assert all(a is b for a, b in zip(current_authorities, authorities, strict=True))
        assert all(port.resolve == resolve and port.send == send for port, resolve, send in ports)
        assert world.admission_hook is trigger
        selected = next(route for route in world.routes if route.route_ref == "compression_compression")
        alternate = world.routes[-1]
        assert selected.expires_at == 100 and alternate.expires_at == 200
        assert all(route.expires_at == 200 for route in world.routes if route is not selected)
        before_graph = graph(runtime)
        retained.update(spent=True, operation=operation, frame=frame, envelope=envelope, output=output,
            result=result, row=row, request_row=request_row, observation=observation, actual=actual,
            clock=adapter._clock, current=world.admission.current, hook=trigger,
            selected=selected, alternate=alternate, graph=before_graph, originals=originals, authorities=authorities)
        record("trigger", dict(before=10, after=100, graph=before_graph, charges=charges(runtime),
            operation_id=id(operation), frame_id=id(frame), envelope_id=id(envelope), output_id=id(output),
            result_id=id(result), row_id=id(row), parent_ids=[id(parent) for parent in row.parents],
            observation_id=id(observation), request_id=id(frame.request), preparation_id=id(frame.preparation),
            sealed_id=id(frame.sealed), completion_id=id(frame.completion), clock_id=id(adapter._clock),
            current_function_id=id(world.admission.current.__func__), hook_id=id(trigger),
            selected=asdict(selected), alternate=asdict(alternate), request=operation.payload, response=output,
            shared_busy=runtime._busy, shared_poisoned=runtime._poisoned, shared_failures=len(runtime._failures),
            native_response=None, native_busy=adapter._busy, native_invalid=adapter._invalid, spent=True))
        world.now = 100
    return trigger, retained


def test_inert_post_result_compression_expiry_catalog():
    a, b = SimpleNamespace(bound=("127.0.0.1", 19001)), SimpleNamespace(bound=("127.0.0.1", 19002))
    world = configure_expiry_world(a, b)
    baseline, profiles, receivers = world.authentic_baseline
    assert len(baseline) == len(world.ports) == 24 and len(world.routes) == 25
    assert all(route.expires_at == 100 for route in baseline)
    assert len({id(world.request), id(world.title_request), id(world.compression_request)}) == 3
    assert world.now == 10 and world.adapter is None
    assert len({id(world.selection_port), id(world.title_selection_port), id(world.compression_selection_port)}) == 3
    assert tuple(world.receiver_bindings.items()) == receivers
    for old, route in zip(baseline, world.routes[:24], strict=True):
        assert replace(route, expires_at=old.expires_at) == old
    selected = next(route for route in world.routes if route.route_ref == "compression_compression")
    alternate = world.routes[-1]
    assert selected.expires_at == 100 and all(route.expires_at == 200 for route in world.routes if route is not selected)
    assert replace(alternate, route_ref=selected.route_ref, expires_at=100, destination=selected.destination) == selected
    assert alternate.destination is world.compression_ports["main"].destination
    assert alternate.destination != world.compression_selection.destination
    assert world.receiver_bindings[id(world.compression_ports["main"])][0] is world.compression_ports["main"]
    assert replace(world.profiles[0], allowed_routes=profiles[0].allowed_routes) == profiles[0]
    assert world.profiles[0].allowed_routes == (*profiles[0].allowed_routes, alternate.route_ref)
    assert world.profiles[0].required_paths == PATHS
    for binding, policy in ((world.binding, world.policy), (world.title_binding, world.title_policy),
                            (world.compression_binding, world.compression_policy)):
        rows = [route for route in world.routes if route.selection_id == binding.selection_id]
        assert {route.path for route in rows} == set(PATHS)
        assert all(route.selection_version == binding.selection_version and route.expires_at > 10 for route in rows)
        assert binding.profiles == policy.profiles == ("restricted_documents",)
    assert [(p.producer_ref, p.revision) for p in (world.policy, world.title_policy, world.compression_policy)] == [
        ("invented_producer", 4), ("title_producer", 2), ("compression_producer", 3)]


    originals = (*world.initial_originals[:9], world.profiles, *world.initial_originals[10:])
    current = (world.request, world.title_request, world.compression_request,
        world.binding, world.title_binding, world.compression_binding, world.policy, world.title_policy,
        world.compression_policy, world.profiles, world.admission.current, world.selection_port.current,
        world.title_selection_port.current, world.compression_selection_port.current, world.result_port.current)
    assert all(a is b if not callable(a) else a == b for a, b in zip(current, originals, strict=True))
    for port in world.ports.values():
        assert not (port.sent if type(port) is ControlledPort else port.observations)
    assert not world.qualifications


def test_post_result_expiry_gate_waits_for_registered_response():
    world = configure_expiry_world(SimpleNamespace(bound=("127.0.0.1", 19001)),
                                   SimpleNamespace(bound=("127.0.0.1", 19002)))
    records = []
    hook, retained = expire_after_registered_result(world, lambda name, row: records.append((name, row)))
    world.admission_hook = hook
    hook()
    original_current = world.admission.current
    clock = lambda: world.now
    runtime = SimpleNamespace(_contents={}, _envelopes={}, _busy=False, _poisoned=False, _failures={}, _retained_bytes=0)
    adapter = world.adapter = SimpleNamespace(_pending=None, _operation=None, _clock=clock, _runtime=runtime,
        _dispatch_result=None, _dispatch_response=None, _busy=True, _invalid=False)
    hook()
    for path in ("main", "compression"):
        adapter._operation = SimpleNamespace(path=path, payload=canonical(dict(task="qualification")))
        adapter._pending = (None, None, compression_reply_bytes())
        hook()
    destination = world.compression_selection.destination
    request = dict(model="invented_model_5", messages=[])
    payload = canonical(dict(task="compression", path="compression", request=request,
        selection=["compression_selection", 6, "chat_completions"], destination=asdict(destination)))
    content, result, stale = object(), object(), object()
    operation = native._PendingOperation(world.compression_request, world.compression_binding,
        world.compression_selection, world.compression_policy, "compression", "compression_compression", payload, content, True)
    adapter._operation = adapter._checked_operation = operation
    for output in (None, b"", compression_reply_bytes()[:-1]):
        adapter._pending = (None, destination, output)
        hook()
    output = compression_reply_bytes()
    envelope = SimpleNamespace(payload=payload, source=world.compression_binding, path="compression", destination=destination)
    adapter._pending = (envelope, destination, output)
    hook()
    adapter._dispatch_result = stale
    runtime._contents[id(stale)] = SimpleNamespace(handle=stale, path="main", binding=tuple(asdict(world.binding).values()))
    hook()
    runtime._contents.clear()
    adapter._dispatch_result = result
    hook()
    assert world.now == 10 and not retained and not records
    binding = tuple(asdict(world.compression_binding).values())
    row = SimpleNamespace(handle=result, payload=output, parents=(content,), binding=binding,
        profiles=("restricted_documents",), path="compression")
    request_row = SimpleNamespace(handle=content, payload=payload, binding=binding,
        profiles=row.profiles, path="compression", parents=())
    runtime._contents[id(content)] = request_row
    runtime._contents[id(result)] = row
    for field, wrong in (("parents", (object(),)), ("handle", object()), ("payload", bytes(bytearray(output))),
                         ("binding", tuple(asdict(world.binding).values())), ("path", "title")):
        original = getattr(row, field)
        setattr(row, field, wrong)
        with pytest.raises(AssertionError):
            hook()
        assert world.now == 10 and not retained and not records
        assert adapter._clock is clock and world.admission.current == original_current
        setattr(row, field, original)
    completion = object()
    adapter._completions = (completion,)
    adapter._aux_pending = native._CompressionFrame(object(), completion, [], b"", (), (), object(),
        request=request, request_bytes=payload, sealed=object(), preparation=SimpleNamespace(value=dict(request=request)))
    observation = SimpleNamespace(envelope=envelope, output=output, error=None, transport_receipt=None)
    port = world.compression_ports["compression"]
    object.__setattr__(port, "observations", (SimpleNamespace(), observation))
    runtime._retained_bytes = len(payload) + len(output)
    hook()
    assert world.now == 100 and len(records) == 1 and retained["result"] is result
    assert retained["row"] is row and retained["operation"] is operation and retained["spent"]
    before = dict(retained)
    hook()
    assert world.now == 100 and retained == before and len(records) == 1
    assert adapter._clock is clock and world.admission.current == original_current


def test_genuine_compression_expiry_after_shared_result(ordinary_effects, caller_boundary, record):
    started = time.monotonic()
    modules = (sys.modules[__name__], native, compression, sys.modules["agent.auxiliary_client"], sys.modules["agent.transports.owned_loopback_http"],
        sys.modules["hermes_content_policy"], sys.modules["tests.agent.test_native_controlled_conversation"])
    assert Path(sys.prefix).resolve() == Path(
        "/home/mersinvald/.local/share/orca/runtime/accountant/hpwa-p10.47/venv").resolve()
    root = Path(__file__).resolve().parents[2]
    for module in modules:
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(root) if module.__name__ != "hermes_content_policy" else path == Path(sys.prefix) / "lib/python3.11/site-packages/hermes_content_policy/__init__.py"
    record("source", dict(interpreter=sys.executable, prefix=sys.prefix,
        modules={module.__name__: dict(path=str(Path(module.__file__).resolve()),
            sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()) for module in modules}))
    a, b = OwnedReceiver(observation_seconds=140.0), OwnedReceiver(observation_seconds=140.0)
    world, result, summary, qualifier = None, None, None, None
    receipts = {}
    verified = False
    try:
        world = configure_expiry_world(a, b)
        hook, trigger = expire_after_registered_result(world, record)
        world.admission_hook = hook
        record("authentic_baseline", dict(routes=[asdict(row) for row in world.authentic_baseline[0]],
            profiles=[asdict(row) for row in world.authentic_baseline[1]]))
        record("authorized_delta", dict(expiries={r.route_ref:r.expires_at for r in world.routes},
            alternate=asdict(world.routes[-1]), profiles=[asdict(row) for row in world.profiles]))
        authority_originals = (*world.initial_originals[:9], world.profiles, *world.initial_originals[10:])
        originals = (world.routes, world.profiles, world.transports, world.selection, world.compression_selection,
            tuple((port, port.resolve, port.send) for port in world.ports.values()))
        listeners = (a.listener, b.listener)
        main_value = json.loads(reply(MAIN_TEXT))
        main_value["model"] = "invented_model_0"
        main_bytes, summary_bytes = canonical(main_value), compression_reply_bytes()
        record("operator", dict(routes=[asdict(route) for route in world.routes],
            profiles=[asdict(profile) for profile in world.profiles],
            main_binding=asdict(world.binding), compression_binding=asdict(world.compression_binding),
            main_policy=asdict(world.policy), compression_policy=asdict(world.compression_policy),
            main_selection=asdict(world.selection), compression_selection=asdict(world.compression_selection),
            endpoints=[asdict(endpoint) for endpoint in world.endpoints], now=world.now))
        for receiver, endpoint, payload in zip((a, b), world.endpoints, (main_bytes, summary_bytes), strict=True):
            receiver.start((ResponseStep(endpoint.request_path.encode(), response_bytes(ACK), QUALIFICATION),
                ResponseStep(endpoint.request_path.encode(), response_bytes(payload))))
        qualifier = qualify_expiry_world(world, record)
        a.wait(1)
        b.wait(1)
        record("qualification_complete", dict(graph=graph(qualifier), charges=charges(qualifier),
            envelopes=[asdict(row.envelope) for row in qualifier._envelopes.values()],
            hops=[lineage.hops for lineage in qualifier._lineages.values()], seconds=time.monotonic()-started))
        assert len(a.rows) == len(b.rows) == 1 and a.rows[0]["settled"] and b.rows[0]["settled"]
        assert a._mark is b._mark is None
        World.bind(world, qualify=False)
        construct(world, model="invented_model_0", max_tokens=None)
        from run_agent import AIAgent
        assert type(world.agent) is AIAgent
        assert world.agent.model == "invented_model_0" and world.agent.max_tokens is None
        assert world.agent.client is world.agent._primary_runtime is None
        assert "temperature" not in vars(world.agent)
        original_agent = dict(vars(world.agent))
        original_clock = adapter_clock = world.adapter._clock
        original_current = world.admission.current
        result = turn(world, USER_TEXT, system_message="Fixed system")
        adapter, runtime = world.adapter, world.adapter._runtime
        invocation, = adapter.invocations
        frame, = adapter._turn_frames
        completion, = adapter._completions
        assert completion.result is result and completion.invocation is invocation and completion.frame is frame
        assert result["final_response"] == MAIN_TEXT and result["final_response"] is invocation.native_response.choices[0].message.content
        assert result["completed"] and not result["failed"] and not result["interrupted"]
        assert completion.payload == canonical(result)
        assert type(invocation.native_response) is native.NativeMainResponse
        assert invocation.response_bytes == main_bytes and invocation.native_response.model == "invented_model_0"
        assert native._response_bytes(invocation.native_response, "invented_model_0") == main_bytes
        assert frame.arguments[0] is USER_TEXT and frame.arguments[2] is None
        main_parents = (frame.admitted, frame.input_content, frame.output_content, *frame.request_contents,
            invocation.request_content, invocation.result_content)
        assert frame.input_content is frame.admitted
        main_contents = (frame.admitted, frame.output_content, *frame.request_contents,
            invocation.request_content, invocation.result_content, completion.content)
        assert len({id(handle) for handle in main_contents}) == len(main_contents)
        assert {id(row.handle) for row in runtime._contents.values()} == {id(handle) for handle in main_contents}
        assert all(x is y for x, y in zip(runtime._contents[id(completion.content)].parents, main_parents, strict=True))
        assert adapter._turn_pending is adapter._operation is None and not world.context._denied
        assert len(main_contents) == 6
        record("main_released", dict(result=result, graph=graph(runtime), charges=charges(runtime),
            envelope=asdict(invocation.envelope), response=invocation.response_bytes,
            completion_id=id(completion), result_id=id(result), frame_id=id(frame), invocation_id=id(invocation),
            native_response_id=id(invocation.native_response), seconds=time.monotonic()-started))
        main_retained_bytes = runtime._retained_bytes
        expected_content_tokens = compression.estimate_messages_tokens_rough(result["messages"])
        original_result = canonical(result)
        prefix = world.agent._cached_system_prompt
        messages = result["messages"]
        message_rows = tuple(messages)
        with pytest.raises(NativeContentPolicyDenied) as refusal:
            with content_policy_scope(world.context):
                bind_compression(world)
                compressor = construct_compressor(world)
                compressor_state = dict(vars(compressor))
                summary = summarize(world, compressor, result)
        error = refusal.value
        chain = [error]
        assert type(error) is NativeContentPolicyDenied and error.args == ("controlled_native_payload_invalid",)
        assert error.__cause__ is error.__context__ is None and error.__suppress_context__ is False
        traceback = error.__traceback__
        locations = []
        while traceback is not None:
            locations.append(dict(file=traceback.tb_frame.f_code.co_filename,
                function=traceback.tb_frame.f_code.co_name, line=traceback.tb_lineno))
            traceback = traceback.tb_next
        expiry_line = 3140
        assert any(row["function"] == "_compression_snapshot" and row["line"] == expiry_line for row in locations)
        record("exception", dict(chain=[dict(type=type(error).__name__, args=error.args, identity=id(error),
            context_id=None if error.__context__ is None else id(error.__context__),
            cause=None, suppressed=error.__suppress_context__) for error in chain], locations=locations,
            predicate="now < route.expires_at", line=expiry_line, shared_selected_grant_executed=True))
        assert summary is None and runtime._poisoned is False and world.context._denied is True
        assert trigger and trigger["clock"] is original_clock is adapter_clock
        assert trigger["current"] == original_current == world.admission.current
        assert trigger["hook"] is world.admission_hook is hook
        assert adapter._pending is adapter._aux_pending is adapter._operation is adapter._checked_operation is adapter._turn_pending is None
        auxiliary, = adapter.auxiliary_invocations
        aux_frame, = (entry[0] for entry in adapter._aux_frames)
        record("compression_partial", dict(graph=graph(runtime), charges=charges(runtime),
            envelope=asdict(auxiliary.envelope), response=auxiliary.response_bytes,
            input_stages=auxiliary.input_stages, output_stages=auxiliary.output_stages,
            source_id=id(auxiliary.source_completion), frame_id=id(aux_frame), record_id=id(auxiliary),
            response_id=id(auxiliary.response_bytes), summary=None))
        assert auxiliary.source_completion is completion and auxiliary.source_completion.result is result
        assert auxiliary.source_frame is frame and auxiliary.source_invocation is invocation
        assert auxiliary.source_handle is world.request and auxiliary.request_handle is world.compression_request
        assert aux_frame.messages is messages is result["messages"]
        assert all(x is y for x, y in zip(messages, message_rows, strict=True))
        assert aux_frame.completion is completion and aux_frame.record is auxiliary
        expected_parents = (*main_parents, completion.content)
        assert all(x is y for x, y in zip(auxiliary.parents, expected_parents, strict=True))
        assert all(handle is world.request for handle in auxiliary.parent_requests)
        assert len(auxiliary.parent_requests) == len(expected_parents)
        assert auxiliary.source_bytes == canonical(messages)
        bridge = runtime._contents[id(aux_frame.source_content)]
        assert bridge.payload == auxiliary.source_bytes
        assert all(x is y for x, y in zip(bridge.parents, auxiliary.parents, strict=True))
        assert auxiliary.selection is world.compression_selection and auxiliary.producer_policy is world.compression_policy
        assert aux_frame.parents is auxiliary.parents and aux_frame.parent_requests is auxiliary.parent_requests
        assert auxiliary.path == auxiliary.task == "compression"
        assert (auxiliary.target.selection_id, auxiliary.target.selection_version) == ("compression_selection", 6)
        assert auxiliary.envelope.destination == auxiliary.observed_destination == world.compression_selection.destination
        assert auxiliary.envelope.path == "compression" and auxiliary.envelope.source == world.compression_binding
        assert auxiliary.request_bytes == auxiliary.envelope.payload
        sealed = json.loads(auxiliary.request_bytes)
        assert sealed["path"] == sealed["task"] == "compression"
        assert sealed["selection"] == ["compression_selection", 6, "chat_completions"]
        assert sealed["destination"] == asdict(world.compression_selection.destination)
        assert len(auxiliary.input_stages) == len(auxiliary.stage_contents) == 7
        facts = [json.loads(stage) for stage in auxiliary.input_stages]
        assert facts[0]["constructor"] == world.adapter._compression_options
        assert facts[0]["previous_summary"] is facts[0]["focus_topic"] is None
        assert facts[0]["source_field"] == "messages" and facts[0]["memory_context"] == ""
        assert facts[1] == dict(content_tokens=expected_content_tokens,
            ratio=0.2, context_length=65536, maximum=3276, summary_budget=2000)
        assert facts[2]["pruned_skill_names"] == [] and USER_TEXT in facts[2]["serialized"]
        assert facts[3] == dict(sampled=facts[2]["serialized"], lean=True, sampling=False,
            original_length=len(facts[2]["serialized"]), slice_count=8, aggregate_cap=160000)
        assert facts[4] == dict(sanitized_memory="", serialized_memory='""', memory_section="",
                               has_user_turn=True, summary_date="2026-11-23")
        assert "current date is 2026-11-23" in facts[5]["prompt"]
        assert "Keep the blue folder." in facts[5]["prompt"]
        assert compression._LEAN_SESSION_LOG_HEADING in facts[5]["prompt"]
        wire = dict(model="invented_model_5", messages=[dict(role="user", content=facts[5]["prompt"])], stream=False)
        request = json.loads(auxiliary.request_bytes)
        assert request["request"] == facts[6]["request"] == wire
        assert facts[6]["arguments"] == aux_frame.preparation.value["arguments"]
        assert aux_frame.request is aux_frame.preparation.value["request"]
        assert aux_frame.request["messages"] is aux_frame.preparation.value["arguments"]["messages"]
        assert aux_frame.request["messages"][0]["content"] == facts[5]["prompt"]
        assert aux_frame.preparation.value["arguments"]["temperature"] is aux_frame.preparation.value["arguments"]["max_tokens"] is None
        assert "temperature" not in wire and "max_tokens" not in wire
        assert auxiliary.result_content is trigger["result"] and auxiliary.native_response is None
        assert runtime._contents[id(auxiliary.result_content)] is trigger["row"]
        assert trigger["row"].payload is auxiliary.response_bytes
        assert trigger["row"].parents == (auxiliary.request_content,)
        assert trigger["row"].parents[0] is auxiliary.request_content
        assert trigger["operation"].content is auxiliary.request_content
        assert trigger["frame"] is not aux_frame
        assert replace(trigger["frame"], record=auxiliary) == aux_frame
        assert aux_frame.preparation is trigger["frame"].preparation
        assert aux_frame.request is trigger["frame"].request and aux_frame.sealed is trigger["frame"].sealed
        assert runtime._busy is runtime._poisoned is False and not runtime._failures
        assert adapter._invalid is True and adapter._busy is False
        with pytest.raises(NativeContentPolicyDenied) as scope_refusal:
            with content_policy_scope(world.context):
                pytest.fail("Denied scope entered")
        assert scope_refusal.value is not error and scope_refusal.value.args == ("controlled_native_payload_denied",)
        record("scope_refusal", dict(error_id=id(scope_refusal.value), reason=scope_refusal.value.args,
            original_id=id(error), distinct=True, shared_busy=runtime._busy, shared_poisoned=runtime._poisoned,
            native_busy=adapter._busy, native_invalid=adapter._invalid, context_denied=world.context._denied))
        assert auxiliary.output_stages == auxiliary.output_contents == ()
        assert auxiliary.output is auxiliary.output_bytes is auxiliary.output_content is None
        assert auxiliary.envelope is trigger["envelope"] and auxiliary.response_bytes is trigger["output"]
        assert auxiliary.request_bytes is trigger["operation"].payload
        assert auxiliary.response_bytes == summary_bytes
        for index, (payload, content) in enumerate(zip(auxiliary.input_stages, auxiliary.stage_contents, strict=True)):
            row = runtime._contents[id(content)]
            assert row.payload == payload
            assert all(x is y for x, y in zip(row.parents, (aux_frame.source_content, *auxiliary.stage_contents[:index]), strict=True))
        row = runtime._contents[id(auxiliary.request_content)]
        assert row.payload == auxiliary.request_bytes
        assert all(x is y for x, y in zip(row.parents, (aux_frame.source_content, *auxiliary.stage_contents), strict=True))
        assert vars(compressor) == compressor_state
        assert compressor._previous_summary is None and compressor.compression_count == 0
        assert canonical(result) == original_result and world.agent._cached_system_prompt is prefix
        assert adapter.auxiliary_local_results == ()
        assert adapter._aux_binding is adapter._fallback_binding is None
        assert not runtime._failures
        current_authorities = (world.request, world.title_request, world.compression_request,
            world.binding, world.title_binding, world.compression_binding, world.policy, world.title_policy,
            world.compression_policy, world.profiles, world.admission.current, world.selection_port.current,
            world.title_selection_port.current, world.compression_selection_port.current, world.result_port.current)
        assert all(x is y if not callable(x) else x == y for x, y in zip(current_authorities, authority_originals, strict=True))
        assert len(frame.request_contents) == 1
        assert len(wire["messages"]) <= 32
        assert all(len(row["content"].encode("utf-8")) <= 16384 for row in wire["messages"])
        assert all(len(payload) <= 262144 for payload in (auxiliary.source_bytes,))
        assert adapter.send_count == adapter.auxiliary_send_count == 1 and adapter.fallback_send_count == 0
        assert not adapter.failed_main_attempts and not adapter.fallback_invocations
        assert len(caller_boundary.dispatches) == 1 and not world.sdk_calls
        assert all(vars(world.agent)[key] is value for key, value in original_agent.items())
        assert all(x is y for x, y in zip(
            (world.routes, world.profiles, world.transports, world.selection, world.compression_selection),
            world.configured_baseline, strict=True))
        assert world.now == 100
        assert [route.route_ref for route in world.routes if route.expires_at <= world.now] == ["compression_compression"]
        assert world.routes[-1].selection_id == "compression_selection" and world.routes[-1].selection_version == 6
        assert world.routes[-1].expires_at > world.now and world.routes[-1].path == "compression"
        for port, resolve, send in originals[5]:
            assert port.resolve == resolve and port.send == send
            if type(port) is ControlledPort:
                assert not port.sent
        for current in (qualifier, runtime):
            rows = tuple(current._contents.values())
            assert sum(len(row.payload) for row in rows) == current._retained_bytes <= 8 * 1024 * 1024
            assert len(rows) + len(current._envelopes) + len(current._failures) <= 128
            assert all(len(row.payload) <= 1024 * 1024 and len(row.parents) <= 32 for row in rows)
            assert all(row.profiles == ("restricted_documents",) for row in rows)
            assert all(lineage.hops <= 128 for lineage in current._lineages.values())
        assert adapter._aux_pending is adapter._operation is adapter._turn_pending is None
        assert adapter._clock is original_clock
        assert len(runtime._envelopes) == 2
        assert len(runtime._contents) == len(main_contents) + 1 + len(auxiliary.stage_contents) + 1 + 1 == 16
        assert all(len(payload) <= 262144 for payload in (*auxiliary.input_stages, auxiliary.request_bytes, invocation.request_bytes))
        assert runtime._retained_bytes == sum(len(row.payload) for row in runtime._contents.values())
        assert runtime._retained_bytes == main_retained_bytes + len(auxiliary.source_bytes) + sum(map(len, auxiliary.input_stages)) + len(auxiliary.request_bytes) + len(auxiliary.response_bytes)
        assert charges(qualifier) == (50, 25, 7241)
        assert charges(runtime) == (16, 2, 30099)
        assert len(auxiliary.response_bytes) == len(summary_bytes) == 312
        assert len(auxiliary.response_bytes) <= 65536 and len(invocation.response_bytes) <= 65536
        a.wait(2)
        b.wait(2)
        for receiver in (a, b):
            mark = receiver.mark_window()
            assert mark.row_count == 2 and mark.active_count == 0
        for name, receiver, listener, operation, payload in zip(("a", "b"), (a, b), listeners,
                (invocation, auxiliary), (main_bytes, summary_bytes), strict=True):
            receipt = receipts[name] = receiver.finish()
            record(name + "_final", asdict(receipt))
            assert_receipt(receipt)
            receiver.check()
            assert len(receipt.connections) == 2 and receipt.connections[receipt.mark.row_count:] == ()
            assert receipt.listener_id == id(listener) and receiver.listener is listener
            port = world.ports["main"] if name == "a" else world.compression_ports["compression"]
            assert len(port.observations) == 2
            for index, connection in enumerate(receipt.connections):
                observation = port.observations[index]
                assert connection.complete and connection.settled
                assert connection.header.endswith(b"\r\n\r\n")
                assert connection.raw == connection.header + connection.body
                assert connection.raw_length == len(connection.raw)
                assert connection.raw_sha256 == hashlib.sha256(connection.raw).hexdigest()
                assert connection.body_length == len(connection.body)
                assert connection.client_reported_peer is not None and connection.host is not None
                assert connection.peer != connection.local
                assert connection.local == connection.bound == receipt.bound and connection.peer[0] == "127.0.0.1"
                assert connection.body == observation.envelope.payload
                assert connection.body_sha256 == hashlib.sha256(observation.envelope.payload).hexdigest()
                header, separator, body = connection.response.partition(b"\r\n\r\n")
                expected = ACK if index == 0 else payload
                assert separator == b"\r\n\r\n" and body == expected
                assert b"Content-Length: " + str(len(body)).encode() in header.split(b"\r\n")
                assert observation.output == body and observation.error is observation.transport_receipt is None
                assert connection.response_accepted == len(connection.response)
                assert connection.response == response_bytes(body)
            data = receipt.connections[1]
            assert port.observations[1].envelope is operation.envelope
            assert port.observations[1].output is operation.response_bytes
            assert data.body == operation.request_bytes
            assert data.response.partition(b"\r\n\r\n")[2] == operation.response_bytes
        assert receipts["a"].bound != receipts["b"].bound
        assert all(receipt.cutoff_at - receipt.mark.started_at >= 0.5 for receipt in receipts.values())
        record("verified", dict(result=result, summary=None, output=None, graph=graph(runtime),
            charges=charges(runtime), qualification_charges=charges(qualifier),
            envelopes=[asdict(row.envelope) for row in runtime._envelopes.values()],
            hops=[lineage.hops for lineage in runtime._lineages.values()],
            main_usage=dict(prompt_tokens=result["prompt_tokens"], completion_tokens=result["completion_tokens"], total_tokens=result["total_tokens"]),
            compression_usage=json.loads(auxiliary.response_bytes)["usage"], estimated_cost_usd=result["estimated_cost_usd"],
            cost_status=result["cost_status"], conversation_exit=True, auxiliary_scope_refused=True,
            agent_attributes_preserved=True, seconds=time.monotonic()-started))
        assert time.monotonic()-started < 145
        world.agent.close()
        world.agent.close()
        assert adapter.invocations == (invocation,) and adapter.auxiliary_invocations == (auxiliary,)
        with pytest.raises(NativeContentPolicyDenied) as caught:
            turn(world, USER_TEXT, system_message="Fixed system")
        assert type(caught.value) is NativeContentPolicyDenied and world.context._denied
        assert caught.value is not error and caught.value is not scope_refusal.value
        assert caught.value.args == ("controlled_native_payload_denied",)
        assert adapter._turn_pending is adapter._operation is None and adapter._constructed.closed
        assert adapter._send_count == adapter._aux_send_count == 1 and adapter._fallback_send_count == 0
        assert len(world.ports["main"].observations) == len(world.compression_ports["compression"].observations) == 2
        verified = True
        record("terminal_refusal", dict(error_type=type(caught.value).__name__, reason=caught.value.args,
            error_id=id(caught.value), closed=adapter._constructed.closed, sends=adapter._send_count,
            compression_sends=adapter._aux_send_count, seconds=time.monotonic()-started))
    finally:
        try:
            if world is not None:
                for runtime_name, current in (("qualification", qualifier), ("native", None if world.adapter is None else world.adapter._runtime)):
                    if current is not None:
                        record(runtime_name + "_retained_final", dict(graph=graph(current), charges=charges(current),
                            envelopes=[asdict(row.envelope) for row in current._envelopes.values()],
                            hops=[lineage.hops for lineage in current._lineages.values()], verified=verified,
                            main_release=result is not None, compression_release=summary is not None))
                for name, port in (("main", world.ports["main"]), ("compression", world.compression_ports["compression"])):
                    record(name + "_observations", [dict(envelope=asdict(row.envelope), output=row.output,
                        error_type=None if row.error is None else type(row.error).__name__, error_id=id(row.error),
                        transport_receipt=row.transport_receipt, started_at=row.started_at, finished_at=row.finished_at)
                        for row in port.observations])
        finally:
            cleanup_errors = []
            for name, receiver in (("a", a), ("b", b)):
                try:
                    if name not in receipts and receiver.thread is not None:
                        receipts[name] = receiver.finish()
                        record(name + "_partial_final", asdict(receipts[name]))
                except BaseException as error:
                    cleanup_errors.append(error)
                finally:
                    try:
                        receiver.close()
                    except BaseException as error:
                        cleanup_errors.append(error)
            record("cleanup", dict(a_closed=a.listener.fileno() == -1, b_closed=b.listener.fileno() == -1,
                a_stopped=a.thread is None or not a.thread.is_alive(), b_stopped=b.thread is None or not b.thread.is_alive(),
                a_sockets_closed=not a._connections, b_sockets_closed=not b._connections,
                a_errors=[type(error).__name__ for error in a.errors], b_errors=[type(error).__name__ for error in b.errors],
                cleanup_errors=[type(error).__name__ for error in cleanup_errors], seconds=time.monotonic()-started))
            assert not cleanup_errors
            assert a.listener.fileno() == b.listener.fileno() == -1
            assert not a._connections and not b._connections
            assert a.thread is None or not a.thread.is_alive()
            assert b.thread is None or not b.thread.is_alive()
