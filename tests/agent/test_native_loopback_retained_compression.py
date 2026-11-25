"""Check one retained first summary through two owned HTTP listeners."""

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import pytest
from hermes_content_policy import ContentPolicyRuntime, PATHS

from agent import native_main_payload as native, context_compressor as compression
from agent.content_policy import NativeContentPolicyDenied, content_policy_scope
from agent.transports.owned_loopback_http import OwnedLoopbackEndpoint, OwnedLoopbackHTTPTransport
from tests.agent.owned_loopback_receiver import OwnedReceiver, ResponseStep, response_bytes
from tests.agent.test_native_controlled_conversation import (
    CompressionWorld, bind_compression, construct_compressor, summarize, construct, turn,
    compression_excluded as compression_excluded,
    excluded as excluded,
)
from tests.agent.test_native_loopback_fallback_denial import (
    ACK, QUALIFICATION, RetainedMainPort, assert_receipt, charges, graph,
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
        if report.nodeid.split("::", 1)[0] != "tests/agent/test_native_loopback_retained_compression.py":
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


def build_compression_world(a, b):
    world = CompressionWorld()
    world.initial_originals = (world.request, world.title_request, world.compression_request,
        world.binding, world.title_binding, world.compression_binding, world.policy, world.title_policy,
        world.compression_policy, world.profiles, world.admission.current, world.selection_port.current,
        world.title_selection_port.current, world.compression_selection_port.current, world.result_port.current)
    world.original_routes = world.routes
    world.original_ports = tuple(world.ports.items())
    replacements, endpoints = {}, []
    for key, role, model, receiver, route_ref in (
        ("main", "p", "invented_model_0", a, "route_main"),
        ("compression_capability_compression", "c", "invented_model_5", b, "compression_compression"),
    ):
        old = world.ports[key]
        destination = replace(old.destination, endpoint_ref="synthetic_loopback_" + role, model_ref=model)
        endpoint = OwnedLoopbackEndpoint(destination.endpoint_ref, receiver.bound[0], receiver.bound[1],
            "/owned-accountant-proof/v1/" + role, destination.provider_ref, destination.model_ref, destination.revision)
        port = RetainedMainPort(OwnedLoopbackHTTPTransport(destination, endpoint=endpoint))
        world.ports[key] = port
        original, admission = world.receiver_bindings.pop(id(old))
        assert original is old
        world.receiver_bindings[id(port)] = (port, admission)
        replacements[route_ref] = destination
        if role == "p":
            world.selection = replace(world.selection, destination=destination)
        else:
            world.compression_ports["compression"] = port
            world.compression_selection = replace(world.compression_selection, destination=destination)
        endpoints.append(endpoint)
    world.routes = tuple(replace(route, destination=replacements[route.route_ref],
        qualification_ref="synthetic_loopback_" + ("main" if route.route_ref == "route_main" else "compression"))
        if route.route_ref in replacements else route for route in world.routes)
    world.transports = tuple((port.destination.transport_ref, port) for port in world.ports.values())
    world.endpoints = tuple(endpoints)
    world.qualifications = []
    return world


def qualify_compression_world(world, record):
    runtime = ContentPolicyRuntime(admission=world.admission, profiles=world.profiles,
        routes=world.routes, transports=world.transports, clock=lambda: world.now, policy_revision=9)
    world.qualifier = runtime
    authorities = {(binding.selection_id, binding.selection_version): (request, binding) for request, binding in (
        (world.request, world.binding), (world.title_request, world.title_binding),
        (world.compression_request, world.compression_binding))}
    for route in world.routes:
        request, binding = authorities[(route.selection_id, route.selection_version)]
        assert world.admission.current(request) is binding
        source = runtime.admit(request, QUALIFICATION)
        result = runtime.dispatch(request, source, path=route.path, route_ref=route.route_ref)
        envelope = tuple(runtime._envelopes.values())[-1].envelope
        assert envelope.source == binding and envelope.payload == QUALIFICATION
        assert envelope.destination == route.destination and envelope.path == route.path
        assert runtime._contents[id(result)].parents == (source,)
        world.qualifications.append((request, source, result, envelope))
        record("qualification", dict(route=asdict(route), envelope=asdict(envelope),
            source=asdict(binding), result=runtime._contents[id(result)].payload, charges=charges(runtime)))
    for port in world.ports.values():
        if type(port) is ControlledPort:
            assert len(port.sent) == 1
            assert any(port.sent[0] is entry[3] for entry in world.qualifications)
            port.sent.clear()
    assert len(world.qualifications) == len(runtime._envelopes) == 3 * len(PATHS) == 24
    assert len(runtime._contents) == 48
    assert len({id(row.handle) for row in runtime._contents.values()}) == 48
    return runtime


def compression_reply_bytes():
    value = json.loads(reply(SUMMARY_TEXT))
    value.update(id="invented_compression_reply", model="invented_model_5")
    return canonical(value)


def test_inert_retained_compression_http_plan():
    a, b = OwnedReceiver(observation_seconds=590.0), OwnedReceiver(observation_seconds=590.0)
    try:
        world = build_compression_world(a, b)
        assert type(world) is CompressionWorld and world.now == 10 and world.adapter is None
        assert len(world.routes) == len(world.ports) == 3 * len(PATHS) == 24
        assert all(route.expires_at == 100 and route.qualification_scope == "synthetic_only" for route in world.routes)
        assert world.compression_ports["compression"] is world.ports["compression_capability_compression"]
        assert len({id(world.request), id(world.title_request), id(world.compression_request)}) == 3
        current = (world.request, world.title_request, world.compression_request,
            world.binding, world.title_binding, world.compression_binding, world.policy, world.title_policy,
            world.compression_policy, world.profiles, world.admission.current, world.selection_port.current,
            world.title_selection_port.current, world.compression_selection_port.current, world.result_port.current)
        assert all(x is y if not callable(x) else x == y for x, y in zip(current, world.initial_originals, strict=True))
        assert (world.policy.producer_ref, world.policy.revision) == ("invented_producer", 4)
        assert (world.title_policy.producer_ref, world.title_policy.revision) == ("title_producer", 2)
        assert (world.compression_policy.producer_ref, world.compression_policy.revision) == ("compression_producer", 3)
        assert (world.compression_binding.selection_id, world.compression_binding.selection_version) == ("compression_selection", 6)
        assert world.profiles[0].allowed_routes == tuple(route.route_ref for route in world.routes)
        assert world.profiles[0].required_paths == PATHS
        for binding in (world.binding, world.title_binding, world.compression_binding):
            assert binding.profiles == ("restricted_documents",)
            routes = tuple(route for route in world.routes if route.selection_id == binding.selection_id)
            assert len(routes) == len(PATHS) and {route.path for route in routes} == set(PATHS)
            assert all(route.selection_version == binding.selection_version for route in routes)
        assert a.bound[0] == b.bound[0] == "127.0.0.1" and a.bound[1] != b.bound[1]
        for key, old in world.original_ports:
            if key not in ("main", "compression_capability_compression"):
                assert world.ports[key] is old and type(old) is ControlledPort and not old.sent
            else:
                assert id(old) not in world.receiver_bindings
        for receiver, endpoint, port, selection, model in zip((a, b), world.endpoints,
                (world.ports["main"], world.compression_ports["compression"]),
                (world.selection, world.compression_selection), ("invented_model_0", "invented_model_5"), strict=True):
            assert port.resolve() is port.destination is selection.destination
            assert selection.destination.model_ref == model
            assert endpoint.port == receiver.bound[1] and endpoint.address == receiver.bound[0]
            assert not port.observations and not hasattr(port, "sent")
            assert dict(world.transports)[port.destination.transport_ref] is port
            assert world.receiver_bindings[id(port)][0] is port
        for original, current in zip(world.original_routes, world.routes, strict=True):
            if original.route_ref not in ("route_main", "compression_compression"):
                assert current is original
            else:
                assert replace(current, destination=original.destination,
                    qualification_ref=original.qualification_ref) == original
        assert a.thread is b.thread is None
    finally:
        a.close()
        b.close()


def test_retained_compression_http_response_recipe():
    payload = compression_reply_bytes()
    assert payload == canonical(json.loads(payload))
    response = native._response(native._decode_response(payload, "invented_model_5"))
    assert type(response) is native.NativeMainResponse and response.model == "invented_model_5"
    content, finish = compression.ContextCompressor._summary_response_content(None, response)
    assert content == SUMMARY_TEXT and content.strip() and finish == "stop"
    assert native._response_bytes(response, "invented_model_5") == payload
    assert json.loads(payload)["usage"] == dict(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    with pytest.raises(NativeContentPolicyDenied):
        native._decode_response(payload, "invented_model_0")
    value = json.loads(payload)
    value["choices"][0]["finish_reason"] = "length"
    with pytest.raises(NativeContentPolicyDenied):
        native._decode_response(canonical(value), "invented_model_5")
    from types import SimpleNamespace
    with pytest.raises(RuntimeError, match="truncated"):
        compression.ContextCompressor._summary_response_content(
            SimpleNamespace(provider="synthetic_native", model="invented_model_0", summary_model="invented_model_5"), value)
    header, separator, body = response_bytes(payload).partition(b"\r\n\r\n")
    assert separator == b"\r\n\r\n" and body == payload
    assert b"Content-Length: " + str(len(body)).encode() in header.split(b"\r\n")


def test_genuine_retained_compression_http(ordinary_effects, caller_boundary, record):
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
    a, b = OwnedReceiver(observation_seconds=590.0), OwnedReceiver(observation_seconds=590.0)
    world, result, summary, qualifier = None, None, None, None
    receipts = {}
    verified = False
    try:
        world = build_compression_world(a, b)
        authority_originals = world.initial_originals
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
        qualifier = qualify_compression_world(world, record)
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
        original_result = canonical(result)
        prefix = world.agent._cached_system_prompt
        messages = result["messages"]
        message_rows = tuple(messages)
        with content_policy_scope(world.context):
            bind_compression(world)
            compressor = construct_compressor(world)
            compressor_state = dict(vars(compressor))
            summary = summarize(world, compressor, result)
        auxiliary, = adapter.auxiliary_invocations
        aux_frame, = (entry[0] for entry in adapter._aux_frames)
        record("compression_released", dict(summary=summary, output=auxiliary.output, graph=graph(runtime), charges=charges(runtime),
            envelope=asdict(auxiliary.envelope), response=auxiliary.response_bytes,
            input_stages=auxiliary.input_stages, output_stages=auxiliary.output_stages,
            source_id=id(auxiliary.source_completion), output_id=id(auxiliary.output), summary_id=id(summary),
            response_id=id(auxiliary.native_response), seconds=time.monotonic()-started))
        assert type(summary) is str and summary.strip() and summary is auxiliary.output["summary"]
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
        assert facts[1] == dict(content_tokens=compression.estimate_messages_tokens_rough(result["messages"]),
            ratio=0.2, context_length=65536, maximum=3276, summary_budget=2000)
        assert facts[2] == dict(serialized=compressor._serialize_for_summary(result["messages"]), pruned_skill_names=[])
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
        assert auxiliary.result_content is not None and auxiliary.native_response is aux_frame.record.native_response
        output = [json.loads(stage) for stage in auxiliary.output_stages]
        assert output[0] == dict(content="<think>scratch</think>## Goal\nKeep the blue folder.",
                                finish_reason="stop", nonempty=True, length_stop=False)
        assert output[1] == dict(stripped="## Goal\nKeep the blue folder.",
                                chosen="## Goal\nKeep the blue folder.", redacted="## Goal\nKeep the blue folder.")
        assert output[2]["restored"] == output[1]["redacted"]
        assert "User asked (deterministic" in output[2]["grounded"]
        assert compression._LEAN_USER_MESSAGES_HEADING in output[2]["augmented"]
        assert output[2]["augmented"] == compressor._augment_summary_lean(output[2]["grounded"], result["messages"])
        assert compression._build_recovery_footer("", len(result["messages"])) == ""
        assert output[3] == dict(prefixed=summary, output={"summary": summary})
        assert auxiliary.output["summary"] is summary and auxiliary.output_bytes == canonical(auxiliary.output)
        assert "scratch" not in summary and summary.startswith(compression.SUMMARY_PREFIX)
        for index, (payload, content) in enumerate(zip(auxiliary.input_stages, auxiliary.stage_contents, strict=True)):
            row = runtime._contents[id(content)]
            assert row.payload == payload
            assert all(x is y for x, y in zip(row.parents, (aux_frame.source_content, *auxiliary.stage_contents[:index]), strict=True))
        row = runtime._contents[id(auxiliary.request_content)]
        assert row.payload == auxiliary.request_bytes
        assert all(x is y for x, y in zip(row.parents, (aux_frame.source_content, *auxiliary.stage_contents), strict=True))
        assert runtime._contents[id(auxiliary.result_content)].parents == (auxiliary.request_content,)
        assert runtime._contents[id(auxiliary.result_content)].payload == auxiliary.response_bytes == summary_bytes
        assert type(auxiliary.native_response) is native.NativeMainResponse
        assert auxiliary.native_response.model == "invented_model_5"
        assert native._response_bytes(auxiliary.native_response, "invented_model_5") == summary_bytes
        assert len(auxiliary.output_stages) == len(auxiliary.output_contents) == 4
        extracted, finish = compressor._summary_response_content(auxiliary.native_response)
        transformed, stripped, augmented = compressor._summary_output(extracted, messages, facts[2]["pruned_skill_names"], True)
        assert auxiliary.output_stages == tuple(canonical(value) for value in (
            dict(content=extracted, finish_reason=finish, nonempty=True, length_stop=False),
            stripped, augmented, dict(prefixed=compressor._with_summary_prefix(transformed), output=auxiliary.output)))
        base = (aux_frame.source_content, auxiliary.request_content, auxiliary.result_content)
        for index, (payload, content) in enumerate(zip(auxiliary.output_stages, auxiliary.output_contents, strict=True)):
            row = runtime._contents[id(content)]
            assert row.payload == payload
            assert all(x is y for x, y in zip(row.parents, (*base, *auxiliary.output_contents[:index]), strict=True))
        row = runtime._contents[id(auxiliary.output_content)]
        assert row.payload == auxiliary.output_bytes == canonical(auxiliary.output)
        assert all(x is y for x, y in zip(row.parents, (*base, *auxiliary.output_contents), strict=True))
        assert vars(compressor) == compressor_state
        assert compressor._previous_summary is None and compressor.compression_count == 0
        assert canonical(result) == original_result and world.agent._cached_system_prompt is prefix
        assert adapter.auxiliary_local_results == ()
        current_authorities = (world.request, world.title_request, world.compression_request,
            world.binding, world.title_binding, world.compression_binding, world.policy, world.title_policy,
            world.compression_policy, world.profiles, world.admission.current, world.selection_port.current,
            world.title_selection_port.current, world.compression_selection_port.current, world.result_port.current)
        assert all(x is y if not callable(x) else x == y for x, y in zip(current_authorities, authority_originals, strict=True))
        assert len(frame.request_contents) == 1
        assert len(wire["messages"]) <= 32
        assert all(len(row["content"].encode("utf-8")) <= 16384 for row in wire["messages"])
        assert all(len(payload) <= 262144 for payload in (auxiliary.source_bytes, auxiliary.output_bytes))
        assert adapter.send_count == adapter.auxiliary_send_count == 1 and adapter.fallback_send_count == 0
        assert not adapter.failed_main_attempts and not adapter.fallback_invocations
        assert len(caller_boundary.dispatches) == 1 and not world.sdk_calls
        assert all(vars(world.agent)[key] is value for key, value in original_agent.items())
        assert (world.routes, world.profiles, world.transports, world.selection, world.compression_selection) == originals[:5]
        assert world.now == 10 and all(route.expires_at == 100 for route in world.routes)
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
        assert len(runtime._envelopes) == 2
        assert len(runtime._contents) == len(main_contents) + 1 + len(auxiliary.stage_contents) + 2 + len(auxiliary.output_contents) + 1 == 21
        assert all(len(payload) <= 262144 for payload in (*auxiliary.input_stages, *auxiliary.output_stages, auxiliary.request_bytes, invocation.request_bytes))
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
        adapter.validate_scope()
        verified = True
        record("verified", dict(result=result, summary=summary, output=auxiliary.output, graph=graph(runtime),
            charges=charges(runtime), qualification_charges=charges(qualifier),
            envelopes=[asdict(row.envelope) for row in runtime._envelopes.values()],
            hops=[lineage.hops for lineage in runtime._lineages.values()],
            main_usage=dict(prompt_tokens=result["prompt_tokens"], completion_tokens=result["completion_tokens"], total_tokens=result["total_tokens"]),
            compression_usage=json.loads(auxiliary.response_bytes)["usage"], estimated_cost_usd=result["estimated_cost_usd"],
            cost_status=result["cost_status"], conversation_exit=True, auxiliary_scope_exit=True,
            agent_attributes_preserved=True, seconds=time.monotonic()-started))
        world.agent.close()
        world.agent.close()
        assert adapter.invocations == (invocation,) and adapter.auxiliary_invocations == (auxiliary,)
        with pytest.raises(NativeContentPolicyDenied) as caught:
            turn(world, USER_TEXT, system_message="Fixed system")
        assert type(caught.value) is NativeContentPolicyDenied and world.context._denied
        assert adapter._turn_pending is adapter._operation is None and adapter._constructed.closed
        assert adapter._send_count == adapter._aux_send_count == 1 and adapter._fallback_send_count == 0
        assert len(world.ports["main"].observations) == len(world.compression_ports["compression"].observations) == 2
        record("terminal_refusal", dict(error_type=type(caught.value).__name__, reason=caught.value.args,
            error_id=id(caught.value), closed=adapter._constructed.closed, sends=adapter._send_count,
            compression_sends=adapter._aux_send_count, seconds=time.monotonic()-started))
    finally:
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
            seconds=time.monotonic()-started))
