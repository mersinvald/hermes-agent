"""Check one retained title through two owned HTTP listeners."""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

import pytest
from hermes_content_policy import ContentPolicyRuntime, PATHS

from agent import native_main_payload as native, title_generator as titles
from agent.content_policy import NativeContentPolicyDenied, content_policy_scope
from agent.transports.owned_loopback_http import OwnedLoopbackEndpoint, OwnedLoopbackHTTPTransport
from tests.agent.owned_loopback_receiver import OwnedReceiver, ResponseStep, response_bytes
from tests.agent.test_native_controlled_conversation import (
    AuxiliaryWorld, bind_title, construct, title_from, turn,
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

EXPECTED_TITLE = "Blue folder retention"
EXPECTED_OUTPUT = b'{"title":"Blue folder retention"}'
MAIN_TEXT = "Keep the blue folder."
USER_TEXT = "Invented folder request."


def build_title_world(a, b):
    world = AuxiliaryWorld()
    world.initial_originals = (world.request, world.title_request, world.binding, world.title_binding,
        world.policy, world.title_policy, world.profiles, world.admission.current,
        world.selection_port.current, world.title_selection_port.current, world.result_port.current)
    world.original_routes = world.routes
    world.original_ports = tuple(world.ports.items())
    replacements, endpoints = {}, []
    for key, role, model, receiver, route_ref in (
        ("main", "p", "invented_model_0", a, "route_main"),
        ("title_capability_title", "c", "invented_model_5", b, "title_title"),
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
            world.title_ports["title"] = port
            world.title_selection = replace(world.title_selection, destination=destination)
        endpoints.append(endpoint)
    world.routes = tuple(replace(route, destination=replacements[route.route_ref],
        qualification_ref="synthetic_loopback_" + ("main" if route.route_ref == "route_main" else "title"))
        if route.route_ref in replacements else route for route in world.routes)
    world.transports = tuple((port.destination.transport_ref, port) for port in world.ports.values())
    world.endpoints = tuple(endpoints)
    world.qualifications = []
    return world


def qualify_title_world(world, record):
    runtime = ContentPolicyRuntime(admission=world.admission, profiles=world.profiles,
        routes=world.routes, transports=world.transports, clock=lambda: world.now, policy_revision=9)
    world.qualifier = runtime
    main, title = world.binding, world.title_binding
    assert tuple(asdict(main).values())[:6] == tuple(asdict(title).values())[:6]
    assert (main.selection_id, main.selection_version) == ("selection", 3)
    assert (title.selection_id, title.selection_version) == ("title_selection", 5)
    for route in world.routes:
        key = (route.selection_id, route.selection_version)
        assert key in ((main.selection_id, main.selection_version), (title.selection_id, title.selection_version))
        request = world.request if key == (main.selection_id, main.selection_version) else world.title_request
        binding = world.admission.current(request)
        assert binding is (main if request is world.request else title)
        source = runtime.admit(request, QUALIFICATION)
        result = runtime.dispatch(request, source, path=route.path, route_ref=route.route_ref)
        envelope = tuple(runtime._envelopes.values())[-1].envelope
        assert envelope.source == binding and envelope.payload == QUALIFICATION
        assert envelope.destination == route.destination and envelope.path == route.path
        assert runtime._contents[id(result)].parents == (source,)
        world.qualifications.append((request, source, result, envelope))
        record("qualification", dict(route=route.route_ref, envelope=asdict(envelope),
            source=asdict(binding), result=runtime._contents[id(result)].payload, charges=charges(runtime)))
    for port in world.ports.values():
        if type(port) is ControlledPort:
            assert len(port.sent) == 1
            port.sent.clear()
    assert len(world.qualifications) == len(runtime._envelopes) == 2 * len(PATHS)
    assert len(runtime._contents) == 2 * len(world.routes)
    return runtime


def title_reply_bytes():
    value = json.loads(reply('{"title":"Blue folder retention"}'))
    value.update(id="invented_title_reply", model="invented_model_5")
    return canonical(value)


def test_inert_retained_title_http_plan():
    a, b = OwnedReceiver(observation_seconds=590.0), OwnedReceiver(observation_seconds=590.0)
    try:
        world = build_title_world(a, b)
        assert type(world) is AuxiliaryWorld and world.now == 10 and world.adapter is None
        assert len(world.routes) == len(world.ports) == 2 * len(PATHS)
        assert all(route.expires_at == 100 and route.qualification_scope == "synthetic_only" for route in world.routes)
        assert world.title_ports["title"] is world.ports["title_capability_title"]
        assert (world.request, world.title_request, world.binding, world.title_binding,
            world.policy, world.title_policy, world.profiles) == world.initial_originals[:7]
        assert (world.admission.current, world.selection_port.current, world.title_selection_port.current,
            world.result_port.current) == world.initial_originals[7:]
        assert tuple(asdict(world.binding).values())[:6] == tuple(asdict(world.title_binding).values())[:6]
        assert (world.policy.producer_ref, world.policy.revision) == ("invented_producer", 4)
        assert (world.title_policy.producer_ref, world.title_policy.revision) == ("title_producer", 2)
        assert world.profiles[0].allowed_routes == tuple(route.route_ref for route in world.routes)
        assert world.profiles[0].required_paths == PATHS
        assert world.binding.profiles == world.title_binding.profiles == world.policy.profiles == world.title_policy.profiles == ("restricted_documents",)
        for binding in (world.binding, world.title_binding):
            routes = tuple(route for route in world.routes if route.selection_id == binding.selection_id)
            assert len(routes) == len(PATHS) and {route.path for route in routes} == set(PATHS)
            assert all(route.selection_version == binding.selection_version for route in routes)
        assert a.bound[0] == b.bound[0] == "127.0.0.1" and a.bound[1] != b.bound[1]
        for key, old in world.original_ports:
            if key not in ("main", "title_capability_title"):
                assert world.ports[key] is old and type(old) is ControlledPort
                assert not old.sent
            else:
                assert id(old) not in world.receiver_bindings
        for receiver, endpoint, port, selection in zip((a, b), world.endpoints,
                (world.ports["main"], world.title_ports["title"]), (world.selection, world.title_selection), strict=True):
            assert port.resolve() is port.destination is selection.destination
            assert endpoint.port == receiver.bound[1] and endpoint.address == receiver.bound[0]
            assert not port.observations and not hasattr(port, "sent")
            assert dict(world.transports)[port.destination.transport_ref] is port
            assert world.receiver_bindings[id(port)][0] is port
        for original, current in zip(world.original_routes, world.routes, strict=True):
            if original.route_ref not in ("route_main", "title_title"):
                assert current is original
            else:
                assert replace(current, destination=original.destination,
                    qualification_ref=original.qualification_ref) == original
        assert a.thread is b.thread is None
    finally:
        a.close()
        b.close()


def test_retained_title_http_response_recipe():
    payload = title_reply_bytes()
    assert payload == canonical(json.loads(payload))
    response = native._response(native._decode_response(payload, "invented_model_5"))
    assert type(response) is native.NativeMainResponse and response.model == "invented_model_5"
    stages, output = titles._title_output_stages(response, lambda: None)
    assert output == {"title": EXPECTED_TITLE} and canonical(output) == EXPECTED_OUTPUT
    assert stages == tuple(canonical(value) for value in (
        '{"title":"Blue folder retention"}', EXPECTED_TITLE, EXPECTED_TITLE, {"words": 3, "accepted": True}))
    assert native._response_bytes(response, "invented_model_5") == payload
    with pytest.raises(NativeContentPolicyDenied):
        native._decode_response(payload, "invented_model_0")
    header, separator, body = response_bytes(payload).partition(b"\r\n\r\n")
    assert separator == b"\r\n\r\n" and body == payload
    assert b"Content-Length: " + str(len(body)).encode() in header.split(b"\r\n")


def test_genuine_retained_title_http(ordinary_effects, caller_boundary, record):
    started = time.monotonic()
    modules = (native, titles, sys.modules["agent.transports.owned_loopback_http"],
        sys.modules["hermes_content_policy"], sys.modules["tests.agent.test_native_controlled_conversation"])
    record("source", dict(interpreter=sys.executable, prefix=sys.prefix,
        modules={module.__name__: dict(path=str(Path(module.__file__).resolve()),
            sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()) for module in modules}))
    a, b = OwnedReceiver(observation_seconds=590.0), OwnedReceiver(observation_seconds=590.0)
    world, result, title, qualifier = None, None, None, None
    receipts = {}
    verified = False
    try:
        world = build_title_world(a, b)
        originals = (world.routes, world.profiles, world.transports, world.selection, world.title_selection,
            tuple((port, port.resolve, port.send) for port in world.ports.values()))
        listeners = (a.listener, b.listener)
        main_value = json.loads(reply(MAIN_TEXT))
        main_value["model"] = "invented_model_0"
        main_bytes, title_bytes = canonical(main_value), title_reply_bytes()
        record("operator", dict(routes=[asdict(route) for route in world.routes],
            profiles=[asdict(profile) for profile in world.profiles],
            main_binding=asdict(world.binding), title_binding=asdict(world.title_binding),
            main_policy=asdict(world.policy), title_policy=asdict(world.title_policy),
            main_selection=asdict(world.selection), title_selection=asdict(world.title_selection),
            endpoints=[asdict(endpoint) for endpoint in world.endpoints], now=world.now))
        for receiver, endpoint, payload in zip((a, b), world.endpoints, (main_bytes, title_bytes), strict=True):
            receiver.start((ResponseStep(endpoint.request_path.encode(), response_bytes(ACK), QUALIFICATION),
                ResponseStep(endpoint.request_path.encode(), response_bytes(payload))))
        qualifier = qualify_title_world(world, record)
        a.wait(1)
        b.wait(1)
        record("qualification_complete", dict(graph=graph(qualifier), charges=charges(qualifier),
            envelopes=[asdict(row.envelope) for row in qualifier._envelopes.values()],
            hops=[lineage.hops for lineage in qualifier._lineages.values()], seconds=time.monotonic()-started))
        assert len(a.rows) == len(b.rows) == 1 and a.rows[0]["settled"] and b.rows[0]["settled"]
        assert a._mark is b._mark is None
        World.bind(world, qualify=False)
        construct(world, model="invented_model_0", max_tokens=None)
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
        record("main_released", dict(result=result, graph=graph(runtime), charges=charges(runtime),
            envelope=asdict(invocation.envelope), response=invocation.response_bytes,
            completion_id=id(completion), result_id=id(result), frame_id=id(frame), invocation_id=id(invocation),
            native_response_id=id(invocation.native_response), seconds=time.monotonic()-started))
        with content_policy_scope(world.context):
            bind_title(world)
            title = title_from(world, result, field="final_response")
        auxiliary, = adapter.auxiliary_invocations
        aux_frame, = (entry[0] for entry in adapter._aux_frames)
        record("title_released", dict(title=title, output=auxiliary.output, graph=graph(runtime), charges=charges(runtime),
            envelope=asdict(auxiliary.envelope), response=auxiliary.response_bytes,
            input_stages=auxiliary.input_stages, output_stages=auxiliary.output_stages,
            source_id=id(auxiliary.source_completion), output_id=id(auxiliary.output), title_id=id(title),
            response_id=id(auxiliary.native_response), seconds=time.monotonic()-started))
        assert type(title) is str and title == EXPECTED_TITLE and title is auxiliary.output["title"]
        assert auxiliary.source_completion is completion and auxiliary.source_completion.result is result
        assert auxiliary.source_frame is frame and auxiliary.source_invocation is invocation
        assert auxiliary.source_handle is world.request and auxiliary.request_handle is world.title_request
        assert aux_frame.field == "final_response" and aux_frame.text is result["final_response"]
        assert aux_frame.completion is completion and aux_frame.record is auxiliary
        expected_parents = (*main_parents, completion.content)
        assert all(x is y for x, y in zip(auxiliary.parents, expected_parents, strict=True))
        assert all(handle is world.request for handle in auxiliary.parent_requests)
        assert len(auxiliary.parent_requests) == len(expected_parents)
        assert auxiliary.source_bytes == canonical(result["final_response"])
        assert runtime._contents[id(aux_frame.source_content)].parents == auxiliary.parents
        assert auxiliary.selection is world.title_selection and auxiliary.producer_policy is world.title_policy
        assert auxiliary.path == "title" and auxiliary.task == "title_generation"
        assert (auxiliary.target.selection_id, auxiliary.target.selection_version) == ("title_selection", 5)
        assert auxiliary.envelope.destination == auxiliary.observed_destination == world.title_selection.destination
        assert auxiliary.envelope.path == "title" and auxiliary.envelope.source == world.title_binding
        assert auxiliary.request_bytes == auxiliary.envelope.payload
        sealed = json.loads(auxiliary.request_bytes)
        assert sealed["path"] == "title" and sealed["task"] == "title_generation"
        assert sealed["selection"] == ["title_selection", 5, "chat_completions"]
        assert sealed["destination"] == asdict(world.title_selection.destination)
        assert sealed["request"]["model"] == "invented_model_5"
        assert sealed["request"] == aux_frame.request and aux_frame.request is aux_frame.preparation.value
        assert len(auxiliary.input_stages) == len(auxiliary.stage_contents) == 7
        expected_stages, request = titles._title_input_stages(result["final_response"], "English", world.title_selection, lambda: None)
        assert auxiliary.input_stages == expected_stages and aux_frame.request == request
        for index, (payload, content) in enumerate(zip(auxiliary.input_stages, auxiliary.stage_contents, strict=True)):
            row = runtime._contents[id(content)]
            assert row.payload == payload
            assert all(x is y for x, y in zip(row.parents, (aux_frame.source_content, *auxiliary.stage_contents[:index]), strict=True))
        row = runtime._contents[id(auxiliary.request_content)]
        assert row.payload == auxiliary.request_bytes
        assert all(x is y for x, y in zip(row.parents, (aux_frame.source_content, *auxiliary.stage_contents), strict=True))
        assert runtime._contents[id(auxiliary.result_content)].parents == (auxiliary.request_content,)
        assert auxiliary.response_bytes == title_bytes and type(auxiliary.native_response) is native.NativeMainResponse
        assert auxiliary.native_response.model == "invented_model_5"
        assert native._response_bytes(auxiliary.native_response, "invented_model_5") == title_bytes
        assert auxiliary.output is aux_frame.record.output and auxiliary.output_bytes == EXPECTED_OUTPUT
        assert len(auxiliary.output_stages) == len(auxiliary.output_contents) == 4
        expected_outputs, output = titles._title_output_stages(auxiliary.native_response, lambda: None)
        assert auxiliary.output_stages == expected_outputs and auxiliary.output == output
        base = (aux_frame.source_content, auxiliary.request_content, auxiliary.result_content)
        for index, (payload, content) in enumerate(zip(auxiliary.output_stages, auxiliary.output_contents, strict=True)):
            row = runtime._contents[id(content)]
            assert row.payload == payload
            assert all(x is y for x, y in zip(row.parents, (*base, *auxiliary.output_contents[:index]), strict=True))
        row = runtime._contents[id(auxiliary.output_content)]
        assert row.payload == auxiliary.output_bytes == canonical(auxiliary.output)
        assert all(x is y for x, y in zip(row.parents, (*base, *auxiliary.output_contents), strict=True))
        assert adapter.send_count == adapter.auxiliary_send_count == 1 and adapter.fallback_send_count == 0
        assert not adapter.failed_main_attempts and not adapter.fallback_invocations
        assert len(caller_boundary.dispatches) == 1 and not world.sdk_calls
        assert all(vars(world.agent)[key] is value for key, value in original_agent.items())
        assert (world.routes, world.profiles, world.transports, world.selection, world.title_selection) == originals[:5]
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
        assert len(runtime._envelopes) == 2
        assert len(runtime._contents) == len(main_contents) + 1 + len(auxiliary.stage_contents) + 2 + len(auxiliary.output_contents) + 1
        assert all(len(payload) <= 262144 for payload in (*auxiliary.input_stages, *auxiliary.output_stages, auxiliary.request_bytes, invocation.request_bytes))
        assert len(auxiliary.response_bytes) <= 65536 and len(invocation.response_bytes) <= 65536
        a.wait(2)
        b.wait(2)
        for receiver in (a, b):
            mark = receiver.mark_window()
            assert mark.row_count == 2 and mark.active_count == 0
        for name, receiver, listener, operation, payload in zip(("a", "b"), (a, b), listeners,
                (invocation, auxiliary), (main_bytes, title_bytes), strict=True):
            receipt = receipts[name] = receiver.finish()
            record(name + "_final", asdict(receipt))
            assert_receipt(receipt)
            receiver.check()
            assert len(receipt.connections) == 2 and receipt.connections[receipt.mark.row_count:] == ()
            assert receipt.listener_id == id(listener) and receiver.listener is listener
            port = world.ports["main"] if name == "a" else world.title_ports["title"]
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
        record("verified", dict(result=result, title=title, output=auxiliary.output, graph=graph(runtime),
            charges=charges(runtime), qualification_charges=charges(qualifier),
            envelopes=[asdict(row.envelope) for row in runtime._envelopes.values()],
            hops=[lineage.hops for lineage in runtime._lineages.values()],
            main_usage=dict(prompt_tokens=result["prompt_tokens"], completion_tokens=result["completion_tokens"], total_tokens=result["total_tokens"]),
            title_usage=json.loads(auxiliary.response_bytes)["usage"], estimated_cost_usd=result["estimated_cost_usd"],
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
        assert len(world.ports["main"].observations) == len(world.title_ports["title"].observations) == 2
        record("terminal_refusal", dict(error_type=type(caught.value).__name__, reason=caught.value.args,
            error_id=id(caught.value), closed=adapter._constructed.closed, sends=adapter._send_count,
            title_sends=adapter._aux_send_count, seconds=time.monotonic()-started))
    finally:
        if world is not None:
            for runtime_name, current in (("qualification", qualifier), ("native", None if world.adapter is None else world.adapter._runtime)):
                if current is not None:
                    record(runtime_name + "_retained_final", dict(graph=graph(current), charges=charges(current),
                        envelopes=[asdict(row.envelope) for row in current._envelopes.values()],
                        hops=[lineage.hops for lineage in current._lineages.values()], verified=verified,
                        main_release=result is not None, title_release=title is not None))
            for name, port in (("main", world.ports["main"]), ("title", world.title_ports["title"])):
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
