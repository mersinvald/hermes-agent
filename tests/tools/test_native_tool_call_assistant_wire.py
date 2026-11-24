"""Retain the stock assistant projection from a genuine closed P source."""

import asyncio
import hashlib
import json
import threading
import time
from contextvars import copy_context
from dataclasses import FrozenInstanceError, replace

import pytest

from agent import native_main_payload as native
from agent.content_policy import NativeContentPolicyDenied, content_policy_scope
from tests.tools.test_native_tool_call_assistant_source import (
    source_world as source_world, enable, receipt as source_receipt,
    ProbeWorld, excluded as excluded,
    invented_offline_configuration as invented_offline_configuration,
    retain_case_durations as retain_case_durations,
    retain_incremental_case_receipts as retain_incremental_case_receipts,
    record_case_start as record_case_start,
)


def capture_closed(world):
    start = time.monotonic()
    with content_policy_scope(world.context):
        producer = enable(world)
        world.run()
        world.agent.close()
    world.source_timings["capture_close"] = time.monotonic() - start
    assert producer.probe.closed and producer.constructed.closed
    assert producer.windows == 3 and producer.probe.stage_index == 8
    return producer.record


@pytest.fixture
def projection_source(source_world):
    return source_world, capture_closed(source_world)


def proof(world):
    owner = native._assistant_wire_owner(world.adapter)
    result = source_receipt(world, native._assistant_source_owner(world.adapter))
    result.update(projection_windows=owner.windows, projection_results=len(owner.records),
        raw_returned=owner.raw_slot[0] is not None, returned_handle=owner.message_content is not None,
        charge_before=owner.charge_before, charge_after=owner.charge_after,
        original_closed=owner.probe.closed and owner.producer.constructed.closed,
        projection_sends=0, source_record_identity=owner.source_result is owner.producer.record)
    data = owner.raw_slot[2]
    result["Wa"] = None if data is None else dict(utf8=data.decode(), hex=data.hex(), bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest())
    result["parents"] = {name: [next(label for label, h in handles(world).items() if h is parent)
        for parent in world.adapter._runtime._content(handle).parents]
        for name, handle in handles(world).items()}
    result["restrictions"] = {name: list(world.adapter._runtime._content(handle).profiles)
        for name, handle in handles(world).items()}
    return result


def handles(world):
    producer = native._assistant_source_owner(world.adapter)
    probe = producer.probe
    result = dict(A=probe.binding.authorization_content, Q=probe.request_content, W=probe.wire_content,
        R=probe.response_content, O=probe.observation_content, Ba=producer.authorization_content,
        Da=producer.message_content)
    owner = native._assistant_wire_owner(world.adapter)
    if owner.message_content is not None:
        result["Wa"] = owner.message_content
    return result


def assert_retained(world, source):
    adapter, runtime = world.adapter, world.adapter._runtime
    owner = native._assistant_wire_owner(adapter)
    payload = owner.raw_slot[2]
    assert payload == b'[{"role":"assistant","content":"","reasoning":null,"finish_reason":"tool_calls","tool_calls":[{"id":"invented-call-1","type":"function","function":{"name":"delegate_task","arguments":"{\\"context\\":null,\\"goal\\":\\"Count blue squares.\\",\\"role\\":\\"leaf\\"}"}}]}]'
    assert len(payload) == 259
    assert hashlib.sha256(payload).hexdigest() == "3c20f50a804e6a4a5022eeae19f1780f532e7721dda64e60889b0911a1f6f9d5"
    assert owner.raw_slot[0] is owner.messages and owner.raw_slot[1] is owner.message_anchors
    assert payload == json.dumps(owner.messages, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    assert owner.inputs[0] is source.message and owner.model is owner.probe.selection.destination.model_ref
    assert owner.messages is not owner.inputs and owner.messages[0] is not source.message
    row, original = owner.messages[0], source.message
    assert list(row) == ["role", "content", "reasoning", "finish_reason", "tool_calls"]
    assert row["tool_calls"] is not original["tool_calls"]
    assert row["tool_calls"][0] is not original["tool_calls"][0]
    assert list(row["tool_calls"][0]) == ["id", "type", "function"]
    assert row["tool_calls"][0]["function"] is original["tool_calls"][0]["function"]
    assert list(row["tool_calls"][0]["function"]) == ["name", "arguments"]
    for name in ("role", "content", "reasoning", "finish_reason"):
        assert row[name] is original[name]
    retained = runtime._content(owner.message_content)
    assert retained.payload is payload and retained.parents == (source.message_content,)
    assert retained.parents[0] is source.message_content and retained.path == "main"
    assert retained.profiles == ("restricted_documents",)
    assert len(runtime._contents) == 8 and len(runtime._envelopes) == 1
    assert runtime._retained_bytes == 3265 + len(source.authorization_bytes) + len(source.message_bytes) + len(payload)
    assert owner.charge_after == (owner.charge_before[0] + 1, owner.charge_before[1] + len(payload))
    assert adapter.send_count == len(world.ports["main"].sent) == 1
    assert not world.sdk_calls and not adapter.invocations and not adapter._completions
    assert owner.producer.windows == 3 and owner.probe.stage_index == 8 and owner.probe.closed
    assert all(tuple(vars(obj).items()) == snapshot for obj, snapshot in owner.snapshots)


def test_actual_assistant_wire(projection_source, record_property, monkeypatch):
    world, source = projection_source
    adapter = world.adapter
    count = len(native._assistant_wire_owners)
    runtime = adapter._runtime
    before = len(runtime._contents), runtime._retained_bytes
    namespace = tuple(vars(adapter).items())
    with content_policy_scope(world.context):
        world.admission_hook = lambda: pytest.fail("Copied source reached an operator callback")
        for invalid in (replace(source), object()):
            with pytest.raises(NativeContentPolicyDenied, match="source_unregistered"):
                adapter.retain_tool_call_assistant_wire(invalid)
        assert len(native._assistant_wire_owners) == count
        assert native._assistant_wire_owner(adapter) is None and not adapter._invalid
        assert (len(runtime._contents), runtime._retained_bytes) == before
        world.admission_hook = None
        errors = []
        context = copy_context()
        def foreign():
            try:
                context.run(adapter.retain_tool_call_assistant_wire, source)
            except NativeContentPolicyDenied as error:
                errors.append(error)
        thread = threading.Thread(target=foreign)
        thread.start()
        thread.join()
        assert len(errors) == 1 and native._assistant_wire_owner(adapter) is None
        rejected = []
        def reenter():
            world.admission_hook = None
            with pytest.raises(NativeContentPolicyDenied, match="reentry_or_thread"):
                _ = adapter.tool_call_assistant_wire_results
            for call in (lambda: runtime._admission.current(adapter._handle),
                         adapter._transport_wrappers[world.selection.destination.transport_ref].resolve,
                         lambda: adapter._around(adapter._clock)):
                with pytest.raises(NativeContentPolicyDenied, match="reentry_or_thread"):
                    call()
            rejected.append(True)
        world.admission_hook = reenter
        world.binding = replace(world.binding, profiles=())
        start = time.monotonic()
        result = adapter.retain_tool_call_assistant_wire(source)
        world.source_timings["projection"] = time.monotonic() - start
    start = time.monotonic()
    assert rejected == [True] and len(native._assistant_wire_owners) == count + 1
    assert result is adapter.tool_call_assistant_wire_results[0]
    assert result.source_result is source and type(result) is native.NativeToolCallAssistantWireResult
    assert result.messages is native._assistant_wire_owner(adapter).messages
    assert_retained(world, source)
    assert native._same_anchors(tuple(value for pair in vars(adapter).items() for value in pair),
                               tuple(value for pair in namespace for value in pair))
    with pytest.raises(FrozenInstanceError):
        result.message_bytes = b"copy"
    with content_policy_scope(world.context):
        with pytest.raises(NativeContentPolicyDenied, match="attempt_spent"):
            adapter.retain_tool_call_assistant_wire(source)
    assert adapter.tool_call_assistant_sources[0] is source
    owner = native._assistant_wire_owner(adapter)
    with pytest.raises(NativeContentPolicyDenied, match="window_missing"):
        owner.current(adapter._handle)
    for obj, name, value in ((owner, "model", "changed"),
                            (world.agent, "_tool_interrupt_reason", "changed"),
                            (owner.producer.response.choices[0], "finish_reason", "stop"),
                            (native._WIRE_CONVERTER, "__kwdefaults__", {"changed": True})):
        with monkeypatch.context() as patch:
            patch.setattr(obj, name, value)
            with pytest.raises(NativeContentPolicyDenied):
                _ = adapter.tool_call_assistant_wire_results
        assert adapter.tool_call_assistant_wire_results[0] is result
    proof_errors = []
    def foreign_proof():
        try:
            _ = adapter.tool_call_assistant_wire_results
        except NativeContentPolicyDenied as error:
            proof_errors.append(error)
    thread = threading.Thread(target=foreign_proof)
    thread.start()
    thread.join()
    assert len(proof_errors) == 1
    assert adapter.tool_call_assistant_wire_results[0] is result
    world.source_timings["local_assertions"] = time.monotonic() - start
    record_property("projection_proof", json.dumps(proof(world)))


def test_selected_expiry_after_derived_output(excluded, record_property):
    world = ProbeWorld()
    world.routes = tuple(replace(route, expires_at=200) if route.path != "main" else route for route in world.routes)
    world.routes = (*world.routes, replace(world.routes[0], route_ref="alternate_main", expires_at=200))
    world.profiles = (replace(world.profiles[0], allowed_routes=tuple(route.route_ref for route in world.routes)),)
    start = time.monotonic()
    world.construct()
    world.source_timings = dict(setup=time.monotonic() - start)
    source = capture_closed(world)
    try:
        with content_policy_scope(world.context):
            def expire():
                owner = native._assistant_wire_owner(world.adapter)
                if owner is not None and owner.message_content is not None and owner.windows == 3:
                    world.now = 100
            final_port = world.adapter._operator_transports[world.routes[-1].destination.transport_ref]
            final_port.resolve_hook = expire
            start = time.monotonic()
            with pytest.raises(NativeContentPolicyDenied, match="selection_expired"):
                world.adapter.retain_tool_call_assistant_wire(source)
            world.source_timings["projection"] = time.monotonic() - start
            final_port.resolve_hook = None
            world.now = 10
        owner = native._assistant_wire_owner(world.adapter)
        assert owner.phase == "denied" and not owner.records and world.adapter.tool_call_assistant_wire_results == ()
        assert_retained(world, source)
        assert len(world.qualification_records) == 9
        assert all(any(route.path == path and route.expires_at > 100 for route in world.routes)
                   for path in native.shared_policy().PATHS)
        assert owner.probe.phase == "released" and not world.adapter._invalid
        with content_policy_scope(world.context):
            with pytest.raises(NativeContentPolicyDenied, match="attempt_spent"):
                world.adapter.retain_tool_call_assistant_wire(source)
    finally:
        record_property("projection_proof", json.dumps(proof(world)))


def test_unregistered_adapter_preflight():
    adapter = object.__new__(native.NativeMainPayloadAdapter)
    class Hostile:
        def __str__(self):
            pytest.fail("Source conversion ran")
        def __eq__(self, other):
            pytest.fail("Source equality ran")
    before = native._assistant_wire_owners
    with pytest.raises(NativeContentPolicyDenied, match="adapter_invalid"):
        adapter.retain_tool_call_assistant_wire(Hostile())
    assert native._assistant_wire_owners is before


def test_original_control_after_actual_output(projection_source, record_property):
    world, source = projection_source
    original = asyncio.CancelledError("Invented projection control")
    try:
        with content_policy_scope(world.context):
            def stop():
                owner = native._assistant_wire_owner(world.adapter)
                if owner is not None and owner.raw_slot[0] is not None:
                    raise original
            world.clock_hook = stop
            start = time.monotonic()
            with pytest.raises(asyncio.CancelledError) as observed:
                world.adapter.retain_tool_call_assistant_wire(source)
            world.source_timings["projection"] = time.monotonic() - start
            world.clock_hook = None
        owner = native._assistant_wire_owner(world.adapter)
        assert observed.value is original and owner.control_exception is original and owner.derive_error is original
        assert owner.raw_slot[0] is owner.messages and owner.raw_slot[2] is owner.message_bytes
        assert len(owner.message_bytes) == 259 and owner.message_content is None
        assert owner.charge_after == owner.charge_before
        assert world.adapter.tool_call_assistant_wire_results == () and not owner.records
        assert len(world.adapter._runtime._contents) == 7 and not world.adapter._invalid
        assert owner.probe.phase == "released" and owner.probe.closed
        assert world.adapter.tool_call_assistant_sources[0] is source
    finally:
        record_property("projection_proof", json.dumps(proof(world)))


@pytest.mark.parametrize("boundary", [50, 49], ids=["expired50", "live49"])
def test_final_required_path_coverage(excluded, record_property, boundary):
    world = ProbeWorld()
    non_main = next(route.path for route in world.routes if route.path != "main")
    world.routes = tuple(replace(route, expires_at=50) if route.path == non_main else route
                         for route in world.routes)
    start = time.monotonic()
    world.construct()
    world.source_timings = dict(setup=time.monotonic() - start)
    source = capture_closed(world)
    ticks, returned = [], []
    actual_clock = world.now
    try:
        with content_policy_scope(world.context):
            def advance():
                owner = native._assistant_wire_owner(world.adapter)
                if owner is not None and owner.windows == 3:
                    ticks.append(world.now)
                    if len(ticks) == 8:
                        world.now = boundary
            world.clock_hook = advance
            start = time.monotonic()
            try:
                if boundary == 50:
                    with pytest.raises(NativeContentPolicyDenied, match="coverage_missing"):
                        returned.append(world.adapter.retain_tool_call_assistant_wire(source))
                else:
                    returned.append(world.adapter.retain_tool_call_assistant_wire(source))
            finally:
                world.source_timings["projection"] = time.monotonic() - start
                actual_clock = world.now
                world.clock_hook = None
                world.now = 10
        owner = native._assistant_wire_owner(world.adapter)
        assert len(ticks) == 8 and ticks == [10] * 8 and actual_clock == boundary
        assert owner.windows == 3 and owner.probe.selection is world.selection
        assert owner.probe.binding.main_route.expires_at == 100 > actual_clock
        assert len(world.qualification_records) == 8
        assert_retained(world, source)
        assert world.adapter.tool_call_assistant_sources[0] is source
        if boundary == 50:
            assert not returned and not owner.records and owner.phase == "denied"
            assert world.adapter.tool_call_assistant_wire_results == ()
            assert not any(route.path == non_main and actual_clock < route.expires_at for route in world.routes)
        else:
            assert len(returned) == 1 and returned[0] is owner.records[0][0]
            assert world.adapter.tool_call_assistant_wire_results[0] is returned[0]
            assert all(any(route.path == path and actual_clock < route.expires_at for route in world.routes)
                       for path in native.shared_policy().PATHS)
    finally:
        world.clock_hook = None
        world.now = 10
        retained = proof(world)
        retained.update(boundary=boundary, actual_clock=actual_clock, clock_callbacks=len(ticks),
            clock_values=ticks, selected_main_expiry=world.adapter._main_route.expires_at,
            route_deadlines=[dict(path=route.path, expires_at=route.expires_at) for route in world.routes],
            returned_results=len(returned), released_results=len(native._assistant_wire_owner(world.adapter).records))
        record_property("final_coverage_proof", json.dumps(retained))
