"""Retain the actual formatter mapping from one genuine terminal issued call."""

import json
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
from hermes_content_policy import Destination, ExecutionBinding, PATHS, ProfilePolicy, RouteBinding

from agent import message_metadata
from agent.content_policy import NativeContentPolicyDenied, content_policy_scope
from agent.native_main_payload import (
    NativeMainSelection, NativeResultPolicy, NativeIssuedCallFormatInput, NativeIssuedCallFormatResult,
    _issued_call_formats, _issued_call_frames, _tool_call_probes,
)
from tests.agent.test_native_main_payload import ControlledPort
from tests.tools.test_native_issued_call_execution import (
    CompactWorld, invented_offline_configuration as invented_offline_configuration, excluded as excluded,
    no_delegation_effects as no_delegation_effects, retain_case_durations as retain_case_durations,
    retain_incremental_case_receipts as retain_incremental_case_receipts, record_case_start as record_case_start,
    retain_failure_contexts as retain_failure_contexts,
)


class FormatWorld(CompactWorld):
    def __init__(self):
        super().__init__()
        self.requests += (object(),)
        f = ExecutionBinding("owner", "execution", "lineage", "document_specialist", 7, 9,
            "selection_4", 6, ("restricted_documents",))
        self.bindings += (f,)
        destination = Destination("invented_4", "synthetic_native", "invented_model_4", "invented_endpoint_4", 1)
        port = ControlledPort(destination)
        port.output = b'{"accepted":true}'
        for path in PATHS:
            self.ports[4, path] = port
            self.routes += (RouteBinding(f"route_4_{path}", f.selection_id, f.selection_version,
                f.consumer_ref, path, destination, 100, f"synthetic_4_{path}", "synthetic_only"),)
        self.selections.append(NativeMainSelection(f.selection_id, f.selection_version, "chat_completions", destination))
        self.selection_ports += (SimpleNamespace(current=self.selection),)
        self.policies += (NativeResultPolicy("producer_4", 4, ("restricted_documents",)),)
        self.transports += ((destination.transport_ref, port),)
        self.profiles = (ProfilePolicy("restricted_documents", tuple(r.route_ref for r in self.routes), PATHS),)
        self.wall_calls = 0
        self.on_wall = None

    def wall_time(self):
        self.wall_calls += 1
        if self.on_wall is not None:
            self.on_wall()
        return 1700000000.0

    def source(self, record_property):
        self.construct()
        self.agent.close()
        self.bind()
        self.bindings = tuple(replace(row, profiles=()) if i < 4 else row for i, row in enumerate(self.bindings))
        self.source_result = self.run()
        runtime = self.adapter._runtime
        indices = {row.handle: i + 1 for i, row in enumerate(runtime._contents.values())}
        record_property("source_proof", json.dumps(dict(qualification_sends=len(self.qualifications),
            source_sends=1, delegate_sends=1, child_sends=1, execution_main_sends=0,
            content_slots=len(runtime._contents), envelopes=len(runtime._envelopes), retained_bytes=runtime._retained_bytes,
            stages=[dict(bytes=len(row.payload), parents=[indices[p] for p in row.parents], path=row.path,
                binding=row.binding, profiles=row.profiles) for row in runtime._contents.values()],
            payloads=[row.payload.decode("utf-8") for row in runtime._contents.values()]), sort_keys=True))
        return self

    def format_bind(self, result=None, **changes):
        options = dict(formatter_request=self.requests[4], formatter_selection=self.selection_ports[4],
            formatter_producer_ref="producer_4", formatter_main_route_ref="route_4_main")
        options.update(changes)
        self.format_token = self.adapter.bind_issued_call_format(self.source_result if result is None else result, **options)
        return self.format_token


class CompactFormatWorld(FormatWorld):
    def __init__(self):
        super().__init__()
        f = replace(self.bindings[4], selection_id=self.bindings[1].selection_id,
            selection_version=self.bindings[1].selection_version)
        self.bindings = (*self.bindings[:4], f)
        self.selections[4] = replace(self.selections[4], selection_id=f.selection_id,
            selection_version=f.selection_version)
        self.routes = tuple(replace(route, selection_id=f.selection_id, selection_version=f.selection_version)
            if route.route_ref == "route_4_main" else route for route in self.routes
            if not route.route_ref.startswith("route_4_") or route.path == "main")
        self.profiles = (ProfilePolicy("restricted_documents", tuple(route.route_ref for route in self.routes), PATHS),)


def terminal_state(world):
    p, e = _tool_call_probes[world.adapter], _issued_call_frames[world.adapter]
    return (p.phase, p.records, p.record_anchors, p.binding, p.closed, p.stage_index,
        p.constructed.closed, p.constructed.state, p.constructed.anchors, e.phase, e.result_anchors,
        e.binding, e.stage_index, e.adapter._constructed.state, e.adapter._constructed.anchors,
        e.frame.phase, e.frame.closed_proof, e.frame.child_adapter._constructed.closed,
        world.adapter.send_count, e.adapter.send_count, e.frame.delegate_sends, e.frame.child_adapter.send_count)


def test_actual_formatter_retains_original_mapping_and_graph(no_delegation_effects, monkeypatch, record_property):
    world = FormatWorld()
    monkeypatch.setattr(message_metadata, "wall_time", world.wall_time)
    started = time.monotonic()
    world.source(record_property)
    record_property("source_seconds", time.monotonic() - started)
    original = terminal_state(world)
    runtime = world.adapter._runtime
    before = len(runtime._contents), len(runtime._envelopes), runtime._retained_bytes
    token = world.format_bind()
    f = _issued_call_formats[world.adapter]
    world.bindings = tuple(replace(row, profiles=()) for row in world.bindings)
    def inspect_during_format():
        with pytest.raises(NativeContentPolicyDenied, match="reentry_or_thread"):
            _ = world.adapter.issued_call_format_results
    world.on_wall = inspect_during_format
    result = world.adapter.run_issued_call_format(token)
    assert type(token) is NativeIssuedCallFormatInput and type(result) is NativeIssuedCallFormatResult
    assert result is world.adapter.issued_call_format_results[0] is f.records[0][0]
    assert result.message is f.partial_results[0][0] is f.message
    assert result.message_bytes is f.partial_results[0][2]
    assert result.message["content"] is world.source_result.aggregate
    assert result.message["name"] is result.message["tool_name"] is world.observation.function_name
    assert result.message["tool_call_id"] is world.observation.call_id
    assert tuple(result.message) == ("role", "name", "tool_name", "content", "tool_call_id", "timestamp")
    assert type(result.message["timestamp"]) is float and result.message["timestamp"] == 1700000000.0
    assert world.wall_calls == 1
    assert result.message_bytes == json.dumps(result.message, ensure_ascii=False, allow_nan=False,
        separators=(",", ":")).encode("utf-8")
    assert not result.message_bytes.endswith(b"\n")
    assert terminal_state(world) == original
    assert f.stage_index == 4 and f.phase == "released" and f.active_call is None
    assert len(runtime._contents) - before[0] == 3 and len(runtime._envelopes) == before[1]
    assert len(world.qualifications) == len(world.routes) == 26
    assert sum(len(p.sent) for p in dict.fromkeys(world.ports.values())) == 3
    assert world.ports[4, "main"].sent == []
    b = result.binding
    for handle, payload, parents in (
        (b.authorization_content, b.authorization_bytes, ()),
        (b.input_content, b.input_bytes, (world.source_result.result_content, b.authorization_content)),
        (result.message_content, result.message_bytes, (b.input_content,)),
    ):
        row = runtime._content(handle)
        assert row.payload is payload and row.parents == parents
        assert row.profiles == ("restricted_documents",)
    assert runtime._content(b.authorization_content).binding[8] == ("restricted_documents",)
    assert all(row.profiles == ("restricted_documents",) for row in runtime._contents.values())
    indices = {row.handle: i + 1 for i, row in enumerate(runtime._contents.values())}
    record_property("positive_proof", json.dumps(dict(source_seconds=time.monotonic() - started,
        before=before, qualification_sends=len(world.qualifications), source_sends=1, delegate_sends=1,
        child_sends=1, execution_main_sends=0, formatter_main_sends=0, content_slots=len(runtime._contents),
        envelopes=len(runtime._envelopes), retained_bytes=runtime._retained_bytes,
        new_bytes=sum(map(len, (b.authorization_bytes, b.input_bytes, result.message_bytes))),
        windows=dict(source=8, execution=8, formatter=f.stage_index),
        stages=[dict(bytes=len(row.payload), parents=[indices[p] for p in row.parents], path=row.path,
            binding=row.binding, profiles=row.profiles) for row in runtime._contents.values()],
        payloads=[row.payload.decode("utf-8") for row in runtime._contents.values()]), sort_keys=True))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(lambda: world.adapter.issued_call_format_results)
        with pytest.raises(NativeContentPolicyDenied, match="reentry_or_thread"):
            future.result()
    before_replay = len(runtime._contents), len(runtime._envelopes), runtime._retained_bytes
    with pytest.raises(NativeContentPolicyDenied, match="attempt_invalid"):
        world.adapter.run_issued_call_format(token)
    assert (len(runtime._contents), len(runtime._envelopes), runtime._retained_bytes) == before_replay
    assert world.adapter.issued_call_format_results[0] is result
    assert terminal_state(world) == original


def retain_partial_proof(world, before, record_property):
    f, runtime = _issued_call_formats[world.adapter], world.adapter._runtime
    assert f.phase == "denied" and world.adapter.issued_call_format_results == ()
    assert sum(len(p.sent) for p in dict.fromkeys(world.ports.values())) == 3
    assert world.wall_calls == 1 and len(f.partial_results) == 1
    message, anchors, payload = f.partial_results[0]
    assert type(message) is dict and message["content"] is world.source_result.aggregate
    if payload is not None:
        assert payload == json.dumps(message, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    indices = {row.handle: i + 1 for i, row in enumerate(runtime._contents.values())}
    record_property("partial_proof", json.dumps(dict(before=before, content_slots=len(runtime._contents),
        envelopes=len(runtime._envelopes), retained_bytes=runtime._retained_bytes,
        added_slots=len(runtime._contents) - before[0], added_bytes=runtime._retained_bytes - before[2],
        raw_message=message, raw_bytes=None if payload is None else payload.decode("utf-8"),
        qualification_sends=len(world.qualifications), source_sends=1, delegate_sends=1, child_sends=1,
        execution_main_sends=0, formatter_main_sends=0, windows=f.stage_index,
        stages=[dict(bytes=len(row.payload), parents=[indices[p] for p in row.parents], path=row.path,
            binding=row.binding, profiles=row.profiles) for row in runtime._contents.values()],
        payloads=[row.payload.decode("utf-8") for row in runtime._contents.values()]), sort_keys=True))
    with pytest.raises((NativeContentPolicyDenied, InterruptedError)):
        world.adapter.run_issued_call_format(world.format_token)
    assert len(runtime._contents) - before[0] in (2, 3)
    return f


def test_copied_source_result_refuses_before_admission(no_delegation_effects, monkeypatch, record_property):
    world = CompactFormatWorld()
    monkeypatch.setattr(message_metadata, "wall_time", world.wall_time)
    world.source(record_property)
    original, runtime = terminal_state(world), world.adapter._runtime
    before = len(runtime._contents), len(runtime._envelopes), runtime._retained_bytes
    copied = replace(world.source_result)
    assert copied == world.source_result and copied is not world.source_result
    with pytest.raises(NativeContentPolicyDenied, match="source_result_changed"):
        world.format_bind(copied)
    assert (len(runtime._contents), len(runtime._envelopes), runtime._retained_bytes) == before
    assert terminal_state(world) == original and world.wall_calls == 0
    with pytest.raises(NativeContentPolicyDenied, match="attempt_spent"):
        world.format_bind()
    assert world.adapter.issued_call_results[0] is world.source_result


def test_initial_formatter_profiles_cover_retained_source_call(no_delegation_effects, monkeypatch, record_property):
    world = CompactFormatWorld()
    monkeypatch.setattr(message_metadata, "wall_time", world.wall_time)
    world.source(record_property)
    world.bindings = (*world.bindings[:4], replace(world.bindings[4], profiles=()))
    world.policies = (*world.policies[:4], replace(world.policies[4], profiles=()))
    original, runtime = terminal_state(world), world.adapter._runtime
    before = len(runtime._contents), len(runtime._envelopes), runtime._retained_bytes
    with pytest.raises(NativeContentPolicyDenied, match="source_profiles_not_authorized"):
        world.format_bind()
    assert (len(runtime._contents), len(runtime._envelopes), runtime._retained_bytes) == before
    assert terminal_state(world) == original and world.wall_calls == 0
    assert world.adapter.issued_call_format_results == ()


def test_control_callback_keeps_raw_formatter_output(no_delegation_effects, monkeypatch, record_property):
    world = CompactFormatWorld()
    monkeypatch.setattr(message_metadata, "wall_time", world.wall_time)
    world.source(record_property)
    runtime, original = world.adapter._runtime, terminal_state(world)
    before = len(runtime._contents), len(runtime._envelopes), runtime._retained_bytes
    token = world.format_bind()
    error = InterruptedError("Invented formatter policy clock interruption")
    def stop():
        f = _issued_call_formats[world.adapter]
        if f.message is not None and f.stage_index == 3:
            raise error
    world.hooks["clock"] = stop
    with pytest.raises(InterruptedError) as caught:
        world.adapter.run_issued_call_format(token)
    assert caught.value is error
    f = retain_partial_proof(world, before, record_property)
    assert f.control_exception is error and f.message_content is None
    assert len(runtime._contents) - before[0] == 2
    assert terminal_state(world) == original


def test_late_selected_expiry_keeps_actual_message_and_charges(no_delegation_effects, monkeypatch, record_property):
    world = CompactFormatWorld()
    world.routes = tuple(replace(route, expires_at=100 if route.route_ref == "route_0_main" else 200)
        for route in world.routes)
    monkeypatch.setattr(message_metadata, "wall_time", world.wall_time)
    world.source(record_property)
    runtime, original = world.adapter._runtime, terminal_state(world)
    before = len(runtime._contents), len(runtime._envelopes), runtime._retained_bytes
    token = world.format_bind()
    def expire_selected():
        if _issued_call_formats[world.adapter].stage_index == 4:
            world.now = 100
    world.hooks["clock"] = expire_selected
    with pytest.raises(NativeContentPolicyDenied, match="selected_expired_or_denied"):
        world.adapter.run_issued_call_format(token)
    f = retain_partial_proof(world, before, record_property)
    assert f.message_content is not None and len(runtime._contents) - before[0] == 3
    assert terminal_state(world) == original
    assert world.routes[0].expires_at == world.now
    assert any(route.route_ref == "route_1_main" and world.now < route.expires_at
        and route.selection_id == world.routes[0].selection_id for route in world.routes)










def test_ordinary_text_contract_refuses_formatted_tool_rows():
    from agent.native_main_payload import _history, _request
    with pytest.raises(NativeContentPolicyDenied):
        _history([dict(role="tool", name="delegate_task", content="Invented result", tool_call_id="invented-call-1")])
    with pytest.raises(NativeContentPolicyDenied):
        _request(dict(model="invented_model", messages=[dict(role="user", content="Invented input")],
            stream=False, tools=[dict(type="function", function=dict(name="delegate_task"))]))
