"""Retain one actual converter list through the original formatter authority."""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from agent import message_metadata
from agent.content_policy import NativeContentPolicyDenied
from agent.native_main_payload import (
    NativeIssuedCallWireResult, _issued_call_formats, _issued_call_wires,
    _issued_format_bytes, _issued_wire_shape,
)
from agent.transports.chat_completions import ChatCompletionsTransport
from tests.tools.test_native_issued_call_format import (
    CompactFormatWorld, terminal_state,
    invented_offline_configuration as invented_offline_configuration, excluded as excluded,
    no_delegation_effects as no_delegation_effects, retain_case_durations as retain_case_durations,
    retain_incremental_case_receipts as retain_incremental_case_receipts, record_case_start as record_case_start,
    retain_failure_contexts as retain_failure_contexts,
)


@pytest.fixture
def record_property(record_property, request):
    from pathlib import Path
    def retain(name, value):
        record_property(name, value)
        xml = request.config.option.xmlpath
        if xml is not None:
            with Path(xml).with_suffix(".properties.jsonl").open("a") as output:
                output.write(json.dumps(dict(nodeid=request.node.nodeid, property=name, value=value)) + "\n")
    return retain


def charges(runtime):
    return len(runtime._contents), len(runtime._envelopes), runtime._retained_bytes


def retain_proof(world, before, original, record_property):
    owner, runtime = _issued_call_wires[world.adapter], world.adapter._runtime
    f = _issued_call_formats[world.adapter]
    raw, anchors, payload = owner.partial_results[0]
    assert raw is owner.messages and raw[0] is not original.message
    assert anchors is owner.message_anchors and payload is owner.message_bytes
    assert raw is not owner.inputs and owner.inputs[0] is original.message
    assert owner.inputs == [original.message] and tuple(raw[0]) == ("role", "name", "content", "tool_call_id")
    assert all(raw[0][key] is original.message[key] for key in raw[0])
    assert payload == _issued_format_bytes(raw)
    assert len(payload) == 415 and hashlib.sha256(payload).hexdigest() == (
        "01c141a03033fe3920d4434eef50e21841b17aad96b92b5c5dabb1c66364c5b8")
    assert len(original.message_bytes) == 466 and hashlib.sha256(original.message_bytes).hexdigest() == (
        "9abee50a0dcd82d70714e9f9ceb06f0162620ec8da7534d7d70f6d27581a8efb")
    assert original.message_bytes == _issued_format_bytes(original.message)
    assert original.message["timestamp"] == 1700000000.0 and "timestamp" not in raw[0]
    assert owner.helper.__class__ is ChatCompletionsTransport
    assert owner.converter[0] is owner.helper and owner.converter[1] is ChatCompletionsTransport.convert_messages
    assert owner.model is original.binding.selection.destination.model_ref and owner.model == "invented_model_4"
    assert charges(runtime) == (before[0] + 1, before[1], before[2] + 415)
    assert charges(runtime) == (27, 3, 25005)
    wf = runtime._content(owner.message_content)
    assert wf.payload is payload and wf.parents == (original.message_content,)
    assert wf.path == "tool_result" and wf.profiles == ("restricted_documents",)
    assert runtime._content(original.message_content).payload is original.message_bytes
    assert f.records[0][0] is original and f.stage_index == 4 and f.phase == "released"
    assert world.wall_calls == 1 and owner.stage_index == 3
    assert len(world.routes) == len(world.qualifications) == 19 and len(world.requests) == 5
    assert len(dict.fromkeys(world.ports.values())) == 5
    assert sum(len(port.sent) for port in dict.fromkeys(world.ports.values())) == 3
    indices = {row.handle: index + 1 for index, row in enumerate(runtime._contents.values())}
    record_property("wire_proof", json.dumps(dict(before=before, after=charges(runtime),
        raw_messages=raw, raw_bytes=payload.decode("utf-8"), original_bytes=original.message_bytes.decode("utf-8"),
        qualification_sends=len(world.qualifications), data_sends=3, wire_sends=0,
        released=len(owner.records), runtime_poisoned=runtime._poisoned, controls=owner.control_exception is not None,
        windows=dict(source=8, execution=8, formatter=f.stage_index, wire=owner.stage_index),
        stages=[dict(bytes=len(row.payload), payload=row.payload.decode("utf-8"),
            parents=[indices[parent] for parent in row.parents], path=row.path, binding=row.binding,
            profiles=row.profiles) for row in runtime._contents.values()]), sort_keys=True))


@pytest.mark.parametrize("mode", ["success", "selected_expiry"])
def test_actual_wire_projection(no_delegation_effects, monkeypatch, record_property, mode):
    world = CompactFormatWorld()
    if mode == "selected_expiry":
        world.routes = tuple(replace(route, expires_at=100 if route.route_ref == "route_4_main" else 200)
            for route in world.routes)
    monkeypatch.setattr(message_metadata, "wall_time", world.wall_time)
    started = time.monotonic()
    world.source(record_property)
    record_property("source_seconds", time.monotonic() - started)
    formatter_started = time.monotonic()
    token = world.format_bind()
    world.bindings = tuple(replace(row, profiles=()) for row in world.bindings)
    original = world.adapter.run_issued_call_format(token)
    record_property("formatter_seconds", time.monotonic() - formatter_started)
    runtime, source_terminal = world.adapter._runtime, terminal_state(world)
    before = charges(runtime)
    assert world.adapter.issued_call_wire_results == ()
    original_namespace = tuple(vars(world.adapter).items())
    copied = replace(original)
    assert copied == original and copied is not original
    with pytest.raises(NativeContentPolicyDenied, match="source_result_changed"):
        world.adapter.retain_issued_call_wire(copied)
    assert world.adapter not in _issued_call_wires and charges(runtime) == before
    callback_refusals, boundaries = [], []
    def inspect_clock():
        owner = _issued_call_wires.get(world.adapter)
        if owner is None or owner.active_call is None:
            return
        assert owner.depth == 1
        if owner.stage_index not in boundaries:
            boundaries.append(owner.stage_index)
            record_property("wire_boundary_" + str(owner.stage_index), json.dumps(dict(
                elapsed=time.monotonic() - wire_started, charges=charges(runtime),
                raw_messages=owner.messages, raw_bytes=None if owner.message_bytes is None
                    else owner.message_bytes.decode("utf-8"), content_retained=owner.message_content is not None)))
        if mode == "success" and not callback_refusals:
            with pytest.raises(NativeContentPolicyDenied, match="reentry_or_thread") as caught:
                _ = world.adapter.issued_call_wire_results
            callback_refusals.append(caught.value)
        if mode == "selected_expiry" and owner.stage_index == 3:
            assert owner.message_content is not None
            world.now = 100
    world.hooks["clock"] = inspect_clock
    wire_started = time.monotonic()
    if mode == "success":
        result = world.adapter.retain_issued_call_wire(original)
        assert type(result) is NativeIssuedCallWireResult
        assert result is world.adapter.issued_call_wire_results[0]
        assert result.source_result is original and result.messages is _issued_call_wires[world.adapter].messages
        assert callback_refusals
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(NativeContentPolicyDenied, match="reentry_or_thread"):
                pool.submit(lambda: world.adapter.issued_call_wire_results).result()
        before_replay = charges(runtime)
        with pytest.raises(NativeContentPolicyDenied, match="attempt_spent"):
            world.adapter.retain_issued_call_wire(original)
        assert charges(runtime) == before_replay and world.adapter.issued_call_wire_results[0] is result
    else:
        with pytest.raises(NativeContentPolicyDenied, match="selected_expired_or_denied"):
            world.adapter.retain_issued_call_wire(original)
        assert world.adapter.issued_call_wire_results == ()
        f = _issued_call_formats[world.adapter]
        assert f.route.route_ref == "route_4_main" and f.route.expires_at == world.now
        assert all(route is f.route or route.expires_at > world.now for route in world.routes)
        assert any(route.path == "main" and route.selection_id == f.route.selection_id
            and route.selection_version == f.route.selection_version and route.expires_at > world.now
            for route in world.routes)
    record_property("wire_seconds", time.monotonic() - wire_started)
    owner = _issued_call_wires[world.adapter]
    assert owner.phase == ("released" if mode == "success" else "denied")
    assert owner.active_call is owner.entry is owner.public is owner.stage_frame is None and owner.depth == 0
    assert terminal_state(world) == source_terminal
    assert tuple(vars(world.adapter)) == tuple(name for name, _ in original_namespace)
    assert all(vars(world.adapter)[name] is value for name, value in original_namespace)
    retain_proof(world, before, original, record_property)


@pytest.mark.parametrize("content", ["e\u0301\u00e9雪😀\x00\n", "x" * 16384], ids=["scalars", "character_limit"])
def test_conversion_preserves_scalar_bytes(content):
    original = dict(role="tool", name="delegate_task", tool_name="delegate_task", content=content,
        tool_call_id="invented-call-1", timestamp=1700000000.0)
    inputs = [original]
    output = ChatCompletionsTransport().convert_messages(inputs, model="invented_model_4")
    _issued_wire_shape(output, original)
    payload = _issued_format_bytes(output)
    assert output is not inputs and output[0] is not original and output[0]["content"] is content
    assert json.loads(payload)[0]["content"] == content and not payload.startswith(b"\xef\xbb\xbf")
    assert not payload.endswith(b"\n") and b'"tool_name"' not in payload and b'"timestamp"' not in payload
    assert tuple(original) == ("role", "name", "tool_name", "content", "tool_call_id", "timestamp")
    if len(content) < 100:
        assert "e\u0301\u00e9雪😀".encode("utf-8") in payload


@pytest.mark.parametrize("content", ["\ud800", "x" * 16385, "雪" * 5462], ids=["surrogate", "characters", "utf8"])
def test_wire_shape_refuses_non_scalar_or_oversize_strings(content):
    original = dict(role="tool", name="delegate_task", content=content, tool_call_id="invented-call-1")
    with pytest.raises(NativeContentPolicyDenied):
        _issued_wire_shape([dict(original)], original)


def test_strict_serialization_refuses_nonfinite_and_surrogate():
    with pytest.raises(ValueError):
        _issued_format_bytes([dict(content=float("nan"))])
    with pytest.raises(UnicodeEncodeError):
        _issued_format_bytes([dict(content="\ud800")])
