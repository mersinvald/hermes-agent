"""Capture the first typed controlled response and actual live assistant return."""

import copy
import hashlib
import json
import threading
import time
from contextvars import copy_context
from dataclasses import FrozenInstanceError, replace

import pytest

from agent import native_main_payload as native
from agent.content_policy import NativeContentPolicyDenied, content_policy_scope
from tests.tools.test_native_tool_call_probe import (
    ProbeWorld, vectors, invented_offline_configuration as invented_offline_configuration,
    excluded as excluded, retain_case_durations as retain_case_durations,
    retain_incremental_case_receipts as retain_incremental_case_receipts,
    record_case_start as record_case_start,
)


@pytest.fixture
def source_world(excluded, record_property):
    start = time.monotonic()
    world = ProbeWorld().construct()
    world.source_timings = dict(setup=time.monotonic() - start)
    yield world
    owner = native._assistant_source_owner(world.adapter)
    if owner is not None:
        record_property("partial_source_proof", json.dumps(receipt(world, owner)))


def enable(world):
    start = time.monotonic()
    world.bind_probe()
    world.adapter.enable_tool_call_assistant_source(world.agent, world.token)
    world.source_timings["bind_enable"] = time.monotonic() - start
    return native._assistant_source_owner(world.adapter)


def receipt(world, owner):
    runtime, probe = world.adapter._runtime, native._tool_call_probes[world.adapter]
    stages = dict(A=probe.binding.authorization_bytes, Q=probe.request_bytes,
                  W=probe.wire_bytes, R=probe.response_bytes, O=probe.observation_bytes,
                  Ba=owner.authorization_bytes, Da=owner.message_bytes)
    return dict(timings=world.source_timings, stages={key: None if data is None else dict(
        utf8=data.decode(), hex=data.hex(), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        for key, data in stages.items()},
        retained_bytes=runtime._retained_bytes, contents=len(runtime._contents), envelopes=len(runtime._envelopes),
        qualification_sends=len(world.qualification_records), probe_sends=world.adapter.send_count,
        capture_sends=0, original_windows=probe.stage_index, capture_windows=owner.windows,
        backing_before_present="_base_url" in dict(owner.before_anchors),
        backing_completed=owner.completed_anchors is not None,
        backing_identity=vars(world.agent).get("_base_url") is owner.backing_value,
        cache_before=owner.cache_before, cache_after=owner.cache, cache_committed=owner.cache_committed,
        cache_identity=vars(world.agent).get("_thinking_pad_cache") is owner.cache,
        response_identity=owner.response.choices[0].message is owner.assistant_message,
        message_identity=owner.raw_slot[0] is owner.message,
        source_released=owner.captured and probe.phase == "released")


def test_actual_live_assistant_source(source_world, record_property, monkeypatch):
    world = source_world
    adapter = world.adapter
    with content_policy_scope(world.context):
        world.bind_probe()
        before = tuple(vars(adapter).items())
        original_count = len(native._assistant_source_owners)
        copied = object.__new__(native.NativeToolCallProbeInput)
        world.admission_hook = lambda: pytest.fail("Copied token reached a callback")
        with pytest.raises(NativeContentPolicyDenied, match="token_invalid"):
            adapter.enable_tool_call_assistant_source(world.agent, copied)
        assert len(native._assistant_source_owners) == original_count
        assert not adapter._invalid
        world.admission_hook = None
        start = time.monotonic()
        adapter.enable_tool_call_assistant_source(world.agent, world.token)
        world.source_timings["enable"] = time.monotonic() - start
        owner = native._assistant_source_owner(adapter)
        assert all(vars(adapter)[name] is value for name, value in before)
        assert len(vars(adapter)) == len(before)
        with pytest.raises(NativeContentPolicyDenied, match="attempt_spent"):
            adapter.enable_tool_call_assistant_source(world.agent, world.token)
        with pytest.raises(NativeContentPolicyDenied, match="window_caller_invalid"):
            with owner.window():
                pytest.fail("An external caller obtained a source window")
        rejected = []
        def reenter():
            world.admission_hook = None
            with pytest.raises(NativeContentPolicyDenied, match="reentry_or_thread"):
                _ = adapter.tool_call_assistant_sources
            rejected.append(True)
        world.admission_hook = reenter
        capture_marks = []
        def clock_mark():
            if owner.capture_frame is not None and not capture_marks:
                capture_marks.append(time.monotonic())
            elif owner.captured and owner.capture_frame is None and len(capture_marks) == 1:
                capture_marks.append(time.monotonic())
                world.clock_hook = None
        world.clock_hook = clock_mark
        start = time.monotonic()
        observation = world.run()
        world.source_timings["original_probe_and_capture"] = time.monotonic() - start
        assert len(capture_marks) == 2
        world.source_timings["capture_between_policy_callbacks"] = capture_marks[1] - capture_marks[0]
        world.source_timings["probe_remainder"] = world.source_timings["original_probe_and_capture"] - (capture_marks[1] - capture_marks[0])
        world.binding = replace(world.binding, profiles=())
    assert rejected == [True]
    record, = adapter.tool_call_assistant_sources
    assert type(record) is native.NativeToolCallAssistantSource
    assert record is owner.record and record.source_observation is observation
    assert record.response is owner.response and record.assistant_message is owner.response.choices[0].message
    assert record.finish_reason is owner.response.choices[0].finish_reason
    assert record.message is owner.raw_slot[0] and record.message_bytes is owner.message_bytes
    assert record.message_bytes == json.dumps(record.message, ensure_ascii=False, allow_nan=False,
                                             separators=(",", ":")).encode("utf-8")
    assert list(record.message) == ["role", "content", "reasoning", "finish_reason", "timestamp", "tool_calls"]
    call = record.message["tool_calls"][0]
    typed_call = record.assistant_message.tool_calls[0]
    assert call["function"]["name"] is typed_call.function.name
    assert call["function"]["arguments"] is typed_call.function.arguments
    assert call["id"] == call["call_id"] == observation.call_id
    assert call["response_item_id"] == "fc_invented-call-1"
    assert type(record.message["timestamp"]) is float
    assert not record.message_bytes.endswith(b"\n")
    assert owner.cache_before == (False, None)
    assert "_base_url" not in dict(owner.before_anchors)
    assert owner.completed_anchors[-1] == ("_base_url", "")
    assert owner.completed_anchors[-1][1] is dict(owner.before_anchors)["base_url"]
    assert owner.completed_anchors is not owner.before_anchors and owner.completed_state is not owner.before_state
    assert all(vars(world.agent)[key] is value for key, value in owner.completed_anchors)
    assert owner.cache == ((world.agent.provider, world.agent.model, world.agent.base_url), False)
    assert owner.constructed.anchors is owner.after_anchors and owner.constructed.state is owner.after_state
    assert owner.completed_anchors is not owner.after_anchors and owner.completed_state is not owner.after_state
    assert all(vars(world.agent)[key] is value for key, value in owner.before_anchors)
    assert vars(world.agent)["_thinking_pad_cache"] is owner.cache
    runtime = adapter._runtime
    assert runtime._content(record.message_content).payload is record.message_bytes
    assert native._same_anchors(runtime._content(record.message_content).parents,
                               (observation.response_content, record.authorization_content))
    assert runtime._content(record.authorization_content).parents == ()
    assert all(row.profiles == ("restricted_documents",) for row in runtime._contents.values())
    assert len(runtime._contents) == 7 and len(runtime._envelopes) == 1
    assert runtime._retained_bytes == 3265 + len(record.authorization_bytes) + len(record.message_bytes)
    fixed = vectors()["stages"]
    for name, payload in dict(A=observation.binding.authorization_bytes, Q=observation.request_bytes,
            W=observation.wire_bytes, R=observation.response_bytes, O=observation.observation_bytes).items():
        assert payload == fixed[name]["utf8"].encode()
    assert adapter.send_count == len(world.ports["main"].sent) == 1
    assert len(world.qualification_records) == 8 and not world.sdk_calls
    assert owner.windows == 3 and native._tool_call_probes[adapter].stage_index == 8
    assert not adapter.invocations and not adapter._completions
    record_property("source_proof", json.dumps(receipt(world, owner)))
    errors = []
    context = copy_context()
    def foreign():
        try:
            context.run(lambda: adapter.tool_call_assistant_sources)
        except NativeContentPolicyDenied as error:
            errors.append(error)
    thread = threading.Thread(target=foreign)
    thread.start()
    thread.join()
    assert len(errors) == 1
    with pytest.raises(FrozenInstanceError):
        record.message_bytes = b"copy"
    for obj, name, replacement in (
        (owner.response.choices[0], "finish_reason", "stop"),
        (world.agent, "model", "changed"),
        (world.agent, "_thinking_pad_cache", ((world.agent.provider, world.agent.model, world.agent.base_url), False)),
        (type(world.agent), "_build_assistant_message", lambda *args: pytest.fail("Replacement ran")),
        (owner, "binding", replace(owner.binding)),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(obj, name, replacement)
            with pytest.raises(NativeContentPolicyDenied):
                _ = adapter.tool_call_assistant_sources
        assert adapter.tool_call_assistant_sources[0] is record
    original = call["function"]["arguments"]
    call["function"]["arguments"] = "{}"
    with pytest.raises(NativeContentPolicyDenied):
        _ = adapter.tool_call_assistant_sources
    call["function"]["arguments"] = original
    assert adapter.tool_call_assistant_sources[0] is record
    world.agent.close()
    assert adapter.tool_call_assistant_sources[0] is record
    with pytest.raises(NativeContentPolicyDenied, match="agent_invalid"):
        with content_policy_scope(world.context):
            adapter.enable_tool_call_assistant_source(world.agent, world.token)


def assert_unpublished(world, owner):
    assert world.adapter._invalid and world.context._denied
    assert native._tool_call_probes[world.adapter].phase == "denied"
    try:
        assert world.adapter.tool_call_assistant_sources == ()
    except NativeContentPolicyDenied:
        pass
    runtime = world.adapter._runtime
    assert owner.raw_slot[0] is owner.message and owner.message_bytes is not None
    assert owner.raw_slot[2] is vars(world.agent)["_thinking_pad_cache"] is owner.cache
    assert owner.cache_committed and owner.constructed.anchors is owner.after_anchors
    assert owner.message_content is not None
    assert runtime._content(owner.message_content).payload is owner.message_bytes
    assert native._same_anchors(runtime._content(owner.message_content).parents,
                               (owner.observation.response_content, owner.authorization_content))
    assert len(runtime._contents) == 7 and len(runtime._envelopes) == 1
    assert runtime._retained_bytes == 3265 + len(owner.authorization_bytes) + len(owner.message_bytes)
    assert world.adapter.send_count == len(world.ports["main"].sent) == 1
    assert not owner.captured and owner.record is None
    assert not world.sdk_calls


def test_selected_expiry_after_actual_output(excluded, record_property):
    world = ProbeWorld()
    world.routes = tuple(replace(route, expires_at=200) if route.path != "main" else route
                         for route in world.routes)
    world.routes = (*world.routes, replace(world.routes[0], route_ref="alternate_main", expires_at=200))
    world.profiles = (replace(world.profiles[0], allowed_routes=tuple(route.route_ref for route in world.routes)),)
    start = time.monotonic()
    world.construct()
    world.source_timings = dict(setup=time.monotonic() - start)
    owner = None
    try:
        with pytest.raises(NativeContentPolicyDenied, match="selection_expired"):
            with content_policy_scope(world.context):
                owner = enable(world)
                def expire():
                    if owner.message_content is not None:
                        world.now = 100
                world.clock_hook = expire
                start = time.monotonic()
                try:
                    world.run()
                finally:
                    world.source_timings["failed_probe_and_capture"] = time.monotonic() - start
        assert_unpublished(world, owner)
        assert len(world.qualification_records) == 9
        assert owner.probe.stage_index == 5 and owner.windows == 3
        assert any(route.route_ref == "alternate_main" and route.expires_at > world.now for route in world.routes)
        assert all(any(route.path == path and route.expires_at > world.now for route in world.routes)
                   for path in native.shared_policy().PATHS)
    finally:
        if owner is not None:
            record_property("partial_source_proof", json.dumps(receipt(world, owner)))


def test_original_control_after_actual_output(source_world):
    import asyncio
    world = source_world
    original = asyncio.CancelledError("Invented source control")
    with pytest.raises(asyncio.CancelledError) as observed:
        with content_policy_scope(world.context):
            owner = enable(world)
            def stop():
                if owner.message_content is not None:
                    raise original
            world.clock_hook = stop
            start = time.monotonic()
            try:
                world.run()
            finally:
                world.source_timings["failed_probe_and_capture"] = time.monotonic() - start
    assert observed.value is original and owner.probe.control_exception is original
    assert_unpublished(world, owner)
    assert len(world.qualification_records) == 8
    assert owner.windows == 3 and owner.probe.stage_index == 5
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            world.adapter.enable_tool_call_assistant_source(world.agent, world.token)
    assert native._assistant_source_owner(world.adapter) is owner


def test_preflight_refusals_preserve_unused_original_token(source_world, monkeypatch):
    world = source_world
    class Hostile:
        def __eq__(self, other):
            pytest.fail("Token equality ran")
        def __str__(self):
            pytest.fail("Token conversion ran")
    with content_policy_scope(world.context):
        world.bind_probe()
        probe = native._tool_call_probes[world.adapter]
        original = probe.binding
        count = len(native._assistant_source_owners)
        snapshot = tuple(vars(world.agent).items())
        world.admission_hook = lambda: pytest.fail("Preflight reached an operator callback")
        for agent, token in ((world.agent, Hostile()), (object(), world.token)):
            with pytest.raises(NativeContentPolicyDenied):
                world.adapter.enable_tool_call_assistant_source(agent, token)
        foreign = object.__new__(native.NativeMainPayloadAdapter)
        with pytest.raises(NativeContentPolicyDenied):
            foreign.enable_tool_call_assistant_source(world.agent, world.token)
        errors = []
        context = copy_context()
        def wrong_thread():
            try:
                context.run(world.adapter.enable_tool_call_assistant_source, world.agent, world.token)
            except NativeContentPolicyDenied as error:
                errors.append(error)
        thread = threading.Thread(target=wrong_thread)
        thread.start()
        thread.join()
        assert len(errors) == 1
        for name, replacement in (("attempted", True), ("phase", "released"), ("binding", replace(original))):
            with monkeypatch.context() as patch:
                patch.setattr(probe, name, replacement)
                with pytest.raises(NativeContentPolicyDenied):
                    world.adapter.enable_tool_call_assistant_source(world.agent, world.token)
        with monkeypatch.context() as patch:
            patch.setattr(world.context, "_denied", True)
            with pytest.raises(NativeContentPolicyDenied):
                world.adapter.enable_tool_call_assistant_source(world.agent, world.token)
        assert native._assistant_source_owner(world.adapter) is None
        assert len(native._assistant_source_owners) == count and probe.binding is original
        assert not probe.attempted and probe.stage_index == 1 and world.adapter.send_count == 0
        assert native._same_anchors(tuple(value for pair in vars(world.agent).items() for value in pair),
                                    tuple(value for pair in snapshot for value in pair))
        assert world.adapter.tool_call_assistant_sources == ()
        world.admission_hook = None
