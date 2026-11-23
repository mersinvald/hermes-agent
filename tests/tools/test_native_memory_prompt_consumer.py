"""Consume one invented memory result through actual native prompt assembly."""

import asyncio
import copy
import hashlib
import json
import threading
from contextvars import copy_context
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_content_policy import (
    PATHS, MAX_PAYLOAD, MAX_RETAINED, ContentPolicyRuntime, Destination, PolicyError, ProfilePolicy, RouteBinding,
)

from agent import prompt_builder, system_prompt
from agent.content_policy import NativeContentContext, NativeContentPolicyDenied, content_policy_scope, guard_tool_result
from agent.native_main_payload import (
    NativeMainPayloadAdapter, NativeMainSelection, NativeMemoryPromptInput, NativeResultPolicy,
    _memory_prompt_frames, _memory_snapshots,
    bind_memory_prompt_consumer, run_memory_prompt_consumer,
)
from tests.agent.test_native_main_payload import (
    ControlledPort, canonical, invented_offline_configuration as invented_offline_configuration,
)
from tests.agent.test_native_controlled_conversation import excluded as excluded
from tests.tools.test_native_memory_snapshot import Hostile, MemoryWorld, getter_observations
from tests.tools.test_native_controlled_delegation import retain_case_durations as retain_case_durations
from tests.tools.test_native_parent_resumption import (
    retain_incremental_case_receipts as retain_incremental_case_receipts,
    record_case_start as record_case_start,
)


class PromptWorld(MemoryWorld):
    def __init__(self, **options):
        super().__init__(**options)
        self.consumer_request = object()
        self.consumer_binding = replace(self.binding, selection_id="consumer_selection", selection_version=1)
        self.consumer_policy = NativeResultPolicy("invented_consumer_producer", 1, ("restricted_documents",))
        self.consumer_ports = {}
        routes = []
        for path in PATHS:
            destination = Destination(f"invented_consumer_{path}", "synthetic_native",
                "invented_consumer_model", "invented_consumer_endpoint", 1)
            port = self.consumer_ports[path] = ControlledPort(destination)
            self.ports["consumer_" + path] = port
            self.receiver_bindings[id(port)] = (port, replace(self.consumer_binding,
                                                              execution_ref="consumer_receiver_" + path))
            routes.append(RouteBinding(f"consumer_{path}_route", "consumer_selection", 1,
                self.consumer_binding.consumer_ref, path, destination, 100,
                f"synthetic_consumer_{path}", "synthetic_only"))
        self.routes = (*self.routes, *routes)
        self.profiles = (ProfilePolicy("restricted_documents", tuple(r.route_ref for r in self.routes), PATHS),)
        self.consumer_selection = NativeMainSelection("consumer_selection", 1, "chat_completions",
                                                       self.consumer_ports["main"].destination)
        self.consumer_context = NativeContentContext(self.admission, self.consumer_request)
        self.consumer_ports["main"].output = canonical(dict(id="invented-memory-consumer-response",
            object="chat.completion", created=1, model="invented_consumer_model",
            choices=[dict(index=0, finish_reason="stop", message=dict(role="assistant", content="Two counts."))],
            usage=dict(prompt_tokens=5, completion_tokens=2, total_tokens=7)))

    def current(self, handle):
        if handle is getattr(self, "consumer_request", None):
            if self.admission_hook:
                self.admission_hook()
            return self.consumer_binding
        return super().current(handle)

    def selected(self, handle):
        if handle is self.consumer_request:
            if self.selection_hook:
                self.selection_hook()
            return self.consumer_selection
        return super().selected(handle)

    def produced(self, ref):
        if ref == "invented_consumer_producer":
            if self.producer_hook:
                self.producer_hook()
            return self.consumer_policy
        return super().produced(ref)

    def qualify(self):
        runtime = ContentPolicyRuntime(admission=self.admission, profiles=self.profiles, routes=self.routes,
            transports=tuple((p.destination.transport_ref, p) for p in self.ports.values()),
            clock=lambda: self.now, policy_revision=9)
        records = []
        for handle, ports, route in ((self.request, self.ports, "route_{}"),
                                    (self.consumer_request, self.consumer_ports, "consumer_{}_route")):
            for path in PATHS:
                content = runtime.admit(handle, b"Invented declared operation")
                result = runtime.dispatch(handle, content, path=path, route_ref=route.format(path))
                envelope, = ports[path].sent
                records.append((path, ports[path], envelope, result, ports[path].output))
                ports[path].sent = []
        for route in getattr(self, "extra_qualifications", ()):
            handle = self.request if route.selection_id == "selection" else self.consumer_request
            port = next(p for p in self.ports.values() if p.destination is route.destination)
            content = runtime.admit(handle, b"Invented alternative qualification")
            result = runtime.dispatch(handle, content, path=route.path, route_ref=route.route_ref)
            envelope, = port.sent
            records.append((route.path, port, envelope, result, port.output))
            port.sent = []
        self.qualification_records = tuple(records)
        assert self.admission.current(self.consumer_request) is self.consumer_binding

    def bind_consumer(self, result=None, **changes):
        if result is None:
            result = self.capture()
        self.source_result = result
        options = dict(consumer_context=self.consumer_context, selection=self.selection_port,
            result_policy=self.result_port, producer_ref="invented_consumer_producer",
            main_route_ref="consumer_main_route", system_message="Use retained notes.",
            session_id="memory-consumer-session")
        options.update(changes)
        with content_policy_scope(self.context):
            self.consumer_token = bind_memory_prompt_consumer(self.adapter, result, **options)
        self.consumer = _memory_prompt_frames[self.adapter]
        return self

    def run(self):
        return run_memory_prompt_consumer(self.consumer_token, user_message="How many counts?", task_id="execution")


def vectors():
    path = Path(__file__).resolve().parents[2] / "docs/evidence/accountant-native-memory-prompt-consumer-20261123/proposed-vectors.json"
    return json.loads(path.read_text())


def test_actual_constructor_prompt_conversation_send_result(excluded, record_property):
    world = PromptWorld().construct()
    with getter_observations(world.store) as observations:
        source = world.capture()
    world.bind_consumer(source)
    result = world.run()
    f, c, p = world.consumer, world.consumer.adapter, world.adapter
    runtime = p._runtime
    turn, = c._turn_frames
    invocation, = c.invocations
    completed, = c._completions
    assert result is completed.result and len(result) == 29
    assert result["final_response"] == "Two counts."
    assert f.phase == "released" and f.terminal and f.stage_index == 8
    assert _memory_snapshots[p].phase == "released" and not _memory_snapshots[p].closed
    assert p._runtime is c._runtime is f.runtime
    assert all(c._transport_wrappers[n] is w for n, w in p._transport_wrappers.items())
    assert c._clock is p._clock and c._admission is p._admission
    assert f.agent._memory_store is f.agent._memory_manager is world.agent._memory_store is world.agent._memory_manager is None
    assert f.agent._cached_system_prompt_static is f.options[5]
    assert f.agent._cached_system_prompt is f.full_prompt is turn.output.active_system_prompt
    assert world.source_result.block is world.store._system_prompt_snapshot["memory"]
    assert sum(row[0] == "call" for row in observations) == 1
    assert "Unrelated red triangles" not in f.full_prompt
    assert p.send_count == 0 and c.send_count == 1 and len(world.qualification_records) == 16
    assert len(world.consumer_ports["main"].sent) == 1
    assert not p.invocations and not world.sdk_calls
    actual = dict(stable_prompt=f.static_prompt.encode(), getter_block=world.source_result.block.encode(),
        full_system_prompt=f.full_prompt.encode(), authorization_bytes=f.authorization_bytes,
        raw_turn_bytes=turn.raw_bytes, turn_context_bytes=turn.output_bytes,
        prepared_bytes=runtime._content(turn.output_content).payload,
        normalized_request_bytes=turn.request_bytes, wire_request_bytes=invocation.request_bytes,
        response_bytes=invocation.response_bytes, native_result_bytes=completed.payload,
        empty_full_system_prompt=f.options[5].encode())
    fixed = vectors()
    for name, data in actual.items():
        assert data == fixed["stages"][name]["utf8"].encode(), name
        assert len(data) == fixed["stages"][name]["bytes"]
        assert hashlib.sha256(data).hexdigest() == fixed["stages"][name]["sha256"]
    source = world.source_result
    for name, data in (("source", source.binding.source_bytes), ("request", source.binding.request_bytes),
                       ("result", source.result_bytes)):
        assert data == fixed["capture"][name]["utf8"].encode()
    handles = dict(S=source.binding.source_content, Q=source.binding.request_content, M=source.result_content,
        A=f.authorization_content, I=turn.admitted, T=turn.output_content, K=turn.request_contents[0],
        W=invocation.request_content, R=invocation.result_content, N=completed.content)
    for name, parents in fixed["ancestry"].items():
        if name == "I_parent_requests":
            continue
        assert runtime._content(handles[name]).parents == tuple(handles[p] for p in parents)
    assert turn.admitted is turn.input_content
    record_property("consumer_proof", json.dumps(dict(stages={k:v.decode() for k,v in actual.items()},
        qualification_sends=16, capture_sends=0, consumer_sends=1,
        source_unchanged=True, original_runtime=True, original_wrappers=True,
        result_identity=True, retained_bytes=runtime._retained_bytes, slots=len(runtime._contents))))


def denied_consumer(world, sends=0):
    f = _memory_prompt_frames.get(world.adapter)
    if f is not None:
        assert f.phase == "denied" and f.terminal
        if f.adapter is not None:
            assert f.adapter._invalid and f.adapter.send_count == sends
            assert not f.adapter._completions
    assert world.consumer_context._denied
    assert world.adapter.send_count == 0 and not world.adapter.invocations
    assert not world.sdk_calls


def test_authenticated_empty_and_no_discovery_or_warning_drain(excluded, monkeypatch):
    world = PromptWorld(empty=True).construct().bind_consumer()
    monkeypatch.setattr(system_prompt, "_ra", excluded)
    monkeypatch.setattr(system_prompt, "drain_truncation_warnings", excluded)
    from run_agent import AIAgent
    monkeypatch.setattr(AIAgent, "_emit_status", excluded)
    warnings = ["Invented unrelated truncation warning"]
    token = prompt_builder._truncation_warnings.set(warnings)
    try:
        result = world.run()
        assert prompt_builder._truncation_warnings.get() is warnings
        assert warnings == ["Invented unrelated truncation warning"]
    finally:
        prompt_builder._truncation_warnings.reset(token)
    f = world.consumer
    assert f.source.block is None and f.full_prompt == "Use retained notes."
    assert f.agent._cached_system_prompt is f.full_prompt
    assert result["final_response"] == "Two counts."
    assert f.adapter.invocations[0].envelope.payload == f.adapter.invocations[0].request_bytes
    assert len(world.qualification_records) == 16 and world.adapter.send_count == 0 and f.adapter.send_count == 1


def test_copied_record_cannot_spend_original_source(excluded):
    world = PromptWorld().construct()
    source = world.capture()
    owner = _memory_snapshots[world.adapter]
    for candidate in (replace(source), copy.copy(source), source.result_bytes, source.block):
        with pytest.raises(NativeContentPolicyDenied):
            world.bind_consumer(candidate)
        assert not hasattr(owner, "memory_prompt_attempt")
        assert not world.consumer_context._denied
    world.bind_consumer(source)
    assert world.run()["completed"]


@pytest.mark.parametrize("change", ["ports", "holder", "options", "closed", "late-bind-failure"])
def test_single_attempt_including_escaping_failure(excluded, change):
    world = PromptWorld().construct()
    source = world.capture()
    original = InterruptedError("Invented bind stop")
    options = {}
    if change == "ports":
        options["selection"] = SimpleNamespace(current=world.selection_port.current)
    elif change == "holder":
        world.consumer_binding = replace(world.consumer_binding, consumer_ref="unrelated_consumer")
    elif change == "options":
        options["system_message"] = Hostile()
    elif change == "closed":
        world.agent.close()
    else:
        def stop():
            f = _memory_prompt_frames.get(world.adapter)
            if f is not None and f.authorization_content is not None:
                raise original
        world.on_clock = stop
    with pytest.raises((NativeContentPolicyDenied, InterruptedError)) as failure:
        world.bind_consumer(source, **options)
    if change == "late-bind-failure":
        assert failure.value is original
        f = _memory_prompt_frames[world.adapter]
        assert f.authorization_content is not None and f.token is None
        assert f.runtime._content(f.authorization_content).payload is f.authorization_bytes
    world.on_clock = None
    spent = _memory_snapshots[world.adapter].memory_prompt_attempt
    replacement = NativeContentContext(world.admission, world.consumer_request)
    with pytest.raises(NativeContentPolicyDenied):
        world.bind_consumer(source, consumer_context=replacement)
    assert _memory_snapshots[world.adapter].memory_prompt_attempt is spent
    assert world.adapter.send_count == 0


def test_success_spends_source_and_token_once(excluded):
    world = PromptWorld().construct().bind_consumer()
    result = world.run()
    frame = world.consumer
    proof = frame.adapter._completions[0]
    with pytest.raises(TypeError):
        NativeMemoryPromptInput()
    with pytest.raises(AttributeError):
        world.consumer_token.block = world.source_result.block
    with pytest.raises(TypeError):
        copy.copy(world.consumer_token)
    with pytest.raises(NativeContentPolicyDenied):
        run_memory_prompt_consumer(object.__new__(NativeMemoryPromptInput),
                                   user_message="How many counts?", task_id="execution")
    assert frame.phase == "released"
    with pytest.raises(NativeContentPolicyDenied):
        world.bind_consumer(world.source_result, consumer_context=NativeContentContext(world.admission, world.consumer_request))
    with pytest.raises(NativeContentPolicyDenied):
        world.run()
    assert frame.adapter.send_count == 1 and frame.adapter._completions == (proof,) and proof.result is result


@pytest.mark.parametrize("name", ["_cached_system_prompt", "_cached_system_prompt_static", "session_id"])
def test_no_cache_or_namespace_reseal(excluded, name):
    world = PromptWorld().construct().bind_consumer()
    setattr(world.consumer.agent, name, "Changed invented state")
    with pytest.raises(NativeContentPolicyDenied):
        world.run()
    denied_consumer(world)
    assert world.consumer.prompt_step == 0


@pytest.mark.parametrize("operation", ["build", "parts", "append", "invalidate", "source-read"])
def test_helpers_outside_authorized_window_refuse_before_discovery(excluded, monkeypatch, operation):
    world = PromptWorld().construct().bind_consumer()
    monkeypatch.setattr(system_prompt, "_ra", excluded)
    f = world.consumer
    with pytest.raises(NativeContentPolicyDenied):
        if operation == "build":
            f.agent._build_system_prompt(f.options[5])
        elif operation == "parts":
            f.agent._build_system_prompt_parts(f.options[5])
        elif operation == "append":
            system_prompt._append_builtin_memory_parts(f.agent, [])
        elif operation == "invalidate":
            system_prompt.invalidate_system_prompt(f.agent)
        else:
            system_prompt._append_builtin_memory_parts(world.agent, [])
    assert f.agent._cached_system_prompt is None
    assert f.adapter.send_count == 0


@pytest.mark.parametrize("operation", ["helper", "turn", "source-probe"])
def test_callback_reentry_has_no_native_window(excluded, monkeypatch, operation):
    world = PromptWorld().construct().bind_consumer()
    f = world.consumer
    monkeypatch.setattr(system_prompt, "_ra", excluded)
    def reenter():
        world.on_lookup = None
        if operation == "helper":
            f.agent._build_system_prompt(f.options[5])
        elif operation == "turn":
            f.agent.run_conversation("How many counts?", system_message=f.options[5], task_id="execution")
        else:
            f.parent._checkpoint()
    world.on_lookup = reenter
    with pytest.raises((NativeContentPolicyDenied, PolicyError)):
        world.run()
    denied_consumer(world)
    assert f.prompt_step == 0


def test_original_token_stays_on_original_thread(excluded):
    world = PromptWorld().construct().bind_consumer()
    context = copy_context()
    errors = []
    def wrong_thread():
        try:
            context.run(world.run)
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=wrong_thread)
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive() and len(errors) == 1 and isinstance(errors[0], NativeContentPolicyDenied)
    denied_consumer(world)


def test_retained_restrictions_and_source_close_after_consumer(excluded):
    world = PromptWorld().construct().bind_consumer()
    source = world.source_result
    world.binding = replace(world.binding, profiles=())
    world.consumer_binding = replace(world.consumer_binding, profiles=())
    result = world.run()
    f, runtime = world.consumer, world.adapter._runtime
    assert result["completed"]
    for handle in (source.binding.source_content, source.binding.request_content, source.result_content,
                   f.authorization_content, f.input_content, f.adapter._completions[0].content):
        assert runtime._content(handle).profiles == ("restricted_documents",)
    assert f.adapter.invocations[0].envelope.profiles == ("restricted_documents",)
    assert source is _memory_snapshots[world.adapter].registered_record
    assert not _memory_snapshots[world.adapter].closed
    world.adapter.validate_scope()
    world.agent.close()
    world.agent.close()
    assert _memory_snapshots[world.adapter].closed and not world.context._denied


@pytest.mark.parametrize("change", ["getter", "store", "version", "producer", "destination", "port", "restriction"])
def test_source_specific_or_consumer_drift_refuses_before_send(excluded, change):
    world = PromptWorld().construct().bind_consumer()
    if change == "getter":
        world.store.format_for_system_prompt = lambda target: world.source_result.block
    elif change == "store":
        world.store._system_prompt_snapshot = dict(world.store._system_prompt_snapshot)
    elif change == "version":
        world.source = replace(world.source, source_version=2)
    elif change == "producer":
        world.memory_policy = replace(world.memory_policy, revision=7)
    elif change == "destination":
        port = world.consumer_ports["main"]
        port.destination = replace(port.destination, revision=2)
    elif change == "port":
        world.selection_port.current = lambda handle: world.consumer_selection
    else:
        world.consumer_binding = replace(world.consumer_binding, profiles=("new_restriction",))
    with pytest.raises((NativeContentPolicyDenied, PolicyError)):
        world.run()
    denied_consumer(world)


@pytest.mark.parametrize("control", [InterruptedError, asyncio.CancelledError, KeyboardInterrupt, SystemExit])
def test_original_trusted_control_identity_after_actual_send(excluded, control):
    world = PromptWorld().construct().bind_consumer()
    original = control("Invented controlled stop")
    def stop(envelope):
        raise original
    world.consumer_ports["main"].hook = stop
    with pytest.raises(control) as failure:
        world.run()
    assert failure.value is original
    denied_consumer(world, sends=1)
    assert len(world.consumer_ports["main"].sent) == 1


@pytest.mark.parametrize("boundary", ["caller-return", "scope-exit", "release"])
def test_late_runtime_poison_retains_real_completion(excluded, boundary):
    world = PromptWorld().construct().bind_consumer()
    f = world.consumer
    observed = []
    def poison():
        ready = (bool(f.adapter._completions) if boundary == "caller-return" else
                 f.terminal if boundary == "scope-exit" else f.phase == "released")
        if ready:
            world.on_clock = None
            def wrong_thread():
                try:
                    f.runtime.admit(world.consumer_request, b"Invented wrong thread")
                except BaseException as error:
                    observed.append(error)
            thread = threading.Thread(target=wrong_thread)
            thread.start()
            thread.join(timeout=10)
            assert not thread.is_alive()
    world.on_clock = poison
    with pytest.raises((NativeContentPolicyDenied, PolicyError)):
        world.run()
    assert len(observed) == 1 and observed[0].args == ("reentrant_or_threaded_use",)
    assert f.adapter.send_count == 1 and f.adapter._completions
    assert f.adapter._completions[0].payload == vectors()["stages"]["native_result_bytes"]["utf8"].encode()
    assert f.phase == "denied" and world.adapter.send_count == 0


@pytest.mark.parametrize("source", [True, False], ids=["P", "C"])
def test_original_selected_expiry_beats_other_live_coverage(excluded, source):
    world = PromptWorld()
    selected = "route_main" if source else "consumer_main_route"
    original = next(route for route in world.routes if route.route_ref == selected)
    alternate = replace(original, route_ref="alternate_selected_main", expires_at=200,
                        qualification_ref="synthetic_alternate_main")
    world.extra_qualifications = (alternate,)
    world.routes = (*tuple(r if r is original else replace(r, expires_at=200) for r in world.routes), alternate)
    world.profiles = tuple(replace(p, allowed_routes=(*p.allowed_routes, alternate.route_ref)) for p in world.profiles)
    world.construct().bind_consumer()
    f = world.consumer
    assert len(world.qualification_records) == 17
    def expire():
        if f.adapter._completions:
            world.now = original.expires_at
    world.on_clock = expire
    with pytest.raises((NativeContentPolicyDenied, PolicyError)):
        world.run()
    assert world.now < alternate.expires_at and world.now == original.expires_at
    assert all(any(r.path == path and r.selection_id == original.selection_id and world.now < r.expires_at
                   for r in world.routes) for path in PATHS)
    assert f.adapter.send_count == 1 and f.adapter._completions and f.phase == "denied"


def test_bounded_real_byte_capacity_failure(excluded, record_property):
    world = PromptWorld().construct().bind_consumer()
    f, runtime = world.consumer, world.adapter._runtime
    remaining = MAX_RETAINED - runtime._retained_bytes - len(vectors()["stages"]["raw_turn_bytes"]["utf8"].encode()) + 1
    fillers = []
    with content_policy_scope(world.context):
        while remaining:
            size = min(MAX_PAYLOAD, remaining)
            fillers.append(runtime.admit(world.request, b"x" * size))
            remaining -= size
    charged = runtime._retained_bytes
    assert len(fillers) == 8
    with pytest.raises((NativeContentPolicyDenied, PolicyError)):
        world.run()
    assert runtime._retained_bytes == charged and f.input_content is None
    assert len(runtime._contents) == 12
    denied_consumer(world)
    record_property("capacity_proof", json.dumps(dict(setup_admissions=8, source_slots=3,
        authorization_slots=1, retained_bytes=charged, failed_raw_bytes=324, consumer_sends=0)))


@pytest.mark.parametrize("operation", ["model", "new-binding", "read", "tool"])
def test_source_terminal_guards_remain_unconditional(excluded, operation):
    world = PromptWorld().construct().bind_consumer()
    before = world.source_result
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            if operation == "model":
                world.agent.run_conversation("How many counts?", task_id="execution")
            elif operation == "new-binding":
                world.adapter.bind_memory_snapshot(world.agent, store=world.store,
                    producer_ref=world.memory_policy.producer_ref)
            elif operation == "read":
                world.adapter.read_memory_snapshot(world.agent, world.token)
            else:
                guard_tool_result(world.agent, "invented_memory_producer")
    assert world.adapter.send_count == world.consumer.adapter.send_count == 0
    assert _memory_snapshots[world.adapter].record is before


def test_ordinary_memory_user_slot_and_warning_status_behavior(monkeypatch):
    world = PromptWorld()
    agent = SimpleNamespace(_memory_store=world.store, _memory_enabled=True, _user_profile_enabled=True)
    parts = []
    system_prompt._append_builtin_memory_parts(agent, parts)
    assert parts == [world.store.format_for_system_prompt("memory"), world.store.format_for_system_prompt("user")]
    agent._memory_enabled = False
    parts = []
    system_prompt._append_builtin_memory_parts(agent, parts)
    assert parts == [world.store.format_for_system_prompt("user")]
    agent._memory_store = None
    statuses = []
    agent._emit_status = statuses.append
    monkeypatch.setattr(system_prompt, "build_system_prompt_parts",
                        lambda *args, **kwargs: dict(stable="Stable", context="Context", volatile="Volatile"))
    token = prompt_builder._truncation_warnings.set(["Invented ordinary warning"])
    try:
        prompt = system_prompt.build_system_prompt(agent)
        assert statuses == ["Invented ordinary warning"]
        assert not prompt_builder.drain_truncation_warnings()
    finally:
        prompt_builder._truncation_warnings.reset(token)
    assert prompt == "Stable\n\nContext\n\nVolatile" and agent._cached_system_prompt_static == "Stable"


@pytest.mark.parametrize("boundary", ["before-send", "after-send"])
def test_cache_freeze_checks_drift_after_actual_build(excluded, boundary):
    world = PromptWorld().construct().bind_consumer()
    f = world.consumer
    def drift(*args):
        if f.cache_frozen:
            f.agent._cached_system_prompt_static = "Changed stable prompt"
    if boundary == "before-send":
        world.on_clock = drift
    else:
        world.consumer_ports["main"].hook = drift
    with pytest.raises((NativeContentPolicyDenied, PolicyError)):
        world.run()
    sends = int(boundary == "after-send")
    denied_consumer(world, sends=sends)
    assert f.cache_frozen and f.prompt_step == 4
    if sends:
        assert f.partial_send[3] is world.consumer_ports["main"].output


@pytest.mark.parametrize("fact", ["owner_ref", "execution_ref", "lineage_ref", "consumer_ref", "auth_epoch", "policy_revision"])
def test_all_first_six_holder_facts_must_match(excluded, fact):
    world = PromptWorld().construct()
    source = world.capture()
    value = getattr(world.consumer_binding, fact)
    world.consumer_binding = replace(world.consumer_binding, **{fact: value + 1 if type(value) is int else "unrelated"})
    with pytest.raises(NativeContentPolicyDenied):
        world.bind_consumer(source)
    assert _memory_snapshots[world.adapter].memory_prompt_attempt is _memory_prompt_frames[world.adapter]
    denied_consumer(world)


def test_bind_requires_active_original_source_scope(excluded):
    world = PromptWorld().construct()
    source = world.capture()
    with pytest.raises(NativeContentPolicyDenied):
        bind_memory_prompt_consumer(world.adapter, source, consumer_context=world.consumer_context,
            selection=world.selection_port, result_policy=world.result_port,
            producer_ref="invented_consumer_producer", main_route_ref="consumer_main_route",
            system_message="Use retained notes.", session_id="memory-consumer-session")
    assert not hasattr(_memory_snapshots[world.adapter], "memory_prompt_attempt")
    world.bind_consumer(source)
    assert world.consumer.phase == "bound" and world.consumer.adapter.send_count == 0


def test_private_constructor_exception_requires_registered_frame(excluded):
    world = PromptWorld().construct()
    options = dict(selection=world.selection_port, result_policy=world.result_port,
        producer_ref="invented_consumer_producer", main_route_ref="consumer_main_route",
        profiles=world.profiles, routes=world.routes, transports=world.adapter._operator_configuration[2],
        clock=world.adapter._clock)
    for changes in (dict(_memory_prompt_frame=object()),
                    dict(_memory_prompt_frame=object(), _delegation_frame=object())):
        with pytest.raises(NativeContentPolicyDenied):
            NativeMainPayloadAdapter(world.consumer_context, **options, **changes)
    assert not world.consumer_context._controlled_main_bound
    assert not world.consumer_context._denied and world.adapter.send_count == 0
