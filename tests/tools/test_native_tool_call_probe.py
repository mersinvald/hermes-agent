"""Observe one invented assistant call through the actual synchronous caller."""

import asyncio
import copy
import hashlib
import json
import threading
from contextvars import copy_context
from dataclasses import replace
from pathlib import Path
from types import FunctionType, MethodType

import pytest
from hermes_content_policy import ContentPolicyRuntime, PATHS, MAX_PAYLOAD, MAX_RETAINED, PolicyError

from agent import chat_completion_helpers as caller
from agent.content_policy import NativeContentPolicyDenied, content_policy_scope
from agent.native_main_payload import (
    NativeToolCallObservation, NativeToolCallProbeInput, _tool_call_probes, _request, _decode_response,
)
from tests.agent.test_native_main_payload import (
    World, canonical, invented_offline_configuration as invented_offline_configuration,
)
from tests.agent.test_native_controlled_conversation import construct, excluded as excluded
from tests.tools.test_native_controlled_delegation import (
    retain_case_durations as retain_case_durations, no_delegation_effects as no_delegation_effects,
)
from tests.tools.test_native_parent_resumption import (
    retain_incremental_case_receipts as retain_incremental_case_receipts,
    record_case_start as record_case_start,
)


SYSTEM = "Request one leaf call."
USER = "Count blue squares."
ARGUMENTS = '{"context":null,"goal":"Count blue squares.","role":"leaf"}'


def vectors():
    return json.loads((Path(__file__).resolve().parents[2]
        / "docs/evidence/accountant-native-tool-call-probe-20261123/vectors.json").read_text())


class ProbeWorld(World):
    def __init__(self):
        super().__init__()
        self.routes = tuple(replace(r, route_ref="main_route") if r.path == "main" else r for r in self.routes)
        self.profiles = (replace(self.profiles[0], allowed_routes=tuple(r.route_ref for r in self.routes)),)
        self.clock_hook = None

    def time(self):
        if self.clock_hook is not None:
            self.clock_hook()
        return self.now

    def qualify(self):
        runtime = ContentPolicyRuntime(admission=self.admission, profiles=self.profiles,
            routes=self.routes, transports=tuple((p.destination.transport_ref, p) for p in self.ports.values()),
            clock=lambda: self.now, policy_revision=9)
        records = []
        for route in self.routes:
            content = runtime.admit(self.request, b"Invented declared operation")
            result = runtime.dispatch(self.request, content, path=route.path, route_ref=route.route_ref)
            port = self.ports[route.path]
            envelope, = port.sent
            records.append((route.path, port, envelope, result, port.output))
            port.sent = []
        self.qualification_records = tuple(records)
        self.ports["main"].output = vectors()["stages"]["R"]["utf8"].encode()

    def construct(self):
        from agent.native_main_payload import bind_controlled_main
        self.qualify()
        self.adapter = bind_controlled_main(self.context, selection=self.selection_port,
            result_policy=self.result_port, producer_ref="invented_producer", main_route_ref="main_route",
            profiles=self.profiles, routes=self.routes,
            transports=tuple((port.destination.transport_ref, port) for port in self.ports.values()), clock=self.time)
        construct(self)
        return self

    def bind_probe(self, **changes):
        options = dict(system_message=SYSTEM, user_message=USER, task_id="execution", expected_arguments_json=ARGUMENTS)
        options.update(changes)
        self.token = self.adapter.bind_tool_call_probe(self.agent, **options)
        return self

    def run(self):
        return self.adapter.run_tool_call_probe(self.agent, self.token)


def test_actual_constructed_caller_send_observation(excluded, record_property):
    world = ProbeWorld().construct()
    agent_anchors = tuple(vars(world.agent).items())
    with content_policy_scope(world.context):
        world.bind_probe()
        result = world.run()
        assert world.adapter.tool_call_observations == (result,)
        owner = _tool_call_probes[world.adapter]
        assert result is owner.records[0] and type(result) is NativeToolCallObservation
        assert owner.phase == "released" and owner.stage_index == 8
    assert all(vars(world.agent)[key] is value for key, value in agent_anchors)
    assert not world.agent.tools and not world.agent.valid_tool_names
    adapter, runtime = world.adapter, world.adapter._runtime
    assert adapter.send_count == len(world.ports["main"].sent) == 1
    assert len(world.qualification_records) == len(PATHS) == 8
    assert not adapter.invocations and not adapter._completions and not adapter._turn_frames
    assert result.envelope is world.ports["main"].sent[0]
    assert result.observed_destination is world.selection.destination
    assert result.response_bytes is world.ports["main"].output
    assert result.envelope.payload is result.wire_bytes
    assert result.binding.token is world.token and result.binding.agent is world.agent
    actual = dict(A=result.binding.authorization_bytes, Q=result.request_bytes, W=result.wire_bytes,
                  R=result.response_bytes, O=result.observation_bytes)
    handles = dict(A=result.binding.authorization_content, Q=result.request_content, W=result.wire_content,
                   R=result.response_content, O=result.observation_content)
    fixed = vectors()
    for key, payload in actual.items():
        assert payload == fixed["stages"][key]["utf8"].encode()
        assert len(payload) == fixed["stages"][key]["bytes"]
        assert hashlib.sha256(payload).hexdigest() == fixed["stages"][key]["sha256"]
        retained = runtime._content(handles[key])
        assert retained.payload is payload
        assert all(a is b for a, b in zip(retained.parents, tuple(handles[k] for k in fixed["ancestry"][key])))
        assert len(retained.parents) == len(fixed["ancestry"][key])
    assert runtime._retained_bytes == 3265 and len(runtime._contents) == 5 and len(runtime._envelopes) == 1
    assert result.arguments_json == ARGUMENTS and result.call_id == "invented-call-1" and result.function_name == "delegate_task"
    assert not world.sdk_calls
    record_property("probe_proof", json.dumps(dict(stages={k:v.decode() for k,v in actual.items()},
        qualification_sends=8, probe_sends=1, retained_bytes=3265, content_slots=5, envelopes=1,
        result_identity=True, original_constructor_anchors=True)))


class Hostile:
    def __str__(self):
        pytest.fail("Input conversion ran")


class StringSubclass(str):
    pass


def assert_denied(world, sends=0):
    owner = _tool_call_probes.get(world.adapter)
    assert owner is not None and owner.phase == "denied"
    assert world.context._denied and world.adapter._invalid
    assert world.adapter.send_count == len(world.ports["main"].sent) == sends
    assert not world.adapter.invocations and not world.adapter._completions and not world.sdk_calls


@pytest.mark.parametrize("changes", [
    dict(system_message=Hostile()), dict(user_message=StringSubclass(USER)), dict(task_id="another_task"),
    dict(expected_arguments_json='{"context":null,"goal":"x","role":"leaf","extra":0}'),
    dict(expected_arguments_json='{"context":null,"goal":"x","goal":"y","role":"leaf"}'),
    dict(expected_arguments_json='{"context":null, "goal":"x","role":"leaf"}'),
    dict(expected_arguments_json='{"context":null,"goal":"","role":"leaf"}'),
    dict(expected_arguments_json='{"context":"other","goal":"x","role":"leaf"}'),
    dict(expected_arguments_json='{"context":null,"goal":"x","role":"orchestrator"}'),
    dict(expected_arguments_json='{"context":null,"goal":"\\ud800","role":"leaf"}'),
    dict(user_message="é" * 8193),
], ids=["opaque", "subclass", "task", "unknown", "duplicate", "noncanonical", "empty", "context", "role", "surrogate", "utf8-bound"])
def test_invalid_inputs_refuse_before_callbacks(excluded, changes):
    world = ProbeWorld().construct()
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            world.admission_hook = lambda: pytest.fail("Invalid input reached an operator callback")
            world.bind_probe(**changes)
    assert_denied(world)
    owner = _tool_call_probes[world.adapter]
    assert owner.binding is None and not world.adapter._runtime._contents


def malformed(kind):
    payload = vectors()["stages"]["R"]["utf8"].encode()
    value = json.loads(payload)
    choice = value["choices"][0]
    call = choice["message"]["tool_calls"][0]
    if kind == "duplicate":
        return payload.replace(b'"created":1', b'"created":1,"created":1')
    if kind == "invalid-utf8":
        return b'\xff'
    if kind == "size":
        return b' ' * 65537
    if kind == "two-calls":
        choice["message"]["tool_calls"].append(copy.deepcopy(call))
    elif kind == "two-choices":
        value["choices"].append(copy.deepcopy(choice))
    elif kind == "unknown":
        call["extra"] = False
    elif kind == "model":
        value["model"] = "another_model"
    elif kind == "stop":
        choice["finish_reason"] = "stop"
    elif kind == "content":
        choice["message"]["content"] = "Unissued text"
    elif kind == "bool-scalar":
        value["created"] = True
    elif kind == "usage":
        value["usage"]["total_tokens"] = 9
    elif kind == "name":
        call["function"]["name"] = "another_function"
    elif kind == "call-id":
        call["id"] = "bad id"
    elif kind == "arguments":
        call["function"]["arguments"] = '{"context":null,"goal":"Another goal.","role":"leaf"}'
    return canonical(value)


@pytest.mark.parametrize("kind", ["duplicate", "invalid-utf8", "size", "two-calls", "two-choices", "unknown",
    "model", "stop", "content", "bool-scalar", "usage", "name", "call-id", "arguments"])
def test_malformed_response_retains_shared_response_before_refusal(excluded, kind, record_property):
    world = ProbeWorld().construct()
    world.ports["main"].output = malformed(kind)
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            world.bind_probe().run()
    assert_denied(world, 1)
    owner, runtime = _tool_call_probes[world.adapter], world.adapter._runtime
    assert owner.envelope is world.ports["main"].sent[0]
    assert owner.response_bytes is world.ports["main"].output
    assert owner.response_content is not None and not owner.records
    assert runtime._content(owner.response_content).payload is owner.response_bytes
    assert runtime._content(owner.response_content).parents == (owner.wire_content,)
    assert len(runtime._contents) == 4 and len(runtime._envelopes) == 1
    assert runtime._retained_bytes == 1616 + len(owner.response_bytes)
    record_property("partial_proof", json.dumps(dict(response_retained=True, sends=1,
        retained_bytes=runtime._retained_bytes, content_slots=4, envelopes=1, observation=False)))


def test_original_argument_binding_and_ordinary_text_refusals(excluded):
    world = ProbeWorld().construct()
    request = json.loads(vectors()["stages"]["Q"]["utf8"])
    with pytest.raises(NativeContentPolicyDenied):
        _request(request)
    with pytest.raises(NativeContentPolicyDenied):
        _decode_response(vectors()["stages"]["R"]["utf8"].encode(), "invented_model")
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            world.bind_probe(expected_arguments_json='{"context":null,"goal":"Another goal.","role":"leaf"}').run()
    assert_denied(world, 1)
    assert _tool_call_probes[world.adapter].response_content is not None


@pytest.mark.parametrize("mode", ["copied", "wrong-thread", "missing-scope", "equal-binding"])
def test_original_token_scope_thread_and_binding_identity(excluded, mode):
    world = ProbeWorld().construct()
    errors = []
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            world.bind_probe()
            if mode == "copied":
                world.token = object.__new__(NativeToolCallProbeInput)
                world.run()
            elif mode == "equal-binding":
                owner = _tool_call_probes[world.adapter]
                owner.binding = replace(owner.binding)
                world.run()
            elif mode == "wrong-thread":
                context = copy_context()
                def invoke():
                    try:
                        context.run(world.run)
                    except BaseException as error:
                        errors.append(error)
                thread = threading.Thread(target=invoke)
                thread.start()
                thread.join()
                assert len(errors) == 1 and type(errors[0]) is NativeContentPolicyDenied
            else:
                with content_policy_scope(None):
                    world.run()
    assert_denied(world)


@pytest.mark.parametrize("fact", ["selection", "producer", "clock", "port", "method"])
def test_changed_original_facts_and_ports_refuse(excluded, fact):
    world = ProbeWorld().construct()
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            world.bind_probe()
            if fact == "selection":
                world.selection = replace(world.selection)
            elif fact == "producer":
                world.policy = replace(world.policy, profiles=("unauthorized",))
            elif fact == "clock":
                world.adapter._clock = lambda: world.now
            elif fact == "port":
                world.ports["main"].destination = replace(world.selection.destination)
            else:
                world.adapter._dispatch_retained_operation = lambda operation: pytest.fail("Replacement ran")
            world.run()
    assert_denied(world)


@pytest.mark.parametrize("mode", ["reenter", "phase-relabel", "generic-caller"])
def test_callback_has_no_owner_window(excluded, mode):
    world = ProbeWorld().construct()
    errors = []
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            world.bind_probe()
            owner = _tool_call_probes[world.adapter]
            def hook():
                world.admission_hook = None
                if mode == "phase-relabel":
                    owner.phase = owner.phase_anchor = "entered"
                try:
                    if mode == "generic-caller":
                        caller.interruptible_api_call(world.agent, json.loads(vectors()["stages"]["Q"]["utf8"]))
                    else:
                        world.run()
                except NativeContentPolicyDenied as error:
                    errors.append(error)
            world.admission_hook = hook
            world.run()
    assert errors
    assert_denied(world)


def test_retained_restrictions_survive_current_profile_removal_and_close(excluded):
    world = ProbeWorld().construct()
    with content_policy_scope(world.context):
        world.bind_probe()
        world.binding = replace(world.binding, profiles=())
        result = world.run()
    runtime = world.adapter._runtime
    assert all(row.profiles == ("restricted_documents",) for row in runtime._contents.values())
    assert result.envelope.profiles == ("restricted_documents",)
    world.agent.close()
    world.agent.close()
    assert world.adapter.tool_call_observations[0] is result
    assert not world.context._denied


@pytest.mark.parametrize("operation", ["replay", "conversation", "executor", "bind", "ordinary-caller", "delegation", "memory", "auxiliary", "fallback", "runtime-admit"])
def test_success_is_terminal_for_new_work(excluded, operation):
    world = ProbeWorld().construct()
    with content_policy_scope(world.context):
        world.bind_probe()
        result = world.run()
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            if operation == "replay":
                world.run()
            elif operation == "conversation":
                world.agent.run_conversation(USER, system_message=SYSTEM, task_id="execution")
            elif operation == "executor":
                from agent.tool_executor import execute_tool_calls_sequential
                execute_tool_calls_sequential(world.agent, [], [], "execution")
            elif operation == "bind":
                world.bind_probe()
            elif operation == "delegation":
                world.adapter.bind_delegation(world.agent, receiver_request=world.request,
                    receiver_selection=world.selection_port, receiver_producer_ref="invented_producer",
                    child_request=world.request, child_selection=world.selection_port,
                    child_producer_ref="invented_producer", delegate_route_ref="route_delegate",
                    child_main_route_ref="main_route", child_session_id="child", child_task_id="execution")
            elif operation == "memory":
                world.adapter.bind_memory_snapshot(world.agent, store=object(), producer_ref="invented_producer")
            elif operation == "auxiliary":
                world.adapter.bind_auxiliary(world.agent, path="title", task="title", request_handle=world.request,
                    selection=world.selection_port, producer_ref="invented_producer", route_ref="route_title", language="en")
            elif operation == "fallback":
                world.adapter.bind_fallback(world.agent, request_handle=world.request, selection=world.selection_port,
                    producer_ref="invented_producer", route_ref="route_fallback")
            elif operation == "runtime-admit":
                try:
                    world.adapter._runtime.admit(world.request, b"Invented later operation")
                except PolicyError:
                    raise NativeContentPolicyDenied("Expected runtime refusal") from None
            else:
                caller.interruptible_api_call(world.agent, json.loads(vectors()["stages"]["Q"]["utf8"]))
    assert_denied(world, 1)
    assert world.adapter.tool_call_observations[0] is result


def test_genuine_bounded_byte_capacity_failure(excluded, record_property):
    world = ProbeWorld().construct()
    runtime = world.adapter._runtime
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            remaining = MAX_RETAINED - MAX_PAYLOAD - 1203 - runtime._retained_bytes
            fills = []
            while remaining:
                size = min(MAX_PAYLOAD, remaining)
                fills.append(runtime.admit(world.request, b"x" * size))
                remaining -= size
            world.bind_probe().run()
    assert_denied(world)
    owner = _tool_call_probes[world.adapter]
    assert len(fills) == 7 and len(runtime._contents) < 16
    assert owner.request_content is not None and owner.wire_content is not None
    assert owner.envelope is None and not owner.records
    record_property("partial_proof", json.dumps(dict(sends=0, fill_admissions=7,
        retained_bytes=runtime._retained_bytes, content_slots=len(runtime._contents), failed_capacity=True)))


def test_bind_requires_active_original_scope(excluded):
    world = ProbeWorld().construct()
    with pytest.raises(NativeContentPolicyDenied, match="scope_invalid"):
        world.bind_probe()
    assert_denied(world)
    assert not world.adapter._runtime._contents


def test_selected_expiry_cannot_use_other_live_coverage(excluded, record_property):
    world = ProbeWorld()
    world.routes = (*world.routes, replace(world.routes[0], route_ref="alternate_main", expires_at=200))
    world.profiles = (replace(world.profiles[0], allowed_routes=tuple(r.route_ref for r in world.routes)),)
    world.construct()
    assert len(world.qualification_records) == 9
    with pytest.raises(NativeContentPolicyDenied, match="selection_expired"):
        with content_policy_scope(world.context):
            world.bind_probe()
            owner = _tool_call_probes[world.adapter]
            def expire():
                if owner.stage_index == 8:
                    world.now = 100
            world.clock_hook = expire
            world.run()
    assert_denied(world, 1)
    assert owner.response_content is not None and len(owner.records) == 1
    assert world.adapter._runtime._retained_bytes == 3265
    record_property("partial_proof", json.dumps(dict(qualification_sends=9, probe_sends=1,
        response_retained=True, observation_retained=True, final_release=False, retained_bytes=3265)))


@pytest.mark.parametrize("boundary", ["caller-return", "scope-exit", "release"])
def test_late_health_poison_keeps_original_observation(excluded, boundary, record_property):
    world = ProbeWorld().construct()
    errors = []
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            world.bind_probe()
            owner, runtime = _tool_call_probes[world.adapter], world.adapter._runtime
            def poison():
                if not owner.records:
                    return
                ready = {"caller-return": owner.stage_index == 5 and owner.dispatch_frame is None,
                    "scope-exit": owner.stage_index == 7 and owner.stage_frame is None,
                    "release": owner.stage_index == 8}[boundary]
                if not ready:
                    return
                world.clock_hook = None
                def wrong_thread():
                    try:
                        runtime.admit(world.request, b"Invented unauthorized thread")
                    except PolicyError as error:
                        errors.append(error)
                thread = threading.Thread(target=wrong_thread)
                thread.start()
                thread.join()
            world.clock_hook = poison
            world.run()
    assert errors and runtime._poisoned
    assert_denied(world, 1)
    result, = world.adapter.tool_call_observations
    assert result.response_bytes is world.ports["main"].output
    assert runtime._content(result.response_content).payload is result.response_bytes
    assert runtime._content(result.observation_content).payload is result.observation_bytes
    assert runtime._retained_bytes == 3265
    record_property("partial_proof", json.dumps(dict(boundary=boundary, sends=1,
        observation_retained=True, final_release=False, retained_bytes=3265)))


@pytest.mark.parametrize("control", [InterruptedError, asyncio.CancelledError, KeyboardInterrupt, SystemExit])
def test_original_control_object_after_actual_response(excluded, control):
    world = ProbeWorld().construct()
    original = control("Invented trusted control")
    with pytest.raises(control) as observed:
        with content_policy_scope(world.context):
            world.bind_probe()
            owner = _tool_call_probes[world.adapter]
            def stop():
                if owner.records and owner.dispatch_frame is None:
                    raise original
            world.clock_hook = stop
            world.run()
    assert observed.value is original
    assert_denied(world, 1)
    assert owner.records[0].response_bytes is world.ports["main"].output
    assert owner.response_content is not None and world.adapter._runtime._retained_bytes == 3265


def test_trusted_send_failure_keeps_attempt_and_envelope(excluded):
    world = ProbeWorld().construct()
    original = InterruptedError("Invented trusted send control")
    def stop(envelope):
        raise original
    world.ports["main"].hook = stop
    with pytest.raises(InterruptedError) as observed:
        with content_policy_scope(world.context):
            world.bind_probe().run()
    assert observed.value is original
    assert_denied(world, 1)
    owner = _tool_call_probes[world.adapter]
    assert owner.envelope is world.ports["main"].sent[0]
    assert owner.response_content is owner.response_bytes is None and not owner.records
    assert world.adapter._runtime._retained_bytes == 1616


def test_source_and_consumer_memory_participants_refuse_probe(excluded):
    from tests.tools.test_native_memory_prompt_consumer import PromptWorld
    world = PromptWorld().construct().bind_consumer()
    world.run()
    for adapter, agent, context in (
        (world.consumer.adapter, world.consumer.agent, world.consumer.context),
        (world.adapter, world.agent, world.context),
    ):
        with pytest.raises(NativeContentPolicyDenied):
            with content_policy_scope(context):
                adapter.bind_tool_call_probe(agent, system_message=SYSTEM, user_message=USER,
                    task_id="execution", expected_arguments_json=ARGUMENTS)
        assert not adapter.tool_call_observations and context._denied
    assert world.adapter.send_count == 0 and world.consumer.adapter.send_count == 1


def test_token_and_records_grant_no_execution_authority(excluded):
    with pytest.raises(TypeError):
        NativeToolCallProbeInput()
    world = ProbeWorld().construct()
    with content_policy_scope(world.context):
        world.bind_probe()
        result = world.run()
    copied = replace(result)
    assert copied == result and copied is not result
    assert world.adapter.tool_call_observations[0] is result
    with pytest.raises(AttributeError):
        world.adapter.tool_call_observations = (copied,)
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            world.adapter.run_tool_call_probe(world.agent, copied)
    assert_denied(world, 1)


def test_released_direct_host_participant_cannot_bind_probe(no_delegation_effects):
    from tests.tools.test_native_controlled_delegation import ThreeHolders
    world = ThreeHolders().bind()
    world.call()
    before = sum(len(p.sent) for p in world.ports.values())
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            world.adapter.bind_tool_call_probe(world.parent, system_message=SYSTEM, user_message=USER,
                task_id="invented_parent_execution", expected_arguments_json=ARGUMENTS)
    assert world.context._denied and not world.adapter.tool_call_observations
    assert sum(len(p.sent) for p in world.ports.values()) == before == 2


def test_read_only_proof_returns_original_registered_record(excluded):
    world = ProbeWorld().construct()
    with content_policy_scope(world.context):
        world.bind_probe()
        result = world.run()
    owner = _tool_call_probes[world.adapter]
    copied = replace(result)
    owner.records = (copied,)
    assert world.adapter.tool_call_observations[0] is result
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            world.run()
    assert_denied(world, 1)
    assert world.adapter.tool_call_observations[0] is result


@pytest.mark.parametrize("target", ["caller", "adapter-before-bind"])
def test_copied_callable_grants_no_probe_window(excluded, target):
    world = ProbeWorld().construct()
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            if target == "adapter-before-bind":
                original = world.adapter._dispatch_retained_operation.__func__
                copied = FunctionType(original.__code__, original.__globals__, original.__name__)
                world.adapter._dispatch_retained_operation = MethodType(copied, world.adapter)
                world.bind_probe()
            else:
                world.bind_probe()
                original = caller.interruptible_api_call
                _tool_call_probes[world.adapter].caller = FunctionType(original.__code__, original.__globals__, original.__name__)
                world.run()
    assert_denied(world)
