"""Associate one original issued call with an actual controlled leaf result."""

import json
import asyncio
import sys
import threading
import time
from contextvars import copy_context
from dataclasses import replace
from functools import cache
from types import SimpleNamespace

import pytest
from hermes_content_policy import ContentPolicyRuntime, Destination, ExecutionBinding, PATHS, ProfilePolicy, RouteBinding

from agent.content_policy import NativeContentContext, NativeContentPolicyDenied, content_policy_scope
from agent.native_main_payload import (
    NativeMainSelection, NativeResultPolicy, NativeIssuedCallInput, NativeIssuedCallResult,
    _issued_call_frames, _tool_call_probes, bind_controlled_main,
)
from tests.agent.test_native_main_payload import (
    ControlledPort, canonical, invented_offline_configuration as invented_offline_configuration,
)
from tests.agent.test_native_controlled_conversation import excluded as excluded
from tests.tools.test_native_controlled_delegation import (
    no_delegation_effects as no_delegation_effects, retain_case_durations as retain_case_durations,
)
from tests.tools.test_native_parent_resumption import (
    retain_incremental_case_receipts as retain_incremental_case_receipts, record_case_start as record_case_start,
)


ARGUMENTS = '{"context":null,"goal":"Count blue squares.","role":"leaf"}'
_construction_seconds = []


@pytest.fixture(scope="module", autouse=True)
def retain_failure_contexts(request):
    from pathlib import Path
    import traceback
    xml = request.config.option.xmlpath
    if xml is None:
        yield
        return
    path = Path(xml).with_suffix(".failures.jsonl")
    class Failures:
        def pytest_runtest_makereport(self, item, call):
            if call.when == "call":
                with path.with_suffix(".construction.jsonl").open("a") as output:
                    output.write(json.dumps(dict(nodeid=item.nodeid, construction_seconds=list(_construction_seconds))) + "\n")
                _construction_seconds.clear()
            if call.excinfo is None:
                return
            error, chain, seen = call.excinfo.value, [], set()
            while error is not None and id(error) not in seen:
                seen.add(id(error))
                chain.append("".join(traceback.format_exception(type(error), error, error.__traceback__, chain=False)))
                error = error.__context__
            with path.open("a") as output:
                output.write(json.dumps(dict(nodeid=item.nodeid, phase=call.when, contexts=chain)) + "\n")
    plugin = Failures()
    request.config.pluginmanager.register(plugin)
    yield
    request.config.pluginmanager.unregister(plugin)


class IssuedWorld:
    def __init__(self):
        self.now = 10
        self.hooks = {}
        self.before_probe = None
        self.requests = tuple(object() for _ in range(4))
        self.bindings = tuple(ExecutionBinding("owner", "execution" if i < 2 else "child_execution", "lineage",
            "document_specialist" if i < 2 else "child_specialist", 7, 9, f"selection_{i}", i + 3,
            ("restricted_documents",)) for i in range(4))
        self.ports, self.routes, self.selections = {}, [], []
        for i, holder in enumerate(self.bindings):
            destination = Destination(f"invented_{i}", "synthetic_native", f"invented_model_{i}",
                                      f"invented_endpoint_{i}", 1)
            port = ControlledPort(destination)
            port.output = b'{"accepted":true}'
            for path in PATHS:
                self.ports[i, path] = port
                self.routes.append(RouteBinding(f"route_{i}_{path}", holder.selection_id, holder.selection_version,
                    holder.consumer_ref, path, destination, 100, f"synthetic_{i}_{path}", "synthetic_only"))
                if path == "main":
                    self.selections.append(NativeMainSelection(holder.selection_id, holder.selection_version,
                                                              "chat_completions", destination))
        self.routes = tuple(self.routes)
        self.profiles = (ProfilePolicy("restricted_documents", tuple(r.route_ref for r in self.routes), PATHS),)
        self.transports = tuple((p.destination.transport_ref, p) for p in dict.fromkeys(self.ports.values()))
        self.policies = tuple(NativeResultPolicy(f"producer_{i}", 4, ("restricted_documents",)) for i in range(4))
        self.admission = SimpleNamespace(current=self.current)
        self.selection_ports = tuple(SimpleNamespace(current=self.selection) for _ in range(4))
        self.result_port = SimpleNamespace(current=self.produced)
        self.clock = self.time
        self.context = NativeContentContext(self.admission, self.requests[0])
        self.qualifications = []

    def hook(self, name):
        if name in self.hooks:
            self.hooks[name]()

    def current(self, handle):
        self.hook("admission")
        for i, request in enumerate(self.requests):
            if handle is request:
                return self.bindings[i]
        for (i, _), port in self.ports.items():
            if handle is port:
                return self.bindings[i]
        raise AssertionError("Unknown capability")

    def selection(self, handle):
        self.hook("selection")
        return next(s for h, s in zip(self.requests, self.selections) if h is handle)

    def produced(self, ref):
        self.hook("producer")
        return next(p for p in self.policies if p.producer_ref == ref)

    def time(self):
        self.hook("clock")
        return self.now

    def construct(self):
        started = time.monotonic()
        from run_agent import AIAgent
        qualifier = ContentPolicyRuntime(admission=self.admission, profiles=self.profiles, routes=self.routes,
            transports=self.transports, clock=self.clock, policy_revision=9)
        for route in self.routes:
            i = int(route.route_ref.split("_")[1])
            request, port = self.requests[i], self.ports[i, route.path]
            content = qualifier.admit(request, b"Invented qualification")
            result = qualifier.dispatch(request, content, path=route.path, route_ref=route.route_ref)
            self.qualifications.append((request, content, result, port.sent[-1]))
            port.sent.clear()
        self.qualifier = qualifier
        self.ports[0, "main"].output = canonical(dict(id="invented_response", object="chat.completion", created=1,
            model="invented_model_0", choices=[dict(index=0, finish_reason="tool_calls", message=dict(role="assistant",
                content=None, tool_calls=[dict(id="invented-call-1", type="function",
                function=dict(name="delegate_task", arguments=ARGUMENTS))]))],
            usage=dict(prompt_tokens=5, completion_tokens=3, total_tokens=8)))
        self.ports[3, "main"].output = canonical(dict(id="invented_child_reply", object="chat.completion", created=12,
            model="invented_model_3", choices=[dict(index=0, finish_reason="stop",
                message=dict(role="assistant", content="Three blue squares."))],
            usage=dict(prompt_tokens=3, completion_tokens=2, total_tokens=5)))
        self.adapter = bind_controlled_main(self.context, selection=self.selection_ports[0], result_policy=self.result_port,
            producer_ref="producer_0", main_route_ref="route_0_main", profiles=self.profiles, routes=self.routes,
            transports=self.transports, clock=self.clock)
        with content_policy_scope(self.context):
            self.agent = AIAgent(controlled_main=self.adapter, provider="synthetic_native", api_mode="chat_completions",
                model="invented_model_0", session_id="invented_source_session", enabled_toolsets=[], quiet_mode=True,
                skip_memory=True, skip_context_files=True, skip_background_review=True)
            if self.before_probe is not None:
                self.before_probe(self)
            token = self.adapter.bind_tool_call_probe(self.agent, system_message="Request one leaf call.",
                user_message="Count blue squares.", task_id="execution", expected_arguments_json=ARGUMENTS)
            self.observation = self.adapter.run_tool_call_probe(self.agent, token)
        _construction_seconds.append(time.monotonic() - started)
        return self

    def bind(self, **changes):
        options = dict(execution_request=self.requests[1], execution_selection=self.selection_ports[1],
            execution_producer_ref="producer_1", execution_main_route_ref="route_1_main",
            receiver_request=self.requests[2], receiver_selection=self.selection_ports[2], receiver_producer_ref="producer_2",
            child_request=self.requests[3], child_selection=self.selection_ports[3], child_producer_ref="producer_3",
            delegate_route_ref="route_2_delegate", child_main_route_ref="route_3_main",
            child_session_id="invented_child_session", child_task_id="child_execution")
        options.update(changes)
        with content_policy_scope(self.context):
            self.token = self.adapter.bind_issued_call_execution(self.observation, **options)
        return self

    def run(self, token=None):
        with content_policy_scope(self.context):
            return self.adapter.run_issued_call_execution(self.token if token is None else token)


class CompactWorld(IssuedWorld):
    def __init__(self):
        super().__init__()
        self.bindings = tuple(replace(holder, selection_id=self.bindings[0 if i < 2 else 2].selection_id,
            selection_version=self.bindings[0 if i < 2 else 2].selection_version)
            for i, holder in enumerate(self.bindings))
        self.routes = tuple(replace(route, selection_id=self.bindings[i].selection_id,
            selection_version=self.bindings[i].selection_version) for route in self.routes
            for i in (int(route.route_ref.split("_")[1]),) if i in (0, 2) or route.path == "main")
        self.selections = [replace(selected, selection_id=holder.selection_id, selection_version=holder.selection_version)
            for selected, holder in zip(self.selections, self.bindings)]
        self.profiles = (replace(self.profiles[0], allowed_routes=tuple(r.route_ref for r in self.routes)),)


def test_actual_issued_call_to_constructed_leaf_result(no_delegation_effects, record_property):
    world = CompactWorld().construct()
    world.agent.close()
    world.agent.close()
    p, runtime = _tool_call_probes[world.adapter], world.adapter._runtime
    source = (p.phase, p.attempted, p.records, p.record_anchors, p.binding, p.stage_index,
              p.proof, p.constructed.state, p.constructed.anchors, world.adapter.send_count)
    world.bind()
    world.bindings = tuple(replace(row, profiles=()) for row in world.bindings)
    result = world.run()
    owner, invocation = _issued_call_frames[world.adapter], result.invocation
    assert type(result) is NativeIssuedCallResult and world.adapter.issued_call_results == (result,)
    assert p.closed and p.constructed.closed
    assert result is owner.result_anchors[0][0] and result.binding is owner.registered_binding
    from agent.native_main_payload import _delegate_frames
    assert owner.frame is result.binding.anchors[-1] is _delegate_frames[result.binding.adapter]
    assert owner.frame.binding is result.binding.anchors[-2] is result.binding.delegation_binding
    assert result.invocation is owner.frame.registered_record
    assert result.observation is world.observation and result.invocation is owner.frame.record
    assert result.return_facts is invocation.return_facts and result.closed_child_proof is owner.frame.closed_proof
    assert (p.phase, p.attempted, p.records, p.record_anchors, p.binding, p.stage_index,
            p.proof, p.constructed.state, p.constructed.anchors, world.adapter.send_count) == source
    assert owner.phase == "released" and owner.active_call is None and owner.stage_index == 8
    assert owner.frame.phase == "released" and owner.frame.terminal and not owner.frame.running
    assert invocation.child_adapter._constructed.closed and invocation.child_completion.result["completed"] is True
    assert invocation.child_completion.result["final_response"] == "Three blue squares."
    assert len(invocation.child_completion.result) == 29
    assert invocation.child_completion.payload == canonical(invocation.child_completion.result)
    assert invocation.child_completion.result is invocation.child_adapter._completions[0].result
    assert result.aggregate.encode("utf-8") is not invocation.aggregate_bytes
    assert result.aggregate.encode("utf-8") == invocation.aggregate_bytes
    assert runtime._content(result.result_content).payload is result.result_bytes
    b = result.binding
    expected_parents = {
        b.authorization_content: (), b.issued_content: (world.observation.observation_content, b.authorization_content),
        invocation.raw_content: (b.issued_content,), invocation.leaf_content: (invocation.raw_content,),
        invocation.request_content: (invocation.raw_content, invocation.leaf_content),
        invocation.child_seed: (invocation.accepted_content,),
        invocation.child_input.admitted: (invocation.child_seed,),
        result.return_facts.returned_content: (invocation.request_content, invocation.accepted_content, invocation.receiver_result),
        result.result_content: (b.issued_content, result.return_facts.returned_content),
    }
    for content, parents in expected_parents.items():
        assert runtime._content(content).parents == parents
    assert runtime._content(invocation.receiver_result).parents == (
        invocation.accepted_content, invocation.child_completion.content, owner.frame.outputs[-2][1])
    assert runtime._content(owner.frame.outputs[-2][1]).parents == (
        invocation.accepted_content, invocation.child_completion.content)
    rows, positional, keywords = invocation.arguments
    assert not positional and tuple(k for k, _ in keywords) == ("goal", "context", "role", "parent_agent")
    assert all(present is (name in {"goal", "context", "role", "parent_agent"}) for name, present, _ in rows)
    assert len(rows) == 12
    z = json.loads(result.result_bytes)
    assert set(z) == {"call", "execution_arguments", "observation", "result"}
    assert z["call"] == json.loads(world.observation.observation_bytes)["call"]
    assert z["call"]["function"]["arguments"] == ARGUMENTS
    assert z["execution_arguments"] == json.loads(invocation.raw_bytes)
    assert z["result"] == json.loads(result.aggregate)
    @cache
    def ancestors(handle):
        return {handle}.union(*(ancestors(p) for p in runtime._content(handle).parents))
    assert set(expected_parents) <= ancestors(result.result_content)
    assert all(row.profiles == ("restricted_documents",) for row in runtime._contents.values())
    assert owner.adapter._runtime is runtime is invocation.child_adapter._runtime
    assert owner.adapter.send_count == 0 and world.adapter.send_count == owner.frame.delegate_sends == invocation.child_adapter.send_count == 1
    assert sum(len(p.sent) for p in dict.fromkeys(world.ports.values())) == 3
    assert len(world.qualifications) == len(world.routes) == 18
    assert len(owner.partial_responses) == 2
    assert owner.partial_responses[0][2] is world.ports[2, "delegate"].output
    assert owner.partial_responses[1][2] is world.ports[3, "main"].output
    indices = {row.handle: i + 1 for i, row in enumerate(runtime._contents.values())}
    record_property("positive_proof", json.dumps(dict(qualification_sends=len(world.qualifications), source_sends=1,
        delegate_sends=1, child_sends=1, execution_main_sends=0, content_slots=len(runtime._contents),
        envelopes=len(runtime._envelopes), retained_bytes=runtime._retained_bytes,
        windows=dict(source=p.stage_index, execution=owner.stage_index),
        native_stages=dict(delegate_inputs=len(owner.frame.inputs), delegate_outputs=len(owner.frame.outputs),
            child_turns=len(invocation.child_adapter._turn_frames), child_results=len(invocation.child_adapter._completions)),
        stages=[dict(bytes=len(row.payload), parents=[indices[p] for p in row.parents], path=row.path,
            binding=row.binding, profiles=row.profiles) for row in runtime._contents.values()],
        payloads=[row.payload.decode() for row in runtime._contents.values()]), sort_keys=True))
    original = result
    with pytest.raises(NativeContentPolicyDenied):
        world.run()
    assert world.adapter.issued_call_results[0] is original and p.phase == "released"
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(owner.context):
            owner.agent.run_conversation("Unissued work", task_id="execution")
    assert sum(len(p.sent) for p in dict.fromkeys(world.ports.values())) == 3


def cheap_source():
    from tests.tools.test_native_tool_call_probe import ProbeWorld
    started = time.monotonic()
    world = ProbeWorld().construct()
    with content_policy_scope(world.context):
        world.bind_probe()
        observation = world.run()
    _construction_seconds.append(time.monotonic() - started)
    world.observation = observation
    world.issued_options = dict(execution_request=world.ports["title"], execution_selection=world.selection_port,
        execution_producer_ref="invented_producer", execution_main_route_ref="main_route",
        receiver_request=world.ports["delegate"], receiver_selection=world.selection_port,
        receiver_producer_ref="invented_producer", child_request=world.ports["a2a"], child_selection=world.selection_port,
        child_producer_ref="invented_producer", delegate_route_ref="route_delegate", child_main_route_ref="main_route",
        child_session_id="invented_child_session", child_task_id="execution")
    return world


@pytest.mark.parametrize("mode", ["copied-observation", "scalar-subclass", "wrong-thread"])
def test_pre_callback_source_identity_and_scalars(no_delegation_effects, mode):
    world = cheap_source()
    if mode == "copied-observation":
        world.observation = replace(world.observation)
    elif mode == "scalar-subclass":
        class Scalar(str):
            pass
        world.issued_options["execution_producer_ref"] = Scalar("invented_producer")
    errors = []
    with content_policy_scope(world.context):
        def forbidden():
            if _issued_call_frames.get(world.adapter) is not None:
                pytest.fail("Invalid input reached an external callback")
        world.admission_hook = forbidden
        if mode == "wrong-thread":
            ctx = copy_context()
            def invoke():
                try:
                    ctx.run(world.adapter.bind_issued_call_execution, world.observation, **world.issued_options)
                except BaseException as error:
                    errors.append(error)
            thread = threading.Thread(target=invoke)
            thread.start()
            thread.join()
            assert len(errors) == 1 and type(errors[0]) is NativeContentPolicyDenied
        else:
            with pytest.raises(NativeContentPolicyDenied):
                world.adapter.bind_issued_call_execution(world.observation, **world.issued_options)
        world.admission_hook = None
    owner = _issued_call_frames[world.adapter]
    assert owner.phase == "denied" and owner.adapter is None and owner.authorization_content is None
    assert _tool_call_probes[world.adapter].phase == "released" and world.adapter.send_count == 1


@pytest.mark.parametrize("index", [0, 1, 2, 3], ids=["source", "execution", "receiver", "child"])
def test_independent_holder_drift_refuses_before_admission(no_delegation_effects, index):
    world = CompactWorld().construct()
    world.bindings = tuple(replace(row, auth_epoch=row.auth_epoch + 1) if i == index else row
                           for i, row in enumerate(world.bindings))
    with pytest.raises(NativeContentPolicyDenied):
        world.bind()
    owner = _issued_call_frames.get(world.adapter)
    if index == 0:
        assert owner is None and world.context._denied
    else:
        assert owner.phase == "denied" and owner.authorization_content is None and owner.adapter is None
    assert world.adapter.send_count == 1 and len(world.adapter._runtime._contents) == 5


def live_alternates(world, selected):
    world.routes = tuple(replace(route, expires_at=100 if route.route_ref == selected else 200)
                         for route in world.routes)
    route = next(r for r in world.routes if r.route_ref == selected)
    world.routes += (replace(route, route_ref=selected + "_alternate", expires_at=200),)
    world.profiles = (replace(world.profiles[0], allowed_routes=tuple(r.route_ref for r in world.routes)),)
    return world


@pytest.mark.parametrize("selected", ["route_1_main", "route_2_delegate", "route_3_main"])
def test_selected_expiry_refuses_live_alternate(no_delegation_effects, selected):
    world = live_alternates(CompactWorld(), selected).construct()
    world.now = 100
    with pytest.raises(NativeContentPolicyDenied, match="selected_expired_or_denied"):
        world.bind()
    owner = _issued_call_frames[world.adapter]
    assert owner.authorization_content is None and owner.adapter is None
    assert world.adapter.send_count == 1 and len(world.qualifications) == 19


@pytest.mark.parametrize("kind", ["admission", "selection", "producer", "clock", "resolve", "send"])
def test_callback_reentry_has_no_execution_window(no_delegation_effects, kind):
    world = CompactWorld().construct()
    errors = []
    def reenter(*args):
        owner = _issued_call_frames.get(world.adapter)
        if owner is None or owner.active_call is None:
            return
        assert owner.depth == 1
        try:
            world.adapter.run_issued_call_execution(owner.token)
        except NativeContentPolicyDenied as error:
            errors.append(error)
    if kind == "resolve":
        world.ports[0, "main"].resolve_hook = reenter
    elif kind == "send":
        world.ports[2, "delegate"].hook = reenter
    else:
        world.hooks[kind] = reenter
    with pytest.raises(NativeContentPolicyDenied):
        world.bind()
        world.run()
    owner = _issued_call_frames[world.adapter]
    assert errors and owner.phase == "denied" and owner.active_call is None
    assert _tool_call_probes[world.adapter].phase == "released"
    assert world.adapter.send_count == 1
    assert sum(len(p.sent) for p in dict.fromkeys(world.ports.values())) == (2 if kind == "send" else 1)
    if kind == "send":
        assert len(owner.partial_responses) == 1
        assert owner.partial_responses[0][2] is world.ports[2, "delegate"].output
        frame, adapter, envelope, operation, pending = owner.partial_operations[0]
        assert frame is owner.registered_frame and adapter is owner.adapter
        assert envelope is frame.record.envelope and operation is frame.partial_sends[0][2]
        assert pending[0] is envelope and adapter._pending is adapter._operation is adapter._checked_operation is None
        assert frame.record.ack_content is frame.record.accepted_content is frame.record.return_facts is None


def test_original_runner_clock_control_and_partial_delegate(no_delegation_effects, record_property):
    from agent.native_main_payload import _DelegateFrame, _controlled_delegate_runner
    world = CompactWorld().construct().bind()
    original = asyncio.CancelledError("Invented guarded clock control")
    guarded = []
    def stop():
        current = sys._getframe()
        while current is not None:
            if (current.f_code is _DelegateFrame.call.__code__
                    and current.f_back.f_code is _controlled_delegate_runner.__wrapped__.__code__):
                owner = _issued_call_frames[world.adapter]
                assert owner.depth == 1 and owner.frame.child is not None
                guarded.append(current.f_locals["callback"])
                raise original
            current = current.f_back
    world.hooks["clock"] = stop
    with pytest.raises(asyncio.CancelledError) as observed:
        world.run()
    owner = _issued_call_frames[world.adapter]
    assert observed.value is original is owner.control_exception
    assert guarded and owner.frame.delegate_sends == 1 and owner.frame.child_adapter.send_count == 0
    assert owner.frame.record.accepted_content is not None and owner.frame.record.child_seed is not None
    assert owner.frame.record.child_completion is None and owner.phase == "denied"
    record_property("partial_proof", json.dumps(dict(delegate_sends=1, child_sends=0,
        content_slots=len(owner.runtime._contents), retained_bytes=owner.runtime._retained_bytes,
        envelopes=len(owner.runtime._envelopes), exact_clock=True, exact_control=True)))


@pytest.mark.parametrize("mode", ["expiry", "health"])
def test_late_release_retains_result_and_closed_child(no_delegation_effects, mode, record_property):
    world = live_alternates(CompactWorld(), "route_0_main").construct().bind()
    owner = _issued_call_frames[world.adapter]
    poisoned = []
    def stop():
        if owner.active_call is None or owner.stage_index != 7:
            return
        world.hooks.pop("clock", None)
        if mode == "expiry":
            world.now = 100
        else:
            def wrong_thread():
                from hermes_content_policy import PolicyError
                try:
                    owner.runtime.admit(world.requests[0], b"Invented unauthorized thread")
                except PolicyError as error:
                    poisoned.append(error)
            thread = threading.Thread(target=wrong_thread)
            thread.start()
            thread.join()
    world.hooks["clock"] = stop
    with pytest.raises(NativeContentPolicyDenied):
        world.run()
    assert owner.phase == "denied" and owner.result_content is not None and owner.result_anchors
    original = world.adapter.issued_call_results[0]
    assert original.invocation.child_adapter._constructed.closed and original.invocation.child_completion is not None
    assert original.return_facts is owner.frame.record.return_facts
    assert owner.runtime._content(original.result_content).payload is original.result_bytes
    assert len(owner.partial_responses) == 2 and owner.frame.delegate_sends == 1
    assert original.invocation.child_adapter.send_count == 1 and world.adapter.send_count == 1
    assert _tool_call_probes[world.adapter].phase == "released"
    assert mode == "expiry" or (poisoned and owner.runtime._poisoned)
    record_property("partial_proof", json.dumps(dict(mode=mode, source_sends=1, delegate_sends=1, child_sends=1,
        result_retained=True, closed_child=True, retained_bytes=owner.runtime._retained_bytes,
        content_slots=len(owner.runtime._contents), envelopes=len(owner.runtime._envelopes))))


def test_copied_token_spends_attempt_without_execution(no_delegation_effects):
    world = CompactWorld().construct().bind()
    with pytest.raises(TypeError):
        NativeIssuedCallInput()
    with pytest.raises(NativeContentPolicyDenied):
        world.run(object.__new__(NativeIssuedCallInput))
    owner = _issued_call_frames[world.adapter]
    assert owner.phase == "denied" and not owner.frame.partial_sends and owner.frame.record is None
    before = owner.runtime._retained_bytes
    with pytest.raises(NativeContentPolicyDenied, match="attempt_spent"):
        world.bind()
    assert owner.runtime._retained_bytes == before and not world.adapter.issued_call_results


def test_genuine_capacity_refusal_keeps_charges(no_delegation_effects, record_property):
    from hermes_content_policy import MAX_PAYLOAD, MAX_RETAINED
    world = CompactWorld()
    fills = []
    def fill(w):
        runtime = w.adapter._runtime
        remaining = MAX_RETAINED - MAX_PAYLOAD - 5000
        while remaining:
            size = min(remaining, MAX_PAYLOAD)
            fills.append(runtime.admit(w.requests[0], b"x" * size))
            remaining -= size
    world.before_probe = fill
    world.construct().bind()
    with pytest.raises(NativeContentPolicyDenied):
        world.run()
    owner = _issued_call_frames[world.adapter]
    assert len(fills) == 7 and owner.issued_content is not None and owner.frame.record.raw_content is not None
    assert owner.frame.delegate_sends == 0 and not owner.frame.partial_sends
    assert world.adapter.send_count == 1 and owner.runtime._retained_bytes > MAX_RETAINED - MAX_PAYLOAD
    record_property("partial_proof", json.dumps(dict(fill_admissions=len(fills), source_sends=1, delegate_sends=0,
        retained_bytes=owner.runtime._retained_bytes, content_slots=len(owner.runtime._contents),
        envelopes=len(owner.runtime._envelopes))))


@pytest.mark.parametrize("change", ["background", "goal"])
def test_actual_capture_refuses_unselected_arguments(no_delegation_effects, monkeypatch, change):
    from functools import wraps
    from tools import delegate_tool
    world = CompactWorld().construct()
    original = delegate_tool.delegate_task
    @wraps(original)
    def altered(*args, **kwargs):
        if change == "background":
            kwargs["background"] = False
        else:
            kwargs["goal"] = "Count red squares."
        return original(*args, **kwargs)
    monkeypatch.setattr(delegate_tool, "delegate_task", altered)
    world.bind()
    with pytest.raises(NativeContentPolicyDenied):
        world.run()
    owner = _issued_call_frames[world.adapter]
    assert owner.phase == "denied" and owner.frame.record is None and not owner.frame.inputs
    assert owner.issued_content is not None and not owner.frame.partial_sends
    assert world.adapter.send_count == 1


@pytest.mark.parametrize("change", ["frame", "method"])
def test_original_execution_identity_refuses_copies(no_delegation_effects, change):
    import copy
    from types import FunctionType, MethodType
    world = CompactWorld().construct().bind()
    owner = _issued_call_frames[world.adapter]
    original = owner.registered_frame
    if change == "frame":
        owner.frame = copy.copy(original)
    else:
        function = owner.adapter._identity.__func__
        owner.adapter._identity = MethodType(FunctionType(function.__code__, function.__globals__, function.__name__),
                                            owner.adapter)
    with pytest.raises(NativeContentPolicyDenied):
        world.run()
    assert owner.phase == "denied" and owner.binding.anchors[-1] is original
    assert original.record is None and not original.partial_sends and not world.adapter.issued_call_results
    assert world.adapter.send_count == 1 and len(owner.runtime._contents) == 7
