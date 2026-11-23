"""Exercise the actual native leaf with independent three-holder declarations."""

import json
import asyncio
from functools import wraps
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
from hermes_content_policy import (
    PATHS, ContentPolicyRuntime, Destination, ExecutionBinding, ProfilePolicy, RouteBinding,
)

from agent.content_policy import NativeContentContext, NativeContentPolicyDenied, content_policy_scope
from agent.native_main_payload import NativeMainSelection, NativeResultPolicy, bind_controlled_main
from tests.agent.test_native_controlled_conversation import excluded as excluded
from tests.agent.test_native_main_payload import invented_offline_configuration as invented_offline_configuration


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":")).encode()


class CaseDurations:
    def pytest_runtest_makereport(self, item, call):
        item.user_properties.append((f"{call.when}_seconds", round(call.duration, 6)))


@pytest.fixture(scope="module", autouse=True)
def retain_case_durations(request):
    plugin = CaseDurations()
    request.config.pluginmanager.register(plugin)
    yield
    request.config.pluginmanager.unregister(plugin)


class DeclaredPort:
    def __init__(self, destination, output):
        self.destination, self.output = destination, output
        self.qualification = []
        self.sent = []
        self.qualifying = False
        self.hook = None

    def resolve(self):
        return self.destination

    def send(self, envelope):
        (self.qualification if self.qualifying else self.sent).append(envelope)
        if self.hook is not None:
            self.hook(envelope)
        return self.output


class ThreeHolders:
    def __init__(self):
        self.now = 10
        self.requests = tuple(object() for _ in range(3))
        self.bindings = (
            ExecutionBinding("invented_owner", "invented_parent_execution", "invented_lineage",
                "invented_parent", 7, 9, "parent_selection", 3, ("restricted_documents",)),
            ExecutionBinding("invented_owner", "invented_child_execution", "invented_lineage",
                "invented_child", 7, 9, "receiver_selection", 4, ("restricted_documents",)),
            ExecutionBinding("invented_owner", "invented_child_execution", "invented_lineage",
                "invented_child", 7, 9, "child_selection", 5, ("restricted_documents",)),
        )
        self.ports, self.routes, self.port_bindings = {}, [], []
        self.selections = []
        for index, binding in enumerate(self.bindings):
            for path in PATHS:
                model = ("invented_parent_model", "invented_receiver_model", "invented_child_model")[index]
                destination = Destination(f"invented_{index}_{path}", "synthetic_native", model,
                                          f"invented_endpoint_{index}", 1)
                if index == 2 and path == "main":
                    destination = Destination("invented_child_main", "synthetic_native", model, "invented_child_endpoint", 1)
                port = DeclaredPort(destination, b'{"accepted":true}')
                self.ports[index, path] = port
                self.port_bindings.append((port, index))
                self.routes.append(RouteBinding(f"route_{index}_{path}", binding.selection_id,
                    binding.selection_version, binding.consumer_ref, path, destination, 100,
                    f"synthetic_inprocess_{index}_{path}", "synthetic_only"))
                if path == "main":
                    self.selections.append(NativeMainSelection(binding.selection_id,
                        binding.selection_version, "chat_completions", destination))
        self.delegate = DeclaredPort(Destination("invented_delegate", "synthetic_native", "invented_receiver",
            "invented_delegate_endpoint", 1), b'{"accepted":true}')
        self.ports["issued"] = self.delegate
        self.port_bindings.append((self.delegate, 1))
        self.routes.append(RouteBinding("issued_delegate", "receiver_selection", 4, "invented_child", "delegate",
            self.delegate.destination, 100, "synthetic_issued_delegate", "synthetic_only"))
        self.forbidden = DeclaredPort(Destination("invented_denied", "synthetic_native", "invented_child_model",
            "invented_denied_endpoint", 1), b'{"accepted":true}')
        self.ports["denied"] = self.forbidden
        self.port_bindings.append((self.forbidden, 2))
        self.routes.append(RouteBinding("denied_child", "child_selection", 5, "invented_child", "main",
            self.forbidden.destination, 100, "synthetic_denied_main", "synthetic_only"))
        self.routes = tuple(self.routes)
        self.profiles = (ProfilePolicy("restricted_documents",
            tuple(r.route_ref for r in self.routes if r.route_ref != "denied_child"), PATHS),)
        self.transports = tuple((p.destination.transport_ref, p) for p in self.ports.values())
        self.policies = tuple(NativeResultPolicy(f"invented_producer_{i}", 4, ("restricted_documents",)) for i in range(3))
        self.admission = SimpleNamespace(current=self.current)
        self.selection = SimpleNamespace(current=self.selected)
        self.policy = SimpleNamespace(current=self.produced)
        self.clock = self.time
        self.qualifications = []
        self.after_accept = None
        self.after_complete = None
        self.context = NativeContentContext(self.admission, self.requests[0])
        self.ports[2, "main"].output = canonical(dict(id="invented_reply", object="chat.completion", created=12,
            model="invented_child_model", choices=[dict(index=0, finish_reason="stop",
                message=dict(role="assistant", content="Two blue squares."))],
            usage=dict(prompt_tokens=3, completion_tokens=2, total_tokens=5)))

    def time(self):
        if self.after_complete is not None and hasattr(self, "adapter"):
            from agent.native_main_payload import _delegate_frames
            frame = _delegate_frames.get(self.adapter)
            if frame is not None and frame.record is not None and frame.record.child_completion is not None:
                callback, self.after_complete = self.after_complete, None
                callback(frame)
        return self.now

    def current(self, handle):
        for index, request in enumerate(self.requests):
            if request is handle:
                return self.bindings[index]
        for port, index in self.port_bindings:
            if port is handle:
                return self.bindings[index]
        raise AssertionError("Unknown capability")

    def selected(self, handle):
        if self.after_accept is not None and hasattr(self, "adapter"):
            from agent.native_main_payload import _delegate_frames
            frame = _delegate_frames.get(self.adapter)
            if frame is not None and frame.record is not None and frame.record.accepted_content is not None:
                callback, self.after_accept = self.after_accept, None
                callback(frame)
        for index, request in enumerate(self.requests):
            if request is handle:
                return self.selections[index]
        raise AssertionError("Unknown selection capability")

    def produced(self, ref):
        return next(p for p in self.policies if p.producer_ref == ref)

    def bind(self):
        from run_agent import AIAgent
        runtime = ContentPolicyRuntime(admission=self.admission, profiles=self.profiles,
            routes=self.routes, transports=self.transports, clock=self.clock, policy_revision=9)
        for port in self.ports.values():
            port.qualifying = True
        for i, request in enumerate(self.requests):
            for path in PATHS:
                content = runtime.admit(request, b"Invented qualification")
                result = runtime.dispatch(request, content, path=path, route_ref=f"route_{i}_{path}")
                self.qualifications.append((runtime, request, path, result, self.ports[i, path].qualification[-1]))
        content = runtime.admit(self.requests[0], b"Invented delegate qualification")
        result = runtime.dispatch(self.requests[0], content, path="delegate", route_ref="issued_delegate")
        self.qualifications.append((runtime, self.requests[0], "delegate", result, self.delegate.qualification[-1]))
        for port in self.ports.values():
            port.qualifying = False
        self.adapter = bind_controlled_main(self.context, selection=self.selection, result_policy=self.policy,
            producer_ref="invented_producer_0", main_route_ref="route_0_main", profiles=self.profiles,
            routes=self.routes, transports=self.transports, clock=self.clock)
        with content_policy_scope(self.context):
            self.parent = AIAgent(controlled_main=self.adapter, provider="synthetic_native",
                api_mode="chat_completions", model="invented_parent_model", session_id="invented_parent_session",
                enabled_toolsets=[], quiet_mode=True, skip_memory=True, skip_context_files=True,
                skip_background_review=True)
            self.adapter.bind_delegation(self.parent, receiver_request=self.requests[1], receiver_selection=self.selection,
                receiver_producer_ref="invented_producer_1", child_request=self.requests[2], child_selection=self.selection,
                child_producer_ref="invented_producer_2", delegate_route_ref="issued_delegate",
                child_main_route_ref="route_2_main", child_session_id="invented_child_session",
                child_task_id="invented_child_execution")
        return self

    def call(self, **changes):
        from tools.delegate_tool import delegate_task
        kwargs = dict(goal="Count blue squares", context="two blue squares", parent_agent=self.parent)
        kwargs.update(changes)
        with content_policy_scope(self.context):
            return delegate_task(**kwargs)


@pytest.fixture
def no_delegation_effects(excluded, monkeypatch):
    from tools import delegate_tool, daemon_pool
    from agent import subagent_lifecycle
    for name in ("_load_config", "_resolve_delegation_credentials", "_get_max_spawn_depth",
                 "_get_max_concurrent_children", "_resolve_workspace_hint", "_build_child_preserving_parent_tools",
                 "_finalize_child_results", "_apply_summary_budget"):
        monkeypatch.setattr(delegate_tool, name, excluded)
    monkeypatch.setattr(daemon_pool, "DaemonThreadPoolExecutor", excluded)
    monkeypatch.setattr(subagent_lifecycle, "bind_subagent_parent", excluded)
    return excluded


def test_blue_squares_actual_native_path(no_delegation_effects, monkeypatch):
    from agent.native_main_payload import _delegate_frames
    from agent import conversation_loop, chat_completion_helpers, native_main_payload
    from run_agent import AIAgent
    from tools import delegate_tool
    constructor, public = AIAgent.__init__, AIAgent.run_conversation
    observed = {}
    for module, name in ((conversation_loop, "_try_controlled_text_turn"),
        (conversation_loop, "build_turn_context"), (chat_completion_helpers, "interruptible_api_call"),
        (native_main_payload, "_complete_controlled_turn"), (delegate_tool, "_build_child_agent"),
        (delegate_tool, "_run_single_child"), (delegate_tool, "_build_child_system_prompt")):
        original = getattr(module, name)
        observed[name] = []
        def observer(original, name):
            @wraps(original)
            def call(*args, **kwargs):
                observed[name].append(threading.get_ident())
                return original(*args, **kwargs)
            return call
        monkeypatch.setattr(module, name, observer(original, name))
    world = ThreeHolders().bind()
    result = world.call()
    assert all(rows == [threading.get_ident()] for rows in observed.values())
    assert AIAgent.__init__ is constructor and AIAgent.run_conversation is public
    expected = '{"results": [{"task_index": 0, "status": "completed", "summary": "Two blue squares.", "api_calls": 1, "duration_seconds": 0.0, "model": "invented_child_model", "exit_reason": "completed", "truncated": false, "tokens": {"input": 3, "output": 2}, "tool_trace": []}], "total_duration_seconds": 0.0}'
    assert result == expected
    frame = _delegate_frames[world.adapter]
    record = frame.record
    assert type(record.child) is AIAgent and record.child_adapter._constructed.agent is record.child
    assert record.leaf_bytes == b'{"context":"two blue squares","goal":"Count blue squares","role":"leaf"}'
    assert record.child_adapter._runtime is world.adapter._runtime
    assert record.child_completion.result["cost_status"] == "unknown"
    assert record.child_completion.result["estimated_cost_usd"] is None
    assert record.child_completion.invocation is record.child_invocation
    assert record.child_invocation.envelope is world.ports[2, "main"].sent[0]
    assert record.return_facts.envelope is world.delegate.sent[0]
    assert frame.phase == "released" and frame.child_adapter._constructed.closed
    assert len(world.delegate.sent) == len(world.ports[2, "main"].sent) == 1
    assert sum(len(p.sent) for p in world.ports.values()) == 2
    assert len(world.qualifications) == sum(len(p.qualification) for p in world.ports.values()) == 25
    assert all(row[0] is not world.adapter._runtime for row in world.qualifications)
    runtime = world.adapter._runtime
    assert runtime._content(record.accepted_content).parents == (record.request_content,)
    assert runtime._content(record.child_seed).parents == (record.accepted_content,)
    assert runtime._content(record.child_input.admitted).parents == (record.child_seed,)
    assert runtime._content(record.return_facts.returned_content).parents == (
        record.request_content, record.accepted_content, record.receiver_result)
    assert runtime._content(record.return_facts.returned_content).payload == expected.encode()
    assert record.child_completion.result["final_response"] == record.child_invocation.native_response.choices[0].message.content
    assert record.child_adapter._system_prefix[0] is frame.prompt
    assert json.loads(runtime._content(record.request_content).payload)["system_prompt"] == frame.prompt
    assert len(frame.inputs) <= 8 and len(frame.outputs) <= 8
    record.child.close()
    world.adapter.validate_scope()
    assert not world.context._denied
    from agent.content_policy import current_context
    assert current_context() is None
    world.bindings = tuple(replace(b, profiles=()) for b in world.bindings)
    world.adapter.validate_scope()
    for operation in (
        lambda: world.parent.run_conversation("later", task_id="invented_parent_execution"),
        lambda: world.adapter.bind_auxiliary(world.parent, path="title", task="title",
            request_handle=world.requests[1], selection=world.selection,
            producer_ref="invented_producer_1", route_ref="route_1_title", language="en"),
        lambda: world.adapter.bind_fallback(world.parent, request_handle=world.requests[1],
            selection=world.selection, producer_ref="invented_producer_1", route_ref="route_1_fallback"),
        world.call,
    ):
        with pytest.raises(NativeContentPolicyDenied):
            operation()
    assert len(world.delegate.sent) == len(world.ports[2, "main"].sent) == 1
    world.parent.close()
    world.parent.close()


class Hostile:
    def __eq__(self, other):
        pytest.fail("Custom equality ran")

    def __bool__(self):
        pytest.fail("Custom Boolean conversion ran")

    def __str__(self):
        pytest.fail("Custom string conversion ran")

    def __repr__(self):
        pytest.fail("Custom representation ran")


class TextSubclass(str):
    pass


@pytest.mark.parametrize("changes", [
    {"tasks": []}, {"tasks": "[]"}, {"tasks": [{"goal": "invented"}]},
    {"goal": ""}, {"goal": " "}, {"goal": TextSubclass("invented")},
    {"goal": Hostile()}, {"context": Hostile()}, {"context": TextSubclass("invented")},
    {"role": "orchestrator"}, {"role": " leaf"}, {"role": Hostile()},
    {"background": True}, {"background": 0}, {"background": Hostile()},
    {"action": "list"}, {"action": "spawn "}, {"action": Hostile()},
    {"max_iterations": 1}, {"output_schema": {}}, {"subagent_id": "invented"},
    {"message": "invented"}, {"credentials_cfg": Hostile()}, {"unknown": Hostile()},
    {"goal": "x" * 16385},
], ids=["empty-list", "string-tasks", "batch", "empty-goal", "blank-goal", "goal-subclass",
    "goal-object", "context-object", "context-subclass", "orchestrator", "role-space", "role-object",
    "background", "background-int", "background-object", "control", "action-space", "action-object",
    "iterations", "schema", "subagent", "message", "credentials", "unknown", "text-bound"])
def test_modes_and_exact_types_refuse_without_effects(no_delegation_effects, changes):
    world = ThreeHolders().bind()
    with pytest.raises((NativeContentPolicyDenied, TypeError)):
        world.call(**changes)
    assert all(not p.sent for p in world.ports.values())
    assert world.context._denied


def test_positional_and_explicit_none_capture(no_delegation_effects):
    from tools.delegate_tool import delegate_task
    from agent.native_main_payload import _delegate_frames
    import inspect
    assert inspect.signature(delegate_task) == inspect.signature(delegate_task.__wrapped__)
    world = ThreeHolders().bind()
    goal, context = "Count blue squares", "two blue squares"
    with content_policy_scope(world.context):
        result = delegate_task(goal, context, role=None, action=None, tasks=None,
            background=False, parent_agent=world.parent, credentials_cfg=None)
    frame = _delegate_frames[world.adapter]
    rows, positional, keywords = frame.record.arguments
    assert positional == (goal, context) and positional[0] is goal
    assert tuple(n for n, _ in keywords) == ("role", "action", "tasks", "background", "parent_agent", "credentials_cfg")
    assert len(rows) == 12 and rows[3] == ("max_iterations", False, None)
    assert rows[2] == ("tasks", True, None)
    assert rows[10][2] is world.parent
    assert json.loads(result)["results"][0]["summary"] == "Two blue squares."


@pytest.mark.parametrize("change", ["receiver-epoch", "child-version", "receiver-producer", "destination", "expiry", "catalog", "method"])
def test_original_holder_and_catalog_freshness(no_delegation_effects, change):
    world = ThreeHolders().bind()
    if change == "receiver-epoch":
        world.bindings = (world.bindings[0], replace(world.bindings[1], auth_epoch=8), world.bindings[2])
    elif change == "child-version":
        world.bindings = (*world.bindings[:2], replace(world.bindings[2], selection_version=6))
    elif change == "receiver-producer":
        world.policies = (world.policies[0], replace(world.policies[1], revision=5), world.policies[2])
    elif change == "destination":
        world.ports[2, "main"].destination = replace(world.ports[2, "main"].destination, endpoint_ref="other")
    elif change == "expiry":
        world.now = 100
    elif change == "catalog":
        world.adapter._operator_routes = tuple(list(world.routes))
    else:
        world.selection.current = lambda handle: world.selections[0]
    with pytest.raises(NativeContentPolicyDenied):
        world.call()
    assert all(not p.sent for p in world.ports.values())


@pytest.mark.parametrize("change", ["retained-profile", "mutated-leaf", "copied-record", "copied-token", "construction-slot"])
def test_post_acceptance_original_anchors(no_delegation_effects, change):
    world = ThreeHolders().bind()
    def alter(frame):
        assert frame.record.accepted_content is not None
        if change == "retained-profile":
            world.bindings = tuple(replace(b, profiles=()) for b in world.bindings)
            assert frame.runtime._content(frame.record.accepted_content).profiles == ("restricted_documents",)
            denied = next(r for r in world.routes if r.route_ref == "denied_child")
            object.__setattr__(frame.binding, "child_main_route", denied)
        elif change == "mutated-leaf":
            frame.leaf["context"] = "changed after retention"
        elif change == "copied-record":
            frame.record = replace(frame.record)
        elif change == "copied-token":
            object.__setattr__(frame.record, "token", object.__new__(type(frame.record.token)))
        else:
            frame.slot = "constructor"
    world.after_accept = alter
    with pytest.raises(NativeContentPolicyDenied):
        world.call()
    assert len(world.delegate.sent) == 1 and not world.ports[2, "main"].sent
    from agent.native_main_payload import _delegate_frames
    assert _delegate_frames[world.adapter].child is None


@pytest.mark.parametrize("boundary", ["delegate", "child"])
@pytest.mark.parametrize("kind", ["interrupt", "cancel", "keyboard", "exit"])
def test_controls_preserve_exception_identity_and_partial_sends(no_delegation_effects, boundary, kind):
    from agent.content_policy import current_context
    from agent.native_main_payload import _delegate_frames
    error = {"interrupt": InterruptedError, "cancel": asyncio.CancelledError,
             "keyboard": KeyboardInterrupt, "exit": SystemExit}[kind]("invented control")
    world = ThreeHolders().bind()
    def stop(envelope):
        raise error
    port = world.delegate if boundary == "delegate" else world.ports[2, "main"]
    port.hook = stop
    with pytest.raises(type(error)) as caught:
        world.call()
    assert caught.value is error and current_context() is None
    frame = _delegate_frames[world.adapter]
    assert frame.invalid and frame.delegate_sends == 1
    assert len(world.delegate.sent) == 1
    assert len(world.ports[2, "main"].sent) == int(boundary == "child")
    assert len(frame.partial_sends) == 1 + int(boundary == "child")
    if boundary == "child":
        assert frame.record.child_input is not None and frame.record.child_request is not None
        assert frame.record.child_completion is None and frame.child_adapter._constructed.closed


@pytest.mark.parametrize("operation", ["reentry", "wrong-thread"])
def test_callback_reentry_and_wrong_thread_refuse(no_delegation_effects, operation):
    world = ThreeHolders().bind()
    if operation == "reentry":
        def recurse(envelope):
            world.call()
        world.delegate.hook = recurse
        with pytest.raises(NativeContentPolicyDenied):
            world.call()
        assert len(world.delegate.sent) == 1 and not world.ports[2, "main"].sent
    else:
        errors = []
        def threaded():
            try:
                world.call()
            except BaseException as error:
                errors.append(error)
        thread = threading.Thread(target=threaded)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive() and len(errors) == 1
        assert isinstance(errors[0], NativeContentPolicyDenied)
        assert all(not p.sent for p in world.ports.values())


def test_shared_retention_capacity_has_no_rollback(no_delegation_effects):
    from hermes_content_policy import PolicyError
    world = ThreeHolders().bind()
    runtime = world.adapter._runtime
    for _ in range(8):
        runtime.admit(world.requests[0], b"x" * 1048576)
    retained = runtime._retained_bytes
    with pytest.raises((NativeContentPolicyDenied, PolicyError)):
        world.call()
    assert runtime._retained_bytes == retained == 8388608
    assert all(not p.sent for p in world.ports.values())


def test_completed_proof_survives_late_destination_denial(no_delegation_effects):
    from agent.native_main_payload import _delegate_frames
    world = ThreeHolders().bind()
    def change(frame):
        assert frame.record.child_completion.result["completed"] is True
        world.bindings = tuple(replace(b, profiles=()) for b in world.bindings)
        object.__setattr__(frame.binding, "child_main_route", next(r for r in world.routes if r.route_ref == "denied_child"))
    world.after_complete = change
    with pytest.raises(NativeContentPolicyDenied):
        world.call()
    frame = _delegate_frames[world.adapter]
    assert frame.record.child_completion.result["final_response"] == "Two blue squares."
    assert frame.record.return_facts is None and frame.invalid
    assert len(world.delegate.sent) == len(world.ports[2, "main"].sent) == 1


def test_unrelated_equal_text_result_is_not_a_return(no_delegation_effects, monkeypatch):
    from agent.native_main_payload import _delegate_frames
    original = ContentPolicyRuntime.return_from_delegate
    unrelated = []
    def substitute(runtime, request, envelope, accepted, result, *, receiver_request):
        assert runtime._content(result).payload == runtime._content(unrelated[0]).payload
        return original(runtime, request, envelope, accepted, unrelated[0], receiver_request=receiver_request)
    monkeypatch.setattr(ContentPolicyRuntime, "return_from_delegate", substitute)
    world = ThreeHolders().bind()
    expected = '{"results": [{"task_index": 0, "status": "completed", "summary": "Two blue squares.", "api_calls": 1, "duration_seconds": 0.0, "model": "invented_child_model", "exit_reason": "completed", "truncated": false, "tokens": {"input": 3, "output": 2}, "tool_trace": []}], "total_duration_seconds": 0.0}'
    qualification_runtime = world.qualifications[-1][0]
    accepted_d = qualification_runtime.accept(world.requests[1], world.qualifications[-1][4])
    unrelated.append(qualification_runtime.derive(world.requests[1], (accepted_d,), expected.encode(), path="tool_result"))
    with pytest.raises(NativeContentPolicyDenied):
        world.call()
    frame = _delegate_frames[world.adapter]
    assert frame.record.child_completion.result["completed"] is True
    assert frame.record.return_facts is None and frame.invalid
    assert len(world.delegate.sent) == len(world.ports[2, "main"].sent) == 1


def test_unregistered_parent_and_direct_native_seams_refuse(no_delegation_effects):
    import copy
    from agent.native_main_payload import _delegate_frames
    from tools import delegate_tool
    world = ThreeHolders().bind()
    copied = copy.copy(world.parent)
    with pytest.raises(NativeContentPolicyDenied):
        delegate_tool.delegate_task(goal="invented", parent_agent=copied)
    assert all(not p.sent for p in world.ports.values())
    for seam in (lambda w: delegate_tool._build_child_agent(0, "invented", None, None, None, 1, 1, w.parent),
                 lambda w: delegate_tool._run_single_child(0, "invented", parent_agent=w.parent)):
        world = ThreeHolders().bind()
        frame = _delegate_frames[world.adapter]
        with pytest.raises(NativeContentPolicyDenied):
            seam(world)
        assert frame.invalid and all(not p.sent for p in world.ports.values())


def test_empty_completed_child_text_cannot_release(no_delegation_effects):
    from agent.native_main_payload import _delegate_frames
    world = ThreeHolders()
    response = json.loads(world.ports[2, "main"].output)
    response["choices"][0]["message"]["content"] = ""
    world.ports[2, "main"].output = canonical(response)
    world.bind()
    with pytest.raises(NativeContentPolicyDenied):
        world.call()
    frame = _delegate_frames[world.adapter]
    assert frame.record.child_completion.result["completed"] is True
    assert frame.record.entry_bytes is frame.record.return_facts is None
    assert len(world.delegate.sent) == len(world.ports[2, "main"].sent) == 1


def test_closed_child_cancellation_still_forbids_release(no_delegation_effects):
    from agent.native_main_payload import _delegate_frames
    from agent.content_policy import current_context
    world = ThreeHolders().bind()
    def cancel_when_closed(frame):
        if frame.child_adapter._constructed.closed:
            frame.child._interrupt_requested = True
        else:
            world.after_complete = cancel_when_closed
    world.after_complete = cancel_when_closed
    with pytest.raises(InterruptedError):
        world.call()
    frame = _delegate_frames[world.adapter]
    assert frame.record.return_facts is not None and frame.child_adapter._constructed.closed
    assert frame.phase == "returned" and frame.invalid and current_context() is None
    assert len(world.delegate.sent) == len(world.ports[2, "main"].sent) == 1


def test_raw_admission_control_keeps_original_exception(no_delegation_effects):
    from agent.native_main_payload import _delegate_frames
    world = ThreeHolders()
    error = InterruptedError("invented raw admission cancellation")
    def declared_clock():
        adapter = getattr(world, "adapter", None)
        frame = _delegate_frames.get(adapter) if adapter is not None else None
        if frame is not None and frame.running and frame.record is None:
            raise error
        return world.now
    world.clock = declared_clock
    world.bind()
    with pytest.raises(InterruptedError) as caught:
        world.call()
    assert caught.value is error
    frame = _delegate_frames[world.adapter]
    assert frame.invalid and frame.record is None and not hasattr(frame, "token")
    assert all(not p.sent for p in world.ports.values())
