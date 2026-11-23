"""Exercise one original operator authorization through the actual parent turn."""

import asyncio
import json
import threading
from dataclasses import replace
from functools import cache
from pathlib import Path

import pytest
from hermes_content_policy import ContentPolicyRuntime, Destination, PATHS, PolicyError, RouteBinding

from agent.content_policy import NativeContentPolicyDenied, content_policy_scope, current_context
from tests.tools.test_native_controlled_delegation import (
    ThreeHolders, DeclaredPort, Hostile, TextSubclass, canonical,
    no_delegation_effects as no_delegation_effects,
    excluded as excluded, invented_offline_configuration as invented_offline_configuration,
    retain_case_durations as retain_case_durations,
)


class IncrementalCaseReceipt:
    def __init__(self, path, pins):
        self.path, self.pins = path, pins

    def write(self, nodeid, phase, outcome, seconds):
        row = dict(nodeid=nodeid, phase=phase, outcome=outcome, seconds=seconds, **self.pins)
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, sort_keys=True) + "\n")

    def pytest_runtest_logreport(self, report):
        self.write(report.nodeid, report.when, report.outcome, report.duration)


@pytest.fixture(scope="module", autouse=True)
def retain_incremental_case_receipts(request):
    xml = request.config.option.xmlpath
    if xml is None:
        yield
        return
    path = Path(xml).with_suffix(".cases.jsonl")
    path.write_text("", encoding="utf-8")
    overrides = dict(item.split("=", 1) for item in request.config.option.override_ini if "=" in item)
    pins = {name: overrides.get("accountant_" + name) for name in ("source_sha256", "test_sha256")}
    plugin = IncrementalCaseReceipt(path, pins)
    request.config.pluginmanager.register(plugin, "accountant_case_receipt")
    yield
    request.config.pluginmanager.unregister(plugin)


@pytest.fixture(autouse=True)
def record_case_start(request):
    plugin = request.config.pluginmanager.getplugin("accountant_case_receipt")
    if plugin is not None:
        plugin.write(request.node.nodeid, "start", "pending", 0.0)


class ResumptionHolders(ThreeHolders):
    def __init__(self):
        super().__init__()
        self.on_current = self.on_selected = self.on_produced = self.on_clock = None
        self.ports[0, "main"].output = canonical(dict(
            id="invented_parent_reply", object="chat.completion", created=13,
            model="invented_parent_model", choices=[dict(index=0, finish_reason="stop",
                message=dict(role="assistant", content="The count is two."))],
            usage=dict(prompt_tokens=5, completion_tokens=4, total_tokens=9)))

    def current(self, handle):
        if self.on_current is not None:
            self.on_current(handle)
        return super().current(handle)

    def selected(self, handle):
        if self.on_selected is not None:
            self.on_selected(handle)
        return super().selected(handle)

    def produced(self, ref):
        if self.on_produced is not None:
            self.on_produced(ref)
        return super().produced(ref)

    def time(self):
        if self.on_clock is not None:
            self.on_clock()
        return super().time()

    def authorize(self, **options):
        with content_policy_scope(self.context):
            self.authorization = self.adapter.bind_parent_resumption(self.parent, **options)
        return self

    def resume(self, token=None):
        with content_policy_scope(self.context):
            return self.adapter.resume_parent(
                self.parent, self.authorization if token is None else token)


def test_example_one_actual_leaf_and_parent(no_delegation_effects):
    from agent.native_main_payload import _delegate_frames, _parent_resumptions
    from run_agent import AIAgent
    world = ResumptionHolders().bind().authorize(
        instruction="Report the count.", system_message="Use the returned count.")
    aggregate = world.call()
    frame = _delegate_frames[world.adapter]
    original = frame.record
    closed_proof = frame.closed_proof
    assert type(original.child) is AIAgent and original.child_adapter._constructed.closed
    result = world.resume()
    owner = _parent_resumptions[world.adapter]
    record = owner.record
    user = "Report the count.\n" + aggregate
    request = dict(model="invented_parent_model", messages=[
        dict(role="system", content="Use the returned count."), dict(role="user", content=user)], stream=False)
    expected = dict(
        final_response="The count is two.", last_reasoning=None,
        messages=[dict(role="user", content=user), dict(role="assistant", content="The count is two.")],
        api_calls=1, completed=True, turn_exit_reason="text_response(finish_reason=stop)", failed=False,
        partial=False, interrupted=False, response_transformed=False, pre_transform_response=None,
        response_previewed=False, model="invented_parent_model", provider="synthetic_native", base_url="",
        input_tokens=5, output_tokens=4, cache_read_tokens=0, cache_write_tokens=0, reasoning_tokens=0,
        prompt_tokens=5, completion_tokens=4, total_tokens=9, last_prompt_tokens=5,
        estimated_cost_usd=None, cost_status="unknown", cost_source=None, service_tier=None,
        session_id="invented_parent_session")
    assert result == expected and len(result) == 29
    assert result is record.result is record.native_completion.result
    assert record.result_bytes == canonical(expected)
    assert record.request == request
    destination = dict(transport_ref="invented_0_main", provider_ref="synthetic_native",
        model_ref="invented_parent_model", endpoint_ref="invented_endpoint_0", revision=1)
    assert record.request_bytes == canonical(dict(api_mode="chat_completions", destination=destination, request=request))
    runtime = world.adapter._runtime
    assert record.input_parents == (original.return_facts.returned_content, owner.binding.authorization_content)
    assert runtime._content(record.input_content).parents == record.input_parents
    assert runtime._content(record.input_content).payload == record.raw_bytes
    @cache
    def ancestors(content):
        return {content}.union(*(ancestors(p) for p in runtime._content(content).parents))
    for content in (*record.preparation_contents, record.request_content,
                    record.native_invocation.result_content, record.result_content):
        lineage = ancestors(content)
        assert record.input_content in lineage
        assert original.return_facts.returned_content in lineage
        assert original.child_completion.content in lineage
        assert "restricted_documents" in runtime._content(content).profiles
    assert record.native_invocation.request_handle is world.requests[0]
    assert original.binding.receiver_request is world.requests[1]
    assert original.binding.child_request is world.requests[2]
    assert frame.record is original and frame.closed_proof is closed_proof
    assert frame.phase == "released" and frame.terminal and not frame.running
    assert owner.phase == "released" and owner.active_call is None
    assert len(frame.partial_sends) == 2
    assert owner.partial_send[1] is record.envelope
    assert owner.partial_send[2] is record.observed_destination
    assert owner.partial_send[3:5] == (0, 1)
    assert owner.partial_send[5] is record.response_bytes
    assert world.adapter.send_count == original.child_adapter.send_count == frame.delegate_sends == 1
    assert sum(len(p.sent) for p in world.ports.values()) == 3
    assert all(not p.sent for key, p in world.ports.items() if isinstance(key, tuple) and key[1] == "tool_result")
    assert len(world.qualifications) == sum(len(p.qualification) for p in world.ports.values()) == 25
    world.bindings = tuple(replace(b, profiles=()) for b in world.bindings)
    world.adapter.validate_scope()
    assert not world.context._denied
    world.parent.close()
    world.parent.close()


def ready(world=None):
    world = ResumptionHolders() if world is None else world
    return world.bind().authorize(instruction="Report the count.", system_message="Use the returned count.")


def owner_of(world):
    from agent.native_main_payload import _parent_resumptions
    return _parent_resumptions[world.adapter]


def assert_denied(world, *, parent_sends=0):
    owner = owner_of(world)
    assert owner.phase == "denied" and owner.active_call is None
    assert world.adapter._invalid and world.context._denied
    if owner.frame.child_adapter is not None:
        assert owner.frame.child_adapter._invalid and owner.frame.child_context._denied
    assert world.adapter.send_count == len(world.ports[0, "main"].sent) == parent_sends
    with pytest.raises(NativeContentPolicyDenied):
        world.resume()


@pytest.mark.parametrize("options", [
    dict(instruction=""), dict(instruction=TextSubclass("text")), dict(instruction=Hostile()),
    dict(instruction="text", system_message=Hostile()),
    dict(instruction="text", system_message=TextSubclass("system")),
])
def test_binding_rejects_input_before_callbacks(no_delegation_effects, options):
    world = ResumptionHolders().bind()
    def forbidden(*args):
        pytest.fail("Malformed input reached an original callback")
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            world.on_current = world.on_selected = world.on_produced = world.on_clock = forbidden
            world.adapter.bind_parent_resumption(world.parent, **options)
    assert world.adapter._invalid and world.context._denied
    assert sum(len(p.sent) for p in world.ports.values()) == 0


@pytest.mark.parametrize("explicit", [False, True])
def test_original_presence_and_capacity_admission(no_delegation_effects, explicit):
    world = ResumptionHolders().bind()
    options = dict(instruction="Report the count.")
    if explicit:
        options["system_message"] = None
    world.authorize(**options)
    owner = owner_of(world)
    b = owner.binding
    assert b.agent is world.parent and b.parent_request is world.requests[0]
    assert b.task_id is world.bindings[0].execution_ref
    rows, positional, keywords = b.options
    assert positional == (world.adapter, world.parent)
    assert dict((n, p) for n, p, _ in rows)["system_message"] is explicit
    assert keywords == tuple(options.items())
    content = world.adapter._runtime._content(b.authorization_content)
    assert content.payload is b.authorization_bytes and content.parents == ()
    assert content.profiles == ("restricted_documents",)
    assert owner.phase == "bound" and owner.record is None
    assert sum(len(p.sent) for p in world.ports.values()) == 0


def test_reentrant_binding_never_issues_token(no_delegation_effects):
    world = ResumptionHolders().bind()
    observations = []
    def reenter(handle):
        from agent.native_main_payload import _parent_resumptions
        if _parent_resumptions.get(world.adapter) is None:
            return
        world.on_current = None
        observations.append(owner_of(world).binding)
        world.adapter.bind_parent_resumption(world.parent, instruction="Nested")
    world.on_current = reenter
    with pytest.raises(NativeContentPolicyDenied):
        world.authorize(instruction="Report the count.")
    assert observations == [None]
    assert owner_of(world).binding is None
    assert_denied(world)


@pytest.mark.parametrize("operation", ["direct", "resume", "wrong_thread", "copied", "unrelated"])
def test_bound_authority_permits_no_work(no_delegation_effects, operation):
    from agent.native_main_payload import NativeParentResumption
    world = ready()
    if operation == "direct":
        call = lambda: world.parent.run_conversation("text", task_id="invented_parent_execution")
    elif operation == "resume":
        call = world.resume
    elif operation == "copied":
        call = lambda: world.resume(object.__new__(NativeParentResumption))
    elif operation == "unrelated":
        other = ready()
        call = lambda: world.resume(other.authorization)
    else:
        errors = []
        def threaded():
            try:
                world.adapter.resume_parent(world.parent, world.authorization)
            except BaseException as error:
                errors.append(error)
        thread = threading.Thread(target=threaded)
        thread.start()
        thread.join()
        assert len(errors) == 1 and isinstance(errors[0], NativeContentPolicyDenied)
        assert_denied(world)
        return
    with pytest.raises(NativeContentPolicyDenied):
        call()
    assert_denied(world)


def test_retained_profiles_and_terminal_operations(no_delegation_effects):
    world = ready()
    aggregate = world.call()
    frame = owner_of(world).frame
    world.bindings = tuple(replace(b, profiles=()) for b in world.bindings)
    result = world.resume()
    owner = owner_of(world)
    assert result["messages"][0]["content"] == "Report the count.\n" + aggregate
    for content in (owner.record.input_content, owner.record.request_content, owner.record.result_content):
        assert "restricted_documents" in world.adapter._runtime._content(content).profiles
    assert frame.phase == "released" and frame.child_adapter._constructed.closed
    world.adapter.validate_scope()
    for operation in (
        world.resume, lambda: world.parent.run_conversation("later", task_id="invented_parent_execution"),
        lambda: world.adapter.dispatch(world.parent, owner.record.request),
        lambda: world.adapter.bind_auxiliary(world.parent, path="title", task="title",
            request_handle=world.requests[1], selection=world.selection,
            producer_ref="invented_producer_1", route_ref="route_1_title", language="en"),
        lambda: world.adapter.bind_fallback(world.parent, request_handle=world.requests[1],
            selection=world.selection, producer_ref="invented_producer_1", route_ref="route_1_fallback"),
        world.call,
    ):
        with pytest.raises(NativeContentPolicyDenied):
            operation()
    assert sum(len(p.sent) for p in world.ports.values()) == 3


@pytest.mark.parametrize("target", ["resume", "public", "dispatch", "helper"])
def test_callback_reentry_before_busy_guard(no_delegation_effects, target):
    from agent.native_main_payload import dispatch_controlled_main
    world = ready()
    world.call()
    observed = []
    def reenter(handle):
        owner = owner_of(world)
        if owner.active_call is None or owner.public_frame is None or owner.stage is None:
            return
        world.on_current = None
        observed.append(world.adapter._busy)
        if target == "resume":
            world.adapter.resume_parent(world.parent, world.authorization)
        elif target == "public":
            args = owner.record.arguments
            world.parent.run_conversation(args[0], args[1], args[2], args[3])
        elif target == "dispatch":
            world.adapter.dispatch(world.parent, {})
        else:
            dispatch_controlled_main(world.parent, {})
    world.on_current = reenter
    with pytest.raises(NativeContentPolicyDenied):
        world.resume()
    assert observed == [False]
    assert_denied(world)


@pytest.mark.parametrize("error", [
    pytest.param(InterruptedError("original"), id="interrupted-send"),
    pytest.param(asyncio.CancelledError("original"), id="cancelled-send"),
    pytest.param(KeyboardInterrupt("original"), id="keyboard-binding"),
    pytest.param(SystemExit("original"), id="system-exit-binding"),
])
def test_original_control_identity_at_trusted_boundaries(no_delegation_effects, error):
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        world = ResumptionHolders().bind()
        def interrupt_binding(handle):
            from agent.native_main_payload import _parent_resumptions
            if _parent_resumptions.get(world.adapter) is not None:
                raise error
        world.on_current = interrupt_binding
        with pytest.raises(type(error)) as raised:
            world.authorize(instruction="Report the count.")
        assert raised.value is error
        assert owner_of(world).binding is None
        assert_denied(world)
        return
    world = ready()
    world.call()
    def interrupt(envelope):
        raise error
    world.ports[0, "main"].hook = interrupt
    with pytest.raises(type(error)) as raised:
        world.resume()
    assert raised.value is error
    owner = owner_of(world)
    assert owner.partial_send[1] is world.ports[0, "main"].sent[0]
    assert owner.partial_send[3:5] == (0, 1)
    assert owner.record.native_completion is None and owner.record.result is None
    assert_denied(world, parent_sends=1)


@pytest.mark.parametrize("mutation", ["input", "result", "delegation"])
def test_actual_object_mutation_denies_release(no_delegation_effects, mutation):
    world = ready()
    world.call()
    owner = owner_of(world)
    if mutation == "delegation":
        owner.frame.record = replace(owner.frame.record)
    elif mutation == "input":
        def mutate(envelope):
            owner.record.request["messages"][-1]["content"] += "changed"
        world.ports[0, "main"].hook = mutate
    else:
        def mutate():
            if world.adapter._completions:
                world.on_clock = None
                world.adapter._completions[0].result["messages"][0]["content"] += "changed"
        world.on_clock = mutate
    with pytest.raises(NativeContentPolicyDenied):
        world.resume()
    assert_denied(world, parent_sends=0 if mutation == "delegation" else 1)
    if mutation == "input":
        assert owner.partial_send[5] is world.ports[0, "main"].output
    if mutation == "result":
        assert world.adapter._completions and world.adapter._completions[0].result is owner.record.result


class ParentExpiryHolders(ResumptionHolders):
    def __init__(self):
        super().__init__()
        self.routes = tuple(replace(r, expires_at=20) if r.route_ref == "route_0_main" else r for r in self.routes)
        destination = Destination("invented_parent_alternate", "synthetic_native", "invented_parent_model",
                                  "invented_parent_alternate_endpoint", 1)
        port = DeclaredPort(destination, b'{"accepted":true}')
        self.ports["parent_alternate"] = port
        self.port_bindings.append((port, 0))
        self.routes += (RouteBinding("parent_alternate", "parent_selection", 3, "invented_parent", "main",
                                    destination, 100, "synthetic_parent_alternate", "synthetic_only"),)
        self.profiles = (replace(self.profiles[0], allowed_routes=(*self.profiles[0].allowed_routes, "parent_alternate")),)
        self.transports = tuple((p.destination.transport_ref, p) for p in self.ports.values())

    def bind(self):
        # Qualify the extra original route before any operator binding.
        port = self.ports["parent_alternate"]
        port.qualifying = True
        runtime = ContentPolicyRuntime(admission=self.admission, profiles=self.profiles,
            routes=self.routes, transports=self.transports, clock=self.clock, policy_revision=9)
        content = runtime.admit(self.requests[0], b"Invented alternate qualification")
        result = runtime.dispatch(self.requests[0], content, path="main", route_ref="parent_alternate")
        self.qualifications.append((runtime, self.requests[0], "main", result, port.qualification[-1]))
        port.qualifying = False
        return super().bind()


@pytest.mark.parametrize("late", [False, True])
def test_original_parent_route_expiry_with_live_coverage(no_delegation_effects, late):
    world = ready(ParentExpiryHolders())
    world.call()
    owner = owner_of(world)
    if late:
        def expire_after_scope():
            if (owner.record is not None and owner.record.native_completion is not None
                    and owner.stage_index == 8 and owner.stage is None):
                world.on_clock = None
                world.now = 20
        world.on_clock = expire_after_scope
    else:
        world.now = 20
    with pytest.raises(NativeContentPolicyDenied):
        world.resume()
    assert world.now == 20
    assert next(r for r in world.routes if r.route_ref == "parent_alternate").expires_at == 100
    assert len(world.qualifications) == sum(len(p.qualification) for p in world.ports.values()) == 26
    assert_denied(world, parent_sends=int(late))
    assert owner.frame.child_adapter._constructed.closed
    if late:
        assert owner.record.native_completion is world.adapter._completions[0]
        assert owner.record.result is owner.record.native_completion.result
        assert current_context() is None


@pytest.mark.parametrize("route", ["issued_delegate", "route_2_main"])
def test_other_used_route_expiry(no_delegation_effects, route):
    world = ResumptionHolders()
    world.routes = tuple(replace(r, expires_at=20) if r.route_ref == route else r for r in world.routes)
    world = ready(world)
    world.call()
    world.now = 20
    with pytest.raises(NativeContentPolicyDenied):
        world.resume()
    assert_denied(world)


def test_authorization_capacity_failure_is_terminal(no_delegation_effects):
    world = ResumptionHolders().bind()
    runtime = world.adapter._runtime
    retained = [runtime.admit(world.requests[0], b"Invented occupied capacity") for _ in range(128)]
    assert len(retained) == 128
    with pytest.raises(Exception):
        world.authorize(instruction="Report the count.")
    owner = owner_of(world)
    assert owner.binding is None and owner.record is None
    assert_denied(world)


def test_complete_over_limit_input_is_refused(no_delegation_effects):
    world = ResumptionHolders().bind().authorize(instruction="x" * 16_384)
    aggregate = world.call()
    owner = owner_of(world)
    assert len(owner.binding.instruction + "\n" + aggregate) > 16_384
    with pytest.raises(NativeContentPolicyDenied):
        world.resume()
    assert owner.record is None
    assert_denied(world)


def test_poison_after_genuine_completion_cannot_release(no_delegation_effects):
    world = ready()
    world.call()
    owner = owner_of(world)
    errors = []
    def poison_after_scope():
        if (owner.record is not None and owner.record.native_completion is not None
                and owner.stage_index == 8 and owner.stage is None):
            world.on_clock = None
            def wrong_thread_admission():
                try:
                    world.adapter._runtime.admit(world.requests[0], b"Invented wrong-thread refusal")
                except BaseException as error:
                    errors.append(error)
            thread = threading.Thread(target=wrong_thread_admission)
            thread.start()
            thread.join()
    world.on_clock = poison_after_scope
    with pytest.raises((NativeContentPolicyDenied, PolicyError)):
        world.resume()
    assert len(errors) == 1 and type(errors[0]) is PolicyError
    assert owner.record.native_completion is world.adapter._completions[0]
    assert owner.record.result is owner.record.native_completion.result
    assert owner.frame.child_adapter._constructed.closed
    assert_denied(world, parent_sends=1)
