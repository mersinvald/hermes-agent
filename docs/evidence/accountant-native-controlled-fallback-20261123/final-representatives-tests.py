"""Check one terminal fallback through actual constructed native entrypoints."""

import asyncio
import copy
import json
import threading
from pathlib import Path
from time import perf_counter
from dataclasses import replace
from types import SimpleNamespace

import pytest
from hermes_content_policy import PATHS, ContentPolicyRuntime, FailedDispatch, ProfilePolicy

from agent import chat_completion_helpers as chat, native_main_payload as native
from agent.content_policy import NativeContentPolicyDenied, content_policy_scope
from tests.agent.test_native_controlled_conversation import (
    CompressionWorld, Hostile, bind_compression, bind_title, construct, construct_compressor,
    summarize, title_from, turn,
    excluded as excluded,
    compression_excluded as compression_excluded,
)
from tests.agent.test_native_main_payload import (
    ControlledPort, World, canonical, reply,
    caller_boundary as caller_boundary,
    invented_offline_configuration as invented_offline_configuration,
)


@pytest.fixture(autouse=True)
def native_case_timing(request):
    timing = dict(nodeid=request.node.nodeid, started=perf_counter())
    yield timing
    timing["total_seconds"] = round(perf_counter() - timing.pop("started"), 6)
    xml = request.config.getoption("xmlpath")
    if xml:
        with Path(xml).with_suffix(".timings.jsonl").open("a") as output:
            output.write(json.dumps(timing, sort_keys=True) + "\n")


class FallbackWorld(World):
    def __init__(self):
        super().__init__()
        self.fallback_request = object()
        self.fallback_binding = replace(self.binding, selection_id="fallback_selection", selection_version=8)
        self.fallback_policy = native.NativeResultPolicy("fallback_producer", 4, self.policy.profiles)
        self.fallback_ports = {}
        routes = []
        for row in self.routes[:len(PATHS)]:
            destination = replace(row.destination, transport_ref=f"fallback_transport_{row.path}",
                                  model_ref="invented_fallback_model", endpoint_ref="invented_fallback_endpoint")
            port = self.fallback_ports[row.path] = ControlledPort(destination)
            self.ports[f"fallback_capability_{row.path}"] = port
            self.receiver_bindings[id(port)] = (port, replace(self.fallback_binding,
                execution_ref=f"fallback_receiver_{row.path}"))
            value = json.loads(reply("The blue folder stays."))
            value.update(id="invented_fallback_reply", model="invented_fallback_model")
            port.output = canonical(value)
            routes.append(replace(row, route_ref=f"fallback_{row.path}", destination=destination,
                selection_id="fallback_selection", selection_version=8,
                qualification_ref=f"synthetic_inprocess_fallback_selection_{row.path}"))
        self.routes += tuple(routes)
        self.profiles = (ProfilePolicy("restricted_documents", tuple(r.route_ref for r in self.routes), PATHS),)
        self.fallback_selection = native.NativeMainSelection("fallback_selection", 8, "chat_completions",
                                                            self.fallback_ports["fallback"].destination)
        self.fallback_selection_port = SimpleNamespace(current=self.fallback_selected)
        self.fallback_qualifications = ()
        self.fallback_selection_hook = None

    def current(self, handle):
        if handle is self.fallback_request:
            if self.admission_hook:
                self.admission_hook()
            return self.fallback_binding
        return super().current(handle)

    def produced(self, producer):
        if producer == "fallback_producer":
            if self.producer_hook:
                self.producer_hook()
            return self.fallback_policy
        return super().produced(producer)

    def fallback_selected(self, handle):
        assert handle is self.fallback_request
        if self.fallback_selection_hook:
            self.fallback_selection_hook()
        return self.fallback_selection

    def qualify(self):
        super().qualify()
        runtime = ContentPolicyRuntime(admission=self.admission, profiles=self.profiles,
            routes=self.routes, transports=tuple((p.destination.transport_ref, p) for p in self.ports.values()),
            clock=lambda: self.now, policy_revision=9)
        records = []
        for path in PATHS:
            source = runtime.admit(self.fallback_request, b"Invented independent fallback qualification")
            result = runtime.dispatch(self.fallback_request, source, path=path, route_ref=f"fallback_{path}")
            records.append((path, source, result, self.fallback_ports[path].sent[0]))
            self.fallback_ports[path].sent = []
        self.fallback_qualifications = tuple(records)



class HistoryFallbackWorld(FallbackWorld, CompressionWorld):
    pass

def bind_fallback(world, **changes):
    options = dict(request_handle=world.fallback_request, selection=world.fallback_selection_port,
                   producer_ref="fallback_producer", route_ref="fallback_fallback")
    options.update(changes)
    with content_policy_scope(world.context):
        world.adapter.bind_fallback(world.agent, **options)


def fallback_world(*, bound=True):
    world = construct(FallbackWorld().bind(), max_tokens=None)
    if bound:
        bind_fallback(world)
    return world


def fail_main(world, error=None, change=None):
    error = ConnectionError() if error is None else error
    def fail(envelope):
        if change:
            change()
        raise error
    world.ports["main"].hook = fail
    return error


def fixed_turn(world, **changes):
    return turn(world, "Keep the blue folder.", system_message="Be concise.", **changes)


@pytest.fixture
def fallback_excluded(compression_excluded, caller_boundary, monkeypatch, native_case_timing):
    from agent import auxiliary_client as auxiliary
    from concurrent import futures
    for module, names in (
        (auxiliary, ("resolve_provider_client",)),
        (chat, ("_reset_stale_streak", "_record_interrupted_provider_wait", "rewrite_prompt_model_identity")),
        (auxiliary, ("_get_cached_client", "_acquire_sync_aux_semaphore", "_call_llm_impl")),
        (futures, ("ThreadPoolExecutor", "ProcessPoolExecutor")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, compression_excluded)
    monkeypatch.setattr(chat, "time", SimpleNamespace(monotonic=compression_excluded, sleep=compression_excluded))
    native_case_timing["setup_seconds"] = round(perf_counter() - native_case_timing["started"], 6)
    return compression_excluded


@pytest.mark.parametrize("error_type", [ConnectionError, TimeoutError])
def test_fallback_fixed_terminal_proof(fallback_excluded, caller_boundary, monkeypatch, error_type):
    derived = []
    derive = ContentPolicyRuntime.derive_for_selection
    def observed(runtime, handle, parents, payload, *, path, parent_requests):
        result = derive(runtime, handle, parents, payload, path=path, parent_requests=parent_requests)
        derived.append((handle, parents, payload, path, parent_requests, result))
        return result
    monkeypatch.setattr(ContentPolicyRuntime, "derive_for_selection", observed)
    world = fallback_world()
    original = dict(vars(world.agent))
    fail_main(world, error_type())
    result = fixed_turn(world)
    failed, = world.adapter.failed_main_attempts
    invocation, = world.adapter.fallback_invocations
    frame = world.adapter._fallback_frame
    expected_request = {"messages": [{"content": "Be concise.", "role": "system"},
                                     {"content": "Keep the blue folder.", "role": "user"}],
                        "model": "invented_model", "stream": False}
    assert failed.request == expected_request and failed.request is caller_boundary.dispatches[-1][2]
    assert failed.turn_frame.request_bytes == canonical(expected_request)
    assert failed.request_bytes == world.ports["main"].sent[0].payload
    assert json.loads(failed.request_bytes)["request"] == expected_request
    assert failed.source_bytes == failed.turn_frame.raw_bytes
    assert failed.send_observation.operation.content is failed.request_content
    assert failed.send_observation.envelope is world.ports["main"].sent[0]
    assert failed.send_observation.exception_type is error_type
    assert (failed.send_observation.send_count_before, failed.send_observation.send_count_after) == (0, 1)
    facts = world.adapter._runtime.failure_facts(world.request, failed.shared_failure)
    assert facts is failed.failure_facts and facts.failure_content is failed.failure_content
    assert facts.envelope is failed.send_observation.envelope
    assert facts.request_content is failed.request_content and facts.response_bytes is None
    failure_payload = canonical(dict(failure_kind="connection_error" if error_type is ConnectionError else "timeout_error",
                                     phase="send", send_count=1, response_bytes=None))
    retained = world.adapter._runtime._contents[id(facts.failure_content)]
    assert retained.payload == failure_payload
    assert invocation.failed_main is failed and invocation.request_handle is world.fallback_request
    assert invocation.request == dict(expected_request, model="invented_fallback_model")
    assert invocation.request["messages"] is failed.request["messages"]
    assert invocation.request_bytes == canonical(invocation.request)
    assert invocation.envelope is world.fallback_ports["fallback"].sent[0]
    assert json.loads(invocation.envelope.payload)["request"] == invocation.request
    assert invocation.observed_destination == world.fallback_selection.destination
    assert invocation.response_bytes == world.fallback_ports["fallback"].output
    assert frame.result is invocation.output is result
    assert frame.result_bytes == invocation.output_bytes == canonical(result)
    assert frame.result_content is invocation.output_content
    assert len(invocation.input_stages) == len(invocation.stage_contents) == 3
    assert len(invocation.output_stages) == len(invocation.output_contents) == 2
    for payload, content in zip((*invocation.input_stages, *invocation.output_stages),
                                (*invocation.stage_contents, *invocation.output_contents)):
        assert any(row[2] == payload and row[-1] is content for row in derived)
    proof = next(row for row in derived if row[-1] is frame.source_content)
    assert all(a is b for a, b in zip(proof[1], failed.parents))
    assert all(a is b for a, b in zip(proof[4], failed.parent_requests))
    assert result == dict(completed=True, failed=False, interrupted=False, partial=False,
        final_response="The blue folder stays.", messages=[*expected_request["messages"],
            dict(role="assistant", content="The blue folder stays.")], fallback=True,
        selected_provider="synthetic_native", selected_model="invented_model", provider="synthetic_native",
        model="invented_fallback_model", base_url="", api_calls=2, main_attempt_usage=None,
        verified_response_usage=dict(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        total_usage_known=False, estimated_cost_usd=None, cost_status="unknown")
    assert world.adapter.send_count == world.adapter.fallback_send_count == 1
    assert world.adapter.auxiliary_send_count == 0
    assert not world.adapter.invocations and not world.adapter._completions
    assert world.adapter._aux_binding is world.adapter._compression_binding is None
    assert all(vars(world.agent)[key] is value for key, value in original.items())
    assert len(world.fallback_qualifications) == 8
    world.adapter.validate_scope()
    world.agent.close()
    world.agent.close()
    assert world.adapter.failed_main_attempts == (failed,)
    assert world.adapter.fallback_invocations == (invocation,)
    denied(world, lambda: fixed_turn(world), 1, 1)


def counts(world, main, fallback, *, successful=0):
    assert world.adapter.send_count == len(world.ports["main"].sent) == main
    assert world.adapter.fallback_send_count == len(world.fallback_ports["fallback"].sent) == fallback
    assert len(world.adapter.invocations) == len(world.adapter._completions) == successful
    assert not world.sdk_calls


def denied(world, callback, main=0, fallback=0, *, successful=0):
    with pytest.raises(NativeContentPolicyDenied):
        callback()
    assert world.context._denied
    counts(world, main, fallback, successful=successful)


def test_bound_main_success_stays_main_and_reason_is_not_proof(fallback_excluded):
    world = fallback_world()
    result = fixed_turn(world)
    assert result["model"] == "invented_model" and result["api_calls"] == 1
    assert result["total_tokens"] == 5 and "fallback" not in result
    counts(world, 1, 0, successful=1)
    assert not world.adapter.failed_main_attempts and not world.adapter.fallback_invocations
    denied(world, lambda: world.agent._try_activate_fallback("timeout"), 1, successful=1)


def test_unbound_failure_keeps_terminal_dispatch(fallback_excluded):
    world = fallback_world(bound=False)
    fail_main(world)
    denied(world, lambda: fixed_turn(world), 1)
    assert not world.adapter.failed_main_attempts and not world.adapter.fallback_invocations


@pytest.mark.parametrize("token", [None, object()])
def test_external_activation_refuses_before_ordinary_effects(fallback_excluded, token):
    world = fallback_world()
    denied(world, lambda: world.agent._try_activate_fallback(controlled_failure=token))


@pytest.mark.parametrize("kind", ["wrapper", "copied_failure"])
def test_shared_failure_requires_native_observation_and_original_token(fallback_excluded, monkeypatch, kind):
    if kind == "wrapper":
        original = native._Transport.send
        def before_port(wrapper, envelope):
            if envelope.path == "main":
                raise ConnectionError()
            return original(wrapper, envelope)
        monkeypatch.setattr(native._Transport, "send", before_port)
    else:
        original = ContentPolicyRuntime.dispatch_attempt
        def copied(runtime, *args, **kwargs):
            failure = original(runtime, *args, **kwargs)
            assert type(failure) is FailedDispatch
            return copy.copy(failure)
        monkeypatch.setattr(ContentPolicyRuntime, "dispatch_attempt", copied)
    world = fallback_world()
    fail_main(world)
    denied(world, lambda: fixed_turn(world), 0 if kind == "wrapper" else 1)
    assert not world.adapter.failed_main_attempts
    assert len(world.adapter._runtime._failures) == 1


@pytest.mark.parametrize("kind", ["interrupted", "explicit", "keyboard", "exit", "runtime", "subclass", "policy"])
def test_ineligible_error_is_terminal_and_controls_keep_identity(fallback_excluded, kind):
    from agent.auxiliary_client import AuxiliaryExplicitCancellation
    class ConnectionSubclass(ConnectionError):
        pass
    errors = dict(interrupted=InterruptedError(Hostile()), explicit=AuxiliaryExplicitCancellation(),
                  keyboard=KeyboardInterrupt(), exit=SystemExit(3), runtime=RuntimeError(),
                  subclass=ConnectionSubclass(), policy=NativeContentPolicyDenied("invented_denial"))
    error = errors[kind]
    world = fallback_world()
    fail_main(world, error)
    expected = InterruptedError if kind == "interrupted" else (
        type(error) if kind in ("explicit", "keyboard", "exit") else NativeContentPolicyDenied)
    with pytest.raises(expected) as caught:
        fixed_turn(world)
    if kind == "interrupted":
        assert type(caught.value) is InterruptedError
        assert caught.value.args == ("Agent interrupted during API call",)
    elif kind in ("explicit", "keyboard", "exit"):
        assert caught.value is error
    counts(world, 1, 0)
    assert world.context._denied and not world.adapter.failed_main_attempts


@pytest.mark.parametrize("kind", ["binding", "producer", "destination", "model", "expiry", "hostile", "qualification", "profiles"])
def test_current_fallback_facts_drift_after_failed_send_refuses(fallback_excluded, kind):
    world = fallback_world()
    def change():
        if kind == "binding":
            world.fallback_binding = replace(world.fallback_binding, auth_epoch=8)
        elif kind == "producer":
            world.fallback_policy = replace(world.fallback_policy, revision=5)
        elif kind == "destination":
            port = world.fallback_ports["fallback"]
            port.destination = replace(port.destination, endpoint_ref="changed_endpoint")
        elif kind == "model":
            world.fallback_selection = replace(world.fallback_selection, destination=replace(
                world.fallback_selection.destination, model_ref="invented_other_model"))
        elif kind == "expiry":
            world.now = 100
        elif kind == "qualification":
            object.__setattr__(world.adapter._fallback_binding.route, "qualification_ref", "changed_qualification")
        elif kind == "profiles":
            world.fallback_binding = replace(world.fallback_binding, profiles=())
        else:
            world.fallback_binding = replace(world.fallback_binding, auth_epoch=Hostile())
    fail_main(world, change=change)
    denied(world, lambda: fixed_turn(world), 1)
    assert not world.adapter.fallback_invocations


@pytest.mark.parametrize("kind", ["send", "model", "malformed", "empty"])
def test_fallback_failure_retains_available_proof(fallback_excluded, kind):
    world = fallback_world()
    fail_main(world)
    port = world.fallback_ports["fallback"]
    if kind == "send":
        def fail(envelope):
            raise TimeoutError()
        port.hook = fail
    elif kind == "malformed":
        port.output = b"{}"
    else:
        output = json.loads(port.output)
        if kind == "model":
            output["model"] = "invented_other_model"
        else:
            output["choices"][0]["message"]["content"] = ""
        port.output = canonical(output)
    denied(world, lambda: fixed_turn(world), 1, 1)
    failed, = world.adapter.failed_main_attempts
    record, = world.adapter.fallback_invocations
    assert record.failed_main is failed and record.envelope is port.sent[0]
    assert record.response_bytes is None if kind == "send" else record.response_bytes == port.output
    assert (record.result_content is None) is (kind == "send")
    assert record.output is None and world.adapter._fallback_frame.result is None
    retained = world.adapter.fallback_invocations
    world.agent.close()
    assert world.adapter.fallback_invocations is retained


@pytest.mark.parametrize("kind", ["source", "request", "proof", "reentry", "repeat", "reason"])
def test_owned_activation_rejects_nested_changes_and_reentry(fallback_excluded, kind):
    world = fallback_world()
    fail_main(world, ConnectionError(Hostile()))
    observed = []
    def tamper():
        frame = world.adapter._fallback_frame
        if frame is None or frame.request is None or observed:
            return
        if kind == "repeat" and not frame.activated:
            return
        observed.append(frame)
        if kind == "source":
            frame.failed_main.turn_frame.arguments[2].append({"role": "user", "content": "stale"})
        elif kind == "request":
            frame.request["messages"][0] = dict(frame.request["messages"][0])
        elif kind == "proof":
            object.__setattr__(frame.failed_main.failure_facts, "envelope", replace(frame.failed_main.failure_facts.envelope))
        elif kind == "reentry":
            world.agent._try_activate_fallback(controlled_failure=frame.token)
        elif kind == "repeat":
            world.adapter._activate_fallback(world.agent, frame.token)
        else:
            world.agent._try_activate_fallback("timeout", controlled_failure=frame.token)
    world.fallback_selection_hook = tamper
    denied(world, lambda: fixed_turn(world, history=[]), 1)
    assert observed and world.adapter.failed_main_attempts


def test_final_cancellation_retains_complete_result_proof(fallback_excluded, monkeypatch):
    original = ContentPolicyRuntime.derive_for_selection
    armed = []
    def cancel_after_final(runtime, handle, parents, payload, *, path, parent_requests):
        result = original(runtime, handle, parents, payload, path=path, parent_requests=parent_requests)
        adapter = world.adapter if world else None
        if (adapter is not None and adapter._fallback_frame is not None
                and adapter._fallback_frame.invocation is not None
                and len(adapter._fallback_frame.invocation.output_contents) == 2
                and payload == adapter._fallback_frame.invocation.output_stages[-1]):
            armed.append(result)
            world.agent._interrupt_requested = True
        return result
    world = None
    monkeypatch.setattr(ContentPolicyRuntime, "derive_for_selection", cancel_after_final)
    world = fallback_world()
    fail_main(world)
    with pytest.raises(InterruptedError) as caught:
        fixed_turn(world)
    assert type(caught.value) is InterruptedError
    assert caught.value.args == ("Agent interrupted during API call",)
    counts(world, 1, 1)
    frame = world.adapter._fallback_frame
    assert armed and frame.result_content is armed[-1]
    assert frame.result_bytes == canonical(frame.result)
    assert frame.invocation.output is frame.result
    assert world.context._denied and not world.adapter._fallback_terminal


def test_prior_success_ancestry_profile_removal_and_terminal_reuse(fallback_excluded, monkeypatch):
    world = fallback_world()
    first = fixed_turn(world)
    earlier = world.adapter._completions[0]
    world.binding = replace(world.binding, profiles=())
    world.fallback_binding = replace(world.fallback_binding, profiles=())
    fail_main(world)
    result = turn(world, "Keep it blue.", first["messages"], system_message="Be concise.")
    failed, = world.adapter.failed_main_attempts
    assert failed.parents[-2] is earlier.invocation.result_content
    assert failed.parents[-1] is earlier.content
    assert all(capability is world.request for capability in failed.parent_requests)
    assert first["total_tokens"] == 5 and result["total_usage_known"] is False
    assert world.adapter._completions == (earlier,)
    assert world.adapter.invocations == (earlier.invocation,)
    counts(world, 2, 1, successful=1)
    assert world.fallback_ports["fallback"].sent[0].profiles == ("restricted_documents",)
    from agent import title_generator
    monkeypatch.setattr(title_generator, "_summarize_user_message", fallback_excluded)
    denied(world, lambda: world.adapter.auxiliary_input(world.agent, source=first, field="final_response"),
           2, 1, successful=1)
    retained = world.adapter.failed_main_attempts, world.adapter.fallback_invocations
    for callback in (lambda: fixed_turn(world),
                     lambda: world.adapter.compression_input(world.agent, source=first),
                     lambda: world.agent._try_activate_fallback(controlled_failure=world.adapter._fallback_frame.token)):
        with pytest.raises(NativeContentPolicyDenied):
            callback()
    assert (world.adapter.failed_main_attempts, world.adapter.fallback_invocations) == retained


def test_wrong_thread_refuses_before_transform(fallback_excluded):
    world = fallback_world()
    errors = []
    def other_thread():
        try:
            fixed_turn(world)
        except BaseException as error:
            errors.append(error)
    worker = threading.Thread(target=other_thread)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert len(errors) == 1 and type(errors[0]) is NativeContentPolicyDenied
    assert world.context._denied
    counts(world, 0, 0)


def test_full_historical_prefix_keeps_bounds_and_original_parents(fallback_excluded, monkeypatch, record_property):
    from agent.title_generator import generate_title
    failures = []
    for name in ("admit", "derive", "derive_for_selection", "dispatch", "dispatch_attempt"):
        original = getattr(ContentPolicyRuntime, name)
        def observe(runtime, *args, _original=original, _name=name, **kwargs):
            from hermes_content_policy import PolicyError
            try:
                return _original(runtime, *args, **kwargs)
            except PolicyError as error:
                failures.append((_name, error.args))
                raise
        monkeypatch.setattr(ContentPolicyRuntime, name, observe)
    world = construct(HistoryFallbackWorld().bind(), max_tokens=None)
    bind_fallback(world)
    steps = []
    try:
        world.ports["main"].output = reply("")
        first = turn(world, "Keep the blue folder.")
        steps.append("main_A")
        with content_policy_scope(world.context):
            bind_title(world)
            bind_compression(world)
            compressor = construct_compressor(world)
            token = world.adapter.auxiliary_input(world.agent, source=first, field="final_response")
            assert generate_title(first["final_response"], controlled_aux=token) is None
            steps.append("local_title_A")
            summarize(world, compressor, first)
            steps.append("compression_A")
        world.ports["main"].output = reply("The blue folder stays.")
        second = turn(world, "Keep it blue.", first["messages"])
        steps.append("main_B")
        with content_policy_scope(world.context):
            title_from(world, second)
            steps.append("sent_title_B")
            summarize(world, compressor, second)
            steps.append("compression_B")
        fail_main(world)
        result = turn(world, "Keep it.", second["messages"])
        assert result is world.adapter._fallback_frame.result
        steps.append("fallback")
    except NativeContentPolicyDenied:
        assert world.context._denied
        assert failures and failures[-1][1] in (("parent_limit",), ("capacity_exhausted",))
    previous = tuple(world.adapter._completed_auxiliary_outputs())
    expected = (world.title_request, world.compression_request, world.title_request, world.compression_request)
    assert all(capability is expected[index] for index, (_, capability) in enumerate(previous))
    runtime = world.adapter._runtime
    slots = len(runtime._contents) + len(runtime._envelopes) + len(runtime._failures)
    assert slots <= 128 and runtime._retained_bytes <= 8 * 1024 * 1024
    assert world.adapter.send_count == len(world.ports["main"].sent)
    assert world.adapter.fallback_send_count == len(world.fallback_ports["fallback"].sent) <= 1
    assert world.adapter.auxiliary_send_count == sum(len(world.ports[key].sent) for key in
        ("title_capability_title", "compression_capability_compression"))
    record_property("historical_bound", json.dumps(dict(steps=steps, failures=failures, slots=slots,
        retained_bytes=runtime._retained_bytes, main_sends=world.adapter.send_count,
        fallback_sends=world.adapter.fallback_send_count, auxiliary_sends=world.adapter.auxiliary_send_count,
        failed_main_records=len(world.adapter.failed_main_attempts))))
    retained = world.adapter.failed_main_attempts, world.adapter.fallback_invocations
    world.agent.close()
    world.agent.close()
    assert (world.adapter.failed_main_attempts, world.adapter.fallback_invocations) == retained


def test_unqualified_siblings_refuse_after_profile_removal(fallback_excluded):
    from agent import auxiliary_client, title_generator
    from agent.memory_manager import MemoryManager
    from plugins.platforms.a2a import tools as a2a_tools
    from run_agent import AIAgent
    from tools import delegate_tool
    for operation in ("memory", "delegate", "a2a", "tool_result", "async", "stream", "background", "real_provider"):
        world = fallback_world()
        world.binding = replace(world.binding, profiles=())
        world.fallback_binding = replace(world.fallback_binding, profiles=())
        operations = {
            "memory": lambda: MemoryManager().prefetch_all("Invented input"),
            "delegate": lambda: delegate_tool.delegate_task(goal="Invented input", parent_agent=world.agent),
            "a2a": lambda: a2a_tools._send_task("document_review", {
                "url": "http://invented.invalid", "streaming": False}, "Invented input", "", follow=False),
            "tool_result": lambda: AIAgent._tool_result_content_for_active_model(
                world.agent, "invented_producer", "Invented output"),
            "async": lambda: asyncio.run(auxiliary_client.async_call_llm(messages=[])),
            "stream": lambda: chat.interruptible_streaming_api_call(world.agent, {}),
            "background": lambda: title_generator.auto_title_session(object(), "invented", "Invented input"),
            "real_provider": lambda: auxiliary_client.call_llm(provider="openai", messages=[]),
        }
        with content_policy_scope(world.context):
            denied(world, operations[operation])
