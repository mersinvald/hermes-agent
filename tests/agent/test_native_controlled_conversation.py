"""Construct real agents for invented controlled text turns."""

import copy
import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
from hermes_content_policy import ContentPolicyRuntime

from agent import conversation_loop, native_main_payload
from agent.content_policy import NativeContentPolicyDenied, content_policy_scope
from tests.agent.test_native_main_payload import (
    World, canonical, reply,
    invented_offline_configuration as invented_offline_configuration,
    caller_boundary as caller_boundary,
)


@pytest.fixture
def excluded(monkeypatch):
    from run_agent import AIAgent
    from agent import agent_init, background_review, turn_finalizer
    from hermes_cli import config, plugins
    import run_agent

    def forbidden(*args, **kwargs):
        pytest.fail("Excluded controlled lifecycle callback ran")

    for module, names in (
        (agent_init, ("_install_safe_stdio", "get_provider_request_timeout", "ContextCompressor",
                      "fetch_model_metadata", "_ra")),
        (config, ("load_config", "load_config_readonly")),
        (plugins, ("discover_plugins",)),
        (background_review, ("cancel_background_review_for_live_turn",)),
        (conversation_loop, ("_install_safe_stdio", "_restore_or_build_system_prompt",
                             "_sanitize_surrogates", "_summarize_user_message_for_log",
                             "set_session_context", "set_current_write_origin", "finalize_turn")),
        (turn_finalizer, ("finalize_turn",)),
        (run_agent, ("OpenAI", "cleanup_vm", "cleanup_browser", "_set_interrupt")),
        (AIAgent, ("_create_request_openai_client", "_create_request_anthropic_client",
                   "_close_request_openai_client", "_close_request_anthropic_client",
                   "_abort_request_openai_client", "_abort_request_anthropic_client",
                   "_touch_activity", "_try_refresh_env_client_credentials",
                   "_ensure_db_session", "_persist_session", "_save_trajectory",
                   "_cleanup_task_resources", "shutdown_memory_provider")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(run_agent, "warnings", SimpleNamespace(warn=forbidden))
    return forbidden


def construct(world=None, **changes):
    from run_agent import AIAgent
    world = World().bind() if world is None else world
    options = dict(controlled_main=world.adapter, provider="synthetic_native",
                   api_mode="chat_completions", model="invented_model", session_id="invented_session",
                   enabled_toolsets=[], skip_memory=True, skip_context_files=True,
                   skip_background_review=True, quiet_mode=True)
    options.update(changes)
    with content_policy_scope(world.context):
        world.agent = AIAgent(**options)
    return world


def turn(world, text="Invented document A", history=None, *, module=False, **changes):
    options = dict(system_message="Fixed system", conversation_history=history, task_id="execution")
    options.update(changes)
    if module:
        return conversation_loop.run_conversation(world.agent, text, **options)
    return world.agent.run_conversation(text, **options)


def refuses(world, callback, sends=0):
    with pytest.raises(NativeContentPolicyDenied):
        callback()
    assert world.context._denied and world.adapter._turn_pending is None
    assert world.adapter.send_count == sends
    assert len(world.ports["main"].sent) == sends
    assert not world.sdk_calls


@pytest.mark.parametrize("module_first", [False, True])
def test_real_constructor_first_and_repeated_entrypoints(excluded, caller_boundary, monkeypatch, module_first):
    derived = []
    derive = ContentPolicyRuntime.derive

    def observed(runtime, handle, parents, payload, *, path):
        content = derive(runtime, handle, parents, payload, path=path)
        derived.append((handle, parents, payload, content))
        return content

    monkeypatch.setattr(ContentPolicyRuntime, "derive", observed)
    world = construct(max_tokens=4096)
    from run_agent import AIAgent
    assert type(world.agent) is AIAgent
    assert world.agent.client is world.agent._primary_runtime is None
    assert "temperature" not in vars(world.agent)
    user = "Invented\ud800 A"
    first = turn(world, user, module=module_first)
    record = world.adapter.invocations[0]
    frame = world.adapter._turn_frames[0]
    complete = world.adapter._completions[0]
    assert frame.arguments[0] is user
    assert b'Invented\\ud800 A' in frame.raw_bytes
    assert b'Invented\\ufffd A' in frame.output_bytes
    assert json.loads(frame.output_bytes)["messages"] == [{"role": "user", "content": "Invented\ufffd A"}]
    assert record.response_bytes == reply()
    assert record.native_response is complete.invocation.native_response
    assert first["final_response"] is record.native_response.choices[0].message.content
    assert record.envelope is world.ports["main"].sent[0]
    assert complete.result is first and complete.payload == canonical(first)
    proof = next(row for row in derived if row[3] is complete.content)
    expected = (frame.admitted, frame.input_content, frame.output_content,
                *frame.request_contents, record.request_content, record.result_content)
    assert proof[0] is world.request and proof[2] == canonical(first)
    assert len(proof[1]) == len(expected)
    assert all(a is b for a, b in zip(proof[1], expected))
    assert first["messages"] is not frame.output.messages
    assert first["messages"][0] is not frame.output.messages[0]
    world.binding = replace(world.binding, profiles=())
    second = turn(world, "Invented B", first["messages"], module=not module_first)
    assert second["messages"] == first["messages"] + [
        {"role": "user", "content": "Invented B"}, {"role": "assistant", "content": "Invented answer A"}]
    assert (second["prompt_tokens"], second["completion_tokens"], second["total_tokens"]) == (6, 4, 10)
    assert first["total_tokens"] == 5 and second["api_calls"] == 1
    second_proof = next(row for row in derived if row[3] is world.adapter._completions[1].content)
    assert second_proof[1][-1] is complete.content
    assert world.adapter._turn_frames[1].arguments[2] is first["messages"]
    assert all(envelope.profiles == ("restricted_documents",) for envelope in world.ports["main"].sent)
    assert all(json.loads(row.request_bytes)["request"]["messages"][0] == {
        "role": "system", "content": "Fixed system"} for row in world.adapter.invocations)
    assert len(caller_boundary.dispatches) == world.adapter.send_count == 2
    assert all(row[0] == threading.get_ident() and row[1] is world.agent for row in caller_boundary.dispatches)
    assert all(not port.sent for path, port in world.ports.items() if path != "main")
    world.agent.close()
    world.agent.close()
    assert len(world.adapter.invocations) == 2
    refuses(world, lambda: turn(world, history=second["messages"]), 2)


def test_empty_text_stays_empty(excluded, caller_boundary):
    world = construct(max_tokens=None)
    world.ports["main"].output = reply("")
    result = turn(world, history=[])
    assert result["final_response"] == ""
    assert result["messages"][-1] == {"role": "assistant", "content": ""}
    assert result["estimated_cost_usd"] is None and result["cost_status"] == "unknown"
    assert "max_tokens" not in json.loads(world.adapter.invocations[0].request_bytes)["request"]
    assert world.adapter.send_count == 1


class Hostile:
    def __eq__(self, other):
        pytest.fail("Custom equality ran")

    def __bool__(self):
        pytest.fail("Custom Boolean conversion ran")

    def __str__(self):
        pytest.fail("Custom string conversion ran")

    def __int__(self):
        pytest.fail("Custom integer conversion ran")


@pytest.mark.parametrize("change", [
    {"tool_delay": Hostile()}, {"max_iterations": True}, {"quiet_mode": 1},
    {"provider": Hostile()}, {"api_mode": Hostile()}, {"model": Hostile()},
    {"session_id": "bad/session"}, {"enabled_toolsets": ()}, {"skip_memory": False},
    {"skip_context_files": False}, {"skip_background_review": False},
    {"api_key": "invented"}, {"base_url": "https://invented.invalid"},
    {"reasoning_config": {}}, {"request_overrides": {}}, {"prefill_messages": []},
    {"step_callback": Hostile()}, {"max_tokens": True}, {"max_tokens": 4097},
    {"model": "other"}, {"controlled_main": object()},
])
def test_constructor_options_refuse_before_callbacks(excluded, caller_boundary, change):
    world = World().bind()
    refuses(world, lambda: construct(world, **change))


def test_constructor_adapter_reuse_and_consumption(excluded, caller_boundary):
    from run_agent import AIAgent
    world = construct()
    refuses(world, lambda: construct(world))
    world.agent.close()
    world = World().bind()
    world.call()
    with pytest.raises(NativeContentPolicyDenied):
        construct(world)
    assert world.adapter.send_count == 1
    with pytest.raises(NativeContentPolicyDenied):
        AIAgent(provider="synthetic_native")


def test_constructor_requires_current_registered_adapter_and_exact_class(excluded, caller_boundary):
    from run_agent import AIAgent
    world = World().bind()
    with pytest.raises(NativeContentPolicyDenied):
        AIAgent(controlled_main=world.adapter)
    assert world.context._denied
    world = World().bind()

    class UnsupportedAgent(AIAgent):
        pass

    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            UnsupportedAgent(controlled_main=world.adapter)
    assert world.context._denied


@pytest.mark.parametrize("changes", [
    {"stream_callback": Hostile()}, {"moa_config": {}}, {"persist_user_message": "invented"},
    {"persist_user_timestamp": 1.0}, {"persist_user_display_kind": "invented"},
    {"persist_user_display_metadata": {}}, {"task_id": None}, {"task_id": "wrong"},
    {"conversation_history": [{"role": "user", "content": "Imported"}]},
    {"system_message": "bad\ud800"},
])
def test_entry_options_refuse_before_input_processing(excluded, caller_boundary, changes):
    world = construct()
    refuses(world, lambda: turn(world, Hostile(), **changes))
    assert not world.adapter._turn_frames


@pytest.mark.parametrize("change", ["context", "bound", "payload", "binding", "provider", "method", "client",
                                         "option_presence", "selection", "producer", "scope"])
def test_current_binding_changes_refuse(excluded, caller_boundary, change, monkeypatch):
    world = construct()
    if change == "context":
        world.agent._native_content_policy_context = object()
    elif change == "bound":
        world.context._controlled_main_bound = Hostile()
    elif change == "payload":
        world.context._controlled_main_payload = object()
    elif change == "binding":
        world.agent._native_controlled_binding = object()
    elif change == "provider":
        world.agent.provider = Hostile()
    elif change == "method":
        world.agent._interruptible_api_call = lambda *a: None
    elif change == "client":
        world.agent.client = object()
    elif change == "option_presence":
        world.agent.temperature = None
    elif change == "selection":
        world.selection = replace(world.selection, selection_version=4)
    elif change == "producer":
        world.policy = replace(world.policy, revision=5)
    else:
        world.context._denied = True
    refuses(world, lambda: turn(world))
    world.agent.close()
    world.agent.close()


@pytest.mark.parametrize("mutation", ["root", "message", "anchor", "response", "usage", "invocation"])
def test_earlier_results_remain_bound(excluded, caller_boundary, mutation):
    world = construct()
    first = turn(world)
    record = world.adapter.invocations[0]
    if mutation == "root":
        first["completed"] = False
    elif mutation == "message":
        first["messages"][0]["content"] = "Changed"
    elif mutation == "anchor":
        first["messages"][0] = dict(first["messages"][0])
    elif mutation == "response":
        object.__setattr__(record.native_response.choices[0].message, "content", "Changed")
    elif mutation == "usage":
        object.__setattr__(record.native_response, "usage", copy.copy(record.native_response.usage))
    else:
        object.__setattr__(record, "result_content", object())
    refuses(world, lambda: turn(world, history=first["messages"]), 1)
    assert len(world.adapter.invocations) == 1


@pytest.mark.parametrize("history_kind", ["missing", "copied", "system"])
def test_repeated_turn_requires_original_history_and_prefix(excluded, caller_boundary, history_kind):
    world = construct()
    first = turn(world)
    history = None if history_kind == "missing" else copy.deepcopy(first["messages"])
    options = {}
    if history_kind == "system":
        history = first["messages"]
        options["system_message"] = "Different prefix"
    refuses(world, lambda: turn(world, history=history, **options), 1)


@pytest.mark.parametrize("when", ["before", "admission", "send", "complete"])
def test_bare_interrupt_preserves_pending_cancellation(excluded, caller_boundary, monkeypatch, when):
    world = construct()
    if when == "before":
        world.agent.interrupt()
    elif when == "admission":
        world.admission_hook = world.agent.interrupt
    elif when == "send":
        world.ports["main"].hook = lambda envelope: world.agent.interrupt()
    else:
        derive = ContentPolicyRuntime.derive

        def cancel(runtime, handle, parents, payload, *, path):
            content = derive(runtime, handle, parents, payload, path=path)
            if b'"final_response"' in payload:
                world.agent.interrupt()
            return content

        monkeypatch.setattr(ContentPolicyRuntime, "derive", cancel)
    with pytest.raises(InterruptedError, match="^Agent interrupted during API call$"):
        turn(world)
    assert world.agent._interrupt_requested is True and world.context._denied
    assert world.agent._execution_thread_id is world.agent._active_request_abort is None
    assert not world.agent._active_children and not world.agent._tool_worker_threads
    assert world.adapter.send_count == (0 if when in ("before", "admission") else 1)
    assert len(world.adapter.invocations) == (0 if when in ("before", "admission") else 1)
    assert not world.adapter._completions and world.adapter._turn_pending is None
    world.agent.close()
    world.agent.close()


@pytest.mark.parametrize("flag", [1, Hostile()])
def test_interrupt_flag_requires_exact_boolean(excluded, caller_boundary, flag):
    world = construct()
    world.agent._interrupt_requested = flag
    refuses(world, lambda: turn(world))


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_process_control_exceptions_propagate(excluded, caller_boundary, exception):
    world = construct()
    error = exception()

    def fail(envelope):
        raise error

    world.ports["main"].hook = fail
    with pytest.raises(exception) as caught:
        turn(world)
    assert caught.value is error
    assert world.context._denied and world.adapter._turn_pending is None
    assert world.adapter.send_count == 1 and not world.adapter._completions
    world.agent.close()


def test_wrong_thread_and_caught_reentry_invalidate(excluded, caller_boundary):
    world = construct()
    errors = []

    def wrong_thread():
        try:
            turn(world)
        except NativeContentPolicyDenied as error:
            errors.append(error)

    thread = threading.Thread(target=wrong_thread)
    thread.start()
    thread.join(timeout=2)
    assert not thread.is_alive() and len(errors) == 1
    assert world.context._denied and world.adapter.send_count == 0
    world = construct()

    def reenter(envelope):
        with pytest.raises(NativeContentPolicyDenied):
            turn(world)

    world.ports["main"].hook = reenter
    refuses(world, lambda: turn(world), 1)


def test_completion_requires_latest_original_response_once(excluded, caller_boundary):
    world = construct()
    first = turn(world)
    frame = world.adapter._turn_frames[-1]
    response = world.adapter.invocations[-1].native_response
    refuses(world, lambda: native_main_payload._complete_controlled_turn(
        world.agent, turn_context=frame.output, response=response), 1)
    assert first["completed"] and len(world.adapter._completions) == 1


def test_eight_turn_limit(excluded, caller_boundary):
    world = construct()
    result = None
    for index in range(8):
        result = turn(world, str(index), None if result is None else result["messages"], module=index % 2 == 1)
    assert result["total_tokens"] == 40
    refuses(world, lambda: turn(world, history=result["messages"]), 8)


def test_cumulative_integer_bound_retains_actual_invocation(excluded, caller_boundary, monkeypatch):
    world = construct()
    monkeypatch.setattr(native_main_payload, "_MAX_INT", 4)
    refuses(world, lambda: turn(world), 1)
    assert len(world.adapter.invocations) == 1 and not world.adapter._completions


def test_close_recognizes_partial_closed_and_reserved_bindings(excluded, caller_boundary, monkeypatch):
    from run_agent import AIAgent
    retained = []
    original = native_main_payload._bind_constructed_main

    def failing(agent, adapter):
        retained.append(agent)
        raise KeyboardInterrupt

    monkeypatch.setattr(native_main_payload, "_bind_constructed_main", failing)
    world = World().bind()
    with pytest.raises(KeyboardInterrupt):
        construct(world)
    partial, = retained
    partial.close()
    partial.close()
    assert world.context._denied
    monkeypatch.setattr(native_main_payload, "_bind_constructed_main", original)
    world = construct()
    world.agent.close()
    world.agent.provider = "custom"
    refuses(world, lambda: turn(world, module=True))
    # A reserved marker always fences generic cleanup.
    AIAgent.close(SimpleNamespace(provider="synthetic_native"))


def test_result_byte_bound_releases_no_partial_result(excluded, caller_boundary, monkeypatch):
    world = construct()
    monkeypatch.setattr(native_main_payload, "_MAX_NATIVE_RESULT", 512)
    refuses(world, lambda: turn(world), 1)
    assert len(world.adapter.invocations) == 1 and not world.adapter._completions


@pytest.mark.parametrize("changed", ["response", "context"])
def test_completion_requires_exact_frame_and_response(excluded, caller_boundary, monkeypatch, changed):
    original = native_main_payload._complete_controlled_turn

    def different(agent, *, turn_context, response):
        return original(agent, turn_context=copy.copy(turn_context) if changed == "context" else turn_context,
                        response=copy.copy(response) if changed == "response" else response)

    monkeypatch.setattr(native_main_payload, "_complete_controlled_turn", different)
    world = construct()
    refuses(world, lambda: turn(world), 1)
    assert len(world.adapter.invocations) == 1 and not world.adapter._completions


@pytest.mark.parametrize("operation,drop_profiles", [
    ("fallback", False), ("compression", True), ("memory", False), ("delegate", True),
    ("a2a", False), ("tool_result", True), ("title", False), ("stream", True),
])
def test_actual_siblings_refuse_after_constructed_turn(excluded, caller_boundary, operation, drop_profiles):
    from agent import chat_completion_helpers, title_generator
    from agent.memory_manager import MemoryManager
    from plugins.platforms.a2a import tools as a2a_tools
    from run_agent import AIAgent
    from tools import delegate_tool
    world = construct()
    turn(world)
    if drop_profiles:
        world.binding = replace(world.binding, profiles=())
    operations = {
        "fallback": lambda: chat_completion_helpers.try_activate_fallback(world.agent),
        "compression": lambda: chat_completion_helpers.handle_max_iterations(world.agent, [], 1),
        "memory": lambda: MemoryManager().prefetch_all("Invented input"),
        "delegate": lambda: delegate_tool.delegate_task(goal="Invented input", parent_agent=world.agent),
        "a2a": lambda: a2a_tools._send_task("document_review", {
            "url": "http://invented.invalid", "streaming": False}, "Invented input", "", follow=False),
        "tool_result": lambda: AIAgent._tool_result_content_for_active_model(
            world.agent, "invented_producer", "Invented output"),
        "title": lambda: title_generator.generate_title("Invented input"),
        "stream": lambda: chat_completion_helpers.interruptible_streaming_api_call(world.agent, {}),
    }
    entered = False
    with pytest.raises(NativeContentPolicyDenied, match="^controlled_native_operation_unsupported$"):
        with content_policy_scope(world.context):
            entered = True
            operations[operation]()
    assert entered and world.context._denied and world.adapter.send_count == 1
    assert all(not port.sent for path, port in world.ports.items() if path != "main")


def test_ordinary_method_keeps_forwarding_and_instrumentation(monkeypatch):
    from run_agent import AIAgent
    from agent import background_review, relay_runtime
    from hermes_cli.observability import relay_shared_metrics
    events = []
    payload = object()
    history = []
    result = {"completed": True}
    agent = SimpleNamespace(provider="ordinary", model="invented_model", session_id="ordinary_session",
                            platform="cli", _conversation_root_id=lambda: "ordinary_session")

    def forwarded(candidate, *args, **kwargs):
        events.append(("forwarded", candidate, args, kwargs))
        return result

    monkeypatch.setattr(background_review, "cancel_background_review_for_live_turn",
                        lambda candidate: events.append(("review", candidate)))
    monkeypatch.setattr(conversation_loop, "run_conversation", forwarded)
    coordinator = SimpleNamespace(
        acquire_conversation=lambda **kw: events.append(("lease", kw)) or object(),
        begin_turn=lambda lease, **kw: events.append(("turn", kw)) or SimpleNamespace(relay_enabled=True),
        finish_logical_calls=lambda *a, **kw: events.append(("finish_calls",)),
        end_turn=lambda *a, **kw: events.append(("end",)),
        release_conversation=lambda *a: events.append(("release",)),
    )
    monkeypatch.setattr(relay_runtime, "SESSION_COORDINATOR", coordinator)
    monkeypatch.setattr(relay_shared_metrics, "start_task_run", lambda **kw: events.append(("start_task",)))
    monkeypatch.setattr(relay_shared_metrics, "finish_task_run", lambda **kw: events.append(("finish_task",)))
    assert AIAgent.run_conversation(agent, payload, "system", history, "task", persist_user_timestamp=12.0) is result
    call = next(event for event in events if event[0] == "forwarded")
    assert call[1] is agent and call[2][0] is payload and call[2][2] is history
    assert call[2][3] == "task" and call[3]["persist_user_timestamp"] == 12.0
    assert [event[0] for event in events] == [
        "review", "lease", "turn", "start_task", "forwarded", "finish_calls", "finish_task", "end", "release"]


def test_ordinary_module_keeps_prologue_forwarding(monkeypatch):
    user, system, history = object(), object(), []
    stream, persisted = object(), object()
    agent = SimpleNamespace(provider="ordinary", _try_refresh_env_client_credentials=lambda: None)
    calls = []
    stop = KeyboardInterrupt()

    def prologue(candidate, *args, **kwargs):
        calls.append((candidate, args, kwargs))
        raise stop

    monkeypatch.setattr(conversation_loop, "build_turn_context", prologue)
    with pytest.raises(KeyboardInterrupt) as caught:
        conversation_loop.run_conversation(agent, user, system, history, "task", stream, persisted,
                                           persist_user_timestamp=12.0, moa_config={"invented": True})
    assert caught.value is stop
    candidate, args, kwargs = calls[0]
    assert candidate is agent
    assert all(a is b for a, b in zip(args[:3], (user, system, history)))
    assert args[3:] == ("task", stream, persisted, 12.0)
    assert kwargs["moa_active"] is True
    assert kwargs["restore_or_build_system_prompt"] is conversation_loop._restore_or_build_system_prompt


def test_owned_binding_cannot_disappear_into_ordinary_fallback(excluded, caller_boundary):
    world = construct()
    binding = world.agent._native_controlled_binding
    del world.agent._native_controlled_binding
    del world.agent._native_controlled_options
    world.agent._native_content_policy_context = None
    world.agent.provider = "ordinary"
    refuses(world, lambda: turn(world))
    world.agent.close()
    assert binding.closed is True


@pytest.mark.parametrize("changed", ["session", "prepared", "options"])
def test_current_changes_inside_send_reject_release(excluded, caller_boundary, changed):
    world = construct()
    def change(envelope):
        if changed == "session":
            world.agent.session_id = "changed"
        elif changed == "prepared":
            world.adapter._turn_frames[-1].output.messages[0]["content"] = "Changed"
        else:
            world.agent.max_tokens = 1

    world.ports["main"].hook = change
    refuses(world, lambda: turn(world), 1)
    assert not world.adapter._completions


def test_unqualified_required_port_never_admits_or_sends(excluded, caller_boundary):
    world = construct(World(missing=("memory",)).bind(qualify=False))
    refuses(world, lambda: turn(world))
    assert not world.adapter._turn_frames


def test_entry_with_foreign_active_context_invalidates_both(excluded, caller_boundary):
    world, foreign = construct(), World().bind()
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(foreign.context):
            turn(world)
    assert world.context._denied and foreign.context._denied
    assert world.adapter.send_count == foreign.adapter.send_count == 0


def test_constructor_defaults_match_the_validation_contract():
    import inspect
    from run_agent import AIAgent
    from agent.agent_init import init_agent
    for function, skipped in ((AIAgent.__init__, "self"), (init_agent, "agent")):
        parameters = inspect.signature(function).parameters
        ordinary = {name: value.default for name, value in parameters.items()
                    if name not in (skipped, "controlled_main")}
        for name, default in ordinary.items():
            required = native_main_payload._CONSTRUCTOR_DEFAULTS[name]
            assert type(required) is type(default) and required == default
        assert set(ordinary) == set(native_main_payload._CONSTRUCTOR_DEFAULTS) - (
            {"tool_delay"} if function is init_agent else set())
        assert parameters["controlled_main"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["controlled_main"].default is None


def test_ordinary_close_keeps_cleanup(monkeypatch):
    from run_agent import AIAgent
    import run_agent
    from tools.process_registry import process_registry
    from tools import computer_use
    from hermes_cli import mem_trim
    events = []
    monkeypatch.setattr(process_registry, "kill_all", lambda **kw: events.append("process"))
    monkeypatch.setattr(run_agent, "cleanup_vm", lambda task: events.append("vm"))
    monkeypatch.setattr(run_agent, "cleanup_browser", lambda task: events.append("browser"))
    monkeypatch.setattr(computer_use, "release_computer_use_session", lambda task: events.append("computer"))
    monkeypatch.setattr(mem_trim, "trim_memory", lambda **kw: events.append("trim"))
    agent = SimpleNamespace(
        provider="ordinary", session_id="invented_session", _session_messages=[],
        shutdown_memory_provider=lambda messages: events.append("memory"),
        client=object(), _close_openai_client=lambda *a, **kw: events.append("client"),
        _close_cached_request_openai_client=lambda **kw: events.append("cached_openai"),
        _close_cached_request_anthropic_client=lambda **kw: events.append("cached_anthropic"),
    )
    AIAgent.close(agent)
    assert events == ["memory", "process", "vm", "browser", "computer", "client",
                      "cached_openai", "cached_anthropic", "trim"]
    assert agent.client is None


class ReservedProviderString(str):
    pass


class ReservedProviderEquality(str):
    def __eq__(self, other):
        pytest.fail("Custom provider equality ran")


class ReservedProviderConversions(str):
    def __bool__(self):
        pytest.fail("Custom provider Boolean conversion ran")

    def __str__(self):
        pytest.fail("Custom provider string conversion ran")

    def strip(self, *args, **kwargs):
        pytest.fail("Custom provider strip ran")


@pytest.mark.parametrize("provider_class", [
    str, ReservedProviderString, ReservedProviderEquality, ReservedProviderConversions,
], ids=["exact", "plain_subclass", "custom_equality", "custom_conversions"])
def test_constructor_reserved_provider_without_adapter_refuses(excluded, provider_class):
    from run_agent import AIAgent
    from agent.content_policy import current_context

    assert current_context() is None
    provider = provider_class("synthetic_native")
    with pytest.raises(NativeContentPolicyDenied):
        AIAgent(provider=provider)
    with pytest.raises(NativeContentPolicyDenied):
        AIAgent(provider=provider, tool_delay=0)
    assert current_context() is None


def test_constructor_exact_ordinary_provider_reaches_ordinary_initialization(excluded, monkeypatch):
    from run_agent import AIAgent
    from agent import agent_init
    from agent.content_policy import current_context

    class OrdinaryInitializationReached(Exception):
        pass

    calls = []

    def ordinary_stdio():
        calls.append("stdio")
        raise OrdinaryInitializationReached

    assert current_context() is None
    monkeypatch.setattr(agent_init, "_install_safe_stdio", ordinary_stdio)
    with pytest.raises(OrdinaryInitializationReached):
        AIAgent(provider="ordinary")
    assert calls == ["stdio"]


class AuxiliaryWorld(World):
    def __init__(self):
        from hermes_content_policy import PATHS, ProfilePolicy
        from agent.native_main_payload import NativeMainSelection, NativeResultPolicy
        super().__init__()
        self.title_request = object()
        self.title_binding = replace(self.binding, selection_id="title_selection", selection_version=5)
        self.title_policy = NativeResultPolicy("title_producer", 2, self.policy.profiles)
        from tests.agent.test_native_main_payload import ControlledPort
        self.title_ports = {}
        title_routes = []
        for row in self.routes:
            destination = replace(row.destination, transport_ref=f"title_transport_{row.path}",
                model_ref="invented_title_model")
            port = self.title_ports[row.path] = ControlledPort(destination)
            self.ports[f"title_capability_{row.path}"] = port
            self.receiver_bindings[id(port)] = (port, replace(self.title_binding,
                execution_ref=f"title_receiver_{row.path}"))
            value = json.loads(reply('{"title":"Invented document review"}'))
            value["model"] = "invented_title_model"
            port.output = canonical(value)
            title_routes.append(replace(row, route_ref=f"title_{row.path}", destination=destination,
                selection_id="title_selection", selection_version=5))
        self.routes += tuple(title_routes)
        self.profiles = (ProfilePolicy("restricted_documents", tuple(r.route_ref for r in self.routes), PATHS),)
        self.title_selection = NativeMainSelection("title_selection", 5, "chat_completions",
                                                   self.title_ports["title"].destination)
        self.title_selection_port = SimpleNamespace(current=self.title_selected)

    def current(self, handle):
        if handle is self.title_request:
            if self.admission_hook:
                self.admission_hook()
            return self.title_binding
        return super().current(handle)

    def produced(self, producer):
        if producer == "title_producer":
            if self.producer_hook:
                self.producer_hook()
            return self.title_policy
        return super().produced(producer)

    def title_selected(self, handle):
        assert handle is self.title_request
        return self.title_selection

    def qualify(self):
        from hermes_content_policy import PATHS
        super().qualify()
        runtime = ContentPolicyRuntime(admission=self.admission, profiles=self.profiles,
            routes=self.routes, transports=tuple((p.destination.transport_ref, p) for p in self.ports.values()),
            clock=lambda: self.now, policy_revision=9)
        for path in PATHS:
            source = runtime.admit(self.title_request, b"Invented independent title qualification")
            runtime.dispatch(self.title_request, source, path=path, route_ref=f"title_{path}")
            self.title_ports[path].sent = []


def auxiliary_world():
    world = construct(AuxiliaryWorld().bind())
    return world


def bind_title(world):
    world.adapter.bind_auxiliary(world.agent, path="title", task="title_generation",
        request_handle=world.title_request, selection=world.title_selection_port,
        producer_ref="title_producer", route_ref="title_title", language="English")


def title_from(world, result, field="user_message", **options):
    from agent.title_generator import generate_title
    token = world.adapter.auxiliary_input(world.agent, source=result, field=field)
    text = (world.adapter._completions[-1].frame.arguments[0] if field == "user_message"
            else result["final_response"])
    return generate_title(text, controlled_aux=token, **options)


@pytest.mark.parametrize("field", ["user_message", "final_response"])
def test_retained_title_main_title_with_independent_selections(excluded, caller_boundary, monkeypatch, field):
    from agent import auxiliary_client as auxiliary
    derived = []
    derive = ContentPolicyRuntime.derive_for_selection
    def observed(runtime, handle, parents, payload, *, path, parent_requests):
        result = derive(runtime, handle, parents, payload, path=path, parent_requests=parent_requests)
        derived.append((handle, parents, payload, path, parent_requests, result))
        return result
    monkeypatch.setattr(ContentPolicyRuntime, "derive_for_selection", observed)
    for name in ("_get_cached_client", "_acquire_sync_aux_semaphore", "_call_llm_impl",
                 "_resolve_task_provider_model", "_relay_auxiliary_metadata"):
        monkeypatch.setattr(auxiliary, name, excluded)
    monkeypatch.setattr(auxiliary.uuid, "uuid4", excluded)
    monkeypatch.setattr(auxiliary, "time", SimpleNamespace(monotonic=excluded))
    world = auxiliary_world()
    user = "<command-message>Invented document " + "A" * 1200 + "</command-message>"
    first = turn(world, user)
    with content_policy_scope(world.context):
        bind_title(world)
        title = title_from(world, first, field)
    record, = world.adapter.auxiliary_invocations
    assert title == "Invented document review" and record.output["title"] is title
    assert record.source_completion.result is first
    assert record.source_frame is world.adapter._turn_frames[0]
    assert record.source_invocation is world.adapter.invocations[0]
    assert record.source_handle is world.request and record.request_handle is world.title_request
    assert record.source_bytes == canonical(user if field == "user_message" else first["final_response"])
    assert record.input_stages[0] == record.source_bytes
    selected_text = user if field == "user_message" else first["final_response"]
    from agent.title_generator import _summarize_user_message, MAX_TITLE_INPUT_CHARS
    assert json.loads(record.input_stages[2]) == _summarize_user_message(selected_text)
    assert json.loads(record.input_stages[3]) == _summarize_user_message(selected_text)[:MAX_TITLE_INPUT_CHARS]
    assert json.loads(record.input_stages[5]) == json.loads(record.request_bytes)["request"]["messages"][0]["content"]
    assert json.loads(record.input_stages[6]) == json.loads(record.request_bytes)["request"]["messages"]
    assert record.output_stages[0] == canonical(record.native_response.choices[0].message.content)
    assert record.output_stages[2] == canonical(title)
    assert len(record.input_stages) == 7 and len(record.output_stages) == 4
    request = json.loads(record.request_bytes)
    assert request["request"]["model"] == "invented_title_model"
    assert request["request"]["messages"][0]["content"].find("Write the title in English") > 0
    assert record.response_bytes is world.title_ports["title"].output
    assert record.envelope is world.title_ports["title"].sent[0]
    assert record.output_bytes == canonical({"title": title})
    bridge = next(row for row in derived if row[-1] is world.adapter._aux_frames[0][0].source_content)
    assert bridge[0] is world.title_request and bridge[1] == record.parents
    assert all(a is b for a, b in zip(bridge[4], record.parent_requests))
    frame = record.source_frame
    required = (frame.admitted, frame.input_content, frame.output_content, *frame.request_contents,
                record.source_invocation.request_content, record.source_invocation.result_content,
                record.source_completion.content)
    assert all(any(parent is item for parent in record.parents) for item in required)
    world.binding = replace(world.binding, profiles=())
    world.title_binding = replace(world.title_binding, profiles=())
    second = turn(world, "Invented second request", first["messages"])
    with content_policy_scope(world.context):
        second_title = title_from(world, second, "final_response")
    assert second_title == title
    assert first["total_tokens"] == 5 and second["total_tokens"] == 10
    assert world.adapter.send_count == 2 and world.adapter.auxiliary_send_count == 2
    assert world.agent.model == "invented_model"
    assert world.adapter.invocations[0] is record.source_invocation
    assert world.adapter.auxiliary_invocations[0] is record
    assert all(e.profiles == ("restricted_documents",) for e in world.title_ports["title"].sent)
    assert all(e.profiles == ("restricted_documents",) for e in world.ports["main"].sent)
    returns = [row for row in derived if row[0] is world.request]
    assert returns and all(any(parent is record.output_content for parent in row[1]) for row in returns)
    assert all(any(capability is world.title_request for capability in row[4]) for row in returns)
    world.agent.close()
    world.agent.close()
    assert len(world.adapter.auxiliary_invocations) == 2


@pytest.mark.parametrize("change", ["owner", "execution", "lineage", "consumer", "epoch", "policy",
    "selection", "model", "destination", "producer", "expiry", "helper", "nested_source", "route"])
def test_retained_title_changed_facts_refuse_before_summary(excluded, caller_boundary, monkeypatch, change):
    from agent import title_generator as titles, skill_commands
    monkeypatch.setattr(skill_commands, "describe_skill_invocation", excluded)
    world = auxiliary_world()
    first = turn(world)
    with content_policy_scope(world.context):
        bind_title(world)
        token = world.adapter.auxiliary_input(world.agent, source=first, field="final_response")
    fields = dict(owner="owner_ref", execution="execution_ref", lineage="lineage_ref",
        consumer="consumer_ref", epoch="auth_epoch", policy="policy_revision", selection="selection_version")
    if change in fields:
        name = fields[change]
        value = getattr(world.title_binding, name)
        world.title_binding = replace(world.title_binding, **{name: value + 1 if type(value) is int else "changed"})
    elif change == "model":
        world.title_selection = replace(world.title_selection,
            destination=replace(world.title_selection.destination, model_ref="changed"))
    elif change == "destination":
        world.title_ports["title"].destination = replace(world.title_ports["title"].destination, endpoint_ref="changed")
    elif change == "producer":
        world.title_policy = replace(world.title_policy, profiles=())
    elif change == "expiry":
        world.now = 100
    elif change == "helper":
        monkeypatch.setattr(titles, "strip_control_wrappers", excluded)
    elif change == "nested_source":
        first["messages"][0]["content"] = "Changed"
    elif change == "route":
        object.__setattr__(world.adapter._aux_binding.route, "qualification_ref", "changed")
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            titles.generate_title(first["final_response"], controlled_aux=token)
    assert world.adapter.send_count == 1 and world.adapter.auxiliary_send_count == 0
    assert not world.title_ports["title"].sent and world.adapter._aux_pending is None


@pytest.mark.parametrize("option", ["timeout", "failure_callback", "main_runtime", "runtime_validator", "destination"])
def test_retained_title_unsupported_options_refuse_without_custom_operations(excluded, caller_boundary, option):
    world = auxiliary_world()
    first = turn(world)
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            bind_title(world)
            title_from(world, first, **{option: Hostile()})
    assert world.adapter.send_count == 1 and world.adapter.auxiliary_send_count == 0


@pytest.mark.parametrize("kind", ["copied_result", "detached_text", "missing", "copied_token", "wrong_field", "pending_main"])
def test_retained_title_input_requires_original_correspondence(excluded, caller_boundary, kind):
    from agent import title_generator as titles
    world = auxiliary_world()
    first = turn(world)
    with content_policy_scope(world.context):
        bind_title(world)
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            if kind == "copied_result":
                world.adapter.auxiliary_input(world.agent, source=dict(first), field="final_response")
            elif kind == "wrong_field":
                world.adapter.auxiliary_input(world.agent, source=first, field="messages")
            elif kind == "pending_main":
                world.adapter.admit_turn(world.agent, user_message="Invented B", system_message="Fixed system",
                    conversation_history=first["messages"], task_id="execution")
                world.adapter.auxiliary_input(world.agent, source=first, field="final_response")
            elif kind == "missing":
                titles.generate_title(first["final_response"])
            else:
                token = world.adapter.auxiliary_input(world.agent, source=first, field="final_response")
                if kind == "copied_token":
                    token = object.__new__(type(token))
                    text = first["final_response"]
                else:
                    text = first["final_response"].encode().decode()
                    assert text is not first["final_response"]
                titles.generate_title(text, controlled_aux=token)
    assert world.adapter.auxiliary_send_count == 0 and world.adapter._aux_pending is None


@pytest.mark.parametrize("field,value", [("max_tokens", True), ("temperature", Hostile()),
    ("stream", True), ("tools", []), ("base_url", "http://invented.invalid"),
    ("api_key", "invented"), ("extra_headers", {}), ("extra_body", {"other": Hostile()}),
    ("reasoning_config", {}), ("route_info", {}), ("latency_info", {}), ("provider", Hostile()),
    ("copied_messages", None), ("copied_extra_body", None), ("copied_model", None)])
def test_retained_decorated_call_checks_exact_sealed_arguments(excluded, caller_boundary, field, value):
    from agent import auxiliary_client as auxiliary
    world = auxiliary_world()
    first = turn(world)
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            bind_title(world)
            token = world.adapter.auxiliary_input(world.agent, source=first, field="final_response")
            stages, request = world.adapter._transform_auxiliary(token)
            sealed = world.adapter.seal_auxiliary(world.agent, token, stages=stages, request=request)
            arguments = dict(request, controlled_aux=sealed)
            if field.startswith("copied_"):
                original_field = field.removeprefix("copied_")
                original = arguments[original_field]
                arguments[original_field] = (original.encode().decode() if type(original) is str
                                             else copy.deepcopy(original))
            else:
                arguments[field] = value
            auxiliary.call_llm(**arguments)
    assert world.adapter.auxiliary_send_count == 0 and world.adapter._aux_pending is None


@pytest.mark.parametrize("when", ["before", "during_send", "output"])
def test_retained_title_cancellation_keeps_actual_sends_and_records(excluded, caller_boundary, monkeypatch, when):
    from agent import title_generator as titles, agent_runtime_helpers
    world = auxiliary_world()
    first = turn(world)
    if when == "output":
        original = agent_runtime_helpers.strip_think_blocks
        def stop(*args):
            world.agent.interrupt()
            return original(*args)
        monkeypatch.setattr(agent_runtime_helpers, "strip_think_blocks", stop)
        world.title_ports["title"].output = canonical(dict(json.loads(world.title_ports["title"].output),
            choices=[{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": "Prose title"}}]))
    with content_policy_scope(world.context):
        bind_title(world)
        token = world.adapter.auxiliary_input(world.agent, source=first, field="final_response")
    if when == "before":
        world.agent.interrupt()
    elif when == "during_send":
        world.title_ports["title"].hook = lambda envelope: world.agent.interrupt()
    with pytest.raises(InterruptedError, match="^Agent interrupted during API call$"):
        with content_policy_scope(world.context):
            titles.generate_title(first["final_response"], controlled_aux=token)
    expected = int(when != "before")
    assert world.adapter.auxiliary_send_count == expected
    assert len(world.adapter.auxiliary_invocations) == expected
    assert all(r.output is None for r in world.adapter.auxiliary_invocations)
    assert world.adapter._aux_pending is None and world.context._denied
    if when == "during_send":
        assert world.adapter.auxiliary_invocations[0].response_bytes is world.title_ports["title"].output


@pytest.mark.parametrize("content", ["", '{"title":"one two three four five six seven eight nine ten eleven twelve thirteen"}'])
def test_retained_title_semantic_none_has_complete_output_proof(excluded, caller_boundary, content):
    world = auxiliary_world()
    first = turn(world)
    value = json.loads(world.title_ports["title"].output)
    value["choices"][0]["message"]["content"] = content
    world.title_ports["title"].output = canonical(value)
    with content_policy_scope(world.context):
        bind_title(world)
        title = title_from(world, first)
    record, = world.adapter.auxiliary_invocations
    assert title is None and record.output == {"title": None}
    assert record.output_content is not None and record.output_bytes == b'{"title":null}'
    assert len(record.output_stages) == 4 and world.adapter.auxiliary_send_count == 1


def test_retained_title_empty_snippet_refuses_without_fabricated_result(excluded, caller_boundary):
    world = auxiliary_world()
    first = turn(world, "   ")
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            bind_title(world)
            title_from(world, first)
    assert world.adapter.auxiliary_send_count == 0 and not world.adapter.auxiliary_invocations


@pytest.mark.parametrize("kind", ["token", "request", "input_stage", "output", "response"])
def test_retained_title_replay_and_nested_mutation_invalidate(excluded, caller_boundary, kind):
    from agent import title_generator as titles, auxiliary_client as auxiliary
    world = auxiliary_world()
    first = turn(world)
    with content_policy_scope(world.context):
        bind_title(world)
        token = world.adapter.auxiliary_input(world.agent, source=first, field="final_response")
        stages, request = world.adapter._transform_auxiliary(token)
        sealed = world.adapter.seal_auxiliary(world.agent, token, stages=stages, request=request)
        response = auxiliary.call_llm(**request, controlled_aux=sealed)
        stages, output = world.adapter._transform_auxiliary(sealed, response=response)
        world.adapter.complete_auxiliary(sealed, response=response, stages=stages, output=output)
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            if kind == "token":
                titles.generate_title(first["final_response"], controlled_aux=token)
            elif kind == "request":
                auxiliary.call_llm(**request, controlled_aux=sealed)
            else:
                record = world.adapter.auxiliary_invocations[0]
                if kind == "input_stage":
                    request["messages"][1]["content"] = "Changed"
                elif kind == "output":
                    record.output["title"] = "Changed"
                else:
                    object.__setattr__(record.native_response.choices[0].message, "content", "Changed")
                turn(world, "Invented B", first["messages"])
    assert world.adapter.auxiliary_send_count == 1 and world.adapter.send_count == 1
    assert len(world.adapter.auxiliary_invocations) == 1


@pytest.mark.parametrize("kind", ["reentry", "thread", "process", "close"])
def test_retained_title_lifecycle_refusal_preserves_owned_records(excluded, caller_boundary, kind):
    from agent import title_generator as titles
    world = auxiliary_world()
    first = turn(world)
    with content_policy_scope(world.context):
        bind_title(world)
        token = world.adapter.auxiliary_input(world.agent, source=first, field="final_response")
    if kind == "close":
        world.agent.close()
        world.agent.close()
        with pytest.raises(NativeContentPolicyDenied):
            titles.generate_title(first["final_response"], controlled_aux=token)
        expected = 0
    elif kind == "thread":
        import contextvars
        with content_policy_scope(world.context):
            copied = contextvars.copy_context()
        failures = []
        def attempt():
            try:
                copied.run(titles.generate_title, first["final_response"], controlled_aux=token)
            except NativeContentPolicyDenied as error:
                failures.append(error)
        worker = threading.Thread(target=attempt)
        worker.start()
        worker.join(timeout=10)
        assert not worker.is_alive() and len(failures) == 1
        expected = 0
    else:
        stop = KeyboardInterrupt()
        def callback(envelope):
            if kind == "process":
                raise stop
            with pytest.raises(NativeContentPolicyDenied):
                world.adapter.auxiliary_input(world.agent, source=first, field="final_response")
        world.title_ports["title"].hook = callback
        with pytest.raises(KeyboardInterrupt if kind == "process" else NativeContentPolicyDenied) as caught:
            with content_policy_scope(world.context):
                titles.generate_title(first["final_response"], controlled_aux=token)
        if kind == "process":
            assert caught.value is stop
        expected = 1
    assert world.adapter.send_count == 1 and world.adapter.auxiliary_send_count == expected
    assert len(world.adapter.auxiliary_invocations) == expected
    assert world.adapter.invocations[0] is world.adapter._completions[0].invocation
    assert world.context._denied and world.adapter._aux_pending is None


@pytest.mark.parametrize("kind", ["stage_bytes", "stage_count", "oversized_stage", "request_identity", "dependency"])
def test_retained_title_preparation_refuses_unproved_stages(excluded, caller_boundary, kind):
    world = auxiliary_world()
    first = turn(world)
    with content_policy_scope(world.context):
        bind_title(world)
        token = world.adapter.auxiliary_input(world.agent, source=first, field="final_response")
        stages, request = world.adapter._transform_auxiliary(token)
    if kind == "stage_bytes":
        stages = (b'"caller label"', *stages[1:])
    elif kind == "stage_count":
        stages = (*stages, b'"eighth"', b'"ninth"')
    elif kind == "oversized_stage":
        stages = (b'x' * 262145, *stages[1:])
    elif kind == "request_identity":
        request = copy.deepcopy(request)
    else:
        world.adapter._runtime.derive_for_selection = None
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            world.adapter.seal_auxiliary(world.agent, token, stages=stages, request=request)
    assert world.adapter.auxiliary_send_count == 0 and not world.adapter.auxiliary_invocations


def test_retained_title_capacity_keeps_every_accepted_parent(excluded, caller_boundary):
    world = auxiliary_world()
    first = turn(world)
    with content_policy_scope(world.context):
        bind_title(world)
    accepted = []
    for _ in range(9):
        try:
            with content_policy_scope(world.context):
                title_from(world, first)
        except NativeContentPolicyDenied:
            break
        accepted.append(world.adapter.auxiliary_invocations[-1])
    else:
        pytest.fail("Auxiliary capacity did not refuse")
    assert 0 < len(accepted) <= 8
    assert all(world.adapter.auxiliary_invocations[i] is record for i, record in enumerate(accepted))
    assert len(world.adapter.auxiliary_invocations) <= 8
    assert world.adapter.auxiliary_send_count == len(world.title_ports["title"].sent)
    assert world.context._denied and world.adapter._aux_pending is None
    for index, record in enumerate(accepted):
        assert all(any(parent is previous.output_content for parent in record.parents)
                   for previous in accepted[:index])


def test_retained_title_missing_dependency_refuses_before_transform(excluded, caller_boundary, monkeypatch):
    from agent import skill_commands
    world = auxiliary_world()
    turn(world)
    monkeypatch.setattr(skill_commands, "describe_skill_invocation", excluded)
    monkeypatch.delattr(ContentPolicyRuntime, "derive_for_selection")
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            bind_title(world)
    assert not world.adapter._aux_frames and world.adapter.auxiliary_send_count == 0


@pytest.mark.parametrize("side", ["source", "target"])
def test_retained_title_profile_removal_keeps_required_destination_coverage(excluded, caller_boundary, side):
    world = auxiliary_world()
    first = turn(world)
    with content_policy_scope(world.context):
        bind_title(world)
        token = world.adapter.auxiliary_input(world.agent, source=first, field="final_response")
    world.binding = replace(world.binding, profiles=())
    world.title_binding = replace(world.title_binding, profiles=())
    port = world.ports["memory"] if side == "source" else world.title_ports["memory"]
    port.destination = replace(port.destination, endpoint_ref="unapproved")
    from agent.title_generator import generate_title
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            generate_title(first["final_response"], controlled_aux=token)
    assert world.adapter.auxiliary_send_count == 0


@pytest.mark.parametrize("change", ["binding", "producer", "destination", "source", "expiry", "failure"])
def test_retained_title_post_send_changes_keep_honest_records(excluded, caller_boundary, change):
    world = auxiliary_world()
    first = turn(world)
    with content_policy_scope(world.context):
        bind_title(world)
    def changed(envelope):
        if change == "binding":
            world.title_binding = replace(world.title_binding, auth_epoch=8)
        elif change == "producer":
            world.title_policy = replace(world.title_policy, revision=3)
        elif change == "destination":
            port = world.title_ports["title"]
            port.destination = replace(port.destination, endpoint_ref="changed")
        elif change == "source":
            first["messages"][0]["content"] = "Changed"
        elif change == "expiry":
            world.now = 100
        else:
            raise RuntimeError("Invented port failure")
    world.title_ports["title"].hook = changed
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            title_from(world, first)
    record, = world.adapter.auxiliary_invocations
    assert record.envelope is world.title_ports["title"].sent[0]
    assert record.output is None and record.output_content is None
    assert record.response_bytes is (None if change == "failure" else world.title_ports["title"].output)
    assert world.adapter.auxiliary_send_count == 1 and world.adapter.send_count == 1
    assert world.context._denied and world.adapter._aux_pending is None


def test_retained_title_original_surrogates_cannot_select_sanitized_source(excluded, caller_boundary):
    world = auxiliary_world()
    first = turn(world, "Invented\ud800 source")
    assert first["messages"][0]["content"] == "Invented\ufffd source"
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            bind_title(world)
            world.adapter.auxiliary_input(world.agent, source=first, field="user_message")
    assert world.adapter.auxiliary_send_count == 0
    assert b'\\ud800' in world.adapter._turn_frames[0].raw_bytes


def test_retained_title_published_records_never_change_during_completion(excluded, caller_boundary):
    from agent import auxiliary_client as auxiliary
    world = auxiliary_world()
    first = turn(world)
    with content_policy_scope(world.context):
        bind_title(world)
        title_from(world, first)
        original, = world.adapter.auxiliary_invocations
        token = world.adapter.auxiliary_input(world.agent, source=first, field="final_response")
        stages, request = world.adapter._transform_auxiliary(token)
        sealed = world.adapter.seal_auxiliary(world.agent, token, stages=stages, request=request)
        response = auxiliary.call_llm(**request, controlled_aux=sealed)
        assert world.adapter.auxiliary_invocations == (original,)
        assert world.adapter.auxiliary_invocations[0] is original
        assert world.adapter.auxiliary_send_count == 2
        pending = world.adapter._aux_pending.record
        assert pending.native_response is response and pending.output is None
        stages, output = world.adapter._transform_auxiliary(sealed, response=response)
        world.adapter.complete_auxiliary(sealed, response=response, stages=stages, output=output)
        assert world.adapter.auxiliary_invocations[0] is original
        assert len(world.adapter.auxiliary_invocations) == 2
        assert world.adapter.auxiliary_invocations[1].native_response is response
        assert world.adapter.auxiliary_invocations[1].output is output
