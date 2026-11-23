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


def test_retained_title_empty_snippet_completes_local_without_send(excluded, caller_boundary):
    world = auxiliary_world()
    first = turn(world, "   ")
    invocation, = world.adapter.invocations
    with content_policy_scope(world.context):
        bind_title(world)
        title = title_from(world, first)
    record, = world.adapter.auxiliary_local_results
    assert title is None and record.output["title"] is title
    assert record.output == {"title": None} and record.output_bytes == b'{"title":null}'
    assert record.source_completion.result is first and record.source_invocation is invocation
    assert record.source_handle is world.request and record.request_handle is world.title_request
    assert record.source_bytes == canonical("   ") == record.input_stages[0]
    assert tuple(map(json.loads, record.input_stages)) == (
        "   ", dict(step="summary", mode="original", value="   "),
        dict(step="wrappers", applied=True, value=""), "",
        dict(step="snippet_decision", max_chars=1000, has_text=False))
    assert record.output_stages == (canonical(dict(outcome="local_no_send", reason="empty_title_snippet")),)
    assert len(record.stage_contents) == 5 and len(record.output_contents) == 1
    assert record.output_content is not record.source_content
    assert record.source_content is world.adapter._aux_frames[0][0].source_content
    assert all(capability is world.request for capability in record.parent_requests)
    assert any(parent is invocation.result_content for parent in record.parents)
    assert world.adapter.send_count == 1 and world.adapter.invocations == (invocation,)
    assert first["total_tokens"] == sum(r.native_response.usage.total_tokens for r in world.adapter.invocations) == 5
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


def block_title_edges(monkeypatch, excluded, *, local=True):
    from agent import title_generator as titles, auxiliary_client as auxiliary
    for name in ("_get_cached_client", "_acquire_sync_aux_semaphore", "_call_llm_impl",
                 "_resolve_task_provider_model", "_relay_auxiliary_metadata"):
        monkeypatch.setattr(auxiliary, name, excluded)
    monkeypatch.setattr(auxiliary.uuid, "uuid4", excluded)
    monkeypatch.setattr(auxiliary, "time", SimpleNamespace(monotonic=excluded))
    for name in ("_auto_title_enabled", "_title_language", "derive_title", "apply_instant_title",
                 "auto_title_session", "_auto_title_session", "maybe_auto_title", "_persist_session_title"):
        monkeypatch.setattr(titles, name, excluded)
    monkeypatch.setattr(titles, "threading", SimpleNamespace(Thread=excluded))
    if local:
        from agent.content_policy import current_context
        dispatch = ContentPolicyRuntime.dispatch
        def no_auxiliary_send(runtime, handle, content, *, path, route_ref):
            context = current_context()
            adapter = None if context is None else context._controlled_main_payload
            if adapter is not None and adapter._aux_pending is not None:
                excluded()
            return dispatch(runtime, handle, content, path=path, route_ref=route_ref)
        monkeypatch.setattr(ContentPolicyRuntime, "dispatch", no_auxiliary_send)
        for name in ("call_llm", "_title_prompt", "_title_output_stages", "_extract_title_text",
                     "_clean_title", "_title_word_decision"):
            monkeypatch.setattr(titles, name, excluded)


def observe_title_derivations(monkeypatch):
    derived = []
    derive = ContentPolicyRuntime.derive
    bridge = ContentPolicyRuntime.derive_for_selection
    def observed(runtime, handle, parents, payload, *, path):
        content = derive(runtime, handle, parents, payload, path=path)
        derived.append((handle, parents, payload, path, (handle,) * len(parents), content))
        return content
    def crossed(runtime, handle, parents, payload, *, path, parent_requests):
        content = bridge(runtime, handle, parents, payload, path=path, parent_requests=parent_requests)
        derived.append((handle, parents, payload, path, parent_requests, content))
        return content
    monkeypatch.setattr(ContentPolicyRuntime, "derive", observed)
    monkeypatch.setattr(ContentPolicyRuntime, "derive_for_selection", crossed)
    return derived


@pytest.mark.parametrize("field,text", [
    ("final_response", ""), ("user_message", ""), ("final_response", " \t\n"),
    ("user_message", "<system-reminder></system-reminder>"),
    ("user_message", '[IMPORTANT: The user has invoked the "" skill. The full skill content is loaded below.]'
        '\nInvented skill body.\nThe user has provided the following instruction alongside the skill invocation: '
        '<system-reminder></system-reminder>'),
])
def test_local_title_actual_empty_sources_have_genuine_complete_proof(
        excluded, caller_boundary, monkeypatch, field, text):
    from dataclasses import FrozenInstanceError
    from agent import title_generator as titles, skill_commands
    from hermes_content_policy import ContentHandle
    derived = observe_title_derivations(monkeypatch)
    block_title_edges(monkeypatch, excluded)
    calls = []
    describe, wrappers = skill_commands.describe_skill_invocation, titles.strip_control_wrappers
    def described(value):
        calls.append(("summary", value))
        return describe(value)
    def wrapped(value):
        calls.append(("wrappers", value))
        return wrappers(value)
    monkeypatch.setattr(skill_commands, "describe_skill_invocation", described)
    monkeypatch.setattr(titles, "strip_control_wrappers", wrapped)
    world = auxiliary_world()
    world.ports["main"].output = reply(text)
    first = turn(world, text)
    main, = world.adapter.invocations
    with content_policy_scope(world.context):
        bind_title(world)
        token = world.adapter.auxiliary_input(world.agent, source=first, field=field)
        selected_text = text if field == "user_message" else first["final_response"]
        assert titles.generate_title(selected_text, controlled_aux=token) is None
    record, = world.adapter.auxiliary_local_results
    frame = world.adapter._aux_frames[0][0]
    assert type(record) is native_main_payload.NativeAuxiliaryLocalResult
    assert not hasattr(record, "__dict__")
    with pytest.raises(FrozenInstanceError):
        record.reason = "changed"
    with pytest.raises(AttributeError):
        world.adapter.auxiliary_local_results = ()
    assert frame.preparation.kind == record.outcome == "local_no_send"
    assert frame.preparation.stages is record.input_stages
    assert frame.preparation.value is record.output and record.output == {"title": None}
    assert record.operation_index == 1 and record.reason == "empty_title_snippet"
    assert record.language == "English" and record.path == "title" and record.task == "title_generation"
    assert record.selection is world.title_selection and record.producer_policy is world.title_policy
    assert record.source is world.binding and record.target is world.title_binding
    assert record.source_completion is world.adapter._completions[0]
    assert record.source_frame is world.adapter._turn_frames[0] and record.source_invocation is main
    assert record.source_handle is world.request and record.request_handle is world.title_request
    assert record.source_bytes == canonical(text) and frame.text is selected_text
    described_source = text.startswith('[IMPORTANT: The user has invoked the ""')
    summary = "<system-reminder></system-reminder>" if described_source else text
    mode = "described" if described_source else "original" if text else "empty_input"
    expected = (text, dict(step="summary", mode=mode, value=summary),
        dict(step="wrappers", applied=bool(text), value=""), "",
        dict(step="snippet_decision", max_chars=1000, has_text=False))
    assert record.input_stages == tuple(canonical(value) for value in expected)
    assert calls == ([("summary", text), ("wrappers", summary)] if text else [])
    previous = []
    for payload, content in zip(record.input_stages, record.stage_contents):
        proof = next(row for row in derived if row[-1] is content)
        assert proof[0] is world.title_request and proof[2] is payload and proof[3] == "title"
        assert proof[1] == (record.source_content, *previous)
        assert all(capability is world.title_request for capability in proof[4])
        previous.append(content)
    decision, = record.output_contents
    decision_proof = next(row for row in derived if row[-1] is decision)
    assert decision_proof[1] == (record.source_content, *record.stage_contents)
    assert decision_proof[2] is record.output_stages[0]
    final = next(row for row in derived if row[-1] is record.output_content)
    assert final[0] is world.title_request and final[1] == (record.source_content, *record.stage_contents, decision)
    assert final[2] is record.output_bytes and record.output_bytes == b'{"title":null}'
    assert type(record.output_content) is ContentHandle
    source_proof = next(row for row in derived if row[-1] is record.source_content)
    assert source_proof[1] is record.parents and source_proof[4] is record.parent_requests
    assert all(capability is world.request for capability in record.parent_requests)
    assert not any(hasattr(record, name) for name in (
        "request", "request_bytes", "envelope", "observed_destination", "response_bytes", "result_content", "native_response"))
    assert frame.request is frame.sealed is frame.record is None
    assert world.adapter.invocations == (main,) and world.adapter.invocations[0] is main
    assert world.adapter.send_count == 1 and world.agent.model == "invented_model"
    assert first["total_tokens"] == sum(r.native_response.usage.total_tokens for r in world.adapter.invocations) == 5
    assert world.adapter.auxiliary_send_count == 0 and world.adapter.auxiliary_invocations == ()
    assert all(not port.sent for path, port in world.ports.items() if path != "main")
    assert not world.sdk_calls and world.adapter._aux_pending is None
    world.agent.close()
    world.agent.close()
    assert world.adapter.auxiliary_local_results[0] is record


def test_local_title_then_main_then_sent_title_keeps_ordered_original_parents(
        excluded, caller_boundary, monkeypatch):
    derived = observe_title_derivations(monkeypatch)
    block_title_edges(monkeypatch, excluded, local=False)
    world = auxiliary_world()
    first = turn(world, "<system-reminder></system-reminder>")
    with content_policy_scope(world.context):
        bind_title(world)
        assert title_from(world, first) is None
    local, = world.adapter.auxiliary_local_results
    world.binding = replace(world.binding, profiles=())
    world.title_binding = replace(world.title_binding, profiles=())
    second = turn(world, "<system-reminder>Keep receipts</system-reminder>", first["messages"])
    with content_policy_scope(world.context):
        assert title_from(world, second) == "Invented document review"
    sent, = world.adapter.auxiliary_invocations
    assert world.adapter.auxiliary_local_results[0] is local
    assert len(world.adapter.invocations) == world.adapter.send_count == 2
    assert world.adapter.auxiliary_send_count == 1
    assert first["total_tokens"] == 5 and second["total_tokens"] == sum(r.native_response.usage.total_tokens for r in world.adapter.invocations) == 10
    assert world.agent.model == "invented_model"
    assert json.loads(sent.input_stages[2]) == json.loads(sent.input_stages[3]) == "Keep receipts"
    assert len(sent.input_stages) == 7 and len(sent.output_stages) == 4
    main_returns = [row for row in derived if row[0] is world.request and any(p is local.output_content for p in row[1])]
    assert main_returns
    for row in main_returns:
        assert row[1][-1] is local.output_content and row[4][-1] is world.title_request
    assert sent.parents[-1] is local.output_content and sent.parent_requests[-1] is world.title_request
    assert all(e.profiles == ("restricted_documents",) for e in (*world.ports["main"].sent, *world.title_ports["title"].sent))
    with content_policy_scope(world.context):
        token = world.adapter.auxiliary_input(world.agent, source=first, field="user_message")
        from agent.title_generator import generate_title
        assert generate_title(local.source_frame.arguments[0], controlled_aux=token) is None
    later = world.adapter.auxiliary_local_results[-1]
    assert later.operation_index == 3
    assert later.parents[-2:] == (local.output_content, sent.output_content)
    proof = next(row for row in derived if row[-1] is later.output_content)
    assert proof[1][-2:] == (local.output_content, sent.output_content)
    assert all(capability is world.title_request for capability in proof[4])
    assert world.adapter.auxiliary_local_results[0] is local and world.adapter.auxiliary_invocations[0] is sent
    assert tuple(world.adapter._completed_auxiliary_outputs()) == (
        (local.output_content, world.title_request), (sent.output_content, world.title_request),
        (later.output_content, world.title_request))
    world.agent.close()
    assert world.adapter.auxiliary_local_results[0] is local


@pytest.mark.parametrize("change", ["source_epoch", "target_epoch", "selection", "producer", "language",
    "destination", "context", "method", "nested_source", "expiry", "main_selection", "route"])
def test_local_title_current_facts_change_during_actual_summary_refuses(
        excluded, caller_boundary, monkeypatch, change):
    from agent import title_generator as titles, skill_commands
    block_title_edges(monkeypatch, excluded)
    world = auxiliary_world()
    world.ports["main"].output = reply(" \t\n")
    first = turn(world)
    original = skill_commands.describe_skill_invocation
    calls = []
    def changed(text):
        calls.append(text)
        result = original(text)
        if change == "source_epoch":
            world.binding = replace(world.binding, auth_epoch=8)
        elif change == "target_epoch":
            world.title_binding = replace(world.title_binding, auth_epoch=8)
        elif change == "selection":
            world.title_selection = replace(world.title_selection, selection_version=6)
        elif change == "producer":
            world.title_policy = replace(world.title_policy, revision=3)
        elif change == "language":
            object.__setattr__(world.adapter._aux_binding, "language", "Changed")
        elif change == "destination":
            port = world.title_ports["title"]
            port.destination = replace(port.destination, endpoint_ref="changed")
        elif change == "context":
            world.agent._native_content_policy_context = object()
        elif change == "method":
            monkeypatch.setattr(world.adapter, "complete_auxiliary_local", excluded)
        elif change == "nested_source":
            first["messages"][0]["content"] = "Changed"
        elif change == "expiry":
            world.now = 100
        elif change == "main_selection":
            world.selection = replace(world.selection, selection_version=4)
        else:
            object.__setattr__(world.adapter._aux_binding.route, "qualification_ref", "changed")
        return result
    monkeypatch.setattr(skill_commands, "describe_skill_invocation", changed)
    with content_policy_scope(world.context):
        bind_title(world)
        token = world.adapter.auxiliary_input(world.agent, source=first, field="final_response")
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            titles.generate_title(first["final_response"], controlled_aux=token)
    assert len(calls) == 1
    assert world.adapter.auxiliary_local_results == world.adapter.auxiliary_invocations == ()
    assert world.adapter.send_count == 1 and world.adapter.auxiliary_send_count == 0
    assert world.context._denied and world.adapter._aux_pending is None


@pytest.mark.parametrize("side", ["source", "target"])
def test_local_title_profile_removal_still_checks_all_captured_routes(
        excluded, caller_boundary, monkeypatch, side):
    block_title_edges(monkeypatch, excluded)
    world = auxiliary_world()
    first = turn(world, "   ")
    with content_policy_scope(world.context):
        bind_title(world)
        token = world.adapter.auxiliary_input(world.agent, source=first, field="user_message")
        preparation = world.adapter._prepare_auxiliary_title(token)
    world.binding = replace(world.binding, profiles=())
    world.title_binding = replace(world.title_binding, profiles=())
    port = world.ports["memory"] if side == "source" else world.title_ports["memory"]
    port.destination = replace(port.destination, endpoint_ref="unapproved")
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            world.adapter.complete_auxiliary_local(world.agent, token,
                stages=preparation.stages, output=preparation.value)
    assert preparation.kind == "local_no_send"
    assert not world.adapter.auxiliary_local_results and not world.adapter.auxiliary_invocations
    assert world.adapter.send_count == 1 and world.adapter.auxiliary_send_count == 0


@pytest.mark.parametrize("kind", ["unprepared", "copied_token", "copied_stages", "copied_output",
    "mutated_output", "mutated_preparation", "mutated_stage", "replay", "published_output"])
def test_local_title_completion_requires_owned_complete_preparation(
        excluded, caller_boundary, monkeypatch, kind):
    block_title_edges(monkeypatch, excluded)
    world = auxiliary_world()
    first = turn(world, "   ")
    with content_policy_scope(world.context):
        bind_title(world)
        token = world.adapter.auxiliary_input(world.agent, source=first, field="user_message")
        if kind == "unprepared":
            stages, output = (canonical("   "),), {"title": None}
        else:
            preparation = world.adapter._prepare_auxiliary_title(token)
            stages, output = preparation.stages, preparation.value
    if kind == "copied_token":
        token = object.__new__(type(token))
    elif kind == "copied_stages":
        stages = tuple(list(stages))
    elif kind == "copied_output":
        output = dict(output)
    elif kind == "mutated_output":
        output["title"] = "Changed"
    elif kind == "mutated_preparation":
        object.__setattr__(preparation, "kind", "dispatch")
    elif kind == "mutated_stage":
        object.__setattr__(preparation, "stages", (canonical("caller claim"), *stages[1:]))
    elif kind in ("replay", "published_output"):
        with content_policy_scope(world.context):
            assert world.adapter.complete_auxiliary_local(world.agent, token, stages=stages, output=output) is output
        published, = world.adapter.auxiliary_local_results
        if kind == "published_output":
            published.output["title"] = "Changed"
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            if kind == "published_output":
                turn(world, "Invented B", first["messages"])
            else:
                world.adapter.complete_auxiliary_local(world.agent, token, stages=stages, output=output)
    expected = int(kind in ("replay", "published_output"))
    assert len(world.adapter.auxiliary_local_results) == expected
    if expected:
        assert world.adapter.auxiliary_local_results[0] is published
    assert world.adapter.send_count == 1 and world.adapter.auxiliary_send_count == 0
    assert not world.adapter.auxiliary_invocations and world.adapter._aux_pending is None
    assert world.context._denied


@pytest.mark.parametrize("kind", ["after_preparation", "derive", "post_publication", "post_publication_policy", "invalid_boolean"])
def test_local_title_cancellation_releases_no_output_and_keeps_proven_records(
        excluded, caller_boundary, monkeypatch, kind):
    from agent import title_generator as titles
    block_title_edges(monkeypatch, excluded)
    original = ContentPolicyRuntime.derive
    world = auxiliary_world()
    first = turn(world, "   ")
    if kind == "derive":
        def cancel(runtime, handle, parents, payload, *, path):
            content = original(runtime, handle, parents, payload, path=path)
            if payload == b'{"title":null}':
                world.agent.interrupt()
            return content
        monkeypatch.setattr(ContentPolicyRuntime, "derive", cancel)
    else:
        def cancel():
            frame = world.adapter._aux_pending
            ready = frame is not None and (frame.local_result is not None if kind.startswith("post_publication")
                                           else frame.preparation is not None)
            if ready:
                if kind == "post_publication_policy":
                    world.title_binding = replace(world.title_binding, auth_epoch=8)
                elif kind == "invalid_boolean":
                    world.agent._interrupt_requested = 1
                else:
                    world.agent.interrupt()
        world.admission_hook = cancel
    with content_policy_scope(world.context):
        bind_title(world)
        token = world.adapter.auxiliary_input(world.agent, source=first, field="user_message")
    before = len(world.adapter._runtime._contents)
    error = NativeContentPolicyDenied if kind in ("invalid_boolean", "post_publication_policy") else InterruptedError
    with pytest.raises(error):
        with content_policy_scope(world.context):
            titles.generate_title("   ", controlled_aux=token)
    assert len(world.adapter.auxiliary_local_results) == int(kind.startswith("post_publication"))
    assert len(world.adapter._runtime._contents) >= before
    if kind == "derive" or kind.startswith("post_publication"):
        assert len(world.adapter._runtime._contents) > before
    assert world.adapter.send_count == 1 and world.adapter.auxiliary_send_count == 0
    assert not world.adapter.auxiliary_invocations and world.adapter._aux_pending is None
    assert world.context._denied
    if kind.startswith("post_publication"):
        published, = world.adapter.auxiliary_local_results
        world.agent.close()
        world.agent.close()
        assert world.adapter.auxiliary_local_results[0] is published


@pytest.mark.parametrize("kind", ["reentry", "process", "close", "thread"])
def test_local_title_lifecycle_refuses_before_excluded_callbacks(
        excluded, caller_boundary, monkeypatch, kind):
    import contextvars
    from agent import title_generator as titles, skill_commands
    block_title_edges(monkeypatch, excluded)
    world = auxiliary_world()
    first = turn(world, "   ")
    original = skill_commands.describe_skill_invocation
    stop = KeyboardInterrupt()
    def during_summary(text):
        if kind == "process":
            raise stop
        if kind == "reentry":
            with pytest.raises(NativeContentPolicyDenied):
                world.adapter.auxiliary_input(world.agent, source=first, field="user_message")
        return original(text)
    monkeypatch.setattr(skill_commands, "describe_skill_invocation", during_summary)
    with content_policy_scope(world.context):
        bind_title(world)
        token = world.adapter.auxiliary_input(world.agent, source=first, field="user_message")
    if kind == "thread":
        with content_policy_scope(world.context):
            copied = contextvars.copy_context()
        failures = []
        def attempt():
            try:
                copied.run(titles.generate_title, "   ", controlled_aux=token)
            except NativeContentPolicyDenied as error:
                failures.append(error)
        worker = threading.Thread(target=attempt)
        worker.start()
        worker.join(timeout=10)
        assert not worker.is_alive() and len(failures) == 1
    else:
        if kind == "close":
            world.agent.close()
            world.agent.close()
        with pytest.raises(KeyboardInterrupt if kind == "process" else NativeContentPolicyDenied) as caught:
            with content_policy_scope(world.context):
                titles.generate_title("   ", controlled_aux=token)
        if kind == "process":
            assert caught.value is stop
    assert world.context._denied and world.adapter._aux_pending is None
    assert not world.adapter.auxiliary_local_results and not world.adapter.auxiliary_invocations
    assert world.adapter.send_count == 1 and world.adapter.auxiliary_send_count == 0


def test_local_title_mixed_capacity_retains_original_order_and_shared_budget(
        excluded, caller_boundary, monkeypatch):
    from agent import title_generator as titles
    block_title_edges(monkeypatch, excluded, local=False)
    world = auxiliary_world()
    first = turn(world, "   ")
    with content_policy_scope(world.context):
        bind_title(world)
    accepted = []
    for index in range(9):
        field = "user_message" if index % 2 == 0 else "final_response"
        try:
            with content_policy_scope(world.context):
                token = world.adapter.auxiliary_input(world.agent, source=first, field=field)
                text = "   " if field == "user_message" else first["final_response"]
                titles.generate_title(text, controlled_aux=token)
        except NativeContentPolicyDenied:
            break
        frame = world.adapter._aux_frames[-1][0]
        record = frame.local_result if frame.local_result is not None else frame.record
        assert all(record.parents[-len(accepted)+i] is previous.output_content for i, previous in enumerate(accepted))
        accepted.append(record)
    else:
        pytest.fail("Shared auxiliary capacity did not refuse")
    assert 1 < len(accepted) <= len(world.adapter._aux_frames) <= 8
    completed = tuple(world.adapter._completed_auxiliary_outputs())
    assert all(content is record.output_content and capability is world.title_request
               for (content, capability), record in zip(completed, accepted))
    assert len(completed) == len(accepted)
    assert len(world.adapter._runtime._contents) + len(world.adapter._runtime._envelopes) <= 128
    assert world.adapter.send_count == 1
    assert world.adapter.auxiliary_send_count == len(world.title_ports["title"].sent)
    assert world.context._denied and world.adapter._aux_pending is None
    assert all(any(original is record for original in (*world.adapter.auxiliary_local_results,
                   *world.adapter.auxiliary_invocations)) for record in accepted)


def test_local_title_changed_resolved_destination_during_derivation_retains_no_invented_result(
        excluded, caller_boundary, monkeypatch):
    from agent import title_generator as titles
    block_title_edges(monkeypatch, excluded)
    world = auxiliary_world()
    first = turn(world, "   ")
    with content_policy_scope(world.context):
        bind_title(world)
        token = world.adapter.auxiliary_input(world.agent, source=first, field="user_message")
    runtime = world.adapter._runtime
    before = len(runtime._contents)
    port = world.title_ports["compression"]
    def changed():
        if len(runtime._contents) > before:
            port.destination = replace(port.destination, endpoint_ref="changed")
    port.resolve_hook = changed
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            titles.generate_title("   ", controlled_aux=token)
    assert len(runtime._contents) > before
    assert not world.adapter.auxiliary_local_results and not world.adapter.auxiliary_invocations
    assert world.adapter.send_count == 1 and world.adapter.auxiliary_send_count == 0
    assert world.context._denied and world.adapter._aux_pending is None


class CompressionWorld(AuxiliaryWorld):
    def __init__(self, *, same_model=False):
        from hermes_content_policy import PATHS, ProfilePolicy
        from agent.native_main_payload import NativeMainSelection, NativeResultPolicy
        from tests.agent.test_native_main_payload import ControlledPort
        super().__init__()
        self.compression_request = object()
        self.compression_binding = replace(self.binding, selection_id="compression_selection", selection_version=6)
        self.compression_policy = NativeResultPolicy("compression_producer", 3, self.policy.profiles)
        self.compression_ports = {}
        routes = []
        for row in self.routes[:len(PATHS)]:
            destination = replace(row.destination, transport_ref=f"compression_transport_{row.path}",
                                  model_ref="invented_model" if same_model else "invented_compression_model")
            port = self.compression_ports[row.path] = ControlledPort(destination)
            self.ports[f"compression_capability_{row.path}"] = port
            self.receiver_bindings[id(port)] = (port, replace(self.compression_binding,
                execution_ref=f"compression_receiver_{row.path}"))
            value = json.loads(reply("<think>scratch</think>## Goal\nKeep the blue folder."))
            value["model"] = destination.model_ref
            port.output = canonical(value)
            routes.append(replace(row, route_ref=f"compression_{row.path}", destination=destination,
                selection_id="compression_selection", selection_version=6))
        self.routes += tuple(routes)
        self.profiles = (ProfilePolicy("restricted_documents", tuple(r.route_ref for r in self.routes), PATHS),)
        self.compression_selection = NativeMainSelection("compression_selection", 6, "chat_completions",
                                                        self.compression_ports["compression"].destination)
        self.compression_selection_port = SimpleNamespace(current=self.compression_selected)
        self.compression_qualifications = ()

    def current(self, handle):
        if handle is self.compression_request:
            if self.admission_hook:
                self.admission_hook()
            return self.compression_binding
        return super().current(handle)

    def produced(self, producer):
        if producer == "compression_producer":
            if self.producer_hook:
                self.producer_hook()
            return self.compression_policy
        return super().produced(producer)

    def compression_selected(self, handle):
        assert handle is self.compression_request
        return self.compression_selection

    def qualify(self):
        from hermes_content_policy import PATHS
        super().qualify()
        runtime = ContentPolicyRuntime(admission=self.admission, profiles=self.profiles,
            routes=self.routes, transports=tuple((p.destination.transport_ref, p) for p in self.ports.values()),
            clock=lambda: self.now, policy_revision=9)
        records = []
        for path in PATHS:
            source = runtime.admit(self.compression_request, b"Invented independent compression qualification")
            result = runtime.dispatch(self.compression_request, source, path=path, route_ref=f"compression_{path}")
            records.append((path, source, result, self.compression_ports[path].sent[0]))
            self.compression_ports[path].sent = []
        self.compression_qualifications = tuple(records)


@pytest.fixture
def compression_excluded(excluded, monkeypatch):
    from agent import context_compressor as compression, auxiliary_client as auxiliary
    import hermes_time
    for module, names in (
        (compression, ("get_model_context_length", "resolve_model_threshold", "_pinned_summary_call_kwargs", "aux_interrupt_protection")),
        (compression.ContextCompressor, ("_compression_cancelled", "_record_aux_compression_call",
            "_fallback_to_main_for_compression", "_clear_compression_failure_cooldown",
            "_build_static_fallback_summary", "_record_compression_failure_cooldown",
            "_load_fallback_compression_streak", "_load_ineffective_compression_count",
            "_sync_micro_compact_to_db")),
        (auxiliary, ("_get_cached_client", "_acquire_sync_aux_semaphore", "_call_llm_impl",
            "_resolve_task_provider_model", "_relay_auxiliary_metadata")),
        (hermes_time, ("now", "get_timezone")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, excluded)
    monkeypatch.setattr(compression, "time", SimpleNamespace(monotonic=excluded))
    return excluded


def compression_world():
    world = construct(CompressionWorld().bind())
    world.ports["main"].output = reply("The blue folder stays.")
    return world


def bind_compression(world, **changes):
    options = dict(request_handle=world.compression_request, selection=world.compression_selection_port,
                   producer_ref="compression_producer", route_ref="compression_compression",
                   context_length=65536, summary_date="2026-11-23")
    options.update(changes)
    world.adapter.bind_compression(world.agent, **options)


def construct_compressor(world, **changes):
    from agent.context_compressor import ContextCompressor
    options = dict(model=world.agent.model, provider="synthetic_native", api_mode="chat_completions",
        summary_model_override=world.compression_selection.destination.model_ref, config_context_length=65536,
        quiet_mode=True, tail_mode="lean", native_main_adapter=world.adapter)
    options.update(changes)
    return ContextCompressor(**options)


def compression_refuses(world, callback, sends=1):
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            callback()
    assert world.context._denied and world.adapter._aux_pending is None
    assert world.adapter.send_count == sends
    assert world.adapter.auxiliary_send_count == len(world.compression_ports["compression"].sent)
    assert not world.sdk_calls


def summarize(world, compressor, result):
    token = world.adapter.compression_input(world.agent, source=result)
    return compressor._generate_summary(result["messages"], controlled_aux=token)


def test_compression_complete_retained_first_summary(compression_excluded, caller_boundary, monkeypatch):
    from agent import context_compressor as compression
    derived = []
    derive = ContentPolicyRuntime.derive_for_selection
    def observed(runtime, handle, parents, payload, *, path, parent_requests):
        result = derive(runtime, handle, parents, payload, path=path, parent_requests=parent_requests)
        derived.append((handle, parents, payload, path, parent_requests, result))
        return result
    monkeypatch.setattr(ContentPolicyRuntime, "derive_for_selection", observed)
    world = compression_world()
    first = turn(world, "Keep the blue folder.")
    prefix = world.agent._cached_system_prompt
    original = canonical(first)
    main_record, = world.adapter.invocations
    with content_policy_scope(world.context):
        bind_compression(world)
        compressor = construct_compressor(world)
        state = dict(vars(compressor))
        summary = summarize(world, compressor, first)
    record, = world.adapter.auxiliary_invocations
    frame = world.adapter._aux_frames[0][0]
    assert type(frame) is native_main_payload._CompressionFrame
    assert type(compressor) is compression.ContextCompressor
    assert record.source_completion.result is first and frame.messages is first["messages"]
    assert record.source_frame is world.adapter._turn_frames[0]
    assert record.source_invocation is main_record and record.source_handle is world.request
    assert record.request_handle is world.compression_request
    assert record.path == record.task == "compression"
    assert record.source_bytes == canonical(first["messages"])
    assert len(record.input_stages) == 7 and len(record.output_stages) == 4
    facts = [json.loads(stage) for stage in record.input_stages]
    assert facts[0]["constructor"] == world.adapter._compression_options
    assert facts[0]["previous_summary"] is facts[0]["focus_topic"] is None
    assert facts[0]["source_field"] == "messages" and facts[0]["memory_context"] == ""
    assert facts[1] == dict(content_tokens=compression.estimate_messages_tokens_rough(first["messages"]),
        ratio=0.2, context_length=65536, maximum=3276, summary_budget=2000)
    assert facts[2] == dict(serialized=compressor._serialize_for_summary(first["messages"]), pruned_skill_names=[])
    assert facts[3] == dict(sampled=facts[2]["serialized"], lean=True, sampling=False,
        original_length=len(facts[2]["serialized"]), slice_count=8, aggregate_cap=160000)
    assert facts[4] == dict(sanitized_memory="", serialized_memory='""', memory_section="",
                           has_user_turn=True, summary_date="2026-11-23")
    assert "current date is 2026-11-23" in facts[5]["prompt"]
    assert "Keep the blue folder." in facts[5]["prompt"]
    assert compression._LEAN_SESSION_LOG_HEADING in facts[5]["prompt"]
    wire = dict(model="invented_compression_model", messages=[dict(role="user", content=facts[5]["prompt"])], stream=False)
    request = json.loads(record.request_bytes)
    assert request["request"] == facts[6]["request"] == wire
    assert facts[6]["arguments"] == frame.preparation.value["arguments"]
    assert frame.request is frame.preparation.value["request"]
    assert frame.request["messages"] is frame.preparation.value["arguments"]["messages"]
    assert frame.request["messages"][0]["content"] == facts[5]["prompt"]
    assert frame.preparation.value["arguments"]["temperature"] is frame.preparation.value["arguments"]["max_tokens"] is None
    assert "temperature" not in wire and "max_tokens" not in wire
    assert record.response_bytes is world.compression_ports["compression"].output
    assert record.envelope is world.compression_ports["compression"].sent[0]
    assert record.result_content is not None and record.native_response is frame.record.native_response
    output = [json.loads(stage) for stage in record.output_stages]
    assert output[0] == dict(content="<think>scratch</think>## Goal\nKeep the blue folder.",
                            finish_reason="stop", nonempty=True, length_stop=False)
    assert output[1] == dict(stripped="## Goal\nKeep the blue folder.",
                            chosen="## Goal\nKeep the blue folder.", redacted="## Goal\nKeep the blue folder.")
    assert output[2]["restored"] == output[1]["redacted"]
    assert "User asked (deterministic" in output[2]["grounded"]
    assert compression._LEAN_USER_MESSAGES_HEADING in output[2]["augmented"]
    assert output[2]["augmented"] == compressor._augment_summary_lean(output[2]["grounded"], first["messages"])
    assert compression._build_recovery_footer("", len(first["messages"])) == ""
    assert output[3] == dict(prefixed=summary, output={"summary": summary})
    assert record.output["summary"] is summary and record.output_bytes == canonical(record.output)
    assert "scratch" not in summary and summary.startswith(compression.SUMMARY_PREFIX)
    bridge = next(row for row in derived if row[-1] is frame.source_content)
    assert bridge[0] is world.compression_request and bridge[1] is frame.parents
    assert all(a is b for a, b in zip(bridge[4], frame.parent_requests))
    assert all(handle is world.request for handle in frame.parent_requests)
    assert world.adapter.send_count == 1 and world.adapter.auxiliary_send_count == 1
    assert world.adapter.invocations == (main_record,) and world.adapter.auxiliary_local_results == ()
    assert first["total_tokens"] == 5 and canonical(first) == original
    assert vars(compressor) == state and compressor._previous_summary is None and compressor.compression_count == 0
    assert world.agent._cached_system_prompt is prefix and len(world.compression_qualifications) == 8


def test_compression_ordered_interleaving_and_return_to_main_title(compression_excluded, caller_boundary, monkeypatch):
    from agent.title_generator import generate_title
    derived = []
    derive = ContentPolicyRuntime.derive_for_selection
    def observed(runtime, handle, parents, payload, *, path, parent_requests):
        result = derive(runtime, handle, parents, payload, path=path, parent_requests=parent_requests)
        derived.append((handle, parents, payload, path, parent_requests, result))
        return result
    monkeypatch.setattr(ContentPolicyRuntime, "derive_for_selection", observed)
    world = compression_world()
    world.ports["main"].output = reply("")
    first = turn(world, "Keep the blue folder.")
    first_bytes = canonical(first)
    prefix = world.agent._cached_system_prompt
    with content_policy_scope(world.context):
        bind_title(world)
        title_binding = world.adapter._aux_binding
        title_anchor = world.adapter._aux_binding_anchor
        bind_compression(world)
        compressor = construct_compressor(world)
        state = dict(vars(compressor))
        local_token = world.adapter.auxiliary_input(world.agent, source=first, field="final_response")
        assert generate_title(first["final_response"], controlled_aux=local_token) is None
        local, = world.adapter.auxiliary_local_results
        first_summary = summarize(world, compressor, first)
    compressed_a, = world.adapter.auxiliary_invocations
    world.binding = replace(world.binding, profiles=())
    world.title_binding = replace(world.title_binding, profiles=())
    world.compression_binding = replace(world.compression_binding, profiles=())
    world.ports["main"].output = reply("The blue folder stays.")
    second = turn(world, "Keep it blue.", first["messages"])
    with content_policy_scope(world.context):
        title = title_from(world, second)
        summary = summarize(world, compressor, second)
    compressed_a_again, sent_title, compressed_b = world.adapter.auxiliary_invocations
    assert compressed_a_again is compressed_a
    assert compressed_a.output["summary"] is first_summary and compressed_b.output["summary"] is summary
    assert sent_title.output["title"] is title
    outputs = (local, compressed_a, sent_title, compressed_b)
    completed = tuple(world.adapter._completed_auxiliary_outputs())
    assert len(completed) == 4
    for (content, capability), record in zip(completed, outputs):
        assert content is record.output_content and capability is record.request_handle
    assert [frame.__class__ for frame, _ in world.adapter._aux_frames] == [
        native_main_payload._AuxiliaryFrame, native_main_payload._CompressionFrame,
        native_main_payload._AuxiliaryFrame, native_main_payload._CompressionFrame]
    for index, (frame, _) in enumerate(world.adapter._aux_frames):
        previous = outputs[:index]
        if previous:
            assert all(a is b.output_content for a, b in zip(frame.parents[-len(previous):], previous))
            assert all(a is b.request_handle for a, b in zip(frame.parent_requests[-len(previous):], previous))
    second_proofs = [row for row in derived if row[0] is world.request and row[3] == "main"]
    assert second_proofs
    for row in second_proofs:
        assert row[1][-2] is local.output_content and row[1][-1] is compressed_a.output_content
        assert row[4][-2] is world.title_request and row[4][-1] is world.compression_request
    title_source = world.adapter._aux_frames[2][0].source_content
    title_proof = next(row for row in derived if row[-1] is title_source)
    assert title_proof[4][-2] is world.title_request and title_proof[4][-1] is world.compression_request
    assert world.adapter._aux_binding is title_binding and world.adapter._aux_binding_anchor is title_anchor
    assert world.adapter.invocations[0] is local.source_invocation
    assert world.adapter.invocations[1] is compressed_b.source_invocation
    assert canonical(first) == first_bytes and world.agent._cached_system_prompt is prefix
    assert first["total_tokens"] == 5 and second["total_tokens"] == 10
    assert world.adapter.send_count == 2 and world.adapter.auxiliary_send_count == 3
    assert vars(compressor) == state and compressor.compression_count == 0
    assert all(envelope.profiles == ("restricted_documents",) for envelope in (
        *world.ports["main"].sent, *world.title_ports["title"].sent,
        *world.compression_ports["compression"].sent))
    retained = world.adapter.auxiliary_invocations
    world.agent.close()
    world.agent.close()
    assert world.adapter.auxiliary_invocations is retained


def test_compression_interleave_without_title_binding(compression_excluded, caller_boundary):
    world = compression_world()
    first = turn(world, "Keep the blue folder.")
    with content_policy_scope(world.context):
        bind_compression(world)
        compressor = construct_compressor(world)
        summarize(world, compressor, first)
    old = world.adapter.auxiliary_invocations[0]
    second = turn(world, "Keep the folder.", first["messages"])
    with content_policy_scope(world.context):
        summarize(world, compressor, second)
    frame = world.adapter._aux_frames[1][0]
    assert frame.parents[-1] is old.output_content and frame.parent_requests[-1] is world.compression_request
    assert world.adapter._aux_binding is None and world.adapter.send_count == 2
    assert world.adapter.auxiliary_send_count == 2 and second["total_tokens"] == 10


@pytest.mark.parametrize("change", [
    {"context_length": True}, {"context_length": 0}, {"context_length": Hostile()},
    {"summary_date": "2026-02-30"}, {"summary_date": "20261123"}, {"summary_date": Hostile()},
    {"producer_ref": "title_producer"}, {"route_ref": "title_compression"},
])
def test_compression_binding_exact_facts_refuse(compression_excluded, caller_boundary, change):
    world = compression_world()
    with content_policy_scope(world.context):
        bind_title(world)
    compression_refuses(world, lambda: bind_compression(world, **change), 0)
    assert world.adapter.auxiliary_send_count == 0


@pytest.mark.parametrize("change", [
    {"provider": Hostile()}, {"model": Hostile()}, {"api_mode": Hostile()},
    {"config_context_length": True}, {"config_context_length": 4096},
    {"quiet_mode": 1}, {"quiet_mode": False}, {"tail_mode": "legacy"},
    {"threshold_percent": 1}, {"summary_target_ratio": 0.3}, {"model_thresholds": {}},
    {"base_url": "invented"}, {"api_key": "invented"}, {"max_tokens": 100},
])
def test_compression_constructor_exact_options_refuse(compression_excluded, caller_boundary, change):
    world = compression_world()
    with content_policy_scope(world.context):
        bind_compression(world)
    compression_refuses(world, lambda: construct_compressor(world, **change), 0)
    assert world.adapter._compression_owner is None and world.adapter.auxiliary_send_count == 0


@pytest.mark.parametrize("change", ["copy_completion", "copy_messages", "equal_row", "foreign", "digest", "wrong_type"])
def test_compression_source_requires_original_completion(compression_excluded, caller_boundary, monkeypatch, change):
    from agent.context_compressor import ContextCompressor
    world = compression_world()
    result = turn(world, "Keep the blue folder.")
    monkeypatch.setattr(ContextCompressor, "_serialize_for_summary", compression_excluded)
    with content_policy_scope(world.context):
        bind_compression(world)
        compressor = construct_compressor(world)
    if change == "copy_completion":
        result = dict(result)
    elif change == "copy_messages":
        result["messages"] = list(result["messages"])
    elif change == "equal_row":
        result["messages"][0] = dict(result["messages"][0])
    elif change == "foreign":
        foreign = compression_world()
        result = turn(foreign, "Keep the blue folder.")
    elif change == "digest":
        result = {"digest": canonical(result).hex()}
    else:
        result = Hostile()
    compression_refuses(world, lambda: world.adapter.compression_input(world.agent, source=result), 1)
    assert world.adapter.auxiliary_send_count == 0


@pytest.mark.parametrize("change", ["main_epoch", "compression_version", "producer_revision", "resolver", "expiry", "remove_binding"])
def test_compression_current_independent_facts_refuse(compression_excluded, caller_boundary, change):
    world = compression_world()
    result = turn(world, "Keep the blue folder.")
    with content_policy_scope(world.context):
        bind_compression(world)
        compressor = construct_compressor(world)
    if change == "main_epoch":
        world.binding = replace(world.binding, auth_epoch=8)
    elif change == "compression_version":
        world.compression_selection = replace(world.compression_selection, selection_version=7)
    elif change == "producer_revision":
        world.compression_policy = replace(world.compression_policy, revision=4)
    elif change == "resolver":
        world.compression_ports["memory"].destination = replace(world.compression_ports["memory"].destination, revision=2)
    elif change == "expiry":
        world.now = 100
    else:
        world.adapter._compression_binding = None
    compression_refuses(world, lambda: summarize(world, compressor, result), 1)
    assert world.adapter.auxiliary_send_count == 0


@pytest.mark.parametrize("kind", ["missing", "copied", "title", "wrong_turns", "focus", "memory", "owner_mutated", "owner_copied", "callback_added", "rebound"])
def test_compression_owned_tokens_and_constructor_refuse(compression_excluded, caller_boundary, kind):
    world = compression_world()
    result = turn(world, "Keep the blue folder.")
    with content_policy_scope(world.context):
        if kind == "title":
            bind_title(world)
        bind_compression(world)
        compressor = construct_compressor(world)
        if kind == "title":
            token = world.adapter.auxiliary_input(world.agent, source=result, field="user_message")
        else:
            token = world.adapter.compression_input(world.agent, source=result)
    if kind == "rebound":
        compression_refuses(world, lambda: construct_compressor(world), 1)
        return
    if kind == "copied":
        token = object.__new__(native_main_payload.NativeAuxiliaryInput)
    elif kind == "missing":
        token = None
    elif kind == "owner_mutated":
        compressor._previous_summary = "Invented stale summary"
    elif kind == "owner_copied":
        compressor = copy.copy(compressor)
    elif kind == "callback_added":
        compressor._compression_cancelled_check = compression_excluded
    turns = list(result["messages"]) if kind == "wrong_turns" else result["messages"]
    options = {"focus_topic": "blue"} if kind == "focus" else {"memory_context": "invented"} if kind == "memory" else {}
    compression_refuses(world, lambda: compressor._generate_summary(turns, controlled_aux=token, **options), 1)
    assert world.adapter.auxiliary_send_count == 0


@pytest.mark.parametrize("when", ["before_source", "after_preparation", "during_send", "after_raw", "before_publication", "post_publication", "invalid_boolean"])
def test_compression_cancellation_retains_honest_proof(compression_excluded, caller_boundary, monkeypatch, when):
    world = compression_world()
    result = turn(world, "Keep the blue folder.")
    published = []
    if when == "during_send":
        world.compression_ports["compression"].hook = lambda envelope: world.agent.interrupt()
    def cancel():
        frame = world.adapter._aux_pending
        if type(frame) is not native_main_payload._CompressionFrame:
            return
        ready = ((frame.preparation is not None) if when in ("after_preparation", "invalid_boolean") else
                 (frame.record is not None and frame.record.native_response is not None) if when == "after_raw" else
                 (frame.record is not None and frame.record.output is not None) if when == "post_publication" else False)
        if ready:
            if when == "post_publication":
                published.append(frame.record)
            world.agent._interrupt_requested = 1 if when == "invalid_boolean" else True
    world.admission_hook = cancel
    if when == "before_publication":
        original = ContentPolicyRuntime.derive
        def cancel_final(runtime, handle, parents, payload, *, path):
            content = original(runtime, handle, parents, payload, path=path)
            if path == "compression" and set(json.loads(payload)) == {"summary"}:
                world.agent._interrupt_requested = True
            return content
        monkeypatch.setattr(ContentPolicyRuntime, "derive", cancel_final)
    with content_policy_scope(world.context):
        bind_compression(world)
        compressor = construct_compressor(world)
        state = dict(vars(compressor))
        token = None if when == "before_source" else world.adapter.compression_input(world.agent, source=result)
    error = NativeContentPolicyDenied if when == "invalid_boolean" else InterruptedError
    with pytest.raises(error) as caught:
        with content_policy_scope(world.context):
            if when == "before_source":
                world.agent._interrupt_requested = True
                world.adapter.compression_input(world.agent, source=result)
            else:
                compressor._generate_summary(result["messages"], controlled_aux=token)
    if error is InterruptedError:
        assert type(caught.value) is InterruptedError and str(caught.value) == "Agent interrupted during API call"
    count = int(when in ("during_send", "after_raw", "before_publication", "post_publication"))
    assert world.adapter.auxiliary_send_count == count
    assert len(world.compression_ports["compression"].sent) == count
    assert len(world.adapter.auxiliary_invocations) == count
    assert world.adapter.send_count == 1 and world.context._denied and world.adapter._aux_pending is None
    assert vars(compressor) == state
    if count:
        record, = world.adapter.auxiliary_invocations
        assert record.envelope is world.compression_ports["compression"].sent[0]
        assert record.response_bytes is world.compression_ports["compression"].output
        assert (record.output is not None) is (when == "post_publication")
        if when == "post_publication":
            assert published and record is published[0] and record.output_content is not None
        world.agent.close()
        world.agent.close()
        assert world.adapter.auxiliary_invocations[0] is record


@pytest.mark.parametrize("kind", ["reentry", "wrong_thread", "close", "pending", "replay"])
def test_compression_lifecycle_refuses(compression_excluded, caller_boundary, kind):
    import contextvars
    world = compression_world()
    result = turn(world, "Keep the blue folder.")
    with content_policy_scope(world.context):
        bind_compression(world)
        compressor = construct_compressor(world)
        token = world.adapter.compression_input(world.agent, source=result)
    if kind == "pending":
        compression_refuses(world, lambda: turn(world, "Invented overlap", result["messages"]), 1)
    elif kind == "close":
        def closed():
            world.agent.close()
            world.agent.close()
            compressor._generate_summary(result["messages"], controlled_aux=token)
        compression_refuses(world, closed)
    elif kind == "wrong_thread":
        errors = []
        with pytest.raises(NativeContentPolicyDenied):
            with content_policy_scope(world.context):
                context = contextvars.copy_context()
                def wrong_thread():
                    try:
                        compressor._generate_summary(result["messages"], controlled_aux=token)
                    except BaseException as error:
                        errors.append(error)
                thread = threading.Thread(target=lambda: context.run(wrong_thread))
                thread.start()
                thread.join(timeout=5)
                assert not thread.is_alive() and len(errors) == 1
                assert type(errors[0]) is NativeContentPolicyDenied
    elif kind == "reentry":
        def reenter(envelope):
            with pytest.raises(NativeContentPolicyDenied):
                compressor._generate_summary(result["messages"], controlled_aux=token)
        world.compression_ports["compression"].hook = reenter
        compression_refuses(world, lambda: compressor._generate_summary(result["messages"], controlled_aux=token))
    else:
        with content_policy_scope(world.context):
            summary = compressor._generate_summary(result["messages"], controlled_aux=token)
        record, = world.adapter.auxiliary_invocations
        assert record.output["summary"] is summary
        compression_refuses(world, lambda: compressor._generate_summary(result["messages"], controlled_aux=token))
        assert world.adapter.auxiliary_invocations[0] is record
    expected = int(kind in ("reentry", "replay"))
    assert world.adapter.auxiliary_send_count == expected and world.adapter.send_count == 1
    assert world.context._denied and world.adapter._aux_pending is None


@pytest.mark.parametrize("kind", ["process", "explicit_cancel", "system_exit"])
def test_compression_base_exception_identity_propagates(compression_excluded, caller_boundary, kind):
    from agent.auxiliary_client import AuxiliaryExplicitCancellation
    failure = (KeyboardInterrupt() if kind == "process" else SystemExit(3) if kind == "system_exit"
               else AuxiliaryExplicitCancellation())
    assert issubclass(AuxiliaryExplicitCancellation, BaseException)
    assert not issubclass(AuxiliaryExplicitCancellation, Exception)
    world = compression_world()
    result = turn(world, "Keep the blue folder.")
    with content_policy_scope(world.context):
        bind_compression(world)
        compressor = construct_compressor(world)
        token = world.adapter.compression_input(world.agent, source=result)
    def stop():
        raise failure
    with pytest.raises(type(failure)) as caught:
        with content_policy_scope(world.context):
            world.admission_hook = stop
            compressor._generate_summary(result["messages"], controlled_aux=token)
    assert caught.value is failure and world.context._denied
    assert world.adapter.auxiliary_send_count == 0 and world.adapter.send_count == 1


@pytest.mark.parametrize("kind", ["empty", "truncated", "raw_bound"])
def test_compression_invalid_output_denies_without_fallback(compression_excluded, caller_boundary, kind):
    world = compression_world()
    result = turn(world, "Keep the blue folder.")
    value = json.loads(world.compression_ports["compression"].output)
    value["choices"][0]["message"]["content"] = " " if kind == "empty" else "A" * 65536 if kind == "raw_bound" else "Partial summary"
    if kind == "truncated":
        value["choices"][0]["finish_reason"] = "length"
    world.compression_ports["compression"].output = canonical(value)
    with content_policy_scope(world.context):
        bind_compression(world)
        compressor = construct_compressor(world)
        state = dict(vars(compressor))
    compression_refuses(world, lambda: summarize(world, compressor, result))
    record, = world.adapter.auxiliary_invocations
    assert record.output is record.output_content is None
    assert world.adapter.auxiliary_send_count == 1 and world.adapter.send_count == 1
    if kind == "raw_bound":
        assert record.response_bytes is None
    else:
        assert record.response_bytes is world.compression_ports["compression"].output
        assert record.native_response is not None and record.result_content is not None
    assert vars(compressor) == state


@pytest.mark.parametrize("kind", ["request", "stage", "helper", "output", "response"])
def test_compression_mutated_proof_refuses(compression_excluded, caller_boundary, monkeypatch, kind):
    from agent import context_compressor as compression, auxiliary_client
    world = compression_world()
    result = turn(world, "Keep the blue folder.")
    with content_policy_scope(world.context):
        bind_compression(world)
        compressor = construct_compressor(world)
        token = world.adapter.compression_input(world.agent, source=result)
        if kind != "helper":
            proof = world.adapter._prepare_compression(token)
            if kind != "stage":
                sealed = world.adapter.seal_auxiliary(world.agent, token, stages=proof.stages, request=proof.value["request"])
                if kind in ("response", "output"):
                    response = auxiliary_client.call_llm(**proof.value["arguments"], controlled_aux=sealed)
                    stages, output = world.adapter._compression_output_stages(sealed, response)
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            if kind == "helper":
                monkeypatch.setattr(compression.ContextCompressor, "_summary_prompt", compression_excluded)
                compressor._generate_summary(result["messages"], controlled_aux=token)
            elif kind == "stage":
                object.__setattr__(proof, "stages", tuple(bytes(bytearray(stage)) for stage in proof.stages))
                world.adapter.seal_auxiliary(world.agent, token, stages=proof.stages, request=proof.value["request"])
            elif kind == "request":
                proof.value["request"]["messages"][0] = dict(proof.value["request"]["messages"][0])
                auxiliary_client.call_llm(**proof.value["arguments"], controlled_aux=sealed)
            else:
                if kind == "response":
                    response = replace(response, choices=tuple(response.choices))
                else:
                    output["summary"] = output["summary"].encode().decode()
                world.adapter.complete_auxiliary(sealed, response=response, stages=stages, output=output)
    assert world.adapter.auxiliary_send_count == int(kind in ("output", "response"))
    assert world.adapter.send_count == 1 and world.context._denied


def test_compression_shared_mixed_capacity_and_prompt_bounds(compression_excluded, caller_boundary):
    from agent.title_generator import generate_title
    world = compression_world()
    world.ports["main"].output = reply("")
    result = turn(world, "Keep the blue folder.")
    with content_policy_scope(world.context):
        bind_title(world)
        bind_compression(world)
        compressor = construct_compressor(world)
        summarize(world, compressor, result)
        for _ in range(7):
            token = world.adapter.auxiliary_input(world.agent, source=result, field="final_response")
            assert generate_title(result["final_response"], controlled_aux=token) is None
        assert len(world.adapter._aux_frames) == 8 and len(world.adapter.auxiliary_local_results) == 7
        retained = len(world.adapter._runtime._contents)
    compression_refuses(world, lambda: world.adapter.compression_input(world.agent, source=result))
    assert len(world.adapter._runtime._contents) == retained
    assert world.adapter.auxiliary_send_count == 1 and world.adapter.send_count == 1


def test_compression_complete_source_cannot_bypass_request_text_bound(compression_excluded, caller_boundary):
    world = compression_world()
    world.ports["main"].output = reply("B" * 16000)
    result = turn(world, "B" * 16000)
    with content_policy_scope(world.context):
        bind_compression(world)
        compressor = construct_compressor(world)
        token = world.adapter.compression_input(world.agent, source=result)
        frame = world.adapter._aux_pending
        assert frame.source_bytes == canonical(result["messages"])
        assert len(frame.source_bytes) > 16384
    compression_refuses(world, lambda: compressor._generate_summary(result["messages"], controlled_aux=token))
    assert world.adapter.auxiliary_send_count == 0 and world.adapter.send_count == 1


@pytest.mark.parametrize("operation", ["compress", "micro", "defrag", "prune", "session", "micro_summary", "unowned_summary"])
def test_compression_proposal_siblings_refuse_after_profile_removal(compression_excluded, caller_boundary, operation):
    world = compression_world()
    result = turn(world, "Keep the blue folder.")
    with content_policy_scope(world.context):
        bind_compression(world)
        compressor = construct_compressor(world)
    world.binding = replace(world.binding, profiles=())
    world.compression_binding = replace(world.compression_binding, profiles=())
    state = dict(vars(compressor))
    storage = SimpleNamespace(get_session=compression_excluded, execute=compression_excluded)
    callbacks = {
        "compress": lambda: compressor.compress(result["messages"]),
        "micro": lambda: compressor._micro_compact(result["messages"]),
        "defrag": lambda: compressor._defrag_rolling_summary(result["messages"]),
        "prune": lambda: compressor.prune_tool_results_only(result["messages"]),
        "session": lambda: compressor.bind_session_state(storage, "invented_session"),
        "micro_summary": lambda: compressor._micro_summarize_one("Invented exchange"),
        "unowned_summary": lambda: compressor._generate_summary(result["messages"]),
    }
    compression_refuses(world, callbacks[operation])
    assert vars(compressor) == state and world.adapter.auxiliary_send_count == 0


@pytest.mark.parametrize("consumer", ["main", "title"])
def test_compression_binding_is_checked_during_another_consumer(compression_excluded, caller_boundary, consumer):
    world = compression_world()
    result = turn(world, "Keep the blue folder.")
    with content_policy_scope(world.context):
        bind_title(world)
        bind_compression(world)
        construct_compressor(world)
    world.compression_policy = replace(world.compression_policy, revision=4)
    callback = (lambda: turn(world, "Invented later turn", result["messages"])) if consumer == "main" else (
        lambda: title_from(world, result))
    compression_refuses(world, callback)
    assert world.adapter.auxiliary_send_count == 0


@pytest.mark.parametrize("consumer", ["title", "local"])
def test_compression_frame_stays_independent_with_equal_main_model(compression_excluded, caller_boundary, consumer):
    from agent.title_generator import generate_title
    world = construct(CompressionWorld(same_model=True).bind())
    result = turn(world, "Keep the blue folder.")
    assert world.selection.destination.model_ref == world.compression_selection.destination.model_ref
    with content_policy_scope(world.context):
        bind_title(world)
        bind_compression(world)
        construct_compressor(world)
        token = world.adapter.compression_input(world.agent, source=result)
    world.binding = replace(world.binding, profiles=())
    world.title_binding = replace(world.title_binding, profiles=())
    world.compression_binding = replace(world.compression_binding, profiles=())
    callback = (lambda: generate_title(result["final_response"], controlled_aux=token)) if consumer == "title" else (
        lambda: world.adapter.complete_auxiliary_local(world.agent, token, stages=(b"{}",), output={"title": None}))
    compression_refuses(world, callback)
    assert world.adapter.auxiliary_send_count == 0


def test_compression_original_catalog_entry_is_required(compression_excluded, caller_boundary):
    from hermes_content_policy import PATHS, ProfilePolicy
    world = CompressionWorld()
    world.routes = tuple(row for row in world.routes if row.route_ref != "compression_compression")
    world.profiles = (ProfilePolicy("restricted_documents", tuple(row.route_ref for row in world.routes), PATHS),)
    construct(world.bind(qualify=False))
    compression_refuses(world, lambda: bind_compression(world), 0)
    assert world.adapter._compression_owner is None and world.adapter.auxiliary_send_count == 0


def test_compression_reductions_keep_original_source_and_skill_markers(compression_excluded, caller_boundary):
    from agent import context_compressor as compression
    marker = compression._skill_pruned_marker("invented_skill")
    user = "Keep the blue folder.\n" + "A" * 7000 + marker + "Z" * 3000
    world = compression_world()
    result = turn(world, user)
    with content_policy_scope(world.context):
        bind_compression(world)
        compressor = construct_compressor(world)
        summary = summarize(world, compressor, result)
    record, = world.adapter.auxiliary_invocations
    serialized = json.loads(record.input_stages[2])
    assert serialized["pruned_skill_names"] == ["invented_skill"]
    assert marker not in serialized["serialized"] and "truncated" in serialized["serialized"]
    assert canonical(user) in record.source_bytes
    assert marker in summary and record.envelope.profiles == ("restricted_documents",)
    assert record.output["summary"] is summary and world.adapter.auxiliary_send_count == 1
