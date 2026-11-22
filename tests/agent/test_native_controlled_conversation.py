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
