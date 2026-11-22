"""Complete invented input through the actual controlled prologue and caller."""

import copy
import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agent.chat_completion_helpers import interruptible_api_call
from agent.content_policy import NativeContentPolicyDenied, content_policy_scope
from agent.native_main_payload import NativePreparedTurn, NativeTurnInput
from agent.turn_context import build_turn_context
from tests.agent.test_native_main_payload import (
    World, reply,
    invented_offline_configuration as invented_offline_configuration,
    caller_boundary as caller_boundary,
)


def prepared_world():
    world = World().bind()
    agent = world.agent
    agent.model = "invented_model"
    agent.enabled_toolsets = []
    agent.tools = []
    agent.valid_tool_names = set()
    agent._memory_store = agent._memory_manager = agent._session_db = None
    agent.compression_enabled = False
    agent._persist_disabled = agent.skip_background_review = True
    agent._cached_system_prompt = None
    agent._fallback_chain = []
    agent.request_overrides = {}
    agent.prefill_messages = []
    return world


def arguments(user="Invented document A", system=None, history=None):
    return dict(user_message=user, system_message=system, conversation_history=history, task_id="execution")


def admit(world, args=None):
    return world.adapter.admit_turn(world.agent, **(args if args is not None else arguments()))


def prepare(world, token, args=None, **changes):
    args = dict(args if args is not None else arguments())
    args.update(changes)
    forbidden = world.forbidden_provider
    return build_turn_context(world.agent, **args, stream_callback=None, persist_user_message=None,
        restore_or_build_system_prompt=forbidden, install_safe_stdio=forbidden,
        sanitize_surrogates=forbidden, summarize_user_message_for_log=forbidden,
        set_session_context=forbidden, set_current_write_origin=forbidden,
        ra=SimpleNamespace(), controlled_input=token)


def request(world, context):
    return world.adapter.prepared_request(world.agent, context.controlled_prepared)


def history(world):
    record = world.adapter.invocations[-1]
    rows = json.loads(record.request_bytes)["request"]["messages"]
    if rows[0]["role"] == "system":
        rows.pop(0)
    return rows + [{"role": "assistant", "content": record.native_response.choices[0].message.content}]


def test_turn_preparation_first_complete_bytes_and_actual_caller(caller_boundary):
    world = caller_boundary.configure(prepared_world())
    args = arguments("Invented\ud800 A", "Invented system")
    with content_policy_scope(world.context):
        token = admit(world, args)
        frame = world.adapter._turn_frames[0]
        assert frame.raw_bytes == b'{"conversation_history":null,"options":{"max_tokens":{"present":false,"value":null},"model":{"present":true,"value":"invented_model"},"stream":{"present":false,"value":false},"temperature":{"present":false,"value":null}},"system_message":"Invented system","task_id":"execution","user_message":"Invented\\ud800 A"}'
        assert world.adapter._runtime._contents[id(frame.admitted)].payload == frame.raw_bytes
        context = prepare(world, token, args)
        assert context.user_message == context.original_user_message == "Invented\ufffd A"
        assert type(context.controlled_prepared) is NativePreparedTurn
        assert context.turn_id == "native-turn-1"
        kwargs = request(world, context)
        assert kwargs == {"model": "invented_model", "stream": False, "messages": [
            {"role": "system", "content": "Invented system"}, {"role": "user", "content": "Invented\ufffd A"}]}
        result = interruptible_api_call(world.agent, kwargs)
    record, = world.adapter.invocations
    assert record.native_response is result
    assert record.response_bytes == reply()
    assert record.request_bytes == b'{"api_mode":"chat_completions","destination":{"endpoint_ref":"invented_endpoint","model_ref":"invented_model","provider_ref":"synthetic_native","revision":1,"transport_ref":"invented_main"},"request":{"messages":[{"content":"Invented system","role":"system"},{"content":"Invented\\ufffd A","role":"user"}],"model":"invented_model","stream":false}}'
    assert record.envelope is world.ports["main"].sent[0]
    assert world.adapter.send_count == 1 and world.sdk_calls == []


def test_turn_preparation_repeated_history_and_exact_parents_after_profile_drop(monkeypatch, caller_boundary):
    world = caller_boundary.configure(prepared_world())
    derivations = []
    derive = world.adapter._runtime.derive

    def observed(handle, parents, payload, *, path):
        result = derive(handle, parents, payload, path=path)
        derivations.append((parents, payload, result))
        return result

    monkeypatch.setattr(world.adapter._runtime, "derive", observed)
    originals = []
    for index in range(3):
        if index:
            world.binding = replace(world.binding, profiles=())
        args = arguments(f"Invented document {index}", "Fixed system", None if not index else history(world))
        with content_policy_scope(world.context):
            context = prepare(world, admit(world, args), args)
            result = interruptible_api_call(world.agent, request(world, context))
        originals.append(result)
        frame = world.adapter._turn_frames[index]
        earlier = tuple(r.result_content for r in world.adapter.invocations[:-1])
        local = [row for row in derivations if row[2] is frame.input_content]
        if index:
            assert len(local) == 1
            assert all(a is b for a, b in zip(local[0][0], (frame.admitted, *earlier)))
        sealed = next(row for row in derivations if row[2] is frame.output_content)
        assert all(a is b for a, b in zip(sealed[0], (frame.admitted, frame.input_content, *earlier)))
        wire = next(row for row in derivations if row[2] is world.adapter.invocations[-1].request_content)
        expected_parents = (frame.admitted, frame.input_content, frame.output_content, *earlier)
        assert len(wire[0]) == len(expected_parents)
        assert all(a is b for a, b in zip(wire[0], expected_parents))
        assert world.adapter._runtime._contents[id(frame.output_content)].payload == sealed[1]
        assert json.loads(sealed[1]) == {"turn_context": json.loads(frame.output_bytes), "request": json.loads(frame.request_bytes)}
        assert world.ports["main"].sent[-1].profiles == ("restricted_documents",)
        assert context.conversation_history == args["conversation_history"]
        assert json.loads(world.adapter.invocations[-1].request_bytes)["request"]["messages"][0] == {
            "role": "system", "content": "Fixed system"}
    assert all(r.native_response is original for r, original in zip(world.adapter.invocations, originals))
    assert world.adapter.send_count == 3 and world.sdk_calls == []


@pytest.mark.parametrize("option,value", [("max_tokens", 4096), ("max_tokens", None),
                                          ("temperature", 0.25), ("temperature", None), ("stream", False)])
def test_turn_preparation_exact_supported_options(option, value):
    world = prepared_world()
    setattr(world.agent, option, value)
    context = prepare(world, admit(world))
    options = json.loads(world.adapter._turn_frames[0].options)
    assert options[option] == {"present": True, "value": value}
    kwargs = request(world, context)
    if value is None:
        assert option not in kwargs
    else:
        assert kwargs[option] == value and type(kwargs[option]) is type(value)
    interruptible_api_call(world.agent, kwargs)
    assert world.adapter.send_count == 1


@pytest.mark.parametrize("field,value", [
    ("enabled_toolsets", ["memory"]), ("tools", [object()]), ("valid_tool_names", {"terminal"}),
    ("_memory_store", object()), ("_memory_manager", object()), ("_session_db", object()),
    ("compression_enabled", True), ("skip_background_review", False), ("_persist_disabled", False),
    ("_fallback_chain", [{"model": "other"}]), ("_fallback_activated", True),
    ("_fallback_index", 1), ("request_overrides", {"temperature": 0.5}),
    ("prefill_messages", [{"role": "user", "content": "Injected"}]),
    ("_pending_cli_user_message", {}), ("_pending_redirect", "Injected"),
    ("_native_command_context", object()), ("_media_context", {}),
    ("_stream_callback", lambda: None), ("ephemeral_system_prompt", "Injected"),
    ("_cached_system_prompt", "Unadmitted"), ("_persist_user_message_override", "Injected"),
    ("_persist_user_display_metadata", {}), ("_external_history", []),
    ("max_tokens", True), ("max_tokens", 4097), ("temperature", 1),
    ("temperature", float("nan")), ("temperature", 2.1), ("stream", True),
    ("reasoning_config", {}), ("extra_headers", {}), ("skip_memory", False),
])
def test_turn_preparation_unsupported_state_refuses_before_processing(field, value):
    world = prepared_world()
    setattr(world.agent, field, value)
    with pytest.raises(NativeContentPolicyDenied):
        admit(world)
    assert world.context._denied and world.adapter.send_count == 0 and world.sdk_calls == []
    assert not world.adapter._turn_frames


@pytest.mark.parametrize("change", ["raw_type", "system_surrogate", "history", "task", "metadata", "huge", "raw_record"])
def test_turn_preparation_input_shape_and_record_limits(change):
    world = prepared_world()
    args = arguments()
    if change == "raw_type":
        args["user_message"] = SimpleNamespace(__str__=world.forbidden_provider)
    elif change == "system_surrogate":
        args["system_message"] = "bad\ud800"
    elif change == "history":
        args["conversation_history"] = [{"role": "user", "content": "Imported"}]
    elif change == "task":
        args["task_id"] = "other_execution"
    elif change == "metadata":
        args["conversation_history"] = [{"role": "user", "content": "Text", "timestamp": 1}]
    elif change == "huge":
        args["user_message"] = "x" * 16_385
    else:
        args["user_message"] = "\ud800" * 16_384
        args["system_message"] = "x" * 16_384
        args["conversation_history"] = [{"role": "user", "content": "y" * 16_384} for _ in range(31)]
    with pytest.raises(NativeContentPolicyDenied):
        admit(world, args)
    assert world.context._denied and world.adapter.send_count == 0 and world.sdk_calls == []


@pytest.mark.parametrize("phase", ["admission", "prologue", "materialization", "dispatch"])
@pytest.mark.parametrize("change", ["profile", "selection", "producer", "state", "option_presence"])
def test_turn_preparation_current_facts_rechecked(phase, change):
    world = prepared_world()

    def mutate():
        if change == "profile":
            world.binding = replace(world.binding, profiles=())
        elif change == "selection":
            world.selection = replace(world.selection, selection_version=4)
        elif change == "producer":
            world.policy = replace(world.policy, revision=5)
        elif change == "state":
            world.agent._memory_manager = object()
        else:
            world.agent.max_tokens = None

    if phase == "admission":
        original = world.adapter._runtime.admit

        def changing(*args):
            result = original(*args)
            mutate()
            return result

        world.adapter._runtime.admit = changing
        operation = lambda: admit(world)
    else:
        token = admit(world)
        if phase == "prologue":
            mutate()
            operation = lambda: prepare(world, token)
        else:
            context = prepare(world, token)
            if phase == "materialization":
                mutate()
                operation = lambda: request(world, context)
            else:
                kwargs = request(world, context)
                mutate()
                operation = lambda: interruptible_api_call(world.agent, kwargs)
    with pytest.raises(NativeContentPolicyDenied):
        operation()
    assert world.context._denied and world.adapter.send_count == 0
    assert world.adapter._turn_pending is None and world.sdk_calls == []


@pytest.mark.parametrize("change", ["input", "history", "system", "task", "sanitize_function", "seal_method"])
def test_turn_preparation_changed_input_or_native_method_refuses(monkeypatch, change):
    world = prepared_world()
    args = arguments(history=[])
    token = admit(world, args)
    if change == "input":
        args["user_message"] += " changed"
    elif change == "history":
        args["conversation_history"] = []
    elif change == "system":
        args["system_message"] = "Injected"
    elif change == "task":
        args["task_id"] = "different"
    elif change == "sanitize_function":
        monkeypatch.setattr("agent.message_sanitization._sanitize_surrogates", world.forbidden_provider)
    else:
        monkeypatch.setattr(world.adapter, "seal_turn", world.forbidden_provider)
    with pytest.raises(NativeContentPolicyDenied):
        prepare(world, token, args)
    assert world.context._denied and world.sdk_calls == [] and world.adapter.send_count == 0


@pytest.mark.parametrize("stage", ["input", "prepared"])
@pytest.mark.parametrize("token_kind", ["unregistered", "copied", "foreign", "consumed"])
def test_turn_preparation_token_identity_and_consumption(stage, token_kind):
    world = prepared_world()
    token = admit(world)
    context = None if stage == "input" else prepare(world, token)
    actual = token if stage == "input" else context.controlled_prepared
    if token_kind == "unregistered":
        actual = object.__new__(NativeTurnInput if stage == "input" else NativePreparedTurn)
    elif token_kind == "copied":
        with pytest.raises(TypeError):
            copy.copy(actual)
        actual = object.__new__(type(actual))
    elif token_kind == "foreign":
        foreign = prepared_world()
        other_token = admit(foreign)
        actual = other_token if stage == "input" else prepare(foreign, other_token).controlled_prepared
    else:
        if stage == "input":
            context = prepare(world, token)
        interruptible_api_call(world.agent, request(world, context))
    with pytest.raises(NativeContentPolicyDenied):
        if stage == "input":
            prepare(world, actual)
        else:
            world.adapter.prepared_request(world.agent, actual)
    assert world.context._denied and world.sdk_calls == []
    assert world.adapter.send_count == (1 if token_kind == "consumed" else 0)


@pytest.mark.parametrize("target", ["messages", "equal_messages", "history", "system", "index", "aux", "token", "partial", "request", "prior_response", "equal_response_part"])
def test_turn_preparation_derived_output_and_retained_results_cannot_change(target):
    world = prepared_world()
    context = prepare(world, admit(world))
    interruptible_api_call(world.agent, request(world, context))
    args = arguments("Invented document B", history=history(world))
    context = prepare(world, admit(world, args), args)
    kwargs = request(world, context)
    if target == "messages":
        context.messages[-1]["content"] = "Changed"
    elif target == "equal_messages":
        context.messages = copy.deepcopy(context.messages)
    elif target == "history":
        context.conversation_history[0]["content"] = "Changed"
    elif target == "system":
        context.active_system_prompt = "Changed"
    elif target == "index":
        context.current_turn_user_idx = 0
    elif target == "aux":
        context.plugin_user_context = "Injected"
    elif target == "token":
        context.controlled_prepared = object.__new__(NativePreparedTurn)
    elif target == "partial":
        del context.ext_prefetch_cache
    elif target == "request":
        kwargs["messages"][-1]["content"] = "Changed"
    elif target == "prior_response":
        object.__setattr__(world.adapter.invocations[0].native_response.choices[0].message, "content", "Changed")
    else:
        response = world.adapter.invocations[0].native_response
        object.__setattr__(response, "usage", copy.copy(response.usage))
    with pytest.raises(NativeContentPolicyDenied):
        interruptible_api_call(world.agent, kwargs)
    assert world.context._denied and world.adapter.send_count == 1 and world.sdk_calls == []


@pytest.mark.parametrize("prefix", [None, "", "Changed prefix"])
def test_turn_preparation_first_system_prefix_cannot_change(prefix):
    world = prepared_world()
    args = arguments(system="First prefix")
    context = prepare(world, admit(world, args), args)
    interruptible_api_call(world.agent, request(world, context))
    args = arguments("Next input", prefix, history(world))
    with pytest.raises(NativeContentPolicyDenied):
        admit(world, args)
    assert world.adapter.send_count == 1 and world.context._denied


def test_turn_preparation_exact_raw_user_limit_and_prepared_utf8_limit():
    world = prepared_world()
    args = arguments("\ud800" * 16_384)
    token = admit(world, args)
    assert b"\\ud800" in world.adapter._turn_frames[0].raw_bytes
    with pytest.raises(NativeContentPolicyDenied):
        prepare(world, token, args)
    assert world.context._denied and world.adapter.send_count == 0


@pytest.mark.parametrize("boundary", ["admission", "prologue", "materialization", "dispatch", "after_send"])
def test_turn_preparation_cancellation_invalidates_without_partial_release(boundary):
    world = prepared_world()
    if boundary == "admission":
        world.agent._interrupt_requested = True
        operation = lambda: admit(world)
    else:
        token = admit(world)
        if boundary == "prologue":
            world.agent._interrupt_requested = True
            operation = lambda: prepare(world, token)
        else:
            context = prepare(world, token)
            if boundary == "materialization":
                world.agent._interrupt_requested = True
                operation = lambda: request(world, context)
            else:
                kwargs = request(world, context)
                if boundary == "dispatch":
                    world.agent._interrupt_requested = True
                else:
                    world.ports["main"].hook = lambda envelope: setattr(world.agent, "_interrupt_requested", True)
                operation = lambda: interruptible_api_call(world.agent, kwargs)
    with pytest.raises(InterruptedError, match="^Agent interrupted during API call$"):
        operation()
    assert world.context._denied and world.adapter._turn_pending is None
    assert world.adapter.send_count == (1 if boundary == "after_send" else 0)
    assert len(world.adapter.invocations) == (1 if boundary == "after_send" else 0)
    assert world.sdk_calls == []


@pytest.mark.parametrize("phase", ["admission", "prologue", "materialization"])
@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_turn_preparation_process_control_exceptions_invalidate_and_propagate(phase, error):
    world = prepared_world()
    if phase == "admission":
        operation = lambda: admit(world)
    else:
        token = admit(world)
        if phase == "prologue":
            operation = lambda: prepare(world, token)
        else:
            context = prepare(world, token)
            operation = lambda: request(world, context)

    def fail():
        raise error("invented")

    world.selection_hook = fail
    with pytest.raises(error, match="invented"):
        operation()
    assert world.context._denied and world.adapter._turn_pending is None and world.adapter.send_count == 0


@pytest.mark.parametrize("phase", ["input", "prepared"])
def test_turn_preparation_wrong_thread_denies(phase):
    world = prepared_world()
    token = admit(world)
    context = None if phase == "input" else prepare(world, token)
    errors = []

    def use():
        try:
            if phase == "input":
                prepare(world, token)
            else:
                request(world, context)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=use)
    thread.start()
    thread.join(5)
    assert not thread.is_alive() and len(errors) == 1
    assert type(errors[0]) is NativeContentPolicyDenied
    assert world.context._denied and world.adapter.send_count == 0


def test_turn_preparation_reentry_invalidates_even_when_operator_catches_denial():
    world = prepared_world()
    seen = []

    def reenter():
        world.admission_hook = None
        try:
            admit(world)
        except NativeContentPolicyDenied:
            seen.append("denied")

    world.admission_hook = reenter
    with pytest.raises(NativeContentPolicyDenied):
        admit(world)
    assert seen == ["denied"] and world.context._denied and world.adapter.send_count == 0


@pytest.mark.parametrize("capacity", ["handles", "bytes"])
def test_turn_preparation_shared_capacity_denies_before_sanitizer(capacity, monkeypatch):
    world = prepared_world()
    runtime = world.adapter._runtime
    if capacity == "handles":
        for _ in range(128):
            runtime.admit(world.request, b"Invented")
    else:
        runtime._retained_bytes = 8 * 1024 * 1024
    from agent import native_main_payload
    called = []
    native = native_main_payload._NATIVE_SANITIZER

    def observed(text):
        called.append(text)
        return native(text)

    # A matching native reference allows this observer to measure admission order.
    monkeypatch.setattr(native_main_payload, "_NATIVE_SANITIZER", observed)
    monkeypatch.setattr("agent.message_sanitization._sanitize_surrogates", observed)
    with pytest.raises(NativeContentPolicyDenied):
        admit(world)
    assert called == [] and world.context._denied and world.adapter.send_count == 0


def test_turn_preparation_one_pending_frame_and_eight_invocations():
    world = prepared_world()
    for index in range(8):
        args = arguments(f"Input {index}", history=None if not index else history(world))
        context = prepare(world, admit(world, args), args)
        interruptible_api_call(world.agent, request(world, context))
    with pytest.raises(NativeContentPolicyDenied):
        admit(world, arguments("Ninth input", history=history(world)))
    assert world.adapter.send_count == len(world.adapter.invocations) == len(world.adapter._turn_frames) == 8
    assert world.context._denied


def test_turn_preparation_pending_frame_cannot_be_replaced():
    world = prepared_world()
    admit(world)
    with pytest.raises(NativeContentPolicyDenied):
        admit(world)
    assert len(world.adapter._turn_frames) == 1 and world.context._denied


@pytest.mark.parametrize("operation", ["forwarder", "loop", "fallback", "compression", "title", "title_prologue",
                                      "memory", "delegate", "a2a", "tool_result", "auxiliary", "stream", "generic_main"])
@pytest.mark.parametrize("profile_drop", [False, True])
def test_turn_preparation_conversation_and_actual_siblings_still_refuse(operation, profile_drop):
    from agent import auxiliary_client, chat_completion_helpers, title_generator
    from agent.conversation_loop import run_conversation
    from agent.content_policy import guard_native_content
    from agent.memory_manager import MemoryManager
    from agent.turn_context import _maybe_title_session_at_turn_start
    from plugins.platforms.a2a import tools as a2a_tools
    from run_agent import AIAgent
    from tools import delegate_tool

    world = prepared_world()
    context = prepare(world, admit(world))
    interruptible_api_call(world.agent, request(world, context))
    if profile_drop:
        world.binding = replace(world.binding, profiles=())
    operations = {
        "forwarder": lambda: AIAgent.run_conversation(world.agent, "Invented input"),
        "loop": lambda: run_conversation(world.agent, "Invented input"),
        "fallback": lambda: chat_completion_helpers.try_activate_fallback(world.agent),
        "compression": lambda: chat_completion_helpers.handle_max_iterations(world.agent, [], 1),
        "title": lambda: title_generator.generate_title("Invented input"),
        "title_prologue": lambda: _maybe_title_session_at_turn_start(world.agent, []),
        "memory": lambda: MemoryManager().prefetch_all("Invented input"),
        "delegate": lambda: delegate_tool.delegate_task(goal="Invented input", parent_agent=world.agent),
        "a2a": lambda: a2a_tools._send_task("document_review", {"url": "http://invented.invalid", "streaming": False}, "Invented input", "", follow=False),
        "tool_result": lambda: AIAgent._tool_result_content_for_active_model(world.agent, "invented_producer", "Invented output"),
        "auxiliary": lambda: auxiliary_client.call_llm(messages=[]),
        "stream": lambda: chat_completion_helpers.interruptible_streaming_api_call(world.agent, {}),
        "generic_main": lambda: guard_native_content("main", world.agent),
    }
    entered = False
    with pytest.raises(NativeContentPolicyDenied, match="^controlled_native_operation_unsupported$"):
        with content_policy_scope(world.context):
            entered = True
            operations[operation]()
    assert entered and world.context._denied
    assert world.adapter.send_count == 1 and world.sdk_calls == []
    assert all(not port.sent for path, port in world.ports.items() if path != "main")


@pytest.mark.parametrize("value", [None, 0, 1, "false", [], object()])
def test_turn_preparation_interrupt_flag_has_no_coercion(value):
    world = prepared_world()
    world.agent._interrupt_requested = value
    with pytest.raises(NativeContentPolicyDenied, match="^controlled_native_interrupt_invalid$"):
        admit(world)
    assert world.context._denied and world.adapter.send_count == 0


@pytest.mark.parametrize("stage", ["admission", "prologue", "materialization"])
def test_turn_preparation_scope_rechecked_after_interrupt_flag_read(stage):
    world = prepared_world()
    active = False

    class MutatingAgent(SimpleNamespace):
        @property
        def _interrupt_requested(self):
            if active:
                world.binding = replace(world.binding, auth_epoch=8)
            return False

    world.agent = MutatingAgent(**vars(world.agent))
    if stage == "admission":
        operation = lambda: admit(world)
    else:
        token = admit(world)
        if stage == "prologue":
            operation = lambda: prepare(world, token)
        else:
            context = prepare(world, token)
            operation = lambda: request(world, context)

    active = True
    with pytest.raises(NativeContentPolicyDenied):
        operation()
    assert world.context._denied and world.adapter.send_count == 0


@pytest.mark.parametrize("operation", ["prepare", "materialize", "seal"])
def test_turn_preparation_foreign_agent_cannot_use_registered_tokens(operation):
    world = prepared_world()
    token = admit(world)
    context = None if operation == "prepare" else prepare(world, token)
    other = SimpleNamespace(**vars(world.agent))
    with pytest.raises(NativeContentPolicyDenied):
        if operation == "prepare":
            world.adapter._prepare_turn(other, token, tuple(arguments().values()), (None,) * 5 + (False,))
        elif operation == "materialize":
            world.adapter.prepared_request(other, context.controlled_prepared)
        else:
            world.adapter.seal_turn(other, token, turn_context=context)
    assert world.context._denied and world.adapter.send_count == 0


@pytest.mark.parametrize("exclusion", ["stream_callback", "persist_user_message", "persist_user_timestamp",
                                       "persist_user_display_kind", "persist_user_display_metadata", "moa_active"])
def test_turn_preparation_prologue_exclusions_are_not_silently_cleared(exclusion):
    world = prepared_world()
    token = admit(world)
    callback = world.forbidden_provider
    kwargs = dict(arguments(), stream_callback=None, persist_user_message=None,
        restore_or_build_system_prompt=callback, install_safe_stdio=callback,
        sanitize_surrogates=callback, summarize_user_message_for_log=callback,
        set_session_context=callback, set_current_write_origin=callback,
        ra=object(), controlled_input=token)
    kwargs[exclusion] = True if exclusion == "moa_active" else "unsupported"
    with pytest.raises(NativeContentPolicyDenied):
        build_turn_context(world.agent, **kwargs)
    assert world.context._denied and world.sdk_calls == []


def test_turn_preparation_matching_cached_prefix_and_empty_ephemeral_are_preserved():
    world = prepared_world()
    world.agent._cached_system_prompt = "Fixed system"
    world.agent.ephemeral_system_prompt = ""
    args = arguments(system="Fixed system")
    context = prepare(world, admit(world, args), args)
    interruptible_api_call(world.agent, request(world, context))
    assert context.active_system_prompt == world.agent._cached_system_prompt == "Fixed system"
    assert world.agent.ephemeral_system_prompt == ""


@pytest.mark.parametrize("path", ["fallback", "title", "compression", "memory", "delegate", "a2a", "tool_result"])
def test_turn_preparation_raw_admission_requires_complete_sibling_coverage(path):
    world = prepared_world()
    world.adapter._runtime._routes = tuple(r for r in world.routes if r.path != path)
    with pytest.raises(NativeContentPolicyDenied):
        admit(world)
    assert world.context._denied and not world.adapter._turn_frames and world.adapter.send_count == 0


@pytest.mark.parametrize("phase", ["seal", "materialization", "dispatch"])
@pytest.mark.parametrize("change", ["cancel", "capacity", "expired_route", "state"])
def test_turn_preparation_shared_proof_failure_releases_no_partial_output(phase, change):
    world = prepared_world()
    token = admit(world)
    if phase == "seal":
        original = world.adapter._runtime.derive

        def failing(handle, parents, payload, *, path):
            if change == "cancel":
                result = original(handle, parents, payload, path=path)
                world.agent._interrupt_requested = True
                return result
            mutate()
            return original(handle, parents, payload, path=path)

        world.adapter._runtime.derive = failing
        operation = lambda: prepare(world, token)
    else:
        context = prepare(world, token)
        if phase == "materialization":
            operation = lambda: request(world, context)
        else:
            kwargs = request(world, context)
            operation = lambda: interruptible_api_call(world.agent, kwargs)

    def mutate():
        if change == "cancel":
            world.agent._interrupt_requested = True
        elif change == "capacity":
            world.adapter._runtime._retained_bytes = 8 * 1024 * 1024
        elif change == "expired_route":
            world.now = 100
        else:
            world.agent._memory_manager = object()

    if phase != "seal":
        mutate()
    expected = InterruptedError if change == "cancel" else NativeContentPolicyDenied
    with pytest.raises(expected):
        operation()
    assert world.context._denied and world.adapter._turn_pending is None and world.adapter.send_count == 0
    assert world.sdk_calls == []


@pytest.mark.parametrize("change", ["truncated", "imported", "input_row_identity", "old_output", "response_identity"])
def test_turn_preparation_prior_history_inspected_before_new_raw_admission(change):
    world = prepared_world()
    context = prepare(world, admit(world))
    interruptible_api_call(world.agent, request(world, context))
    args = arguments("Next input", history=history(world))
    if change == "truncated":
        args["conversation_history"].pop(0)
    elif change == "imported":
        args["conversation_history"][0]["content"] = "Imported"
    elif change == "input_row_identity":
        first = world.adapter._turn_frames[0]
        # The first input had no history, so the next frame supplies these anchors.
        token = admit(world, args)
        args["conversation_history"][0] = dict(args["conversation_history"][0])
        with pytest.raises(NativeContentPolicyDenied):
            prepare(world, token, args)
        assert first.output is context and world.adapter.send_count == 1
        return
    elif change == "old_output":
        context.messages[0]["content"] = "Changed"
    else:
        record = world.adapter.invocations[0]
        object.__setattr__(record, "native_response", copy.copy(record.native_response))
    retained = len(world.adapter._runtime._contents)
    with pytest.raises(NativeContentPolicyDenied):
        admit(world, args)
    assert len(world.adapter._runtime._contents) == retained
    assert world.context._denied and world.adapter.send_count == 1


def test_turn_preparation_ordinary_prologue_keeps_default_and_original_behavior():
    from tests.agent.test_turn_context import _FakeAgent, _build
    agent = _FakeAgent()
    context = _build(agent)
    assert context.controlled_prepared is None
    assert context.user_message == context.messages[-1]["content"]
    assert context.active_system_prompt == agent._cached_system_prompt


def unsupported_runtime_label(kind, supported, calls):
    class EqualLabel:
        def __eq__(self, other):
            calls.append("equality")
            return True

        def __str__(self):
            calls.append("conversion")
            return supported

    class EqualString(str):
        def __eq__(self, other):
            calls.append("equality")
            return True

        def __str__(self):
            calls.append("conversion")
            return supported

    class PlainString(str):
        def __str__(self):
            calls.append("conversion")
            return supported

    if kind == "equal_object":
        return EqualLabel()
    if kind == "equal_string":
        return EqualString(supported)
    return PlainString(supported)


def runtime_label_operation(world, phase, prior_send):
    if prior_send:
        context = prepare(world, admit(world))
        interruptible_api_call(world.agent, request(world, context))
    args = arguments("Next invented input", history=history(world) if prior_send else None)
    if phase == "admission":
        return lambda: admit(world, args)
    token = admit(world, args)
    if phase == "prologue":
        return lambda: prepare(world, token, args)
    context = prepare(world, token, args)
    return lambda: request(world, context)


def assert_runtime_label_denial(world, operation, calls, monkeypatch):
    from agent import native_main_payload
    frames = world.adapter._turn_frames
    retained = len(world.adapter._runtime._contents)
    sent = world.adapter.send_count
    sanitizer_calls = []

    def blocked_sanitizer(value):
        sanitizer_calls.append(value)
        pytest.fail("Sanitizer ran after unsupported runtime label")

    monkeypatch.setattr(native_main_payload, "_NATIVE_SANITIZER", blocked_sanitizer)
    monkeypatch.setattr("agent.message_sanitization._sanitize_surrogates", blocked_sanitizer)
    with pytest.raises(NativeContentPolicyDenied, match="^controlled_native_payload_invalid$"):
        operation()
    assert calls == [] and sanitizer_calls == []
    assert world.context._denied and world.adapter._turn_pending is None
    assert world.adapter._preparation_enabled
    assert world.adapter._turn_frames is frames
    assert len(world.adapter._runtime._contents) == retained
    with pytest.raises(NativeContentPolicyDenied):
        operation()
    with pytest.raises(NativeContentPolicyDenied):
        interruptible_api_call(world.agent, {"model": "invented_model", "messages": [
            {"role": "user", "content": "Invented refused input"}]})
    assert calls == [] and sanitizer_calls == []
    assert world.adapter.send_count == len(world.ports["main"].sent) == sent
    assert len(world.adapter.invocations) == sent and world.sdk_calls == []
    assert all(not port.sent for path, port in world.ports.items() if path != "main")


@pytest.mark.parametrize("phase", ["admission", "prologue", "materialization"])
@pytest.mark.parametrize("field,supported", [("api_mode", "chat_completions"),
                                            ("provider", "synthetic_native")])
@pytest.mark.parametrize("kind", ["equal_object", "equal_string", "plain_string"])
@pytest.mark.parametrize("prior_send", [False, True])
def test_turn_preparation_runtime_labels_require_exact_strings(
    phase, field, supported, kind, prior_send, monkeypatch,
):
    world = prepared_world()
    operation = runtime_label_operation(world, phase, prior_send)
    calls = []
    setattr(world.agent, field, unsupported_runtime_label(kind, supported, calls))
    assert_runtime_label_denial(world, operation, calls, monkeypatch)


@pytest.mark.parametrize("phase", ["admission", "prologue", "materialization"])
@pytest.mark.parametrize("kind", ["equal_object", "equal_string", "plain_string", "different_model"])
@pytest.mark.parametrize("prior_send", [False, True])
def test_turn_preparation_actual_model_read_requires_exact_selected_string(
    phase, kind, prior_send, monkeypatch,
):
    world = prepared_world()
    active = False
    reads, calls = [], []
    value = ("other_invented_model" if kind == "different_model" else
             unsupported_runtime_label(kind, "invented_model", calls))

    class ModelAgent(SimpleNamespace):
        @property
        def model(self):
            if active:
                reads.append("model")
                return value
            return self.__dict__["model"]

    world.agent = ModelAgent(**vars(world.agent))
    operation = runtime_label_operation(world, phase, prior_send)
    assert type(vars(world.agent)["model"]) is str
    assert vars(world.agent)["model"] == "invented_model"
    active = True
    assert_runtime_label_denial(world, operation, calls, monkeypatch)
    assert reads == ["model"]
