"""Check public glue with inert data and one genuine constructed episode."""

import asyncio
import builtins
import hashlib
import json
import sys
import time
from dataclasses import fields, replace
from pathlib import Path
from types import FunctionType, SimpleNamespace

import pytest

from agent import conversation_loop, issued_call_episode as episode
from agent import native_main_payload as native
from agent.content_policy import NativeContentPolicyDenied, current_context
from run_agent import AIAgent
from tests.tools.test_native_issued_call_episode import EpisodeWorld, CallerWorld, ORDER
from tests.tools.test_native_issued_call_continuation import (
    no_delegation_effects as no_delegation_effects,
    invented_offline_configuration as invented_offline_configuration,
    excluded as excluded,
    retain_incremental_case_receipts as retain_incremental_case_receipts,
    record_case_start as record_case_start,
    retain_failure_contexts as retain_failure_contexts,
)


@pytest.fixture
def record(record_property, request):
    def retain(name, value):
        record_property(name, json.dumps(value))
        path = Path(request.config.option.xmlpath).with_suffix(".receipts.jsonl")
        with path.open("a") as output:
            output.write(json.dumps(dict(nodeid=request.node.nodeid, property=name, value=value)) + "\n")
    return retain


def clone(function, namespace):
    copied = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__)
    copied.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    return copied


def inert_gate(*, active=None, fail=None):
    """Run original glue code without native authority or a native registry."""
    class Agent:
        pass

    class Adapter:
        pass

    class Binding:
        pass

    agent, adapter, binding = Agent(), Adapter(), Binding()
    context = object()
    binding.agent, binding.adapter = agent, adapter
    adapter._constructed, adapter._turn_agent = binding, agent
    adapter._context = adapter._bound_context = context
    agent._native_controlled_binding = binding
    agent._native_content_policy_context = context
    values = {}
    for name in ("Probe", "Execution", "Formatter", "Consumer"):
        cls = getattr(episode, "NativeIssuedCallEpisode" + name)
        values[name.lower()] = cls(*(object() for _ in fields(cls)))
    values["probe"] = episode.NativeIssuedCallEpisodeProbe(
        "Request one leaf call.", "Count blue squares.", "execution", "{}")
    plan = episode.NativeIssuedCallConversationPlan(adapter, **values)
    calls, invalidated = [], []
    result = {}

    def execute(*args, **kwargs):
        calls.append((args, kwargs))
        if fail is not None:
            raise fail
        return object(), result

    namespace = dict(episode.__dict__)
    namespace.update(AIAgent=Agent, NativeMainPayloadAdapter=Adapter,
        native=SimpleNamespace(_owned_constructed_binding=lambda value: binding if value is agent else None,
            _ConstructedMain=Binding, _adapters=(adapter,),
            _invalidate_constructed_attempt=lambda *args: invalidated.append(args)),
        current_context=lambda: active)
    anchors = list(episode._PUBLIC_EPISODE_CODE)
    anchors[4] = (None, None, execute)
    checker = clone(episode._check_public_episode_code, namespace)
    anchors[-1] = (SimpleNamespace(checker=checker), "checker", checker, checker.__code__,
        checker.__globals__, checker.__defaults__, checker.__kwdefaults__, ())
    # This local checker only checks the inert glue. It grants no native authority.
    namespace["_PUBLIC_EPISODE_CODE"] = anchors
    namespace["_check_public_episode_code"] = checker
    # Keep the original checker body, with inert rows that describe local callables.
    for i, row in enumerate(anchors[:-1]):
        fn = execute if i == 4 else (lambda: None)
        obj = SimpleNamespace(function=fn)
        anchors[i] = (obj, "function", fn, fn.__code__, fn.__globals__, fn.__defaults__, fn.__kwdefaults__, ())
    anchors[4] = (anchors[4][0], "function", execute, execute.__code__, execute.__globals__,
        execute.__defaults__, execute.__kwdefaults__, ())
    anchors[4][0].function = execute
    gate = clone(episode._try_controlled_issued_call_conversation, namespace)
    obj = SimpleNamespace(gate=gate)
    anchors[3] = (obj, "gate", gate, gate.__code__, gate.__globals__, gate.__defaults__, gate.__kwdefaults__, tuple(gate.__kwdefaults__.items()))
    options = dict(user_message=plan.probe.user_message, system_message=plan.probe.system_message,
        task_id=plan.probe.task_id, controlled_issued_call=plan)
    return SimpleNamespace(agent=agent, adapter=adapter, binding=binding, plan=plan, options=options,
        gate=gate, calls=calls, invalidated=invalidated, result=result, namespace=namespace)


def inert_entry(entry, fake):
    """Run the original public entry against an inert episode module."""
    def imported(name, *args, **kwargs):
        if name == "agent":
            return SimpleNamespace(issued_call_episode=fake)
        return builtins.__import__(name, *args, **kwargs)
    original = AIAgent.run_conversation if entry == "class" else conversation_loop.run_conversation
    namespace = dict(original.__globals__, __builtins__=dict(vars(builtins), __import__=imported))
    function = clone(original, namespace)
    return function


@pytest.mark.parametrize("entry", ["class", "module"])
def test_public_episode_unselected(monkeypatch, entry):
    calls, result = [], {}
    def text(*args):
        calls.append(args)
        return True, result
    monkeypatch.setattr(conversation_loop, "_try_controlled_text_turn", text)
    agent = object()
    args = tuple(object() for _ in range(10))
    function = AIAgent.run_conversation if entry == "class" else conversation_loop.run_conversation
    assert function(agent, *args) is result
    assert len(calls) == 1 and all(a is b for a, b in zip(calls[0], (agent, *args)))
    assert episode._try_controlled_issued_call_conversation(object(), object()) == (False, None)


@pytest.mark.parametrize("field", [
    "user_message", "system_message", "task_id", "conversation_history", "stream_callback",
    "persist_user_message", "persist_user_timestamp", "persist_user_display_kind",
    "persist_user_display_metadata", "moa_config",
])
def test_public_episode_argument_refusal(field):
    world = inert_gate()
    world.options[field] = object()
    with pytest.raises(NativeContentPolicyDenied, match="^controlled_native_payload_denied$"):
        world.gate(world.agent, **world.options)
    assert world.calls == [] and world.invalidated == [(world.agent, world.adapter)]


@pytest.mark.parametrize("field", ["user_message", "system_message", "task_id"])
def test_public_episode_probe_binding(field):
    world = inert_gate()
    original = world.options[field]
    copied = (original + " ")[:-1]
    assert copied == original and copied is not original
    world.options[field] = copied
    with pytest.raises(NativeContentPolicyDenied):
        world.gate(world.agent, **world.options)
    assert world.calls == []


@pytest.mark.parametrize("case", ["plan", "agent", "adapter", "probe", "execution", "formatter", "consumer",
    "agent_adapter", "context"])
def test_public_episode_plan_identity(case):
    world = inert_gate()
    agent = world.agent
    if case == "plan":
        world.options["controlled_issued_call"] = object()
    elif case == "agent":
        agent = object()
    elif case in ("adapter", "probe", "execution", "formatter", "consumer"):
        field = "source_adapter" if case == "adapter" else case
        world.options["controlled_issued_call"] = replace(world.plan, **{field: object()})
    elif case == "agent_adapter":
        world.adapter._turn_agent = object()
    else:
        world.agent._native_content_policy_context = object()
    with pytest.raises(NativeContentPolicyDenied):
        world.gate(agent, **world.options)
    assert world.calls == []
    assert bool(world.invalidated) is (case != "agent")


@pytest.mark.parametrize("entry", ["class", "module"])
def test_public_episode_active_scope(entry):
    world = inert_gate(active=object())
    with pytest.raises(NativeContentPolicyDenied):
        fake = SimpleNamespace(**world.namespace)
        inert_entry(entry, fake)(world.agent, **world.options)
    assert world.calls == [] and world.invalidated


@pytest.mark.parametrize("entry", ["class", "module"])
def test_public_episode_dispatch(entry):
    """Check each original entry with an inert import and no policy claim."""
    calls, result = [], {}
    def gate(*args, **kwargs):
        calls.append((args, kwargs))
        return True, result
    def checker(anchors):
        pass
    anchors = [None] * 5
    anchors[3] = (None, None, gate)
    fake = SimpleNamespace(_check_public_episode_code=checker, _PUBLIC_EPISODE_CODE=anchors,
        NativeContentPolicyDenied=NativeContentPolicyDenied)
    anchors[-1] = (fake, "_check_public_episode_code", checker, checker.__code__, checker.__globals__,
        checker.__defaults__, checker.__kwdefaults__)
    function = inert_entry(entry, fake)
    args = tuple(object() for _ in range(11))
    plan = object()
    assert function(*args, controlled_issued_call=plan) is result
    assert len(calls) == 1 and all(a is b for a, b in zip(calls[0][0], args))
    assert calls[0][1] == dict(controlled_issued_call=plan)


def projection_data():
    usage = lambda p, c, t: SimpleNamespace(prompt_tokens=p, completion_tokens=c, total_tokens=t)
    messages = [dict(role="system", content="Request one leaf call."),
        dict(role="user", content="Count blue squares."),
        dict(role="assistant", content=None, tool_calls=[dict(id="call", type="function",
            function=dict(name="delegate_task", arguments="{}"))]),
        dict(role="tool", tool_call_id="call", name="delegate_task", tool_name="delegate_task",
            content="Three blue squares.", timestamp=1700000000.0)]
    source = SimpleNamespace(response=SimpleNamespace(usage=usage(5, 3, 8)))
    result = SimpleNamespace(request_bytes=json.dumps(dict(messages=messages)).encode(),
        completion=SimpleNamespace(final_response="The leaf counted three blue squares.", usage=usage(11, 7, 18)),
        binding=SimpleNamespace(selection=SimpleNamespace(destination=SimpleNamespace(model_ref="invented_model_5"))))
    return result, source, messages


def test_public_episode_projection():
    """Pure data establishes the dictionary mapping, with no authority claim."""
    result, source, messages = projection_data()
    output = episode._project_issued_call_conversation(result, source, "source_session")
    assert type(output) is dict
    assert set(output) == set((
        "final_response last_reasoning messages api_calls completed turn_exit_reason failed partial interrupted "
        "response_transformed pre_transform_response response_previewed model provider base_url input_tokens "
        "output_tokens cache_read_tokens cache_write_tokens reasoning_tokens prompt_tokens completion_tokens "
        "total_tokens last_prompt_tokens estimated_cost_usd cost_status cost_source service_tier session_id").split())
    assert output["messages"] == messages + [dict(role="assistant", content=result.completion.final_response)]
    assert all(a is not b for a, b in zip(output["messages"], messages))
    assert (output["prompt_tokens"], output["completion_tokens"], output["total_tokens"]) == (16, 10, 26)
    assert (output["input_tokens"], output["output_tokens"], output["last_prompt_tokens"]) == (16, 10, 11)
    assert output["api_calls"] == 2 and output["completed"] is True
    assert output["model"] == "invented_model_5" and output["session_id"] == "source_session"
    assert all(output[key] == 0 for key in ("cache_read_tokens", "cache_write_tokens", "reasoning_tokens"))
    assert all(output[key] is None for key in (
        "last_reasoning", "pre_transform_response", "estimated_cost_usd", "cost_source", "service_tier"))
    assert all(output[key] is False for key in (
        "failed", "partial", "interrupted", "response_transformed", "response_previewed"))
    assert (output["provider"], output["base_url"], output["cost_status"], output["turn_exit_reason"]) == (
        "synthetic_native", "", "unknown", "text_response(finish_reason=stop)")
    second = episode._project_issued_call_conversation(result, source, "source_session")
    assert second is not output and second["messages"] is not output["messages"]


@pytest.mark.parametrize("case", ["bool", "negative", "inconsistent", "component_bound", "sum_bound", "bytes_bound"])
def test_public_episode_projection_refusal(case):
    result, source, _ = projection_data()
    if case == "bytes_bound":
        result.completion.final_response = "x" * native._MAX_NATIVE_RESULT
    elif case == "sum_bound":
        source.response.usage = SimpleNamespace(prompt_tokens=native._MAX_INT, completion_tokens=0,
            total_tokens=native._MAX_INT)
    else:
        source.response.usage.prompt_tokens = dict(bool=True, negative=-1, inconsistent=4,
            component_bound=native._MAX_INT+1)[case]
    with pytest.raises(NativeContentPolicyDenied):
        episode._project_issued_call_conversation(result, source, "source_session")


@pytest.mark.parametrize("stage", ["C_bind", "probe_run", "close", "getter", "exit_C"])
def test_public_episode_failure(monkeypatch, stage):
    """Extraction preserves original exception objects in the inert host."""
    control = asyncio.CancelledError("Invented cancellation")
    world = CallerWorld(monkeypatch, failure=stage, error=control)
    with pytest.raises(asyncio.CancelledError) as caught:
        world.invoke()
    assert caught.value is control and world.events == ORDER[:ORDER.index(stage)+1]
    assert world.active is None


def test_public_episode_gate_exception():
    control = asyncio.CancelledError("Invented public cancellation")
    world = inert_gate(fail=control)
    with pytest.raises(asyncio.CancelledError) as caught:
        world.gate(world.agent, **world.options)
    assert caught.value is control and world.invalidated == [(world.agent, world.adapter)]


@pytest.mark.parametrize("name", ["entry", "gate", "checker", "projector", "decoder", "encoder", "executor", "host"])
def test_public_episode_code_replacement(monkeypatch, name):
    rows = episode._PUBLIC_EPISODE_CODE
    index = dict(entry=0, gate=3, executor=4, projector=5, decoder=6, encoder=7, host=2, checker=len(rows)-1)[name]
    row = rows[index]
    ran = []
    def replacement(*args, **kwargs):
        ran.append(True)
        raise AssertionError("Replacement executed")
    with monkeypatch.context() as patch:
        patch.setattr(row[0], row[1], replacement)
        # The captured entry checks the original gate and checker before use.
        with pytest.raises(NativeContentPolicyDenied):
            rows[0][2](object(), "user", controlled_issued_call=object())
        assert ran == []


def test_genuine_public_issued_call_conversation(tmp_path, monkeypatch, record, no_delegation_effects):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("model: invented_model\nmemory:\n  enabled: false\n")
    root = Path(__file__).resolve().parents[2]
    paths = {name: str(Path(sys.modules[name].__file__).resolve())
             for name in ("agent.native_main_payload", "run_agent", "agent.issued_call_episode",
                          "agent.conversation_loop", "agent.content_policy", "hermes_content_policy")}
    assert all(Path(path).is_relative_to(root) for name, path in paths.items()
               if name != "hermes_content_policy")
    assert Path(sys.prefix).resolve() == Path(
        "/home/mersinvald/.local/share/orca/runtime/accountant/hpwa-p10.47/venv").resolve()
    record("loaded_paths", dict(paths=paths, interpreter=sys.executable, prefix=sys.prefix,
        sha256={name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in paths.items()}))
    record("phase", "source_preparation")
    world = EpisodeWorld("positive").prepare_source()
    record("phase", "source_prepared")
    source_session = world.agent.session_id
    plan = episode.NativeIssuedCallConversationPlan(world.adapter, world.probe_input,
        world.execution_input, world.formatter_input, world.consumer_input)
    started, previous, continuation_before = time.monotonic(), [], []

    def observe():
        owners = [owner for owner in native._issued_continuation_owners if owner.parent is world.adapter]
        owner = owners[0] if owners else None
        runtime = world.adapter._runtime
        if owner is not None and owner.phase == "paired" and not continuation_before:
            continuation_before.append((len(runtime._contents), len(runtime._envelopes), runtime._retained_bytes))
        boundary = (len(runtime._contents), len(runtime._envelopes),
                    None if owner is None else owner.phase, None if owner is None else owner.windows,
                    *(getattr(rows.get(world.adapter), "phase", None) for rows in (
                        native._tool_call_probes, native._issued_call_frames, native._issued_call_formats)))
        if previous == [boundary]:
            return
        previous[:] = [boundary]
        indices = {row.handle: i for i, row in enumerate(runtime._contents.values())}
        record("boundary", dict(elapsed=time.monotonic()-started, boundary=boundary,
            retained_bytes=runtime._retained_bytes,
            sends=[len(world.ports[i, "main"].sent) for i in range(6)],
            graph=[dict(index=indices[row.handle], bytes=len(row.payload), hex=row.payload.hex(),
                sha256=hashlib.sha256(row.payload).hexdigest(), parents=[indices[p] for p in row.parents],
                binding=row.binding, profiles=row.profiles, path=row.path)
                for row in runtime._contents.values()]))

    world.hooks["clock"] = observe
    record("phase", "public_entry")
    public = world.agent.run_conversation(plan.probe.user_message,
        system_message=plan.probe.system_message, task_id=plan.probe.task_id, controlled_issued_call=plan)
    owner, = [owner for owner in native._issued_continuation_owners if owner.parent is world.adapter]
    result = owner.records[0][0]
    completion = result.completion
    runtime = owner.runtime
    assert completion is result.completion is owner.completion
    assert current_context() is None and owner.phase == "released" and owner.windows == 6
    assert result.binding is owner.binding and result.pair is owner.pair
    assert result.request["model"] is world.selections[5].destination.model_ref
    probe = native._tool_call_probes[world.adapter]
    issued = native._issued_call_frames[world.adapter]
    formatter = native._issued_call_formats[world.adapter]
    assert result.tools is probe.request["tools"] is result.request["tools"]
    assert all(result.request["messages"][i] is probe.request["messages"][i] for i in range(2))
    assert all(result.request["messages"][i+2] is result.pair.messages[i] for i in range(2))
    assert set(result.request) == {"model", "messages", "tools", "stream", "tool_choice"}
    assert result.request["stream"] is False and result.request["tool_choice"] == "none"
    assert result.invocation.native_response is completion.response is owner.response
    assert result.invocation.response_bytes is world.ports[5, "main"].output
    assert completion.normalized is result.normalized and completion.usage is result.normalized.usage
    assert completion.final_response == "The leaf counted three blue squares."
    assert probe.closed and probe.constructed.closed
    assert issued.phase == formatter.phase == "released"
    assert issued.frame.child_adapter._constructed.closed
    source = result.pair.assistant_result.source_result
    formatted = result.pair.tool_result.source_result
    issued_result = formatted.source_result
    assert source.source_observation is issued_result.observation is issued_result.binding.observation
    assert source.source_observation is probe.records[0]
    assert source.response is native._assistant_source_owner(world.adapter).response
    assert issued_result is issued.result_anchors[0][0]
    assert formatted is formatter.records[0][0]
    assert issued_result.invocation is issued.frame.record
    assert issued_result.invocation.child_completion is not None
    assert issued_result.return_facts is issued_result.invocation.return_facts
    assert formatted.message["content"] is issued_result.aggregate
    assert all(row.profiles == ("restricted_documents",) for row in runtime._contents.values())
    assert runtime._poisoned is False and owner.control_exception is None
    assert owner.charge_after == (len(runtime._contents), len(runtime._envelopes), runtime._retained_bytes)
    assert len(owner.policy_observations) > 0
    assert all(route.expires_at > world.now for route in world.routes)
    assert len(world.routes) <= 128 and len(world.profiles) <= 32 and len(world.transports) <= 128
    assert all(len(row.payload) <= 262144 for row in runtime._contents.values())
    assert len(result.invocation.response_bytes) <= 65536
    assert all(port.sent[0].destination == port.destination for port in dict.fromkeys(world.ports.values()) if port.sent)
    assert owner.selection.destination is owner.route.destination is world.selections[5].destination
    assert [len(world.ports[i, "main"].sent) for i in range(6)] == [1, 0, 1, 1, 0, 1]
    assert len(world.qualifications) == 27 and len(runtime._envelopes) == 4
    h, k, w, r, n = (owner.history_content, owner.request_content, owner.wire_content,
                      owner.response_content, owner.normalized_content)
    j, a = result.pair.message_content, result.binding.authorization_content
    for handle, parents in ((h, (j, a)), (k, (h,)), (w, (h, k)), (r, (w,)),
                            (n, (h, k, w, r)), (completion.completion_content, (j, a, h, k, w, r, n))):
        assert runtime._content(handle).parents == parents
    assert len(continuation_before) == 1
    assert len(runtime._contents) == continuation_before[0][0] + 6
    assert len(runtime._envelopes) == continuation_before[0][1] + 1
    assert runtime._retained_bytes == sum(len(row.payload) for row in runtime._contents.values())
    assert type(public) is dict and public["completed"] is True
    assert public["final_response"] is completion.final_response
    assert public["messages"] == json.loads(result.request_bytes)["messages"] + [
        dict(role="assistant", content=completion.final_response)]
    assert len(public["messages"]) == 5
    assert [row["role"] for row in public["messages"]] == ["system", "user", "assistant", "tool", "assistant"]
    assert all(public["messages"][i] is not result.request["messages"][i] for i in range(4))
    assert public["model"] is result.binding.selection.destination.model_ref
    assert public["model"] == "invented_model_5" and world.agent.model == "invented_model_0"
    assert public["session_id"] is source_session and source_session == "invented_source_session"
    assert result.agent.session_id == "invented_consumer_session" != source_session
    assert public["api_calls"] == 2
    assert (public["prompt_tokens"], public["completion_tokens"], public["total_tokens"]) == (16, 10, 26)
    assert (public["input_tokens"], public["output_tokens"], public["last_prompt_tokens"]) == (16, 10, 11)
    assert len(json.dumps(public, sort_keys=True, ensure_ascii=True, allow_nan=False,
        separators=(",", ":")).encode()) <= native._MAX_NATIVE_RESULT
    assert world.agent.client is world.agent._primary_runtime is None
    assert result.agent.client is result.agent._primary_runtime is None
    record("public_result", public)
    observe()
    before = (len(runtime._contents), len(runtime._envelopes), runtime._retained_bytes)
    with pytest.raises(NativeContentPolicyDenied):
        world.agent.run_conversation(plan.probe.user_message, system_message=plan.probe.system_message,
            task_id=plan.probe.task_id, controlled_issued_call=plan)
    assert before == (len(runtime._contents), len(runtime._envelopes), runtime._retained_bytes)
    record("final", dict(charges=before, owner_phase=owner.phase, windows=owner.windows,
        elapsed=time.monotonic()-started, completion=completion.completion_bytes.decode(),
        request=result.request_bytes.decode(), policy_observations=owner.policy_observations))



@pytest.mark.parametrize("stage", [None, "C_bind", "probe_run", "close", "getter", "exit_C", "projection"])
def test_public_episode_executor_scope(monkeypatch, stage):
    """Check extracted public sequencing with inert ports and no native authority."""
    control = asyncio.CancelledError("Invented public executor cancellation")
    world = CallerWorld(monkeypatch, failure=stage, error=control)
    world.agent.session_id = "source_session"
    projected = {}
    def project(result, source, session):
        assert world.active is world.child
        assert result is world.result and source is world.source and session is world.agent.session_id
        world.events.append("projection")
        if stage == "projection":
            raise control
        return projected
    namespace = dict(episode.__dict__)
    execute = clone(episode._execute_issued_call_episode, namespace)
    checker = clone(episode._check_public_episode_code, namespace)
    anchors = list(episode._PUBLIC_EPISODE_CODE)
    components = {
        4: execute, 5: project, 8: episode.current_context, 9: episode.content_policy_scope,
        10: episode.content_policy_scope.__wrapped__, 11: episode.bind_issued_call_continuation,
        12: episode.run_issued_call_continuation, 13: episode.issued_call_continuation_results,
        14: type(world.agent).close, 25: type(world.adapter).tool_call_assistant_sources.fget,
        len(anchors)-1: checker,
    }
    for index in range(15, 25):
        components[index] = getattr(type(world.adapter), anchors[index][1])
    for index, function in components.items():
        obj = SimpleNamespace(function=function)
        anchors[index] = (obj, "function", function, function.__code__, function.__globals__,
            function.__defaults__, function.__kwdefaults__, tuple((function.__kwdefaults__ or {}).items()))
    namespace.update(_PUBLIC_EPISODE_CODE=anchors, _check_public_episode_code=checker)
    options = dict(source_agent=world.agent, source_adapter=world.adapter, **world.inputs,
        project_conversation=True)
    if stage is None:
        completion, output = execute(**options)
        assert completion is world.completion and output is projected
        assert world.events == ORDER[:-1] + ["projection", "exit_C"]
    else:
        with pytest.raises(asyncio.CancelledError) as caught:
            execute(**options)
        assert caught.value is control
        expected = ORDER[:-1] + ["projection"] if stage == "projection" else ORDER[:ORDER.index(stage)+1]
        if stage == "exit_C":
            expected = ORDER[:-1] + ["projection", "exit_C"]
        assert world.events == expected
    assert world.active is None


@pytest.mark.parametrize("target", ["checker_code", "projector_code", "entry_defaults"])
def test_public_episode_callable_metadata(monkeypatch, target):
    def changed(*args, **kwargs):
        raise AssertionError("Changed code executed")
    anchors = episode._PUBLIC_EPISODE_CODE
    with monkeypatch.context() as patch:
        if target == "entry_defaults":
            patch.setitem(AIAgent.run_conversation.__kwdefaults__, "controlled_issued_call", object())
        else:
            function = anchors[-1][2] if target == "checker_code" else anchors[5][2]
            patch.setattr(function, "__code__", changed.__code__)
        with pytest.raises(NativeContentPolicyDenied):
            anchors[0][2](object(), "user", controlled_issued_call=object())


def test_public_episode_projection_exact_bounds():
    result, source, _ = projection_data()
    output = episode._project_issued_call_conversation(result, source, "source_session")
    encoded = json.dumps(output, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode()
    request = json.loads(result.request_bytes)
    request["messages"][0]["content"] += "x" * (native._MAX_NATIVE_RESULT - len(encoded))
    result.request_bytes = json.dumps(request).encode()
    bounded = episode._project_issued_call_conversation(result, source, "source_session")
    assert len(json.dumps(bounded, sort_keys=True, ensure_ascii=True, allow_nan=False,
        separators=(",", ":")).encode()) == native._MAX_NATIVE_RESULT
    result, source, _ = projection_data()
    source.response.usage = SimpleNamespace(prompt_tokens=native._MAX_INT-21, completion_tokens=3,
        total_tokens=native._MAX_INT-18)
    assert episode._project_issued_call_conversation(result, source, "source_session")["total_tokens"] == native._MAX_INT
