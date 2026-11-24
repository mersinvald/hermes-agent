"""Qualify the new caller and one original synthetic episode."""

import asyncio
import hashlib
import json
import sys
import time
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
from hermes_content_policy import ContentPolicyRuntime, PATHS, PolicyError

from agent import issued_call_episode as episode
from agent import native_main_payload as native
from agent.content_policy import NativeContentPolicyDenied, content_policy_scope, current_context
from run_agent import AIAgent
from tests.tools.test_native_issued_call_continuation import ContinuationWorld

ARGUMENTS = '{"context":null,"goal":"Count blue squares.","role":"leaf"}'


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode()


@pytest.fixture
def record(record_property, request):
    def retain(name, value):
        record_property(name, json.dumps(value))
        with Path(request.config.option.xmlpath).with_suffix(".receipts.jsonl").open("a") as output:
            output.write(json.dumps(dict(nodeid=request.node.nodeid, property=name, value=value)) + "\n")
    return retain


class EpisodeWorld(ContinuationWorld):
    def prepare_source(self):
        self.qualifier = ContentPolicyRuntime(admission=self.admission, profiles=self.profiles,
            routes=self.routes, transports=self.transports, clock=self.clock, policy_revision=9)
        qualifications = [(self.requests[int(route.route_ref.split("_")[1])], route) for route in self.routes]
        qualifications += [(self.requests[5], next(route for route in self.routes if route.path == path
            and (route.consumer_ref, route.selection_id, route.selection_version)
                == (self.bindings[5].consumer_ref, self.bindings[5].selection_id, self.bindings[5].selection_version)))
            for path in PATHS if path != "main"]
        for request, route in qualifications:
            port = self.ports[int(route.route_ref.split("_")[1]), route.path]
            content = self.qualifier.admit(request, b"Invented qualification")
            result = self.qualifier.dispatch(request, content, path=route.path, route_ref=route.route_ref)
            self.qualifications.append((request, content, result, port.sent[-1]))
            port.sent.clear()
        self.ports[0, "main"].output = canonical(dict(id="invented_response", object="chat.completion", created=1,
            model="invented_model_0", choices=[dict(index=0, finish_reason="tool_calls", message=dict(role="assistant",
                content=None, tool_calls=[dict(id="invented-call-1", type="function",
                    function=dict(name="delegate_task", arguments=ARGUMENTS))]))],
            usage=dict(prompt_tokens=5, completion_tokens=3, total_tokens=8)))
        self.ports[3, "main"].output = canonical(dict(id="invented_child_reply", object="chat.completion", created=12,
            model="invented_model_3", choices=[dict(index=0, finish_reason="stop",
                message=dict(role="assistant", content="Three blue squares."))],
            usage=dict(prompt_tokens=3, completion_tokens=2, total_tokens=5)))
        self.ports[5, "main"].output = canonical(dict(id="invented_continuation_reply", object="chat.completion", created=21,
            model="invented_model_5", choices=[dict(index=0, finish_reason="stop",
                message=dict(role="assistant", content="The leaf counted three blue squares."))],
            usage=dict(prompt_tokens=11, completion_tokens=7, total_tokens=18)))
        self.adapter = native.bind_controlled_main(self.context, selection=self.selection_ports[0],
            result_policy=self.result_port, producer_ref="producer_0", main_route_ref="route_0_main",
            profiles=self.profiles, routes=self.routes, transports=self.transports, clock=self.clock)
        with content_policy_scope(self.context):
            self.agent = AIAgent(controlled_main=self.adapter, provider="synthetic_native", api_mode="chat_completions",
                model="invented_model_0", session_id="invented_source_session", enabled_toolsets=[], quiet_mode=True,
                skip_memory=True, skip_context_files=True, skip_background_review=True)
        self.probe_input = episode.NativeIssuedCallEpisodeProbe(
            "Request one leaf call.", "Count blue squares.", "execution", ARGUMENTS)
        self.execution_input = episode.NativeIssuedCallEpisodeExecution(
            self.requests[1], self.selection_ports[1], "producer_1", "route_1_main",
            self.requests[2], self.selection_ports[2], "producer_2",
            self.requests[3], self.selection_ports[3], "producer_3", "route_2_delegate",
            "route_3_main", "invented_child_session", "child_execution")
        self.formatter_input = episode.NativeIssuedCallEpisodeFormatter(
            self.requests[4], self.selection_ports[4], "producer_4", "route_4_main")
        self.consumer_input = episode.NativeIssuedCallEpisodeConsumer(
            self.consumer_context, self.selection_ports[5], "producer_5", "route_5_main",
            self.selections[5].destination.model_ref, "invented_consumer_session")
        return self

    def invoke(self):
        return episode.run_controlled_issued_call_episode(self.agent, self.adapter,
            probe=self.probe_input, execution=self.execution_input,
            formatter=self.formatter_input, consumer=self.consumer_input)


def test_genuine_issued_call_episode(tmp_path, monkeypatch, record):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("model: invented_model\nmemory:\n  enabled: false\n")
    root = Path(__file__).resolve().parents[2]
    paths = {name: str(Path(sys.modules[name].__file__).resolve())
             for name in ("agent.native_main_payload", "run_agent", "agent.issued_call_episode")}
    assert all(Path(path).is_relative_to(root) for path in paths.values())
    assert Path(sys.prefix).resolve() == Path(
        "/home/mersinvald/.local/share/orca/runtime/accountant/hpwa-p10.47/venv").resolve()
    record("loaded_paths", dict(paths=paths, interpreter=sys.executable, prefix=sys.prefix))
    world = EpisodeWorld("positive").prepare_source()
    started, previous, continuation_before = time.monotonic(), [], []

    def observe():
        owners = [owner for owner in native._issued_continuation_owners if owner.parent is world.adapter]
        owner = owners[0] if owners else None
        runtime = world.adapter._runtime
        if owner is not None and owner.phase == "paired" and not continuation_before:
            continuation_before.append((len(runtime._contents), len(runtime._envelopes), runtime._retained_bytes))
        boundary = (len(runtime._contents), len(runtime._envelopes),
                    None if owner is None else owner.phase, None if owner is None else owner.windows)
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
    completion = world.invoke()
    owner, = [owner for owner in native._issued_continuation_owners if owner.parent is world.adapter]
    result = owner.records[0][0]
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
    observe()
    before = (len(runtime._contents), len(runtime._envelopes), runtime._retained_bytes)
    with pytest.raises(NativeContentPolicyDenied):
        world.invoke()
    assert before == (len(runtime._contents), len(runtime._envelopes), runtime._retained_bytes)
    record("final", dict(charges=before, owner_phase=owner.phase, windows=owner.windows,
        elapsed=time.monotonic()-started, completion=completion.completion_bytes.decode(),
        request=result.request_bytes.decode(), policy_observations=owner.policy_observations))


class CallerWorld:
    def __init__(self, monkeypatch, *, failure=None, error=None, mutate=False, wrong_result=False):
        self.events, self.calls, self.active = [], {}, None
        self.parent, self.child = object(), object()
        self.token, self.probe_token, self.issued_token, self.format_token = (object() for _ in range(4))
        self.observation, self.source, self.issued, self.formatted = (object() for _ in range(4))
        self.tool, self.assistant, self.pair, self.completion = (object() for _ in range(4))
        self.result = SimpleNamespace(completion=self.completion)
        self.inputs = {}
        for name in ("Probe", "Execution", "Formatter", "Consumer"):
            cls = getattr(episode, "NativeIssuedCallEpisode" + name)
            values = tuple(self.child if field.name == "consumer_context" else object() for field in fields(cls))
            self.inputs[name.lower()] = cls(*values)
        self.original = {name: tuple(getattr(value, field.name) for field in fields(value))
                         for name, value in self.inputs.items()}
        world = self

        def event(name, args=(), kwargs=None):
            world.events.append(name)
            world.calls[name] = (args, kwargs or {})
            if failure == name:
                raise error

        class Agent:
            def close(self):
                event("close", (self,))

        class Adapter:
            _bound_context = world.parent

            def bind_tool_call_probe(self, agent, **kwargs):
                event("probe_bind", (agent,), kwargs)
                return world.probe_token

            def enable_tool_call_assistant_source(self, agent, token):
                event("enable", (agent, token))

            def run_tool_call_probe(self, agent, token):
                event("probe_run", (agent, token))
                return world.observation

            @property
            def tool_call_assistant_sources(self):
                event("source")
                return (world.source,)

            def bind_issued_call_execution(self, observation, **kwargs):
                event("execution_bind", (observation,), kwargs)
                return world.issued_token

            def run_issued_call_execution(self, token):
                event("execution_run", (token,))
                return world.issued

            def bind_issued_call_format(self, issued, **kwargs):
                event("formatter_bind", (issued,), kwargs)
                return world.format_token

            def run_issued_call_format(self, token):
                event("formatter_run", (token,))
                return world.formatted

            def retain_issued_call_wire(self, formatted):
                event("Wf", (formatted,))
                return world.tool

            def retain_tool_call_assistant_wire(self, source):
                event("Wa", (source,))
                return world.assistant

            def retain_issued_call_pair(self, assistant, tool):
                event("J", (assistant, tool))
                return world.pair

        @contextmanager
        def scope(context):
            name = "P" if context is world.parent else "C"
            event("enter_" + name, (context,))
            previous, world.active = world.active, context
            try:
                yield
                event("exit_" + name, (context,))
            finally:
                world.active = previous

        def bind(adapter, **kwargs):
            event("C_bind", (adapter,), kwargs)
            if mutate:
                for value in world.inputs.values():
                    for field in fields(value):
                        object.__setattr__(value, field.name, object())
                adapter._bound_context = object()
            return world.token

        def run(token, pair):
            event("C_run", (token, pair))
            assert world.active is world.child
            return world.result

        def getter(token):
            event("getter", (token,))
            assert world.active is world.child
            return (SimpleNamespace(completion=world.completion),) if wrong_result else (world.result,)

        def context():
            event("context")
            return world.active

        self.agent, self.adapter = Agent(), Adapter()
        for name, value in dict(AIAgent=Agent, NativeMainPayloadAdapter=Adapter,
                content_policy_scope=scope, current_context=context, bind_issued_call_continuation=bind,
                run_issued_call_continuation=run, issued_call_continuation_results=getter).items():
            monkeypatch.setattr(episode, name, value)

    def invoke(self, **changes):
        options = dict(source_agent=self.agent, source_adapter=self.adapter, **self.inputs)
        options.update(changes)
        return episode.run_controlled_issued_call_episode(**options)


ORDER = ["context", "C_bind", "enter_P", "probe_bind", "enable", "probe_run", "close", "source",
         "execution_bind", "execution_run", "formatter_bind", "formatter_run", "Wf", "Wa", "J",
         "exit_P", "enter_C", "C_run", "getter", "exit_C"]


@pytest.mark.parametrize("field", ["source_agent", "source_adapter", "probe", "execution", "formatter", "consumer"])
def test_caller_exact_type_preflight(monkeypatch, field):
    world = CallerWorld(monkeypatch)
    original = dict(source_agent=world.agent, source_adapter=world.adapter, **world.inputs)[field]
    subclass = type("Derived", (type(original),), {})
    value = object.__new__(subclass)
    with pytest.raises(NativeContentPolicyDenied, match="^issued_call_episode_input_type$"):
        world.invoke(**{field: value})
    assert world.events == []


def test_caller_outside_scope_preflight(monkeypatch):
    world = CallerWorld(monkeypatch)
    world.active = object()
    with pytest.raises(NativeContentPolicyDenied, match="^issued_call_episode_scope_active$"):
        world.invoke()
    assert world.events == ["context"]


@pytest.mark.parametrize("mutate", [False, True])
def test_caller_capture_order_forwarding_and_completion(monkeypatch, mutate):
    world = CallerWorld(monkeypatch, mutate=mutate)
    assert world.invoke() is world.completion
    assert world.events == ORDER and world.active is None
    for stage, name in (("C_bind", "consumer"), ("probe_bind", "probe"),
                        ("execution_bind", "execution"), ("formatter_bind", "formatter")):
        kwargs = world.calls[stage][1]
        assert tuple(kwargs) == tuple(field.name for field in fields(world.inputs[name]))
        assert all(actual is expected for actual, expected in zip(kwargs.values(), world.original[name]))
    for stage, expected in {
        "C_bind": (world.adapter,), "enter_P": (world.parent,), "probe_bind": (world.agent,),
        "enable": (world.agent, world.probe_token), "probe_run": (world.agent, world.probe_token),
        "close": (world.agent,), "execution_bind": (world.observation,),
        "execution_run": (world.issued_token,), "formatter_bind": (world.issued,),
        "formatter_run": (world.format_token,), "Wf": (world.formatted,), "Wa": (world.source,),
        "J": (world.assistant, world.tool), "enter_C": (world.child,),
        "C_run": (world.token, world.pair), "getter": (world.token,),
    }.items():
        assert all(actual is value for actual, value in zip(world.calls[stage][0], expected))


@pytest.mark.parametrize("stage", ORDER[1:])
def test_caller_first_exception_identity(monkeypatch, stage):
    control = asyncio.CancelledError("Invented caller cancellation")
    world = CallerWorld(monkeypatch, failure=stage, error=control)
    with pytest.raises(asyncio.CancelledError) as caught:
        world.invoke()
    assert caught.value is control and world.events == ORDER[:ORDER.index(stage)+1]
    assert world.active is None


def test_caller_shared_exception_context_identity(monkeypatch):
    try:
        try:
            try:
                raise NativeContentPolicyDenied("controlled_native_issued_continuation_coverage_missing")
            except NativeContentPolicyDenied:
                raise PolicyError("callback_failed") from None
        except PolicyError:
            raise PolicyError("post_send_rejected") from None
    except PolicyError as error:
        original, callback, native_error = error, error.__context__, error.__context__.__context__
    world = CallerWorld(monkeypatch, failure="C_run", error=original)
    with pytest.raises(PolicyError) as caught:
        world.invoke()
    assert caught.value is original and original.__context__ is callback
    assert callback.__context__ is native_error and original.__cause__ is None
    assert world.events == ORDER[:ORDER.index("C_run")+1]


def test_caller_getter_requires_result_identity(monkeypatch):
    world = CallerWorld(monkeypatch, wrong_result=True)
    with pytest.raises(NativeContentPolicyDenied, match="^issued_call_episode_result_identity$"):
        world.invoke()
    assert world.events == ORDER[:-1] and world.active is None
