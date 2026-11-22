"""Invented controlled ports at the actual native main boundary."""

import copy
import json
import socket
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
import httpx
from hermes_content_policy import (
    PATHS, ContentPolicyRuntime, Destination, ExecutionBinding, ProfilePolicy, RouteBinding,
)

from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
from agent.content_policy import NativeContentContext, NativeContentPolicyDenied, content_policy_scope
from agent.native_main_payload import NativeMainSelection, NativeResultPolicy, bind_controlled_main


@pytest.fixture(autouse=True)
def invented_offline_configuration(tmp_path, monkeypatch):
    home = tmp_path / "invented_hermes"
    home.mkdir()
    (home / "config.yaml").write_text("model: invented_model\nmemory:\n  enabled: false\n")
    monkeypatch.setenv("HERMES_HOME", str(home))

    def no_network(*args, **kwargs):
        pytest.fail("Network callback ran")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(httpx.Client, "send", no_network)
    monkeypatch.setattr(httpx.AsyncClient, "send", no_network)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":")).encode()


def reply(content="Invented answer A"):
    return canonical({
        "id": "invented_reply", "object": "chat.completion", "created": 12,
        "model": "invented_model", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    })


class ControlledPort:
    def __init__(self, destination):
        self.destination = destination
        self.sent = []
        self.output = reply()
        self.hook = None
        self.resolve_hook = None

    def resolve(self):
        if self.resolve_hook:
            self.resolve_hook()
        return self.destination

    def send(self, envelope):
        assert envelope.destination == self.destination
        self.sent.append(envelope)
        if self.hook:
            self.hook(envelope)
        return self.output


class World:
    def __init__(self, *, missing=()):
        self.request = object()
        self.binding = ExecutionBinding("owner", "execution", "lineage", "document_specialist",
                                        7, 9, "selection", 3, ("restricted_documents",))
        self.now = 10
        self.admission_hook = self.selection_hook = self.producer_hook = None
        self.ports = {}
        self.routes = []
        self.receiver_bindings = {}
        for path in PATHS:
            destination = Destination(f"invented_{path}", "synthetic_native", "invented_model",
                                      "invented_endpoint", 1)
            port = self.ports[path] = ControlledPort(destination)
            self.receiver_bindings[id(port)] = (port, replace(self.binding,
                execution_ref=f"receiver_{path}", consumer_ref="document_specialist"))
            if path not in missing:
                self.routes.append(RouteBinding(f"route_{path}", "selection", 3,
                    "document_specialist", path, destination, 100,
                    f"synthetic_inprocess_{path}", "synthetic_only"))
        self.routes = tuple(self.routes)
        route_refs = tuple(r.route_ref for r in self.routes)
        self.profiles = (
            ProfilePolicy("restricted_documents", route_refs, PATHS),
            ProfilePolicy("extra_restriction", tuple(r for r in route_refs if r != "route_main"), PATHS),
        )
        self.selection = NativeMainSelection("selection", 3, "chat_completions",
                                             self.ports["main"].destination)
        self.policy = NativeResultPolicy("invented_producer", 4, ("restricted_documents",))
        self.admission = SimpleNamespace(current=self.current)
        self.selection_port = SimpleNamespace(current=self.selected)
        self.result_port = SimpleNamespace(current=self.produced)
        self.context = NativeContentContext(self.admission, self.request)
        self.agent = SimpleNamespace(_native_content_policy_required=True,
            _native_content_policy_context=self.context, api_mode="chat_completions",
            provider="synthetic_native")
        self.adapter = None
        self.sdk_calls = []
        self.qualification_records = ()
        self.agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=self.forbidden_provider)))
        self.agent._run_codex_stream = self.forbidden_provider
        self.agent._anthropic_messages_create = self.forbidden_provider

    def current(self, handle):
        if self.admission_hook:
            self.admission_hook()
        if handle is self.request:
            return self.binding
        registered, binding = self.receiver_bindings[id(handle)]
        assert registered is handle
        return binding

    def selected(self, handle):
        assert handle is self.request
        if self.selection_hook:
            self.selection_hook()
        return self.selection

    def produced(self, producer):
        assert producer == "invented_producer"
        if self.producer_hook:
            self.producer_hook()
        return self.policy

    def qualify(self):
        runtime = ContentPolicyRuntime(admission=self.admission, profiles=self.profiles,
            routes=self.routes, transports=tuple((p.destination.transport_ref, p) for p in self.ports.values()),
            clock=lambda: self.now, policy_revision=9)
        records = []
        for path in PATHS:
            content = runtime.admit(self.request, b"Invented declared operation")
            result = runtime.dispatch(self.request, content, path=path, route_ref=f"route_{path}")
            port = self.ports[path]
            envelope, = port.sent
            assert envelope.payload == b"Invented declared operation"
            assert envelope.destination == port.destination
            records.append((path, port, envelope, result, port.output))
            port.sent = []
        self.qualification_records = tuple(records)

    def bind(self, *, qualify=True):
        if qualify:
            self.qualify()
        self.adapter = bind_controlled_main(self.context, selection=self.selection_port,
            result_policy=self.result_port, producer_ref="invented_producer", main_route_ref="route_main",
            profiles=self.profiles, routes=self.routes,
            transports=tuple((port.destination.transport_ref, port) for port in self.ports.values()),
            clock=lambda: self.now)
        return self

    def forbidden_provider(self, *args, **kwargs):
        self.sdk_calls.append("provider")
        pytest.fail("SDK provider callback ran")

    def make_client(self, *args, **kwargs):
        self.sdk_calls.append("client")
        pytest.fail("SDK client construction ran")

    def call(self, request=None):
        if request is None:
            request = {"model": "invented_model", "messages": [
                {"role": "user", "content": "Invented document A"}]}
        return _dispatch_nonstreaming_api_request(self.agent, request, make_client=self.make_client)


def test_complete_native_payload_and_result_release():
    world = World().bind()
    with content_policy_scope(world.context):
        result = world.call()
    expected = b'{"api_mode":"chat_completions","destination":{"endpoint_ref":"invented_endpoint","model_ref":"invented_model","provider_ref":"synthetic_native","revision":1,"transport_ref":"invented_main"},"request":{"messages":[{"content":"Invented document A","role":"user"}],"model":"invented_model"}}'
    record, = world.adapter.invocations
    assert world.ports["main"].sent[0].payload == expected == record.request_bytes
    assert record.response_bytes == reply()
    assert record.native_response is result
    assert record.envelope is world.ports["main"].sent[0]
    assert record.observed_destination == world.selection.destination
    assert record.result_content is not record.request_content
    assert record.request_handle is world.request
    assert result.choices[0].message.content == "Invented answer A"
    assert result.usage.total_tokens == 5
    assert world.adapter.send_count == 1 and world.sdk_calls == []
    assert len(world.qualification_records) == 8
    expected_result = b'{"choices":[{"finish_reason":"stop","index":0,"message":{"content":"Invented answer A","role":"assistant"}}],"created":12,"id":"invented_reply","model":"invented_model","object":"chat.completion","usage":{"completion_tokens":2,"prompt_tokens":3,"total_tokens":5}}'
    assert record.response_bytes == expected_result


def test_reuse_keeps_parent_restrictions_after_source_profiles_drop():
    world = World().bind()
    result = world.call()
    world.binding = replace(world.binding, profiles=())
    request = {"model": "invented_model", "messages": [
        {"role": "user", "content": "Invented document A"},
        {"role": "assistant", "content": result.choices[0].message.content},
        {"role": "user", "content": "Invented document B"},
    ]}
    with content_policy_scope(world.context):
        world.call(request)
    assert world.ports["main"].sent[1].profiles == ("restricted_documents",)
    assert world.adapter.send_count == 2 and world.sdk_calls == []


@pytest.mark.parametrize("path", PATHS)
def test_exact_controlled_shared_ports_support_their_declared_operation(path):
    world = World()
    runtime = ContentPolicyRuntime(admission=world.admission, profiles=world.profiles,
        routes=world.routes, transports=tuple((p.destination.transport_ref, p) for p in world.ports.values()),
        clock=lambda: world.now, policy_revision=9)
    payload = b"Invented declared operation"
    source = runtime.admit(world.request, payload)
    result = runtime.dispatch(world.request, source, path=path, route_ref=f"route_{path}")
    assert result is not source
    assert world.ports[path].sent[0].payload == payload
    assert sum(len(p.sent) for p in world.ports.values()) == 1


@pytest.mark.parametrize("path", PATHS[1:])
def test_missing_required_sibling_route_prevents_native_main(path):
    world = World(missing=(path,)).bind(qualify=False)
    with pytest.raises(NativeContentPolicyDenied):
        world.call()
    assert world.adapter.send_count == 0 and world.sdk_calls == []


def next_request(world, content="Invented document B"):
    previous = world.adapter.invocations[-1]
    messages = json.loads(previous.request_bytes)["request"]["messages"]
    messages += [{"role": "assistant", "content": previous.native_response.choices[0].message.content},
                 {"role": "user", "content": content}]
    return {"model": "invented_model", "messages": messages}


def denied(world, callback, sends=0):
    with pytest.raises(NativeContentPolicyDenied):
        callback()
    assert world.context._denied
    assert world.adapter.send_count == sends and world.sdk_calls == []
    with pytest.raises(NativeContentPolicyDenied):
        world.call()
    assert world.adapter.send_count == sends


def test_every_accepted_result_handle_remains_a_derive_parent(monkeypatch):
    world = World().bind()
    parents = []
    derive = ContentPolicyRuntime.derive

    def observed(runtime, handle, prior, payload, *, path):
        parents.append(prior)
        return derive(runtime, handle, prior, payload, path=path)

    monkeypatch.setattr(ContentPolicyRuntime, "derive", observed)
    world.call()
    world.call(next_request(world))
    world.call(next_request(world, "Invented document C"))
    first, second, third = world.adapter.invocations
    assert parents == [(first.result_content,), (first.result_content, second.result_content)]
    assert third.request_bytes == world.ports["main"].sent[-1].payload
    assert first.result_content is not second.result_content


def test_new_restriction_denies_reuse_before_controlled_send():
    world = World().bind()
    world.call()
    request = next_request(world)
    world.binding = replace(world.binding, profiles=("restricted_documents", "extra_restriction"))
    denied(world, lambda: world.call(request), sends=1)


@pytest.mark.parametrize("policy", [None, NativeResultPolicy("other_producer", 4, ("restricted_documents",)),
    NativeResultPolicy("invented_producer", 5, ("restricted_documents",)),
    NativeResultPolicy("invented_producer", 4, ()),
    NativeResultPolicy("invented_producer", 4, ("restricted_documents", "extra_restriction"))])
def test_unknown_or_changed_producer_policy_denies_reuse(policy):
    world = World().bind()
    world.call()
    request = next_request(world)
    world.policy = policy
    denied(world, lambda: world.call(request), sends=1)


def test_additional_output_restriction_refuses_adapter_binding():
    world = World()
    world.policy = replace(world.policy, profiles=("restricted_documents", "extra_restriction"))
    with pytest.raises(NativeContentPolicyDenied):
        world.bind()
    assert world.context._denied
    assert sum(len(port.sent) for port in world.ports.values()) == 0


@pytest.mark.parametrize("field,value", [
    ("owner_ref", "other_owner"), ("execution_ref", "other_execution"),
    ("lineage_ref", "other_lineage"), ("consumer_ref", "other_consumer"),
    ("auth_epoch", 8), ("policy_revision", 10), ("selection_id", "other_selection"),
    ("selection_version", 4), ("profiles", ("extra_restriction",)),
])
@pytest.mark.parametrize("boundary", ["before", "send"])
def test_complete_current_binding_facts_survive_callback_boundaries(field, value, boundary):
    world = World().bind()

    def change(*args):
        world.binding = replace(world.binding, **{field: value})

    if boundary == "before":
        change()
    else:
        world.ports["main"].hook = change
    denied(world, world.call, sends=int(boundary == "send"))
    assert world.adapter.invocations == ()


@pytest.mark.parametrize("field,value", [("provider", "other"), ("api_mode", "codex_responses")])
@pytest.mark.parametrize("boundary", ["before", "send"])
def test_actual_agent_provider_and_mode_are_current(field, value, boundary):
    world = World().bind()

    def change(*args):
        setattr(world.agent, field, value)

    if boundary == "before":
        change()
    else:
        world.ports["main"].hook = change
    denied(world, world.call, sends=int(boundary == "send"))


@pytest.mark.parametrize("change", ["request_model", "selection_id", "selection_version",
    "selection_model", "selection_endpoint", "destination", "sibling_destination", "expiry"])
def test_request_selection_and_actual_destination_mismatch_prevent_send(change):
    world = World().bind()
    request = {"model": "invented_model", "messages": [{"role": "user", "content": "Invented document A"}]}
    if change == "request_model":
        request["model"] = "other_model"
    elif change == "selection_id":
        world.selection = replace(world.selection, selection_id="other_selection")
    elif change == "selection_version":
        world.selection = replace(world.selection, selection_version=4)
    elif change.startswith("selection_"):
        field = "model_ref" if change == "selection_model" else "endpoint_ref"
        world.selection = replace(world.selection,
            destination=replace(world.selection.destination, **{field: "other"}))
    elif change in ("destination", "sibling_destination"):
        port = world.ports["main" if change == "destination" else "fallback"]
        port.destination = replace(port.destination, endpoint_ref="other_endpoint")
    else:
        world.now = 100
    denied(world, lambda: world.call(request))


@pytest.mark.parametrize("change", ["selection", "destination", "producer", "request", "context", "clock", "callback"])
def test_changes_during_authorized_send_reject_output_and_count_the_send(change):
    world = World().bind()
    request = {"model": "invented_model", "messages": [{"role": "user", "content": "Invented document A"}]}

    def mutate(envelope):
        if change == "selection":
            world.selection = replace(world.selection, selection_version=4)
        elif change == "destination":
            world.ports["main"].destination = replace(world.selection.destination, revision=2)
        elif change == "producer":
            world.policy = replace(world.policy, revision=5)
        elif change == "request":
            request["messages"][0]["content"] = "Changed invented request"
        elif change == "context":
            world.agent._native_content_policy_context = NativeContentContext(world.admission, world.request)
        elif change == "clock":
            world.now = 100
        else:
            world.ports["main"].send = lambda envelope: reply()

    world.ports["main"].hook = mutate
    denied(world, lambda: world.call(request), sends=1)
    assert world.adapter.invocations == ()


@pytest.mark.parametrize("change", ["admission", "handle", "selection_port", "result_port", "detach", "copy", "foreign", "method"])
def test_changed_captured_identities_never_fall_through_to_sdk(change):
    world = World().bind()
    if change == "admission":
        world.context.admission = SimpleNamespace(current=world.current)
    elif change == "handle":
        world.context.request_handle = object()
    elif change == "selection_port":
        world.adapter._selection_port = SimpleNamespace(current=lambda _: world.selection)
    elif change == "result_port":
        world.adapter._result_port = SimpleNamespace(current=lambda _: world.policy)
    elif change == "detach":
        world.context._controlled_main_payload = None
    elif change == "copy":
        world.context._controlled_main_payload = copy.copy(world.adapter)
    elif change == "foreign":
        world.context._controlled_main_payload = World().bind().adapter
    else:
        world.selection_port.current = lambda _: world.selection
    denied(world, world.call)


@pytest.mark.parametrize("kind", ["thread", "reentry"])
def test_thread_transfer_and_caught_reentry_invalidate_context(kind):
    world = World().bind()
    if kind == "reentry":
        def reenter(envelope):
            with pytest.raises(NativeContentPolicyDenied):
                world.call()
        world.ports["main"].hook = reenter
        denied(world, world.call, sends=1)
    else:
        errors = []
        def run():
            try:
                world.call()
            except NativeContentPolicyDenied as exc:
                errors.append(exc)
        thread = threading.Thread(target=run)
        thread.start()
        thread.join(5)
        assert not thread.is_alive() and len(errors) == 1
        denied(world, world.call)


@pytest.mark.parametrize("change", ["response", "message", "usage", "record_copy", "response_copy", "handle_copy"])
def test_retained_original_response_and_handle_identity_are_required(change):
    world = World().bind()
    response = world.call()
    request = next_request(world)
    record = world.adapter.invocations[0]
    if change == "response":
        object.__setattr__(response, "model", "other_model")
    elif change == "message":
        object.__setattr__(response.choices[0].message, "content", "Changed invented answer")
    elif change == "usage":
        object.__setattr__(response.usage, "total_tokens", object())
    elif change == "record_copy":
        world.adapter._records = (copy.copy(record),)
    elif change == "response_copy":
        object.__setattr__(record, "native_response", copy.copy(response))
    else:
        object.__setattr__(record, "result_content", copy.copy(record.result_content))
    denied(world, lambda: world.call(request), sends=1)


def test_eight_record_limit_denies_before_ninth_send():
    world = World().bind()
    world.call()
    for index in range(7):
        world.call(next_request(world, f"Invented document {index}"))
    assert len(world.adapter.invocations) == 8
    denied(world, lambda: world.call(next_request(world)), sends=8)


@pytest.mark.parametrize("profiles_dropped", [False, True])
@pytest.mark.parametrize("operation", ["fallback", "compression", "memory", "delegate", "a2a", "tool_result",
    "title", "auxiliary", "auxiliary_impl", "stream", "loop", "forwarder", "prologue", "title_prologue"])
def test_actual_sibling_guards_refuse_after_successful_controlled_scope_entry(operation, profiles_dropped):
    from agent import auxiliary_client, chat_completion_helpers, title_generator
    from agent.conversation_loop import run_conversation
    from agent.memory_manager import MemoryManager
    from agent.turn_context import build_turn_context, _maybe_title_session_at_turn_start
    from plugins.platforms.a2a import tools as a2a_tools
    from run_agent import AIAgent
    from tools import delegate_tool

    world = World().bind()
    world.call()
    if profiles_dropped:
        world.binding = replace(world.binding, profiles=())
    callback = world.forbidden_provider
    operations = {
        "fallback": lambda: chat_completion_helpers.try_activate_fallback(world.agent),
        "compression": lambda: chat_completion_helpers.handle_max_iterations(world.agent, [], 1),
        "memory": lambda: MemoryManager().prefetch_all("Invented input"),
        "delegate": lambda: delegate_tool.delegate_task(goal="Invented input", parent_agent=world.agent),
        "a2a": lambda: a2a_tools._send_task("document_review", {"url": "http://invented.invalid", "streaming": False}, "Invented input", "", follow=False),
        "tool_result": lambda: AIAgent._tool_result_content_for_active_model(world.agent, "invented_producer", "Invented output"),
        "title": lambda: title_generator.generate_title("Invented input"),
        "auxiliary": lambda: auxiliary_client.call_llm(messages=[]),
        "auxiliary_impl": lambda: auxiliary_client._call_llm_impl(messages=[]),
        "stream": lambda: chat_completion_helpers.interruptible_streaming_api_call(world.agent, {"model": "invented_model"}),
        "loop": lambda: run_conversation(world.agent, "Invented input"),
        "forwarder": lambda: AIAgent.run_conversation(world.agent, "Invented input"),
        "title_prologue": lambda: _maybe_title_session_at_turn_start(world.agent, []),
        "prologue": lambda: build_turn_context(world.agent, "Invented input", None, [], "task", None, None,
            restore_or_build_system_prompt=callback, install_safe_stdio=callback,
            sanitize_surrogates=callback, summarize_user_message_for_log=callback,
            set_session_context=callback, set_current_write_origin=callback, ra=object()),
    }
    entered = False
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            entered = True
            operations[operation]()
            pytest.fail("Sibling function returned")
    assert entered and world.context._denied
    assert world.adapter.send_count == 1 and world.sdk_calls == []
    assert all(not port.sent for path, port in world.ports.items() if path != "main")


@pytest.mark.parametrize("mode,provider", [("codex_responses", "synthetic_native"),
    ("anthropic_messages", "anthropic"), ("bedrock_converse", "bedrock"),
    ("chat_completions", "moa"), ("unknown", "custom")])
def test_sibling_nonstreaming_modes_cannot_use_the_adapter(mode, provider):
    world = World().bind()
    world.agent.api_mode, world.agent.provider = mode, provider
    denied(world, world.call)


@pytest.mark.parametrize("context_kind", ["absent", "ordinary", "missing_required"])
@pytest.mark.parametrize("streaming", [False, True])
def test_reserved_marker_without_adapter_denies_before_client_or_sdk(context_kind, streaming):
    from agent.chat_completion_helpers import interruptible_streaming_api_call
    world = World()
    if context_kind in ("absent", "missing_required"):
        world.agent._native_content_policy_context = None
        world.agent._native_content_policy_required = context_kind == "missing_required"
    else:
        world.binding = replace(world.binding, profiles=())
    with pytest.raises(NativeContentPolicyDenied):
        if streaming:
            interruptible_streaming_api_call(world.agent, {"model": "invented_model"})
        else:
            world.call()
    assert world.sdk_calls == []


class Unsupported:
    calls = []

    def __str__(self):
        self.calls.append("str")
        raise AssertionError("Conversion ran")

    def __eq__(self, other):
        self.calls.append("eq")
        raise AssertionError("Equality ran")

    def __hash__(self):
        self.calls.append("hash")
        return 1


class MappingSubclass(dict):
    def __iter__(self):
        pytest.fail("Mapping subclass iteration ran")


class BytesSubclass(bytes):
    def decode(self, *args, **kwargs):
        pytest.fail("Bytes subclass conversion ran")


@pytest.mark.parametrize("shape", ["object", "mapping_subclass", "key_object", "key_int", "model_object",
    "model_subclass", "messages_object", "messages_tuple", "message_object", "content_object", "role_object",
    "tools", "tool_result", "image", "cycle", "surrogate", "text_limit", "message_limit", "byte_limit",
    "temperature_int", "temperature_nan", "temperature_inf", "temperature_negative", "temperature_large",
    "tokens_bool", "tokens_zero", "tokens_large", "streaming"])
def test_unsupported_request_shapes_deny_without_custom_conversion_equality_or_hashing(shape):
    world = World().bind()
    request = {"model": "invented_model", "messages": [{"role": "user", "content": "Invented document A"}]}
    message = request["messages"][0]
    if shape == "object":
        request = Unsupported()
    elif shape == "mapping_subclass":
        request = MappingSubclass(request)
    elif shape == "key_object":
        request[Unsupported()] = "invented"
    elif shape == "key_int":
        request[1] = "invented"
    elif shape == "model_object":
        request["model"] = Unsupported()
    elif shape == "model_subclass":
        request["model"] = type("ModelSubclass", (str,), {})("invented_model")
    elif shape == "messages_object":
        request["messages"] = Unsupported()
    elif shape == "messages_tuple":
        request["messages"] = tuple(request["messages"])
    elif shape == "message_object":
        request["messages"][0] = Unsupported()
    elif shape == "content_object":
        message["content"] = Unsupported()
    elif shape == "role_object":
        message["role"] = Unsupported()
    elif shape == "tools":
        request["tools"] = []
    elif shape == "tool_result":
        message["role"] = "tool"
    elif shape == "image":
        message["content"] = [{"type": "image_url", "url": "http://invented.invalid"}]
    elif shape == "cycle":
        message["content"] = request
    elif shape == "surrogate":
        message["content"] = "\ud800"
    elif shape == "text_limit":
        message["content"] = "A" * 16_385
    elif shape == "message_limit":
        request["messages"] = [dict(message) for _ in range(33)]
    elif shape == "byte_limit":
        request["messages"] = [{"role": "user", "content": "\x01" * 16_384} for _ in range(3)]
    elif shape.startswith("temperature_"):
        request["temperature"] = {"temperature_int": 1, "temperature_nan": float("nan"),
            "temperature_inf": float("inf"), "temperature_negative": -0.1,
            "temperature_large": 2.1}[shape]
    elif shape.startswith("tokens_"):
        request["max_tokens"] = {"tokens_bool": True, "tokens_zero": 0, "tokens_large": 4097}[shape]
    else:
        request["stream"] = True
    Unsupported.calls.clear()
    denied(world, lambda: world.call(request))
    assert Unsupported.calls == []


@pytest.mark.parametrize("shape", ["object", "bytes_subclass", "oversize", "utf8", "duplicate_keys",
    "noncanonical", "nan", "deep", "nodes", "unknown", "model", "choices", "tools", "role",
    "tokens_bool", "tokens_sum", "created_float", "content", "surrogate", "finish_reason"])
def test_unsupported_result_shapes_reject_release_after_one_authorized_send(shape):
    world = World().bind()
    value = json.loads(reply())
    payload = None
    if shape == "object":
        payload = Unsupported()
    elif shape == "bytes_subclass":
        payload = BytesSubclass(reply())
    elif shape == "oversize":
        payload = b"x" * 65_537
    elif shape == "utf8":
        payload = b"\xff"
    elif shape == "duplicate_keys":
        payload = b'{"id":"invented_reply","id":"other"}'
    elif shape == "noncanonical":
        payload = reply() + b"\n"
    elif shape == "nan":
        payload = b'{"created":NaN}'
    elif shape == "deep":
        payload = b"[" * 9 + b"0" + b"]" * 9
    elif shape == "nodes":
        payload = canonical([0] * 2049)
    elif shape == "unknown":
        value["other"] = "Invented"
    elif shape == "model":
        value["model"] = "other_model"
    elif shape == "choices":
        value["choices"].append(value["choices"][0])
    elif shape == "tools":
        value["choices"][0]["message"]["tool_calls"] = []
    elif shape == "role":
        value["choices"][0]["message"]["role"] = "user"
    elif shape == "tokens_bool":
        value["usage"]["prompt_tokens"] = True
    elif shape == "tokens_sum":
        value["usage"]["total_tokens"] = 6
    elif shape == "created_float":
        value["created"] = 12.0
    elif shape == "content":
        value["choices"][0]["message"]["content"] = {}
    elif shape == "surrogate":
        value["choices"][0]["message"]["content"] = "\ud800"
    else:
        value["choices"][0]["finish_reason"] = "length"
    if payload is None:
        payload = canonical(value)
    world.ports["main"].output = payload
    Unsupported.calls.clear()
    denied(world, world.call, sends=1)
    assert world.adapter.invocations == () and Unsupported.calls == []


def test_supported_options_and_unicode_preserve_complete_bytes():
    world = World().bind()
    world.ports["main"].output = reply("Invented reply \u2603")
    request = {"model": "invented_model", "messages": [
        {"role": "system", "content": "Invented specialist"},
        {"role": "user", "content": "Invented document \u2603"}],
        "max_tokens": 4096, "temperature": 0.0, "stream": False}
    response = world.call(request)
    assert response.choices[0].message.content == "Invented reply \u2603"
    payload = world.ports["main"].sent[0].payload
    assert b'"max_tokens":4096' in payload and b'"temperature":0.0' in payload
    assert b'"stream":false' in payload and b"\\u2603" in payload
    assert json.loads(payload)["request"] == request


@pytest.mark.parametrize("change", ["initial_assistant", "repeat_user", "changed_history", "forged_answer"])
def test_unqualified_or_changed_history_never_becomes_a_result_parent(change):
    world = World().bind()
    if change in ("initial_assistant", "repeat_user"):
        role = "assistant" if change == "initial_assistant" else "user"
        request = {"model": "invented_model", "messages": [
            {"role": role, "content": "Invented unqualified text"},
            {"role": "user", "content": "Invented document A"}]}
        denied(world, lambda: world.call(request))
    else:
        world.call()
        request = next_request(world)
        request["messages"][0 if change == "changed_history" else 1]["content"] = "Invented replacement"
        denied(world, lambda: world.call(request), sends=1)


def test_shortened_profile_required_paths_cannot_qualify_main():
    world = World()
    world.profiles = (replace(world.profiles[0], required_paths=("main",)),)
    with pytest.raises(NativeContentPolicyDenied):
        world.bind(qualify=False)
    assert world.context._denied and sum(len(p.sent) for p in world.ports.values()) == 0


@pytest.mark.parametrize("port", ["admission", "selection", "producer", "resolve"])
def test_changes_inside_pre_send_trusted_callbacks_make_zero_sends(port):
    world = World().bind()
    changed = False
    def change():
        nonlocal changed
        if not changed:
            changed = True
            world.binding = replace(world.binding, auth_epoch=8)
    if port == "resolve":
        world.ports["main"].resolve_hook = change
    elif port == "admission":
        world.admission_hook = change
    elif port == "selection":
        world.selection_hook = change
    else:
        world.producer_hook = change
    denied(world, world.call)
    assert changed


def test_ordinary_provider_without_controlled_adapter_preserves_native_sdk_behavior():
    world = World()
    world.binding = replace(world.binding, profiles=())
    world.agent.provider = "custom"
    output = object()
    calls = []
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: calls.append(kwargs) or output)))
    request = {"model": "invented_model", "messages": [{"role": "user", "content": "Invented ordinary document"}]}
    result = _dispatch_nonstreaming_api_request(world.agent, request, make_client=lambda *_: client)
    assert result is output and calls == [request]


@pytest.mark.parametrize("field", ["binding", "destination", "route_path", "producer_profiles"])
def test_unsupported_trusted_fact_shapes_deny_before_custom_operations(field):
    world = World()
    if field == "binding":
        world.binding = replace(world.binding, owner_ref=Unsupported())
    elif field == "destination":
        world.selection = replace(world.selection, destination=Unsupported())
    elif field == "route_path":
        world.routes = (replace(world.routes[0], path=Unsupported()), *world.routes[1:])
    else:
        world.policy = replace(world.policy, profiles=(Unsupported(),))
    Unsupported.calls.clear()
    with pytest.raises(NativeContentPolicyDenied):
        world.bind(qualify=False)
    assert Unsupported.calls == [] and world.context._denied
    assert sum(len(p.sent) for p in world.ports.values()) == 0


def test_missing_producer_policy_denies_before_first_send():
    world = World()
    world.policy = None
    with pytest.raises(NativeContentPolicyDenied):
        world.bind()
    assert world.context._denied and sum(len(p.sent) for p in world.ports.values()) == 0


def test_callback_failure_counts_authorized_send_without_releasing_result():
    world = World().bind()
    def fail(envelope):
        raise RuntimeError("Invented callback failure")
    world.ports["main"].hook = fail
    denied(world, world.call, sends=1)
    assert world.adapter.invocations == ()


@pytest.mark.parametrize("entrypoint", ["async_call_llm", "_async_call_llm_impl"])
@pytest.mark.asyncio
async def test_async_auxiliary_refuses_after_scope_entry_and_profile_drop(entrypoint):
    from agent import auxiliary_client
    world = World().bind()
    world.call()
    world.binding = replace(world.binding, profiles=())
    entered = False
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            entered = True
            await getattr(auxiliary_client, entrypoint)(messages=[])
            pytest.fail("Async auxiliary function returned")
    assert entered and world.context._denied
    assert world.adapter.send_count == 1 and world.sdk_calls == []
