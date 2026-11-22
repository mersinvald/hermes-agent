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


@pytest.mark.parametrize("kind", ["copy", "foreign_active_scope"])
def test_rejected_context_attachment_invalidates_original_adapter(kind):
    world = World().bind()
    if kind == "copy":
        copied = copy.copy(world.context)
        world.agent._native_content_policy_context = copied
        denied(world, world.call)
        world.agent._native_content_policy_context = world.context
    else:
        entered = False
        with pytest.raises(NativeContentPolicyDenied):
            with content_policy_scope(world.context):
                entered = True
                world.agent._native_content_policy_context = NativeContentContext(world.admission, world.request)
                world.call()
                pytest.fail("Foreign context returned")
        assert entered
        world.agent._native_content_policy_context = world.context
    denied(world, world.call)


@pytest.fixture
def caller_boundary(monkeypatch):
    from agent import chat_completion_helpers as helpers
    selector = helpers.should_use_direct_api_call

    def forbidden(*args, **kwargs):
        pytest.fail("Controlled caller reached scheduling or lifecycle machinery")

    for name in ("should_use_direct_api_call", "direct_api_call", "_context_thread_target",
                 "_check_stale_giveup", "_resolve_direct_stale_timeout",
                 "_derive_stream_stale_timeout", "_inline_nonstream_hard_timeout",
                 "_reset_stale_streak", "_join_worker_for_relay_teardown"):
        monkeypatch.setattr(helpers, name, forbidden)
    monkeypatch.setattr(helpers, "threading", SimpleNamespace(
        Thread=forbidden, Timer=forbidden, Lock=forbidden, Event=forbidden,
        get_ident=threading.get_ident,
    ))
    dispatch = helpers._dispatch_nonstreaming_api_request
    dispatches = []

    def observed(agent, request, *, make_client):
        dispatches.append((threading.get_ident(), agent, request))
        return dispatch(agent, request, make_client=make_client)

    monkeypatch.setattr(helpers, "_dispatch_nonstreaming_api_request", observed)

    def configure(world):
        for name in ("_touch_activity", "_create_request_openai_client",
                     "_create_request_anthropic_client", "_abort_request_openai_client",
                     "_abort_request_anthropic_client", "_close_request_openai_client",
                     "_close_request_anthropic_client"):
            setattr(world.agent, name, forbidden)
        return world

    def call(world, request=None):
        if request is None:
            request = {"model": "invented_model", "messages": [
                {"role": "user", "content": "Invented document A"}]}
        return helpers.interruptible_api_call(world.agent, request)

    return SimpleNamespace(configure=configure, call=call, dispatches=dispatches, selector=selector)


@pytest.mark.parametrize("platform,direct_setting", [("cli", False), ("cron", True)])
def test_controlled_caller_complete_bytes_identity_and_bound_thread(caller_boundary, platform, direct_setting):
    world = caller_boundary.configure(World().bind())
    world.agent.platform = platform
    assert caller_boundary.selector(world.agent) is direct_setting
    owner_thread = threading.get_ident()
    send_threads = []
    world.ports["main"].hook = lambda envelope: send_threads.append(threading.get_ident())
    request = {"model": "invented_model", "messages": [
        {"role": "user", "content": "Invented document A"}]}
    with content_policy_scope(world.context):
        result = caller_boundary.call(world, request)
    record, = world.adapter.invocations
    expected = b'{"api_mode":"chat_completions","destination":{"endpoint_ref":"invented_endpoint","model_ref":"invented_model","provider_ref":"synthetic_native","revision":1,"transport_ref":"invented_main"},"request":{"messages":[{"content":"Invented document A","role":"user"}],"model":"invented_model"}}'
    expected_result = b'{"choices":[{"finish_reason":"stop","index":0,"message":{"content":"Invented answer A","role":"assistant"}}],"created":12,"id":"invented_reply","model":"invented_model","object":"chat.completion","usage":{"completion_tokens":2,"prompt_tokens":3,"total_tokens":5}}'
    assert record.request_bytes == expected == world.ports["main"].sent[0].payload
    assert record.response_bytes == expected_result
    assert result is record.native_response
    assert record.envelope is world.ports["main"].sent[0]
    assert send_threads == [owner_thread]
    assert caller_boundary.dispatches == [(owner_thread, world.agent, request)]
    assert world.adapter.send_count == 1 and world.sdk_calls == []
    assert not hasattr(world.agent, "_active_request_abort")
    assert len(world.qualification_records) == len(PATHS)


def caller_denied(world, boundary, request=None, *, sends=0, records=0, reason=None):
    with pytest.raises(NativeContentPolicyDenied) as caught:
        boundary.call(world, request)
    if reason is not None:
        assert str(caught.value) == reason
    assert world.context._denied
    assert world.adapter.send_count == sends == len(world.ports["main"].sent)
    assert len(world.adapter.invocations) == records
    with pytest.raises(NativeContentPolicyDenied):
        boundary.call(world, request)
    assert world.adapter.send_count == sends and world.sdk_calls == []


def test_controlled_caller_active_scope_without_agent_attachment(caller_boundary):
    world = caller_boundary.configure(World().bind())
    world.agent._native_content_policy_context = None
    world.agent._native_content_policy_required = False
    with content_policy_scope(world.context):
        result = caller_boundary.call(world)
    assert world.adapter.invocations[0].native_response is result
    assert world.adapter.send_count == 1 and len(caller_boundary.dispatches) == 1


def test_controlled_caller_reuse_keeps_every_parent_after_profile_drop(caller_boundary, monkeypatch):
    world = caller_boundary.configure(World().bind())
    parents = []
    derive = ContentPolicyRuntime.derive

    def observed(runtime, handle, prior, payload, *, path):
        parents.append(prior)
        return derive(runtime, handle, prior, payload, path=path)

    monkeypatch.setattr(ContentPolicyRuntime, "derive", observed)
    first_response = caller_boundary.call(world)
    world.binding = replace(world.binding, profiles=())
    with content_policy_scope(world.context):
        second_response = caller_boundary.call(world, next_request(world))
        third_response = caller_boundary.call(world, next_request(world, "Invented document C"))
    first, second, third = world.adapter.invocations
    assert first.native_response is first_response
    assert second.native_response is second_response
    assert third.native_response is third_response
    assert len(parents) == 2 and len(parents[0]) == 1 and len(parents[1]) == 2
    assert parents[0][0] is parents[1][0] is first.result_content
    assert parents[1][1] is second.result_content
    assert all(envelope.profiles == ("restricted_documents",) for envelope in world.ports["main"].sent)
    assert third.request_bytes == world.ports["main"].sent[2].payload
    assert world.adapter.send_count == 3 and len(caller_boundary.dispatches) == 3
    assert {call[0] for call in caller_boundary.dispatches} == {threading.get_ident()}


@pytest.mark.parametrize("boundary", ["before", "send"])
@pytest.mark.parametrize("fact", ["binding", "selection", "destination", "producer", "request", "mode", "provider"])
def test_controlled_caller_current_fact_mismatch_denies(caller_boundary, fact, boundary):
    world = caller_boundary.configure(World().bind())
    request = {"model": "invented_model", "messages": [
        {"role": "user", "content": "Invented document A"}]}

    def change(*args):
        if fact == "binding":
            world.binding = replace(world.binding, auth_epoch=8)
        elif fact == "selection":
            world.selection = replace(world.selection, selection_version=4)
        elif fact == "destination":
            world.ports["main"].destination = replace(world.selection.destination, revision=2)
        elif fact == "producer":
            world.policy = replace(world.policy, revision=5)
        elif fact == "request":
            request["model"] = "other_model"
        elif fact == "mode":
            world.agent.api_mode = "codex_responses"
        else:
            world.agent.provider = "custom"

    if boundary == "before":
        change()
    else:
        world.ports["main"].hook = change
    caller_denied(world, caller_boundary, request, sends=int(boundary == "send"),
                  reason="controlled_native_payload_denied")


@pytest.mark.parametrize("policy", [None, NativeResultPolicy("invented_producer", 5, ("restricted_documents",))])
def test_controlled_caller_changed_producer_policy_denies_reuse(caller_boundary, policy):
    world = caller_boundary.configure(World().bind())
    caller_boundary.call(world)
    request = next_request(world)
    world.binding = replace(world.binding, profiles=())
    world.policy = policy
    caller_denied(world, caller_boundary, request, sends=1, records=1,
                  reason="controlled_native_payload_denied")


@pytest.mark.parametrize("retained", ["response", "result_handle"])
def test_controlled_caller_original_retained_identities_govern_reuse(caller_boundary, retained):
    world = caller_boundary.configure(World().bind())
    result = caller_boundary.call(world)
    record, = world.adapter.invocations
    request = next_request(world)
    assert record.native_response is result
    if retained == "response":
        replacement = replace(record, native_response=copy.copy(result))
    else:
        replacement = replace(record, result_content=copy.copy(record.result_content))
    world.adapter._records = (replacement,)
    caller_denied(world, caller_boundary, request, sends=1, records=1,
                  reason="controlled_native_payload_denied")


@pytest.mark.parametrize("authority", ["invalid", "stale", "detached", "copied_adapter", "foreign_adapter", "copied_context"])
def test_controlled_caller_invalid_authority_never_reaches_lifecycle(caller_boundary, authority):
    world = caller_boundary.configure(World().bind())
    if authority == "invalid":
        world.context.invalidate()
    elif authority == "stale":
        world.now = 100
    elif authority == "detached":
        world.context._controlled_main_payload = None
    elif authority == "copied_adapter":
        world.context._controlled_main_payload = copy.copy(world.adapter)
    elif authority == "foreign_adapter":
        world.context._controlled_main_payload = World().bind().adapter
    else:
        world.agent._native_content_policy_context = copy.copy(world.context)
    caller_denied(world, caller_boundary)
    world.agent._native_content_policy_context = world.context
    with pytest.raises(NativeContentPolicyDenied):
        caller_boundary.call(world)
    assert world.adapter.send_count == 0


def test_controlled_caller_foreign_active_context_invalidates_original(caller_boundary):
    world = caller_boundary.configure(World().bind())
    with pytest.raises(NativeContentPolicyDenied, match="^content_policy_context_missing$"):
        with content_policy_scope(world.context):
            world.agent._native_content_policy_context = NativeContentContext(world.admission, world.request)
            caller_boundary.call(world)
    assert world.context._denied and world.adapter.send_count == 0
    world.agent._native_content_policy_context = world.context
    caller_denied(world, caller_boundary)


@pytest.mark.parametrize("kind", ["absent", "ordinary", "required"])
def test_controlled_caller_reserved_marker_without_adapter_denies(caller_boundary, kind):
    world = caller_boundary.configure(World())
    if kind == "ordinary":
        world.binding = replace(world.binding, profiles=())
    else:
        world.agent._native_content_policy_context = None
        world.agent._native_content_policy_required = kind == "required"
    reason = "content_policy_context_missing" if kind == "required" else "controlled_native_adapter_missing"
    with pytest.raises(NativeContentPolicyDenied, match=f"^{reason}$"):
        caller_boundary.call(world)
    assert world.sdk_calls == [] and not world.ports["main"].sent
    if kind == "ordinary":
        assert world.context._denied


def test_controlled_caller_wrong_thread_denies_without_moving_authority(caller_boundary):
    world = caller_boundary.configure(World().bind())
    errors = []

    def run():
        try:
            caller_boundary.call(world)
        except NativeContentPolicyDenied as exc:
            errors.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    worker.join(5)
    assert not worker.is_alive() and len(errors) == 1
    assert str(errors[0]) == "controlled_native_payload_denied"
    caller_denied(world, caller_boundary)


@pytest.mark.parametrize("mode,provider", [
    ("anthropic_messages", "anthropic"), ("bedrock_converse", "bedrock"),
    ("chat_completions", "moa"), ("unknown", "custom"),
])
def test_controlled_caller_unsupported_modes_stay_refused(caller_boundary, mode, provider):
    world = caller_boundary.configure(World().bind())
    world.agent.api_mode, world.agent.provider = mode, provider
    caller_denied(world, caller_boundary, reason="controlled_native_payload_denied")


@pytest.mark.parametrize("output", [None, b"\xff", b'{"id":"A","id":"B"}', b"[]"],
                         ids=["not_bytes", "utf8", "duplicate_keys", "wrong_shape"])
def test_controlled_caller_malformed_result_keeps_send_count(caller_boundary, output):
    world = caller_boundary.configure(World().bind())
    world.ports["main"].output = output
    caller_denied(world, caller_boundary, sends=1, reason="controlled_native_payload_denied")


@pytest.mark.parametrize("boundary", ["resolve", "send"])
def test_controlled_caller_callback_failure_keeps_fixed_policy_reason(caller_boundary, boundary):
    world = caller_boundary.configure(World().bind())

    def fail(*args):
        raise RuntimeError("Invented callback failure")

    if boundary == "resolve":
        world.ports["main"].resolve_hook = fail
    else:
        world.ports["main"].hook = fail
    caller_denied(world, caller_boundary, sends=int(boundary == "send"),
                  reason="controlled_native_payload_denied")


def test_controlled_caller_cancellation_before_dispatch_invalidates(caller_boundary):
    world = caller_boundary.configure(World().bind())
    world.agent._interrupt_requested = True
    with pytest.raises(InterruptedError, match="^Agent interrupted during API call$"):
        caller_boundary.call(world)
    assert world.context._denied and world.adapter._invalid
    assert world.adapter.send_count == 0 and world.adapter.invocations == ()
    assert caller_boundary.dispatches == [] and not world.ports["main"].sent
    world.agent._interrupt_requested = False
    caller_denied(world, caller_boundary)


def test_controlled_caller_cancellation_during_callback_rejects_caller_release(caller_boundary):
    world = caller_boundary.configure(World().bind())
    world.agent._interrupt_requested = False

    def cancel(envelope):
        world.agent._interrupt_requested = True

    world.ports["main"].hook = cancel
    released = []
    with pytest.raises(InterruptedError, match="^Agent interrupted during API call$"):
        released.append(caller_boundary.call(world))
    assert released == [] and world.context._denied and world.adapter._invalid
    record, = world.adapter.invocations
    assert record.response_bytes == reply()
    assert record.envelope is world.ports["main"].sent[0]
    assert world.adapter.send_count == 1 and len(caller_boundary.dispatches) == 1
    world.agent._interrupt_requested = False
    caller_denied(world, caller_boundary, sends=1, records=1)
    assert world.adapter.invocations[0] is record


class UnsupportedInterrupt(Unsupported):
    def __bool__(self):
        self.calls.append("bool")
        raise AssertionError("Boolean conversion ran")

    def __int__(self):
        self.calls.append("int")
        raise AssertionError("Integer conversion ran")


@pytest.mark.parametrize("boundary", ["before", "send"])
@pytest.mark.parametrize("value", [None, 0, 1, "false", (), {}, UnsupportedInterrupt()],
                         ids=["none", "zero", "one", "string", "tuple", "mapping", "custom"])
def test_controlled_caller_unsupported_interrupt_flags_deny_without_conversion(caller_boundary, value, boundary):
    world = caller_boundary.configure(World().bind())
    Unsupported.calls.clear()
    world.agent._interrupt_requested = False
    if boundary == "before":
        world.agent._interrupt_requested = value
    else:
        world.ports["main"].hook = lambda envelope: setattr(world.agent, "_interrupt_requested", value)
    count = int(boundary == "send")
    caller_denied(world, caller_boundary, sends=count, records=count,
                  reason="controlled_native_interrupt_invalid")
    assert Unsupported.calls == []
    world.agent._interrupt_requested = False
    caller_denied(world, caller_boundary, sends=count, records=count)


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_controlled_caller_process_control_exceptions_propagate_and_invalidate(caller_boundary, exception):
    world = caller_boundary.configure(World().bind())
    error = exception("Invented process control")

    def fail(envelope):
        raise error

    world.ports["main"].hook = fail
    with pytest.raises(exception) as caught:
        caller_boundary.call(world)
    assert caught.value is error
    assert world.context._denied and world.adapter._invalid
    assert world.adapter.send_count == 1 and world.adapter.invocations == ()
    caller_denied(world, caller_boundary, sends=1)


@pytest.mark.parametrize("profiles_dropped", [False, True])
@pytest.mark.parametrize("operation", ["loop", "forwarder", "prologue", "title_prologue"])
def test_controlled_caller_startup_refuses_after_valid_scope_entry(caller_boundary, operation, profiles_dropped):
    from agent.conversation_loop import run_conversation
    from agent.turn_context import build_turn_context, _maybe_title_session_at_turn_start
    from run_agent import AIAgent

    world = caller_boundary.configure(World().bind())
    caller_boundary.call(world)
    if profiles_dropped:
        world.binding = replace(world.binding, profiles=())
    callback = world.forbidden_provider
    operations = {
        "loop": lambda: run_conversation(world.agent, "Invented input"),
        "forwarder": lambda: AIAgent.run_conversation(world.agent, "Invented input"),
        "title_prologue": lambda: _maybe_title_session_at_turn_start(world.agent, []),
        "prologue": lambda: build_turn_context(world.agent, "Invented input", None, [], "task", None, None,
            restore_or_build_system_prompt=callback, install_safe_stdio=callback,
            sanitize_surrogates=callback, summarize_user_message_for_log=callback,
            set_session_context=callback, set_current_write_origin=callback, ra=object()),
    }
    entered = False
    with pytest.raises(NativeContentPolicyDenied, match="^controlled_native_operation_unsupported$"):
        with content_policy_scope(world.context):
            entered = True
            operations[operation]()
    assert entered and world.context._denied
    assert world.adapter.send_count == 1 and world.sdk_calls == []
    assert len(world.adapter.invocations) == 1
    assert all(not port.sent for path, port in world.ports.items() if path != "main")


@pytest.mark.parametrize("operation", ["fallback", "compression", "memory", "delegate", "a2a", "tool_result",
                                      "title", "auxiliary", "auxiliary_impl", "stream"])
def test_controlled_caller_siblings_refuse_after_scope_entry_and_profile_drop(caller_boundary, operation):
    from agent import auxiliary_client, chat_completion_helpers, title_generator
    from agent.memory_manager import MemoryManager
    from plugins.platforms.a2a import tools as a2a_tools
    from run_agent import AIAgent
    from tools import delegate_tool

    world = caller_boundary.configure(World().bind())
    caller_boundary.call(world)
    world.binding = replace(world.binding, profiles=())
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
    }
    entered = False
    with pytest.raises(NativeContentPolicyDenied, match="^controlled_native_operation_unsupported$"):
        with content_policy_scope(world.context):
            entered = True
            operations[operation]()
    assert entered and world.context._denied
    assert world.adapter.send_count == 1 and world.sdk_calls == []
    assert all(not port.sent for path, port in world.ports.items() if path != "main")


@pytest.mark.parametrize("platform", ["cli", "cron"])
@pytest.mark.parametrize("attachment", ["absent", "ordinary", "missing_required"])
def test_controlled_caller_ordinary_calls_keep_selector_and_lifecycle(monkeypatch, platform, attachment):
    from agent import chat_completion_helpers as helpers

    world = World()
    world.binding = replace(world.binding, profiles=())
    agent = world.agent
    agent.provider, agent.platform = "custom", platform
    agent._interrupt_requested = False
    agent._consecutive_stale_streams = 0
    agent._native_content_policy_required = attachment != "absent"
    if attachment != "ordinary":
        agent._native_content_policy_context = None
    events = []
    responses = []
    owner_thread = threading.get_ident()
    response = object()

    def create(**kwargs):
        responses.append((threading.get_ident(), kwargs))
        return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    agent._compute_non_stream_stale_timeout = lambda kwargs: float("inf")
    agent._touch_activity = lambda text: events.append(("activity", text))
    agent._create_request_openai_client = lambda **kwargs: events.append(("client", kwargs)) or client
    agent._close_request_openai_client = lambda client, **kwargs: events.append(("close", kwargs))
    selector = helpers.should_use_direct_api_call
    selections = []

    def observed(candidate):
        selected = selector(candidate)
        selections.append(selected)
        return selected

    monkeypatch.setattr(helpers, "should_use_direct_api_call", observed)
    request = {"model": "invented_model", "messages": [
        {"role": "user", "content": "Invented ordinary document"}]}
    if attachment == "missing_required":
        with pytest.raises(NativeContentPolicyDenied, match="^content_policy_context_missing$"):
            helpers.interruptible_api_call(agent, request)
        assert responses == []
        assert all(event[0] == "activity" for event in events)
    else:
        assert helpers.interruptible_api_call(agent, request) is response
        assert len(responses) == 1 and responses[0][1] == request
        assert (responses[0][0] == owner_thread) is (platform == "cron")
        assert [event[0] for event in events].count("client") == 1
        assert [event for event in events if event[0] == "close"] == [
            ("close", {"reason": "request_complete"})]
    assert selections == [platform == "cron"]
    assert any(event[0] == "activity" for event in events)
    assert world.adapter is None and not world.ports["main"].sent


class InterruptReadAgent(SimpleNamespace):
    def __getattribute__(self, name):
        if name == "_interrupt_requested":
            return vars(self)["read_interrupt"]()
        return super().__getattribute__(name)


def change_on_interrupt_read(world, boundary, change):
    reads = []

    def read_interrupt():
        reads.append(None)
        if len(reads) == (1 if boundary == "before" else 2):
            change()
        return False

    world.agent = InterruptReadAgent(**vars(world.agent), read_interrupt=read_interrupt)
    return reads


def test_controlled_caller_correction_retained_payload_false_marker_cannot_select_moa(caller_boundary):
    world = caller_boundary.configure(World().bind())
    first_response = caller_boundary.call(world)
    record, = world.adapter.invocations
    assert first_response is record.native_response
    world.binding = replace(world.binding, profiles=())
    world.context._controlled_main_bound = False
    world.agent.provider = "moa"
    assert world.context._controlled_main_payload is world.adapter
    with pytest.raises(NativeContentPolicyDenied, match="^controlled_native_adapter_missing$"):
        caller_boundary.call(world, next_request(world))
    assert world.context._denied and world.adapter._invalid
    assert world.adapter.invocations == (record,)
    assert world.adapter.send_count == len(world.ports["main"].sent) == 1
    assert len(caller_boundary.dispatches) == 1 and world.sdk_calls == []


@pytest.mark.parametrize("boundary", ["entry", "before", "release"])
@pytest.mark.parametrize("marker", [None, 0, 1, "false", (), {}, UnsupportedInterrupt()],
                         ids=["none", "zero", "one", "string", "tuple", "mapping", "custom"])
def test_controlled_caller_correction_non_boolean_bound_markers_deny_without_conversion(
        caller_boundary, marker, boundary):
    world = caller_boundary.configure(World().bind())
    Unsupported.calls.clear()

    def change():
        world.context._controlled_main_bound = marker

    if boundary == "entry":
        change()
    else:
        change_on_interrupt_read(world, boundary, change)
    released = []
    with pytest.raises(NativeContentPolicyDenied, match="^controlled_native_adapter_missing$"):
        released.append(caller_boundary.call(world))
    count = int(boundary == "release")
    assert released == [] and Unsupported.calls == []
    assert world.context._denied and world.adapter._invalid
    assert world.adapter.send_count == len(world.ports["main"].sent) == count
    assert len(world.adapter.invocations) == len(caller_boundary.dispatches) == count
    assert world.sdk_calls == []
    if count:
        record, = world.adapter.invocations
        assert record.native_response.choices[0].message.content == "Invented answer A"
        assert record.response_bytes == reply()


@pytest.mark.parametrize("boundary", ["before", "release"])
@pytest.mark.parametrize("fact", ["context", "detached", "copied_adapter", "marker_false", "binding"])
def test_controlled_caller_correction_interrupt_read_rechecks_current_attachment(
        caller_boundary, boundary, fact):
    world = caller_boundary.configure(World().bind())
    foreign = World().bind() if fact == "context" else None

    def change():
        if fact == "context":
            world.agent._native_content_policy_context = foreign.context
        elif fact == "detached":
            world.context._controlled_main_payload = None
        elif fact == "copied_adapter":
            world.context._controlled_main_payload = copy.copy(world.adapter)
        elif fact == "marker_false":
            world.context._controlled_main_bound = False
        else:
            world.binding = replace(world.binding, auth_epoch=8)

    reads = change_on_interrupt_read(world, boundary, change)
    released = []
    with pytest.raises(NativeContentPolicyDenied) as caught:
        released.append(caller_boundary.call(world))
    assert str(caught.value) in (
        "controlled_native_adapter_missing", "controlled_native_attachment_invalid",
        "controlled_native_payload_denied", "content_policy_context_missing",
    )
    count = int(boundary == "release")
    assert released == [] and len(reads) == count + 1
    assert world.context._denied
    assert world.adapter.send_count == len(world.ports["main"].sent) == count
    assert len(world.adapter.invocations) == len(caller_boundary.dispatches) == count
    assert world.sdk_calls == []
    if foreign is not None:
        assert foreign.adapter.send_count == 0 and foreign.sdk_calls == []
    if count:
        record, = world.adapter.invocations
        assert record.envelope is world.ports["main"].sent[0]
        assert record.response_bytes == reply()
        assert record.native_response.choices[0].message.content == "Invented answer A"
