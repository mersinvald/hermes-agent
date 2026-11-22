"""Carry invented main payloads through operator-bound shared dispatch.

Only the selected synchronous native boundary can use this adapter.
The operator supplies genuine controlled ports for the complete shared policy.
"""

from __future__ import annotations

import json
import math
import re
import sys
import threading
import weakref
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from types import MethodType, SimpleNamespace
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from agent.turn_context import TurnContext
    from hermes_content_policy import (
        ContentHandle, Destination, DispatchEnvelope, ExecutionBinding,
        ProfilePolicy, RouteBinding, TransportPort,
    )

from agent.content_policy import (
    NativeContentContext, NativeContentPolicyDenied, current_context, shared_policy,
)
from agent.message_sanitization import _sanitize_surrogates as _NATIVE_SANITIZER

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z", re.ASCII)
_MAX_INT = (1 << 53) - 1
_MAX_NATIVE_RESULT = 262_144
_adapters = weakref.WeakSet()
_constructed_bindings = weakref.WeakSet()


class NativeTurnInput:
    __slots__ = ()

    def __new__(cls):
        raise TypeError("Native turn tokens require adapter admission")


class NativePreparedTurn:
    __slots__ = ()

    def __new__(cls):
        raise TypeError("Native turn tokens require adapter preparation")


@dataclass(frozen=True, slots=True)
class _TurnFrame:
    token: NativeTurnInput
    arguments: tuple
    raw_bytes: bytes
    admitted: object
    input_content: object
    options: bytes
    input_anchors: tuple
    output: object = None
    output_anchors: tuple = ()
    output_bytes: bytes | None = None
    output_content: object = None
    prepared: NativePreparedTurn | None = None
    request_bytes: bytes | None = None
    request_contents: tuple = ()


def _history(value):
    if value is None:
        return
    _require(type(value) is list and len(value) <= 31)
    for row in value:
        _keys(row, ("role", "content"))
        _require(type(row["role"]) is str and row["role"] in ("user", "assistant"))
        _text(row["content"])


def _turn_shape(arguments):
    _require(type(arguments) is tuple and len(arguments) == 4)
    user, system, history, task = arguments
    _require(type(user) is str and len(user) <= 16_384)
    if system is not None:
        _text(system)
    _history(history)
    _identifier(task)


def _turn_bytes(arguments, options):
    _turn_shape(arguments)
    user, system, history, task = arguments
    payload = _json(dict(user_message=user, system_message=system,
                         conversation_history=history, task_id=task,
                         options=json.loads(options)))
    _require(len(payload) <= 262_144)
    return payload


def _native_state(agent, system):
    namespace = object.__getattribute__(agent, "__dict__")
    _require(type(namespace) is dict)
    for name, cls in (("enabled_toolsets", list), ("tools", list), ("valid_tool_names", set)):
        _require(type(namespace.get(name)) is cls and len(namespace[name]) == 0)
    for name, expected in (("compression_enabled", False), ("_persist_disabled", True),
                           ("skip_background_review", True)):
        _require(namespace.get(name) is expected)
    for name in ("_memory_store", "_memory_manager", "_session_db"):
        _require(name in namespace and namespace[name] is None)
    snapshot = {}
    empty = {
        "_fallback_chain": list, "prefill_messages": list,
        "request_overrides": dict, "_session_messages": list,
        "_plugin_user_context": str, "_ext_prefetch_cache": str,
    }
    absent = (
        "_fallback_model", "fallback_model", "_active_fallback", "_primary_runtime",
        "_stream_callback", "ephemeral_system_prompt", "_native_command_context",
        "_pending_cli_user_message", "_pending_redirect", "_compression_warning",
        "_persist_user_message_override", "_persist_user_message_timestamp",
        "_persist_user_display_kind", "_persist_user_display_metadata",
        "_relay_pending_turn_id", "_media_context", "_external_history",
        "reasoning_config", "extra_body", "extra_headers", "_request_overrides",
        "_active_fallback_provider", "_fallback_original_model", "_moa_runtime",
    )
    for name in (*empty, *absent, "_cached_system_prompt", "stream", "skip_memory",
                 "_fallback_activated", "_fallback_index"):
        present = name in namespace
        value = namespace.get(name)
        if name in empty:
            _require(value is None or (type(value) is empty[name] and len(value) == 0))
        elif name == "_cached_system_prompt":
            _require(value is None or (type(value) is str and value in ("", system)))
        elif name == "ephemeral_system_prompt":
            _require(value is None or (type(value) is str and value == ""))
        elif name == "stream":
            _require(not present or value is False)
        elif name == "skip_memory":
            _require(not present or value is True)
        elif name == "_fallback_activated":
            _require(not present or value is False)
        elif name == "_fallback_index":
            _require(not present or (type(value) is int and value == 0))
        else:
            _require(value is None)
        snapshot[name] = {"present": present, "value": value}
    options = {}
    for name in ("model", "max_tokens", "temperature"):
        present, value = name in namespace, namespace.get(name)
        if name == "model":
            _identifier(value)
        elif value is not None:
            if name == "max_tokens":
                _integer(value, 4096)
            else:
                _require(type(value) is float and math.isfinite(value) and 0 <= value <= 2)
        options[name] = {"present": present, "value": value}
    options["stream"] = {"present": "stream" in namespace, "value": False}
    snapshot["options"] = options
    return _json(snapshot), _json(options)


def _output_bytes(output):
    from agent.turn_context import TurnContext
    _require(type(output) is TurnContext)
    _require(type(vars(output)) is dict)
    _require(set(vars(output)) == {field.name for field in fields(TurnContext)})
    _text(output.user_message)
    _text(output.original_user_message)
    _history(output.messages)
    _history(output.conversation_history)
    if output.active_system_prompt is not None:
        _text(output.active_system_prompt)
    _identifier(output.effective_task_id)
    _identifier(output.turn_id)
    _integer(output.current_turn_user_idx, 31, 0)
    _require(output.should_review_memory is False)
    _require(type(output.plugin_user_context) is str and output.plugin_user_context == "")
    _require(type(output.ext_prefetch_cache) is str and output.ext_prefetch_cache == "")
    _require(output.preflight_compression_blocked is False)
    payload = _json({f.name: getattr(output, f.name) for f in fields(TurnContext)
                     if f.name != "controlled_prepared"})
    _require(len(payload) <= 262_144)
    return payload


def _output_anchors(output):
    return (output.messages, *output.messages, output.conversation_history,
            *(output.conversation_history or ()))


def _input_anchors(arguments):
    return (*arguments, *(arguments[2] or ()))


@dataclass(frozen=True, slots=True)
class NativeMainSelection:
    selection_id: str
    selection_version: int
    api_mode: str
    destination: Destination


@dataclass(frozen=True, slots=True)
class NativeResultPolicy:
    producer_ref: str
    revision: int
    profiles: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NativeMainMessage:
    role: str
    content: str
    tool_calls: None = None


@dataclass(frozen=True, slots=True)
class NativeMainChoice:
    index: int
    message: NativeMainMessage
    finish_reason: str


@dataclass(frozen=True, slots=True)
class NativeMainUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class NativeMainResponse:
    id: str
    object: str
    created: int
    model: str
    choices: tuple[NativeMainChoice, ...]
    usage: NativeMainUsage


@dataclass(frozen=True, slots=True)
class NativeMainInvocation:
    request_handle: object
    source: ExecutionBinding
    selection: NativeMainSelection
    producer_policy: NativeResultPolicy
    request_bytes: bytes
    request_content: ContentHandle
    envelope: DispatchEnvelope
    observed_destination: Destination
    response_bytes: bytes
    result_content: ContentHandle
    native_response: NativeMainResponse


class NativeSelectionPort(Protocol):
    def current(self, request_handle: object) -> NativeMainSelection: ...


class NativeResultPolicyPort(Protocol):
    def current(self, producer_ref: str) -> NativeResultPolicy: ...


def _require(condition):
    if not condition:
        raise NativeContentPolicyDenied("controlled_native_payload_invalid")


def _identifier(value):
    _require(type(value) is str and 1 <= len(value) <= 128)
    _require(_ID.fullmatch(value) is not None)


def _integer(value, maximum=_MAX_INT, minimum=1):
    _require(type(value) is int and minimum <= value <= maximum)


def _profiles(value):
    _require(type(value) is tuple and len(value) <= 32)
    for item in value:
        _identifier(item)
    _require(len(set(value)) == len(value))


def _row(value, cls):
    _require(type(value) is cls)
    return tuple(object.__getattribute__(value, f.name) for f in fields(cls))


def _binding(value):
    row = _row(value, shared_policy().ExecutionBinding)
    for i in (0, 1, 2, 3, 6):
        _identifier(row[i])
    for i in (4, 5, 7):
        _integer(row[i])
    _profiles(row[8])
    return row


def _destination(value):
    row = _row(value, shared_policy().Destination)
    for item in row[:4]:
        _identifier(item)
    _integer(row[4])
    return row


def _selection(value):
    row = _row(value, NativeMainSelection)
    _identifier(row[0])
    _integer(row[1])
    _require(type(row[2]) is str and row[2] == "chat_completions")
    return (*row[:3], _destination(row[3]))


def _producer(value):
    row = _row(value, NativeResultPolicy)
    _identifier(row[0])
    _integer(row[1])
    _profiles(row[2])
    return row


def _keys(value, required, optional=()):
    _require(type(value) is dict and len(value) <= len(required) + len(optional))
    for key in value:
        _require(type(key) is str and len(key) <= 32)
    _require(set(required) <= value.keys() <= set((*required, *optional)))


def _text(value):
    _require(type(value) is str and len(value) <= 16_384)
    _require(not any(0xD800 <= ord(c) <= 0xDFFF for c in value))
    _require(len(value.encode("utf-8")) <= 16_384)


def _request(value):
    _keys(value, ("model", "messages"), ("max_tokens", "temperature", "stream"))
    _identifier(value["model"])
    if "max_tokens" in value:
        _integer(value["max_tokens"], 4096)
    if "temperature" in value:
        temperature = value["temperature"]
        _require(type(temperature) is float and math.isfinite(temperature))
        _require(0 <= temperature <= 2)
    if "stream" in value:
        _require(value["stream"] is False)
    messages = value["messages"]
    _require(type(messages) is list and 1 <= len(messages) <= 32)
    for message in messages:
        _keys(message, ("role", "content"))
        _require(type(message["role"]) is str)
        _require(message["role"] in ("system", "user", "assistant"))
        _text(message["content"])


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _response_shape(value, model):
    _keys(value, ("id", "object", "created", "model", "choices", "usage"))
    _identifier(value["id"])
    _identifier(value["model"])
    _require(type(value["object"]) is str and value["object"] == "chat.completion")
    _integer(value["created"], minimum=0)
    choices = value["choices"]
    _require(type(choices) is list and len(choices) == 1)
    choice = choices[0]
    _keys(choice, ("index", "message", "finish_reason"))
    _integer(choice["index"], 0, 0)
    _require(type(choice["finish_reason"]) is str and choice["finish_reason"] == "stop")
    message = choice["message"]
    _keys(message, ("role", "content"))
    _require(type(message["role"]) is str and message["role"] == "assistant")
    _text(message["content"])
    usage = value["usage"]
    _keys(usage, ("prompt_tokens", "completion_tokens", "total_tokens"))
    for number in usage.values():
        _integer(number, 1_000_000, 0)
    _require(usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"])
    _require(value["model"] == model)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result)
        result[key] = value
    return result


def _decode_response(payload, model):
    _require(type(payload) is bytes and len(payload) <= 65_536)
    text = payload.decode("utf-8", errors="strict")
    quoted = escaped = False
    depth = 0
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            _require(depth <= 8)
        elif char in "]}":
            depth -= 1
            _require(depth >= 0)
    _require(depth == 0 and not quoted)
    value = json.loads(text, object_pairs_hook=_pairs,
                       parse_constant=lambda _: _require(False))
    pending = [value]
    nodes = 0
    while pending:
        node = pending.pop()
        nodes += 1
        _require(nodes <= 2048)
        if type(node) is dict:
            pending.extend(node.values())
            nodes += len(node)
        elif type(node) is list:
            pending.extend(node)
    _response_shape(value, model)
    _require(_json(value) == payload)
    return value


def _response(value):
    choice = value["choices"][0]
    return NativeMainResponse(
        value["id"], value["object"], value["created"], value["model"],
        (NativeMainChoice(0, NativeMainMessage(**choice["message"]), "stop"),),
        NativeMainUsage(**value["usage"]),
    )


def _response_bytes(response, model):
    values = _row(response, NativeMainResponse)
    _require(type(values[4]) is tuple and len(values[4]) == 1)
    choice = _row(values[4][0], NativeMainChoice)
    message = _row(choice[1], NativeMainMessage)
    _require(message[2] is None)
    usage = _row(values[5], NativeMainUsage)
    value = dict(zip(("id", "object", "created", "model"), values[:4]))
    value["choices"] = [{"index": choice[0], "finish_reason": choice[2],
                         "message": {"role": message[0], "content": message[1]}}]
    value["usage"] = dict(zip(("prompt_tokens", "completion_tokens", "total_tokens"), usage))
    _response_shape(value, model)
    return _json(value)


def _method(callback):
    if type(callback) is MethodType:
        return callback.__self__, callback.__func__
    return (callback,)


def _same_method(current, captured):
    actual = _method(current)
    return len(actual) == len(captured) and all(a is b for a, b in zip(actual, captured))


class _Admission:
    def __init__(self, adapter):
        self.adapter = adapter

    def current(self, handle):
        adapter = self.adapter
        _require(handle is adapter._handle)
        return adapter._around(lambda: adapter._admission.current(handle))


class _Transport:
    def __init__(self, adapter, port, *, main):
        self.adapter, self.port, self.main = adapter, port, main

    def resolve(self):
        actual = self.adapter._around(self.port.resolve)
        _destination(actual)
        return actual

    def send(self, envelope):
        adapter = self.adapter
        _require(self.main and adapter._pending is None)
        _require(type(envelope) is shared_policy().DispatchEnvelope)
        _binding(envelope.source)
        _destination(envelope.destination)
        _profiles(envelope.profiles)
        _require(type(envelope.payload) is bytes and len(envelope.payload) <= 262_144)
        _require(type(envelope.path) is str)
        _identifier(envelope.recipient_ref)
        adapter._checkpoint()
        _require(envelope.payload == adapter._request_bytes and envelope.path == "main")
        _require(envelope.source == adapter._source)
        _require(envelope.destination == adapter._selected.destination)
        _require(envelope.recipient_ref == adapter._source.consumer_ref)
        actual = self.resolve()
        _require(actual == envelope.destination)
        adapter._pending = (envelope, actual, None)
        adapter._send_count += 1
        output = adapter._around(lambda: self.port.send(envelope))
        _require(type(output) is bytes and len(output) <= 65_536)
        adapter._pending = (envelope, actual, output)
        _require(self.resolve() == actual)
        adapter._checkpoint()
        return output


class NativeMainPayloadAdapter:
    """Retain complete bytes and accepted handles for one controlled context."""

    def __init__(
        self, context: NativeContentContext, *, selection: NativeSelectionPort,
        result_policy: NativeResultPolicyPort, producer_ref: str, main_route_ref: str,
        profiles: tuple[ProfilePolicy, ...], routes: tuple[RouteBinding, ...],
        transports: tuple[tuple[str, TransportPort], ...], clock: Callable[[], int],
    ) -> None:
        _require(type(context) is NativeContentContext)
        _require(current_context(SimpleNamespace(_native_content_policy_context=context)) is context)
        _require(not context._controlled_main_bound and not context._denied)
        self._context = context
        self._bound_context = context
        self._admission, self._handle = context.admission, context.request_handle
        self._selection_port, self._result_port = selection, result_policy
        self._producer_ref, self._route_ref = producer_ref, main_route_ref
        self._clock = clock
        self._thread = threading.get_ident()
        self._busy = self._invalid = False
        self._records = ()
        self._record_anchors = ()
        self._send_count = 0
        self._pending = self._agent = self._kwargs = None
        self._expected = self._holder = self._initial_selection = self._initial_producer = None
        self._preparation_enabled = False
        self._turn_agent = self._turn_arguments = self._turn_state = self._turn_facts = None
        self._turn_raw = self._turn_options = None
        self._turn_frames = ()
        self._constructed = None
        self._completions = ()
        self._completion_anchors = ()
        self._turn_pending = None
        self._system_prefix = ()
        context._controlled_main_payload = self
        context._controlled_main_bound = True
        try:
            _require(len(_adapters) < 128)
            _adapters.add(self)
            _identifier(producer_ref)
            _identifier(main_route_ref)
            self._captured = (self._admission, self._handle, selection, result_policy, clock)
            self._methods = tuple((port, name, _method(getattr(port, name))) for port, name in (
                (self._admission, "current"), (selection, "current"), (result_policy, "current"),
            ))
            self._methods += tuple((self, name, _method(getattr(self, name))) for name in (
                "admit_turn", "seal_turn", "prepared_request", "_prepare_turn", "_turn_request",
                "_turn_operation",
            ))
            _require(type(transports) is tuple and len(transports) <= 128)
            policy = shared_policy()
            _require(type(profiles) is tuple and len(profiles) <= 32)
            for profile in profiles:
                _row(profile, policy.ProfilePolicy)
                _require(type(profile.required_paths) is tuple)
                _require(all(type(path) is str for path in profile.required_paths))
                _require(profile.required_paths == policy.PATHS)
            _require(type(routes) is tuple and len(routes) <= 128)
            for route in routes:
                row = _row(route, policy.RouteBinding)
                for index in (0, 1, 3, 7):
                    _identifier(row[index])
                for index in (2, 6):
                    _integer(row[index])
                _require(type(row[4]) is str and row[4] in policy.PATHS)
                _destination(row[5])
                _require(type(row[8]) is str and row[8] == "synthetic_only")
            selected = tuple(r for r in routes if r.route_ref == main_route_ref)
            _require(len(selected) == 1 and selected[0].path == "main")
            main_transport = selected[0].destination.transport_ref
            wrappers = []
            for pair in transports:
                _require(type(pair) is tuple and len(pair) == 2)
                name, port = pair
                _identifier(name)
                self._methods += tuple((port, method, _method(getattr(port, method)))
                                       for method in ("resolve", "send"))
                wrappers.append((name, _Transport(self, port, main=name == main_transport)))
            self._main_route = selected[0]
            self._busy = True
            self._checkpoint()
            self._runtime = policy.ContentPolicyRuntime(
                admission=_Admission(self), profiles=profiles, routes=routes,
                transports=tuple(wrappers), clock=lambda: self._around(clock),
                policy_revision=self._source.policy_revision,
            )
            self._captured_runtime = self._runtime
        except Exception:
            self._deny()
        finally:
            self._busy = False

    @property
    def invocations(self) -> tuple[NativeMainInvocation, ...]:
        return self._records

    @property
    def send_count(self) -> int:
        return self._send_count

    def _deny(self):
        self._invalidate()
        raise NativeContentPolicyDenied("controlled_native_payload_denied") from None

    def _invalidate(self):
        self._invalid = True
        self._turn_pending = None
        if type(self._bound_context) is NativeContentContext:
            self._bound_context._denied = True

    def _identity(self):
        if self._constructed is not None:
            _verify_constructed(self._constructed)
        context = self._context
        _require(context is self._bound_context)
        _require(not self._invalid and not context._denied)
        _require(type(self) is NativeMainPayloadAdapter)
        _require(self in _adapters)
        if "_runtime" in vars(self):
            _require(self._runtime is self._captured_runtime)
        _require(context._controlled_main_bound is True and context._controlled_main_payload is self)
        _require(threading.get_ident() == self._thread)
        actual = (context.admission, context.request_handle, self._selection_port,
                  self._result_port, self._clock)
        _require(all(a is b for a, b in zip(actual, self._captured)))
        _require(self._admission is self._captured[0] and self._handle is self._captured[1])
        if self._agent is not None:
            _require(current_context(self._agent) is context)
        if self._turn_agent is not None:
            _require(current_context(self._turn_agent) is context)
        for port, name, method in self._methods:
            _require(_same_method(getattr(port, name), method))

    def _snapshot(self):
        self._identity()
        self._verify_records()
        self._verify_turns()
        self._verify_completions()
        source = self._admission.current(self._handle)
        source_row = _binding(source)
        selected = self._selection_port.current(self._handle)
        selection_row = _selection(selected)
        producer = self._result_port.current(self._producer_ref)
        producer_row = _producer(producer)
        self._identity()
        _require(producer_row[0] == self._producer_ref)
        _require(selection_row[:2] == (source_row[6], source_row[7]))
        _require(selection_row[3][1] == "synthetic_native")
        _destination(self._main_route.destination)
        _require(selected.destination == self._main_route.destination)
        _require(set(producer_row[2]) <= set(source_row[8]) or bool(self._records))
        if self._holder is not None:
            _require(source_row[:8] == self._holder)
            _require(selection_row == self._initial_selection)
            _require(producer_row == self._initial_producer)
        payload = None
        if self._agent is not None:
            mode, provider = self._agent.api_mode, self._agent.provider
            _require(type(mode) is str and mode == "chat_completions")
            _require(type(provider) is str and provider == "synthetic_native")
            _request(self._kwargs)
            _require(self._kwargs["model"] == selected.destination.model_ref)
            payload = _json({"api_mode": mode, "destination": dict(zip(
                ("transport_ref", "provider_ref", "model_ref", "endpoint_ref", "revision"),
                selection_row[3])), "request": self._kwargs})
            _require(len(payload) <= 262_144)
        self._source, self._selected, self._producer_policy = source, selected, producer
        if self._turn_agent is not None:
            model = self._turn_agent.model
            _require(type(model) is str and model == selected.destination.model_ref)
            if self._turn_facts is not None:
                _require((source_row, selection_row, producer_row) == self._turn_facts)
        return source_row, selection_row, producer_row, payload

    def _verify_records(self):
        _require(type(self._records) is tuple and len(self._records) == len(self._record_anchors))
        for record, anchor in zip(self._records, self._record_anchors):
            _require(record is anchor[0])
            _row(record, NativeMainInvocation)
            for actual, original in zip((record.native_response, record.request_content,
                    record.result_content, record.envelope, record.request_handle), anchor[1:6]):
                _require(actual is original)
            _require(type(record.request_bytes) is bytes and type(record.response_bytes) is bytes)
            _require((record.request_bytes, record.response_bytes) == anchor[6:8])
            _require(_binding(record.source) == anchor[8])
            _require(_selection(record.selection) == self._initial_selection)
            _require(_producer(record.producer_policy) == self._initial_producer)
            _require(_destination(record.observed_destination) == self._initial_selection[3])
            _require(_response_bytes(record.native_response, record.selection.destination.model_ref)
                     == record.response_bytes)
            if self._preparation_enabled:
                response = record.native_response
                actual = (response.choices, response.choices[0], response.choices[0].message, response.usage)
                _require(all(a is b for a, b in zip(actual, anchor[9])))
                actual = (record.source, record.selection, record.producer_policy, record.observed_destination)
                _require(all(a is b for a, b in zip(actual, anchor[10])))

    def _verify_turns(self):
        if self._turn_agent is None:
            return
        from agent.message_sanitization import _sanitize_surrogates
        _require(_sanitize_surrogates is _NATIVE_SANITIZER)
        mode, provider = self._turn_agent.api_mode, self._turn_agent.provider
        _require(type(mode) is str and mode == "chat_completions")
        _require(type(provider) is str and provider == "synthetic_native")
        system = self._system_prefix[0] if self._system_prefix else None
        if self._turn_arguments is not None:
            _turn_shape(self._turn_arguments)
            system = self._turn_arguments[1]
        state, options = _native_state(self._turn_agent, system)
        if self._turn_state is not None:
            _require((state, options) == (self._turn_state, self._turn_options))
            _require(_turn_bytes(self._turn_arguments, options) == self._turn_raw)
        _require(type(self._turn_frames) is tuple and len(self._turn_frames) <= 8)
        for frame in self._turn_frames:
            _require(type(frame) is _TurnFrame)
            _require(type(frame.token) is NativeTurnInput)
            _require(_turn_bytes(frame.arguments, frame.options) == frame.raw_bytes)
            actual = _input_anchors(frame.arguments)
            _require(len(actual) == len(frame.input_anchors))
            _require(all(a is b for a, b in zip(actual, frame.input_anchors)))
            if frame.output is not None:
                _require(_output_bytes(frame.output) == frame.output_bytes)
                actual = _output_anchors(frame.output)
                _require(len(actual) == len(frame.output_anchors))
                _require(all(a is b for a, b in zip(actual, frame.output_anchors)))
                _require(frame.output.controlled_prepared is frame.prepared)

    def _verify_completions(self):
        _require(type(self._completions) is tuple and len(self._completions) <= 8)
        _require(type(self._completion_anchors) is tuple)
        _require(len(self._completions) == len(self._completion_anchors))
        for index, completion in enumerate(self._completions):
            anchor = self._completion_anchors[index]
            _require(completion is anchor[0])
            _require(all(a is b for a, b in zip(
                (completion.frame, completion.invocation, completion.parents, completion.payload,
                 completion.content, completion.result, completion.anchors), anchor[1:])))
            _require(type(completion) is _NativeCompletion)
            _require(completion.frame is self._turn_frames[index])
            _require(completion.invocation is self._records[index])
            _require(type(completion.payload) is bytes and len(completion.payload) <= _MAX_NATIVE_RESULT)
            _require(_native_result_bytes(completion.result) == completion.payload)
            actual = _native_result_anchors(completion.result)
            _require(len(actual) == len(completion.anchors))
            _require(all(a is b for a, b in zip(actual, completion.anchors)))

    def _interrupt_turn(self):
        interrupted = getattr(self._turn_agent, "_interrupt_requested", False)
        self._checkpoint()
        if type(interrupted) is not bool:
            raise NativeContentPolicyDenied("controlled_native_interrupt_invalid")
        if interrupted is True:
            raise InterruptedError("Agent interrupted during API call")

    @contextmanager
    def _turn_operation(self, agent):
        try:
            _require(not self._busy)
            self._busy = True
            _require(self._turn_agent is agent)
            initial = self._checkpoint()
            if self._turn_state is None and self._turn_arguments is not None:
                self._turn_state, self._turn_options = _native_state(agent, self._turn_arguments[1])
                self._turn_raw = _turn_bytes(self._turn_arguments, self._turn_options)
                self._turn_facts = initial[:3]
            self._interrupt_turn()
            yield
            self._interrupt_turn()
        except InterruptedError:
            self._invalidate()
            raise InterruptedError("Agent interrupted during API call") from None
        except NativeContentPolicyDenied:
            self._invalidate()
            raise
        except Exception:
            self._deny()
        except BaseException:
            self._invalidate()
            raise
        finally:
            self._busy = False

    def admit_turn(
        self, agent: object, *, user_message: str,
        system_message: str | None,
        conversation_history: list[dict[str, str]] | None,
        task_id: str,
    ) -> NativeTurnInput:
        self._preparation_enabled = True
        if self._turn_agent is None:
            self._turn_agent = agent
        arguments = (user_message, system_message, conversation_history, task_id)
        if not self._busy and self._turn_pending is None and self._turn_state is None:
            self._turn_arguments = arguments
        with self._turn_operation(agent):
            _require(self._turn_pending is None and len(self._records) < 8)
            _require(len(self._turn_frames) < 8)
            state, options = _native_state(agent, system_message)
            raw = _turn_bytes(arguments, options)
            _require(task_id == self._source.execution_ref)
            if self._system_prefix:
                _require(system_message == self._system_prefix[0])
            if self._records:
                previous = self._records[-1]
                history = json.loads(previous.request_bytes)["request"]["messages"]
                prefix = history.pop(0)["content"] if history[0]["role"] == "system" else None
                _require(prefix == system_message)
                history.append({"role": "assistant", "content": previous.native_response.choices[0].message.content})
                _require(conversation_history == history)
            else:
                _require(conversation_history is None or conversation_history == [])
            self._turn_arguments, self._turn_raw = arguments, raw
            self._turn_state, self._turn_options = state, options
            self._turn_facts = self._checkpoint()[:3]
            admitted = self._runtime.admit(self._handle, raw)
            parents = (admitted, *(r.result_content for r in self._records),
                       *(row.content for row in self._completions))
            content = (self._runtime.derive(self._handle, parents, raw, path="main")
                       if self._records else admitted)
            self._checkpoint()
            token = object.__new__(NativeTurnInput)
            frame = _TurnFrame(token, arguments, raw, admitted, content, options, _input_anchors(arguments))
            self._turn_frames += (frame,)
            self._turn_pending = frame
            if not self._system_prefix:
                self._system_prefix = (system_message,)
            return token

    def _prepare_turn(self, agent, token, arguments, exclusions):
        from agent.turn_context import TurnContext

        with self._turn_operation(agent):
            frame = self._turn_pending
            _require(frame is not None and type(token) is NativeTurnInput and token is frame.token)
            _require(frame.output is None)
            _require(all(value is None for value in exclusions[:-1]) and exclusions[-1] is False)
            _require(len(arguments) == len(frame.arguments))
            _require(all(a is b for a, b in zip(arguments, frame.arguments)))
            _require(_turn_bytes(arguments, frame.options) == frame.raw_bytes)
            # The shared registry retains the complete raw bytes before this call.
            user, system, history, task = frame.arguments
            sanitized = _NATIVE_SANITIZER(user)
            copied = None if history is None else [dict(row) for row in history]
            messages = ([] if copied is None else [dict(row) for row in copied])
            messages.append({"role": "user", "content": sanitized})
            output = TurnContext(
                user_message=sanitized, original_user_message=sanitized,
                messages=messages, conversation_history=copied,
                active_system_prompt=system, effective_task_id=task,
                turn_id=f"native-turn-{len(self._records) + 1}",
                current_turn_user_idx=len(messages) - 1,
            )
            encoded = _output_bytes(output)
            frame = replace(frame, output=output, output_anchors=_output_anchors(output), output_bytes=encoded)
            self._turn_frames = (*self._turn_frames[:-1], frame)
            self._turn_pending = frame
        prepared = self.seal_turn(agent, token, turn_context=output)
        _require(output.controlled_prepared is prepared)
        return output

    def seal_turn(
        self, agent: object, turn: NativeTurnInput, *, turn_context: TurnContext,
    ) -> NativePreparedTurn:
        with self._turn_operation(agent):
            frame = self._turn_pending
            _require(frame is not None and type(turn) is NativeTurnInput and turn is frame.token)
            _require(frame.output is turn_context and frame.output is not None and frame.prepared is None)
            from agent.turn_context import TurnContext
            user, system, history, task = frame.arguments
            clean = _NATIVE_SANITIZER(user)
            expected = TurnContext(clean, clean, [*(history or []), {"role": "user", "content": clean}],
                                   history, system, task, f"native-turn-{len(self._records) + 1}", len(history or []))
            _require(_output_bytes(expected) == frame.output_bytes)
            request = self._turn_request(frame)
            encoded = _json({"turn_context": json.loads(frame.output_bytes), "request": request})
            _require(len(encoded) <= 262_144)
            parents = (frame.admitted, frame.input_content, *(r.result_content for r in self._records),
                       *(row.content for row in self._completions))
            content = self._runtime.derive(self._handle, parents, encoded, path="main")
            self._checkpoint()
            prepared = object.__new__(NativePreparedTurn)
            turn_context.controlled_prepared = prepared
            frame = replace(frame, prepared=prepared, output_content=content, request_bytes=_json(request))
            self._turn_frames = (*self._turn_frames[:-1], frame)
            self._turn_pending = frame
            return prepared

    def _turn_request(self, frame):
        options = json.loads(frame.options)
        output = json.loads(frame.output_bytes)
        messages = output["messages"]
        if output["active_system_prompt"] is not None:
            messages.insert(0, {"role": "system", "content": output["active_system_prompt"]})
        request = {"model": options["model"]["value"], "messages": messages, "stream": False}
        for name in ("max_tokens", "temperature"):
            if options[name]["value"] is not None:
                request[name] = options[name]["value"]
        _request(request)
        return request

    def prepared_request(self, agent: object, prepared: NativePreparedTurn) -> dict:
        with self._turn_operation(agent):
            frame = self._turn_pending
            _require(frame is not None and type(prepared) is NativePreparedTurn and prepared is frame.prepared)
            request = self._turn_request(frame)
            _require(_json(request) == frame.request_bytes)
            content = self._runtime.derive(
                self._handle,
                (frame.admitted, frame.input_content, frame.output_content,
                 *(r.result_content for r in self._records),
                 *(row.content for row in self._completions)),
                frame.request_bytes, path="main",
            )
            self._checkpoint()
            frame = replace(frame, request_contents=(*frame.request_contents, content))
            self._turn_frames = (*self._turn_frames[:-1], frame)
            self._turn_pending = frame
            return request

    def _checkpoint(self):
        first = self._snapshot()
        second = self._snapshot()
        _require(first == second)
        if self._expected is not None:
            _require(first == self._expected)
        if self._holder is None:
            self._holder, self._initial_selection, self._initial_producer = first[0][:8], first[1], first[2]
        return first

    def _around(self, callback):
        self._checkpoint()
        output = callback()
        self._checkpoint()
        return output

    def validate_scope(self) -> None:
        try:
            _require(not self._busy)
            self._busy = True
            self._checkpoint()
        except Exception:
            self._deny()
        except BaseException:
            if self._preparation_enabled:
                self._invalidate()
            raise
        finally:
            self._busy = False

    def dispatch(self, agent: object, api_kwargs: dict) -> NativeMainResponse:
        try:
            _require(not self._busy and len(self._records) < 8)
            self._busy = True
            frame = None
            if self._preparation_enabled:
                frame = self._turn_pending
                _require(self._turn_agent is agent and frame is not None and frame.prepared is not None)
                self._interrupt_turn()
            _request(api_kwargs)
            if frame is not None:
                _require(_json(api_kwargs) == frame.request_bytes)
            self._agent, self._kwargs = agent, api_kwargs
            self._expected = self._checkpoint()
            self._request_bytes = self._expected[3]
            messages = api_kwargs["messages"]
            if self._records:
                previous = self._records[-1]
                prior_messages = json.loads(previous.request_bytes)["request"]["messages"]
                for record in self._records:
                    _require(record.request_handle is self._handle)
                    _require(_producer(record.producer_policy) == self._initial_producer)
                    _require(_response_bytes(record.native_response, api_kwargs["model"]) == record.response_bytes)
                prior_messages += [{"role": "assistant", "content": previous.native_response.choices[0].message.content}]
                _require(len(messages) == len(prior_messages) + 1)
                _require(messages[:-1] == prior_messages)
            else:
                _require(len(messages) in (1, 2))
                if len(messages) == 2:
                    _require(messages[0]["role"] == "system")
            _require(messages[-1]["role"] == "user")
            if frame is not None:
                request_content = self._runtime.derive(
                    self._handle,
                    (frame.admitted, frame.input_content, frame.output_content,
                     *(r.result_content for r in self._records),
                     *(row.content for row in self._completions)),
                    self._request_bytes, path="main",
                )
            elif self._records:
                request_content = self._runtime.derive(
                    self._handle, tuple(r.result_content for r in self._records),
                    self._request_bytes, path="main",
                )
            else:
                request_content = self._runtime.admit(self._handle, self._request_bytes)
            result_content = self._runtime.dispatch(
                self._handle, request_content, path="main", route_ref=self._route_ref,
            )
            self._checkpoint()
            _require(self._pending is not None)
            envelope, actual, output = self._pending
            value = _decode_response(output, api_kwargs["model"])
            response = _response(value)
            self._checkpoint()
            record = NativeMainInvocation(
                self._handle, self._source, self._selected, self._producer_policy,
                self._request_bytes, request_content, envelope, actual, output,
                result_content, response,
            )
            self._records += (record,)
            self._record_anchors += ((record, response, request_content, result_content,
                envelope, self._handle, self._request_bytes, output, _binding(record.source),
                (response.choices, response.choices[0], response.choices[0].message, response.usage),
                (record.source, record.selection, record.producer_policy, record.observed_destination)),)
            if frame is not None:
                self._turn_pending = None
                self._turn_arguments = self._turn_raw = self._turn_state = self._turn_options = self._turn_facts = None
                self._interrupt_turn()
            return response
        except InterruptedError:
            if not self._preparation_enabled:
                self._deny()
            self._invalidate()
            raise InterruptedError("Agent interrupted during API call") from None
        except Exception:
            self._deny()
        except BaseException:
            if self._preparation_enabled:
                self._invalidate()
            raise
        finally:
            self._agent = self._kwargs = self._pending = self._expected = None
            self._busy = False


def bind_controlled_main(
    context: NativeContentContext, *, selection: NativeSelectionPort,
    result_policy: NativeResultPolicyPort, producer_ref: str, main_route_ref: str,
    profiles: tuple[ProfilePolicy, ...], routes: tuple[RouteBinding, ...],
    transports: tuple[tuple[str, TransportPort], ...], clock: Callable[[], int],
) -> NativeMainPayloadAdapter:
    try:
        return NativeMainPayloadAdapter(
            context, selection=selection, result_policy=result_policy, producer_ref=producer_ref,
            main_route_ref=main_route_ref, profiles=profiles, routes=routes,
            transports=transports, clock=clock,
        )
    except Exception:
        if type(context) is NativeContentContext:
            context.invalidate()
        raise NativeContentPolicyDenied("controlled_native_payload_denied") from None


def _attached(context):
    adapter = context._controlled_main_payload
    if type(adapter) is not NativeMainPayloadAdapter or adapter._context is not context:
        context.invalidate()
        raise NativeContentPolicyDenied("controlled_native_attachment_invalid")
    return adapter


def validate_controlled_scope(context: NativeContentContext) -> None:
    _attached(context).validate_scope()


def invalidate_controlled_context(context: NativeContentContext) -> None:
    adapter = context._controlled_main_payload
    if type(adapter) is NativeMainPayloadAdapter:
        adapter._invalid = True
        adapter._turn_pending = None
        if type(adapter._context) is NativeContentContext:
            adapter._context._denied = True


def _prepare_controlled_turn(agent, token, arguments, exclusions):
    context = current_context(agent)
    try:
        _require(context is not None and context._controlled_main_bound is True)
        return _attached(context)._prepare_turn(agent, token, arguments, exclusions)
    except BaseException:
        if type(context) is NativeContentContext:
            context._denied = True
            invalidate_controlled_context(context)
        raise


def dispatch_controlled_main(agent: object, api_kwargs: dict) -> tuple[bool, NativeMainResponse | None]:
    context = current_context(agent)
    if context is not None and context._controlled_main_bound:
        return True, _attached(context).dispatch(agent, api_kwargs)
    provider = getattr(agent, "provider", None)
    if type(provider) is str and provider == "synthetic_native":
        if context is not None:
            context.invalidate()
        raise NativeContentPolicyDenied("controlled_native_adapter_missing")
    return False, None


# Keep these defaults equal to the existing constructor API.
_CONSTRUCTOR_DEFAULTS = {
    'base_url': None,
    'api_key': None,
    'provider': None,
    'api_mode': None,
    'acp_command': None,
    'acp_args': None,
    'command': None,
    'args': None,
    'model': '',
    'max_iterations': sys.maxsize,
    'enabled_toolsets': None,
    'disabled_toolsets': None,
    'save_trajectories': False,
    'verbose_logging': False,
    'quiet_mode': False,
    'tool_progress_mode': 'all',
    'ephemeral_system_prompt': None,
    'log_prefix_chars': 100,
    'log_prefix': '',
    'providers_allowed': None,
    'providers_ignored': None,
    'providers_order': None,
    'provider_sort': None,
    'provider_require_parameters': False,
    'provider_data_collection': None,
    'openrouter_min_coding_score': None,
    'session_id': None,
    'tool_progress_callback': None,
    'tool_start_callback': None,
    'tool_complete_callback': None,
    'thinking_callback': None,
    'reasoning_callback': None,
    'clarify_callback': None,
    'read_terminal_callback': None,
    'read_preview_callback': None,
    'drive_preview_callback': None,
    'read_window_below_callback': None,
    'setup_mcp_callback': None,
    'tour_callback': None,
    'step_callback': None,
    'stream_delta_callback': None,
    'interim_assistant_callback': None,
    'tool_gen_callback': None,
    'status_callback': None,
    'notice_callback': None,
    'notice_clear_callback': None,
    'event_callback': None,
    'reaction_callback': None,
    'max_tokens': None,
    'reasoning_config': None,
    'service_tier': None,
    'request_overrides': None,
    'prefill_messages': None,
    'platform': None,
    'user_id': None,
    'user_id_alt': None,
    'user_name': None,
    'chat_id': None,
    'chat_name': None,
    'chat_type': None,
    'thread_id': None,
    'gateway_session_key': None,
    'skip_context_files': False,
    'load_soul_identity': False,
    'skip_memory': False,
    'skip_background_review': False,
    'session_db': None,
    'parent_session_id': None,
    'iteration_budget': None,
    'run_budget_seconds': None,
    'fallback_model': None,
    'credential_pool': None,
    'checkpoints_enabled': False,
    'checkpoint_max_snapshots': 20,
    'checkpoint_max_total_size_mb': 500,
    'checkpoint_max_file_size_mb': 10,
    'pass_session_id': False,
    'requested_provider': None,
    'capabilities': None,
    "tool_delay": None,
}


def _namespace(agent):
    try:
        namespace = object.__getattribute__(agent, "__dict__")
    except (AttributeError, TypeError):
        return {}
    return namespace if type(namespace) is dict else {}


def _owned_constructed_binding(agent):
    return next((binding for binding in _constructed_bindings if binding.agent is agent), None)


def _controlled_attempt(agent, adapter=None, provider=None):
    namespace = _namespace(agent)
    if (adapter is not None or _owned_constructed_binding(agent) is not None
            or "_native_controlled_binding" in namespace
            or "_native_controlled_options" in namespace):
        return True
    provider = namespace.get("provider") if provider is None else provider
    if isinstance(provider, str) and str.__eq__(provider, "synthetic_native") is True:
        return True
    active = current_context()
    attached = namespace.get("_native_content_policy_context")
    for context in (attached, active):
        if context is None:
            continue
        if type(context) is not NativeContentContext:
            return True
        if (context._controlled_main_bound is not False
                or context._controlled_main_payload is not None):
            return True
    return False


def _invalidate_constructed_attempt(agent, adapter=None):
    namespace = _namespace(agent)
    binding = _owned_constructed_binding(agent)
    candidates = [adapter]
    if type(binding) is _ConstructedMain and binding in _constructed_bindings:
        candidates.append(binding.adapter)
    for candidate in candidates:
        if type(candidate) is NativeMainPayloadAdapter and candidate in _adapters:
            candidate._invalidate()
    for context in (namespace.get("_native_content_policy_context"), current_context()):
        if type(context) is NativeContentContext:
            context._denied = True
            invalidate_controlled_context(context)


def _validate_constructor_options(agent, adapter, supplied):
    try:
        from run_agent import AIAgent
        _require(type(agent) is AIAgent)
        _require(type(adapter) is NativeMainPayloadAdapter and adapter in _adapters)
        _require(adapter._constructed is None and adapter._turn_agent is None)
        _require(not adapter._records and not adapter._turn_frames and not adapter._busy)
        _require(current_context() is adapter._context)
        _require(current_context(agent) is adapter._context)
        supported = {"provider", "api_mode", "model", "session_id", "quiet_mode",
                     "enabled_toolsets", "skip_memory", "skip_context_files",
                     "skip_background_review", "max_tokens"}
        options = {}
        for name, default in _CONSTRUCTOR_DEFAULTS.items():
            if name == "tool_delay" and name not in supplied:
                continue
            _require(name in supplied)
            value = supplied[name]
            if name not in supported:
                _require(type(value) is type(default))
                _require(value == default)
            options[name] = value
        for name, expected in (("provider", "synthetic_native"),
                               ("api_mode", "chat_completions")):
            _require(type(options[name]) is str and options[name] == expected)
        _identifier(options["model"])
        _identifier(options["session_id"])
        for name in ("quiet_mode", "skip_memory", "skip_context_files", "skip_background_review"):
            _require(options[name] is True)
        tools = options["enabled_toolsets"]
        _require(type(tools) is list and len(tools) == 0)
        if options["max_tokens"] is not None:
            _integer(options["max_tokens"], 4096)
        snapshot = _json(options)
        adapter.validate_scope()
        _require(_json(options) == snapshot)
        _require(options["model"] == adapter._selected.destination.model_ref)
        return options
    except BaseException:
        _invalidate_constructed_attempt(agent, adapter)
        raise


def _initialize_controlled_main(agent, options):
    """Set inert state without configuration, discovery, or resource creation."""
    namespace = _namespace(agent)
    _require(not namespace)
    for name, value in options.items():
        namespace[name] = value
    namespace.update(
        tools=[], valid_tool_names=set(), _persist_disabled=True,
        compression_enabled=False, _memory_store=None, _memory_manager=None,
        _session_db=None, context_compressor=None, _primary_runtime=None,
        _fallback_model=None, _fallback_chain=[], prefill_messages=[],
        _session_messages=[], request_overrides={}, _client_kwargs={},
        client=None, _anthropic_client=None, _cached_system_prompt=None,
        base_url="", api_key=None, _native_content_policy_required=True,
        _native_controlled_options=_json(options),
        _interrupt_requested=False, _interrupt_message=None,
        _tool_interrupt_reason=None, _interrupt_thread_signal_pending=False,
        _execution_thread_id=None, _active_request_abort=None,
        _active_children=set(), _active_children_lock=threading.Lock(),
        _tool_worker_threads=set(), _tool_worker_threads_lock=threading.Lock(),
        _pending_redirect=None, _pending_redirect_lock=threading.Lock(),
        _hard_interrupt_requested=threading.Event(),
    )


@dataclass(eq=False)
class _ConstructedMain:
    agent: object
    adapter: NativeMainPayloadAdapter
    state: bytes
    anchors: tuple
    methods: tuple
    closed: bool = False


def _constructed_state(namespace):
    dynamic = {"_interrupt_requested", "_interrupt_message", "_tool_interrupt_reason",
               "_interrupt_thread_signal_pending"}
    opaque = {"_native_controlled_binding", "_native_content_policy_context",
              "_active_children_lock", "_tool_worker_threads_lock",
              "_pending_redirect_lock", "_hard_interrupt_requested"}
    value = {}
    for name, item in namespace.items():
        _require(type(name) is str)
        if name in dynamic or name in opaque:
            continue
        if name == "_native_controlled_options":
            _require(type(item) is bytes)
            value[name] = item.decode("ascii")
        elif name in ("valid_tool_names", "_active_children", "_tool_worker_threads"):
            _require(type(item) is set and len(item) == 0)
            value[name] = []
        else:
            _require(type(item) in (str, int, bool, list, dict, type(None)))
            if type(item) in (list, dict):
                _require(len(item) == 0)
            value[name] = item
    return _json(value)


def _verify_constructed(binding):
    from run_agent import AIAgent
    _require(type(binding) is _ConstructedMain and binding in _constructed_bindings)
    _require(type(binding.closed) is bool and binding.closed is False)
    agent, adapter = binding.agent, binding.adapter
    _require(type(agent) is AIAgent)
    _require(type(adapter) is NativeMainPayloadAdapter and adapter in _adapters)
    namespace = _namespace(agent)
    _require(namespace.get("_native_controlled_binding") is binding)
    _require(adapter._constructed is binding and adapter._turn_agent is agent)
    _require(namespace.get("_native_content_policy_context") is adapter._bound_context)
    _require(set(namespace) == {name for name, _ in binding.anchors})
    dynamic = {"_interrupt_requested", "_interrupt_message", "_tool_interrupt_reason",
               "_interrupt_thread_signal_pending"}
    for name, original in binding.anchors:
        if name not in dynamic:
            _require(namespace[name] is original)
    _require(_constructed_state(namespace) == binding.state)
    if type(namespace["_interrupt_requested"]) is not bool:
        raise NativeContentPolicyDenied("controlled_native_interrupt_invalid")
    _require(type(namespace["_interrupt_thread_signal_pending"]) is bool)
    _require(namespace["_interrupt_message"] is None)
    reason = namespace["_tool_interrupt_reason"]
    _require(reason is None or (type(reason) is str and reason == "user interrupt"))
    for port, name, method in binding.methods:
        _require(_same_method(getattr(port, name), method))


def _bind_constructed_main(agent, adapter: NativeMainPayloadAdapter) -> None:
    try:
        from run_agent import AIAgent
        from agent import conversation_loop
        _require(type(agent) is AIAgent)
        _require(type(adapter) is NativeMainPayloadAdapter and adapter in _adapters)
        _require(adapter._constructed is None and adapter._turn_agent is None)
        _require(not adapter._records and not adapter._turn_frames)
        _require(current_context() is adapter._bound_context)
        adapter.validate_scope()
        agent._native_content_policy_context = adapter._bound_context
        binding = _ConstructedMain(agent, adapter, b"", (), ())
        agent._native_controlled_binding = binding
        adapter._constructed = binding
        adapter._turn_agent = agent
        adapter._preparation_enabled = True
        binding.methods = tuple((port, name, _method(getattr(port, name))) for port, name in (
            (agent, "run_conversation"), (agent, "close"), (agent, "interrupt"),
            (agent, "_interruptible_api_call"),
            (conversation_loop, "_try_controlled_text_turn"),
            (conversation_loop, "build_turn_context"),
            (sys.modules[__name__], "_complete_controlled_turn"),
            (adapter, "_verify_completions"),
        ))
        binding.anchors = tuple(vars(agent).items())
        binding.state = _constructed_state(vars(agent))
        _constructed_bindings.add(binding)
        adapter.validate_scope()
    except BaseException:
        _invalidate_constructed_attempt(agent, adapter)
        raise


def _constructed_adapter(agent):
    try:
        binding = _namespace(agent).get("_native_controlled_binding")
        _verify_constructed(binding)
        _require(current_context(agent) is binding.adapter._bound_context)
        binding.adapter.validate_scope()
        return binding.adapter
    except Exception:
        _invalidate_constructed_attempt(agent)
        raise NativeContentPolicyDenied("controlled_native_payload_denied") from None
    except BaseException:
        _invalidate_constructed_attempt(agent)
        raise


@dataclass(frozen=True, slots=True)
class _NativeCompletion:
    frame: _TurnFrame
    invocation: NativeMainInvocation
    parents: tuple
    payload: bytes
    content: object
    result: dict
    anchors: tuple


def _native_result_anchors(result):
    return (result, *result.values(), *result["messages"],
            *(value for row in result["messages"] for value in row.values()))


def _native_result_bytes(result):
    _require(type(result) is dict)
    for name, value in result.items():
        _require(type(name) is str)
        if name == "messages":
            _history(value)
        else:
            _require(type(value) in (str, int, bool, type(None)))
    return _json(result)


def _complete_controlled_turn(agent, *, turn_context: TurnContext,
                              response: NativeMainResponse) -> dict:
    adapter = _constructed_adapter(agent)
    with adapter._turn_operation(agent):
        _require(adapter._turn_pending is None)
        _require(len(adapter._turn_frames) == len(adapter._records) == len(adapter._completions) + 1)
        frame, invocation = adapter._turn_frames[-1], adapter._records[-1]
        _require(frame.output is turn_context and frame.prepared is not None)
        _require(response is invocation.native_response)
        _require(_response_bytes(response, agent.model) == invocation.response_bytes)
        messages = [dict(row) for row in turn_context.messages]
        messages.append({"role": "assistant", "content": response.choices[0].message.content})
        totals = [sum(getattr(record.native_response.usage, name) for record in adapter._records)
                  for name in ("prompt_tokens", "completion_tokens", "total_tokens")]
        for total in totals:
            _integer(total, _MAX_INT, 0)
        prompt, completion, total = totals
        _require(prompt + completion == total)
        result = dict(
            final_response=response.choices[0].message.content, last_reasoning=None,
            messages=messages, api_calls=1, completed=True,
            turn_exit_reason="text_response(finish_reason=stop)", failed=False,
            partial=False, interrupted=False, response_transformed=False,
            pre_transform_response=None, response_previewed=False,
            model=agent.model, provider="synthetic_native", base_url="",
            input_tokens=prompt, output_tokens=completion, cache_read_tokens=0,
            cache_write_tokens=0, reasoning_tokens=0, prompt_tokens=prompt,
            completion_tokens=completion, total_tokens=total,
            last_prompt_tokens=response.usage.prompt_tokens, estimated_cost_usd=None,
            cost_status="unknown", cost_source=None, service_tier=None,
            session_id=agent.session_id,
        )
        payload = _native_result_bytes(result)
        _require(len(payload) <= _MAX_NATIVE_RESULT)
        parents = (frame.admitted, frame.input_content, frame.output_content,
                   *frame.request_contents, invocation.request_content, invocation.result_content,
                   *(row.content for row in adapter._completions))
        content = adapter._runtime.derive(adapter._handle, parents, payload, path="main")
        adapter._interrupt_turn()
        completed = _NativeCompletion(
            frame, invocation, parents, payload, content, result, _native_result_anchors(result),
        )
        adapter._completions += (completed,)
        adapter._completion_anchors += ((completed, frame, invocation, parents, payload,
                                         content, result, completed.anchors),)
        adapter._interrupt_turn()
        return result


def _close_constructed_main(agent) -> bool:
    if not _controlled_attempt(agent):
        return False
    _invalidate_constructed_attempt(agent)
    binding = _owned_constructed_binding(agent)
    if binding is not None:
        binding.closed = True
    return True
