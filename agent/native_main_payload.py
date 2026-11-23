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
        FailedDispatch, ProfilePolicy, RouteBinding, TransportPort,
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
class _DelegateFrames(weakref.WeakKeyDictionary):
    def __setitem__(self, key, value):
        super().__setitem__(key, weakref.ref(value))

    def __getitem__(self, key):
        return super().__getitem__(key)()

    def get(self, key, default=None):
        reference = super().get(key)
        return reference() if reference is not None else default


_delegate_frames = _DelegateFrames()


class NativeDelegateInput:
    __slots__ = ()

    def __new__(cls):
        raise TypeError("Delegate tokens require retained input")


@dataclass(frozen=True, slots=True)
class NativeDelegateBinding:
    parent_request: object
    receiver_request: object
    child_request: object
    receiver_selection: NativeSelectionPort
    child_selection: NativeSelectionPort
    receiver_producer_ref: str
    child_producer_ref: str
    delegate_route: object
    child_main_route: object
    child_session_id: str
    child_task_id: str
    snapshots: tuple
    anchors: tuple


@dataclass(frozen=True, slots=True)
class NativeDelegateInvocation:
    token: NativeDelegateInput
    binding: NativeDelegateBinding
    arguments: tuple
    argument_anchors: tuple
    raw_bytes: bytes
    raw_content: object
    leaf_bytes: bytes
    leaf_content: object
    request_content: object
    envelope: object
    ack_bytes: bytes | None
    ack_content: object
    accepted_content: object
    child_seed: object
    child: object
    child_adapter: object
    child_input: object
    child_request: object
    child_invocation: object
    child_completion: object
    entry_bytes: bytes | None
    aggregate_bytes: bytes | None
    receiver_result: object
    return_facts: object


class NativeTurnInput:
    __slots__ = ()

    def __new__(cls):
        raise TypeError("Native turn tokens require adapter admission")


class NativeAuxiliaryInput:
    __slots__ = ()

    def __new__(cls):
        raise TypeError("Auxiliary tokens require retained input")


class NativeAuxiliaryRequest:
    __slots__ = ()

    def __new__(cls):
        raise TypeError("Auxiliary tokens require retained preparation")


class NativePreparedTurn:
    __slots__ = ()

    def __new__(cls):
        raise TypeError("Native turn tokens require adapter preparation")


class NativeFallbackInput:
    __slots__ = ()

    def __new__(cls):
        raise TypeError("Fallback tokens require an observed main failure")


@dataclass(frozen=True, slots=True)
class _FallbackBinding:
    handle: object
    port: object
    producer_ref: str
    route: object
    holder: tuple
    selected: tuple
    producer: tuple
    original_profiles: tuple


@dataclass(frozen=True, slots=True)
class _FailedSendObservation:
    operation: object
    envelope: object
    observed_destination: object
    exception_type: type
    send_count_before: int
    send_count_after: int


@dataclass(frozen=True, slots=True)
class NativeFailedMainAttempt:
    turn_frame: object
    request_handle: object
    source: object
    selection: object
    producer_policy: object
    source_bytes: bytes
    parents: tuple
    parent_requests: tuple
    request: dict
    request_bytes: bytes
    request_content: object
    send_observation: _FailedSendObservation
    shared_failure: object
    failure_facts: object
    failure_content: object


@dataclass(frozen=True, slots=True)
class _FallbackFrame:
    token: NativeFallbackInput
    failed_main: NativeFailedMainAttempt
    binding: _FallbackBinding
    parents: tuple
    parent_requests: tuple
    source_content: object
    input_stages: tuple = ()
    stage_contents: tuple = ()
    request: dict | None = None
    request_bytes: bytes | None = None
    request_anchors: tuple = ()
    activated: bool = False
    invocation: object = None
    result: dict | None = None
    result_bytes: bytes | None = None
    result_content: object = None
    result_anchors: tuple = ()


@dataclass(frozen=True, slots=True)
class NativeFallbackInvocation:
    failed_main: NativeFailedMainAttempt
    request_handle: object
    source: object
    selection: object
    producer_policy: object
    parents: tuple
    parent_requests: tuple
    input_stages: tuple
    stage_contents: tuple
    request: dict
    request_bytes: bytes
    request_content: object
    envelope: object = None
    observed_destination: object = None
    response_bytes: bytes | None = None
    result_content: object = None
    native_response: object = None
    output_stages: tuple = ()
    output_contents: tuple = ()
    output: dict | None = None
    output_bytes: bytes | None = None
    output_content: object = None


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


@dataclass(frozen=True, slots=True)
class NativeAuxiliaryInvocation:
    source_completion: object
    source_frame: object
    source_invocation: NativeMainInvocation
    source_handle: object
    request_handle: object
    source: object
    target: object
    path: str
    task: str
    selection: NativeMainSelection
    producer_policy: NativeResultPolicy
    source_bytes: bytes
    parents: tuple
    parent_requests: tuple
    input_stages: tuple
    stage_contents: tuple
    request_bytes: bytes
    request_content: object
    envelope: object
    observed_destination: object
    response_bytes: bytes | None
    result_content: object
    native_response: NativeMainResponse | None
    output_stages: tuple = ()
    output_contents: tuple = ()
    output_bytes: bytes | None = None
    output: dict | None = None
    output_content: object = None


@dataclass(frozen=True, slots=True)
class NativeAuxiliaryLocalResult:
    source_completion: object
    source_frame: object
    source_invocation: NativeMainInvocation
    source_handle: object
    request_handle: object
    source: ExecutionBinding
    target: ExecutionBinding
    path: str
    task: str
    selection: NativeMainSelection
    producer_policy: NativeResultPolicy
    language: str
    operation_index: int
    outcome: str
    reason: str
    source_bytes: bytes
    parents: tuple
    parent_requests: tuple
    source_content: ContentHandle
    input_stages: tuple[bytes, ...]
    stage_contents: tuple[ContentHandle, ...]
    output_stages: tuple[bytes, ...]
    output_contents: tuple[ContentHandle, ...]
    output_bytes: bytes
    output: dict
    output_content: ContentHandle


@dataclass(frozen=True, slots=True)
class _AuxiliaryTitlePreparation:
    kind: str
    stages: tuple[bytes, ...]
    value: dict


@dataclass(frozen=True, slots=True)
class _AuxiliaryBinding:
    handle: object
    port: object
    producer_ref: str
    route: object
    language: str
    source: tuple
    selected: tuple
    producer: tuple
    original_profiles: tuple


@dataclass(frozen=True, slots=True)
class _AuxiliaryFrame:
    token: NativeAuxiliaryInput
    completion: object
    field: str
    text: str
    source_bytes: bytes
    parents: tuple
    parent_requests: tuple
    source_content: object
    stages: tuple = ()
    stage_contents: tuple = ()
    request: dict | None = None
    request_bytes: bytes | None = None
    request_anchors: tuple = ()
    sealed: NativeAuxiliaryRequest | None = None
    record: NativeAuxiliaryInvocation | None = None
    preparation: _AuxiliaryTitlePreparation | None = None
    preparation_anchors: tuple = ()
    local_result: NativeAuxiliaryLocalResult | None = None


@dataclass(frozen=True, slots=True)
class _CompressionBinding:
    auxiliary: _AuxiliaryBinding
    context_length: int
    summary_date: str


@dataclass(frozen=True, slots=True)
class _CompressionPreparation:
    stages: tuple[bytes, ...]
    value: dict
    has_user_turn: bool
    pruned_skill_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _CompressionFrame:
    token: NativeAuxiliaryInput
    completion: object
    messages: list
    source_bytes: bytes
    parents: tuple
    parent_requests: tuple
    source_content: ContentHandle
    stages: tuple = ()
    stage_contents: tuple = ()
    request: dict | None = None
    request_bytes: bytes | None = None
    request_anchors: tuple = ()
    sealed: NativeAuxiliaryRequest | None = None
    record: NativeAuxiliaryInvocation | None = None
    preparation: _CompressionPreparation | None = None
    preparation_anchors: tuple = ()


@dataclass(frozen=True, slots=True)
class _PendingOperation:
    handle: object
    source: object
    selected: NativeMainSelection
    producer: NativeResultPolicy
    path: str
    route_ref: str
    payload: bytes
    content: object
    auxiliary: bool


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
    _response_shape_base(value, model)
    _require(value["choices"][0]["finish_reason"] == "stop")


def _compression_response_shape(value, model):
    _response_shape_base(value, model)
    _require(value["choices"][0]["finish_reason"] in ("stop", "length"))


def _response_shape_base(value, model):
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
    _require(type(choice["finish_reason"]) is str)
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
    value = _decode_response_payload(payload)
    _response_shape(value, model)
    return value


def _decode_compression_response(payload, model):
    value = _decode_response_payload(payload)
    _compression_response_shape(value, model)
    return value


def _decode_response_payload(payload):
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
    _require(_json(value) == payload)
    return value


def _response(value):
    choice = value["choices"][0]
    return NativeMainResponse(
        value["id"], value["object"], value["created"], value["model"],
        (NativeMainChoice(0, NativeMainMessage(**choice["message"]), choice["finish_reason"]),),
        NativeMainUsage(**value["usage"]),
    )


def _response_bytes(response, model):
    value = _response_value(response)
    _response_shape(value, model)
    return _json(value)


def _compression_response_bytes(response, model):
    value = _response_value(response)
    _compression_response_shape(value, model)
    return _json(value)


def _response_value(response):
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
    return value


def _method(callback):
    if type(callback) is MethodType:
        return callback.__self__, callback.__func__
    return (callback,)


def _same_method(current, captured):
    actual = _method(current)
    return len(actual) == len(captured) and all(a is b for a, b in zip(actual, captured))


def _route_facts(route):
    row = _row(route, shared_policy().RouteBinding)
    for index in (0, 1, 3, 7):
        _identifier(row[index])
    for index in (2, 6):
        _integer(row[index])
    _require(type(row[4]) is str and row[4] in shared_policy().PATHS)
    _destination(row[5])
    _require(type(row[8]) is str and row[8] == "synthetic_only")
    return row


def _destination_value(destination):
    return dict(zip(("transport_ref", "provider_ref", "model_ref", "endpoint_ref", "revision"),
                    _destination(destination)))


def _selection_value(selection):
    row = _selection(selection)
    return dict(selection_id=row[0], selection_version=row[1], api_mode=row[2],
                destination=_destination_value(selection.destination))


def _fallback_failed_bytes(failed):
    frame, facts = failed.turn_frame, failed.failure_facts
    payload = _json(dict(source=json.loads(failed.source_bytes),
        preparation=json.loads(frame.output_bytes), source_binding=_binding(failed.source),
        selection=_selection_value(failed.selection), producer_policy=_producer(failed.producer_policy),
        request=failed.request, normalized_request=json.loads(failed.request_bytes),
        prepared_request=json.loads(frame.request_bytes), failure=dict(failure_kind=facts.failure_kind,
            phase=facts.phase, send_count=facts.send_count, response_bytes=facts.response_bytes)))
    _require(len(payload) <= 262_144)
    return payload


def _fallback_result_bytes(result):
    keys = ("completed", "failed", "interrupted", "partial", "final_response", "messages", "fallback",
            "selected_provider", "selected_model", "provider", "model", "base_url", "api_calls",
            "main_attempt_usage", "verified_response_usage", "total_usage_known", "estimated_cost_usd", "cost_status")
    _keys(result, keys)
    for name, expected in (("completed", True), ("failed", False), ("interrupted", False),
                           ("partial", False), ("fallback", True), ("total_usage_known", False)):
        _require(result[name] is expected)
    _text(result["final_response"])
    for name in ("selected_provider", "selected_model", "provider", "model"):
        _identifier(result[name])
    for name, expected in (("base_url", ""), ("cost_status", "unknown")):
        _require(type(result[name]) is str and result[name] == expected)
    _require(result["main_attempt_usage"] is None and result["estimated_cost_usd"] is None)
    _require(type(result["api_calls"]) is int and result["api_calls"] == 2)
    _request(dict(model=result["model"], messages=result["messages"]))
    usage = result["verified_response_usage"]
    _keys(usage, ("prompt_tokens", "completion_tokens", "total_tokens"))
    for value in usage.values():
        _integer(value, 1_000_000, 0)
    _require(usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"])
    payload = _json(result)
    _require(len(payload) <= 262_144)
    return payload


class _Admission:
    def __init__(self, adapter):
        self.adapter = adapter

    def current(self, handle):
        adapter = self.adapter
        frame = _delegate_frames.get(adapter)
        if frame is not None:
            return frame.current(handle)
        _require(handle is adapter._handle or (adapter._aux_binding is not None
                 and handle is adapter._aux_binding.handle) or (adapter._compression_binding is not None
                 and handle is adapter._compression_binding.auxiliary.handle) or (adapter._fallback_binding is not None
                 and handle is adapter._fallback_binding.handle))
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
        frame = _delegate_frames.get(adapter)
        if frame is not None and frame.running:
            return frame.send(self, envelope)
        operation = adapter._operation
        _require(type(operation) is _PendingOperation and adapter._pending is None)
        _require(adapter._busy and operation is adapter._checked_operation)
        _require(type(envelope) is shared_policy().DispatchEnvelope)
        _binding(envelope.source)
        _destination(envelope.destination)
        _profiles(envelope.profiles)
        _require(type(envelope.payload) is bytes and len(envelope.payload) <= 262_144)
        _require(type(envelope.path) is str)
        _identifier(envelope.recipient_ref)
        adapter._checkpoint()
        _require(envelope.payload == operation.payload and envelope.path == operation.path)
        _require(envelope.source == operation.source)
        _require(envelope.destination == operation.selected.destination)
        _require(self is adapter._transport_wrappers[envelope.destination.transport_ref])
        _require(envelope.recipient_ref == operation.source.consumer_ref)
        actual = self.resolve()
        _require(actual == envelope.destination)
        adapter._interrupt_operation()
        adapter._pending = (envelope, actual, None)
        before = adapter._send_count
        if operation.path == "fallback":
            _integer(adapter._fallback_send_count, _MAX_INT - 1, 0)
            adapter._fallback_send_count += 1
        elif operation.auxiliary:
            _integer(adapter._aux_send_count, _MAX_INT - 1, 0)
            adapter._aux_send_count += 1
        else:
            _integer(adapter._send_count, _MAX_INT - 1, 0)
            adapter._send_count += 1
        try:
            output = self.port.send(envelope)
        except (ConnectionError, TimeoutError) as error:
            if (type(error) in (ConnectionError, TimeoutError) and operation.path == "main"
                    and adapter._fallback_binding is not None):
                adapter._failed_send_observation = _FailedSendObservation(
                    operation, envelope, actual, type(error), before, adapter._send_count,
                )
            raise
        adapter._pending = (envelope, actual, output)
        adapter._checkpoint()
        _require(type(output) is bytes and len(output) <= 65_536)
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
        _delegation_frame: object = None,
    ) -> None:
        if _delegation_frame is not None:
            _require(type(_delegation_frame) is _DelegateFrame)
            _delegation_frame.open_view(self, context)
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
        self._aux_binding = None
        self._compression_binding = None
        self._compression_owner = None
        self._fallback_binding = None
        self._fallback_frame = None
        self._fallback_frame_anchor = None
        self._fallback_records = self._fallback_anchors = ()
        self._failed_main_attempts = self._failed_main_anchors = ()
        self._fallback_send_count = 0
        self._failed_send_observation = None
        self._fallback_phase = None
        self._fallback_expected = None
        self._fallback_expected_anchor = None
        self._fallback_terminal = False
        self._aux_pending = None
        self._aux_frames = ()
        self._aux_records = ()
        self._aux_anchors = ()
        self._aux_local_results = ()
        self._aux_local_anchors = ()
        self._aux_send_count = 0
        self._operation = self._checked_operation = None
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
            self._operator_routes = routes
            self._operator_profiles = profiles
            self._transport_wrappers = dict(wrappers)
            self._operator_transports = dict(transports)
            self._operator_resolvers = tuple((port, _method(port.resolve)) for _, port in transports)
            self._operator_configuration = (routes, profiles, transports, tuple(wrappers))
            self._main_route = selected[0]
            self._busy = True
            self._checkpoint()
            if _delegation_frame is not None:
                self._runtime = _delegation_frame.runtime
            else:
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

    @property
    def failed_main_attempts(self) -> tuple[NativeFailedMainAttempt, ...]:
        return self._failed_main_attempts

    @property
    def fallback_invocations(self) -> tuple[NativeFallbackInvocation, ...]:
        return self._fallback_records

    @property
    def fallback_send_count(self) -> int:
        return self._fallback_send_count

    def _deny(self):
        self._invalidate()
        raise NativeContentPolicyDenied("controlled_native_payload_denied") from None

    def _invalidate(self):
        self._invalid = True
        frame = _delegate_frames.get(self) or getattr(self, "_delegate_parent_frame", None)
        if frame is not None:
            frame.invalidate()
        frame = self._aux_pending
        if type(frame) in (_AuxiliaryFrame, _CompressionFrame) and type(frame.record) is NativeAuxiliaryInvocation:
            self._accept_aux_record(frame.record)
        self._turn_pending = self._aux_pending = None
        self._operation = self._checked_operation = None
        if type(self._bound_context) is NativeContentContext:
            self._bound_context._denied = True

    def _identity(self):
        frame = _delegate_frames.get(self) or getattr(self, "_delegate_parent_frame", None)
        delegate_scope = False
        if frame is not None:
            frame.verify()
            delegate_scope = self is frame.parent and frame.child_scope()
        _require(threading.get_ident() == self._thread)
        if self._compression_binding is not None and self._turn_agent is not None:
            interrupted = _namespace(self._turn_agent).get("_interrupt_requested", False)
            _require(type(interrupted) is bool)
            if interrupted is True:
                raise InterruptedError("Agent interrupted during API call")
        if self._constructed is not None:
            _verify_constructed(self._constructed)
        context = self._context
        _require(context is self._bound_context)
        _require(not self._invalid and not context._denied)
        _require(type(self) is NativeMainPayloadAdapter)
        _require(self in _adapters)
        if "_runtime" in vars(self):
            _require(self._runtime is self._captured_runtime)
        if "_operator_configuration" in vars(self):
            routes, profiles, transports, wrappers = self._operator_configuration
            _require(self._operator_routes is routes and self._operator_profiles is profiles)
            _require(type(self._operator_transports) is dict and len(self._operator_transports) == len(transports))
            _require(type(self._transport_wrappers) is dict and len(self._transport_wrappers) == len(wrappers))
            for (name, port), (wrapped_name, wrapper) in zip(transports, wrappers):
                _require(self._operator_transports.get(name) is port)
                _require(self._transport_wrappers.get(wrapped_name) is wrapper)
                _require(wrapper.adapter is self and wrapper.port is port)
        if self._aux_binding is not None or hasattr(self, "_bound_aux_binding"):
            _require(hasattr(self, "_bound_aux_binding"))
            _require(self._aux_binding is self._bound_aux_binding)
            values, anchors = self._aux_binding_anchor
            _require(all(a is b for a, b in zip(_row(self._aux_binding, _AuxiliaryBinding), values)))
            _require(_same_anchors(_nested_anchors(values)[1:], anchors))
        if self._compression_binding is not None or hasattr(self, "_bound_compression_binding"):
            _require(hasattr(self, "_bound_compression_binding"))
            _require(type(self._compression_binding) is _CompressionBinding)
            _require(self._compression_binding is self._bound_compression_binding)
            _require(_same_anchors(_nested_anchors(self._compression_binding), self._compression_binding_anchor))
            _require(_same_anchors(_compression_constants(), self._compression_constants))
        if self._compression_owner is not None or hasattr(self, "_compression_owner_anchor"):
            _require(hasattr(self, "_compression_owner_anchor"))
            _require(_same_anchors(_nested_anchors(self._compression_options), self._compression_options_anchor))
            owner, namespace, payload, anchors = self._compression_owner_anchor
            _require(self._compression_owner is owner and vars(owner) is namespace)
            _require(namespace.get("_native_main_adapter") is self)
            _require(all(type(key) is str for key in namespace))
            state = {key: value for key, value in namespace.items() if key != "_native_main_adapter"}
            _require(all(type(value) in (str, int, float, bool, type(None)) for value in state.values()))
            _require(_json(state) == payload)
            _require(_same_anchors(_nested_anchors(namespace), anchors))
        _require(context._controlled_main_bound is True and context._controlled_main_payload is self)
        _require(threading.get_ident() == self._thread)
        actual = (context.admission, context.request_handle, self._selection_port,
                  self._result_port, self._clock)
        _require(all(a is b for a, b in zip(actual, self._captured)))
        _require(self._admission is self._captured[0] and self._handle is self._captured[1])
        if self._agent is not None and not delegate_scope:
            _require(current_context(self._agent) is context)
        if self._turn_agent is not None and not delegate_scope:
            _require(current_context(self._turn_agent) is context)
        for port, name, method in self._methods:
            _require(_same_method(getattr(port, name), method))
        if self._fallback_binding is not None or hasattr(self, "_bound_fallback_binding"):
            _require(self._fallback_binding is self._bound_fallback_binding)
            _require(_same_anchors(_nested_anchors(self._fallback_binding), self._fallback_binding_anchor))

    def _snapshot(self):
        self._identity()
        self._verify_records()
        self._verify_turns()
        self._verify_completions()
        self._verify_auxiliary()
        self._verify_fallback()
        if self._fallback_binding is not None:
            _require(_same_anchors(_nested_anchors(self._operator_configuration), self._fallback_catalog_anchor))
        source = self._admission.current(self._handle)
        self._identity()
        source_row = _binding(source)
        selected = self._selection_port.current(self._handle)
        self._identity()
        selection_row = _selection(selected)
        producer = self._result_port.current(self._producer_ref)
        producer_row = _producer(producer)
        self._identity()
        _require(producer_row[0] == self._producer_ref)
        _require(selection_row[:2] == (source_row[6], source_row[7]))
        _require(selection_row[3][1] == "synthetic_native")
        _destination(self._main_route.destination)
        _require(selected.destination == self._main_route.destination)
        retained_delegate_profiles = ()
        delegation = _delegate_frames.get(self) or getattr(self, "_delegate_parent_frame", None)
        if delegation is not None:
            delegation.verify()
            retained_delegate_profiles = delegation.initial[0 if self is delegation.parent else 2][0][8]
        _require(set(producer_row[2]) <= set((*source_row[8], *retained_delegate_profiles)) or bool(self._records))
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
        if self._aux_binding is not None:
            self._aux_snapshot(source_row)
        if self._compression_binding is not None:
            self._compression_snapshot(source_row)
        if self._fallback_binding is not None:
            self._fallback_snapshot(source_row)
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

    def _interrupt_operation(self):
        if self._turn_agent is not None:
            self._interrupt_turn()

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
            if self._compression_binding is not None and self._turn_agent is not None:
                interrupted = _namespace(self._turn_agent).get("_interrupt_requested", False)
                self._invalidate()
                if type(interrupted) is bool and interrupted is True:
                    raise InterruptedError("Agent interrupted during API call") from None
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
        if self._fallback_frame is not None or self._fallback_terminal:
            self._deny()
        self._preparation_enabled = True
        if self._turn_agent is None:
            self._turn_agent = agent
        arguments = (user_message, system_message, conversation_history, task_id)
        if not self._busy and self._turn_pending is None and self._turn_state is None:
            self._turn_arguments = arguments
        with self._turn_operation(agent):
            self._model_open()
            _require(self._turn_pending is None and self._aux_pending is None and len(self._records) < 8)
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
            delegation = getattr(self, "_delegate_parent_frame", None)
            if delegation is not None:
                delegation.check_child_input(agent, arguments)
                admitted = self._runtime.derive(self._handle, (delegation.record.child_seed,), raw, path="main")
            else:
                admitted = self._runtime.admit(self._handle, raw)
            parents = (admitted, *(r.result_content for r in self._records),
                       *(row.content for row in self._completions))
            content = (self._derive(self._handle, parents, raw, path="main")
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
            self._model_open()
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
            self._model_open()
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
            content = self._derive(self._handle, parents, encoded, path="main")
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
            self._model_open()
            frame = self._turn_pending
            _require(frame is not None and type(prepared) is NativePreparedTurn and prepared is frame.prepared)
            request = self._turn_request(frame)
            _require(_json(request) == frame.request_bytes)
            content = self._derive(
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

    @property
    def auxiliary_invocations(self) -> tuple[NativeAuxiliaryInvocation, ...]:
        return self._aux_records

    @property
    def auxiliary_send_count(self) -> int:
        return self._aux_send_count

    @property
    def auxiliary_local_results(self) -> tuple[NativeAuxiliaryLocalResult, ...]:
        return self._aux_local_results

    def _completed_auxiliary_outputs(self):
        for frame, _ in self._aux_frames:
            record = (frame.local_result if type(frame) is _AuxiliaryFrame
                      and frame.local_result is not None else frame.record)
            if record is not None and record.output_content is not None:
                yield record.output_content, record.request_handle

    def _derive(self, handle, parents, payload, *, path):
        completed = tuple(self._completed_auxiliary_outputs())
        previous = tuple(content for content, _ in completed)
        if not previous:
            return self._runtime.derive(handle, parents, payload, path=path)
        parents = (*parents, *previous)
        return self._runtime.derive_for_selection(
            handle, parents, payload, path=path,
            parent_requests=(*(self._handle for _ in parents[:-len(previous)]),
                             *(capability for _, capability in completed)),
        )

    def _callback_scope(self):
        _require(not self._invalid and not self._bound_context._denied)
        _require(threading.get_ident() == self._thread)
        _require(self._context is self._bound_context)
        _require(current_context(self._turn_agent) is self._bound_context)
        _require(_namespace(self._turn_agent).get("_native_controlled_binding") is self._constructed)

    def _model_open(self):
        _require(self not in _delegate_frames and not hasattr(self, "_delegate_frame"))
        _require(self._fallback_frame is None and self._fallback_terminal is False)

    def bind_delegation(
        self, agent: object, *, receiver_request: object,
        receiver_selection: NativeSelectionPort, receiver_producer_ref: str,
        child_request: object, child_selection: NativeSelectionPort,
        child_producer_ref: str, delegate_route_ref: str,
        child_main_route_ref: str, child_session_id: str, child_task_id: str,
    ) -> None:
        try:
            _require(self not in _delegate_frames)
            _DelegateFrame(self, agent, receiver_request, receiver_selection, receiver_producer_ref,
                child_request, child_selection, child_producer_ref, delegate_route_ref,
                child_main_route_ref, child_session_id, child_task_id)
        except BaseException:
            self._invalidate()
            raise

    def _verify_fallback(self):
        _integer(self._fallback_send_count, 1, 0)
        _require(type(self._fallback_terminal) is bool)
        _require(self._fallback_phase is None or (type(self._fallback_phase) is str
                 and self._fallback_phase in ("observe", "authenticate", "activate", "activating",
                                              "activated", "send", "complete", "released")))
        if self._fallback_expected is not None:
            _require(_same_anchors(_nested_anchors(self._fallback_expected), self._fallback_expected_anchor))
        for records, anchors, cls in (
            (self._failed_main_attempts, self._failed_main_anchors, NativeFailedMainAttempt),
            (self._fallback_records, self._fallback_anchors, NativeFallbackInvocation),
        ):
            _require(type(records) is tuple and type(anchors) is tuple)
            _require(len(records) == len(anchors) and len(records) <= 1)
            for record, anchor in zip(records, anchors):
                _require(type(record) is cls and record is anchor[0])
                _require(_same_anchors(_nested_anchors(record), anchor[1]))
        frame = self._fallback_frame
        if frame is not None:
            _require(type(frame) is _FallbackFrame)
            _require(frame is self._fallback_frame_anchor[0])
            _require(_same_anchors(_nested_anchors(frame), self._fallback_frame_anchor[1]))
            _require(frame.failed_main is self._failed_main_attempts[0])
            _require(frame.binding is self._fallback_binding)
            _require(type(frame.token) is NativeFallbackInput and type(frame.activated) is bool)
            if frame.request is not None:
                _request(frame.request)
                _require(_json(frame.request) == frame.request_bytes)
                _require(_same_anchors(_nested_anchors(frame.request), frame.request_anchors))
            if frame.invocation is not None:
                record = frame.invocation
                _require(record is self._fallback_records[0])
                if record.native_response is not None:
                    _require(_response_bytes(record.native_response, frame.binding.selected[3][2])
                             == record.response_bytes)
            if frame.result is not None:
                _require(_fallback_result_bytes(frame.result) == frame.result_bytes)
                _require(_same_anchors(_nested_anchors(frame.result), frame.result_anchors))

    def _retain_fallback_frame(self, frame):
        _require(type(frame) is _FallbackFrame)
        self._fallback_frame = frame
        self._fallback_frame_anchor = frame, _nested_anchors(frame)

    def _retain_fallback_invocation(self, record):
        _require(type(record) is NativeFallbackInvocation)
        self._fallback_records = (record,)
        self._fallback_anchors = ((record, _nested_anchors(record)),)
        self._retain_fallback_frame(replace(self._fallback_frame, invocation=record))

    def _fallback_snapshot(self, main):
        binding = self._fallback_binding
        target = self._admission.current(binding.handle)
        self._identity()
        target_row = _binding(target)
        selected = binding.port.current(binding.handle)
        self._identity()
        selected_row = _selection(selected)
        producer = self._result_port.current(binding.producer_ref)
        self._identity()
        producer_row = _producer(producer)
        _require(target_row[:6] == main[:6] and target_row[:8] == binding.holder[:8])
        _require(selected_row == binding.selected and selected_row[:2] == target_row[6:8])
        _require(producer_row == binding.producer and producer_row[2] == self._initial_producer[2])
        _require(set(producer_row[2]) <= set(binding.original_profiles))
        _require(set(producer_row[2]) <= set(self._original_profiles))
        for other in (self._aux_binding, self._compression_binding.auxiliary
                      if self._compression_binding is not None else None):
            if other is not None:
                _require(binding.handle is not other.handle and binding.port is not other.port)
                _require(binding.producer_ref != other.producer_ref and binding.selected[0] != other.selected[0])
        now = self._clock()
        _integer(now, _MAX_INT, 0)
        self._identity()
        for route in self._operator_routes:
            _route_facts(route)
            _require(now < route.expires_at)
            port = self._operator_transports[route.destination.transport_ref]
            method = next(method for original, method in self._operator_resolvers if port is original)
            self._callback_scope()
            _require(_same_method(port.resolve, method))
            actual = port.resolve()
            self._callback_scope()
            _require(_same_method(port.resolve, method))
            _destination(actual)
            _require(actual == route.destination)
        declared = tuple(route.path for route in self._operator_routes
            if route.selection_id == selected.selection_id
            and route.selection_version == selected.selection_version and route.consumer_ref == target.consumer_ref)
        _require(len(declared) == len(shared_policy().PATHS) and set(declared) == set(shared_policy().PATHS))
        facts = target_row, selected_row, producer_row
        if self._fallback_expected is not None:
            _require(facts == self._fallback_expected)
        self._fallback_current = target, selected, producer

    def bind_fallback(
        self, agent: object, *, request_handle: object,
        selection: NativeSelectionPort, producer_ref: str, route_ref: str,
    ) -> None:
        with self._turn_operation(agent):
            self._model_open()
            _require(self._constructed is not None and self._fallback_binding is None)
            _require(self._turn_pending is None and self._aux_pending is None)
            _identifier(producer_ref)
            _identifier(route_ref)
            _require(request_handle is not self._handle and selection is not self._selection_port)
            _require(producer_ref != self._producer_ref)
            from agent import chat_completion_helpers as chat
            from run_agent import AIAgent
            methods = (
                (selection, "current"), (AIAgent, "_try_activate_fallback"),
                (chat, "try_activate_fallback"),
                *((self._runtime, name) for name in
                  ("dispatch_attempt", "failure_facts", "derive_for_selection", "dispatch")),
                *((self, name) for name in (
                    "bind_fallback", "_begin_fallback", "_activate_fallback", "_dispatch_fallback",
                    "_complete_fallback_turn", "_verify_fallback", "_fallback_snapshot",
                    "_fallback_derive", "_retain_fallback_frame", "_retain_fallback_invocation",
                    "_model_open", "dispatch", "_dispatch_retained_operation", "_identity", "_snapshot",
                    "_checkpoint", "_around", "_interrupt_turn", "_interrupt_operation", "_turn_operation")),
                *((wrapper, name) for wrapper in self._transport_wrappers.values() for name in ("resolve", "send")),
                *((sys.modules[__name__], name) for name in (
                    "_fallback_result_bytes", "_fallback_failed_bytes", "_route_facts",
                    "_nested_anchors", "_same_anchors", "_response_bytes", "_decode_response",
                    "_controlled_fallback_activation", "_destination_value", "_selection_value")),
            )
            self._methods += tuple((port, name, _method(getattr(port, name))) for port, name in methods)
            target = self._around(lambda: self._admission.current(request_handle))
            holder = _binding(target)
            selected = self._around(lambda: selection.current(request_handle))
            selected_row = _selection(selected)
            producer = self._around(lambda: self._result_port.current(producer_ref))
            producer_row = _producer(producer)
            _require(holder[:6] == self._holder[:6] and holder[6] != self._holder[6])
            _require(selected_row[:2] == holder[6:8] and selected_row[3][1] == "synthetic_native")
            _require(producer_row[0] == producer_ref and producer_row[2] == self._initial_producer[2])
            _require(set(producer_row[2]) <= set(holder[8]))
            for other in (self._aux_binding, self._compression_binding.auxiliary
                          if self._compression_binding is not None else None):
                if other is not None:
                    _require(request_handle is not other.handle and selection is not other.port)
                    _require(producer_ref != other.producer_ref and holder[6] != other.selected[0])
            routes = tuple(route for route in self._operator_routes if route.route_ref == route_ref)
            _require(len(routes) == 1)
            route = routes[0]
            _route_facts(route)
            _require(route.path == "fallback" and route.selection_id == selected.selection_id)
            _require(route.selection_version == selected.selection_version)
            _require(route.consumer_ref == target.consumer_ref and route.destination == selected.destination)
            self._fallback_binding = _FallbackBinding(request_handle, selection, producer_ref,
                route, holder, selected_row, producer_row, holder[8])
            self._bound_fallback_binding = self._fallback_binding
            self._fallback_binding_anchor = _nested_anchors(self._fallback_binding)
            self._fallback_catalog_anchor = _nested_anchors(self._operator_configuration)
            self._interrupt_turn()

    def _begin_fallback(self, agent: object, *, failure: FailedDispatch) -> NativeFallbackInput:
        _require(type(self._fallback_phase) is str)
        _require(self._busy and self._turn_agent is agent and self._fallback_phase == "observe")
        _require(self._fallback_binding is not None and self._fallback_frame is None)
        self._fallback_phase = "authenticate"
        self._interrupt_turn()
        operation, observation, frame = self._operation, self._failed_send_observation, self._turn_pending
        _require(type(operation) is _PendingOperation and operation is self._checked_operation)
        _require(operation.path == "main" and operation.handle is self._handle)
        _require(type(observation) is _FailedSendObservation and observation.operation is operation)
        _require(frame is not None and frame.prepared is not None and self._pending is not None)
        envelope, actual, output = self._pending
        _require(envelope is observation.envelope and actual is observation.observed_destination and output is None)
        _require(observation.exception_type is ConnectionError or observation.exception_type is TimeoutError)
        _integer(observation.send_count_before, _MAX_INT - 1, 0)
        _integer(observation.send_count_after, _MAX_INT, 1)
        _require(observation.send_count_before + 1 == observation.send_count_after == self._send_count)
        _require(type(failure) is shared_policy().FailedDispatch)
        facts = self._runtime.failure_facts(self._handle, failure)
        _row(facts, shared_policy().DispatchFailureFacts)
        _require(facts.request_content is operation.content and facts.envelope is envelope)
        _require(facts.source is envelope.source and facts.profiles is envelope.profiles)
        _destination(facts.observed_destination)
        _require(facts.observed_destination == actual == operation.selected.destination)
        _require(type(facts.route_ref) is str and facts.route_ref == operation.route_ref)
        _require(type(facts.failure_kind) is str and facts.failure_kind ==
                 ("connection_error" if observation.exception_type is ConnectionError else "timeout_error"))
        _require(type(facts.phase) is str and facts.phase == "send")
        _require(type(facts.send_count) is int and facts.send_count == 1 and facts.response_bytes is None)
        parents = (frame.admitted, frame.input_content, frame.output_content,
                   *frame.request_contents, operation.content, facts.failure_content)
        capabilities = tuple(self._handle for _ in parents)
        auxiliary_outputs = []
        latest_main = -1
        for auxiliary, _ in self._aux_frames:
            record = (auxiliary.local_result if type(auxiliary) is _AuxiliaryFrame
                      and auxiliary.local_result is not None else auxiliary.record)
            _require(record is not None and record.output_content is not None)
            present = tuple(index for index, completed in enumerate(self._completions)
                            if any(parent is completed.content for parent in auxiliary.parents))
            _require(present and present == tuple(range(present[-1] + 1)))
            _require(present[-1] >= latest_main)
            latest_main = present[-1]
            auxiliary_outputs.append((latest_main, record.output_content, record.request_handle))
        for index, completion in enumerate(self._completions):
            earlier = completion.frame
            prior = (earlier.admitted, earlier.input_content, earlier.output_content,
                     *earlier.request_contents, completion.invocation.request_content,
                     completion.invocation.result_content, completion.content)
            parents += prior
            capabilities += tuple(self._handle for _ in prior)
            for preceding_main, content, capability in auxiliary_outputs:
                if preceding_main == index:
                    parents += (content,)
                    capabilities += (capability,)
        failed = NativeFailedMainAttempt(frame, self._handle, operation.source, operation.selected,
            operation.producer, frame.raw_bytes, parents, capabilities, self._kwargs,
            operation.payload, operation.content, observation, failure, facts, facts.failure_content)
        self._failed_main_attempts = (failed,)
        self._failed_main_anchors = ((failed, _nested_anchors(failed)),)
        token = object.__new__(NativeFallbackInput)
        self._retain_fallback_frame(_FallbackFrame(token, failed, self._fallback_binding, parents, capabilities, None))
        self._interrupt_turn()
        payload = _fallback_failed_bytes(failed)
        content = self._runtime.derive_for_selection(self._fallback_binding.handle, parents, payload,
            path="fallback", parent_requests=capabilities)
        self._retain_fallback_frame(replace(self._fallback_frame, source_content=content))
        request = dict(self._kwargs, model=self._fallback_binding.selected[3][2])
        _request(request)
        stages = (payload, _json(dict(selection=_selection_value(self._fallback_current[1]), request=request)),
                  _json(dict(api_mode="chat_completions", destination=_destination_value(self._fallback_current[1].destination), request=request)))
        _stage_tuple(stages)
        contents = ()
        for stage in stages:
            contents += (self._fallback_derive(stage, (content, *contents)),)
            self._retain_fallback_frame(replace(self._fallback_frame,
                input_stages=stages[:len(contents)], stage_contents=contents))
            self._interrupt_turn()
        self._retain_fallback_frame(replace(self._fallback_frame, input_stages=stages, stage_contents=contents,
            request=request, request_bytes=_json(request), request_anchors=_nested_anchors(request)))
        self._operation = self._checked_operation = self._pending = None
        self._fallback_phase = "activate"
        _require(agent._try_activate_fallback(controlled_failure=token) is True)
        _require(self._fallback_phase == "activated" and self._fallback_frame.activated is True)
        return token

    def _fallback_derive(self, payload, additional):
        _require(type(payload) is bytes and len(payload) <= 262_144)
        frame = self._fallback_frame
        self._interrupt_turn()
        content = self._runtime.derive_for_selection(frame.binding.handle,
            (*frame.parents, *additional), payload, path="fallback",
            parent_requests=(*frame.parent_requests, *(frame.binding.handle for _ in additional)))
        return content

    def _activate_fallback(self, agent: object, token: NativeFallbackInput) -> bool:
        try:
            frame = self._fallback_frame
            _require(type(self._fallback_phase) is str and type(frame) is _FallbackFrame)
            _require(self._busy and self._turn_agent is agent and self._fallback_phase == "activate")
            _require(type(token) is NativeFallbackInput and frame is not None and token is frame.token)
            _require(frame.activated is False and frame.invocation is None and self._operation is None)
            self._fallback_phase = "activating"
            self._interrupt_turn()
            self._retain_fallback_frame(replace(frame, activated=True))
            self._interrupt_turn()
            self._fallback_phase = "activated"
            return True
        except BaseException:
            self._invalidate()
            raise

    def _dispatch_fallback(self, agent: object, token: NativeFallbackInput) -> NativeMainResponse:
        frame = self._fallback_frame
        _require(type(self._fallback_phase) is str and type(frame) is _FallbackFrame)
        _require(self._busy and self._turn_agent is agent and self._fallback_phase == "activated")
        _require(type(token) is NativeFallbackInput and token is frame.token and frame.activated is True)
        _require(frame.invocation is None and self._operation is None and self._pending is None)
        self._fallback_phase = "send"
        self._interrupt_turn()
        source, selected, producer = self._fallback_current
        normalized = frame.input_stages[-1]
        _require(normalized == _json(dict(api_mode="chat_completions",
            destination=_destination_value(selected.destination), request=frame.request)))
        content = self._fallback_derive(normalized, (frame.source_content, *frame.stage_contents))
        operation = _PendingOperation(frame.binding.handle, source, selected, producer,
            "fallback", frame.binding.route.route_ref, normalized, content, False)
        record = NativeFallbackInvocation(frame.failed_main, frame.binding.handle, source, selected, producer,
            frame.parents, frame.parent_requests, frame.input_stages, frame.stage_contents,
            frame.request, frame.request_bytes, content)
        self._retain_fallback_invocation(record)
        self._operation = self._checked_operation = operation
        self._dispatch_result = self._dispatch_response = None
        try:
            result, envelope, actual, output, response = self._dispatch_retained_operation(operation)
        finally:
            pending = self._pending
            if pending is not None:
                envelope, actual, output = pending
                record = replace(record, envelope=envelope, observed_destination=actual,
                    response_bytes=output if type(output) is bytes and len(output) <= 65_536 else None,
                    result_content=self._dispatch_result, native_response=self._dispatch_response)
                self._retain_fallback_invocation(record)
            self._operation = self._checked_operation = self._pending = None
        _require(result is record.result_content and response is record.native_response)
        self._interrupt_turn()
        self._fallback_phase = "complete"
        return response

    def _complete_fallback_turn(self, agent: object, *, turn_context: TurnContext,
                                response: NativeMainResponse) -> dict:
        with self._turn_operation(agent):
            frame = self._fallback_frame
            _require(frame is not None and self._fallback_phase == "complete" and not self._fallback_terminal)
            _require(frame.failed_main.turn_frame is self._turn_pending)
            _require(turn_context is frame.failed_main.turn_frame.output)
            record = frame.invocation
            _require(response is record.native_response and record.result_content is not None)
            _require(_response_bytes(response, frame.binding.selected[3][2]) == record.response_bytes)
            _require(bool(response.choices[0].message.content))
            result = dict(completed=True, failed=False, interrupted=False, partial=False,
                final_response=response.choices[0].message.content,
                messages=[*(dict(row) for row in frame.request["messages"]),
                          dict(role="assistant", content=response.choices[0].message.content)],
                fallback=True, selected_provider=frame.failed_main.selection.destination.provider_ref,
                selected_model=frame.failed_main.selection.destination.model_ref,
                provider=record.observed_destination.provider_ref, model=response.model, base_url="",
                api_calls=2, main_attempt_usage=None, verified_response_usage=dict(
                    prompt_tokens=response.usage.prompt_tokens, completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens), total_usage_known=False,
                estimated_cost_usd=None, cost_status="unknown")
            payload = _fallback_result_bytes(result)
            stages = (_json(dict(raw_response=json.loads(record.response_bytes), native_response=_response_value(response))), payload)
            _stage_tuple(stages)
            contents = ()
            base = (frame.source_content, *frame.stage_contents, record.request_content, record.result_content)
            for stage in stages:
                contents += (self._fallback_derive(stage, (*base, *contents)),)
                record = replace(record, output_stages=stages[:len(contents)], output_contents=contents)
                self._retain_fallback_invocation(record)
                self._interrupt_turn()
            content = self._fallback_derive(payload, (*base, *contents))
            record = replace(record, output=result, output_bytes=payload, output_content=content)
            self._retain_fallback_invocation(record)
            self._retain_fallback_frame(replace(self._fallback_frame, result=result, result_bytes=payload,
                result_content=content, result_anchors=_nested_anchors(result)))
            self._interrupt_turn()
            self._fallback_terminal = True
            self._fallback_phase = "released"
            self._turn_pending = None
            self._turn_arguments = self._turn_raw = self._turn_state = self._turn_options = self._turn_facts = None
            self._interrupt_turn()
            _require(result is self._fallback_frame.result)
            return result

    def _aux_snapshot(self, main):
        binding = self._aux_binding
        from agent import title_generator as titles
        _require(titles._controlled_title_constants() == self._title_constants)
        target = self._admission.current(binding.handle)
        self._identity()
        target_row = _binding(target)
        selected = binding.port.current(binding.handle)
        self._identity()
        selection_row = _selection(selected)
        producer = self._result_port.current(binding.producer_ref)
        producer_row = _producer(producer)
        self._identity()
        _require(target_row[:6] == main[:6])
        _require(target_row[:8] == binding.source[:8])
        _require(selection_row == binding.selected)
        _require(selection_row[:2] == target_row[6:8])
        _require(producer_row == binding.producer)
        _require(producer_row[2] == self._initial_producer[2])
        _require(set(producer_row[2]) <= set(binding.original_profiles))
        _require(set(producer_row[2]) <= set(self._original_profiles))
        _require(selected.destination == binding.route.destination)
        now = self._clock()
        _integer(now, _MAX_INT, 0)
        self._identity()
        if self._compression_binding is None:
            for route in self._operator_routes:
                _require(type(route) is shared_policy().RouteBinding)
                _integer(route.expires_at)
                _require(now < route.expires_at)
                port = self._operator_transports[route.destination.transport_ref]
                method = next(method for original, method in self._operator_resolvers if port is original)
                self._callback_scope()
                _require(_same_method(port.resolve, method))
                actual = port.resolve()
                self._callback_scope()
                _require(_same_method(port.resolve, method))
                _destination(actual)
                _require(actual == route.destination)
        self._identity()
        self._aux_current = target, selected, producer

    def _verify_auxiliary(self):
        _integer(self._send_count, _MAX_INT, 0)
        _integer(self._aux_send_count, _MAX_INT, 0)
        _require(type(self._aux_records) is tuple and len(self._aux_records) <= 8)
        _require(len(self._aux_records) == len(self._aux_anchors))
        for record, anchor in zip(self._aux_records, self._aux_anchors):
            _require(record is anchor[0])
            actual = _row(record, NativeAuxiliaryInvocation)
            _require(all(a is b for a, b in zip(actual, anchor[1])))
            _require(_same_anchors(_nested_anchors(actual)[1:], anchor[2]))
            if record.native_response is not None:
                compression = any(type(frame) is _CompressionFrame and frame.record is record
                                  for frame, _ in self._aux_frames)
                verify = _compression_response_bytes if compression else _response_bytes
                _require(verify(record.native_response, record.selection.destination.model_ref) == record.response_bytes)
            if record.output is not None:
                _require(_json(record.output) == record.output_bytes)
        _require(type(self._aux_local_results) is tuple and len(self._aux_local_results) <= 8)
        _require(len(self._aux_local_results) == len(self._aux_local_anchors))
        for record, anchor in zip(self._aux_local_results, self._aux_local_anchors):
            _require(record is anchor[0])
            actual = _row(record, NativeAuxiliaryLocalResult)
            _require(all(a is b for a, b in zip(actual, anchor[1])))
            _require(_same_anchors(_nested_anchors(actual)[1:], anchor[2]))
            _require(_json(record.output) == record.output_bytes)
        _require(type(self._aux_frames) is tuple and len(self._aux_frames) <= 8)
        for frame, anchor in self._aux_frames:
            _require(type(frame) in (_AuxiliaryFrame, _CompressionFrame))
            actual = _row(frame, type(frame))
            _require(all(a is b for a, b in zip(actual, anchor[0])))
            _require(_same_anchors(_nested_anchors(actual)[1:], anchor[1]))
            _require(any(frame.completion is row for row in self._completions))
            if type(frame) is _CompressionFrame:
                _require(frame.messages is frame.completion.result["messages"])
                _require(_json(frame.messages) == frame.source_bytes)
                if frame.preparation is not None:
                    preparation = frame.preparation
                    _require(type(preparation) is _CompressionPreparation)
                    _require(_same_anchors(_nested_anchors(preparation), frame.preparation_anchors))
                    _require(len(preparation.stages) == 7)
                    _stage_tuple(preparation.stages)
            else:
                selected = (frame.completion.frame.arguments[0] if frame.field == "user_message"
                            else frame.completion.result["final_response"])
                _require(frame.text is selected and _json(selected) == frame.source_bytes)
                if frame.preparation is not None:
                    preparation = frame.preparation
                    _require(type(preparation) is _AuxiliaryTitlePreparation)
                    _require(preparation.kind in ("dispatch", "local_no_send"))
                    _require(_same_anchors(_nested_anchors(preparation), frame.preparation_anchors))
                    _stage_tuple(preparation.stages)
                if frame.local_result is not None:
                    _require(frame.record is None and frame.sealed is None and frame.request is None)
                    _require(any(frame.local_result is record for record in self._aux_local_results))
            if frame.record is not None and frame.record.native_response is not None:
                verify = _compression_response_bytes if type(frame) is _CompressionFrame else _response_bytes
                _require(verify(frame.record.native_response,
                    frame.record.selection.destination.model_ref) == frame.record.response_bytes)

    def _retain_aux_frame(self, frame):
        values = _row(frame, type(frame))
        entry = (frame, (values, _nested_anchors(values)[1:]))
        if self._aux_pending is None:
            self._aux_frames += (entry,)
        else:
            _require(self._aux_frames[-1][0] is self._aux_pending)
            self._aux_frames = (*self._aux_frames[:-1], entry)
        self._aux_pending = frame

    def _accept_aux_record(self, record):
        if any(original is record for original in self._aux_records):
            return
        _require(len(self._aux_records) < 8)
        values = _row(record, NativeAuxiliaryInvocation)
        self._aux_records += (record,)
        self._aux_anchors += ((record, values, _nested_anchors(values)[1:]),)

    def _retain_aux_record(self, record, frame=None):
        frame = self._aux_pending if frame is None else frame
        _require(not any(original is frame.record for original in self._aux_records))
        if record.output is not None or self._invalid:
            self._accept_aux_record(record)
        if self._aux_pending is None:
            updated = replace(frame, record=record)
            values = _row(updated, type(updated))
            self._aux_frames = (*self._aux_frames[:-1],
                                (updated, (values, _nested_anchors(values)[1:])))
        else:
            self._retain_aux_frame(replace(frame, record=record))

    def bind_auxiliary(
        self, agent: object, *, path: str, task: str,
        request_handle: object, selection: NativeSelectionPort,
        producer_ref: str, route_ref: str, language: str,
    ) -> None:
        with self._turn_operation(agent):
            self._model_open()
            _require(self._constructed is not None and self._aux_binding is None)
            _require(self._turn_pending is None and self._aux_pending is None)
            _require(type(path) is str and path == "title")
            _require(type(task) is str and task == "title_generation")
            _identifier(producer_ref)
            _identifier(route_ref)
            _require(type(language) is str and len(language) <= 128)
            _require(not any(0xD800 <= ord(c) <= 0xDFFF for c in language))
            _require(request_handle is not self._handle)
            if self._compression_binding is not None:
                other = self._compression_binding.auxiliary
                _require(request_handle is not other.handle and selection is not other.port)
                _require(producer_ref != other.producer_ref)
            _require(callable(getattr(self._runtime, "derive_for_selection", None)))
            from agent import title_generator as titles, auxiliary_client as auxiliary
            from agent import skill_commands, agent_runtime_helpers
            self._methods += tuple((port, name, _method(getattr(port, name))) for port, name in (
                (selection, "current"), (self._runtime, "derive_for_selection"),
                (self._runtime, "dispatch"), (self._runtime, "derive"),
                *((self, name) for name in (
                    "bind_auxiliary", "auxiliary_input", "seal_auxiliary", "dispatch_auxiliary",
                    "complete_auxiliary", "complete_auxiliary_local", "_prepare_auxiliary_title",
                    "_completed_auxiliary_outputs", "_transform_auxiliary", "_dispatch_retained_operation",
                    "_verify_auxiliary", "_aux_snapshot", "_callback_scope", "_accept_aux_record")),
                *((titles, name) for name in (
                    "generate_title", "_controlled_title_constants", "_title_input_stages", "_title_output_stages",
                    "_title_preparation_stages",
                    "_summarize_user_message", "strip_control_wrappers", "_extract_title_text", "_clean_title",
                    "_title_snippet", "_title_prompt", "_title_word_decision")),
                (skill_commands, "describe_skill_invocation"),
                (agent_runtime_helpers, "strip_think_blocks"),
                (auxiliary, "call_llm"),
                (auxiliary.call_llm, "__wrapped__"),
                *((sys.modules[__name__], name) for name in (
                    "_try_controlled_auxiliary_call", "_controlled_title_entry", "_auxiliary_attempt",
                    "_auxiliary_owner", "_call_arguments_bytes", "_stage_tuple", "_response_bytes", "_decode_response", "_response", "_response_shape",
                    "_compression_call_arguments_bytes", "_native_result_anchors", "_native_result_bytes",
                    "_response_shape_base", "_decode_response_payload", "_decode_compression_response",
                    "_compression_response_shape", "_compression_response_bytes", "_response_value")),
            ))
            self._title_constants = titles._controlled_title_constants()
            target = self._around(lambda: self._admission.current(request_handle))
            target_row = _binding(target)
            selected = self._around(lambda: selection.current(request_handle))
            selection_row = _selection(selected)
            producer = self._around(lambda: self._result_port.current(producer_ref))
            producer_row = _producer(producer)
            _require(target_row[:6] == self._holder[:6])
            _require(target_row[6:8] != self._holder[6:8])
            _require(selection_row[:2] == target_row[6:8])
            if self._compression_binding is not None:
                _require(selection_row[0] != self._compression_binding.auxiliary.selected[0])
            _require(selection_row[3][1] == "synthetic_native")
            _require(producer_row[0] == producer_ref and producer_row[2] == self._initial_producer[2])
            _require(set(producer_row[2]) <= set(target_row[8]))
            routes = tuple(r for r in self._operator_routes if r.route_ref == route_ref)
            _require(len(routes) == 1)
            route = routes[0]
            _require(route.path == path and route.selection_id == selected.selection_id)
            _require(route.selection_version == selected.selection_version)
            _require(route.consumer_ref == target.consumer_ref and route.destination == selected.destination)
            self._aux_binding = _AuxiliaryBinding(request_handle, selection, producer_ref, route,
                language, target_row, selection_row, producer_row, target_row[8])
            self._bound_aux_binding = self._aux_binding
            values = _row(self._aux_binding, _AuxiliaryBinding)
            self._aux_binding_anchor = (values, _nested_anchors(values)[1:])
            self._interrupt_turn()

    def auxiliary_input(self, agent: object, *, source: dict, field: str) -> NativeAuxiliaryInput:
        with self._turn_operation(agent):
            self._model_open()
            _require(self._aux_binding is not None and self._turn_pending is None and self._aux_pending is None)
            _require(len(self._aux_frames) < 8)
            _require(type(source) is dict and type(field) is str)
            _require(field in ("user_message", "final_response"))
            completion = next((row for row in self._completions if row.result is source), None)
            _require(completion is not None)
            frame, invocation = completion.frame, completion.invocation
            text = frame.arguments[0] if field == "user_message" else source["final_response"]
            _text(text)
            payload = _json(text)
            _require(len(payload) <= 262_144)
            completed = tuple(self._completed_auxiliary_outputs())
            parents = (frame.admitted, frame.input_content, frame.output_content,
                       *frame.request_contents, invocation.request_content, invocation.result_content,
                       completion.content, *(row.content for row in self._completions
                                             if row is not completion),
                       *(content for content, _ in completed))
            previous = len(completed)
            parent_requests = (*(self._handle for _ in parents[:len(parents)-previous]),
                               *(capability for _, capability in completed))
            content = self._runtime.derive_for_selection(self._aux_binding.handle, parents,
                payload, path="title", parent_requests=parent_requests)
            self._interrupt_turn()
            token = object.__new__(NativeAuxiliaryInput)
            self._retain_aux_frame(_AuxiliaryFrame(token, completion, field, text, payload,
                                                  parents, parent_requests, content))
            return token

    def _checked_aux_frame(self, token, *, sealed=False):
        frame = self._aux_pending
        _require(type(frame) in (_AuxiliaryFrame, _CompressionFrame) and self._turn_pending is None)
        binding, _, _ = self._aux_frame_binding(frame)
        _require(binding is not None)
        expected = frame.sealed if sealed else frame.token
        cls = NativeAuxiliaryRequest if sealed else NativeAuxiliaryInput
        _require(type(token) is cls and token is expected)
        if type(frame) is _AuxiliaryFrame:
            from agent import title_generator as titles
            _require(titles._controlled_title_constants() == self._title_constants)
        self._interrupt_turn()
        return frame

    def _prepare_auxiliary_title(self, token):
        from agent import title_generator as titles
        with self._turn_operation(self._turn_agent):
            frame = self._checked_aux_frame(token)
            _require(type(frame) is _AuxiliaryFrame)
            _require(frame.sealed is None and frame.preparation is None)
            kind, stages, value = titles._title_preparation_stages(
                frame.text, self._aux_binding.language, self._aux_current[1], self._interrupt_turn)
            _require(kind in ("dispatch", "local_no_send"))
            _stage_tuple(stages)
            preparation = _AuxiliaryTitlePreparation(kind, stages, value)
            self._interrupt_turn()
            self._retain_aux_frame(replace(frame, preparation=preparation,
                preparation_anchors=_nested_anchors(preparation)))
            self._interrupt_turn()
            return preparation

    def _transform_auxiliary(self, token, *, response=None):
        from agent import title_generator as titles
        if response is None:
            preparation = self._prepare_auxiliary_title(token)
            with self._turn_operation(self._turn_agent):
                self._checked_aux_frame(token)
                _require(preparation.kind == "dispatch")
                return preparation.stages, preparation.value
        with self._turn_operation(self._turn_agent):
            frame = self._checked_aux_frame(token, sealed=response is not None)
            _require(type(frame) is _AuxiliaryFrame)
            _require(frame.record is not None and response is frame.record.native_response)
            stages, output = titles._title_output_stages(response, self._interrupt_turn)
            self._aux_output = (output, stages, output["title"])
            return stages, output

    def seal_auxiliary(self, agent: object, source: NativeAuxiliaryInput, *,
                       stages: tuple[bytes, ...], request: dict) -> NativeAuxiliaryRequest:
        with self._turn_operation(agent):
            frame = self._checked_aux_frame(source)
            _require(frame.sealed is None)
            proof = frame.preparation
            binding, path, task = self._aux_frame_binding(frame)
            if type(frame) is _CompressionFrame:
                _require(type(proof) is _CompressionPreparation)
            else:
                _require(type(proof) is _AuxiliaryTitlePreparation and proof.kind == "dispatch")
            original_request = proof.value["request"] if type(frame) is _CompressionFrame else proof.value
            _require(proof.stages is stages and original_request is request)
            _require(_same_anchors(_nested_anchors(proof), frame.preparation_anchors))
            _stage_tuple(stages)
            contents = []
            for stage in stages:
                self._interrupt_turn()
                contents.append(self._runtime.derive(binding.handle,
                    (frame.source_content, *contents), stage, path=path))
            payload = _json(dict(path=path, task=task,
                selection=list(binding.selected[:3]),
                destination=dict(zip(("transport_ref", "provider_ref", "model_ref", "endpoint_ref", "revision"),
                                     binding.selected[3])), request=request))
            _require(len(payload) <= 262_144)
            token = object.__new__(NativeAuxiliaryRequest)
            self._retain_aux_frame(replace(frame, stages=stages, stage_contents=tuple(contents),
                request=request, request_bytes=payload, request_anchors=_nested_anchors(request), sealed=token))
            self._interrupt_turn()
            return token

    def _dispatch_retained_operation(self, operation):
        _require(type(operation) is _PendingOperation and operation is self._operation)
        _require(operation is self._checked_operation and self._busy and self._pending is None)
        self._interrupt_operation()
        self._dispatch_result = self._dispatch_response = None
        dispatcher = (self._runtime.dispatch_attempt if self._fallback_binding is not None
                      and operation.path == "main" else self._runtime.dispatch)
        result = dispatcher(operation.handle, operation.content,
                            path=operation.path, route_ref=operation.route_ref)
        if self._fallback_binding is not None and type(result) is shared_policy().FailedDispatch:
            return result
        self._dispatch_result = result
        self._checkpoint()
        _require(self._pending is not None)
        envelope, actual, output = self._pending
        decoder = _decode_compression_response if type(self._aux_pending) is _CompressionFrame else _decode_response
        value = decoder(output, operation.selected.destination.model_ref)
        response = _response(value)
        self._dispatch_response = response
        self._checkpoint()
        return result, envelope, actual, output, response

    def dispatch_auxiliary(self, request: NativeAuxiliaryRequest, *, arguments: dict) -> NativeMainResponse:
        with self._turn_operation(self._turn_agent):
            frame = self._checked_aux_frame(request, sealed=True)
            _require(frame.record is None and self._operation is None)
            _keys(arguments, (*_AUX_CALL_DEFAULTS, "controlled_aux"))
            _require(arguments.pop("controlled_aux") is request)
            _require(_same_anchors(_nested_anchors(frame.request), frame.request_anchors))
            original_arguments = (frame.preparation.value["arguments"] if type(frame) is _CompressionFrame
                                  else frame.request)
            _require(all(arguments[name] is original_arguments[name] for name in _AUX_CALL_DEFAULTS))
            binding, path, task = self._aux_frame_binding(frame)
            normalizer = (_compression_call_arguments_bytes if type(frame) is _CompressionFrame
                          else _call_arguments_bytes)
            _require(normalizer(arguments) == normalizer(original_arguments))
            if type(frame) is _CompressionFrame:
                _require(frame.request is frame.preparation.value["request"])
                _require(frame.request["messages"] is original_arguments["messages"])
                _require(_json(_compression_wire(arguments)) == _json(frame.request))
                _require(_json(frame.preparation.value) == frame.stages[-1])
            content = self._runtime.derive(binding.handle,
                (frame.source_content, *frame.stage_contents), frame.request_bytes, path=path)
            source, selected, producer = (self._compression_current if type(frame) is _CompressionFrame
                                          else self._aux_current)
            operation = _PendingOperation(binding.handle, source, selected, producer,
                path, binding.route.route_ref, frame.request_bytes, content, True)
            self._operation = self._checked_operation = operation
            result = response = None
            try:
                result, envelope, actual, output, response = self._dispatch_retained_operation(operation)
            finally:
                if self._pending is not None:
                    envelope, actual, output = self._pending
                    record = NativeAuxiliaryInvocation(frame.completion, frame.completion.frame,
                        frame.completion.invocation, self._handle, binding.handle,
                        self._source, source, path, task, selected, producer,
                        frame.source_bytes, frame.parents, frame.parent_requests, frame.stages,
                        frame.stage_contents, frame.request_bytes, content, envelope, actual,
                        output if type(output) is bytes and len(output) <= 65_536 else None,
                        self._dispatch_result, self._dispatch_response)
                    self._retain_aux_record(record, frame)
                self._operation = self._checked_operation = self._pending = None
            self._interrupt_turn()
            return response

    def complete_auxiliary(self, request: NativeAuxiliaryRequest, *, response: NativeMainResponse,
                           stages: tuple[bytes, ...], output: dict) -> dict:
        from agent import title_generator as titles
        with self._turn_operation(self._turn_agent):
            frame = self._checked_aux_frame(request, sealed=True)
            record = frame.record
            _require(record is not None and response is record.native_response)
            _require(record.output is None)
            _stage_tuple(stages)
            binding, path, task = self._aux_frame_binding(frame)
            if type(frame) is _CompressionFrame:
                expected, expected_output = self._compression_response_transform(frame, response)
                key = "summary"
                _keys(output, (key,))
                _require(type(output[key]) is str and len(output[key]) > 0)
            else:
                expected, expected_output = titles._title_output_stages(response, self._interrupt_turn)
                key = "title"
                _keys(output, (key,))
                _require(type(output[key]) in (str, type(None)))
            _require(stages == expected)
            _require(_json(output) == _json(expected_output))
            _require(output is self._aux_output[0] and stages is self._aux_output[1])
            _require(output[key] is self._aux_output[2])
            contents = []
            for stage in stages:
                self._interrupt_turn()
                contents.append(self._runtime.derive(binding.handle,
                    (frame.source_content, record.request_content, record.result_content, *contents),
                    stage, path=path))
            payload = _json(output)
            _require(len(payload) <= 262_144)
            parents = (frame.source_content, record.request_content, record.result_content, *contents)
            content = self._derive_auxiliary_output(binding, parents, payload, path)
            if type(frame) is _CompressionFrame:
                self._interrupt_turn()
            self._retain_aux_record(replace(record, output_stages=stages, output_contents=tuple(contents),
                output_bytes=payload, output=output, output_content=content))
            self._interrupt_turn()
            self._aux_pending = None
            return output

    def complete_auxiliary_local(
        self,
        agent: object,
        source: NativeAuxiliaryInput,
        *,
        stages: tuple[bytes, ...],
        output: dict,
    ) -> dict:
        with self._turn_operation(agent):
            frame = self._checked_aux_frame(source)
            _require(type(frame) is _AuxiliaryFrame)
            _require(frame.sealed is None and frame.record is None and frame.local_result is None)
            proof = frame.preparation
            _require(type(proof) is _AuxiliaryTitlePreparation and proof.kind == "local_no_send")
            _require(proof.stages is stages and proof.value is output)
            _require(_same_anchors(_nested_anchors(proof), frame.preparation_anchors))
            _stage_tuple(stages)
            _keys(output, ("title",))
            _require(output["title"] is None and len(stages) == 5)
            payload = _json(output)
            _require(len(payload) <= 262_144)
            contents = []
            for stage in stages:
                self._interrupt_turn()
                contents.append(self._runtime.derive(self._aux_binding.handle,
                    (frame.source_content, *contents), stage, path="title"))
            output_stages = (_json(dict(outcome="local_no_send", reason="empty_title_snippet")),)
            self._interrupt_turn()
            decision = self._runtime.derive(self._aux_binding.handle,
                (frame.source_content, *contents), output_stages[0], path="title")
            parents = (frame.source_content, *contents, decision)
            self._interrupt_turn()
            content = self._derive_auxiliary_output(self._aux_binding, parents, payload, "title")
            self._interrupt_turn()
            target, selected, producer = self._aux_current
            record = NativeAuxiliaryLocalResult(frame.completion, frame.completion.frame,
                frame.completion.invocation, self._handle, self._aux_binding.handle,
                self._source, target, "title", "title_generation", selected, producer,
                self._aux_binding.language, len(self._aux_frames), "local_no_send", "empty_title_snippet",
                frame.source_bytes, frame.parents, frame.parent_requests, frame.source_content,
                stages, tuple(contents), output_stages, (decision,), payload, output, content)
            values = _row(record, NativeAuxiliaryLocalResult)
            self._aux_local_results += (record,)
            self._aux_local_anchors += ((record, values, _nested_anchors(values)[1:]),)
            self._retain_aux_frame(replace(frame, stages=stages, stage_contents=tuple(contents), local_result=record))
            self._interrupt_turn()
            self._aux_pending = None
            return output

    def _aux_frame_binding(self, frame):
        if type(frame) is _AuxiliaryFrame:
            _require(self._aux_binding is not None)
            return self._aux_binding, "title", "title_generation"
        _require(type(frame) is _CompressionFrame and self._compression_binding is not None)
        return self._compression_binding.auxiliary, "compression", "compression"

    def _derive_auxiliary_output(self, binding, parents, payload, path):
        completed = tuple(self._completed_auxiliary_outputs())
        if not completed:
            return self._runtime.derive(binding.handle, parents, payload, path=path)
        return self._runtime.derive_for_selection(
            binding.handle, (*parents, *(content for content, _ in completed)), payload, path=path,
            parent_requests=(*(binding.handle for _ in parents), *(handle for _, handle in completed)))

    def _compression_snapshot(self, main):
        binding = self._compression_binding.auxiliary
        target = self._admission.current(binding.handle)
        self._identity()
        target_row = _binding(target)
        selected = binding.port.current(binding.handle)
        self._identity()
        selection_row = _selection(selected)
        producer = self._result_port.current(binding.producer_ref)
        producer_row = _producer(producer)
        self._identity()
        _require(target_row[:6] == main[:6] and target_row[:8] == binding.source[:8])
        _require(selection_row == binding.selected and selection_row[:2] == target_row[6:8])
        _require(producer_row == binding.producer and producer_row[2] == self._initial_producer[2])
        _require(set(producer_row[2]) <= set(binding.original_profiles))
        _require(set(producer_row[2]) <= set(self._original_profiles))
        _require(selected.destination == binding.route.destination)
        now = self._clock()
        _integer(now, _MAX_INT, 0)
        self._identity()
        for route in self._operator_routes:
            _require(type(route) is shared_policy().RouteBinding)
            _integer(route.expires_at)
            _require(now < route.expires_at)
            port = self._operator_transports[route.destination.transport_ref]
            method = next(method for original, method in self._operator_resolvers if port is original)
            self._callback_scope()
            _require(_same_method(port.resolve, method))
            actual = port.resolve()
            self._callback_scope()
            _require(_same_method(port.resolve, method))
            _destination(actual)
            _require(actual == route.destination)
        self._identity()
        self._compression_current = target, selected, producer

    def bind_compression(
        self, agent: object, *, request_handle: object,
        selection: NativeSelectionPort, producer_ref: str, route_ref: str,
        context_length: int, summary_date: str,
    ) -> None:
        with self._turn_operation(agent):
            self._model_open()
            _require(self._constructed is not None and self._compression_binding is None)
            _require(self._turn_pending is None and self._aux_pending is None)
            _integer(context_length)
            _require(type(summary_date) is str and len(summary_date) == 10)
            _require(re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", summary_date) is not None)
            from datetime import date
            _require(date.fromisoformat(summary_date).isoformat() == summary_date)
            _identifier(producer_ref)
            _identifier(route_ref)
            _require(request_handle is not self._handle and selection is not self._selection_port)
            _require(producer_ref != self._producer_ref)
            if self._aux_binding is not None:
                other = self._aux_binding
                _require(request_handle is not other.handle and selection is not other.port)
                _require(producer_ref != other.producer_ref)
            from agent import context_compressor as compression, auxiliary_client as auxiliary
            from agent import agent_runtime_helpers
            methods = (
                (selection, "current"),
                *((self._runtime, name) for name in ("derive_for_selection", "derive", "dispatch")),
                *((self, name) for name in (
                    "bind_compression", "compression_input", "seal_auxiliary", "dispatch_auxiliary",
                    "complete_auxiliary", "_prepare_compression", "_compression_output_stages",
                    "_compression_response_transform", "_compression_snapshot", "_aux_frame_binding",
                    "_derive_auxiliary_output", "_checked_aux_frame", "_completed_auxiliary_outputs",
                    "_dispatch_retained_operation", "_verify_auxiliary", "_retain_aux_frame",
                    "_retain_aux_record", "_accept_aux_record", "_callback_scope", "_interrupt_turn",
                    "_interrupt_operation", "_snapshot", "_checkpoint", "_identity", "_turn_operation", "_around")),
                *((compression, name) for name in _COMPRESSION_FUNCTIONS),
                (agent_runtime_helpers, "strip_think_blocks"),
                (auxiliary, "call_llm"), (auxiliary.call_llm, "__wrapped__"),
                *((sys.modules[__name__], name) for name in (
                    "_controlled_compression_constructor", "_controlled_compression_entry",
                    "_compression_constants", "_compression_wire", "_compression_call_arguments_bytes",
                    "_try_controlled_auxiliary_call", "_auxiliary_attempt", "_auxiliary_owner",
                    "_stage_tuple")),
            )
            self._methods += tuple((port, name, _method(getattr(port, name))) for port, name in methods)
            target = self._around(lambda: self._admission.current(request_handle))
            target_row = _binding(target)
            selected = self._around(lambda: selection.current(request_handle))
            selection_row = _selection(selected)
            producer = self._around(lambda: self._result_port.current(producer_ref))
            producer_row = _producer(producer)
            _require(target_row[:6] == self._holder[:6])
            _require(target_row[6] != self._holder[6] and selection_row[:2] == target_row[6:8])
            if self._aux_binding is not None:
                _require(selection_row[0] != self._aux_binding.selected[0])
            _require(selection_row[3][1] == "synthetic_native")
            _require(producer_row[0] == producer_ref and producer_row[2] == self._initial_producer[2])
            _require(set(producer_row[2]) <= set(target_row[8]))
            routes = tuple(r for r in self._operator_routes if r.route_ref == route_ref)
            _require(len(routes) == 1)
            route = routes[0]
            _require(route.path == "compression" and route.selection_id == selected.selection_id)
            _require(route.selection_version == selected.selection_version)
            _require(route.consumer_ref == target.consumer_ref and route.destination == selected.destination)
            binding = _AuxiliaryBinding(request_handle, selection, producer_ref, route, "",
                target_row, selection_row, producer_row, target_row[8])
            self._compression_binding = _CompressionBinding(binding, context_length, summary_date)
            self._bound_compression_binding = self._compression_binding
            self._compression_binding_anchor = _nested_anchors(self._compression_binding)
            self._compression_constants = _compression_constants()
            self._interrupt_turn()

    def compression_input(self, agent: object, *, source: dict) -> NativeAuxiliaryInput:
        with self._turn_operation(agent):
            self._model_open()
            _require(self._compression_binding is not None and self._compression_owner is not None)
            _require(self._turn_pending is None and self._aux_pending is None and len(self._aux_frames) < 8)
            _require(type(source) is dict)
            completion = next((row for row in self._completions if row.result is source), None)
            _require(completion is not None)
            frame, invocation = completion.frame, completion.invocation
            _require(source["completed"] is True)
            for key in ("interrupted", "failed", "partial", "response_transformed", "response_previewed"):
                _require(source[key] is False)
            _require(source["pre_transform_response"] is None)
            messages = source["messages"]
            _history(messages)
            payload = _json(messages)
            _require(len(payload) <= 262_144)
            completed = tuple(self._completed_auxiliary_outputs())
            parents = (frame.admitted, frame.input_content, frame.output_content,
                       *frame.request_contents, invocation.request_content, invocation.result_content,
                       completion.content, *(row.content for row in self._completions if row is not completion),
                       *(content for content, _ in completed))
            count = len(parents) - len(completed)
            parent_requests = (*(self._handle for _ in range(count)), *(handle for _, handle in completed))
            binding = self._compression_binding.auxiliary
            self._interrupt_turn()
            content = self._runtime.derive_for_selection(binding.handle, parents, payload,
                path="compression", parent_requests=parent_requests)
            self._interrupt_turn()
            token = object.__new__(NativeAuxiliaryInput)
            self._retain_aux_frame(_CompressionFrame(token, completion, messages, payload,
                                                    parents, parent_requests, content))
            self._interrupt_turn()
            return token

    def _prepare_compression(self, token: NativeAuxiliaryInput) -> _CompressionPreparation:
        with self._turn_operation(self._turn_agent):
            frame = self._checked_aux_frame(token)
            _require(type(frame) is _CompressionFrame and frame.preparation is None and frame.sealed is None)
            owner = self._compression_owner
            self._interrupt_turn()
            facts = owner._summary_source(frame.messages, "", retained=True)
            self._interrupt_turn()
            prompt = owner._summary_prompt(facts, None, self._compression_binding.summary_date)
            self._interrupt_turn()
            arguments = dict(_AUX_CALL_DEFAULTS, task="compression", provider=owner.provider,
                model=owner.summary_model, api_mode=owner.api_mode, stream=False,
                strict_destination=True, messages=[dict(role="user", content=prompt)])
            _compression_call_arguments_bytes(arguments)
            value = dict(arguments=arguments, request=_compression_wire(arguments))
            stages = (
                _json(dict(constructor=self._compression_options, source_field="messages",
                           previous_summary=None, focus_topic=None, memory_context="")),
                _json(facts["budget"]),
                _json(dict(serialized=facts["serialized"], pruned_skill_names=facts["markers"])),
                _json(dict(sampled=facts["sampled"], lean=True,
                           sampling=len(facts["serialized"]) > owner._SUMMARY_INPUT_MAX_CHARS,
                           original_length=len(facts["serialized"]), slice_count=owner._SAMPLED_INPUT_SLICES,
                           aggregate_cap=owner._SUMMARY_INPUT_MAX_CHARS)),
                _json(dict(sanitized_memory=facts["sanitized_memory"],
                           serialized_memory=facts["serialized_memory"], memory_section=facts["memory_section"],
                           has_user_turn=facts["has_user_turn"], summary_date=self._compression_binding.summary_date)),
                _json(dict(prompt=prompt)),
                _json(value),
            )
            _stage_tuple(stages)
            proof = _CompressionPreparation(stages, value, facts["has_user_turn"], tuple(facts["markers"]))
            self._interrupt_turn()
            self._retain_aux_frame(replace(frame, preparation=proof, preparation_anchors=_nested_anchors(proof)))
            self._interrupt_turn()
            return proof

    def _compression_response_transform(self, frame, response):
        _require(type(frame) is _CompressionFrame and response is frame.record.native_response)
        _require(_compression_response_bytes(response, frame.record.selection.destination.model_ref) == frame.record.response_bytes)
        owner, proof = self._compression_owner, frame.preparation
        self._interrupt_turn()
        content, finish_reason = owner._summary_response_content(response)
        self._interrupt_turn()
        summary, strip_facts, output_facts = owner._summary_output(
            content, frame.messages, proof.pruned_skill_names, proof.has_user_turn)
        self._interrupt_turn()
        prefixed = owner._with_summary_prefix(summary)
        self._interrupt_turn()
        output = dict(summary=prefixed)
        stages = (_json(dict(content=content, finish_reason=finish_reason, nonempty=True, length_stop=False)),
                  _json(strip_facts), _json(output_facts), _json(dict(prefixed=prefixed, output=output)))
        _stage_tuple(stages)
        return stages, output

    def _compression_output_stages(
        self, request: NativeAuxiliaryRequest, response: NativeMainResponse,
    ) -> tuple[tuple[bytes, ...], dict]:
        with self._turn_operation(self._turn_agent):
            frame = self._checked_aux_frame(request, sealed=True)
            _require(type(frame) is _CompressionFrame and frame.record is not None)
            stages, output = self._compression_response_transform(frame, response)
            self._aux_output = output, stages, output["summary"]
            self._interrupt_turn()
            return stages, output

    def _checkpoint(self):
        first = self._snapshot()
        second = self._snapshot()
        _require(first == second)
        if self._expected is not None:
            _require(first == self._expected)
        if self._holder is None:
            self._holder, self._initial_selection, self._initial_producer = first[0][:8], first[1], first[2]
            self._original_profiles = first[0][8]
        return first

    def _around(self, callback):
        try:
            self._checkpoint()
            output = callback()
            self._checkpoint()
            return output
        except BaseException as error:
            _remember_delegate_control(_delegate_frames.get(self) or getattr(self, "_delegate_parent_frame", None), error)
            raise

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
            self._model_open()
            _require(not self._busy and self._aux_pending is None and len(self._records) < 8)
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
            if self._fallback_binding is not None:
                self._fallback_expected = (_binding(self._fallback_current[0]),
                    _selection(self._fallback_current[1]), _producer(self._fallback_current[2]))
                self._fallback_expected_anchor = _nested_anchors(self._fallback_expected)
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
                request_content = self._derive(
                    self._handle,
                    (frame.admitted, frame.input_content, frame.output_content,
                     *(r.result_content for r in self._records),
                     *(row.content for row in self._completions)),
                    self._request_bytes, path="main",
                )
            elif self._records:
                request_content = self._derive(
                    self._handle, tuple(r.result_content for r in self._records),
                    self._request_bytes, path="main",
                )
            else:
                request_content = self._runtime.admit(self._handle, self._request_bytes)
            operation = _PendingOperation(
                self._handle, self._source, self._selected, self._producer_policy,
                "main", self._route_ref, self._request_bytes, request_content, False,
            )
            self._operation = self._checked_operation = operation
            outcome = self._dispatch_retained_operation(operation)
            if self._fallback_binding is not None and type(outcome) is shared_policy().FailedDispatch:
                self._fallback_phase = "observe"
                token = self._begin_fallback(agent, failure=outcome)
                return self._dispatch_fallback(agent, token)
            result_content, envelope, actual, output, response = outcome
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
            self._operation = self._checked_operation = None
            self._fallback_expected = None
            self._fallback_expected_anchor = None
            self._busy = False


def _nested_anchors(value):
    result = []
    policy = shared_policy()
    records = (NativeMainResponse, NativeMainChoice, NativeMainMessage, NativeMainUsage,
               NativeMainSelection, NativeResultPolicy, policy.ExecutionBinding,
               policy.Destination, policy.DispatchEnvelope, policy.RouteBinding,
               NativeAuxiliaryInvocation, NativeAuxiliaryLocalResult, _AuxiliaryTitlePreparation,
               _CompressionBinding, _AuxiliaryBinding, _CompressionPreparation,
               _FallbackBinding, _FailedSendObservation, NativeFailedMainAttempt,
               _FallbackFrame, NativeFallbackInvocation, _TurnFrame, _PendingOperation,
               NativeDelegateBinding, NativeDelegateInvocation,
               policy.ProfilePolicy, policy.DispatchFailureFacts)
    def visit(item):
        result.append(item)
        if type(item) is dict:
            for key, child in item.items():
                _require(type(key) is str)
                visit(key)
                visit(child)
        elif type(item) in (list, tuple):
            for child in item:
                visit(child)
        elif type(item) in records:
            for field in fields(item):
                visit(object.__getattribute__(item, field.name))
    visit(value)
    return tuple(result)


def _same_anchors(actual, original):
    return len(actual) == len(original) and all(a is b for a, b in zip(actual, original))


def _stage_tuple(stages):
    _require(type(stages) is tuple and 1 <= len(stages) <= 8)
    for stage in stages:
        _require(type(stage) is bytes and len(stage) <= 262_144)


_AUX_CALL_DEFAULTS = dict(task=None, provider=None, model=None, base_url=None, api_key=None,
    main_runtime=None, messages=None, temperature=None, max_tokens=None, tools=None, timeout=None,
    extra_body=None, reasoning_config=None, extra_headers=None, api_mode=None, stream=False,
    stream_options=None, route_info=None, latency_info=None, strict_destination=False)


def _call_arguments_bytes(arguments):
    _keys(arguments, tuple(_AUX_CALL_DEFAULTS))
    for name in ("task", "provider", "model", "api_mode"):
        _require(type(arguments[name]) is str)
        _identifier(arguments[name])
    _require(arguments["task"] == "title_generation")
    _require(arguments["provider"] == "synthetic_native")
    _require(arguments["api_mode"] == "chat_completions")
    _require(type(arguments["max_tokens"]) is int and arguments["max_tokens"] == 64)
    _require(type(arguments["temperature"]) is float and arguments["temperature"] == 0.3)
    _require(arguments["stream"] is False and arguments["strict_destination"] is True)
    for name in ("base_url", "api_key", "main_runtime", "tools", "timeout", "reasoning_config",
                 "extra_headers", "stream_options", "route_info", "latency_info"):
        _require(arguments[name] is None)
    _request(dict(model=arguments["model"], messages=arguments["messages"]))
    from agent.title_generator import _TITLE_RESPONSE_FORMAT
    _require(type(arguments["extra_body"]) is dict)
    _require(_safe_json(arguments["extra_body"]) == _json({"response_format": _TITLE_RESPONSE_FORMAT}))
    return _json(arguments)



def _compression_wire(arguments):
    request = dict(model=arguments["model"], messages=arguments["messages"], stream=False)
    _request(request)
    return request


def _compression_call_arguments_bytes(arguments: dict) -> bytes:
    _keys(arguments, tuple(_AUX_CALL_DEFAULTS))
    for name, expected in (("task", "compression"), ("provider", "synthetic_native"),
                           ("api_mode", "chat_completions")):
        _require(type(arguments[name]) is str and arguments[name] == expected)
    _identifier(arguments["model"])
    _require(arguments["stream"] is False and arguments["strict_destination"] is True)
    for name in _AUX_CALL_DEFAULTS:
        if name not in ("task", "provider", "model", "api_mode", "messages", "stream", "strict_destination"):
            _require(arguments[name] is None)
    _compression_wire(arguments)
    return _json(arguments)


_COMPRESSION_HELPERS = (
    "__init__", "_generate_summary", "_compute_summary_budget", "_summary_source", "_summary_prompt", "_summary_budget_facts",
    "_summary_response_content", "_summary_output", "_serialize_for_summary", "_sample_summary_input",
    "_bound_summary_input", "_augment_summary_lean", "_ground_historical_task_snapshot",
    "_latest_user_task_snapshot", "_transcript_has_real_user_turn", "_is_synthetic_compression_user_turn",
    "_has_compressed_summary_metadata", "_is_context_summary_content", "classify_summary_content",
    "_starts_with_summary_prefix", "_strip_summary_prefix", "_with_summary_prefix",
    "_validate_summary_user_provenance", "_resolve_context_length", "context_length", "max_summary_tokens",
)
_COMPRESSION_FUNCTIONS = (
    "_build_anchor_index", "_build_recovery_footer", "_build_verbatim_user_section",
    "_collect_ghosted_skill_names", "_content_text_for_contains", "_extract_pruned_skill_names",
    "_image_part_label", "_redact_compaction_text", "_reinject_pruned_skill_markers",
    "_response_finish_reason", "_skill_pruned_marker", "_skill_view_call_sites", "_synthetic_user_row",
    "sanitize_memory_context", "estimate_messages_tokens_rough", "redact_sensitive_text", "call_llm",
)


_COMPRESSION_CONSTANT_NAMES = (
    'COMPRESSED_SUMMARY_METADATA_KEY',
    'COMPRESSION_CONTINUATION_USER_CONTENT',
    'HISTORICAL_TASK_HEADING',
    'LEGACY_SUMMARY_PREFIX',
    'MAX_ITERATIONS_SUMMARY_REQUEST',
    'SKILL_PRUNED_MARKER_PREFIX',
    'SUMMARY_PREFIX',
    'TODO_INJECTION_HEADER',
    '_ACTIVE_TASK_MAX_CHARS',
    '_ANCHOR_NOISE',
    '_ANCHOR_PATTERNS',
    '_BACKGROUND_PROCESS_NOTIFICATION_PREFIX',
    '_HISTORICAL_SUMMARY_PREFIXES',
    '_HISTORICAL_TASK_SECTION_RE',
    '_LEAN_ANCHOR_BUDGET_CHARS',
    '_LEAN_ANCHOR_HEADING',
    '_LEAN_RECOVERY_HEADING',
    '_LEAN_SESSION_LOG_BUDGET_TOKENS',
    '_LEAN_SESSION_LOG_HEADING',
    '_LEAN_USER_MESSAGES_BUDGET_CHARS',
    '_LEAN_USER_MESSAGES_HEADING',
    '_LEAN_USER_MESSAGE_MAX_CHARS',
    '_LEGACY_COMPRESSION_CONTINUATION_USER_CONTENT',
    '_MAX_PRUNED_SKILL_MARKERS',
    '_MEDIA_DIRECTIVE_RE',
    '_MERGED_SUMMARY_DELIMITER',
    '_MIN_SUMMARY_TOKENS',
    '_NO_USER_TASK_SENTINEL',
    '_PRUNED_SKILLS_SECTION_HEADING',
    '_SKILL_PRUNED_MARKER_RE',
    '_SKILL_VIEW_PRUNE_MIN_CHARS',
    '_SUMMARY_END_MARKER',
    '_SUMMARY_RATIO',
    '_SUMMARY_TOKENS_CEILING',
    '_TRUNCATED_SUMMARY_MARKER',
)


_COMPRESSION_LOOP_CONSTANTS = ('_CODEX_ACK_CONTINUATION_NUDGE', '_CODEX_INCOMPLETE_NUDGE', '_DROPPED_TOOLCALL_NUDGE_CONTENT', '_EMPTY_TOOL_RESPONSE_NUDGE', '_LENGTH_CONTINUATION_DROPPED_TOOLS_PREFIX', '_LENGTH_CONTINUATION_NETWORK_STUB', '_LENGTH_CONTINUATION_OUTPUT_LIMIT')


def _compression_constants():
    from agent import context_compressor as compression, conversation_loop
    values = (*(vars(compression)[name] for name in _COMPRESSION_CONSTANT_NAMES),
              *(vars(conversation_loop)[name] for name in _COMPRESSION_LOOP_CONSTANTS))
    cls = vars(compression.ContextCompressor)
    descriptors = tuple(cls[name] for name in _COMPRESSION_HELPERS)
    constants = tuple(value for name, value in sorted(cls.items()) if name.isupper())
    anchors = (*_nested_anchors(values)[1:], *_nested_anchors(descriptors)[1:],
               *_nested_anchors(constants)[1:])
    members = []
    for value in anchors:
        if type(value) in (set, frozenset):
            _require(all(type(member) is str for member in value))
            members.extend(sorted(value))
    return (compression.ContextCompressor, *anchors, *members,
            *_nested_anchors(_COMPRESSION_DEFAULTS), *_nested_anchors(_AUX_CALL_DEFAULTS),
            _COMPRESSION_HELPERS, _COMPRESSION_FUNCTIONS, _COMPRESSION_CONSTANT_NAMES, _COMPRESSION_LOOP_CONSTANTS)


_COMPRESSION_DEFAULTS = dict(threshold_percent=0.50, protect_first_n=3, protect_last_n=20,
    summary_target_ratio=0.20, quiet_mode=True, base_url="", api_key="",
    abort_on_summary_failure=False, max_tokens=None, model_thresholds=None,
    threshold_tokens_cap=None, proactive_prune_tokens=0, proactive_prune_min_result_chars=8000,
    proactive_prune_min_reclaim_tokens=4096, min_tail_user_messages=1, tail_mode="lean")


def _controlled_compression_constructor(
    compressor: object, adapter: NativeMainPayloadAdapter | None, *, options: dict,
) -> bool:
    _require(type(options) is dict and all(type(key) is str for key in options))
    if adapter is None and not _auxiliary_attempt(provider=options.get("provider")):
        return False
    context = current_context()
    try:
        _require(type(adapter) is NativeMainPayloadAdapter and adapter in _adapters)
        _require(context is adapter._bound_context)
        with adapter._turn_operation(adapter._turn_agent):
            from agent.context_compressor import ContextCompressor
            _require(type(compressor) is ContextCompressor and vars(compressor) == {})
            _require(adapter._compression_binding is not None and adapter._compression_owner is None)
            _require(all(type(key) is str for key in options))
            _require(set(options) == {*_COMPRESSION_DEFAULTS, "model", "provider", "api_mode",
                                     "summary_model_override", "config_context_length"})
            for name, expected in _COMPRESSION_DEFAULTS.items():
                actual = options[name]
                _require(type(actual) is type(expected))
                _require(actual == expected)
            for name in ("model", "provider", "api_mode", "summary_model_override"):
                _identifier(options[name])
            _integer(options["config_context_length"])
            binding = adapter._compression_binding
            _require(options["provider"] == "synthetic_native" and options["api_mode"] == "chat_completions")
            _require(options["model"] == adapter._initial_selection[3][2])
            _require(options["summary_model_override"] == binding.auxiliary.selected[3][2])
            _require(options["config_context_length"] == binding.context_length)
            state = dict(options)
            state.pop("summary_model_override")
            state.pop("config_context_length")
            state.update(summary_model=options["summary_model_override"],
                _config_context_length=binding.context_length, _resolved_context_length=binding.context_length,
                _max_summary_tokens=None,
                _previous_summary=None, _summary_has_user_turn=None, _session_db=None, _session_id="",
                compression_count=0, _micro_compact_enabled=False, _micro_compact_cursor=0,
                _micro_compact_rolling_summary="", _micro_compact_passes=0,
                _micro_compact_consecutive_failures=0, _micro_compact_tokens_saved_total=0,
                _summary_failure_cooldown_until=0.0, _last_summary_error=None,
                _last_compression_telemetry=None, _active_compression_telemetry=None,
                _compression_telemetry_seed=None)
            _require(all(type(value) in (str, int, float, bool, type(None)) for value in state.values()))
            adapter._interrupt_turn()
            vars(compressor).update(state, _native_main_adapter=adapter)
            state["_max_summary_tokens"] = compressor.max_summary_tokens
            adapter._compression_owner = compressor
            adapter._compression_options = dict(options)
            adapter._compression_options_anchor = _nested_anchors(adapter._compression_options)
            adapter._compression_owner_anchor = (compressor, vars(compressor), _json(state),
                                                   _nested_anchors(vars(compressor)))
            adapter._interrupt_turn()
            return True
    except BaseException:
        if type(adapter) is NativeMainPayloadAdapter:
            adapter._invalidate()
        if type(context) is NativeContentContext:
            context.invalidate()
        raise


def _controlled_compression_entry(
    compressor: object, turns: list, token: NativeAuxiliaryInput | None, *,
    focus_topic: str | None, memory_context: str,
) -> NativeMainPayloadAdapter | None:
    namespace = object.__getattribute__(compressor, "__dict__")
    owned = any(adapter._compression_owner is compressor for adapter in _adapters)
    if not owned and namespace.get("_native_main_adapter") is None and not _auxiliary_attempt(
            token=token, provider=namespace.get("provider")):
        return None
    adapter = _auxiliary_owner(token, NativeAuxiliaryInput)
    with adapter._turn_operation(adapter._turn_agent):
        _require(adapter._compression_owner is compressor)
        frame = adapter._checked_aux_frame(token)
        _require(type(frame) is _CompressionFrame and frame.preparation is None and frame.sealed is None)
        _require(type(turns) is list and turns is frame.messages)
        _require(focus_topic is None and type(memory_context) is str and memory_context == "")
        _require(_json(turns) == frame.source_bytes)
        return adapter


def _refuse_compression_owner(compressor):
    namespace = object.__getattribute__(compressor, "__dict__")
    adapter = namespace.get("_native_main_adapter")
    owners = tuple(a for a in _adapters if a._compression_owner is compressor)
    if adapter is not None or owners or _auxiliary_attempt(provider=namespace.get("provider")):
        for owner in owners:
            owner._invalidate()
        from agent.content_policy import guard_native_content
        guard_native_content("compression", compressor)
        raise NativeContentPolicyDenied("controlled_native_operation_unsupported")

def _safe_json(value):
    def check(item, depth=0):
        _require(depth <= 8)
        _require(type(item) in (str, int, float, bool, dict, list, type(None)))
        if type(item) is dict:
            _require(len(item) <= 32)
            for key, child in item.items():
                _require(type(key) is str)
                check(child, depth + 1)
        elif type(item) is list:
            _require(len(item) <= 32)
            for child in item:
                check(child, depth + 1)
    check(value)
    payload = _json(value)
    _require(len(payload) <= 262_144)
    return payload


def _auxiliary_attempt(*, token=None, provider=None):
    if token is not None:
        return True
    if isinstance(provider, str) and str.__eq__(provider, "synthetic_native") is True:
        return True
    context = current_context()
    return context is not None and (context._controlled_main_bound is not False
                                    or context._controlled_main_payload is not None)


def _auxiliary_owner(token, cls):
    context = current_context()
    if token is None and type(context) is NativeContentContext and context._controlled_main_bound is True:
        context.check("title")
    candidates = [a for a in _adapters if type(token) is cls and any(
        token is (frame.sealed if cls is NativeAuxiliaryRequest else frame.token)
        for frame, _ in a._aux_frames)]
    try:
        _require(type(token) is cls and len(candidates) == 1)
        adapter = candidates[0]
        _require(context is adapter._bound_context)
        return adapter
    except BaseException:
        for adapter in candidates:
            adapter._invalidate()
        if type(context) is NativeContentContext:
            context._denied = True
            invalidate_controlled_context(context)
        raise NativeContentPolicyDenied("controlled_native_payload_denied") from None


def _controlled_title_entry(user_message, token, options):
    if not _auxiliary_attempt(token=token):
        return None
    adapter = _auxiliary_owner(token, NativeAuxiliaryInput)
    with adapter._turn_operation(adapter._turn_agent):
        frame = adapter._checked_aux_frame(token)
        _require(type(frame) is _AuxiliaryFrame)
        _require(all(value is None for value in options))
        _require(type(user_message) is str and user_message is frame.text)
        _require(_json(user_message) == frame.source_bytes)
        return adapter


def _try_controlled_auxiliary_call(*, arguments):
    _require(type(arguments) is dict)
    _require(all(type(key) is str for key in arguments))
    if not _auxiliary_attempt(token=arguments.get("controlled_aux"), provider=arguments.get("provider")):
        return False, None
    token = arguments.get("controlled_aux")
    adapter = _auxiliary_owner(token, NativeAuxiliaryRequest)
    return True, adapter.dispatch_auxiliary(token, arguments=arguments)


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
        adapter._invalidate()
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


def _controlled_fallback_activation(agent, reason, token):
    if token is None and not _controlled_attempt(agent):
        return False, False
    if token is None:
        _invalidate_constructed_attempt(agent)
        raise NativeContentPolicyDenied("controlled_native_operation_unsupported")
    binding = _owned_constructed_binding(agent)
    try:
        _require(reason is None and type(token) is NativeFallbackInput and binding is not None)
        _verify_constructed(binding)
        adapter = binding.adapter
        _require(current_context(agent) is adapter._bound_context)
        return True, adapter._activate_fallback(agent, token)
    except BaseException:
        _invalidate_constructed_attempt(agent)
        raise


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
    return _nested_anchors(result)


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
        content = adapter._derive(adapter._handle, parents, payload, path="main")
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
    owned = _owned_constructed_binding(agent)
    frame = getattr(owned.adapter, "_delegate_parent_frame", None) if owned is not None else None
    if (frame is not None and owned.closed and not frame.invalid
            and frame.phase in ("returned", "released") and frame.child is agent):
        frame.verify()
        return True
    _invalidate_constructed_attempt(agent)
    binding = _owned_constructed_binding(agent)
    if binding is not None:
        binding.closed = True
    return True


class _DelegateFrame:
    """Own one authenticated, synchronous delegation on the binding thread."""

    def __init__(self, parent, agent, receiver, receiver_selection, receiver_producer,
                 child_request, child_selection, child_producer, delegate_route,
                 child_route, session, task):
        _require(parent._constructed is _owned_constructed_binding(agent))
        _verify_constructed(parent._constructed)
        _require(not hasattr(parent, "_delegate_frame"))
        _require(not hasattr(parent, "_delegate_parent_frame"))
        parent.validate_scope()
        _require(parent._turn_agent is agent and not parent._busy)
        _require(not parent._turn_frames and not parent._records and not parent._completions)
        _require(parent._turn_pending is parent._aux_pending is None)
        _require(parent._aux_binding is parent._compression_binding is parent._fallback_binding is None)
        for value in (receiver_producer, child_producer, delegate_route, child_route, session, task):
            _identifier(value)
        self.parent, self.agent, self.runtime = parent, agent, parent._runtime
        self.thread = threading.get_ident()
        self.running = self.invalid = self.checking = self.terminal = False
        self.phase = self.phase_anchor = self.record = self.record_anchor = None
        self.child_adapter = self.child_context = self.child = None
        self.slot = self.slot_anchor = None
        self.delegate_sends = 0
        self.partial_sends = ()
        self.partial_anchors = ()
        self.control_exception = None
        self.inputs = self.outputs = ()
        self.methods = parent._methods
        self.catalog = parent._operator_configuration
        self.catalog_anchor = _nested_anchors(self.catalog)
        dr = tuple(r for r in parent._operator_routes if r.route_ref == delegate_route)
        mr = tuple(r for r in parent._operator_routes if r.route_ref == child_route)
        _require(len(dr) == len(mr) == 1 and dr[0].path == "delegate" and mr[0].path == "main")
        self.receiver_port = parent._operator_transports[dr[0].destination.transport_ref]
        self.capabilities = (parent._handle, receiver, child_request)
        self.selections = (parent._selection_port, receiver_selection, child_selection)
        self.producers = (parent._producer_ref, receiver_producer, child_producer)
        self.extra_methods = tuple((port, "current", _method(port.current)) for port in self.selections)
        self.extra_methods += tuple((self.runtime, name, _method(getattr(self.runtime, name))) for name in (
            "admit", "derive", "dispatch", "accept", "derive_for_selection", "return_from_delegate", "_content"))
        from tools import delegate_tool
        from run_agent import AIAgent
        self.extra_methods += tuple((port, name, _method(getattr(port, name))) for port, name in (
            (AIAgent, "__init__"), (NativeMainPayloadAdapter, "__init__"),
            (delegate_tool, "delegate_task"), (delegate_tool, "_build_child_agent"),
            (delegate_tool, "_run_single_child"), (delegate_tool, "_build_child_system_prompt"),
            (delegate_tool, "_normalize_role")))
        self.extra_methods += tuple((parent, name, _method(getattr(parent, name))) for name in (
            "_identity", "_snapshot", "_checkpoint", "_around", "_model_open", "validate_scope",
            "_verify_records", "_verify_turns", "_verify_completions", "bind_delegation"))
        facts, originals = self.facts()
        p, r, m = (row[0] for row in facts)
        _require(p[0] == r[0] and p[2] == r[2] and p[4:6] == r[4:6])
        _require(p[1] != r[1] and p[3] != r[3] and r[:6] == m[:6])
        _require(task == m[1])
        for index, route in ((1, dr[0]), (2, mr[0])):
            holder, selected, producer = facts[index]
            _require((route.consumer_ref, route.selection_id, route.selection_version)
                     == (holder[3], holder[6], holder[7]))
            if index == 2:
                _require(selected[3] == _destination(route.destination))
            _require(set(producer[2]) <= set(holder[8]))
        port_binding = parent._admission.current(self.receiver_port)
        _require(_binding(port_binding) == r)
        self.initial = facts
        self.binding = NativeDelegateBinding(parent._handle, receiver, child_request,
            receiver_selection, child_selection, receiver_producer, child_producer,
            dr[0], mr[0], session, task,
            (originals, port_binding, self.catalog, self.methods, self.extra_methods),
            _nested_anchors((originals, port_binding, self.catalog, self.methods, self.extra_methods)))
        self.binding_anchor = _row(self.binding, NativeDelegateBinding)
        self.snapshot_anchor = _nested_anchors(self.binding.snapshots)
        self.capability_anchor = self.capabilities
        self.selection_anchor = self.selections
        self.initial_anchor = _nested_anchors(self.initial)
        parent._delegate_frame = self
        _delegate_frames[parent] = self
        self.checkpoint()

    def invalidate(self):
        if (self.slot == "close" and self.phase == "returned" and not self.invalid
                and current_context() is None):
            self.verify()
            _require(_same_anchors(self.closed_proof, self.proof()))
            return
        self.invalid = self.terminal = True
        for adapter in (self.parent, self.child_adapter):
            if adapter is not None:
                adapter._invalid = True
                adapter._bound_context._denied = True

    def child_scope(self):
        return (self.running and self.phase in ("accepted", "constructed", "completed", "returned")
                and current_context() in (None, self.child_context))

    def verify(self):
        _require(_delegate_frames.get(self.parent) is self and not self.invalid)
        _require(self.parent._delegate_frame is self)
        _require(self.capabilities is self.capability_anchor and self.selections is self.selection_anchor)
        _require(_same_anchors(_nested_anchors(self.initial), self.initial_anchor))
        _require(threading.get_ident() == self.thread)
        _require(self.phase is self.phase_anchor and self.slot is self.slot_anchor)
        _require(self.parent._runtime is self.runtime is self.parent._captured_runtime)
        _require(self.parent._operator_configuration is self.catalog)
        _require(type(self.partial_sends) is tuple and len(self.partial_sends) <= 2)
        _require(_same_anchors(_nested_anchors(self.partial_sends)[1:], self.partial_anchors))
        _integer(self.delegate_sends, 1, 0)
        _require(_same_anchors(_nested_anchors(self.catalog), self.catalog_anchor))
        _require(_same_anchors(_row(self.binding, NativeDelegateBinding), self.binding_anchor))
        _require(_same_anchors(_nested_anchors(self.binding.snapshots), self.snapshot_anchor))
        _verify_constructed(self.parent._constructed)
        _require(self.parent._turn_agent is self.agent)
        _require(not self.parent._turn_frames and not self.parent._records and not self.parent._completions)
        _require(self.parent._aux_binding is self.parent._compression_binding is self.parent._fallback_binding is None)
        for port, name, method in (*self.methods, *self.extra_methods):
            _require(_same_method(getattr(port, name), method))
        if self.record is not None:
            _require(self.record is self.registered_record)
            _require(_same_anchors(_row(self.record, NativeDelegateInvocation), self.record_anchor))
            _require(_same_anchors(_nested_anchors(_row(self.record, NativeDelegateInvocation)[2:])[1:], self.record_nested))
            _require(self.record.binding is self.binding and self.record.token is self.token)
            _require(self.record.arguments is self.arguments)
            _require(_same_anchors(_nested_anchors(self.arguments), self.record.argument_anchors))
            _require(self.encode_arguments(self.arguments) == self.record.raw_bytes)
            if self.leaf is not None:
                _require(_json(self.leaf) == self.record.leaf_bytes)
            if hasattr(self, "plan_anchor"):
                _require(_same_anchors(_nested_anchors((self.prompt, self.constructor_options))[1:], self.plan_anchor))
                _text(self.prompt)
                _require(_json(dict(raw=json.loads(self.record.raw_bytes), leaf=self.leaf,
                    system_prompt=self.prompt, constructor=self.constructor_options,
                    entry=dict(user_message=self.leaf["goal"], system_message=self.prompt,
                               task_id=self.binding.child_task_id))) == self.plan)
                if self.record.request_content is not None:
                    _require(self.runtime._content(self.record.request_content).payload is self.plan)
            if self.child_adapter is not None:
                adapter = self.child_adapter
                _require(adapter._delegate_parent_frame is self)
                _require(adapter._aux_binding is adapter._compression_binding is adapter._fallback_binding is None)
                _require(adapter._context is self.child_context)
                if hasattr(adapter, "_runtime"):
                    _require(adapter._runtime is self.runtime)
                if self.child is not None:
                    _require(self.record.child is self.child)
                    binding = adapter._constructed
                    if binding.closed:
                        _require(self.phase in ("returned", "released"))
                        _require(_same_anchors(self.closed_proof, self.proof()))
                        _require(adapter._invalid and adapter._bound_context._denied)
                        _require(_constructed_state(_namespace(self.child)) == binding.state)
                        for name, original in binding.anchors:
                            if name not in ("_interrupt_requested", "_interrupt_message", "_tool_interrupt_reason",
                                            "_interrupt_thread_signal_pending"):
                                _require(_namespace(self.child).get(name) is original)
                        interrupted = _namespace(self.child).get("_interrupt_requested")
                        _require(type(interrupted) is bool)
                        if interrupted:
                            error = InterruptedError("Agent interrupted during delegation")
                            _remember_delegate_control(self, error)
                            raise error
                        adapter._verify_turns()
                        adapter._verify_records()
                        adapter._verify_completions()
                    else:
                        _verify_constructed(binding)

    def call(self, callback, *args):
        try:
            return callback(*args)
        except BaseException as error:
            _remember_delegate_control(self, error)
            raise

    def facts(self):
        values, originals = [], []
        for handle, selection, producer_ref in zip(self.capabilities, self.selections, self.producers):
            source = self.call(self.parent._admission.current, handle)
            selected = self.call(selection.current, handle)
            producer = self.call(self.parent._result_port.current, producer_ref)
            row = _binding(source), _selection(selected), _producer(producer)
            _require(row[1][:2] == row[0][6:8] and row[1][3][1] == "synthetic_native")
            _require(row[2][0] == producer_ref)
            values.append(row)
            originals.append((source, selected, producer))
        return tuple(values), tuple(originals)

    def checkpoint(self):
        self.verify()
        _require(not self.checking)
        self.checking = True
        try:
            for _ in range(2):
                facts, _ = self.facts()
                for actual, original in zip(facts, self.initial):
                    _require(actual[0][:8] == original[0][:8] and actual[1:] == original[1:])
                    _require(set(actual[0][8]) <= set(original[0][8]))
                port = _binding(self.call(self.parent._admission.current, self.receiver_port))
                _require(port[:8] == self.initial[1][0][:8])
                _require(set(port[8]) <= set(self.initial[1][0][8]))
                now = self.call(self.parent._clock)
                _integer(now, minimum=0)
                allowed = set(r.route_ref for r in self.parent._operator_routes)
                inherited = set(p for row in self.initial for p in row[0][8])
                inherited.update(p for row in self.initial for p in row[2][2])
                for profile in self.parent._operator_profiles:
                    if profile.profile_ref in inherited:
                        allowed.intersection_update(profile.allowed_routes)
                _require(self.binding.delegate_route.route_ref in allowed)
                _require(self.binding.child_main_route.route_ref in allowed)
                for row in facts:
                    for path in shared_policy().PATHS:
                        _require(any(r.route_ref in allowed and r.path == path
                            and (r.consumer_ref, r.selection_id, r.selection_version)
                            == (row[0][3], row[0][6], row[0][7]) and now < r.expires_at
                            for r in self.parent._operator_routes))
                for route in self.parent._operator_routes:
                    _route_facts(route)
                    actual = self.call(self.parent._operator_transports[route.destination.transport_ref].resolve)
                    _require(_destination(actual) == _destination(route.destination))
                interrupted = _namespace(self.agent).get("_interrupt_requested")
                _require(type(interrupted) is bool)
                if interrupted:
                    raise InterruptedError("Agent interrupted during delegation")
                if self.child is not None and not self.child_adapter._constructed.closed:
                    interrupted = _namespace(self.child).get("_interrupt_requested")
                    _require(type(interrupted) is bool)
                    if interrupted:
                        raise InterruptedError("Agent interrupted during delegation")
                self.verify()
        except BaseException as error:
            _remember_delegate_control(self, error)
            raise
        finally:
            self.checking = False

    def current(self, handle):
        self.verify()
        valid = next((h for h in self.capabilities if handle is h), None)
        if valid is not self.binding.parent_request:
            _require(self.running and self.record is not None and self.record.token is self.token)
            _require(self.phase in ("captured", "dispatched", "accepted", "constructed", "completed", "returned"))
        if valid is None:
            wrapper = next((w for w in self.parent._transport_wrappers.values() if handle is w), None)
            _require(wrapper is not None)
            valid = wrapper.port
            _require(valid is self.receiver_port)
        self.call(self.parent._checkpoint)
        try:
            result = self.parent._admission.current(valid)
        except BaseException as error:
            _remember_delegate_control(self, error)
            raise
        self.call(self.parent._checkpoint)
        _binding(result)
        return result

    def retain(self, **changes):
        self.verify()
        self.record = replace(self.record, **changes)
        self.registered_record = self.record
        self.record_anchor = _row(self.record, NativeDelegateInvocation)
        self.record_nested = _nested_anchors(self.record_anchor[2:])[1:]

    def set_slot(self, slot):
        self.verify()
        _require(self.record is not None and self.record.token is self.token)
        self.slot = self.slot_anchor = slot

    def transition(self, token, previous, phase):
        self.checkpoint()
        _require(token is self.token and self.record.token is token and self.phase == previous)
        self.phase = self.phase_anchor = phase
        self.checkpoint()

    def stage(self, payload, parents, *, output=False, handle=None, parent_requests=None):
        _require(type(payload) is bytes and len(payload) <= _MAX_NATIVE_RESULT)
        stages = self.outputs if output else self.inputs
        _require(len(stages) < 8)
        handle = self.binding.parent_request if handle is None else handle
        if not parents:
            content = self.runtime.admit(handle, payload)
        elif parent_requests is None:
            content = self.runtime.derive(handle, parents, payload, path="delegate")
        else:
            content = self.runtime.derive_for_selection(handle, parents, payload,
                path="tool_result" if output else "main", parent_requests=parent_requests)
        if output:
            self.outputs += ((payload, content),)
        else:
            self.inputs += ((payload, content),)
        return content

    def encode_arguments(self, arguments):
        rows, positional, keywords = arguments
        def value(name, item):
            return {"parent_request": list(self.initial[0][0][:8])} if name == "parent_agent" else item
        return _json(dict(arguments=[(n, p, value(n, v)) for n, p, v in rows],
            positional=[value(rows[i][0], v) for i, v in enumerate(positional)],
            keywords=[(n, value(n, v)) for n, v in keywords]))

    def capture(self, args, kwargs, signature):
        self.checkpoint()
        _require(not self.running and not self.terminal and self.record is None)
        self.running = self.terminal = True
        _require(all(type(k) is str for k in kwargs))
        bound = signature.bind(*args, **kwargs)
        values = bound.arguments
        _require(values.get("parent_agent") is self.agent)
        _require(values.get("credentials_cfg") is None)
        for name in ("tasks", "max_iterations", "output_schema", "subagent_id", "message"):
            _require(values.get(name) is None)
        goal, context = values.get("goal"), values.get("context")
        _text(goal)
        _require(bool(goal.strip()))
        if context is not None:
            _text(context)
        for name, cls, expected in (("role", str, "leaf"), ("background", bool, False), ("action", str, "spawn")):
            item = values.get(name)
            _require(item is None or (type(item) is cls and item == expected))
        self.arguments = (tuple((name, name in values, values.get(name)) for name in signature.parameters),
                          args, tuple(kwargs.items()))
        self.leaf = None
        raw = self.encode_arguments(self.arguments)
        raw_content = self.stage(raw, ())
        self.token = object.__new__(NativeDelegateInput)
        self.record = NativeDelegateInvocation(token=self.token, binding=self.binding,
            arguments=self.arguments, argument_anchors=_nested_anchors(self.arguments),
            raw_bytes=raw, raw_content=raw_content, leaf_bytes=b"", leaf_content=None,
            request_content=None, envelope=None, ack_bytes=None, ack_content=None,
            accepted_content=None, child_seed=None, child=None, child_adapter=None,
            child_input=None, child_request=None, child_invocation=None, child_completion=None,
            entry_bytes=None, aggregate_bytes=None, receiver_result=None, return_facts=None)
        self.registered_record = self.record
        self.record_anchor = _row(self.record, NativeDelegateInvocation)
        self.record_nested = _nested_anchors(self.record_anchor[2:])[1:]
        self.phase = self.phase_anchor = "captured"
        self.slot = self.slot_anchor = "entry"
        return self.token

    def send(self, wrapper, envelope):
        self.checkpoint()
        adapter = self.parent if self.phase == "captured" else self.child_adapter
        _require(adapter is not None and self.slot in ("dispatch", "runner"))
        operation = adapter._operation
        _require(type(operation) is _PendingOperation and operation is adapter._checked_operation)
        _require(adapter._busy and adapter._pending is None)
        _require(type(envelope) is shared_policy().DispatchEnvelope)
        _require(envelope.payload is operation.payload and envelope.path == operation.path)
        _require(_binding(envelope.source) == _binding(operation.source))
        _require(_destination(envelope.destination) == _destination(operation.selected.destination))
        _require(self.parent._transport_wrappers[envelope.destination.transport_ref] is wrapper)
        recipient = self.initial[1][0][3] if adapter is self.parent else self.initial[2][0][3]
        _require(type(envelope.recipient_ref) is str and envelope.recipient_ref == recipient)
        _require(wrapper.resolve() == envelope.destination)
        adapter._pending = (envelope, envelope.destination, None)
        if adapter is self.parent:
            _require(self.delegate_sends == 0 and operation.path == "delegate")
            self.delegate_sends += 1
            self.retain(envelope=envelope)
        else:
            _require(adapter is self.child_adapter and adapter._send_count == 0 and operation.path == "main")
            adapter._send_count += 1
            self.retain(child_input=adapter._turn_pending, child_request=adapter._kwargs)
        self.partial_sends += ((adapter, envelope, operation),)
        self.partial_anchors = _nested_anchors(self.partial_sends)[1:]
        try:
            output = wrapper.port.send(envelope)
        except BaseException as error:
            _remember_delegate_control(self, error)
            raise
        adapter._pending = (envelope, envelope.destination, output)
        if adapter is self.parent:
            self.retain(ack_bytes=output)
            _require(type(output) is bytes and output == b'{"accepted":true}')
        else:
            _require(type(output) is bytes and len(output) <= 65_536)
        self.checkpoint()
        return output

    def dispatch_request(self):
        self.checkpoint()
        parent, binding = self.parent, self.binding
        _require(self.phase == "captured" and self.slot == "entry")
        operation = _PendingOperation(binding.parent_request, parent._source,
            NativeMainSelection(binding.delegate_route.selection_id, binding.delegate_route.selection_version,
                "chat_completions", binding.delegate_route.destination), parent._producer_policy,
            "delegate", binding.delegate_route.route_ref,
            self.runtime._content(self.record.request_content).payload, self.record.request_content, False)
        self.set_slot("dispatch")
        parent._operation = parent._checked_operation = operation
        parent._busy = True
        try:
            ack = self.runtime.dispatch(binding.parent_request, operation.content,
                path="delegate", route_ref=binding.delegate_route.route_ref)
            self.retain(ack_content=ack)
            _require(self.runtime._content(ack).payload == b'{"accepted":true}')
        finally:
            parent._busy = False
            parent._operation = parent._checked_operation = parent._pending = None
            self.set_slot(None)
        self.transition(self.token, "captured", "dispatched")
        accepted = self.runtime.accept(binding.receiver_request, self.record.envelope)
        self.retain(accepted_content=accepted)
        seed = self.stage(self.record.leaf_bytes, (accepted,), handle=binding.child_request,
            parent_requests=(binding.receiver_request,))
        self.retain(child_seed=seed)
        self.transition(self.token, "dispatched", "accepted")

    def open_view(self, adapter, context):
        self.checkpoint()
        _require(self.slot == "view" and self.phase == "accepted" and self.child_adapter is None)
        _require(current_context() is None and context is self.child_context)
        _require(context.admission is self.parent._admission and context.request_handle is self.binding.child_request)
        self.child_adapter = adapter
        adapter._delegate_parent_frame = self

    def check_child_input(self, agent, arguments):
        self.checkpoint()
        _require(self.phase == "constructed" and self.slot == "runner" and agent is self.child)
        _require(not self.child_adapter._turn_frames)
        _require(arguments[0] is self.leaf["goal"] and arguments[1] is self.prompt)
        _require(arguments[2] is None and arguments[3] is self.binding.child_task_id)

    def proof(self):
        adapter = self.child_adapter
        return (adapter._turn_frames, adapter._records, adapter._record_anchors,
                adapter._completions, adapter._completion_anchors,
                self.record, self.record_anchor, adapter._constructed.state,
                adapter._constructed.anchors)

    def retain_completion(self, result):
        self.checkpoint()
        _require(self.phase == "constructed" and self.slot == "runner")
        adapter = self.child_adapter
        _require(len(adapter._turn_frames) == len(adapter._records) == len(adapter._completions) == 1)
        completion = adapter._completions[0]
        _require(completion.result is result)
        adapter._verify_completions()
        self.retain(child_input=adapter._turn_frames[0],
            child_invocation=adapter._records[0], child_completion=completion)

    def complete(self, result, duration):
        self.checkpoint()
        _require(self.record.child_completion.result is result)
        completion = self.record.child_completion
        _require(result["completed"] is True and result["failed"] is result["interrupted"] is result["partial"] is False)
        _require(type(result["api_calls"]) is int and result["api_calls"] == 1)
        _text(result["final_response"])
        _require(bool(result["final_response"].strip()))
        _require(type(duration) is float and math.isfinite(duration) and 0 <= duration <= 600)
        entry = dict(task_index=0, status="completed", summary=result["final_response"], api_calls=1,
            duration_seconds=duration, model=result["model"], exit_reason="completed", truncated=False,
            tokens=dict(input=result["input_tokens"], output=result["output_tokens"]), tool_trace=[])
        payload = json.dumps(entry, ensure_ascii=False).encode("utf-8")
        self.retain(entry_bytes=payload)
        self.stage(payload, (self.record.accepted_content, completion.content), output=True,
            handle=self.binding.receiver_request,
            parent_requests=(self.binding.receiver_request, self.binding.child_request))
        self.transition(self.token, "constructed", "completed")
        return entry

    def finish(self, entry):
        self.checkpoint()
        _require(self.phase == "completed")
        _require(self.delegate_sends == self.child_adapter._send_count == 1 and len(self.partial_sends) == 2)
        _require(json.dumps(entry, ensure_ascii=False).encode("utf-8") == self.record.entry_bytes)
        aggregate = json.dumps(dict(results=[entry], total_duration_seconds=entry["duration_seconds"]), ensure_ascii=False)
        payload = aggregate.encode("utf-8")
        self.retain(aggregate_bytes=payload)
        content = self.stage(payload, (self.record.accepted_content, self.record.child_completion.content,
            self.outputs[-1][1]), output=True, handle=self.binding.receiver_request,
            parent_requests=(self.binding.receiver_request, self.binding.child_request, self.binding.receiver_request))
        self.retain(receiver_result=content)
        facts = self.runtime.return_from_delegate(self.binding.parent_request, self.record.envelope,
            self.record.accepted_content, content, receiver_request=self.binding.receiver_request)
        self.retain(return_facts=facts)
        self.check_return()
        self.transition(self.token, "completed", "returned")
        return aggregate

    def check_return(self):
        record, facts = self.record, self.record.return_facts
        _require(type(facts) is shared_policy().DelegateReturnFacts)
        _require(facts.envelope is record.envelope and facts.accepted_content is record.accepted_content)
        _require(facts.receiver_result is record.receiver_result)
        returned = self.runtime._content(facts.returned_content)
        _require(returned.payload is self.runtime._content(record.receiver_result).payload)
        _require(returned.payload == record.aggregate_bytes and returned.path == "tool_result")
        _require(_same_anchors(returned.parents,
            (record.request_content, record.accepted_content, record.receiver_result)))
        _require(_binding(facts.sender)[:8] == self.initial[0][0][:8])
        _require(_binding(facts.receiver)[:8] == self.initial[1][0][:8])


def _delegate_owner(agent):
    binding = _owned_constructed_binding(agent)
    return _delegate_frames.get(binding.adapter) if binding is not None else None


def _remember_delegate_control(frame, error):
    import asyncio
    if frame is not None and isinstance(error, (InterruptedError, asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
        if frame.control_exception is None:
            frame.control_exception = error


def _capture_delegate_call(function):
    from functools import wraps
    from inspect import signature
    public_signature = signature(function)

    @wraps(function)
    def captured(*args, **kwargs):
        agent = args[10] if len(args) > 10 else kwargs.get("parent_agent")
        frame = _delegate_owner(agent)
        if frame is None:
            return function(*args, **kwargs)
        try:
            frame.capture(args, kwargs, public_signature)
            return function(*args, **kwargs)
        except BaseException:
            frame.invalidate()
            if frame.control_exception is not None:
                raise frame.control_exception
            raise
        finally:
            frame.running = False
    return captured


def _delegate_boundary(function):
    from functools import wraps

    @wraps(function)
    def checked(arguments):
        frame = _delegate_owner(arguments["parent_agent"])
        try:
            return function(arguments)
        except BaseException:
            if frame is not None:
                frame.invalidate()
            raise
    return checked


def _controlled_delegate_entry(arguments):
    frame = _delegate_owner(arguments["parent_agent"])
    if frame is None:
        return False, None
    from contextlib import ExitStack
    from agent.content_policy import content_policy_scope
    from tools import delegate_tool
    try:
        frame.checkpoint()
        _require(frame.running and frame.phase == "captured" and frame.slot == "entry")
        for name, _, original in frame.record.arguments[0]:
            _require(arguments[name] is original)
        goal, context, role = arguments["goal"], arguments["context"], arguments["role"]
        top_role = delegate_tool._normalize_role(role)
        single_task = {"goal": goal, "context": context, "role": top_role}
        effective_role = delegate_tool._normalize_role(single_task.get("role") or top_role)
        _child_context = single_task.get("context")
        leaf = dict(goal=single_task["goal"], context=_child_context, role=effective_role)
        frame.retain(leaf_bytes=_json(leaf))
        frame.leaf = leaf
        content = frame.stage(frame.record.leaf_bytes, (frame.record.raw_content,))
        frame.retain(leaf_content=content)
        frame.scopes = ExitStack()
        try:
            frame.set_slot("builder")
            child = delegate_tool._build_child_agent(0, goal, context, None, None,
                sys.maxsize, 1, frame.agent, role="leaf", _native_delegate_token=frame.token)
            frame.set_slot("runner")
            entry = delegate_tool._run_single_child(0, goal, child, frame.agent,
                _native_delegate_token=frame.token)
            aggregate = frame.finish(entry)
            frame.closed_proof = frame.proof()
            frame.set_slot(None)
        except BaseException:
            frame.invalidate()
            raise
        finally:
            active_error = sys.exc_info()
            if active_error[0] is not None:
                try:
                    frame.scopes.__exit__(*active_error)
                except BaseException:
                    pass
            else:
                frame.scopes.close()
        frame.checkpoint()
        frame.check_return()
        _require(aggregate.encode("utf-8") == frame.record.aggregate_bytes)
        frame.transition(frame.token, "returned", "released")
        return True, aggregate
    except BaseException as error:
        frame.invalidate()
        if frame.control_exception is not None:
            raise frame.control_exception
        if isinstance(error, Exception) and not isinstance(error, (InterruptedError, NativeContentPolicyDenied)):
            raise NativeContentPolicyDenied("controlled_native_delegation_denied") from None
        raise


@_delegate_boundary
def _controlled_delegate_builder(arguments):
    frame = _delegate_owner(arguments["parent_agent"])
    token = arguments.get("_native_delegate_token")
    if frame is None:
        _require(token is None)
        return False, None
    from run_agent import AIAgent
    from agent.content_policy import content_policy_scope
    from tools.delegate_tool import _build_child_system_prompt
    frame.checkpoint()
    _require(frame.running and frame.phase == "captured" and frame.slot == "builder")
    _require(token is frame.token)
    for name in ("task_index", "max_iterations", "task_count"):
        _require(type(arguments[name]) is int)
    _require(arguments["task_index"] == 0 and arguments["task_count"] == 1)
    _require(arguments["max_iterations"] == sys.maxsize)
    _require(arguments["goal"] is frame.leaf["goal"] and arguments["context"] is frame.leaf["context"])
    _require(type(arguments["role"]) is str and arguments["role"] == "leaf")
    for name in ("toolsets", "model", "override_provider", "override_base_url", "override_api_key",
                 "override_api_mode", "override_request_overrides", "override_max_tokens",
                 "override_acp_command", "override_acp_args"):
        _require(arguments[name] is None)
    frame.prompt = _build_child_system_prompt(frame.leaf["goal"], frame.leaf["context"], workspace_path=None, role="leaf")
    _text(frame.prompt)
    b = frame.binding
    frame.constructor_options = dict(provider="synthetic_native", api_mode="chat_completions",
        model=frame.initial[2][1][3][2], session_id=b.child_session_id, enabled_toolsets=[],
        quiet_mode=True, skip_memory=True, skip_context_files=True, skip_background_review=True)
    plan = _json(dict(raw=json.loads(frame.record.raw_bytes), leaf=frame.leaf,
        system_prompt=frame.prompt, constructor=frame.constructor_options,
        entry=dict(user_message=frame.leaf["goal"], system_message=frame.prompt, task_id=b.child_task_id)))
    frame.plan = plan
    frame.plan_anchor = _nested_anchors((frame.prompt, frame.constructor_options))[1:]
    content = frame.stage(plan, (frame.record.raw_content, frame.record.leaf_content))
    frame.retain(request_content=content)
    frame.set_slot("entry")
    frame.dispatch_request()
    frame.checkpoint()
    frame.scopes.enter_context(content_policy_scope(None))
    frame.scopes.callback(_close_delegate_child, frame)
    context = NativeContentContext(frame.parent._admission, b.child_request)
    frame.child_context = context
    frame.set_slot("view")
    routes, profiles, transports, _ = frame.catalog
    adapter = NativeMainPayloadAdapter(context, selection=b.child_selection,
        result_policy=frame.parent._result_port, producer_ref=b.child_producer_ref,
        main_route_ref=b.child_main_route.route_ref, profiles=profiles, routes=routes,
        transports=transports, clock=frame.parent._clock, _delegation_frame=frame)
    _require(adapter is frame.child_adapter and adapter._runtime is frame.runtime)
    frame.retain(child_adapter=adapter)
    frame.scopes.enter_context(content_policy_scope(context))
    frame.set_slot("constructor")
    child = AIAgent(controlled_main=adapter, **frame.constructor_options)
    frame.retain(child=child)
    frame.child = child
    frame.transition(frame.token, "accepted", "constructed")
    return True, child


@_delegate_boundary
def _controlled_delegate_runner(arguments):
    frame = _delegate_owner(arguments["parent_agent"])
    options = arguments["_kwargs"]
    token = options.get("_native_delegate_token")
    if frame is None:
        _require(token is None)
        return False, None
    frame.checkpoint()
    _require(frame.running and frame.phase == "constructed" and frame.slot == "runner")
    _require(token is frame.token and len(options) == 1)
    _require(type(arguments["task_index"]) is int and arguments["task_index"] == 0)
    _require(arguments["goal"] is frame.leaf["goal"] and arguments["child"] is frame.child)
    _require(all(arguments[name] is None for name in ("owner_session_id", "owner_transport", "owner_session_record")))
    start = frame.parent._clock()
    _integer(start, minimum=0)
    result = frame.child.run_conversation(user_message=frame.leaf["goal"], system_message=frame.prompt,
        task_id=frame.binding.child_task_id)
    frame.retain_completion(result)
    end = frame.parent._clock()
    _integer(end, minimum=0)
    entry = frame.complete(result, float(end - start))
    return True, entry


def _close_delegate_child(frame):
    if frame.child is None:
        return
    if frame.invalid:
        try:
            frame.child.close()
        except BaseException:
            pass
        return
    frame.checkpoint()
    _require(frame.phase == "returned" and current_context() is None)
    _require(_same_anchors(frame.closed_proof, frame.proof()))
    frame.set_slot("close")
    frame.child.close()
    frame.set_slot(None)
    frame.checkpoint()
