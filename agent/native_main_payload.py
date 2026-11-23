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
from tools.memory_tool import MemoryStore as _BUILTIN_MEMORY_STORE

_MEMORY_GETTER = _BUILTIN_MEMORY_STORE.format_for_system_prompt
_MEMORY_GETTER_CODE = _MEMORY_GETTER.__code__
_MEMORY_CLASS_ANCHORS = tuple(value for pair in vars(_BUILTIN_MEMORY_STORE).items() for value in pair)

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
_parent_resumptions = _DelegateFrames()
_memory_snapshots = _DelegateFrames()
_memory_prompt_frames = _DelegateFrames()
_tool_call_probes = _DelegateFrames()
_issued_call_frames = _DelegateFrames()


@dataclass(frozen=True, slots=True)
class NativeMemorySource:
    store: object
    target: str
    owner_ref: str
    profile_ref: str
    source_version: int
    producer_policy: NativeResultPolicy


class NativeMemorySnapshotInput:
    __slots__ = ()

    def __new__(cls):
        raise TypeError("Memory tokens require operator admission")


@dataclass(frozen=True, slots=True)
class NativeMemorySnapshotBinding:
    token: NativeMemorySnapshotInput
    agent: object
    request_handle: object
    store: object
    target: str
    producer_policy: NativeResultPolicy
    selection: NativeMainSelection
    source_bytes: bytes
    source_content: object
    request_bytes: bytes
    request_content: object
    options: tuple
    snapshots: tuple
    anchors: tuple
    thread: int


@dataclass(frozen=True, slots=True)
class NativeMemorySnapshotResult:
    binding: NativeMemorySnapshotBinding
    block: str | None
    result_bytes: bytes
    result_content: object
    parents: tuple


def _memory_require(condition, reason):
    if not condition:
        raise NativeContentPolicyDenied("controlled_native_memory_" + reason)


def _memory_failure(adapter, error):
    owner = _memory_snapshots.get(adapter)
    adapter._invalidate()
    if owner is not None and owner.control_exception is not None:
        raise owner.control_exception
    if not isinstance(error, Exception):
        raise error
    if isinstance(error, InterruptedError):
        raise error
    if (type(error) is NativeContentPolicyDenied and len(error.args) == 1
            and type(error.args[0]) is str and error.args[0].startswith("controlled_native_memory_")):
        raise error
    if type(error) is shared_policy().PolicyError and error.args == ("capacity_exhausted",):
        raise NativeContentPolicyDenied("controlled_native_memory_capacity_denied") from None
    if type(error) is shared_policy().PolicyError and error.args == ("runtime_stale",):
        raise NativeContentPolicyDenied("controlled_native_memory_runtime_unhealthy") from None
    raise NativeContentPolicyDenied("controlled_native_memory_authority_denied") from None


def _capture_memory_binding(function):
    from functools import wraps
    from inspect import signature
    public_signature = signature(function)

    @wraps(function)
    def captured(*args, **kwargs):
        adapter = args[0] if args else kwargs.get("self")
        try:
            _memory_require(type(adapter) is NativeMainPayloadAdapter and adapter in _adapters, "adapter_invalid")
            if adapter in _tool_call_probes:
                adapter._deny()
            _memory_require(all(type(name) is str for name in kwargs), "options_invalid")
            bound = public_signature.bind(*args, **kwargs).arguments
            target = bound.get("target", "memory")
            _memory_require(type(target) is str and target == "memory", "target_invalid")
            producer = bound["producer_ref"]
            _memory_require(type(producer) is str and bool(producer), "producer_invalid")
            _identifier(producer)
            _memory_require(type(bound["store"]) is _BUILTIN_MEMORY_STORE, "store_invalid")
            options = (tuple((name, name in bound, bound.get(name))
                             for name in public_signature.parameters), args, tuple(kwargs.items()))
            _MemorySnapshot(adapter, bound["agent"], bound["store"], producer, target, options)
            return function(*args, **kwargs)
        except BaseException as error:
            if type(adapter) is NativeMainPayloadAdapter and adapter in _adapters:
                _memory_failure(adapter, error)
            raise
    return captured


class NativeParentResumption:
    __slots__ = ()

    def __new__(cls):
        raise TypeError("Parent tokens require operator admission")


@dataclass(frozen=True, slots=True)
class NativeParentResumptionBinding:
    token: NativeParentResumption
    agent: object
    delegation_frame: object
    delegation_binding: NativeDelegateBinding
    parent_request: object
    selection: NativeMainSelection
    producer_policy: NativeResultPolicy
    main_route: object
    instruction: str
    system_message: str | None
    task_id: str
    options: tuple
    authorization_bytes: bytes
    authorization_content: object
    snapshots: tuple
    anchors: tuple
    thread: int


@dataclass(frozen=True, slots=True)
class NativeParentResumptionInvocation:
    binding: NativeParentResumptionBinding
    delegation: NativeDelegateInvocation
    return_facts: object
    returned_content: object
    closed_child_proof: tuple
    arguments: tuple
    argument_anchors: tuple
    raw_bytes: bytes
    input_parents: tuple
    input_content: object
    turn_frame: _TurnFrame | None
    preparation_contents: tuple
    request: dict | None
    request_bytes: bytes | None
    request_content: object
    envelope: object
    observed_destination: object
    response_bytes: bytes | None
    native_invocation: NativeMainInvocation | None
    native_completion: _NativeCompletion | None
    result: dict | None
    result_bytes: bytes | None
    result_content: object


def _resumption_callback(adapter, callback, *args):
    issued = _issued_window(adapter)
    if issued is not None:
        return issued.callback(callback, *args)
    probe = _tool_call_probes.get(adapter)
    if probe is not None:
        return probe.callback(callback, *args)
    prompt = _memory_prompt_frames.get(adapter)
    owner = (prompt if prompt is not None and prompt.active_call is not None else
             _memory_snapshots.get(adapter) or _parent_resumptions.get(adapter))
    return callback(*args) if owner is None else owner.callback(callback, *args)


def _capture_resumption_binding(function):
    from functools import wraps
    from inspect import signature
    public_signature = signature(function)

    @wraps(function)
    def captured(*args, **kwargs):
        adapter = args[0] if args else kwargs.get("self")
        try:
            _require(type(adapter) is NativeMainPayloadAdapter and adapter in _adapters)
            if adapter in _tool_call_probes:
                adapter._deny()
            _require(all(type(name) is str for name in kwargs))
            bound = public_signature.bind(*args, **kwargs).arguments
            instruction = bound["instruction"]
            _text(instruction)
            _require(bool(instruction))
            system = bound.get("system_message")
            if system is not None:
                _text(system)
            options = (tuple((name, name in bound, bound.get(name))
                             for name in public_signature.parameters), args, tuple(kwargs.items()))
            owner = _ParentResumption(adapter, bound["agent"], options)
            return function(*args, **kwargs)
        except BaseException:
            if type(adapter) is NativeMainPayloadAdapter and adapter in _adapters:
                owner = _parent_resumptions.get(adapter)
                adapter._invalidate()
                if owner is not None and owner.control_exception is not None:
                    raise owner.control_exception
            raise
    return captured


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
    memory_prompt = _memory_prompt_owner(agent)
    if memory_prompt is not None:
        memory_prompt.cache_state(namespace)
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
            _require(memory_prompt is not None or value is None
                     or (type(value) is str and value in ("", system)))
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
        snapshot[name] = {"present": present, "value": (None if memory_prompt is not None
                          and name == "_cached_system_prompt" else value)}
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

    def memory_source(self, request_handle: object, producer_ref: str) -> NativeMemorySource: ...


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
        formatter = _issued_call_format_window(adapter)
        if formatter is not None:
            return formatter.current(handle)
        issued = _issued_window(adapter)
        if issued is not None:
            return issued.current(handle)
        probe = _tool_call_probes.get(adapter)
        if probe is not None:
            return probe.current(handle)
        memory = _memory_prompt_frames.get(adapter)
        if memory is not None and memory.owns_window():
            return memory.current(handle)
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
        formatter = _issued_call_format_window(self.adapter)
        if formatter is not None:
            return formatter.resolve(self)
        issued = _issued_window(self.adapter)
        if issued is not None:
            return issued.resolve(self)
        memory = _memory_prompt_frames.get(self.adapter)
        if memory is not None and memory.owns_window():
            return memory.resolve(self)
        actual = self.adapter._around(self.port.resolve)
        _destination(actual)
        return actual

    def send(self, envelope):
        adapter = self.adapter
        issued = _issued_window(adapter)
        if issued is not None:
            return issued.send(self, envelope)
        probe = _tool_call_probes.get(adapter)
        if probe is not None:
            return probe.send(self, envelope)
        memory = _memory_prompt_frames.get(adapter)
        if memory is not None and memory.owns_window():
            return memory.send(self, envelope)
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
        owner = _parent_resumptions.get(adapter)
        if owner is not None:
            owner.sending(operation, envelope, actual, before)
        if operation.path == "fallback":
            _integer(adapter._fallback_send_count, _MAX_INT - 1, 0)
            adapter._fallback_send_count += 1
        elif operation.auxiliary:
            _integer(adapter._aux_send_count, _MAX_INT - 1, 0)
            adapter._aux_send_count += 1
        else:
            _integer(adapter._send_count, _MAX_INT - 1, 0)
            adapter._send_count += 1
        if owner is not None:
            owner.partial_send = owner.partial_anchor = (*owner.partial_send[:4], adapter._send_count, None)
        try:
            output = _resumption_callback(adapter, self.port.send, envelope)
        except (ConnectionError, TimeoutError) as error:
            if (type(error) in (ConnectionError, TimeoutError) and operation.path == "main"
                    and adapter._fallback_binding is not None):
                adapter._failed_send_observation = _FailedSendObservation(
                    operation, envelope, actual, type(error), before, adapter._send_count,
                )
            raise
        adapter._pending = (envelope, actual, output)
        if owner is not None:
            owner.sent(envelope, actual, output, before, adapter._send_count)
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
        _memory_prompt_frame: object = None,
        _issued_call_frame: object = None,
    ) -> None:
        _require(_delegation_frame is None or _memory_prompt_frame is None)
        _require(_issued_call_frame is None or (_delegation_frame is None and _memory_prompt_frame is None))
        if _issued_call_frame is not None:
            _require(type(_issued_call_frame) is _IssuedCallExecution)
            _issued_call_frame.open_view(self, context)
        if _memory_prompt_frame is not None:
            _require(type(_memory_prompt_frame) is _MemoryPrompt)
            _memory_prompt_frame.open_view(self, context)
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
                wrapper = (_Transport(self, port, main=name == main_transport)
                           if _memory_prompt_frame is None and _issued_call_frame is None else
                           (_memory_prompt_frame or _issued_call_frame).parent._transport_wrappers[name])
                wrappers.append((name, wrapper))
            self._operator_routes = routes
            self._operator_profiles = profiles
            self._transport_wrappers = dict(wrappers)
            self._operator_transports = dict(transports)
            self._operator_resolvers = tuple((port, _method(port.resolve)) for _, port in transports)
            self._operator_configuration = (routes, profiles, transports, tuple(wrappers))
            if _memory_prompt_frame is not None:
                self._operator_configuration = _memory_prompt_frame.catalog
            if _issued_call_frame is not None:
                self._operator_configuration = _issued_call_frame.catalog
            self._main_route = selected[0]
            self._busy = True
            self._checkpoint()
            if _issued_call_frame is not None:
                self._runtime = _issued_call_frame.runtime
            elif _memory_prompt_frame is not None:
                self._runtime = _memory_prompt_frame.runtime
            elif _delegation_frame is not None:
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

    def bind_issued_call_format(
        self, result: NativeIssuedCallResult, *,
        formatter_request: object,
        formatter_selection: NativeSelectionPort,
        formatter_producer_ref: str,
        formatter_main_route_ref: str,
    ) -> NativeIssuedCallFormatInput:
        try:
            owner = _IssuedCallFormat(self, result, (formatter_request, formatter_selection,
                formatter_producer_ref, formatter_main_route_ref), sys._getframe())
            return owner.bind()
        except BaseException as error:
            owner = _issued_call_formats.get(self)
            if owner is not None:
                owner.phase = owner.phase_anchor = "denied"
                owner.active_call = owner.call_anchor = owner.entry = owner.public = None
                if owner.control_exception is not None:
                    raise owner.control_exception
            if not isinstance(error, Exception) or isinstance(error, InterruptedError):
                raise
            if type(error) is NativeContentPolicyDenied:
                raise
            raise NativeContentPolicyDenied("controlled_native_issued_format_denied") from None

    def run_issued_call_format(
        self, authorization: NativeIssuedCallFormatInput,
    ) -> NativeIssuedCallFormatResult:
        try:
            owner = _issued_call_formats.get(self)
            _issued_format_require(owner is not None and owner.parent is self, "binding_missing")
            return owner.run(authorization, sys._getframe())
        except BaseException as error:
            owner = _issued_call_formats.get(self)
            if owner is not None:
                owner.phase = owner.phase_anchor = "denied"
                owner.active_call = owner.call_anchor = owner.entry = owner.public = None
                if owner.control_exception is not None:
                    raise owner.control_exception
            if not isinstance(error, Exception) or isinstance(error, InterruptedError):
                raise
            if type(error) is NativeContentPolicyDenied:
                raise
            raise NativeContentPolicyDenied("controlled_native_issued_format_denied") from None

    @property
    def issued_call_format_results(self) -> tuple[NativeIssuedCallFormatResult, ...]:
        owner = _issued_call_formats.get(self)
        if owner is None:
            return ()
        owner.scope_access()
        return tuple(result for result, _ in owner.records)

    def bind_issued_call_execution(
        self, observation: NativeToolCallObservation, *,
        execution_request: object,
        execution_selection: NativeSelectionPort,
        execution_producer_ref: str,
        execution_main_route_ref: str,
        receiver_request: object,
        receiver_selection: NativeSelectionPort,
        receiver_producer_ref: str,
        child_request: object,
        child_selection: NativeSelectionPort,
        child_producer_ref: str,
        delegate_route_ref: str,
        child_main_route_ref: str,
        child_session_id: str,
        child_task_id: str,
    ) -> NativeIssuedCallInput:
        try:
            owner = _IssuedCallExecution(self, observation, (
                execution_request, execution_selection, execution_producer_ref, execution_main_route_ref,
                receiver_request, receiver_selection, receiver_producer_ref,
                child_request, child_selection, child_producer_ref, delegate_route_ref,
                child_main_route_ref, child_session_id, child_task_id), sys._getframe())
            return owner.bind()
        except BaseException as error:
            _issued_failure(self, error)

    def run_issued_call_execution(
        self, authorization: NativeIssuedCallInput,
    ) -> NativeIssuedCallResult:
        try:
            owner = _issued_call_frames.get(self)
            _issued_require(owner is not None and owner.parent is self, "binding_missing")
            return owner.run(authorization, sys._getframe())
        except BaseException as error:
            _issued_failure(self, error)

    @property
    def issued_call_results(self) -> tuple[NativeIssuedCallResult, ...]:
        owner = _issued_call_frames.get(self)
        if owner is None:
            return ()
        _issued_require(type(owner.result_anchors) is tuple and len(owner.result_anchors) <= 1, "records_changed")
        for result, anchor in owner.result_anchors:
            _issued_require(_same_anchors(_row(result, NativeIssuedCallResult), anchor)
                            and result.binding is owner.registered_binding, "result_changed")
        return tuple(result for result, _ in owner.result_anchors)

    def bind_tool_call_probe(
        self, agent: object, *, system_message: str, user_message: str,
        task_id: str, expected_arguments_json: str,
    ) -> NativeToolCallProbeInput:
        try:
            _probe_require(type(self) is NativeMainPayloadAdapter and self in _adapters, "adapter_invalid")
            owner = _ToolCallProbe(self, agent,
                (system_message, user_message, task_id, expected_arguments_json), sys._getframe())
            return owner.bind()
        except BaseException as error:
            _probe_failure(self, error)

    def run_tool_call_probe(
        self, agent: object, authorization: NativeToolCallProbeInput,
    ) -> NativeToolCallObservation:
        try:
            owner = _tool_call_probes.get(self)
            _probe_require(owner is not None, "binding_missing")
            _probe_require(current_context() is self._bound_context, "scope_invalid")
            owner.access()
            return owner.run(agent, authorization, sys._getframe())
        except BaseException as error:
            _probe_failure(self, error)

    @property
    def tool_call_observations(self) -> tuple[NativeToolCallObservation, ...]:
        owner = _tool_call_probes.get(self)
        if owner is None:
            return ()
        anchors = owner.record_anchors
        _probe_require(type(anchors) is tuple and len(anchors) <= 1, "records_changed")
        records = []
        for record, values in anchors:
            _probe_require(_same_anchors(_row(record, NativeToolCallObservation), values)
                           and record.binding is owner.registered_binding, "record_changed")
            records.append(record)
        return tuple(records)

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
        probe = _tool_call_probes.get(self)
        if probe is not None:
            probe.phase = probe.phase_anchor = "denied"
        prompt = _memory_prompt_frames.get(self)
        if prompt is not None:
            prompt.invalidate()
        memory = _memory_snapshots.get(self)
        if memory is not None:
            memory.phase = memory.phase_anchor = "denied"
            memory.active_call = memory.call_frame = memory.public_frame = None
            memory.active_anchor = memory.getter_frame = None
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
        issued = _issued_window(self)
        issued_scope = False
        if issued is not None:
            issued.access()
            issued_scope = self is issued.parent and issued.source_active()
        prompt = _memory_prompt_frames.get(self)
        probe_scope = False
        if prompt is not None and prompt.owns_window():
            prompt.verify()
            probe_scope = self is prompt.parent and prompt.source_probe_active()
        memory = _memory_snapshots.get(self)
        if memory is not None:
            memory.verify()
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
        if self._constructed is not None and not (memory is not None and memory.closed) and not (
                issued_scope and issued.source_owner.closed):
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
                wrapper_owner = prompt.parent if prompt is not None and self is prompt.adapter else self
                if issued is not None and self is issued.adapter:
                    wrapper_owner = issued.parent
                _require(wrapper.adapter is wrapper_owner and wrapper.port is port)
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
        if self._agent is not None and not (delegate_scope or probe_scope or issued_scope):
            _require(current_context(self._agent) is context)
        if self._turn_agent is not None and not (delegate_scope or probe_scope or issued_scope):
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
        source = _resumption_callback(self, self._admission.current, self._handle)
        self._identity()
        source_row = _binding(source)
        selected = _resumption_callback(self, self._selection_port.current, self._handle)
        self._identity()
        selection_row = _selection(selected)
        producer = _resumption_callback(self, self._result_port.current, self._producer_ref)
        producer_row = _producer(producer)
        self._identity()
        _require(producer_row[0] == self._producer_ref)
        _require(selection_row[:2] == (source_row[6], source_row[7]))
        _require(selection_row[3][1] == "synthetic_native")
        _destination(self._main_route.destination)
        _require(selected.destination == self._main_route.destination)
        prompt = _memory_prompt_frames.get(self)
        retained_delegate_profiles = (prompt.profile_union if prompt is not None
                                      and self is prompt.adapter else ())
        memory = _memory_snapshots.get(self)
        if memory is not None:
            retained_delegate_profiles = memory.original_profiles
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
                if prompt is not None and self is prompt.adapter:
                    _require(source_row[:8] == self._turn_facts[0][:8]
                             and set(source_row[8]) <= set(prompt.consumer_facts[0][8])
                             and (selection_row, producer_row) == self._turn_facts[1:])
                else:
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
        if self in _tool_call_probes:
            self._deny()
        owner = _memory_prompt_frames.get(self) or _parent_resumptions.get(self)
        stage = None
        try:
            _memory_require(self not in _memory_snapshots, "model_work_denied")
            if owner is not None and owner.active_call is not None:
                stage = owner.enter_stage()
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
            if type(owner) is _MemoryPrompt:
                raise
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
            if stage is not None:
                owner.exit_stage(stage)

    def admit_turn(
        self, agent: object, *, user_message: str,
        system_message: str | None,
        conversation_history: list[dict[str, str]] | None,
        task_id: str,
    ) -> NativeTurnInput:
        owner = _memory_prompt_frames.get(self) or _parent_resumptions.get(self)
        if type(owner) is _MemoryPrompt:
            owner.check_input(agent, user_message, system_message, conversation_history, task_id)
        elif owner is not None:
            owner.access()
            _require(owner.phase == "entered" and owner.record.input_content is None)
            _require(_same_anchors((user_message, system_message, conversation_history, task_id),
                                   owner.record.arguments))
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
            if type(owner) is _MemoryPrompt:
                admitted = self._runtime.derive_for_selection(self._handle,
                    (owner.source.result_content, owner.authorization_content), raw, path="main",
                    parent_requests=(owner.parent._handle, self._handle))
                owner.input_content, owner.raw_bytes = admitted, raw
                owner.transition("entered", "admitted")
            elif owner is not None:
                admitted = self._runtime.derive(self._handle, owner.record.input_parents, raw, path="main")
                owner.retain(raw_bytes=raw, input_content=admitted)
                owner.transition("entered", "admitted")
            elif delegation is not None:
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
            if owner is not None:
                owner.progress()
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
            prompt = _memory_prompt_frames.get(self)
            if prompt is not None:
                system = prompt.build_prompt(agent, system)
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
            owner = _memory_prompt_frames.get(self) or _parent_resumptions.get(self)
            if owner is not None:
                owner.progress()
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
            prompt = _memory_prompt_frames.get(self)
            if prompt is not None:
                system = prompt.full_prompt
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
            owner = _memory_prompt_frames.get(self) or _parent_resumptions.get(self)
            if owner is not None:
                owner.progress()
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
            owner = _memory_prompt_frames.get(self) or _parent_resumptions.get(self)
            if owner is not None:
                owner.progress()
                owner.prepared_request = request
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
        if self in _tool_call_probes:
            self._deny()
        _memory_require(self not in _memory_snapshots, "model_work_denied")
        owner = _memory_prompt_frames.get(self) or _parent_resumptions.get(self)
        if owner is None:
            _require(self not in _delegate_frames and not hasattr(self, "_delegate_frame"))
        else:
            owner.access()
            owner.check_stage()
            owner.checkpoint()
        _require(self._fallback_frame is None and self._fallback_terminal is False)

    def bind_delegation(
        self, agent: object, *, receiver_request: object,
        receiver_selection: NativeSelectionPort, receiver_producer_ref: str,
        child_request: object, child_selection: NativeSelectionPort,
        child_producer_ref: str, delegate_route_ref: str,
        child_main_route_ref: str, child_session_id: str, child_task_id: str,
    ) -> None:
        try:
            if self in _tool_call_probes:
                self._deny()
            _memory_require(self not in _memory_snapshots, "new_binding_denied")
            _require(self not in _delegate_frames)
            _DelegateFrame(self, agent, receiver_request, receiver_selection, receiver_producer_ref,
                child_request, child_selection, child_producer_ref, delegate_route_ref,
                child_main_route_ref, child_session_id, child_task_id)
        except BaseException:
            self._invalidate()
            raise

    @_capture_resumption_binding
    def bind_parent_resumption(
        self, agent: object, *, instruction: str,
        system_message: str | None = None,
    ) -> NativeParentResumption:
        return _parent_resumptions[self].bind(instruction, system_message)

    def resume_parent(
        self, agent: object, authorization: NativeParentResumption,
    ) -> dict:
        _require(type(self) is NativeMainPayloadAdapter and self in _adapters)
        owner = _parent_resumptions.get(self)
        try:
            _require(owner is not None)
            return owner.resume(agent, authorization)
        except BaseException:
            self._invalidate()
            if owner is not None and owner.control_exception is not None:
                raise owner.control_exception
            raise

    @_capture_memory_binding
    def bind_memory_snapshot(
        self, agent: object, *, store: object,
        producer_ref: str, target: str = "memory",
    ) -> NativeMemorySnapshotInput:
        return _memory_snapshots[self].bind()

    def read_memory_snapshot(
        self, agent: object, source: NativeMemorySnapshotInput,
    ) -> NativeMemorySnapshotResult:
        _memory_require(type(self) is NativeMainPayloadAdapter and self in _adapters, "adapter_invalid")
        try:
            owner = _memory_snapshots.get(self)
            _memory_require(owner is not None, "binding_missing")
            return owner.read(agent, source, sys._getframe())
        except BaseException as error:
            _memory_failure(self, error)

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
        probe = _tool_call_probes.get(self)
        if probe is not None:
            return probe.observe_operation(operation, result)
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
        probe = _tool_call_probes.get(self)
        if probe is not None:
            return probe.checkpoint()
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
        formatter = _issued_call_format_window(self)
        if formatter is not None:
            return formatter.around(callback)
        issued = _issued_window(self)
        if issued is not None:
            return issued.around(callback)
        prompt = _memory_prompt_frames.get(self)
        if prompt is not None and prompt.owns_window() and self is prompt.parent:
            return prompt.around(callback)
        try:
            self._checkpoint()
            output = _resumption_callback(self, callback)
            self._checkpoint()
            return output
        except BaseException as error:
            _remember_delegate_control(_delegate_frames.get(self) or getattr(self, "_delegate_parent_frame", None), error)
            raise

    def validate_scope(self) -> None:
        try:
            issued = _issued_call_frames.get(self)
            if issued is not None and self is issued.adapter:
                issued.scope_access()
                return
            probe = _tool_call_probes.get(self)
            if probe is not None:
                probe.scope_access()
                return
            prompt = _memory_prompt_frames.get(self)
            if prompt is not None and self is prompt.adapter:
                prompt.access()
                prompt.checkpoint()
            memory = _memory_snapshots.get(self)
            if memory is not None:
                memory.scope_access()
            _require(not self._busy)
            self._busy = True
            self._checkpoint()
            if memory is not None:
                memory.checkpoint()
        except BaseException as error:
            if _memory_snapshots.get(self) is not None:
                _memory_failure(self, error)
            if isinstance(error, Exception):
                self._deny()
            if self._preparation_enabled:
                self._invalidate()
            raise
        finally:
            self._busy = False

    def dispatch(self, agent: object, api_kwargs: dict) -> NativeMainResponse:
        owner = _memory_prompt_frames.get(self) or _parent_resumptions.get(self)
        stage = None
        try:
            if owner is not None:
                stage = owner.enter_stage()
                _require(owner.phase == ("request-prepared" if type(owner) is _MemoryPrompt else "admitted")
                         and api_kwargs is owner.prepared_request)
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
            if owner is not None:
                owner.progress()
            if frame is not None:
                self._turn_pending = None
                self._turn_arguments = self._turn_raw = self._turn_state = self._turn_options = self._turn_facts = None
                self._interrupt_turn()
            return response
        except InterruptedError:
            if type(owner) is _MemoryPrompt:
                self._invalidate()
                raise
            if not self._preparation_enabled:
                self._deny()
            self._invalidate()
            raise InterruptedError("Agent interrupted during API call") from None
        except Exception:
            if type(owner) is _MemoryPrompt:
                self._invalidate()
                raise
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
            if stage is not None:
                owner.exit_stage(stage)


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


def dispatch_controlled_main(agent: object, api_kwargs: dict) -> tuple[bool, NativeMainResponse | NativeToolCallObservation | None]:
    context = current_context(agent)
    if context is not None and context._controlled_main_bound:
        adapter = _attached(context)
        probe = _tool_call_probes.get(adapter)
        if probe is not None:
            try:
                return True, probe.dispatch(agent, api_kwargs, sys._getframe())
            except BaseException as error:
                _probe_failure(adapter, error)
        return True, adapter.dispatch(agent, api_kwargs)
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
    prompt = _memory_prompt_owner(agent)
    if prompt is not None:
        prompt.cache_state(namespace)
    for name, original in binding.anchors:
        if name not in dynamic and not (prompt is not None and name in _MEMORY_CACHE_NAMES):
            _require(namespace[name] is original)
    state = namespace if prompt is None else dict(namespace, _cached_system_prompt=None,
                                                _cached_system_prompt_static=None)
    _require(_constructed_state(state) == binding.state)
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
        prompt = _memory_prompt_frames.get(adapter)
        if prompt is not None:
            prompt.agent = agent
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
        if _memory_prompt_frames.get(adapter) is not None:
            agent._cached_system_prompt_static = None
        binding.anchors = tuple(vars(agent).items())
        binding.state = _constructed_state(vars(agent))
        _constructed_bindings.add(binding)
        adapter.validate_scope()
    except BaseException:
        _invalidate_constructed_attempt(agent, adapter)
        raise


def _constructed_adapter(agent):
    try:
        probe_binding = _owned_constructed_binding(agent)
        if probe_binding is not None and probe_binding.adapter in _tool_call_probes:
            probe_binding.adapter._deny()
        binding = _namespace(agent).get("_native_controlled_binding")
        owned = _owned_constructed_binding(agent)
        if owned is not None:
            _memory_require(owned.adapter not in _memory_snapshots, "model_work_denied")
        owner = (_memory_prompt_frames.get(owned.adapter) or _parent_resumptions.get(owned.adapter)
                 if owned is not None else None)
        if owner is not None:
            owner.access()
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
        owner = _memory_prompt_frames.get(adapter) or _parent_resumptions.get(adapter)
        if owner is not None:
            owner.progress()
            owner.transition("sent", "completed")
        adapter._interrupt_turn()
        return result


def _close_constructed_main(agent) -> bool:
    if not _controlled_attempt(agent):
        return False
    owned = _owned_constructed_binding(agent)
    probe = _tool_call_probes.get(owned.adapter) if owned is not None else None
    if probe is not None and probe.phase == "released":
        try:
            return probe.close()
        except BaseException as error:
            _probe_failure(owned.adapter, error)
    memory = _memory_snapshots.get(owned.adapter) if owned is not None else None
    if memory is not None and memory.phase == "released":
        try:
            _memory_require(current_context() in (None, owned.adapter._bound_context), "scope_invalid")
            memory.scope_access()
            memory.checkpoint()
            memory.closed = True
            owned.closed = True
            memory.checkpoint()
            return True
        except BaseException as error:
            _memory_failure(owned.adapter, error)
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


class _MemorySnapshot:
    """Own one authenticated read of the original frozen memory block."""

    def __init__(self, parent, agent, store, producer_ref, target, options):
        from run_agent import AIAgent
        _memory_require(type(agent) is AIAgent, "agent_invalid")
        _memory_require(parent not in _memory_snapshots and not hasattr(parent, "_memory_snapshot_owner"),
                        "new_binding_denied")
        _memory_require(parent not in _delegate_frames and parent not in _parent_resumptions
                        and not hasattr(parent, "_delegate_parent_frame")
                        and not hasattr(parent, "_delegate_frame"), "parent_unavailable")
        _memory_require(parent._constructed is _owned_constructed_binding(agent)
                        and parent._constructed is not None and parent._turn_agent is agent, "agent_invalid")
        _memory_require(threading.get_ident() == parent._thread, "thread_invalid")
        _memory_require(current_context() is parent._bound_context
                        and current_context(agent) is parent._bound_context, "scope_invalid")
        _memory_require(not parent._busy and not parent._invalid and not parent._bound_context._denied,
                        "parent_unavailable")
        _memory_require(not parent._turn_frames and not parent._records and not parent._completions
                        and not parent._aux_frames and not parent._aux_records
                        and not parent._aux_local_results and not parent._fallback_records
                        and not parent._failed_main_attempts, "parent_unavailable")
        _memory_require(parent._pending is parent._operation is parent._checked_operation
                        is parent._turn_pending is parent._aux_pending is None, "parent_unavailable")
        _memory_require(parent._aux_binding is parent._compression_binding is parent._compression_owner
                        is parent._fallback_binding is parent._fallback_frame is None
                        and not parent._fallback_terminal, "parent_unavailable")
        _verify_constructed(parent._constructed)
        self.parent, self.agent, self.store = parent, agent, store
        self.producer_ref, self.target, self.options = producer_ref, target, options
        self.option_anchors = _nested_anchors(options)
        self.runtime, self.thread = parent._runtime, parent._thread
        self.original_profiles = parent._original_profiles
        self.phase = self.phase_anchor = "bound"
        self.depth = 0
        self.closed = self.checking = False
        self.active_call = self.active_anchor = self.call_frame = self.public_frame = None
        self.getter_frame = self.getter_method = None
        self.getter_entered = self.getter_returned = False
        self.getter_block = self.result_bytes = self.result_content = None
        self.source_content = self.request_content = None
        self.source = self.binding = self.registered_binding = self.record = self.registered_record = None
        self.control_exception = None
        self.constructed = parent._constructed
        self.catalog = parent._operator_configuration
        self.catalog_anchor = _nested_anchors(self.catalog)
        self.native_facts = tuple((name, getattr(parent, name)) for name in (
            "_handle", "_admission", "_selection_port", "_result_port", "_clock",
            "_main_route", "_route_ref", "_producer_ref", "_original_profiles", "_holder"))
        self.initial_source, self.selection, self.main_producer = parent._source, parent._selected, parent._producer_policy
        self.initial_facts = (_binding(self.initial_source), _selection(self.selection), _producer(self.main_producer))
        self.initial_anchor = _nested_anchors((self.initial_source, self.selection, self.main_producer))[1:]
        self.proof = (parent._turn_frames, parent._records, parent._completions,
                      parent._aux_frames, parent._aux_records, parent._aux_local_results,
                      parent._failed_main_attempts, parent._fallback_records)
        self.methods = tuple((port, name, _method(getattr(port, name))) for port, name in (
            (self.runtime, "_check_operator"), (self.runtime, "_operator"),
            (self.runtime, "admit"), (self.runtime, "derive"), (self.runtime, "_content"),
            (self.runtime._admission, "current"),
            (parent._result_port, "current"),
            (parent, "bind_memory_snapshot"), (parent, "read_memory_snapshot"),
            (parent, "_identity"), (parent, "_snapshot"), (parent, "_checkpoint"),
            (parent, "_around"), (parent, "_model_open"), (parent, "validate_scope"),
            (parent, "_invalidate"), (parent, "_turn_operation"),
            (sys.modules[__name__], "_resumption_callback"),
            (sys.modules[__name__], "_constructed_adapter"),
            (sys.modules[__name__], "_close_constructed_main")))
        self.methods += tuple((self, name, _method(getattr(self, name))) for name in (
            "verify", "health", "callback", "checkpoint", "source_facts", "store_facts",
            "bind", "read", "scope_access", "access", "content"))
        parent._memory_snapshot_owner = self
        _memory_snapshots[parent] = self

    def health(self):
        _memory_require(self.parent._runtime is self.runtime is self.parent._captured_runtime, "runtime_changed")
        for port, name, method in self.methods:
            reason = "source_lookup_changed" if name == "memory_source" else "method_changed"
            _memory_require(_same_method(getattr(port, name, None), method), reason)
        self.runtime._check_operator()

    def callback(self, callback, *args):
        _memory_require(self.depth == 0, "reentry_denied")
        self.health()
        self.depth += 1
        try:
            prompt = _memory_prompt_frames.get(self.parent)
            if prompt is not None and prompt.active_call is not None:
                return prompt.callback(callback, *args)
            return callback(*args)
        except BaseException as error:
            import asyncio
            if isinstance(error, (InterruptedError, asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                if self.control_exception is None:
                    self.control_exception = error
            raise
        finally:
            self.depth -= 1
            if self.control_exception is None:
                self.health()
                self.verify()

    def store_facts(self):
        from tools import memory_tool
        _memory_require(memory_tool.MemoryStore is _BUILTIN_MEMORY_STORE
                        and type(self.store) is _BUILTIN_MEMORY_STORE, "store_changed")
        _memory_require(_same_anchors(tuple(value for pair in vars(_BUILTIN_MEMORY_STORE).items() for value in pair),
                                     _MEMORY_CLASS_ANCHORS), "store_class_changed")
        _memory_require(_BUILTIN_MEMORY_STORE.format_for_system_prompt is _MEMORY_GETTER
                        and _MEMORY_GETTER.__code__ is _MEMORY_GETTER_CODE
                        and _same_method(self.store.format_for_system_prompt, (self.store, _MEMORY_GETTER)),
                        "getter_changed")
        namespace = object.__getattribute__(self.store, "__dict__")
        snapshot = namespace.get("_system_prompt_snapshot")
        _memory_require(type(snapshot) is dict and len(snapshot) == 2, "snapshot_invalid")
        _memory_require(all(type(key) is str for key in snapshot)
                        and set(snapshot) == {"memory", "user"}, "snapshot_invalid")
        _memory_require(all(type(value) is str for value in snapshot.values()), "snapshot_invalid")
        block = snapshot[self.target]
        _memory_require(not any(0xD800 <= ord(char) <= 0xDFFF for char in block), "snapshot_invalid")
        flags = namespace.get("memory_enabled"), namespace.get("user_profile_enabled")
        _memory_require(flags[0] is True and type(flags[1]) is bool, "flags_invalid")
        limits = namespace.get("memory_char_limit"), namespace.get("user_char_limit")
        _memory_require(all(type(value) is int and 1 <= value <= _MAX_INT for value in limits), "limits_invalid")
        return namespace, snapshot, tuple(snapshot.items()), flags, limits

    def source_facts(self):
        port = self.parent._result_port
        policy = self.callback(port.current, self.producer_ref)
        _memory_require(type(policy) is NativeResultPolicy, "producer_type_invalid")
        row = _producer(policy)
        _memory_require(row[0] == self.producer_ref, "producer_changed")
        _memory_require(set(row[2]) <= set(self.original_profiles), "producer_restrictions_invalid")
        source = self.callback(port.memory_source, self.parent._handle, self.producer_ref)
        _memory_require(type(source) is NativeMemorySource, "source_type_invalid")
        values = _row(source, NativeMemorySource)
        _memory_require(source.store is self.store and source.target is self.target, "source_association_changed")
        _memory_require(type(source.target) is str and source.target == "memory", "target_invalid")
        _identifier(source.owner_ref)
        _identifier(source.profile_ref)
        _memory_require(source.owner_ref == self.initial_facts[0][0], "source_owner_invalid")
        _memory_require(type(source.source_version) is int and 1 <= source.source_version <= _MAX_INT,
                        "source_version_invalid")
        _memory_require(source.producer_policy is policy, "producer_identity_changed")
        if self.source is not None:
            _memory_require(source is self.source and _same_anchors(values, self.source_anchor), "source_changed")
            _memory_require(policy is self.producer and row == self.producer_facts, "producer_changed")
        return source, policy, row

    def verify(self):
        p = self.parent
        _memory_require(_memory_snapshots.get(p) is self and p._memory_snapshot_owner is self, "owner_changed")
        _memory_require(self.phase is self.phase_anchor and self.phase in
                        ("bound", "entered", "completed", "released"), "owner_denied")
        _memory_require(self.binding is self.registered_binding and self.record is self.registered_record,
                        "binding_changed")
        if self.phase == "bound":
            _memory_require(self.record is None and not self.getter_entered and not self.getter_returned
                            and self.result_bytes is self.result_content is None, "capture_repeated")
        elif self.phase in ("completed", "released"):
            _memory_require(self.record is not None and self.getter_returned
                            and self.result_content is not None, "result_changed")
        if self.phase in ("entered", "completed"):
            _memory_require(self.active_call is not None, "call_invalid")
        _memory_require(threading.get_ident() == self.thread == p._thread, "thread_invalid")
        _memory_require(self.depth == 0, "reentry_denied")
        _memory_require(not p._invalid and not p._bound_context._denied, "owner_denied")
        _memory_require(_same_anchors(_nested_anchors(self.options), self.option_anchors), "options_changed")
        _memory_require(p._operator_configuration is self.catalog
                        and _same_anchors(_nested_anchors(self.catalog), self.catalog_anchor), "catalog_changed")
        _memory_require(all(getattr(p, name) is value for name, value in self.native_facts), "holder_changed")
        _memory_require(_same_anchors(_nested_anchors((self.initial_source, self.selection, self.main_producer))[1:],
                                     self.initial_anchor), "holder_changed")
        _memory_require(_same_anchors((p._turn_frames, p._records, p._completions,
                        p._aux_frames, p._aux_records, p._aux_local_results,
                        p._failed_main_attempts, p._fallback_records), self.proof), "parent_changed")
        _memory_require(p._constructed is self.constructed and p._turn_agent is self.agent
                        and _owned_constructed_binding(self.agent) is self.constructed, "agent_changed")
        _memory_require(p._aux_binding is p._compression_binding is p._compression_owner
                        is p._fallback_binding is p._fallback_frame is None
                        and not p._fallback_terminal and p not in _delegate_frames
                        and p not in _parent_resumptions, "parent_changed")
        _memory_require(p._pending is p._operation is p._checked_operation
                        is p._turn_pending is p._aux_pending is None, "parent_changed")
        namespace = _namespace(self.agent)
        _memory_require(namespace.get("_memory_store") is None
                        and namespace.get("_memory_manager") is None, "agent_memory_changed")
        if self.closed:
            b = self.constructed
            _memory_require(self.phase == "released" and b.closed is True, "parent_changed")
            _memory_require(namespace.get("_native_controlled_binding") is b
                            and namespace.get("_native_content_policy_context") is p._bound_context, "agent_changed")
            _memory_require(set(namespace) == {name for name, _ in b.anchors}
                            and _constructed_state(namespace) == b.state, "agent_changed")
            dynamic = {"_interrupt_requested", "_interrupt_message", "_tool_interrupt_reason",
                       "_interrupt_thread_signal_pending"}
            _memory_require(all(namespace[name] is original for name, original in b.anchors if name not in dynamic),
                            "agent_changed")
            for port, name, method in b.methods:
                _memory_require(_same_method(getattr(port, name), method), "method_changed")
        else:
            _verify_constructed(self.constructed)
        interrupted = namespace.get("_interrupt_requested")
        _memory_require(type(interrupted) is bool and type(namespace.get("_interrupt_thread_signal_pending")) is bool
                        and namespace.get("_interrupt_message") is None
                        and (namespace.get("_tool_interrupt_reason") is None
                             or namespace.get("_tool_interrupt_reason") == "user interrupt"), "control_invalid")
        if interrupted:
            raise InterruptedError("Agent interrupted during memory snapshot")
        if self.active_call is not None:
            self.access()
        if self.source is not None:
            _memory_require(_same_anchors(_row(self.source, NativeMemorySource), self.source_anchor), "source_changed")
            _memory_require(_producer(self.producer) == self.producer_facts, "producer_changed")
            actual = self.store_facts()
            _memory_require(actual[0] is self.store_snapshot[0] and actual[1] is self.store_snapshot[1]
                            and _same_anchors(_nested_anchors(actual[0]), self.store_anchor), "snapshot_changed")
        if self.binding is not None:
            b = self.binding
            _memory_require(b is self.registered_binding
                            and _same_anchors(_row(b, NativeMemorySnapshotBinding), self.binding_anchor), "binding_changed")
            _memory_require(_same_anchors(_nested_anchors(b.snapshots), b.anchors), "binding_changed")
            self.content(b.source_content, b.source_bytes, ())
            self.content(b.request_content, b.request_bytes, ())
        if self.record is not None:
            r = self.record
            _memory_require(r is self.registered_record
                            and _same_anchors(_row(r, NativeMemorySnapshotResult), self.record_anchor), "result_changed")
            _memory_require(r.binding is self.binding and r.block is self.getter_block
                            and r.result_bytes is self.result_bytes and r.result_content is self.result_content,
                            "result_changed")
            _memory_require(_same_anchors(r.parents, (self.binding.request_content, self.binding.source_content)),
                            "result_changed")
            _memory_require(r.result_bytes == _json(dict(target=self.target, block=r.block)), "result_changed")
            self.content(r.result_content, r.result_bytes, r.parents)

    def content(self, handle, payload, parents):
        record = self.runtime._content(handle)
        _memory_require(record.payload is payload and _same_anchors(record.parents, parents), "content_changed")
        _memory_require(set((*self.original_profiles, *self.producer_facts[2])) <= set(record.profiles),
                        "retained_restrictions_missing")

    def checkpoint(self):
        _memory_require(not self.checking, "reentry_denied")
        self.checking = True
        try:
            self.health()
            self.parent._checkpoint()
            for _ in range(2):
                source = self.callback(self.parent._admission.current, self.parent._handle)
                selected = self.callback(self.parent._selection_port.current, self.parent._handle)
                main_producer = self.callback(self.parent._result_port.current, self.parent._producer_ref)
                facts = _binding(source), _selection(selected), _producer(main_producer)
                _memory_require(facts[0][:8] == self.initial_facts[0][:8]
                                and set(facts[0][8]) <= set(self.original_profiles), "holder_changed")
                _memory_require(selected is self.selection and facts[1] == self.initial_facts[1], "selection_changed")
                _memory_require(main_producer is self.main_producer and facts[2] == self.initial_facts[2], "producer_changed")
                if self.source is not None:
                    self.source_facts()
                now = self.callback(self.parent._clock)
                _integer(now, minimum=0)
                _memory_require(now < self.parent._main_route.expires_at, "selection_expired")
                allowed = {r.route_ref for r in self.parent._operator_routes}
                for profile in self.parent._operator_profiles:
                    if profile.profile_ref in self.original_profiles:
                        allowed.intersection_update(profile.allowed_routes)
                for path in shared_policy().PATHS:
                    _memory_require(any(r.path == path and r.route_ref in allowed
                        and (r.consumer_ref, r.selection_id, r.selection_version)
                            == (facts[0][3], facts[0][6], facts[0][7]) and now < r.expires_at
                        for r in self.parent._operator_routes), "coverage_missing")
                for route in self.parent._operator_routes:
                    actual = self.callback(self.parent._operator_transports[route.destination.transport_ref].resolve)
                    _memory_require(actual is route.destination
                                    and _destination(actual) == _destination(route.destination), "destination_changed")
                self.health()
                self.verify()
        finally:
            self.checking = False

    def bind(self):
        p = self.parent
        try:
            _memory_require(self.depth == 0 and self.binding is None, "binding_invalid")
            lookup = getattr(p._result_port, "memory_source", None)
            _memory_require(callable(lookup), "source_lookup_missing")
            self.methods += ((p._result_port, "memory_source", _method(lookup)),)
            p._busy = True
            self.store_snapshot = self.store_facts()
            self.store_anchor = _nested_anchors(self.store_snapshot[0])
            self.checkpoint()
            source, producer, facts = self.source_facts()
            self.source, self.producer, self.producer_facts = source, producer, facts
            self.source_anchor = _row(source, NativeMemorySource)
            self.checkpoint()
            source_bytes = _json(dict(block=self.store_snapshot[1][self.target], target=self.target,
                flags=dict(memory_enabled=self.store_snapshot[3][0], user_profile_enabled=self.store_snapshot[3][1]),
                limits=dict(memory_char_limit=self.store_snapshot[4][0], user_char_limit=self.store_snapshot[4][1]),
                source=dict(owner_ref=source.owner_ref, profile_ref=source.profile_ref, source_version=source.source_version),
                producer=facts, holder=self.initial_facts[0], selection=_selection_value(self.selection),
                main_producer=self.initial_facts[2], options=[(name, present) for name, present, _ in self.options[0]]))
            request_bytes = _json(dict(target=self.target))
            _memory_require(len(source_bytes) <= _MAX_NATIVE_RESULT, "source_size_invalid")
            self.source_bytes, self.request_bytes = source_bytes, request_bytes
            self.source_content = self.runtime.admit(p._handle, source_bytes)
            self.health()
            self.content(self.source_content, source_bytes, ())
            self.checkpoint()
            self.request_content = self.runtime.admit(p._handle, request_bytes)
            self.health()
            self.content(self.request_content, request_bytes, ())
            self.checkpoint()
            snapshots = (self.source, self.source_anchor, self.initial_source, self.selection, self.main_producer,
                         self.producer, self.producer_facts, self.store_snapshot, self.constructed,
                         self.constructed.methods, self.catalog, self.methods)
            token = object.__new__(NativeMemorySnapshotInput)
            self.binding = NativeMemorySnapshotBinding(token, self.agent, p._handle, self.store, self.target,
                producer, self.selection, source_bytes, self.source_content, request_bytes, self.request_content,
                self.options, snapshots, _nested_anchors(snapshots), self.thread)
            self.registered_binding = self.binding
            self.binding_anchor = _row(self.binding, NativeMemorySnapshotBinding)
            self.checkpoint()
            return token
        finally:
            p._busy = False

    def access(self):
        _memory_require(self.depth == 0 and threading.get_ident() == self.thread, "reentry_denied")
        _memory_require(self.active_call is self.active_anchor and self.active_call is not None, "call_invalid")
        frame, public = self.call_frame, self.public_frame
        _memory_require(frame is not None and frame.f_code is type(self).read.__code__
                        and public is frame.f_back and public.f_code is NativeMainPayloadAdapter.read_memory_snapshot.__code__,
                        "call_invalid")
        _memory_require(frame.f_locals.get("self") is self and frame.f_locals.get("agent") is self.agent
                        and frame.f_locals.get("token") is self.binding.token
                        and public.f_locals.get("self") is self.parent
                        and public.f_locals.get("agent") is self.agent
                        and public.f_locals.get("source") is self.binding.token, "call_invalid")
        current = sys._getframe()
        while current is not None and current is not frame:
            current = current.f_back
        _memory_require(current is frame, "call_invalid")

    def scope_access(self):
        _memory_require(self.depth == 0 and self.active_call is None and not self.checking, "reentry_denied")
        self.verify()

    def read(self, agent, token, public_frame):
        _memory_require(self.depth == 0 and self.active_call is None, "reentry_denied")
        _memory_require(self.phase == "bound" and self.binding is not None and self.record is None
                        and not self.getter_entered and not self.getter_returned, "capture_repeated")
        _memory_require(agent is self.agent and type(token) is NativeMemorySnapshotInput
                        and token is self.binding.token, "token_invalid")
        _memory_require(threading.get_ident() == self.thread, "thread_invalid")
        _memory_require(current_context() is self.parent._bound_context
                        and current_context(agent) is self.parent._bound_context, "scope_invalid")
        self.active_call = self.active_anchor = object()
        self.call_frame, self.public_frame = sys._getframe(), public_frame
        self.phase = self.phase_anchor = "entered"
        self.parent._busy = True
        try:
            self.checkpoint()
            self.access()
            self.getter_method = self.store.format_for_system_prompt
            self.getter_frame = self.call_frame
            self.health()
            self.getter_entered = True
            block = self.getter_method(self.target)
            self.getter_block, self.getter_returned = block, True
            _memory_require(block is None or type(block) is str, "getter_result_invalid")
            self.result_bytes = _json(dict(block=block, target=self.target))
            _memory_require(len(self.result_bytes) <= _MAX_NATIVE_RESULT, "result_size_invalid")
            _memory_require(block is (self.store_snapshot[1][self.target] or None), "getter_result_changed")
            self.health()
            self.checkpoint()
            parents = (self.binding.request_content, self.binding.source_content)
            self.result_content = self.runtime.derive(self.parent._handle, parents, self.result_bytes, path="memory")
            self.health()
            self.record = NativeMemorySnapshotResult(self.binding, block, self.result_bytes, self.result_content, parents)
            self.registered_record = self.record
            self.record_anchor = _row(self.record, NativeMemorySnapshotResult)
            self.phase = self.phase_anchor = "completed"
            self.checkpoint()
            self.phase = self.phase_anchor = "released"
            self.checkpoint()
            return self.record
        except BaseException as error:
            _memory_failure(self.parent, error)
        finally:
            self.parent._busy = False
            self.active_call = self.active_anchor = self.call_frame = self.public_frame = None
            self.getter_frame = None


_MEMORY_CACHE_NAMES = ("_cached_system_prompt", "_cached_system_prompt_static")


class NativeMemoryPromptInput:
    __slots__ = ()

    def __new__(cls):
        raise TypeError("Memory prompt tokens require operator admission")


def _memory_prompt_owner(agent):
    binding = _owned_constructed_binding(agent)
    frame = _memory_prompt_frames.get(binding.adapter) if binding is not None else None
    return frame if frame is not None and frame.adapter is binding.adapter else None


def _memory_prompt_access(agent, kind, system=None):
    binding = _owned_constructed_binding(agent)
    if binding is None:
        _require(not _controlled_attempt(agent))
        return None
    owner = _memory_prompt_frames.get(binding.adapter)
    try:
        _memory_require(binding.adapter not in _memory_snapshots, "prompt_work_denied")
        _require(owner is not None and owner.adapter is binding.adapter)
        return owner.prompt_access(agent, kind, system)
    except BaseException:
        binding.adapter._invalidate()
        raise


class _MemoryPrompt:
    """Own one source attempt and one original runtime consumer view."""

    def __init__(self, source_owner, source, context, selection, result_policy,
                 producer, route, system, session, entry):
        self.source_owner, self.source = source_owner, source
        self.parent, self.context = source_owner.parent, context
        self.runtime, self.thread = source_owner.runtime, source_owner.thread
        self.options = (context, selection, result_policy, producer, route, system, session)
        self.option_anchors = _nested_anchors(self.options)
        self.catalog = source_owner.catalog
        self.catalog_anchor = source_owner.catalog_anchor
        self.adapter = self.agent = self.token = None
        self.phase = self.phase_anchor = "bound"
        self.depth = self.stage_index = 0
        self.active_call = self.active_anchor = object()
        self.entry = entry
        self.public_frame = self.stage = self.stage_anchor = self.probe = None
        self.control_exception = None
        self.checking = self.constructing = self.terminal = False
        self.consumer_source = self.consumer_selection = self.consumer_producer = None
        self.consumer_facts = None
        self.profile_union = source_owner.original_profiles
        self.authorization_bytes = self.authorization_content = None
        self.input_content = self.raw_bytes = self.arguments = None
        self.prepared_request = self.partial_send = None
        self.full_prompt = self.static_prompt = None
        self.prompt_frame = self.forward_frame = self.builder_frame = self.parts_frame = None
        self.prompt_step = 0
        self.cache_frozen = False
        self.methods = ()
        from agent import conversation_loop, system_prompt
        from run_agent import AIAgent
        self.stage_codes = tuple(_method(f)[-1].__code__ for f in (
            conversation_loop._try_controlled_text_turn, self.parent.admit_turn,
            self.parent._prepare_turn, self.parent.seal_turn, self.parent.prepared_request,
            self.parent.dispatch, _complete_controlled_turn,
            conversation_loop._try_controlled_text_turn))
        self.methods = tuple((port, name, _method(getattr(port, name))) for port, name in (
            (AIAgent, "__init__"), (AIAgent, "_build_system_prompt"),
            (NativeMainPayloadAdapter, "__init__"),
            (system_prompt, "build_system_prompt"), (system_prompt, "build_system_prompt_parts"),
            (system_prompt, "_append_builtin_memory_parts"),
            (sys.modules[__name__], "bind_memory_prompt_consumer"),
            (sys.modules[__name__], "run_memory_prompt_consumer"),
            (sys.modules[__name__], "_memory_prompt_access"),
            (sys.modules[__name__], "_prepare_controlled_turn"),
            (sys.modules[__name__], "_complete_controlled_turn"),
            (self.runtime, "_check_operator"), (self.runtime, "_operator"),
            (self.runtime, "admit"), (self.runtime, "derive"),
            (self.runtime, "derive_for_selection"), (self.runtime, "dispatch"),
            (self.runtime, "_content"), (self.runtime._admission, "current")))
        self.methods += tuple((self, name, _method(getattr(self, name))) for name in (
            "access", "verify", "checkpoint", "callback", "source_probe", "source_probe_active",
            "owns_window", "current", "around", "resolve", "send", "open_view", "check_input",
            "enter_stage", "check_stage", "exit_stage", "progress", "transition", "build_prompt",
            "prompt_access", "cache_state", "invalidate"))
        self.runtime_anchors = (self.runtime._admission, self.runtime._clock,
                                self.runtime._transports)
        self.runtime_anchor_facts = _nested_anchors(self.runtime_anchors)

    def invalidate(self):
        self.phase = self.phase_anchor = "denied"
        self.terminal = True
        if self.adapter is not None:
            self.adapter._invalid = True
        self.context._denied = True

    def owns_window(self):
        if self.active_call is None:
            return False
        self.access()
        return True

    def access(self):
        _require(self.depth == 0 and threading.get_ident() == self.thread)
        _require(self.active_call is self.active_anchor and self.active_call is not None)
        _require(self.phase is self.phase_anchor and self.phase != "denied")
        frame = sys._getframe(1)
        while frame is not None and frame is not self.entry:
            frame = frame.f_back
        _require(frame is self.entry and frame is not None)
        if self.token is None:
            _require(frame.f_code is bind_memory_prompt_consumer.__code__)
            _require(frame.f_locals.get("source_adapter") is self.parent
                     and frame.f_locals.get("source") is self.source)
        else:
            _require(frame.f_code is run_memory_prompt_consumer.__code__)
            _require(frame.f_locals.get("token") is self.token)
            _require(frame.f_locals.get("user_message") is self.arguments[0]
                     and frame.f_locals.get("task_id") is self.arguments[3])
        _require(_memory_prompt_frames.get(self.parent) is self)
        _require(getattr(self.source_owner, "memory_prompt_attempt", None) is self)

    def source_probe_active(self):
        self.access()
        if self.probe is None:
            return False
        frame = sys._getframe(1)
        suspended, caller, _ = self.probe
        while frame is not None and frame is not caller:
            frame = frame.f_back
        _require(frame is caller and frame.f_code in (type(self).checkpoint.__code__, type(self).around.__code__))
        _require(suspended.f_code is type(self).source_probe.__wrapped__.__code__
                 and suspended.f_locals.get("self") is self and frame.f_locals.get("self") is self)
        return True

    @contextmanager
    def source_probe(self):
        self.access()
        _require(self.probe is None)
        self.probe = (sys._getframe(), sys._getframe(2), object())
        try:
            yield
        finally:
            self.probe = None

    def callback(self, callback, *args):
        self.access()
        self.depth += 1
        try:
            return callback(*args)
        except BaseException as error:
            import asyncio
            if isinstance(error, (InterruptedError, asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                if self.control_exception is None:
                    self.control_exception = error
            raise
        finally:
            self.depth -= 1

    def verify(self):
        self.access()
        p, m = self.parent, self.source_owner
        _require(_memory_snapshots.get(p) is m and m.record is m.registered_record is self.source)
        _require(m.phase == "released" and not m.closed and m.active_call is None
                 and m.getter_entered and m.getter_returned)
        _require(self.source.binding is m.binding is m.registered_binding)
        _require(p._runtime is p._captured_runtime is self.runtime)
        _require(not p._invalid and not p._bound_context._denied and not p._constructed.closed)
        _require(_same_anchors(_nested_anchors(self.options), self.option_anchors))
        _require(p._operator_configuration is self.catalog
                 and _same_anchors(_nested_anchors(self.catalog), self.catalog_anchor))
        _require(_same_anchors((self.runtime._admission, self.runtime._clock, self.runtime._transports),
                               self.runtime_anchors))
        _require(_same_anchors(_nested_anchors(self.runtime._transports),
                               _nested_anchors(self.runtime_anchors[2])))
        for port, name, method in self.methods:
            _require(_same_method(getattr(port, name), method))
        if self.adapter is not None:
            c = self.adapter
            _require(_memory_prompt_frames.get(c) is self and c._memory_prompt_frame is self)
            _require(c._bound_context is self.context and self.context.admission is p._admission)
            if hasattr(c, "_runtime"):
                _require(c._runtime is c._captured_runtime is self.runtime)
            _require(not c._invalid and not self.context._denied)
            if c._constructed is not None:
                _require(not c._constructed.closed and c._constructed.agent is self.agent)
        _require(p.send_count == p.auxiliary_send_count == p.fallback_send_count == 0)
        if self.consumer_facts is not None:
            _require(_binding(self.consumer_source)[:8] == self.consumer_facts[0][:8])
            _require(_selection(self.consumer_selection) == self.consumer_facts[1])
            _require(_producer(self.consumer_producer) == self.consumer_facts[2])
        for handle, payload, parents in (
            (self.authorization_content, self.authorization_bytes, ()),
            (self.input_content, self.raw_bytes, (self.source.result_content, self.authorization_content)),
        ):
            if handle is not None:
                content = self.runtime._content(handle)
                _require(content.payload is payload and _same_anchors(content.parents, parents))
                _require(set(self.profile_union) <= set(content.profiles))

    def checkpoint(self):
        self.verify()
        _require(not self.checking)
        self.checking = True
        try:
            with self.source_probe():
                self.source_owner.checkpoint()
            for _ in range(2):
                source = self.callback(self.parent._admission.current, self.context.request_handle)
                selected = self.callback(self.options[1].current, self.context.request_handle)
                producer = self.callback(self.options[2].current, self.options[3])
                facts = _binding(source), _selection(selected), _producer(producer)
                _require(facts[0][:6] == self.source_owner.initial_facts[0][:6])
                _require(facts[1][:2] == facts[0][6:8] and facts[1][3][1] == "synthetic_native")
                _require(facts[2][0] == self.options[3])
                _require(set(facts[2][2]) <= set((*self.profile_union, *facts[0][8])))
                if self.consumer_facts is None:
                    self.consumer_source, self.consumer_selection, self.consumer_producer = source, selected, producer
                    self.consumer_facts = facts
                    self.profile_union = tuple(sorted(set((*self.profile_union, *facts[0][8], *facts[2][2]))))
                    routes = tuple(r for r in self.catalog[0] if r.route_ref == self.options[4])
                    _require(len(routes) == 1 and routes[0].path == "main")
                    self.main_route = routes[0]
                original = self.consumer_facts
                _require(facts[0][:8] == original[0][:8] and set(facts[0][8]) <= set(original[0][8]))
                _require(selected is self.consumer_selection and facts[1] == original[1])
                _require(producer is self.consumer_producer and facts[2] == original[2])
                _require((self.main_route.consumer_ref, self.main_route.selection_id, self.main_route.selection_version)
                         == (facts[0][3], facts[0][6], facts[0][7]))
                _require(selected.destination is self.main_route.destination)
                now = self.callback(self.parent._clock)
                _integer(now, minimum=0)
                _require(now < self.parent._main_route.expires_at and now < self.main_route.expires_at)
                allowed = set(r.route_ref for r in self.catalog[0])
                for profile in self.catalog[1]:
                    if profile.profile_ref in self.profile_union:
                        allowed.intersection_update(profile.allowed_routes)
                _require(self.main_route.route_ref in allowed and self.parent._route_ref in allowed)
                for holder in (self.source_owner.initial_facts[0], original[0]):
                    for path in shared_policy().PATHS:
                        _require(any(r.route_ref in allowed and r.path == path
                            and (r.consumer_ref, r.selection_id, r.selection_version) == (holder[3], holder[6], holder[7])
                            and now < r.expires_at for r in self.catalog[0]))
                for route in self.catalog[0]:
                    _route_facts(route)
                    actual = self.callback(self.parent._operator_transports[route.destination.transport_ref].resolve)
                    _require(actual is route.destination)
                for agent in (self.source_owner.agent, self.agent):
                    if agent is not None:
                        flag = _namespace(agent).get("_interrupt_requested")
                        _require(type(flag) is bool)
                        if flag:
                            raise InterruptedError("Agent interrupted during memory prompt")
                self.runtime._check_operator()
                self.verify()
        finally:
            self.checking = False

    def current(self, handle):
        self.verify()
        _require(handle is self.parent._handle or handle is self.context.request_handle)
        return self.callback(self.parent._admission.current, handle)

    def around(self, callback):
        self.verify()
        with self.source_probe():
            self.parent._checkpoint()
        output = self.callback(callback)
        with self.source_probe():
            self.parent._checkpoint()
        return output

    def resolve(self, wrapper):
        self.verify()
        _require(any(wrapper is original for original in self.parent._transport_wrappers.values()))
        actual = self.around(wrapper.port.resolve)
        _destination(actual)
        return actual

    def send(self, wrapper, envelope):
        self.access()
        self.check_stage()
        self.checkpoint()
        c = self.adapter
        _require(self.stage_index == 6 and self.phase == "request-prepared")
        operation = c._operation
        _require(type(operation) is _PendingOperation and operation is c._checked_operation)
        _require(c._busy and c._pending is None and self.partial_send is None and c._send_count == 0)
        _require(type(envelope) is shared_policy().DispatchEnvelope)
        _require(operation.handle is self.context.request_handle and operation.path == "main")
        _require(envelope.payload is operation.payload and envelope.path == operation.path)
        _require(envelope.source == operation.source and envelope.destination == self.main_route.destination)
        _require(envelope.recipient_ref == self.consumer_facts[0][3])
        _require(self.parent._transport_wrappers[envelope.destination.transport_ref] is wrapper)
        actual = wrapper.resolve()
        _require(actual is self.main_route.destination and actual == envelope.destination
                 and c._kwargs is self.prepared_request)
        c._pending = (envelope, actual, None)
        c._send_count += 1
        self.partial_send = (operation, envelope, actual, None)
        output = self.callback(wrapper.port.send, envelope)
        c._pending = (envelope, actual, output)
        self.partial_send = (operation, envelope, actual, output)
        self.transition("request-prepared", "sent")
        _require(type(output) is bytes and len(output) <= 65_536)
        _require(wrapper.resolve() is actual)
        self.checkpoint()
        return output

    def open_view(self, adapter, context):
        self.access()
        _require(self.constructing and self.adapter is None and context is self.context)
        _require(current_context() is None and not context._controlled_main_bound and not context._denied)
        _require(context.admission is self.parent._admission)
        self.adapter = adapter
        adapter._memory_prompt_frame = self
        _memory_prompt_frames[adapter] = self

    def check_input(self, agent, user, system, history, task):
        self.access()
        _require(self.phase == "entered" and self.input_content is None)
        _require(agent is self.agent and _same_anchors((user, system, history, task), self.arguments))

    def enter_stage(self):
        self.access()
        _require(self.stage is None and self.stage_index < len(self.stage_codes))
        caller = sys._getframe(1)
        code = self.stage_codes[self.stage_index]
        while caller is not None and caller.f_code is not code:
            caller = caller.f_back
        _require(caller is not None)
        _require(caller.f_locals.get("agent") is self.agent)
        public = caller
        from run_agent import AIAgent
        while public is not None and public.f_code is not AIAgent.run_conversation.__code__:
            public = public.f_back
        _require(public is not None and public.f_back is self.entry)
        if self.public_frame is None:
            self.public_frame = public
        _require(public is self.public_frame)
        _require(_same_anchors(tuple(public.f_locals[n] for n in (
            "user_message", "system_message", "conversation_history", "task_id")), self.arguments))
        self.stage = self.stage_anchor = (self.active_call, caller, object())
        self.stage_index += 1
        self.checkpoint()
        return self.stage

    def check_stage(self):
        self.access()
        _require(self.stage is self.stage_anchor and self.stage is not None)
        _require(self.stage[0] is self.active_call)
        caller = sys._getframe(1)
        while caller is not None and caller is not self.stage[1]:
            caller = caller.f_back
        _require(caller is self.stage[1])

    def exit_stage(self, stage):
        if self.phase != "denied":
            self.check_stage()
            _require(self.stage is stage)
            self.checkpoint()
        self.stage = self.stage_anchor = None

    def transition(self, previous, phase):
        self.access()
        _require(self.phase == previous)
        self.phase = self.phase_anchor = phase
        self.checkpoint()

    def progress(self):
        self.check_stage()
        if self.stage_index == 4:
            self.transition("prompt-built", "sealed")
        elif self.stage_index == 5:
            self.transition("sealed", "request-prepared")

    def cache_state(self, namespace):
        self.access()
        full, static = (namespace.get(name) for name in _MEMORY_CACHE_NAMES)
        if self.cache_frozen:
            _require(full is self.full_prompt and static is self.static_prompt)
        elif self.prompt_step == 4:
            _require(full is None and static is self.options[5])
        else:
            _require(full is None and static is None)

    def prompt_access(self, agent, kind, system):
        from agent import system_prompt
        from run_agent import AIAgent
        self.check_stage()
        self.verify()
        _require(agent is self.agent and self.stage_index == 3 and not self.cache_frozen)
        _require(self.prompt_frame is not None and self.phase == "admitted")
        caller = sys._getframe(2)
        if kind == "build":
            _require(self.prompt_step == 0 and system is self.options[5])
            _require(caller.f_code is system_prompt.build_system_prompt.__code__)
            forward = caller.f_back
            _require(forward.f_code is AIAgent._build_system_prompt.__code__ and forward.f_back is self.prompt_frame)
            self.builder_frame, self.forward_frame = caller, forward
            self.prompt_step = 1
        elif kind == "parts":
            _require(self.prompt_step == 1 and system is self.options[5])
            _require(caller.f_code is system_prompt.build_system_prompt_parts.__code__
                     and caller.f_back is self.builder_frame)
            self.parts_frame = caller
            self.prompt_step = 2
        elif kind == "append":
            _require(self.prompt_step == 2 and caller.f_code is system_prompt._append_builtin_memory_parts.__code__
                     and caller.f_back is self.parts_frame)
            self.prompt_step = 3
        elif kind == "joined":
            _require(self.prompt_step == 3 and caller is self.builder_frame and system is self.options[5])
            _require(_namespace(agent)["_cached_system_prompt_static"] is system)
            self.prompt_step = 4
        else:
            _require(False)
        return self.source

    def build_prompt(self, agent, system):
        self.check_stage()
        self.checkpoint()
        _require(agent is self.agent and system is self.options[5] and self.prompt_step == 0)
        _require(self.prompt_frame is None and self.phase == "admitted")
        self.prompt_frame = sys._getframe()
        try:
            full = agent._build_system_prompt(system)
            expected = "\n\n".join(part for part in (system, self.source.block) if part)
            _require(self.prompt_step == 4 and type(full) is str and full == expected)
            _text(full)
            self.static_prompt, self.full_prompt = system, full
            _require(_namespace(agent)["_cached_system_prompt"] is None)
            agent._cached_system_prompt = full
            self.cache_frozen = True
            self.transition("admitted", "prompt-built")
            return full
        finally:
            self.prompt_frame = None


def bind_memory_prompt_consumer(
    source_adapter: NativeMainPayloadAdapter,
    source: NativeMemorySnapshotResult, *,
    consumer_context: NativeContentContext,
    selection: NativeSelectionPort,
    result_policy: NativeResultPolicyPort,
    producer_ref: str, main_route_ref: str,
    system_message: str, session_id: str,
) -> NativeMemoryPromptInput:
    _require(type(source_adapter) is NativeMainPayloadAdapter and source_adapter in _adapters)
    owner = _memory_snapshots.get(source_adapter)
    _require(owner is not None and type(source) is NativeMemorySnapshotResult)
    _require(source is owner.record is owner.registered_record and source.binding is owner.registered_binding)
    _require(current_context() is source_adapter._bound_context and threading.get_ident() == owner.thread)
    _require(not hasattr(owner, "memory_prompt_attempt"))
    owner.memory_prompt_attempt = object()
    frame = None
    try:
        for value in (producer_ref, main_route_ref, system_message, session_id):
            _require(type(value) is str)
        for value in (producer_ref, main_route_ref, session_id):
            _identifier(value)
        _text(system_message)
        _require(type(consumer_context) is NativeContentContext)
        _require(consumer_context is not source_adapter._bound_context
                 and consumer_context.request_handle is not source_adapter._handle)
        _require(consumer_context.admission is source_adapter._admission
                 and selection is source_adapter._selection_port and result_policy is source_adapter._result_port)
        frame = _MemoryPrompt(owner, source, consumer_context, selection, result_policy,
                              producer_ref, main_route_ref, system_message, session_id, sys._getframe())
        owner.memory_prompt_attempt = frame
        _memory_prompt_frames[source_adapter] = frame
        from agent.content_policy import content_policy_scope
        from run_agent import AIAgent
        frame.checkpoint()
        _require(not consumer_context._controlled_main_bound and not consumer_context._denied)
        frame.constructing = True
        with content_policy_scope(None):
            NativeMainPayloadAdapter(consumer_context, selection=selection, result_policy=result_policy,
                producer_ref=producer_ref, main_route_ref=main_route_ref, profiles=frame.catalog[1],
                routes=frame.catalog[0], transports=frame.catalog[2], clock=source_adapter._clock,
                _memory_prompt_frame=frame)
            with content_policy_scope(consumer_context):
                frame.agent = AIAgent(controlled_main=frame.adapter, provider="synthetic_native",
                    api_mode="chat_completions", model=frame.consumer_selection.destination.model_ref,
                    session_id=session_id, enabled_toolsets=[], quiet_mode=True, skip_memory=True,
                    skip_context_files=True, skip_background_review=True, max_tokens=None)
                constructor = dict(provider="synthetic_native", api_mode="chat_completions",
                    model=frame.agent.model, session_id=session_id, enabled_toolsets=[], quiet_mode=True,
                    skip_memory=True, skip_context_files=True, skip_background_review=True)
                frame.authorization_bytes = _json(dict(constructor=constructor,
                    consumer=frame.consumer_facts[0], controlled_main_present=True,
                    main_route_ref=main_route_ref, producer_policy=frame.consumer_facts[2],
                    producer_ref=producer_ref, session_id=session_id, system_message=system_message))
                _require(len(frame.authorization_bytes) <= _MAX_NATIVE_RESULT)
                frame.authorization_content = frame.runtime.admit(consumer_context.request_handle,
                                                                  frame.authorization_bytes)
                frame.checkpoint()
        frame.constructing = False
        frame.checkpoint()
        token = object.__new__(NativeMemoryPromptInput)
        frame.token = token
        return token
    except BaseException:
        if frame is not None:
            frame.invalidate()
            if frame.control_exception is not None:
                raise frame.control_exception
        elif type(consumer_context) is NativeContentContext and consumer_context is not source_adapter._bound_context:
            consumer_context._denied = True
        raise
    finally:
        if frame is not None:
            frame.entry = frame.active_call = frame.active_anchor = None


def run_memory_prompt_consumer(
    token: NativeMemoryPromptInput, *,
    user_message: str, task_id: str,
) -> dict:
    _require(type(token) is NativeMemoryPromptInput)
    frame = next((f for a in _memory_prompt_frames for f in (_memory_prompt_frames.get(a),)
                  if f is not None and f.token is token), None)
    _require(frame is not None)
    try:
        _require(frame.phase == "bound" and frame.active_call is None and not frame.terminal)
        _require(threading.get_ident() == frame.thread)
        _text(user_message)
        _identifier(task_id)
        _require(task_id == frame.consumer_facts[0][1])
        _require(current_context() in (None, frame.parent._bound_context))
        frame.arguments = (user_message, frame.options[5], None, task_id)
        frame.entry = sys._getframe()
        frame.active_call = frame.active_anchor = object()
        frame.transition("bound", "entered")
        from agent.content_policy import content_policy_scope
        with content_policy_scope(None):
            with content_policy_scope(frame.context):
                result = frame.agent.run_conversation(user_message=user_message,
                    system_message=frame.options[5], conversation_history=None, task_id=task_id)
                frame.checkpoint()
                _require(frame.stage_index == 8 and frame.stage is None and frame.phase == "completed")
                _require(len(frame.adapter._completions) == 1
                         and result is frame.adapter._completions[0].result)
                frame.adapter._verify_turns()
                frame.adapter._verify_records()
                frame.adapter._verify_completions()
                frame.terminal = True
            frame.checkpoint()
        frame.transition("completed", "released")
        frame.checkpoint()
        return result
    except BaseException:
        frame.invalidate()
        if frame.control_exception is not None:
            raise frame.control_exception
        raise
    finally:
        frame.entry = frame.public_frame = frame.active_call = frame.active_anchor = None


class _ParentResumption:
    """Retain one operator authorization and its actual parent proof."""

    def __init__(self, parent, agent, options):
        _require(parent not in _parent_resumptions)
        frame = _delegate_frames.get(parent)
        _require(type(frame) is _DelegateFrame and frame.parent is parent and frame.agent is agent)
        _require(not hasattr(frame, "parent_resumption"))
        _require(threading.get_ident() == frame.thread == parent._thread)
        _require(current_context() is parent._bound_context)
        _require(current_context(agent) is parent._bound_context)
        _require(parent._constructed is _owned_constructed_binding(agent))
        _require(not frame.running and not frame.terminal and frame.record is None and frame.phase is None)
        _require(not parent._busy and parent._pending is parent._operation is parent._checked_operation is None)
        _require(parent._turn_pending is parent._aux_pending is None)
        _require(not parent._turn_frames and not parent._records and not parent._completions)
        _require(parent._aux_binding is parent._compression_binding is parent._fallback_binding is None)
        self.parent, self.agent, self.frame = parent, agent, frame
        self.options, self.option_anchors = options, _nested_anchors(options)
        self.phase = self.phase_anchor = "bound"
        self.binding = self.registered_binding = self.record = self.registered_record = None
        self.depth = 0
        self.active_call = self.call_frame = self.public_frame = None
        self.active_anchor = self.stage = self.stage_anchor = None
        self.stage_index = 0
        self.control_exception = None
        self.prepared_request = None
        self.partial_send = None
        self.partial_anchor = None
        self.proof = self.parent_proof()
        frame.parent_resumption = self
        _parent_resumptions[parent] = self

    def parent_proof(self):
        p = self.parent
        return p._turn_frames, p._records, p._record_anchors, p._completions, p._completion_anchors

    def callback(self, callback, *args):
        self.depth += 1
        try:
            output = callback(*args)
            return output
        except BaseException as error:
            import asyncio
            if isinstance(error, (InterruptedError, asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                if self.control_exception is None:
                    self.control_exception = error
            raise
        finally:
            self.depth -= 1

    def verify(self):
        _require(_parent_resumptions.get(self.parent) is self)
        _require(self.frame.parent_resumption is self and self.frame.parent is self.parent)
        _require(self.phase is self.phase_anchor and self.phase != "denied")
        _require(threading.get_ident() == self.frame.thread)
        _require(_same_anchors(_nested_anchors(self.options), self.option_anchors))
        _require(_same_anchors(self.parent_proof(), self.proof))
        _require(all(type(rows) is tuple and len(rows) <= 1 for rows in self.proof))
        _require(self.partial_send is self.partial_anchor)
        if self.binding is not None:
            b = self.binding
            _require(b is self.registered_binding)
            _require(_same_anchors(_row(b, NativeParentResumptionBinding), self.binding_anchor))
            _require(b.snapshots is self.snapshots and b.anchors is self.snapshot_anchors)
            _require(_same_anchors(b.snapshots, (self.frame.binding.snapshots, self.frame.initial,
                self.parent._constructed, self.parent._constructed.methods, self.frame.catalog, self.methods)))
            _require(b.agent is self.agent and b.delegation_frame is self.frame)
            _require(b.delegation_binding is self.frame.binding and b.parent_request is self.parent._handle)
            _require(self.parent._main_route is b.main_route and self.parent._route_ref is self.route_ref)
            _require(self.parent._producer_ref is self.frame.producers[0])
            _require(b.options is self.options and b.thread == self.frame.thread)
            _require(self.frame.runtime._content(b.authorization_content).payload is b.authorization_bytes)
            for port, name, method in self.methods:
                _require(_same_method(getattr(port, name), method))
        if self.record is not None:
            r = self.record
            _require(r is self.registered_record and r.binding is self.binding)
            _require(_same_anchors(_row(r, NativeParentResumptionInvocation), self.record_anchor))
            _require(_same_anchors(self.mutable_anchors(r), self.record_nested))
            _require(_same_anchors(_nested_anchors(r.arguments), r.argument_anchors))
            _require(r.delegation is self.frame.record and r.closed_child_proof is self.frame.closed_proof)
            _require(r.return_facts is r.delegation.return_facts)
            _require(r.returned_content is r.return_facts.returned_content)
            _require(_same_anchors(r.input_parents, (r.returned_content, self.binding.authorization_content)))
            _require(r.arguments[0].encode("utf-8") == self.binding.instruction.encode("utf-8")
                     + b"\n" + r.delegation.aggregate_bytes)
            _require(_NATIVE_SANITIZER(r.arguments[0]) == r.arguments[0])
            if r.input_content is not None:
                content = self.frame.runtime._content(r.input_content)
                _require(content.payload is r.raw_bytes and _same_anchors(content.parents, r.input_parents))
            if r.turn_frame is not None:
                _require(_same_anchors(self.parent._turn_frames, (r.turn_frame,)))
                _require(r.turn_frame.admitted is r.input_content)
                _require(_same_anchors(r.turn_frame.arguments, r.arguments))
            if r.native_completion is not None:
                _require(r.native_completion.result is r.result)
                _require(r.native_completion.payload is r.result_bytes)
                _require(_native_result_bytes(r.result) == r.result_bytes)
                _require(self.frame.runtime._content(r.result_content).payload is r.result_bytes)

    def checkpoint(self):
        self.frame.runtime._check_operator()
        self.frame.checkpoint()
        self.parent._checkpoint()
        self.frame.runtime._check_operator()
        self.verify()

    def bind(self, instruction, system):
        from agent import conversation_loop
        self.checkpoint()
        _require(self.binding is None and self.active_call is None and self.depth == 0)
        _verify_constructed(self.parent._constructed)
        originals = self.frame.binding.snapshots[0]
        source, selected, producer = originals[0]
        task = source.execution_ref
        _identifier(task)
        self.methods = tuple((port, name, _method(getattr(port, name))) for port, name in (
            (self.parent, "bind_parent_resumption"), (self.parent, "resume_parent"),
            (self.parent, "_dispatch_retained_operation"),
            (sys.modules[__name__], "_constructed_adapter"),
            (sys.modules[__name__], "_prepare_controlled_turn"),
            (sys.modules[__name__], "dispatch_controlled_main"),
            (sys.modules[__name__], "_resumption_callback"),
            (self.frame.runtime, "_check_operator"), (self.frame.runtime, "_operator"),
            (conversation_loop, "_try_controlled_text_turn")))
        self.methods += tuple((self, name, _method(getattr(self, name))) for name in (
            "callback", "verify", "checkpoint", "access", "_access", "enter_stage", "check_stage",
            "exit_stage", "retain", "mutable_anchors", "transition", "progress", "access_progress",
            "sending", "sent", "resume", "parent_proof"))
        self.route_ref = self.parent._route_ref
        self.stage_codes = tuple(_method(function)[-1].__code__ for function in (
            conversation_loop._try_controlled_text_turn, self.parent.admit_turn,
            self.parent._prepare_turn, self.parent.seal_turn, self.parent.prepared_request,
            self.parent.dispatch, _complete_controlled_turn, conversation_loop._try_controlled_text_turn))
        snapshots = (self.frame.binding.snapshots, self.frame.initial, self.parent._constructed,
                     self.parent._constructed.methods, self.frame.catalog, self.methods)
        self.snapshots, self.snapshot_anchors = snapshots, _nested_anchors(snapshots)
        encoded = _json(dict(instruction=instruction, system_message=system, task_id=task,
            parent=_binding(source), selection=_selection_value(selected), producer=_producer(producer),
            route=_route_facts(self.parent._main_route)[:5],
            options=[(name, present) for name, present, _ in self.options[0]]))
        _require(len(encoded) <= _MAX_NATIVE_RESULT)
        content = self.frame.runtime.admit(self.parent._handle, encoded)
        self.checkpoint()
        token = object.__new__(NativeParentResumption)
        self.binding = NativeParentResumptionBinding(token, self.agent, self.frame, self.frame.binding,
            self.parent._handle, selected, producer, self.parent._main_route, instruction, system, task,
            self.options, encoded, content, snapshots, self.snapshot_anchors, self.frame.thread)
        self.registered_binding = self.binding
        self.binding_anchor = _row(self.binding, NativeParentResumptionBinding)
        self.checkpoint()
        return token

    def access(self):
        try:
            self._access()
        except BaseException:
            self.frame.invalidate()
            raise

    def _access(self):
        _require(self.depth == 0 and self.active_call is not None and self.call_frame is not None)
        _require(self.active_call is self.active_anchor)
        _require(self.phase in ("entered", "admitted", "sent", "completed"))
        _require(threading.get_ident() == self.binding.thread)
        # Only the original public call can own native model work.
        code = _method(self.agent.run_conversation)[1].__code__
        current = sys._getframe(1)
        while current is not None and current.f_code is not code:
            current = current.f_back
        _require(current is not None and current.f_back is self.call_frame)
        if self.public_frame is None:
            self.public_frame = current
        _require(current is self.public_frame)
        arguments = self.record.arguments
        _require(_same_anchors(tuple(current.f_locals[name] for name in (
            "user_message", "system_message", "conversation_history", "task_id")), arguments))
        self.verify()

    def enter_stage(self):
        self.access()
        _require(self.stage is None and self.stage_index < len(self.stage_codes))
        code = self.stage_codes[self.stage_index]
        caller = sys._getframe(1)
        while caller is not None and caller.f_code is not code:
            caller = caller.f_back
        _require(caller is not None)
        self.stage = self.stage_anchor = (self.active_call, caller, object())
        self.stage_index += 1
        self.checkpoint()
        return self.stage

    def check_stage(self):
        _require(self.stage is not None and self.stage is self.stage_anchor)
        _require(self.stage[0] is self.active_call)
        caller = sys._getframe(1)
        while caller is not None and caller is not self.stage[1]:
            caller = caller.f_back
        _require(caller is self.stage[1])

    def exit_stage(self, stage):
        if self.phase != "denied":
            self.access()
            _require(self.stage is stage is self.stage_anchor)
            self.checkpoint()
        self.stage = self.stage_anchor = None

    def retain(self, **changes):
        self.record = replace(self.record, **changes)
        self.registered_record = self.record
        self.record_anchor = _row(self.record, NativeParentResumptionInvocation)
        self.record_nested = self.mutable_anchors(self.record)

    def mutable_anchors(self, record):
        return _nested_anchors((record.turn_frame, record.request, record.result))[1:]

    def transition(self, previous, phase):
        self.checkpoint()
        _require(self.phase == previous and self.active_call is not None and self.depth == 0)
        self.phase = self.phase_anchor = phase
        self.checkpoint()

    def progress(self):
        self.access_progress()
        p, changes = self.parent, {}
        if p._turn_frames:
            f = p._turn_frames[0]
            changes.update(turn_frame=f, preparation_contents=(
                *((f.output_content,) if f.output_content is not None else ()), *f.request_contents))
            _require(len(changes["preparation_contents"]) <= 8)
        if p._records:
            invocation = p._records[0]
            changes.update(native_invocation=invocation, request_content=invocation.request_content,
                request_bytes=invocation.request_bytes, envelope=invocation.envelope,
                observed_destination=invocation.observed_destination, response_bytes=invocation.response_bytes)
        if p._completions:
            completed = p._completions[0]
            changes.update(native_completion=completed, result=completed.result,
                result_bytes=completed.payload, result_content=completed.content)
        self.proof = self.parent_proof()
        self.retain(**changes)
        self.verify()

    def access_progress(self):
        # Native progress appends before the next read-only verification.
        _require(self.depth == 0 and self.active_call is not None)
        _require(threading.get_ident() == self.binding.thread and self.phase != "denied")
        self.check_stage()

    def sending(self, operation, envelope, actual, before):
        self.access()
        self.check_stage()
        self.checkpoint()
        _require(self.phase == "admitted" and self.partial_send is None and before == 0)
        _require(operation is self.parent._operation and operation.path == "main")
        _require(operation.handle is self.binding.parent_request)
        _require(not self.frame.running and self.frame.phase == "released" and self.frame.terminal)
        _require(self.parent._kwargs is self.prepared_request)
        _require(self.parent._kwargs["messages"][-1]["content"].encode("utf-8")
                 == self.record.arguments[0].encode("utf-8"))
        self.partial_send = self.partial_anchor = (operation, envelope, actual, before, None, None)
        self.retain(request=self.parent._kwargs, request_bytes=operation.payload,
            request_content=operation.content, envelope=envelope, observed_destination=actual)

    def sent(self, envelope, actual, output, before, after):
        operation, original, destination, count, _, _ = self.partial_send
        _require(envelope is original and actual is destination and before == count == 0 and after == 1)
        self.partial_send = self.partial_anchor = (operation, envelope, actual, before, after, output)
        self.access()
        self.check_stage()
        self.retain(response_bytes=output)
        self.transition("admitted", "sent")

    def resume(self, agent, authorization):
        _require(self.depth == 0 and self.active_call is None and self.phase == "bound")
        _require(type(authorization) is NativeParentResumption)
        _require(self.binding is not None and authorization is self.binding.token and agent is self.agent)
        _require(current_context() is self.parent._bound_context)
        _require(current_context(agent) is self.parent._bound_context)
        _require(not self.frame.running and self.frame.terminal and self.frame.phase == "released")
        _require(not self.parent._busy and self.parent._operation is self.parent._pending is None)
        _require(self.parent._checked_operation is self.parent._turn_pending is self.parent._aux_pending is None)
        _require(not self.frame.runtime._busy)
        _require(self.frame.child_adapter._constructed.closed)
        self.checkpoint()
        self.frame.check_return()
        aggregate = self.frame.record.aggregate_bytes
        _require(type(aggregate) is bytes)
        text = aggregate.decode("utf-8", errors="strict")
        _require(text.encode("utf-8") == aggregate)
        user = self.binding.instruction + "\n" + text
        _text(user)
        _require(_NATIVE_SANITIZER(user) == user)
        arguments = (user, self.binding.system_message, None, self.binding.task_id)
        _, options = _native_state(agent, self.binding.system_message)
        raw = _turn_bytes(arguments, options)
        self.record = NativeParentResumptionInvocation(self.binding, self.frame.record,
            self.frame.record.return_facts, self.frame.record.return_facts.returned_content,
            self.frame.closed_proof, arguments, _nested_anchors(arguments), raw,
            (self.frame.record.return_facts.returned_content, self.binding.authorization_content),
            None, None, (), None, None, None, None, None, None, None, None, None, None, None)
        self.retain()
        self.active_call = object()
        self.active_anchor = self.active_call
        self.call_frame = sys._getframe()
        try:
            self.transition("bound", "entered")
            result = agent.run_conversation(user_message=user, system_message=self.binding.system_message,
                conversation_history=None, task_id=self.binding.task_id)
            self.checkpoint()
            _require(self.phase == "completed" and result is self.record.result)
            _require(self.stage_index == len(self.stage_codes) and self.stage is None)
            self.parent._verify_turns()
            self.parent._verify_records()
            self.parent._verify_completions()
            _require(self.parent._operation is self.parent._checked_operation is self.parent._pending is None)
            _require(self.parent._turn_pending is None and not self.parent._busy)
            _require(not self.frame.runtime._busy)
            self.transition("completed", "released")
            self.active_call = None
            self.checkpoint()
            return result
        finally:
            self.active_call = self.call_frame = self.public_frame = None
            self.active_anchor = self.stage = self.stage_anchor = None


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
        owner = _parent_resumptions.get(self.parent)
        if owner is not None:
            owner.phase = owner.phase_anchor = "denied"
            owner.active_call = None
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
        owner = _parent_resumptions.get(self.parent)
        if owner is None:
            _require(not self.parent._turn_frames and not self.parent._records and not self.parent._completions)
        else:
            _require(type(owner) is _ParentResumption)
            if owner.binding is not None:
                for port, name, method in owner.methods:
                    _require(_same_method(getattr(port, name), method))
            owner.verify()
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
            issued = _issued_window(self.parent)
            if issued is not None:
                return issued.delegate_call(self, callback, *args)
            return _resumption_callback(self.parent, callback, *args)
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
        issued = _issued_window(self.parent)
        if issued is not None:
            return issued.delegate_checkpoint(self)
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
                _require(now < self.binding.delegate_route.expires_at)
                _require(now < self.binding.child_main_route.expires_at)
                owner = _parent_resumptions.get(self.parent)
                if owner is not None and owner.binding is not None:
                    _require(now < owner.binding.main_route.expires_at)
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
            issued = _issued_window(self.parent)
            result = (self.parent._admission.current(valid) if issued is None else
                      issued.callback(self.parent._admission.current, valid))
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
        issued = _issued_window(self.parent)
        raw_content = (self.stage(raw, ()) if issued is None else
                       issued.capture(self, raw, self.arguments))
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
            issued = _issued_window(self.parent)
            output = (wrapper.port.send(envelope) if issued is None else
                      issued.callback(wrapper.port.send, envelope))
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
    issued = _issued_window(frame.parent)
    start = frame.parent._clock() if issued is None else frame.call(frame.parent._clock)
    _integer(start, minimum=0)
    result = frame.child.run_conversation(user_message=frame.leaf["goal"], system_message=frame.prompt,
        task_id=frame.binding.child_task_id)
    frame.retain_completion(result)
    end = frame.parent._clock() if issued is None else frame.call(frame.parent._clock)
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


class NativeToolCallProbeInput:
    __slots__ = ()

    def __new__(cls):
        raise TypeError("Probe tokens require operator admission")


@dataclass(frozen=True, slots=True)
class NativeToolCallProbeBinding:
    token: NativeToolCallProbeInput
    agent: object
    request_handle: object
    selection: NativeMainSelection
    producer_policy: NativeResultPolicy
    main_route: object
    system_message: str
    user_message: str
    task_id: str
    expected_arguments_json: str
    authorization_bytes: bytes
    authorization_content: object
    anchors: tuple
    thread: int


@dataclass(frozen=True, slots=True)
class NativeToolCallObservation:
    binding: NativeToolCallProbeBinding
    request_bytes: bytes
    request_content: object
    wire_bytes: bytes
    wire_content: object
    envelope: object
    observed_destination: object
    response_bytes: bytes
    response_content: object
    call_id: str
    function_name: str
    arguments_json: str
    observation_bytes: bytes
    observation_content: object


def _probe_require(condition, reason):
    if not condition:
        raise NativeContentPolicyDenied("controlled_native_tool_probe_" + reason)


def _probe_failure(adapter, error):
    owner = _tool_call_probes.get(adapter)
    adapter._invalidate()
    if owner is not None and owner.control_exception is not None:
        raise owner.control_exception
    if not isinstance(error, Exception) or isinstance(error, InterruptedError):
        raise error
    if type(error) is NativeContentPolicyDenied:
        raise error
    if type(error) is shared_policy().PolicyError and error.args == ("capacity_exhausted",):
        raise NativeContentPolicyDenied("controlled_native_tool_probe_capacity_denied") from None
    raise NativeContentPolicyDenied("controlled_native_tool_probe_authority_denied") from None


def _tool_probe_arguments(value):
    _text(value)
    decoded = _decode_response_payload(value.encode("utf-8"))
    _keys(decoded, ("context", "goal", "role"))
    _require(decoded["context"] is None)
    _require(type(decoded["role"]) is str and decoded["role"] == "leaf")
    _text(decoded["goal"])
    _require(bool(decoded["goal"]))
    return decoded


def _tool_probe_schema():
    return {"type": "function", "function": {"name": "delegate_task", "parameters": {
        "type": "object", "additionalProperties": False,
        "properties": {"context": {"type": ["string", "null"]}, "goal": {"type": "string"},
                       "role": {"type": "string", "enum": ["leaf"]}},
        "required": ["context", "goal", "role"]}}}


def _tool_probe_request(value):
    _keys(value, ("model", "messages", "stream", "tools", "tool_choice"))
    _identifier(value["model"])
    _require(value["stream"] is False)
    messages = value["messages"]
    _require(type(messages) is list and len(messages) == 2)
    for row, role in zip(messages, ("system", "user")):
        _keys(row, ("role", "content"))
        _require(type(row["role"]) is str and row["role"] == role)
        _text(row["content"])
    # Exact scalar types also apply inside the fixed schema.
    _require(_safe_json(value["tools"]) == _json([_tool_probe_schema()]))
    _require(_safe_json(value["tool_choice"]) == _json({"type": "function", "function": {"name": "delegate_task"}}))
    _require(len(_json(value)) <= _MAX_NATIVE_RESULT)


def _tool_probe_response(payload, model, expected_arguments_json):
    value = _decode_response_payload(payload)
    _keys(value, ("id", "object", "created", "model", "choices", "usage"))
    _identifier(value["id"])
    _identifier(value["model"])
    _require(value["model"] == model)
    _require(type(value["object"]) is str and value["object"] == "chat.completion")
    _integer(value["created"], minimum=0)
    choices = value["choices"]
    _require(type(choices) is list and len(choices) == 1)
    choice = choices[0]
    _keys(choice, ("index", "message", "finish_reason"))
    _integer(choice["index"], 0, 0)
    _require(type(choice["finish_reason"]) is str and choice["finish_reason"] == "tool_calls")
    message = choice["message"]
    _keys(message, ("role", "content", "tool_calls"))
    _require(type(message["role"]) is str and message["role"] == "assistant" and message["content"] is None)
    calls = message["tool_calls"]
    _require(type(calls) is list and len(calls) == 1)
    call = calls[0]
    _keys(call, ("id", "type", "function"))
    _identifier(call["id"])
    _require(type(call["type"]) is str and call["type"] == "function")
    function = call["function"]
    _keys(function, ("name", "arguments"))
    _require(type(function["name"]) is str and function["name"] == "delegate_task")
    _tool_probe_arguments(function["arguments"])
    _require(function["arguments"] == expected_arguments_json)
    usage = value["usage"]
    _keys(usage, ("prompt_tokens", "completion_tokens", "total_tokens"))
    for number in usage.values():
        _integer(number, 1_000_000, 0)
    _require(usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"])
    return value


class _ToolCallProbe:
    """Own one synthetic protocol observation without execution authority."""

    def __init__(self, parent, agent, inputs, public):
        from run_agent import AIAgent
        _probe_require(parent not in _tool_call_probes and not hasattr(parent, "_tool_probe_owner"), "attempt_spent")
        self.parent, self.agent, self.inputs = parent, agent, inputs
        self.public_bind, self.bind_frame = public, None
        self.run_frame = self.public_run = self.dispatch_frame = None
        self.active_call = self.call_anchor = self.stage_frame = self.stage_anchor = None
        self.depth = self.stage_index = 0
        self.phase = self.phase_anchor = "bound"
        self.attempted = self.closed = False
        self.binding = self.registered_binding = self.binding_anchor = None
        self.records = self.record_anchors = ()
        self.request = self.request_anchors = self.operation = None
        self.request_bytes = self.request_content = self.wire_bytes = self.wire_content = None
        self.envelope = self.observed_destination = self.response_bytes = self.response_content = None
        self.observation_bytes = self.observation_content = None
        self.partial_send = None
        self.control_exception = None
        self.runtime, self.thread = parent._runtime, parent._thread
        self.constructed = parent._constructed
        self.catalog = parent._operator_configuration
        self.catalog_anchors = _nested_anchors(self.catalog)
        self.original_profiles = parent._original_profiles
        self.source, self.selection, self.producer = parent._source, parent._selected, parent._producer_policy
        self.initial = _binding(self.source), _selection(self.selection), _producer(self.producer)
        self.original_anchors = _nested_anchors((self.source, self.selection, self.producer))[1:]
        self.native_facts = tuple((name, getattr(parent, name)) for name in (
            "_context", "_bound_context", "_handle", "_admission", "_selection_port", "_result_port", "_clock",
            "_main_route", "_route_ref", "_producer_ref", "_holder", "_original_profiles", "_constructed"))
        self.proof = tuple(getattr(parent, name) for name in (
            "_turn_frames", "_records", "_record_anchors", "_completions", "_completion_anchors",
            "_aux_frames", "_aux_records", "_aux_local_results", "_failed_main_attempts", "_fallback_records"))
        # Register the original attempt before any operator callback.
        parent._tool_probe_owner = self
        _tool_call_probes[parent] = self
        _probe_require(type(agent) is AIAgent and self.constructed is _owned_constructed_binding(agent)
                       and self.constructed is not None and parent._turn_agent is agent, "agent_invalid")
        _probe_require(not parent._busy and not parent._invalid and not parent._bound_context._denied, "owner_denied")
        _probe_require(all(not value for value in self.proof) and parent._send_count == 0, "owner_used")
        _probe_code_current()
        self.methods = self.capture_methods()
        self.methods_anchor = self.methods
        self.runtime_facts = tuple((name, getattr(self.runtime, name)) for name in (
            "_admission", "_clock", "_profiles", "_routes", "_transports", "_operator_snapshot", "_ports", "_route_rows"))
        self.runtime_ports = tuple(self.runtime._ports.items())
        self.runtime_routes = _nested_anchors(self.runtime._route_rows)
        for value in inputs:
            _text(value)
        _identifier(inputs[2])
        _tool_probe_arguments(inputs[3])
        _probe_require(inputs[2] == self.initial[0][1], "task_invalid")
        self.input_anchors = tuple(inputs)
        self.verify()

    def capture_methods(self):
        from agent import chat_completion_helpers as caller
        self.caller = caller.interruptible_api_call
        self.caller_functions = (caller.interruptible_api_call, caller._controlled_main_api_call,
                                 caller._dispatch_nonstreaming_api_request, dispatch_controlled_main)
        entries = [
            (self.runtime, name) for name in ("_check_operator", "_operator", "_content", "admit", "derive", "dispatch")]
        entries += [(self.runtime._admission, "current")]
        entries += [(self.parent, name) for name in (
            "bind_tool_call_probe", "run_tool_call_probe", "_checkpoint", "_identity", "validate_scope",
            "_around", "_model_open", "_turn_operation", "_invalidate", "_dispatch_retained_operation")]
        entries += [(caller, name) for name in ("interruptible_api_call", "_controlled_main_api_call",
                                               "_dispatch_nonstreaming_api_request")]
        entries += [(sys.modules[__name__], name) for name in (
            "dispatch_controlled_main", "_tool_probe_request", "_tool_probe_response", "_tool_probe_arguments",
            "_tool_probe_schema", "_probe_failure", "_resumption_callback", "_close_constructed_main")]
        entries += [(wrapper, name) for wrapper in self.parent._transport_wrappers.values() for name in ("send", "resolve")]
        entries += [(self, name) for name in (
            "verify", "checkpoint", "access", "scope_access", "callback", "bind", "run", "dispatch", "send",
            "observe_operation", "window", "caller_access", "content", "close", "current")]
        return tuple((obj, name, _method(getattr(obj, name))) for obj, name in entries)

    def access(self):
        _probe_require(self.depth == 0 and threading.get_ident() == self.thread, "reentry_or_thread")
        p = self.parent
        issued = _issued_window(p)
        if issued is not None:
            issued.access()
            if issued.source_active():
                _probe_require(self.phase == "released" and self.attempted and self.run_frame is None,
                               "source_window_invalid")
                return
        active = current_context()
        _probe_require((active is p._bound_context or (self.phase == "released" and self.run_frame is None
                       and active is None)) and current_context(self.agent) is p._bound_context, "scope_invalid")
        if self.bind_frame is not None:
            frame, public = self.bind_frame, self.public_bind
            _probe_require(frame.f_code is _ToolCallProbe.bind.__code__ and frame.f_back is public
                           and public.f_code is NativeMainPayloadAdapter.bind_tool_call_probe.__code__, "bind_frame_invalid")
            _probe_require(public.f_locals.get("agent") is self.agent and public.f_locals.get("self") is p
                           and _same_anchors(tuple(public.f_locals.get(name) for name in (
                               "system_message", "user_message", "task_id", "expected_arguments_json")), self.input_anchors),
                           "bind_input_changed")
        elif self.run_frame is not None:
            frame, public = self.run_frame, self.public_run
            _probe_require(self.active_call is self.call_anchor and self.active_call is not None, "call_invalid")
            _probe_require(frame.f_code is _ToolCallProbe.run.__code__ and frame.f_back is public
                           and public.f_code is NativeMainPayloadAdapter.run_tool_call_probe.__code__, "run_frame_invalid")
            _probe_require(public.f_locals.get("agent") is self.agent and public.f_locals.get("self") is p
                           and public.f_locals.get("authorization") is self.binding.token
                           and frame.f_locals.get("authorization") is self.binding.token, "token_invalid")
        else:
            _probe_require(self.phase in ("bound", "released"), "window_missing")
            return
        _probe_require(frame.f_locals.get("self") is self, "owner_changed")
        current = sys._getframe()
        while current is not None and current is not frame:
            current = current.f_back
        _probe_require(current is frame, "frame_missing")
        if self.stage_frame is not None:
            _probe_require(self.stage_frame is self.stage_anchor, "stage_changed")
            current = sys._getframe()
            while current is not None and current is not self.stage_frame:
                current = current.f_back
            _probe_require(current is self.stage_frame, "stage_missing")

    def verify(self):
        _probe_code_current()
        p = self.parent
        _probe_require(_tool_call_probes.get(p) is self and p._tool_probe_owner is self, "owner_changed")
        _probe_require(self.phase is self.phase_anchor and self.phase in (
            "bound", "entered", "sent", "observed", "released"), "owner_denied")
        self.access()
        _probe_require(not p._invalid and not p._bound_context._denied, "owner_denied")
        _probe_require(p._runtime is self.runtime is p._captured_runtime, "runtime_changed")
        _probe_require(all(getattr(p, name) is value for name, value in self.native_facts), "native_facts_changed")
        _probe_require(p._operator_configuration is self.catalog
                       and _same_anchors(_nested_anchors(self.catalog), self.catalog_anchors), "catalog_changed")
        _probe_require(_same_anchors(_nested_anchors((self.source, self.selection, self.producer))[1:],
                                    self.original_anchors), "original_facts_changed")
        _probe_require(_same_anchors(tuple(getattr(p, name) for name in (
            "_turn_frames", "_records", "_record_anchors", "_completions", "_completion_anchors",
            "_aux_frames", "_aux_records", "_aux_local_results", "_failed_main_attempts", "_fallback_records")),
            self.proof), "ordinary_records_changed")
        _probe_require(p not in _memory_snapshots and p not in _memory_prompt_frames and p not in _delegate_frames
                       and p not in _parent_resumptions and not hasattr(p, "_delegate_parent_frame")
                       and not hasattr(p, "_delegate_frame"), "participant_unavailable")
        _probe_require(p._aux_binding is p._compression_binding is p._compression_owner is p._fallback_binding
                       is p._fallback_frame is p._aux_pending is p._turn_pending is None
                       and not p._fallback_terminal, "participant_unavailable")
        _probe_require(p._operation is p._checked_operation is self.operation, "operation_changed")
        _probe_require(type(p._send_count) is int and p._send_count == (0 if self.partial_send is None else 1)
                       and p._aux_send_count == p._fallback_send_count == 0 and p._send_count <= 8, "send_count_changed")
        _probe_require(self.caller is _probe_caller_module.interruptible_api_call
                       and _same_anchors(self.caller_functions, (
                           _probe_caller_module.interruptible_api_call, _probe_caller_module._controlled_main_api_call,
                           _probe_caller_module._dispatch_nonstreaming_api_request, dispatch_controlled_main)), "caller_changed")
        for name in ("bind_tool_call_probe", "run_tool_call_probe", "_dispatch_retained_operation",
                     "_checkpoint", "_identity", "validate_scope", "_around", "_model_open", "_turn_operation", "_invalidate"):
            _probe_require(_same_method(getattr(p, name), (p, getattr(NativeMainPayloadAdapter, name))), "adapter_method_changed")
        _probe_require(self.methods is self.methods_anchor, "methods_changed")
        _probe_require(all(getattr(self.runtime, name) is value for name, value in self.runtime_facts)
                       and _same_anchors(_nested_anchors(self.runtime._route_rows), self.runtime_routes)
                       and len(self.runtime._ports) == len(self.runtime_ports)
                       and all(self.runtime._ports.get(name) is port for name, port in self.runtime_ports), "runtime_ports_changed")
        for obj, name, method in self.methods:
            _probe_require(_same_method(getattr(obj, name, None), method), "method_changed")
        if not self.closed:
            p._identity()
        else:
            b, namespace = self.constructed, _namespace(self.agent)
            _probe_require(b.closed is True and self.phase == "released"
                           and namespace.get("_native_controlled_binding") is b, "agent_changed")
            _probe_require(set(namespace) == {name for name, _ in b.anchors}
                           and _constructed_state(namespace) == b.state, "agent_changed")
            dynamic = {"_interrupt_requested", "_interrupt_message", "_tool_interrupt_reason", "_interrupt_thread_signal_pending"}
            _probe_require(all(namespace[name] is original for name, original in b.anchors if name not in dynamic), "agent_changed")
            for obj, name, method in b.methods:
                _probe_require(_same_method(getattr(obj, name), method), "method_changed")
        namespace = _namespace(self.agent)
        interrupted = namespace.get("_interrupt_requested")
        _probe_require(type(interrupted) is bool, "control_invalid")
        if interrupted:
            raise InterruptedError("Agent interrupted during API call")
        if self.binding is not None:
            _probe_require(self.binding is self.registered_binding
                           and _same_anchors(_row(self.binding, NativeToolCallProbeBinding), self.binding_anchor), "binding_changed")
            self.content(self.binding.authorization_content, self.binding.authorization_bytes, ())
        if self.request is not None:
            _tool_probe_request(self.request)
            _probe_require(_same_anchors(_nested_anchors(self.request), self.request_anchors)
                           and _json(self.request) == self.request_bytes, "request_changed")
        for handle, payload, parents in (
            (self.request_content, self.request_bytes, (self.binding.authorization_content,) if self.binding else ()),
            (self.wire_content, self.wire_bytes, (self.binding.authorization_content, self.request_content) if self.binding else ()),
            (self.response_content, self.response_bytes, (self.wire_content,)),
            (self.observation_content, self.observation_bytes,
             (self.binding.authorization_content, self.request_content, self.wire_content, self.response_content) if self.binding else ()),
        ):
            if handle is not None:
                self.content(handle, payload, parents)
        _probe_require(type(self.records) is tuple and len(self.records) <= 1 and len(self.records) == len(self.record_anchors),
                       "records_changed")
        for record, anchor in zip(self.records, self.record_anchors):
            _probe_require(record is anchor[0]
                           and _same_anchors(_row(record, NativeToolCallObservation), anchor[1]), "record_changed")
        if self.operation is not None:
            _probe_require(_same_anchors(_row(self.operation, _PendingOperation), self.operation_anchor), "operation_changed")
        if self.envelope is not None:
            _probe_require(_same_anchors(_nested_anchors(self.envelope), self.envelope_anchor), "envelope_changed")

    def callback(self, callback, *args):
        self.verify()
        self.runtime._check_operator()
        self.depth += 1
        try:
            result = callback(*args)
            if (self.partial_send is not None and len(args) == 1 and args[0] is self.envelope
                    and _same_method(callback, _method(self.parent._operator_transports[
                        self.selection.destination.transport_ref].send))):
                self.response_bytes = result
                self.partial_send = (*self.partial_send[:5], result)
                self.parent._pending = (self.envelope, self.observed_destination, result)
            return result
        except BaseException as error:
            import asyncio
            if isinstance(error, (InterruptedError, asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                if self.control_exception is None:
                    self.control_exception = error
            raise
        finally:
            self.depth -= 1
            if self.control_exception is None:
                self.runtime._check_operator()
                self.verify()

    def checkpoint(self):
        self.verify()
        self.runtime._check_operator()
        p = self.parent
        first = None
        for _ in range(2):
            source = self.callback(p._admission.current, p._handle)
            selected = self.callback(p._selection_port.current, p._handle)
            producer = self.callback(p._result_port.current, p._producer_ref)
            facts = _binding(source), _selection(selected), _producer(producer)
            _probe_require(facts[0][:8] == self.initial[0][:8]
                           and set(facts[0][8]) <= set(self.original_profiles), "holder_changed")
            _probe_require(selected is self.selection and facts[1] == self.initial[1], "selection_changed")
            _probe_require(producer is self.producer and facts[2] == self.initial[2]
                           and set(facts[2][2]) <= set(self.original_profiles), "producer_changed")
            _probe_require(first is None or facts == first, "facts_unstable")
            first = facts
            now = self.callback(p._clock)
            _integer(now, minimum=0)
            _probe_require(now < p._main_route.expires_at, "selection_expired")
            allowed = {r.route_ref for r in p._operator_routes}
            for profile in p._operator_profiles:
                if profile.profile_ref in self.original_profiles:
                    allowed.intersection_update(profile.allowed_routes)
            for path in shared_policy().PATHS:
                _probe_require(any(r.path == path and r.route_ref in allowed
                    and (r.consumer_ref, r.selection_id, r.selection_version) == (facts[0][3], facts[0][6], facts[0][7])
                    and now < r.expires_at for r in p._operator_routes), "coverage_missing")
            for route in p._operator_routes:
                actual = self.callback(p._operator_transports[route.destination.transport_ref].resolve)
                _probe_require(actual is route.destination, "destination_changed")
            self.runtime._check_operator()
        return (*first, self.wire_bytes)

    def content(self, handle, payload, parents):
        record = self.runtime._content(handle)
        _probe_require(type(handle) is shared_policy().ContentHandle and record.payload is payload
                       and _same_anchors(record.parents, parents), "content_changed")
        _probe_require(set((*self.original_profiles, *self.initial[2][2])) <= set(record.profiles), "restrictions_missing")

    @contextmanager
    def window(self):
        self.access()
        _probe_require(self.stage_frame is None and self.stage_index < 8, "window_invalid")
        self.stage_index += 1
        self.stage_frame = self.stage_anchor = sys._getframe().f_back.f_back
        try:
            self.checkpoint()
            yield
            self.checkpoint()
        finally:
            self.stage_frame = self.stage_anchor = None

    def current(self, handle):
        try:
            self.access()
            _probe_require(handle is self.parent._handle and (self.bind_frame is not None or self.run_frame is not None),
                           "admission_window_missing")
            return self.callback(self.parent._admission.current, handle)
        except BaseException as error:
            _probe_failure(self.parent, error)

    def scope_access(self):
        self.access()
        self.checkpoint()

    def bind(self):
        self.bind_frame = sys._getframe()
        try:
            with self.window():
                p = self.parent
                authorization_bytes = _json(dict(holder=self.initial[0], producer_policy=self.initial[2],
                    main_route_ref=p._route_ref, system_message=self.inputs[0], user_message=self.inputs[1],
                    task_id=self.inputs[2], expected_arguments_json=self.inputs[3]))
                _require(len(authorization_bytes) <= _MAX_NATIVE_RESULT)
                self.authorization_bytes = authorization_bytes
                self.authorization_content = self.runtime.admit(p._handle, authorization_bytes)
                self.checkpoint()
                token = object.__new__(NativeToolCallProbeInput)
                anchors = (self.constructed, self.source, self.catalog, self.methods, self.input_anchors,
                           self.original_anchors, self.native_facts, self.runtime)
                self.binding = NativeToolCallProbeBinding(token, self.agent, p._handle, self.selection,
                    self.producer, p._main_route, *self.inputs, authorization_bytes, self.authorization_content,
                    anchors, self.thread)
                self.registered_binding = self.binding
                self.binding_anchor = _row(self.binding, NativeToolCallProbeBinding)
            return token
        finally:
            self.bind_frame = self.public_bind = None

    def caller_access(self, public):
        self.access()
        frames = []
        frame = public
        for function in reversed(self.caller_functions):
            _probe_require(frame is not None and frame.f_code is function.__code__
                           and frame.f_globals is function.__globals__, "caller_invalid")
            _probe_require(frame.f_locals.get("agent") is self.agent
                           and frame.f_locals.get("api_kwargs") is self.request, "caller_input_changed")
            frames.append(frame)
            frame = frame.f_back
        _probe_require(frame is self.run_frame, "caller_invalid")
        return tuple(frames)

    def dispatch(self, agent, api_kwargs, public):
        self.caller_frames = self.caller_access(public)
        _probe_require(agent is self.agent and api_kwargs is self.request and self.phase == "entered"
                       and self.stage_index == 5 and self.operation is None, "dispatch_invalid")
        p = self.parent
        _probe_require(not p._busy and p._pending is None and p._send_count < 8, "capacity_denied")
        self.dispatch_frame = sys._getframe()
        p._busy = True
        try:
            self.operation = _PendingOperation(p._handle, self.source, self.selection, self.producer,
                "main", p._route_ref, self.wire_bytes, self.wire_content, False)
            self.operation_anchor = _row(self.operation, _PendingOperation)
            p._operation = p._checked_operation = self.operation
            self.checkpoint()
            result = p._dispatch_retained_operation(self.operation)
            _probe_require(len(self.records) == 1 and result is self.records[0], "response_changed")
            return result
        finally:
            p._operation = p._checked_operation = self.operation = None
            p._busy = False
            self.dispatch_frame = None

    def send(self, wrapper, envelope):
        self.access()
        p, operation = self.parent, self.operation
        _probe_require(self.dispatch_frame is not None and operation is p._operation is p._checked_operation
                       and type(operation) is _PendingOperation and p._busy and p._pending is None
                       and self.partial_send is None, "send_invalid")
        self.checkpoint()
        _probe_require(type(envelope) is shared_policy().DispatchEnvelope and envelope.path == "main"
                       and envelope.payload is self.wire_bytes and _binding(envelope.source)[:8] == self.initial[0][:8]
                       and set(envelope.source.profiles) <= set(self.original_profiles)
                       and set(self.original_profiles) <= set(envelope.profiles)
                       and envelope.destination == self.selection.destination
                       and envelope.recipient_ref == self.source.consumer_ref
                       and wrapper is p._transport_wrappers[self.selection.destination.transport_ref], "envelope_invalid")
        actual = wrapper.resolve()
        _probe_require(actual is self.selection.destination, "destination_changed")
        self.envelope, self.observed_destination = envelope, actual
        self.envelope_anchor = _nested_anchors(envelope)
        self.partial_send = (operation, envelope, actual, 0, 1, None)
        p._pending = (envelope, actual, None)
        p._send_count += 1
        output = self.callback(wrapper.port.send, envelope)
        # Retain the actual return before any protocol parsing or later checkpoint.
        self.response_bytes = output
        self.partial_send = (operation, envelope, actual, 0, 1, output)
        p._pending = (envelope, actual, output)
        self.phase = self.phase_anchor = "sent"
        self.checkpoint()
        _probe_require(type(output) is bytes, "response_type_invalid")
        _probe_require(wrapper.resolve() is actual, "destination_changed")
        return output

    def observe_operation(self, operation, result):
        _probe_require(operation is self.operation and self.partial_send is not None, "operation_changed")
        self.response_content = result
        self.checkpoint()
        value = _tool_probe_response(self.response_bytes, self.selection.destination.model_ref, self.inputs[3])
        call = value["choices"][0]["message"]["tool_calls"][0]
        observation = _json(dict(api_calls=self.parent._send_count, call=call, executed=False,
                                 request=self.request, response=value, status="tool_call_observed"))
        _require(len(observation) <= _MAX_NATIVE_RESULT)
        self.observation_bytes = observation
        self.observation_content = self.runtime.derive(self.parent._handle,
            (self.binding.authorization_content, self.request_content, self.wire_content, result), observation, path="main")
        record = NativeToolCallObservation(self.binding, self.request_bytes, self.request_content,
            self.wire_bytes, self.wire_content, self.envelope, self.observed_destination,
            self.response_bytes, result, call["id"], call["function"]["name"], call["function"]["arguments"],
            observation, self.observation_content)
        self.records = (record,)
        self.record_anchors = ((record, _row(record, NativeToolCallObservation)),)
        self.parent._dispatch_response = record
        self.phase = self.phase_anchor = "observed"
        self.checkpoint()
        return record

    def run(self, agent, authorization, public):
        _probe_require(self.phase == "bound" and not self.attempted and self.run_frame is None
                       and self.depth == 0 and agent is self.agent and self.binding is not None
                       and type(authorization) is NativeToolCallProbeInput
                       and authorization is self.binding.token, "attempt_invalid")
        self.attempted = True
        self.run_frame, self.public_run = sys._getframe(), public
        self.active_call = self.call_anchor = object()
        self.phase = self.phase_anchor = "entered"
        from agent.content_policy import content_policy_scope
        try:
            with content_policy_scope(self.parent._bound_context):
                with self.window():
                    self.checkpoint()
                with self.window():
                    request = dict(model=self.selection.destination.model_ref,
                        messages=[dict(role="system", content=self.inputs[0]), dict(role="user", content=self.inputs[1])],
                        stream=False, tools=[_tool_probe_schema()],
                        tool_choice={"type": "function", "function": {"name": "delegate_task"}})
                    _tool_probe_request(request)
                    self.request_bytes = _json(request)
                    self.request, self.request_anchors = request, _nested_anchors(request)
                    self.request_content = self.runtime.derive(self.parent._handle,
                        (self.binding.authorization_content,), self.request_bytes, path="main")
                with self.window():
                    self.wire_bytes = _json(dict(api_mode="chat_completions",
                        destination=_destination_value(self.selection.destination), request=request))
                    _require(len(self.wire_bytes) <= _MAX_NATIVE_RESULT)
                    self.wire_content = self.runtime.derive(self.parent._handle,
                        (self.binding.authorization_content, self.request_content), self.wire_bytes, path="main")
                with self.window():
                    result = self.caller(agent, request)
                with self.window():
                    _probe_require(len(self.records) == 1 and result is self.records[0], "response_changed")
                with self.window():
                    self.caller_result = result
            # The inner scope validates its exit before final release.
            with self.window():
                _probe_require(result is self.caller_result is self.records[0], "response_changed")
                self.phase = self.phase_anchor = "released"
            return result
        finally:
            self.run_frame = self.public_run = self.active_call = self.call_anchor = None
            self.parent._pending = None

    def close(self):
        _probe_require(self.phase == "released" and self.depth == 0 and self.run_frame is None, "close_invalid")
        self.checkpoint()
        self.closed = True
        self.constructed.closed = True
        self.checkpoint()
        return True


def _probe_code_current():
    for obj, name, function, code in _TOOL_PROBE_CODE:
        _probe_require(getattr(obj, name, None) is function and function.__code__ is code, "original_code_changed")


from agent import chat_completion_helpers as _probe_caller_module

_TOOL_PROBE_CODE = tuple((obj, name, getattr(obj, name), getattr(obj, name).__code__)
    for obj, names in (
        (_probe_caller_module, ("interruptible_api_call", "_controlled_main_api_call", "_dispatch_nonstreaming_api_request")),
        (sys.modules[__name__], ("dispatch_controlled_main", "_tool_probe_request", "_tool_probe_response",
            "_tool_probe_arguments", "_tool_probe_schema", "_probe_code_current")),
        (NativeMainPayloadAdapter, ("bind_tool_call_probe", "run_tool_call_probe", "_dispatch_retained_operation",
            "_checkpoint", "_identity", "validate_scope", "_around", "_model_open", "_turn_operation", "_invalidate")),
        (_ToolCallProbe, ("__init__", "capture_methods", "verify", "checkpoint", "access", "scope_access", "callback",
            "bind", "run", "dispatch", "send", "observe_operation", "window", "caller_access", "content", "close", "current")),
        (_Transport, ("resolve", "send")), (_Admission, ("current",)),
    ) for name in names)


class NativeIssuedCallInput:
    __slots__ = ()

    def __new__(cls):
        raise TypeError("Issued-call tokens require operator admission")


@dataclass(frozen=True, slots=True)
class NativeIssuedCallBinding:
    token: NativeIssuedCallInput
    source_adapter: object
    observation: NativeToolCallObservation
    execution_request: object
    selection: NativeMainSelection
    producer_policy: NativeResultPolicy
    agent: object
    adapter: object
    delegation_binding: NativeDelegateBinding
    authorization_bytes: bytes
    authorization_content: object
    issued_bytes: bytes
    issued_content: object
    anchors: tuple
    thread: int


@dataclass(frozen=True, slots=True)
class NativeIssuedCallResult:
    binding: NativeIssuedCallBinding
    observation: NativeToolCallObservation
    invocation: NativeDelegateInvocation
    return_facts: object
    closed_child_proof: tuple
    aggregate: str
    result_bytes: bytes
    result_content: object


def _issued_require(condition, reason):
    if not condition:
        raise NativeContentPolicyDenied("controlled_native_issued_call_" + reason)


def _issued_window(adapter):
    owner = _issued_call_frames.get(adapter)
    if owner is None:
        frame = getattr(adapter, "_delegate_parent_frame", None)
        owner = _issued_call_frames.get(frame.parent) if type(frame) is _DelegateFrame else None
    return owner if owner is not None and (owner.active_call is not None or adapter is not owner.parent) else None


def _issued_failure(source, error):
    owner = _issued_call_frames.get(source)
    if owner is not None:
        owner.phase = owner.phase_anchor = "denied"
        owner.active_call = owner.call_anchor = None
        if owner.adapter is not None:
            owner.adapter._invalidate()
        if owner.control_exception is not None:
            raise owner.control_exception
    if not isinstance(error, Exception) or isinstance(error, InterruptedError):
        raise error
    if type(error) is NativeContentPolicyDenied:
        raise error
    if type(error) is shared_policy().PolicyError and error.args == ("capacity_exhausted",):
        raise NativeContentPolicyDenied("controlled_native_issued_call_capacity_denied") from None
    raise NativeContentPolicyDenied("controlled_native_issued_call_authority_denied") from None


def _validate_issued_call_source(
    source: NativeMainPayloadAdapter,
    observation: NativeToolCallObservation,
    view: _IssuedCallExecution,
) -> None:
    view.access()
    _issued_require(source is view.parent and observation is view.observation
                    and view.source_frame is None, "source_invalid")
    view.source_frame = sys._getframe()
    try:
        probe = view.source_owner
        _issued_require(_tool_call_probes.get(source) is probe and source._tool_probe_owner is probe,
                        "source_owner_changed")
        _issued_require(probe.phase == "released" and probe.attempted is True
                        and probe.active_call is probe.run_frame is probe.bind_frame is None,
                        "source_not_released")
        _issued_require(type(observation) is NativeToolCallObservation and len(probe.records) == 1
                        and probe.records[0] is observation and probe.record_anchors[0][0] is observation,
                        "observation_invalid")
        _issued_require(observation.binding is probe.binding is probe.registered_binding
                        and source._constructed is probe.constructed
                        and source._send_count == 1 and not source._busy, "source_changed")
        probe.verify()
        _issued_require(observation.observation_content is probe.observation_content
                        and observation.observation_bytes is probe.observation_bytes
                        and observation.response_content is probe.response_content
                        and observation.response_bytes is probe.response_bytes
                        and observation.envelope is probe.envelope, "source_content_changed")
        call = json.loads(observation.observation_bytes)["call"]
        _issued_require(call == view.call_value and call["id"] == observation.call_id
                        and call["function"]["name"] == observation.function_name == "delegate_task"
                        and call["function"]["arguments"] == observation.arguments_json, "call_changed")
    finally:
        view.source_frame = None


class _IssuedCallExecution:
    """Own one issued call and its original delegated result."""

    def __init__(self, source, observation, options, public):
        _issued_require(type(source) is NativeMainPayloadAdapter and source in _adapters, "source_invalid")
        _issued_require(source not in _issued_call_frames and not hasattr(source, "_issued_call_owner"), "attempt_spent")
        self.parent, self.observation, self.options = source, observation, options
        self.source_owner = _tool_call_probes.get(source)
        self.runtime, self.thread = source._runtime, source._thread
        self.catalog, self.catalog_anchors = source._operator_configuration, _nested_anchors(source._operator_configuration)
        self.public = public
        self.entry = self.stage_frame = self.stage_anchor = self.source_frame = None
        self.active_call = self.call_anchor = object()
        self.depth = self.stage_index = 0
        self.phase = self.phase_anchor = "binding"
        self.adapter = self.agent = self.context = self.frame = None
        self.registered_frame = None
        self.binding = self.registered_binding = self.binding_anchor = None
        self.token = object.__new__(NativeIssuedCallInput)
        self.registered_token = self.token
        self.result_anchors = ()
        self.initial = self.originals = None
        self.authorization_bytes = self.authorization_content = self.issued_bytes = self.issued_content = None
        self.result_bytes = self.result_content = self.aggregate = None
        self.control_exception = None
        self.partial_responses = self.partial_response_anchors = ()
        self.partial_operations = self.partial_operation_anchors = ()
        self.execution_methods = self.execution_method_anchor = ()
        self.attempted = self.checking = False
        self.methods = ()
        source._issued_call_owner = self
        _issued_require(len(_issued_call_frames) < 128, "capacity_denied")
        _issued_call_frames[source] = self
        _issued_require(current_context() is source._bound_context and self.source_owner is not None, "source_scope_invalid")
        _issued_require(type(observation) is NativeToolCallObservation, "observation_invalid")
        for index in (2, 3, 6, 9, 10, 11, 12, 13):
            _identifier(options[index])
        self.option_anchors = _nested_anchors(options)
        self.call_value = json.loads(observation.observation_bytes)["call"]
        self.call_bytes = _json(self.call_value)
        self.decoded = _tool_probe_arguments(observation.arguments_json)
        _issued_require(bool(self.decoded["goal"].strip()), "goal_invalid")
        self.decoded_anchors = _nested_anchors(self.decoded)
        self.capabilities = (source._handle, options[0], options[4], options[7])
        _issued_require(len({id(h) for h in self.capabilities}) == 4, "capabilities_invalid")
        self.selections = (source._selection_port, options[1], options[5], options[8])
        self.producers = (source._producer_ref, options[2], options[6], options[9])
        refs = (source._route_ref, options[3], options[10], options[11])
        self.routes = tuple(next((r for r in self.catalog[0] if r.route_ref == ref), None) for ref in refs)
        _issued_require(all(r is not None for r in self.routes)
                        and tuple(r.path for r in self.routes) == ("main", "main", "delegate", "main"), "routes_invalid")
        self.receiver_port = source._operator_transports[self.routes[2].destination.transport_ref]
        from run_agent import AIAgent
        from tools import delegate_tool
        self.methods = tuple((obj, name, _method(getattr(obj, name))) for obj, name in (
            *((port, "current") for port in self.selections), (source._admission, "current"),
            (source._result_port, "current"), (source, "_clock"),
            *((port, name) for _, port in self.catalog[2] for name in ("resolve", "send")),
            *((self.runtime, name) for name in ("admit", "derive", "derive_for_selection", "dispatch",
                "accept", "return_from_delegate", "_check_operator", "_content")),
            (AIAgent, "__init__"),
            *((delegate_tool, name) for name in ("delegate_task", "_build_child_agent", "_run_single_child",
                "_build_child_system_prompt", "_normalize_role")),
            *((self, name) for name in ("access", "verify", "checkpoint", "callback", "source_active",
                "window", "bind", "run", "current", "around", "resolve", "send", "open_view",
                "capture", "check_arguments", "delegate_call", "delegate_checkpoint", "scope_access"))))
        self.methods_anchor = self.methods
        self.method_codes = tuple((method[-1], method[-1].__code__) for _, _, method in self.methods
                                  if hasattr(method[-1], "__code__"))

    def access(self):
        _issued_require(self.depth == 0 and threading.get_ident() == self.thread, "reentry_or_thread")
        _issued_require(_issued_call_frames.get(self.parent) is self and self.parent._issued_call_owner is self,
                        "owner_changed")
        _issued_require(self.phase is self.phase_anchor and self.phase in ("binding", "bound", "running", "released"),
                        "owner_denied")
        _issued_require(self.active_call is self.call_anchor and self.active_call is not None
                        and self.entry is not None, "window_missing")
        _issued_require(self.entry.f_locals.get("self") is self and self.entry.f_back is self.public, "frame_changed")
        bind = self.entry.f_code is _IssuedCallExecution.bind.__code__
        _issued_require(bind or self.entry.f_code is _IssuedCallExecution.run.__code__, "frame_invalid")
        _issued_require(self.public.f_code is (NativeMainPayloadAdapter.bind_issued_call_execution.__code__ if bind
                        else NativeMainPayloadAdapter.run_issued_call_execution.__code__)
                        and self.public.f_locals.get("self") is self.parent, "public_frame_invalid")
        if bind:
            _issued_require(self.public.f_locals.get("observation") is self.observation, "observation_changed")
            names = ("execution_request", "execution_selection", "execution_producer_ref", "execution_main_route_ref",
                "receiver_request", "receiver_selection", "receiver_producer_ref", "child_request", "child_selection",
                "child_producer_ref", "delegate_route_ref", "child_main_route_ref", "child_session_id", "child_task_id")
            _issued_require(_same_anchors(tuple(self.public.f_locals.get(n) for n in names), self.options), "options_changed")
        else:
            _issued_require(self.public.f_locals.get("authorization") is self.token
                            and self.entry.f_locals.get("authorization") is self.token, "token_invalid")
        current = sys._getframe()
        frames = []
        while current is not None and current is not self.entry:
            frames.append(current)
            current = current.f_back
        _issued_require(current is self.entry, "frame_missing")
        if self.stage_frame is not None:
            _issued_require(self.stage_frame is self.stage_anchor
                            and (self.stage_frame is self.entry or self.stage_frame in frames), "stage_missing")
        frame = self.frame or (_delegate_frames.get(self.adapter) if self.adapter is not None else None)
        child = frame.child_context if frame is not None else None
        _issued_require(any(current_context() is c for c in (self.parent._bound_context, self.context, child, None)),
                        "scope_invalid")

    def source_active(self):
        self.access()
        if self.source_frame is None:
            return False
        frame = self.source_frame
        _issued_require(frame.f_code is _validate_issued_call_source.__code__
                        and frame.f_locals.get("view") is self and frame.f_locals.get("source") is self.parent
                        and frame.f_locals.get("observation") is self.observation, "source_frame_invalid")
        current = sys._getframe()
        while current is not None and current is not frame:
            current = current.f_back
        _issued_require(current is frame and self.stage_frame is not None, "source_window_missing")
        return True

    def verify(self):
        self.access()
        for obj, name, function, code in _ISSUED_CALL_CODE:
            _issued_require(getattr(obj, name, None) is function and function.__code__ is code, "original_code_changed")
        _issued_require(self.parent._runtime is self.parent._captured_runtime is self.runtime
                        and self.parent._operator_configuration is self.catalog, "runtime_changed")
        _issued_require(_same_anchors(_nested_anchors(self.catalog), self.catalog_anchors)
                        and _same_anchors(_nested_anchors(self.options), self.option_anchors)
                        and _same_anchors(_nested_anchors(self.decoded), self.decoded_anchors)
                        and _json(self.call_value) == self.call_bytes, "facts_changed")
        _issued_require(self.methods is self.methods_anchor, "methods_changed")
        _issued_require(self.token is self.registered_token, "token_changed")
        for obj, name, original in self.methods:
            _issued_require(_same_method(getattr(obj, name, None), original), "method_changed")
        for function, code in self.method_codes:
            _issued_require(function.__code__ is code, "method_code_changed")
        _issued_require(type(self.partial_responses) is tuple and len(self.partial_responses) <= 2
                        and _same_anchors(_nested_anchors(self.partial_responses)[1:], self.partial_response_anchors),
                        "partial_response_changed")
        _issued_require(type(self.partial_operations) is tuple
                        and len(self.partial_operations) == len(self.partial_responses)
                        and _same_anchors(_nested_anchors(self.partial_operations)[1:], self.partial_operation_anchors),
                        "partial_operation_changed")
        if self.adapter is not None and hasattr(self.adapter, "_captured"):
            a = self.adapter
            _issued_require(type(a) is NativeMainPayloadAdapter and _issued_call_frames.get(a) is self and a._issued_call_frame is self
                            and not a._invalid and not a._bound_context._denied
                            and a._bound_context is self.context and a._handle is self.options[0], "execution_changed")
            if hasattr(a, "_runtime"):
                _issued_require(a._runtime is a._captured_runtime is self.runtime, "runtime_changed")
            _issued_require(self.execution_methods is self.execution_method_anchor, "execution_methods_changed")
            for obj, name, original, code in self.execution_methods:
                _issued_require(obj is a and _same_method(getattr(a, name), original)
                                and original[-1].__code__ is code, "execution_method_changed")
        if self.registered_frame is not None:
            _issued_require(self.frame is self.registered_frame is _delegate_frames.get(self.adapter)
                            and self.adapter._delegate_frame is self.frame, "delegate_owner_changed")
            for name in _ISSUED_FRAME_METHODS:
                _issued_require(_same_method(getattr(self.frame, name), (self.frame, getattr(_DelegateFrame, name))),
                                "delegate_method_changed")
            child = self.frame.child_adapter
            if child is not None:
                _issued_require(type(child) is NativeMainPayloadAdapter, "child_adapter_changed")
                for name in _ISSUED_ADAPTER_METHODS:
                    _issued_require(_same_method(getattr(child, name), (child, getattr(NativeMainPayloadAdapter, name))),
                                    "child_method_changed")
        if self.binding is not None:
            _issued_require(self.binding is self.registered_binding
                            and _same_anchors(_row(self.binding, NativeIssuedCallBinding), self.binding_anchor)
                            and self.frame is self.binding.anchors[-1]
                            and self.frame.binding is self.binding.anchors[-2] is self.binding.delegation_binding
                            and self.execution_methods is self.binding.anchors[-3], "binding_changed")
        for result, anchor in self.result_anchors:
            _issued_require(_same_anchors(_row(result, NativeIssuedCallResult), anchor), "result_changed")

    def callback(self, callback, *args):
        self.verify()
        _issued_require(any(_same_method(callback, method) for _, _, method in self.methods), "callback_invalid")
        self.runtime._check_operator()
        response_slot = None
        frame = self.registered_frame
        if (frame is not None and len(args) == 1 and frame.partial_sends
                and args[0] is frame.partial_sends[-1][1]
                and any(_same_method(callback, _method(port.send)) for _, port in self.catalog[2])):
            caller = sys._getframe(1)
            _issued_require(caller.f_code is _DelegateFrame.send.__code__ and caller.f_locals.get("self") is frame
                            and caller.f_locals.get("envelope") is args[0] and self.stage_index == 4
                            and self.phase == "running" and frame.running, "send_callback_window_invalid")
            _issued_require(len(self.partial_operations) < 2
                            and len(frame.partial_sends) == len(self.partial_operations) + 1,
                            "partial_capacity_denied")
            adapter, envelope, operation = frame.partial_sends[-1]
            pending = adapter._pending
            _issued_require(type(operation) is _PendingOperation
                            and adapter._operation is adapter._checked_operation is operation
                            and type(pending) is tuple and pending[0] is envelope,
                            "partial_operation_changed")
            response_slot = len(self.partial_operations), adapter, envelope, operation, pending
            self.partial_operations += ((frame, adapter, envelope, operation, pending),)
            self.partial_operation_anchors = _nested_anchors(self.partial_operations)[1:]
            self.partial_responses += ((adapter, envelope, None),)
            self.partial_response_anchors = _nested_anchors(self.partial_responses)[1:]
        self.depth += 1
        try:
            output = callback(*args)
            if response_slot is not None:
                index, adapter, envelope, operation, pending = response_slot
                self.partial_responses = (*self.partial_responses[:index], (adapter, envelope, output))
                self.partial_response_anchors = _nested_anchors(self.partial_responses)[1:]
                _issued_require(adapter._pending is pending and adapter._operation is adapter._checked_operation is operation,
                                "partial_operation_changed")
            return output
        except BaseException as error:
            import asyncio
            if isinstance(error, (InterruptedError, asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                if self.control_exception is None:
                    self.control_exception = error
            raise
        finally:
            self.depth -= 1
            if self.control_exception is None:
                self.runtime._check_operator()
                self.verify()

    def checkpoint(self):
        self.verify()
        _issued_require(not self.checking, "checkpoint_reentry")
        self.checking = True
        try:
            _validate_issued_call_source(self.parent, self.observation, self)
            for _ in range(2):
                facts, originals = [], []
                for handle, selection, producer_ref in zip(self.capabilities, self.selections, self.producers):
                    source = self.callback(self.parent._admission.current, handle)
                    selected = self.callback(selection.current, handle)
                    producer = self.callback(self.parent._result_port.current, producer_ref)
                    row = _binding(source), _selection(selected), _producer(producer)
                    _issued_require(row[1][:2] == row[0][6:8] and row[1][3][1] == "synthetic_native"
                                    and row[2][0] == producer_ref, "authority_invalid")
                    facts.append(row)
                    originals.append((source, selected, producer))
                facts, originals = tuple(facts), tuple(originals)
                p, e, r, m = (f[0] for f in facts)
                _issued_require(p[:6] == e[:6] and e[0] == r[0] and e[2] == r[2]
                                and e[4:6] == r[4:6] and e[1] != r[1] and e[3] != r[3]
                                and r[:6] == m[:6] and m[1] == self.options[13], "holders_invalid")
                _issued_require(originals[0][1] is self.source_owner.selection
                                and originals[0][2] is self.source_owner.producer
                                and facts[0][0][:8] == self.source_owner.initial[0][:8]
                                and facts[0][1:] == self.source_owner.initial[1:], "source_authority_changed")
                if self.initial is None:
                    self.initial, self.originals = facts, originals
                    self.initial_anchors = _nested_anchors((facts, originals))[1:]
                    self.profile_union = tuple(sorted({p for f in facts for p in (*f[0][8], *f[2][2])}
                        | set(self.source_owner.original_profiles)))
                _issued_require(_same_anchors(_nested_anchors((self.initial, self.originals))[1:], self.initial_anchors),
                                "initial_changed")
                for index, (actual, original) in enumerate(zip(facts, self.initial)):
                    _issued_require(actual[0][:8] == original[0][:8] and set(actual[0][8]) <= set(original[0][8])
                                    and actual[1:] == original[1:]
                                    and originals[index][1] is self.originals[index][1]
                                    and originals[index][2] is self.originals[index][2]
                                    and set(actual[2][2]) <= set(self.profile_union), "authority_changed")
                port = _binding(self.callback(self.parent._admission.current, self.receiver_port))
                _issued_require(port[:8] == self.initial[2][0][:8]
                                and set(port[8]) <= set(self.initial[2][0][8]), "receiver_port_changed")
                now = self.callback(self.parent._clock)
                _integer(now, minimum=0)
                allowed = {route.route_ref for route in self.catalog[0]}
                for profile in self.catalog[1]:
                    if profile.profile_ref in self.profile_union:
                        allowed.intersection_update(profile.allowed_routes)
                for i, route in enumerate(self.routes):
                    _route_facts(route)
                    _issued_require(now < route.expires_at and route.route_ref in allowed, "selected_expired_or_denied")
                    holder = self.initial[i][0]
                    _issued_require((route.consumer_ref, route.selection_id, route.selection_version)
                                    == (holder[3], holder[6], holder[7]), "selected_holder_invalid")
                    if i != 2:
                        _issued_require(self.originals[i][1].destination is route.destination, "selected_destination_changed")
                for row in self.initial:
                    for path in shared_policy().PATHS:
                        _issued_require(any(route.route_ref in allowed and route.path == path
                            and (route.consumer_ref, route.selection_id, route.selection_version)
                            == (row[0][3], row[0][6], row[0][7]) and now < route.expires_at
                            for route in self.catalog[0]), "coverage_missing")
                for route in self.catalog[0]:
                    actual = self.callback(self.parent._operator_transports[route.destination.transport_ref].resolve)
                    _issued_require(actual is route.destination, "destination_changed")
                frame = self.frame or (_delegate_frames.get(self.adapter) if self.adapter is not None else None)
                for agent in (self.source_owner.agent, self.agent, frame.child if frame is not None else None):
                    if agent is not None:
                        flag = _namespace(agent).get("_interrupt_requested")
                        _issued_require(type(flag) is bool, "control_invalid")
                        if flag:
                            raise InterruptedError("Agent interrupted during issued-call execution")
                for handle, payload, parents in (
                    (self.authorization_content, self.authorization_bytes, ()),
                    (self.issued_content, self.issued_bytes,
                     (self.observation.observation_content, self.authorization_content)),
                    (self.result_content, self.result_bytes,
                     (self.issued_content, frame.record.return_facts.returned_content)
                     if frame is not None and frame.record is not None and frame.record.return_facts is not None else ()),
                ):
                    if handle is not None:
                        content = self.runtime._content(handle)
                        _issued_require(content.payload is payload and _same_anchors(content.parents, parents)
                                        and set(self.profile_union) <= set(content.profiles), "content_changed")
                self.runtime._check_operator()
                self.verify()
        finally:
            self.checking = False

    @contextmanager
    def window(self):
        self.access()
        _issued_require(self.stage_frame is None and self.stage_index < 8, "capacity_denied")
        self.stage_index += 1
        self.stage_frame = self.stage_anchor = self.entry
        try:
            self.checkpoint()
            yield
            self.checkpoint()
        finally:
            self.stage_frame = self.stage_anchor = None

    def current(self, handle):
        self.verify()
        valid = next((h for h in self.capabilities if handle is h), None)
        if valid is None:
            wrapper = next((w for _, w in self.catalog[3] if handle is w), None)
            _issued_require(wrapper is not None and wrapper.port is self.receiver_port, "capability_invalid")
            valid = self.receiver_port
        refresh = self.stage_index == 4 and self.frame is not None
        if refresh:
            _issued_require(_same_method(self.adapter._checkpoint,
                            (self.adapter, NativeMainPayloadAdapter._checkpoint)), "checkpoint_method_changed")
            self.adapter._checkpoint()
        result = self.callback(self.parent._admission.current, valid)
        if refresh:
            self.adapter._checkpoint()
        return result

    def around(self, callback):
        self.access()
        _issued_require(_same_method(callback, _method(self.parent._clock)), "clock_invalid")
        self.checkpoint()
        output = self.callback(callback)
        self.checkpoint()
        return output

    def resolve(self, wrapper):
        self.verify()
        _issued_require(any(wrapper is w for _, w in self.catalog[3]), "wrapper_invalid")
        actual = self.callback(wrapper.port.resolve)
        _destination(actual)
        return actual

    def send(self, wrapper, envelope):
        self.access()
        frame = self.frame
        _issued_require(frame is not None and frame is _delegate_frames.get(self.adapter)
                        and frame.running and self.phase == "running" and self.stage_index == 4,
                        "send_window_invalid")
        return frame.send(wrapper, envelope)

    def open_view(self, adapter, context):
        self.access()
        _issued_require(self.phase == "binding" and self.stage_index == 2 and self.adapter is None
                        and current_context() is None and context is self.context, "constructor_window_invalid")
        _issued_require(context.admission is self.parent._admission and context.request_handle is self.options[0],
                        "constructor_context_invalid")
        self.adapter = adapter
        adapter._issued_call_frame = self
        _issued_call_frames[adapter] = self
        self.execution_methods = tuple((adapter, name, _method(getattr(adapter, name)),
                                        getattr(adapter, name).__func__.__code__)
            for name in _ISSUED_ADAPTER_METHODS)
        self.execution_method_anchor = self.execution_methods

    def delegate_call(self, frame, callback, *args):
        self.access()
        _issued_require(frame is _delegate_frames.get(self.adapter) or (
            self.phase == "binding" and self.stage_index == 2 and frame.parent is self.adapter
            and not hasattr(self.adapter, "_delegate_frame")), "delegate_frame_invalid")
        if _same_method(callback, (self.adapter, NativeMainPayloadAdapter._checkpoint)):
            caller = sys._getframe(2)
            _issued_require(caller.f_code is _DelegateFrame.current.__code__
                            and caller.f_locals.get("self") is frame and not args, "internal_checkpoint_invalid")
            return self.adapter._checkpoint()
        return self.callback(callback, *args)

    def delegate_checkpoint(self, frame):
        self.access()
        _issued_require(frame is _delegate_frames.get(self.adapter) and frame.parent is self.adapter,
                        "delegate_frame_invalid")
        caller = sys._getframe(1)
        _issued_require(caller.f_code is _DelegateFrame.checkpoint.__code__
                        and caller.f_locals.get("self") is frame, "checkpoint_frame_invalid")
        _issued_require(not frame.checking, "checkpoint_reentry")
        frame.checking = True
        try:
            self.checkpoint()
            _issued_require(frame.runtime is self.runtime and frame.agent is self.agent
                            and frame.binding.parent_request is self.options[0]
                            and frame.binding.receiver_request is self.options[4]
                            and frame.binding.child_request is self.options[7]
                            and frame.binding.delegate_route is self.routes[2]
                            and frame.binding.child_main_route is self.routes[3], "delegation_changed")
            for actual, original in zip(frame.initial, self.initial[1:]):
                _issued_require(actual[0][:8] == original[0][:8] and actual[1:] == original[1:]
                                and set(actual[0][8]) <= set(original[0][8]), "delegation_authority_changed")
            if frame.record is not None:
                self.check_arguments(frame.arguments)
                _issued_require(self.runtime._content(frame.record.raw_content).parents == (self.issued_content,),
                                "raw_parent_changed")
        except BaseException as error:
            _remember_delegate_control(frame, error)
            raise
        finally:
            frame.checking = False
        _DelegateFrame.verify(frame)
        _issued_require(_delegate_frames.get(self.adapter) is frame and self.adapter._delegate_frame is frame,
                        "delegate_owner_changed")
        for name in _ISSUED_FRAME_METHODS:
            _issued_require(_same_method(getattr(frame, name), (frame, getattr(_DelegateFrame, name))),
                            "delegate_method_changed")

    def check_arguments(self, arguments):
        self.access()
        rows, positional, keywords = arguments
        names = ("goal", "context", "role", "parent_agent")
        _issued_require(type(positional) is tuple and not positional
                        and tuple(name for name, _ in keywords) == names, "arguments_presence_invalid")
        _issued_require(len(rows) == 12 and all(type(present) is bool and present is (name in names)
                        for name, present, _ in rows), "arguments_presence_invalid")
        _issued_require(tuple(name for name, _, _ in rows) == (
            "goal", "context", "tasks", "max_iterations", "role", "background", "output_schema", "action",
            "subagent_id", "message", "parent_agent", "credentials_cfg"), "argument_rows_changed")
        values = dict(goal=self.decoded["goal"], context=self.decoded["context"], role=self.decoded["role"],
                      parent_agent=self.agent)
        _issued_require(all(value is values[name] for name, value in keywords)
                        and all(value is values.get(name) for name, _, value in rows), "arguments_changed")
        _issued_require(self.observation.arguments_json == _json(self.decoded).decode("ascii")
                        and self.observation.call_id == self.call_value["id"]
                        and self.observation.function_name == self.call_value["function"]["name"] == "delegate_task",
                        "call_changed")

    def capture(self, frame, raw, arguments):
        self.access()
        _issued_require(frame is self.frame is _delegate_frames.get(self.adapter) and self.phase == "running"
                        and self.stage_index == 4 and current_context() is self.context, "capture_window_invalid")
        self.check_arguments(arguments)
        _issued_require(len(frame.inputs) < 8, "capacity_denied")
        content = self.runtime.derive(self.options[0], (self.issued_content,), raw, path="delegate")
        frame.inputs += ((raw, content),)
        return content

    def scope_access(self):
        if self.active_call is not None:
            self.access()
            self.checkpoint()
            return
        _issued_require(self.phase == self.phase_anchor == "released" and self.depth == 0
                        and threading.get_ident() == self.thread and self.result_anchors
                        and current_context() in (None, self.parent._bound_context, self.context), "terminal_scope_invalid")
        result, anchor = self.result_anchors[0]
        _issued_require(_same_anchors(_row(result, NativeIssuedCallResult), anchor)
                        and self.frame.phase == "released" and self.frame.terminal
                        and not self.frame.running and self.frame.child_adapter._constructed.closed,
                        "terminal_proof_changed")

    def bind(self):
        self.entry = sys._getframe()
        from agent.content_policy import content_policy_scope
        from run_agent import AIAgent
        try:
            with self.window():
                rows = self.initial
                value = dict(call=self.call_value,
                    execution=dict(holder=rows[1][0], selection=rows[1][1], producer_policy=rows[1][2],
                                   main_route_ref=self.options[3]),
                    receiver=dict(holder=rows[2][0], selection=rows[2][1], producer_policy=rows[2][2],
                                  delegate_route_ref=self.options[10]),
                    child=dict(holder=rows[3][0], selection=rows[3][1], producer_policy=rows[3][2],
                               main_route_ref=self.options[11], session_id=self.options[12], task_id=self.options[13]))
                self.authorization_bytes = _json(value)
                _issued_require(len(self.authorization_bytes) <= _MAX_NATIVE_RESULT, "capacity_denied")
                self.authorization_content = self.runtime.admit(self.options[0], self.authorization_bytes)
                self.issued_bytes = _json(dict(authorization=value, observation=json.loads(self.observation.observation_bytes)))
                _issued_require(len(self.issued_bytes) <= _MAX_NATIVE_RESULT, "capacity_denied")
                self.issued_content = self.runtime.derive_for_selection(self.options[0],
                    (self.observation.observation_content, self.authorization_content), self.issued_bytes,
                    path="delegate", parent_requests=(self.parent._handle, self.options[0]))
            with self.window():
                with content_policy_scope(None):
                    self.context = NativeContentContext(self.parent._admission, self.options[0])
                    routes, profiles, transports, _ = self.catalog
                    a = NativeMainPayloadAdapter(self.context, selection=self.options[1],
                        result_policy=self.parent._result_port, producer_ref=self.options[2], main_route_ref=self.options[3],
                        profiles=profiles, routes=routes, transports=transports, clock=self.parent._clock,
                        _issued_call_frame=self)
                    with content_policy_scope(self.context):
                        self.agent = AIAgent(controlled_main=a, provider="synthetic_native", api_mode="chat_completions",
                            model=self.originals[1][1].destination.model_ref, session_id=self.options[12],
                            enabled_toolsets=[], quiet_mode=True, skip_memory=True, skip_context_files=True,
                            skip_background_review=True)
                        a.bind_delegation(self.agent, receiver_request=self.options[4], receiver_selection=self.options[5],
                            receiver_producer_ref=self.options[6], child_request=self.options[7], child_selection=self.options[8],
                            child_producer_ref=self.options[9], delegate_route_ref=self.options[10],
                            child_main_route_ref=self.options[11], child_session_id=self.options[12], child_task_id=self.options[13])
                        self.frame = _delegate_frames[a]
                        self.registered_frame = self.frame
                self.binding = NativeIssuedCallBinding(self.token, self.parent, self.observation, self.options[0],
                    self.originals[1][1], self.originals[1][2], self.agent, a, self.frame.binding,
                    self.authorization_bytes, self.authorization_content, self.issued_bytes, self.issued_content,
                    (self.option_anchors, self.catalog_anchors, self.methods, self.initial_anchors,
                     self.execution_methods, self.registered_frame.binding, self.registered_frame), self.thread)
                self.registered_binding = self.binding
                self.binding_anchor = _row(self.binding, NativeIssuedCallBinding)
                self.phase = self.phase_anchor = "bound"
            return self.token
        finally:
            self.active_call = self.call_anchor = self.entry = self.public = None

    def run(self, authorization, public):
        _issued_require(self.phase == self.phase_anchor == "bound" and not self.attempted
                        and type(authorization) is NativeIssuedCallInput and authorization is self.token
                        and current_context() is self.parent._bound_context, "attempt_invalid")
        self.attempted = True
        self.entry, self.public = sys._getframe(), public
        self.active_call = self.call_anchor = object()
        self.phase = self.phase_anchor = "running"
        from agent.content_policy import content_policy_scope
        from tools.delegate_tool import delegate_task
        try:
            with self.window():
                self.frame.checkpoint()
            with self.window():
                with content_policy_scope(self.context):
                    self.aggregate = delegate_task(goal=self.decoded["goal"], context=self.decoded["context"],
                        role=self.decoded["role"], parent_agent=self.agent)
            with self.window():
                frame = self.frame
                self.check_arguments(frame.record.arguments)
                frame.checkpoint()
                frame.check_return()
                _issued_require(frame.phase == "released" and frame.terminal and not frame.running
                                and frame.child_adapter._constructed.closed
                                and _same_anchors(frame.closed_proof, frame.proof())
                                and self.aggregate.encode("utf-8") == frame.record.aggregate_bytes
                                and self.adapter._send_count == 0, "result_association_invalid")
            with self.window():
                self.result_bytes = _json(dict(call=self.call_value,
                    execution_arguments=json.loads(frame.record.raw_bytes),
                    observation=json.loads(self.observation.observation_bytes), result=json.loads(self.aggregate)))
                _issued_require(len(self.result_bytes) <= _MAX_NATIVE_RESULT, "capacity_denied")
                self.result_content = self.runtime.derive(self.options[0],
                    (self.issued_content, frame.record.return_facts.returned_content), self.result_bytes, path="tool_result")
                result = NativeIssuedCallResult(self.binding, self.observation, frame.record,
                    frame.record.return_facts, frame.closed_proof, self.aggregate, self.result_bytes, self.result_content)
                self.result_anchors = ((result, _row(result, NativeIssuedCallResult)),)
            with self.window():
                frame.checkpoint()
            with self.window():
                self.phase = self.phase_anchor = "released"
            return result
        finally:
            self.active_call = self.call_anchor = self.entry = self.public = None


_ISSUED_ADAPTER_METHODS = ("_identity", "_snapshot", "_checkpoint", "_around", "validate_scope", "_model_open",
    "_invalidate", "_deny", "bind_delegation", "_verify_records", "_verify_turns", "_verify_completions")
_ISSUED_FRAME_METHODS = tuple(n for n, f in vars(_DelegateFrame).items() if callable(f))

_ISSUED_CALL_CODE = tuple((obj, name, getattr(obj, name), getattr(obj, name).__code__)
    for obj, names in (
        (_IssuedCallExecution, tuple(n for n, f in vars(_IssuedCallExecution).items() if callable(f))),
        (NativeMainPayloadAdapter, ("bind_issued_call_execution", "run_issued_call_execution", "__init__",
            *_ISSUED_ADAPTER_METHODS)),
        (_DelegateFrame, _ISSUED_FRAME_METHODS),
        (sys.modules[__name__], ("_validate_issued_call_source", "_issued_window", "_issued_failure",
            "_resumption_callback", "_controlled_delegate_runner")),
    ) for name in names)

_ISSUED_CALL_CODE += tuple((function, "__wrapped__", function.__wrapped__, function.__wrapped__.__code__)
    for function in (_IssuedCallExecution.window, _controlled_delegate_builder, _controlled_delegate_runner))


class NativeIssuedCallFormatInput:
    __slots__ = ()

    def __new__(cls):
        raise TypeError("Formatter tokens require operator admission")


@dataclass(frozen=True, slots=True)
class NativeIssuedCallFormatBinding:
    token: NativeIssuedCallFormatInput
    source_adapter: object
    source_result: NativeIssuedCallResult
    formatter_request: object
    selection: NativeMainSelection
    producer_policy: NativeResultPolicy
    authorization_bytes: bytes
    authorization_content: object
    input_bytes: bytes
    input_content: object
    anchors: tuple
    thread: int


@dataclass(frozen=True, slots=True)
class NativeIssuedCallFormatResult:
    binding: NativeIssuedCallFormatBinding
    source_result: NativeIssuedCallResult
    message: dict
    message_bytes: bytes
    message_content: object
    anchors: tuple


_issued_call_formats = _DelegateFrames()


def _issued_format_require(condition, reason):
    if not condition:
        raise NativeContentPolicyDenied("controlled_native_issued_format_" + reason)


def _issued_call_format_window(adapter):
    owner = _issued_call_formats.get(adapter)
    return owner if owner is not None and owner.active_call is not None else None


def _issued_format_bytes(message):
    return json.dumps(message, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _issued_format_content(runtime, handle, payload, parents, profiles):
    record = runtime._content(handle)
    _issued_format_require(record.payload is payload and _same_anchors(record.parents, parents)
        and set(profiles) <= set(record.profiles), "content_changed")
    return record


def _issued_format_constructed(adapter, closed):
    binding, agent = adapter._constructed, adapter._turn_agent
    _issued_format_require(type(binding) is _ConstructedMain and binding in _constructed_bindings
        and binding.agent is agent and binding.adapter is adapter and binding.closed is closed, "constructor_changed")
    namespace = _namespace(agent)
    _issued_format_require(namespace.get("_native_controlled_binding") is binding
        and namespace.get("_native_content_policy_context") is adapter._bound_context
        and set(namespace) == {name for name, _ in binding.anchors}
        and _constructed_state(namespace) == binding.state, "attached_context_changed")
    dynamic = {"_interrupt_requested", "_interrupt_message", "_tool_interrupt_reason", "_interrupt_thread_signal_pending"}
    _issued_format_require(all(namespace[name] is value for name, value in binding.anchors if name not in dynamic),
        "constructor_anchor_changed")
    for obj, name, method in binding.methods:
        _issued_format_require(_same_method(getattr(obj, name), method), "constructor_method_changed")
    flag = namespace.get("_interrupt_requested")
    _issued_format_require(type(flag) is bool and type(namespace.get("_interrupt_thread_signal_pending")) is bool,
        "control_invalid")
    if flag:
        raise InterruptedError("Agent interrupted during issued-call format")
    _issued_format_require(namespace.get("_interrupt_message") is None
        and namespace.get("_tool_interrupt_reason") in (None, "user interrupt"), "control_changed")


def _validate_issued_call_format_source(source, result, view) -> None:
    view.access()
    _issued_format_require(source is view.parent and result is view.source_result and view.source_frame is None
        and view.stage_frame is view.entry, "source_window_invalid")
    view.source_frame = sys._getframe()
    try:
        view.source_active()
        p, e = view.probe, view.execution
        _issued_format_require(type(p) is _ToolCallProbe and type(e) is _IssuedCallExecution
            and _tool_call_probes.get(source) is p and source._tool_probe_owner is p
            and _issued_call_frames.get(source) is e and source._issued_call_owner is e
            and e.parent is p.parent is source, "source_owner_changed")
        _issued_format_require(p.phase == p.phase_anchor == e.phase == e.phase_anchor == "released"
            and p.attempted is e.attempted is True and p.stage_index == e.stage_index == 8
            and p.depth == e.depth == 0 and not e.checking
            and p.active_call is p.run_frame is p.bind_frame is e.active_call is e.entry is e.public is None,
            "source_not_terminal")
        _issued_format_require(type(result) is NativeIssuedCallResult and len(e.result_anchors) == 1
            and e.result_anchors[0][0] is result
            and _same_anchors(_row(result, NativeIssuedCallResult), e.result_anchors[0][1]), "source_result_changed")
        o = result.observation
        _issued_format_require(o is e.observation and type(o) is NativeToolCallObservation
            and len(p.records) == len(p.record_anchors) == 1 and p.records[0] is p.record_anchors[0][0] is o
            and _same_anchors(_row(o, NativeToolCallObservation), p.record_anchors[0][1])
            and o.binding is p.binding is p.registered_binding
            and _same_anchors(_row(p.binding, NativeToolCallProbeBinding), p.binding_anchor), "observation_changed")
        _issued_format_require(source._constructed is p.constructed and p.closed is p.constructed.closed
            and source._send_count == 1 and source._aux_send_count == source._fallback_send_count == 0
            and not source._busy and not source._invalid and not source._bound_context._denied,
            "source_state_changed")
        _issued_format_require(all(getattr(source, name) is value for name, value in p.native_facts)
            and _same_anchors(tuple(getattr(source, name) for name in (
                "_turn_frames", "_records", "_record_anchors", "_completions", "_completion_anchors",
                "_aux_frames", "_aux_records", "_aux_local_results", "_failed_main_attempts", "_fallback_records")), p.proof),
            "source_records_changed")
        _issued_format_require(source._operation is source._checked_operation is p.operation
            and source._pending is None and p.partial_send[4] == 1
            and _same_anchors(_row(p.partial_send[0], _PendingOperation), p.operation_anchor)
            and _same_anchors(_nested_anchors(p.envelope), p.envelope_anchor), "source_send_changed")
        _issued_format_require(o.observation_content is p.observation_content and o.observation_bytes is p.observation_bytes
            and o.response_content is p.response_content and o.response_bytes is p.response_bytes
            and o.envelope is p.envelope and p.envelope.payload is p.wire_bytes, "source_payload_changed")
        _tool_probe_request(p.request)
        _issued_format_require(_same_anchors(_nested_anchors(p.request), p.request_anchors)
            and _json(p.request) == p.request_bytes, "source_request_changed")
        _issued_format_require(_same_anchors(_nested_anchors((p.source, p.selection, p.producer))[1:], p.original_anchors)
            and _same_anchors(_nested_anchors(e.options), e.option_anchors)
            and _same_anchors(_nested_anchors(e.decoded), e.decoded_anchors)
            and _same_anchors(_nested_anchors((e.initial, e.originals))[1:], e.initial_anchors), "source_facts_changed")
        b, frame = result.binding, e.frame
        _issued_format_require(b is e.binding is e.registered_binding
            and _same_anchors(_row(b, NativeIssuedCallBinding), e.binding_anchor)
            and b.token is e.token is e.registered_token and b.source_adapter is source and b.observation is o
            and frame is e.registered_frame is b.anchors[-1] is _delegate_frames.get(e.adapter)
            and frame.binding is b.delegation_binding is b.anchors[-2]
            and b.anchors[-3] is e.execution_methods is e.execution_method_anchor, "execution_binding_changed")
        a = e.adapter
        _issued_format_require(type(a) is NativeMainPayloadAdapter and a in _adapters
            and a is b.adapter and a._issued_call_frame is e and _issued_call_frames.get(a) is e
            and a._delegate_frame is frame and a._context is a._bound_context is e.context
            and a._handle is e.options[0] and a._turn_agent is b.agent is e.agent
            and not a._invalid and not a._bound_context._denied and not a._busy and a._send_count == 0
            and not a._turn_frames and not a._records and not a._completions, "execution_changed")
        _issued_format_require(frame.phase == frame.phase_anchor == "released" and frame.terminal
            and not frame.running and not frame.invalid and not frame.checking and frame.slot is frame.slot_anchor is None
            and frame.agent is e.agent and frame.parent is a and frame.delegate_sends == 1, "delegate_not_terminal")
        invocation = result.invocation
        _issued_format_require(invocation is frame.record is frame.registered_record
            and invocation.binding is frame.binding and invocation.token is frame.token
            and _same_anchors(_row(invocation, NativeDelegateInvocation), frame.record_anchor)
            and _same_anchors(_nested_anchors(frame.record_anchor[2:])[1:], frame.record_nested)
            and _same_anchors(_nested_anchors(invocation.arguments), invocation.argument_anchors)
            and frame.encode_arguments(invocation.arguments) == invocation.raw_bytes, "invocation_changed")
        rows, positional, keywords = invocation.arguments
        values = dict(goal=e.decoded["goal"], context=e.decoded["context"], role=e.decoded["role"], parent_agent=e.agent)
        _issued_format_require(not positional and tuple(n for n, _ in keywords) == tuple(values)
            and len(rows) == 12 and all(type(present) is bool and present is (name in values)
                and value is values.get(name) for name, present, value in rows)
            and all(value is values[name] for name, value in keywords), "call_presence_changed")
        call = json.loads(o.observation_bytes)["call"]
        _issued_format_require(call == e.call_value and _json(call) == e.call_bytes
            and call["id"] == o.call_id and "|" not in o.call_id and o.call_id.strip() == o.call_id
            and call["function"]["name"] == o.function_name == "delegate_task"
            and call["function"]["arguments"] == o.arguments_json == _json(e.decoded).decode("ascii"), "call_changed")
        for text in (o.call_id, o.function_name, o.arguments_json, result.aggregate):
            _text(text)
        _issued_format_require(result.aggregate is e.aggregate and result.aggregate.encode("utf-8") == invocation.aggregate_bytes
            and result.result_bytes is e.result_bytes and result.result_content is e.result_content,
            "aggregate_changed")
        _issued_format_require(result.result_bytes == _json(dict(call=call,
            execution_arguments=json.loads(invocation.raw_bytes), observation=json.loads(o.observation_bytes),
            result=json.loads(result.aggregate))), "result_bytes_changed")
        child = frame.child_adapter
        _issued_format_require(type(child) is NativeMainPayloadAdapter and child is invocation.child_adapter
            and child._delegate_parent_frame is frame and child._turn_agent is invocation.child is frame.child
            and child._send_count == 1 and child._invalid and child._bound_context._denied
            and not child._busy and child._context is child._bound_context is frame.child_context,
            "closed_child_changed")
        completion = invocation.child_completion
        _issued_format_require(len(child._completions) == len(child._records) == len(child._turn_frames) == 1
            and child._completions[0] is completion and child._records[0] is invocation.child_invocation
            and child._turn_frames[0] is invocation.child_input and completion.invocation is invocation.child_invocation
            and len(completion.result) == 29 and _native_result_bytes(completion.result) == completion.payload
            and _same_anchors(_native_result_anchors(completion.result), completion.anchors), "completion_changed")
        child._verify_turns()
        child._verify_records()
        child._verify_completions()
        _issued_format_require(result.closed_child_proof is frame.closed_proof
            and _same_anchors(frame.closed_proof, frame.proof()), "closed_proof_changed")
        for adapter, closed in ((source, p.closed), (a, False), (child, True)):
            _issued_format_constructed(adapter, closed)
            _issued_format_require(adapter._runtime is adapter._captured_runtime is view.runtime
                and adapter._admission is source._admission
                and adapter._bound_context.admission is source._admission
                and adapter._bound_context.request_handle is adapter._handle
                and adapter._bound_context._controlled_main_payload is adapter
                and adapter._bound_context._controlled_main_bound is True, "runtime_context_changed")
        for owner in (p, e):
            _issued_format_require(owner.methods is owner.methods_anchor, "source_methods_changed")
            for obj, name, method in owner.methods:
                _issued_format_require(_same_method(getattr(obj, name), method), "source_method_changed")
        for obj, name, method, code in e.execution_methods:
            _issued_format_require(_same_method(getattr(obj, name), method) and method[-1].__code__ is code,
                "execution_method_changed")
        for obj, name, method in (*frame.methods, *frame.extra_methods):
            _issued_format_require(_same_method(getattr(obj, name), method), "delegate_method_changed")
        for name in _ISSUED_FRAME_METHODS:
            _issued_format_require(_same_method(getattr(frame, name), (frame, getattr(_DelegateFrame, name))),
                "frame_method_changed")
        for adapter in (source, a, child):
            for name in _ISSUED_ADAPTER_METHODS:
                _issued_format_require(_same_method(getattr(adapter, name), (adapter, getattr(NativeMainPayloadAdapter, name))),
                    "adapter_method_changed")
        for function, code in e.method_codes:
            _issued_format_require(function.__code__ is code, "source_code_changed")
        _issued_format_require(all(getattr(view.runtime, name) is value for name, value in p.runtime_facts)
            and _same_anchors(_nested_anchors(view.runtime._route_rows), p.runtime_routes)
            and len(view.runtime._ports) == len(p.runtime_ports)
            and all(view.runtime._ports.get(name) is port for name, port in p.runtime_ports), "runtime_ports_changed")
        for handle, payload, parents, profiles in (
            (p.binding.authorization_content, p.binding.authorization_bytes, (), p.original_profiles),
            (p.request_content, p.request_bytes, (p.binding.authorization_content,), p.original_profiles),
            (p.wire_content, p.wire_bytes, (p.binding.authorization_content, p.request_content), p.original_profiles),
            (p.response_content, p.response_bytes, (p.wire_content,), p.original_profiles),
            (p.observation_content, p.observation_bytes,
                (p.binding.authorization_content, p.request_content, p.wire_content, p.response_content), p.original_profiles),
            (e.authorization_content, e.authorization_bytes, (), e.profile_union),
            (e.issued_content, e.issued_bytes, (p.observation_content, e.authorization_content), e.profile_union),
            (invocation.raw_content, invocation.raw_bytes, (e.issued_content,), e.profile_union),
            (invocation.leaf_content, invocation.leaf_bytes, (invocation.raw_content,), e.profile_union),
            (invocation.request_content, frame.plan, (invocation.raw_content, invocation.leaf_content), e.profile_union),
            (invocation.accepted_content, frame.plan, (invocation.request_content,), e.profile_union),
            (invocation.child_seed, invocation.leaf_bytes, (invocation.accepted_content,), e.profile_union),
            (completion.content, completion.payload, completion.parents, e.profile_union),
            (frame.outputs[-2][1], invocation.entry_bytes, (invocation.accepted_content, completion.content), e.profile_union),
            (invocation.receiver_result, invocation.aggregate_bytes,
                (invocation.accepted_content, completion.content, frame.outputs[-2][1]), e.profile_union),
            (result.return_facts.returned_content, invocation.aggregate_bytes,
                (invocation.request_content, invocation.accepted_content, invocation.receiver_result), e.profile_union),
            (result.result_content, result.result_bytes,
                (e.issued_content, result.return_facts.returned_content), e.profile_union),
        ):
            _issued_format_content(view.runtime, handle, payload, parents, profiles)
        turn, main = invocation.child_input, invocation.child_invocation
        _issued_format_require(turn.input_content is turn.admitted and len(turn.request_contents) == 1
            and main.request_handle is frame.binding.child_request
            and child._kwargs is None and type(invocation.child_request) is dict
            and _json(dict(api_mode="chat_completions",
                destination=_destination_value(main.selection.destination), request=invocation.child_request)) == main.request_bytes
            and main.envelope.payload is main.request_bytes
            and main.observed_destination is main.envelope.destination
            and _destination(main.envelope.destination) == _destination(main.selection.destination),
            "child_request_changed")
        prepared_bytes = _json(dict(turn_context=json.loads(turn.output_bytes), request=json.loads(turn.request_bytes)))
        for handle, payload, parents in (
            (turn.admitted, turn.raw_bytes, (invocation.child_seed,)),
            (turn.output_content, prepared_bytes, (turn.admitted, turn.input_content)),
            (turn.request_contents[0], turn.request_bytes, (turn.admitted, turn.input_content, turn.output_content)),
            (main.request_content, main.request_bytes, (turn.admitted, turn.input_content, turn.output_content)),
            (main.result_content, main.response_bytes, (main.request_content,)),
            (invocation.ack_content, invocation.ack_bytes, (invocation.request_content,)),
        ):
            content = view.runtime._content(handle)
            _issued_format_require(content.payload == payload and _same_anchors(content.parents, parents)
                and set(e.profile_union) <= set(content.profiles), "child_stage_changed")
        facts = result.return_facts
        _issued_format_require(type(facts) is shared_policy().DelegateReturnFacts and facts is invocation.return_facts
            and facts.envelope is invocation.envelope and facts.accepted_content is invocation.accepted_content
            and facts.receiver_result is invocation.receiver_result and facts.sender is invocation.envelope.source
            and _binding(facts.sender)[:8] == e.initial[1][0][:8]
            and _binding(facts.receiver)[:8] == e.initial[2][0][:8]
            and set(e.profile_union) <= set(facts.profiles), "return_changed")
        issued = view.runtime._envelopes.get(id(invocation.envelope))
        acceptance = view.runtime._delegate_acceptances.get(id(invocation.accepted_content))
        _issued_format_require(issued is not None and issued.sender_handle is e.options[0]
            and acceptance is not None and acceptance.receiver_request is e.options[4]
            and acceptance.receiver is facts.receiver
            and view.runtime._delegate_returns.get(id(invocation.envelope)) is invocation.envelope,
            "registered_return_changed")
        view.runtime._delegate_anchors(issued, acceptance, invocation.receiver_result)
        entry = json.loads(invocation.entry_bytes)
        _issued_format_require(invocation.entry_bytes == json.dumps(entry, ensure_ascii=False).encode("utf-8")
            and result.aggregate == json.dumps(dict(results=[entry], total_duration_seconds=entry["duration_seconds"]),
                ensure_ascii=False) and entry["summary"] == completion.result["final_response"], "entry_changed")
        view.runtime._check_operator()
        if view.pending_handle is not None:
            view.current_row = view.callback(source._admission.current, view.pending_handle)
    except BaseException as error:
        import asyncio
        if isinstance(error, (InterruptedError, asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
            if view.control_exception is None:
                view.control_exception = error
        raise
    finally:
        view.source_frame = None


class _IssuedCallFormat:
    """Retain one actual formatter result from an original terminal issued call."""

    def __init__(self, source, result, options, public):
        _issued_format_require(type(source) is NativeMainPayloadAdapter and source in _adapters, "source_invalid")
        _issued_format_require(source not in _issued_call_formats and not hasattr(source, "_issued_call_format_owner"),
            "attempt_spent")
        self.parent, self.source_result, self.options = source, result, options
        self.probe, self.execution = _tool_call_probes.get(source), _issued_call_frames.get(source)
        self.runtime, self.thread, self.catalog = source._runtime, source._thread, source._operator_configuration
        self.catalog_anchors, self.option_anchors = _nested_anchors(self.catalog), _nested_anchors(options)
        self.public, self.entry = public, None
        self.active_call = self.call_anchor = object()
        self.source_frame = self.stage_frame = None
        self.depth = self.stage_index = 0
        self.phase = self.phase_anchor = "binding"
        self.checking = self.attempted = False
        self.token = self.registered_token = object.__new__(NativeIssuedCallFormatInput)
        self.binding = self.registered_binding = self.binding_anchor = None
        self.authorization_bytes = self.authorization_content = self.input_bytes = self.input_content = None
        self.message_content = self.message = self.message_bytes = self.message_anchors = None
        self.partial_results = self.partial_anchors = self.records = ()
        self.initial = self.originals = None
        self.source_snapshots = self.graph_snapshots = None
        self.pending_handle = self.current_row = None
        self.control_exception = None
        self.methods = self.methods_anchor = ()
        source._issued_call_format_owner = self
        _issued_format_require(len(_issued_call_formats) < 128, "capacity_denied")
        _issued_call_formats[source] = self
        _issued_format_require(current_context() is None or current_context() is source._bound_context, "scope_invalid")
        _identifier(options[2])
        _identifier(options[3])
        self.route = next((route for route in self.catalog[0] if route.route_ref == options[3]), None)
        _issued_format_require(self.route is not None and self.route.path == "main", "route_missing")
        self.selected_routes = tuple(next((route for route in self.catalog[0]
            if route.path == "main" and route.destination is original[1].destination
            and (route.consumer_ref, route.selection_id, route.selection_version)
                == (row[0][3], row[0][6], row[0][7])), None)
            for original, row in zip(self.execution.originals, self.execution.initial)) + (self.route,)
        _issued_format_require(all(route is not None for route in self.selected_routes), "selected_route_missing")
        _issued_format_require(all(options[1] is not port for port in self.execution.selections)
            and options[2] not in self.execution.producers, "independent_formatter_ports_required")
        self.wall_time = _format_timestamp_module.wall_time
        self.wall_method = _method(self.wall_time)
        self.wall_code = getattr(self.wall_time, "__code__", None)
        self.formatter = _format_helpers.make_tool_result_message
        self.helper_constants = tuple((name, getattr(_format_helpers, name)) for name in (
            "_UNTRUSTED_TOOL_NAMES", "_UNTRUSTED_TOOL_PREFIXES", "_UNTRUSTED_WRAP_MIN_CHARS",
            "_DELIMITER_TOKEN_RE", "_UPSTREAM_ELISION_PATTERNS", "_ELISION_SCAN_MIN_CHARS",
            "_ELISION_SCAN_MAX_CHARS", "_UPSTREAM_ELISION_NOTICE"))
        self.methods = tuple((obj, name, _method(getattr(obj, name))) for obj, name in (
            (source._admission, "current"), (source._result_port, "current"), (source, "_clock"),
            (options[1], "current"),
            *((port, "current") for port in self.execution.selections),
            *((port, name) for _, port in self.catalog[2] for name in ("resolve", "send")),
            *((self.runtime, name) for name in ("_check_operator", "_content", "_delegate_anchors", "_delegate_ancestry",
                "_call", "_current", "_now", "_coverage", "_fresh", "admit", "derive", "derive_for_selection", "dispatch", "accept", "return_from_delegate")),
            *((self, name) for name in _ISSUED_FORMAT_METHODS)))
        self.methods_anchor = self.methods
        self.method_codes = tuple((method[-1], method[-1].__code__) for _, _, method in self.methods
            if hasattr(method[-1], "__code__"))

    def access(self):
        _issued_format_require(self.depth == 0 and threading.get_ident() == self.thread, "reentry_or_thread")
        _issued_format_require(_issued_call_formats.get(self.parent) is self
            and self.parent._issued_call_format_owner is self and self.token is self.registered_token, "owner_changed")
        _issued_format_require(self.phase is self.phase_anchor and self.phase in ("binding", "bound", "running")
            and self.active_call is self.call_anchor and self.active_call is not None and self.entry is not None,
            "window_missing")
        _issued_format_require(self.entry.f_locals.get("self") is self and self.entry.f_back is self.public
            and self.entry.f_globals is globals() and self.public.f_globals is globals(), "frame_changed")
        bind = self.entry.f_code is _IssuedCallFormat.bind.__code__
        _issued_format_require(bind or self.entry.f_code is _IssuedCallFormat.run.__code__, "frame_invalid")
        _issued_format_require(self.public.f_code is (NativeMainPayloadAdapter.bind_issued_call_format.__code__
            if bind else NativeMainPayloadAdapter.run_issued_call_format.__code__)
            and self.public.f_locals.get("self") is self.parent, "public_frame_invalid")
        if bind:
            _issued_format_require(self.public.f_locals.get("result") is self.source_result
                and _same_anchors(tuple(self.public.f_locals.get(name) for name in (
                    "formatter_request", "formatter_selection", "formatter_producer_ref", "formatter_main_route_ref")),
                    self.options), "bind_input_changed")
        else:
            _issued_format_require(self.public.f_locals.get("authorization") is self.token
                and self.entry.f_locals.get("authorization") is self.token, "token_changed")
        current = sys._getframe()
        while current is not None and current is not self.entry:
            current = current.f_back
        _issued_format_require(current is self.entry and current_context() in (None, self.parent._bound_context),
            "scope_or_frame_invalid")
        _issued_format_require(self.stage_frame is None or self.stage_frame is self.entry, "stage_changed")

    def verify(self):
        self.access()
        _issued_format_require(NativeMainPayloadAdapter.issued_call_format_results.fget is _ISSUED_FORMAT_GETTER,
            "getter_changed")
        for obj, name, function, code in (*_TOOL_PROBE_CODE, *_ISSUED_CALL_CODE, *_ISSUED_FORMAT_CODE):
            _issued_format_require(getattr(obj, name, None) is function and function.__code__ is code, "code_changed")
        _issued_format_require(self.methods is self.methods_anchor and self.parent._runtime is self.runtime
            and self.parent._operator_configuration is self.catalog
            and _same_anchors(_nested_anchors(self.catalog), self.catalog_anchors)
            and _same_anchors(_nested_anchors(self.options), self.option_anchors), "facts_changed")
        for obj, name, method in self.methods:
            _issued_format_require(_same_method(getattr(obj, name), method), "method_changed")
        for function, code in self.method_codes:
            _issued_format_require(function.__code__ is code, "method_code_changed")
        _issued_format_require(_format_timestamp_module.wall_time is self.wall_time
            and _same_method(self.wall_time, self.wall_method)
            and getattr(self.wall_time, "__code__", None) is self.wall_code
            and self.formatter is _format_helpers.make_tool_result_message
            and all(getattr(_format_helpers, name) is value for name, value in self.helper_constants), "formatter_changed")
        if self.source_snapshots is not None:
            for obj, pairs, anchors in self.source_snapshots:
                actual = tuple(vars(obj).items())
                _issued_format_require(tuple(name for name, _ in actual) == tuple(name for name, _ in pairs)
                    and _same_anchors(tuple(value for _, value in actual), tuple(value for _, value in pairs))
                    and _same_anchors(_nested_anchors(tuple(value for _, value in actual))[1:], anchors),
                    "terminal_state_changed")
            for handle, original, anchor in self.graph_snapshots:
                content = self.runtime._content(handle)
                _issued_format_require(content is original and _same_anchors(tuple(getattr(content, f.name)
                    for f in fields(content)), anchor), "ancestry_changed")
            for adapter, closed in ((self.parent, self.probe.closed), (self.execution.adapter, False),
                    (self.execution.frame.child_adapter, True)):
                _issued_format_constructed(adapter, closed)
        if self.binding is not None:
            _issued_format_require(self.binding is self.registered_binding
                and _same_anchors(_row(self.binding, NativeIssuedCallFormatBinding), self.binding_anchor)
                and _same_anchors(self.binding.anchors, (self.options, self.catalog_anchors, self.initial_anchors,
                    self.source_snapshots, self.graph_snapshots, self.methods, self.wall_time)), "binding_changed")
        _issued_format_require(len(self.partial_results) <= 1
            and _same_anchors(_nested_anchors(self.partial_results)[1:], self.partial_anchors), "partial_changed")
        if self.message is not None:
            _issued_format_require(_same_anchors(_nested_anchors(self.message), self.message_anchors)
                and _issued_format_bytes(self.message) == self.message_bytes, "message_changed")

    def checkpoint(self):
        self.verify()
        _issued_format_require(self.stage_frame is self.entry and not self.checking, "checkpoint_reentry")
        self.checking = True
        try:
            _validate_issued_call_format_source(self.parent, self.source_result, self)
            if self.source_snapshots is None:
                e, p = self.execution, self.probe
                objects = (self.parent, p, e, e.adapter, e.frame, e.frame.child_adapter)
                self.source_snapshots = tuple((obj, tuple(vars(obj).items()),
                    _nested_anchors(tuple(vars(obj).values()))[1:]) for obj in objects)
                graph = self.runtime._delegate_ancestry(self.source_result.result_content)
                self.graph_snapshots = tuple((row.handle, row, tuple(getattr(row, f.name) for f in fields(row))) for row in graph)
                self.profiles = tuple(sorted(set(e.profile_union) | {p for row in graph for p in row.profiles}))
            for _ in range(2):
                facts, originals = [], []
                handles = (*self.execution.capabilities, self.options[0])
                _issued_format_require(len({id(handle) for handle in handles}) == 5, "capabilities_invalid")
                for handle, selection, producer_ref in zip(handles,
                        (*self.execution.selections, self.options[1]), (*self.execution.producers, self.options[2])):
                    admitted = self.callback(self.parent._admission.current, handle)
                    selected = self.callback(selection.current, handle)
                    producer = self.callback(self.parent._result_port.current, producer_ref)
                    row = _binding(admitted), _selection(selected), _producer(producer)
                    _issued_format_require(row[0][6:8] == row[1][:2] and row[1][3][1] == "synthetic_native"
                        and row[2][0] == producer_ref, "authority_invalid")
                    facts.append(row)
                    originals.append((admitted, selected, producer))
                facts, originals = tuple(facts), tuple(originals)
                for i in range(4):
                    original, actual = self.execution.initial[i], facts[i]
                    _issued_format_require(actual[0][:8] == original[0][:8] and set(actual[0][8]) <= set(original[0][8])
                        and actual[1:] == original[1:] and originals[i][1] is self.execution.originals[i][1]
                        and originals[i][2] is self.execution.originals[i][2], "source_authority_changed")
                f = facts[4]
                _issued_format_require(all(originals[4][1].destination is not item[1].destination for item in originals[:4]),
                    "independent_formatter_selection_required")
                _issued_format_require(f[0][:6] == facts[1][0][:6] and set(f[2][2]) <= set(self.profiles)
                    and self.route.destination is originals[4][1].destination, "formatter_holder_invalid")
                if self.initial is None:
                    _issued_format_require(set(self.profiles) <= set(f[0][8]), "source_profiles_not_authorized")
                    self.initial, self.originals = facts, originals
                    self.initial_anchors = _nested_anchors((facts, originals))[1:]
                    self.profiles = tuple(sorted(set(self.profiles) | set(f[0][8]) | set(f[2][2])))
                _issued_format_require(_same_anchors(_nested_anchors((self.initial, self.originals))[1:],
                    self.initial_anchors), "initial_changed")
                for actual, original, current, retained in zip(facts, self.initial, originals, self.originals):
                    _issued_format_require(actual[0][:8] == original[0][:8] and set(actual[0][8]) <= set(original[0][8])
                        and actual[1:] == original[1:] and current[1] is retained[1] and current[2] is retained[2],
                        "authority_changed")
                receiver = _binding(self.callback(self.parent._admission.current, self.execution.receiver_port))
                _issued_format_require(receiver[:8] == self.execution.initial[2][0][:8]
                    and set(receiver[8]) <= set(self.execution.initial[2][0][8]), "receiver_changed")
                now = self.callback(self.parent._clock)
                _integer(now, minimum=0)
                allowed = {r.route_ref for r in self.catalog[0]}
                _issued_format_require(set(self.profiles) <= {p.profile_ref for p in self.catalog[1]}, "profile_missing")
                for profile in self.catalog[1]:
                    if profile.profile_ref in self.profiles:
                        _issued_format_require(profile.required_paths == shared_policy().PATHS, "profile_changed")
                        allowed.intersection_update(profile.allowed_routes)
                for i, route in (*enumerate(self.selected_routes), (2, self.execution.routes[2])):
                    _route_facts(route)
                    holder = self.initial[i][0]
                    _issued_format_require(route.route_ref in allowed and now < route.expires_at
                        and (route.consumer_ref, route.selection_id, route.selection_version) == (holder[3], holder[6], holder[7]),
                        "selected_expired_or_denied")
                    if route is self.selected_routes[i]:
                        _issued_format_require(self.originals[i][1].destination is route.destination, "destination_changed")
                for holder, _, _ in facts:
                    for path in shared_policy().PATHS:
                        _issued_format_require(any(route.route_ref in allowed and route.path == path
                            and (route.consumer_ref, route.selection_id, route.selection_version) == (holder[3], holder[6], holder[7])
                            and now < route.expires_at for route in self.catalog[0]), "coverage_missing")
                for route in self.catalog[0]:
                    actual = self.callback(self.parent._operator_transports[route.destination.transport_ref].resolve)
                    _issued_format_require(actual is route.destination, "destination_changed")
                for handle, payload, parents in ((self.authorization_content, self.authorization_bytes, ()),
                        (self.input_content, self.input_bytes, (self.source_result.result_content, self.authorization_content)),
                        (self.message_content, self.message_bytes, (self.input_content,))):
                    if handle is not None:
                        _issued_format_content(self.runtime, handle, payload, parents, self.profiles)
                self.runtime._check_operator()
                self.verify()
        finally:
            self.checking = False

    def callback(self, callback, *args):
        self.verify()
        _issued_format_require(self.stage_frame is self.entry and any(_same_method(callback, method)
            for _, _, method in self.methods), "callback_invalid")
        self.runtime._check_operator()
        self.depth += 1
        try:
            return callback(*args)
        except BaseException as error:
            import asyncio
            if isinstance(error, (InterruptedError, asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                if self.control_exception is None:
                    self.control_exception = error
            raise
        finally:
            self.depth -= 1
            if self.control_exception is None:
                try:
                    self.verify()
                    self.runtime._check_operator()
                except BaseException as error:
                    import asyncio
                    if isinstance(error, (InterruptedError, asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                        if self.control_exception is None:
                            self.control_exception = error
                    raise

    def source_active(self):
        self.access()
        frame = self.source_frame
        _issued_format_require(frame is not None and frame.f_code is _validate_issued_call_format_source.__code__
            and frame.f_locals.get("source") is self.parent and frame.f_locals.get("result") is self.source_result
            and frame.f_locals.get("view") is self and self.stage_frame is self.entry, "source_frame_invalid")
        current = sys._getframe()
        while current is not None and current is not frame:
            current = current.f_back
        _issued_format_require(current is frame, "source_frame_missing")
        return True

    @contextmanager
    def window(self):
        self.access()
        _issued_format_require(self.stage_frame is None and self.stage_index < 4, "capacity_denied")
        self.stage_index += 1
        self.stage_frame = self.entry
        try:
            self.checkpoint()
            yield
            self.checkpoint()
        finally:
            self.stage_frame = None

    def current(self, handle):
        self.verify()
        caller = sys._getframe(1)
        _issued_format_require(caller.f_code is _Admission.current.__code__ and caller.f_locals.get("adapter") is self.parent
            and self.stage_frame is self.entry and any(handle is h for h in (*self.execution.capabilities, self.options[0])),
            "admission_frame_invalid")
        frame = caller.f_back
        _issued_format_require(frame.f_code is type(self.runtime)._call.__code__
            and frame.f_locals.get("self") is self.runtime, "runtime_callback_frame_invalid")
        frame = frame.f_back
        _issued_format_require(frame.f_code is type(self.runtime)._current.__code__
            and frame.f_locals.get("self") is self.runtime and frame.f_locals.get("request_handle") is handle,
            "runtime_frame_invalid")
        while frame is not None and frame.f_code not in (type(self.runtime).admit.__code__, type(self.runtime).derive.__code__,
                type(self.runtime).derive_for_selection.__code__):
            frame = frame.f_back
        _issued_format_require(frame is not None and frame.f_locals.get("self") is self.runtime
            and frame.f_locals.get("request_handle") is self.options[0] and frame.f_back is self.entry
            and ((self.stage_index == 1 and frame.f_code in (type(self.runtime).admit.__code__,
                type(self.runtime).derive_for_selection.__code__)) or
                (self.stage_index == 3 and frame.f_code is type(self.runtime).derive.__code__)), "operation_invalid")
        _issued_format_require(self.pending_handle is None, "admission_reentry")
        self.pending_handle, self.current_row = handle, None
        try:
            _validate_issued_call_format_source(self.parent, self.source_result, self)
            return self.current_row
        finally:
            self.pending_handle = self.current_row = None

    def around(self, callback):
        self.access()
        caller = sys._getframe(1)
        _issued_format_require(caller.f_code is NativeMainPayloadAdapter._around.__code__
            and caller.f_locals.get("self") is self.parent and self.runtime._busy
            and self.stage_frame is self.entry and _same_method(callback, _method(self.parent._clock)), "clock_invalid")
        frame = caller.f_back
        _issued_format_require(frame.f_code is self.runtime._clock.__code__
            and frame.f_back.f_code is type(self.runtime)._call.__code__
            and frame.f_back.f_locals.get("self") is self.runtime, "clock_frame_invalid")
        return self.callback(callback)

    def resolve(self, wrapper):
        self.verify()
        caller = sys._getframe(1)
        _issued_format_require(caller.f_code is _Transport.resolve.__code__ and caller.f_locals.get("self") is wrapper
            and any(wrapper is w for _, w in self.catalog[3]) and self.runtime._busy
            and self.stage_frame is self.entry, "resolve_invalid")
        frame = caller.f_back
        _issued_format_require(frame.f_code is type(self.runtime)._call.__code__
            and frame.f_locals.get("self") is self.runtime
            and frame.f_back.f_code is type(self.runtime)._coverage.__code__
            and frame.f_back.f_locals.get("self") is self.runtime, "resolve_frame_invalid")
        actual = self.callback(wrapper.port.resolve)
        _issued_format_require(any(wrapper.port is self.parent._operator_transports[r.destination.transport_ref]
            and actual is r.destination for r in self.catalog[0]), "resolve_destination_changed")
        return actual

    def bind(self):
        self.entry = sys._getframe()
        try:
            with self.window():
                call = self.execution.call_value
                authorization = dict(format_mode="delegate_result_v1", source_call=call,
                    formatter_holder=self.initial[4][0], selection=self.initial[4][1],
                    producer_policy=self.initial[4][2], main_route_ref=self.options[3])
                self.authorization_bytes = _json(authorization)
                _issued_format_require(len(self.authorization_bytes) <= _MAX_NATIVE_RESULT, "capacity_denied")
                self.authorization_content = self.runtime.admit(self.options[0], self.authorization_bytes)
                self.input_bytes = _json(dict(call=call, content=self.source_result.aggregate,
                    effect_disposition=dict(present=False), name=self.source_result.observation.function_name,
                    tool_call_id=self.source_result.observation.call_id))
                _issued_format_require(len(self.input_bytes) <= _MAX_NATIVE_RESULT
                    and len(self.authorization_bytes) + len(self.input_bytes) <= 786432, "capacity_denied")
                self.input_content = self.runtime.derive_for_selection(self.options[0],
                    (self.source_result.result_content, self.authorization_content), self.input_bytes, path="tool_result",
                    parent_requests=(self.execution.options[0], self.options[0]))
                self.binding = NativeIssuedCallFormatBinding(self.token, self.parent, self.source_result, self.options[0],
                    self.originals[4][1], self.originals[4][2], self.authorization_bytes, self.authorization_content,
                    self.input_bytes, self.input_content, (self.options, self.catalog_anchors, self.initial_anchors,
                        self.source_snapshots, self.graph_snapshots, self.methods, self.wall_time), self.thread)
                self.registered_binding = self.binding
                self.binding_anchor = _row(self.binding, NativeIssuedCallFormatBinding)
                self.phase = self.phase_anchor = "bound"
            return self.token
        finally:
            self.active_call = self.call_anchor = self.entry = self.public = None

    def run(self, authorization, public):
        _issued_format_require(self.phase == self.phase_anchor == "bound" and not self.attempted
            and type(authorization) is NativeIssuedCallFormatInput and authorization is self.token
            and threading.get_ident() == self.thread and self.depth == 0
            and current_context() in (None, self.parent._bound_context), "attempt_invalid")
        self.attempted = True
        self.entry, self.public = sys._getframe(), public
        self.active_call = self.call_anchor = object()
        self.phase = self.phase_anchor = "running"
        try:
            with self.window():
                pass
            with self.window():
                _issued_format_require(not self.partial_results, "partial_capacity_denied")
                self.partial_results = ((None, None, None),)
                self.partial_anchors = _nested_anchors(self.partial_results)[1:]
                self.depth += 1
                try:
                    message = self.formatter(self.source_result.observation.function_name, self.source_result.aggregate,
                        self.source_result.observation.call_id)
                    anchors = _nested_anchors(message)
                    self.partial_results = ((message, anchors, None),)
                    self.partial_anchors = _nested_anchors(self.partial_results)[1:]
                    payload = _issued_format_bytes(message)
                    self.partial_results = ((message, anchors, payload),)
                    self.partial_anchors = _nested_anchors(self.partial_results)[1:]
                except BaseException as error:
                    import asyncio
                    if isinstance(error, (InterruptedError, asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                        if self.control_exception is None:
                            self.control_exception = error
                    raise
                finally:
                    self.depth -= 1
                self.verify()
                self.runtime._check_operator()
                self.checkpoint()
                _issued_format_require(type(message) is dict and tuple(message) == (
                    "role", "name", "tool_name", "content", "tool_call_id", "timestamp")
                    and message["role"] == "tool"
                    and message["name"] is message["tool_name"] is self.source_result.observation.function_name
                    and message["content"] is self.source_result.aggregate
                    and message["tool_call_id"] is self.source_result.observation.call_id
                    and type(message["timestamp"]) is float and math.isfinite(message["timestamp"]), "message_invalid")
                for value in message.values():
                    if type(value) is str:
                        _text(value)
                _issued_format_require(len(payload) <= _MAX_NATIVE_RESULT
                    and len(payload) + len(self.authorization_bytes) + len(self.input_bytes) <= 786432, "capacity_denied")
                self.message, self.message_bytes, self.message_anchors = message, payload, anchors
                self.message_content = self.runtime.derive(self.options[0], (self.input_content,), payload, path="tool_result")
            with self.window():
                result = NativeIssuedCallFormatResult(self.binding, self.source_result, self.message, self.message_bytes,
                    self.message_content, self.message_anchors)
            self.records = ((result, _row(result, NativeIssuedCallFormatResult)),)
            self.phase = self.phase_anchor = "released"
            return result
        finally:
            self.active_call = self.call_anchor = self.entry = self.public = None

    def scope_access(self):
        _issued_format_require(self.depth == 0 and threading.get_ident() == self.thread, "reentry_or_thread")
        _issued_format_require(_issued_call_formats.get(self.parent) is self and self.parent._issued_call_format_owner is self
            and type(self.records) is tuple and len(self.records) <= 1, "records_changed")
        for result, anchor in self.records:
            _issued_format_require(_same_anchors(_row(result, NativeIssuedCallFormatResult), anchor)
                and result.binding is self.registered_binding and result.source_result is self.source_result
                and _same_anchors(_nested_anchors(result.message), result.anchors)
                and _issued_format_bytes(result.message) == result.message_bytes, "record_changed")


from agent import tool_dispatch_helpers as _format_helpers
from agent import message_metadata as _format_timestamp_module

_ISSUED_FORMAT_METHODS = ("__init__", "access", "verify", "checkpoint", "callback", "source_active", "window", "current",
    "around", "resolve", "bind", "run", "scope_access")
_ISSUED_FORMAT_CODE = tuple((obj, name, getattr(obj, name), getattr(obj, name).__code__)
    for obj, names in (
        (_IssuedCallFormat, _ISSUED_FORMAT_METHODS),
        (NativeMainPayloadAdapter, ("bind_issued_call_format", "run_issued_call_format")),
        (sys.modules[__name__], ("_validate_issued_call_format_source", "_issued_call_format_window",
            "_issued_format_require", "_issued_format_bytes", "_issued_format_content", "_issued_format_constructed",
            "_row", "_binding", "_selection", "_producer", "_json", "_text", "_method", "_same_method",
            "_nested_anchors", "_same_anchors", "_constructed_state", "_namespace", "_native_result_anchors",
            "_native_result_bytes", "_tool_probe_request", "_identifier", "_integer", "_route_facts", "_destination_value")),
        (_format_helpers, ("make_tool_result_message", "_normalize_tool_call_id", "_is_untrusted_tool",
            "_detect_upstream_elision", "_maybe_append_elision_notice", "_tool_output_risk_metadata",
            "_neutralize_delimiters", "_maybe_wrap_untrusted", "stamp_message_timestamp")),
        (_format_timestamp_module, ("stamp_message_timestamp",)),
    ) for name in names)
_ISSUED_FORMAT_CODE += ((_IssuedCallFormat.window, "__wrapped__", _IssuedCallFormat.window.__wrapped__,
    _IssuedCallFormat.window.__wrapped__.__code__),)

_ISSUED_FORMAT_GETTER = NativeMainPayloadAdapter.issued_call_format_results.fget
_ISSUED_FORMAT_CODE += ((NativeMainPayloadAdapter.issued_call_format_results, "fget", _ISSUED_FORMAT_GETTER,
    _ISSUED_FORMAT_GETTER.__code__),)
