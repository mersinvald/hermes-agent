"""Run one controlled episode through the original component entries."""

import json
import sys
from dataclasses import dataclass

from agent import conversation_loop
from agent import native_main_payload as native

from agent.content_policy import (
    NativeContentContext, NativeContentPolicyDenied, content_policy_scope, current_context,
)
from agent.native_main_payload import (
    NativeMainPayloadAdapter, NativeSelectionPort, NativeIssuedCallContinuationCompletion,
    bind_issued_call_continuation, run_issued_call_continuation, issued_call_continuation_results,
)
from run_agent import AIAgent


@dataclass(frozen=True, slots=True)
class NativeIssuedCallEpisodeProbe:
    system_message: str
    user_message: str
    task_id: str
    expected_arguments_json: str


@dataclass(frozen=True, slots=True)
class NativeIssuedCallEpisodeExecution:
    execution_request: object
    execution_selection: NativeSelectionPort
    execution_producer_ref: str
    execution_main_route_ref: str
    receiver_request: object
    receiver_selection: NativeSelectionPort
    receiver_producer_ref: str
    child_request: object
    child_selection: NativeSelectionPort
    child_producer_ref: str
    delegate_route_ref: str
    child_main_route_ref: str
    child_session_id: str
    child_task_id: str


@dataclass(frozen=True, slots=True)
class NativeIssuedCallEpisodeFormatter:
    formatter_request: object
    formatter_selection: NativeSelectionPort
    formatter_producer_ref: str
    formatter_main_route_ref: str


@dataclass(frozen=True, slots=True)
class NativeIssuedCallEpisodeConsumer:
    consumer_context: NativeContentContext
    consumer_selection: NativeSelectionPort
    consumer_producer_ref: str
    consumer_main_route_ref: str
    target_model: str
    session_id: str


def run_controlled_issued_call_episode(
    source_agent: AIAgent,
    source_adapter: NativeMainPayloadAdapter,
    *,
    probe: NativeIssuedCallEpisodeProbe,
    execution: NativeIssuedCallEpisodeExecution,
    formatter: NativeIssuedCallEpisodeFormatter,
    consumer: NativeIssuedCallEpisodeConsumer,
) -> NativeIssuedCallContinuationCompletion:
    """Return the original completion after the original scope validators pass."""
    completion, _ = _execute_issued_call_episode(
        source_agent, source_adapter, probe=probe, execution=execution,
        formatter=formatter, consumer=consumer, project_conversation=False,
    )
    return completion


def _execute_issued_call_episode(
    source_agent: AIAgent,
    source_adapter: NativeMainPayloadAdapter,
    *,
    probe: NativeIssuedCallEpisodeProbe,
    execution: NativeIssuedCallEpisodeExecution,
    formatter: NativeIssuedCallEpisodeFormatter,
    consumer: NativeIssuedCallEpisodeConsumer,
    project_conversation: bool,
) -> tuple[NativeIssuedCallContinuationCompletion, dict | None]:
    """Run the original host sequence and its optional public projection."""
    anchors = _PUBLIC_EPISODE_CODE if project_conversation else None

    def checkpoint():
        if anchors is None:
            return
        row = anchors[-1]
        checker = row[2]
        if (_check_public_episode_code is not checker
                or checker.__code__ is not row[3]
                or checker.__globals__ is not row[4]
                or checker.__defaults__ is not row[5]
                or checker.__kwdefaults__ is not row[6]):
            raise NativeContentPolicyDenied("controlled_native_payload_denied")
        checker(anchors)

    checkpoint()
    if (type(source_agent) is not AIAgent
            or type(source_adapter) is not NativeMainPayloadAdapter
            or type(probe) is not NativeIssuedCallEpisodeProbe
            or type(execution) is not NativeIssuedCallEpisodeExecution
            or type(formatter) is not NativeIssuedCallEpisodeFormatter
            or type(consumer) is not NativeIssuedCallEpisodeConsumer):
        raise NativeContentPolicyDenied("issued_call_episode_input_type")
    if (anchors[8][2] if project_conversation else current_context)() is not None:
        raise NativeContentPolicyDenied("issued_call_episode_scope_active")
    agent, adapter = source_agent, source_adapter
    parent_context = adapter._bound_context
    probe_refs = (
        probe.system_message,
        probe.user_message,
        probe.task_id,
        probe.expected_arguments_json,
    )
    execution_refs = (
        execution.execution_request,
        execution.execution_selection,
        execution.execution_producer_ref,
        execution.execution_main_route_ref,
        execution.receiver_request,
        execution.receiver_selection,
        execution.receiver_producer_ref,
        execution.child_request,
        execution.child_selection,
        execution.child_producer_ref,
        execution.delegate_route_ref,
        execution.child_main_route_ref,
        execution.child_session_id,
        execution.child_task_id,
    )
    formatter_refs = (
        formatter.formatter_request,
        formatter.formatter_selection,
        formatter.formatter_producer_ref,
        formatter.formatter_main_route_ref,
    )
    consumer_refs = (
        consumer.consumer_context,
        consumer.consumer_selection,
        consumer.consumer_producer_ref,
        consumer.consumer_main_route_ref,
        consumer.target_model,
        consumer.session_id,
    )
    source_session_id = agent.session_id if project_conversation else None
    projection = None
    checkpoint()
    continuation = (anchors[11][2] if project_conversation else bind_issued_call_continuation)(adapter,
        consumer_context=consumer_refs[0],
        consumer_selection=consumer_refs[1],
        consumer_producer_ref=consumer_refs[2],
        consumer_main_route_ref=consumer_refs[3],
        target_model=consumer_refs[4],
        session_id=consumer_refs[5],
    )
    checkpoint()
    with (anchors[9][2] if project_conversation else content_policy_scope)(parent_context):
        checkpoint()
        token = (anchors[15][2].__get__(adapter) if project_conversation else adapter.bind_tool_call_probe)(agent,
            system_message=probe_refs[0],
            user_message=probe_refs[1],
            task_id=probe_refs[2],
            expected_arguments_json=probe_refs[3],
        )
        checkpoint()
        (anchors[16][2].__get__(adapter) if project_conversation else adapter.enable_tool_call_assistant_source)(agent, token)
        checkpoint()
        observation = (anchors[17][2].__get__(adapter) if project_conversation else adapter.run_tool_call_probe)(agent, token)
        checkpoint()
        (anchors[14][2].__get__(agent) if project_conversation else agent.close)()
        checkpoint()
        source, = (anchors[25][2](adapter) if project_conversation
                   else adapter.tool_call_assistant_sources)
        checkpoint()
        issued_token = (anchors[18][2].__get__(adapter) if project_conversation else adapter.bind_issued_call_execution)(observation,
            execution_request=execution_refs[0],
            execution_selection=execution_refs[1],
            execution_producer_ref=execution_refs[2],
            execution_main_route_ref=execution_refs[3],
            receiver_request=execution_refs[4],
            receiver_selection=execution_refs[5],
            receiver_producer_ref=execution_refs[6],
            child_request=execution_refs[7],
            child_selection=execution_refs[8],
            child_producer_ref=execution_refs[9],
            delegate_route_ref=execution_refs[10],
            child_main_route_ref=execution_refs[11],
            child_session_id=execution_refs[12],
            child_task_id=execution_refs[13],
        )
        checkpoint()
        issued = (anchors[19][2].__get__(adapter) if project_conversation else adapter.run_issued_call_execution)(issued_token)
        checkpoint()
        format_token = (anchors[20][2].__get__(adapter) if project_conversation else adapter.bind_issued_call_format)(issued,
            formatter_request=formatter_refs[0],
            formatter_selection=formatter_refs[1],
            formatter_producer_ref=formatter_refs[2],
            formatter_main_route_ref=formatter_refs[3],
        )
        checkpoint()
        formatted = (anchors[21][2].__get__(adapter) if project_conversation else adapter.run_issued_call_format)(format_token)
        checkpoint()
        tool = (anchors[22][2].__get__(adapter) if project_conversation else adapter.retain_issued_call_wire)(formatted)
        checkpoint()
        assistant = (anchors[23][2].__get__(adapter) if project_conversation else adapter.retain_tool_call_assistant_wire)(source)
        checkpoint()
        pair = (anchors[24][2].__get__(adapter) if project_conversation else adapter.retain_issued_call_pair)(assistant, tool)
        checkpoint()
    checkpoint()
    with (anchors[9][2] if project_conversation else content_policy_scope)(consumer_refs[0]):
        checkpoint()
        result = (anchors[12][2] if project_conversation else run_issued_call_continuation)(continuation, pair)
        checkpoint()
        records = (anchors[13][2] if project_conversation else issued_call_continuation_results)(continuation)
        checkpoint()
        if len(records) != 1 or records[0] is not result:
            raise NativeContentPolicyDenied("issued_call_episode_result_identity")
        checkpoint()
        if project_conversation:
            checkpoint()
            projection = anchors[5][2](result, source, source_session_id)
            checkpoint()
    return result.completion, projection


@dataclass(frozen=True, slots=True)
class NativeIssuedCallConversationPlan:
    source_adapter: NativeMainPayloadAdapter
    probe: NativeIssuedCallEpisodeProbe
    execution: NativeIssuedCallEpisodeExecution
    formatter: NativeIssuedCallEpisodeFormatter
    consumer: NativeIssuedCallEpisodeConsumer


def _try_controlled_issued_call_conversation(
    agent, user_message, system_message=None, conversation_history=None,
    task_id=None, stream_callback=None, persist_user_message=None,
    persist_user_timestamp=None, persist_user_display_kind=None,
    persist_user_display_metadata=None, moa_config=None, *,
    controlled_issued_call=None,
) -> tuple[bool, dict | None]:
    """Select one episode without ordinary conversation preprocessing."""
    if controlled_issued_call is None:
        return False, None
    authenticated = None
    try:
        anchors = _PUBLIC_EPISODE_CODE
        row = anchors[-1]
        checker = row[2]
        if (_check_public_episode_code is not checker
                or checker.__code__ is not row[3]
                or checker.__globals__ is not row[4]
                or checker.__defaults__ is not row[5]
                or checker.__kwdefaults__ is not row[6]):
            raise NativeContentPolicyDenied("controlled_native_payload_denied")
        checker(anchors)
        if type(agent) is AIAgent:
            binding = native._owned_constructed_binding(agent)
            if (type(binding) is native._ConstructedMain
                    and type(binding.adapter) is NativeMainPayloadAdapter
                    and binding.adapter in native._adapters):
                authenticated = binding
        plan = controlled_issued_call
        if type(plan) is not NativeIssuedCallConversationPlan:
            raise NativeContentPolicyDenied("controlled_native_payload_denied")
        adapter, probe, execution, formatter, consumer = (
            plan.source_adapter, plan.probe, plan.execution, plan.formatter, plan.consumer,
        )
        if (type(agent) is not AIAgent
                or type(adapter) is not NativeMainPayloadAdapter
                or type(probe) is not NativeIssuedCallEpisodeProbe
                or type(execution) is not NativeIssuedCallEpisodeExecution
                or type(formatter) is not NativeIssuedCallEpisodeFormatter
                or type(consumer) is not NativeIssuedCallEpisodeConsumer):
            raise NativeContentPolicyDenied("controlled_native_payload_denied")
        namespace = object.__getattribute__(agent, "__dict__")
        if (authenticated is None or authenticated.adapter is not adapter
                or adapter._constructed is not authenticated
                or adapter._turn_agent is not agent
                or namespace.get("_native_controlled_binding") is not authenticated
                or namespace.get("_native_content_policy_context") is not adapter._bound_context
                or adapter._context is not adapter._bound_context):
            raise NativeContentPolicyDenied("controlled_native_payload_denied")
        if (type(user_message) is not str or user_message is not probe.user_message
                or type(system_message) is not str or system_message is not probe.system_message
                or type(task_id) is not str or task_id is not probe.task_id
                or any(value is not None for value in (
                    conversation_history, stream_callback, persist_user_message,
                    persist_user_timestamp, persist_user_display_kind,
                    persist_user_display_metadata, moa_config,
                )) or current_context() is not None):
            raise NativeContentPolicyDenied("controlled_native_payload_denied")
        checker(anchors)
        _, result = anchors[4][2](
            agent, adapter, probe=probe, execution=execution,
            formatter=formatter, consumer=consumer, project_conversation=True,
        )
        checker(anchors)
        return True, result
    except BaseException:
        if authenticated is not None:
            native._invalidate_constructed_attempt(agent, authenticated.adapter)
        raise


def _project_issued_call_conversation(result, source, source_session_id: str) -> dict:
    """Copy authenticated parent results without a policy operation."""
    totals = []
    for usage in (source.response.usage, result.completion.usage):
        counts = tuple(getattr(usage, name) for name in (
            "prompt_tokens", "completion_tokens", "total_tokens",
        ))
        if (any(type(value) is not int or not 0 <= value <= native._MAX_INT for value in counts)
                or counts[0] + counts[1] != counts[2]):
            raise NativeContentPolicyDenied("controlled_native_payload_denied")
        totals.append(counts)
    prompt, completion, total = (sum(values) for values in zip(*totals))
    if any(value > native._MAX_INT for value in (prompt, completion, total)):
        raise NativeContentPolicyDenied("controlled_native_payload_denied")
    messages = _PUBLIC_EPISODE_CODE[6][2](result.request_bytes)["messages"]
    messages.append({"role": "assistant", "content": result.completion.final_response})
    projected = dict(
        final_response=result.completion.final_response, last_reasoning=None,
        messages=messages, api_calls=2, completed=True,
        turn_exit_reason="text_response(finish_reason=stop)", failed=False,
        partial=False, interrupted=False, response_transformed=False,
        pre_transform_response=None, response_previewed=False,
        model=result.binding.selection.destination.model_ref,
        provider="synthetic_native", base_url="",
        input_tokens=prompt, output_tokens=completion, cache_read_tokens=0,
        cache_write_tokens=0, reasoning_tokens=0, prompt_tokens=prompt,
        completion_tokens=completion, total_tokens=total,
        last_prompt_tokens=totals[1][0], estimated_cost_usd=None,
        cost_status="unknown", cost_source=None, service_tier=None,
        session_id=source_session_id,
    )
    payload = _PUBLIC_EPISODE_CODE[7][2](projected, sort_keys=True, ensure_ascii=True, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    if len(payload) > native._MAX_NATIVE_RESULT:
        raise NativeContentPolicyDenied("controlled_native_payload_denied")
    return projected


def _check_public_episode_code(anchors):
    """Check local callable identity before any captured callable runs."""
    if _PUBLIC_EPISODE_CODE is not anchors:
        raise NativeContentPolicyDenied("controlled_native_payload_denied")
    for obj, name, function, code, namespace, defaults, kwdefaults, keywords in anchors:
        if (getattr(obj, name) is not function or function.__code__ is not code
                or function.__globals__ is not namespace or function.__defaults__ is not defaults
                or function.__kwdefaults__ is not kwdefaults
                or (kwdefaults is not None and (
                    tuple(kwdefaults) != tuple(key for key, _ in keywords)
                    or any(kwdefaults[key] is not value for key, value in keywords)))):
            raise NativeContentPolicyDenied("controlled_native_payload_denied")


_PUBLIC_EPISODE_CODE = tuple(
    (obj, name, function, function.__code__, function.__globals__, function.__defaults__,
     function.__kwdefaults__, tuple((function.__kwdefaults__ or {}).items()))
    for obj, name in (
        (AIAgent, "run_conversation"),
        (conversation_loop, "run_conversation"),
        (sys.modules[__name__], "run_controlled_issued_call_episode"),
        (sys.modules[__name__], "_try_controlled_issued_call_conversation"),
        (sys.modules[__name__], "_execute_issued_call_episode"),
        (sys.modules[__name__], "_project_issued_call_conversation"),
        (json, "loads"), (json, "dumps"),
        (sys.modules[__name__], "current_context"),
        (sys.modules[__name__], "content_policy_scope"),
        (content_policy_scope, "__wrapped__"),
        (sys.modules[__name__], "bind_issued_call_continuation"),
        (sys.modules[__name__], "run_issued_call_continuation"),
        (sys.modules[__name__], "issued_call_continuation_results"),
        (AIAgent, "close"),
        *((NativeMainPayloadAdapter, name) for name in (
            "bind_tool_call_probe", "enable_tool_call_assistant_source", "run_tool_call_probe",
            "bind_issued_call_execution", "run_issued_call_execution", "bind_issued_call_format",
            "run_issued_call_format", "retain_issued_call_wire", "retain_tool_call_assistant_wire",
            "retain_issued_call_pair",
        )),
        (NativeMainPayloadAdapter.tool_call_assistant_sources, "fget"),
        (sys.modules[__name__], "_check_public_episode_code"),
    )
    for function in (getattr(obj, name),)
)
