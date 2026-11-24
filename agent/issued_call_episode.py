"""Run one controlled episode through the original component entries."""

from dataclasses import dataclass

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
    if (type(source_agent) is not AIAgent
            or type(source_adapter) is not NativeMainPayloadAdapter
            or type(probe) is not NativeIssuedCallEpisodeProbe
            or type(execution) is not NativeIssuedCallEpisodeExecution
            or type(formatter) is not NativeIssuedCallEpisodeFormatter
            or type(consumer) is not NativeIssuedCallEpisodeConsumer):
        raise NativeContentPolicyDenied("issued_call_episode_input_type")
    if current_context() is not None:
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
    continuation = bind_issued_call_continuation(adapter,
        consumer_context=consumer_refs[0],
        consumer_selection=consumer_refs[1],
        consumer_producer_ref=consumer_refs[2],
        consumer_main_route_ref=consumer_refs[3],
        target_model=consumer_refs[4],
        session_id=consumer_refs[5],
    )
    with content_policy_scope(parent_context):
        token = adapter.bind_tool_call_probe(agent,
            system_message=probe_refs[0],
            user_message=probe_refs[1],
            task_id=probe_refs[2],
            expected_arguments_json=probe_refs[3],
        )
        adapter.enable_tool_call_assistant_source(agent, token)
        observation = adapter.run_tool_call_probe(agent, token)
        agent.close()
        source, = adapter.tool_call_assistant_sources
        issued_token = adapter.bind_issued_call_execution(observation,
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
        issued = adapter.run_issued_call_execution(issued_token)
        format_token = adapter.bind_issued_call_format(issued,
            formatter_request=formatter_refs[0],
            formatter_selection=formatter_refs[1],
            formatter_producer_ref=formatter_refs[2],
            formatter_main_route_ref=formatter_refs[3],
        )
        formatted = adapter.run_issued_call_format(format_token)
        tool = adapter.retain_issued_call_wire(formatted)
        assistant = adapter.retain_tool_call_assistant_wire(source)
        pair = adapter.retain_issued_call_pair(assistant, tool)
    with content_policy_scope(consumer_refs[0]):
        result = run_issued_call_continuation(continuation, pair)
        records = issued_call_continuation_results(continuation)
        if len(records) != 1 or records[0] is not result:
            raise NativeContentPolicyDenied("issued_call_episode_result_identity")
    return result.completion
