# Controlled native text-turn consumer contract

This implementation follows the approved P10.10 design and the P10.11 task clarifications.
The proposal label describes the consumer API for coordinator acceptance.
It does not authorize private processing or deployment.

The operator supplies a genuine current `NativeContentContext`, admission capability, selection port, producer policy, routes, transports, and clock.
All eight shared paths require truthful declared operations and current qualification.
Delegate and A2A qualification require authenticated receiver bindings.
No constructor option supplies these bindings or creates default routes.

Within `content_policy_scope(context)`, construct one exact `AIAgent` with the registered, unconsumed adapter:

```python
agent = AIAgent(
    controlled_main=adapter,
    provider="synthetic_native",
    api_mode="chat_completions",
    model=selected_model,
    session_id=explicit_session_id,
    enabled_toolsets=[],
    skip_memory=True,
    skip_context_files=True,
    skip_background_review=True,
    quiet_mode=True,
    max_tokens=None,
)
first = agent.run_conversation(user, system, task_id=execution_ref)
```

Use the existing module entrypoint with the same agent for another turn:

```python
from agent.conversation_loop import run_conversation

second = run_conversation(
    agent, next_user, system,
    conversation_history=first["messages"], task_id=execution_ref,
)
agent.close()
```

The model and session ID require bounded exact strings.
`max_tokens` permits `None` or an exact integer from 1 through 4096.
Every other constructor option must retain its exact existing default.
The constructor validates forwarder-only `tool_delay` before the warning callback.
It creates inert native state directly without configuration, clients, discovery, or persistence.

Normal module imports remain outside this qualification and can load configuration or tool modules.

Both existing entrypoints call `_try_controlled_text_turn` before their ordinary guards and lifecycle callbacks.
The helper retains the exact original argument objects and requires the current execution reference.
It accepts only text input, one stable optional system prefix, and original retained history.
Later turns require the exact prior returned `messages` list.
Streaming, MoA, persistence overrides, timestamps, and display metadata must remain `None`.

The helper uses `admit_turn`, actual `build_turn_context`, `prepared_request`, actual `interruptible_api_call`, and the existing dispatcher.
The prologue seals its output once.
The controlled caller sends exactly once on the bound thread.
The result keeps the accepted assistant text, including empty text.
No finalizer, transform, retry, nudge, pricing, persistence, or second model call follows.

`_complete_controlled_turn` accepts the latest consumed prepared frame and its exact original response.
It derives the complete canonical result through the shared public `derive` method before release.
Parents include raw, prepared, materialized request, invocation request/result, and all earlier complete result handles.
Later raw, prepared, and request proofs also retain earlier complete result handles.
The adapter retains bytes, handles, returned dictionaries, nested object anchors, and invocation correspondence together.
Any later change to retained results, responses, usage, labels, state, attachments, or methods denies reuse.
`NativeMainInvocation` and all existing public adapter signatures remain unchanged.

The result uses the approved native fields in `native-api.json`.
Token counters are exact cumulative integers bounded by `2**53 - 1`.
Every accepted invocation contributes once.
Cache and reasoning counters remain zero, and cost remains unknown.

The implementation checks sums and the complete 262144-byte result bound before release.
Overflow preserves actual sends and accepted records, with no partial result.
Tests inject smaller integer and result bounds to exercise these guards.
Normal response bounds and eight turns cannot naturally reach the integer limit.

Close recognizes valid, malformed, partial, denied, canceled, and closed controlled attempts before generic cleanup.
It invalidates owned pending state and remains idempotent.
Accepted records remain inspectable, and every later turn denies.
Cancellation remains pending and uses the accepted fixed interruption error.
Process-control exceptions invalidate pending proof and propagate unchanged.

The local bound is eight turns with one pending frame.
Shared limits remain 128 handles and 8 MiB.
There is no in-flight preemption, deadline, watchdog, or sensitive cross-thread dispatch claim.
Full P10/P06, auxiliary native success, real providers, egress, durable restoration, source/financial qualification, and private processing remain open.
The coordinator owns acceptance and merge.
