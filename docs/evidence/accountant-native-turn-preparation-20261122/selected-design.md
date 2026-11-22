# Proposed controlled turn preparation

Date: 2026-11-22. Status: proposal for root selection, without implementation authority.
Native inputs are exact candidate `f9c0f8745c504f2a0fb0d9f443c7f25576f8f6e3` and corrected source `8767a634aea9c3b47aef3e08860cc29d0a528738`.
[Source pins](source-pins.json) bind every inspected file. Line references below refer to that candidate.

## Component and stopping boundary

Add complete-byte controlled preparation to the existing `build_turn_context()` prologue.
Keep the existing `TurnContext` representation and accepted `interruptible_api_call()` execution path.
The component ends when a checked prologue result and its exact main request are available.
Focused composition checks then pass that request through the actual accepted caller.
No new agent, conversation loop, transport, runner, or generic authorization framework belongs here.

Complete `AIAgent.run_conversation()` startup remains refused in this component.
This boundary avoids admitting the large loop and finalizer through a local preparation exception.
Root must select their later integration separately. This proposal creates no task or implementation change.

## Source diagnosis

| Existing boundary | Current behavior and required treatment |
| --- | --- |
| `run_agent.py:8687`, `AIAgent.run_conversation()` | Its guard at line 8707 refuses before background-review cancellation, Relay admission, durable lease machinery, or task instrumentation. Keep this first caller refusal. |
| `agent/conversation_loop.py:1899`, `run_conversation()` | Its guard at line 1941 refuses before MoA decoding, credential refresh, and prologue entry. Keep this refusal. |
| `agent/turn_context.py:563`, `build_turn_context()` | Its guard at line 592 refuses before stdio setup, session recovery, sanitation, logs, or history copying. Add only a token-bound controlled preparation branch here. |
| `agent/turn_context.py:266`, title helper | Its title guard precedes checks for database, session, or platform. Controlled preparation must omit this call entirely. |
| `agent/chat_completion_helpers.py:1912`, kwargs builder | Generic kwargs construction consults transports, provider profiles, cache scope, and overrides. It does not supply the accepted minimal request contract automatically. Keep it outside this component. |
| `agent/conversation_loop.py:2322` through actual send | Message repair, context selection, normalization, middleware, Relay, and auxiliary work remain unqualified for controlled data. Do not enter this section. |
| `agent/turn_finalizer.py:129`, `finalize_turn()` | Persistence, output transforms, hooks, memory sync, and background reviews remain outside the accepted controlled boundary. Keep complete conversation startup refused. |

Changing every generic `Path.main` guard would admit these operations without complete local content proof.
Calling `validate_native_scope()` alone would prove current adapter facts but would not bind the complete prologue payload.
The proposed exception must require both an exact adapter-owned input token and complete registered input bytes.
Every other generic main guard and native sibling refusal remains mandatory.

## Exact native interface

The proposed interface lives on the existing `NativeMainPayloadAdapter`.
The two token types are opaque, with empty slots and no public constructor authority.
Only their exact registered identities work. Copying, serialization, labels, or equal data grant no authority.

```python
class NativeTurnInput:
    __slots__ = ()
class NativePreparedTurn:
    __slots__ = ()
# New NativeMainPayloadAdapter methods:
def admit_turn(
    self, agent: object, *, user_message: str,
    system_message: str | None,
    conversation_history: list[dict[str, str]] | None,
    task_id: str,
) -> NativeTurnInput: ...
def seal_turn(
    self, agent: object, turn: NativeTurnInput, *,
    turn_context: TurnContext,
) -> NativePreparedTurn: ...
def prepared_request(
    self, agent: object, prepared: NativePreparedTurn,
) -> dict: ...
```

Append `controlled_input: NativeTurnInput | None = None` to the keyword-only parameters of existing `build_turn_context()`.
Append `controlled_prepared: NativePreparedTurn | None = None` to `TurnContext` with an ordinary default of `None`.
Keep the existing caller and dispatcher signatures unchanged.
Use deferred annotations to avoid a module import cycle.

`admit_turn()` captures the actual agent, current context, adapter, thread, raw arguments, and supported option snapshot.
Its private record retains complete canonical input bytes and the shared input handle.
Capture `model`, `max_tokens`, `temperature`, and `stream=False` from the supported native option snapshot.
Reject all other request overrides before preparation.

`seal_turn()` accepts only the matching pending input and the exact complete output from the controlled prologue.
It retains the output bytes, output object anchors, derived handle, and one pending prepared token.
`prepared_request()` revalidates these anchors and returns a fresh minimal request dictionary.
Changed dictionaries or prologue results must deny at dispatch.

Existing adapter dispatch checks its pending prepared token when preparation preceded dispatch.
It requires exact canonical request equality, matching current agent/context, and the same retained source and result identities.
It consumes the prepared token once, while retaining its lineage anchors.
The existing direct caller contract remains unchanged without preparation.
Its existing shape, history, producer, and result-parent checks still apply.

## Complete input and local lineage

Boundary shape validation and canonical serialization precede text processing.
Allow exact built-in strings, exact lists and dictionaries, and fixed keys only.
Use the accepted canonical JSON rules, including ASCII escapes for raw surrogate code points.
Reject custom conversion hooks, unsupported metadata, multimodal content, tool rows, and unknown fields before sanitation.

The input record includes complete user text, optional system text, complete history, task identity, and every supported request option.
Limit each raw and prepared record to 262,144 canonical bytes.
Keep the existing 32-message and 16,384-byte text limits for main requests.
Raw user text can contain surrogates within that raw-record bound. System and retained history text must satisfy accepted text rules.
The shared 128-handle and 8 MiB retention limits remain unchanged.

A digest can supplement this record but cannot replace any text or registered handle.

The first turn accepts no history. It can have one explicit invented system message.
Later history must exactly match the previous accepted request messages and original retained response text.
Keep the system message separate when comparing the native history representation.
Plain history copies require this correspondence to the same registered results. Original response objects remain retained identity anchors.

Check every prior `NativeMainInvocation`, original `NativeMainResponse`, producer policy, and shared result handle before any text processing.
Arbitrary imported history, equal copied response objects, and changed prior text deny.

Existing adapter limits remain in force, including at most eight invocations.

`admit_turn()` uses the existing adapter `ContentPolicyRuntime`, admission port, and exact request capability.
It calls shared `admit()` with the complete raw input bytes.
For subsequent turns, derive the complete local input from that input handle and every earlier accepted result handle.
Require current complete bindings and eight-path coverage before returning the input token.
Do not allocate a fresh runtime or use the ordinary empty-control admission path.

The controlled prologue validates the input token before sanitation, history copying, prompt composition, or any preview generation.
It compares all actual prologue arguments with captured input before processing. Changed arguments deny.
It uses the pinned native `_sanitize_surrogates()` for the current user text.
It preserves prior accepted history and response text exactly.
It copies plain history, appends the sanitized user row, and uses only the admitted explicit system text.

It constructs the existing `TurnContext` without prompt discovery, timestamps, persisted sidecars, or injected context.
No unadmitted local text can enter that representation.

`seal_turn()` reconstructs the permitted transformation from retained raw bytes and checks the complete output correspondence.
It derives the complete output bytes with the raw input handle and every earlier accepted result as parents.
The output includes all prepared text, history, system text, indices, identifiers, and fixed empty auxiliary fields.
Revalidate facts and exact object identities before releasing the prepared token or prologue result.
Do not release a partially checked `TurnContext` after failure.

`prepared_request()` constructs only the accepted `model`, `messages`, and supported bounded scalar options.
The system text and history must fit existing adapter request rules.
Dispatch derives the complete wire request from the raw input, prepared output, and every earlier accepted result handle.
The shared registry therefore joins earlier restrictions even when current source profiles become empty.
Retain input and prepared anchors beside existing invocation records without changing the public `NativeMainInvocation` shape.
Unknown profiles, changed producer policy, stale bindings, missing sibling coverage, or capacity exhaustion deny before another send.

## Local-processing interface and shared dependency

The accepted adapter exposes dispatch and scope validation, but no complete-byte prologue admission interface.
Its current `dispatch()` first admits request bytes after the caller has already built them.
That timing cannot prove authority before native input sanitation or prompt preparation.
The concrete consumer of the proposed local interface is the controlled branch of `build_turn_context()`.

Existing shared `admit()` and `derive()` already retain complete bytes, exact handles, parent identities, and lineage profiles.
The proposed native interface uses those operations on the same runtime.
No shared-engine API change is a mandatory dependency for this component.
Do not add an arbitrary local callback executor, generic permission framework, digest-only checker, or new shared path.
Keep the immutable local 0.1.0 wheel and unchanged engine. Native manifests and dependency delivery remain outside this proposal.

## Operator boundary and controlled exclusions

The trusted operator supplies the existing admission, selection, result-policy, clock, routes, and genuine invented ports for all eight paths.
Message content cannot select these ports, profiles, destinations, identities, or the controlled preparation mode.
The same currently sealed adapter governs the preparation token and main dispatch.
All destinations remain `synthetic_only` with provider `synthetic_native` and nonstreaming `chat_completions`.

| Required state | Existing support and additional controlled requirement |
| --- | --- |
| No tools | Use `enabled_toolsets=[]`. Verify actual `agent.tools == []` and an empty valid-tool set because native additions can change the snapshot. |
| No memory | Use `skip_memory=True` without a requested memory toolset. Verify both memory store and manager are absent. |
| No background review | Use existing `skip_background_review=True`. Preparation must not invoke review or lifecycle callbacks. |
| No compression | Require exact `compression_enabled=False`. Omit estimators, compressor callbacks, recovery, and overflow warning callbacks. |
| No fallback | Existing absent `fallback_model` produces an empty chain. Require empty actual chain, no active fallback, and no runtime restoration callback. |
| No title or persistence | Require `_session_db=None` and `_persist_disabled=True`. These options alone do not skip the title guard, so omit the title helper explicitly. |
| No plugins or discovery | Omit hooks, MCP refresh, environment refresh, gateway notes, prompt builders, external memory, and bot-tool injection. Existing ordinary options do not establish this exclusion alone. |

Require `stream_callback=None`, `moa_active=False`, and absent persistence overrides and display metadata.
Require no native command/media context, pending CLI handoff, redirect text, auxiliary injected context, or external history restoration.
Require no nonempty cached or ephemeral prompt beyond the admitted explicit system text.
Require the supplied bounded `task_id` to equal the current execution reference.

Use a deterministic local turn identifier without a persisted session or clock lookup.
Reject unsupported state before preprocessing. Do not silently clear state or convert a refusal into ordinary execution.

Controlled preparation omits stdio installation, logs, preview callbacks, provider construction, persistence, and generic lifecycle work.

## Thread, cancellation, and failure semantics

Admission, preparation, sealing, request materialization, and dispatch stay on the original adapter thread.
No worker, watchdog, timer, lease refresher, heartbeat, or capability transfer follows.
Cross-thread use and reentrancy invalidate the controlled context.
Require exact context, adapter, operator-port, method, and payload identities around each local proof operation.
Recheck current native scope after every cancellation-flag read, as the corrected caller does.

Use the accepted exact Boolean interruption rule, with missing flags interpreted as `False` only within this bounded scope.
Cancellation before preparation releases no processed text and makes zero sends.
Cancellation between preparation and dispatch invalidates pending tokens and makes zero sends.
Cancellation after a completed callback preserves actual send counts and accepted internal records without caller release.

Use the same fixed `InterruptedError` text as the accepted caller.
Process-control exceptions invalidate pending preparation and propagate unchanged.
No in-flight callback preemption, asynchronous cancellation, or deadline guarantee follows.

## Exact proposed native change allowlist

| Path | Permitted future change |
| --- | --- |
| `agent/native_main_payload.py` | Add the bounded token records and three methods above. Extend parent selection in dispatch and private preparation-anchor checks. |
| `agent/turn_context.py` | Add the keyword, optional result field, and controlled preparation branch before the generic guard. Keep ordinary prologue behavior intact. |
| `tests/agent/test_native_turn_context.py` | Add invented input, local transformation, identity, refusal, and actual-caller composition examples. |
| `tests/agent/test_native_main_payload.py` | Add only directly affected preparation/dispatch compatibility cases. Preserve accepted caller and adapter cases. |
| `docs/evidence/<root-selected-component>/` | Record exact native source, current focused results, dependency reuse, and failure scopes. |

No change to `run_agent.py`, `agent/conversation_loop.py`, `agent/turn_finalizer.py`, or `agent/chat_completion_helpers.py` belongs to this component.
No change to `agent/content_policy.py`, shared engine, dependency manifests, routes, permissions, configuration, or deployments follows.
Existing tests for ordinary `build_turn_context()` must continue through their original branch.
The sanitizer remains an unchanged native function. The controlled branch must not trust a caller-supplied replacement callback.

## Focused acceptance examples

| Example | Required observation |
| --- | --- |
| First invented turn | Actual `build_turn_context()` prepares explicit system/user text after complete admission. Actual accepted caller sends the exact resulting request once. |
| Local transformation | A raw surrogate in invented user text remains in retained raw bytes. The prepared text exactly matches the pinned sanitizer. No preview or callback sees it before admission. |
| Repeated invented turn | Use original retained response identity and exact accepted history. Every earlier result remains a parent of local derivation and main dispatch. |
| Current profile drop | Empty current profiles cannot remove restrictions retained by earlier input and result handles. Inspect outgoing bytes and registered parent identities. |
| Invalid input or token | Reject missing, copied, foreign, reused, stale, or wrong-thread tokens before sanitation. Blocked callback counters remain zero. |
| Mutated derived text | Changed history, cached prompt, `TurnContext`, request dictionary, or prior response denies before another send. |
| Stale facts or producer | Mutations during admission, cancellation reads, or sealing invalidate the context and release no prepared output. |
| Unsupported activity | Tools, memory, title, hooks, fallback, compression, stream, MoA, metadata, and persistence refuse before their callbacks. |
| Cancellation | Exercise pre-prepare, post-prepare, pre-send, and post-callback boundaries. Preserve exact counts, fixed exceptions, and accepted internal records. |
| Unopened boundaries | Both conversation entrypoints and actual native siblings refuse after valid controlled scope entry, including after profile drop. |

Use the changed-behavior selection, directly affected ordinary prologue compatibility, and one actual prologue-to-caller composition group.
Reuse byte-identical installed dependency evidence. Do not duplicate the historical 187, 233, 105, or 137 selections without a concrete gap.
These are future acceptance examples. This documentation task executes none of them.

## Mandatory dependencies and separate qualification

The component requires accepted corrected native source, the exact existing wheel, and genuine invented operator ports.
It also requires complete retained input/result history.
The preparation interface and native prologue consumer must land together under the later scope selected by root.
No positive native sibling behavior follows from shared all-path coverage.
Real providers/egress, private deployed-base comparison, durable/serialized restoration, and sensitive cross-thread support require separate qualification.
P06 private inference, source completeness, bank semantics, financial writes, deployment, and activation remain open.
