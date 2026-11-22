# Native controlled main payload proposal

Date: 2026-11-22. Issue: `hpwa-p10.5`. Status: root decision required.
Native base: `a31286bd59f8d701676c1cca82a9e699e416fb7c`.
PWA contract pin: `1219df6a72b29920741a5ea91e19955300723b65`.

This proposal supports one native function with invented content and controlled ports.
It grants no private inference or production permission.
Root owns the implementation decision, Beads writes, final acceptance, and publication.

## Source boundary

The original P10 requires policy before sensitive calls and across eight paths.
Architecture section 17 preserves these requirements and separates earlier control checks from payload dispatch.
The P10.4 follow-up requires complete main request/result bytes and result reuse.
The [source pins](pins.json) identify these immutable sources.

At the native base, `agent/content_policy.py` has 243 lines.
`NativeContentContext._current()` calls `require_ordinary()` before its binding checks.
`NativeContentContext.check()` creates a fresh shared runtime and admits `b""`.
`checked_native_call()` checks authority around an unrelated native callback.
These functions prove ordinary control checks and sensitive refusal only.

The selected function is `agent.chat_completion_helpers._dispatch_nonstreaming_api_request()` at base lines 920-1009.
It dispatches Codex, Anthropic, Bedrock, MoA, and OpenAI-compatible calls.
The normal OpenAI-compatible branch constructs a client at line 1006.
Its current callback invokes `request_client.chat.completions.create(**api_kwargs)` at line 1009.

The proposed interception precedes the existing guard at line 936.
Only an explicitly bound controlled adapter can take this interception.
It returns the accepted controlled response directly.
It never constructs a request client or invokes the original SDK callback.
Without this adapter, the existing function continues its current behavior.

`AIAgent.run_conversation()`, conversation prologues, and streaming retain their existing guards.
This component does not admit a complete sensitive conversation through those earlier boundaries.
Its positive claim concerns the selected actual function only.

## Shared API and delivery

The sole engine remains `hermes_content_policy` from `hermes-content-policy==0.1.0`.
Its engine SHA-256 is `28fa3b07c5d6f8a64b9aa970fbdb8a212ac2e7a03d9e4fa6d3020c9ddf0554aa`.
The accepted wheel SHA-256 is `c10b082397cc05e9cdaf5c1991b82c0834af3d78729eb0c91b53aaae96b7c2bc`.
The exact wheel path and handoff receipt appear in [pins.json](pins.json).

After root approval, install that wheel into a task-owned environment.
Use `--no-index --no-deps` with its exact filesystem path.
Do not add a dependency manifest, checkout import path, engine copy, or PWA server dependency.
The artifact stays immutable.

The unchanged public methods suffice for the proposed record boundary:

```python
ContentPolicyRuntime(
    *, admission: AdmissionPort, profiles: tuple[ProfilePolicy, ...],
    routes: tuple[RouteBinding, ...],
    transports: tuple[tuple[str, TransportPort], ...],
    clock: Callable[[], int], policy_revision: int,
) -> None
ContentPolicyRuntime.admit(request_handle: object, payload: bytes) -> ContentHandle
ContentPolicyRuntime.derive(
    request_handle: object, parents: tuple[ContentHandle, ...], payload: bytes,
    *, path: Path,
) -> ContentHandle
ContentPolicyRuntime.dispatch(
    request_handle: object, content: ContentHandle, *, path: Path, route_ref: str,
) -> ContentHandle
AdmissionPort.current(request_handle: object) -> ExecutionBinding
TransportPort.resolve() -> Destination
TransportPort.send(envelope: DispatchEnvelope) -> bytes
```

The native wrapper retains the exact bytes returned by its controlled port.
Successful shared `dispatch()` returns the corresponding accepted result handle.
The adapter pairs that handle with those retained bytes after all freshness checks.
It reads no private engine registry and unwraps no content capability.
`accept()` does not release result bytes and is unnecessary for this main operation.

## Exact native types and signatures

Add `agent/native_main_payload.py` after approval.
Use frozen dataclasses with slots and exact field types.
The response supports one textual choice without tools or streaming.

```python
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
class NativeSelectionPort(Protocol):
    def current(self, request_handle: object) -> NativeMainSelection: ...
class NativeResultPolicyPort(Protocol):
    def current(self, producer_ref: str) -> NativeResultPolicy: ...
class NativeMainPayloadAdapter:
    def __init__(
        self, context: NativeContentContext, *,
        selection: NativeSelectionPort, result_policy: NativeResultPolicyPort,
        producer_ref: str, main_route_ref: str,
        profiles: tuple[ProfilePolicy, ...], routes: tuple[RouteBinding, ...],
        transports: tuple[tuple[str, TransportPort], ...],
        clock: Callable[[], int],
    ) -> None: ...
    def dispatch(self, agent: object, api_kwargs: dict) -> NativeMainResponse: ...
def bind_controlled_main(
    context: NativeContentContext, *, selection: NativeSelectionPort,
    result_policy: NativeResultPolicyPort, producer_ref: str, main_route_ref: str,
    profiles: tuple[ProfilePolicy, ...], routes: tuple[RouteBinding, ...],
    transports: tuple[tuple[str, TransportPort], ...], clock: Callable[[], int],
) -> NativeMainPayloadAdapter: ...
def dispatch_controlled_main(
    agent: object, api_kwargs: dict,
) -> tuple[bool, NativeMainResponse | None]: ...
```

`bind_controlled_main()` attaches one adapter to an existing exact registered native context.
It rejects replacement and captures that context, its admission port, and its exact request handle.
The native context gains an optional private adapter slot, initially `None`.
Other context methods keep their ordinary-only behavior.

`dispatch_controlled_main()` first uses `current_context(agent)`.
It returns `(False, None)` when the context has no controlled adapter.
An attached adapter must have the exact captured context and attachment identity.
Copying an adapter, context, handle, or response grants no authority.
Any failed controlled operation invalidates the native context and its adapter.

The adapter creates one shared runtime on its construction thread.
It never transfers that runtime or its content handles across threads.
It rejects reentry, thread changes, adapter replacement, and changed trusted port identities.
These checks protect adapter state and add no independent permission rules.
The shared engine owns route, profile, lineage, destination, and content decisions.

## Supported operation and encoding

Support only `agent.api_mode == "chat_completions"` and `agent.provider == "synthetic_native"`.
The selected provider must equal `agent.provider` and `selection.destination.provider_ref`.
The selected model must equal `api_kwargs["model"]` and `selection.destination.model_ref`.
The current selection identifier and version must equal the authenticated execution binding.
The chosen route destination must equal the current selection destination.

The request must be an exact built-in dictionary.
Its required keys are `model` and `messages`.
Its only optional keys are `max_tokens`, `temperature`, and `stream`.
`model` is an exact nonempty string with at most 128 ASCII identifier characters.

`max_tokens` is an exact integer from 1 through 4096.
`temperature` is an exact finite float from 0 through 2.
`stream`, when present, must be the exact value `False`.

`messages` is an exact list with 1 through 32 exact dictionaries.
Each message has exactly `role` and `content`.
Both fields are exact strings.
Content permits valid Unicode scalar values and has at most 16,384 UTF-8 bytes.

The first request permits one optional system message followed by one user message.
Later requests contain the previous request messages, its accepted assistant message, and one new user message.
This grammar preserves earlier messages and permits controlled result reuse.

Tools, images, files, arbitrary objects, mappings, subclasses, iterators, cycles, and unknown keys deny.

Validate the complete bounded shape before conversion, content comparison, hashing, or callback release.
Check exact container and key types before membership tests or key lookup.
Check string bounds and surrogates before UTF-8 encoding.
Use explicit finite scalar bounds.
Reject booleans in integer fields.
No user object receives `str()`, custom equality, hashing, JSON conversion, or attribute traversal.

Encode the complete request as UTF-8 JSON without a byte-order mark or newline.
Use `sort_keys=True`, `ensure_ascii=True`, `allow_nan=False`, and `separators=(",", ":")`.
The complete envelope payload has exactly three fields:

```json
{"api_mode":"chat_completions","destination":{"endpoint_ref":"invented_endpoint","model_ref":"invented_model","provider_ref":"invented_provider","revision":1,"transport_ref":"invented_main"},"request":{"messages":[{"content":"Invented document A","role":"user"}],"model":"invented_model"}}
```

Reject request bytes above 262,144 bytes before shared admission or port invocation.
The payload includes all accepted request fields, API mode, and the complete selected destination.
It is not a digest or empty control admission.

The controlled result must be exact `bytes` with at most 65,536 bytes.
Decode strict UTF-8 JSON and reject duplicate keys, invalid numbers, excessive nesting, and excessive nodes.
Use a bounded pre-parser depth check before JSON decoding.
The result permits at most eight container levels and 2,048 total JSON nodes.
The only accepted result schema matches all `NativeMainResponse` fields above.

`object` must be `chat.completion`.
`choices` must contain one item with index zero, role `assistant`, and finish reason `stop`.
The wire message has exactly `role` and `content`, with no tools.
Token counts are exact integers from zero through 1,000,000.
Their sum must equal `total_tokens`.

`created` is an exact integer from zero through 9,007,199,254,740,991.
Response identifiers use the same bounded identifier grammar as the request model.
The response model must equal the selected model.
Canonical re-encoding must equal the received result bytes.

The native adapter constructs the frozen response from validated primitives after shared acceptance.
It retains that exact response object beside its accepted result handle.
It revalidates and re-encodes retained response objects before reuse.
Changed objects or unsupported fields deny before comparison.

## Operator port boundary and eight-path closure

The caller does not supply routes, profiles, selection, provenance, or ports through request fields.
Only the explicit in-process operator factory supplies these objects.
The selection port reads current authenticated selection facts.
The result-policy port reads current authenticated producer facts.
The admission port returns the original current execution binding without adapter-added profiles.

The controlled main port sends the exact shared envelope payload to an invented in-process service.
Its `resolve()` returns the actual destination used for that operation.
Its `send()` executes one synchronous operation without retries, redirects, SDK calls, or network access.
The wrapper compares the observed destination immediately before and after the controlled send.
Shared dispatch also checks current destination and selection around the callback.
There is no real provider callback with `synthetic_only` qualification.

The shared runtime requires live route coverage for every required path.
It has no public representation for an explicitly refused path.
Therefore the native adapter must not fabricate sibling routes or qualification references.
It accepts an operator-supplied complete controlled policy universe.
Each profile keeps `required_paths == PATHS` with all eight original names.

The offline operator fixture must bind real invented in-process ports for all eight shared paths.
Each port must implement its declared controlled operation and destination truthfully.
Each delegate/A2A port must also have the authenticated receiver binding required by shared dispatch.
Direct bounded shared dispatch checks qualify these exact fixture ports for their declared paths.
These are new port checks, not repeated historical engine suites.
Removing a required route or changing its actual destination must prevent even main admission.

Native support remains narrower than this controlled shared universe.
Only the selected actual native function connects to the main port.
Existing native guards refuse sensitive fallback, title, compression, memory, delegate, A2A, and tool-result processing.
Streaming, sibling API modes, conversation prologues, and auxiliary main calls gain no positive exception.
Tests must prove these refusals with a controlled adapter attached.
Native refusal therefore closes those native paths without granting a native send capability.

This proposal asks root to approve this distinction explicitly.
Complete controlled shared coverage is a prerequisite, not proof of native support for sibling operations.
If root requires coverage to represent refusal itself, the unchanged API cannot express that fact.
The smallest alternative seam is an explicit shared refusal binding for a consumer, selection, version, and path.

Shared coverage would accept that binding while shared dispatch would deny its path before any send.
That alternative requires separate shared source review and a new accepted artifact.
It is not part of the implementation proposed here.

## Invocation records, freshness, and reuse

Retain at most eight invocation records within one adapter.
Each record contains the following facts:

```python
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
```

The pending operation keeps its envelope and result bytes until shared dispatch finishes.
No pending object is a reusable invocation record.
No response escapes when shared dispatch rejects its result.
Count a controlled send once `send()` starts, even if later checks reject the response.

Before the first request, validate current binding and selection shapes without custom equality.
Validate all binding identifiers, positive bounded revisions, and exact unique profile tuples.
The shared runtime validates the complete binding and policy configuration again.
Use the exact original request handle for every shared operation.

The trusted port must return exact current producer policy.
Its profile tuple must be a subset of the authenticated source profile tuple.
This first component refuses producers with additional restrictions that the source does not already carry.
It never invents a binding or drops an output restriction to make dispatch succeed.
Unknown producer policy denies before the controlled send.

Admit the first complete request bytes with shared `admit()`.
For later requests, use shared `derive()` with every retained accepted result handle as a parent.
Then use shared `dispatch(..., path="main", route_ref=main_route_ref)`.
The wrapper retains the complete returned bytes and observed destination inside `send()`.
Shared dispatch must return its result handle before native result construction and release.

After every trusted callback, compare current binding, selection, producer policy, context, and port identities.
The full binding must remain unchanged during one invocation.
The owner, execution, lineage, consumer, authentication epoch, policy revision, selection identifier, and version remain fixed across invocations.
Source profiles can change between invocations only under retained shared lineage rules.
The adapter never removes profile restrictions retained in earlier results.

Recheck producer policy for every retained record before reuse.
Unknown policy, changed revision, changed profile tuple, or changed producer identity denies reuse and invalidates the context.
Check the identity and canonical bytes of the original response object.
The next request must contain the accepted assistant text in the required conversation position.
All earlier result handles remain parents even when current source profiles become ordinary.
Shared `derive()` therefore preserves their profile union and current required-path coverage.

Recheck current facts after shared acceptance and after response construction.
If a callback changed trusted facts, discard the result and block later operations.
Retain the authorized send count in the failure evidence.
No post-send rejection claim implies that the earlier authorized send did not happen.

## Proposed edit and check scope

After root approval, runtime edits affect `agent/content_policy.py` and `agent/chat_completion_helpers.py`.
The new adapter lives in `agent/native_main_payload.py`.
Actual-function tests live in `tests/agent/test_native_main_payload.py`.
The evidence directory contains this proposal, pins, command records, and final scoped results.
No other runtime boundary gains a positive exception.

Use [validation-plan.md](validation-plan.md) for change-based checks.
Preserve the earlier 137 native and 21 compatibility passes at source `3b4fe67e` as historical evidence.
Those cases prove no positive native payload dispatch.
Rerun only changed adapter paths and directly affected ordinary compatibility cases.

Full P10, real egress qualification, durable restoration, and sensitive cross-thread transfer remain open.
Production ingress, configuration, deployment, model calls, private content, and financial logic remain outside this component.
Do not push or open a native PR before root reviews the completed source and confirms publication.
