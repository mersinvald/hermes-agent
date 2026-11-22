# Native admission and refusal component

Runtime/test source: `3b4fe67e8e708532fbe78ee3a088d26353a64d11`.
Base: `089fadba6cfeacc5af9ebffcec1e6f72ae9eed28`.
Root approved the bounded component through `msg_46c9b42c553e`.
Root supplied the accepted shared wheel through `msg_f882209c55c8`.

The component refuses sensitive native processing before preprocessing or provider dispatch.
It does not enable sensitive inference, qualify a native destination, or complete P10.
The edit set contains the approved 19 runtime files and eight new test files.
Native dependency manifests remain unchanged.

## Shared package and native contract

Native guards lazily import `hermes_content_policy` from the accepted standalone wheel.
The installed engine retains SHA256 `28fa3b07c5d6f8a64b9aa970fbdb8a212ac2e7a03d9e4fa6d3020c9ddf0554aa`.
[dependency-handoff.json](dependency-handoff.json) records the exact artifact and immutable receipt hashes.
The upstream artifact receipt retains its candidate status.
The separate root receipt and reply accept that exact artifact.
Native validation did not install the PWA server distribution.

The native layer uses shared `ExecutionBinding` and `ContentPolicyRuntime.admit` for ordinary admission checks.
Each check admits empty control bytes in its own runtime.
No shared content handle crosses threads.
The native layer does not send request or result payloads through shared `dispatch`.
It therefore makes no positive native content-propagation or destination-equality claim.
Sensitive source profiles always refuse before those payload operations.

This component adds no second route intersection engine, financial classifier, P01 fact producer, or production qualification scope.
`synthetic_only` remains unchanged in the installed authoritative engine.
No real SDK callback registers as a synthetic transport.
No configured endpoint, fake peer, or successful test proves local egress.

## Admission and current authority

`pwa_http.content_policy` accepts an exact boolean flag, positive epoch/revision values, and explicit source assignments.
An empty profile tuple denotes an explicitly authenticated ordinary source.
Absent or disabled policy retains the existing unqualified legacy surface.
That surface remains outside the new enforcement claim.
Missing assignments, ambiguous owners, missing adapters, stale bindings, or missing dependencies deny enabled work.

Admission reuses `NativePwaOwnership.authorize`, current platform authorization, and the existing canonical conversation lookup.
The adapter binds exact event objects after authenticated native admission.
A copied event or a matching `SessionSource` value grants no preprocessing capability.
Configuration or adapter replacement invalidates existing enabled contexts.
Independent epoch, revision, owner, execution, lineage, and selection checks remain explicit.

The existing durable execution row and managed-model lease supply execution and selection facts.
An opaque owner reference identifies the authenticated principal and concierge.
It does not substitute the process lease owner for the principal.
The selected model version does not substitute for the authentication epoch.
Policy refusal releases a captured managed lease and records a failed execution through existing ledger methods.

## Actual native call-path coverage

| Boundary | Implemented behavior | Actual-function evidence |
| --- | --- | --- |
| Ingress and preprocessing | Refuse constrained profiles before command admission, media preprocessing, or context expansion. Require exact admitted event identity. | Real ownership and temporary SQLite. Both preprocessing functions execute their new guards. |
| Main | Check before prologue and request-local client dispatch. Recheck before ordinary result release. | Real forwarder, loop, prologue, dispatch, and streaming entrypoints. Fake client observes ordinary arguments. |
| Fallback | Refuse constrained or stale context before route mutation. | Actual `try_activate_fallback` entrypoint refuses. No positive constrained fallback claim. |
| Direct summary | Refuse before summary conversion. Check direct Codex and managed summary dispatch sites. | Actual iteration-summary entrypoint refuses across five API/provider configurations. These are entry refusals, not positive branch qualification. |
| Title and auxiliary | Refuse before snippets, validators, workers, sync calls, or async provider awaits. Preserve managed strict destination routing. | Actual title and sync/async auxiliary functions. One ordinary title uses real resolution and a fake HTTP transport. |
| Compression | Refuse before full/micro summary conversion and compression entry. | Actual compressor methods refuse without a provider call. |
| Memory | Guard manager callbacks and scheduling. Recheck queued work before its callback. | Actual manager methods and a real blocked executor with an invented callback. |
| Delegate | Refuse before child construction, run, finalization, or background scheduling. Ordinary children retain the private context. | Actual delegate construction/run/finalization and both background dispatch entrypoints. |
| A2A | Refuse before peer discovery, redaction, audit, persistence, or transport. | Actual `_send_task` for unary, streaming, and follow modes. Fake HTTP ports receive zero sensitive calls. |
| Tool results | Reject unknown or constrained producer policy before dispatch, conversion, and append. Invalidate subsequent enabled reuse. | Both actual executors and native string/image conversion entrypoint. Earlier ordinary context cannot continue after rejection. |
| Recovery | Require current authenticated restoration for enabled re-entry. Preserve stored commands/results. | Actual command replay refusal and actual completion restore into a queue, followed by native delivery refusal. |

The tests use invented `document-review` content and `restricted-documents` profiles.
They contain no financial fields or Accountant imports.
All eight shared path names remain required for future constrained support.
An absent adapter does not establish that a path cannot run.

When an ordinary authorized callback changes the epoch, the adapter rejects its returned result.
The test counts that callback as one actual authorized invocation.
It does not describe the earlier invocation as prevented.
If client construction changes the source to constrained content, the adapter refuses before the provider invocation.

## Durable state and unsupported paths

Owned idempotent receipt readback preserves the original command identity and fingerprint.
Readback does not authorize image preprocessing or another model call.
Enabled queued commands without current in-process admission receive an existing failed execution outcome and remain stored as not applied.
Restored completions retain their pending durable result rows when delivery refuses.
Missing policy fields in restored JSON never establish ordinary authority.

This component does not restore serialized capabilities or durable policy authority.
It refuses enabled completion re-entry rather than treating a missing JSON marker as permission.
It does not qualify peer receivers, memory-provider internals, context-engine extensions, OCR, vision, embeddings, or native compaction.
It does not support sensitive streaming, child-thread handoff, background delegation, or cross-process transfer.
Ordinary native routing retains its existing provider behavior, including opaque SDK internals.
Those internals have no sensitive transport qualification in this component.

Unknown trusted tool-result policy also refuses ordinary enabled tool reuse.
The production loader currently supplies no qualified result-profile adapter.
This limitation prevents unknown returned content from silently becoming ordinary text.
The legacy surface and explicitly authenticated ordinary calls without such unsupported results retain their existing behavior.

## Validation and remaining acceptance gates

[checks.json](checks.json) records exact selections, source pins, report hashes, failures, and durations.
The final affected selection uses the prescribed native runner and the accepted installed wheel.
All 137 new native cases pass at the runtime/test source above.
All 21 selected compatibility cases pass at that same source.
Both selections have zero failures and skips.
Their elapsed times are 11.98 seconds and 20.77 seconds.

The compatibility selection covers command identity, failure cleanup, managed title routing, image retention, resolvers, and completion ownership.
No complete native, PWA, or Accountant suite ran.
No additional shared service simulation ran in this task.
Historical synthetic service evidence retains its P10.1 scope.

The early edit run recorded 44 passes and eight setup failures from missing `aiohttp`.
Locked environment setup corrected that missing optional dependency.
The next run recorded 95 passes and one incorrect decoded-result field assertion.
A later run recorded 59 passes and one missing fixture model-initialization step.
The corrected runs retain their separate report scopes.
None of these failures is hidden by a retry.

The accepted P10.1 results remain 109 policy cases and 33 unchanged transport cases at their recorded sources.
Its installed wheel smoke retains its original scope.
The five PostgreSQL-dependent facade skips remain unverified.
P10.3 standalone Python checks and its initial fixture failure retain their own recorded scopes.

This task adds no production installation, deployment, private inference, source-completeness, or egress evidence.
Serialized/durable transfer, deployed private-base differences, financial activation, and full P10 remain separate acceptance gates.
Root owns source acceptance and the native publication decision.
