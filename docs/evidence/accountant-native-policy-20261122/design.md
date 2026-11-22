# Native shared content policy proposal

Date: 2026-11-22. Task: `hpwa-p10.2`. Status: design gate, runtime approval pending.
Native base: `089fadba6cfeacc5af9ebffcec1e6f72ae9eed28`.
[source-pins.json](source-pins.json) identifies every inspected source object.
[source-map.md](source-map.md) identifies the native call sites and gaps.
[allowlist.json](allowlist.json) defines the proposed edit boundary.

## Concrete component

Add one native adapter layer around the authoritative P10.1 policy engine.
Bind content policy before inbound preprocessing, agent construction, title launch, or memory callbacks.
Use the same bindings at native dispatch and result boundaries.
Keep all policy decisions in `ContentPolicyRuntime`.
The native layer supplies authenticated facts, content handles, and actual call-site observations.
It must not become another profile intersection or destination policy engine.

This component can prove native refusal and controlled native call-path behavior.
It cannot authorize sensitive production inference with the accepted shared API.
`RouteBinding.qualification_scope` accepts only `synthetic_only`.
Keep that restriction intact.
Never wrap a real provider callback inside a synthetic-qualified transport.
No port name, private URL, or endpoint label proves local egress.

Recommend this bounded component for approval.
Sensitive real-provider admission must return a fixed policy denial before preprocessing.
Positive shared-contract cases use invented data and operator-bound offline ports in tests.
Call-site tests must execute changed native functions and count provider invocations.
Separate service simulations remain shared-contract evidence only.

## Ownership and delivery decision

The PWA wheel currently owns `hermes_pwa_server.content_destination_policy`.
Its distribution requires Python `==3.13.*` and installs server, database, and authentication dependencies.
Native Hermes supports Python `>=3.11,<3.14`.
Adding the PWA distribution to native requirements would couple unrelated applications and exclude supported interpreters.
A checkout path, `PYTHONPATH`, copied engine, or import fallback would hide this constraint.

Propose a separate dependency worktree rooted at accepted PWA main.
Move the unchanged engine into `packages/content-policy/src/hermes_content_policy/__init__.py` there.
Build distribution `hermes-content-policy`, version `0.1.0`, with no runtime dependencies and Python `>=3.11,<3.14`.
Keep the PWA import path as a compatibility re-export from `hermes_content_policy`.
The re-export must preserve public object identity.
The shared distribution owns the sole implementation after extraction.

Use `from hermes_content_policy import ContentPolicyRuntime, ExecutionBinding, Destination` in the native adapter.
Load that dependency lazily when the operator enables native content policy.
Add native optional extra `content-policy = ["hermes-content-policy>=0.1.0,<0.3"]` only after root approves artifact delivery.
Use an immutable artifact hash and source receipt for validation.
Do not resolve an unreviewed package from an index under this proposed name.

Root must supply the dependency worktree, exact approved artifact, and publication target.
The companion change must retain the engine source hash and PWA test imports.
No dependency file or cross-repository edit belongs to this design commit.
If extraction cannot proceed, stop native imports and dependency edits.
Do not substitute a second engine.

## Authenticated admission

Add operator configuration under `pwa_http.content_policy` in existing `config.yaml` parsing.
The section contains `enabled`, `policy_revision`, `auth_epoch`, and source-ID-to-profile assignments.
Each enabled source needs an explicit profile tuple, including `[]` for ordinary authenticated content.
Missing assignments, unknown profiles, or a missing host adapter deny that source.
An absent section keeps the existing unqualified legacy surface outside this component enforcement claim.
It does not authorize sensitive work or establish an ordinary binding.

Install one trusted admission adapter when `NativePwaHttp` constructs its existing ingress.
The adapter reuses `NativePwaOwnership.authorize`, configured source identity, and native command ownership.
The request cannot supply authoritative profiles, destinations, epochs, or admission handles.
The command DTO remains unchanged.

For platform ingress, resolve the authenticated adapter event to one configured owner and source assignment.
Reject absent or ambiguous owner matches.
A `SessionSource` constructed by an untrusted caller is not an authentication capability.
Retain platform authorization and the native Telegram binding checks.

Before command acceptance, bind the authenticated principal and canonical conversation to the operator assignment.
Before execution, bind the actual execution ID and process owner from the durable native execution row.
Use the compression root as `lineage_ref` and the captured model ID/version as selection facts.
Use an opaque server-side owner reference for principal plus concierge identity.
Do not confuse that owner with `_command_owner`, which identifies a process lease.
Do not substitute the conversation generation or model version for `auth_epoch`.

The operator owns positive `auth_epoch` and `policy_revision` values.
The adapter checks current authorization and configuration on each operation.
Configuration replacement invalidates live handles instead of silently refreshing them.
Expiry and selection changes require a new admitted execution.
Compression preserves the root and effective profile union.

Inbound preprocessing currently precedes `TurnRunner.run_sync` and native model capture.
Guard `_prepare_profile_scoped_inbound_message_text` and `_prepare_inbound_message_text` at entry.
For enabled policy, require authenticated source admission before either function reaches media or context expansion.
Sensitive admission denies here because preprocessing adapters and production qualifications remain absent.
Capture full execution/selection facts later under the existing managed-model lease before any eligible payload dispatch.
No sensitive execution reaches that later step in this component.

The early check is a refusal boundary, not a provisional shared execution capability.
Positive sensitive preprocessing needs execution capture before preprocessing and qualified adapters for every preprocessing hop.
That ordering change remains a separate acceptance gate.

`TurnRunner.run_sync` must recheck admission before agent construction and cache reuse.
Attach a private, exact-identity policy context to the executing agent and scoped call context.
`AIAgent.run_conversation` must reject missing, stale, or foreign context for policy-enabled execution before its prologue.
Direct unmanaged CLI/TUI entrypoints remain outside this bounded integration.
Their unchanged behavior does not establish sensitive admission support.

## Native adapter contract

Add `agent/content_policy.py` for native binding and call adapters.
It imports the shared engine and exposes native boundary helpers, without duplicating route rules.
Keep configuration parsing in `gateway/pwa_config.py` and authenticated source lookup in existing ownership code.
Use explicit context capture for threads instead of global environment state.
Do not add model tools, mutate prompts, alter toolsets, or insert policy text into messages.

An enabled execution needs a current authenticated binding even when its profiles are empty.
Empty profiles mean the authenticated operator selected ordinary processing.
Missing context means denial for enabled execution.
Untrusted strings in messages, result dictionaries, or restored JSON cannot create authority.

For controlled native dispatch, the adapter supplies one captured transport to shared `dispatch`.
`resolve` reads the exact request-local client, final model, API mode, and endpoint binding.
Operator mappings bind these observations to bounded shared `Destination` identifiers.
The adapter checks request override fields before transport release.
Unknown clients, hidden retry/failover behavior, or unsupported API modes deny sensitive dispatch.

The production host loader exposes no synthetic transport registration setting.
Tests inject offline ports through the native adapter constructor and invoke actual changed native functions.
Those fixtures do not register provider SDK callbacks as synthetic ports.
Positive offline adapter tests do not make production transport adapters available.

The shared API accepts bytes and returns opaque content handles.
Native request objects and response objects therefore stay in a bounded adapter-owned invocation record.
The transport receives canonical request bytes and returns bounded response bytes.
The adapter returns the captured native response only after shared dispatch accepts the result.
Do not read private shared registries or add a handle-unwrapping escape hatch.
Streaming remains unsupported for sensitive content until each emitted chunk has a qualified retention and release contract.

The current engine binds itself to one thread and rejects reentrancy.
Use it only on its owner thread.
Do not share it directly with title, memory, tool, or delegate workers.
These paths deny sensitive work before scheduling until an authenticated cross-thread handoff exists.
Tests can call synchronous changed functions on the owner thread with bounded offline ports.
Asynchronous tests verify denial before any awaitable provider dispatch.

## Required path closure

| Path | Native boundary and proposed behavior |
| --- | --- |
| Main | Guard execution before prologue. Observe request-local clients in `_dispatch_nonstreaming_api_request`. Guard streaming entry before worker launch. Sensitive real clients deny. |
| Fallback | Check context before `try_activate_fallback` mutates routing. Recheck each resolved replacement at dispatch. No fallback inherits permission. |
| Title | Capture context before title worker launch. Check `generate_title` before snippet processing and `call_llm`. Validator exceptions cannot waive policy denial. |
| Compression | Check `_generate_summary`, `_micro_summarize_one`, and direct `handle_max_iterations` before content conversion or provider calls. Native compaction and context-engine extensions remain unsupported. |
| Memory | Check manager entrypoints before provider callbacks and before worker submission. Cover prefetch, turn/session hooks, sync, tools, compression, writes, and delegation results. External provider internals remain unsupported. |
| Delegate | Check `delegate_task`, `_build_child_agent`, `_run_single_child`, and finalization. Sensitive child construction, thread dispatch, and background launch deny. Returned summaries cannot clear constraints. |
| A2A | Check `_send_task` before discovery, redaction, audit, persistence, streaming, or HTTP. Serialized peer authentication and receiving enforcement remain absent, so sensitive peer sends deny. |
| Tool result | Check both executor append paths and `_tool_result_content_for_active_model` before conversion or vision preprocessing. Trusted producer constraints join existing content through shared `derive`. Unknown provenance denies reuse. |

All constrained profiles require all eight paths.
Do not remove a required path because its adapter is absent.
An unsupported-path denial does not prove that the feature cannot run.
Closure checks must reject before the first sensitive provider call.
Do not claim eight-path positive native support from eight labels in a fake service graph.

Direct summaries have separate Codex, Anthropic, Bedrock, and OpenAI branches.
The actual test matrix must call the changed summary function for each branch and assert zero sensitive provider calls.
The same rule applies to sync and async auxiliary calls, including stream retries and credential/model replacement.
Keep ordinary auxiliary resolution and existing managed-title strict destination checks intact.

Tool results can promote an ordinary lineage through an authenticated producer.
Use shared `derive` to join constraints before append, summary, memory, or the next model call.
Reject unknown or missing trusted-result policy instead of converting it into an ordinary string.
Preserve existing side-effect disposition, canonical append order, persistence, and image checks.
After rejection, invalidate the enabled execution so an earlier ordinary handle cannot continue.

## Durability and real service limits

The durable command ledger remains authoritative for command and execution state.
It currently stores no shared content-policy capability or authenticated restoration record.
Do not serialize exact-identity handles or trust profile strings from task/result JSON.
For an enabled source, replay without current authenticated restoration must deny before preprocessing or model dispatch.
Do not mark the original command completed merely because the runtime refused replay.
Retain the ledger and report a truthful policy failure through existing outcome mechanisms.

Background delegation already has durable task and completion records.
Recovery does not restore shared policy authority.
Guard restored completion consumption for policy-enabled targets before a new agent turn or memory callback.
Retain the stored result and existing ownership filters.
Do not silently drop policy-enabled records or reclassify missing context as ordinary.

Serialized authentication, durable policy restoration, and cross-thread capability transfer need separate accepted designs.
Actual deployed private-base differences also need a source comparison at the deployed commit.
Do not use original checkout HEAD `711d3bc5ac1c7e91bff6a25beade35e6c5215924` or image prefix `f1990df3` as proof.
No P01 attestation, destination fact, or local-model assertion originates in this component.
Infra stays read-only at `9b766df2925968f12687b008e9f452c3a37360b0`.

## Generic specialist

Use invented specialist `document-review` with profile `restricted-documents` in native tests.
It has no financial classification, account fields, or Accountant import.
Its operator selects the profile before any concierge or specialist model sees the document.
The same shared binding, route equality, profile union, and denial rules apply.
An explicitly authenticated ordinary specialist uses an empty profile tuple.
These cases show a generic contract without a finance-only authority framework.

## Decision required from root

Approve or revise the exact native runtime/test paths in [allowlist.json](allowlist.json).
Confirm that this component delivers native denial plus controlled call-path tests, with sensitive real-provider dispatch still denied.
Assign the separate shared-package worktree and exact artifact delivery target before imports or dependency edits.
Confirm the operator configuration and enabled-source replay denial described above.
Broader production qualifications, persistent restoration, and publication require separate explicit scope.

This design stage changes only this evidence directory.
It runs no native, PWA, or Accountant tests and makes no network calls.
Root retains Beads writes, final acceptance, cross-repository decisions, merge, and publication.
