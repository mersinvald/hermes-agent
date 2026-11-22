# Controlled native construction and text turns

Status: completed implementation for coordinator acceptance.
Source/API freeze: `62d5eaf0033343eb66b749360274bb5246d6d1da`.
Native base: `2a84496fad0724fb3f79b9c37cef3f2c283e781c`.
Canonical PWA publication: `095d81d5299a775f43fe2458e7ac3eb8ef22bb45`.
The final evidence commit changes no runtime or test source.

The approved design comes from immutable PWA source `072eda7696f2b727139b070aa9fb879c76934b56`.
Its SHA256 is `83494927245f5b5386c84bc53d60341e0ab703a01524c8381e457779f5d1ab4b`.
[Input receipts](inputs.json) bind that design, original P10, architecture sections 17.8/17.9, and five protected receipts.
The receipts retain public pins without private content.

The explicit keyword-only `controlled_main` branch precedes ordinary initialization and deprecated-argument warnings.
It validates every constructor option before operator callbacks, configuration, conversion, discovery, or client creation.
It binds one exact constructed `AIAgent` to one genuine, current, unconsumed adapter on its bound thread.
It creates supported inert state directly.
The implementation creates no second agent, client, worker, or alternate loop.

Both existing conversation entrypoints call one private helper before ordinary lifecycle callbacks.
The helper uses the accepted admission and prologue, actual interruptible caller, actual dispatcher, and shared public result derivation.
The prologue seals once, and each turn sends once.
Complete results retain original responses, full bytes, handles, dictionaries, nested anchors, and earlier result correspondence.
Later operations reject changes to earlier results or bindings.

[The exact API](native-api.json) and [consumer contract](consumer-proposal.md) describe arguments, results, ownership, and limits.
`NativeMainInvocation` and existing public adapter signatures remain unchanged.
All generic sibling guards remain intact.
Controlled close invalidates owned pending state before generic cleanup, including malformed, partial, canceled, denied, and closed attempts.
It retains accepted records and refuses later turns.

## Focused verification

The [final log](final-focused.txt) records 100 passed cases across six files in 49.3 runner seconds.
There are zero failures, skips, or retries.
The new integration file contributes 90 cases.
Ten unchanged existing cases cover preparation, label refusal, ordinary constructor routing, caller selection, prologue behavior, and cleanup.
The estimated 140-test progress denominator is not the executed count.

Tests use `scripts/run_tests.sh`, invented temporary configuration, blocked network callbacks, and zero retries.
Positive integration constructs real `AIAgent` instances without constructor replacement, `init_agent` patches, or later state clearing.
Only qualified operator transports simulate invented responses.
Tests inspect original identities, complete retained bytes, and genuine derivation parents through the public shared method.
Excluded lifecycle callbacks fail on use.

The integration cases cover first and repeated turns through both entrypoints, stable prefixes, empty text, and retained profile restrictions.
They cover constructor/entry refusal, current binding changes, prior result mutation, cancellation, thread refusal, reentry, capacity, and close.
Actual siblings refuse after successful controlled entry, including after current profiles become empty.
Ordinary method/module forwarding and ordinary close retain their behavior.
Smaller injected result and integer bounds exercise overflow guards without a partial release.

[The source manifest](tested-source.json) pins every implementation and selected test file.
[Preservation checks](preservation.json) confirm the allowlist and unchanged protected boundaries, manifests, lockfiles, and accepted tests.
The task does not read package-lock contents, change another checkout, or mutate canonical Beads.

## Earlier scopes and failures

Earlier runs overlap the final group and add no distinct accepted cases.
They used evolving source without exact source manifests.
The raw logs retain their original output and whitespace.

- [First](edit-first.txt): 3 passed in 6.3 seconds for initial real constructor success.

- [Group](edit-group.txt): 66 passed and two failed in 39.5 seconds for cancellation expectations and partial-construction invalidation.

- [Correction](edit-correction.txt): 17 passed in 10.8 seconds for corrections, siblings, forwarding, and result guards.

- [Expanded](edit-complete.txt): 84 passed and one failed in 46.0 seconds for the missing-route construction expectation.

- [Last](edit-last.txt): 10 passed in 5.3 seconds for admission, defaults, close, callback, and cancellation checks.


The accepted dispatcher retains its actual invocation before reporting send-time cancellation.
The constructor now invalidates its adapter if initialization or binding raises.
Missing shared coverage refuses admission before input processing or sending.
These earlier failures are not complete passing selections.
No test run overlapped a source edit within its imported scope.

## Dependency and static scope

[The installed smoke](installed-smoke.json) uses one task-owned offline environment with 84 cached locked native dependencies.
It installs the exact accepted shared wheel with `--no-index --no-deps --offline`.
Engine version 0.1.0 and both accepted wheel/engine hashes remain unchanged.
The smoke checks imports and public API only.
The smoke includes no native build, index lookup, copied-engine import, PWA server installation, or duplicate installed functional selection.

One preliminary smoke invocation used a temporary script path and failed to import the native repository module.
The corrected stdin invocation used normal repository import resolution and passed.
It still imported the shared engine from the wheel installed in the task environment.

[Static comparison](ruff.json) finds no new F/E9 diagnostics.
Two existing diagnostics remain: an unused proxy import and the unresolved `RateLimitState` annotation.
Changed runtime and tests pass the scoped whitespace check.
Raw failure logs retain quoted diagnostic whitespace.

[STE baseline](ste-before.txt) and [STE candidate](ste-after.txt) use the same global linter and Python prose mode.
No existing Python prose score or em-dash count increases.
New Python prose, the consumer contract, this report, and PR text have no mechanical violations.
The linter does not certify technical correctness.

## Publication and limits

The feature branch remains available for review.
The draft PR targets `mersinvald/hermes-agent:mkl`.
The worker reports configured checks against its exact final head after publication.
Zero checks do not imply CI success.
The coordinator retains final acceptance, canonical Beads, and merge ownership.

The local bounds remain eight turns and 262144 result bytes.
Cumulative counters require exact integers bounded by `2**53 - 1`.
Shared limits remain 128 handles and 8 MiB.
Cancellation remains pending, and process-control exceptions propagate unchanged.
There is no in-flight preemption, deadline, heartbeat, watchdog, or sensitive cross-thread dispatch claim.

Full P10/P06, real providers/egress, auxiliary native success, durable or serialized restoration, source/financial qualification, and private processing remain open.
This work includes no private/model processing, financial write, production test, deployment, or workflow activation.
