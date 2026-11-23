# Native empty-snippet local title result

This component implements the selected P10.15 design for actual synchronous `generate_title()`.
An actual empty snippet returns proven `None` through a separate local-result record.
The retained source, title capability, current policy, and shared runtime remain required.
Full P10/P06 remains open.

## Accepted inputs

Native base: `4d47543f91aaf94d4abc3bac2c04a47a411a1be2`, published through native PR7.
PWA publication: `08bde9978829cbf0ab1f7c7c5efd0d65de981ed7`, published through PWA PR37.
The design source is `f75aa9491cb44f64362bdbd14121cb62936b4a88`.
Its path is `docs/mesh/evidence/accountant-native-retained-title-contracts-20261123/next-component-design.md`.
Its SHA256 is `70f0699773c424ed29fe279158360809e55daf3c78984019209e50e391863641`.

[Input verification](input-verification.json) records the protected receipt checks.
[Dependency receipts](dependency-receipts.json) record the accepted dependency delivery.
The original P10 requirements and architecture section 17.11 came from the accepted PWA Git objects.
The implementation does not depend on the protected checkout HEAD.

Root approved one precise test change in orchestration message `msg_4a5394fffe06`.
The old whitespace denial test now checks proven local completion.
Its zero-send assertion remains unchanged.
[Preservation checks](preservation.json) record unchanged original definitions, signatures, and other test assertions.
Beads access used read-only mode against the canonical repository.
Root retains Beads writes, source acceptance, merge, and closure.

## Runtime contract

The only runtime edits are `agent/native_main_payload.py` and `agent/title_generator.py`.
`NativeAuxiliaryInvocation`, `NativeMainInvocation`, constructor APIs, and existing public adapter signatures remain unchanged.
The new `NativeAuxiliaryLocalResult` is frozen and slotted.
The read-only `auxiliary_local_results` tuple exposes completed local records.
The public `complete_auxiliary_local()` method accepts only the owned pending token and exact registered stage/output objects.

The local proof retains full source bytes, five input records, one output decision, and `b'{"title":null}'`.
Actual helpers record empty-input early return, summary choice, wrapper execution, and the complete snippet.
A local result creates no request, envelope, observed send destination, model response, or model-result handle.
The original title language remains a binding fact.
No model applied it.

Source preparation runs once per operation.
The sent path retains seven input records, four output records, and complete response verification.
The registered frozen preparation and its complete identity anchors replace duplicate source execution at sealing.
One traversal preserves completed local and sent output order, handles, and original capabilities.
Later main/title derivations inherit those outputs through the accepted shared selection bridge.

Local completion uses the existing eight-operation auxiliary budget and current byte/stage bounds.
Shared limits remain 32 parents, 128 handles/envelopes, 8 MiB retained bytes, 1 MiB payloads, and 128 hops.
The shared runtime can refuse before a native limit.
No failed operation gains an invented output parent.
A refusal after publication retains the completed record and releases no output.

## Verification scope

The final focused selection passed 80 cases in 478.1 runner seconds, with zero failures or retries.
[The final command](validation-command.json) selects changed behavior and directly affected compatibility.
[The final log](final-focused.log) and [timing receipt](final-run.json) contain the exact result.
This selection is not a fresh full native or PWA suite.
[Source pins](tested-source.json) bind the four tested files before and after execution.
The [consumer contract](consumer-proposal.md) fixes the supported seam and remaining refusals.

Tests use actual constructed agents, the existing adapter, actual `generate_title()`, and the exact installed dependency.
Local controls block model, dispatch, SDK, client, configuration, persistence, thread, prompt, parsing, and ordinary fallback callbacks.
They cover original fields, complete bytes/identities, policy changes, profile removal, inherited coverage, cancellation, replay, reentry, capacity, and close.
The ordering case proves main A, local title A, main B, sent title B, and later local reuse.
Main model, response identity, token totals, and send counters remain separate from local results.

Edit feedback logs retain initial failures and intermediate selections.
Initial failures exposed detached test text, a nonexistent counter, and an old helper assumption.
Other failures exposed exception assertions inside the policy scope.
Later tests correct those assumptions.
Their counts do not add to the final selection.

No interrupted run occurred in this task.
The historical 198 native cases and 300 shared cases were not rerun.

[The installed smoke](installed-smoke.json) verifies the exact 0.2.0 dependency, engine bytes, API/import, and one representative shared derivation.
Native imports use this task worktree.
This is not an installed native functional selection.
[Setup](environment-setup.log) uses cached locked native dependencies without installing the native project.
[Wheel delivery](wheel-install.log) installs the accepted wheel with offline, no-index, and no-deps options.
No dependency manifest or lock changed.

## Remaining gates

Background, async, streaming, instant persistence, private, and sibling consumers retain refusal.
No real provider inference, live egress, service operation, financial write, build, deployment, or activation ran.
Durable restoration, sensitive cross-thread, source, financial, production, and full P10/P06 qualification remain open.
Configured GitHub checks apply only to the exact published head.
Zero checks or statuses grant no CI success claim.
