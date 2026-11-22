# Controlled native main payload component

Issue: `hpwa-p10.5`. Root approved design commit `82273832f186289c95401eaaa2e66065e48ee93c` in message `msg_a5ae36e16d64`.
Native base: `a31286bd59f8d701676c1cca82a9e699e416fb7c`.
Canonical PWA contract: `1219df6a72b29920741a5ea91e19955300723b65`.
The final source commit appears in [checks.json](checks.json).

The actual `_dispatch_nonstreaming_api_request()` now intercepts one explicit controlled operation before its existing generic guard.
It supports synchronous nonstreaming `chat_completions` with provider `synthetic_native`.
The operator must bind a current authenticated native context and complete controlled shared policy.
The adapter sends complete canonical request bytes through shared `dispatch()`.
It retains complete response bytes, accepted opaque handles, and the exact frozen native response object.

The selected helper returns a native response after shared acceptance and current fact checks.
It never constructs a client or calls the original SDK branch for a controlled request.
Unsupported mode, provider, attachment, or shape invalidates the context and raises.
The reserved provider marker without a valid adapter denies before native client construction.
Ordinary providers without a controlled adapter retain their existing behavior.

## Current facts and retained results

The adapter checks current owner, execution, lineage, consumer, authentication epoch, policy revision, selection, destination, and producer policy.
It repeats checks around callbacks and before response release.
It also checks the actual agent mode, provider, request model, attachment identity, trusted port identities, and request bytes.
It rejects reentry and transfer to another thread.

Each later request derives from every earlier accepted result handle.
The shared runtime retains earlier profile restrictions when the current source profile tuple becomes empty.
The native adapter retains and revalidates the original response object before reuse.
Unknown producer policy or any changed producer policy denies reuse.
The first producer profile tuple must be a subset of the original authenticated source profiles.
Additional output restrictions deny this bounded component.

An attached adapter seals its context against every generic native operation.
This seal remains after the current source profiles become empty.
Rejecting a copied or mismatched controlled context also invalidates its attached adapter and original context.
Scope activation checks current facts without granting a generic operation.

The internal scope helpers are `validate_native_scope(context)` and `validate_controlled_scope(context: NativeContentContext) -> None`.
The internal invalidation helper is `invalidate_controlled_context(context: NativeContentContext) -> None`.
The selected interception uses `dispatch_controlled_main(agent: object, api_kwargs: dict) -> tuple[bool, NativeMainResponse | None]`.
[consumer-api.json](consumer-api.json) records the exact native types, signatures, limits, and source hashes.

## Bytes and path coverage

Requests carry the complete accepted native request, API mode, and selected destination in canonical UTF-8 JSON.
The request limit is 262,144 bytes.
Results require one canonical textual response, with a 65,536-byte limit, eight container levels, and 2,048 JSON nodes.
The adapter retains at most eight accepted invocation records.
Unsupported shapes deny before custom conversion, equality, hashing, or callback release.

The operator supplies genuine invented in-process ports for all eight shared paths.
Each positive fixture qualifies those exact port instances through their declared shared operations before native checks.
Delegate and A2A qualification uses current authenticated receiver bindings.
The adapter creates no sibling routes or qualification references.
Every supplied profile retains the exact eight-path requirement.

These checks establish controlled shared-port coverage.
Only the selected native main function has positive controlled support.
Actual native fallback, title, compression, memory, delegate, A2A, tool results, streaming, auxiliary calls, and prologues continue refusal.
Sibling tests enter the controlled scope successfully before they invoke the actual function.
Conversation startup remains outside positive support.

Native `send_count` counts a controlled callback invocation even when later checks reject its result.
The evidence includes post-send stale facts, malformed responses, callback failure, and zero result release.
Shared-port qualification operations have separate retained records.
Their counts do not form part of native `send_count`.

## Checks and delivery

[checks.json](checks.json) records focused runner commands, exact source pins, case counts, timing, and checkpoint limits.
The final complete selection covers the new component and directly affected ordinary and refusal compatibility cases.
Earlier edit checkpoints retain their own source limits.
[writing-check.json](writing-check.json) compares existing Python prose with the same-linter baseline.

The task-owned `.venv` uses Python 3.11.15 and the exact accepted local wheel.
The installed engine SHA-256 remains `28fa3b07c5d6f8a64b9aa970fbdb8a212ac2e7a03d9e4fa6d3020c9ddf0554aa`.
The wheel SHA-256 remains `c10b082397cc05e9cdaf5c1991b82c0834af3d78729eb0c91b53aaae96b7c2bc`.
The install used `--no-index --no-deps`.
No PWA server distribution, engine copy, checkout import path, or manifest change supplies the dependency.

Historical P10.2 evidence retains 137 native passes and 21 selected compatibility passes at source `3b4fe67e`.
Those earlier checks establish ordinary control binding and sensitive refusal only.
They establish no positive native payload propagation.
This task did not rerun the full historical selections.

## Remaining boundaries

Production ingress, configuration, real provider transports, deployment, private inference, and financial activation remain outside this component.
Full P10, real destination/egress qualification, durable restoration, serialized transfer, and sensitive cross-thread support remain open.
Other native boundaries, existing tests, shared source, dependency manifests, hooks, and Git configuration retain their bytes.
The original checkout and its concurrent `package-lock.json` contents received no reads or edits.
The feature branch remains local until root reviews the completed source and confirms publication.
Root retains Beads writes and final acceptance.
