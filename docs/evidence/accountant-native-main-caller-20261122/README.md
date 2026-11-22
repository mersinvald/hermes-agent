# Controlled native main caller

The current correction uses source `8767a634aea9c3b47aef3e08860cc29d0a528738`.
Read [correction/README.md](correction/README.md) and [correction/consumer-api.json](correction/consumer-api.json) for its validation contract and exact pins.
The correction retains 32 red failures and records 137 passing cases on the corrected source.
The text below and the initial reports describe source `8e7fecd24b119a90b2936726358b66acdeb0b0f7`.
The initial `SHA256SUMS` applies to evidence commit `143d64987cb8aa651b89029a651ae885dd9b665c`, before this README update.

This component forwards controlled calls through `interruptible_api_call(agent, api_kwargs)` on the thread bound to the adapter.
It extends the accepted P10.5 payload adapter at native base `5a68423dfab155604999f4da223ff8cde22dbe46`.
The canonical design is PWA base `df34b86c05c671d3e7832a7822fe8e54a77a70fd`, original P10 and architecture sections 17.4/17.5.
Root selected this component and retains acceptance, Beads writes, and merge.

The caller detects controlled attachments in the active scope or agent namespace, and the reserved `synthetic_native` provider marker.
It checks these markers before the ordinary selector, worker creation, or lifecycle callbacks.
The existing `current_context()` resolves and validates controlled context identity.
The new private helper `_controlled_main_api_call(agent, api_kwargs: dict, *, context)` calls the accepted dispatch helper exactly once.
Its nested `refuse_client(*args, **kwargs)` denies any attempt to construct an SDK client.

The accepted adapter still checks actual mode/provider, complete request bytes, selected destination, producer policy, current bindings, and original response identity.
The caller adds no destination, route, or policy decision.

The nested `check_interrupt()` reads `_interrupt_requested` before dispatch and after the helper returns.
A missing flag means `False` for the bounded fixture.
The caller supports only exact Boolean values.
`True` causes `InterruptedError` and context invalidation.
Other values cause the fixed reason `controlled_native_interrupt_invalid`, without truthiness or conversion.

Controlled exceptions invalidate the context and propagate.
Cleanup does not suppress `KeyboardInterrupt`, `SystemExit`, or `GeneratorExit`.

Cancellation after an authorized callback preserves the actual send count and any accepted internal invocation record.
The caller raises before it releases that result.
An accepted internal record does not establish successful caller release.
Later use of the invalidated context still denies after cancellation clears.

This component is synchronous.
It has no in-flight preemption, asynchronous interruption, callback deadline, heartbeat, timer, worker, abort hook, or watchdog.
The function name grants none of these properties.
Wrong-thread calls deny rather than transfer capability.
Only nonstreaming `chat_completions` with a current operator-bound `synthetic_native` adapter has positive controlled support.
The function `direct_api_call()` remains unchanged and gains no separate controlled qualification.

Every earlier accepted result remains a derive parent.
Current source-profile drop cannot remove retained lineage restrictions.
All eight shared paths remain mandatory.
Startup, conversation, prologue, streaming, other modes/providers, fallback, title, compression, memory, delegate/A2A, and unqualified tool results remain refused.
The new cases enter a valid controlled scope before actual startup and sibling refusal checks.

Ordinary calls retain their selector, dispatch path, activity, and client cleanup.
This includes ordinary missing-context refusal after the ordinary lifecycle starts.

The additive tests use complete invented canonical bytes and the actual caller and accepted adapter.
They block network access and fail on controlled scheduling, heartbeat/watchdog construction, direct selection, or SDK lifecycle callbacks.
The exact wheel and unchanged engine pass one installed dependency/API/import smoke.
The task uses a new environment with cached locked dependencies, plus the local wheel through `--no-index --no-deps`.
No package-index lookup, native build, engine copy, checkout import path, PWA server dependency, or manifest change occurs.

The final selection passed 105 cases at source `8e7fecd24b119a90b2936726358b66acdeb0b0f7` in 25.9 runner seconds.
It includes 86 additive caller cases, 11 representative accepted-adapter cases, and eight affected ordinary compatibility cases.
Failures, skips, and retries were zero.
The wrapper elapsed time was 26.301 seconds.
The selection does not rerun the full 187 P10.5 cases or historical native/PWA/Accountant suites.

`final-focused.json` records the exact source commit, command, hashes, deadline, and exit code.
`final-focused.log` records the complete final selection.
`checks.json` separates reproduction, intermediate feedback, and final results.
`pins.json`, `consumer-api.json`, and `resource-preservation.json` bind the sources, API, and allowed changes.
`writing-check.json` records the STE linter comparison.
The linter checks mechanical English patterns and does not certify technical correctness.

Full P10, real providers/egress, durable/serialized restoration, sensitive cross-thread support, source qualification, and financial acceptance remain open.
No private processing, model call, production activation, deployment, or financial write occurs.
The draft PR publishes source for root review.
No empty GitHub check state establishes CI success.
