# P10.49 continuation consolidation

Both selected whole harness cases passed on the frozen source.
Each case used a fresh process and source graph.
The task changed evidence only.
Root retains acceptance, merge, Beads writes, and closure.

| Case | Command seconds | Pytest call seconds | Final phase | Windows | Contents | Envelopes | Retained bytes |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: |
| positive | 478.301369 | 468.541408 | released | 6 | 38 | 4 | 32851 |
| coverage_expiry | 331.262805 | 328.232857 | denied | 3 | 35 | 4 | 32125 |

The command total is 809.564174 seconds.
Command time includes runner preparation.
Task wall time includes preparation, waits, static checks, and publication.
The task clock starts at 13:20:25 UTC on November 24, 2026.
The hard task limit is 1800 seconds.

The plan reserves 60 seconds for final work and 90 seconds for other delays.
The floor receipts precede each command.

The [results](results.json) retain exact IDs, commands, PIDs, times, XML metadata, and original case outcomes.
The [pins](pins.json) retain the input, source, dependency, and environment hashes.
The [raw manifest](raw-manifest.json) retains private artifact hashes and modes.

The [positive view](positive.json) and [coverage view](coverage_expiry.json) retain actual graph facts.
These views omit payload text, hexadecimal payload copies, and ephemeral handle identities.
They keep byte counts, payload hashes, ordered parents, bindings, profiles, sends, windows, and stage facts.
The private XML and original property receipts retain the full raw evidence.

The private directory has mode 0700.
Raw files have mode 0600.

The positive test replaces `native._IssuedCallPair.proof_access` inside the active consumer scope.
Its original assertions require `pair_code_changed` and an empty replacement call list.
The original test also checks copied tokens, reentry, a spent attempt, getters, completion, and scope exit.
The passed assertions establish these facts.
No separate getter observation marker exists.

Static source flow connects `issued_call_continuation_results` to continuation `proof_access`, `verify`, and `pair_proof`.
The pair proof checks the pinned class methods before it calls the original getter.
The [qualification map](qualification.json) records the exact source lines and assertion limits.
The denial thread does not qualify sensitive execution across threads.

Coverage expiry retains the original error chain:
`PolicyError('post_send_rejected')`, `PolicyError('callback_failed')`, then `NativeContentPolicyDenied('controlled_native_issued_continuation_coverage_missing')`.
The original assertions establish this chain, live main grants, retained raw output, and absent result records.
The final policy observation records time 50.
The original fallback grants expire at time 50.
All original main grants remain live at that time.

The raw output has 299 bytes and remains outside the content graph after refusal.
The coverage view records its exact hash.

Normalization and completion remain absent.
The runtime ends poisoned in phase `denied`.
Charges remain exact in the coverage view.

Both runner summaries report one passed test and zero failed tests.
Both XML files and original call receipts report a pass.
The retained logs contain no timeout or unhandled callback or thread exception marker.
The canonical runner suppresses successful pytest output.
The retained logs therefore do not certify the absence of every pytest warning.
The task preserves this visibility limit.

The first supervisor used the wrong success text during its own result check.
It searched for `1 passed`, but the canonical runner prints `1 tests passed`.
The private correction retains the original false field and the matching XML, call receipt, and runner summary.
This correction changed no harness and caused no retry.

The task ran no control, selected expiry, guard, host episode, or compatibility case.
Historical failed P10.47 tasks, both native50 failures, and the original P10.54 failure remain unchanged.
Corrected P10.54 live capture remains UNEXECUTED.
This checkpoint does not accept full P10/P06, private inference, providers, egress, durable transfer, or cross-thread support.
Source completeness, financial writes, production, manifests, index delivery, deployment, and activation remain open.
The Mac collector and local Firefly staging remain outside scope.
