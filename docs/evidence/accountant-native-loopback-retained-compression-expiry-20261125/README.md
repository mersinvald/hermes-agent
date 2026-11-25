# Native compression expiry after a complete HTTP response

The bounded current0.4 proof passed once. It returned no compression summary.
Only the selected compression route expired after the complete raw HTTP response.
All sibling routes and the same-selection alternate remained live.

The first refusal came from `now < route.expires_at` in native `_compression_snapshot` at line 3140.
The original exception contexts retained four objects, from outer to inner:

1. `NativeContentPolicyDenied("controlled_native_payload_denied")`.
2. `PolicyError("post_send_rejected")`.
3. `PolicyError("callback_failed")`.
4. `NativeContentPolicyDenied("controlled_native_payload_invalid")`.

The shared selected-grant predicate remained downstream and did not execute.
The synthetic fixture clock advanced once, from 10 to 100.
The hook inspected pending bytes and the original forwarding observation before that transition.
It did not call an adapter operation or policy method.

Qualification used 25 routes, 50 distinct contents, and 25 envelopes.
Two qualification sends used HTTP. The other 23 used original in-process ports.
One original port handled two qualification sends.
Qualification retained 7241 bytes, separate from native data charges.

The genuine main turn completed through the original constructed agent.
The partial compression record retained its original source, ordered parents, seven input stages, request, envelope, and raw response.
Shared retention ended at 15 contents, two envelopes, and 29787 bytes.
The 312-byte response remained outside shared content retention.
No decoded compression response, result handle, output stage, or summary existed.

Both receiver windows ended after qualification and data responses settled.
Each window contained two accepted connections and an empty quiet suffix.
Both listeners, accepted sockets, and receiver threads closed.
The agent closed twice, then refused another turn without another send.
These observations make no packet or kernel absence claim.

The test reused the accepted positive71 proof and response recipe without another execution.
Its original recipe failure remains recorded in [controls](controls.json).
Both new controls passed before source freeze. The genuine history passed with zero retries.
No executable edit followed that history.

All 12255 prior blobs and modes remain exact.
Production code, shared code, APIs, fixtures, runner, pytest core, clocks, registered methods, and limits remain unchanged.
Raw logs, XML, JSONL, headers, and socket metadata remain in the private receipt directory.

See [contracts](contracts.json), [operator facts](operator-table.json), [results](results.json), and [costs](costs.json).
See [pins](pins.json), [source freeze](source-freeze.json), [preservation](preservation.json), and [checks](checks.json).
The [writing check](writing-check.json) records the prose check. [SHA256SUMS](SHA256SUMS) covers the other eleven files.

This result accepts no full P10 or operational gate.
P06, private, provider, egress, durable, thread, source, financial, production, manifest, index, deployment, and activation gates remain open.
Root owns acceptance, merge, Beads closure, and canonical reconciliation.
