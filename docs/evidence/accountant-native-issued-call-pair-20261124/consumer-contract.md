# Local issued call transcript

`retain_issued_call_pair(assistant, tool)` retains one local transcript from two original registered converter results.
Call this method on the original P adapter, in its original context and thread.
The frozen, slotted result has seven fields in this order:

```text
assistant_result
tool_result
model
messages
message_bytes
message_content
anchors
```

`issued_call_pair_results` returns at most one original result.
The getter checks the retained proof. It grants no operation window or authority.
Copied records fail before callbacks or attempt registration.
A registered attempt remains spent after release or failure.

The module retains at most 128 owners. It does not reclaim or reset owners.

Both input chains must reach the same original observation object and its original content handle.
The proof includes the actual child, completion, acknowledgment, accepted request, shared return, formatter mapping, timestamp, and converter records.
Matching IDs, text, hashes, arguments, models, or destinations do not prove this association.

P stays closed after source capture. E and F register before Wa freezes the P namespace.
The new operation adds no P namespace attribute.
All original operations, frames, windows, attempts, constructors, caches, and source records keep their terminal states.

`model` retains the exact original F model object.
Each converter keeps its own original model object and proof.
A later provider model has no qualification from this operation.

The input list contains the original assistant converter row, then the original tool converter row.
The original sanitizer runs once after proof and current authority checks.
The operation retains its actual output list, nested anchors, and complete serialized bytes before later checks.
It accepts only the same two rows, nested values, order, and bytes.
Any repair, insertion, deletion, duplicate removal, argument rewrite, or unknown shape causes refusal.

Serialization uses insertion order, strict UTF-8, compact separators, `ensure_ascii=False`, and `allow_nan=False`.
It adds no BOM or final newline. It preserves Unicode, argument text, timestamps, and the actual child duration.
For complete valid one-row converter arrays, the byte check is:

```python
j = b"[" + wa[1:-1] + b"," + wf[1:-1] + b"]"
len(j) == len(wa) + len(wf) - 1
```

This check follows original association. Arithmetic cannot establish association.
The genuine positive produces its own Wa259, Wf415, and J673.
Their lengths happen to match the earlier separate records. Those records do not supply this pair.
[Vectors](vectors.json) retain the new complete bytes, graph, identities, charges, times, and sends.

J has ordered parents `(Wa, Wf)` under the original F request on `main`.
The shared operation independently authenticates P and F through `parent_requests=(original_P_request, original_F_request)`.
Their first six holder facts must match. Retained restrictions join monotonically.
J adds one content and its actual byte charge.
It adds no send, envelope, agent, capability, route, destination, acceptance, return, or authority.

Three new bounded windows cover proof, sanitizer plus derivation, and final release.
Every callback boundary checks original proof, thread, reentry, captured methods, code, configuration, and runtime health.
Checks cover original P/E/R/M/F holders, selections, producers, selected grants, receiver binding, ports, and actual destinations.
Current profile removal does not remove retained restrictions or path coverage requirements.

After resolvers, the operation reads one final policy time.
It checks every original selected grant, then all required paths for every original holder at that same time.
Alternate live coverage cannot replace an expired selected grant.

Failure publishes zero pair results. Earlier charges and source objects remain retained.
Actual sanitizer output stays available when later checks refuse release.
A J charge exists only when shared derivation returns its registered handle.
Missing handles and unknown bytes remain absent or unknown.
The first original control exception keeps its identity through normalization and cleanup.

Native limits remain 262144 bytes per value and 786432 combined input and output bytes.
Each string remains limited to 16384 characters and 16384 UTF-8 bytes.
Shared limits remain 32 parents, 128 combined slots, 8 MiB retained bytes, 1 MiB payloads, and 128 hops.
Capacity refusal grants no allowance or reset.

J is a local two-row transcript segment. It is not a complete provider request or native continuation.
Request integration still needs an original consumer, complete prefix, tool definitions, model, options, and actual serialized request.
Send integration still needs current authority, destination proof, actual transport bytes, limits, and controls.
Response integration still needs the actual returned object and native normalizer.
Consumer integration still needs admitted tool-role continuation.

Full P10/P06, native manifests, index delivery, real inference, private sources, egress, durable restoration, and sensitive cross-thread support remain open.
Source completeness, financial writes, production, deployment, and activation remain open.
The online Mac collector and local Firefly staging remain outside scope.
Root retains source acceptance, merge, Beads closure, and all wider activation decisions.

## Separate expiry harness qualification

The [corrected expiry case](expiry-qualification/README.md) passes once on unchanged runtime, test, and API bytes.
Its actual J has 673 bytes and ordered original parents `(Wa, Wf)`.
At final time 50, required fallback coverage expires while every selected main grant remains live until 100.
The operation retains J and publishes zero pair results. The owner stays spent.

The original terminal predicates prove closed P and child state after the final clock callback.
Exception propagation leaves the original scope before the test catches the expected refusal.
The qualification changes no original guard, denial, clock, callback, frame, or operation.
The original failed task and all earlier scope limits remain exact.
Root retains acceptance and all downstream gates above.
