# Authenticated frozen-memory snapshot contract

This component captures one built-in memory block on the original thread.
It does not send a request or consume memory in a prompt.
The original constructed parent keeps `_memory_store` and `_memory_manager` at `None`.

The original result-policy port implements the optional `memory_source(P, producer_ref)` lookup.
Ordinary hosts and paths need only the existing `current()` method.
The lookup selects the registered source from the exact original P capability and producer reference.
A supplied store, path, record, block, profile label, or digest does not grant source authority.

`NativeMemorySource` has six ordered fields: `store`, `target`, `owner_ref`, `profile_ref`, `source_version`, and `producer_policy`.
The source record must have this exact type.
Its store and target must be the original argument objects.

The owner must match P. The host profile reference identifies the host profile.
That reference does not add a content restriction.
The source version must be a positive built-in integer.

The producer policy must be the exact original `current(producer_ref)` result, with unchanged revision and restrictions.
Its restrictions must fit the original authenticated P profile set before admission.

The host calls `bind_memory_snapshot(agent, *, store, producer_ref, target="memory")` in the matching active scope.
The target must be the built-in string `"memory"`.
The binder captures original option presence, positional identities, and ordered keyword pairs before default substitution.
It checks built-in types before callbacks or conversions.
The current thread must own an unused, constructed, idle parent.
Delegation, resumption, auxiliary work, compression, fallback, pending work, and terminal parents cannot bind this operation.

The owner registers before callbacks.
The binder anchors the original source record, fields, source lookup, producer lookup, runtime methods, and registered source version.
It also anchors the actual `MemoryStore` class, original getter code and method, namespace, frozen mapping, flags, limits, and block.
The store must enable memory and use built-in Boolean flags and positive built-in integer limits.
Substitution or later mutation denies the operation.
Binding and reading never call the loader or touch source files.

The source bytes contain the complete selected frozen block, target, flags, limits, source facts, producer facts, and original holder/selection facts.
They also retain original option presence.
They do not copy the unrelated user block.
The request bytes are exactly `b'{"target":"memory"}'`.
All bytes use the existing native `_json()` serialization without truncation or a terminal LF.

The original runtime admits one independent source root and one request root under P.
The token becomes available only after both admissions and final binding checks succeed.
`NativeMemorySnapshotBinding` retains the fifteen ordered fields in [native-api.json](native-api.json).
The token exposes no public data fields.

The host calls `read_memory_snapshot(agent, source)` with the original agent and token.
The active public read frame, original call identity, getter method, thread, and callback depth guard access.
Equal arguments or a phase label do not grant access.
The actual getter runs once and returns its exact string object or `None`.
The test observer records the actual getter frame without replacing the getter.

The result bytes contain the complete getter result and target.
The original runtime derives the result with path `memory` and exactly these ordered parents:

```python
(binding.request_content, binding.source_content)
```

The five-field result retains the original binding, exact block object, complete bytes, genuine result handle, and ordered parents.
It creates no dispatch envelope, acknowledgement, usage, model cost, or conversation result dictionary.
An independent source root does not claim fabricated main or child ancestry.

Original P holder facts, model, selection, producers, catalogs, ports, methods, and destinations remain current.
All eight paths remain covered: `main`, `fallback`, `title`, `compression`, `memory`, `delegate`, `a2a`, and `tool_result`.
Source and producer restrictions survive removal from current P profiles.
New restrictions and revision drift refuse.
The original clock must keep the selected grant valid through final release, even with independent live coverage.
The owner checks the anchored original runtime health methods before and after callback, getter, derivation, and release boundaries.

The phases are `bound`, `entered`, `completed`, `released`, and `denied`.
Retained binding, result, and getter observations constrain each phase.
A relabeled phase cannot authorize another getter call.

Any escaping failure permanently denies the owner and invalidates the created participant and context.
Original control exceptions retain identity across shared normalization.
Actual partial getter observations and admission charges remain retained.
No retry, capacity recovery, source reload, replacement runtime, or fabricated completed result follows.

After release, only valid read-only scope validation and idempotent close remain.
Repeated capture, new bindings, model work, tool work, and terminal-parent reuse refuse.
Current profile removal does not reopen authority.
All original native and shared limits remain unchanged.

This scope qualifies invented loaded state only.
Real source provisioning, private memory, prompt consumption, writes, model calls, asynchronous transfer, and durable ownership remain unqualified.
Full P10/P06 and all private, egress, durable, cross-thread, source, financial, production, native-manifest, and index gates remain open.
Deployment, workflow activation, and the Mac collector remain outside scope.
