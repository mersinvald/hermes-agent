# Probe consumer contract

`NativeMainPayloadAdapter.bind_tool_call_probe()` admits one operator authorization under the active original P scope.
`run_tool_call_probe()` spends that original token once on its owning thread.
Both methods use the original unused constructed `AIAgent`, its original adapter, and its original constructor anchors.
The agent keeps empty tool lists and unchanged prompt caches.

The opaque slotted token has no public fields or direct constructor.
The binding and observation records are frozen and slotted. Each has exactly fourteen ordered fields with the selected annotations.
The read-only `tool_call_observations` property returns a bounded tuple of the original registered observation.
Equal records, copied tokens, copied context, and phase labels grant no window.

The exact original ports, clock, catalog, runtime, transport wrappers, selected main destination, holder facts, and producer policy remain current.
The operator qualifies all eight paths with genuine shared sends before probe binding.
The fixed positive uses eight qualification sends and one separate probe send.
The selected-expiry fixture uses nine qualifications. Its alternate main route cannot replace the original selected grant.

The binder checks exact built-in strings before operator callbacks.
It requires the original task ID and canonical JSON arguments with exactly `context`, `goal`, and `role` keys.
Context is null. Goal is nonempty and bounded. Role is `leaf`.
Duplicate keys, unknown keys, subclasses, unsupported values, invalid scalars, and noncanonical strings refuse.

The helper creates its fixed request internally. Only this synthetic request advertises the fixed `delegate_task` schema.
No caller supplies tools, tool choice, history, streaming settings, or destination configuration.
The registered execution schema stays unchanged.

The actual `interruptible_api_call()` calls `_controlled_main_api_call()` and `_dispatch_nonstreaming_api_request()` synchronously.
The exact registered branch in `dispatch_controlled_main()` uses the original pending operation and checked transport.
It calls the original shared dispatch once. It pins caller frames, request identity, adapter, methods, envelope, resolved destination, and returned observation.
No SDK, network, provider, worker, executor, registry dispatch, public conversation, or tool function runs.
Boundary cancellation checks cannot preempt an active callback or impose an in-flight callback deadline.

Ordinary `_request`, `_history`, `_response_shape_base`, `_response_shape`, and `_decode_response` bodies remain byte-identical.
The sole old-signature exception extends the `dispatch_controlled_main` return union with `NativeToolCallObservation`.
Its parameters/defaults and every other old signature remain unchanged. All old record fields and annotations remain unchanged.
The observation passes through the caller as the original object. It is not a text completion or a 29-field conversation result.

## Complete retained graph

All native JSON uses sorted keys, ASCII escapes, compact separators, UTF-8, and no trailing LF.
The [vectors](vectors.json) contain every byte. The fixed five payloads total 3265 retained bytes.

| Stage | Bytes | Public operation and ordered parents |
| --- | ---: | --- |
| A | 403 | `admit(P, authorization_bytes)` |
| Q | 506 | `derive(P, (A,), request_bytes, path="main")` |
| W | 707 | `derive(P, (A,Q), wire_bytes, path="main")` |

| Stage | Bytes | Public operation and ordered parents |
| --- | ---: | --- |
| R | 444 | Actual `dispatch(P, W, path="main", route_ref="main_route")` |
| O | 1205 | `derive(P, (A,Q,W,R), observation_bytes, path="main")` |

The graph uses five content slots and one genuine envelope.
R is the original shared response handle. The observation retains the normalized request, wire wrapper, envelope, resolved destination, and actual response.
It also retains the original decoded call strings and complete observation bytes.
The fixed response has one assistant call, null content, `finish_reason="tool_calls"`, the original model, and usage 5/3/8.

The parser checks response size, canonical encoding, scalar types, unknown fields, duplicate keys, one choice, and one call.
It checks the function name, call ID, canonical matching arguments, and actual usage totals.
Malformed protocol bytes retain R and actual charges before refusal.
A trusted send failure can retain its envelope and attempted send without a returned response handle.
Late failures retain every genuine stage that already exists.

## Lifecycle and limits

One attempt registers before callbacks. Eight guarded windows cover binding, run entry, request, wire, caller, observation check, caller return, and release.
The genuine inner scope validates exit before release. Scope exit adds no data stage or slot allowance.
The external P scope remains active after the helper returns.

Original selected expiry, full path coverage, retained restrictions, controls, and runtime health govern every checkpoint through final release.
Current profile removal cannot remove retained restrictions. Unauthorized producer restrictions refuse.
The probe send consumes existing main capacity and increments the existing `send_count`.
Legacy main invocation and completion records stay separate and unchanged.

An escaping bind/run failure permanently denies the owner and context.
Original trusted `InterruptedError`, `CancelledError`, `KeyboardInterrupt`, and `SystemExit` objects survive shared error normalization.
The implementation does not clear flags, retry, delete partial charges, replace responses, replace runtimes, or restore authority.
Successful release closes all new work on the participant. Read-only proof and idempotent close remain available.

Native limits remain eight main/auxiliary slots and stages, 262144-byte values, 65536-byte responses, and 32 messages.
Text, including complete arguments, remains bounded by 16384 characters and 16384 UTF-8 bytes.
Shared limits remain 32 parents, 128 combined slots, 8 MiB retained bytes, 1 MiB payload, and 128 hops.
The accepted shared package remains `hermes-content-policy==0.4.0`.

Actual execution, tool-role construction/consumption, public conversation integration, memory reuse, fallback, and multi-call loops remain unsupported.
A later round trip requires a separately selected original issued-call-to-execution/result association.
Parent text resumption does not supply that relation.
No configuration, route, permission, dependency, manifest, lock, hook, package, artifact, or shared API publication changes follow.
