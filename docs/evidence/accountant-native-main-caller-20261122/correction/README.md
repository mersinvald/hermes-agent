# Controlled caller context correction

The corrected source is `8767a634aea9c3b47aef3e08860cc29d0a528738`.
The review candidate was `143d64987cb8aa651b89029a651ae885dd9b665c`.
The reviewed runtime was `8e7fecd24b119a90b2936726358b66acdeb0b0f7`.
The original task remains succeeded within its recorded scope.
This correction implements finding R1 under task `task_841a16c3991a`.
Root retains canonical Beads, acceptance, and merge.

The caller recognizes a retained payload even when its bound marker becomes `False`.
The lower helper could then select ordinary dispatch.
The MoA branch does not call `make_client`, so `refuse_client` could not refuse that branch.
The new local regression retains one accepted invocation, drops current profiles, keeps the payload, clears the marker, and selects MoA.
It requires fixed refusal and no further SDK callbacks or controlled sends.

The caller now requires the captured registered `NativeContentContext` and its captured sealed adapter.
It requires the exact `True` bound marker before native scope validation.
It uses `current_context()` and `validate_native_scope()` before dispatch and before caller release.
It checks attachment identity before and after native scope validation.
It repeats these checks after each cancellation-flag read.

Malformed markers deny without Boolean conversion.
Rejection marks the context denied and calls the existing adapter invalidator without marker conversion.

Only `_controlled_main_api_call` changes in runtime source.
The adapter, native policy, shared engine, ordinary dispatch, and direct caller retain their earlier source bytes.
The 187 accepted P10.5 cases and 86 initial caller cases retain all their test bytes.
The 32 new cases are additive.

`red.json` and `red.log` retain all 32 failures against the unchanged reviewed runtime bytes.
The tests in that run were new working-tree tests with their own hash.
That run is not unchanged `8e7fecd` test evidence.
`green.json` records 32 passing correction cases in 9.2 runner seconds.

`final-focused.json` records 137 passing cases in 34.7 runner seconds, with zero failures, skips, and retries.
Its wrapper elapsed time was 35.082 seconds.
It includes 86 initial caller cases, 32 correction cases, 11 representative adapter cases, and eight ordinary compatibility cases.
Do not add counts from these overlapping scopes.
The full 187-case P10.5 selection was not rerun.

Each report records the exact command and source hashes.
`offline.py` supplies invented temporary configuration and blocks socket and HTTP callbacks across the selected files.
The established caller fixture also blocks SDK, lifecycle, scheduling, heartbeat, and watchdog callbacks for controlled calls.
The final selection retains pre-send and post-call cancellation, process-control exceptions, context invalidation, actual send counts, and accepted internal records.
Caller rejection after a completed invocation does not remove that record or reduce its send count.

The task reuses its existing `.venv` and exact accepted dependency.
The installed engine hash still matches the original installed smoke evidence.
`../installed-smoke.json` and `../resource-preservation.json` retain their original source scopes.
No install, sync, build, duplicate installed selection, or historical suite run occurred.
`consumer-api.json`, `pins.json`, and `SHA256SUMS` identify the current contract and evidence.
The parent API and reports remain historical evidence.

The corrected caller remains synchronous and supports only the accepted invented `synthetic_native` adapter with nonstreaming `chat_completions`.
It has no in-flight preemption, callback deadline, heartbeat, or watchdog.
Full P10, real providers/egress, private/model processing, restoration, sensitive cross-thread support, and financial acceptance remain open.
No production, deployment, live service, financial write, or Beads sync occurred.
The PR remains draft and requires root review.
An empty configured check list does not establish CI success.
