# Qualify the corrected P10.45 expiry harness

Run one narrow qualification in the existing P10.45 worktree and retained draft PR21.
Preserve original failed Task `task_b80c8266a131`, Dispatch `ctx_a00e25dedaae`, reports, ledgers, source pins, failures, and original handoff.
This is a separate qualification task. Do not change the recorded failed outcome or retry the entire implementation.

The exact current candidate and executable pins are:

```text
/home/mersinvald/orca/workspaces/hermes-agent/hpwa-p10.45
candidate b1a76fe8d3294a49db4e30edb4b84265d35ebf70
runtime SHA256 e18ed40a2642a6255eea61a3feb1212815a35c11a72e82037744db5698600a6a
test SHA256 636cc79ea3bfba9a8818c15c1065e9c0205fce720e3d7b9078246371283d641e
API SHA256 5916de14b4803a4045fcff7d76e7d5cfe0e9020c37b4d686a15c35f2772650cb
original handoff /home/mersinvald/.local/share/orca/runtime/accountant/hpwa-p10.45/receipts/worker-handoff.json
original handoff SHA256 8424b6ddb89e1d12926d6111e580abe18cb2b5deff93737eb6fbd41942149145
```

Root read the complete new runtime diff, new test file, README, scope correction patch, original scope implementation, and measured ledger.
The prior expiry execution retained actual J and refused correctly at 50 with zero pair results.
Its test then caught the exception inside `content_policy_scope`, so normal scope exit ran the old validation.

The original scope correctly invalidated expired P, causing a later test failure.

The committed harness moves the expected exception outside the scope, as the passing original-control case already does.
The original context manager skips its normal final validation when the body propagates an exception.
This is a concrete harness correction. It does not authorize a scope guard or runtime change.

Root requires the corrected expiry postconditions to execute before source acceptance.

## Exact scope and budget

Keep every runtime, test, old evidence, dependency, fixture, helper, callback, API, and limit byte unchanged.
Run only:

```text
tests/tools/test_native_issued_call_pair.py::test_final_non_main_coverage_after_J
```

Use the unchanged canonical runner, exact function selector, per-file isolation, `maxfail=1`, zero retries, and incremental external receipts.
Set a 290-supervisor-second cap for this single execution.
The prior failing supervisor used 267.7471921660035 seconds, with 213.49102498203865 seconds through Wa.

Do not rerun the successful positive, control, compatibility, setup, delivery, installation, shared smoke, or other histories.

The original task ledger remains 809.214237332992 of 900 seconds, with 90.78576266700804 seconds unused.
Root adds 240 seconds for this concrete missing qualification. The combined component allowance becomes 1140 seconds.
This separate task may use at most 315 supervisor seconds, including the execution, static evidence, hooks, and publication.

The remaining 15.78576266700804 seconds of combined allowance stays reserved for root metadata verification.
Count every failed command and actual supervisor duration. No retry, file split, or new process resets the total.

Use the already complete task-local P10.45 environment unchanged.
Keep the existing isolated home, cache, temporary files, bytecode location, interpreter, and accepted shared wheel.
No source, fixture, environment, import smoke, delivery, install, build, index lookup, or profiler change follows.

If the selected case fails or times out, preserve its exact result and stop this qualification without another genuine run.

## Required observations

Keep original timestamps, child duration, complete actual graph, parent order, handle identities, restrictions, and charges.
Record policy time, eight final-window clock calls, all selected-main deadlines, and the expired required non-main grant.
Require actual J retention, zero released pair results, spent owner state, and all existing corrected test assertions.

Record original P/source/child terminal state without clearing a denial or changing a clock to pass the operation.

Record complete raw output, actual J bytes and ordered `(Wa, Wf)` parents, and separate qualification/data sends.
Do not copy the prior 673-byte or 27894-byte totals into the new actual history.
Retain exact runtime and test snapshots and source hashes for the selected execution.

## Evidence and publication

Add a focused bundle under the existing owned evidence directory:

```text
docs/evidence/accountant-native-issued-call-pair-20261124/expiry-qualification/
```

Preserve the original external handoff and all original Task reports byte-for-byte.
Keep original compressed artifacts, failed scope, inverse patches, source preservation, and original ledger immutable.
Append a concise qualification section to the current README and consumer contract.

Update the current manifest while preserving its previous bytes in the new qualification bundle.
Do not relabel earlier scopes as executions of final runtime or claim a fresh combined whole-file pass.

Run only relevant hash, byte, lineage, scope, link, full-task whitespace, and new-prose STE checks.
No extra functional representative or environment smoke is necessary.
Commit qualification evidence through normal hooks and push normally to the retained same branch and draft PR21.

No force, admin bypass, hook/configuration changes, branch deletion, or merge.

Record exact published head, tree, source/evidence/dependency pins, PR state, zero-check limitations, and clean final task status.
Write a separate external qualification handoff with its own ledger and original report hashes.
Root retains source acceptance, merge, Beads closure, reconciliation, and all wider gates.

Full P10/P06, provider request, native continuation, private/egress/durable/source/financial/production gates remain open.
Report the qualification outcome with `worker_done`.
