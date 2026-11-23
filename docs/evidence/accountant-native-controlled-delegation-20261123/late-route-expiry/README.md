# Expiry of original delegation routes

This follow-up closes one late expiry boundary in the registered native delegation frame.
It starts from PR11 head `93d88a4aa29d34fac1bf7e7360e08a0700cf878b` and tree `d4bf8f415e2f5d4e4c51a89713ccb24d09df6030`.
The original successful task and its evidence remain unchanged.

The additive fixture declares the issued delegate route with expiry `20` before qualification or binding.
All 24 independent P/R/M coverage routes retain expiry `100`.
The original declared clock starts at `10`.
Its existing completion callback advances the clock to `20` only after the actual child closes.
Shared return has already completed at that point, and the original child proof remains sealed.
No destination or proof changes.

| Scope | Outcome | Wall seconds |
| --- | --- | ---: |
| `late-issued-expiry-red` | Expected failure on the exact original runtime source | 36.260 |
| `late-issued-expiry-green` | Four focused cases passed on corrected source | 113.353 |

The red case reached phase `released` and returned the aggregate at the issued route expiry.
The genuine completion, shared return, and closed child all remained present.
Independent coverage remained unexpired.
The retained test properties confirm one delegate send, one child main send, zero return sends, and 25 qualification sends.

The runtime patch adds only these two requirements after the original clock observation and integer check:

```python
_require(now < self.binding.delegate_route.expires_at)
_require(now < self.binding.child_main_route.expires_at)
```

All capability, selection, profile, destination, coverage, anchor, and cancellation checks remain in place.
The shared engine, public API, original clock observation, and all other frame methods remain unchanged.
No new runtime callback or authority exists.

The green expiry case denies aggregate release at phase `returned`.
It retains the same genuine completion, shared return, closed child, sealed proof, and exact send counts.
The native participants remain invalid after denial.
The selected child-main route remains fresh in this regression, which isolates expiry of the issued delegate route.

The other fresh cases cover the genuine blue-squares success, closed-child cancellation, and original raw-admission interruption identity.
The selection uses `scripts/run_tests.sh`, zero retries, and task-owned HOME under `.task-home/late-route-expiry`.
The JUnit receipts retain exact case IDs, setup/call durations, source/test hashes, and actual observations.
The `record_property` compatibility notice does not remove those recorded properties.

[The comparison receipt](comparisons.json) proves that removal of the two new predicates reproduces the entire original runtime file.
The prior test file remains an exact byte prefix, including every original fixture and assertion.
The original raw-interruption wrapper AST and its inverse source comparison also remain unchanged.
[The baseline](baseline.json) pins every prior evidence file, and the follow-up verifies their original hashes.

[The source freeze](source-freeze.json) records the corrected source and unchanged API/consumer/dependency pins.
The installed shared engine retains its accepted hash.
This follow-up repeats no dependency delivery or installed functional smoke.
It restarts no 53-case, cumulative native/shared, or historical selection.
Prior results retain their original source and scope, with no fresh aggregate claim.

Only `agent/native_main_payload.py`, the additive test suffix, and this new evidence directory change.
The pre-existing untracked `.task-home/` remains outside Git and outside the correction.
The publication receipt records the updated draft PR and exact source publication.
Root retains source acceptance, merge, and canonical Beads authority.
No merge or Beads closure follows this task.

Full P10/P06, private, egress, durable, cross-thread, source, financial, production, native manifest, and index gates remain open.
Deployment, activation, and the online Mac SMS collector remain outside scope.
