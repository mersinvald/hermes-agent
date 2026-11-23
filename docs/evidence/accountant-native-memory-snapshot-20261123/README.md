# Native frozen-memory snapshot evidence

P10.29 implements one authenticated read-only capture of invented built-in frozen memory.
The actual `MemoryStore` getter returns its exact string object or `None` on the owning thread.
The original shared runtime retains complete source, request, and result bytes.
Capture makes zero model calls and zero sends.

The selected proposal comes from PWA publication `1b05162d82db17b9f6290ae5bfd0eb27323b711e` and candidate `ca5a9a9d892f61bc0a0075f6d399d3e42a8191a3`.
Its SHA256 is `d65672a0fcd0ff76b97e35f2e793e4c44ca727a1654d8cc32b459710db7aa9d5`.
The source-map SHA256 is `15e1bf2fd7d8824fdee2e614080eacc8f4a73bbafb6fd1ee48ad3fa7c433c77f`.
Native base is `e54325fa147202988164024cf9da33b6c7336b84`.
Source commit is `090f73950df25df5152ccfdfb3024ed1ec083a88`.

[The consumer contract](consumer-contract.md) defines authority, bytes, lifecycle, failure, and terminal behavior.
[The API](native-api.json) records the optional port method, host bind/read methods, opaque token, and ordered records.
[The source freeze](source-freeze.json) pins source, test, API, consumer, dependency, and byte evidence.
[The source boundary](source-boundary.json) verifies 217 original signatures, 28 annotated shapes, and all original constructors.
It also verifies 11522 unchanged other base paths.

[The source patch](source-boundary.patch) uses zero context lines to pass the repository whitespace check.
Use `git apply --unidiff-zero` to apply it and add `--reverse --check` for the inverse check.

The actual parent keeps both memory fields at `None`.
The operator source lookup binds the exact original store, target, producer policy, owner, profile reference, and version.
The host profile reference does not become a content restriction.
The owner checks the original capabilities, ports, methods, producers, catalogs, destinations, full coverage, selected-grant expiry, and runtime health through release.
Retained restrictions survive current profile removal.

The test fixtures load invented files through the actual constructor and loader before binding.
Binding and reading do not load files, change the store, consume a prompt, or call a provider.
Negative tests change test-owned objects only to check refusal.
The narrow test trace observes the unchanged getter frame. It does not replace the getter or run a performance profiler.

[Observed bytes](observed-bytes.json) contain complete source, request, getter, result, empty result, and retained failure observations.
The nonempty source has 1463 bytes.
Its two admissions and one derivation consume three slots and 2153 retained bytes.
Its parents have the exact order `(request_content, source_content)`.
The source is one independent authenticated frozen-memory root.
It does not claim main or child ancestry.

| Fixed value | UTF-8 bytes | SHA256 |
| --- | ---: | --- |
| Request | 19 | `c1115f389258bbd38c28b6d0754ebeaaa4e63fbf5c933e8c49dd68bf285d0b0c` |
| Getter block | 359 | `ccf4ac249b0c29904c33330172bd9f4410bfeccfd3f0590d6bb0cef7abb36e67` |
| Result | 671 | `66decc8c2d60a7c82e086aa93eb2cc0f69ae41049d62fd63421528b97219c10a` |
| Empty result | 32 | `f87661743e6e96ece438d3a69d2ba49aebb1cc8e00661af59a64fd346faa5c2f` |

Each new fixture uses one original P holder and eight genuine catalog qualification dispatches before native binding.
Those qualification sends remain separate from capture.
Capture does not use a memory dispatch route or qualify a provider request.
The default terminal representative uses its unchanged three-holder fixture and 25 qualification sends.
Its two consumer sends belong to that separate historical path, not memory capture.

The first actual nonempty check passed in 16.786 wall seconds before test expansion.
The final focused selection passed all 83 new cases in 85.710 wall seconds.
Two main representatives passed in 5.690 seconds.
One default terminal representative passed in 44.473 seconds.
All selections used the unchanged `scripts/run_tests.sh` with zero retries.

| Separate scope | Outcome | Wall seconds |
| --- | --- | ---: |
| First actual nonempty getter | 1 pass | 16.786 |
| Feedback | 8 passes | 21.217 |
| Intermediate focused selection | 82 passes | 86.542 |
| Added phase-relabel regression before correction | 1 expected failure | 4.901 |

| Final source scope | Outcome | Wall seconds |
| --- | --- | ---: |
| Final focused selection | 83 passes | 85.710 |
| Main representatives | 2 passes | 5.690 |
| Default terminal representative | 1 pass | 44.473 |

The red case observed a second getter call after a phase-label change.
The final source also checks retained result and getter facts before access.
The corrected case denies access before another getter call.
The final expiry cases keep all eight paths live after original selected-grant expiry.

[Validation scopes](validation-scopes.json) retain exact case IDs, source/test pins, outcomes, and timings for every selection.
The new selection records setup, call, and teardown times separately.
The unchanged main representatives record case totals without separate setup properties.
Their successful checks ran once.
Intermediate scopes and the expected failure do not form a fresh cumulative success.

The capacity case adds eight genuine filler admissions without changing runtime limits or private state.
The two capture roots remain charged when result derivation exceeds retained byte capacity by one byte.
The actual getter result and complete result bytes remain retained, without a completed result handle.
Another case retains one genuine source admission after a control exception prevents token issuance.
Late poison cases retain actual getter/result observations and deny release after getter, derivation, and release boundaries.

[Dependency evidence](dependency.json) pins the unchanged accepted `hermes-content-policy==0.4.0` wheel, engine, shared API, and installed RECORD.
One task-owned offline, no-index, no-dependency install delivered that exact wheel.
One small installed shared import/API/admission/RECORD smoke passed.
Native functional checks imported worktree source. They do not qualify an installed native distribution.

Historical shared capacity evidence retains its old scope. No historical 128-admission selection ran again.

[Runner isolation](runner-isolation.json) records external environment, state, caches, snapshots, and raw receipt paths.
Isolation was ready before the first check.
The source runners, fixtures, manifests, locks, hooks, and Git settings remain unchanged.
The original ignored HOME and original native package-lock content remain outside inspection.

Ruff and patch-whitespace checks passed.
[Writing checks](writing-check.json) record the same STE mode for the original and final Python prose.
The manifest covers every published evidence file except itself.

Root owns acceptance, merge, Beads closure, and canonical reconciliation.
Real source association provisioning, private memory, prompt consumption, writes, models, asynchronous transfer, and durable ownership remain unqualified.
Full P10/P06 and all private, egress, durable, cross-thread, source, financial, production, native-manifest, and index gates remain open.
Deployment, workflow activation, and the Mac collector remain outside scope.
