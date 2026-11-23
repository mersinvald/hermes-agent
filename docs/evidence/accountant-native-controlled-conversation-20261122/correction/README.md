# Reserved-provider constructor correction

Status: R1 implementation complete for root acceptance.
Correction source: `3a8483330014d5a29131b0daf2c8955b8e63fc15`.
Reviewed source: `62d5eaf0033343eb66b749360274bb5246d6d1da`.
Original evidence: `4fe5dea92a40837cb74df0cf92cb24f10f414165`.
Public API SHA256: `01a5bb5b432843b5db10b59d503c05154321b3db550265cde99cfc8382e278c2`.

The [review input](inputs.json) requires only `P10.11-R1-reserved-provider-detection`.
Root withdrew R2 after it checked actual instance exposure.
This correction adds no binding observer or R2 probe.
Original succeeded Task `task_85614dc1044a` and its recorded evidence remain unchanged.

`_controlled_attempt` now recognizes the reserved value in a string subclass through `isinstance` and direct `str.__eq__`.
This check calls no custom equality, Boolean conversion, string conversion, or `strip` method.
Both constructor gates use this detector.
An attempt without the explicit adapter raises `NativeContentPolicyDenied` before ordinary initialization.
The exact-string option validator remains unchanged.
The [runtime patch](corrected-runtime.patch) changes one comparison in this detector.

Five additive cases call the actual `AIAgent` constructor without an active or attached controlled context or supplied adapter.
They cover the exact reserved string, a plain subclass, custom equality, custom conversions, and an exact ordinary provider.
Custom methods and excluded ordinary callbacks fail on use.
The reserved cases also require denial before a deprecated-argument warning.
The ordinary control reaches the first ordinary initialization callback through an explicit sentinel.
The existing exact controlled constructor case completes an invented text turn.

| Selection | Passed | Failed | Runner seconds |
| --- | ---: | ---: | ---: |
| [Red](red.txt), exact reviewed runtime and new tests | 3 | 3 | 3.7 |
| [Failed intermediate](failed-green.txt), incorrect patch location | 3 | 3 | 2.7 |
| [Green](green.txt), corrected detector | 6 | 0 | 2.0 |
| [Current affected selection](current.txt) | 105 | 0 | 49.4 |

The red failures reach `agent/agent_init.py:687`, which calls `_install_safe_stdio`.
They reproduce R1 before normalization or client setup.
The intermediate patch matched the dispatcher comparison and left the constructor detector unchanged.
Its [source pin](failed-green-source.json), [patch](failed-green-runtime.patch), log, and results remain available.
The final source restores that dispatcher comparison and corrects only the constructor detector.

The red and green selections contain the same six IDs.
They overlap the final selection and add no independent passing counts.
The final selection contains the original 100 IDs and five additive constructor cases across six files.
It directly tests the corrected detector.
It is not a historical cumulative suite.
The estimated 142-test denominator is not the executed count.

[Selected IDs](selection-ids.json) and [per-file outcomes](current-results/) record all 105 results.
The small [result recorder](record_selection.py) records IDs and outcomes without changes to execution.
The [current manifest](current-source.json) pins runtime, tests, API, and recorder bytes.
The [red manifest](red-source.json) separately pins the exact reviewed runtime and new tests.
[Run receipts](verification.json) retain commands, source references, counts, overlap, and practical limits.
Each individual run receipt also records elapsed wall time and its exit code.

All runs use `scripts/run_tests.sh`, invented temporary configuration, blocked network callbacks, and zero retries.
No source edit overlaps an imported test scope.
The original 100 passes in 49.3 seconds remain valid only for their exact original source.
[Preservation evidence](preservation.json) confirms unchanged original files and assertions.
The parent README contains only an inserted current notice.
The original `checksums.json` and result files remain byte-identical.

The task uses the existing environment and unchanged accepted installed wheel.
It includes no install, sync, build, index lookup, or repeated installed functional selection.
The API, result fields, shared engine, dependencies, and native manifests remain unchanged.
[Scoped Ruff checks](ruff.json) report zero F/E9 diagnostics.
The STE checks use the same global linter and Python prose mode as the original evidence.
These checks do not certify technical correctness.

Root retains acceptance, canonical Beads, and merge ownership.
PR6 remains draft and unmerged on the existing feature branch.
The worker reports the exact published head and its configured checks at handoff.
Zero checks do not imply CI success.

Full P10/P06, real providers, egress, private processing, durable restoration, and source or financial qualification remain open.
This task includes no service operation, model processing, financial write, production test, deployment, or activation.
It changes no hooks, saved Git configuration, upstreams, credentials, other checkout, or Beads state.
