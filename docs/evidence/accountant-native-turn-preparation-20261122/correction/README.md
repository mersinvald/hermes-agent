# Exact runtime labels for controlled turn preparation

This correction addresses root finding `P10.9-R1-runtime-label-types` on reviewed commit `3ed4ef6d8011b1a40866eb7b08fb110c5e2f5f29`.
Root supplied a source finding and ran no runtime test or probe.
Root retains acceptance and merge.
The original task remains succeeded within its recorded scope.

The turn checks previously compared actual mode and provider labels before exact string guards.
An unsupported object or string subclass could compare equal and receive local preparation proof.
The actual model property also reached equality without a type guard.
A valid model string in the agent namespace did not validate that separate property result.

The correction captures each actual scalar once in its existing check.
It requires exact built-in `str` before equality.
The actual model must still equal the selected destination model.
The existing fixed denial reasons invalidate the context and pending token.
Public signatures, byte formats, sticky mode, state snapshots, and ordinary/direct paths retain their existing contracts.

## Regression scope

The appended tests use `prepared_world`, invented configuration, network blocks, and excluded callbacks that fail on use.
They cover admission, pending prologue, and pending materialization before and after one accepted send.
Mode and provider cases use an equality object, an equality string subclass, and a plain string subclass.
Custom equality returns `True` and records each call.
Custom string conversion also records each call.

The model fixture supplies an exact namespace string and a separate controlled property result.
It tests the same unsupported values and an exact string with a different model label.
Denial must precede custom equality, conversion, sanitizer calls, new frames, new shared content, and further sends.
The context must remain denied after another local operation and an attempted caller send.
The model property must have exactly one read during the rejected operation.

## Red, green, and current records

The [red manifest](red-source.json) pins unchanged reviewed runtime and the exact appended test bytes.
The [raw red log](red.log) records 54 failures and 6 passes, with 187 cases deselected.
Thirty-six cases returned without denial.
Eighteen prologue cases reached the blocked sanitizer.
The six different-model cases passed on the reviewed runtime.
These are runtime regression failures, distinct from the frozen initial fixture failures.

The red report does not target a fixed failure count.

The [green manifest](green-source.json) pins corrected runtime and the same test bytes.
The [green log](green.log) records all 60 new cases passing.
The [current manifest](current-source.json) and [current run](current-run.json) identify the final affected selection.
The [verification receipt](verification.json) supplies exact commands, counts, times, overlap, and limits.
Green and current source/test hashes must match.
These overlapping counts are not additive.

The final selection repeats the initial 208-case filter and explicitly includes both new regression names.
The initial 208 record remains unchanged in the parent directory.
This selection does not repeat the historical 187/233/105/137 suites or installed-package selections.
The existing task-owned `.venv` uses unchanged `hermes-content-policy==0.1.0`.
No installation, dependency sync, build, package lookup, or shared-engine change follows.

## Pins and preservation

The current [API receipt](native-api.json) retains the unchanged public API.
The [API comparison](api-preservation.json) checks complete public definitions against the reviewed Git objects.
The [preservation receipt](preservation.json) verifies the frozen initial checksums and all protected source paths.
Every original turn test remains a complete prefix of the appended test file.
All initial logs retain their exact bytes and whitespace.

The parent README has only an inserted current notice.
The parent checksum file remains frozen and does not certify current bytes.
The correction checksum file covers current source, tests, and correction evidence except itself.

The same STE linter checks baseline and current prose.
Its score measures mechanical prose patterns and does not certify correctness.
Whitespace checks exclude raw captured logs.
Ruff checks the two changed Python files with `F,E9`.
Normal commit hooks also run.

## Practical limits

Draft PR5 stays draft and does not merge.
Exact-head configured checks require a separate report after publication.
Zero checks do not establish CI success.
Full P10/P06, conversation startup, real providers/egress, private/model processing, restoration, and sensitive cross-thread support remain open.
Source completeness, source/financial qualification, financial writes, activation, and deployment also remain open.
No live service, offensive probe, Beads write or sync, configuration change, new authority framework, or new metadata source follows.
