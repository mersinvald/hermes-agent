# Retained native first-summary evidence

Source commit: `9234aced8395379fa60aefe07cd5d0eb723b906d`.
Native base: `a951a7156c6da9beadd0dff6f0eed85554afd116`.
Selected design: PWA `12d2d7df20a3b8077e5b536d2cea6fdb476d5230`.
Design SHA256: `7b7e52e862be363c3107b14a75f452aa936c358bf41d884fc9a0fb84cbcb1582`.

The implementation returns a proven synchronous first-summary proposal through the existing adapter and dispatcher.
It retains complete source, request, response, transformation, and output proof with independent compression authority.
It does not adopt history or change cached prefixes, rolling/session state, telemetry, cooldown, or compression counts.
The [consumer contract](consumer-contract.md) defines the exact API, state boundary, and unchanged limits.

Root owns source acceptance, merge, Beads closure, and later component selection.
Full P10/P06 and all recorded external gates remain open.

## Validation scopes

| Scope | Passed | Failed | Runner seconds | Pin |
| --- | --- | --- | --- | --- |
| Initial focused selection | 98 | 2 | 663.4 | Initial source |
| Publication correction | 4 | 1 | 306.3 | Current runtime, earlier fixture |
| Corrected source-bound fixture | 1 | 0 | 17.9 | Current source |
| Separate capacity/refusal selection | 26 | 0 | 330.1 | Initial source |

These results do not form a fresh combined suite.
All commands use `scripts/run_tests.sh` with zero file retries.
The receipts contain exact commands, process timings, file timings, and source hashes.

The initial selection found a missing cancellation checkpoint before compression proof publication.
It also rejected the unowned initial history in the test before compression input.
The correction adds the compression-only checkpoint and preserves the actual output records after later cancellation.

The four passing correction cases cover first-summary release, full interleaving, cancellation before publication, and cancellation after publication.
The corrected source-bound case uses one actual native completion with more than 16384 bytes of complete original message text.
Its actual summary prompt exceeds the unchanged request-text limit and refuses without another send or fallback.

The preceding large fixture used four genuine main completions.
It reached the existing 262144-byte main-preparation record limit before compression input.
The [fixture decision](source-bound-fixture-decision.json) preserves that refusal and changes no limit.

The separate 26-case scope covers constructor/binding scalar refusals, empty/truncated/raw-bound output, and mixed auxiliary capacity.
Its mixed case completes seven title-local operations and one compression operation before the ninth input refuses.
It does not roll back shared handles or increase a bound.

The initial focused selection also covers ordinary serialization/budget/prompt behavior, focus, first/iterative/legacy summaries, fallback, and empty/truncated output.
It includes only a few existing main/title/local representatives.
The current correction changes no ordinary compressor source.

- [Initial command and outcome](final-run.json), [raw output](final-focused.log).
- [Publication correction](correction-run.json), [raw output](correction-focused.log).
- [Final fixture receipt](source-bound-run.json), [raw output](source-bound-focused.log).
- [Earlier scopes](intermediate-scopes.json), including every retained failed feedback run.

The [publication patch](correction.patch) and [fixture patch](source-bound-fixture.patch) reconstruct both failed source revisions by their exact hashes.
The original freeze, correction freeze, and API snapshots remain separate.

The 663-second initial selection exceeded the ten-minute target.
The correction reused unchanged scopes and did not repeat that selection.
No historical 80/198/native selection, 300-case shared selection, or cumulative PWA/Accountant/PDF/OCR/storage suite ran.

## Dependency and preservation

The task-owned environment uses cached locked native dependencies without installing the native project.
The first setup command selected an absent dependency group.
The [corrected setup](environment-setup-corrected.log) selects the locked `dev` extra offline.
The [dependency receipt](dependency-receipt.json) records the exact offline/no-index/no-deps wheel delivery.

Wheel SHA256: `b6dd72e900cea284b16f0aee0cb706824ce7e495176fb826c96fb508bfea5e6a`.
Engine SHA256: `576e057d0b59053a254a5a766509526824e14640081fdb31ba58b1b03517a625`.

One [dependency smoke](dependency-smoke.json) checks the installed API, import paths, and an invented shared bridge/dispatch/return.
Native imports use the task source. This is not an installed native functional selection.
No PWA package install, index lookup, wheel build, engine copy, manifest edit, or deployment occurred.

The [input receipt](input-verification.json) verifies the immutable design and protected root receipts.
The Beads command used read-only sandbox mode. No Beads mutation or sync occurred.
Protected lock/package-lock contents were not read.
The metadata receipts compare those files around the commits.

The [API record](native-api.json) preserves every old adapter signature and record field list.
The old test files remain exact byte prefixes of their additive replacements.
The [template comparison](template-preservation.json) verifies exact original prompt expressions and unchanged actual helpers.

The prompt method preserves 21 complete `Assign` operations and one complete `AugAssign` operation.
Its five new assignments read retained facts and the retained date.
The earlier value 24 counts selected value expressions across `_summary_source()` and `_summary_prompt()`.
The [metric receipt](template-metric-correction.json) preserves the earlier report and defines both counts.

Reproduce the source-only comparison from the repository root:

```sh
python3 -I -S -B docs/evidence/accountant-native-retained-compression-20261123/template-metric-compare.py
```

This clarification changes no runtime or validation scope.

Ruff reports no findings. Its unchanged `run_agent.py` directive warning remains in the raw stderr file.
Existing Python STE scores do not increase. New evidence prose has zero findings and no em dashes.
The [STE comparison](ste-comparison.json) uses the same global linter and mode before and after the change.
Raw pytest logs keep their original whitespace.

`checksums.json` binds the evidence files. `source-commit.json` binds the seven owned source/test files.
Publication and configured-check results belong to the exact final PR head.
Zero configured checks grants no CI success claim.
