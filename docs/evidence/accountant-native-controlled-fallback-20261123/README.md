# Authenticated terminal native fallback

Source commit `9982e597b8ec93febc17993783e5fa72086994dc` implements one controlled synthetic native fallback.
The actual constructed agent keeps its main model, provider, bindings, and ordinary fallback state.
Only an observed exact main transport failure with original shared proof can start fallback.
The result retains unknown main usage and cost. Later model operations refuse.

## Pins

| Item | Pin |
| --- | --- |
| Native base | `a905202143476bdf2d0ed7d0a7115e373e8121d9` |
| Base tree | `f97bb03862dc41ea469014a347253b1dddefb51d` |
| Accepted shared publication | PWA `9d97078083e213f1df8143aec0ab9137792567d3` |

| Item | Pin |
| --- | --- |
| Current source commit | `9982e597b8ec93febc17993783e5fa72086994dc` |
| Current source tree | `b6394b5debce6f26dc66443c7386ad42847e5e31` |
| Adapter SHA256 | `bcb2bff0a6f4fc33b426078783a39aa892707ec6205a275e0b4c29920b6fc47b` |

| Item | Pin |
| --- | --- |
| Native API SHA256 | `07cc88df9776e220c38f8289d983eea72bea2acdf48cbbe04ecbcf477b3a46d7` |
| Consumer contract SHA256 | `f14fd1113de097354becf239f12be7556b14b45ba6e1222b463964ad9bb0d6b5` |
| Dependency receipt SHA256 | `dc50592a9168b5cc62b4990152a9a27bffca8872cada8344e20afc2670f32c91` |

[Source verification](source-verification.json), [compatibility verification](compatibility-verification.json), and [source commit verification](source-commit.json) bind the owned changes.
[The frozen source](frozen-source.json) records every current runtime and test hash.
[The consumer contract](consumer-contract.md) defines the behavior and exact constructor clarification.
[Input verification](input-verification.json) binds all accepted receipts, proposal bytes, source map, and artifact pins.
[Original requirements](original-requirements.json) retain the original P10 section from its accepted PWA object.

## Dependency delivery

The task uses the accepted `hermes-content-policy==0.3.0` wheel.
Its SHA256 is `d2a98ca9dbf09c78726c1bc75d5097752bb281ab4e922b3ea83062cc8bfddb0c`.
The installed engine SHA256 is `d9c7608d00f9f83c87cf0fa6eb12f6a70453faebf17ef63838d2d297ad2ebace`.
The shared API SHA256 is `db73a486c43ba96c545a227a01b95ba146a2d8c68168782468f597c0e9b2fcf4`.
The accepted artifact receipt SHA256 is `2ea2a22e6883b99efc531be360a7e02b2b97ac93761d30069f90254a67c1d781`.

[The dependency receipt](dependency-receipt.json) records the exact wheel path, offline commands, and unchanged native manifest hashes.
[The smoke](dependency-smoke.json) checks the installed version, engine bytes, and additive API.
The task built no wheel and installed no PWA package.

## Focused checks

Every native selection used `scripts/run_tests.sh`, zero file retries, the exact installed wheel, and an isolated task HOME.
The table records separate source revisions and completed scopes.
[Scope receipts](scopes.json) preserve exact commands, hashes, IDs, outcomes, timings, and original failed scopes.

| Scope | Source | Result | Wall seconds |
| --- | --- | --- | ---: |
| `corrected-focused` | Current corrected runtime and tests | 4 passed | 298.861 |
| `final-native` | Before chronology correction | 36 passed, 1 setup failure | 432.552 |
| `final-history` | Before chronology correction | 1 passed | 887.887 |
| `final-representatives` | Before chronology correction | 6 passed | 29.225 |

The current four cases cover the older-source title chronology, minimal ConnectionError success, failed fallback send, and corrected unqualified sibling checks.
[Their JUnit result](corrected-focused.xml) and [case timings](corrected-focused.timings.jsonl) retain exact IDs and setup times.
The older-source regression took 231.036 seconds. The current complete minimal case took 39.642 seconds.

Root found the chronology issue during ordinary source review.
The previous grouping used the auxiliary source completion as its operation position.
The correction uses original completion identities in each retained auxiliary parent list.
It checks the complete main prefix and monotonic auxiliary order before it constructs fallback parents.
[The correction patch](chronology-correction.patch) contains this sole runtime correction.
[Provenance checks](test-correction-provenance.json) confirm that the original fixtures and 36 passing test bodies remain byte-identical.

The sibling setup failure occurred before its callbacks because the test removed profiles before scope entry.
The corrected case enters the valid scope, removes profiles, and calls each actual sibling in a fresh fixture.
It checks memory, delegate, A2A, tool results, async, streaming, background title, and real-provider refusal with zero sends.

The old native scope retains both exact exception successes, native/shared failure corroboration, drift denials, cancellation, process controls, and partial proof.
It also retains reentry, repeated activation, nested source/request/proof changes, prior successful ancestry, wrong-thread use, terminal reuse, and close checks.
The six unchanged representatives cover ordinary fallback, actual main entrypoints, local title, and compression.
[Their exact IDs](representative-case-ids.json) bind that separate six-case result.
There is no fresh cumulative pass claim for all these cases on the corrected runtime.

The earliest scopes retain three invalid test-sentinel setups and the unsupported illustrative constructor arguments.
The task stopped the original 32-port two-case timing scope at 148.565 seconds without a completion claim.
The separate minimal timing case passed in 97.853 seconds before record-check placement changed.
The 12-case early refusal scope passed in 27.441 seconds. The 13-case control scope passed in 114.068 seconds.
No earlier scope replaces the current four-case check.

## Historical prefix and limits

The separate full prefix completed before its original 15-minute ceiling.
The deadline controller found the process already settled and sent no signal.
[Its JUnit property](final-history.xml) records the complete six-operation prefix.
The terminal turn adds failed main C and fallback.

The result retained 89 shared slots and 95771 bytes, with three main sends, three auxiliary sends, and one fallback send.
Its one failed-main record remains separate from the two successful main completions.
This is synthetic proof on the recorded earlier runtime.

The current regression retains main A, main B, and a later local title sourced from A.
Failed main C then starts fallback.
It checks complete parent order and each original capability after current profile removal.
It preserves earlier successful totals and retains separate fallback usage.

All limits remain unchanged: one pending operation, one fallback attempt, eight main slots, eight auxiliary inputs, and eight input/output stages.
Native source, stage, request, and final payloads remain within 262144 bytes. Raw response bytes remain within 65536 bytes.
Requests keep 32 messages and 16384 UTF-8 bytes per text.
Shared limits remain 32 parents, 128 combined slots, 8 MiB retention, 1 MiB payloads, and 128 hops.

## Static checks and publication

Ruff found zero issues in the changed runtime and test files.
It retained one existing invalid-noqa warning in unchanged code.
The global STE linter reports zero new violations in owned prose.
Existing runtime comment and docstring scores remain unchanged.
The task preserves every original record field list, native constructor, ordinary fallback body, and controlled caller.
The task preserves generic guards, native manifests, lockfiles, shared source, and existing test definitions.

The source commit used normal Git hooks.
A later evidence commit and draft PR retain this source. The publication receipt records their exact pins.
Root owns source and artifact acceptance, merge, and Beads.

Full P10/P06, real providers and egress, private inference, durable restoration, and sensitive cross-thread support remain open.
Source completeness, financial, production, deployment, and activation gates remain open.
The online Mac SMS collector remains outside scope.
