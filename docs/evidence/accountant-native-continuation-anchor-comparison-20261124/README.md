# Live anchor comparison evaluation

Outcome: reject the allocation hypothesis at the cheap benefit gate.
This draft contains evidence only.
Native runtime source, existing tests, and the public API remain exact.

The [input verification](input-verification.json) checks 19 Git objects, seven protected receipts, five environment files, and the shared wheel.
The base is `5ceb15437ec9318cbd62903a550d20f8743a3c0f`.
Its tree is `fa9d7c816619fe89f65577ff25ec6d785ac33979`, with 12,012 paths.
The selected proposal has SHA256 `31c02a473cfd6f8a71d9b4d0ec1650e0d9d0b03aa4b4de7d4144c400ddeadde7`.

The [prototype](prototype.py) uses explicit original type and traversal support bindings in a separate namespace.
The unchanged `_nested_anchors` and `_same_anchors` functions supply the independent reference.
No original component receives a patch.
The prototype reads current recognized types on each call and current dataclass fields on each visit.
It consumes every live traversal after an identity mismatch.
The [source pins](source-pins.json) retain the exact prototype, reference, verifier, test, and API hashes.

The [experiment](experiment.py) passed all 27 differential cases.
Cases cover identity, aliases, containers, temporary roots, ordered keys, fields, anchor lengths, multiple roots, mutation, restoration, invalid keys, and recursion.
The metadata cases concern unrecognized namespaces.
Recognized field freshness also has a static source check.
No integrated verifier or authority qualification follows from these pure cases.

The experiment uses the pinned historical request, raw response, and normalized data.
It constructs original recognized records and the existing fixture catalog without preparation, qualification, admission, or send calls.
It preserves destination aliases, declaration order for recognized records, and repeated alias visits.
Nested request dictionaries retain receipt order.
The request root follows the native request field order used by the experiment.
The transport defaults use the same `(None, None)` operands as the original transport normalizer.

Five paired batches alternate reference and candidate order.
Each batch uses 1,200 calls per shape and implementation.
The [result](experiment-result.json) records every batch duration, workload size, median, and median absolute deviation.
The [command receipt](experiment-command.json) records 5.225382267 seconds, including imports and setup, within the 30-second cap.

| Comparison shape | Anchor visits | Reference median, microseconds | Candidate median, microseconds |
| --- | ---: | ---: | ---: |
| Callable defaults | 4 | 3.582 | 2.908 |
| Catalog | 373 | 163.248 | 152.116 |
| Authority | 26 | 14.443 | 12.881 |

| Comparison shape | Anchor visits | Reference median, microseconds | Candidate median, microseconds |
| --- | ---: | ---: | ---: |
| Request | 89 | 32.111 | 28.779 |
| Response | 17 | 10.334 | 9.527 |
| Normalized and usage | 22 | 11.623 | 8.848 |
| Transport defaults | 2 | 2.876 | 2.297 |

The sums of the per-shape medians are 238.215719 and 217.356780 microseconds.
The measured reduction is 8.756324%, below the required 10%.
No shape shows a regression beyond the recorded noise rule.
Defaults take only a few microseconds per call.
These small pure comparisons supply no causal episode cost or full-episode speedup claim.
No additional batch or performance expansion follows the failed gate.

Native implementation elapsed time is zero.
No runtime or test edit, genuine history, installed-native test, shared smoke, model call, or service operation ran.
The historical 544.579277353012-second passing command retains its original source and scope.
This task does not repeat that command or transfer its authority to the prototype.

The [preservation receipt](preservation.json) checks all base paths and modes, five immutable files, and the wheel after the experiment.
The [ledger](ledger.json) retains the wall-clock allowance and validation limits.
The [failure log](required-command-failures.log) retains one corrected static AST check failure.
Python 3.12 and Python 3.11 produce different AST dump schemas.
The corrected static check uses the measured Python 3.11 interpreter.
The comparison experiment ran once.

Beads49, full P10/P06, and private, provider, egress, durable, source, financial, production, manifest, index, deployment, and activation gates remain open.
Both original native50 Tasks remain FAILED with their historical scopes unchanged.
Mac collection and Firefly remain outside this scope.
Root retains acceptance, merge, Beads closure, and every wider decision.
