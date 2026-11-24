# Fresh record-type membership evaluation

Outcome: reject the fixed membership prototype at the benefit gate.
This draft contains evidence only.
Native runtime source, existing tests, and the public API remain exact.

The base is `0847ff73eadfabcc92b62504674eae670f0fcc41`.
Its tree is `d42e6b937f82868f24f03a916d7666cd3f32095e`, with 12,027 paths.
The [run receipt](run.json) records the input pins, budgets, checks, and preservation scope.
The input receipt has SHA256 `361f3b2008fa3bcd49edb929ce5bcabc0551a401897df433fa3712d8c9f89470`.

The [prototype](prototype.py) keeps the original record tuple and traversal branches.
Each invocation creates its own lazy membership index after eight original tuple checks.
An explicit loop requires every record entry to have the exact builtin metaclass `type`.
Later lookups use the index only when the item type also has the exact builtin metaclass.
All other lookups retain the original tuple equality protocol.
The prototype reads fields on every recognized record visit and retains exact `object.__getattribute__` access.

The [experiment](experiment.py) compiles only the accepted pure walker from pinned Git source as its independent reference.
Both namespaces receive explicit original type and traversal support bindings.
Both comparison paths use the original `_same_anchors` function.
No original component receives a patch.
The namespace protocol cases use local classes and restore their metadata.

All 104 differential cases passed before timing.
The cases compare exact tuple lengths and each reference identity.
They cover every recognized record type, aliases, roots, keys, containers, opaque values, mutation, restoration, and metadata changes.
They also cover later failures, recursion, custom metaclass equality order, truth conversion, reflected comparisons, and unhashable classes.
Custom hash methods must not run.
Class bases and instance classes changed successfully in local cases on Python 3.11.15.

A local plain class rejected a metaclass change on that interpreter.
Metadata changes between alias visits still required fresh field reads.
Nonclass record aliases and nonplain record catalogs retained tuple membership.
These pure checks supply no native authority qualification.

The data builder repeats the pinned P10.52 construction exactly.
It uses the same seven shapes, sizes, field order, and alias relationships.
Historical request, response, and normalized bytes come from the pinned receipts.
Only the existing fixture constructor ran.
No preparation, qualification, admission, send, storage, model, or service operation ran.
The old experiment main and its rejected allocation candidate did not run.

The single timing command took 5.525281442 seconds, including imports and setup.
Five paired batches alternate order and use 1,200 repetitions per shape.
The [result](result.json) retains every batch, median, and noise value.
The first callable-default reference batch contains a large outlier.
The report retains that batch and applies the fixed median rule without another measurement.

| Shape | Anchors | Reference median, microseconds | Candidate median, microseconds | Candidate change |
| --- | ---: | ---: | ---: | ---: |
| callable_defaults | 4 | 3.624193 | 3.850495 | +6.244194% |
| catalog | 373 | 169.553637 | 135.787106 | -19.914955% |
| authority | 26 | 14.359456 | 15.261450 | +6.281534% |

| Shape | Anchors | Reference median, microseconds | Candidate median, microseconds | Candidate change |
| --- | ---: | ---: | ---: | ---: |
| request | 89 | 32.598863 | 27.419366 | -15.888583% |
| response | 17 | 10.424297 | 11.801976 | +13.216030% |
| normalized_usage | 22 | 11.902551 | 15.005884 | +26.072843% |
| transport_defaults | 2 | 2.941996 | 3.172273 | +7.827226% |

The sums of the per-shape medians are 245.404993 and 212.298549 microseconds.
The aggregate reduction is 13.490534%, below the required 20%.
The normalized/usage shape regressed 26.072843% and exceeded the fixed noise allowance.
The noise rule allows 10% plus three times the larger median absolute deviation for each shape.
The aggregate difference exceeds three times the summed noise.
The failed gate stops expansion regardless of that confidence check.

Native implementation elapsed time is zero.
No runtime edit, test edit, genuine history, installed-native suite, shared smoke, profiler, or baseline repeat follows.
The historical 544.579277353012-second passing command retains its historical scope.
This evaluation supplies no CPU attribution or causal episode speedup claim.

The [manifest](manifest.json) checks the other five evidence files.
The final Ruff F/E9 check covers the two new Python files.
An earlier wider check found one unchanged native F401 import at line 6271.
The run receipt retains that feedback and the corrected evidence-only check scope.
The STE check found long paragraphs, which this document splits.

The protected external handoff records publication metadata and final costs.
No dependency installation, cache copy, package metadata change, package index access, build, or Git configuration change occurred.
The five immutable environment files passed checks before and after the experiment.
The task used a separate `HERMES_HOME` and the ignored `.venv` link to the pinned warm environment.

Beads49, full P10/P06, egress, durable, source, financial, production, manifests, index, deployment, and activation gates remain open.
Both original native50 Tasks remain FAILED.
Mac collection and Firefly remain outside this scope.
Root retains acceptance, merge, Beads closure, and wider decisions.
