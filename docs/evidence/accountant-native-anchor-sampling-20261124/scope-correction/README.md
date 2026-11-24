# Canonical path scope correction

This repair corrects the collector filename classifier.
It does not change the original failed Task or its budget overrun.
Every earlier evidence file remains exact.

Original capture scope: FAILED.
Verified filtered observation view: PASS.

Corrected collector: unit-qualified only.
Corrected live capture: UNEXECUTED.

Original functional history: PASSED once at its original pin.
Original Task and budget: FAILED and unchanged.

The classifier resolves absolute filenames and both roots before containment checks.
Relative filenames and resolution failures remain external.
The selected standard-library allowlist remains exact.
External descriptors suppress filenames, qualified names, first lines, and current lines.
In-root aliases use canonical project-relative names.
The sampler bounds, triggers, hooks, frame inspection, safe metadata, and collection loop remain exact.

The canonical runner passed all 26 unit cases once.
These include the original 13 cases and 13 classifier cases.
The new cases cover real symlink escapes, nested escapes, prefix collisions, relative paths, loops, missing files, and aliases.
They also cover selected and unselected standard-library files.
The tests use temporary files and compiled code descriptors.

The [audit script](audit.py) executes only the exact classifier helper from the corrected plugin source.
It imports no native runtime and collects no sample.
The [audit result](audit-result.json) confirms all 27 descriptor redactions.
It checks the observation payload separately from historical qualification fields.
Ordered coordinate lists retain multiple entries when redaction maps their lines to null.
Counts, code IDs, depths, timestamps, structural sizes, native positions, and example order remain exact.

The script reuses the existing public sample by SHA256.
It creates no second sample and publishes no private original.
The public sample remains `407446fd5d531e3bc6767c93c22385f81f0486531f857a7cb2935d3153d166c6`.
The private original remains `67c02424f6d7685a03e8de15c4ba1ca62579f047bdc034f057b2d34e2f92f7c4`.
Its file mode remains 0600, and its exact parent mode remains 0700.

Run the audit with the existing warm Python environment:

```sh
.venv/bin/python docs/evidence/accountant-native-anchor-sampling-20261124/scope-correction/audit.py \
  --private-original /home/mersinvald/.local/share/orca/runtime/accountant/hpwa-p10.54/sample.original.json \
  --output /tmp/native-anchor-scope-audit.json
```

The [repair receipt](result.json) records exact source pins, unit IDs, durations, preservation checks, and costs.
The [manifest](manifest.json) checks the other new evidence files.
The protected repair handoff records the final publication pins and costs.
Ruff F/E9 covers only the changed plugin and unit files.
The unchanged native F401 at line 6271 remains outside scope.

No new genuine history, compatibility history, shared smoke, profiler, or corrected live capture ran.
No dependency installation, cache copy, index query, wheel build, or Git configuration change occurred.
Root retains source acceptance, merge, Beads writes, and closure.
Beads49 and all wider gates remain open.
Both original native50 Tasks remain FAILED.
