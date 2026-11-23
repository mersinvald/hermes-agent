# Controlled native tool-call observation

This component observes one synthetic assistant function call through the actual synchronous native caller and shared checked transport.
It stops before execution. The call ID, function name, and arguments grant no execution authority.

The accepted native base is `cf6879d42c15576e4cbb68f9b19a270530528d46`.
PWA publication `6b47e7d71ed63f9bfc269e2b22bd71c1406678b8` contains the selected P10.32 proposal.
Its candidate is `4443638f984ac9bdfc6663fbd25710dc52841e94`.
The proposal SHA256 is `71014420d0740c5ff198cec82f7e9e06c050e97ac3a9e51a902da9f116bef42c`.

The [consumer contract](consumer-contract.md) defines the implemented boundary.
[Source preservation](source-preservation.json) records the sole annotation exception and unchanged text validators.
[API data](native-api.json) records the exact token, methods, and fourteen-field records.
[Source pins](source-freeze.json) identify the final runtime, tests, consumer contract, API, and dependency inputs.

[Complete vectors](vectors.json) retain all five payloads and their ordered ancestry.
[Observed bytes](observed-bytes.json) contain the actual focused positive proof.
[Validation scopes](validation-scopes.json) retain each failed, intermediate, corrected, and final selection separately.
Each scope includes exact IDs, setup/call/teardown durations, source/test pins, and external raw receipt checksums.
Their union is not a new aggregate run.

The focused selection passed 64 cases at its recorded intermediate source pin.
The final selection passed five cases after the proof getter and callable checks changed.
[Source preservation](source-preservation.json) identifies those exact changes. Every other runtime function definition stayed byte-identical.

| Separate scope | Passes | Wall seconds |
| --- | ---: | ---: |
| Focused source | 64 | 456.036 |
| Final callable/proof access | 5 | 28.598 |
| Text representative with phase records | 1 | 3.838 |

All selections used zero retries. The eight-case feedback selection took 60.378 seconds.
The initial failed attempt and its corrected 14.233-second positive remain separate evidence.
These overlapping selections do not form a fresh aggregate success.

[Dependency data](dependency.json) records the one offline install and one shared import/API/admission/RECORD smoke.
[Runner isolation](runner-isolation.json) records external environment, state, caches, and receipt paths.
This evidence claims no installed native distribution or installed native functional selection.

[The format correction](format-correction.md) records the zero-context source patch and full-task whitespace scope.
It preserves all original functional evidence and source identities.

Root owns acceptance, merge, Beads closure, and canonical reconciliation.
Full P10/P06 and private/real-provider/egress/durable/cross-thread/source/financial/production/native-manifest/index gates remain open.
The online Mac collector stays outside scope.
