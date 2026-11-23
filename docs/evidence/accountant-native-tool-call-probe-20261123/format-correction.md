# Evidence patch format correction

Correction task: `task_46d39ad575eb`, dispatch `ctx_6545e12d1fab`.
Original succeeded task `task_b852ac8db51b` and candidate `f43f58cd91d598d3e10a7c45cc7d82e255261310` remain in history.
The accepted base stays `cf6879d42c15576e4cbb68f9b19a270530528d46`.

Root found fifteen blank context lines with trailing spaces in the embedded runtime patch.
The original base-to-candidate whitespace check failed on those artifact lines. Runtime, test, and byte-proof checks passed within their original scopes.
This correction changes the patch format and its evidence references only.

## Exact patch generation and application

```sh
git diff --no-ext-diff --no-textconv --unified=0 cf6879d42c15576e4cbb68f9b19a270530528d46 f43f58cd91d598d3e10a7c45cc7d82e255261310 -- agent/native_main_payload.py
```

| Runtime source patch | Bytes | SHA256 |
| --- | ---: | --- |
| Original context format | 46075 | `9c9fb0d08e074f89c911113008d90d01d7cd831d00b76e6288535c59e4989577` |
| Corrected zero-context format | 41863 | `b4f0133693b6fe32fd925f47d9db2118a1f0e8581905b1405874a42385696314` |

Forward and inverse checks use `git apply --unidiff-zero` against owned source copies outside the checkout.
Forward application matches the original candidate bytes. Inverse application recovers the accepted base bytes.
The [static check record](static-checks.json) retains the original receipts and identifies the new round trip.

The original `git diff --check` covered uncommitted tracked changes.
It did not cover the then-untracked patch artifact. Its original result and scope remain unchanged.
New whitespace checks cover staged changes, the full base-to-index change, and the full base-to-correction-commit change.
The external correction receipt retains the exact commands and results.

The original manifest SHA256 is `6d9d5de828ec567fe26e5eab89f97ae0ea8c480db7540eeb5aedfa88dbfefe90`.
The updated `SHA256SUMS` covers this report and the corrected artifact.
The external publication receipt records the final manifest hash without a checksum cycle.
Its path is `/home/mersinvald/.local/share/orca/runtime/accountant/hpwa-p10.33-format-correction/receipts/correction-publication.json`.

## Unchanged identities and validation

| Artifact | Unchanged SHA256 |
| --- | --- |
| Runtime | `b7ccad81dda1051f03c7f0dab8d6535acfe33d2a2b284d157f17b33968d0162e` |
| Tests | `606f6a0ca2256dda472bb3e7a4ddbce6d5e783d056a29aa6e05c2382561a9a28` |
| Native API | `7854cecda074b6ba02921a367e07458aace393800ad22e81496a4647be754617` |
| Consumer contract | `891daa8a5900b1b16e830db9de2657818a5a097e836a92ad64c9d1d1b5666399` |

Complete vectors, observations, source-preservation proof, dependency bytes, limits, assertions, and functional reports remain byte-identical.
All original private receipts and `worker-handoff.json` remain unchanged.
The [eight validation scopes](validation-scopes.json) retain their original source/test pins and durations.
The focused 64-case/456.036-second scope and final callable/proof five-case/28.598-second scope remain separate.
This correction adds no functional result or combined success claim.

The protected initial root receipts retain their explicit format hold.
They establish no final acceptance.

| Initial root receipt | SHA256 |
| --- | --- |
| Source verification | `58a1b332779fc69610cfe8775be9c73a04ebf73934d1ed046a2b668059704520` |
| Evidence verification | `42d80de344891dedf2da7f428be1178912c063f9c82eb8ee73158bc31b393255` |

Only static hash, link, patch, whitespace, and prose checks run in this correction.
No application import, collection, functional test, install, build, smoke, or profiler runs.
Root owns review, acceptance, merge, Beads closure, and later reconciliation. All original external gates remain open.
