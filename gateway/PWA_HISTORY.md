# Native owner-history search and synchronization

S04 extends the existing private PWA service and existing SessionDB. The reviewed
wire is defined by the PWA repository's `contracts/owner-history-http.md` and
`contracts/v1/owner-history.schema.json`. No additional agent runner or transcript
database is introduced.

## Protected-history fence (schema 31)

The additive `native_pwa_history_fence` singleton holds a private generation and
two transient replacement flags. It is installed with the native schema upgrade;
existing sessions and messages retain their values. Its triggers are independent
of FTS state and are installed after column reconciliation. Normal ignored session
setup, appended messages and `session_meta` bookkeeping do not increment the
generation. Protected display rewrites (including equal-size content changes),
deletion, key replacement, ownership, compression lineage and native title changes
do increment it. Model metadata compares only authorization-relevant markers and
validity/size boundaries, so unrelated provider usage updates do not starve scans.

SQLite BEFORE INSERT fires even for `INSERT OR IGNORE`. A private pending flag is
therefore overwritten for each attempted row, and only an actual AFTER INSERT
applies the replacement increment. Session replacement also detects the native
unique-title collision. The existing FTS triggers write their own indexes, not
the session/message tables; multirow IGNORE/REPLACE and rollback are covered with
real SQLite behavior. No global recursive-trigger setting is changed.

The generation conservatively spans the native profile. A rewrite belonging to
another owner can require a new sweep, without exposing its identity, text or
counts. It is carried only inside opaque operation state/keyed lineage versions.
It also supplements existing N07 history cursors, closing the old equal-size edit
gap in MAX/COUNT/SUM signatures. Cursor conflict is an explicit refresh condition,
never an empty successful history result.

Adding a continuation under an already-compressed parent also fences the
existing lineage, even if the parent needs no further UPDATE. Malformed or
oversized child metadata conservatively fences a loss of provable lineage;
it is never treated as a trusted continuation. Recognized, fully owned branch
and delegate inserts leave the parent lineage unchanged. Native creation may
subsequently fill a missing child owner/profile in an UPDATE within that same
transaction. That actual ownership backfill conservatively invalidates sweeps,
even for a new independent branch. Initial captures stay read-only; there is no
additional high-water registration write to suppress this conservative case.

Duplicate branch/delegate marker keys require conservative raw comparison:
SQLite extracts the first duplicate key while the Python ownership resolver
uses the last. Such child insertions are conservatively fenced as ambiguous
instead of relying on SQLite's first-key classification. On reopen the installer
reconciles only its exact owned trigger names inside a savepoint. Changed or
missing protection on an existing singleton increments the generation once;
unchanged reopen does not. Failure rolls back definitions and generation
atomically. The upgrade regression installs actual original schema31 trigger
definitions before reopening with the corrected implementation.

This schema prerequisite alone does not advertise search or sync availability.
The owner-authorized scan and HTTP mounts require their separate runtime checks.

## Search and sync runtime

The opt-in `pwa_http` service mounts authenticated `GET /v1/pwa/search`,
`GET /v1/pwa/sync`, and checkpoint probes on the latter. It advertises
`owner_history_search` and `history_sync`. Existing private service credentials,
trusted principal bindings, Host validation and source authorization apply.
No additional configuration, FTS dependency, agent process or model request is
needed. Install the committed source archive with the existing PWA HTTP runbook;
it includes `gateway/pwa_history_scan.py` and `hermes_state_pwa_scan.py`.

Queries are trimmed literal Unicode casefold substrings, at most 256 code points
and 1,024 UTF-8 bytes; control/surrogate characters are rejected before trimming.
Operators, quotes and wildcard characters have no special meaning. Matching
follows credential redaction and native display projection. Snippets contain at
most 500 actual source code points; `match_truncated` records a folded match
whose source span cannot fit. No offsets or synthetic ellipses are generated.
System, session metadata, subordinate delegation and foreign content are absent.
Retained compacted messages and separately owned user branches remain eligible.
Unsupported or oversized display content produces explicit partial coverage.

A request considers at most 500 session candidates, 256 message/segment steps,
32 groups, and its aggregate limit (search 1–50; sync 1–100). Search reserves one
slot for each first root header, allowing the facade to match a manual title;
empty or underfilled continuations are normal. A required header that cannot fit
fails explicitly. An individual sync message too large for a whole page becomes
an explicit oversized omission. A partially full page defers an unpublished
message to its continuation, never advancing past it. The existing response byte
limit, request timeout (default 15 seconds), and 5-second SQLite progress bound
remain active. Threaded reads may finish after HTTP cancellation; they neither
write history nor execute work.

Each sweep fixes message/session upper row IDs and the protected generation.
Membership, lineage order, display text and title metadata stay within that
snapshot. Appends and independent roots appear on the next sweep; protected
changes return conflict. Current activity/execution projection may refresh and
is not a claim of byte-identical responses. The actual native incremental writer
start/completion path is covered alongside compression and equal-size rewrite.

Opaque handles are scoped to the principal, configured binding, currently
allowed source set, query/root/limit, and snapshot. They expire absolutely using
`pwa_http.cursor_ttl` (default 900 seconds, configured bound 30–3,600), including
multi-page operations. Process restart, shutdown and bounded LRU eviction also
invalidate them. The registry permits 4,096 entries and 4 MiB of encoded state,
with a 256 KiB per-native-state ceiling; it stores no result transcript bodies.
Replaying a valid cursor repeats membership/positions without extending the
operation deadline. Canonical root and source authorization run before reads and
after async work, before publication.

Terminal coverage is `complete` or `partial`, never inferred from an empty page.
Reasons accumulate across continuations. Complete covers only the captured,
provably authorized retained display text; it excludes unavailable legacy
ownership, future rows and non-text interpretation. A checkpoint probe compares
snapshot/configuration/authority and returns `unchanged` or `refresh_required`.
It is a conservative refresh indicator, not a mutation log or delta feed. Start
a new finite sweep on refresh, conflict, expiry or restart.
