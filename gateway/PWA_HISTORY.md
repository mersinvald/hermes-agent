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

This schema prerequisite alone does not advertise search or sync availability.
The owner-authorized scan and HTTP mounts require their separate runtime checks.
