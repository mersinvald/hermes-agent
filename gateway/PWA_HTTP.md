# Restricted native PWA transport (N07)

The opt-in listener runs inside the existing `GatewayRunner`, borrowing the
`SessionStore` SQLite handle and `NativeConversationIngress`. Native leases,
durable N02 commands, provider execution and gateway drain/recovery remain the
execution authority. This service never constructs a separate agent executor.

Wire reference: PWA contract pin
`a55d1d94ccc21cf0781b802a1d0e4a60a514c677`,
`contracts/native-pwa-http.md` and `contracts/v1/native-pwa.schema.json`.
All routes are below `/v1/pwa/`. Health/capabilities, conversation list/create,
inspect/history, execution lookup, recovery/events, Telegram binding, command submit
and receipt lookup require the private facade
credential and the strict trusted-principal header. There are no browser login,
general agent, arbitrary file/database, or administrative mutation routes.

## Configuration

Disabled by default. Example values below are synthetic; they are not production
registration or credentials. Operator binding is explicit, with no signup or
email/role-based identity inference.

```yaml
gateway:
  pwa_http:
    enabled: true
    concierge_id: synthetic-concierge
    host: 127.0.0.1
    port: 8766
    allowed_hosts: ["127.0.0.1:8766"]
    token_env: HERMES_PWA_FACADE_TOKEN
    models:
      default_model_id: daily
      entries:
        - model_id: daily       # Opaque browser-visible ID.
          display_name: Daily
          provider: openrouter  # Native-only route identity.
          model: synthetic/model
          capabilities:
            text_input: supported
            image_input: unknown
            tools: supported
            reasoning_controls: unknown
    bindings:
      - issuer: https://synthetic.invalid
        subject: synthetic-owner
        default_source_id: telegram-owner
        sources:
          - source_id: telegram-owner
            platform: telegram
            chat_id: "synthetic-chat"
            chat_type: dm
            user_id: "synthetic-native-user"
            # Optional: thread_id, scope_id, session_ids (explicit legacy IDs).
```

Provide the service credential through the named secret environment variable,
at least 32 bytes; it is neither serialized into config nor returned to callers.
Ordinary native platform source authorization must also allow the configured
source. Multiplex profiles, partial/malformed settings, missing credentials and
distinct configured Telegram identities collapsing to the same native channel key fail
closed. Repeating the identical configured native source across principals is
allowed: they share its native dialogue visibility and channel selection, while
command receipts remain principal-scoped. Non-loopback IP binds additionally
require `private_network: true`, paired `tls_cert`/`tls_key`, and exact Host
values. TLS requires at least TLS1.2. Infrastructure must independently restrict
this endpoint to the facade; a configuration declaration is not network isolation.

Optional finite transport limits: `max_body_bytes` (default524288),
`max_response_bytes` (default2097152), `max_requests` (default32),
`request_timeout` seconds (default15), and `cursor_ttl` seconds (default900).
These bound HTTP observation/admission, not active native conversation count.
Port0 selects an ephemeral port for isolated tests. Disabled configuration should
contain only `enabled: false`. Reload bindings by restarting the gateway; existing
durable rows do not confer permission when their configuration binding changes.

The `models` block is optional. Omitting it preserves ordinary native model
resolution and leaves managed model capabilities unavailable. When present, it
is strict: the default must name exactly one of one to 100 unique entries, and
every entry requires an opaque ID, display name, native provider/model route and
four explicit capability states (`supported`, `unsupported` or `unknown`). A
malformed block fails configuration loading. Credential/provider resolution is
dynamic: an unresolved route remains a bounded unavailable catalog entry without
exposing its upstream identity. A temporarily unavailable default blocks only
default initialization or an execution already selecting it. Other available
allowlisted selections remain usable and native never substitutes a model.

## Ownership and persistence

Schema28 adds native owner/create-ID assignments and the N04 delivery-attempt
table without changing existing transcripts. Every read and retry checks current
principal/source binding, native source authorization and all retained lineage
peers. Explicit legacy IDs may fill missing peer evidence but cannot override
contradictory evidence. User dialogue branches remain discoverable; subordinate
delegations do not become top-level conversations. Unknown metadata or ownership
produces partial discovery coverage, never a complete-history claim.

New root+owner+create-ID reservation is one native SQLite transaction. Alias
failure after that commit leaves the same reserved root for a retry. Creation
does not change Telegram selection or end a native session. Native-owned aliases
continue through shared admission. `await ingress.native_conversation(...)`
provides a minimal authorized native DTO; S02 adds application metadata.

Conversation/history pagination cursors live in a bounded process-local map (4096 entries), bound to
principal, configuration fingerprint, resource/query and page size. They expire
on TTL, eviction or restart; loss causes409 and a canonical read restart, never a
command resend. Cursor bytes contain no principal or foreign resource identity.
History preserves lineage/row order, compacted display rows and deduplicated
copies. It excludes rewind-deleted rows, withholds system/unsupported/oversized
content, and marks affected pages partial. A changing lineage or compaction
snapshot produces conflict rather than an inconsistent successful page. SQL
reads have a five-second progress deadline and return bounded projections.

Native title, tool name and text pass through forced native credential redaction;
redaction failure withholds the field/content. Access and parser logs are disabled
for this listener because raw HTTP request lines and header errors can contain
credentials. HTTP errors contain static messages only.

## Checkpoint verification and limits

Use the native hermetic runner with Python3.13.3:

```sh
HERMES_PYTHON=/private/tmp/hermes-pwa-native313-venv/bin/python \
HERMES_TEST_FILE_RETRIES=0 scripts/run_tests.sh \
  tests/gateway/test_pwa_http.py \
  tests/gateway/test_native_commands.py \
  tests/gateway/test_conversation_control.py -q
```

Tests use real loopback HTTP, isolated SQLite, the real gateway entrypoint and
shared native runner/lease/command paths with a synthetic model executor. They
cover forged scope, source revocation, postcommit creation failure, retry,
disconnect, lineage changes, redaction and startup/shutdown. They are not live
Zitadel, Telegram, model/provider quality, browser, TLS/CNI or deployment evidence.
Event/replay/recovery and Telegram binding/final delivery are mounted here.
Telegram capabilities reflect the principal’s configured native sources; final
delivery additionally requires the local Telegram adapter and local execution.
Explicit cancellation control is available with local native execution. Token,
model-routing and clarification observers remain unavailable. No live deployment
or production credentials are used here.

`POST commands` accepts the closed input/cancellation response union. Cancellation
receipts distinguish request acknowledgement from native and known remote task
observations; unknown descendants remain unknown. `GET commands/{command_id}`
accepts `remote_limit` (1–100, default50) and an opaque `remote_cursor` only for
cancellation receipts. Exact execution lookup and recovery capture the accepted
cancellation flag in the same native transaction as their other observations.
Compact recovery carries at most10 private control receipts with whole counts
and command-lookup cursors, without remote task IDs or endpoint details. Closing
the HTTP observer never submits cancellation or retries an uncertain remote write.


## Recovery, execution and bounded subscriptions

`GET conversations/{root}/recovery` returns the native recovery DTO, not the
browser facade DTO. `GET conversations/{root}/executions/{execution_id}` reads
that exact durable native execution, including entries outside the bounded recent
snapshot. Both require a canonical dialogue root and current source authorization.
An open row belonging to another process instance or awaiting completion
persistence is projected `unknown`, never fabricated as a live local owner.

`GET conversations/{root}/events` uses finite SSE subscriptions. Each frame is
`event: recovery`, `id: <encoded cursor>`, and `data: <one-line native recovery>`.
The initial frame always captures a snapshot; subsequent frames contain bounded
replay plus the next snapshot/cursor. Supply either `cursor` query or
`Last-Event-ID` on SSE; recovery accepts only the query. Encoding is unpadded
base64url of canonical JSON (sorted keys, compact separators, UTF-8); the encoded
value is at most 2048 bytes. Malformed encoding/noncanonical JSON returns400;
canonical JSON with an invalid semantic position returns a native `gap`.
Retention/epoch loss produces `gap` or `expired`, never a command resend.

Optional `event_limits` uses the native EventLimits fields: defaults are
`max_count: 4096`, `max_bytes: 8388608`, `max_age_seconds: 86400`,
`max_event_bytes: 16384`, `batch_count: 100`, `snapshot_count: 50`,
`max_subscribers: 64`. Config validates positive integers and caps these at
65536 records, 64MiB, 7days, 64KiB/event, 100/batch, 100/snapshot and
128 subscribers. Event detail may expire; durable journal identity does not.

`event_poll_interval` defaults1 second (1–10), `stream_write_timeout` defaults5
seconds (1–30), and `stream_max_seconds` defaults30 seconds (1–300 total, including
initial capture). Every awaited capture/prepare/write/poll wait is clamped to the
remaining lifetime. Initial capture also has `request_timeout`. Each stream has one poll
and one write at a time, with no unbounded queue, and each frame must fit
`max_response_bytes`. Subscription capacity is independent of `max_requests` so
stream readers do not consume ordinary command/request slots. Each publication
reauthorizes after awaited I/O; failure after headers closes the stream without
fabricating domain events. Disconnect, expiry, slow writes and listener shutdown
release subscriptions only. A failed/expired stream may close its transport
without a final chunk marker rather than wait on an unbounded EOF drain. Already
received complete SSE frames remain valid. These closures do not cancel
executions or command recovery.

## Telegram selection and delivery observation

The existing `pwa_http.enabled` is the single opt-in; no additional Telegram flag
is needed. It installs N04 final-only normal output for configured Telegram
identities only. Other platforms and unconfigured native channels retain their
existing behavior. Explicit native Telegram control/approval prompts remain
native; ambiguous/PWA internal progress stays silent. Browser conversation
creation and native routing aliases never implicitly move Telegram selection.

`GET/PUT conversations/{root}/telegram-binding` reads or selects the configured
Telegram channel for this authorized root. The selected root can differ from the
requested root, but is separately authorized before publication. Inaccessible
selection is404, not a disclosed foreign ID. Missing pointer is explicitly
`unbound` with null IDs/version0; a non-Telegram source is unavailable404.
PUT requires `{schema_version: "1.0", expected_binding_version: <integer>}`.
First PUT with version0 atomically persists the pointer to the already durable
root; it creates no extra dialogue. A changed root increments version, selecting
the same root keeps it, and stale/exhausted versions return409. Values are bounded
at9007199254740991. Native primary routing DB failure rolls back selection even
if the legacy JSON mirror remains writable. Neither selection ends, reopens nor
interrupts either dialogue.

Eligibility is rechecked at final dispatch against the current native binding
and source authorization. A send already in flight cannot be retracted by a later
switch. Delivery records contain no payload or credentials; `attempting` projects
as `unknown` and is not automatically retried. Delivery and execution outcomes
are separate. A native execution includes at most the configured Telegram
channel's persisted delivered/skipped/unknown record; absence omits `deliveries`.
Delivery projection and recovery cursor are read in the same SQLite transaction.
An optional `delivery_changed` observation follows the successful attempting-only
CAS. Observation failure rotates the recovery epoch while preserving the durable
delivery result and never resending. Repeated completion/CAS emits no duplicate.

## Private retained image storage (M01)

`GET images/policy`, `POST conversations/{root}/images`, and
`GET conversations/{root}/images/{image_id}[/{original|preview}]` implement the
reviewed PWA `bdc0627` storage contract. POST carries one raw JPEG/PNG/static WebP
and `X-Hermes-PWA-Upload-Id`, with 201 on durable creation, 200 on an identical
retry and 409 on changed bytes/type for the same native owner/root/upload ID.
Image commands remain unavailable; this storage path does not send image bytes,
EXIF, GPS, filename or filesystem paths into a model, title utility or delegate.
The later reviewed native admission must own transcript association through the
existing MessageEvent/GatewayRunner seam; no independent executor was introduced.

The native-owned `${HERMES_HOME}/pwa-images` directory retains exact originals
outside expiring media caches. The owner scope contains the configured concierge,
issuer/subject, actual profile and validated native source identity. PWA cache
`native_owner_id` labels are not an independent native authority. Every operation
checks actual native ownership; binary responses recheck it after held-FD reads.
Source reassignments cannot inherit the previous owner's image IDs.

Optional `pwa_http.images` has these defaults and bounded configuration:

- `max_original_bytes: 20971520` (1 through 20971520).
- `max_images_per_message: 10` (1 through 10; future admission policy only).
- `max_decoded_pixels: 40000000` and `max_dimension: 16384`, configurable lower.
- `workers: 1` (1–2), covering streaming, codecs, reads and downstream writes.
- `min_free_bytes: 268435456` (positive, at most 16 GiB). Admission reserves this
  profile filesystem headroom plus every configured in-flight original/preview
  allowance. Full disk/quota failure rejects writes without deleting old images.

The current maxima are bounded implementation transport/admission ceilings, not
an amendment preventing later reviewed changes to ADR 005 defaults. Existing
JSON request/response caps are unchanged. Request transport auto-decompression is
disabled; explicit Content-Encoding is rejected before bytes are interpreted.
Codecs reject MIME mismatches, invalid/truncated pixels, animation, dimensions
and pixel count excess. Originals preserve metadata byte-for-byte. Previews are
fresh oriented/resized RGB JPEG pixels (2048px edge, 8MiB cap), stripping metadata
including EXIF/GPS/XMP/ICC/comments. Original dimensions refer to encoded pixels;
preview dimensions refer to oriented/reduced pixels.

Uploads stream into mode0700 private staging directories and mode0600 files;
owned descriptors pin every component without following symlinks. A closed
manifest and bounded digests bind scope, upload ID, image ID and byte variants.
Original/preview/manifest/stage directory are fsynced before atomic nonempty
record-directory publication, followed by parent fsync before acknowledgement.
Concurrent identical uploads converge on one random image identity. Retries
verify retained digests and fsync the parent again, including after lost ACK or
uncertain previous fsync. Only known unpublished staging data is cleaned.
Published originals have no automatic deletion/TTL/GC; backup remains a separate
pilot gate. An existing corrupt record fails closed without overwrite.

Native request_timeout covers upload and downstream response writes. Codec
threads keep their stage and worker slot through repeated cancellation until
actual completion; cancellation never releases the slot while a thread writes.
A slow/closed binary response closes transport without sending a fabricated JSON
body after headers. Native startup/runtime ownership and the retained history DB
remain in their existing process; images add no native DB schema.
