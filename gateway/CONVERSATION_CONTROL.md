# Native conversation ingress and durable commands

`NativeConversationIngress` is an opt-in object installed **inside the running
messaging gateway**, before the first turn. It is intended for the restricted
native `/v1/pwa` transport. This patch does not mount that route, authenticate
HTTP, or enable the broad HTTP agent API. The application facade must call the
native transport; it must not instantiate GatewayRunner or access SQLite itself.

Construct it with the gateway's existing SessionDB and explicit server-provisioned
`Principal(issuer, subject) -> tuple[ConversationGrant(session_id, SessionSource)]`
assignments. A grant may name any retained compression alias. Never derive a
principal or grant from JSON body fields. The native HTTP handler must authenticate
the facade, receive its authenticated principal context, reject unrecognized
fields, and authorize every inspect/history/send invocation through this object.
Gateway source authorization also runs for sends; native channel/user/profile
identity comes from the grant, not the caller. Multiplexed-profile deployments
are not supported by this single-database object; install a separately scoped
native endpoint per profile instead.

Operations:

- `inspect(principal, conversation_id)` returns stable root `conversation_id`,
  `native_root_session_id`, current-tip `native_session_id`, ordered
  `native_session_ids`, and nullable `active_execution`.
- `history(...)` returns retained native messages in separate lineage `segments`,
  including in-place-compacted display rows. Errors propagate; an unavailable
  history is never represented as an empty conversation. Pagination, rewind
  presentation and facade redaction remain follow-up work.
- `send(..., text, mode="send" | "queue")` routes into the existing runner.
  Active sends return `{disposition: "steered" | "queued", execution: ...}`.
  Idle sends currently await the runner's ordinary result. This is not a command
  receipt or a network contract. Browser adapters must use `submit`, below.

Execution projections contain `execution_id`, `conversation_id`, `origin`
(`pwa` or the native platform), and `execution_state` (`starting`, `running`, or
`unknown` after early agent-reference cleanup). They never expose agent objects,
credentials, or internal routing keys. Selected/observed model attribution is
not yet part of this projection. Execution identity rotates at the existing
native queued-continuation boundary. It is retained across compression and only
released by the generation-scoped turn-lease cleanup, not by early `/stop` cleanup.

The existing TurnState is the lifecycle authority. After native persistence and
lineage resolution, owner lookup/publication has no intervening await. Aliases
steer the actual owner's agent or append to its existing FIFO before loading a
second transcript. Existing process-local and durable writer leases remain in
place. A persisted native row is required for opted-in admission. Missing grant
IDs are unavailable, but do not disable unrelated grants.

Browser sends use an independent root-based native routing alias. SessionStore
persists this alias without ending/reopening a transcript or switching Telegram's
current routing key. Source channel and user identity remain intact for auth and
context. The discriminator is excluded from transport source encoding/decoding;
trusted SessionEntry persistence retains it for native restart/background source
lookup. If that alias is missing, background source resolution fails closed
instead of falling back to Telegram's current conversation. Command events retain their identity through native continuation and plugin
rewrites. The legacy pending-message spool skips these events: their text must
not be replayed as an anonymous user input. Unrelated native delegation recovery
retains its existing behavior.

Durable command operations use contract pin
`fba9a1bea7c86834d672f3a1524128f6c61ebd76`:

- `submit(principal, command)` accepts schema `1.0` text `send`, `steer`, and
  `queue`. It returns only after the native SessionDB commits the receipt.
  `command_receipt(principal, command_id)` applies current grant and native
  source authorization before returning a retained receipt. Neither operation
  mounts HTTP. The later authenticated transport must inject this existing
  ingress instance, not start another gateway or agent.
- Command IDs are scoped to concierge ID plus principal issuer/subject in this
  profile-local database. Canonical payload fingerprints include the stable
  target root. Same-ID retries replay the persisted receipt; changed text,
  action, target, or preconditions conflict. IDs and receipts are retained;
  absence is not evidence that a fresh resend is safe. Unsupported actions,
  model/binding preconditions, schema versions, and proxy execution reject.
- Idle send enters the existing runner. Active send steers the real execution;
  if its initial input has not started, send becomes a queued follow-up instead.
  Explicit queue always queues. A queued command gets its resulting execution
  identity when claimed. Different IDs with identical text remain different
  inputs in FIFO order.
- Initial command input and its applied receipt commit in one native transcript
  transaction under the actual native writer lease. At existing safe steer
  drain boundaries, the tool-row amendment and all consumed command receipts
  commit together before changing live provider content. Existing multimodal
  storage projection is retained. `applied` means this local input/control
  boundary committed, not that a provider observed it or produced a response.
- A turn's unconsumed steer becomes exactly one queued fallback on the same
  command row. Requested action and old target remain; effective action becomes
  queue and the future resulting execution differs from the target. Direct
  send-to-queue before input start has no fallback marker because no steer was
  accepted. Repeated close/reconcile cannot manufacture another fallback.
- Accepted dispatch is independent of the awaiting caller. Startup and a
  30-second native recovery tick reconcile provably dead process instances,
  including reused PIDs via process birth evidence, while protecting a live
  native writer lease. Pre-input crash reservations can be retried with their
  same execution ID; post-input crashes retain applied evidence and an unknown
  execution outcome and never automatically repeat the provider call. Reopening
  a pre-input execution advances its start order so legacy resume exclusion
  follows the actual latest execution, including interposed native turns.
- Final generation-fenced runner cleanup closes pre-agent/handoff failures;
  early control cleanup cannot replace a worker still holding its native lease.
  Legacy startup resume is excluded only when the latest execution is addressed
  by the command journal. Old completed receipts do not disable later unrelated
  Telegram-only recovery. Persistence failures propagate rather than returning
  accepted/applied success; queued work can remain pending until storage,
  adapter capacity, or authorization is restored.

The additive schema migration introduces native execution and command tables in
SessionDB; it leaves existing transcript rows intact. The journal is a mailbox
and application record, not a second executor. No exactly-once model, network,
tool, or remote side-effect guarantee is made. Unknown external outcomes require
later reconciliation or an explicit new user decision, not automatic input replay.

Limitations before enabling a browser transport:

- Grants are explicit existing-session assignments. Discovering/provisioning all
  owner history and new browser conversations is a later native/facade operation.
- N03 owns events, broadcast/replay, cursor/epoch,
  retention gaps and canonical snapshot recovery. No event or connection
  subscription protocol is introduced here.
- Telegram final-only delivery requires the explicit N04 channel policy below;
  constructing an ingress alone retains ordinary native delivery behavior.
- Cancel/redirect/remote effects, model/catalog/version control, browser sessions,
  SSO/CSRF, image handling, and global diagnostics are outside this foundation.

Tests use the real native ingress/runner/TurnRunner/FIFO path with deterministic
model and transport doubles and isolated native SQLite. They do not establish
HTTP, production, remote specialist, or physical-device acceptance. Patch
packaging includes the new module and modified session/runtime files; the local
packaging test verifies runtime selection and archive bytes without publishing.

## N03 event observations and recovery

Contract pin: PWA `8ef59e8b9c84d205a49de2233bee0b0cd51a5284`.
Schema 29 adds bounded native event detail and execution outcome/origin/time
observations. Native journal transitions and their events commit together.
`applied` means committed native input, never provider or remote success.
A generic owner close without a recorded result remains `unknown`. Legacy native
execution timestamps remain null. A different ingress owner after restart is
shown as unknown until native liveness/lease reconciliation establishes its state.

The native feed is `ingress.events`; HTTP mounts it through N07. Use
`await feed.recover(principal, conversation_id, cursor)` or
`subscription = feed.subscribe(...)`, `await subscription.poll()`, and
`subscription.close()` in the transport's `finally`. A subscriber holds only its
cursor, with one bounded poll in flight. Closing all subscriptions does not stop
native work. Every poll authorizes before and after I/O. `feed.close()` releases
subscriptions during service shutdown. Native conversation metadata comes from
N07's authorized `native_conversation` provider, not invented app title/model data.
Its active execution projection is replaced from the atomic journal snapshot.

Configure `EventLimits` through `NativeConversationIngress(event_limits=...)`.
Defaults are 4096 events total, 8 MiB encoded event bytes, 24 hours, 16 KiB per
event, 100 inspected events per replay page, 50 recent executions and scoped
receipts per snapshot, and 64 concurrent subscriptions. Batch/snapshot hard limits
are 100. Count and byte bounds apply at every write; age expiry is also enforced
on every recovery read. Native retained transcript/history is independent. Recent
execution/command indexes keep snapshot reads bounded as the journal grows.

Cursors are exclusive and scoped to conversation epochs. Identical retained
cursors return identical event ordering. A restart or optional observation loss
rotates epoch and returns a gap; an explicitly supplied old scoped epoch may
still retrieve its retained detail, alongside the new snapshot boundary. Count,
byte or age eviction (including slow readers) returns `expired`. Invalid,
foreign, future and mismatching event cursors return a gap without revealing
another conversation's identities or counts. Private command records of another
principal explicitly granted the same conversation are skipped while advancing
the inspected cursor; these filtered positions are not retention loss. Snapshot
receipts always filter exact principal/concierge scope. `*_has_more` records the
recent-window limit; it is never a claim of full transcript or event coverage.

Execution/receipt snapshots and replay boundaries share one SessionDB transaction.
Storage failures fail the request, never return empty success. A native worker's
known terminal outcome is retained by exact execution/owner identity if its close
transaction fails. Before dequeuing the next native input, the same gateway
route/generation asynchronously waits for retry; it does not invoke the provider
again, replace the worker, or enter generic error transcript persistence. Queued
commands progress automatically after storage recovers. Draining exits the wait;
a process crash before persistence conservatively leaves unknown recovery.

Available observations: committed execution lifecycle, command application/full
receipt transitions, and native tool start/completion using the real call ID plus
a bounded tool identifier. Tool callbacks preserve existing consumers/threading
and are restored after each turn without changing cached tools/prompts. Argument,
result, prompt-preview, provider-credential and internal reasoning bodies are not
included. An optional tool-observation failure changes epoch rather than failing
or repeating the actual tool.

`event_stream` does **not** mean live assistant/token text. Message delta, selected
or observed model routing, native delivery events, structured clarification and
remote delegation/cancellation are unavailable in this N03 slice. A clarify or
delegate tool lifecycle is only that tool's observation, not evidence of a remote
task state or permission to resume it. N05/N06 supply remote control/structured
clarification; N04 supplies native delivery state. U02 can refresh canonical final
history after completion and must decide any later token-stream integration
without enabling provider streaming merely to observe it.

Tests use real SessionDB, leases, native ingress, GatewayRunner and its FIFO;
provider/tool responses are synthetic. They do not establish live external,
Telegram network, browser transport or device acceptance.

## Telegram selection and final delivery (N04)

Install `TelegramConversationChannel(existing_ingress)` after trusted source/grant
configuration and before accepting input. It uses the existing runner, SessionStore
and SessionDB. N07 must mount this policy explicitly before advertising Telegram
binding/final delivery capability; this patch adds no HTTP endpoint. Its static
`trusted_channel_sources()` default uses native grants; N07's provider supplies
configured defensive source copies and branch-aware `ingress._resolve` results.
The same resolver is injected into managed SessionStore selection, so authorized
user branches retain their own root and compression aliases share their dialogue.

`inspect(principal, conversation_id)` returns the current root/tip/version for that
trusted Telegram channel. `select(principal, conversation_id, expected_version)`
checks current authorization and atomically persists its native pointer/version.
A different root increments the version; selecting the same root does not. Native
`/new` and `/resume` use this selection path without ending, interrupting or
reopening either dialogue. Existing writers and delegated work continue. The first
message for a fresh configured channel persists a root and enables its binding
before routing input. Persistence failures do not admit or send work. PWA sends
never select Telegram implicitly. Telegram input routes through the selected root
alias before adapter busy/FIFO processing; no output is submitted as input.

Both origins buffer normal assistant deltas/interim output until completion. A
Telegram-origin execution retains native tool/status and explicit clarification,
approval and selection-confirmation interactions. PWA-origin work receives no
additional Telegram progress or interactive-prompt mirroring. Automatic spoken
final synthesis is not emitted by this policy; explicit successful `MEDIA:` final
attachments use the existing native delivery splitter. Failed turns deliver their
normalized failure text without successful-turn attachment uploads. Other channels
and non-opted-in native conversations retain their existing behavior. A restored
internal async-delegation wake carries its route alias but not its original
execution origin. Such ambiguous/system-origin progress is suppressed in managed
routes until trustworthy provenance exists; the Telegram transport alone cannot
establish Telegram origin. N05/N06 must validate persisted origin restoration and
interactive continuation behavior before expanding that capability.

Every completion first reserves the native durable `(execution_id, channel_key)`
delivery record. The trusted channel key is an opaque digest of configured routing
identity; records contain neither response bodies nor credentials. Each text/media
dispatch rechecks the current root, binding version and native source authorization.
Browser logout does not itself revoke Telegram permission. Native reconnect,
Markdown fallback and media anchor-retry boundaries recheck immediately before
Bot API dispatch. A switch after a send has started cannot retract that request.
Uncertain media/text sends do not invoke an alternate resend path. Repeated
completion callbacks cannot repeat a reserved delivery, and switching back does
not replay a skipped final.

`delivered`, `skipped` and `unknown` describe delivery only. Execution completion
and command application remain independent. An interrupted `attempting` record is
projected as `unknown` and is never automatically retried. Partial multi-message
sends, unconfirmed media results and failed acknowledgements are conservative
`unknown`; there is no exactly-once Telegram claim. Durable reservation failure
prevents external send; a terminal-write failure retains unknown reservation.
Reservation, terminal-write and read failures are contained at the outbound
boundary so the native adapter cannot turn them into an unguarded generic agent
error reply. A missing/unreadable delivery record returns no receipt claim; the
authorized lookup remains unavailable until storage can establish its state.

For N03/N07 integration, `delivery(principal, conversation_id, execution_id)` is a
narrow authorized snapshot lookup with current source authorization and root
matching. The successful attempting-only CAS in `native_delivery_finish` is the
post-commit event-observation boundary. N03 can observe there after integration;
this patch emits no wire event and adds no delivery array to execution recovery.
The native delivery record remains authoritative if an observer fails. Tests use
real runner, FIFO, SQLite and Telegram send paths with synthetic model/Bot doubles;
HTTP, live Telegram and device acceptance remain separate review gates.

## N05 explicit cancellation and known remote-task coverage

Schema30 adds control receipts and actual dispatch provenance to SessionDB.
`submit(principal, cancel_command)` addresses one exact conversation/execution.
The stable ID shares the native input command fingerprint domain and conflicts
atomically across send, steer, queue and cancel. An accepted receipt is durable
control admission. It has no input application state. Native request state and
observed N03 execution state are separate; completed or interrupted native work
does not establish downstream compute termination or rollback of completed effects.

The controller signals existing native `hard_interrupt` mechanisms, including
managed detached children selected by copied execution/owner provenance. It never
invalidates generations, releases a live lease or starts replacement work. Queued
requests and unconsumed steer fallback remain retained for later executions.
Native interrupt signalling happens independently of remote control capacity.
Disconnect, listener close and SSO expiry do not cancel executions. Managed
children retain ownership under stale-monitor warnings; configured native
operation timeouts remain actual native failure behavior. Unknown restored child
provenance is never reconstructed from a Telegram or gateway route alias.

Each actual outbound A2A dispatch reserves an opaque indexed row before I/O and
records validated task/context evidence when available. A lost response remains
unknown coverage, never a resend. Cancellation uses the exact saved RPC route,
configured peer origin/path identity, tenant and protocol version. Changed peers,
foreign task/context owners and superseded execution claims fail closed. The
latest native task claim and every prior write reservation fence cancellation
across duplicate dispatch aliases. Global A2A transcript entries are not authority.

Remote control uses a shared aiohttp client with no environment proxy, redirect,
compression expansion or write retry. Each RPC has a total deadline of at most
30 seconds and a strict JSON-RPC envelope/body limit of 256KiB. Native controller
constructor limits default to 4 concurrent control jobs, one target per scheduling
turn, a 30-second total turn deadline, and 5-second polling after a complete scan.
Concurrency is configurable 1–64, polling 1–300 seconds and the turn deadline up to
90 seconds. These limit control/observation I/O only; they impose no native
execution admission cap. Per-command `remote_after` and `next_poll_at` persist
fair scanning; a large slow execution yields its slot between targets. Pending
commands remain in SQLite rather than an unbounded Python queue. The resolver
shares/deduplicates DNS and bounds residual lookup tasks by connector capacity.
Timed-out HTTP callers close their socket/read task and cannot later send. OS
getaddrinfo itself may finish later; physical cancellation of OS DNS is not claimed.

GetTask preflight must validate the exact recorded identity before one CancelTask
(or legacy tasks/cancel). The write reservation commits before I/O. A sent or
possibly sent request is reconciled by reads only, including across restart and
fresh manual commands. A known-not-sent preflight failure remains failed for its
original stable command; only a fresh explicit cancel command may retry. Attempt
rows retain old evidence. Target `request_origin` distinguishes this command from
a shared prior reserved write and an unattempted target. A newer manual retry
never silently becomes an older command's own acknowledgement.

Native recovery atomically captures the execution journal, shared existence of
an accepted cancel request, private control receipts and event cursor. Shared
`remote_cancellation=requested` means only accepted user intent, with unconfirmed
remote outcome. Exact-principal `cancel_state_changed` events contain command ID,
change kind and optional opaque dispatch ID, without tool arguments/results,
reasoning, peer URLs, raw remote identities or credentials. Recovery includes at
most 10 recent control receipts with empty target pages and explicit coverage;
whole-execution counts and an opaque offset-zero cursor lead to GET command
pagination (default 50/max 100). Controllers traverse every indexed target
independently of that page size. Cursors bind principal, command and execution;
process restart requires page 1 again. Known counts count indexed dispatch records,
including lost task identities. Unobserved descendants remain unknown even when
all known tasks report canceled.

The standalone native patch provides ingress/controller seams; native HTTP
capability and mounted command response unions require the reviewed N05 wire and
primary integration. The existing browser facade performs app metadata enrichment.
Tests use actual native runner, SQLite, leases, local process interrupt and native
A2A hooks with synthetic provider/peer responses. They are not live provider,
Telegram or downstream specialist acceptance evidence.

## N06 native redirect and clarification checkpoint

The native control paths implement the N06 wire at PWA commit
`f847a5159c62be06df3b1f8b8ed3e395aef9c122`. This checkpoint covers native,
managed child, and Telegram clarification. `remote_clarification` remains
unavailable: ordinary A2A input-required text or permission data does not establish
an actionable question protocol. The optional configured structured-question
adapter is a separate N06 phase; no production peer support is claimed here.

A redirect reserves one public command ID across all input/control kinds. Its
native interruption and queued input are private phases of that same identity.
Public command lookups, events and snapshot input receipts never expose those
phases as separate cancel/send receipts. The accepted interruption does not free
a native writer: the original execution must close and its actual native lease
must release or transfer before the new direction enters the existing FIFO.
Queued requests and unconsumed steer fallbacks remain ahead of the new direction.

Remote confirmation and native release are independent gates. A live managed
child or a previous process's unavailable child inventory leaves the dispatch
frontier uncertain, even when no remote task ID is known. Explicit confirmation
waives only that remote uncertainty. The release transaction records its decision
basis once and queues the input once; late task observations update current
coverage without rewriting the decision that released an already applied input.
The one control observation pump visits at most 16 pending redirects per turn;
it never owns a conversation or starts a replacement executor.

Questions originate in the actual native callback or an actual registered child,
with execution/owner/child binding retained privately. The existing gateway entry
lock orders durable answer claims before Event wakeup; only the exact waiter ACK
records delivery. A committed claim whose wakeup read fails may be handed off to
the same live entry after storage recovery. A lost waiter or failed ACK remains
unknown and is never replayed into a later worker. Browser disconnect, logout,
and listener shutdown do not cancel the native waiter. Native cancel and route
cleanup wake only the matching pending native entry. Expiry and cancellation
return explicit unresolved/expired/cancelled observations, never an answer.

Telegram buttons and PWA answers share one claim. Telegram Other changes only
its still-pending native question to text and increments the durable revision
under the same lock. Old PWA revisions and old choice buttons cannot answer the
new text revision. This path does not interpret any text as write approval;
permission_bridge remains independent.

Question GET routes are `/v1/pwa/conversations/{root}/clarifications` and its
`/{question}` exact lookup. Pages select unresolved questions in admission order,
with a fixed traversal upper bound, page limits 1..100 (default 50), a 65,536-byte
individual question bound and a 131,072-byte whole-page bound. Whole questions
are retained at byte boundaries. Opaque cursors expire after one hour and bind
principal, concierge, current native grant/source, root, page limit and process
event epoch. Malformed cursors are invalid input; foreign, expired, changed-grant
and restarted traversal cursors return a generic recovery gap. Recovery includes
at most ten questions and their continuation cursor captured with its native
journal/event boundary. Shared question events contain only identity, revision
and state; answer command IDs/receipts remain private to their exact principal.
