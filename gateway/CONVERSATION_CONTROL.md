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
- N04 owns delivery-time Telegram binding checks and origin-sensitive delivery.
  This seam retains existing native adapter progress/final behavior; it must not
  yet be enabled as a user-facing PWA transport claiming final-only mirroring.
- Cancel/redirect/remote effects, model/catalog/version control, browser sessions,
  SSO/CSRF, image handling, and global diagnostics are outside this foundation.

Tests use the real native ingress/runner/TurnRunner/FIFO path with deterministic
model and transport doubles and isolated native SQLite. They do not establish
HTTP, production, remote specialist, or physical-device acceptance. Patch
packaging includes the new module and modified session/runtime files; the local
packaging test verifies runtime selection and archive bytes without publishing.
