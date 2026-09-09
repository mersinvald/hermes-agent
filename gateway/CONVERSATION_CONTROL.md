# Native conversation ingress foundation

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
  receipt or a network contract. N02 must normalize admission/results and attach
  durable command IDs before a browser can interpret it as accepted work.

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
instead of falling back to Telegram's current conversation. Native async task
recovery, pending queue spool recovery, and browser-origin retention across all
restart paths still require N02/N03 integration tests; do not infer recovery
acceptance from source lookup alone.

Limitations before enabling a browser transport:

- Grants are explicit existing-session assignments. Discovering/provisioning all
  owner history and new browser conversations is a later native/facade operation.
- N02 owns durable receipts, retry deduplication, bounded admission, queue/steer
  consumption and exactly-one fallback across finish, crash and restart races.
  Current FIFO and steer acceptance are process-local. Tail-of-turn arrival and
  interrupted/drained queues do not have durable delivery guarantees.
- N03 owns detachable request lifetime, events, broadcast/replay, cursor/epoch,
  retention gaps and canonical snapshot recovery. No connection lifetime promise
  or execution-status retention is introduced here.
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
