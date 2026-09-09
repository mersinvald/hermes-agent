# Native Hermes HITL Candidate

## H01: Native Telegram Grant Inspection and Revocation

The H01 candidate starts at native `748fb13209f06fe8e8f843c61eee141ff42da2d7`.
An authorized human sends `/grants` in the existing Telegram bot, selects a
recorded grant, inspects the full scope, and selects **Revoke future use**.
The command is handled before the agent queue and does not enter the model,
interrupt a running task, or resolve a clarification or approval waiter.
Human sender/chat values come from the actual native message/callback. The
strict native callback authorization runs even when intake permits DM pairing.

The existing permission service API remains the sole contract: owner-scoped
`GET /grants?after=N` and `POST /grants/{id}/revoke {}`. No new endpoint, database
migration, model tool, runtime wrapper, bot or permission-writing API is added.
The native bridge's separate credential, fixed HTTPS origin, default TLS trust,
proxy/redirect rejection, bounded parsing and error redaction remain in force.
Every new imported module and the command registry are in the patch publisher.

Each service page is limited to 100 records and 4 MiB, and the entire page is
validated before five rows are displayed. Cursor ordering, unique IDs, owner,
scope shape, resource mapping, template constraints, expiry and revoked state
must be valid; malformed or oversized responses fail closed. The detail view
renders the complete immutable record with escaped actor/backend/tool/resource,
constraints, expiry and a display fingerprint. Details exceeding Telegram's
bounded message size receive no revoke button. Listing shows **not revoked
(subject to current service policy)**, expired or revoked; recorded scope alone
does not prove current contract applicability or effective execution authority.

The service does not expose a grant version/CAS or grant-by-ID endpoint. H01
relies on its existing immutable random ID, cursor, scope and expiry, with no ID
reuse and only monotonic revocation. A local SHA-256 fingerprint of those
immutable fields binds the display; it is not a server version token. Inspection
and revoke refresh the exact cursor via `after=cursor-1`, require the same ID and
fingerprint, and reject changed or missing records. Backend revocation remains
atomic with future consumption. It cannot recall an already authorized call.

Per-adapter views have five-minute leases, at most 128 owner/chat slots, five
retained rows per view and at most 128 previous-page cursors. Each button nonce
is bound to its native sender, chat, topic, message and displayed grant. A nonce
is consumed before any await. Refresh, replacement, restart, expiry or uncertain
Telegram delivery invalidates old buttons. Late HTTP/Telegram responses cannot
install old authority or replace a newer view. No views or decisions are
persisted or automatically recovered. The permission ledger is preserved.

A lost revoke acknowledgment triggers one read-only reconciliation, never a
second POST. The UI distinguishes confirmed revocation, observed revocation
after a lost response, refusal and unconfirmed state. Refresh permits a new
explicit human attempt after inspection; the bridge never retries a mutation,
executes a tool or resumes a task automatically. Native UI follows the adapter's
existing English message convention; no locale framework was introduced.

Validation uses Python 3.13.3 and `scripts/run_tests.sh` with
`HERMES_PYTHON=/private/tmp/hermes-pwa-native313-venv/bin/python`. The 44 new
grant tests cover real loopback HTTPS/native dispatch, 103-record service
pagination and owner isolation, ledger reopen, wrong sender/chat/message/topic,
malformed and changed scope, revoked/expired states, view limits, restart,
replacement, duplicate clicks, lost acknowledgments and delayed responses.
The existing 85 native permission tests and five TLS trust tests also passed,
including consumed-operation replay. Another 41 native approval/clarification,
command-routing and packaging checks, and 61 command-registry tests passed.
The registry keeps `/grants` out of Slack's 50 native-command slots so existing
commands are preserved. Focused Ruff checks and compilation also passed.
Telegram delivery is fake and all grants
belong to the synthetic fixture in temporary storage. No real Telegram update,
standing-template activation, HA control, deployment or live acceptance is
claimed. Production standing templates remain disabled.

```sh
HERMES_PYTHON=/private/tmp/hermes-pwa-native313-venv/bin/python \
  scripts/run_tests.sh tests/permissions -- \
  --permissions-source=/Users/mersinvald/dev/mkl.dev/infra/services/ai/mcp-permissions -q
```

The earlier evidence below applies to its explicitly named historical sources.

Source candidate on `18a0b6c8df807acbd64c65dbbf66199f635f2664`.
No production acceptance, publishing, deployment, or real Telegram/HA control
is claimed. The permission service's current HTTP JSON API is authoritative;
these tests import explicitly selected infra source without modifying it.

## P2 Lifecycle Fix

`RemoteApprovalResolution` separates a successful provider decision from
`waiter_active`. A confirmed HTTP decision remains recorded if the original
native queue entry disappears while HTTP is in flight. The resolver never
recreates that entry, signals a removed waiter, or resolves a new request in
the same chat. Network I/O remains outside the shared queue lock.

The native Telegram callback then reports:

> Permission recorded; request no longer waiting; no action automatically executed

This is a permission receipt, not a completed action. Service refusal (HTTP 403)
is reported as rejected; transport/503 uncertainty is unconfirmed. No POST is
retried after a duplicate click or failed Telegram acknowledgment. Existing
nonce binding, canonical action rendering, choice scopes and operation IDs are
unchanged. Legacy command approval resolvers retain their existing return types.

## Tests

TLS follow-up: use the standard `SSL_CERT_FILE` environment variable pointing
to a bundle containing existing public roots plus the required private CA.
Do not replace public roots with a private-only file. No `ca_file` bridge
setting, client SSL context override, or disabled verification is needed.
Five subprocess tests exercise the unmocked urllib client against the real
service HTTP API, including merged trust success, wrong/untrusted CA, hostname
mismatch and plain HTTP rejection. Both default SSL and native Telegram's
HTTPX client retain the public-root fingerprint subset. A deliberately broken
proxy environment does not intercept bridge traffic. Test CA keys are generated
only under the test temporary directory and never included in image layers.
The permissions test total is now **62** (57 lifecycle tests plus 5 TLS tests).

Future configuration example, not enabled or deployed by this candidate:

```yaml
mcp_permissions:
  enabled: false
  url: https://permissions.ai.svc.cluster.local
  owner: <permission-service-owner-id>
a2a_agents:
  groundskeeper:
    expected_permission_actor: <stable-Gateway-apiKey.name>
```

The actual URL must match the cert-manager certificate SAN. Mount the merged
bundle separately and set `SSL_CERT_FILE` to its path; keep the human token in
the native process secret environment, never in tool/peer credentials.

From this worktree, with the candidate's existing test environment:

```sh
scripts/run_tests.sh tests/permissions -- \
  --permissions-source=/Users/mersinvald/dev/mkl.dev/infra/services/ai/mcp-permissions -q
```

Result: **57 passed**, comprising the previous 44 and 13 added race/refusal
cases. Real loopback TLS routes commit once/standing decisions before delaying
their responses. Coverage includes native timeout, interrupt, handover, direct
bridge-client integration, replacement pending entries, failed Telegram ack,
expiry, changed actor mapping, 503, and unchanged no-retry uncertainty behavior.
Telegram Query/Bot I/O is fake; no Telegram client is started or polled.

Selected native regression overlap: **110 passed** (not disjoint from previous
candidate regression runs):

```sh
scripts/run_tests.sh \
  tests/gateway/test_telegram_approval_buttons.py \
  tests/tools/test_approval_interrupt.py \
  tests/tools/test_request_tool_approval.py \
  tests/tools/test_approval_plugin_hooks.py \
  tests/plugins/test_a2a_streaming_progress.py \
  tests/plugins/test_a2a_follow_and_artifacts.py \
  tests/gateway/test_native_progress_delivery_contract.py \
  tests/gateway/test_run_cleanup_progress.py \
  tests/gateway/test_telegram_progress_edit_transient.py -q -j 6
```

The parent-owned combined native flow also ran against this source in a
network-disabled, read-only Linux container: **11 passed, zero skipped**.
Only the driver explicitly reattempted execution. Backend calls/effects stayed
zero after approval callbacks, with replay and standing-resource bounds retained.
It emitted two existing unawaited `_watch_stdio_children` coroutine warnings
in `tools/mcp_tool.py`; this fix does not change that code.

Reproduction using the already available pinned artifacts (no pulls/builds):

```sh
KILO=/var/folders/hl/yl5w820n4yj6kp8hm1thtzxm0000gn/T/kilo
SERVICE=/Users/mersinvald/dev/mkl.dev/infra/services/ai/mcp-permissions
docker run --rm --pull=never --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges --user 65532:65532 --tmpfs /tmp:rw,mode=1777 \
  --mount "type=bind,src=$KILO/hitl-foundation-artifacts,dst=/artifact,readonly" \
  --mount "type=bind,src=$KILO/native-permissions-runtime,dst=/runtime,readonly" \
  --mount "type=bind,src=$SERVICE,dst=/service,readonly" \
  --mount "type=bind,src=$PWD,dst=/hermes,readonly" \
  -e PYTHONPATH=/artifact/hermes-deps:/runtime/deps:/service/src:/hermes \
  -e PYTHONDONTWRITEBYTECODE=1 -e HOME=/tmp python:3.13-slim \
  python /service/tests/test_native_human_flow.py \
  --gateway-binary /artifact/app/agentgateway \
  --kagent-probe /runtime/kagent-probe --kagent-server /artifact/kagent-server \
  --hermes-source /hermes
```

That runner verifies artifact hashes and rejects skipped acceptance. Missing
artifacts must be reported as unavailable, not counted as passed.

Full parent-service regression run: **109 passed, 18 skipped**. These skips
are not acceptance passes; the separate combined native gate above ran all 11
tests with no skips. The first generic run had six container-layout/import
failures, resolved by using a sufficiently deep service mount, its `src` working
directory, and installed dependency paths that survive child environment reset.
No service source changes were made:

```sh
docker run --rm --pull=never --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges --user 65532:65532 --tmpfs /tmp:rw,mode=1777 \
  --mount "type=bind,src=$KILO/native-permissions-runtime/deps,dst=/usr/local/lib/python3.13/site-packages,readonly" \
  --mount "type=bind,src=$SERVICE,dst=/workspace/services/ai/mcp-permissions,readonly" \
  --mount "type=bind,src=/Users/mersinvald/dev/mkl.dev/agentgateway,dst=/agentgateway,readonly" \
  --workdir /workspace/services/ai/mcp-permissions/src \
  -e PYTHONDONTWRITEBYTECODE=1 -e HOME=/tmp python:3.13-slim \
  python -m pytest /workspace/services/ai/mcp-permissions/tests -q -p no:cacheprovider
```

## Known Limits

- A committed grant cannot be recalled by removing a Hermes waiter. The service
  remains authoritative for inspection, revocation, expiry and consumption.
- If HTTP confirmation is lost, Hermes reports uncertainty without retrying.
  Explicit read-only service inspection can establish current state; no poller,
  conversation store, replacement UI or automatic recovery is added here.
- If Telegram acknowledgment and message editing both fail, the receipt may not
  be visible. The consumed nonce prevents a second POST. No delivery is claimed.
- No callback automatically executes MCP, restarts A2A, or implies an external
  action completed. Native consumption remains subject to operation-ID guards.
- The unrelated baseline command-path test previously failed identically on
  unchanged `18a0b6c`; it is not fixed or counted as passing here.
