# Restricted native PWA transport (N07)

The opt-in listener runs inside the existing `GatewayRunner`, borrowing the
`SessionStore` SQLite handle and `NativeConversationIngress`. Native leases,
durable N02 commands, provider execution and gateway drain/recovery remain the
execution authority. This service never constructs a separate agent executor.

Wire reference: PWA contract pin
`e0b1e40581d8aba1f1a59985c0f8268ed3b3e111`,
`contracts/native-pwa-http.md` and `contracts/v1/native-pwa.schema.json`.
All routes are below `/v1/pwa/`. Health/capabilities, conversation list/create,
inspect/history, command submit and receipt lookup require the private facade
credential and the strict trusted-principal header. There are no browser login,
general agent, arbitrary file/database, event, or administrative mutation routes.

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
ambiguous source assignments fail closed. Non-loopback IP binds additionally
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

Opaque cursors live in a bounded process-local map (4096 entries), bound to
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
Event/replay/snapshot/cancellation and Telegram final-delivery capabilities remain
unavailable until their native packages are integrated and verified. No live
deployment or production credentials are used here.
