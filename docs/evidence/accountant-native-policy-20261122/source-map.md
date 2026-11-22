# Native call-site inspection

All native line references use `089fadba6cfeacc5af9ebffcec1e6f72ae9eed28`.
All PWA references use accepted main `2a05257840cee4bd421466c1f420df850e6980fb`.
The P10.1 source map supplied the starting points.
This inspection extends those points into actual native dispatch and preprocessing functions.
It does not establish complete source coverage behind provider or plugin adapters.

| Native source | Observed behavior | Required integration |
| --- | --- | --- |
| `gateway/pwa_http.py:65`, `:120` | Listener installs ownership, models, and ingress. Authentication validates bearer and principal binding. | Install the operator-bound admission adapter through this existing constructor. |
| `gateway/pwa_config.py:63`, `:74`, `:404` | Source and owner bindings have no content profiles or auth epoch. Config fingerprint is a hash. | Parse explicit profile assignments and positive epoch/revision. Do not treat a fingerprint as an epoch. |
| `gateway/pwa_ownership.py:140` | Authorization checks current principal, source, assignment, and compression rows. | Reuse this check for current facts. Never accept request-supplied profiles. |
| `gateway/native_commands.py:227`, `:313` | Submit validates command shape and authorization before durable admission. | Bind source admission before acceptance. Preserve the DTO and idempotency. |
| `gateway/conversation_control.py:263`, `:273` | Execution open records process owner and native execution ID. | Retain execution identity and owner checks. Missing restored policy authority must deny enabled replay. |
| `gateway/run.py:20181`, `:20597` | Preprocessing can invoke vision, STT, video, and context expansion before `run_sync`. | Deny sensitive work at both entries before these operations. |
| `gateway/run.py:5960`, `:6080`, `:6479` | Managed execution captures model selection under a lease, then constructs or reuses the agent. | Recheck source admission first. Bind execution and selection before agent use. |
| `gateway/run.py:7245`, `:7341` | Image guard and native execution context attach near `run_conversation`. | Preserve both. Their location is too late to authorize preprocessing. |
| `run_agent.py:8685`, `agent/conversation_loop.py:1899` | Forwarder and loop start per-turn work before the main model call. | Reject enabled missing/stale context before the prologue. |
| `agent/conversation_loop.py:3302` | Main pre-request image guard observes the agent and kwargs. | Preserve image validation. It does not observe every final client. |
| `agent/chat_completion_helpers.py:920`, `:3438`, `:4082` | Nonstream dispatch selects clients by API mode. Stream dispatch constructs a request-local client inside a worker. | Check the exact final client before dispatch. Deny unsupported sensitive stream/thread paths. |
| `agent/chat_completion_helpers.py:2507`, `:2999` | Fallback replaces runtime. Iteration summaries dispatch outside the normal loop, with retry branches. | Guard replacement and every direct-summary branch. |
| `agent/turn_context.py:266`, `:561`, `:1494` | Prologue starts title and memory work before main dispatch. | Carry enabled context and reject unsupported callbacks before launch. |
| `agent/title_generator.py:341`, `:545`, `:721` | Titles derive snippets and use auxiliary calls in a worker. Validator exceptions proceed. | Reject stale policy independently of the advisory runtime validator. Preserve title hooks. |
| `agent/auxiliary_client.py:9438`, `:9729`, `:9836`, `:10758`, `:10801` | Sync/async helpers resolve and retry clients. Strict destination applies only to sync calls. | Guard both entrypoints before resolution and any payload call. Strict origin checks do not supply shared authority. |
| `agent/context_compressor.py:4918`, `:5240`, `:7054` | Full and micro summaries call auxiliary routing. Output returns as text. | Guard before conversion/call and retain constraints on accepted output. |
| `agent/conversation_compression.py:3831`, `:5167` | Compression can invoke memory hooks and rotate session state. | Keep lineage root stable. Unqualified extension callbacks deny sensitive work. |
| `agent/memory_manager.py:594`, `:703`, `:744`, `:948`, `:970`, `:984`, `:1113`, `:1214`, `:1325` | Manager forwards queries, turns, messages, writes, and delegation results. Some operations schedule threads. | Check before callbacks and scheduling. Provider internals and worker handoff remain unsupported. |
| `tools/delegate_tool.py:1702`, `:2067`, `:2576`, `:3656`, `:3829` | Child construction inherits routing/fallback. Finalization sends summaries to memory and hooks. | Bind before child construction. Deny sensitive threads/background launch and unqualified return paths. |
| `tools/async_delegation.py:248`, `:327`, `:347`, `:404` | Durable ledger stores tasks/results and restores completion events. It stores no shared capability. | Retain durability. Deny enabled re-entry without authenticated restoration. |
| `plugins/platforms/a2a/tools.py:129`, `:295`, `:438`, `:474` | Peer resolution precedes discovery and payload HTTP. Send can stream and returns redacted text. | Deny sensitive work before discovery. Redaction cannot clear policy. |
| `plugins/platforms/a2a/provenance.py:99` | Native provenance prepares durable remote dispatch facts. | Keep provenance. It does not authenticate content-policy receivers or their downstream routes. |
| `agent/tool_executor.py:1098`, `:1844`, `:1952`, `:2764` | Concurrent/sequential paths convert results, append canonical messages, then emit completion. | Join authenticated producer constraints before conversion and append. |
| `run_agent.py:7402` | Tool-result conversion can preprocess image content for a nonvision model. | Guard before this preprocessing. Retain existing multimodal behavior for ordinary authenticated content. |

## Reused evidence

P10.1 runtime and tests use frozen source `2b66c7504fb8863208d5dda6c991307fd004a0ca`.
The accepted evidence records 109 policy cases and 33 unchanged transport cases as passing.
It also records one installed wheel/API/import smoke with no real model or network call.
Reuse those results for unchanged code.
The five PostgreSQL-dependent facade skips remain unverified.
Do not represent them as native or facade integration passes.

The accepted PWA module SHA256 is `28fa3b07c5d6f8a64b9aa970fbdb8a212ac2e7a03d9e4fa6d3020c9ddf0554aa`.
Its route parser requires `synthetic_only` and a `synthetic_` qualification reference.
Its runtime enforces one-thread use and rejects reentrancy.
Exact envelope identity cannot cross serialization or process restart.
These are implementation constraints for the native design, not missing production qualifications to infer away.

Infra remains read-only and needs no additional inspection for this component.
The original native checkout and its working files were not used.
This worker did not read or edit `.serena` content.
