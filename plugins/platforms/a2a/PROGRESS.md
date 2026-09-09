# Native A2A Progress Candidate

Opt in per configured peer in native `config.yaml`; the model endpoint and
permissions are unchanged. No prompt prelude, extra model call, service or
direct messaging API is involved. Raw `display.tool_progress` can remain off.

```yaml
a2a_agents:
  groundskeeper:
    # Keep the existing URL, auth, timeout and tenant configuration.
    display_name: "Специалист по дому"
    progress_notify: true
    streaming: true
    progress_messages:
      dispatch: "Передаю запрос специалисту по дому."
      follow: "Проверяю ход существующей задачи специалиста."
      tool: "Специалист выполняет инструмент."
      tool_result: "Специалист получил результат инструмента."
      tool_error: "Источник вернул ошибку специалисту."
      "tool:GetLiveContext": "Специалист запрашивает актуальные данные Home Assistant"
      "tool:load_memory": "Специалист проверяет сохранённый контекст"
```

`display_name` defaults to `Specialist`. English defaults exist for dispatch,
tool, tool result and tool error; `{peer}` substitutes the configured display
name. Catalog keys match exact observed function names, not model arguments.
Unknown names use the generic tool phrase. Catalogs are local configuration,
never peer-provided prose. `progress_notify` and `streaming` default to false.

The existing status callback receives `tool_activity` and the gateway uses its
native edit-in-place status lane, scoped to the originating chat/topic. It
allows one in-flight delivery plus one latest pending status per turn. Updates
are coalesced, not lost behind an in-flight send; duplicate text is suppressed.
At least two seconds separate delivery completion and the next update. Each
TurnRunner owns its message ID, so handover cannot reuse another turn's bubble.
Native finalization closes callbacks and drains before final delivery/cleanup.
Successful runs register the native post-delivery callback for `cleanup_progress`;
stale, canceled or timed-out deliveries delete any late acknowledged own ID.
Notification failure never retries a peer action or blindly creates a new bubble.

The drain is bounded to 30 seconds and shields an already-started native adapter
operation so late acknowledgements can still be cleaned up. Adapter network
timeouts remain in force. An unacknowledged remote Telegram send cannot provide
an ID to delete; transport timeout/cancellation cannot guarantee exactly-once
remote delivery. Deletes are best-effort, not a claimed transactional guarantee.

Streaming uses the discovered JSON-RPC version and `capabilities.streaming`:
`message/stream` for 0.3 and `SendStreamingMessage` for 1.0. Without capability,
the existing unary request still emits dispatch status. A send is never retried
or downgraded on HTTP/auth/RPC errors or stream interruption. Interrupted tasks
with established IDs are persisted as unknown, not canceled. Existing explicit
input-required continuation and conditional GetTask negotiation are retained.

Explicit read-only observation uses the existing tool:

```json
{"agent":"groundskeeper","action":"follow","task_id":"<persisted-task>","context_id":"<persisted-context>"}
```

No message/data is accepted for follow. It requires persisted configured-peer,
origin and context binding, validates GetTask, returns terminal/pending snapshots
directly, or subscribes via `tasks/resubscribe` (0.3) / `SubscribeToTask` (1.0).
Discovery, authorization or subscription errors never restart work. Unsupported
subscriptions report failure; they do not resend. SSE event IDs are bounded,
deduplicated and persisted with intermediate task state. The pinned SDK's
SubscribeToTask request has no replay cursor: stored IDs are diagnostic only,
not sent as invented resume parameters. Gapless replay is not promised.

SSE is bounded to 256 KiB per event/request, 8 MiB per response, 4096 events and
the configured timeout (a blocking read may additionally consume its socket
timeout). JSON-RPC request IDs and established peer task/context IDs must match.
Final pending questions take priority over artifacts. UI statuses never render
peer text, tool arguments/results, task IDs, credentials or reasoning. Final
model output excludes marked thought parts; pending question DataParts retain
their explicit-user-answer semantics. Artifact append requires a preceding
artifact and cannot follow `lastChunk`. Replace updates retain artifact order;
append joins text fragments without injecting newlines. All nonempty final text
artifacts are returned, not only the first. Marked thoughts and tool DataParts
are excluded from final output (explicit pending-question DataParts are retained).
Function events require bounded string name/id and a structurally valid payload;
function responses require an object `response`. Go ADK omits empty `args`, which
is accepted, but a present non-object args value is not progress evidence.

Source references checked locally: kagent v0.10.0 `go/adk/pkg/a2a/executor.go`
emits non-partial function events even without LLM token streaming;
Go ADK v2.1.0 uses `adk_type=function_call/function_response` and `adk_thought`.
kagent also recognizes `kagent_type`. This is immediate peer-task event
visibility, not reconstructed recursive traces or a latency reduction.

## Review Gate

`test_native_kagent_progress.py` compiles a test-only fixture inside an isolated
kagent v0.10.0 checkout at `ae008d1e19fbb195ba3c8205fe345d17309a9158`. It uses the
actual KAgentExecutor, ADK function tool, native event queue and controller
PassthroughRequestHandler with its legacy compatibility transport. Go model
token streaming remains false. No runtime source in kagent is modified.

The local Docker Gateway uses the pilot's read-only-verified deployed image:
`dcr.home.lan.mkl.dev/library/agentgateway@sha256:12a133d46327ee3b4fd12ad913ad30f575007eafce427641719f6b702b8d528e`.
Fixed peer path, API-key authentication, request-body bounds and method guards
remain enabled; wrong key/path/query, CancelTask and malformed follow are denied.
Both protocol versions prove non-EOF tool/result delivery while native executor
barriers remain held, plus explicit read-only follow on the same running task.

Hermes runs the native AIAgent loop with a tool-only first model response and
no raw tool progress. Actual TelegramAdapter send/edit/delete methods run with
mocked Bot transport. Observed order is send(101), edit(101), edit(101),
final(102), delete(101); native post-delivery cleanup owns the deletion.
Both parent and specialist use exactly two model calls, no narration calls.
Separate event-barrier tests cover pending coalescence, bounded drain timeout,
cancellation, handover and deletion of late acknowledgements without touching
the successor's message. Transport tests cover malformed events, credential
rotation across artifact fragments, event-ID deduplication and pending priority.

Reproduce the opt-in native test (cached modules, no dependency mutation):

```sh
HERMES_PYTHON=/path/to/test/python KAGENT_SOURCE=/path/to/isolated/kagent-v0.10.0 \
  /path/to/test/python -m pytest tests/gateway/test_native_kagent_progress.py -k launch -q -s -o addopts=''
```

The launcher uses GOPROXY=off and GOFLAGS=-mod=readonly and removes only its
owned temporary Go test file/container. Production UI remains untested. No
production polling, messages, HA actions, deployments, image build/publication
or infrastructure commits were performed. Source review precedes packaging.

The publisher's exact runtime allowlist now includes all seven changed runtime
files, checks unchanged dependency manifests and pinned upstream layers, and
handles the one explicitly new module. Review source first; then run the native
publisher for arm64 and amd64 with a unique commit-derived Harbor candidate tag.
Do not deploy this source checkpoint without completing the above gates.

The final gateway/A2A selection passed 371 tests. Executor segmentation/context
propagation separately passed 35 tests with one skip. The pinned Go fixture
passed both 0.3/1.0 subtests, each invoking the real Gateway/Telegram contract
test. The original 9e4d715 checkpoint had 344 gateway/A2A tests. Combining
both selections in one pytest process exposes a registry-discovery ordering
failure in `TestMultiAgentRouting.test_path_routed_agent_card_uses_prefix_and_canonical_path`.
The identical failure was reproduced on unchanged b80e2d1 with that test plus
`test_tool_batch_segmentation.py`; it is not fixed or hidden by this patch.
The local uv version warns about upstream `exclude-newer = "14 days"`; test
dependencies were supplied via a disposable `--no-project --python 3.13`
environment, without changing project dependency files.
