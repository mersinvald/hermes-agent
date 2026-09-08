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
allows one pending delivery per turn and at least two seconds between updates;
fast/duplicate stages are dropped rather than queued. Message IDs are registered
with existing `cleanup_progress`. Notification failure never retries an action.

Streaming uses the discovered JSON-RPC version and `capabilities.streaming`:
`message/stream` for 0.3 and `SendStreamingMessage` for 1.0. Without capability,
the existing unary request still emits dispatch status. A send is never retried
or downgraded on HTTP/auth/RPC errors or stream interruption. Interrupted tasks
with established IDs are persisted as unknown, not canceled. Existing explicit
input-required continuation and conditional GetTask negotiation are retained.

SSE is bounded to 256 KiB per event/request, 8 MiB per response, 4096 events and
the configured timeout (a blocking read may additionally consume its socket
timeout). JSON-RPC request IDs and established peer task/context IDs must match.
Final pending questions take priority over artifacts. UI statuses never render
peer text, tool arguments/results, task IDs, credentials or reasoning. Final
model output excludes marked thought parts; pending question DataParts retain
their explicit-user-answer semantics. Artifact append requires a preceding
artifact and cannot follow `lastChunk`.

Source references checked locally: kagent v0.10.0 `go/adk/pkg/a2a/executor.go`
emits non-partial function events even without LLM token streaming;
Go ADK v2.1.0 uses `adk_type=function_call/function_response` and `adk_thought`.
kagent also recognizes `kagent_type`. This is immediate peer-task event
visibility, not reconstructed recursive traces or a latency reduction.

## Review Gate

Local tests cover native AIAgent with a no-text tool-call model response,
real loopback HTTP (unary and both SSE bindings), native gateway/Telegram
status routing with mocked send/edit, pre-completion notification, one edited
bubble, no extra model request, and cleanup-ID registration. Transport tests
cover pending priority, credential/auth rejection, mismatched IDs, malformed
streams, interruption without replay, and optional callbacks.

Not yet validated: actual kagent Go executor plus model stub; isolated
AgentGateway method authorization and non-EOF forwarding; full Telegram API
formatting/deletion; production UI. No production polling, messages, HA actions,
deployments, image build or publication were performed for this candidate.
These are activation gates, not claimed successes.

The publisher's exact runtime allowlist now includes all seven changed runtime
files, checks unchanged dependency manifests and pinned upstream layers, and
handles the one explicitly new module. Review source first; then run the native
publisher for arm64 and amd64 with a unique commit-derived Harbor candidate tag.
Do not deploy this source checkpoint without completing the above gates.

Validation checkpoint: gateway/A2A selection passed 344 tests; executor
segmentation/context propagation separately passed 35 with one skip. Combining
both selections in one pytest process exposes a registry-discovery ordering
failure in `TestMultiAgentRouting.test_path_routed_agent_card_uses_prefix_and_canonical_path`.
The identical failure was reproduced on unchanged b80e2d1 with that test plus
`test_tool_batch_segmentation.py`; it is not fixed or hidden by this patch.
The local uv version warns about upstream `exclude-newer = "14 days"`; test
dependencies were supplied via a disposable `--no-project --python 3.13`
environment, without changing project dependency files.
