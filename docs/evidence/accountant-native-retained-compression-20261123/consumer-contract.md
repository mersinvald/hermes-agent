# Retained first-summary proposal

This component returns one proven summary from an original successful native completion.
It uses synchronous `synthetic_native`, nonstreaming `chat_completions`, and the existing checked dispatcher.
The adapter owns one actual `ContextCompressor` and an independent compression binding.

## Public API

```python
adapter.bind_compression(
    agent,
    request_handle=compression_capability,
    selection=compression_selection,
    producer_ref="compression_producer",
    route_ref="compression_compression",
    context_length=65536,
    summary_date="2026-11-23",
)
compressor = ContextCompressor(
    model=agent.model,
    provider="synthetic_native",
    api_mode="chat_completions",
    summary_model_override="invented_compression_model",
    config_context_length=65536,
    quiet_mode=True,
    tail_mode="lean",
    native_main_adapter=adapter,
)
token = adapter.compression_input(agent, source=completion)
summary = compressor._generate_summary(
    completion["messages"], controlled_aux=token,
)
```

The host must supply the genuine capability, selection port, producer, and route before adapter binding.
The example identifiers refer to invented operator fixtures. They grant no authority by themselves.
The exact valid calendar date and positive context length are explicit binding facts.
The controlled constructor does not resolve a model context length or a timezone.

The source is the original complete `completion["messages"]` list.
Every original dictionary, row, value, and nested identity remains bound to the successful main completion.
Copied completions, copied rows, foreign tokens, changed helpers, and changed constructor state refuse.

Only the constructor and summary method gain optional keyword arguments.
All prior adapter signatures and main/title/local record fields remain unchanged.
Ordinary construction keeps its defaults and lazy optional dependency behavior.

## Proof and release

The existing ordered auxiliary ledger holds title-local, title-sent, and compression-sent frames.
Exact closed frame types select the retained binding, path, and task.
The adapter checks every installed binding at each checkpoint.
Later main and auxiliary operations inherit completed outputs with their original capabilities and operation order.
The public `derive_for_selection()` method preserves selection boundaries.

Seven input records retain these complete facts:

1. Constructor options, original source field, absent previous summary, absent focus, and empty memory.

2. Actual rough estimate, ratio, context length, maximum, and chosen summary budget.

3. Complete serialization and ordered skill markers before reduction.

4. Complete sampled text, sampling decision, original length, slice count, and aggregate cap.

5. Memory sanitization and escaping, user provenance, and the retained date.

6. The actual complete first-summary prompt and lean session-log instructions.

7. Every normalized call argument and the original complete request dictionary.

The preparation retains the original call arguments, request dictionary, message list, and prompt objects.
Sealing uses that registered preparation without another source transformation.
The actual wire request contains only `model`, `messages`, and `stream=False`.
Temperature and output-token cap arguments are `None` and do not enter the wire request.

Four output records retain these complete facts:

1. Extracted content, finish reason, nonempty decision, and length-stop decision.

2. Think stripping, trim choice, and redaction.

3. Marker restoration, task grounding, lean augmentation, and user provenance.

4. Prefix normalization and the exact final `{"summary": prefixed_summary}` output.

The invocation retains the actual envelope, destination, raw response, native response, and shared result handle when available.
A length stop reaches the actual summary response helper and refuses without fallback.
Main and title response validation still requires `stop`.

The method returns the retained summary field by identity after final policy and cancellation checks.
Empty or truncated output refuses. Policy denial and cancellation do not become semantic `None`.
A refusal after publication can retain completed proof but releases no summary.
Explicit cancellation and process-control exceptions retain their identities.
Agent cancellation raises the fixed bare `InterruptedError`.

## State and limits

The proposal does not adopt history or assign `_previous_summary`.
It does not change cached prefixes, messages, rolling state, session state, telemetry, cooldown, or `compression_count`.
Main records, sends, response identities, model, and token totals remain unchanged.
One actual compression send increments `auxiliary_send_count` once.
The title-local collection remains title-only.


- One pending main or auxiliary operation. Eight total auxiliary inputs. Eight input/output stages per operation.

- Source, stage, request, and final records: 262144 bytes each. Raw responses: 65536 bytes.

- Requests: 32 messages. Each message: 16384 UTF-8 bytes.

- Shared parents: 32. Shared handles and envelopes: 128.

- Shared retained bytes: 8 MiB. Shared payload: 1 MiB. Shared hops: 128.

Shared capacity can refuse before a native maximum.
The adapter keeps complete proof and retains genuine handles when capacity refuses.

Both history-compression host entries refuse before their existing effects.
Full compression, micro-summary, automatic compaction, fallback, persistence, background, async, streaming, and real providers remain excluded.
No deployment, activation, financial write, or private inference follows.
Full P10/P06 and all recorded external gates remain open.
Root owns source acceptance, merge, Beads closure, and later component selection.
