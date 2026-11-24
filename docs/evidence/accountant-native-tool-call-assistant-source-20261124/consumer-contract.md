# Consumer contract

Use the original live agent, adapter, scope, and unused probe token.
Select the mode before the original probe sends.

```python
authorization = adapter.bind_tool_call_probe(
    agent,
    system_message=system_message,
    user_message=user_message,
    task_id=task_id,
    expected_arguments_json=expected_arguments_json,
)
adapter.enable_tool_call_assistant_source(agent, authorization)
observation = adapter.run_tool_call_probe(agent, authorization)
(source,) = adapter.tool_call_assistant_sources
```

The returned `observation` remains the original caller result.
`source.source_observation` is that exact object.
`source.response` is the first local stock typed object from the exact retained response bytes.
`source.assistant_message` is its original `choices[0].message`.
`source.finish_reason` is the original choice field.
`source.message` is the actual native builder mapping.

The complete mapping retains its field order, metadata presence, nested identities, and timestamp.
`message_bytes` contains the complete strict UTF-8 serialization.
`message_content` authenticates those bytes with ordered parents `(R, Ba)` through original P on `main`.
`authorization_content` is the single Ba root for complete selected builder-only input facts.
R remains the original response handle.
The authorization digest grants no capability.

The original strict response check runs before typed materialization.
The stock forwarder calls the original `build_assistant_message` helper once.
No caller supplies a replacement message or builder.
Reasoning and stream callbacks remain absent or `None` in the actual controlled constructor.

The owner binds callback code and callable defaults before source callbacks.
The original timestamp helper uses its original wall clock.
The policy clock remains separate.

The new owner permits two distinct constructor transitions.
First, root permits the exact missing backing-field completion before source callbacks.
Second, the actual builder may create only its stock cache tuple.
The owner retains original, completed, and actual post-builder snapshots.
It checks all other namespace entries and identities.
An arbitrary injected tuple fails the registered-cache check.

The mode retains at most 128 strong owners.
It reclaims no owner, slot, byte charge, or window.
The original eight P windows remain exact.
The source mode uses three separate bounded windows.
The mode adds zero sends, agents, envelopes, or capabilities.

A copied-token preflight registers no owner and spends no valid original token.
A repeat opt-in call refuses.
The mode refuses closed agents, attempted probes, terminal records, and foreign bindings.
Callback reentry and foreign threads cannot read a source result.

The getter returns original records only after source checks and original probe release.
A denied or partial owner supplies no result.
Getter access grants no admission or execution authority.

Every callback retains the original selected grant, destination, holder, producer, restrictions, coverage, and runtime health checks.
Alternate live coverage cannot replace the expired selected grant.
Source restrictions remain in the retained graph after current labels disappear.
Late refusal keeps raw output, complete bytes, actual cache state, and charged handles.
The original control exception keeps its identity through shared normalization.

Native values remain bounded at 262144 bytes.
Strings remain bounded at 16384 characters and UTF-8 bytes.
Shared limits remain 32 parents, 128 combined slots, 8 MiB retained bytes, 1 MiB payloads, and 128 hops.
The default probe and ordinary direct dispatch keep their original behavior and charges.

This component stops before assistant/source association, transcript append, tool execution, pairing, sanitization, persistence, or another request.
Root selects any later component separately.
