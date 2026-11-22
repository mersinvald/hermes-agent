# Bounded native validation plan

Status: proposed after runtime approval. This design stage runs no runtime tests.
Use invented documents, temporary `HERMES_HOME`, real native imports, and fake provider/transport ports.
Do not read user configuration or credentials.
Use `scripts/run_tests.sh` for every native Python selection.

## Actual changed native functions

| Test file | Functions exercised | Required assertions |
| --- | --- | --- |
| `tests/gateway/test_native_content_policy.py` | `NativeConversationIngress.submit`, preprocessing entries, `TurnRunner.run_sync` | Missing adapter/assignment denies before preprocessing and agent construction. Current owner, execution, lineage, epoch, selection, and revision must agree. |
| `tests/agent/test_content_policy.py` | Native binding helpers and shared engine import | Authenticated ordinary and constrained bindings differ from absent context. Shared import identity and byte/result bounds hold. |
| `tests/agent/test_content_policy_dispatch.py` | `_dispatch_nonstreaming_api_request`, streaming entry, `try_activate_fallback`, `handle_max_iterations` | Call actual functions with fake clients. Denied real/unknown clients receive zero payload calls. Check each API mode and direct-summary retry. |
| `tests/agent/test_content_policy_auxiliary.py` | `call_llm`, `async_call_llm`, `generate_title`, full/micro summary methods | Deny before sync/async provider dispatch and title thread launch. Validator exceptions grant no permission. Ordinary calls preserve existing behavior. |
| `tests/agent/test_content_policy_memory.py` | Manager prefetch, sync, tool, lifecycle, write, compression, and delegation methods | Deny before callback or executor submission. A provider name never proves its internal destination. |
| `tests/tools/test_content_policy_delegation.py` | `delegate_task`, child construction/run, finalization, completion restore | Deny sensitive child dispatch and restored re-entry. Retain records and ordinary ownership behavior. Preserve constraints on returned results. |
| `tests/plugins/test_a2a_content_policy.py` | `_send_task` | Deny before discovery, audit, persistence, or HTTP. Fake post/stream functions receive zero sensitive calls. |
| `tests/agent/test_content_policy_tool_results.py` | Both tool executors and `_tool_result_content_for_active_model` | Authenticated producer promotion constrains earlier handles. Unknown context cannot become ordinary through string or image conversion. |

Fake transports count dispatches and capture exact call arguments.
Use failing sentinels at provider entrypoints and preprocessing callbacks for negative cases.
Call real resolver functions against temporary configuration when destination resolution changes.
Patch only the final provider/transport boundary, clock, and authenticated host facts.
Do not replace changed native functions with mocks or read source text in tests.

## Controlled service simulations

Use one `document-review` specialist with no financial code.
Exercise staged shared `dispatch` and `accept` using the authoritative installed package.
Retain the profile union on summaries, replies, and tool-result promotion.
Record these as in-process service simulations.
They do not establish native child threads, peer authentication, durable recovery, or local egress.
Reuse P10.1 policy results instead of repeating its complete 109-case selection.

## Feedback and final affected slice

During edits, select only the new file for the changed boundary.
Target feedback within two minutes.
Record slow dependency setup before expanding validation.
Use one final run of the eight new files, with a ten-minute limit.
Stop and report a partial scope if setup or execution exceeds that limit.

Run existing compatibility cases only for paths changed by the accepted implementation.
Select explicit tests for command idempotency, ordinary title strict routing, image retention, and ordinary completion ownership.
Record exact node IDs after inspecting the implemented diff.
Do not run complete native, PWA, or Accountant suites.
Do not repeat complete installed selections or rebuild historical artifacts.

The shared-package companion needs one artifact/import smoke after extraction.
It must prove one implementation and identity across old PWA and new native imports.
Its Python 3.11/3.13 compatibility claim needs actual interpreter validation or an explicit unresolved gate.
Do not expand native validation into the complete synthetic suite by default.

## Evidence and acceptance limits

Record source commit, exact selection, duration, exit code, failures, and skips.
Retain partial and failed reports with their original scopes.
Hash reports without embedding content or private diagnostics.
Lint new prose with the global STE linter.
Check staged paths before commit and final status after commit.

Acceptance must distinguish native denial, native fake-provider invocation, and shared service simulation.
Sensitive real-provider dispatch remains denied in this proposal.
Serialized authentication, durable policy restoration, concurrent handoff, deployed differences, and actual egress remain separate gates.
Full P10 and financial activation remain open.
