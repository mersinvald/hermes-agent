"""Real conversation-loop exits retain provider receipts through native HTTP."""

import json
import time
from dataclasses import replace
from types import SimpleNamespace
from urllib.parse import urlencode
from unittest.mock import patch

import pytest

from agent.conversation_loop import run_conversation as _real_loop
from hermes_constants import PARTIAL_STREAM_STUB_ID
from hermes_state_events import timestamp
from tests.gateway.test_pwa_http import service, model_config
from tests.gateway.test_pwa_inspector import get
from tests.gateway.test_pwa_inspector_facts import scope
from tests.run_agent.test_run_agent import (
    agent as agent,
    TestRunConversation as _ConversationHarness,
    _mock_response,
    _mock_tool_call,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("first_kind", [
    "normal", "length", "tool_length", "content_filter", "partial_stream",
    "partial_stream_usage", "empty", "malformed", "exception",
])
async def test_received_response_boundary_preserves_usage_and_retry_behavior(
    monkeypatch, tmp_path, agent, first_kind,
):
    async with service(monkeypatch, tmp_path, models=model_config()) as (
        server, client, native,
    ):
        db, root = native[2], native[4].session_id
        start = time.time() - 1
        server.ownership.config = replace(
            server.ownership.config,
            inspector_model_names=("actual/model",),
            inspector_provider_names=("custom",),
        )
        owner = server.ingress._command_owner
        db.native_execution_open(root, "response-turn", owner, origin="pwa")
        _ConversationHarness()._setup_agent(agent)
        agent.session_id, agent.platform = root, "pwa"
        agent.model, agent.provider = "upstream/daily", "custom"
        agent.base_url = "http://127.0.0.1:1/v1"
        agent.tool_use_enforcement = "off"
        agent._api_max_retries = 2
        monkeypatch.setattr("agent.conversation_loop.jittered_backoff", lambda *a, **k: 0)
        if first_kind == "normal":
            from hermes_cli.observability import native_inspector

            observe = native_inspector.observe_response

            def repeated(*args, **kwargs):
                observe(*args, **kwargs)
                observe(*args, **kwargs)

            monkeypatch.setattr(native_inspector, "observe_response", repeated)
        first = _mock_response(
            content="PRIVATE first half ",
            finish_reason=first_kind if first_kind in {"length", "content_filter"} else "stop",
            usage={"prompt_tokens": 19, "completion_tokens": 5, "total_tokens": 24},
        )
        first.model = "actual/model"
        if first_kind == "tool_length":
            first.choices[0].finish_reason = "length"
            first.choices[0].message.content = ""
            first.choices[0].message.tool_calls = [_mock_tool_call(
                name="read_file", arguments='{"path":"PRIVATE', call_id="truncated",
            )]
        elif first_kind in {"partial_stream", "partial_stream_usage"}:
            first.id = PARTIAL_STREAM_STUB_ID
            first.choices[0].finish_reason = "length"
            if first_kind == "partial_stream":
                first.usage = None
        elif first_kind == "empty":
            first.choices[0].message.content = ""
        elif first_kind == "malformed":
            first = SimpleNamespace(choices=[], model="actual/model", usage=first.usage)
        elif first_kind == "exception":
            first = RuntimeError("PRIVATE transport failure")
        continuation = _mock_response(
            content="PRIVATE final answer",
            usage={"prompt_tokens": 31, "completion_tokens": 7, "total_tokens": 38},
        )
        continuation.model = "actual/model"
        replies = [first] if first_kind in {"normal", "content_filter"} else [first, continuation]
        agent.client.chat.completions.create.side_effect = replies
        with (
            scope(server, root, "response-turn"),
            patch("agent.conversation_loop.run_conversation", _real_loop),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("run_agent.handle_function_call") as execute_tool,
        ):
            result = agent.run_conversation("PRIVATE user request")
        assert agent.client.chat.completions.create.call_count == len(replies)
        assert result["completed"] is (first_kind != "content_filter")
        execute_tool.assert_not_called()  # Truncated tool arguments never execute.
        db.native_execution_close("response-turn", owner, outcome="completed")
        task = (await get(client, "/v1/pwa/inspector/tasks"))["rows"][0]["task_ref"]
        facts = await get(client, f"/v1/pwa/inspector/tasks/{task}/facts")
        calls = facts["model_calls"]
        # Error retries and tool-argument retries share a logical request; text
        # continuations/empty-response retries start a new logical request.
        separate = first_kind in {"length", "partial_stream", "partial_stream_usage", "empty"}
        assert len(calls) == (2 if separate else 1)
        first_call = calls[0]
        assert first_call["state"] == (
            "failed" if first_kind in {"content_filter", "partial_stream", "partial_stream_usage", "empty"} else "completed"
        )
        assert first_call["duration_ms"] >= 0
        assert first_call["requested_model"] == "upstream/daily"
        assert first_call["provider"] == "custom"
        assert first_call["actual_model"] == (None if first_kind.startswith("partial_stream") else "actual/model")
        expected_input = (
            None if first_kind == "partial_stream" else
            50 if first_kind in {"tool_length", "malformed"} else
            31 if first_kind == "exception" else 19
        )
        assert first_call["input_tokens"] == expected_input
        assert first_call["output_tokens"] == (
            None if expected_input is None else 12 if expected_input == 50 else 7 if expected_input == 31 else 5
        )
        assert first_call["attempt_count"] == (2 if first_kind in {"tool_length", "malformed", "exception"} else 1)
        assert first_call["failed_attempt_count"] == (
            1 if first_kind in {"content_filter", "partial_stream", "partial_stream_usage", "empty", "malformed", "exception"} else 0
        )
        query = urlencode({"from": timestamp(start), "to": timestamp(time.time())})
        metrics = await get(client, "/v1/pwa/inspector/execution-metrics?" + query)
        row = metrics["rows"][0]
        assert row["model_call_count"] == len(calls)
        assert row["input_tokens"] == (31 if first_kind in {"partial_stream", "exception"} else 19 if len(replies) == 1 else 50)
        assert row["output_tokens"] == (7 if first_kind in {"partial_stream", "exception"} else 5 if len(replies) == 1 else 12)
        # The native receipt does not rerun or move ordinary session accounting.
        assert agent.session_api_calls == (0 if first_kind == "content_filter" else 2 if first_kind == "empty" else 1)
        assert "PRIVATE" not in json.dumps([facts, metrics])


@pytest.mark.parametrize("mode", ["anthropic_messages", "codex_responses"])
def test_boundary_normalization_preserves_response_and_tool_names(monkeypatch, mode):
    from copy import deepcopy
    from hermes_cli.observability import native_inspector
    from agent.transports.anthropic import AnthropicTransport
    from agent.transports.codex import ResponsesApiTransport

    if mode == "anthropic_messages":
        transport = AnthropicTransport()
        response = SimpleNamespace(
            stop_reason="tool_use", model="actual/model", usage=None,
            content=[SimpleNamespace(type="tool_use", name="mcp__read_file", id="call", input={"path": "PRIVATE"})],
        )
        from tools.registry import registry

        monkeypatch.setattr(registry, "get_entry", lambda name: object() if name == "read_file" else None)
    else:
        transport = ResponsesApiTransport()
        response = SimpleNamespace(
            status="incomplete", incomplete_details={"reason": "content_filter"},
            output=[], model="actual/model", usage=None,
        )
    actor = SimpleNamespace(
        session_id="session", model="requested/model", provider="custom", api_mode=mode,
        _is_anthropic_oauth=True, _get_transport=lambda: transport,
        _usage_summary_for_api_request_hook=lambda response: None,
    )
    before, captured = deepcopy(response), []
    monkeypatch.setattr(native_inspector, "handles_hook", lambda name: True)
    monkeypatch.setattr(native_inspector, "observe_lifecycle", lambda name, **values: captured.append(values))
    native_inspector.observe_response(actor, response, api_request_id="request", started_at=1, ended_at=2, retry_count=0)
    assert response == before
    assert captured[0]["response_completed"] is (mode == "anthropic_messages")
    normalized = transport.normalize_response(response, strip_tool_prefix=True)
    if mode == "anthropic_messages":
        assert response.content[0].name == "mcp__read_file"
        assert normalized.tool_calls[0].name == "read_file"
    else:
        assert normalized.finish_reason == "content_filter"
