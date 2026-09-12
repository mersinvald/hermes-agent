"""Actual A2A tool dispatch retains the bound native caller, never a guess."""

import json
import time
from urllib.parse import urlencode
from unittest.mock import patch

import pytest

from agent.conversation_loop import run_conversation as _real_loop
from hermes_cli.lifecycle import invoke_hook
from hermes_state_events import timestamp
from plugins.platforms.a2a import protocol, tools
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
@pytest.mark.parametrize("caller_role", ["primary", "leaf", "orchestrator", "uncaptured"])
async def test_a2a_tool_worker_preserves_actual_caller(monkeypatch, tmp_path, agent, caller_role):
    async with service(monkeypatch, tmp_path, models=model_config()) as (server, client, native):
        db, root = native[2], native[4].session_id
        owner, start = server.ingress._command_owner, time.time() - 1
        db.native_execution_open(root, "dispatch-turn", owner, origin="pwa")
        _ConversationHarness()._setup_agent(agent)
        agent.session_id = root if caller_role == "primary" else "actual-child"
        agent.platform = "pwa" if caller_role == "primary" else "subagent"
        agent.is_subagent = caller_role != "primary"
        agent.tool_use_enforcement = "off"
        agent.valid_tool_names.add("a2a_call")
        agent.client.chat.completions.create.side_effect = [
            _mock_response(content="", tool_calls=[_mock_tool_call(
                name="a2a_call", arguments=json.dumps({"agent": "peer", "message": "PRIVATE arithmetic"}),
            )]),
            _mock_response(content="PRIVATE final answer"),
        ]
        monkeypatch.setattr(tools, "_load_config", lambda: {
            "a2a_agents": {"peer": {"url": "http://peer.test/agent/"}},
        })
        monkeypatch.setattr(tools, "_fetch_card", lambda *_: protocol.build_agent_card(
            name="peer", url="http://peer.test/agent/", description="synthetic",
        ))
        sent = []

        def post(url, body, headers, timeout):
            sent.append(body["method"])
            return {"jsonrpc": "2.0", "id": body["id"], "result": {"task": {
                "id": "remote-task", "contextId": "remote-context",
                "status": {"state": "TASK_STATE_COMPLETED"},
            }}}

        monkeypatch.setattr(tools, "_http_post_json", post)

        def dispatch(name, args, *a, **kw):
            assert name == "a2a_call"
            return tools.a2a_call(args)

        with (
            scope(server, root, "dispatch-turn"),
            patch("agent.conversation_loop.run_conversation", _real_loop),
            patch("run_agent.handle_function_call", side_effect=dispatch),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            if caller_role in {"leaf", "orchestrator"}:
                invoke_hook("subagent_start", parent_session_id=root, parent_turn_id="parent-turn",
                            child_session_id=agent.session_id, child_role=caller_role)
            result = agent.run_conversation("PRIVATE user request")
        assert result["completed"] is True
        assert sent == ["SendMessage"]
        db.native_execution_close("dispatch-turn", owner, outcome="completed")
        task = (await get(client, "/v1/pwa/inspector/tasks"))["rows"][0]["task_ref"]
        facts = await get(client, f"/v1/pwa/inspector/tasks/{task}/facts")
        remote = next(item for item in facts["delegations"] if item["delegation_kind"] == "a2a")
        local = next((item for item in facts["delegations"] if item["delegation_kind"] == "local"), None)
        expected = None if caller_role == "uncaptured" else (
            local["participant_ref"] if local else facts["model_calls"][0]["agent_ref"]
        )
        assert remote["parent_agent_ref"] == expected
        assert remote["peer_name"] == "peer"
        query = urlencode({"from": timestamp(start), "to": timestamp(time.time())})
        metrics = await get(client, "/v1/pwa/inspector/execution-metrics?" + query)
        group = next(item for item in metrics["rows"][0]["delegations"] if item["delegation_kind"] == "a2a")
        assert group["agent_ref"] == expected
        assert group["agent_role"] == (None if caller_role == "uncaptured" else caller_role)
        assert group["call_count"] == 1
        assert "PRIVATE" not in json.dumps([facts, metrics])
