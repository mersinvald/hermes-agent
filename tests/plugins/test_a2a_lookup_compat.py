"""A single read-only method fallback must never become a write retry."""
import copy
import json
import urllib.error

import pytest

from plugins.platforms.a2a import protocol, tools


@pytest.fixture
def lookup_peer(monkeypatch):
    config = {"a2a_agents": {"peer": {"url": "http://peer.test", "auth": {
        "type": "bearer", "token": "SYNTHETIC_LOOKUP_A"}}}}
    monkeypatch.setattr(tools, "_load_config", lambda: config)
    monkeypatch.setattr(tools, "_fetch_card", lambda *_: protocol.build_agent_card(
        name="peer", url="http://peer.test/rpc", description="test", tenant="fixture"))
    calls = []
    pending = {"id": "task-1", "contextId": "context-1", "status": {
        "state": "input-required", "message": {"parts": [{"text": "Which color?"}]}}}
    options = {"first": None, "fallback": None, "send": None}

    def post(url, body, headers, timeout):
        calls.append((url, copy.deepcopy(body), dict(headers), timeout))
        mode = {"GetTask": "first", "tasks/get": "fallback", "SendMessage": "send"}[body["method"]]
        override = options[mode]
        if isinstance(override, Exception):
            raise override
        if override:
            return override(body)
        if mode == "first":
            return {"jsonrpc": "2.0", "id": body["id"], "error": {"code": -32601, "message": "Method not found"}}
        task = copy.deepcopy(pending)
        if mode == "send":
            task["status"]["state"] = "TASK_STATE_COMPLETED"
            task["status"]["message"]["parts"][0]["text"] = "blue"
        return {"jsonrpc": "2.0", "id": body["id"], "result": task}

    monkeypatch.setattr(tools, "_http_post_json", post)
    args = {"agent": "peer", "message": "", "context_id": "context-1", "task_id": "task-1",
            "data": [{"decision_type": "approve", "ask_user_answers": [{"answer": ["blue"]}]}]}
    return config, calls, options, pending, args


def test_lookup_fallback_keeps_write_version_task_auth_and_origin(lookup_peer):
    _, calls, _, _, args = lookup_peer
    assert "completed" in tools.a2a_call(args)
    assert [call[1]["method"] for call in calls] == ["GetTask", "tasks/get", "SendMessage"]
    assert [call[2]["A2A-Version"] for call in calls] == ["1.0", "0.3", "1.0"]
    assert len({call[1]["id"] for call in calls}) == 3
    assert all(call[0] == "http://peer.test/rpc" for call in calls)
    assert all(call[2]["Authorization"] == "Bearer SYNTHETIC_LOOKUP_A" for call in calls)
    assert calls[0][1]["params"] == calls[1][1]["params"] == {"id": "task-1", "tenant": "fixture"}
    message = calls[-1][1]["params"]["message"]
    assert message["contextId"] == "context-1" and message["taskId"] == "task-1"
    assert message["role"] == "ROLE_USER" and message["parts"][0]["data"] == args["data"][0]
    assert "kind" not in message["parts"][0]


@pytest.mark.parametrize("code", [-32001, -32002, -32600, -32602, "-32601", -32601.0, None])
def test_other_rpc_errors_never_fallback(lookup_peer, code):
    _, calls, options, _, args = lookup_peer
    options["first"] = lambda body: {"jsonrpc": "2.0", "id": body["id"], "error": {"code": code, "message": "no"}}
    assert tools.a2a_call(args).startswith("Error:") and len(calls) == 1


@pytest.mark.parametrize("mutation", [
    {"id": "wrong"}, {"id": None}, {"jsonrpc": "1.0"}, {"jsonrpc": None}, {"result": {}},
    {"error": {"code": -32601}}, {"error": {"code": -32601, "message": []}}])
def test_malformed_method_not_found_does_not_retry(lookup_peer, mutation):
    _, calls, options, _, args = lookup_peer
    options["first"] = lambda body: {"jsonrpc": "2.0", "id": body["id"],
        "error": {"code": -32601, "message": "not found"}, **mutation}
    assert tools.a2a_call(args).startswith("Error:") and len(calls) == 1


@pytest.mark.parametrize("stage", ["first", "fallback", "send"])
@pytest.mark.parametrize("error", [TimeoutError(), urllib.error.URLError("ambiguous"),
    json.JSONDecodeError("bad", "?", 0), *[urllib.error.HTTPError("http://peer.test", code, "no", {}, None)
        for code in (400, 401, 403, 404, 429, 500, 503)]])
def test_transport_errors_never_retry(lookup_peer, stage, error):
    _, calls, options, _, args = lookup_peer
    options[stage] = error
    assert tools.a2a_call(args).startswith("Error:")
    assert len(calls) == {"first": 1, "fallback": 2, "send": 3}[stage]
    assert sum(call[1]["method"] == "SendMessage" for call in calls) <= 1


@pytest.mark.parametrize("mutation", [{"id": "different"}, {"contextId": "different"},
    *[{"status": {"state": state}} for state in ("completed", "canceled", "failed", "working", "auth-required")]])
def test_fallback_still_requires_same_pending_task(lookup_peer, mutation):
    _, calls, _, pending, args = lookup_peer
    pending.update(mutation)
    assert tools.a2a_call(args).startswith("Error:") and len(calls) == 2


@pytest.mark.parametrize("payload", [[], {}, {"jsonrpc": "2.0", "id": "wrong", "result": {}},
    {"jsonrpc": "2.0", "result": {}}, {"jsonrpc": "1.0", "result": {}}])
def test_malformed_fallback_response_never_sends(lookup_peer, payload):
    _, calls, options, _, args = lookup_peer
    options["fallback"] = lambda body: payload
    assert tools.a2a_call(args).startswith("Error:") and len(calls) == 2


@pytest.mark.parametrize("stage", ["fallback", "send"])
def test_repeated_method_not_found_stops(lookup_peer, stage):
    _, calls, options, _, args = lookup_peer
    options[stage] = lambda body: {"jsonrpc": "2.0", "id": body["id"], "error": {"code": -32601, "message": "no"}}
    assert tools.a2a_call(args).startswith("Error:")
    assert len(calls) == (2 if stage == "fallback" else 3)


def test_credential_rotation_during_read_retains_snapshot(lookup_peer):
    config, calls, options, pending, args = lookup_peer

    def fallback(body):
        config["a2a_agents"]["peer"]["auth"]["token"] = "SYNTHETIC_LOOKUP_B"
        return {"jsonrpc": "2.0", "id": body["id"], "result": pending}

    options["fallback"] = fallback
    assert "completed" in tools.a2a_call(args)
    assert all(call[2]["Authorization"] == "Bearer SYNTHETIC_LOOKUP_A" for call in calls)


def test_legacy_peer_does_not_get_a_second_legacy_attempt(lookup_peer, monkeypatch):
    _, calls, options, _, args = lookup_peer
    monkeypatch.setattr(tools, "_fetch_card", lambda *_: {"protocolVersion": "0.3", "url": "http://peer.test/rpc"})
    options["fallback"] = lambda body: {"jsonrpc": "2.0", "id": body["id"], "error": {"code": -32601, "message": "no"}}
    assert tools.a2a_call(args).startswith("Error:") and len(calls) == 1
