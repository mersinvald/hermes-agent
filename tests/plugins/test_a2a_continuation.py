"""Bounded structured continuations; no automatic decisions or authority claims."""
import copy

import pytest

from plugins.platforms.a2a import protocol, tools


@pytest.fixture
def peer(monkeypatch):
    card = protocol.build_agent_card(name="test", url="http://peer.test/rpc", description="test")
    pending = {"id": "task-1", "contextId": "context-1", "status": {
        "state": "TASK_STATE_INPUT_REQUIRED", "message": {"parts": [{"text": "Which room?"}]}}}
    response = copy.deepcopy(pending)
    calls = []
    monkeypatch.setattr(tools, "_fetch_card", lambda *_: card)
    monkeypatch.setattr(tools, "_resolve_peer", lambda _: {"url": "http://peer.test", "timeout": 1})

    def post(url, body, headers, timeout):
        calls.append((url, body, headers))
        return {"result": pending if body["method"] in ("GetTask", "tasks/get") else {"task": response}}

    monkeypatch.setattr(tools, "_http_post_json", post)
    return card, pending, response, calls


@pytest.mark.parametrize("state", ["TASK_STATE_INPUT_REQUIRED", "input-required"])
def test_pending_projection_and_identifiers(peer, state):
    _, _, response, calls = peer
    response["status"]["state"] = state
    response["artifacts"] = [{"parts": [{"text": "Stale answer"}]}]
    output = tools.a2a_call({"agent": "test", "message": "hello"})
    assert all(text in output for text in ("Which room?", "task-1", "context-1", "input-required", "task_id"))
    assert "Stale answer" not in output
    assert "taskId" not in calls[0][1]["params"]["message"]


@pytest.mark.parametrize("version", ["1.0", "0.3"])
def test_multiple_answers_are_data_not_text(peer, version):
    card, _, response, calls = peer
    card["supportedInterfaces"][0]["protocolVersion"] = version
    response["status"]["state"] = "completed"
    decision = {"decision_type": "approve", "ask_user_answers": [
        {"answer": ["room-a", "room-b"]}, {"answer": ["tomorrow"]}]}
    args = {"agent": "test", "message": "", "task_id": "task-1", "context_id": "context-1", "data": [decision]}
    saved = copy.deepcopy(args)
    output = tools.a2a_call(args)
    assert "completed" in output and args == saved
    assert len(calls) == 2
    wire = calls[1][1]["params"]["message"]
    assert wire["taskId"] == "task-1" and wire["contextId"] == "context-1"
    assert wire["parts"][0]["data"] == decision and "text" not in wire["parts"][0]
    assert calls[1][2]["A2A-Version"] == version
    assert calls[1][1]["method"] == ("SendMessage" if version == "1.0" else "message/send")
    assert wire["role"] == ("ROLE_USER" if version == "1.0" else "user")
    assert ("kind" in wire["parts"][0]) == (version == "0.3")


def test_does_not_generate_approval(peer):
    data = [{"ask_user_answers": [{"answer": ["room"]}]}]
    tools.a2a_call({"agent": "test", "message": "", "context_id": "context-1", "task_id": "task-1", "data": data})
    assert peer[3][-1][1]["params"]["message"]["parts"][0]["data"] == data[0]
    assert "decision_type" not in data[0]


@pytest.mark.parametrize("data", [True, {}, [], [True], [{}], [{1: "value"}], [{"x": float("nan")}],
    [{"x": float("inf")}], [{"x": "\ud800"}], [{"x": "x" * 65537}], [{"x": (1, 2)}],
    [{"x": [1] * 4097}], [{"x": "sk-abcdefghij1234567890ABCD"}]])
def test_invalid_data_never_sends(peer, data):
    output = tools.a2a_call({"agent": "test", "message": "", "task_id": "task-1", "context_id": "context-1", "data": data})
    assert output.startswith("Error:")
    assert not peer[3]


def test_nested_and_cyclic_data_rejected(peer):
    data = {"x": "leaf"}
    for _ in range(18):
        data = {"x": data}
    with pytest.raises(ValueError):
        tools._validate_data([data])
    data = {}
    data["x"] = data
    with pytest.raises(ValueError):
        tools._validate_data([data])


@pytest.mark.parametrize("change", [{"id": "other-task"}, {"contextId": "other-context"},
    {"status": {"state": "completed"}}, {"status": {"state": "canceled"}}, {}])
def test_preflight_rejects_unknown_mismatched_or_terminal(peer, change):
    _, pending, _, calls = peer
    if change:
        pending.update(change)
    else:
        pending.clear()
    output = tools.a2a_call({"agent": "test", "message": "answer", "task_id": "task-1", "context_id": "context-1"})
    assert output.startswith("Error:") and len(calls) == 1


def test_response_cannot_silently_create_new_task(peer):
    peer[2]["id"] = "new-task"
    output = tools.a2a_call({"agent": "test", "message": "answer", "task_id": "task-1", "context_id": "context-1"})
    assert "did not resume" in output


@pytest.mark.parametrize("state", ["completed", "canceled", "failed", "TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"])
def test_terminal_states_render_normally(peer, state):
    peer[2]["status"]["state"] = state
    output = tools.a2a_call({"agent": "test", "message": "hello"})
    assert tools._short_state(state) in output and "calling a2a_call again" not in output


def test_cross_origin_card_rejected(peer):
    peer[0]["supportedInterfaces"][0]["url"] = "http://different.test/"
    assert "origin" in tools.a2a_call({"agent": "test", "message": "hello"})
    assert not peer[3]


def test_unknown_protocol_rejected(peer):
    peer[0]["supportedInterfaces"][0]["protocolVersion"] = "2.0"
    assert "unsupported" in tools.a2a_call({"agent": "test", "message": "hello"})
    assert not peer[3]


def test_incompatible_payloads_rejected(peer):
    for args in ({"message": {"data": {}}}, {"message": "hello", "task_id": "task-1"},
                 {"message": "hello", "data": [{"answer": "room"}]}):
        assert tools.a2a_call({"agent": "test", **args}).startswith("Error:")
    assert not peer[3]
