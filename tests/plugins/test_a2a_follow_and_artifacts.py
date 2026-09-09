import pytest

from plugins.platforms.a2a import protocol, tools
from tests.plugins.test_a2a_streaming_progress import setup_peer, wire, status


@pytest.mark.parametrize("data", [None, "malicious", [], {}, {"name": "clock", "id": "c"},
    {"name": [], "id": "c", "response": {}}, {"name": "clock", "id": [], "response": {}},
    {"name": "clock", "id": "c", "response": "secret"}, {"name": "clock", "id": "c", "response": []}])
def test_malformed_response_is_not_progress(setup_peer, monkeypatch, data):
    notices = []
    wire(monkeypatch, [status(status={"state": "working", "message": {"parts": [
        {"metadata": {"adk_type": "function_response"}, "data": data}]}}), status("completed")])
    tools.a2a_call({"agent": "peer", "message": "synthetic"}, status_callback=lambda *a: notices.append(a))
    assert len(notices) == 1


def test_all_artifacts_and_replace_append(setup_peer, monkeypatch):
    def artifact(aid, text, **extra):
        return {"kind": "artifact-update", "contextId": "c", "taskId": "t",
                "artifact": {"artifactId": aid, "parts": [{"text": text}]}, **extra}
    wire(monkeypatch, [artifact("a", "old"), artifact("a", "first"),
        artifact("a", " answer", append=True, lastChunk=True), artifact("b", "second answer"),
        artifact("empty", ""), status("completed")])
    result = tools.a2a_call({"agent": "peer", "message": "synthetic"})
    assert "first answer\n\nsecond answer" in result and "old" not in result


def test_unary_artifacts_exclude_thoughts_and_tool_data():
    result = tools._reply_text_from_result({"artifacts": [{"parts": [{"text": "first"},
        {"text": "hidden", "metadata": {"adk_thought": True}}, {"data": {"response": "secret"}}]},
        {"parts": [{"text": "second"}]}]})
    assert result == "first\n\nsecond"


def test_pending_question_keeps_data_without_thought_text():
    result = tools._reply_text_from_result({"status": {"state": "input-required", "message": {
        "parts": [{"text": "private thought", "metadata": {"adk_thought": True}},
                  {"data": {"question": "Which room?"}}]}}})
    assert "Which room?" in result and "private thought" not in result


@pytest.fixture
def persisted():
    protocol.persist_message("c", "agent", "", "t", peer_task={"version": 1, "peer": "peer",
        "origin": tools._peer_origin("http://peer.test"), "task_id": "t", "context_id": "c", "state": "unknown"})


@pytest.mark.parametrize("version,method", [("0.3", "tasks/resubscribe"), ("1.0", "SubscribeToTask")])
def test_follow_only_reads_existing_task(setup_peer, persisted, monkeypatch, version, method):
    setup_peer[1]["protocolVersion"] = version
    calls = wire(monkeypatch, [status("completed")])
    lookups = []
    def get(url, body, headers, timeout):
        lookups.append(body)
        assert body["method"] == ("tasks/get" if version == "0.3" else "GetTask")
        return {"jsonrpc": "2.0", "id": body["id"], "result": {"id": "t", "contextId": "c",
                "status": {"state": "working"}, "artifacts": [{"artifactId": "prior", "parts": [{"text": "retained"}]}]}}
    monkeypatch.setattr(tools, "_http_post_json", get)
    result = tools.a2a_call({"action": "follow", "agent": "peer", "task_id": "t", "context_id": "c"})
    assert "completed" in result and "retained" in result
    assert len(lookups) == len(calls) == 1
    assert calls[0][0]["method"] == method and "message" not in calls[0][0]["params"]


def test_follow_requires_history_binding(setup_peer, monkeypatch):
    monkeypatch.setattr(tools, "_http_post_json", lambda *_: pytest.fail("unbound read"))
    result = tools.a2a_call({"action": "follow", "agent": "peer", "task_id": "t", "context_id": "c"})
    assert "persisted task" in result


def test_follow_terminal_snapshot_never_subscribes(setup_peer, persisted, monkeypatch):
    def get(url, body, headers, timeout):
        assert body["method"] == "tasks/get"
        return {"jsonrpc": "2.0", "id": body["id"], "result": {"id": "t", "contextId": "c",
            "status": {"state": "input-required", "message": {"parts": [{"data": {"question": "Which?"}}]}},
            "artifacts": [{"parts": [{"text": "old"}]}]}}
    monkeypatch.setattr(tools, "_http_post_json", get)
    result = tools.a2a_call({"action": "follow", "agent": "peer", "task_id": "t", "context_id": "c"})
    assert "Which?" in result and "old" not in result


def test_discovery_failure_never_downgrades_to_unary(setup_peer, monkeypatch):
    def fail(*args):
        raise TimeoutError()
    monkeypatch.setattr(tools, "_fetch_card", fail)
    monkeypatch.setattr(tools, "_http_post_json", lambda *_: pytest.fail("unary downgrade"))
    result = tools.a2a_call({"agent": "peer", "message": "synthetic"})
    assert "no message was sent" in result


def test_event_id_persisted_after_interruption(setup_peer, monkeypatch):
    wire(monkeypatch, [status()], event_ids=["event-1"])
    tools.a2a_call({"agent": "peer", "message": "synthetic"})
    last = protocol.load_conversation("c")[-1]["peer_task"]
    assert last["state"] == "unknown" and last["last_event_id"] == "event-1"


def test_duplicate_artifact_chunk_and_rotated_auth_snapshot(setup_peer, monkeypatch):
    setup_peer[0]["auth"] = {"type": "bearer", "token": "credential-fragment"}
    initial = {"kind": "artifact-update", "taskId": "t", "contextId": "c",
               "artifact": {"artifactId": "a", "parts": [{"text": "credential-"}]}}
    appended = {"kind": "artifact-update", "taskId": "t", "contextId": "c", "append": True,
                "artifact": {"artifactId": "a", "parts": [{"text": "fragment"}]}}
    requests = wire(monkeypatch, [initial, appended, appended, status("completed")], event_ids=["one", "two", "two", "three"],
        on_open=lambda: setup_peer[0]["auth"].update(token="rotated-credential"))
    result = tools.a2a_call({"agent": "peer", "message": "synthetic"})
    assert "credential-fragment" not in result and "[REDACTED]fragment" not in result
    assert result.endswith("[REDACTED]")
    assert requests[0][1]["Authorization"] == "Bearer credential-fragment"
    assert "credential-fragment" not in str(protocol.load_conversation("c"))


@pytest.mark.parametrize("changes", [{"contextId": "other"}, {"id": "other"}])
def test_follow_lookup_mismatch_does_not_subscribe(setup_peer, persisted, monkeypatch, changes):
    def get(url, body, headers, timeout):
        return {"jsonrpc": "2.0", "id": body["id"], "result": {
            "id": "t", "contextId": "c", "status": {"state": "working"}, **changes}}
    monkeypatch.setattr(tools, "_http_post_json", get)
    result = tools.a2a_call({"action": "follow", "agent": "peer", "task_id": "t", "context_id": "c"})
    assert "did not match" in result
