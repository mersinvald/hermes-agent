"""Regressions for P2a/P2b on runtime baseline fadf8fd6 (synthetic auth only)."""
import http.client
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import logging
import os
import socket
import subprocess
import sys
import traceback
import threading
import urllib.error

import pytest

from plugins.platforms.a2a import protocol, tools


@pytest.fixture
def holdauth(monkeypatch):
    config = {"a2a_agents": {"alpha": {"url": "http://peer.test/?secret=query-secret",
        "auth": {"type": "bearer", "token": "SYNTHETIC_PRIVATE_AUTH"},
        "capabilities": ["holdauth"]}}}
    monkeypatch.setattr(tools, "_load_config", lambda: config)
    return config


@pytest.mark.parametrize("through_http", [False, True])
def test_holdauth_invalid_header_public_tools_no_socket_or_exception_leak(holdauth, monkeypatch, caplog, capsys, through_http):
    holdauth["a2a_agents"]["alpha"]["auth"]["token"] += "\n"
    attempted = []

    def forbidden(*args):
        attempted.append(args)
        raise AssertionError("socket forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(http.client.HTTPConnection, "debuglevel", 1)
    caplog.set_level(logging.DEBUG)
    headers = {"Authorization": "Bearer " + holdauth["a2a_agents"]["alpha"]["auth"]["token"]}
    if through_http:
        # Exercise the actual HTTP helper's defensive check through both tools.
        monkeypatch.setattr(tools, "_auth_header", lambda auth: headers)
    outputs = [tools.a2a_call({"agent": "alpha", "message": "hello"}),
               tools.a2a_orchestrate({"capability": "holdauth", "message": "hello"})]
    for helper, args in ((tools._http_get_json, ("http://peer.test",)),
                         (tools._http_post_json, ("http://peer.test", {}))):
        with pytest.raises(ValueError) as caught:
            helper(*args, headers, 1)
        outputs.append("".join(traceback.format_exception(caught.value)))
        assert caught.value.__context__ is None and caught.value.__cause__ is None
        logging.getLogger("holdauth").error("%s", outputs[-1])
    assert ("invalid HTTP header configuration" if through_http else "invalid bearer configuration") in outputs[0]
    assert not attempted
    persisted = [protocol.load_conversation(context) for context in protocol.list_conversations()]
    assert "SYNTHETIC_PRIVATE_AUTH" not in json.dumps(persisted)
    assert "SYNTHETIC_PRIVATE_AUTH" not in "\n".join(outputs) + caplog.text
    assert "query-secret" not in "\n".join(outputs) + caplog.text
    captured = capsys.readouterr()
    assert "SYNTHETIC_PRIVATE_AUTH" not in captured.out + captured.err


@pytest.mark.parametrize("error", [ValueError("Bearer SYNTHETIC_PRIVATE_AUTH\\n query-secret"),
    urllib.error.URLError("http://peer.test/?secret=query-secret"),
    urllib.error.HTTPError("http://peer.test/?secret=query-secret", 401,
                           "Bearer SYNTHETIC_PRIVATE_AUTH", {}, None), TimeoutError("SYNTHETIC_PRIVATE_AUTH")])
def test_holdauth_all_error_paths(holdauth, monkeypatch, caplog, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(tools, "_http_get_json", fail)
    monkeypatch.setattr(tools, "_http_post_json", fail)
    monkeypatch.setattr(protocol, "load_conversation", fail)
    outputs = [tools.a2a_discover({"url": "http://peer.test/?secret=query-secret"}),
               tools.a2a_call({"agent": "alpha", "message": "hello"}),
               tools.a2a_call({"agent": "alpha", "message": "answer", "task_id": "real-task-1", "context_id": "ctx"}),
               tools.a2a_history({"context_id": "ctx"}),
               tools.a2a_orchestrate({"capability": "holdauth", "message": "hello"})]
    for output in outputs:
        assert "Error:" in output
        assert "SYNTHETIC_PRIVATE_AUTH" not in output and "query-secret" not in output
    if isinstance(error, urllib.error.HTTPError):
        assert "HTTP 401" in outputs[0]
    assert "SYNTHETIC_PRIVATE_AUTH" not in caplog.text


def test_holdauth_state_query_error_uses_safe_category(holdauth, monkeypatch):
    monkeypatch.setattr(tools, "_fetch_card", lambda *args: {})
    monkeypatch.setattr(tools, "_http_post_json", lambda *args: {"error": {
        "message": "Bearer SYNTHETIC_PRIVATE_AUTH http://peer.test/?secret=query-secret"}})
    for task in ("", "real-task-1"):
        output = tools.a2a_call({"agent": "alpha", "message": "hello", "task_id": task, "context_id": "ctx"})
        assert output.startswith("Error:")
        assert "SYNTHETIC_PRIVATE_AUTH" not in output and "query-secret" not in output


@pytest.fixture
def holdauth_tasks(holdauth, monkeypatch):
    holdauth["a2a_agents"]["beta"] = {"url": "http://second.test"}
    monkeypatch.setattr(tools, "_fetch_card", lambda *args: {})
    calls = []
    state = {"value": "TASK_STATE_INPUT_REQUIRED"}

    def post(url, body, headers, timeout):
        calls.append(body)
        task = "real-task-1" if "peer.test" in url else "real-task-2"
        result = {"id": task, "contextId": "shared-context", "status": {
            "state": state["value"], "message": {"parts": [{"text":
                "Which room? Public normal text. Bearer SYNTHETIC_PRIVATE_AUTH"}]}}}
        return {"result": {"task": result}}

    monkeypatch.setattr(tools, "_http_post_json", post)
    return calls, state


def test_holdauth_history_real_task_peer_collision_and_fresh_process(holdauth_tasks):
    for name in ("alpha", "beta"):
        assert "input-required" in tools.a2a_call({"agent": name, "message": "hello"})
    records = protocol.load_conversation("shared-context")
    assert [r["task_id"] for r in records] == ["real-task-1", "real-task-2"]
    assert all(r["task_id"] != r["request_id"] for r in records)
    assert [r["peer_task"]["peer"] for r in records] == ["alpha", "beta"]
    source = "from plugins.platforms.a2a.tools import a2a_history; print(a2a_history({'context_id': 'shared-context'}))"
    output = subprocess.check_output([sys.executable, "-c", source], text=True, env=os.environ.copy())
    tasks = [json.loads(line[6:]) for line in output.splitlines() if line.startswith("Task: ")]
    assert tasks == [{"peer": "alpha", "task_id": "real-task-1", "context_id": "shared-context", "state": "input-required"},
                     {"peer": "beta", "task_id": "real-task-2", "context_id": "shared-context", "state": "input-required"}]
    assert "Which room? Public normal text." in output
    assert "SYNTHETIC_PRIVATE_AUTH" not in output
    assert "SYNTHETIC_PRIVATE_AUTH" not in json.dumps(records)


def test_holdauth_latest_state_and_peer_origin_binding(holdauth_tasks, holdauth):
    tools.a2a_call({"agent": "alpha", "message": "hello"})
    output = tools.a2a_call({"agent": "beta", "message": "answer", "context_id": "shared-context", "task_id": "real-task-1"})
    assert "different configured peer origin" in output
    original_url = holdauth["a2a_agents"]["alpha"]["url"]
    holdauth["a2a_agents"]["alpha"]["url"] = "http://changed.test"
    assert "different configured peer origin" in tools.a2a_call({"agent": "alpha", "message": "answer", "context_id": "shared-context", "task_id": "real-task-1"})
    holdauth["a2a_agents"]["alpha"]["url"] = original_url
    holdauth_tasks[1]["value"] = "TASK_STATE_COMPLETED"
    tools.a2a_call({"agent": "alpha", "message": "hello", "context_id": "shared-context"})
    output = tools.a2a_history({"context_id": "shared-context"})
    tasks = [json.loads(line[6:]) for line in output.splitlines() if line.startswith("Task: ")]
    assert tasks == [{"peer": "alpha", "task_id": "real-task-1", "context_id": "shared-context", "state": "completed"}]
    assert "recorded state input-required" in output


def test_holdauth_legacy_history_is_unknown_not_rpc_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "_conv_dir", lambda: tmp_path)
    (tmp_path / "legacy.jsonl").write_text(json.dumps({"role": "agent", "text": "old question", "task_id": "old-rpc-id"}) + "\n")
    output = tools.a2a_history({"context_id": "legacy"})
    assert "old question" in output and "peer/task/state unknown" in output
    assert "old-rpc-id" not in output and "Task: " not in output


@pytest.mark.parametrize("payload", [
    {"contextId": "ctx", "status": {"state": "input-required"}},
    {"id": "real-task-1", "status": {"state": "input-required"}},
    {"id": "real-task-1", "contextId": "../ctx", "status": {"state": "input-required"}},
])
def test_holdauth_missing_id_and_context_filename_collision(holdauth, monkeypatch, payload):
    monkeypatch.setattr(tools, "_fetch_card", lambda *args: {})
    monkeypatch.setattr(tools, "_http_post_json", lambda *args: {"result": {"task": payload}})
    output = tools.a2a_call({"agent": "alpha", "message": "hello"})
    if "id" not in payload or "contextId" not in payload:
        assert "lacks valid task/context" in output
    else:
        assert "real-task-1" in output
        assert not protocol.load_conversation("ctx")
        assert protocol.load_conversation("../ctx")[0]["peer_task"]["context_id"] == "../ctx"


def test_holdauth_supplied_context_cannot_be_silently_changed(holdauth_tasks):
    output = tools.a2a_call({"agent": "alpha", "message": "hello", "context_id": "requested-context"})
    assert "different context" in output
    assert not protocol.load_conversation("shared-context")


def test_holdauth_real_http_error_body_and_success_echo(holdauth, caplog):
    mode = {"value": "http"}

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            assert self.headers["Authorization"] == "Bearer SYNTHETIC_PRIVATE_AUTH"
            echo = "Public normal text. " + self.headers["Authorization"]
            self.send_response(403 if mode["value"] == "http" else 200)
            self.end_headers()
            response = ({"error": {"message": echo + " http://peer.test/?secret=query-secret"}}
                        if mode["value"] != "success" else {"result": {"task": {
                            "id": "real-task-1", "contextId": "ctx", "status": {"state": "completed"},
                            "artifacts": [{"parts": [{"text": echo}]}]}}})
            self.wfile.write(json.dumps(response).encode())

    server = HTTPServer(("127.0.0.1", 0), Endpoint)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    holdauth["a2a_agents"]["alpha"]["url"] = f"http://127.0.0.1:{server.server_port}"
    try:
        for value in ("http", "rpc", "success"):
            mode["value"] = value
            output = tools.a2a_call({"agent": "alpha", "message": "hello"})
            assert "SYNTHETIC_PRIVATE_AUTH" not in output and "query-secret" not in output
            if value == "success":
                assert "Public normal text." in output
            else:
                assert output.startswith("Error:")
        assert "SYNTHETIC_PRIVATE_AUTH" not in json.dumps(protocol.load_conversation("ctx")) + caplog.text
    finally:
        server.shutdown()
        worker.join()
        server.server_close()


@pytest.mark.parametrize("agent, recoverable", [("http://peer.test", True),
    ("http://peer.test/?secret=query-secret", False), ("http://user:password@peer.test", False)])
def test_holdauth_direct_peer_history_does_not_store_url_credentials(holdauth_tasks, monkeypatch, agent, recoverable):
    output = tools.a2a_call({"agent": agent, "message": "hello"})
    assert "real-task-1" in output
    records = protocol.load_conversation("shared-context")
    assert bool(records[0].get("peer_task")) is recoverable
    assert "query-secret" not in json.dumps(records) and "password" not in json.dumps(records)
