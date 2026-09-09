"""Native transport progress: no model narration, external services or secrets."""
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from plugins.platforms.a2a import protocol, tools, streaming


@pytest.fixture
def setup_peer(monkeypatch):
    peer = {"url": "http://peer.test", "streaming": True, "progress_notify": True,
            "display_name": "Specialist", "progress_messages": {
                "dispatch": "Передаю запрос специалисту по дому.",
                "tool:GetLiveContext": "Специалист запрашивает актуальные данные Home Assistant"}}
    card = {"protocolVersion": "0.3", "capabilities": {"streaming": True}}
    monkeypatch.setattr(tools, "_resolve_peer", lambda _: peer)
    monkeypatch.setattr(tools, "_fetch_card", lambda *_: card)
    monkeypatch.setattr(tools, "_load_config", lambda: {})
    return peer, card


def status(state="working", **changes):
    return {"kind": "status-update", "taskId": "t", "contextId": "c",
            "status": {"state": state}, **changes}


def wire(monkeypatch, events, *, content_type="text/event-stream", event_ids=None, on_open=None):
    requests = []

    class Response(io.BytesIO):
        headers = Message()
    Response.headers["Content-Type"] = content_type

    class Opener:
        def open(self, request, **kwargs):
            body = json.loads(request.data)
            requests.append((body, dict(request.header_items())))
            if on_open:
                on_open()
            payload = b": keepalive\n\ndata:\n\n"
            for index, event in enumerate(events):
                envelope = {"jsonrpc": "2.0", "id": body["id"], "result": event}
                if event_ids:
                    payload += b"id: " + event_ids[index].encode() + b"\n"
                payload += b"data: " + json.dumps(envelope).encode() + b"\n\n"
            return Response(payload)

    monkeypatch.setattr(streaming.urllib.request, "build_opener", lambda *_: Opener())
    return requests


@pytest.mark.parametrize("version,method", [("0.3", "message/stream"), ("1.0", "SendStreamingMessage")])
def test_stream_versions_and_pending_priority(setup_peer, monkeypatch, version, method):
    setup_peer[1]["protocolVersion"] = version
    events = [{"kind": "artifact-update", "taskId": "t", "contextId": "c",
               "artifact": {"artifactId": "a", "parts": [{"text": "stale"}]}},
              status("input-required", status={"state": "input-required", "message": {
                  "parts": [{"data": {"ask_user": "Which room?"}}]}})]
    if version == "1.0":
        events = [{"artifactUpdate": events[0]}, {"statusUpdate": events[1]}]
    calls = wire(monkeypatch, events)
    notices = []
    result = tools.a2a_call({"agent": "peer", "message": "synthetic"},
                            status_callback=lambda *args: notices.append(args))
    assert "Which room?" in result and "stale" not in result
    assert "task_id" in result and "context_id" in result
    assert calls[0][0]["method"] == method
    assert calls[0][1]["A2a-version"] == version
    assert notices == [("tool_activity", "Передаю запрос специалисту по дому.")]


def test_function_metadata_only_not_thoughts_or_arguments(setup_peer, monkeypatch):
    parts = [{"text": "PRIVATE REASONING", "metadata": {"adk_thought": True}},
             {"data": {"id": "call-1", "name": "GetLiveContext", "args": {"private": "VALUE"}},
              "metadata": {"adk_type": "function_call"}},
             {"data": {"id": "call-1", "name": "GetLiveContext", "response": {"private": "VALUE"}},
              "metadata": {"adk_type": "function_response"}}]
    wire(monkeypatch, [status(status={"state": "working", "message": {"parts": parts}}),
                      status("completed", status={"state": "completed", "message": {"parts": [
                          {"text": "PRIVATE", "metadata": {"adk_thought": True}}, {"text": "final"}]}})])
    notices = []
    result = tools.a2a_call({"agent": "peer", "message": "synthetic"}, status_callback=lambda *a: notices.append(a))
    assert len(notices) == 3
    assert "Home Assistant" in notices[1][1]
    assert "received a tool result" in notices[2][1]
    assert "final" in result and "PRIVATE" not in result
    assert "VALUE" not in repr(notices) and "REASONING" not in repr(notices)


@pytest.mark.parametrize("bad", [status(contextId="other"), status(taskId="other"),
                               status(taskId=""), status("not-a-state")])
def test_mismatch_and_disconnect_never_replay(setup_peer, monkeypatch, bad):
    calls = wire(monkeypatch, [status(), bad])
    result = tools.a2a_call({"agent": "peer", "message": "synthetic"})
    assert "Do not replay" in result and len(calls) == 1
    records = protocol.load_conversation("c")
    assert records[-1]["peer_task"]["state"] == "unknown"


@pytest.mark.parametrize("kind", ["eof", "wrong-type", "oversize"])
def test_invalid_stream_never_unary(setup_peer, monkeypatch, kind):
    events = [status()] if kind != "oversize" else [status(extra="x" * streaming.MAX_EVENT_BYTES)]
    calls = wire(monkeypatch, events, content_type="application/json" if kind == "wrong-type" else "text/event-stream")
    monkeypatch.setattr(tools, "_http_post_json", lambda *_: pytest.fail("unary replay"))
    result = tools.a2a_call({"agent": "peer", "message": "synthetic"})
    assert result.startswith("Error:") and len(calls) == 1


def test_unsupported_capability_unary_still_notifies(setup_peer, monkeypatch):
    setup_peer[1]["capabilities"]["streaming"] = False
    notices = []

    def post(*args):
        assert notices  # Deterministic before the blocking unary call.
        return {"result": {"id": "t", "contextId": "c", "status": {"state": "completed"}}}

    monkeypatch.setattr(tools, "_http_post_json", post)
    result = tools.a2a_call({"agent": "peer", "message": "synthetic"}, status_callback=lambda *a: notices.append(a))
    assert "completed" in result and len(notices) == 1


def test_invalid_auth_never_announces_dispatch(setup_peer):
    setup_peer[0]["auth"] = {"type": "bearer", "token": "bad\nheader"}
    notices = []
    result = tools.a2a_call({"agent": "peer", "message": "synthetic"}, status_callback=lambda *a: notices.append(a))
    assert result.startswith("Error:") and not notices


@pytest.mark.parametrize("state", ["failed", "canceled", "rejected", "auth-required"])
def test_terminal_state_preserved(setup_peer, monkeypatch, state):
    wire(monkeypatch, [status(state)])
    result = tools.a2a_call({"agent": "peer", "message": "synthetic"})
    assert state in result


def test_unknown_function_uses_configured_generic_phrase(setup_peer, monkeypatch):
    setup_peer[0]["progress_messages"]["tool"] = "Локальный общий статус"
    wire(monkeypatch, [status(status={"state": "working", "message": {"parts": [
        {"metadata": {"kagent_type": "function_call"}, "data": {"id": "call-1", "name": "arbitrary_remote_name", "args": {}}}]}}),
        status("completed")])
    notices = []
    tools.a2a_call({"agent": "peer", "message": "synthetic"}, status_callback=lambda *a: notices.append(a))
    assert notices[-1][1] == "Локальный общий статус"
    assert "arbitrary_remote_name" not in repr(notices)


def test_request_auth_snapshot_redacts_final_and_progress(setup_peer, monkeypatch):
    setup_peer[0]["auth"] = {"type": "bearer", "token": "synthetic-secret"}
    setup_peer[0]["progress_messages"]["dispatch"] = "Sending synthetic-secret"
    calls = wire(monkeypatch, [status("completed", status={"state": "completed", "message": {
        "parts": [{"text": "SYNTHETIC-SECRET"}]}})])
    notices = []
    result = tools.a2a_call({"agent": "peer", "message": "synthetic"}, status_callback=lambda *a: notices.append(a))
    assert "synthetic-secret" not in (result + repr(notices)).lower()
    assert calls[0][1]["Authorization"] == "Bearer synthetic-secret"


@pytest.mark.parametrize("callback_mode", ["none", "raises", "disabled"])
def test_callback_is_optional(setup_peer, monkeypatch, callback_mode):
    wire(monkeypatch, [status("completed")])
    def callback(*args):
        raise RuntimeError("presentation failure")
    if callback_mode == "disabled":
        setup_peer[0]["progress_notify"] = False
        callback = lambda *_: pytest.fail("disabled notice")
    result = tools.a2a_call({"agent": "peer", "message": "synthetic"},
                           status_callback=None if callback_mode == "none" else callback)
    assert "completed" in result


def test_real_http_initial_notice_before_peer_finishes(setup_peer):
    entered, release, observed = threading.Event(), threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            entered.set()
            assert release.wait(5)
            data = {"jsonrpc": "2.0", "id": body["id"], "result": status("completed")}
            self.wfile.write(b"data: " + json.dumps(data).encode() + b"\n\n")
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    setup_peer[0]["url"] = f"http://127.0.0.1:{server.server_port}"
    try:
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(tools.a2a_call, {"agent": "peer", "message": "synthetic"},
                                 status_callback=lambda *_: observed.set())
            try:
                assert entered.wait(5) and observed.is_set()
                assert not future.done()
            finally:
                release.set()
            assert "completed" in future.result(5)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
