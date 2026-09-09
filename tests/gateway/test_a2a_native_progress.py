"""No-text model tool call traverses native executor, registry and Telegram status."""
import asyncio
import json
import threading
import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.gateway.test_telegram_status_update import _install_fake_telegram


@pytest.fixture(autouse=True)
def isolate_native_discovery(monkeypatch):
    # Native agent imports discover built-ins; don't change later card fixtures.
    from tools.registry import registry
    for name in ("_tools", "_toolset_checks", "_toolset_aliases"):
        monkeypatch.setattr(registry, name, dict(getattr(registry, name)))


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_version", [None, "0.3", "1.0"])
async def test_tool_only_model_emits_native_status_before_blocked_peer(monkeypatch, stream_version):
    _install_fake_telegram(monkeypatch)
    from gateway.config import PlatformConfig, Platform
    from gateway.platforms.base import SendResult
    from gateway.run import TurnRunner
    from plugins.platforms.telegram.adapter import TelegramAdapter
    from plugins.platforms.a2a import tools
    from tools.registry import registry
    from run_agent import AIAgent
    from tests.run_agent.test_run_agent import _mock_response, _mock_tool_call

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="synthetic"))
    adapter._bot = MagicMock()
    seen = asyncio.Event()
    stage_seen = asyncio.Event()

    async def send(chat_id, content, **kwargs):
        assert chat_id == "origin-dm"
        assert content == "Передаю запрос специалисту по дому."
        seen.set()
        return SendResult(success=True, message_id="progress-1")

    adapter.send = AsyncMock(side_effect=send)
    async def edit(chat_id, message_id, content, **kwargs):
        assert chat_id == "origin-dm" and message_id == "progress-1"
        assert content == "Специалист запрашивает актуальные данные Home Assistant"
        stage_seen.set()
        return SendResult(success=True, message_id="progress-1")
    adapter.edit_message = AsyncMock(side_effect=edit)
    ctx = SimpleNamespace(_status_adapter=adapter, _status_chat_id="origin-dm",
        _status_thread_metadata=None, _loop_for_step=asyncio.get_running_loop(),
        _run_still_current=lambda: True, _cleanup_progress=True, _cleanup_msg_ids=[],
        source=SimpleNamespace(platform=Platform.TELEGRAM))
    turn = TurnRunner(None, ctx)
    entered, release = threading.Event(), threading.Event()
    release_stage = threading.Event()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, body):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def do_GET(self):
            self.respond({"protocolVersion": stream_version or "0.3",
                          "capabilities": {"streaming": bool(stream_version)}})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            entered.set()
            if stream_version:
                assert body["method"] == ("message/stream" if stream_version == "0.3" else "SendStreamingMessage")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                assert release_stage.wait(10)
                def emit(state, parts):
                    event = {"kind": "status-update", "taskId": "t", "contextId": "c",
                             "status": {"state": state, "message": {"parts": parts}}}
                    if stream_version == "1.0":
                        event = {"statusUpdate": event}
                    self.wfile.write(b"data: " + json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": event}).encode() + b"\n\n")
                    self.wfile.flush()
                emit("working", [{"data": {"id": "call-1", "name": "GetLiveContext", "args": {"private": "never display"}},
                                  "metadata": {"adk_type": "function_call"}}])
                assert release.wait(10)
                emit("completed", [{"text": "synthetic result"}])
                return
            assert release.wait(10)
            self.respond({"result": {"id": "t", "contextId": "c", "status": {"state": "completed"},
                           "artifacts": [{"parts": [{"text": "synthetic result"}]}]}})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    Path(os.environ["HERMES_HOME"], "config.yaml").write_text(json.dumps({"a2a_agents": {"peer": {
        "url": f"http://127.0.0.1:{server.server_port}", "progress_notify": True, "streaming": bool(stream_version),
        "progress_messages": {"dispatch": "Передаю запрос специалисту по дому.",
                              "tool:GetLiveContext": "Специалист запрашивает актуальные данные Home Assistant"}}}}))
    schema = tools._SCHEMAS["a2a_call"]["function"]
    with patch("run_agent.get_tool_definitions", return_value=[tools._SCHEMAS["a2a_call"]]), \
         patch("run_agent.check_toolset_requirements", return_value={}), patch("run_agent.OpenAI"):
        agent = AIAgent(api_key="synthetic", base_url="http://127.0.0.1:9/v1", quiet_mode=True,
                        skip_context_files=True, skip_memory=True)
    registry.register(name="a2a_call", toolset="a2a", schema=schema,
                      handler=tools.a2a_call)
    agent.status_callback = turn._status_callback_sync
    agent.tool_progress_callback = None
    agent.client = MagicMock()
    agent._cached_system_prompt = "Synthetic test."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", tool_calls=[_mock_tool_call("a2a_call", json.dumps({"agent": "peer", "message": "synthetic"}))], finish_reason="tool_calls"),
        _mock_response(content="Finished.", finish_reason="stop")]
    running = asyncio.create_task(asyncio.to_thread(agent.run_conversation, "synthetic"))
    try:
        await asyncio.wait_for(seen.wait(), 8)
        assert await asyncio.to_thread(entered.wait, 2)
        assert not running.done()
        assert agent.client.chat.completions.create.call_count == 1
        if stream_version:
            await asyncio.sleep(2.05)  # Exercise the real per-turn edit throttle.
            release_stage.set()
            await asyncio.wait_for(stage_seen.wait(), 5)
            assert not running.done()
            assert agent.client.chat.completions.create.call_count == 1
    finally:
        release_stage.set()
        release.set()
        try:
            await asyncio.wait_for(running, 10)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
    assert agent.client.chat.completions.create.call_count == 2
    await turn.finish_activity()
    assert ctx._cleanup_msg_ids == ["progress-1"]
    adapter.send.assert_awaited_once()
    assert adapter.edit_message.await_count == (1 if stream_version else 0)
