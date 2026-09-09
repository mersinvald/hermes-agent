"""Opt-in pinned Go executor -> deployed-image local Gateway -> native Hermes UX.

Run launch with KAGENT_SOURCE pointing at an isolated v0.10.0 checkout and
HERMES_PYTHON at the test interpreter. All requests and credentials are synthetic.
"""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import urllib.error
import urllib.request
import urllib.parse
import uuid

import pytest
import yaml

IMAGE = "dcr.home.lan.mkl.dev/library/agentgateway@sha256:12a133d46327ee3b4fd12ad913ad30f575007eafce427641719f6b702b8d528e"
PEER_PATH = "/api/a2a/kagent-system/groundskeeper-108157884/"


@pytest.mark.skipif(not os.environ.get("KAGENT_SOURCE") or bool(os.environ.get("NATIVE_CONTROLLER_URL")), reason="explicit isolated native Go fixture required")
def test_launch_native_fixture():
    source = Path(os.environ["KAGENT_SOURCE"])
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    assert revision == "ae008d1e19fbb195ba3c8205fe345d17309a9158"
    target = source / "go/core/internal/a2a/hermes_progress_fixture_test.go"
    assert not target.exists(), "do not overwrite another worker fixture"
    fixture = Path(__file__).parent / "fixtures/kagent_progress_test.go"
    shutil.copyfile(fixture, target)
    try:
        env = {**os.environ, "GOPROXY": "off", "GOFLAGS": "-mod=readonly",
               "HERMES_ROOT": str(Path(__file__).resolve().parents[2])}
        result = subprocess.run(["go", "test", "./core/internal/a2a", "-run", "^TestHermesNativeProgress$", "-count=1", "-v", "-timeout", "240s"],
                                cwd=source / "go", env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=270)
        print(result.stdout)
        assert result.returncode == 0
    finally:
        target.unlink()


@pytest.fixture
def gateway(tmp_path):
    controller = os.environ["NATIVE_CONTROLLER_URL"]
    backend = "host.docker.internal:" + controller.rsplit(":", 1)[1]
    key = "fixture-progress-key"
    auth = {"mode": "strict", "keys": [{"keyHash": "sha256:" + hashlib.sha256(key.encode()).hexdigest(),
        "metadata": {"name": "hermes-mike", "role": "caller"}}],
        "location": {"header": {"name": "authorization", "prefix": "Bearer "}}}
    guard = 'apiKey.name == "hermes-mike" && apiKey.role == "caller" && request.pathAndQuery == request.path'
    routes = []
    for method, path in [("GET", PEER_PATH + ".well-known/agent-card.json"), ("POST", PEER_PATH)]:
        policies = {"apiKey": auth, "authorization": {"rules": [{"require": guard}]},
                    "transformations": {"request": {"remove": ["authorization", "x-user", "x-agent-name", "x-share-token"], "set": {"x-user-id": '"fixture-owner"'}}}}
        if method == "POST":
            policies["buffer"] = {"request": {"maxBytes": 131072, "failureMode": "failClosed"}}
            policies["authorization"]["rules"].append({"require": 'json(request.body).jsonrpc == "2.0" && json(request.body).method in ["SendMessage", "message/send", "GetTask", "tasks/get", "SendStreamingMessage", "message/stream", "SubscribeToTask", "tasks/resubscribe"]'})
            policies["authorization"]["rules"].append({"require": '!(json(request.body).method in ["SubscribeToTask", "tasks/resubscribe"]) || (type(json(request.body).params.id) == string && size(json(request.body).params.id) > 0 && size(json(request.body).params.id) <= 1024)'})
        routes.append({"name": "fixture-" + method.lower(), "matches": [{"path": {"exact": path}, "method": method}],
                       "backends": [{"host": backend}], "policies": policies})
    config = tmp_path / "gateway.yaml"
    config.write_text(yaml.safe_dump({"binds": [{"port": 8888, "listeners": [{"protocol": "HTTP", "routes": routes}]}]}))
    name = "hermes-progress-test-" + uuid.uuid4().hex[:10]
    log = (tmp_path / "gateway.log").open("w+")
    process = subprocess.Popen(["docker", "run", "--pull=never", "--rm", "--name", name, "--memory", "256m", "-p", "127.0.0.1::8888",
        "-v", f"{config}:/gateway.yaml:ro", IMAGE, "-f", "/gateway.yaml"], stdout=log, stderr=subprocess.STDOUT)
    try:
        endpoint = None
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if process.poll() is not None:
                log.seek(0)
                pytest.fail(log.read())
            inspected = subprocess.run(["docker", "inspect", name, "--format", '{{(index (index .NetworkSettings.Ports "8888/tcp") 0).HostPort}}'], capture_output=True, text=True)
            if inspected.returncode == 0 and inspected.stdout.strip().isdigit():
                endpoint = "http://127.0.0.1:" + inspected.stdout.strip()
                try:
                    with urllib.request.urlopen(urllib.request.Request(endpoint + PEER_PATH + ".well-known/agent-card.json", headers={"Authorization": "Bearer " + key}), timeout=1) as response:
                        assert json.load(response)["capabilities"]["streaming"]
                    break
                except (OSError, ValueError):
                    pass
            time.sleep(0.1)
        else:
            log.seek(0)
            pytest.fail("Gateway did not become ready: " + log.read())
        urllib.request.urlopen(os.environ["NATIVE_CONTROL_URL"] + "/url?" + urllib.parse.urlencode({"url": endpoint + PEER_PATH}), timeout=2).close()
        yield endpoint, key
    finally:
        subprocess.run(["docker", "stop", "-t", "1", name], capture_output=True, timeout=10)
        process.wait(timeout=10)
        log.close()


@pytest.mark.skipif(not os.environ.get("NATIVE_CONTROLLER_URL"), reason="launched by pinned Go fixture")
@pytest.mark.asyncio
async def test_native_contract(gateway, monkeypatch):
    from tests.gateway.test_telegram_status_update import _install_fake_telegram
    _install_fake_telegram(monkeypatch)
    from gateway.config import Platform, PlatformConfig
    from gateway.run import TurnRunner
    from plugins.platforms.telegram.adapter import TelegramAdapter
    from plugins.platforms.a2a import tools
    from run_agent import AIAgent
    from tests.run_agent.test_run_agent import _mock_response, _mock_tool_call
    endpoint, key = gateway
    control = os.environ["NATIVE_CONTROL_URL"]
    events = []
    seen = {stage: asyncio.Event() for stage in ("start", "tool", "result")}
    labels = {"start": "Передаю запрос специалисту.", "tool": "Специалист запрашивает время.", "result": "Специалист получил результат инструмента."}
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="synthetic-unused"))
    async def capture(operation, **kwargs):
        text = kwargs.get("text", "")
        mid = kwargs.get("message_id", 102 if text == "Final" else 101)
        events.append((operation, mid, text, time.monotonic()))
        for stage, label in labels.items():
            if text == label:
                seen[stage].set()
        return SimpleNamespace(message_id=mid)
    async def send(**kw):
        return await capture("send", **kw)
    async def edit(**kw):
        return await capture("edit", **kw)
    async def delete(**kw):
        return await capture("delete", **kw)
    adapter._bot = SimpleNamespace(send_message=AsyncMock(side_effect=send),
        edit_message_text=AsyncMock(side_effect=edit),
        delete_message=AsyncMock(side_effect=delete), send_chat_action=AsyncMock())
    ctx = SimpleNamespace(_status_adapter=adapter, _status_chat_id="424242", _status_thread_metadata=None,
        _loop_for_step=asyncio.get_running_loop(), _run_still_current=lambda: True, _cleanup_progress=True,
        _cleanup_msg_ids=[], session_key="fixture", run_generation=1, source=SimpleNamespace(platform=Platform.TELEGRAM))
    turn = TurnRunner(None, ctx)
    Path(os.environ["HERMES_HOME"], "config.yaml").write_text(yaml.safe_dump({"a2a_agents": {"peer": {
        "url": endpoint + PEER_PATH, "auth": {"type": "bearer", "token": key}, "streaming": True, "progress_notify": True,
        "progress_messages": {"dispatch": labels["start"], "tool:GetDateTime": labels["tool"], "tool_result": labels["result"]}}}}))
    # The fixed path and authenticated native body guard stay in place.
    for path, token, method in [(PEER_PATH, "invalid", "message/stream"), (PEER_PATH + "?x=1", key, "message/stream"), ("/other/", key, "message/stream"), (PEER_PATH, key, "CancelTask"), (PEER_PATH, key, "SubscribeToTask")]:
        req = urllib.request.Request(endpoint + path, data=json.dumps({"jsonrpc": "2.0", "id": "denied", "method": method, "params": {}}).encode(), headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(req, timeout=2)
        assert error.value.code in (401, 403, 404)
    with patch("run_agent.get_tool_definitions", return_value=[tools._SCHEMAS["a2a_call"]]), patch("run_agent.check_toolset_requirements", return_value={}), patch("run_agent.OpenAI"):
        agent = AIAgent(api_key="synthetic", base_url="http://127.0.0.1:9/v1", quiet_mode=True, skip_memory=True, skip_context_files=True)
    agent.client = MagicMock()
    agent.status_callback = turn._status_callback_sync
    agent.tool_progress_callback = None
    agent._cached_system_prompt = "Synthetic test only."
    agent._use_prompt_caching = False
    agent.compression_enabled = agent.save_trajectories = False
    agent.tool_delay = 0
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", tool_calls=[_mock_tool_call("a2a_call", json.dumps({"agent": "peer", "message": "synthetic"}))], finish_reason="tool_calls"),
        _mock_response(content="Final", finish_reason="stop")]
    running = asyncio.create_task(asyncio.to_thread(agent.run_conversation, "synthetic"))
    following = None
    try:
        await asyncio.wait_for(seen["start"].wait(), 8)
        await asyncio.wait_for(seen["tool"].wait(), 8)
        assert not running.done() and agent.client.chat.completions.create.call_count == 1
        from plugins.platforms.a2a import protocol
        records = [record for context in protocol.list_conversations() for record in protocol.load_conversation(context)
                   if record.get("peer_task")]
        task = records[-1]["peer_task"]
        subscribed = asyncio.Event()
        loop = asyncio.get_running_loop()
        following = asyncio.create_task(asyncio.to_thread(tools.a2a_call,
            {"agent": "peer", "action": "follow", "task_id": task["task_id"], "context_id": task["context_id"]},
            status_callback=lambda *_: loop.call_soon_threadsafe(subscribed.set)))
        observation = asyncio.create_task(subscribed.wait())
        await asyncio.wait([following, observation], timeout=4, return_when=asyncio.FIRST_COMPLETED)
        if following.done():
            assert not following.result().startswith("Error:"), following.result()
        assert subscribed.is_set()
        observation.cancel()
        urllib.request.urlopen(control + "/tool", timeout=2).close()
        await asyncio.wait_for(seen["result"].wait(), 8)
        assert not running.done() and agent.client.chat.completions.create.call_count == 1
    finally:
        urllib.request.urlopen(control + "/tool", timeout=2).close()
        urllib.request.urlopen(control + "/final", timeout=2).close()
        result = await asyncio.wait_for(running, 10)
        if following is not None:
            follow_result = await asyncio.wait_for(following, 10)
        await turn.finish_activity()
    assert agent.client.chat.completions.create.call_count == 2
    assert "Synthetic final answer" in json.dumps(result)
    assert "Synthetic final answer" in follow_result, follow_result
    await adapter.send("424242", "Final", metadata={"notify": True})
    await adapter.pop_post_delivery_callback("fixture", generation=1)()
    assert [e[0] for e in events] == ["send", "edit", "edit", "send", "delete"]
    assert events[-1][1] == 101 and events[-2][1] == 102
    assert all(b[3] - a[3] >= 1.99 for a, b in zip(events[:2], events[1:3]))
    print("NATIVE_PROOF=" + json.dumps({"version": os.environ["NATIVE_VERSION"], "events": events, "parent_calls": 2, "stream_false": True}))
