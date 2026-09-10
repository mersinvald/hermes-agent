"""Actual native agent loop + OpenAI SDK + local HTTP, no provider calls."""
import asyncio
import base64
import io
import json

import pytest
from aiohttp import web
from PIL import Image

import run_agent
from agent.conversation_loop import run_conversation as real_conversation_loop
from tests.gateway.test_native_commands import command
from tests.gateway.test_pwa_http import service
from tests.gateway.test_pwa_image_commands import initialize, models, upload

RealAgent = run_agent.AIAgent


async def closed(db, root):
    async def wait():
        while db.native_execution(root) is not None or any(
            r["phase"] in {"queued", "assigned"} for r in db.native_command_rows(root)
        ):
            await asyncio.sleep(0.01)
    await asyncio.wait_for(wait(), 15)


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "error"), [
    (400, "Only 'text' content type is supported."),
    (413, "Request too large"),
    (400, "Invalid PNG image."),
    (400, "image exceeds 5 MB maximum"),
    (0, "catalog removed"),
    (-1, "read byte limit"),
    (-2, "retained pixel limit"),
    (-3, "serialized request limit"),
    (-4, "history plus current pixel limit"),
    (-5, "additional request image limit"),
    (-6, "additional request text limit"),
    (-7, "late compression tip read limit"),
])
async def test_real_sdk_images_history_and_no_text_retry(monkeypatch, tmp_path, status, error):
    requests, received = [], asyncio.Queue()
    rejection = False
    async def completion(request):
        raw = await request.read()
        assert len(raw) < 48 * 1024 * 1024
        payload = json.loads(raw)
        requests.append(payload)
        received.put_nowait(payload)
        if rejection:
            return web.json_response({"error": {"message": error, "type": "invalid_request_error"}}, status=status)
        return web.json_response({"id": "synthetic", "object": "chat.completion", "created": 1,
            "model": "upstream/daily", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Synthetic image reply."},
                                                     "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})
    app = web.Application()
    app.router.add_post("/v1/chat/completions", completion)
    endpoint = web.AppRunner(app)
    await endpoint.setup()
    site = web.TCPSite(endpoint, "127.0.0.1", 0)
    await site.start()
    origin = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/v1"
    try:
        async with service(monkeypatch, tmp_path, models=models(), resolver=lambda _: {
            "provider": "custom", "base_url": origin, "api_key": "synthetic-local-only",
        }) as (server, client, native):
            class Agent(RealAgent):
                def __init__(self, **kwargs):
                    kwargs.update(skip_memory=True, skip_context_files=True, quiet_mode=True)
                    super().__init__(**kwargs)
                    self._disable_streaming = True
            monkeypatch.setattr(run_agent, "AIAgent", Agent)
            monkeypatch.setattr("agent.conversation_loop.run_conversation", real_conversation_loop)
            monkeypatch.setattr(run_agent, "get_tool_definitions", lambda **_: [])
            monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda **_: {})
            monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *a, **k: 200000)
            root, db = native[4].session_id, native[2]
            # The older synthetic-loop fixture intentionally omits this handle.
            # A real AIAgent must receive the actual shared native SessionDB.
            server.runner._session_db = db
            await initialize(client, root)
            image = await upload(client, root)
            first = command(root, "exact caption", expected_model_version=1)
            first["payload"]["image_ids"] = [image["image_id"]]
            response = await client.post("/v1/pwa/commands", json=first)
            assert response.status == 200, await response.text()
            initial = await asyncio.wait_for(received.get(), 15)
            await closed(db, root)
            assert len(requests) == 1
            parts = [p for m in initial["messages"] if isinstance(m.get("content"), list)
                     for p in m["content"] if p.get("type") == "image_url"]
            assert len(parts) == 1
            data = base64.b64decode(parts[0]["image_url"]["url"].split(",", 1)[1])
            with Image.open(io.BytesIO(data)) as pixels:
                assert not pixels.getexif() and pixels.size == (24, 48)
            assert b"SYNTHETIC PRIVATE" not in data
            # A text-only continuation must carry retained managed image pixels.
            response = await client.post("/v1/pwa/commands", json=command(root, "continue", id="second"))
            assert response.status == 200, await response.text()
            continuation = await asyncio.wait_for(received.get(), 15)
            await closed(db, root)
            assert any(p.get("image_url") == parts[0]["image_url"] for m in continuation["messages"]
                       if isinstance(m.get("content"), list) for p in m["content"])
            rejection = True
            before = db.get_messages_as_conversation(root)
            with db._read_ctx() as conn:
                retained = [tuple(row) for row in conn.execute(
                    "SELECT id,content,api_content,display_metadata FROM messages WHERE session_id=? ORDER BY id", (root,))]
            decoded = []
            if status == 0:
                server.models.config = None
            elif status == -1:
                # Reopen image storage to prove activation is durable, then
                # prevent even the first context decoder from being entered.
                server.images.close()
                monkeypatch.setattr("gateway.pwa_image_policy.MAX_CONTEXT_SOURCE_BYTES", 1)
                original_decode = db._rows_to_conversation
                def decode(*args, **kwargs):
                    decoded.append(True)
                    return original_decode(*args, **kwargs)
                monkeypatch.setattr(db, "_rows_to_conversation", decode)
            elif status == -2:
                monkeypatch.setattr("gateway.pwa_image_policy.MAX_REQUEST_PIXEL_BYTES", len(data) - 1)
            elif status == -3:
                monkeypatch.setattr("gateway.pwa_image_policy.MAX_REQUEST_BODY_BYTES", 128)
            elif status == -4:
                monkeypatch.setattr("gateway.pwa_image_policy.MAX_REQUEST_PIXEL_BYTES", len(data) + 1)
            elif status in {-5, -6}:
                original_build = RealAgent._build_api_kwargs
                def build(agent, *args, **kwargs):
                    result = original_build(agent, *args, **kwargs)
                    additional = parts if status == -5 else [{"type": "text", "text": "extra" * 10000}]
                    result["messages"] = [*result["messages"], {"role": "user", "content": additional}]
                    return result
                monkeypatch.setattr(RealAgent, "_build_api_kwargs", build)
                if status == -5:
                    monkeypatch.setattr("gateway.pwa_image_policy.MAX_REQUEST_PIXEL_BYTES", len(data) + 1)
                else:
                    monkeypatch.setattr("gateway.pwa_image_policy.MAX_REQUEST_BODY_BYTES", 32768)
            elif status == -7:
                monkeypatch.setattr("gateway.pwa_image_policy.MAX_CONTEXT_SOURCE_BYTES", 65536)
                def rotated_tip(session_id):
                    assert session_id == root
                    db.publish_compression_child(parent_session_id=root, child_session_id="bounded-tip",
                        source=native[5].platform.value,
                        messages=[*before, {"role": "assistant", "content": "retained" * 20000}],
                        require_compression_lease=False)
                    decoded.append("rotated")
                    return "bounded-tip"
                monkeypatch.setattr(db, "resolve_resume_session_id", rotated_tip)
            third = command(root, "continue again", id="third")
            if status == -4:
                third["payload"]["image_ids"] = [image["image_id"]]
                third["expected_model_version"] = 1
            response = await client.post("/v1/pwa/commands", json=third)
            assert response.status == 200, await response.text()
            if status > 0:
                await asyncio.wait_for(received.get(), 15)
            await closed(db, root)
            assert len(requests) == (3 if status > 0 else 2)
            if status == -1:
                assert decoded == []
                monkeypatch.setattr(db, "_rows_to_conversation", original_decode)
            if status < 0:
                with db._read_ctx() as conn:
                    after = [tuple(row) for row in conn.execute(
                        "SELECT id,content,api_content,display_metadata FROM messages WHERE session_id=? ORDER BY id", (root,))]
                assert after[:len(retained)] == retained
            if status == -7:
                assert decoded == ["rotated"]
            with db._read_ctx() as conn:
                assert conn.execute("SELECT observed_state FROM native_executions WHERE conversation_id=? ORDER BY created_order DESC LIMIT 1", (root,)).fetchone()[0] == "failed"
    finally:
        await endpoint.cleanup()
