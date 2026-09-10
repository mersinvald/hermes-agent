"""Actual native agent loop + OpenAI SDK + local HTTP, no provider calls."""
import asyncio
import base64
import io

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
])
async def test_real_sdk_images_history_and_no_text_retry(monkeypatch, tmp_path, status, error):
    requests, received = [], asyncio.Queue()
    rejection = False
    async def completion(request):
        payload = await request.json()
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
            if status == 0:
                server.models.config = None
            response = await client.post("/v1/pwa/commands", json=command(root, "continue again", id="third"))
            assert response.status == 200, await response.text()
            if status:
                await asyncio.wait_for(received.get(), 15)
            await closed(db, root)
            assert len(requests) == (3 if status else 2)
            with db._read_ctx() as conn:
                assert conn.execute("SELECT observed_state FROM native_executions WHERE conversation_id=? ORDER BY created_order DESC LIMIT 1", (root,)).fetchone()[0] == "failed"
    finally:
        await endpoint.cleanup()
