"""Private image routes on real native HTTP and the retained profile filesystem."""
import asyncio
from dataclasses import replace

import pytest

from gateway.pwa_http import NativePwaHttp
from tests.gateway.test_pwa_http import OWNER, headers, service
from tests.gateway.test_pwa_images import picture


def upload_headers(upload_id="one", mime="image/png"):
    return {"Content-Type": mime, "X-Hermes-PWA-Upload-Id": upload_id}


@pytest.mark.asyncio
async def test_http_original_preview_retry_restart_foreign_and_text_only(monkeypatch, tmp_path):
    async with service(monkeypatch, tmp_path) as (server, client, native):
        root = native[4].session_id
        path = f"/v1/pwa/conversations/{root}/images"
        data = picture(exif=True)
        policy = await (await client.get("/v1/pwa/images/policy")).json()
        assert policy["max_images_per_message"] == 10 and policy["max_original_bytes"] == 20*1024*1024
        first = await client.post(path, data=data, headers=upload_headers())
        assert first.status == 201, await first.text()
        descriptor = await first.json()
        again = await client.post(path, data=data, headers=upload_headers())
        assert again.status == 200 and await again.json() == descriptor
        assert (await client.post(path, data=data, headers=upload_headers(mime="image/jpeg"))).status == 409
        image_path = path + "/" + descriptor["image_id"]
        original = await client.get(image_path + "/original")
        assert original.status == 200 and await original.read() == data
        assert original.headers["Cache-Control"] == "private, no-store"
        preview = await client.get(image_path + "/preview")
        assert preview.status == 200 and b"SYNTHETIC PRIVATE" not in await preview.read()
        foreign = await client.get(image_path + "/original", headers=headers(replace(OWNER, subject="foreign")))
        assert foreign.status in {403, 404} and data not in await foreign.read()
        assert (await client.get(image_path.replace(root, "foreign-root") + "/original")).status == 404
        assert (await client.post("/v1/pwa/commands", json={"schema_version": "1.0", "image_ids": [descriptor["image_id"]]})).status == 400
        server.images.close()
        # Store recreation is the native listener restart's retained-image seam.
        assert (await (await client.get(image_path + "/original")).read()) == data
        assert (await client.post(path, data=data, headers=upload_headers())).status == 200
        await server.close()
        del server.runner.conversation_ingress
        replacement = NativePwaHttp(server.runner)
        try:
            await replacement.start()
            import aiohttp
            async with aiohttp.ClientSession(headers=headers()) as new_client:
                response = await new_client.get(f"http://127.0.0.1:{replacement.port}" + image_path + "/original")
                assert response.status == 200 and await response.read() == data
        finally:
            await replacement.close()
            replacement.ingress.stop_command_recovery()


@pytest.mark.asyncio
async def test_validation_chunk_caps_and_concurrent_worker_bound(monkeypatch, tmp_path):
    async with service(monkeypatch, tmp_path) as (server, client, native):
        path = f"/v1/pwa/conversations/{native[4].session_id}/images"
        for data, mime in [(b"%PDF-1.4", "application/pdf"), (picture(), "image/jpeg"), (b"broken", "image/png"), (b"heic", "image/heic")]:
            assert (await client.post(path, data=data, headers=upload_headers(mime=mime))).status == 400
        server.config = replace(server.config, images=replace(server.config.images, max_original_bytes=100))
        async def too_big():
            yield b"x" * 60
            yield b"y" * 60
        assert (await client.post(path, data=too_big(), headers=upload_headers())).status == 413
        started, release = asyncio.Event(), asyncio.Event()
        async def slow():
            yield b"x"
            started.set()
            await release.wait()
            yield b"y"
        pending = asyncio.create_task(client.post(path, data=slow(), headers=upload_headers()))
        await started.wait()
        for _ in range(100):
            if server.images.workers:
                break
            await asyncio.sleep(0.01)
        overflow = await client.post(path, data=picture(), headers=upload_headers("second"))
        assert overflow.status == 429
        release.set()
        assert (await pending).status == 400
        assert server.images.workers == 0


@pytest.mark.asyncio
async def test_native_grant_revocation_before_publication_preserves_old_images(monkeypatch, tmp_path):
    async with service(monkeypatch, tmp_path) as (server, client, native):
        path = f"/v1/pwa/conversations/{native[4].session_id}/images"
        data = picture()
        first = await client.post(path, data=data, headers=upload_headers())
        assert first.status == 201
        descriptor = await first.json()
        original_prepare = server.images.store.prepare
        def revoke(*args):
            result = original_prepare(*args)
            server.runner._is_user_authorized_for_source = lambda _: False
            return result
        monkeypatch.setattr(server.images.store, "prepare", revoke)
        response = await client.post(path, data=data, headers=upload_headers("two"))
        assert response.status == 404
        assert (await client.get(path + "/" + descriptor["image_id"] + "/original")).status == 404
        from hermes_constants import get_hermes_home
        assert len(list((get_hermes_home() / "pwa-images").iterdir())) == 1


@pytest.mark.asyncio
async def test_native_encoded_json_rejected_without_transport_decompression(monkeypatch, tmp_path):
    async with service(monkeypatch, tmp_path) as (_, client, _):
        response = await client.post("/v1/pwa/commands", data=b"not gzip",
                                     headers={"Content-Type": "application/json", "Content-Encoding": "gzip"})
        assert response.status == 400
        assert (await client.get("/v1/pwa/health")).status == 200
