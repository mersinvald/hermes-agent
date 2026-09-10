"""Owner image commands through native HTTP, durable queue and real turn lease."""
import base64
import asyncio
import copy
import hashlib
import io
import json
from dataclasses import replace

import pytest
from PIL import Image

from gateway.pwa_image_policy import PIXEL_POLICY, model_pixels
from gateway.pwa_images import ImageUnavailable, PrivateImageStore
from tests.gateway.test_pwa_images import picture
from tests.gateway.test_pwa_image_http import upload_headers
from tests.gateway.test_pwa_http import OWNER, headers, model_config, service
from tests.gateway.test_native_commands import JournalAgent, command, started


def models():
    result = model_config()
    result["entries"][0]["capabilities"] = {**result["entries"][0]["capabilities"], "image_input": "supported"}
    return result


def resolver(_):
    return {"provider": "custom", "base_url": "https://synthetic.invalid/v1", "api_key": "synthetic"}


def configure_agent(monkeypatch):
    original = JournalAgent.__init__
    def init(self, **kwargs):
        original(self, **kwargs)
        self.model, self.provider, self.base_url = kwargs["model"], kwargs["provider"], kwargs["base_url"]
    monkeypatch.setattr(JournalAgent, "__init__", init)


async def upload(client, root, upload_id="one", data=None):
    response = await client.post(f"/v1/pwa/conversations/{root}/images",
                                 data=data or picture(exif=True), headers=upload_headers(upload_id))
    assert response.status == 201, await response.text()
    return await response.json()


async def initialize(client, root):
    response = await client.get(f"/v1/pwa/conversations/{root}/model")
    assert response.status == 200, await response.text()


@pytest.mark.asyncio
async def test_images_durable_order_active_send_queue_replay_and_history(monkeypatch, tmp_path):
    async with service(monkeypatch, tmp_path, models=models(), resolver=resolver) as (server, client, native):
        configure_agent(monkeypatch)
        root, db = native[4].session_id, native[2]
        await initialize(client, root)
        descriptors = [await upload(client, root, str(i), picture(size=(2500 + i, 80))) for i in range(2)]
        ids = [d["image_id"] for d in descriptors]
        first = command(root, "  authored caption  ", expected_model_version=1)
        first["payload"]["image_ids"] = ids
        response = await client.post("/v1/pwa/commands", json=first)
        assert response.status == 200, await response.text()
        receipt = await response.json()
        await started()
        wire = JournalAgent.effects[0]
        assert wire[0] == {"type": "text", "text": "  authored caption  "}
        for index, part in enumerate(wire[1:]):
            data = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
            with Image.open(io.BytesIO(data)) as image:
                assert image.width == 2500 + index and image.width > descriptors[index]["preview"]["width"]
                assert not image.getexif() and "icc_profile" not in image.info
        row = db.native_command_lookup(server.ingress._command_scope(OWNER), "c1")
        stored = json.loads(row["payload_json"])
        assert stored["payload"] == first["payload"]
        assert [i["image_id"] for i in stored["_native_images"]] == ids
        assert all(i["policy"] == PIXEL_POLICY for i in stored["_native_images"])
        assert "_native_images" not in json.dumps(receipt)
        expected = hashlib.sha256(("native-command-v1\n" + json.dumps(first, sort_keys=True, separators=(",", ":"))).encode()).hexdigest()
        assert receipt["payload_fingerprint"] == expected
        history = await (await client.get(f"/v1/pwa/conversations/{root}/history")).json()
        assert history["messages"][0]["content"] == "  authored caption  "
        assert history["messages"][0]["image_ids"] == ids
        assert "representations" not in json.dumps(history) and "data:image" not in json.dumps(history)
        second = command(root, "", id="c2", expected_model_version=1)
        second["payload"]["image_ids"] = list(reversed(ids))
        queued = await client.post("/v1/pwa/commands", json=second)
        assert queued.status == 200, await queued.text()
        assert (await queued.json())["effective_action"] == "queue"
        steer = {**second, "command_id": "steer", "type": "steer", "target_execution_id": row["resulting_execution_id"]}
        assert (await client.post("/v1/pwa/commands", json=steer)).status == 400
        assert db.native_command_lookup(row["scope"], "steer") is None
        replay = await client.post("/v1/pwa/commands", json=first)
        assert replay.status == 200 and (await replay.json())["payload_fingerprint"] == expected
        changed = copy.deepcopy(first)
        changed["payload"]["image_ids"].reverse()
        assert (await client.post("/v1/pwa/commands", json=changed)).status == 409
        assert (await client.post("/v1/pwa/commands", json={**first, "_native_images": []})).status == 400
        JournalAgent.gates[0].set()
        await started()
        assert len(JournalAgent.effects) == 2
        assert JournalAgent.effects[1][0] == {"type": "text", "text": ""}
        assert [i["image_url"]["url"] for i in JournalAgent.effects[1][1:]] == [i["image_url"]["url"] for i in wire[:0:-1]]
        server.runner._is_user_authorized_for_source = lambda _: False
        assert (await client.post("/v1/pwa/commands", json=first)).status == 404


@pytest.mark.asyncio
async def test_all_images_authorized_before_admission_and_route_unknown_rejected(monkeypatch, tmp_path):
    async with service(monkeypatch, tmp_path, models=models(), resolver=resolver) as (server, client, native):
        root, db = native[4].session_id, native[2]
        await initialize(client, root)
        image = await upload(client, root)
        request = command(root, "caption", expected_model_version=1)
        for ids in ([image["image_id"], "im_" + "f" * 32 + "_" + "e" * 64], [image["image_id"]] * 2):
            request["payload"]["image_ids"] = ids
            assert (await client.post("/v1/pwa/commands", json=request)).status in {400, 404}
            assert db.native_command_lookup(server.ingress._command_scope(OWNER), "c1") is None
        request["payload"]["image_ids"] = [image["image_id"]]
        assert (await client.post("/v1/pwa/commands", json=request, headers=headers(replace(OWNER, subject="foreign")))).status == 403
        server.models.config = replace(server.models.config, entries=tuple(
            replace(r, capabilities=replace(r.capabilities, image_input="unknown")) for r in server.models.config.entries))
        assert (await client.post("/v1/pwa/commands", json=request)).status == 400
        assert db.native_command_rows(root) == []


@pytest.mark.parametrize(("fmt", "mime"), [("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")])
def test_immutable_full_pixels_original_preserved_and_restart(tmp_path, fmt, mime):
    store = PrivateImageStore(tmp_path / "pwa-images")
    scope, root = ["synthetic"], "root"
    original = picture(fmt, exif=True)
    stage = store.stage()
    try:
        stage.write(original)
        descriptor = store.prepare(stage, scope, root, "one", mime)
        descriptor, _ = store.publish(stage, scope, descriptor)
    finally:
        stage.close()
    image_id = descriptor["image_id"]
    info, pixels = model_pixels(store, scope, root, image_id)
    assert store.read(scope, root, image_id, "original")[1] == original
    assert b"SYNTHETIC PRIVATE" not in pixels
    store.close()
    store = PrivateImageStore(tmp_path / "pwa-images")
    try:
        assert model_pixels(store, scope, root, image_id, expected=info) == (info, pixels)
        with pytest.raises(ImageUnavailable):
            model_pixels(store, ["foreign"], root, image_id)
        with pytest.raises(ImageUnavailable):
            model_pixels(store, scope, root, image_id, expected={**info, "sha256": "f" * 64})
        derivative = tmp_path / "pwa-images" / (image_id.split("_")[-1] + "-" + PIXEL_POLICY)
        (derivative / "pixels").unlink()
        (derivative / "manifest").unlink()
        derivative.rmdir()
        assert model_pixels(store, scope, root, image_id, expected=info) == (info, pixels)
        (derivative / "pixels").write_bytes(b"corrupt")
        with pytest.raises(ImageUnavailable):
            model_pixels(store, scope, root, image_id, expected=info)
        assert (derivative / "pixels").read_bytes() == b"corrupt"
        assert store.read(scope, root, image_id, "original")[1] == original
    finally:
        store.close()


@pytest.mark.asyncio
async def test_redirect_sidecar_survives_promotion_and_historical_retry(monkeypatch, tmp_path):
    async with service(monkeypatch, tmp_path, models=models(), resolver=resolver) as (server, client, native):
        configure_agent(monkeypatch)
        root, db = native[4].session_id, native[2]
        await initialize(client, root)
        image = await upload(client, root)
        initial = await client.post("/v1/pwa/commands", json=command(root))
        assert initial.status == 200
        await started()
        execution = db.native_execution(root)["execution_id"]
        direction = command(root, "new image direction", id="redirect", kind="redirect",
                            target_execution_id=execution, expected_model_version=1)
        direction["payload"]["image_ids"] = [image["image_id"]]
        invalid = copy.deepcopy(direction)
        invalid["payload"]["image_ids"].append("im_" + "e" * 32 + "_" + "f" * 64)
        assert (await client.post("/v1/pwa/commands", json=invalid)).status == 404
        assert not db.native_execution_cancel_requested(execution)
        response = await client.post("/v1/pwa/commands", json=direction)
        assert response.status == 200, await response.text()
        scope = server.ingress._command_scope(OWNER)
        control = db.native_control_lookup(scope, "redirect")
        frozen = json.loads(control["payload_json"])["_native_images"]
        # Recreate native image storage before the deferred input runs.
        server.images.close()
        JournalAgent.gates[0].set()
        await started()
        row = db.native_command_lookup(scope, "redirect")
        assert row["phase"] == "applied"
        assert json.loads(row["payload_json"])["_native_images"] == frozen
        assert JournalAgent.effects[1][0]["text"] == "new image direction"
        assert (await client.post("/v1/pwa/commands", json=direction)).status == 200
        changed = copy.deepcopy(direction)
        changed["payload"]["text"] += " changed"
        assert (await client.post("/v1/pwa/commands", json=changed)).status == 409


@pytest.mark.asyncio
async def test_queued_media_rechecks_actual_captured_route(monkeypatch, tmp_path):
    async with service(monkeypatch, tmp_path, models=models(), resolver=resolver) as (server, client, native):
        configure_agent(monkeypatch)
        root, db = native[4].session_id, native[2]
        await initialize(client, root)
        image = await upload(client, root)
        assert (await client.post("/v1/pwa/commands", json=command(root))).status == 200
        await started()
        queued = command(root, "must remain complete", id="media", kind="queue", expected_model_version=1)
        queued["payload"]["image_ids"] = [image["image_id"]]
        assert (await client.post("/v1/pwa/commands", json=queued)).status == 200
        selected = await client.put(f"/v1/pwa/conversations/{root}/model", json={
            "schema_version": "1.0", "mutation_id": "change", "model_id": "deep", "expected_model_version": 1})
        assert selected.status == 200, await selected.text()
        JournalAgent.gates[0].set()
        scope = server.ingress._command_scope(OWNER)
        async def failed():
            while db.native_command_lookup(scope, "media")["phase"] in {"queued", "assigned"}:
                await asyncio.sleep(0.01)
        await asyncio.wait_for(failed(), 10)
        assert db.native_command_lookup(scope, "media")["phase"] == "not_applied"
        with db._read_ctx() as conn:
            assert conn.execute("SELECT observed_state FROM native_executions WHERE conversation_id=? ORDER BY created_order DESC LIMIT 1", (root,)).fetchone()[0] == "failed"
        assert JournalAgent.effects == ["first"]
        # Historical retry remains the already admitted command after selection
        # changes; it creates no new execution.
        assert (await client.post("/v1/pwa/commands", json=queued)).status == 200


@pytest.mark.asyncio
async def test_compression_preserves_authored_caption_all_refs_and_sync(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from agent.context_compressor import (
        COMPRESSED_SUMMARY_METADATA_KEY, SUMMARY_PREFIX, _SUMMARY_END_MARKER,
        split_user_originated_turn,
    )
    from gateway.pwa_image_policy import NativeMediaGuard
    from tests.gateway.test_pwa_owner_history import sweep
    async with service(monkeypatch, tmp_path, models=models(), resolver=resolver) as (server, client, native):
        configure_agent(monkeypatch)
        root, db = native[4].session_id, native[2]
        await initialize(client, root)
        ids = [(await upload(client, root, str(i)))["image_id"] for i in range(2)]
        body = command(root, "  exact authored text  ", expected_model_version=1)
        body["payload"]["image_ids"] = ids
        assert (await client.post("/v1/pwa/commands", json=body)).status == 200
        await started()
        JournalAgent.gates[0].set()
        async def done():
            while db.native_execution(root) is not None:
                await asyncio.sleep(0.01)
        await asyncio.wait_for(done(), 10)
        user = db.get_messages_as_conversation(root)[0]
        carrier = {**user, "content": f"{SUMMARY_PREFIX}\nSynthetic summary\n{_SUMMARY_END_MARKER}\n\n  exact authored text  ",
                   "display_kind": "hidden", COMPRESSED_SUMMARY_METADATA_KEY: True}
        _, display = split_user_originated_turn(carrier)
        assert display["display_metadata"]["pwa_images"] == user["display_metadata"]["pwa_images"]
        db.publish_compression_child(parent_session_id=root, child_session_id="compressed-tip",
            source=native[5].platform.value, messages=[carrier], require_compression_lease=False)
        history = await (await client.get(f"/v1/pwa/conversations/{root}/history")).json()
        image_rows = [m for m in history["messages"] if "image_ids" in m]
        assert image_rows and all(m["image_ids"] == ids and m["content"] == "  exact authored text  " for m in image_rows)
        pages = await sweep(client, "sync", limit=1)
        synced = [m for p in pages for g in p["groups"] for m in g["messages"] if "image_ids" in m]
        assert len(synced) == len(image_rows) and all(m["image_ids"] == ids for m in synced)
        guard = NativeMediaGuard(server.images, SimpleNamespace(conversation_id=root),
                                 db.native_image_command(root), "daily")
        replay = guard.history([carrier])[0]
        assert "Synthetic summary" in replay["content"][0]["text"]
        assert len(replay["content"]) == 3


@pytest.mark.parametrize("caption", ["x" * 20000, "📷" * 20000, "x" * 65537, "📷" * 100000,
                                     "\0" * 65536, "\0" * 65537],
                         ids=["large-ascii", "large-unicode", "omitted-ascii", "omitted-unicode",
                              "nul-boundary", "omitted-nul"])
@pytest.mark.asyncio
async def test_large_image_caption_sql_history_and_sync_keep_all_refs(monkeypatch, tmp_path, caption):
    from tests.gateway.test_pwa_owner_history import sweep
    async with service(monkeypatch, tmp_path, models=models(), resolver=resolver) as (server, client, native):
        configure_agent(monkeypatch)
        root, db = native[4].session_id, native[2]
        await initialize(client, root)
        with Image.effect_noise((256, 256), 100).convert("RGB") as noisy:
            buffer = io.BytesIO()
            noisy.save(buffer, format="PNG")
            data = buffer.getvalue()
        ids = [(await upload(client, root, str(i), data))["image_id"] for i in range(10)]
        body = command(root, caption, expected_model_version=1)
        body["payload"]["image_ids"] = ids
        # Encode real UTF-8 so the accepted maximum Unicode caption fits the
        # native request-byte limit instead of aiohttp's default ASCII escapes.
        response = await client.post("/v1/pwa/commands", data=json.dumps(body, ensure_ascii=False).encode(),
                                     headers={"Content-Type": "application/json"})
        assert response.status == 200, await response.text()
        await started()
        with db._read_ctx() as conn:
            stored = conn.execute("SELECT id,length(CAST(content AS BLOB)),length(display_metadata) FROM messages WHERE session_id=? AND role='user'",
                                  (root,)).fetchone()
        assert stored[2] > 16384
        assert stored[1] > 65536
        projected = db.native_pwa_history_rows(root, 0, stored[0], 1)[0]
        display = json.loads(projected["image_display"])
        assert set(display) == {"text", "text_length", "image_ids"}
        assert display["image_ids"] == ids and display["text_length"] == len(caption)
        assert projected["display_metadata"] is None
        if len(caption) > 65536:
            assert display["text"] is None
            assert stored[1] > 65536 and projected["content"] is None
        else:
            assert display["text"] == caption
        history_response = await client.get(f"/v1/pwa/conversations/{root}/history")
        assert history_response.status == 200, await history_response.text()
        history = await history_response.json()
        pages = await sweep(client, "sync", limit=1)
        rows = history["messages"] + [m for p in pages for g in p["groups"] for m in g["messages"]]
        assert len(rows) == 2
        for row in rows:
            assert row["image_ids"] == ids
            assert row["content"] == (caption if len(caption) <= 65536 else None)
            assert row["content_state"] == ("available" if len(caption) <= 65536 else "omitted")
            assert row["omission_reason"] == (None if len(caption) <= 65536 else "oversized")
        for forbidden in ("representations", "data:image", "pixels", "_native_images", "pwa_images"):
            assert forbidden not in json.dumps([history, pages, projected])
