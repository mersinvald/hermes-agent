"""Synthetic images exercise retained storage, real codecs and descriptor FDs."""
import concurrent.futures
import errno
import io
import os
from dataclasses import replace

from PIL import Image, PngImagePlugin
import pytest

from gateway.pwa_images import ImageConflict, ImageLimits, ImageRejected, ImageUnavailable, PrivateImageStore

SCOPE = ["concierge", "https://issuer.invalid", "owner", {"user_id": "native-owner"}]


def picture(fmt="PNG", *, exif=False, size=(48, 24)):
    with Image.new("RGB", size, (30, 120, 210)) as image:
        image.putpixel((0, 0), (200, 30, 50))
        buffer = io.BytesIO()
        options = {}
        if exif:
            metadata = Image.Exif()
            metadata[274] = 6
            metadata[270] = "SYNTHETIC PRIVATE CAPTURE CONTEXT"
            metadata[34853] = {1: "N", 2: (0.0, 0.0, 0.0), 3: "E", 4: (0.0, 0.0, 0.0)}
            options["exif"] = metadata
            if fmt == "PNG":
                info = PngImagePlugin.PngInfo()
                info.add_text("Comment", "SYNTHETIC PRIVATE COMMENT")
                options["pnginfo"] = info
        image.save(buffer, format=fmt, **options)
        return buffer.getvalue()


def upload(store, data, upload_id="one", mime="image/png", scope=SCOPE):
    stage = store.stage()
    try:
        for offset in range(0, len(data), 19):
            stage.write(data[offset:offset+19])
        descriptor = store.prepare(stage, scope, "root", upload_id, mime)
        return store.publish(stage, scope, descriptor)
    finally:
        stage.close()


@pytest.fixture
def store(tmp_path):
    current = PrivateImageStore(tmp_path / "pwa-images")
    yield current
    current.close()


@pytest.mark.parametrize(("fmt", "mime"), [("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")])
def test_original_retained_preview_pixels_without_metadata(store, fmt, mime):
    data = picture(fmt, exif=True)
    descriptor, created = upload(store, data, mime=mime)
    assert created and descriptor["admission_state"] == "unavailable"
    assert (descriptor["original"]["width"], descriptor["original"]["height"]) == (48, 24)
    assert (descriptor["preview"]["width"], descriptor["preview"]["height"]) == (24, 48)
    assert store.read(SCOPE, "root", descriptor["image_id"], "original")[1] == data
    with Image.open(io.BytesIO(data)) as original:
        assert original.getexif().get_ifd(34853)[1] == "N"
    preview = store.read(SCOPE, "root", descriptor["image_id"], "preview")[1]
    with Image.open(io.BytesIO(preview)) as image:
        assert image.format == "JPEG" and image.size == (24, 48)
        assert not image.getexif()
        assert not set(image.info) & {"exif", "xmp", "icc_profile", "comment"}
    assert b"SYNTHETIC PRIVATE" not in preview


def test_concurrent_same_upload_id_restart_lost_ack_and_conflict(store, tmp_path):
    data = picture()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: upload(store, data), range(8)))
    assert sum(created for _, created in results) == 1
    descriptors = [descriptor for descriptor, _ in results]
    assert all(descriptor == descriptors[0] for descriptor in descriptors)
    restarted = PrivateImageStore(tmp_path / "pwa-images")
    try:
        assert upload(restarted, data) == (descriptors[0], False)
        for changed, mime in [(picture(size=(10, 10)), "image/png"), (data, "image/jpeg")]:
            with pytest.raises(ImageConflict):
                upload(restarted, changed, mime=mime)
        assert restarted.read(SCOPE, "root", descriptors[0]["image_id"], "original")[1] == data
    finally:
        restarted.close()
    assert len(list((tmp_path / "pwa-images").iterdir())) == 1


@pytest.mark.parametrize("payload,mime", [(b"not an image", "image/png"), (b"%PDF-1.4", "application/pdf"), (picture(), "image/jpeg"), (picture("JPEG")[:-80], "image/jpeg")])
def test_invalid_bytes_do_not_publish(store, tmp_path, payload, mime):
    with pytest.raises(ImageRejected):
        upload(store, payload, mime=mime)
    assert not list((tmp_path / "pwa-images").iterdir())


def test_animation_decompression_and_byte_limits(store, tmp_path):
    buffer = io.BytesIO()
    with Image.new("RGB", (16, 16), "red") as first, Image.new("RGB", (16, 16), "blue") as second:
        first.save(buffer, format="WEBP", save_all=True, append_images=[second])
    with pytest.raises(ImageRejected):
        upload(store, buffer.getvalue(), mime="image/webp")
    store.limits = replace(store.limits, max_decoded_pixels=100)
    with pytest.raises(ImageRejected):
        upload(store, picture())
    store.limits = replace(store.limits, max_original_bytes=10)
    with pytest.raises(ImageRejected):
        upload(store, picture())
    assert not list((tmp_path / "pwa-images").iterdir())


def test_scope_root_symlink_and_corrupt_payload_fail_closed(store, tmp_path):
    descriptor, _ = upload(store, picture())
    image_id = descriptor["image_id"]
    with pytest.raises(ImageUnavailable):
        store.read([*SCOPE[:-1], {"user_id": "reassigned"}], "root", image_id, "original")
    with pytest.raises(ImageUnavailable):
        store.read(SCOPE, "foreign-root", image_id, "original")
    folder = tmp_path / "pwa-images" / image_id.rsplit("_", 1)[1]
    original = folder / "original"
    held = original.read_bytes()
    original.write_bytes(b"x" * len(held))
    with pytest.raises(ImageUnavailable):
        store.read(SCOPE, "root", image_id, "original")
    original.unlink()
    outside = tmp_path / "private-elsewhere"
    outside.write_bytes(held)
    original.symlink_to(outside)
    with pytest.raises(ImageUnavailable):
        store.read(SCOPE, "root", image_id, "original")
    assert outside.read_bytes() == held
    actual = tmp_path / "original-folder"
    folder.rename(actual)
    folder.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ImageUnavailable):
        store.read(SCOPE, "root", image_id, "original")


def test_publication_fsync_failure_retains_original_and_retry(store, tmp_path, monkeypatch):
    data = picture()
    fsync = os.fsync
    def interrupted(fd):
        if fd == store.fd:
            raise OSError(errno.ENOSPC, "synthetic no room during durable publication")
        return fsync(fd)
    monkeypatch.setattr(os, "fsync", interrupted)
    with pytest.raises(OSError):
        upload(store, data)
    records = list((tmp_path / "pwa-images").iterdir())
    assert len(records) == 1 and (records[0] / "original").read_bytes() == data
    monkeypatch.setattr(os, "fsync", fsync)
    descriptor, created = upload(store, data)
    assert not created and store.read(SCOPE, "root", descriptor["image_id"], "original")[1] == data


def test_disk_headroom_rejects_without_deleting_published_original(store, tmp_path):
    descriptor, _ = upload(store, picture())
    store.limits = replace(store.limits, min_free_bytes=10**30)
    with pytest.raises(OSError):
        upload(store, picture(), "second")
    assert store.read(SCOPE, "root", descriptor["image_id"], "original")[1] == picture()
    assert len(list((tmp_path / "pwa-images").iterdir())) == 1


def test_failed_unpublished_original_or_manifest_keeps_existing_image(store, tmp_path, monkeypatch):
    descriptor, _ = upload(store, picture())
    import gateway.pwa_images as images
    original_file = images._file
    def fail_manifest(name, *, parent, write=False):
        if name == "manifest" and write:
            raise OSError(errno.EDQUOT, "synthetic quota")
        return original_file(name, parent=parent, write=write)
    monkeypatch.setattr(images, "_file", fail_manifest)
    with pytest.raises(OSError):
        upload(store, picture(), "second")
    assert store.read(SCOPE, "root", descriptor["image_id"], "original")[1] == picture()
    assert len(list((tmp_path / "pwa-images").iterdir())) == 1


def test_corrupted_original_is_not_acknowledged_by_idempotent_retry(store, tmp_path):
    descriptor, _ = upload(store, picture())
    folder = tmp_path / "pwa-images" / descriptor["image_id"].rsplit("_", 1)[1]
    (folder / "original").write_bytes(b"corrupt")
    with pytest.raises(ImageUnavailable):
        upload(store, picture())
    assert (folder / "original").read_bytes() == b"corrupt"


def test_descriptor_rejects_decoded_pixel_product(store, tmp_path):
    import json
    descriptor, _ = upload(store, picture())
    folder = tmp_path / "pwa-images" / descriptor["image_id"].rsplit("_", 1)[1]
    manifest = folder / "manifest"
    value = json.loads(manifest.read_bytes())
    value["descriptor"]["original"].update(width=16384, height=16384)
    manifest.write_text(json.dumps(value))
    with pytest.raises(ImageUnavailable):
        store.read(SCOPE, "root", descriptor["image_id"])


def test_stage_constructor_failure_closes_owned_fd_and_directory(store, tmp_path, monkeypatch):
    import gateway.pwa_images as images
    opened = []
    real_dir = images._dir
    def remember_dir(*args, **kwargs):
        fd = real_dir(*args, **kwargs)
        opened.append(fd)
        return fd
    def no_space(*args, **kwargs):
        raise OSError(errno.ENOSPC, "synthetic original-create failure")
    monkeypatch.setattr(images, "_dir", remember_dir)
    monkeypatch.setattr(images, "_file", no_space)
    with pytest.raises(OSError):
        store.stage()
    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])
    assert not list((tmp_path / "pwa-images").iterdir())


@pytest.mark.asyncio
async def test_repeated_cancellation_retains_stage_and_worker_until_thread_finishes(store, tmp_path):
    import asyncio
    import threading
    from gateway.pwa_image_http import completed_thread
    stage = store.stage()
    loop = asyncio.get_running_loop()
    entered, released = asyncio.Event(), asyncio.Event()
    unblock = threading.Event()
    def write_in_thread():
        loop.call_soon_threadsafe(entered.set)
        assert unblock.wait(10)
        stage.write(b"synthetic bytes")
    async def own_stage_and_slot():
        try:
            await completed_thread(write_in_thread)
        finally:
            stage.close()
            released.set()
    task = asyncio.create_task(own_stage_and_slot())
    await entered.wait()
    try:
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert task.cancelling() == 2 and not task.done() and not released.is_set()
        assert os.fstat(stage.fd)
        assert len(list((tmp_path / "pwa-images").iterdir())) == 1
    finally:
        unblock.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert released.is_set() and stage.fd is None
    assert not list((tmp_path / "pwa-images").iterdir())
