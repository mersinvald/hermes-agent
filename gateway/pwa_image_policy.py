"""Native PWA attachment admission. Originals never enter a model request.

The versioned pixel representation is retained separately from the UI preview.
Metadata disclosure has no opt-in switch here: destination-wide policy guards
must exist before a later policy can disclose extracted metadata.
"""
from __future__ import annotations

import base64
import errno
import hashlib
import io
import json
import os
import secrets

from PIL import Image, ImageOps

from gateway.pwa_images import (
    IMAGE_ID, ImageRejected, ImageUnavailable, _canonical, _closed_json, _dir, _file,
)

PIXEL_POLICY = "pwa-pixels-jpeg-v1"
MAX_PIXEL_BYTES = 20 * 1024 * 1024
MAX_COMMAND_PIXEL_BYTES = 24 * 1024 * 1024
SIDECAR = "_native_images"


def validate_image_payload(command):
    payload = command.get("payload")
    if (not isinstance(payload, dict) or set(payload) - {"text", "image_ids"}
            or not isinstance(payload.get("text"), str) or len(payload["text"]) > 100000
            or len(payload["text"].encode()) > 409600):
        raise ValueError("invalid message payload")
    ids = payload.get("image_ids", [])
    if (not isinstance(ids, list) or ("image_ids" in payload and not 1 <= len(ids) <= 10)
            or any(not isinstance(i, str) or not IMAGE_ID.fullmatch(i) for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("invalid image identities")
    if not ids and not payload["text"].strip():
        raise ValueError("expected text or images")
    if ids:
        if command.get("type") not in {"send", "queue", "redirect"}:
            raise ValueError("images are unavailable for this action")
        version = command.get("expected_model_version")
        if type(version) is not int or not 1 <= version <= 9007199254740991:
            raise ValueError("image admission requires a model version")
    return ids


def require_image_route(catalog, model_id):
    config = getattr(catalog, "config", None)
    route = config.route(model_id) if config is not None else None
    if route is None or route.capabilities.image_input != "supported":
        raise ImageRejected("selected native route does not support image input")
    return catalog.resolve(model_id)


class _BoundedPixels(io.BytesIO):
    def write(self, data):
        if self.tell() + len(data) > MAX_PIXEL_BYTES:
            raise ImageRejected("full-resolution image exceeds model byte limit")
        return super().write(data)


def fresh_pixels(original):
    """Full dimensions, applied orientation, RGB JPEG; no copied metadata."""
    with Image.open(io.BytesIO(original)) as decoded:
        decoded.load()
        with ImageOps.exif_transpose(decoded) as oriented:
            with Image.new("RGB", oriented.size, "white") as pixels:
                if "A" in oriented.getbands():
                    with oriented.convert("RGBA") as rgba:
                        pixels.paste(rgba, mask=rgba.getchannel("A"))
                else:
                    with oriented.convert("RGB") as rgb:
                        pixels.paste(rgb)
                with _BoundedPixels() as output:
                    pixels.save(output, format="JPEG", quality=95, subsampling=0)
                    return output.getvalue(), pixels.size


def model_pixels(store, scope, root, image_id, *, expected=None):
    """Verify retained authority, then read/publish one immutable derivative.

    A missing derivative can be rebuilt from the retained original. For an
    admitted command its exact digest must match; a codec change cannot silently
    alter an already accepted image. Corruption is never replaced in place.
    """
    descriptor, original = store.read(scope, root, image_id, "original")
    key = IMAGE_ID.fullmatch(image_id)[1] + "-" + PIXEL_POLICY
    identity = {"image_id": image_id, "original_sha256": descriptor["original"]["sha256"],
                "policy": PIXEL_POLICY}

    def read():
        with store._record(key) as fd:
            with _file("manifest", parent=fd) as file:
                raw = file.read(4097)
            if len(raw) > 4096:
                raise ImageUnavailable("model image unavailable")
            try:
                info = _closed_json(raw)
                if (set(info) != {*identity, "sha256", "byte_size", "width", "height"}
                        or any(info[k] != v for k, v in identity.items())
                        or type(info["byte_size"]) is not int or not 1 <= info["byte_size"] <= MAX_PIXEL_BYTES
                        or any(type(info[k]) is not int or not 1 <= info[k] <= 16384 for k in ("width", "height"))
                        or info["width"] * info["height"] > 40_000_000):
                    raise ValueError()
                with _file("pixels", parent=fd) as file:
                    data = file.read(info["byte_size"] + 1)
                if len(data) != info["byte_size"] or hashlib.sha256(data).hexdigest() != info["sha256"]:
                    raise ValueError()
                if expected is not None and info != expected:
                    raise ValueError()
                return info, data
            except (ValueError, TypeError, KeyError):
                raise ImageUnavailable("model image unavailable") from None

    try:
        return read()
    except FileNotFoundError:
        pass
    data, (width, height) = fresh_pixels(original)
    info = {**identity, "sha256": hashlib.sha256(data).hexdigest(), "byte_size": len(data),
            "width": width, "height": height}
    if expected is not None and info != expected:
        raise ImageUnavailable("retained image representation cannot be reconstructed")
    store.headroom(len(data))
    stage = ".pixels-" + secrets.token_hex(24)
    os.mkdir(stage, 0o700, dir_fd=store.fd)
    fd = _dir(stage, parent=store.fd)
    published = False
    try:
        for name, value in (("pixels", data), ("manifest", _canonical(info))):
            with _file(name, parent=fd, write=True) as file:
                file.write(value)
                file.flush()
                os.fsync(file.fileno())
        os.fsync(fd)
        try:
            os.rename(stage, key, src_dir_fd=store.fd, dst_dir_fd=store.fd)
            published = True
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
        os.fsync(store.fd)
        return read()
    finally:
        if not published:
            for name in ("pixels", "manifest"):
                try:
                    os.unlink(name, dir_fd=fd)
                except FileNotFoundError:
                    pass
            os.rmdir(stage, dir_fd=store.fd)
        os.close(fd)


def prepare_images(store, scope, root, ids, *, expected=None):
    if len(ids) > store.limits.max_images_per_message:
        raise ImageRejected("too many images")
    if expected is not None and (not isinstance(expected, list) or len(expected) != len(ids)):
        raise ImageUnavailable("retained image admission unavailable")
    images, total = [], 0
    for index, image_id in enumerate(ids):
        info, _ = model_pixels(store, scope, root, image_id,
                               expected=expected[index] if expected is not None else None)
        total += info["byte_size"]
        if total > MAX_COMMAND_PIXEL_BYTES:
            raise ImageRejected("image command exceeds aggregate model byte limit")
        images.append(info)
    return images


def command_content(images, row, model_id):
    """Reauthorize at the captured native route, including durable replay."""
    from gateway.conversation_control import Principal
    body = json.loads(row["payload_json"])
    ids = body["payload"].get("image_ids", [])
    if not ids:
        return None
    validate_image_payload(body)
    _, issuer, subject = json.loads(row["scope"])
    principal = Principal(issuer, subject)
    root = row["conversation_id"]
    scope = images.owner_scope(principal, root)
    require_image_route(images.parent.models, model_id)
    frozen = body.get(SIDECAR)
    if not isinstance(frozen, list) or len(frozen) != len(ids):
        raise ImageUnavailable("retained image admission unavailable")
    parts = [{"type": "text", "text": body["payload"]["text"]}]
    total = 0
    for image_id, expected in zip(ids, frozen):
        _, data = model_pixels(images.image_store(), scope, root, image_id, expected=expected)
        total += len(data)
        if total > MAX_COMMAND_PIXEL_BYTES:
            raise ImageRejected("image command exceeds aggregate model byte limit")
        parts.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"), "detail": "high"}})
    if images.owner_scope(principal, root) != scope:
        raise PermissionError("image owner changed")
    require_image_route(images.parent.models, model_id)
    return parts


def display_images(body):
    ids = body.get("payload", {}).get("image_ids", [])
    if not ids:
        return None
    frozen = body.get(SIDECAR)
    if not isinstance(frozen, list) or [i.get("image_id") for i in frozen] != ids:
        raise ImageUnavailable("retained image admission unavailable")
    return {"image_ids": ids, "text": body["payload"]["text"], "representations": frozen}


class ManagedImageError(RuntimeError):
    """The complete native media turn must fail without a text retry."""


def _image_urls(messages):
    urls = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                image = part.get("image_url")
                urls.append(image.get("url") if isinstance(image, dict) else image)
    return urls


class NativeMediaGuard:
    """One captured native turn; validate the actual request before dispatch.

    This first policy permits no image shrinking, stripping or native provider
    fallback. The private gateway's downstream failover receives the same
    stripped pixels; an image rejection ends the turn truthfully.
    """
    def __init__(self, images, execution, row, model_id):
        self.images, self.execution, self.model_id = images, execution, model_id
        self.scope = row["scope"]
        self.required = []
        self.current = []

    def content(self, row):
        parts = command_content(self.images, row, self.model_id)
        if parts is not None:
            self.required.extend(_image_urls([{"content": parts}]))
        return parts

    def history(self, history):
        result = []
        for message in history or []:
            metadata = message.get("display_metadata") or {}
            display = metadata.get("pwa_images") if isinstance(metadata, dict) else None
            if display is not None:
                if not isinstance(display, dict) or set(display) != {"text", "image_ids", "representations"}:
                    raise ManagedImageError("Retained image history is unavailable.")
                row = {"scope": self.scope, "conversation_id": self.execution.conversation_id,
                       "payload_json": json.dumps({"type": "send", "expected_model_version": 1,
                           "payload": {"text": display["text"], "image_ids": display["image_ids"]},
                           SIDECAR: display["representations"]})}
                parts = self.content(row)
                if isinstance(message.get("content"), str):
                    # A native compression carrier can also contain its summary;
                    # preserve it in API context while display stays authored.
                    parts[0]["text"] = message["content"]
                message = {**message, "content": parts}
            result.append(message)
        return result

    def reloaded_history(self, history):
        self.required = []
        result = self.history(history)
        self.required.extend(self.current)
        return result

    def check_route(self, agent):
        from gateway.conversation_control import Principal
        _, issuer, subject = json.loads(self.scope)
        self.images.owner_scope(Principal(issuer, subject), self.execution.conversation_id)
        route = require_image_route(self.images.parent.models, self.model_id)
        if (agent.model != route.model
                or agent.provider != route.runtime.get("provider")
                or str(agent.base_url or "").rstrip("/") != str(route.runtime.get("base_url") or "").rstrip("/")):
            raise ManagedImageError("The captured image route changed; no replacement request was sent.")

    def before_request(self, agent, kwargs):
        self.check_route(agent)
        extra = kwargs.get("extra_body") or {}
        if (kwargs.get("model", agent.model) != agent.model
                or not isinstance(extra, dict)
                or set(extra) & {"messages", "model", "input", "contents", "system"}):
            raise ManagedImageError("Request overrides cannot replace the native image request.")
        # Preserve order and multiplicity while allowing additional native tool
        # images. A transformed/missing managed attachment is never a text retry.
        actual = iter(_image_urls([m for m in kwargs.get("messages", []) if m.get("role") == "user"]))
        for required in self.required:
            if not any(value == required for value in actual):
                raise ManagedImageError("The native request cannot preserve all retained images.")

    @staticmethod
    def reject_error(error):
        from agent.message_sanitization import _looks_like_image_content_rejection
        if isinstance(error, ManagedImageError):
            raise error
        status = getattr(error, "status_code", None)
        body = str(getattr(error, "body", None) or error)
        if status == 413 or _looks_like_image_content_rejection(body):
            raise ManagedImageError("The selected route rejected the image request; no text-only retry was sent.") from None
