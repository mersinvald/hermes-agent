"""Private retained PWA originals; never a model attachment or a media cache.

Native HTTP supplies a verified owner/conversation scope. A published record is
immutable and never automatically removed. Paths and input metadata never enter
native execution. Model admission remains unavailable until the M02 route gate.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import secrets
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

MIME_FORMATS = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}
IMAGE_ID = re.compile(r"im_[0-9a-f]{32}_([0-9a-f]{64})\Z")
UPLOAD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
PREVIEW_MAX_BYTES = 8 * 1024 * 1024


class ImageRejected(ValueError):
    pass


class ImageConflict(Exception):
    pass


class ImageUnavailable(LookupError):
    pass


@dataclass(frozen=True)
class ImageLimits:
    max_original_bytes: int = 20 * 1024 * 1024
    max_images_per_message: int = 10
    max_decoded_pixels: int = 40_000_000
    max_dimension: int = 16_384
    workers: int = 1
    min_free_bytes: int = 256 * 1024 * 1024

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise ValueError("invalid private image limits")
        for key, number in value.items():
            upper = (2 if key == "workers" else 16 * 1024**3
                     if key == "min_free_bytes" else getattr(cls, key))
            if type(number) is not int or not 1 <= number <= upper:
                raise ValueError("invalid private image limit")
        return cls(**value)

    def policy(self):
        return {"schema_version": "1.0", "max_original_bytes": self.max_original_bytes,
                "max_images_per_message": self.max_images_per_message,
                "max_decoded_pixels": self.max_decoded_pixels,
                "max_dimension": self.max_dimension,
                "supported_content_types": list(MIME_FORMATS),
                "admission_state": "unavailable"}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _dir(name, *, parent=None):
    return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)


def _file(name, *, parent, write=False):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL if write else os.O_RDONLY
    fd = os.open(name, flags | os.O_NOFOLLOW, 0o600, dir_fd=parent)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ImageUnavailable("private image unavailable")
        return os.fdopen(fd, "wb" if write else "rb")
    except BaseException:
        os.close(fd)
        raise



def _digest_file(file, maximum):
    digest, size = hashlib.sha256(), 0
    file.seek(0)
    while chunk := file.read(65536):
        size += len(chunk)
        if size > maximum:
            raise ImageRejected("image exceeds byte limit")
        digest.update(chunk)
    file.seek(0)
    return digest.hexdigest(), size


def _closed_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate image manifest key")
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))


class StagedImage:
    def __init__(self, store):
        self.store = store
        self.name = ".stage-" + secrets.token_hex(24)
        self.fd = None
        self.original = None
        self.identity = None
        self.size = 0
        self.published = False
        self.ready = False
        os.mkdir(self.name, 0o700, dir_fd=store.fd)
        try:
            info = os.stat(self.name, dir_fd=store.fd, follow_symlinks=False)
            self.identity = (info.st_dev, info.st_ino)
            self.fd = _dir(self.name, parent=store.fd)
            opened = os.fstat(self.fd)
            if (opened.st_dev, opened.st_ino) != self.identity:
                raise ImageUnavailable("private stage unavailable")
            self.original = _file("original", parent=self.fd, write=True)
        except BaseException:
            self.close()
            raise

    def write(self, chunk):
        self.size += len(chunk)
        if self.size > self.store.limits.max_original_bytes:
            raise ImageRejected("image exceeds byte limit")
        self.store.headroom(len(chunk))
        self.original.write(chunk)

    def seal(self):
        self.original.flush()
        os.fsync(self.original.fileno())
        self.original.close()

    def close(self):
        if self.original is not None:
            self.original.close()
        # Only this attempt's held staging FD is eligible for cleanup. Once a
        # rename succeeds, an uncertain fsync/response must retain the original.
        try:
            if not self.published:
                if self.fd is not None:
                    for name in ("original", "preview", "manifest"):
                        try:
                            os.unlink(name, dir_fd=self.fd)
                        except FileNotFoundError:
                            pass
                try:
                    info = os.stat(self.name, dir_fd=self.store.fd, follow_symlinks=False)
                    if (info.st_dev, info.st_ino) == self.identity:
                        os.rmdir(self.name, dir_fd=self.store.fd)
                except FileNotFoundError:
                    pass
        finally:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None



class PrivateImageStore:
    def __init__(self, root: Path, limits=None):
        self.limits = limits or ImageLimits()
        # The parent is the trusted profile's HERMES_HOME. Every added component
        # is opened without following links, including an existing images dir.
        parent = _dir(root.parent)
        try:
            try:
                os.mkdir(root.name, 0o700, dir_fd=parent)
                os.fsync(parent)
            except FileExistsError:
                pass
            self.fd = _dir(root.name, parent=parent)
        finally:
            os.close(parent)
        if os.fstat(self.fd).st_mode & 0o077:
            os.close(self.fd)
            raise ImageUnavailable("private image directory permissions unavailable")

    def close(self):
        os.close(self.fd)

    def headroom(self, extra=0):
        available = os.fstatvfs(self.fd)
        # Cover every allowed in-flight original and derivative before consuming
        # the profile PVC shared with native history. Never free retained media.
        reserved = self.limits.workers * (self.limits.max_original_bytes + PREVIEW_MAX_BYTES)
        if available.f_bavail * available.f_frsize < self.limits.min_free_bytes + reserved + extra:
            raise OSError(errno.ENOSPC, "private image storage headroom unavailable")

    def stage(self):
        self.headroom()
        return StagedImage(self)

    @staticmethod
    def key(scope, root, upload_id):
        if not UPLOAD_ID.fullmatch(upload_id):
            raise ImageRejected("invalid upload identity")
        return hashlib.sha256(_canonical([scope, root, upload_id])).hexdigest()

    def prepare(self, stage, scope, root, upload_id, content_type):
        """Decode a fully bounded staged upload, without trusting MIME/magic."""
        key = self.key(scope, root, upload_id)
        stage.seal()
        with _file("original", parent=stage.fd) as original:
            digest, size = _digest_file(original, self.limits.max_original_bytes)
        try:
            with self._record(key) as record:
                previous = self._manifest(record, scope, root)
                self._verify_record(record, previous)
                if (previous["original"]["sha256"] != digest
                        or previous["original"]["content_type"] != content_type
                        or previous["original"]["byte_size"] != size):
                    raise ImageConflict("upload identity already used")
                return previous
        except FileNotFoundError:
            pass
        if content_type not in MIME_FORMATS:
            raise ImageRejected("unsupported image type")
        self.headroom()
        try:
            with _file("original", parent=stage.fd) as original:
                digest, size = _digest_file(original, self.limits.max_original_bytes)
                with Image.open(original) as encoded:
                    width, height = encoded.size
                    if (encoded.format != MIME_FORMATS[content_type]
                            or not 1 <= width <= self.limits.max_dimension
                            or not 1 <= height <= self.limits.max_dimension
                            or width * height > self.limits.max_decoded_pixels
                            or getattr(encoded, "n_frames", 1) != 1):
                        raise ImageRejected("unsupported image encoding")
                    encoded.verify()
                original.seek(0)
                with Image.open(original) as decoded:
                    decoded.load()  # Reject truncated pixel payloads, not just headers.
                    oriented = ImageOps.exif_transpose(decoded)
                    try:
                        oriented.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                        # A fresh pixel image deliberately carries no metadata,
                        # ICC profile, comment, EXIF or XMP. Flatten alpha to white.
                        pixels = Image.new("RGB", oriented.size, "white")
                        if "A" in oriented.getbands():
                            rgba = oriented.convert("RGBA")
                            try:
                                pixels.paste(rgba, mask=rgba.getchannel("A"))
                            finally:
                                rgba.close()
                        else:
                            pixels.paste(oriented.convert("RGB"))
                        try:
                            preview_width, preview_height = pixels.size
                            with _file("preview", parent=stage.fd, write=True) as preview:
                                pixels.save(preview, format="JPEG", quality=85)
                                preview.flush()
                                os.fsync(preview.fileno())
                        finally:
                            pixels.close()
                    finally:
                        oriented.close()
            with _file("preview", parent=stage.fd) as preview:
                preview_digest, preview_size = _digest_file(preview, PREVIEW_MAX_BYTES)
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError,
                SyntaxError, EOFError, ValueError) as exc:
            if isinstance(exc, OSError) and exc.errno is not None:
                raise
            raise ImageRejected("invalid image bytes") from exc
        descriptor = {
            "schema_version": "1.0", "image_id": "im_" + secrets.token_hex(32 // 2) + "_" + key,
            "conversation_id": root, "upload_id": upload_id,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "retained": True, "admission_state": "unavailable",
            "original": {"content_type": content_type, "byte_size": size, "sha256": digest,
                         "width": width, "height": height},
            "preview": {"content_type": "image/jpeg", "byte_size": preview_size,
                        "sha256": preview_digest, "width": preview_width, "height": preview_height},
        }
        with _file("manifest", parent=stage.fd, write=True) as manifest:
            manifest.write(_canonical({"scope": scope, "descriptor": descriptor}))
            manifest.flush()
            os.fsync(manifest.fileno())
        os.fsync(stage.fd)
        stage.ready = True
        return descriptor

    def publish(self, stage, scope, descriptor):
        key = IMAGE_ID.fullmatch(descriptor["image_id"])[1]
        try:
            # An existing complete record is also the restart/lost-ACK path.
            with self._record(key) as record:
                previous = self._manifest(record, scope, descriptor["conversation_id"])
                self._verify_record(record, previous)
                if (previous["upload_id"] != descriptor["upload_id"]
                        or previous["original"] != descriptor["original"]):
                    raise ImageConflict("upload identity already used")
            os.fsync(self.fd)
            return previous, False
        except FileNotFoundError:
            pass
        if not stage.ready:
            raise ImageUnavailable("retained upload disappeared")
        try:
            self.headroom()
            os.rename(stage.name, key, src_dir_fd=self.fd, dst_dir_fd=self.fd)
            stage.published = True
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            with self._record(key) as record:
                previous = self._manifest(record, scope, descriptor["conversation_id"])
                self._verify_record(record, previous)
                if (previous["upload_id"] != descriptor["upload_id"]
                        or previous["original"] != descriptor["original"]):
                    raise ImageConflict("upload identity already used") from None
            os.fsync(self.fd)
            return previous, False
        os.fsync(self.fd)
        return descriptor, True

    def _record(self, key):
        from contextlib import contextmanager

        @contextmanager
        def opened():
            fd = _dir(key, parent=self.fd)
            try:
                yield fd
            finally:
                os.close(fd)
        return opened()

    def _manifest(self, fd, scope, root):
        with _file("manifest", parent=fd) as file:
            raw = file.read(16385)
        if len(raw) > 16384:
            raise ImageUnavailable("private image unavailable")
        try:
            value = _closed_json(raw)
            if set(value) != {"scope", "descriptor"} or value["scope"] != scope:
                raise ValueError()
            descriptor = value["descriptor"]
            if (set(descriptor) != {"schema_version", "image_id", "conversation_id", "upload_id",
                                  "created_at", "retained", "admission_state", "original", "preview"}
                    or descriptor["schema_version"] != "1.0" or descriptor["conversation_id"] != root
                    or not IMAGE_ID.fullmatch(descriptor["image_id"])
                    or descriptor["retained"] is not True or descriptor["admission_state"] != "unavailable"
                    or self.key(scope, root, descriptor["upload_id"]) != IMAGE_ID.fullmatch(descriptor["image_id"])[1]):
                raise ValueError()
            for variant in ("original", "preview"):
                info = descriptor[variant]
                cap = 20 * 1024 * 1024 if variant == "original" else PREVIEW_MAX_BYTES
                if (set(info) != {"content_type", "byte_size", "sha256", "width", "height"}
                        or info["content_type"] not in MIME_FORMATS
                        or (variant == "preview" and info["content_type"] != "image/jpeg")
                        or not re.fullmatch(r"[0-9a-f]{64}", info["sha256"])
                        or type(info["byte_size"]) is not int or not 1 <= info["byte_size"] <= cap
                        or any(type(info[k]) is not int or not 1 <= info[k] <= (2048 if variant == "preview" else 16384) for k in ("width", "height"))
                        or info["width"] * info["height"] > 40_000_000):
                    raise ValueError()
            return descriptor
        except (ValueError, TypeError, KeyError, RecursionError):
            raise ImageUnavailable("private image unavailable") from None

    @staticmethod
    def _verify_record(fd, descriptor):
        for variant in ("original", "preview"):
            expected = descriptor[variant]
            with _file(variant, parent=fd) as file:
                digest, size = _digest_file(file, expected["byte_size"])
            if digest != expected["sha256"] or size != expected["byte_size"]:
                raise ImageUnavailable("private image unavailable")

    def read(self, scope, root, image_id, variant=None):
        match = IMAGE_ID.fullmatch(image_id)
        if not match or variant not in (None, "original", "preview"):
            raise ImageUnavailable("private image unavailable")
        try:
            with self._record(match[1]) as fd:
                descriptor = self._manifest(fd, scope, root)
                if descriptor["image_id"] != image_id:
                    raise ImageUnavailable("private image unavailable")
                if variant is None:
                    self._verify_record(fd, descriptor)
                    return descriptor, None
                info = descriptor[variant]
                with _file(variant, parent=fd) as file:
                    data = file.read(info["byte_size"] + 1)
                if len(data) != info["byte_size"] or hashlib.sha256(data).hexdigest() != info["sha256"]:
                    raise ImageUnavailable("private image unavailable")
                return descriptor, data
        except (OSError, ValueError):
            raise ImageUnavailable("private image unavailable") from None
