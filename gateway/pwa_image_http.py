"""Bounded private image HTTP inside the existing authenticated native listener."""
from __future__ import annotations

import asyncio

from aiohttp import web

from gateway.pwa_config import source_identity
from gateway.pwa_images import ImageConflict, ImageRejected, PrivateImageStore, UPLOAD_ID
from hermes_constants import get_hermes_home


async def completed_thread(function, *args):
    """Keep the worker slot/stage alive if a request times out during codec I/O."""
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
            # A second shutdown/request cancellation must not cancel the task
            # tracking the still-running executor thread or release its stage.
            if task.cancelled():
                raise
        except BaseException:
            if cancelled:
                raise asyncio.CancelledError() from None
            raise
        else:
            if cancelled:
                raise asyncio.CancelledError()
            return result



class NativeImageHttp:
    def __init__(self, parent):
        self.parent = parent
        self.store = None
        self.workers = 0

    def close(self):
        if self.store is not None:
            self.store.close()
            self.store = None

    def scope(self, request, principal, root):
        if self.parent._authenticate(request) != principal:
            raise PermissionError("image owner unavailable")
        projection, grant = self.parent.ingress._authorize(principal, root)
        if projection["conversation_id"] != root:
            raise PermissionError("canonical image conversation required")
        return [*self.parent.ownership.scope(principal),
                self.parent.db._own_profile_name(), source_identity(grant.source)]

    async def handle(self, request, principal, parts):
        from gateway.pwa_http import RequestError

        self.parent._query(request)
        if request.method == "GET" and parts == ["images", "policy"]:
            return self.parent.config.images.policy(), 200
        if (len(parts) not in (3, 4, 5) or parts[0] != "conversations" or parts[2] != "images"
                or request.method not in ("GET", "POST")
                or (request.method == "POST" and len(parts) != 3)
                or (request.method == "GET" and len(parts) == 3)
                or (len(parts) == 5 and parts[4] not in ("original", "preview"))):
            raise RequestError(404, "capability_unavailable")
        if self.workers >= self.parent.config.images.workers:
            raise RequestError(429, "native_unavailable")
        self.workers += 1
        stage = None
        try:
            root = parts[1]
            scope = self.scope(request, principal, root)
            if self.store is None:
                self.store = PrivateImageStore(get_hermes_home() / "pwa-images", self.parent.config.images)
            if request.method == "POST":
                for name in ("Content-Type", "X-Hermes-PWA-Upload-Id"):
                    if len(request.headers.getall(name, [])) != 1:
                        raise RequestError(400, "invalid_request")
                upload_id = request.headers["X-Hermes-PWA-Upload-Id"]
                if not UPLOAD_ID.fullmatch(upload_id):
                    raise RequestError(400, "invalid_request")
                limit = self.parent.config.images.max_original_bytes
                if request.content_length is not None and request.content_length > limit:
                    raise RequestError(413, "invalid_request")
                stage = self.store.stage()
                async for chunk in request.content.iter_chunked(65536):
                    if stage.size + len(chunk) > limit:
                        raise RequestError(413, "invalid_request")
                    await completed_thread(stage.write, chunk)
                if not stage.size:
                    raise RequestError(400, "invalid_request")
                descriptor = await completed_thread(self.store.prepare, stage, scope, root,
                                                    upload_id, request.headers["Content-Type"])
                if self.scope(request, principal, root) != scope:
                    raise PermissionError("image owner changed")
                # No await between final native authority check and publication.
                descriptor, created = self.store.publish(stage, scope, descriptor)
                return descriptor, 201 if created else 200
            variant = parts[4] if len(parts) == 5 else None
            descriptor, data = await completed_thread(self.store.read, scope, root, parts[3], variant)
            if self.scope(request, principal, root) != scope:
                raise PermissionError("image owner changed")
            if variant is None:
                return descriptor, 200
            response = web.StreamResponse(headers={
                "Content-Type": descriptor[variant]["content_type"],
                "Content-Length": str(len(data)), "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff", "Content-Disposition": "attachment",
            })
            try:
                await response.prepare(request)
                await response.write(data)
                await response.write_eof()
            except (Exception, asyncio.CancelledError):
                response.force_close()
                if request.transport is not None:
                    request.transport.close()
            return response, 200
        except ImageConflict:
            raise RequestError(409, "conflict") from None
        except ImageRejected:
            raise RequestError(400, "invalid_request") from None
        finally:
            try:
                if stage is not None:
                    stage.close()
            finally:
                self.workers -= 1
