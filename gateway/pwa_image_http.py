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
        return self.owner_scope(principal, root)

    def owner_scope(self, principal, root):
        projection, grant = self.parent.ingress._authorize(principal, root)
        if projection["conversation_id"] != root:
            raise PermissionError("canonical image conversation required")
        if not self.parent.runner._is_user_authorized_for_source(grant.source):
            raise PermissionError("image source unavailable")
        return [*self.parent.ownership.scope(principal),
                self.parent.db._own_profile_name(), source_identity(grant.source)]

    def image_store(self):
        if self.store is None:
            self.store = PrivateImageStore(get_hermes_home() / "pwa-images", self.parent.config.images)
        return self.store

    def admission_state(self):
        models = self.parent.models.config
        return ("requires_model_check" if models is not None
                and not self.parent.runner._get_proxy_url()
                and any(route.capabilities.image_input == "supported" for route in models.entries)
                else "unavailable")

    def descriptor(self, descriptor):
        return {**descriptor, "admission_state": self.admission_state()}

    async def prepare_command(self, principal, body, *, previous=None):
        import json
        from gateway.pwa_image_policy import SIDECAR, prepare_images, require_image_route
        from gateway.pwa_http import RequestError
        from hermes_state_commands import CommandConflict

        if self.workers >= self.parent.config.images.workers:
            raise RequestError(429, "native_unavailable")
        self.workers += 1
        try:
            root = body["conversation_id"]
            scope = self.owner_scope(principal, root)
            if previous is None:
                selected = self.parent.db.native_model_selection(root)
                if not selected or selected["model_version"] != body["expected_model_version"]:
                    raise CommandConflict("model version changed")
                require_image_route(self.parent.models, selected["model_id"])
            frozen = json.loads(previous["payload_json"]).get(SIDECAR) if previous else None
            if previous is not None and frozen is None:
                raise ImageRejected("retained image admission unavailable")
            result = await completed_thread(
                lambda: prepare_images(self.image_store(), scope, root,
                                       body["payload"]["image_ids"], expected=frozen))
            if self.owner_scope(principal, root) != scope:
                raise PermissionError("image owner changed")
            return result
        finally:
            self.workers -= 1

    async def handle(self, request, principal, parts):
        from gateway.pwa_http import RequestError

        self.parent._query(request)
        if request.method == "GET" and parts == ["images", "policy"]:
            from gateway.pwa_image_policy import (
                MAX_COMMAND_PIXEL_BYTES, MAX_CONTEXT_SOURCE_BYTES, MAX_REQUEST_BODY_BYTES,
                MAX_REQUEST_PIXEL_BYTES, PIXEL_POLICY,
            )
            state = self.admission_state()
            return {**self.parent.config.images.policy(), "admission_state": state,
                    "actions": {"send": state, "queue": state, "redirect": state, "steer": "unavailable"},
                    "active_send": "queue", "model_representation": PIXEL_POLICY,
                    "max_command_pixel_bytes": MAX_COMMAND_PIXEL_BYTES,
                    "max_context_source_bytes": MAX_CONTEXT_SOURCE_BYTES,
                    "max_request_pixel_bytes": MAX_REQUEST_PIXEL_BYTES,
                    "max_request_body_bytes": MAX_REQUEST_BODY_BYTES,
                    "metadata_disclosure": "withheld"}, 200
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
            self.image_store()
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
                return self.descriptor(descriptor), 201 if created else 200
            variant = parts[4] if len(parts) == 5 else None
            descriptor, data = await completed_thread(self.store.read, scope, root, parts[3], variant)
            if self.scope(request, principal, root) != scope:
                raise PermissionError("image owner changed")
            if variant is None:
                return self.descriptor(descriptor), 200
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
