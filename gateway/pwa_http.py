"""Restricted private facade transport inside the existing GatewayRunner."""

from __future__ import annotations

import asyncio
import base64
import hmac
import json
import logging
import os
import re
import secrets
import ssl
import time
from collections import OrderedDict

from aiohttp import web

from gateway.conversation_control import NativeConversationIngress
from gateway.pwa_config import (
    PwaHttpConfig,
    canonical,
    closed,
    identifier,
    parse_principal,
)
from gateway.pwa_ownership import NativePwaOwnership
from gateway.pwa_image_http import NativeImageHttp
from gateway.pwa_history_scan import NativeHistoryScan
from gateway.pwa_models import (
    ModelCatalogUnavailable,
    ModelRouteUnavailable,
    NativeModelCatalog,
)
from gateway.session import ChannelBindingConflict
from gateway.config import Platform
from gateway.telegram_conversations import TelegramConversationChannel, channel_key
from gateway.native_clarification import ClarificationRecoveryGap
from hermes_state_commands import CommandConflict

CONTRACT_PIN = "5b6242a8dc036b7b81a40e9fafd8afb5788ef2a5"


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result
    def constant(_):
        raise ValueError("nonfinite JSON number")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


class RequestError(Exception):
    def __init__(self, status, code):
        self.status, self.code = status, code


class NativePwaHttp:
    def __init__(self, runner):
        self.runner = runner
        self.config = PwaHttpConfig.from_dict(runner.config.pwa_http)
        if not self.config.enabled or runner.config.multiplex_profiles:
            raise ValueError("native PWA listener is not enabled for this profile")
        token = os.environ.get(self.config.token_env, "")
        if (
            len(token.encode()) < 32
            or len(token) > 4096
            or any(ord(c) < 33 or ord(c) > 126 for c in token)
        ):
            raise ValueError("native PWA facade credential unavailable")
        self._token = token
        self.db = runner.session_store._db
        if self.db is None:
            raise RuntimeError("native PWA SessionDB unavailable")
        self.ownership = NativePwaOwnership(runner, self.db, self.config, secret=token)
        self.models = NativeModelCatalog(
            self.config.models, runner._resolve_managed_model_provider
        )
        self.ingress = NativeConversationIngress(
            runner,
            self.db,
            {},
            concierge_id=self.config.concierge_id,
            grant_provider=self.ownership,
            event_limits=self.config.event_limits,
            model_catalog=self.models,
        )
        self.telegram_channel = TelegramConversationChannel(self.ingress)
        self.history_scan = NativeHistoryScan(self)
        self.images = NativeImageHttp(self)
        self._streams = set()
        self._close_event = asyncio.Event()
        self._http_runner = None
        self._site = None
        self._closing = False
        self._requests = 0
        self._cursors = OrderedDict()
        self.port = None

    def _error(self, status, code):
        messages = {"authentication_required": "Facade authentication required.",
                    "authorization_denied": "Resource unavailable.", "invalid_request": "Invalid native request.",
                    "conflict": "Native request conflicts with current state.",
                    "recovery_gap": "Question traversal is no longer available; recover current state.",
                    "native_unavailable": "Native service unavailable.",
                    "capability_unavailable": "Native capability unavailable.", "internal_error": "Native request failed."}
        return web.json_response({"schema_version": "1.0", "error": {"code": code,
            "message": messages[code], "retryable": status in (429, 503)}}, status=status,
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})

    def _authenticate(self, request):
        for header in ("Authorization", "X-Hermes-PWA-Principal", "Host"):
            if len(request.headers.getall(header, [])) != 1:
                raise RequestError(401 if header == "Authorization" else 400,
                    "authentication_required" if header == "Authorization" else "invalid_request")
        authorization = request.headers["Authorization"]
        if not hmac.compare_digest(authorization.encode(), ("Bearer " + self._token).encode()):
            raise RequestError(401, "authentication_required")
        if request.headers["Host"].lower() not in self.config.allowed_hosts:
            raise RequestError(400, "invalid_request")
        if "Origin" in request.headers or "Cookie" in request.headers or "Content-Encoding" in request.headers:
            raise RequestError(400, "invalid_request")
        raw = request.headers["X-Hermes-PWA-Principal"]
        if len(raw) > 4096:
            raise RequestError(400, "invalid_request")
        try:
            decoded = base64.b64decode(raw + "=" * (-len(raw) % 4), altchars=b"-_", validate=True)
            if base64.urlsafe_b64encode(decoded).decode().rstrip("=") != raw:
                raise ValueError("noncanonical principal")
            principal, concierge = parse_principal(strict_json(decoded.decode("utf-8")))
        except (ValueError, TypeError, UnicodeError, RecursionError):
            raise RequestError(400, "invalid_request") from None
        if concierge != self.ownership.config.concierge_id:
            raise RequestError(403, "authorization_denied")
        try:
            self.ownership.binding(principal)
        except PermissionError:
            raise RequestError(403, "authorization_denied") from None
        return principal

    def _query(self, request, allowed=()):
        if set(request.query) - set(allowed) or any(len(request.query.getall(k)) != 1 for k in request.query):
            raise ValueError("invalid native query")
        if request.method == "GET" and (request.can_read_body or request.content_length):
            raise ValueError("GET body unavailable")
        return request.query

    async def _body(self, request):
        if request.content_type != "application/json":
            raise ValueError("expected JSON")
        return strict_json((await request.read()).decode("utf-8"))

    def _page(self, request):
        query = self._query(request, ("limit", "cursor"))
        value = query.get("limit", "50")
        if not value.isascii() or not value.isdecimal() or not 1 <= int(value) <= 100:
            raise ValueError("invalid page size")
        return int(value), query.get("cursor")

    def _scope(self, principal, query, limit):
        return (principal, self.ownership.config.fingerprint, query, limit)

    def _model_scope(self, principal):
        return canonical([
            "model",
            self.config.concierge_id,
            principal.issuer,
            principal.subject,
        ])

    def _cursor(self, token, scope):
        entry = self._cursors.get(token)
        if entry is None or entry[0] < time.monotonic() or entry[1] != scope:
            raise CommandConflict("invalid or expired native cursor")
        self._cursors.move_to_end(token)
        return entry[2]

    def _new_cursor(self, state, scope):
        now = time.monotonic()
        for token in tuple(self._cursors):
            if self._cursors[token][0] < now:
                self._cursors.pop(token)
        while len(self._cursors) >= 4096:
            self._cursors.popitem(last=False)
        token = secrets.token_urlsafe(32)
        self._cursors[token] = (now + self.config.cursor_ttl, scope, state)
        return token

    async def _history(self, request, principal, root):
        limit, cursor = self._page(request)
        projection, _ = await asyncio.to_thread(self.ingress._authorize, principal, root)
        lineage = projection["native_session_ids"]
        scope = self._scope(principal, "history:" + projection["conversation_id"], limit)
        signature, upper = await asyncio.to_thread(self.db.native_pwa_history_signature, lineage)
        state = self._cursor(cursor, scope) if cursor else {"signature": signature, "segment": 0, "after": 0}
        if state["signature"] != signature:
            raise CommandConflict("native history changed")
        segment, after, messages = state["segment"], state["after"], []
        size = len(canonical(lineage).encode()) + 4096
        partial = False
        while segment < len(lineage) and len(messages) < limit:
            rows = await asyncio.to_thread(self.db.native_pwa_history_rows, lineage[segment], after, upper[segment], limit-len(messages)+1)
            if not rows:
                segment, after = segment + 1, 0
                continue
            paused = False
            for row in rows:
                if row["session_id"] != lineage[segment]:
                    raise RuntimeError("native history lineage mismatch")
                message = self.ownership.message(row)
                length = len(canonical(message).encode())
                if not messages and size + length > self.config.max_response_bytes:
                    message.update(content=None, content_state="omitted", omission_reason="oversized")
                    length = len(canonical(message).encode())
                if len(messages) == limit or size + length > self.config.max_response_bytes:
                    paused = True
                    break
                messages.append(message)
                size += length
                after = row["id"]
                partial |= message["content_state"] != "available"
            if paused:
                break
        final_projection, _ = await asyncio.to_thread(self.ingress._authorize, principal, root)
        if final_projection["native_session_ids"] != lineage:
            raise CommandConflict("native history lineage changed")
        current, _ = await asyncio.to_thread(self.db.native_pwa_history_signature, lineage)
        if current != signature:
            raise CommandConflict("native history changed")
        next_cursor = None if segment >= len(lineage) else self._new_cursor(
            {"signature": signature, "segment": segment, "after": after}, scope)
        return {"schema_version": "1.0", "conversation_id": projection["conversation_id"],
                "native_session_ids": lineage, "messages": messages, "next_cursor": next_cursor,
                "history_state": "partial" if partial else "available"}

    def _authorize_root(self, principal, root):
        projection, grant = self.ingress._authorize(principal, root)
        if projection["conversation_id"] != root:
            raise PermissionError("canonical native root required")
        if not self.runner._is_user_authorized_for_source(grant.source):
            raise PermissionError("native source is not authorized")
        return projection, grant

    def _event_cursor(self, request, *, stream=False):
        query = self._query(request, ("cursor",))
        headers = request.headers.getall("Last-Event-ID", [])
        if headers and (not stream or len(headers) != 1 or "cursor" in query):
            raise ValueError("ambiguous native event cursor")
        raw = query.get("cursor", headers[0] if headers else None)
        if raw is None:
            return None
        if not raw or len(raw) > 2048:
            raise ValueError("invalid native event cursor")
        decoded = base64.b64decode(
            raw + "=" * (-len(raw) % 4), altchars=b"-_", validate=True
        )
        if base64.urlsafe_b64encode(decoded).decode().rstrip("=") != raw:
            raise ValueError("noncanonical native event cursor")
        value = strict_json(decoded.decode("utf-8"))
        if canonical(value).encode() != decoded:
            raise ValueError("noncanonical native event cursor")
        # Valid JSON with an invalid semantic position is a native gap, not an
        # HTTP syntax error. Null must not accidentally become initial capture.
        return {} if value is None else value

    async def _binding(self, request, principal, root):
        self._query(request)
        _, grant = self._authorize_root(principal, root)
        if grant.source.platform != Platform.TELEGRAM:
            raise RequestError(404, "capability_unavailable")
        if request.method == "PUT":
            body = await self._body(request)
            closed(
                body,
                {"schema_version", "expected_binding_version"},
                {"schema_version", "expected_binding_version"},
            )
            if body["schema_version"] != "1.0":
                raise ValueError("invalid binding version")
            binding = await self.telegram_channel.select(
                principal, root, body["expected_binding_version"]
            )
        else:
            entry = self.runner.session_store.lookup_by_session_key(
                self.runner._session_key_for_source(grant.source)
            )
            if entry is None:
                return dict(
                    schema_version="1.0",
                    conversation_id=root,
                    channel="telegram",
                    state="unbound",
                    selected_conversation_id=None,
                    native_session_id=None,
                    binding_version=0,
                )
            binding = await self.telegram_channel.inspect(principal, root)
        return dict(
            schema_version="1.0",
            conversation_id=root,
            channel="telegram",
            state="selected",
            selected_conversation_id=binding["conversation_id"],
            native_session_id=binding["native_session_id"],
            binding_version=binding["binding_version"],
        )

    async def _model(self, request, principal, root):
        self._query(request)
        self._authorize_root(principal, root)
        if self.config.models is None:
            raise RequestError(404, "capability_unavailable")
        if request.method == "GET":
            selection = await asyncio.to_thread(self.db.native_model_selection, root)
            if selection is None:
                try:
                    self.models.resolve_default()
                except ModelRouteUnavailable:
                    raise RequestError(503, "native_unavailable") from None
                selection = await asyncio.to_thread(
                    self.db.native_model_initialize,
                    root,
                    self.config.models.default_model_id,
                )
            self._authorize_root(principal, root)
            return {"schema_version": "1.0", **selection}

        body = await self._body(request)
        closed(
            body,
            {
                "schema_version",
                "mutation_id",
                "model_id",
                "expected_model_version",
            },
            {
                "schema_version",
                "mutation_id",
                "model_id",
                "expected_model_version",
            },
        )
        if body["schema_version"] != "1.0":
            raise ValueError("invalid model mutation version")
        identifier(body["mutation_id"])
        model_id = identifier(body["model_id"])
        if len(model_id) > 200:
            raise ValueError("invalid model id")
        expected = body["expected_model_version"]
        if type(expected) is not int or not 0 <= expected <= 9007199254740991:
            raise ValueError("invalid expected model version")
        stored = {
            **body,
            "conversation_id": root,
        }
        scope = self._model_scope(principal)
        result = await asyncio.to_thread(
            self.db.native_model_mutation_replay, scope, stored
        )
        if result is None:
            try:
                self.models.resolve(model_id)
            except LookupError:
                raise RequestError(409, "conflict") from None
            except ModelRouteUnavailable:
                raise RequestError(503, "native_unavailable") from None
            result = await asyncio.to_thread(
                self.db.native_model_mutate, scope, stored
            )
        self._authorize_root(principal, root)
        return result

    async def _events(self, request, principal, root):
        deadline = time.monotonic() + self.config.stream_max_seconds

        def budget(maximum):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("native stream lifetime expired")
            return min(maximum, remaining)

        if self._closing or not self.runner._running or self.runner._draining:
            raise RequestError(503, "native_unavailable")
        cursor = self._event_cursor(request, stream=True)
        self._authorize_root(principal, root)
        feed = self.ingress.events
        if len(feed._subscriptions) >= self.db._native_events_limits.max_subscribers:
            raise RequestError(429, "native_unavailable")
        subscription = feed.subscribe(principal, root, cursor)
        task = asyncio.current_task()
        self._streams.add(task)
        response = None
        try:
            async with asyncio.timeout(budget(self.config.request_timeout)):
                result = await subscription.poll()
            while not self._closing:
                data = canonical(result).encode()
                if len(data) > self.config.max_response_bytes:
                    raise RequestError(503, "native_unavailable")
                event_id = base64.urlsafe_b64encode(
                    canonical(result["cursor"]).encode()
                ).rstrip(b"=")
                frame = (
                    b"event: recovery\nid: " + event_id + b"\ndata: " + data + b"\n\n"
                )
                if len(frame) > self.config.max_response_bytes:
                    raise RequestError(503, "native_unavailable")
                # No awaited work between current authority and each publication.
                self._authenticate(request)
                self._authorize_root(principal, root)
                if response is None:
                    response = web.StreamResponse(
                        headers={
                            "Content-Type": "text/event-stream",
                            "Cache-Control": "no-store",
                            "X-Content-Type-Options": "nosniff",
                            "X-Accel-Buffering": "no",
                        }
                    )
                    async with asyncio.timeout(budget(self.config.stream_write_timeout)):
                        await response.prepare(request)
                async with asyncio.timeout(budget(self.config.stream_write_timeout)):
                    self._authenticate(request)
                    self._authorize_root(principal, root)
                    await response.write(frame)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    await asyncio.wait_for(
                        self._close_event.wait(),
                        min(remaining, self.config.event_poll_interval),
                    )
                    break
                except TimeoutError:
                    pass
                if (
                    time.monotonic() >= deadline
                    or request.transport is None
                    or request.transport.is_closing()
                ):
                    break
                async with asyncio.timeout(
                    budget(self.config.request_timeout)
                ):
                    result = await subscription.poll()
            if response is not None:
                async with asyncio.timeout(budget(self.config.stream_write_timeout)):
                    await response.write_eof()
            return response
        except Exception:
            if response is None or not response.prepared:
                raise
            # Headers already committed: close, never fabricate success/domain
            # events or expose exception text. Client reconnects with its cursor.
            response.force_close()
            # Do not leave aiohttp's final EOF drain outside our write/lifetime
            # budget after a slow or interrupted stream.
            if request.transport is not None:
                request.transport.close()
            return response
        finally:
            subscription.close()
            self._streams.discard(task)

    async def _dispatch(self, request, principal):
        tail = request.path.removeprefix("/v1/pwa/")
        if request.method == "GET" and tail == "health":
            self._query(request)
            if not self.runner._running and not self._closing:
                raise RequestError(503, "native_unavailable")
            return {
                "schema_version": "1.0",
                "service": "native_pwa",
                "state": "draining"
                if self._closing or self.runner._draining
                else "ready",
            }, 200
        if self._closing or not self.runner._running or self.runner._draining:
            raise RequestError(503, "native_unavailable")
        parts = tail.split("/")
        if parts == ["images", "policy"] or (len(parts) >= 3 and parts[0] == "conversations" and parts[2] == "images"):
            return await self.images.handle(request, principal, parts)
        if request.method == "GET" and tail == "capabilities":
            self._query(request)
            capabilities = [
                {"capability_id": name, "availability": "available"}
                for name in (
                    "command_admission",
                    "durable_commands",
                    "history",
                    "owner_history_search",
                    "history_sync",
                    "event_stream",
                    "event_replay",
                    "snapshot_recovery",
                )
            ]
            if self.runner._get_proxy_url():
                for capability in capabilities[:2]:
                    capability.update(
                        availability="unavailable",
                        reason="Local native execution is required.",
                    )
            sources = self.ownership.binding(principal).sources
            telegram = any(
                s.source.platform == Platform.TELEGRAM
                and self.runner._is_user_authorized_for_source(s.source)
                for s in sources
            )
            for name in ("telegram_binding", "telegram_final_delivery"):
                available = telegram and (
                    name == "telegram_binding"
                    or bool(self.runner.adapters.get(Platform.TELEGRAM))
                    and not self.runner._get_proxy_url()
                )
                capabilities.append({
                    "capability_id": name,
                    "availability": "available" if available else "unavailable",
                    **(
                        {}
                        if available
                        else {
                            "reason": "Configured native Telegram channel unavailable."
                        }
                    ),
                })
            local_control = not self.runner._get_proxy_url()
            capabilities.append({
                "capability_id": "remote_cancellation",
                "availability": "available" if local_control else "unavailable",
                **({} if local_control else {"reason": "Local native execution is required."}),
            })
            capabilities.extend([
                {"capability_id": kind, "availability": "available" if local_control else "unavailable",
                 **({} if local_control else {"reason": "Local native execution is required."})}
                for kind in ("redirect", "native_clarification")
            ])
            capabilities.append({"capability_id": "remote_clarification", "availability": "unavailable",
                                 "reason": "No configured observed specialist question protocol adapter."})
            model_availability = (
                "available" if self.config.models is not None else "unavailable"
            )
            for name in ("model_catalog", "conversation_model_selection"):
                capabilities.append({
                    "capability_id": name,
                    "availability": model_availability,
                    **(
                        {}
                        if model_availability == "available"
                        else {"reason": "Managed model catalog is not configured."}
                    ),
                })
            return {"schema_version": "1.0", "capabilities": capabilities}, 200
        if request.method == "GET" and tail == "models":
            self._query(request)
            try:
                return self.models.catalog(), 200
            except ModelCatalogUnavailable:
                raise RequestError(404, "capability_unavailable") from None
        if tail in {"search", "sync"} and request.method == "GET":
            return await self.history_scan.request(request, principal, tail), 200
        if tail == "conversations" and request.method == "GET":
            limit, cursor = self._page(request)
            scope = self._scope(principal, "conversations", limit)
            after = self._cursor(cursor, scope) if cursor else ""
            roots, next_after, coverage = await asyncio.to_thread(
                self.ownership.discover, principal, after, limit
            )
            conversations = [
                await self.ingress.native_conversation(principal, root)
                for root in roots
            ]
            return {
                "schema_version": "1.0",
                "conversations": conversations,
                "coverage": coverage,
                "next_cursor": self._new_cursor(next_after, scope)
                if next_after is not None
                else None,
            }, 200
        if tail == "conversations" and request.method == "POST":
            self._query(request)
            body = await self._body(request)
            closed(
                body, {"schema_version", "create_id"}, {"schema_version", "create_id"}
            )
            if body["schema_version"] != "1.0":
                raise ValueError("invalid create version")
            result, created = await self.ownership.create(
                self.ingress, principal, identifier(body["create_id"])
            )
            return result, 201 if created else 200
        parts = tail.split("/")
        if len(parts) >= 3 and parts[0] == "conversations":
            root = identifier(parts[1])
            if (
                len(parts) == 3
                and parts[2] == "telegram-binding"
                and request.method in {"GET", "PUT"}
            ):
                return await self._binding(request, principal, root), 200
            if request.method == "GET" and len(parts) in (3, 4) and parts[2] == "clarifications":
                if len(parts) == 4:
                    self._query(request)
                    return await self.ingress.clarifications.page(principal, root, identity=identifier(parts[3])), 200
                query = self._query(request, ("cursor", "limit"))
                raw_limit = query.get("limit", "50")
                if not re.fullmatch(r"[1-9][0-9]{0,2}", raw_limit) or int(raw_limit) > 100:
                    raise ValueError("invalid question page limit")
                return await self.ingress.clarifications.page(principal, root,
                    cursor=query.get("cursor"), limit=int(raw_limit)), 200
            if request.method == "GET" and len(parts) == 3 and parts[2] == "recovery":
                cursor = self._event_cursor(request)
                self._authorize_root(principal, root)
                return await self.ingress.events.recover(principal, root, cursor), 200
            if request.method == "GET" and len(parts) == 4 and parts[2] == "executions":
                self._query(request)
                self._authorize_root(principal, root)
                return await self.ingress.events.execution(
                    principal, root, identifier(parts[3])
                ), 200
            if (
                len(parts) == 3
                and parts[2] == "model"
                and request.method in {"GET", "PUT"}
            ):
                return await self._model(request, principal, root), 200
        if (
            request.method == "GET"
            and len(parts) in (2, 3)
            and parts[0] == "conversations"
        ):
            root = identifier(parts[1])
            if len(parts) == 3 and parts[2] == "history":
                return await self._history(request, principal, root), 200
            if len(parts) == 2:
                self._query(request)
                return await self.ingress.native_conversation(principal, root), 200
        if tail == "commands" and request.method == "POST":
            self._query(request)
            return await self.ingress.submit(principal, await self._body(request)), 200
        if request.method == "GET" and len(parts) == 2 and parts[0] == "commands":
            query = self._query(request, ("remote_cursor", "remote_limit"))
            raw_limit = query.get("remote_limit", "50")
            if not re.fullmatch(r"[1-9][0-9]{0,2}", raw_limit) or int(raw_limit) > 100:
                raise ValueError("invalid remote page limit")
            result = await self.ingress.command_receipt(
                principal, identifier(parts[1]),
                remote_cursor=identifier(query["remote_cursor"]) if "remote_cursor" in query else None,
                remote_limit=int(raw_limit),
            )
            if query and result.get("receipt_kind") not in {"cancel", "redirect"}:
                raise ValueError("remote pagination requires a cancellation receipt")
            return result, 200
        raise RequestError(404, "capability_unavailable")

    async def handle(self, request):
        admitted = False
        try:
            principal = self._authenticate(request)
            parts = request.path.split("/")
            if (
                request.method == "GET"
                and len(parts) == 6
                and parts[:4] == ["", "v1", "pwa", "conversations"]
                and parts[5] == "events"
            ):
                return await self._events(request, principal, identifier(parts[4]))
            if self._requests >= self.config.max_requests:
                raise RequestError(429, "native_unavailable")
            self._requests += 1
            admitted = True
            async with asyncio.timeout(self.config.request_timeout):
                result, status = await self._dispatch(request, principal)
            if (
                len(parts) >= 6
                and parts[:4] == ["", "v1", "pwa", "conversations"]
                and parts[5] in {"recovery", "executions", "telegram-binding", "clarifications"}
            ):
                self._authenticate(request)
                self._authorize_root(principal, identifier(parts[4]))
                if parts[5] == "telegram-binding" and result["state"] == "selected":
                    self._authorize_root(principal, result["selected_conversation_id"])
            if isinstance(result, web.StreamResponse):
                return result
            data = canonical(result).encode()
            if len(data) > self.config.max_response_bytes:
                raise RequestError(503, "native_unavailable")
            return web.Response(
                body=data,
                status=status,
                content_type="application/json",
                headers={
                    "Cache-Control": "no-store",
                    "X-Content-Type-Options": "nosniff",
                },
            )
        except RequestError as exc:
            return self._error(exc.status, exc.code)
        except ClarificationRecoveryGap:
            return self._error(409, "recovery_gap")
        except (CommandConflict, ChannelBindingConflict):
            return self._error(409, "conflict")
        except PermissionError:
            return self._error(404, "authorization_denied")
        except web.HTTPRequestEntityTooLarge:
            return self._error(413, "invalid_request")
        except (ValueError, TypeError, UnicodeError, RecursionError):
            return self._error(400, "invalid_request")
        except LookupError:
            return self._error(404, "authorization_denied")
        except (TimeoutError, OSError):
            return self._error(503, "native_unavailable")
        except Exception:
            return self._error(503, "native_unavailable")
        finally:
            if admitted:
                self._requests -= 1

    async def start(self):
        context = None
        if self.config.tls_cert:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.load_cert_chain(self.config.tls_cert, self.config.tls_key)
        app = web.Application(client_max_size=self.config.max_body_bytes,
                              handler_args={"auto_decompress": False})
        app.router.add_route("*", "/{path:.*}", self.handle)
        protocol_logger = logging.Logger("native_pwa_protocol")
        protocol_logger.disabled = True  # Parser failures can contain raw header/query bytes.
        self._http_runner = web.AppRunner(app, access_log=None, logger=protocol_logger,
            shutdown_timeout=5, keepalive_timeout=15, max_line_size=8192, max_field_size=8192)
        try:
            await self._http_runner.setup()
            self._site = web.TCPSite(self._http_runner, self.config.host, self.config.port, ssl_context=context)
            await self._site.start()
            self.port = self._site._server.sockets[0].getsockname()[1]
        except BaseException:
            await self._http_runner.cleanup()
            self._http_runner = None
            raise

    async def close(self):
        self._closing = True
        self._close_event.set()
        self.ingress.events.close()
        for task in tuple(self._streams):
            task.cancel()
        if self._streams:
            await asyncio.gather(*tuple(self._streams), return_exceptions=True)
        if self._http_runner is not None:
            await self._http_runner.cleanup()
            self._http_runner = None
        self._cursors.clear()
        self.history_scan.handles.close()
        self.images.close()
