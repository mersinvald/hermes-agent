"""Restricted private facade transport inside the existing GatewayRunner."""
from __future__ import annotations

import asyncio
import base64
import hmac
import json
import logging
import os
import secrets
import ssl
import time
from collections import OrderedDict

from aiohttp import web

from gateway.conversation_control import NativeConversationIngress
from gateway.pwa_config import PwaHttpConfig, canonical, closed, identifier, parse_principal
from gateway.pwa_ownership import NativePwaOwnership
from hermes_state_commands import CommandConflict

CONTRACT_PIN = "e0b1e40581d8aba1f1a59985c0f8268ed3b3e111"


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
        if len(token.encode()) < 32 or len(token) > 4096 or any(ord(c) < 33 or ord(c) > 126 for c in token):
            raise ValueError("native PWA facade credential unavailable")
        self._token = token
        self.db = runner.session_store._db
        if self.db is None:
            raise RuntimeError("native PWA SessionDB unavailable")
        self.ownership = NativePwaOwnership(runner, self.db, self.config, secret=token)
        self.ingress = NativeConversationIngress(runner, self.db, {}, concierge_id=self.config.concierge_id, grant_provider=self.ownership)
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

    async def _dispatch(self, request, principal):
        tail = request.path.removeprefix("/v1/pwa/")
        if request.method == "GET" and tail == "health":
            self._query(request)
            if not self.runner._running and not self._closing:
                raise RequestError(503, "native_unavailable")
            return {"schema_version": "1.0", "service": "native_pwa", "state": "draining" if self._closing or self.runner._draining else "ready"}, 200
        if self._closing or not self.runner._running or self.runner._draining:
            raise RequestError(503, "native_unavailable")
        if request.method == "GET" and tail == "capabilities":
            self._query(request)
            capabilities = [{"capability_id": name, "availability": "available"} for name in ("command_admission", "durable_commands", "history")]
            if self.runner._get_proxy_url():
                for capability in capabilities[:2]:
                    capability.update(availability="unavailable", reason="Local native execution is required.")
            capabilities += [{"capability_id": name, "availability": "unavailable", "reason": "Native integration is not yet available."} for name in ("event_stream", "event_replay", "snapshot_recovery", "remote_cancellation", "telegram_final_delivery")]
            return {"schema_version": "1.0", "capabilities": capabilities}, 200
        if tail == "conversations" and request.method == "GET":
            limit, cursor = self._page(request)
            scope = self._scope(principal, "conversations", limit)
            after = self._cursor(cursor, scope) if cursor else ""
            roots, next_after, coverage = await asyncio.to_thread(self.ownership.discover, principal, after, limit)
            conversations = [await self.ingress.native_conversation(principal, root) for root in roots]
            return {"schema_version": "1.0", "conversations": conversations, "coverage": coverage,
                    "next_cursor": self._new_cursor(next_after, scope) if next_after is not None else None}, 200
        if tail == "conversations" and request.method == "POST":
            self._query(request)
            body = await self._body(request)
            closed(body, {"schema_version", "create_id"}, {"schema_version", "create_id"})
            if body["schema_version"] != "1.0":
                raise ValueError("invalid create version")
            result, created = await self.ownership.create(self.ingress, principal, identifier(body["create_id"]))
            return result, 201 if created else 200
        parts = tail.split("/")
        if request.method == "GET" and len(parts) in (2, 3) and parts[0] == "conversations":
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
            self._query(request)
            return await self.ingress.command_receipt(principal, identifier(parts[1])), 200
        raise RequestError(404, "capability_unavailable")

    async def handle(self, request):
        admitted = False
        try:
            principal = self._authenticate(request)
            if self._requests >= self.config.max_requests:
                raise RequestError(429, "native_unavailable")
            self._requests += 1
            admitted = True
            async with asyncio.timeout(self.config.request_timeout):
                result, status = await self._dispatch(request, principal)
            data = canonical(result).encode()
            if len(data) > self.config.max_response_bytes:
                raise RequestError(503, "native_unavailable")
            return web.Response(body=data, status=status, content_type="application/json",
                                headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})
        except RequestError as exc:
            return self._error(exc.status, exc.code)
        except CommandConflict:
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
        app = web.Application(client_max_size=self.config.max_body_bytes)
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
        if self._http_runner is not None:
            await self._http_runner.cleanup()
            self._http_runner = None
        self._cursors.clear()
