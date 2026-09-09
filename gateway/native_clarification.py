"""Task-bound native clarification over the existing blocking gateway primitive.

No permission decision is interpreted here. A native waiter receives only its
own exact, durably claimed text or offered selection.
"""

import asyncio
import hashlib
import json
import secrets
import time
import uuid

from agent.native_execution_context import (
    current_native_execution,
    NativeExecutionOrigin,
)
from hermes_state_commands import CommandConflict, canonical
from tools import clarify_gateway


def manager_for_current_execution():
    context = current_native_execution()
    if not context or context[1] is None:
        return None
    return getattr(context[1].ingress, "clarifications", None)


class NativeWaiter:
    def __init__(self, manager, origin, identity):
        self.manager, self.origin, self.identity = manager, origin, identity
        self.claim_token = secrets.token_hex(32)
        self.claim = None

    def claim_answer(self, entry, scope, body):
        row, created = self.manager.db.native_question_claim(
            scope, body, self.origin, self.claim_token
        )
        if created:
            question = self.manager.db.native_question_lookup(self.identity)
            # DB commit precedes wake; a read failure is recovered from that exact
            # durable claim while the same entry/owner still exists.
            response = self.manager.db._native_question_answer(
                question, body["payload"]["answer"]
            )
            self.claim = (scope, body["command_id"])
            entry.response = response
            entry.event.set()
        return row

    def channel_answer(self, entry, response):
        if not response:
            self.cancel(entry)
            return True
        question = self.manager.db.native_question_lookup(self.identity)
        shape = json.loads(question["question_json"])
        if shape["kind"] == "text":
            answer = dict(kind="text", text=str(response))
        else:
            try:
                labels = (
                    json.loads(response)
                    if shape["kind"] == "multi_select"
                    else [response]
                )
            except (ValueError, TypeError):
                return False
            if not isinstance(labels, list) or not labels:
                return False
            from tools.clarify_gateway import _label_matches

            ids = []
            for label in labels:
                matches = [
                    option["id"]
                    for option in shape["options"]
                    if _label_matches(str(label), option["label"])
                ]
                if len(matches) != 1 or matches[0] in ids:
                    return False
                ids.append(matches[0])
            answer = dict(kind="selection", option_ids=ids)
        body = dict(
            schema_version="1.0",
            type="clarification_response",
            command_id="native-answer-" + self.identity,
            conversation_id=self.origin.conversation_id,
            target_execution_id=self.origin.execution_id,
            payload=dict(
                clarification_id=self.identity,
                question_revision=question["question_revision"],
                answer=answer,
            ),
        )
        from gateway.native_redirect import validate_control

        validate_control(body)
        scope = (
            "native-channel:" + hashlib.sha256(entry.session_key.encode()).hexdigest()
        )
        try:
            self.claim_answer(entry, scope, body)
            return True
        except CommandConflict:
            return False

    def other(self, entry):
        try:
            self.manager.db.native_question_other(self.identity, self.origin)
        except CommandConflict:
            return False
        entry.choices = None
        entry.multi_select = False
        entry.awaiting_text = True
        return True

    def finish_wait(self, entry):
        if self.claim is not None:
            try:
                acknowledged = self.manager.db.native_question_ack(
                    self.identity, self.origin, self.claim_token
                )
            except Exception:
                acknowledged = False
            if not acknowledged:
                entry.response = "[clarification answer delivery is unresolved]"
        elif not entry.event.is_set():
            try:
                self.manager.db.native_question_terminal(
                    self.identity, self.origin, "expired"
                )
                entry.response = "[clarification expired without an answer]"
            except Exception:
                entry.response = "[clarification expiry observation is unresolved]"

    def cancel(self, entry):
        try:
            self.manager.db.native_question_terminal(
                self.identity, self.origin, "cancelled"
            )
            entry.response = "[clarification cancelled without resuming the question]"
        except Exception:
            # A failed journal write must not strand a waiter removed by native
            # route cleanup. Recovery records the lost waiter as unknown.
            entry.response = "[clarification cancellation observation is unresolved]"
        entry.event.set()


class NativeClarificationController:
    def __init__(self, ingress):
        self.ingress, self.db = ingress, ingress.db
        self._after = 0
        self._cursor_key = secrets.token_bytes(32)
        self._children = {}

    def bind_child(self, child):
        import weakref
        from functools import partial

        context = current_native_execution()
        if not context or context[1] is not self.ingress.cancellations:
            return
        identity = uuid.uuid4().hex

        def discarded(_):
            with clarify_gateway._lock:
                self._children.pop(identity, None)

        with clarify_gateway._lock:
            self._children[identity] = (weakref.ref(child, discarded), context[0])
        child._native_clarification_binding = (self, identity)
        child.clarify_callback = partial(self.callback, child_id=identity)

    def unbind_child(self, identity):
        with clarify_gateway._lock:
            if self._children.pop(identity, None) is None:
                return
            for entry in tuple(clarify_gateway._entries.values()):
                if entry.managed and entry.managed.manager is self:
                    row = self.db.native_question_lookup(entry.clarify_id)
                    if (
                        row
                        and json.loads(row["origin_json"]).get("child_id") == identity
                        and not entry.event.is_set()
                    ):
                        entry.managed.cancel(entry)

    def has_children(self, origin):
        with clarify_gateway._lock:
            return any(
                reference() is not None and child_origin == origin
                for reference, child_origin in self._children.values()
            )

    def register(
        self, question, choices, *, multi_select=False, session_key="", child_id=None
    ):
        context = current_native_execution()
        if not context or context[1] is not self.ingress.cancellations:
            raise CommandConflict("native question provenance unavailable")
        origin = context[0]
        if child_id is not None:
            binding = self._children.get(child_id)
            if not binding or binding[0]() is None or binding[1] != origin:
                raise CommandConflict("actual native child question owner unavailable")
        if (
            not isinstance(question, str)
            or not question.strip()
            or len(question) > 8192
        ):
            raise ValueError("native question exceeds prompt limit")
        choices = list(choices or [])
        if len(choices) > 4 or any(
            not isinstance(label, str) or not label or len(label) > 1024
            for label in choices
        ):
            raise ValueError("native question options exceed limit")
        shape = dict(
            prompt=question,
            kind=("multi_select" if multi_select else "single_select")
            if choices
            else "text",
            options=[
                dict(id="o" + str(index + 1), label=label)
                for index, label in enumerate(choices)
            ],
        )
        if len(canonical(shape).encode()) > 60000:
            raise ValueError("native question exceeds byte limit")
        identity = uuid.uuid4().hex
        timeout = float(clarify_gateway.get_clarify_timeout())
        expires = time.time() + timeout if timeout > 0 else None
        waiter = NativeWaiter(self, origin, identity)
        with clarify_gateway._lock:
            self.db.native_question_register(
                origin,
                identity,
                origin_kind="native_child" if child_id else "native",
                origin_key=identity,
                origin_context=dict(child_id=child_id, session_key=session_key),
                question=shape,
                expires_at=expires,
            )
            return clarify_gateway.register(
                identity, session_key, question, choices, multi_select, managed=waiter
            )

    def callback(self, question, choices, multi_select=False, *, child_id=None):
        entry = self.register(
            question, choices, multi_select=multi_select, child_id=child_id
        )
        return clarify_gateway.wait_for_response(
            entry.clarify_id, clarify_gateway.get_clarify_timeout()
        )

    async def submit(self, principal, body):
        scope = self.ingress._command_scope(principal)
        old = self.db.native_control_lookup(scope, body["command_id"])
        if old:
            # Duplicate lookup still checks the full canonical request and type.
            self.db._execute_write(
                lambda conn: self.db._native_control_duplicate(conn, scope, body)
            )
            return await self.ingress.controls.receipt(principal, body["command_id"])
        identity = body["payload"]["clarification_id"]
        with clarify_gateway._lock:
            entry = clarify_gateway._entries.get(identity)
            if (
                not entry
                or not entry.managed
                or entry.managed.manager is not self
                or entry.event.is_set()
            ):
                raise CommandConflict("exact native question waiter unavailable")
            entry.managed.claim_answer(entry, scope, body)
        self.ingress.controls.wake()
        return await self.ingress.controls.receipt(principal, body["command_id"])

    def cancel_execution(self, origin):
        with clarify_gateway._lock:
            for entry in tuple(clarify_gateway._entries.values()):
                waiter = entry.managed
                if (
                    waiter
                    and waiter.manager is self
                    and waiter.origin == origin
                    and not entry.event.is_set()
                ):
                    waiter.cancel(entry)

    async def reconcile(self):
        rows = await asyncio.to_thread(
            self.db.native_question_pending, after=self._after, limit=16
        )
        if not rows and self._after:
            self._after = 0
            rows = await asyncio.to_thread(self.db.native_question_pending, limit=16)
        for row in rows:
            self._after = row["ordinal"]
            origin = NativeExecutionOrigin(
                row["conversation_id"], row["execution_id"], row["owner"]
            )
            with clarify_gateway._lock:
                entry = clarify_gateway._entries.get(row["clarification_id"])
                waiter = entry.managed if entry else None
                if waiter and waiter.manager is self and waiter.origin == origin:
                    if (
                        row["state"] == "answer_recorded"
                        and not entry.event.is_set()
                        and row["claim_token"] == waiter.claim_token
                    ):
                        command = self.db.native_control_lookup(
                            row["answer_scope"], row["answer_command_id"]
                        )
                        answer = json.loads(command["payload_json"])["payload"][
                            "answer"
                        ]
                        entry.response = self.db._native_question_answer(row, answer)
                        waiter.claim = (row["answer_scope"], row["answer_command_id"])
                        entry.event.set()
                    continue
            # A lost process-local waiter never gets replayed into a new worker.
            from hermes_state_commands import process_owner_dead

            if row["owner"] == self.ingress._command_owner or process_owner_dead(
                row["owner"]
            ):
                self.db.native_question_terminal(
                    row["clarification_id"], origin, "unknown"
                )

    def _available(self):
        return frozenset(
            identity
            for identity, entry in clarify_gateway._entries.items()
            if entry.managed
            and entry.managed.manager is self
            and not entry.event.is_set()
        )

    def _scope(self, principal, root):
        from gateway.conversation_control import _source_identity

        projection, grant = self.ingress._authorize(principal, root)
        if not self.ingress.runner._is_user_authorized_for_source(grant.source):
            raise PermissionError("native source is not authorized")
        return canonical([
            self.ingress._command_scope(principal),
            projection["conversation_id"],
            grant.session_id,
            [
                getattr(value, "value", value)
                for value in _source_identity(grant.source)
            ],
        ])

    def _cursor(self, scope, epoch, upper, after, limit):
        import base64
        import hmac

        value = (
            base64
            .urlsafe_b64encode(
                canonical([
                    epoch,
                    upper,
                    after,
                    limit,
                    int(time.time()) + 3600,
                ]).encode()
            )
            .decode()
            .rstrip("=")
        )
        signature = hmac.new(
            self._cursor_key, (scope + "\n" + value).encode(), hashlib.sha256
        ).hexdigest()
        return value + "." + signature

    def _decode(self, scope, cursor, limit):
        import base64
        import hmac
        import re

        if (
            not isinstance(cursor, str)
            or len(cursor) > 256
            or not re.fullmatch(r"[A-Za-z0-9_-]+\.[a-f0-9]{64}", cursor)
        ):
            raise ValueError("invalid question cursor")
        value, signature = cursor.split(".")
        expected = hmac.new(
            self._cursor_key, (scope + "\n" + value).encode(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected, signature):
            raise ClarificationRecoveryGap()
        try:
            values = json.loads(
                base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
            )
            epoch, upper, after, page_limit, expires = values
            if (
                not isinstance(epoch, str)
                or any(
                    type(number) is not int or number < 0
                    for number in (upper, after, page_limit, expires)
                )
                or after > upper
            ):
                raise ValueError()
        except (ValueError, TypeError):
            raise ValueError("invalid question cursor") from None
        if page_limit != limit or expires < time.time():
            raise ClarificationRecoveryGap()
        return epoch, upper, after

    def recovery_snapshot(
        self,
        root,
        scope,
        cursor=None,
        *,
        delivery_channel_key=None,
        question_scope=None,
    ):
        with clarify_gateway._lock:
            available = self._available()
            result = self.db.native_event_recovery(
                root,
                scope,
                cursor,
                delivery_channel_key=delivery_channel_key,
                clarification_available=available,
            )
            page = result.pop("question_page")
            next_cursor = (
                self._cursor(
                    question_scope,
                    result["question_epoch"],
                    page["upper"],
                    page["after"],
                    10,
                )
                if page["has_more"]
                else None
            )
            result["pending_clarifications"] = page["questions"]
            result["clarification_coverage"] = dict(
                limit=10,
                has_more=page["has_more"],
                next_cursor=next_cursor,
                selection="unresolved",
                max_response_bytes=131072,
            )
            return result

    async def page(self, principal, root, *, cursor=None, limit=50, identity=None):
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("invalid question page limit")
        projection, _ = self.ingress._authorize(principal, root)
        root = projection["conversation_id"]
        scope = self._scope(principal, root)
        decoded = self._decode(scope, cursor, limit) if cursor is not None else None

        def capture():
            from hermes_state_events import timestamp
            from hermes_state_clarifications import question_view

            with clarify_gateway._lock:
                available = self._available()

                def read(conn):
                    epoch = self.db.native_event_epoch(root)
                    if decoded and decoded[0] != epoch:
                        raise ClarificationRecoveryGap()
                    head = conn.execute(
                        "SELECT sequence FROM native_event_heads WHERE conversation_id=? AND epoch=?",
                        (root, epoch),
                    ).fetchone()
                    result = dict(
                        schema_version="1.0",
                        conversation_id=root,
                        captured_at=timestamp(time.time()),
                        event_cursor=dict(epoch=epoch, sequence=head[0] if head else 0),
                    )
                    if identity is not None:
                        row = self.db._native_question_row(conn, identity)
                        if not row or row["conversation_id"] != root:
                            raise LookupError("question unavailable")
                        result["clarification"] = question_view(
                            row, identity in available
                        )
                        return result
                    page = self.db._native_question_page(
                        conn,
                        root,
                        limit=limit,
                        after=decoded[2] if decoded else 0,
                        upper=decoded[1] if decoded else None,
                        available=available,
                    )
                    result.update(
                        questions=page["questions"],
                        has_more=page["has_more"],
                        next_cursor=self._cursor(
                            scope, epoch, page["upper"], page["after"], limit
                        )
                        if page["has_more"]
                        else None,
                        coverage=dict(
                            selection="unresolved",
                            limit=limit,
                            max_response_bytes=131072,
                            consistency="page_capture",
                            order="question_admission_ascending",
                        ),
                    )
                    if len(canonical(result).encode()) > 131072:
                        raise ValueError("question page exceeds response bound")
                    return result

                return self.db._execute_write(read)

        result = await asyncio.to_thread(capture)
        if self._scope(principal, root) != scope:
            raise ClarificationRecoveryGap()
        return result


class ClarificationRecoveryGap(Exception):
    """Opaque cursor no longer names a valid authorized traversal."""
