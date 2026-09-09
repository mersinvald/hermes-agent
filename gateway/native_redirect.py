"""N06 redirect gates over existing native interruption, ownership and FIFO."""

import asyncio
import hmac
import json
import logging
import re

from agent.native_execution_context import NativeExecutionOrigin
from hermes_state_events import timestamp

logger = logging.getLogger(__name__)


def validate_control(command):
    required = {
        "schema_version",
        "command_id",
        "conversation_id",
        "target_execution_id",
        "type",
        "payload",
    }
    allowed = set(required)
    if command.get("type") == "redirect":
        allowed.add("expected_model_version")
    if set(command) < required or set(command) - allowed:
        raise ValueError("unsupported control fields or preconditions")
    if not isinstance(command.get("target_execution_id"), str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", command["target_execution_id"]
    ):
        raise ValueError("invalid execution target")
    payload = command.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("invalid control payload")
    if command["type"] == "redirect":
        expected_model_version = command.get("expected_model_version")
        if expected_model_version is not None and (
            type(expected_model_version) is not int
            or not 1 <= expected_model_version <= 9007199254740991
        ):
            raise ValueError("invalid expected model version")
        text = payload.get("text")
        if (
            set(payload) != {"text"}
            or not isinstance(text, str)
            or not text.strip()
            or len(text) > 100000
        ):
            raise ValueError("expected nonempty redirect text")
        if len(text.encode("utf-8")) > 409600:
            raise ValueError("redirect text exceeds byte limit")
    elif command["type"] == "redirect_confirm":
        if (
            set(payload)
            != {
                "redirect_command_id",
                "confirmation_revision",
                "allow_unresolved_remote",
            }
            or not isinstance(payload["redirect_command_id"], str)
            or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", payload["redirect_command_id"]
            )
            or type(payload["confirmation_revision"]) is not int
            or not 1 <= payload["confirmation_revision"] <= 9007199254740991
            or payload["allow_unresolved_remote"] is not True
        ):
            raise ValueError("invalid redirect confirmation")

    elif command["type"] == "clarification_response":
        if (
            set(payload) != {"clarification_id", "question_revision", "answer"}
            or not isinstance(payload["clarification_id"], str)
            or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", payload["clarification_id"]
            )
            or type(payload["question_revision"]) is not int
            or not 1 <= payload["question_revision"] <= 9007199254740991
        ):
            raise ValueError("invalid question target")
        answer = payload["answer"]
        if not isinstance(answer, dict):
            raise ValueError("invalid question answer")
        if answer.get("kind") == "text":
            text = answer.get("text")
            if (
                set(answer) != {"kind", "text"}
                or not isinstance(text, str)
                or not text.strip()
                or len(text) > 100000
                or len(text.encode()) > 409600
            ):
                raise ValueError("invalid text answer")
        elif answer.get("kind") == "selection":
            ids = answer.get("option_ids")
            if (
                set(answer) != {"kind", "option_ids"}
                or not isinstance(ids, list)
                or not 1 <= len(ids) <= 4
                or any(
                    not isinstance(identity, str)
                    or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", identity)
                    for identity in ids
                )
                or len(set(ids)) != len(ids)
            ):
                raise ValueError("invalid selection answer")
        else:
            raise ValueError("unsupported question answer kind")


class NativeControlController:
    """One bounded observation pump, never a model executor or root owner."""

    def __init__(self, ingress):
        self.ingress = ingress
        self.db = ingress.db
        self._timer = None
        self._running = None
        self._after = 0
        self._stopped = False

    def wake(self):
        if self._stopped or self._running is not None:
            return
        if self._timer is not None:
            self._timer.cancel()
        self._timer = asyncio.get_running_loop().call_later(0, self._tick)

    def stop(self):
        self._stopped = True
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def start(self):
        self._stopped = False
        self.wake()

    def _tick(self):
        self._timer = None
        if self._stopped or self._running is not None:
            return
        self._running = asyncio.create_task(self._reconcile())
        self._running.add_done_callback(self._finished)

    def _finished(self, task):
        self._running = None
        if not task.cancelled() and task.exception():
            logger.warning("Native control reconciliation remains pending")
        if not self._stopped:
            self._timer = asyncio.get_running_loop().call_later(1, self._tick)

    def frontier(self, row):
        # Another process's native child inventory cannot be recreated from a
        # route alias or a closed root journal row. Keep that uncertainty explicit.
        if row["owner"] != self.ingress._command_owner:
            return "uncertain"
        execution = self.db.native_execution(row["conversation_id"])
        if execution and execution["execution_id"] == row["execution_id"]:
            return "uncertain"
        from tools.async_delegation import native_execution_has_pending_children

        origin = NativeExecutionOrigin(
            row["conversation_id"], row["execution_id"], row["owner"]
        )
        questions = getattr(self.ingress, "clarifications", None)
        pending_child = questions is not None and questions.has_children(origin)
        return (
            "uncertain"
            if pending_child or native_execution_has_pending_children(origin.record())
            else "closed"
        )

    async def _reconcile(self):
        rows = await asyncio.to_thread(
            self.db.native_redirect_pending, after=self._after, limit=16
        )
        if not rows and self._after:
            self._after = 0
            rows = await asyncio.to_thread(self.db.native_redirect_pending, limit=16)
        for row in rows:
            if self._stopped:
                break
            self._after = row["ordinal"]
            try:
                await asyncio.to_thread(
                    self.db.native_execution_reconcile, row["conversation_id"]
                )
                frontier = self.frontier(row)
                released = await asyncio.to_thread(
                    self.db.native_redirect_progress,
                    row["scope"],
                    row["command_id"],
                    frontier=frontier,
                )
                if released:
                    self.ingress._commands_finished(row["conversation_id"])
            except Exception:
                logger.warning("Native redirect remains pending reconciliation")
        clarifications = getattr(self.ingress, "clarifications", None)
        if clarifications is not None:
            await clarifications.reconcile()

    async def receipt(
        self, principal, command_id, *, remote_cursor=None, remote_limit=50
    ):
        if type(remote_limit) is not int or not 1 <= remote_limit <= 100:
            raise ValueError("invalid remote page limit")
        scope = self.ingress._command_scope(principal)
        row = await asyncio.to_thread(self.db.native_control_lookup, scope, command_id)
        if row is None:
            raise LookupError("control unavailable")
        self.ingress._authorize(principal, row["conversation_id"])
        if row["kind"] != "redirect" and (
            remote_cursor is not None or remote_limit != 50
        ):
            raise ValueError("remote pagination requires cancellation or redirect")
        after = 0
        if remote_cursor is not None:
            if not isinstance(remote_cursor, str) or not re.fullmatch(
                r"[0-9a-f]{1,16}\.[0-9a-f]{64}", remote_cursor
            ):
                raise ValueError("invalid remote cursor")
            after = int(remote_cursor.split(".")[0], 16)
            if not hmac.compare_digest(
                self.ingress.cancellations._cursor(row, after), remote_cursor
            ):
                raise ValueError("invalid remote cursor")
        view = await asyncio.to_thread(
            self.db.native_control_snapshot,
            scope,
            command_id,
            after=after,
            limit=remote_limit,
        )
        _, grant = self.ingress._authorize(principal, row["conversation_id"])
        if not self.ingress.runner._is_user_authorized_for_source(grant.source):
            raise PermissionError("native source is not authorized")
        return self.receipt_view(view)

    def receipt_view(self, view):
        row = view["command"]
        if "kind" not in row:
            return self.ingress.cancellations.receipt_view(view)
        common = dict(
            schema_version="1.0",
            receipt_kind=row["kind"],
            command_id=row["command_id"],
            conversation_id=row["conversation_id"],
            target_execution_id=row["execution_id"],
            payload_fingerprint=row["fingerprint"],
            receipt_state="accepted",
            durability="durable",
            recorded_at=timestamp(row["recorded_at"]),
        )
        if row["kind"] == "redirect_confirm":
            return dict(
                **common,
                redirect_command_id=row["reference_id"],
                confirmation_revision=row["reference_revision"],
                allow_unresolved_remote=True,
                confirmation_state="recorded",
            )
        if row["kind"] == "clarification_response":
            question = view["question"]
            return dict(
                **common,
                clarification_id=row["reference_id"],
                question_revision=row["reference_revision"],
                origin=question["origin"],
                answer_state=row["answer_state"],
                delivery_evidence=row["delivery_evidence"],
                question_state=question["state"],
            )
        state = view["redirect"]
        stop = self.ingress.cancellations.receipt_view(view["cancellation"])
        basis = (
            json.loads(state["release_basis_json"])
            if state["release_basis_json"]
            else None
        )
        input_row = view["input"]
        application = input_row["phase"] if input_row else "pending"
        if application in {"queued", "assigned"}:
            application = "pending"
        if basis is not None:
            if input_row is None:
                raise RuntimeError("released redirect input evidence unavailable")
            phase = "queued" if application == "pending" else application
            native_release = "released"
        elif not view["native_released"]:
            native_release = (
                "unknown"
                if stop["native"]["observed_state"] == "unknown"
                else "pending"
            )
            phase = "awaiting_native_release"
        else:
            native_release, phase = "released", "awaiting_confirmation"
        return dict(
            **common,
            phase=phase,
            native_release=native_release,
            confirmation=dict(
                required=bool(state["confirmation_required"]),
                revision=state["confirmation_revision"]
                if state["confirmation_required"]
                else None,
                confirmed=state["confirmation_command_id"] is not None,
            ),
            dispatch_frontier=state["dispatch_frontier"],
            release_basis=basis,
            queue_policy="after_retained_inputs",
            queued_and_unconsumed_steers="retained",
            new_direction=dict(
                application_state=application,
                resulting_execution_id=input_row["resulting_execution_id"]
                if input_row
                else None,
            ),
            **{
                key: stop[key]
                for key in (
                    "native",
                    "remote_coverage",
                    "remote_targets",
                    "next_remote_cursor",
                )
            },
        )
