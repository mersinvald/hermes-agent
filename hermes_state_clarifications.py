"""Exact native question and answer claims in the native journal transaction."""

import json
import time

from hermes_state_commands import CommandConflict, canonical


class NativeClarificationStateMixin:
    @staticmethod
    def _native_question_row(conn, identity):
        row = conn.execute(
            "SELECT * FROM native_clarifications WHERE clarification_id=?", (identity,)
        ).fetchone()
        return dict(row) if row else None

    def native_question_lookup(self, identity):
        with self._read_ctx() as conn:
            return self._native_question_row(conn, identity)

    def _native_question_event(self, conn, row):
        self._native_question_execution_state(conn, row)
        self._native_event_append(
            conn,
            row["conversation_id"],
            row["execution_id"],
            "clarification_state_changed",
            dict(
                clarification_id=row["clarification_id"],
                question_revision=row["question_revision"],
                state=row["state"],
            ),
        )

    def _native_question_execution_state(self, conn, row):
        if row["origin"] != "native":
            return
        pending = conn.execute(
            "SELECT 1 FROM native_clarifications WHERE execution_id=? AND origin='native' AND state IN ('pending','answer_recorded') LIMIT 1",
            (row["execution_id"],),
        ).fetchone()
        state = "input_required" if pending else "running"
        changed = conn.execute(
            "UPDATE native_executions SET observed_state=?,updated_at=? WHERE execution_id=? AND owner=? AND state='open' AND observed_state IN ('starting','running','input_required') AND observed_state<>?",
            (state, time.time(), row["execution_id"], row["owner"], state),
        ).rowcount
        if changed:
            self._native_event_append(
                conn,
                row["conversation_id"],
                row["execution_id"],
                "execution_state_changed",
                {"state": state},
            )

    @staticmethod
    def _native_question_owner(conn, origin, *, allow_closed=False):
        execution = conn.execute(
            "SELECT * FROM native_executions WHERE conversation_id=? AND execution_id=? AND owner=? AND (state='open' OR (? AND state='closed'))",
            (origin.conversation_id, origin.execution_id, origin.owner, allow_closed),
        ).fetchone()
        cancelled = conn.execute(
            "SELECT 1 FROM native_cancel_commands WHERE execution_id=?",
            (origin.execution_id,),
        ).fetchone()
        if not execution or cancelled:
            raise CommandConflict(
                "native question owner is no longer accepting questions"
            )
        return execution

    def native_question_register(
        self,
        origin,
        identity,
        *,
        origin_kind,
        origin_key,
        origin_context,
        question,
        expires_at,
    ):
        if origin_kind not in {"native", "native_child", "remote"}:
            raise ValueError("invalid native question origin")

        def write(conn):
            self._native_question_owner(
                conn, origin, allow_closed=origin_kind == "native_child"
            )
            now = time.time()
            conn.execute(
                "INSERT INTO native_clarifications(clarification_id,question_revision,conversation_id,execution_id,owner,origin,origin_key,origin_json,question_json,state,expires_at,recorded_at,updated_at) VALUES(?,1,?,?,?,?,?,?,?,'pending',?,?,?)",
                (
                    identity,
                    origin.conversation_id,
                    origin.execution_id,
                    origin.owner,
                    origin_kind,
                    origin_key,
                    canonical(origin_context),
                    canonical(question),
                    expires_at,
                    now,
                    now,
                ),
            )
            row = self._native_question_row(conn, identity)
            self._native_question_event(conn, row)
            return row

        return self._execute_write(write)

    @staticmethod
    def _native_question_answer(question, answer):
        shape = json.loads(question["question_json"])
        if answer["kind"] == "text":
            if shape["kind"] != "text":
                raise CommandConflict("question requires an offered selection")
            return answer["text"]
        options = {option["id"]: option["label"] for option in shape["options"]}
        ids = answer["option_ids"]
        if (
            shape["kind"] == "text"
            or (shape["kind"] == "single_select" and len(ids) != 1)
            or any(identity not in options for identity in ids)
        ):
            raise CommandConflict("answer is not an offered selection")
        labels = [options[identity] for identity in ids]
        return (
            json.dumps(labels, ensure_ascii=False)
            if shape["kind"] == "multi_select"
            else labels[0]
        )

    def native_question_claim(self, scope, body, origin, claim_token):
        """Caller holds the exact live native entry lock across commit and wake."""

        def write(conn):
            old = self._native_control_duplicate(conn, scope, body)
            if old:
                return old, False
            payload = body["payload"]
            question = self._native_question_row(conn, payload["clarification_id"])
            execution = self._native_question_owner(
                conn,
                origin,
                allow_closed=bool(question and question["origin"] == "native_child"),
            )
            if (
                not question
                or question["question_revision"] != payload["question_revision"]
                or (
                    question["conversation_id"],
                    question["execution_id"],
                    question["owner"],
                )
                != (body["conversation_id"], body["target_execution_id"], origin.owner)
                or question["execution_id"] != origin.execution_id
                or question["state"] != "pending"
                or question["expires_at"] is not None
                and question["expires_at"] <= time.time()
            ):
                raise CommandConflict("question is no longer pending for this owner")
            self._native_question_answer(question, payload["answer"])
            row = self._native_control_insert(
                conn,
                scope,
                body,
                execution,
                (question["clarification_id"], question["question_revision"]),
            )
            conn.execute(
                "UPDATE native_clarifications SET state='answer_recorded',answer_scope=?,answer_command_id=?,claim_token=?,updated_at=? WHERE clarification_id=?",
                (
                    scope,
                    body["command_id"],
                    claim_token,
                    time.time(),
                    question["clarification_id"],
                ),
            )
            self._native_control_event(conn, row, "recorded")
            self._native_question_event(
                conn, self._native_question_row(conn, question["clarification_id"])
            )
            return row, True

        return self._execute_write(write)

    def native_question_other(self, identity, origin):
        def write(conn):
            row = self._native_question_row(conn, identity)
            self._native_question_owner(
                conn, origin, allow_closed=bool(row and row["origin"] == "native_child")
            )
            if (
                not row
                or row["origin"] not in {"native", "native_child"}
                or (row["execution_id"], row["owner"])
                != (origin.execution_id, origin.owner)
                or row["state"] != "pending"
                or row["question_revision"] >= 9007199254740991
                or row["expires_at"] is not None
                and row["expires_at"] <= time.time()
            ):
                raise CommandConflict("native question cannot change input kind")
            shape = json.loads(row["question_json"])
            if shape["kind"] == "text":
                return row
            shape.update(kind="text", options=[])
            conn.execute(
                "UPDATE native_clarifications SET question_revision=question_revision+1,question_json=?,updated_at=? WHERE clarification_id=?",
                (canonical(shape), time.time(), identity),
            )
            row = self._native_question_row(conn, identity)
            self._native_question_event(conn, row)
            return row

        return self._execute_write(write)

    def native_question_ack(self, identity, origin, claim_token):
        """Evidence comes only from the exact native waiter accepting its claim."""

        def write(conn):
            question = self._native_question_row(conn, identity)
            if not question or (
                question["execution_id"],
                question["owner"],
                question["claim_token"],
            ) != (origin.execution_id, origin.owner, claim_token):
                raise CommandConflict("native question claim unavailable")
            if question["state"] == "answered":
                return True
            if question["state"] != "answer_recorded":
                return False
            row = self._native_control_row(
                conn, question["answer_scope"], question["answer_command_id"]
            )
            if row["answer_state"] != "recorded":
                return False
            now = time.time()
            conn.execute(
                "UPDATE native_control_commands SET answer_state='delivered',delivery_evidence='native_waiter_handoff',updated_at=? WHERE scope=? AND command_id=?",
                (now, row["scope"], row["command_id"]),
            )
            conn.execute(
                "UPDATE native_clarifications SET state='answered',updated_at=? WHERE clarification_id=?",
                (now, identity),
            )
            self._native_control_event(conn, row, "delivered")
            self._native_question_event(conn, self._native_question_row(conn, identity))
            return True

        return self._execute_write(write)

    def native_question_terminal(self, identity, origin, state):
        if state not in {"cancelled", "expired", "unknown"}:
            raise ValueError("invalid question terminal observation")

        def write(conn):
            question = self._native_question_row(conn, identity)
            if not question or (question["execution_id"], question["owner"]) != (
                origin.execution_id,
                origin.owner,
            ):
                raise CommandConflict("native question owner unavailable")
            if question["state"] not in {"pending", "answer_recorded"}:
                return False
            now = time.time()
            conn.execute(
                "UPDATE native_clarifications SET state=?,updated_at=? WHERE clarification_id=?",
                (state, now, identity),
            )
            if question["answer_command_id"]:
                row = self._native_control_row(
                    conn, question["answer_scope"], question["answer_command_id"]
                )
                answer_state = "unknown" if state == "unknown" else "not_delivered"
                conn.execute(
                    "UPDATE native_control_commands SET answer_state=?,updated_at=? WHERE scope=? AND command_id=?",
                    (answer_state, now, row["scope"], row["command_id"]),
                )
                self._native_control_event(conn, row, answer_state)
            self._native_question_event(conn, self._native_question_row(conn, identity))
            return True

        return self._execute_write(write)

    def native_question_pending(self, *, after=0, limit=100):
        if (
            type(after) is not int
            or after < 0
            or type(limit) is not int
            or not 1 <= limit <= 100
        ):
            raise ValueError("invalid question page")
        with self._read_ctx() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM native_clarifications WHERE ordinal>? AND state IN ('pending','answer_recorded') ORDER BY ordinal LIMIT ?",
                    (after, limit),
                )
            ]

    def _native_question_page(
        self, conn, root, *, limit=10, after=0, upper=None, available=frozenset()
    ):
        if upper is None:
            upper = conn.execute(
                "SELECT COALESCE(MAX(ordinal),0) FROM native_clarifications WHERE conversation_id=?",
                (root,),
            ).fetchone()[0]
        rows = conn.execute(
            "SELECT * FROM native_clarifications WHERE conversation_id=? AND ordinal>? AND ordinal<=? AND state IN ('pending','answer_recorded','unknown') ORDER BY ordinal LIMIT ?",
            (root, after, upper, limit + 1),
        ).fetchall()
        questions, used, last = [], 2048, after
        for row in rows[:limit]:
            value = question_view(dict(row), row["clarification_id"] in available)
            size = len(canonical(value).encode())
            if size > 65536:
                raise ValueError("stored native question exceeds response bound")
            if used + size > 131072:
                break
            questions.append(value)
            used += size + 1
            last = row["ordinal"]
        return dict(
            questions=questions,
            has_more=len(rows) > len(questions),
            after=last,
            upper=upper,
        )


def question_view(row, available=False):
    from hermes_state_events import timestamp

    shape = json.loads(row["question_json"])
    pending = row["state"] == "pending"
    available = bool(
        available
        and pending
        and (row["expires_at"] is None or row["expires_at"] > time.time())
    )
    reason = (
        None if available else ("not_pending" if not pending else "owner_unavailable")
    )
    if row["origin"] == "remote":
        available = False
        reason = "adapter_disabled" if pending else "not_pending"
    if row["state"] == "unknown":
        reason = "reconciliation_required"
    return dict(
        schema_version="1.0",
        clarification_id=row["clarification_id"],
        conversation_id=row["conversation_id"],
        execution_id=row["execution_id"],
        question_revision=row["question_revision"],
        origin=row["origin"],
        answer_protocol="hermes.clarification.v1"
        if row["origin"] == "remote"
        else "native_callback_v1",
        state=row["state"],
        question=shape["prompt"],
        input_kind=shape["kind"],
        options=[
            dict(option_id=option["id"], label=option["label"])
            for option in shape["options"]
        ],
        expires_at=timestamp(row["expires_at"])
        if row["expires_at"] is not None
        else None,
        response_available=available,
        unavailable_reason=reason,
    )
