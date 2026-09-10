"""Durable N06 control phases in SessionDB's existing transaction domain.

Control rows reserve one public identity. Redirect's interrupt and input rows are
private phases under that identity, never independent public commands.
"""

import hmac
import json
import time

from hermes_state_cancellation import fingerprint
from hermes_state_commands import CommandConflict, canonical


class NativeControlStateMixin:
    @staticmethod
    def _native_control_row(conn, scope, command_id):
        row = conn.execute(
            "SELECT * FROM native_control_commands WHERE scope=? AND command_id=?",
            (scope, command_id),
        ).fetchone()
        return dict(row) if row else None

    def native_control_lookup(self, scope, command_id):
        with self._read_ctx() as conn:
            return self._native_control_row(conn, scope, command_id)

    def _native_control_cross_kind(self, conn, scope, command_id):
        if self._native_control_row(conn, scope, command_id):
            raise CommandConflict("command ID reused with a different payload")

    def _native_control_duplicate(self, conn, scope, body):
        row = self._native_control_row(conn, scope, body["command_id"])
        if row:
            if row["kind"] != body["type"] or not hmac.compare_digest(
                row["fingerprint"], fingerprint(body)
            ):
                raise CommandConflict("command ID reused with a different payload")
            return row
        for table in ("native_commands", "native_cancel_commands"):
            if conn.execute(
                f"SELECT 1 FROM {table} WHERE scope=? AND command_id=?",
                (scope, body["command_id"]),
            ).fetchone():
                raise CommandConflict("command ID reused with a different payload")
        return None

    def _native_control_insert(self, conn, scope, body, execution, reference=None):
        now = time.time()
        conn.execute(
            "INSERT INTO native_control_commands(scope,command_id,kind,conversation_id,execution_id,owner,fingerprint,payload_json,reference_id,reference_revision,recorded_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                scope,
                body["command_id"],
                body["type"],
                body["conversation_id"],
                body["target_execution_id"],
                execution["owner"],
                fingerprint(body),
                canonical(body),
                reference[0] if reference else None,
                reference[1] if reference else None,
                now,
                now,
            ),
        )
        return self._native_control_row(conn, scope, body["command_id"])

    def _native_control_event(self, conn, row, change, dispatch_id=None):
        kind = row["kind"]
        payload = dict(receipt_kind=kind, command_id=row["command_id"], change=change)
        if kind == "redirect_confirm":
            payload.update(
                redirect_command_id=row["reference_id"],
                confirmation_revision=row["reference_revision"],
            )
            event = "redirect_confirmation_state_changed"
        elif kind == "clarification_response":
            payload.update(
                clarification_id=row["reference_id"],
                question_revision=row["reference_revision"],
            )
            event = "clarification_response_state_changed"
        else:
            event = "redirect_state_changed"
            if dispatch_id is not None:
                payload["dispatch_id"] = dispatch_id
        self._native_event_append(
            conn,
            row["conversation_id"],
            row["execution_id"],
            event,
            payload,
            scope=row["scope"],
        )

    def native_redirect_admit(self, scope, body, *, images=None):
        def write(conn):
            # An immutable replay is authoritative even if the conversation's
            # selection moved after the original admission.
            old = self._native_control_duplicate(conn, scope, body)
            if old:
                return old, False
            if "expected_model_version" in body:
                selected = conn.execute(
                    "SELECT model_version FROM native_conversation_models "
                    "WHERE conversation_id=?",
                    (body["conversation_id"],),
                ).fetchone()
                if (
                    selected is None
                    or selected["model_version"] != body["expected_model_version"]
                ):
                    raise CommandConflict("model version changed")
            execution = conn.execute(
                "SELECT * FROM native_executions WHERE conversation_id=? ORDER BY created_order DESC LIMIT 1",
                (body["conversation_id"],),
            ).fetchone()
            if (
                not execution
                or execution["execution_id"] != body["target_execution_id"]
            ):
                raise CommandConflict("target execution is no longer current")
            row = self._native_control_insert(conn, scope, body, execution)
            if images is not None:
                conn.execute(
                    "UPDATE native_control_commands SET payload_json=? WHERE scope=? AND command_id=?",
                    (canonical({**body, "_native_images": images}), scope, body["command_id"]),
                )
                row = self._native_control_row(conn, scope, body["command_id"])
            conn.execute(
                "INSERT INTO native_redirect_state(scope,command_id,updated_at) VALUES(?,?,?)",
                (scope, body["command_id"], time.time()),
            )
            # Same logical ID. The N06 discriminator exists before any observer
            # can emit the interrupt phase, in this same commit.
            self._native_cancel_insert(conn, scope, body, execution)
            self._native_redirect_update(conn, row, frontier="uncertain", release=False)
            return row, True

        return self._execute_write(write)

    def native_redirect_confirm(self, scope, body):
        def write(conn):
            old = self._native_control_duplicate(conn, scope, body)
            if old:
                return old, False
            payload = body["payload"]
            target = self._native_control_row(
                conn, scope, payload["redirect_command_id"]
            )
            state = conn.execute(
                "SELECT * FROM native_redirect_state WHERE scope=? AND command_id=?",
                (scope, payload["redirect_command_id"]),
            ).fetchone()
            if (
                not target
                or target["kind"] != "redirect"
                or not state
                or (target["conversation_id"], target["execution_id"])
                != (body["conversation_id"], body["target_execution_id"])
                or not state["confirmation_required"]
                or state["confirmation_revision"] != payload["confirmation_revision"]
                or state["confirmation_command_id"] is not None
                or state["release_basis_json"] is not None
            ):
                raise CommandConflict("redirect confirmation is no longer pending")
            row = self._native_control_insert(
                conn,
                scope,
                body,
                target,
                (target["command_id"], payload["confirmation_revision"]),
            )
            conn.execute(
                "UPDATE native_redirect_state SET confirmation_command_id=?,confirmed_revision=?,updated_at=? WHERE scope=? AND command_id=?",
                (
                    body["command_id"],
                    payload["confirmation_revision"],
                    time.time(),
                    scope,
                    target["command_id"],
                ),
            )
            self._native_control_event(conn, row, "recorded")
            self._native_control_event(conn, target, "confirmation_recorded")
            return row, True

        return self._execute_write(write)

    @staticmethod
    def _native_redirect_released(conn, row):
        execution = conn.execute(
            "SELECT * FROM native_executions WHERE conversation_id=? AND execution_id=? AND owner=?",
            (row["conversation_id"], row["execution_id"], row["owner"]),
        ).fetchone()
        if not execution or execution["state"] == "open":
            return False
        lease = conn.execute(
            "SELECT holder FROM session_turn_leases WHERE conversation_id=?",
            (row["conversation_id"],),
        ).fetchone()
        if not lease:
            return True
        if execution["lease_holder"] and lease[0] != execution["lease_holder"]:
            return True  # A different native lease already owns the root.
        from hermes_state import _compression_lock_holder_process_is_dead

        return _compression_lock_holder_process_is_dead(lease[0])

    def _native_redirect_update(self, conn, row, *, frontier, release):
        state = conn.execute(
            "SELECT * FROM native_redirect_state WHERE scope=? AND command_id=?",
            (row["scope"], row["command_id"]),
        ).fetchone()
        now = time.time()
        known = conn.execute(
            "SELECT count(*) FROM native_remote_dispatches WHERE execution_id=?",
            (row["execution_id"],),
        ).fetchone()[0]
        released = self._native_redirect_released(conn, row)
        conn.execute(
            "UPDATE native_redirect_state SET dispatch_frontier=?,updated_at=? WHERE scope=? AND command_id=?",
            (frontier, now, row["scope"], row["command_id"]),
        )
        if state["release_basis_json"] is not None:
            return False  # Preserve the actual past gate, even after late evidence.
        needs_confirmation = bool(known or frontier == "uncertain")
        if needs_confirmation and not state["confirmation_required"]:
            if state["confirmation_revision"] >= 9007199254740991:
                raise CommandConflict("confirmation revision exhausted")
            conn.execute(
                "UPDATE native_redirect_state SET confirmation_required=1,confirmation_revision=confirmation_revision+1 WHERE scope=? AND command_id=?",
                (row["scope"], row["command_id"]),
            )
            self._native_control_event(conn, row, "confirmation_required")
        elif not needs_confirmation and not state["confirmation_command_id"]:
            # A provisional warning can resolve before release. Retain the
            # issuance counter privately; any later warning gets a fresh token.
            conn.execute(
                "UPDATE native_redirect_state SET confirmation_required=0 WHERE scope=? AND command_id=?",
                (row["scope"], row["command_id"]),
            )
        state = conn.execute(
            "SELECT * FROM native_redirect_state WHERE scope=? AND command_id=?",
            (row["scope"], row["command_id"]),
        ).fetchone()
        if (
            not release
            or not released
            or (state["confirmation_required"] and not state["confirmation_command_id"])
        ):
            return False
        if conn.execute(
            "SELECT 1 FROM native_commands WHERE scope=? AND command_id=?",
            (row["scope"], row["command_id"]),
        ).fetchone():
            raise CommandConflict(
                "redirect input phase already exists without release evidence"
            )
        basis = dict(
            remote_gate="explicit_confirmation"
            if state["confirmation_required"]
            else "closed_observed_frontier",
            known_count_at_release=known,
            confirmation_revision=state["confirmed_revision"],
        )
        tail = conn.execute(
            "SELECT COALESCE(MAX(queue_order),0)+1 FROM native_commands"
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO native_commands(scope,command_id,conversation_id,fingerprint,payload_json,requested_action,effective_action,target_execution_id,phase,queue_order,recorded_at) VALUES(?,?,?,?,?,'redirect','queue',?,'queued',?,?)",
            (
                row["scope"],
                row["command_id"],
                row["conversation_id"],
                row["fingerprint"],
                row["payload_json"],
                row["execution_id"],
                tail,
                row["recorded_at"],
            ),
        )
        conn.execute(
            "UPDATE native_redirect_state SET release_basis_json=?,released_at=?,updated_at=? WHERE scope=? AND command_id=?",
            (canonical(basis), now, now, row["scope"], row["command_id"]),
        )
        self._native_control_event(conn, row, "native_release")
        self._native_control_event(conn, row, "input_queued")
        return True

    def native_redirect_progress(self, scope, command_id, *, frontier, release=True):
        if frontier not in {"closed", "uncertain"}:
            raise ValueError("invalid dispatch frontier")

        def write(conn):
            row = self._native_control_row(conn, scope, command_id)
            if not row or row["kind"] != "redirect":
                raise LookupError("redirect unavailable")
            return self._native_redirect_update(
                conn, row, frontier=frontier, release=release
            )

        return self._execute_write(write)

    def native_redirect_pending(self, *, after=0, limit=100):
        if (
            type(after) is not int
            or after < 0
            or type(limit) is not int
            or not 1 <= limit <= 100
        ):
            raise ValueError("invalid redirect page")
        with self._read_ctx() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    "SELECT c.* FROM native_control_commands c JOIN native_redirect_state r ON r.scope=c.scope AND r.command_id=c.command_id WHERE c.ordinal>? AND r.release_basis_json IS NULL ORDER BY c.ordinal LIMIT ?",
                    (after, limit),
                )
            ]

    def _native_control_snapshot(self, conn, scope, command_id, *, after=0, limit=0):
        row = self._native_control_row(conn, scope, command_id)
        if row is None:
            return None
        result = dict(command=row)
        if row["kind"] == "redirect":
            result.update(
                redirect=dict(
                    conn.execute(
                        "SELECT * FROM native_redirect_state WHERE scope=? AND command_id=?",
                        (scope, command_id),
                    ).fetchone()
                ),
                cancellation=self._native_cancel_snapshot_from_conn(
                    conn, scope, command_id, after=after, limit=limit
                ),
                native_released=self._native_redirect_released(conn, row),
            )
            input_row = conn.execute(
                "SELECT * FROM native_commands WHERE scope=? AND command_id=?",
                (scope, command_id),
            ).fetchone()
            result["input"] = dict(input_row) if input_row else None
        elif row["kind"] == "clarification_response":
            question = conn.execute(
                "SELECT * FROM native_clarifications WHERE clarification_id=? AND question_revision=?",
                (row["reference_id"], row["reference_revision"]),
            ).fetchone()
            result["question"] = dict(question) if question else None
        return result

    def native_control_snapshot(self, scope, command_id, *, after=0, limit=50):
        if (
            type(limit) is not int
            or not 0 <= limit <= 100
            or type(after) is not int
            or after < 0
        ):
            raise ValueError("invalid control page")
        return self._execute_write(
            lambda conn: self._native_control_snapshot(
                conn, scope, command_id, after=after, limit=limit
            )
        )
