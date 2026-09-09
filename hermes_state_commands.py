"""Native command journal operations, in SessionDB's transactions and lease domain.

No dispatcher, connection, or alternative database lives here. A committed input
is local application evidence, never evidence of provider or remote effects.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from datetime import datetime, timezone


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def receipt(row):
    return {
        "schema_version": "1.0",
        "command_id": row["command_id"],
        "conversation_id": row["conversation_id"],
        "payload_fingerprint": row["fingerprint"],
        "receipt_state": "accepted",
        "durability": "durable",
        "application_state": {
            "queued": "pending",
            "steer": "pending",
            "assigned": "pending",
        }.get(row["phase"], row["phase"]),
        "requested_action": row["requested_action"],
        "effective_action": row["effective_action"],
        "target_execution_id": row["target_execution_id"],
        "resulting_execution_id": row["resulting_execution_id"],
        "fallback": {"reason": "unconsumed_steer"} if row["fallback"] else None,
        "recorded_at": datetime.fromtimestamp(
            row["recorded_at"], timezone.utc
        ).isoformat(),
    }


def process_owner():
    import psutil

    return f"pid={os.getpid()}:birth={psutil.Process().create_time()}"


def process_owner_dead(owner):
    from hermes_state import _compression_lock_holder_process_is_dead

    if _compression_lock_holder_process_is_dead(owner):
        return True
    # PID reuse (including PID 1 after container replacement) must not turn
    # a dead journal owner into an immortal one. Unknown birth evidence stays
    # protected; never infer death merely from another gateway UUID.
    try:
        import psutil

        parts = dict(part.split("=", 1) for part in owner.split(":") if "=" in part)
        return psutil.Process(int(parts["pid"])).create_time() != float(parts["birth"])
    except (KeyError, ValueError, psutil.Error):
        return False


class CommandConflict(ValueError):
    pass


class NativeCommandStateMixin:
    def native_command_lookup(self, scope, command_id):
        with self._read_ctx() as conn:
            row = conn.execute(
                "SELECT * FROM native_commands WHERE scope=? AND command_id=?",
                (scope, command_id),
            ).fetchone()
            return dict(row) if row else None

    def native_command_rows(self, root, *, phase=None):
        with self._read_ctx() as conn:
            sql = "SELECT * FROM native_commands WHERE conversation_id=?"
            args = [root]
            if phase:
                sql += " AND phase=?"
                args.append(phase)
            return [
                dict(r)
                for r in conn.execute(sql + " ORDER BY queue_order, ordinal", args)
            ]

    def native_execution(self, root):
        with self._read_ctx() as conn:
            row = conn.execute(
                "SELECT * FROM native_executions WHERE conversation_id=? "
                "AND state='open'",
                (root,),
            ).fetchone()
            return dict(row) if row else None

    def native_command_admit(self, scope, body, *, effect, target=None, result=None):
        payload = canonical(body)
        fingerprint = hashlib.sha256(
            ("native-command-v1\n" + payload).encode()
        ).hexdigest()

        def write(conn):
            resolved_effect, resolved_target = effect, target
            self._native_control_cross_kind(conn, scope, body["command_id"])
            old = conn.execute(
                "SELECT * FROM native_commands WHERE scope=? AND command_id=?",
                (scope, body["command_id"]),
            ).fetchone()
            if old:
                if not hmac.compare_digest(old["fingerprint"], fingerprint):
                    raise CommandConflict("command ID reused with a different payload")
                return dict(old), False
            if conn.execute(
                "SELECT 1 FROM native_cancel_commands WHERE scope=? AND command_id=?",
                (scope, body["command_id"]),
            ).fetchone():
                raise CommandConflict("command ID reused with a different payload")
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
            # The owner can close in its worker thread while admission waits
            # for SQLite. Resolve that race in this same write transaction.
            active = conn.execute(
                "SELECT * FROM native_executions WHERE conversation_id=? "
                "AND state='open'",
                (body["conversation_id"],),
            ).fetchone()
            expected = body.get("target_execution_id")
            if expected and (not active or active["execution_id"] != expected):
                raise CommandConflict("target execution is no longer active")
            if body["type"] == "steer" and not active:
                raise CommandConflict("steer requires an active execution")
            if active:
                resolved_target = active["execution_id"]
                resolved_effect = (
                    "steer"
                    if body["type"] == "steer"
                    or (body["type"] == "send" and active["input_started"])
                    else "queue"
                )
            elif resolved_effect == "steer":
                resolved_effect = "queue"
            if (
                not active
                and conn.execute(
                    "SELECT 1 FROM native_commands WHERE conversation_id=? AND phase IN ('queued','assigned') LIMIT 1",
                    (body["conversation_id"],),
                ).fetchone()
            ):
                resolved_effect = "queue"
            phase = "steer" if resolved_effect == "steer" else "queued"
            resolved_result = resolved_target if phase == "steer" else None
            order = conn.execute(
                "SELECT COALESCE(MAX(queue_order),0)+1 FROM native_commands"
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO native_commands "
                "(scope,command_id,conversation_id,fingerprint,payload_json,requested_action,"
                "effective_action,target_execution_id,resulting_execution_id,phase,queue_order,recorded_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    scope,
                    body["command_id"],
                    body["conversation_id"],
                    fingerprint,
                    payload,
                    body["type"],
                    resolved_effect,
                    resolved_target,
                    resolved_result,
                    phase,
                    order,
                    time.time(),
                ),
            )
            self._native_command_event(conn, scope, body["command_id"])
            return dict(
                conn.execute(
                    "SELECT * FROM native_commands WHERE scope=? AND command_id=?",
                    (scope, body["command_id"]),
                ).fetchone()
            ), True

        return self._execute_write(write)

    def native_execution_open(
        self, root, execution_id, owner, *, command=None, origin="unknown"
    ):
        def write(conn):
            active = conn.execute(
                "SELECT execution_id,owner FROM native_executions "
                "WHERE conversation_id=? AND state='open'",
                (root,),
            ).fetchone()
            if active and (active["execution_id"], active["owner"]) != (
                execution_id,
                owner,
            ):
                raise CommandConflict("conversation already has an execution owner")
            if command:
                row = conn.execute(
                    "SELECT * FROM native_commands WHERE scope=? AND command_id=?",
                    command,
                ).fetchone()
                if (
                    not row
                    or row["phase"] != "queued"
                    or row["conversation_id"] != root
                ):
                    raise CommandConflict("command is not available for execution")
                if (
                    row["resulting_execution_id"]
                    and row["resulting_execution_id"] != execution_id
                ):
                    raise CommandConflict(
                        "command already has another execution identity"
                    )
                conn.execute(
                    "UPDATE native_commands SET phase='assigned',resulting_execution_id=? "
                    "WHERE scope=? AND command_id=?",
                    (execution_id, *command),
                )
            order = conn.execute(
                "SELECT COALESCE(MAX(created_order),0)+1 FROM native_executions"
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO native_executions (execution_id,conversation_id,owner,state,created_order) VALUES (?,?,?,'open',?) "
                "ON CONFLICT(execution_id) DO UPDATE SET owner=excluded.owner,state='open',created_order=excluded.created_order "
                "WHERE native_executions.state != 'open' AND native_executions.input_started=0",
                (execution_id, root, owner, order),
            )
            opened = conn.execute(
                "SELECT state,owner,conversation_id FROM native_executions WHERE execution_id=?",
                (execution_id,),
            ).fetchone()
            if not opened or tuple(opened) != ("open", owner, root):
                raise CommandConflict("execution identity cannot be reopened")
            now = time.time()
            conn.execute(
                "UPDATE native_executions SET origin=?,observed_state='starting',created_at=COALESCE(created_at,?),updated_at=? WHERE execution_id=? AND owner=?",
                (
                    origin
                    if origin in {"pwa", "telegram", "native", "system"}
                    else "unknown",
                    now,
                    now,
                    execution_id,
                    owner,
                ),
            )
            self._native_event_append(
                conn,
                root,
                execution_id,
                "execution_state_changed",
                {"state": "starting"},
            )
            if command:
                self._native_command_event(conn, *command)

        self._execute_write(write)

    def native_execution_close(
        self, execution_id, owner, *, crashed=False, recover=False, outcome=None
    ):
        if outcome not in {None, "completed", "failed", "interrupted", "cancelled"}:
            raise ValueError("invalid observed execution outcome")

        def write(conn):
            row = conn.execute(
                "SELECT * FROM native_executions WHERE execution_id=? AND owner=? AND state='open'",
                (execution_id, owner),
            ).fetchone()
            if not row:
                return False
            conn.execute(
                "UPDATE native_executions SET state=? WHERE execution_id=? AND owner=?",
                ("unknown" if crashed else "closed", execution_id, owner),
            )
            observed = outcome or "unknown"
            conn.execute(
                "UPDATE native_executions SET observed_state=?,updated_at=? WHERE execution_id=?",
                (observed, time.time(), execution_id),
            )
            self._native_event_append(
                conn,
                row["conversation_id"],
                execution_id,
                "execution_state_changed",
                {"state": observed},
            )
            changed = conn.execute(
                "SELECT scope,command_id FROM native_commands WHERE (target_execution_id=? AND phase='steer') OR (resulting_execution_id=? AND phase='assigned')",
                (execution_id, execution_id),
            ).fetchall()
            # Fallback is a phase transition on the original row. A repeated
            # close cannot allocate a second request or a new client identity.
            tail = conn.execute(
                "SELECT COALESCE(MAX(queue_order),0) FROM native_commands"
            ).fetchone()[0]
            pending = conn.execute(
                "SELECT scope,command_id FROM native_commands WHERE target_execution_id=? "
                "AND phase='steer' ORDER BY ordinal",
                (execution_id,),
            ).fetchall()
            for offset, cmd in enumerate(pending, 1):
                conn.execute(
                    "UPDATE native_commands SET phase='queued',effective_action='queue',fallback=1,"
                    "resulting_execution_id=NULL,queue_order=? WHERE scope=? AND command_id=?",
                    (tail + offset, cmd["scope"], cmd["command_id"]),
                )
            conn.execute(
                "UPDATE native_commands SET phase=? WHERE resulting_execution_id=? AND phase='assigned'",
                (
                    "queued" if recover and not row["input_started"] else "not_applied",
                    execution_id,
                ),
            )
            for cmd in changed:
                self._native_command_event(conn, *cmd)
            return True

        return self._execute_write(write)

    def native_execution_command_managed(self, root):
        with self._read_ctx() as conn:
            latest = conn.execute(
                "SELECT execution_id FROM native_executions WHERE conversation_id=? ORDER BY created_order DESC LIMIT 1",
                (root,),
            ).fetchone()
            if latest is None:
                return False
            return (
                conn.execute(
                    "SELECT 1 FROM native_commands WHERE conversation_id=? AND "
                    "(resulting_execution_id=? OR target_execution_id=?) LIMIT 1",
                    (root, latest["execution_id"], latest["execution_id"]),
                ).fetchone()
                is not None
            )

    def native_execution_has_lease(self, root):
        with self._read_ctx() as conn:
            return (
                conn.execute(
                    "SELECT 1 FROM session_turn_leases WHERE conversation_id=?", (root,)
                ).fetchone()
                is not None
            )

    def native_execution_reconcile(self, root):
        """Recover only a provably dead local owner; unknown liveness is protected."""
        from hermes_state import _compression_lock_holder_process_is_dead

        row = self.native_execution(root)
        if not row or not process_owner_dead(row["owner"]):
            return False
        # Do not override a different, live native writer merely because the
        # journal owner died. Native lease reclamation remains authoritative.
        with self._read_ctx() as conn:
            lease = conn.execute(
                "SELECT holder,expires_at FROM session_turn_leases WHERE conversation_id=?",
                (root,),
            ).fetchone()
        if (
            lease
            and lease["expires_at"] > time.time()
            and not _compression_lock_holder_process_is_dead(lease["holder"])
        ):
            return False
        return self.native_execution_close(
            row["execution_id"], row["owner"], crashed=True, recover=True
        )

    def native_command_refresh_input(
        self, execution_id, owner, session_id, holder, command, message
    ):
        """Retain the prologue's API sidecar after early atomic input admission."""

        def write(conn):
            row = conn.execute(
                "SELECT input_session_id,input_row_id FROM native_commands WHERE scope=? AND command_id=? AND phase='applied'",
                command,
            ).fetchone()
            if not row or (row["input_session_id"], row["input_row_id"]) != (
                session_id,
                message.get("_row_id"),
            ):
                return  # Compression's own copy/backfill path owns a new row.
            if not conn.execute(
                "SELECT 1 FROM native_executions WHERE execution_id=? AND owner=? AND state='open'",
                (execution_id, owner),
            ).fetchone():
                raise CommandConflict("execution owner changed")
            self._check_transcript_write_guards(
                conn, session_id, None, turn_lease_holder=holder
            )
            conn.execute(
                "UPDATE messages SET api_content=? WHERE id=? AND session_id=? AND active=1",
                (message.get("api_content"), row["input_row_id"], session_id),
            )

        self._execute_write(write)

    def native_command_apply(
        self, execution_id, owner, session_id, holder, message, *, command=None
    ):
        """Commit a user row or tool amendment together with command evidence.

        Callers publish returned content/markers only after commit. A provider
        must never run on an amendment whose transaction failed.
        """

        def write(conn):
            execution = conn.execute(
                "SELECT * FROM native_executions WHERE execution_id=? AND owner=? AND state='open'",
                (execution_id, owner),
            ).fetchone()
            if (
                not execution
                or self._session_turn_lease_key_on_conn(conn, session_id)
                != execution["conversation_id"]
            ):
                raise CommandConflict("execution owner changed")
            if not holder:
                raise RuntimeError("durable native input requires a native turn lease")
            self._check_transcript_write_guards(
                conn, session_id, None, turn_lease_holder=holder
            )
            if message["role"] == "user":
                rows = []
                if command:
                    row = conn.execute(
                        "SELECT * FROM native_commands WHERE scope=? AND command_id=?",
                        command,
                    ).fetchone()
                    if (
                        not row
                        or row["phase"] != "assigned"
                        or row["resulting_execution_id"] != execution_id
                    ):
                        raise CommandConflict(
                            "command input is already applied or unavailable"
                        )
                    rows = [row]
                # Ordinary native input uses its established persistence path;
                # only command-backed input is appended by this atomic seam.
                result = dict(message)
                if command:
                    inserted, tools = self._insert_message_rows(
                        conn, session_id, [result]
                    )
                    conn.execute(
                        "UPDATE sessions SET message_count=message_count+?,tool_call_count=tool_call_count+? WHERE id=?",
                        (inserted, tools, session_id),
                    )
                conn.execute(
                    "UPDATE native_executions SET input_started=1,lease_holder=?,observed_state='running',updated_at=? WHERE execution_id=? AND owner=?",
                    (holder, time.time(), execution_id, owner),
                )
                self._native_event_append(
                    conn,
                    execution["conversation_id"],
                    execution_id,
                    "execution_state_changed",
                    {"state": "running"},
                )
            else:
                rows = conn.execute(
                    "SELECT * FROM native_commands WHERE target_execution_id=? AND phase='steer' ORDER BY ordinal",
                    (execution_id,),
                ).fetchall()
                if not rows:
                    return None
                row_id = message.get("_row_id")
                target = conn.execute(
                    "SELECT role,active FROM messages WHERE id=? AND session_id=?",
                    (row_id, session_id),
                ).fetchone()
                if not target or target["role"] != "tool" or not target["active"]:
                    raise CommandConflict("current tool input is no longer available")
                from agent.prompt_builder import format_steer_marker

                marker = format_steer_marker(
                    "\n".join(
                        json.loads(r["payload_json"])["payload"]["text"] for r in rows
                    )
                )
                from agent.tool_dispatch_helpers import (
                    _is_multimodal_tool_result,
                    _multimodal_text_summary,
                    project_tool_content_for_storage,
                )

                content = message.get("content") or ""
                if isinstance(content, str):
                    content += marker
                elif _is_multimodal_tool_result(content):
                    content = {
                        **content,
                        "content": [
                            *content["content"],
                            {"type": "text", "text": marker},
                        ],
                        "text_summary": _multimodal_text_summary(content) + marker,
                    }
                elif isinstance(content, list):
                    content = [*content, {"type": "text", "text": marker}]
                else:
                    raise ValueError("unsupported native tool content")
                stored = project_tool_content_for_storage(content)
                conn.execute(
                    "UPDATE messages SET content=?,api_content=NULL WHERE id=? AND session_id=? AND active=1",
                    (self._encode_content(stored), row_id, session_id),
                )
                result = {**message, "content": content}
            for row in rows:
                conn.execute(
                    "UPDATE native_commands SET phase='applied',input_session_id=?,input_row_id=? "
                    "WHERE scope=? AND command_id=?",
                    (
                        session_id,
                        result.get("_row_id"),
                        row["scope"],
                        row["command_id"],
                    ),
                )
                self._native_command_event(conn, row["scope"], row["command_id"])
            return result

        return self._execute_write(write, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S)
