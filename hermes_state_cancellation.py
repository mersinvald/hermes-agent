"""Execution-scoped control evidence in the existing SessionDB transaction domain."""

import hashlib
import hmac
import json
import time
from uuid import uuid4

from hermes_state_commands import CommandConflict, canonical


def fingerprint(body):
    return hashlib.sha256(
        ("native-command-v1\n" + canonical(body)).encode()
    ).hexdigest()


def binding_key(binding):
    fields = (
        "peer_name",
        "configured_endpoint_fingerprint",
        "rpc_endpoint",
        "protocol_version",
        "tenant",
        "configured_tenant",
    )
    return hashlib.sha256(
        canonical({key: binding[key] for key in fields}).encode()
    ).hexdigest()


def _latest_task_owner(conn, row):
    return conn.execute(
        "SELECT * FROM native_remote_dispatches WHERE binding_key=? AND task_id=? ORDER BY ordinal DESC LIMIT 1",
        (row["binding_key"], row["task_id"]),
    ).fetchone()


class NativeCancellationStateMixin:
    def _native_remote_rows(self, conn, execution_id, scope, command_id, after, limit):
        # Prefer this command's own attempt, retaining earlier failed history.
        # Otherwise expose only a prior reserved write as shared observation.
        return conn.execute(
            "SELECT d.*,a.request_state AS cancel_request_state,a.reason AS cancel_reason,a.scope AS cancel_scope,a.command_id AS cancel_command_id,a.write_reserved FROM native_remote_dispatches d LEFT JOIN native_remote_cancel_attempts a ON a.rowid=COALESCE((SELECT own.rowid FROM native_remote_cancel_attempts own WHERE own.dispatch_id=d.dispatch_id AND own.scope=:scope AND own.command_id=:command),(SELECT prior.rowid FROM native_remote_cancel_attempts prior JOIN native_remote_dispatches pd USING(dispatch_id) WHERE prior.write_reserved=1 AND pd.binding_key=d.binding_key AND pd.task_id=d.task_id ORDER BY prior.rowid DESC LIMIT 1)) WHERE d.execution_id=:execution AND d.ordinal>:after ORDER BY d.ordinal LIMIT :limit",
            dict(
                scope=scope,
                command=command_id,
                execution=execution_id,
                after=after,
                limit=limit,
            ),
        ).fetchall()

    def _native_cancel_event(self, conn, scope, command_id, change, dispatch_id=None):
        row = conn.execute(
            "SELECT * FROM native_cancel_commands WHERE scope=? AND command_id=?",
            (scope, command_id),
        ).fetchone()
        if row is None:
            return
        payload = dict(receipt_kind="cancel", command_id=command_id, change=change)
        if dispatch_id is not None:
            payload["dispatch_id"] = dispatch_id
        self._native_event_append(
            conn,
            row["conversation_id"],
            row["execution_id"],
            "cancel_state_changed",
            payload,
            scope=scope,
        )

    def native_cancel_lookup(self, scope, command_id):
        with self._read_ctx() as conn:
            row = conn.execute(
                "SELECT * FROM native_cancel_commands WHERE scope=? AND command_id=?",
                (scope, command_id),
            ).fetchone()
            return dict(row) if row else None

    def native_cancel_admit(self, scope, body):
        digest = fingerprint(body)

        def write(conn):
            old = conn.execute(
                "SELECT * FROM native_cancel_commands WHERE scope=? AND command_id=?",
                (scope, body["command_id"]),
            ).fetchone()
            if old:
                if not hmac.compare_digest(old["fingerprint"], digest):
                    raise CommandConflict("command ID reused with a different payload")
                return dict(old), False
            if conn.execute(
                "SELECT 1 FROM native_commands WHERE scope=? AND command_id=?",
                (scope, body["command_id"]),
            ).fetchone():
                raise CommandConflict("command ID reused with a different payload")
            execution = conn.execute(
                "SELECT * FROM native_executions WHERE conversation_id=? AND execution_id=?",
                (body["conversation_id"], body["target_execution_id"]),
            ).fetchone()
            if not execution:
                raise CommandConflict("execution unavailable")
            conn.execute(
                "INSERT INTO native_cancel_commands(scope,command_id,conversation_id,execution_id,owner,fingerprint,payload_json,native_request_state,recorded_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    scope,
                    body["command_id"],
                    body["conversation_id"],
                    execution["execution_id"],
                    execution["owner"],
                    digest,
                    canonical(body),
                    "pending" if execution["state"] == "open" else "not_running",
                    time.time(),
                ),
            )
            row = conn.execute(
                "SELECT * FROM native_cancel_commands WHERE scope=? AND command_id=?",
                (scope, body["command_id"]),
            ).fetchone()
            self._native_cancel_event(conn, scope, body["command_id"], "accepted")
            return dict(row), True

        return self._execute_write(write)

    def native_cancel_note_native(self, scope, command_id, state):
        if state not in {"requested", "not_running", "unknown", "failed"}:
            raise ValueError("invalid native request state")

        def write(conn):
            changed = conn.execute(
                "UPDATE native_cancel_commands SET native_request_state=? WHERE scope=? AND command_id=? AND native_request_state<>?",
                (state, scope, command_id, state),
            ).rowcount
            if changed:
                self._native_cancel_event(conn, scope, command_id, "native_request")

        self._execute_write(write)

    def native_cancel_commands(self, *, after=0, limit=100):
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("invalid cancel batch limit")
        with self._read_ctx() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM native_cancel_commands WHERE ordinal>? ORDER BY ordinal LIMIT ?",
                    (after, limit),
                )
            ]

    def native_remote_dispatch_prepare(
        self, origin, binding, request_id, *, task_id=None, context_id=None
    ):
        """Reserve before outbound I/O. Lost responses retain attempted coverage.

        A known continuation may be claimed only from actual recorded evidence
        in this conversation. Latest ordinal is the exact task-owner fence.
        """
        values = (
            dict(binding)
            if binding
            else dict.fromkeys((
                "peer_name",
                "configured_endpoint_fingerprint",
                "rpc_endpoint",
                "protocol_version",
                "tenant",
                "configured_tenant",
            ))
        )
        key = binding_key(values) if binding else None

        def write(conn):
            execution = conn.execute(
                "SELECT 1 FROM native_executions WHERE conversation_id=? AND execution_id=? AND owner=?",
                (origin.conversation_id, origin.execution_id, origin.owner),
            ).fetchone()
            if not execution:
                raise CommandConflict("execution provenance unavailable")
            # A cancellation accepted before dispatch prevents new remote work;
            # no I/O has started, so this is a real not-sent outcome.
            if conn.execute(
                "SELECT 1 FROM native_cancel_commands WHERE execution_id=? LIMIT 1",
                (origin.execution_id,),
            ).fetchone():
                raise CommandConflict("execution cancellation requested")
            if task_id:
                known = conn.execute(
                    "SELECT * FROM native_remote_dispatches WHERE binding_key=? AND task_id=? ORDER BY ordinal DESC LIMIT 1",
                    (key, task_id),
                ).fetchone()
                if (
                    not known
                    or known["conversation_id"] != origin.conversation_id
                    or known["context_id"] != context_id
                ):
                    raise CommandConflict("remote task ownership unavailable")
                # Resuming a task whose cancellation was already attempted is
                # ambiguous; never revive it by writing a new continuation.
                if conn.execute(
                    "SELECT 1 FROM native_remote_cancel_attempts a JOIN native_remote_dispatches d USING(dispatch_id) WHERE d.binding_key=? AND d.task_id=? AND a.write_reserved=1 LIMIT 1",
                    (key, task_id),
                ).fetchone():
                    raise CommandConflict("remote task cancellation already requested")
            now, identity = time.time(), uuid4().hex
            fields = (
                "peer_name",
                "configured_endpoint_fingerprint",
                "rpc_endpoint",
                "protocol_version",
                "tenant",
                "configured_tenant",
            )
            conn.execute(
                "INSERT INTO native_remote_dispatches(dispatch_id,conversation_id,execution_id,owner,binding_key,peer_name,configured_endpoint_fingerprint,rpc_endpoint,protocol_version,tenant,configured_tenant,request_id,phase,task_id,context_id,recorded_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    identity,
                    origin.conversation_id,
                    origin.execution_id,
                    origin.owner,
                    key,
                    *(values[k] for k in fields),
                    request_id,
                    "attempted",
                    task_id,
                    context_id if task_id else None,
                    now,
                    now,
                ),
            )
            return identity

        return self._execute_write(write)

    def native_remote_dispatch_observe(
        self, origin, dispatch_id, task_id, context_id, state
    ):
        from plugins.platforms.a2a.cancellation import bounded_identifier, TASK_STATES

        if (
            not bounded_identifier(task_id)
            or not bounded_identifier(context_id)
            or state not in TASK_STATES
        ):
            raise ValueError("invalid remote task evidence")

        def write(conn):
            row = conn.execute(
                "SELECT * FROM native_remote_dispatches WHERE dispatch_id=? AND conversation_id=? AND execution_id=? AND owner=?",
                (
                    dispatch_id,
                    origin.conversation_id,
                    origin.execution_id,
                    origin.owner,
                ),
            ).fetchone()
            if not row:
                raise CommandConflict("dispatch unavailable")
            if row["task_id"] and (
                row["task_id"] != task_id or row["context_id"] != context_id
            ):
                raise CommandConflict("task context changed")
            foreign = conn.execute(
                "SELECT 1 FROM native_remote_dispatches WHERE binding_key=? AND task_id=? AND (conversation_id!=? OR context_id!=?) LIMIT 1",
                (row["binding_key"], task_id, origin.conversation_id, context_id),
            ).fetchone()
            if foreign:
                raise CommandConflict("remote task ownership unavailable")
            conn.execute(
                "UPDATE native_remote_dispatches SET task_id=?,context_id=?,observed_state=?,phase='observed',updated_at=? WHERE dispatch_id=?",
                (task_id, context_id, state, time.time(), dispatch_id),
            )

        self._execute_write(write)

    def native_remote_targets(
        self, execution_id, *, after=0, limit=100, scope=None, command_id=None
    ):
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("invalid remote batch limit")
        with self._read_ctx() as conn:
            if scope is None:
                latest = conn.execute(
                    "SELECT scope,command_id FROM native_cancel_commands WHERE execution_id=? ORDER BY ordinal DESC LIMIT 1",
                    (execution_id,),
                ).fetchone()
                if latest:
                    scope, command_id = latest
            return [
                dict(row)
                for row in self._native_remote_rows(
                    conn, execution_id, scope, command_id, after, limit
                )
            ]

    def native_remote_cancel_reserve(self, scope, command_id, dispatch_id):
        """One cancellation write for this exact task claim, across commands.

        The initial state is unknown BEFORE I/O: a process exit after reservation
        cannot distinguish a not-yet-sent request from a lost acknowledgement.
        Recovery performs reads only for every reserved attempt.
        """

        def write(conn):
            command = conn.execute(
                "SELECT * FROM native_cancel_commands WHERE scope=? AND command_id=?",
                (scope, command_id),
            ).fetchone()
            target = conn.execute(
                "SELECT * FROM native_remote_dispatches WHERE dispatch_id=?",
                (dispatch_id,),
            ).fetchone()
            if (
                not command
                or not target
                or (
                    command["conversation_id"],
                    command["execution_id"],
                    command["owner"],
                )
                != (target["conversation_id"], target["execution_id"], target["owner"])
                or not target["task_id"]
                or not target["binding_key"]
            ):
                raise CommandConflict("cancellation target unavailable")
            current = _latest_task_owner(conn, target)
            if not current or (
                current["execution_id"],
                current["owner"],
                current["context_id"],
            ) != (target["execution_id"], target["owner"], target["context_id"]):
                raise CommandConflict("remote task owner superseded")
            if conn.execute(
                "SELECT 1 FROM native_remote_cancel_attempts a JOIN native_remote_dispatches d USING(dispatch_id) WHERE d.binding_key=? AND d.task_id=? AND a.write_reserved=1 LIMIT 1",
                (target["binding_key"], target["task_id"]),
            ).fetchone():
                raise CommandConflict("remote cancellation already reserved")
            if conn.execute(
                "SELECT 1 FROM native_remote_cancel_attempts WHERE dispatch_id=? AND scope=? AND command_id=?",
                (dispatch_id, scope, command_id),
            ).fetchone():
                raise CommandConflict("command cancellation already attempted")
            now = time.time()
            conn.execute(
                "INSERT INTO native_remote_cancel_attempts VALUES (?,?,?,?,?,?,?,?)",
                (
                    dispatch_id,
                    scope,
                    command_id,
                    "unknown",
                    "attempt_reserved",
                    now,
                    now,
                    1,
                ),
            )

            self._native_cancel_event(
                conn, scope, command_id, "remote_request", dispatch_id
            )

        self._execute_write(write)

    def native_remote_cancel_note(
        self, dispatch_id, observation, *, update_request=True
    ):
        def write(conn):
            before = conn.execute(
                "SELECT a.scope,a.command_id,a.request_state,d.observed_state FROM native_remote_dispatches d JOIN native_remote_cancel_attempts a ON a.rowid=COALESCE((SELECT own.rowid FROM native_remote_cancel_attempts own WHERE own.dispatch_id=d.dispatch_id ORDER BY own.rowid DESC LIMIT 1),(SELECT prior.rowid FROM native_remote_cancel_attempts prior JOIN native_remote_dispatches pd USING(dispatch_id) WHERE pd.binding_key=d.binding_key AND pd.task_id=d.task_id AND prior.write_reserved=1 ORDER BY prior.rowid DESC LIMIT 1)) WHERE d.dispatch_id=?",
                (dispatch_id,),
            ).fetchone()
            if update_request:
                conn.execute(
                    "UPDATE native_remote_cancel_attempts SET request_state=?,reason=?,updated_at=? WHERE rowid=(SELECT rowid FROM native_remote_cancel_attempts WHERE dispatch_id=? ORDER BY rowid DESC LIMIT 1)",
                    (
                        observation.cancel_state,
                        observation.reason,
                        time.time(),
                        dispatch_id,
                    ),
                )
            if observation.task_state != "unknown":
                conn.execute(
                    "UPDATE native_remote_dispatches SET observed_state=?,updated_at=? WHERE dispatch_id=?",
                    (observation.task_state, time.time(), dispatch_id),
                )

            if before and (
                (update_request and before["request_state"] != observation.cancel_state)
                or (
                    observation.task_state != "unknown"
                    and before["observed_state"] != observation.task_state
                )
            ):
                self._native_cancel_event(
                    conn,
                    before["scope"],
                    before["command_id"],
                    "remote_request" if update_request else "remote_observation",
                    dispatch_id,
                )

        self._execute_write(write)

    def native_cancel_snapshot(self, scope, command_id, *, after=0, limit=50):
        if (
            type(limit) is not int
            or not 1 <= limit <= 100
            or type(after) is not int
            or after < 0
        ):
            raise ValueError("invalid remote page")

        return self._execute_write(
            lambda conn: self._native_cancel_snapshot_from_conn(
                conn, scope, command_id, after=after, limit=limit
            )
        )

    def _native_cancel_snapshot_from_conn(
        self, conn, scope, command_id, *, after=0, limit=0
    ):
        # Recovery intentionally captures no target details, but whole counts.
        command = conn.execute(
            "SELECT * FROM native_cancel_commands WHERE scope=? AND command_id=?",
            (scope, command_id),
        ).fetchone()
        if not command:
            return None
        execution = conn.execute(
            "SELECT * FROM native_executions WHERE execution_id=? AND owner=? AND conversation_id=?",
            (command["execution_id"], command["owner"], command["conversation_id"]),
        ).fetchone()
        counts = conn.execute(
            "SELECT COUNT(*),COALESCE(SUM(task_id IS NULL),0) FROM native_remote_dispatches WHERE execution_id=?",
            (command["execution_id"],),
        ).fetchone()
        rows = self._native_remote_rows(
            conn, command["execution_id"], scope, command_id, after, limit + 1
        )
        return dict(
            command=dict(command),
            execution=dict(execution) if execution else None,
            targets=[dict(row) for row in rows[:limit]],
            known_count=counts[0],
            task_id_unknown_count=counts[1],
            has_more=len(rows) > limit,
        )

    def native_execution_cancel_requested(self, execution_id):
        with self._read_ctx() as conn:
            return bool(
                conn.execute(
                    "SELECT 1 FROM native_cancel_commands WHERE execution_id=? LIMIT 1",
                    (execution_id,),
                ).fetchone()
            )

    def native_cancel_for_execution(self, execution_id):
        with self._read_ctx() as conn:
            row = conn.execute(
                "SELECT * FROM native_cancel_commands WHERE execution_id=? ORDER BY ordinal DESC LIMIT 1",
                (execution_id,),
            ).fetchone()
            return dict(row) if row else None

    def native_cancel_recovery_rows(self, *, after=0, limit=100):
        with self._read_ctx() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    "SELECT c.* FROM native_cancel_commands c WHERE c.ordinal>? AND c.next_poll_at<=? AND NOT EXISTS(SELECT 1 FROM native_cancel_commands newer WHERE newer.execution_id=c.execution_id AND newer.ordinal>c.ordinal) AND (c.native_request_state='pending' OR EXISTS (SELECT 1 FROM native_remote_dispatches d WHERE d.execution_id=c.execution_id AND d.task_id IS NOT NULL AND d.observed_state NOT IN ('completed','canceled','failed','rejected') AND NOT EXISTS (SELECT 1 FROM native_remote_cancel_attempts a WHERE a.dispatch_id=d.dispatch_id AND a.scope=c.scope AND a.command_id=c.command_id AND a.write_reserved=0 AND a.request_state IN ('failed','rejected','not_needed')))) ORDER BY c.ordinal LIMIT ?",
                    (after, time.time(), limit),
                )
            ]

    def native_remote_cancel_no_write(
        self, scope, command_id, dispatch_id, observation
    ):
        state = {"confirmed": "not_needed", "already_completed": "not_needed"}.get(
            observation.cancel_state, observation.cancel_state
        )

        def write(conn):
            command = conn.execute(
                "SELECT * FROM native_cancel_commands WHERE scope=? AND command_id=?",
                (scope, command_id),
            ).fetchone()
            target = conn.execute(
                "SELECT * FROM native_remote_dispatches WHERE dispatch_id=?",
                (dispatch_id,),
            ).fetchone()
            if (
                not command
                or not target
                or command["execution_id"] != target["execution_id"]
                or command["owner"] != target["owner"]
            ):
                raise CommandConflict("cancellation target unavailable")
            now = time.time()
            changed = conn.execute(
                "INSERT INTO native_remote_cancel_attempts VALUES (?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING",
                (
                    dispatch_id,
                    scope,
                    command_id,
                    state,
                    observation.reason,
                    now,
                    now,
                    0,
                ),
            ).rowcount
            if observation.task_state != "unknown":
                conn.execute(
                    "UPDATE native_remote_dispatches SET observed_state=?,updated_at=? WHERE dispatch_id=?",
                    (observation.task_state, now, dispatch_id),
                )

            if changed:
                self._native_cancel_event(
                    conn, scope, command_id, "remote_request", dispatch_id
                )

        self._execute_write(write)

    def native_remote_task_owned(self, root, binding, task_id, context_id):
        with self._read_ctx() as conn:
            row = conn.execute(
                "SELECT conversation_id,context_id FROM native_remote_dispatches WHERE binding_key=? AND task_id=? ORDER BY ordinal DESC LIMIT 1",
                (binding_key(binding), task_id),
            ).fetchone()
            return bool(
                row
                and row["conversation_id"] == root
                and row["context_id"] == context_id
            )

    def native_cancel_defer(self, scope, command_id, seconds, after):
        self._execute_write(
            lambda c: c.execute(
                "UPDATE native_cancel_commands SET next_poll_at=?,remote_after=? WHERE scope=? AND command_id=?",
                (time.time() + seconds, after, scope, command_id),
            )
        )
