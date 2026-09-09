"""Bounded native replay; journal facts and events share SessionDB transactions.

Event detail expires. Native execution rows, command receipts and transcript rows
remain the recovery authority. Nothing in this module dispatches agent work.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from hermes_state_commands import canonical, receipt


def timestamp(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


@dataclass(frozen=True)
class EventLimits:
    max_count: int = 4096
    max_bytes: int = 8 * 1024 * 1024
    max_age_seconds: int = 86400
    max_event_bytes: int = 16384
    batch_count: int = 100
    snapshot_count: int = 50
    max_subscribers: int = 64

    def __post_init__(self):
        for value in self.__dict__.values():
            if type(value) is not int or value < 1:
                raise ValueError("event limits must be positive integers")
        if self.max_event_bytes > self.max_bytes:
            raise ValueError("event size exceeds retention byte limit")
        if self.batch_count > 100 or self.snapshot_count > 100:
            raise ValueError("wire batches are limited to 100 records")


class NativeEventStateMixin:
    def native_events_enable(self, limits=None):
        # One service epoch per installation; restarting never claims continuity
        # of non-durable worker callbacks. Do not replace an already live epoch.
        if not getattr(self, "_native_events_epoch", None):
            self._native_events_limits = limits or EventLimits()
            self._native_events_epoch = uuid4().hex
            self._native_events_overrides = {}

    def native_event_mark_gap(self, root):
        # Called outside write callbacks (including after a failed transaction).
        # Share SessionDB's writer lock: observation loss must linearize before
        # or after a canonical capture, never in the middle of its boundary.
        with self._lock:
            self._native_events_overrides[root] = uuid4().hex

    def native_event_epoch(self, root):
        if root in self._native_events_overrides:
            return self._native_events_overrides[root]
        return hashlib.sha256(
            (self._native_events_epoch + "\0" + root).encode()
        ).hexdigest()

    def _native_event_prune(self, conn, now):
        limits = self._native_events_limits
        rows = conn.execute(
            "SELECT ordinal,conversation_id,epoch,sequence,occurred_at,byte_count "
            "FROM native_events ORDER BY ordinal DESC"
        ).fetchall()
        count = size = 0
        victims = []
        for row in rows:
            count += 1
            size += row["byte_count"]
            if (
                count > limits.max_count
                or size > limits.max_bytes
                or row["occurred_at"] < now - limits.max_age_seconds
            ):
                victims.append(row)
        for row in victims:
            conn.execute(
                "UPDATE native_event_heads SET evicted_through=MAX(evicted_through,?) "
                "WHERE conversation_id=? AND epoch=?",
                (row["sequence"], row["conversation_id"], row["epoch"]),
            )
            conn.execute("DELETE FROM native_events WHERE ordinal=?", (row["ordinal"],))
        # Historical epoch heads without retained detail have no further role.
        for row in conn.execute(
            "SELECT conversation_id,epoch FROM native_event_heads AS h WHERE NOT EXISTS "
            "(SELECT 1 FROM native_events e WHERE e.conversation_id=h.conversation_id AND e.epoch=h.epoch)"
        ).fetchall():
            if row["epoch"] != self.native_event_epoch(row["conversation_id"]):
                conn.execute(
                    "DELETE FROM native_event_heads WHERE conversation_id=? AND epoch=?",
                    tuple(row),
                )

    def _native_event_append(
        self, conn, root, execution_id, kind, payload, *, scope=None
    ):
        if not getattr(self, "_native_events_epoch", None):
            return  # Opt-in native ingress only; legacy native behavior preserved.
        now = time.time()
        epoch = self.native_event_epoch(root)
        conn.execute(
            "INSERT INTO native_event_heads(conversation_id,epoch) VALUES (?,?) ON CONFLICT DO NOTHING",
            (root, epoch),
        )
        conn.execute(
            "UPDATE native_event_heads SET sequence=sequence+1 WHERE conversation_id=? AND epoch=?",
            (root, epoch),
        )
        sequence = conn.execute(
            "SELECT sequence FROM native_event_heads WHERE conversation_id=? AND epoch=?",
            (root, epoch),
        ).fetchone()[0]
        event = dict(
            schema_version="1.0",
            event_id=uuid4().hex,
            epoch=epoch,
            sequence=sequence,
            conversation_id=root,
            execution_id=execution_id,
            type=kind,
            occurred_at=timestamp(now),
            payload=payload,
        )
        body = canonical(event)
        size = len(body.encode())
        if size > self._native_events_limits.max_event_bytes:
            raise ValueError("native event exceeds configured byte limit")
        conn.execute(
            "INSERT INTO native_events(conversation_id,epoch,sequence,event_id,principal_scope,occurred_at,body,byte_count) VALUES (?,?,?,?,?,?,?,?)",
            (root, epoch, sequence, event["event_id"], scope, now, body, size),
        )
        self._native_event_prune(conn, now)

    def _native_command_event(self, conn, scope, command_id):
        row = conn.execute(
            "SELECT * FROM native_commands WHERE scope=? AND command_id=?",
            (scope, command_id),
        ).fetchone()
        control = self._native_control_row(conn, scope, command_id)
        if control:
            if control["kind"] != "redirect" or row["requested_action"] != "redirect":
                raise ValueError("inconsistent internal control input phase")
            change = {
                "applied": "input_applied", "not_applied": "input_not_applied",
                "unknown": "input_unknown",
            }.get(row["phase"], "input_queued")
            self._native_control_event(conn, control, change)
            return
        view = receipt(row)
        self._native_event_append(
            conn,
            row["conversation_id"],
            row["resulting_execution_id"],
            "command_application_changed",
            dict(
                command_id=command_id,
                application_state=view["application_state"],
                receipt=view,
            ),
            scope=scope,
        )

    def native_activity_event(self, execution_id, owner, kind, payload):
        def write(conn):
            row = conn.execute(
                "SELECT * FROM native_executions WHERE execution_id=? AND owner=? AND state='open'",
                (execution_id, owner),
            ).fetchone()
            if row:
                self._native_event_append(
                    conn, row["conversation_id"], execution_id, kind, payload
                )

        self._execute_write(write)

    def native_event_recovery(
        self, root, scope, cursor=None, *, delivery_channel_key=None, clarification_available=frozenset()
    ):
        """Capture journal state and replay boundary in one SQLite transaction.

        A bounded native history API, not this snapshot, pages older transcripts.
        Old-epoch details are returned only when that exact scoped epoch is asked
        for; the gap response's cursor points at the new epoch's snapshot boundary.
        """
        limits = self._native_events_limits

        def read(conn):
            # Capture only after _execute_write owns the writer lock and SQLite
            # transaction; an observer may have rotated while we waited.
            epoch = self.native_event_epoch(root)
            now = time.time()
            self._native_event_prune(conn, now)
            head = conn.execute(
                "SELECT * FROM native_event_heads WHERE conversation_id=? AND epoch=?",
                (root, epoch),
            ).fetchone()
            high = head["sequence"] if head else 0
            result_cursor = dict(epoch=epoch, sequence=high)
            status, reason, events = "current", None, []
            if cursor is not None:
                if (
                    not isinstance(cursor, dict)
                    or set(cursor) - {"epoch", "sequence", "event_id"}
                    or not isinstance(cursor.get("epoch"), str)
                    or len(cursor["epoch"]) > 256
                    or type(cursor.get("sequence")) is not int
                    or cursor["sequence"] < 0
                    or (
                        "event_id" in cursor
                        and (
                            not isinstance(cursor["event_id"], str)
                            or len(cursor["event_id"]) > 256
                        )
                    )
                ):
                    status, reason = "gap", "invalid_cursor"
                else:
                    requested = conn.execute(
                        "SELECT * FROM native_event_heads WHERE conversation_id=? AND epoch=?",
                        (root, cursor["epoch"]),
                    ).fetchone()
                    if cursor["epoch"] != epoch:
                        status, reason = "gap", "epoch_changed_or_cursor_unavailable"
                    elif cursor["sequence"] > high:
                        status, reason = "gap", "cursor_ahead"
                    elif (
                        requested and cursor["sequence"] < requested["evicted_through"]
                    ):
                        status, reason = "expired", "retention_evicted_or_slow_consumer"
                    elif "event_id" in cursor:
                        matching = conn.execute(
                            "SELECT event_id FROM native_events WHERE conversation_id=? AND epoch=? AND sequence=?",
                            (root, epoch, cursor["sequence"]),
                        ).fetchone()
                        if not matching or matching[0] != cursor["event_id"]:
                            status, reason = "gap", "cursor_event_unavailable"
                    if requested and cursor["sequence"] <= requested["sequence"]:
                        inspected = conn.execute(
                            "SELECT body,principal_scope,sequence,event_id FROM native_events WHERE conversation_id=? AND epoch=? AND sequence>? ORDER BY sequence LIMIT ?",
                            (
                                root,
                                cursor["epoch"],
                                cursor["sequence"],
                                limits.batch_count,
                            ),
                        ).fetchall()
                        events = [
                            json.loads(r["body"])
                            for r in inspected
                            if r["principal_scope"] is None
                            or r["principal_scope"] == scope
                        ]
                        if status == "current" and inspected:
                            # Private commands of another explicit grantee are
                            # inspected but not exposed. Such positions are not gaps.
                            result_cursor = dict(
                                epoch=epoch, sequence=inspected[-1]["sequence"]
                            )

            executions = [
                dict(r)
                for r in conn.execute(
                    "SELECT e.*,m.model_id,m.model_version,"
                    "EXISTS(SELECT 1 FROM native_cancel_commands c WHERE c.execution_id=e.execution_id) AS cancel_requested "
                    "FROM native_executions e LEFT JOIN native_execution_models m "
                    "ON m.execution_id=e.execution_id WHERE e.conversation_id=? "
                    "ORDER BY e.created_order DESC LIMIT ?",
                    (root, limits.snapshot_count + 1),
                )
            ]
            for row in executions[: limits.snapshot_count]:
                self._native_execution_delivery(conn, row, delivery_channel_key)
            commands = [
                receipt(r)
                for r in conn.execute(
                    "SELECT * FROM native_commands WHERE conversation_id=? AND scope=? AND requested_action IN ('send','steer','queue') ORDER BY ordinal DESC LIMIT ?",
                    (root, scope, limits.snapshot_count + 1),
                )
            ]
            control_ids = conn.execute(
                "SELECT c.command_id,c.recorded_at FROM native_cancel_commands c WHERE c.conversation_id=? AND c.scope=? AND NOT EXISTS(SELECT 1 FROM native_control_commands n WHERE n.scope=c.scope AND n.command_id=c.command_id) UNION ALL SELECT command_id,recorded_at FROM native_control_commands WHERE conversation_id=? AND scope=? ORDER BY recorded_at DESC,command_id DESC LIMIT 11",
                (root, scope, root, scope),
            ).fetchall()
            controls = [self._native_control_snapshot(conn, scope, row[0])
                        or self._native_cancel_snapshot_from_conn(conn, scope, row[0])
                        for row in control_ids[:10]]
            result = dict(
                schema_version="1.0",
                status=status,
                cursor=result_cursor,
                events=events,
                captured_at=timestamp(now),
                executions=executions[: limits.snapshot_count],
                command_receipts=commands[: limits.snapshot_count],
                control_receipts=controls,
                question_epoch=epoch,
                question_page=self._native_question_page(conn, root, available=clarification_available),
                control_receipt_coverage=dict(limit=10, has_more=len(control_ids)>10,
                                              target_details="command_lookup"),
                coverage=dict(
                    executions_has_more=len(executions) > limits.snapshot_count,
                    commands_has_more=len(commands) > limits.snapshot_count,
                ),
            )
            if reason:
                result["detail"] = dict(reason=reason)
            return result

        # Retention maintenance and the canonical capture use the existing short
        # write transaction, which excludes concurrent journal transitions.
        return self._execute_write(read)

    @staticmethod
    def _native_execution_delivery(conn, row, channel_key):
        if channel_key is None:
            return
        delivery = conn.execute(
            "SELECT state,binding_version FROM native_channel_deliveries "
            "WHERE conversation_id=? AND execution_id=? AND channel_key=?",
            (row["conversation_id"], row["execution_id"], channel_key),
        ).fetchone()
        if delivery:
            row["deliveries"] = [
                {
                    "channel": "telegram",
                    "state": "unknown"
                    if delivery["state"] == "attempting"
                    else delivery["state"],
                    "binding_version": delivery["binding_version"],
                }
            ]

    def native_event_execution(self, root, execution_id, *, delivery_channel_key=None):
        def read(conn):
            found = conn.execute(
                "SELECT e.*,m.model_id,m.model_version,"
                "EXISTS(SELECT 1 FROM native_cancel_commands c WHERE c.execution_id=e.execution_id) AS cancel_requested "
                "FROM native_executions e LEFT JOIN native_execution_models m "
                "ON m.execution_id=e.execution_id "
                "WHERE e.conversation_id=? AND e.execution_id=?",
                (root, execution_id),
            ).fetchone()
            if found is None:
                raise LookupError("native execution unavailable")
            row = dict(found)
            self._native_execution_delivery(conn, row, delivery_channel_key)
            return row

        return self._execute_write(read)
