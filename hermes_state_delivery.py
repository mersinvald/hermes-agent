"""Final channel delivery reservations within the native SessionDB.

Rows contain routing evidence, never response bodies. An interrupted attempting
row is an unknown remote outcome and is never automatically retransmitted.
"""

import time


class NativeDeliveryStateMixin:
    def native_delivery_lookup(self, execution_id, channel_key):
        with self._read_ctx() as conn:
            row = conn.execute(
                "SELECT * FROM native_channel_deliveries WHERE execution_id=? AND channel_key=?",
                (execution_id, channel_key),
            ).fetchone()
            return dict(row) if row else None

    def native_delivery_reserve(self, execution_id, channel_key, root, version):
        def write(conn):
            cursor = conn.execute(
                "INSERT OR IGNORE INTO native_channel_deliveries "
                "(execution_id,channel_key,conversation_id,binding_version,state,recorded_at) "
                "VALUES (?,?,?,?,'attempting',?)",
                (execution_id, channel_key, root, version, time.time()),
            )
            row = conn.execute(
                "SELECT * FROM native_channel_deliveries WHERE execution_id=? AND channel_key=?",
                (execution_id, channel_key),
            ).fetchone()
            return dict(row), cursor.rowcount == 1

        return self._execute_write(write)

    def native_delivery_finish(self, execution_id, channel_key, state):
        if state not in {"delivered", "skipped", "unknown"}:
            raise ValueError("invalid delivery outcome")

        def write(conn):
            changed = (
                conn.execute(
                    "UPDATE native_channel_deliveries SET state=?,completed_at=? "
                    "WHERE execution_id=? AND channel_key=? AND state='attempting'",
                    (state, time.time(), execution_id, channel_key),
                ).rowcount
                == 1
            )
            if changed:
                return dict(
                    conn.execute(
                        "SELECT * FROM native_channel_deliveries WHERE execution_id=? AND channel_key=?",
                        (execution_id, channel_key),
                    ).fetchone()
                )
            return None

        row = self._execute_write(write)
        # The authoritative delivery CAS has already committed. Observation loss
        # must neither roll back its outcome nor trigger another remote send.
        if row and getattr(self, "_native_events_limits", None) is not None:
            try:
                self._execute_write(
                    lambda conn: self._native_event_append(
                        conn,
                        row["conversation_id"],
                        execution_id,
                        "delivery_changed",
                        {
                            "channel": "telegram",
                            "state": state,
                            "binding_version": row["binding_version"],
                        },
                    )
                )
            except Exception:
                self.native_event_mark_gap(row["conversation_id"])
        return row is not None
