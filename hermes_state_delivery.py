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
        return self._execute_write(
            lambda conn: (
                conn.execute(
                    "UPDATE native_channel_deliveries SET state=?,completed_at=? "
                    "WHERE execution_id=? AND channel_key=? AND state='attempting'",
                    (state, time.time(), execution_id, channel_key),
                ).rowcount
                == 1
            )
        )
