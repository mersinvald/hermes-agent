"""Durable final-delivery reservation and monotonic completion evidence."""

from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest
from hermes_state import SessionDB


def test_delivery_reservation_is_shared_and_completed_once(tmp_path):
    first = SessionDB(tmp_path / "state.db")
    second = SessionDB(tmp_path / "state.db")
    with ThreadPoolExecutor(2) as pool:
        rows = list(
            pool.map(
                lambda db: db.native_delivery_reserve("e1", "channel", "root", 3),
                [first, second],
            )
        )
    assert sum(created for row, created in rows) == 1
    assert first.native_delivery_finish("e1", "channel", "delivered")
    assert not second.native_delivery_finish("e1", "channel", "unknown")
    assert second.native_delivery_lookup("e1", "channel")["state"] == "delivered"
    first.close()
    second.close()
    reopened = SessionDB(tmp_path / "state.db")
    row, created = reopened.native_delivery_reserve("e1", "channel", "root", 8)
    assert not created and row["state"] == "delivered" and row["binding_version"] == 3
    reopened.close()


def test_interrupted_attempt_and_storage_failure_never_claim_success(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.native_delivery_reserve("e1", "channel", "root", 1)
    db.close()
    db = SessionDB(tmp_path / "state.db")
    row, created = db.native_delivery_reserve("e1", "channel", "root", 2)
    assert not created and row["state"] == "attempting"
    db._execute_write(
        lambda conn: conn.execute(
            "CREATE TRIGGER reject_delivery BEFORE INSERT ON native_channel_deliveries BEGIN SELECT RAISE(ABORT,'synthetic storage failure'); END"
        )
    )
    with pytest.raises(sqlite3.DatabaseError):
        db.native_delivery_reserve("e2", "channel", "root", 2)
    assert db.native_delivery_lookup("e2", "channel") is None
    db.close()
