"""Additive native authority/delivery migration preserves existing transcripts."""

import sqlite3

import pytest

from hermes_state import SessionDB


def test_upgrade_preserves_native_state_and_durable_new_records(tmp_path):
    path = tmp_path / "native.db"
    db = SessionDB(path)
    db.create_session("retained", "telegram", user_id="owner")
    db.append_message("retained", "user", "retained native history")
    db.close()
    # An actual pre-N07 database, including existing N02 journal tables.
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE native_pwa_conversations")
        conn.execute("DROP TABLE native_channel_deliveries")
        conn.execute("UPDATE schema_version SET version=27")
    db = SessionDB(path)
    assert db.get_messages("retained")[0]["content"] == "retained native history"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "INSERT INTO native_pwa_conversations VALUES (?,?,?,?,?,?,?,?)",
            ("concierge", "https://issuer.invalid", "owner", "retained", "{}",
             "created", "create-1", 1.0),
        )
        conn.execute(
            "INSERT INTO native_channel_deliveries VALUES (?,?,?,?,?,?,?)",
            ("execution", "channel-hash", "retained", 1, "attempting", 1.0, None),
        )
    db.close()
    db = SessionDB(path)
    try:
        assert db.get_messages("retained")[0]["content"] == "retained native history"
        with sqlite3.connect(path) as conn:
            assert conn.execute("SELECT state FROM native_channel_deliveries").fetchone() == ("attempting",)
            assert conn.execute("SELECT create_id FROM native_pwa_conversations").fetchone() == ("create-1",)
            # The same principal/create ID cannot reserve a second root.
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO native_pwa_conversations VALUES (?,?,?,?,?,?,?,?)",
                    ("concierge", "https://issuer.invalid", "owner", "other", "{}",
                     "created", "create-1", 2.0),
                )
            # One native execution/channel has one durable send attempt.
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO native_channel_deliveries VALUES (?,?,?,?,?,?,?)",
                    ("execution", "channel-hash", "retained", 2, "delivered", 2.0, 2.0),
                )
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute("UPDATE native_channel_deliveries SET state='success'")
    finally:
        db.close()
