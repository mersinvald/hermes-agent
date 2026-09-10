"""Actual SQLite snapshot/byte guards before native context decoding."""
from contextlib import contextmanager

import pytest

import hermes_state
from hermes_state import ManagedMediaReadError, SessionDB


@pytest.mark.parametrize("field", ["content", "api_content", "display_metadata"])
def test_read_limit_precedes_row_decode_and_preserves_state(tmp_path, monkeypatch, field):
    db = SessionDB(tmp_path / "native.db")
    try:
        db.create_session("root", "telegram")
        kwargs = {field: ([{"type": "text", "text": "x" * 4096}] if field == "content"
                          else {"private": "x" * 4096} if field == "display_metadata" else "x" * 4096)}
        content = kwargs.pop("content", "caption")
        db.append_message("root", "user", content, **kwargs)
        before = db.get_messages_as_conversation("root")
        decoded = []
        original = db._rows_to_conversation
        def decode(*args, **kwargs):
            decoded.append(True)
            return original(*args, **kwargs)
        monkeypatch.setattr(db, "_rows_to_conversation", decode)
        with pytest.raises(ManagedMediaReadError, match="byte limit"):
            db.get_messages_as_conversation("root", max_materialized_bytes=1024)
        assert decoded == []
        assert db.get_messages_as_conversation("root") == before
        assert db.get_messages_as_conversation("root", max_materialized_bytes=16384) == before
        with db._read_ctx() as conn:
            assert not conn.in_transaction
    finally:
        db.close()


def test_byte_preflight_and_fetch_share_sqlite_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "native.db"
    db, peer = SessionDB(path), SessionDB(path)
    try:
        assert db._wal_active
        db.create_session("root", "telegram")
        db.append_message("root", "user", "first")
        original = db._read_ctx
        appended, transactions = [], []
        class Connection:
            def __init__(self, conn):
                self.conn = conn
            def __getattr__(self, name):
                return getattr(self.conn, name)
            def execute(self, sql, *args):
                cursor = self.conn.execute(sql, *args)
                if sql.startswith("SELECT COALESCE(SUM(") and not appended:
                    assert self.conn.in_transaction
                    peer.append_message("root", "assistant", "later" * 4096)
                    appended.append(True)
                return cursor
        @contextmanager
        def read():
            with original() as conn:
                try:
                    yield Connection(conn)
                finally:
                    transactions.append(conn.in_transaction)
        monkeypatch.setattr(db, "_read_ctx", read)
        rows = db.get_messages_as_conversation("root", max_materialized_bytes=1024)
        assert [r["content"] for r in rows] == ["first"]
        assert appended and not any(transactions)
        with pytest.raises(ManagedMediaReadError, match="byte limit"):
            db.get_messages_as_conversation("root", max_materialized_bytes=1024)
        assert len(db.get_messages_as_conversation("root")) == 2
    finally:
        peer.close()
        db.close()


def test_bounded_read_deadline_releases_snapshot_without_decoding(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "native.db")
    try:
        db.create_session("root", "telegram")
        db.append_message("root", "user", "retained")
        monkeypatch.setattr(hermes_state, "MANAGED_MEDIA_READ_TIMEOUT", -1)
        monkeypatch.setattr(db, "_rows_to_conversation", lambda *a, **k: pytest.fail("decoded timed-out read"))
        with pytest.raises(ManagedMediaReadError, match="deadline"):
            db.get_messages_as_conversation("root", max_materialized_bytes=1024)
        with db._read_ctx() as conn:
            assert not conn.in_transaction
    finally:
        db.close()


@pytest.mark.parametrize("rotated", [False, True], ids=["durable-parent", "adopted-child"])
def test_compression_reload_obeys_native_limit_and_releases_lock(tmp_path, monkeypatch, rotated):
    from tests.agent.test_compression_concurrent_fork import _build_agent_with_db
    db = SessionDB(tmp_path / "native.db")
    try:
        db.create_session("root", "telegram")
        target = "root"
        if rotated:
            db.end_session("root", "compression")
            db.create_session("tip", "telegram", parent_session_id="root")
            target = "tip"
        db.append_message(target, "user", [{"type": "text", "text": "retained" * 4096}])
        agent = _build_agent_with_db(db, "root")
        agent._native_media_read_limit = 1024
        decoded = []
        original = db._rows_to_conversation
        def decode(*args, **kwargs):
            decoded.append(True)
            return original(*args, **kwargs)
        monkeypatch.setattr(db, "_rows_to_conversation", decode)
        with pytest.raises(ManagedMediaReadError, match="byte limit"):
            agent._compress_context([{"role": "user", "content": "stale"}], "sys", approx_tokens=120000)
        assert decoded == []
        agent.context_compressor.compress.assert_not_called()
        assert db.get_compression_lock_holder("root") is None
        assert agent.session_id == "root"
        assert len(db.get_messages_as_conversation(target)) == 1
    finally:
        db.close()
