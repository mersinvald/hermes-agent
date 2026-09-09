"""Native display-history mutation fences, independent of FTS availability."""
from __future__ import annotations

from hermes_state_pwa import bounded_read


_SESSION_FIELDS = ("id", "source", "user_id", "chat_id", "chat_type", "thread_id",
                   "profile_name", "parent_session_id", "origin_json", "title", "started_at")
_MESSAGE_FIELDS = ("id", "session_id", "role", "content", "tool_call_id", "tool_calls",
                   "tool_name", "timestamp", "active", "compacted", "display_kind", "display_metadata")


def _ownership_config(alias):
    value = f"{alias}.model_config"
    # Invalid/non-object JSON changes affect authorization too. Known objects
    # compare only the two markers consumed by the native lineage resolver.
    return f"""CASE WHEN length({value})>16384 THEN json_object('_unavailable',1)
        WHEN {value} IS NULL THEN json_object('_branched_from',NULL,'_delegate_from',NULL)
        WHEN json_valid({value}) THEN CASE WHEN json_type({value})='object'
            THEN json_object('_branched_from',json_extract({value},'$._branched_from'),
                             '_delegate_from',json_extract({value},'$._delegate_from'))
            ELSE {value} END ELSE {value} END"""


def install_history_fence(cursor):
    """Install after column reconciliation, before version advancement.

    Existing FTS triggers write only their own indexes, never sessions/messages.
    A BEFORE insert sets its pending flag for *this* row; an ignored insert has
    no AFTER insert. The next BEFORE overwrites that stale flag. Actual REPLACE
    fires AFTER insert even with SQLite's default recursive_triggers disabled.
    All flag/counter writes participate in the native transaction and rollback.
    """
    cursor.execute("INSERT OR IGNORE INTO native_pwa_history_fence(singleton) VALUES(1)")
    session_changed = " OR ".join(f"old.{field} IS NOT new.{field}" for field in _SESSION_FIELDS)
    session_changed += " OR COALESCE(old.end_reason='compression',0) IS NOT COALESCE(new.end_reason='compression',0)"
    session_changed += f" OR ({_ownership_config('old')}) IS NOT ({_ownership_config('new')})"
    message_changed = " OR ".join(f"old.{field} IS NOT new.{field}" for field in _MESSAGE_FIELDS)
    statements = [
        """CREATE TRIGGER IF NOT EXISTS pwa_history_session_before_insert BEFORE INSERT ON sessions BEGIN
            UPDATE native_pwa_history_fence SET pending_session_replace=EXISTS(
                SELECT 1 FROM sessions WHERE id=new.id OR (new.title IS NOT NULL AND title=new.title))
                WHERE singleton=1; END""",
        """CREATE TRIGGER IF NOT EXISTS pwa_history_session_after_insert AFTER INSERT ON sessions BEGIN
            UPDATE native_pwa_history_fence SET generation=generation+pending_session_replace,
                pending_session_replace=0 WHERE singleton=1; END""",
        """CREATE TRIGGER IF NOT EXISTS pwa_history_session_delete AFTER DELETE ON sessions BEGIN
            UPDATE native_pwa_history_fence SET generation=generation+1 WHERE singleton=1; END""",
        f"""CREATE TRIGGER IF NOT EXISTS pwa_history_session_update
            AFTER UPDATE OF {','.join(_SESSION_FIELDS)},end_reason,model_config ON sessions
            WHEN {session_changed} BEGIN
            UPDATE native_pwa_history_fence SET generation=generation+1 WHERE singleton=1; END""",
        """CREATE TRIGGER IF NOT EXISTS pwa_history_message_before_insert BEFORE INSERT ON messages BEGIN
            UPDATE native_pwa_history_fence SET pending_message_replace=EXISTS(
                SELECT 1 FROM messages WHERE id=new.id AND (role!='session_meta' OR new.role!='session_meta'))
                WHERE singleton=1; END""",
        """CREATE TRIGGER IF NOT EXISTS pwa_history_message_after_insert AFTER INSERT ON messages BEGIN
            UPDATE native_pwa_history_fence SET generation=generation+pending_message_replace,
                pending_message_replace=0 WHERE singleton=1; END""",
        """CREATE TRIGGER IF NOT EXISTS pwa_history_message_delete AFTER DELETE ON messages
            WHEN old.role!='session_meta' BEGIN
            UPDATE native_pwa_history_fence SET generation=generation+1 WHERE singleton=1; END""",
        f"""CREATE TRIGGER IF NOT EXISTS pwa_history_message_update
            AFTER UPDATE OF {','.join(_MESSAGE_FIELDS)} ON messages
            WHEN (old.role!='session_meta' OR new.role!='session_meta') AND ({message_changed}) BEGIN
            UPDATE native_pwa_history_fence SET generation=generation+1 WHERE singleton=1; END""",
    ]
    for statement in statements:
        cursor.execute(statement)


def history_snapshot(db):
    # One SQLite statement captures a coherent generation/high-water tuple.
    # Upper IDs are private fencing material, never caller offsets or counts.
    with bounded_read(db) as connection:
        row = connection.execute("""SELECT generation,
            (SELECT COALESCE(MAX(id),0) FROM messages WHERE role NOT IN ('session_meta','system')),
            (SELECT COALESCE(MAX(rowid),0) FROM sessions)
            FROM native_pwa_history_fence WHERE singleton=1""").fetchone()
        if row is None:
            raise RuntimeError("native history fence unavailable")
        return {"generation": row[0], "message_upper": row[1], "session_upper": row[2]}
