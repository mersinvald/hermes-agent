"""Native display-history mutation fences, independent of FTS availability."""
from __future__ import annotations

from hermes_state_pwa import bounded_read


_SESSION_FIELDS = ("id", "source", "user_id", "chat_id", "chat_type", "thread_id",
                   "profile_name", "parent_session_id", "origin_json", "title", "started_at")
_MESSAGE_FIELDS = ("id", "session_id", "role", "content", "tool_call_id", "tool_calls",
                   "tool_name", "timestamp", "active", "compacted", "display_kind", "display_metadata")


def _duplicate_markers(value):
    return f"""(SELECT COUNT(*)-COUNT(DISTINCT key) FROM json_each({value})
        WHERE key IN ('_branched_from','_delegate_from'))>0"""


def _ownership_config(alias):
    value = f"{alias}.model_config"
    # Invalid/non-object JSON changes affect authorization too. Known objects
    # compare only the two markers consumed by the native lineage resolver.
    return f"""CASE WHEN length({value})>16384 THEN json_object('_unavailable',1)
        WHEN {value} IS NULL THEN json_object('_branched_from',NULL,'_delegate_from',NULL)
        WHEN json_valid({value}) THEN CASE WHEN json_type({value})='object'
            THEN CASE WHEN {_duplicate_markers(value)} THEN {value}
                ELSE json_object('_branched_from',json_extract({value},'$._branched_from'),
                             '_delegate_from',json_extract({value},'$._delegate_from')) END
            ELSE {value} END ELSE {value} END"""


def _bounded_object(value, *, markers=False):
    result = f"CASE WHEN {_duplicate_markers(value)} THEN NULL ELSE {value} END" if markers else value
    return f"""CASE WHEN {value} IS NULL THEN '{{}}'
        WHEN length({value})<=16384 AND json_valid({value}) THEN
            CASE WHEN json_type({value})='object' THEN {result} END END"""


def install_history_fence(cursor):
    """Install after column reconciliation, before version advancement.

    Existing FTS triggers write only their own indexes, never sessions/messages.
    A BEFORE insert sets its pending flag for *this* row; an ignored insert has
    no AFTER insert. The next BEFORE overwrites that stale flag. Actual REPLACE
    fires AFTER insert even with SQLite's default recursive_triggers disabled.
    All flag/counter writes participate in the native transaction and rollback.
    """
    session_changed = " OR ".join(f"old.{field} IS NOT new.{field}" for field in _SESSION_FIELDS)
    session_changed += " OR COALESCE(old.end_reason='compression',0) IS NOT COALESCE(new.end_reason='compression',0)"
    session_changed += f" OR ({_ownership_config('old')}) IS NOT ({_ownership_config('new')})"
    message_changed = " OR ".join(f"old.{field} IS NOT new.{field}" for field in _MESSAGE_FIELDS)
    config = _bounded_object("new.model_config", markers=True)
    origin = _bounded_object("new.origin_json")
    statements = [
        """CREATE TRIGGER IF NOT EXISTS pwa_history_session_before_insert BEFORE INSERT ON sessions BEGIN
            UPDATE native_pwa_history_fence SET pending_session_replace=EXISTS(
                SELECT 1 FROM sessions WHERE id=new.id OR (new.title IS NOT NULL AND title=new.title))
                WHERE singleton=1; END""",
        """CREATE TRIGGER IF NOT EXISTS pwa_history_session_after_insert AFTER INSERT ON sessions BEGIN
            UPDATE native_pwa_history_fence SET generation=generation+pending_session_replace,
                pending_session_replace=0 WHERE singleton=1; END""",
        f"""CREATE TRIGGER IF NOT EXISTS pwa_history_compression_insert AFTER INSERT ON sessions
            WHEN EXISTS(SELECT 1 FROM sessions WHERE id=new.parent_session_id AND end_reason='compression')
                AND (({config}) IS NULL OR ({origin}) IS NULL OR
                    (new.source!='tool' AND json_extract(({config}),'$._delegate_from') IS NULL
                     AND json_extract(({config}),'$._branched_from') IS NOT new.parent_session_id))
            BEGIN UPDATE native_pwa_history_fence SET generation=generation+1 WHERE singleton=1; END""",
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
    # SQLite json_extract uses the first duplicate key; Python json.loads uses
    # the last. Definition repair must also reach already opened schema31 DBs.
    # Repair only our exact names, atomically, and invalidate old cursors once.
    cursor.execute("SAVEPOINT pwa_history_fence_install")
    try:
        fresh = cursor.execute("SELECT 1 FROM native_pwa_history_fence WHERE singleton=1").fetchone() is None
        cursor.execute("INSERT OR IGNORE INTO native_pwa_history_fence(singleton) VALUES(1)")
        changed = False
        for statement in statements:
            name = statement.split("IF NOT EXISTS ", 1)[1].split()[0]
            existing = cursor.execute("SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?", (name,)).fetchone()
            normalize = lambda sql: " ".join(sql.split()).replace("CREATE TRIGGER IF NOT EXISTS", "CREATE TRIGGER")
            if existing and normalize(existing[0]) != normalize(statement):
                cursor.execute(f'DROP TRIGGER "{name}"')
                changed = True
            elif not existing:
                changed = True
            cursor.execute(statement)
        if changed and not fresh:
            cursor.execute("UPDATE native_pwa_history_fence SET generation=generation+1 WHERE singleton=1")
        cursor.execute("RELEASE pwa_history_fence_install")
    except BaseException:
        cursor.execute("ROLLBACK TO pwa_history_fence_install")
        cursor.execute("RELEASE pwa_history_fence_install")
        raise


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

def history_candidates(db, after, upper, limit=500):
    """Bounded discovery keys, never transcript payloads or arbitrary offsets."""
    with bounded_read(db) as connection:
        rows = connection.execute("""SELECT
            CASE WHEN length(id)<=256 THEN id END AS id,
            substr(source,1,65) AS source,substr(user_id,1,513) AS user_id,
            substr(chat_id,1,513) AS chat_id FROM sessions
            WHERE id>? AND rowid<=? ORDER BY id LIMIT ?""", (after, upper, limit)).fetchall()
        if any(row["id"] is None for row in rows):
            raise RuntimeError("native history discovery key unavailable")
        return [dict(row) for row in rows]
