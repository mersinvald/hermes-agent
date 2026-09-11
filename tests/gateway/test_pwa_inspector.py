"""Real native HTTP history, ownership and retained-source inspection."""

import json
import hashlib
import sqlite3
from dataclasses import replace
from types import SimpleNamespace
import os
from pathlib import Path

import pytest

from gateway.conversation_control import Principal
from agent.native_execution_context import NativeExecutionOrigin
from gateway.pwa_config import OwnerBinding, canonical
from hermes_state import SessionDB
from tests.gateway.test_pwa_http import OWNER, headers, service


def execution(server, db, root, key, *, close=True):
    owner = server.ingress._command_owner
    db.native_execution_open(root, key, owner, origin="pwa")
    for activity in ("repeat-1", "repeat-2"):
        db.native_activity_event(
            key,
            owner,
            "tool_changed",
            {
                "activity_id": activity,
                "state": "completed",
                "detail": {
                    "tool_name": "terminal",
                    "unreviewed": "PRIVATE TOOL OUTPUT",
                },
            },
        )
    if close:
        db.native_execution_close(key, owner, outcome="completed")


async def get(client, path):
    response = await client.get(path)
    value = await response.json()
    assert response.status == 200, value
    return value


@pytest.mark.asyncio
async def test_finished_tasks_retain_activity_without_a_recovery_cursor(
    monkeypatch, tmp_path
):
    async with service(monkeypatch, tmp_path) as (server, client, native):
        db, root = native[2], native[4].session_id
        execution(server, db, root, "older")
        execution(server, db, root, "newer")
        await get(client, "/v1/pwa/inspector/runtime-events")
        first = await get(client, "/v1/pwa/inspector/tasks?limit=1")
        assert first["has_more"] and first["next_cursor"]
        task = first["rows"][0]
        assert task["ordinal"] == 1 and task["duration_ms"] >= 0
        assert task["completed_at"] == task["updated_at"]
        assert first["summary"]["task_count"] == 1
        assert first["summary"]["duration_count"] == 1
        second = await get(
            client, "/v1/pwa/inspector/tasks?limit=1&cursor=" + first["next_cursor"]
        )
        assert second["rows"][0]["task_ref"] != task["task_ref"]
        assert second["rows"][0]["ordinal"] != task["ordinal"]
        assert not second["has_more"]
        detail = await get(client, "/v1/pwa/inspector/tasks/" + task["task_ref"])
        tools = [e for e in detail["events"] if e["kind"] == "tool_changed"]
        assert len(tools) == 2
        assert tools[0]["activity_ref"] != tools[1]["activity_ref"]
        assert all(e["tool_name"] == "terminal" for e in tools)
        assert len(task["tools"]) == 2
        assert task["tools"][0]["invocation_ref"] != task["tools"][1]["invocation_ref"]
        assert "PRIVATE" not in json.dumps(detail)
        assert "newer" not in json.dumps(detail) and root not in json.dumps(detail)


@pytest.mark.asyncio
async def test_participants_require_real_remote_dispatch_provenance(
    monkeypatch, tmp_path
):
    from tests.state.test_native_cancellation import binding

    async with service(monkeypatch, tmp_path) as (server, client, native):
        db, root = native[2], native[4].session_id
        execution(server, db, root, "with-remote")
        origin = NativeExecutionOrigin(
            root, "with-remote", server.ingress._command_owner
        )
        first = db.native_remote_dispatch_prepare(
            origin, binding("private-peer"), "attempt-1"
        )
        second = db.native_remote_dispatch_prepare(
            origin, binding("private-peer"), "attempt-2"
        )
        db.native_remote_dispatch_observe(
            origin, second, "private-remote-task", "private-context", "completed"
        )
        page = await get(client, "/v1/pwa/inspector/tasks")
        participants = page["rows"][0]["participants"]
        assert [p["state"] for p in participants] == ["attempted", "completed"]
        assert participants[0]["participant_ref"] == participants[1]["participant_ref"]
        assert participants[0]["invocation_ref"] != participants[1]["invocation_ref"]
        assert all(
            x not in json.dumps(page)
            for x in (
                "private-peer",
                "private-remote-task",
                "private-context",
                first,
                second,
            )
        )


@pytest.mark.asyncio
async def test_expired_retained_details_and_invalid_cursor_are_distinct(
    monkeypatch, tmp_path
):
    async with service(monkeypatch, tmp_path) as (server, client, native):
        db, root = native[2], native[4].session_id
        execution(server, db, root, "one")
        page = await get(client, "/v1/pwa/inspector/tasks")
        path = "/v1/pwa/inspector/tasks/" + page["rows"][0]["task_ref"]
        first = await get(client, path + "?limit=1")
        assert first["has_more"]
        invalid = await client.get(path + "?cursor=forged")
        assert invalid.status == 409
        db._execute_write(
            lambda conn: conn.execute("UPDATE native_events SET occurred_at=0")
        )
        detail = await get(client, path)
        assert detail["history_state"] == "expired" and detail["events"] == []
        assert detail["task"]["state"] == "completed"
        continued = await get(client, path + "?limit=1&cursor=" + first["next_cursor"])
        assert continued["history_state"] == "expired"


@pytest.mark.asyncio
async def test_native_admin_authority_is_explicit_and_never_grants_foreign_payload(
    monkeypatch, tmp_path
):
    async with service(monkeypatch, tmp_path) as (server, client, native):
        db, root = native[2], native[4].session_id
        execution(server, db, root, "own")
        denied = await client.get("/v1/pwa/inspector/tasks?scope=all")
        assert denied.status == 404
        alice = server.ownership.config.bindings[0]
        other_principal = Principal(OWNER.issuer, "other")
        source = replace(
            alice.sources[0], source=replace(alice.sources[0].source, user_id="foreign")
        )
        other = OwnerBinding(other_principal, (source,), source.source_id)
        monkeypatch.setattr(
            server.runner,
            "_is_user_authorized_for_source",
            lambda candidate: (
                candidate.user_id in {alice.sources[0].source.user_id, "foreign"}
            ),
        )
        config = replace(
            server.ownership.config,
            bindings=(replace(alice, inspector_admin=True), other),
        )
        server.ownership.config = config
        foreign_root = "foreign-private-root"
        db.create_session(
            foreign_root,
            source.source.platform.value,
            user_id="foreign",
            chat_id=source.source.chat_id,
            chat_type=source.source.chat_type,
            thread_id=source.source.thread_id,
        )
        execution(server, db, foreign_root, "foreign-private-execution")
        page = await get(client, "/v1/pwa/inspector/tasks?scope=all")
        assert len(page["rows"]) == 2
        foreign = next(r for r in page["rows"] if r["owner_scope"] == "other")
        own_key = hashlib.sha256(
            canonical([OWNER.issuer, OWNER.subject, config.concierge_id]).encode()
        ).hexdigest()
        filtered = await get(
            client, "/v1/pwa/inspector/tasks?scope=all&owner_keys=" + own_key
        )
        assert (
            len(filtered["rows"]) == 1 and filtered["rows"][0]["owner_scope"] == "self"
        )
        rejected = await client.get(
            "/v1/pwa/inspector/tasks/"
            + foreign["task_ref"]
            + "?scope=all&owner_keys="
            + own_key
        )
        assert rejected.status == 404
        detail = await get(
            client, "/v1/pwa/inspector/tasks/" + foreign["task_ref"] + "?scope=all"
        )
        assert all(
            e["tool_name"] is None and e["payload"]["state"] == "withheld"
            for e in detail["events"]
        )
        assert "foreign-private" not in json.dumps(detail)
        response = await client.get("/v1/pwa/inspector/tasks/" + foreign["task_ref"])
        assert response.status == 404
        response = await client.get(
            "/v1/pwa/inspector/tasks/" + foreign["task_ref"],
            headers=headers(other_principal),
        )
        assert response.status == 404


@pytest.mark.asyncio
async def test_owner_rebinding_during_detail_read_withholds_captured_rows(
    monkeypatch, tmp_path
):
    async with service(monkeypatch, tmp_path) as (server, client, native):
        db, root = native[2], native[4].session_id
        execution(server, db, root, "one")
        page = await get(client, "/v1/pwa/inspector/tasks")
        original = server.inspector.detail_read

        def rebind(*args):
            result = original(*args)
            server.ownership.config = replace(server.ownership.config, bindings=())
            return result

        monkeypatch.setattr(server.inspector, "detail_read", rebind)
        response = await client.get(
            "/v1/pwa/inspector/tasks/" + page["rows"][0]["task_ref"]
        )
        assert response.status == 404


def test_existing_execution_schema_upgrades_without_fabricating_completion(tmp_path):
    path = tmp_path / "old.db"
    db = SessionDB(path)
    db.native_execution_open("root", "legacy", "owner")
    db.close()
    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE native_executions DROP COLUMN completed_at")
    conn.execute("ALTER TABLE native_executions DROP COLUMN activity_evicted")
    conn.execute("ALTER TABLE native_events DROP COLUMN inspector_payload")
    conn.commit()
    conn.close()
    db = SessionDB(path)
    try:
        row = db.native_execution("root")
        assert row["completed_at"] is None and row["activity_evicted"] == 0
    finally:
        db.close()


@pytest.mark.asyncio
async def test_owner_payload_capture_is_separate_from_normal_replay_and_expires(
    monkeypatch, tmp_path
):
    from gateway.native_events import NativeActivityObserver

    async with service(monkeypatch, tmp_path) as (server, client, native):
        db, root = native[2], native[4].session_id
        owner = server.ingress._command_owner
        db.native_execution_open(root, "payload-task", owner, origin="pwa")
        before = await server.ingress.events.recover(OWNER, root)
        Path(os.environ["HERMES_HOME"], "config.yaml").write_text(
            json.dumps({
                "a2a_agents": {
                    "reviewed": {
                        "url": "https://peer.invalid",
                        "auth": {"type": "bearer", "token": "peer-key-known"},
                    },
                }
            })
        )
        agent = SimpleNamespace(
            api_key="model-key-known",
            tool_start_callback=None,
            tool_complete_callback=None,
        )
        context = SimpleNamespace(
            ingress=server.ingress,
            execution=SimpleNamespace(
                execution_id="payload-task", conversation_id=root
            ),
        )
        observer = NativeActivityObserver(context, agent).install()
        args = {"agent": "reviewed", "message": "Owner request peer-key-known"}
        try:
            agent.tool_start_callback("a2a-invocation", "a2a_call", args)
            agent.tool_complete_callback(
                "a2a-invocation", "a2a_call", args, "Owner result model-key-known"
            )
            agent.tool_start_callback(
                "file-invocation",
                "read_file",
                {"path": "/owned/report.txt", "limit": 20},
            )
            agent.tool_complete_callback(
                "file-invocation",
                "read_file",
                {"path": "/owned/report.txt"},
                "FILE SECRET",
            )
            agent.tool_start_callback(
                "terminal-invocation", "terminal", {"command": "TERMINAL SECRET"}
            )
            for identity, arguments in (
                (
                    "suspect-invocation",
                    {
                        "agent": "reviewed",
                        "message": "sk-syntheticUnconfiguredCredential",
                    },
                ),
                (
                    "data-invocation",
                    {"agent": "reviewed", "data": {"private": "DATA SECRET"}},
                ),
                (
                    "url-invocation",
                    {"agent": "https://private-peer.invalid", "message": "URL SECRET"},
                ),
                ("oversized-invocation", {"agent": "reviewed", "message": "x" * 8192}),
            ):
                agent.tool_start_callback(identity, "a2a_call", arguments)
        finally:
            observer.restore()
        normal = await server.ingress.events.recover(OWNER, root, before["cursor"])
        assert normal["events"]
        assert all(
            text not in json.dumps(normal)
            for text in (
                "Owner request",
                "Owner result",
                "/owned/report.txt",
                "FILE SECRET",
                "TERMINAL SECRET",
            )
        )
        page = await get(client, "/v1/pwa/inspector/tasks")
        detail_path = "/v1/pwa/inspector/tasks/" + page["rows"][0]["task_ref"]
        detail = await get(client, detail_path)
        captures = [e["payload"] for e in detail["events"]]
        assert any(
            p["text"] and "Owner request" in p["text"] and p["state"] == "redacted"
            for p in captures
        )
        assert any(p["text"] and "Owner result" in p["text"] for p in captures)
        assert any(p["text"] and "/owned/report.txt" in p["text"] for p in captures)
        assert any(p["reason"] == "oversized" for p in captures)
        assert sum(p["reason"] == "credential_boundary_unproven" for p in captures) >= 4
        assert all(
            secret not in json.dumps(detail)
            for secret in (
                "peer-key-known",
                "model-key-known",
                "FILE SECRET",
                "TERMINAL SECRET",
                "sk-syntheticUnconfiguredCredential",
                "DATA SECRET",
                "URL SECRET",
                "private-peer.invalid",
            )
        )
        db._execute_write(
            lambda conn: conn.execute("UPDATE native_events SET occurred_at=0")
        )
        expired = await get(client, detail_path)
        assert expired["history_state"] == "expired" and expired["events"] == []


@pytest.mark.asyncio
async def test_runtime_event_logs_use_typed_events_without_body_or_stderr_inference(
    monkeypatch, tmp_path
):
    async with service(monkeypatch, tmp_path) as (server, client, native):
        db, root = native[2], native[4].session_id
        owner = server.ingress._command_owner
        db.native_execution_open(root, "runtime", owner, origin="pwa")
        for kind in ("warning", "error", "tool_changed"):
            db.native_activity_event(
                "runtime",
                owner,
                kind,
                {"state": "running", "detail": {"stderr": "SECRET"}},
            )
        page = await get(client, "/v1/pwa/inspector/runtime-events")
        mapped = {r["kind"]: r["severity"] for r in page["rows"]}
        assert mapped["warning"] == "WARN" and mapped["error"] == "ERROR"
        assert mapped["tool_changed"] == "INFO" and "SECRET" not in json.dumps(page)
        detail = await get(
            client, "/v1/pwa/inspector/tasks/" + page["rows"][0]["task_ref"]
        )
        assert detail["task"]["task_ref"] == page["rows"][0]["task_ref"]


def test_explicit_completion_is_cleared_when_unconsumed_execution_reopens(tmp_path):
    db = SessionDB(tmp_path / "reopen.db")
    try:
        db.native_execution_open("root", "retry", "owner")
        db.native_execution_close("retry", "owner", outcome="interrupted")
        with db._read_ctx() as conn:
            row = conn.execute(
                "SELECT completed_at FROM native_executions WHERE execution_id='retry'"
            ).fetchone()
        assert row["completed_at"] is not None
        db.native_execution_open("root", "retry", "new-owner")
        assert db.native_execution("root")["completed_at"] is None
    finally:
        db.close()


@pytest.mark.parametrize("value", [False, True, "true", 1, None])
def test_native_admin_config_requires_an_explicit_boolean(value):
    from gateway.pwa_config import PwaHttpConfig
    from tests.gateway.test_pwa_http import config_for
    from tests.gateway.test_42039_duplicate_user_message import _source

    raw = config_for(_source())
    assert PwaHttpConfig.from_dict(raw).bindings[0].inspector_admin is False
    raw["bindings"][0]["inspector_admin"] = value
    if type(value) is bool:
        assert PwaHttpConfig.from_dict(raw).bindings[0].inspector_admin is value
    else:
        with pytest.raises(ValueError):
            PwaHttpConfig.from_dict(raw)
