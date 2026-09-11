"""Real authenticated native HTTP over retained lifecycle facts and SQL totals."""

import json
import time
import sqlite3
from dataclasses import replace
from urllib.parse import urlencode
from unittest.mock import patch

import pytest

from agent.native_execution_context import NativeExecutionOrigin, native_execution_scope
from agent.conversation_loop import run_conversation as _real_loop
from hermes_cli.lifecycle import invoke_hook
from hermes_state_events import timestamp
from tests.gateway.test_pwa_http import service, model_config
from tests.gateway.test_pwa_inspector import get
from tests.run_agent.test_run_agent import (
    agent as agent,  # Re-export the shared pytest fixture.
    TestRunConversation as _ConversationHarness,
    _mock_response,
)


def scope(server, root, execution):
    return native_execution_scope(
        NativeExecutionOrigin(root, execution, server.ingress._command_owner),
        server.ingress.cancellations,
    )


def call(server, root, execution, request, *, retry=False):
    start = time.time() - 0.02
    values = dict(
        session_id=root,
        api_request_id=request,
        turn_id="turn",
        model="upstream/daily",
        provider="provider-a",
        started_at=start,
    )
    with scope(server, root, execution):
        invoke_hook(
            "pre_api_request",
            **values,
            retry_count=0,
            messages=[{"content": "PRIVATE PROMPT"}],
        )
        if retry:
            invoke_hook(
                "api_request_error",
                **values,
                ended_at=start + 0.005,
                error={"type": "APITimeoutError", "message": "PRIVATE ERROR"},
                status_code=504,
            )
            invoke_hook("pre_api_request", **values, retry_count=1)
        invoke_hook(
            "post_api_request",
            **values,
            ended_at=start + 0.01,
            response_model="upstream/daily",
            usage={"input_tokens": 11, "output_tokens": 7},
            accounting_cost={
                "amount": 0.01,
                "currency": "USD",
                "basis": "estimated",
                "source": "native_accounting",
            },
            assistant_message="PRIVATE ANSWER",
        )


@pytest.mark.asyncio
async def test_execution_facts_and_sql_aggregate_retain_real_calls_tools_children(
    monkeypatch, tmp_path
):
    async with service(monkeypatch, tmp_path, models=model_config()) as (
        server,
        client,
        native,
    ):
        db, root = native[2], native[4].session_id
        start = time.time() - 1
        db.native_execution_open(
            root, "captured", server.ingress._command_owner, origin="pwa"
        )
        call(server, root, "captured", "one", retry=True)
        call(server, root, "captured", "two")
        with scope(server, root, "captured"):
            invoke_hook(
                "subagent_start",
                parent_session_id=root,
                parent_turn_id="turn",
                child_session_id="child",
                child_role="leaf",
                child_goal="PRIVATE CHILD GOAL",
            )
            invoke_hook(
                "pre_api_request",
                session_id="child",
                api_request_id="child-call",
                model="upstream/daily",
                provider="provider-a",
                started_at=time.time(),
            )
            invoke_hook(
                "subagent_stop",
                parent_session_id=root,
                parent_turn_id="turn",
                child_session_id="child",
                child_role="leaf",
                child_status="completed",
                duration_ms=12,
                child_summary="PRIVATE SUMMARY",
            )
        for state in ("running", "completed"):
            db.native_activity_event(
                "captured",
                server.ingress._command_owner,
                "tool_changed",
                dict(
                    activity_id="actual-tool",
                    state=state,
                    detail={"tool_name": "read_file"},
                ),
            )
        db.native_execution_close(
            "captured", server.ingress._command_owner, outcome="completed"
        )
        page = await get(client, "/v1/pwa/inspector/tasks")
        task = page["rows"][0]["task_ref"]
        facts = await get(client, f"/v1/pwa/inspector/tasks/{task}/facts?limit=1")
        assert facts["model_calls_has_more"]
        assert facts["model_calls"][0]["attempt_count"] == 2
        assert facts["model_calls"][0]["input_tokens"] == 11
        assert facts["model_calls"][0]["cost"]["amount"] == 0.01
        assert facts["delegations"][0]["delegation_kind"] == "local"
        assert facts["delegations"][0]["duration_ms"] >= 0
        second = await get(
            client,
            f"/v1/pwa/inspector/tasks/{task}/facts?"
            + urlencode({"limit": 1, "cursor": facts["model_calls_next_cursor"]}),
        )
        assert (
            second["model_calls"][0]["call_ref"] != facts["model_calls"][0]["call_ref"]
        )
        query = urlencode({"from": timestamp(start), "to": timestamp(time.time())})
        metrics = await get(client, "/v1/pwa/inspector/execution-metrics?" + query)
        row = metrics["rows"][0]
        assert row["execution_count"] == 1
        assert row["model_call_count"] == 3
        assert row["model_call_completed_count"] == 2
        assert row["model_call_active_count"] == 0  # abandoned child is unknown
        assert row["input_tokens"] == 22 and row["cost_usd"] == 0.02
        assert (
            row["tool_call_count"] == 1 and row["tools"][0]["tool_name"] == "read_file"
        )
        assert row["local_delegation_count"] == 1
        assert row["models"][0]["task_refs"] == [task]
        assert "PRIVATE" not in json.dumps([facts, metrics])
        (tmp_path / "native-inspector-dtos.json").write_text(
            json.dumps({"task_facts": facts, "execution_metrics": metrics})
        )


@pytest.mark.asyncio
async def test_retention_and_missing_native_origin_are_explicit(monkeypatch, tmp_path):
    async with service(monkeypatch, tmp_path, models=model_config()) as (
        server,
        client,
        native,
    ):
        db, root = native[2], native[4].session_id
        db.native_execution_open(
            root, "retained", server.ingress._command_owner, origin="telegram"
        )
        call(server, root, "retained", "one")
        invoke_hook(
            "pre_api_request",
            session_id=root,
            api_request_id="untrusted",
            started_at=time.time(),
        )
        db.native_execution_close(
            "retained", server.ingress._command_owner, outcome="completed"
        )
        task = (await get(client, "/v1/pwa/inspector/tasks"))["rows"][0]["task_ref"]
        first = await get(client, f"/v1/pwa/inspector/tasks/{task}/facts")
        assert len(first["model_calls"]) == 1
        db._execute_write(
            lambda conn: conn.execute(
                "UPDATE native_inspector_facts SET updated_at=?", (time.time() - 90000,)
            )
        )
        expired = await get(client, f"/v1/pwa/inspector/tasks/{task}/facts")
        assert expired["model_calls"] == []
        assert expired["coverage"]["expired_execution_count"] == 1
        assert "retained_facts_expired" in expired["coverage"]["limitations"]
        response = await client.get(
            "/v1/pwa/inspector/execution-metrics?from=2026-01-01&to=2026-01-03"
        )
        assert response.status == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["pwa", "telegram"])
async def test_normal_agent_loop_captures_real_hook_usage_and_actual_route(
    monkeypatch, tmp_path, agent, origin  # noqa: F811 - shared pytest fixture
):
    async with service(monkeypatch, tmp_path, models=model_config()) as (
        server,
        client,
        native,
    ):
        db, root = native[2], native[4].session_id
        server.ownership.config = replace(
            server.ownership.config,
            inspector_model_names=("Qwen/Qwen3.8-27B-262k",),
            inspector_provider_names=("custom",),
        )
        db.native_execution_open(
            root, "normal-turn", server.ingress._command_owner, origin=origin
        )
        _ConversationHarness()._setup_agent(agent)
        agent.session_id, agent.platform = root, origin
        agent.model, agent.provider = "upstream/daily", "custom"
        agent.base_url = "http://127.0.0.1:1/v1"
        agent.tool_use_enforcement = "off"
        response = _mock_response(
            content="PRIVATE ANSWER",
            usage={"prompt_tokens": 19, "completion_tokens": 5, "total_tokens": 24},
        )
        response.model = "Qwen/Qwen3.8-27B-262k"
        agent.client.chat.completions.create.return_value = response
        with (
            scope(server, root, "normal-turn"),
            patch("agent.conversation_loop.run_conversation", _real_loop),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("PRIVATE USER MESSAGE")
        assert result["completed"] is True
        db.native_execution_close(
            "normal-turn", server.ingress._command_owner, outcome="completed"
        )
        task = (await get(client, "/v1/pwa/inspector/tasks"))["rows"][0]["task_ref"]
        facts = await get(client, f"/v1/pwa/inspector/tasks/{task}/facts")
        assert len(facts["model_calls"]) == 1
        call = facts["model_calls"][0]
        assert (
            call["state"] == "completed"
            and call["input_tokens"] == 19
            and call["output_tokens"] == 5
        )
        assert call["requested_model"] == "upstream/daily"
        assert (
            call["actual_model"] == "Qwen/Qwen3.8-27B-262k"
            and call["provider"] == "custom"
        )
        assert "PRIVATE" not in json.dumps(facts)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["facts", "metrics"])
async def test_native_facts_recheck_owner_after_await(monkeypatch, tmp_path, route):
    async with service(monkeypatch, tmp_path, models=model_config()) as (
        server,
        client,
        native,
    ):
        db, root = native[2], native[4].session_id
        start = time.time() - 1
        db.native_execution_open(
            root, "revoked", server.ingress._command_owner, origin="pwa"
        )
        call(server, root, "revoked", "one")
        db.native_execution_close(
            "revoked", server.ingress._command_owner, outcome="completed"
        )
        task = (await get(client, "/v1/pwa/inspector/tasks"))["rows"][0]["task_ref"]
        method = "read_detail" if route == "facts" else "aggregate"
        original = getattr(server.inspector.facts, method)

        def revoke(*args):
            result = original(*args)
            server.ownership.config = replace(server.ownership.config, bindings=())
            return result

        monkeypatch.setattr(server.inspector.facts, method, revoke)
        path = (
            f"/v1/pwa/inspector/tasks/{task}/facts"
            if route == "facts"
            else "/v1/pwa/inspector/execution-metrics?"
            + urlencode({"from": timestamp(start), "to": timestamp(time.time())})
        )
        response = await client.get(path)
        assert response.status == 404


def test_old_native_database_migrates_without_inventing_model_capture(tmp_path):
    from hermes_state import SessionDB

    path = tmp_path / "legacy.db"
    db = SessionDB(path)
    db.native_execution_open("root", "old", "owner")
    db.close()
    with sqlite3.connect(path) as conn:
        for column in ("facts_started_at", "facts_ref", "facts_evicted"):
            conn.execute("ALTER TABLE native_executions DROP COLUMN " + column)
        conn.execute("DROP TABLE native_inspector_facts")
    db = SessionDB(path)
    try:
        row = db.native_execution("root")
        assert (
            row["facts_started_at"] is None
            and row["facts_ref"] is None
            and row["facts_evicted"] == 0
        )
        assert db.native_inspector_fact("old", "model", "old") is None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_admin_facts_withhold_foreign_usage_and_collapse_private_named_groups(
    monkeypatch, tmp_path
):
    from gateway.conversation_control import Principal
    from gateway.pwa_config import OwnerBinding
    from tests.gateway.test_pwa_http import OWNER

    async with service(monkeypatch, tmp_path, models=model_config()) as (
        server,
        client,
        native,
    ):
        db = native[2]
        start = time.time() - 1
        alice = server.ownership.config.bindings[0]
        source = replace(
            alice.sources[0], source=replace(alice.sources[0].source, user_id="foreign")
        )
        other = OwnerBinding(
            Principal(OWNER.issuer, "foreign"), (source,), source.source_id
        )
        monkeypatch.setattr(
            server.runner,
            "_is_user_authorized_for_source",
            lambda candidate: (
                candidate.user_id in {alice.sources[0].source.user_id, "foreign"}
            ),
        )
        server.ownership.config = replace(
            server.ownership.config,
            bindings=(replace(alice, inspector_admin=True), other),
        )
        root = "foreign-root"
        db.create_session(
            root,
            source.source.platform.value,
            user_id="foreign",
            chat_id=source.source.chat_id,
            chat_type=source.source.chat_type,
            thread_id=source.source.thread_id,
        )
        db.native_execution_open(
            root, "foreign", server.ingress._command_owner, origin="telegram"
        )
        call(server, root, "foreign", "one")
        call(server, root, "foreign", "two")
        origin = NativeExecutionOrigin(root, "foreign", server.ingress._command_owner)
        db.native_inspector_record(
            origin,
            "model",
            root + "\0two",
            {
                "requested_model": "upstream/deep",
                "trace_id": "1" * 32,
                "observation_id": "2" * 16,
            },
        )
        db.native_execution_close(
            "foreign", server.ingress._command_owner, outcome="completed"
        )
        task = (await get(client, "/v1/pwa/inspector/tasks?scope=all"))["rows"][0][
            "task_ref"
        ]
        denied = await client.get(f"/v1/pwa/inspector/tasks/{task}/facts")
        assert denied.status == 404
        facts = await get(client, f"/v1/pwa/inspector/tasks/{task}/facts?scope=all")
        for fact in facts["model_calls"]:
            assert all(
                fact[k] is None
                for k in (
                    "input_tokens",
                    "output_tokens",
                    "cost",
                    "requested_model",
                    "actual_model",
                    "provider",
                    "trace_ref",
                    "observation_ref",
                    "native_correlation_ref",
                )
            )
        metrics = await get(
            client,
            "/v1/pwa/inspector/execution-metrics?"
            + urlencode({
                "from": timestamp(start),
                "to": timestamp(time.time()),
                "scope": "all",
            }),
        )
        row = metrics["rows"][0]
        assert row["model_call_count"] == 2 and len(row["models"]) == 1
        assert row["input_tokens"] is None and row["cost_usd"] is None
        assert row["usage_known_count"] == row["cost_known_count"] == 0
        assert row["models"][0]["call_count"] == 2


@pytest.mark.parametrize(
    "bad", [True, ["bad model"], ["x" * 201], ["duplicate", "duplicate"]]
)
def test_observed_model_vocabulary_rejects_unreviewed_shape(bad):
    from gateway.pwa_config import PwaHttpConfig
    from tests.gateway.test_pwa_http import config_for
    from gateway.session import SessionSource
    from gateway.config import Platform

    raw = config_for(
        SessionSource(
            platform=Platform.TELEGRAM, chat_id="test", chat_type="dm", user_id="test"
        )
    )
    raw["inspector_model_names"] = bad
    with pytest.raises(ValueError):
        PwaHttpConfig.from_dict(raw)
