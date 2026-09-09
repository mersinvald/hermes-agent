"""Real private HTTP validates controls and serves bounded exact native questions."""

import asyncio
import copy
import json

import pytest

from agent.native_execution_context import NativeExecutionOrigin, native_execution_scope
from tests.gateway.test_pwa_http import service, OWNER
from tests.gateway.test_native_commands import command, started, finish, JournalAgent
from tests.gateway.test_native_events import provider_loop
from tests.gateway.test_native_cancellation import initialized_interrupt_state
from tools import clarify_gateway


async def questions(client, root):
    response = await client.get(f"/v1/pwa/conversations/{root}/clarifications")
    assert response.status == 200, await response.text()
    return await response.json()


def request(root, execution, question):
    return dict(
        schema_version="1.0",
        command_id="answer-1",
        type="clarification_response",
        conversation_id=root,
        target_execution_id=execution,
        payload=dict(
            clarification_id=question,
            question_revision=1,
            answer=dict(kind="text", text="actual native answer"),
        ),
    )


@pytest.mark.asyncio
async def test_actual_native_callback_http_claim_ack_and_recovery(
    monkeypatch, tmp_path
):
    observed = []
    async with service(monkeypatch, tmp_path) as (server, client, native):
        root = native[4].session_id
        beginning = server.db.native_event_recovery(
            root, server.ingress._command_scope(OWNER)
        )["cursor"]

        def loop(agent, *args, **kwargs):
            result = provider_loop(agent, *args, **kwargs)
            observed.append(agent.clarify_callback("Native exact question?", None))
            return result

        monkeypatch.setattr("agent.conversation_loop.run_conversation", loop)
        assert (await client.post("/v1/pwa/commands", json=command(root))).status == 200
        await started()
        JournalAgent.gates[0].set()
        for _ in range(100):
            page = await questions(client, root)
            if page["questions"]:
                break
            await asyncio.sleep(0.01)
        question = page["questions"][0]
        assert question["origin"] == "native" and question["response_available"]
        body = request(root, question["execution_id"], question["clarification_id"])
        response = await client.post("/v1/pwa/commands", json=body)
        assert response.status == 200, await response.text()
        value = await response.json()
        assert value["receipt_kind"] == "clarification_response"
        await finish(server.ingress)
        assert observed == ["actual native answer"]
        receipt = await (await client.get("/v1/pwa/commands/answer-1")).json()
        assert (
            receipt["answer_state"] == "delivered"
            and receipt["delivery_evidence"] == "native_waiter_handoff"
        )
        recovery_response = await client.get(f"/v1/pwa/conversations/{root}/recovery")
        assert recovery_response.status == 200, await recovery_response.text()
        recovery = await recovery_response.json()
        assert [
            row["receipt_kind"] for row in recovery["snapshot"]["control_receipts"]
        ] == ["clarification_response"]
        assert all(
            row["command_id"] != "answer-1"
            for row in recovery["snapshot"]["command_receipts"]
        )
        assert "actual native answer" not in json.dumps(
            recovery["snapshot"]["control_receipts"]
        )
        assert (
            await client.get("/v1/pwa/commands/answer-1?remote_limit=50")
        ).status == 400


@pytest.mark.asyncio
async def test_native_http_rejects_malformed_answers_before_any_durable_mutation(
    monkeypatch, tmp_path
):
    async with service(monkeypatch, tmp_path) as (server, client, native):
        root = native[4].session_id
        original = request(root, "missing-execution", "missing-question")
        variants = []
        for change in [
            dict(question_revision=True),
            dict(question_revision=0),
            dict(question_revision=9007199254740992),
            dict(extra="unknown"),
            dict(answer=dict(kind="approval", approved=True)),
            dict(answer=dict(kind="text", text="")),
            dict(answer=dict(kind="text", text="x", owner="forged")),
            dict(answer=dict(kind="selection", option_ids=[])),
            dict(answer=dict(kind="selection", option_ids=["o1", "o1"])),
            dict(answer=dict(kind="selection", option_ids=["o1"] * 5)),
            dict(answer=dict(kind="text", text="x" * 100001)),
        ]:
            body = copy.deepcopy(original)
            body["payload"].update(change)
            variants.append(body)
        variants.append({**original, "owner": "forged"})
        for body in variants:
            response = await client.post("/v1/pwa/commands", json=body)
            assert response.status == 400, await response.text()
        assert (
            server.db._execute_write(
                lambda conn: conn.execute(
                    "SELECT count(*) FROM native_control_commands"
                ).fetchone()[0]
            )
            == 0
        )


@pytest.mark.asyncio
async def test_question_pages_are_scoped_bounded_and_restart_gaps(
    monkeypatch, tmp_path
):
    async with service(monkeypatch, tmp_path) as (server, client, native):
        root = native[4].session_id
        await server.ingress.submit(OWNER, command(root))
        await started()
        execution = server.ingress._owner(root)[2]
        origin = NativeExecutionOrigin(
            root, execution.execution_id, server.ingress._command_owner
        )
        entries = []
        with native_execution_scope(origin, server.ingress.cancellations):
            for _ in range(13):
                entries.append(
                    server.ingress.clarifications.register(
                        "bounded question", None, session_key="page-test"
                    )
                )
        response = await client.get(
            f"/v1/pwa/conversations/{root}/clarifications?limit=10"
        )
        first = await response.json()
        assert len(first["questions"]) == 10 and first["has_more"]
        cursor = first["next_cursor"]
        response = await client.get(
            f"/v1/pwa/conversations/{root}/clarifications",
            params=dict(limit=10, cursor=cursor),
        )
        last = await response.json()
        assert len(last["questions"]) == 3 and not last["has_more"]
        server.ingress.clarifications._cursor_key = b"restart-secret"
        response = await client.get(
            f"/v1/pwa/conversations/{root}/clarifications",
            params=dict(limit=10, cursor=cursor),
        )
        assert (
            response.status == 409
            and (await response.json())["error"]["code"] == "recovery_gap"
        )
        for entry in entries:
            entry.managed.cancel(entry)
            clarify_gateway.wait_for_response(entry.clarify_id, 0.1)
        JournalAgent.gates[0].set()


@pytest.mark.asyncio
async def test_question_byte_bound_and_private_control_snapshot(monkeypatch, tmp_path):
    async with service(monkeypatch, tmp_path) as (server, client, native):
        root = native[4].session_id
        await server.ingress.submit(OWNER, command(root))
        await started()
        execution = server.ingress._owner(root)[2]
        origin = NativeExecutionOrigin(
            root, execution.execution_id, server.ingress._command_owner
        )
        entries = []
        with native_execution_scope(origin, server.ingress.cancellations):
            for _ in range(4):
                entries.append(
                    server.ingress.clarifications.register(
                        "🙂" * 8192, ["🙂" * 1024] * 4, session_key="large"
                    )
                )
        response = await client.get(
            f"/v1/pwa/conversations/{root}/clarifications?limit=100"
        )
        raw = await response.read()
        assert response.status == 200 and len(raw) <= 131072
        page = json.loads(raw)
        assert len(page["questions"]) == 2 and page["has_more"]
        for question in page["questions"]:
            assert len(question["question"]) == 8192
        # Signature remains opaque to a foreign root/principal; malformed cursors
        # are invalid input rather than silently restarting traversal.
        response = await client.get(
            f"/v1/pwa/conversations/{root}/clarifications?cursor=invalid"
        )
        assert response.status == 400
        first = entries[0]
        body = request(root, execution.execution_id, first.clarify_id)
        body["payload"]["answer"] = dict(kind="selection", option_ids=["o1"])
        assert (await client.post("/v1/pwa/commands", json=body)).status == 200
        scope = server.ingress._command_scope(OWNER)
        with clarify_gateway._lock:
            captured = server.db.native_event_recovery(root, "another-explicit-grantee")
        assert all(
            row["command"]["command_id"] != "answer-1"
            for row in captured["control_receipts"]
        )
        assert server.db.native_control_lookup(scope, "answer-1") is not None
        for entry in entries:
            if not entry.event.is_set():
                entry.managed.cancel(entry)
            clarify_gateway.wait_for_response(entry.clarify_id, 0.1)
        JournalAgent.gates[0].set()


@pytest.mark.asyncio
async def test_http_redirect_confirmation_never_waives_live_native_writer(
    monkeypatch, tmp_path
):
    async with service(monkeypatch, tmp_path) as (server, client, native):
        root = native[4].session_id
        beginning = server.db.native_event_recovery(
            root, server.ingress._command_scope(OWNER)
        )["cursor"]

        def loop(agent, *args, **kwargs):
            result = provider_loop(agent, *args, **kwargs)
            return {
                **result,
                "interrupted": bool(agent._interrupt_requested),
                "completed": not agent._interrupt_requested,
            }

        monkeypatch.setattr("agent.conversation_loop.run_conversation", loop)
        assert (await client.post("/v1/pwa/commands", json=command(root))).status == 200
        await started()
        execution = server.ingress._owner(root)[2]
        redirect = command(
            root,
            "new direction",
            "redirect-1",
            kind="redirect",
            target_execution_id=execution.execution_id,
        )
        response = await client.post("/v1/pwa/commands", json=redirect)
        assert response.status == 200, await response.text()
        accepted = await response.json()
        assert (
            accepted["receipt_kind"] == "redirect"
            and accepted["new_direction"]["application_state"] == "pending"
        )
        confirmation = {
            **redirect,
            "type": "redirect_confirm",
            "command_id": "confirm-1",
            "payload": {
                "redirect_command_id": "redirect-1",
                "confirmation_revision": accepted["confirmation"]["revision"],
                "allow_unresolved_remote": True,
            },
        }
        response = await client.post("/v1/pwa/commands", json=confirmation)
        assert response.status == 200, await response.text()
        assert (await response.json())["confirmation_state"] == "recorded"
        await server.ingress.controls._reconcile()
        assert JournalAgent.effects == ["first"]
        assert server.ingress._owner(root)[2] == execution
        JournalAgent.gates[0].set()
        await started()
        response = await client.get("/v1/pwa/commands/redirect-1")
        assert response.status == 200, await response.text()
        applied = await response.json()
        assert (
            applied["receipt_kind"] == "redirect"
            and applied["new_direction"]["application_state"] == "applied"
        )
        assert applied["release_basis"]["remote_gate"] == "explicit_confirmation"
        assert applied["native_release"] == "released"
        replay = await server.ingress.events.recover(OWNER, root, beginning)
        related = [
            event
            for event in replay["events"]
            if event["payload"].get("command_id") == "redirect-1"
        ]
        assert related and {event["type"] for event in related} == {
            "redirect_state_changed"
        }
        assert "input_applied" in {event["payload"]["change"] for event in related}
        assert all(
            receipt["command_id"] != "redirect-1"
            for receipt in replay["snapshot"]["command_receipts"]
        )
        JournalAgent.gates[1].set()
        await finish(server.ingress)
