"""Reconcile HTTP ownership, event recovery and Telegram's current binding."""
import pytest

from gateway.telegram_conversations import TelegramConversationChannel, channel_key
from tests.gateway.test_native_commands import JournalAgent, command, finish, started
from tests.gateway.test_native_events import provider_loop
from tests.gateway.test_pwa_http import OWNER, service


@pytest.mark.asyncio
async def test_http_created_conversations_share_provider_recovery_and_channel_selection(monkeypatch, tmp_path):
    constructor = JournalAgent.__init__

    def initialize(self, **kwargs):
        constructor(self, **kwargs)
        self.session_id = kwargs["session_id"]

    monkeypatch.setattr(JournalAgent, "__init__", initialize)
    async with service(monkeypatch, tmp_path) as (server, client, native):
        monkeypatch.setattr("agent.conversation_loop.run_conversation", provider_loop)
        runner, _, db, store, entry, source, adapter = native
        policy = TelegramConversationChannel(server.ingress)
        root_a = entry.session_id
        created = await client.post("/v1/pwa/conversations", json={"schema_version": "1.0", "create_id": "integrated-b"})
        assert created.status == 201
        root_b = (await created.json())["conversation_id"]
        subscription = server.ingress.events.subscribe(OWNER, root_a)
        before = await subscription.poll()
        assert before["snapshot"]["conversation"]["conversation_id"] == root_a
        response = await client.post("/v1/pwa/commands", json=command(root_a, "A", "integrated-a"))
        assert response.status == 200
        await started()
        execution_a = policy.execution(server.ingress._owner(root_a)[0])
        running = await subscription.poll()
        assert running["snapshot"]["conversation"]["active_execution"]["execution_id"] == execution_a.execution_id
        binding = await policy.inspect(OWNER, root_a)
        await policy.select(OWNER, root_b, binding["binding_version"])
        response = await client.post("/v1/pwa/commands", json=command(root_b, "B", "integrated-b"))
        assert response.status == 200
        await started()
        execution_b = policy.execution(server.ingress._owner(root_b)[0])
        assert execution_b.execution_id != execution_a.execution_id
        # Closing an observer cannot interrupt either active native execution.
        subscription.close()
        assert db.native_execution(root_a) and db.native_execution(root_b)
        JournalAgent.gates[0].set()
        JournalAgent.gates[1].set()
        await finish(server.ingress)
        assert db.native_delivery_lookup(execution_a.execution_id, channel_key(source))["state"] == "skipped"
        assert db.native_delivery_lookup(execution_b.execution_id, channel_key(source))["state"] == "delivered"
        assert len([m for m in adapter.sent if (m.get("metadata") or {}).get("notify")]) == 1
        for root, text, execution in [(root_a, "A", execution_a), (root_b, "B", execution_b)]:
            recovered = await server.ingress.events.recover(OWNER, root)
            assert recovered["snapshot"]["conversation"]["active_execution"] is None
            assert recovered["snapshot"]["recent_executions"][0]["execution_id"] == execution.execution_id
            assert recovered["snapshot"]["recent_executions"][0]["state"] == "completed"
            response = await client.get(f"/v1/pwa/conversations/{root}/history")
            history = await response.json()
            assert response.status == 200, history
            assert [m["content"] for m in history["messages"] if m["role"] == "user"] == [text]
            assert any(m["role"] == "session_meta" for m in db.get_messages(root))
            assert all(m["role"] != "session_meta" for m in history["messages"])
        runner._is_user_authorized_for_source = lambda _: False
        with pytest.raises(PermissionError):
            await server.ingress.events.recover(OWNER, root_b)
        response = await client.get(f"/v1/pwa/conversations/{root_b}/history")
        assert response.status == 404
