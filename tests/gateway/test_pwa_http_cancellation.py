"""Actual HTTP cancellation, receipt pagination and shared recovery capture."""

import pytest

from agent.native_execution_context import NativeExecutionOrigin
from tests.gateway.test_native_cancellation import initialized_interrupt_state, settled
from tests.gateway.test_native_commands import JournalAgent, command, finish, started
from tests.gateway.test_native_events import provider_loop
from tests.gateway.test_pwa_http import service
from tests.gateway.test_pwa_http_mount import actual_writer, cursor, frame
from tests.state.test_native_cancellation import cancel


@pytest.mark.asyncio
async def test_mounted_cancel_keeps_writer_and_projects_private_recovery(monkeypatch, tmp_path):
    async with service(monkeypatch, tmp_path) as (server, client, native):
        actual_writer(monkeypatch)

        def loop(agent, *args, **kwargs):
            result = provider_loop(agent, *args, **kwargs)
            return {**result, "interrupted": bool(agent._interrupt_requested),
                    "completed": not agent._interrupt_requested}

        monkeypatch.setattr("agent.conversation_loop.run_conversation", loop)
        db, root = native[2], native[4].session_id
        capabilities = await (await client.get('/v1/pwa/capabilities')).json()
        assert next(c for c in capabilities['capabilities']
                    if c['capability_id'] == 'remote_cancellation')['availability'] == 'available'
        sent = await client.post('/v1/pwa/commands', json=command(root))
        assert sent.status == 200
        first = await started()
        execution = server.ingress._owner(root)[2]
        origin = NativeExecutionOrigin(root, execution.execution_id, server.ingress._command_owner)
        # Real durable dispatch evidence without a configured/cancellable peer;
        # no remote network traffic is created by this integration test.
        for index in range(2):
            db.native_remote_dispatch_prepare(origin, None, f'unknown-dispatch-{index}')
        initial = await (await client.get(f'/v1/pwa/conversations/{root}/recovery')).json()
        body = cancel(root, execution.execution_id)
        accepted = await client.post('/v1/pwa/commands', json=body)
        assert accepted.status == 200
        receipt = await accepted.json()
        assert receipt['receipt_kind'] == 'cancel' and receipt['durability'] == 'durable'
        await settled(server.ingress.cancellations)
        assert first._interrupt_requested
        assert server.ingress._owner(root)[2] == execution
        exact = await (await client.get(
            f'/v1/pwa/conversations/{root}/executions/{execution.execution_id}')).json()
        assert exact['remote_cancellation']['state'] == 'requested'
        one = await (await client.get('/v1/pwa/commands/cancel-1',
                                    params={'remote_limit': '1'})).json()
        assert len(one['remote_targets']) == 1 and one['remote_coverage']['has_more']
        two = await (await client.get('/v1/pwa/commands/cancel-1', params={
            'remote_limit': '1', 'remote_cursor': one['next_remote_cursor']})).json()
        assert len(two['remote_targets']) == 1 and not two['remote_coverage']['has_more']
        assert one['remote_targets'][0]['dispatch_id'] != two['remote_targets'][0]['dispatch_id']
        for query in ({'remote_limit': '0'}, {'remote_limit': '101'}, {'remote_limit': '01'},
                      {'remote_cursor': 'forged'}, {'unknown': '1'}):
            assert (await client.get('/v1/pwa/commands/cancel-1', params=query)).status == 400
        input_id = command(root)['command_id']
        assert (await client.get(f'/v1/pwa/commands/{input_id}',
                                 params={'remote_limit': '50'})).status == 400
        duplicate = await client.post('/v1/pwa/commands', json=body)
        assert duplicate.status == 200
        assert (await duplicate.json())['payload_fingerprint'] == receipt['payload_fingerprint']
        stream = await client.get(f'/v1/pwa/conversations/{root}/events',
                                  params={'cursor': cursor(initial['cursor'])})
        recovered = await frame(stream)
        stream.close()
        snapshot = recovered['snapshot']
        assert snapshot['recent_executions'][0]['remote_cancellation']['state'] == 'requested'
        control = snapshot['control_receipts'][0]
        assert control['remote_targets'] == [] and control['remote_coverage']['known_count'] == 2
        assert control['remote_coverage']['has_more'] and control['next_remote_cursor']
        assert any(e['type'] == 'cancel_state_changed' for e in recovered['events'])
        assert server.ingress._owner(root)[2] == execution
        JournalAgent.gates[0].set()
        await finish(server.ingress)
        assert server.ingress._owner(root) is None
        final = await (await client.get(
            f'/v1/pwa/conversations/{root}/executions/{execution.execution_id}')).json()
        assert final['remote_cancellation']['state'] == 'requested'
        assert final['state'] == 'interrupted'
