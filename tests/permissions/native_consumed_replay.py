"""Run alongside the pinned service's 11 native gates, in loopback-only Docker.

PYTHONPATH must include the selected service tests/src and Hermes source.
Set the existing NATIVE_GATEWAY_BINARY, NATIVE_KAGENT_PROBE,
NATIVE_KAGENT_SERVER and NATIVE_HERMES_SOURCE artifact paths.
No production endpoints or real human identities are accepted by the lab.
"""
import json

import pytest

from conftest import config, params
from test_native_human_flow import (
    ACTOR, OWNER, PEER, WORKLOAD, detail, lab, streaming_peer, tool, ui,
)


def test_own_consumed_replay_ref(lab, ui, monkeypatch):
    from tools import mcp_tool
    from plugins.platforms.a2a.tools import a2a_call

    name = "own-consumed-replay-ref"
    cfg = {"url": str(lab.client.base_url) + "/mcp",
           "headers": {"Authorization": "Bearer " + WORKLOAD}}
    ui.cfg["mcp_servers"] = {name: {**cfg, "expected_permission_actor": ACTOR}}
    monkeypatch.setattr(mcp_tool, "_servers", {})
    mcp_tool._ensure_mcp_loop()
    try:
        server = mcp_tool._run_on_mcp_loop(lambda: mcp_tool._connect_server(name, cfg), timeout=15)
        mcp_tool._servers[name] = server
        handler = mcp_tool._make_tool_handler(name, "fixture_set_value", 10)
        reference = tool(lab)["error"]["data"]["request_id"]
        canonical = detail(lab, reference)
        assert canonical["action"]["actor"] == ACTOR and canonical["action"]["owner"] == OWNER
        response = lab.human.post(f"/requests/{reference}/decision",
                                  json={"digest": canonical["digest"], "choice": "once"})
        assert response.status_code == 200
        assert detail(lab, reference)["state"] == "approved"
        baseline = lab.fixture.calls, lab.fixture.effects
        args = json.loads(params())["arguments"]
        assert "result" in json.loads(handler(args))
        assert (lab.fixture.calls - baseline[0], lab.fixture.effects - baseline[1]) == (1, 1)
        assert detail(lab, reference)["outcome"] == "result_received"
        before = (lab.fixture.calls, lab.fixture.effects, len(lab.hook.passes), len(lab.hook.responses))
        audit = lab.hook.ledger.listing(OWNER, "audit")
        requests = lab.hook.ledger.listing(OWNER, "requests")
        grants = lab.hook.ledger.listing(OWNER, "grants")
        posts = []
        original = ui.bridge._http
        def readonly(remote, user, chat, body=None):
            assert body is None, "Replay must not POST a decision"
            posts.append(remote.reference)
            return original(remote, user, chat)
        monkeypatch.setattr(ui.bridge, "_http", readonly)
        ui.approval.register_gateway_notify("native-flow", lambda _: pytest.fail("Replay must not enqueue UI"))
        direct = json.loads(handler(args))
        evidence = streaming_peer(lab, ui)
        a2a = a2a_call({"agent": PEER, "message": "Repeat identical synthetic operation"})
        # A2A wraps the native JSON feedback in its existing peer envelope.
        for output in (json.dumps(direct), a2a):
            assert '"permission_status": "consumed"' in output
            assert '"approval_prompt_sent": false' in output
            assert '"action_executed": false' in output
            assert '"priorOutcome": "result_received"' in output
            assert reference in output and "permission_required" not in output
        assert len(posts) == 2 and set(posts) == {reference}
        assert not ui.adapter._bot.send_message.called and not ui.approval._gateway_queues
        assert before == (lab.fixture.calls, lab.fixture.effects, len(lab.hook.passes), len(lab.hook.responses))
        assert audit == lab.hook.ledger.listing(OWNER, "audit")
        assert requests == lab.hook.ledger.listing(OWNER, "requests")
        assert grants == lab.hook.ledger.listing(OWNER, "grants")
        assert sum("execute" in item for item in evidence()) == 1
    finally:
        mcp_tool.shutdown_mcp_servers()
