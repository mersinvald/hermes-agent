"""Owner-controller transport only; no native owner admission is implied here."""

import copy
import json
from dataclasses import replace
from urllib.error import URLError

import pytest

from plugins.platforms.a2a import cancellation as control, tools


@pytest.fixture
def peer(monkeypatch):
    config = {
        "a2a_agents": {
            "peer": {
                "url": "http://peer.test/agent/specialist/",
                "tenant": "configured-tenant",
                "timeout": 180,
                "auth": {"type": "bearer", "token": "SYNTHETIC_CANCEL_TOKEN"},
            }
        }
    }
    monkeypatch.setattr(tools, "_load_config", lambda: config)
    target = control.RecordedTask(
        "dispatch-1",
        "peer",
        control.endpoint_fingerprint(config["a2a_agents"]["peer"]["url"]),
        "http://peer.test/agent/specialist/",
        "1.0",
        "card-tenant",
        "task-1",
        "context-1",
        configured_tenant="configured-tenant",
    )
    calls, reservations = [], []
    options = {
        "GetTask": "working",
        "CancelTask": "canceled",
        "tasks/get": "working",
        "tasks/cancel": "canceled",
    }

    def post(url, body, headers, timeout):
        calls.append((url, copy.deepcopy(body), dict(headers), timeout))
        response = options[body["method"]]
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return response(body)
        state = (
            response
            if headers["A2A-Version"].startswith("0.3")
            else "TASK_STATE_" + response.replace("-", "_").upper()
        )
        return {
            "jsonrpc": "2.0",
            "id": body["id"],
            "result": {
                "id": "task-1",
                "contextId": "context-1",
                "status": {"state": state},
            },
        }

    client = control.TaskControllerClient(post=post)
    return config, target, client, calls, reservations, options


@pytest.mark.parametrize("version", ["1.0", "1.0.0", "0.3", "0.3.0"])
@pytest.mark.asyncio
async def test_once_then_read_reconciliation_preserves_exact_binding(peer, version):
    _, target, client, calls, reserved, options = peer
    target = replace(target, protocol_version=version)
    result = await client.cancel_once(
        target, before_send=lambda: reserved.append("durable")
    )
    assert result == control.ControlObservation(
        "acknowledged", "canceled", "cancel_rpc_accepted", True
    )
    get, cancel = (
        ("tasks/get", "tasks/cancel")
        if version.startswith("0.3")
        else ("GetTask", "CancelTask")
    )
    assert [c[1]["method"] for c in calls] == [get, cancel]
    assert reserved == ["durable"]
    assert len({c[1]["id"] for c in calls}) == 2
    assert all(
        c[0] == target.rpc_endpoint and c[2]["A2A-Version"] == version and c[3] == 30
        for c in calls
    )
    assert all(
        c[1]["params"] == {"id": "task-1", "tenant": "card-tenant"} for c in calls
    )
    assert all(c[2]["Authorization"] == "Bearer SYNTHETIC_CANCEL_TOKEN" for c in calls)
    options[get] = "canceled"
    assert (
        await client.observe(target, cancellation_requested=True)
    ).cancel_state == "confirmed"
    assert [c[1]["method"] for c in calls] == [get, cancel, get]


@pytest.mark.parametrize(
    "state,expected",
    [
        ("completed", "already_completed"),
        ("canceled", "confirmed"),
        ("failed", "rejected"),
        ("rejected", "rejected"),
    ],
)
@pytest.mark.asyncio
async def test_already_terminal_never_writes(peer, state, expected):
    _, target, client, calls, reserved, options = peer
    options["GetTask"] = state
    result = await client.cancel_once(target, before_send=lambda: reserved.append(True))
    assert result.cancel_state == expected and result.task_state == state
    assert not result.write_attempted and not reserved and len(calls) == 1


@pytest.mark.parametrize(
    "state", ["working", "submitted", "input-required", "auth-required", "unknown"]
)
@pytest.mark.asyncio
async def test_native_pending_task_states_can_request_cancel(peer, state):
    _, target, client, calls, reserved, options = peer
    options["GetTask"] = state
    assert (
        await client.cancel_once(target, before_send=lambda: reserved.append(True))
    ).write_attempted
    assert len(calls) == 2 and reserved == [True]


@pytest.mark.asyncio
async def test_durable_reservation_failure_prevents_cancel_write(peer):
    _, target, client, calls, _, _ = peer

    def unavailable():
        raise OSError("storage unavailable")

    result = await client.cancel_once(target, before_send=unavailable)
    assert result.cancel_state == "failed" and not result.write_attempted
    assert len(calls) == 1


@pytest.mark.parametrize("stage", ["GetTask", "CancelTask"])
@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("SYNTHETIC_CANCEL_TOKEN"),
        URLError("secret"),
        ValueError("broken response"),
    ],
)
@pytest.mark.asyncio
async def test_ambiguous_transport_never_retries_write(peer, stage, error):
    _, target, client, calls, _, options = peer
    options[stage] = error
    result = await client.cancel_once(target, before_send=lambda: None)
    assert result.cancel_state == ("unknown" if stage == "CancelTask" else "failed")
    assert result.write_attempted == (stage == "CancelTask")
    assert len(calls) == (2 if stage == "CancelTask" else 1)
    assert "TOKEN" not in str(result) and "secret" not in str(result)
    options["GetTask"] = "working"
    assert (
        await client.observe(target, cancellation_requested=True)
    ).cancel_state == "requested"
    assert sum(c[1]["method"] == "CancelTask" for c in calls) == (stage == "CancelTask")


@pytest.mark.parametrize(
    "mutation",
    [
        {"jsonrpc": "1.0"},
        {"id": "foreign"},
        {"id": True},
        {"error": {"code": -32601, "message": "no"}},
        {
            "result": {
                "id": "foreign",
                "contextId": "context-1",
                "status": {"state": "TASK_STATE_WORKING"},
            }
        },
        {
            "result": {
                "id": "task-1",
                "contextId": "foreign",
                "status": {"state": "TASK_STATE_WORKING"},
            }
        },
        {
            "result": {
                "id": "task-1",
                "contextId": "context-1",
                "status": {"state": "working"},
            }
        },
        {
            "result": {
                "id": "task-1",
                "contextId": "context-1",
                "status": {"state": "TASK_STATE_IMAGINARY"},
            }
        },
    ],
)
@pytest.mark.parametrize("stage", ["GetTask", "CancelTask"])
@pytest.mark.asyncio
async def test_strict_envelope_task_context_and_version(peer, mutation, stage):
    _, target, client, calls, _, options = peer
    options[stage] = lambda body: {
        "jsonrpc": "2.0",
        "id": body["id"],
        "result": {
            "id": "task-1",
            "contextId": "context-1",
            "status": {"state": "TASK_STATE_WORKING"},
        },
        **mutation,
    }
    result = await client.cancel_once(target, before_send=lambda: None)
    assert result.cancel_state == ("unknown" if stage == "CancelTask" else "failed")
    assert len(calls) == (2 if stage == "CancelTask" else 1)


@pytest.mark.parametrize("stage", ["GetTask", "CancelTask"])
@pytest.mark.asyncio
async def test_valid_rpc_rejection_is_not_method_fallback(peer, stage):
    _, target, client, calls, _, options = peer
    options[stage] = lambda body: {
        "jsonrpc": "2.0",
        "id": body["id"],
        "error": {"code": -32601, "message": "unsupported"},
    }
    result = await client.cancel_once(target, before_send=lambda: None)
    assert result.cancel_state == "rejected"
    assert len(calls) == (2 if stage == "CancelTask" else 1)


@pytest.mark.parametrize(
    "change",
    [
        {"url": "http://foreign.test/agent/specialist/"},
        {"url": "http://peer.test/agent/other/"},
        {"url": "http://peer.test/agent/specialist"},
        {"tenant": "foreign"},
    ],
)
@pytest.mark.asyncio
async def test_changed_configured_route_or_tenant_prevents_any_request(peer, change):
    config, target, client, calls, _, _ = peer
    config["a2a_agents"]["peer"].update(change)
    assert (
        await client.cancel_once(target, before_send=lambda: None)
    ).cancel_state == "failed"
    assert not calls


@pytest.mark.parametrize(
    "change",
    [
        {"rpc_endpoint": "http://foreign.test/agent/specialist/"},
        {"rpc_endpoint": "http://user:secret@peer.test/agent/specialist/"},
        {"peer_name": "http://peer.test"},
        {"task_id": "SYNTHETIC_CANCEL_TOKEN"},
        {"context_id": ""},
        {"context_id": "x" * 1025},
        {"protocol_version": "2.0"},
    ],
)
@pytest.mark.asyncio
async def test_invalid_recorded_binding_never_reaches_peer(peer, change):
    _, target, client, calls, _, _ = peer
    assert (
        await client.cancel_once(replace(target, **change), before_send=lambda: None)
    ).cancel_state == "failed"
    assert not calls


@pytest.mark.parametrize(
    "payload",
    [b'{"a":1,"a":2}', b"NaN", b"\xff", b"x" * (control.MAX_RESPONSE_BYTES + 1)],
)
def test_control_response_parser_is_strict_and_bounded(monkeypatch, payload):
    with pytest.raises(control.ControlProtocolError):
        control._decode_response(payload)
