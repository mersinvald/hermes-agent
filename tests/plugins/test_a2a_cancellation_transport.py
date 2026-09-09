"""Real loopback HTTP deadlines and bounded residual DNS, no external peer."""

import asyncio
import json
import time
from dataclasses import replace

import pytest
from aiohttp import web

from plugins.platforms.a2a import cancellation as control
from tests.plugins.test_a2a_cancellation import peer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["headers", "trickle", "redirect", "oversize", "proxy"]
)
async def test_actual_transport_deadline_redirect_byte_and_proxy_boundary(
    peer, monkeypatch, mode
):
    config, target, _, _, _, _ = peer
    calls = []
    disconnected = asyncio.Event()

    async def endpoint(request):
        body = await request.json()
        calls.append(body["method"])
        payload = {
            "jsonrpc": "2.0",
            "id": body["id"],
            "result": {
                "id": target.task_id,
                "contextId": target.context_id,
                "status": {
                    "state": "TASK_STATE_WORKING"
                    if body["method"] == "GetTask"
                    else "TASK_STATE_CANCELED"
                },
            },
        }
        if body["method"] == "GetTask" or mode == "proxy":
            return web.json_response(payload)
        if mode == "redirect":
            raise web.HTTPTemporaryRedirect(location="/forbidden")
        if mode == "oversize":
            return web.Response(
                body=b" " * (control.MAX_RESPONSE_BYTES + 1),
                content_type="application/json",
            )
        if mode == "headers":
            for _ in range(200):
                if request.transport is None or request.transport.is_closing():
                    disconnected.set()
                    break
                await asyncio.sleep(0.02)
            return web.json_response(payload)
        response = web.StreamResponse(headers={"Content-Type": "application/json"})
        await response.prepare(request)
        try:
            for byte in json.dumps(payload).encode():
                await response.write(bytes([byte]))
                await asyncio.sleep(0.03)
        except (ConnectionResetError, RuntimeError):
            disconnected.set()
        return response

    app = web.Application()
    app.router.add_post("/agent/specialist/", endpoint)
    app.router.add_post("/forbidden", lambda request: pytest.fail("redirect followed"))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    url = (
        f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/agent/specialist/"
    )
    config["a2a_agents"]["peer"].update(url=url, timeout=1)
    target = replace(
        target,
        rpc_endpoint=url,
        configured_endpoint_fingerprint=control.endpoint_fingerprint(url),
    )
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.setenv(key, "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    client = control.TaskControllerClient(max_connections=2)
    reserved = []
    begin = time.monotonic()
    try:
        observed = await client.cancel_once(
            target, before_send=lambda: reserved.append(True)
        )
        elapsed = time.monotonic() - begin
        assert elapsed < 2.5
        assert calls == ["GetTask", "CancelTask"] and reserved == [True]
        assert observed.write_attempted
        assert observed.cancel_state == (
            "acknowledged" if mode == "proxy" else "unknown"
        )
        assert len(client._session.connector._acquired) == 0
        if mode in {"headers", "trickle"}:
            await asyncio.wait_for(disconnected.wait(), 1)
        # Explicit reconciliation reads. The write is never automatically retried.
        await client.observe(target, cancellation_requested=True)
        assert calls == ["GetTask", "CancelTask", "GetTask"]
    finally:
        await client.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_dns_waiters_are_deduplicated_bounded_and_cannot_later_send(monkeypatch):
    resolver = control._BoundedResolver(2)
    gate = asyncio.Event()
    calls = []

    async def resolve(host, port, family):
        calls.append(host)
        await gate.wait()
        return []

    monkeypatch.setattr(resolver._resolver, "resolve", resolve)

    async def deadline(host):
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(resolver.resolve(host, 80), 0.02)

    await asyncio.gather(deadline("one"), deadline("one"), deadline("two"))
    assert calls == ["one", "two"] and len(resolver._pending) == 2
    with pytest.raises(OSError, match="capacity"):
        await resolver.resolve("three", 80)
    gate.set()
    await asyncio.gather(*resolver._pending.values())
    await asyncio.sleep(0)
    assert not resolver._pending
    await resolver.close()


@pytest.mark.asyncio
async def test_timed_out_http_dns_wait_never_sends_after_resolver_finishes(
    peer, monkeypatch
):
    import socket

    config, target, _, _, _, _ = peer
    calls = []

    async def endpoint(request):
        body = await request.json()
        calls.append(body["method"])
        return web.json_response(
            dict(
                jsonrpc="2.0",
                id=body["id"],
                result=dict(
                    id=target.task_id,
                    contextId=target.context_id,
                    status=dict(state="TASK_STATE_WORKING"),
                ),
            )
        )

    app = web.Application()
    app.router.add_post("/agent/specialist/", endpoint)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    url = f"http://delayed-dns.invalid:{port}/agent/specialist/"
    config["a2a_agents"]["peer"].update(url=url, timeout=1)
    target = replace(
        target,
        rpc_endpoint=url,
        configured_endpoint_fingerprint=control.endpoint_fingerprint(url),
    )
    gate = asyncio.Event()

    async def resolve(host, port, family):
        await gate.wait()
        return [
            dict(
                hostname=host,
                host="127.0.0.1",
                port=port,
                family=socket.AF_INET,
                proto=0,
                flags=0,
            )
        ]

    client = control.TaskControllerClient(max_connections=1)
    client._resolver = control._BoundedResolver(1)
    monkeypatch.setattr(client._resolver._resolver, "resolve", resolve)
    try:
        assert (await client.observe(target)).task_state == "unknown"
        gate.set()
        await asyncio.sleep(0.05)
        assert calls == []
        assert not client._session.connector._acquired
        assert (await client.observe(target)).task_state == "working"
        assert calls == ["GetTask"]
    finally:
        gate.set()
        await client.close()
        await runner.cleanup()
