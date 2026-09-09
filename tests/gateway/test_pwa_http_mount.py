"""Authenticated HTTP -> existing native runner, delivery ledger and event feed."""

import asyncio
import base64
import copy
import json
from dataclasses import replace

import pytest
from aiohttp import ClientPayloadError
from aiohttp import web

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.pwa_config import PwaHttpConfig, canonical
from gateway.pwa_http import NativePwaHttp
from gateway.telegram_conversations import channel_key
from tests.gateway.test_native_commands import (
    JournalAgent,
    command,
    finish,
    started,
    setup,
)
from tests.gateway.test_native_events import provider_loop
from tests.gateway.test_pwa_http import (
    OWNER,
    TOKEN,
    service,
    config_for,
    headers,
    native_row,
)


def cursor(value):
    return base64.urlsafe_b64encode(canonical(value).encode()).decode().rstrip("=")


async def frame(response):
    lines = []
    while True:
        line = await asyncio.wait_for(response.content.readline(), 5)
        if not line:
            return None
        if line == b"\n":
            break
        lines.append(line.decode().rstrip("\n"))
    fields = dict(line.split(": ", 1) for line in lines)
    assert fields["event"] == "recovery"
    result = json.loads(fields["data"])
    assert fields["id"] == cursor(result["cursor"])
    return result


async def closed_body(response):
    try:
        return await response.content.read()
    except ClientPayloadError:
        # A hard lifetime/failed write closes the transport without waiting for
        # an unbounded chunked EOF drain. Already received frames remain valid.
        return b""


def actual_writer(monkeypatch):
    initialize = JournalAgent.__init__

    def init(self, **kwargs):
        initialize(self, **kwargs)
        self.session_id = kwargs["session_id"]

    monkeypatch.setattr(JournalAgent, "__init__", init)
    monkeypatch.setattr("agent.conversation_loop.run_conversation", provider_loop)


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["pwa", "telegram"])
async def test_http_binding_cas_two_roots_exact_execution_and_delivery_replay(
    monkeypatch, tmp_path, origin
):
    async with service(monkeypatch, tmp_path) as (server, client, native):
        actual_writer(monkeypatch)
        runner, _, db, store, entry, source, adapter = native
        root = entry.session_id
        initial = await (
            await client.get(f"/v1/pwa/conversations/{root}/recovery")
        ).json()
        selected = await (
            await client.get(f"/v1/pwa/conversations/{root}/telegram-binding")
        ).json()
        assert selected["binding_version"] == 1
        other = await (
            await client.post(
                "/v1/pwa/conversations",
                json={"schema_version": "1.0", "create_id": "other"},
            )
        ).json()
        other = other["conversation_id"]
        native_task = None
        if origin == "pwa":
            await client.post("/v1/pwa/commands", json=command(root, "A", "a"))
        else:
            native_task = asyncio.create_task(
                runner._handle_message(MessageEvent(text="A", source=replace(source)))
            )
        await started()
        path = f"/v1/pwa/conversations/{other}/telegram-binding"
        # Race two identical expectations: only the first can change the pointer.
        answers = await asyncio.gather(*[
            client.put(
                path, json={"schema_version": "1.0", "expected_binding_version": 1}
            )
            for _ in range(2)
        ])
        assert sorted(r.status for r in answers) == [200, 409]
        repeated = await client.put(
            path, json={"schema_version": "1.0", "expected_binding_version": 2}
        )
        assert (
            repeated.status == 200 and (await repeated.json())["binding_version"] == 2
        )
        await client.post("/v1/pwa/commands", json=command(other, "B", "b"))
        await started()
        assert JournalAgent.effects == ["A", "B"]
        JournalAgent.gates[0].set()
        JournalAgent.gates[1].set()
        await finish(server.ingress)
        if native_task:
            await native_task
        for conversation, expected in [(root, "skipped"), (other, "delivered")]:
            recovered = await (
                await client.get(
                    f"/v1/pwa/conversations/{conversation}/recovery",
                    params={"cursor": cursor(initial["cursor"])},
                )
            ).json()
            execution = recovered["snapshot"]["recent_executions"][0]
            assert execution["state"] == "completed"
            assert execution["deliveries"] == [
                {"channel": "telegram", "state": expected, "binding_version": 2}
            ]
            eid = execution["execution_id"]
            exact = await client.get(
                f"/v1/pwa/conversations/{conversation}/executions/{eid}"
            )
            assert exact.status == 200 and await exact.json() == execution
            assert (
                await client.get(
                    f"/v1/pwa/conversations/{other if conversation == root else root}/executions/{eid}"
                )
            ).status == 404
            assert len([
                e for e in recovered["events"] if e["type"] == "delivery_changed"
            ]) == (1 if conversation == root else 0)
            assert not db.native_delivery_finish(eid, channel_key(source), "delivered")
            history = await (
                await client.get(f"/v1/pwa/conversations/{conversation}/history")
            ).json()
            assert len([r for r in history["messages"] if r["role"] == "user"]) == 1
        assert (
            len([m for m in adapter.sent if (m.get("metadata") or {}).get("notify")])
            == 1
        )
        assert not any(m.get("content") in {"A", "B"} for m in adapter.sent)


@pytest.mark.asyncio
async def test_two_streams_disconnect_and_listener_close_do_not_cancel_writer(
    monkeypatch, tmp_path
):
    async with service(monkeypatch, tmp_path) as (server, client, native):
        actual_writer(monkeypatch)
        root = native[4].session_id
        path = f"/v1/pwa/conversations/{root}/events"
        one, two = await asyncio.gather(client.get(path), client.get(path))
        first, second = await asyncio.gather(frame(one), frame(two))
        assert first["cursor"] == second["cursor"]
        assert server._requests == 0 and len(server.ingress.events._subscriptions) == 2
        await client.post("/v1/pwa/commands", json=command(root))
        await started()
        a, b = await asyncio.gather(frame(one), frame(two))
        assert a["events"] == b["events"] and a["events"]
        one.close()
        assert native[2].native_execution(root)
        await server.close()
        assert not server.ingress.events._subscriptions
        assert native[2].native_execution(root) and JournalAgent.effects == ["first"]
        JournalAgent.gates[0].set()
        await finish(server.ingress)
        assert not native[2].native_execution(root)
        two.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "suffix,header,status",
    [
        ("?cursor=%%%", {}, 400),
        ("?cursor=" + cursor(None), {}, 200),
        ("?cursor=" + cursor({"epoch": "wrong", "sequence": -1}), {}, 200),
        ("?cursor=" + cursor({}) + "&cursor=" + cursor({}), {}, 400),
        ("?cursor=" + cursor({}), {"Last-Event-ID": cursor({})}, 400),
        ("", {"Last-Event-ID": "!"}, 400),
    ],
)
async def test_stream_cursor_encoding_and_semantic_gaps(
    monkeypatch, tmp_path, suffix, header, status
):
    async with service(monkeypatch, tmp_path) as (_, client, native):
        response = await client.get(
            f"/v1/pwa/conversations/{native[4].session_id}/events" + suffix,
            headers=header,
        )
        assert response.status == status
        if status == 200:
            assert (await frame(response))["status"] == "gap"
        response.close()


@pytest.mark.asyncio
async def test_stream_capacity_is_separate_and_lifetime_releases_subscriber(
    monkeypatch, tmp_path
):
    async with service(monkeypatch, tmp_path) as (server, client, native):
        server.config = replace(server.config, max_requests=1, stream_max_seconds=1)
        native[2]._native_events_limits = replace(
            native[2]._native_events_limits, max_subscribers=1
        )
        path = f"/v1/pwa/conversations/{native[4].session_id}/events"
        response = await client.get(path)
        await frame(response)
        assert (await client.get(path)).status == 429
        assert (await client.get("/v1/pwa/health")).status == 200
        assert await closed_body(response) == b""
        assert not server.ingress.events._subscriptions


@pytest.mark.asyncio
@pytest.mark.parametrize("after_prepare", [False, True])
async def test_authorization_is_checked_after_await_before_stream_publication(
    monkeypatch, tmp_path, after_prepare
):
    async with service(monkeypatch, tmp_path) as (server, client, native):
        runner = native[0]
        if after_prepare:
            original = web.StreamResponse.prepare

            async def prepare(self, request):
                result = await original(self, request)
                if self.content_type == "text/event-stream":
                    runner._is_user_authorized_for_source = lambda _: False
                return result

            monkeypatch.setattr(web.StreamResponse, "prepare", prepare)
        else:
            original = server.ingress.events.recover

            async def recover(*args, **kwargs):
                result = await original(*args, **kwargs)
                runner._is_user_authorized_for_source = lambda _: False
                return result

            monkeypatch.setattr(server.ingress.events, "recover", recover)
        response = await client.get(
            f"/v1/pwa/conversations/{native[4].session_id}/events"
        )
        if after_prepare:
            assert response.status == 200 and await closed_body(response) == b""
        else:
            assert response.status == 404
        assert not server.ingress.events._subscriptions


@pytest.mark.asyncio
async def test_slow_writer_is_closed_without_execution_control(monkeypatch, tmp_path):
    async with service(monkeypatch, tmp_path) as (server, client, native):
        original = web.StreamResponse.write

        async def blocked(self, data):
            if self.content_type == "text/event-stream":
                await asyncio.Event().wait()
            return await original(self, data)

        monkeypatch.setattr(web.StreamResponse, "write", blocked)
        server.config = replace(server.config, stream_write_timeout=1)
        response = await client.get(
            f"/v1/pwa/conversations/{native[4].session_id}/events"
        )
        assert await closed_body(response) == b""
        assert not server.ingress.events._subscriptions
        assert JournalAgent.effects == []


@pytest.mark.asyncio
async def test_binding_unbound_cas_and_primary_write_failure_roll_back(
    monkeypatch, tmp_path
):
    async with service(monkeypatch, tmp_path) as (server, client, native):
        db, store, entry = native[2:5]
        # Configured fresh channel, existing authorized root, no channel pointer.
        with store._lock:
            store._entries.pop(entry.session_key)
            store._save()
        path = f"/v1/pwa/conversations/{entry.session_id}/telegram-binding"
        unbound = await (await client.get(path)).json()
        assert unbound["state"] == "unbound" and unbound["binding_version"] == 0
        original = db.replace_gateway_routing_entries
        monkeypatch.setattr(
            db,
            "replace_gateway_routing_entries",
            lambda *a, **k: (_ for _ in ()).throw(OSError("private failure")),
        )
        response = await client.put(
            path, json={"schema_version": "1.0", "expected_binding_version": 0}
        )
        assert response.status == 503
        assert store.lookup_by_session_key(entry.session_key) is None
        monkeypatch.setattr(db, "replace_gateway_routing_entries", original)
        count = db._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        response = await client.put(
            path, json={"schema_version": "1.0", "expected_binding_version": 0}
        )
        assert (
            response.status == 200 and (await response.json())["binding_version"] == 1
        )
        assert db._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == count
        assert (
            await client.put(
                path, json={"schema_version": "1.0", "expected_binding_version": 0}
            )
        ).status == 409


@pytest.mark.asyncio
async def test_binding_selected_inaccessible_root_is_never_disclosed(
    monkeypatch, tmp_path
):
    async with service(monkeypatch, tmp_path) as (server, client, native):
        db, store, entry, source = native[2:6]
        native_row(db, replace(source, user_id="foreign"), "private-root")
        old = store.lookup_by_session_key(entry.session_key)
        old.session_id = "private-root"
        response = await client.get(
            f"/v1/pwa/conversations/{entry.session_id}/telegram-binding"
        )
        assert response.status == 404 and "private-root" not in await response.text()


@pytest.mark.asyncio
async def test_exact_execution_outside_snapshot_bound_and_delivery_event_gap(
    monkeypatch, tmp_path
):
    async with service(monkeypatch, tmp_path) as (server, client, native):
        actual_writer(monkeypatch)
        db, root, source = native[2], native[4].session_id, native[5]
        initial = db.native_event_recovery(root, server.ingress._command_scope(OWNER))[
            "cursor"
        ]
        native[2]._native_events_limits = replace(
            native[2]._native_events_limits, snapshot_count=1
        )
        original = db._native_event_append

        def append(conn, root, eid, kind, *args, **kwargs):
            if kind == "delivery_changed":
                raise OSError("optional observation lost")
            return original(conn, root, eid, kind, *args, **kwargs)

        monkeypatch.setattr(db, "_native_event_append", append)
        ids = []
        for index in range(2):
            receipt = await (
                await client.post(
                    "/v1/pwa/commands", json=command(root, str(index), str(index))
                )
            ).json()
            await started()
            ids.append(db.native_execution(root)["execution_id"])
            JournalAgent.gates[index].set()
            await finish(server.ingress)
        recovery = await (
            await client.get(
                f"/v1/pwa/conversations/{root}/recovery",
                params={"cursor": cursor(initial)},
            )
        ).json()
        assert (
            recovery["status"] == "gap"
            and recovery["snapshot"]["coverage"]["executions_has_more"]
        )
        assert recovery["snapshot"]["recent_executions"][0]["execution_id"] == ids[1]
        exact = await client.get(f"/v1/pwa/conversations/{root}/executions/{ids[0]}")
        body = await exact.json()
        assert (
            exact.status == 200
            and body["state"] == "completed"
            and body["deliveries"][0]["state"] == "delivered"
        )
        assert (
            db.native_delivery_lookup(ids[0], channel_key(source))["state"]
            == "delivered"
        )
        assert JournalAgent.effects == ["0", "1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("second_platform", ["telegram", "discord"])
async def test_configured_shared_identity_and_distinct_colliding_native_key(
    monkeypatch, tmp_path, second_platform
):
    native = setup(monkeypatch, tmp_path)
    runner, old, db, store, entry, source, _ = native
    old.stop_command_recovery()
    del runner.conversation_ingress
    raw = config_for(source)
    shared = copy.deepcopy(raw["bindings"][0])
    shared["subject"] = "second-owner"
    raw["bindings"].append(shared)
    runner.config.pwa_http = raw
    monkeypatch.setenv("HERMES_PWA_FACADE_TOKEN", TOKEN)
    server = NativePwaHttp(runner)
    try:
        assert len(server.ownership.config.bindings) == 2
        assert (await server.telegram_channel.inspect(OWNER, entry.session_id))[
            "binding_version"
        ] == 1
    finally:
        await server.close()
        server.ingress.stop_command_recovery()
    del runner.conversation_ingress
    raw["bindings"][1]["sources"][0]["user_id"] = "distinct-user"
    # Explicitly exercise a native key policy collapsing distinct source IDs.
    monkeypatch.setattr(runner, "_session_key_for_source", lambda _: entry.session_key)
    raw["bindings"][1]["sources"][0]["platform"] = second_platform
    if second_platform == "telegram":
        with pytest.raises(ValueError, match="distinct native identities"):
            NativePwaHttp(runner)
    else:
        mixed = NativePwaHttp(runner)
        await mixed.close()
    runner.conversation_ingress.stop_command_recovery()
    db.close()


@pytest.mark.parametrize(
    "change",
    [
        {"event_limits": {"max_subscribers": 0}},
        {"event_limits": {"max_count": True}},
        {"event_limits": {"max_count": 65537}},
        {"event_limits": {"unknown": 1}},
        {"event_limits": {"max_bytes": 1}},
        {"stream_max_seconds": 301},
        {"stream_write_timeout": 0},
        {"event_poll_interval": 0},
    ],
)
def test_event_and_stream_config_is_strict_and_bounded(change):
    from tests.gateway.test_42039_duplicate_user_message import _source

    raw = config_for(_source())
    raw.update(change)
    with pytest.raises(ValueError):
        PwaHttpConfig.from_dict(raw)


@pytest.mark.asyncio
async def test_delivery_snapshot_and_cursor_share_transaction(monkeypatch, tmp_path):
    import threading

    async with service(monkeypatch, tmp_path) as (server, client, native):
        db, root, source = native[2], native[4].session_id, native[5]
        eid = "synthetic-snapshot-execution"
        db.native_execution_open(root, eid, server.ingress._command_owner, origin="pwa")
        db.native_delivery_reserve(eid, channel_key(source), root, 1)
        before = db.native_event_recovery(root, server.ingress._command_scope(OWNER))[
            "cursor"
        ]
        entered, competing = threading.Event(), threading.Event()
        original = db._native_execution_delivery

        def projection(conn, row, key):
            entered.set()
            assert competing.wait(5)
            original(conn, row, key)

        monkeypatch.setattr(db, "_native_execution_delivery", projection)

        def commit():
            assert entered.wait(5)
            competing.set()
            db.native_delivery_finish(eid, channel_key(source), "delivered")

        worker = asyncio.create_task(asyncio.to_thread(commit))
        response = await client.get(
            f"/v1/pwa/conversations/{root}/recovery", params={"cursor": cursor(before)}
        )
        first = await response.json()
        assert response.status == 200
        assert (
            first["snapshot"]["recent_executions"][0]["deliveries"][0]["state"]
            == "unknown"
        )
        assert first["cursor"] == before
        await worker
        second = await (
            await client.get(
                f"/v1/pwa/conversations/{root}/recovery",
                params={"cursor": cursor(first["cursor"])},
            )
        ).json()
        assert (
            second["snapshot"]["recent_executions"][0]["deliveries"][0]["state"]
            == "delivered"
        )
        assert [e["type"] for e in second["events"]] == ["delivery_changed"]


@pytest.mark.asyncio
async def test_branch_and_compression_binding_resolve_exact_root(monkeypatch, tmp_path):
    async with service(monkeypatch, tmp_path) as (server, client, native):
        db, entry, source = native[2], native[4], native[5]
        root = entry.session_id
        native_row(
            db,
            source,
            "user-branch",
            parent_session_id=root,
            model_config={"_branched_from": root},
        )
        binding = await client.put(
            "/v1/pwa/conversations/user-branch/telegram-binding",
            json={"schema_version": "1.0", "expected_binding_version": 1},
        )
        assert (
            binding.status == 200
            and (await binding.json())["selected_conversation_id"] == "user-branch"
        )
        db.end_session("user-branch", "compression")
        native_row(db, source, "branch-tip", parent_session_id="user-branch")
        selected = await (
            await client.get("/v1/pwa/conversations/user-branch/telegram-binding")
        ).json()
        assert (
            selected["selected_conversation_id"] == "user-branch"
            and selected["native_session_id"] == "branch-tip"
        )
        assert selected["binding_version"] == 2
        for tail in [
            "telegram-binding",
            "recovery",
            "events",
            "executions/nonexistent",
        ]:
            response = await client.get("/v1/pwa/conversations/branch-tip/" + tail)
            assert response.status == 404
        # A user branch is independent; compression aliases never become paths.
        assert (
            await client.get(f"/v1/pwa/conversations/{root}/recovery")
        ).status == 200


@pytest.mark.asyncio
async def test_exact_execution_pending_completion_and_foreign_owner_are_unknown(
    monkeypatch, tmp_path
):
    async with service(monkeypatch, tmp_path) as (server, client, native):
        db, root = native[2], native[4].session_id
        eid = "pending-execution"
        db.native_execution_open(root, eid, server.ingress._command_owner, origin="pwa")
        with server.ingress._completion_lock:
            server.ingress._completion_observations[eid] = None
        response = await client.get(f"/v1/pwa/conversations/{root}/executions/{eid}")
        assert response.status == 200 and (await response.json())["state"] == "unknown"
        with server.ingress._completion_lock:
            server.ingress._completion_observations.clear()
        db._execute_write(
            lambda conn: conn.execute(
                "UPDATE native_executions SET owner='other-instance' WHERE execution_id=?",
                (eid,),
            )
        )
        response = await client.get(f"/v1/pwa/conversations/{root}/executions/{eid}")
        assert response.status == 200 and (await response.json())["state"] == "unknown"


@pytest.mark.asyncio
async def test_nontelegram_capabilities_and_binding_are_unavailable(
    monkeypatch, tmp_path
):
    async with service(monkeypatch, tmp_path) as (server, client, native):
        owner = server.ownership.config.bindings[0]
        source = replace(owner.sources[0].source, platform=Platform.DISCORD)
        binding = replace(owner, sources=(replace(owner.sources[0], source=source),))
        server.ownership.config = replace(server.ownership.config, bindings=(binding,))
        native_row(native[2], source, "discord-root")
        response = await client.get(
            "/v1/pwa/conversations/discord-root/telegram-binding"
        )
        assert (
            response.status == 404
            and (await response.json())["error"]["code"] == "capability_unavailable"
        )
        caps = (await (await client.get("/v1/pwa/capabilities")).json())["capabilities"]
        assert all(
            c["availability"] == "unavailable"
            for c in caps
            if c["capability_id"] in {"telegram_binding", "telegram_final_delivery"}
        )
        assert (
            await client.get("/v1/pwa/conversations/discord-root/recovery")
        ).status == 200


@pytest.mark.asyncio
async def test_oversized_first_stream_frame_fails_before_headers(monkeypatch, tmp_path):
    async with service(monkeypatch, tmp_path) as (server, client, native):
        original = server.ingress.events.recover

        async def recover(*args, **kwargs):
            result = await original(*args, **kwargs)
            result["detail"] = {"reason": "x" * server.config.max_response_bytes}
            return result

        monkeypatch.setattr(server.ingress.events, "recover", recover)
        response = await client.get(
            f"/v1/pwa/conversations/{native[4].session_id}/events"
        )
        assert response.status == 503 and response.content_type == "application/json"
        assert not server.ingress.events._subscriptions


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["initial", "late_write"])
async def test_total_stream_lifetime_bounds_capture_and_late_write(monkeypatch, tmp_path, stage):
    import time
    async with service(monkeypatch, tmp_path) as (server, client, native):
        actual_writer(monkeypatch)
        root = native[4].session_id
        await client.post("/v1/pwa/commands", json=command(root))
        await started()
        server.config = replace(server.config, stream_max_seconds=1, stream_write_timeout=5, request_timeout=5)
        recover = server.ingress.events.recover
        async def delayed(*args, **kwargs):
            if stage == "initial":
                await asyncio.Event().wait()
            result = await recover(*args, **kwargs)
            await asyncio.sleep(0.6)
            return result
        monkeypatch.setattr(server.ingress.events, "recover", delayed)
        if stage == "late_write":
            write = web.StreamResponse.write
            async def blocked(self, data):
                if self.content_type == "text/event-stream":
                    await asyncio.Event().wait()
                return await write(self, data)
            monkeypatch.setattr(web.StreamResponse, "write", blocked)
        start = time.monotonic()
        response = await client.get(f"/v1/pwa/conversations/{root}/events")
        await closed_body(response)
        assert time.monotonic() - start < 2.5  # generous scheduler margin, below either configured5s limit
        assert response.status == (503 if stage == "initial" else 200)
        assert not server.ingress.events._subscriptions
        assert native[2].native_execution(root) and JournalAgent.effects == ["first"]
        JournalAgent.gates[0].set()
        await finish(server.ingress)
