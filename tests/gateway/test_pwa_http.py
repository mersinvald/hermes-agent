"""Real HTTP/native ingress, durable journal and isolated SQLite behavior."""
import asyncio
import base64
import copy
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import replace

import aiohttp
import pytest

from gateway.conversation_control import Principal
from gateway.pwa_config import PwaHttpConfig
from gateway.pwa_http import NativePwaHttp
from gateway.run import _managed_conversation_title_destination
from tests.gateway.test_native_commands import JournalAgent, command, finish, setup, started

TOKEN = "synthetic-facade-secret-" + "s" * 40
OWNER = Principal("https://issuer.invalid", "owner")


@pytest.mark.parametrize(
    "timeout", [None, True, 0, -1, float("nan"), float("inf"), "7", 10**400]
)
def test_invalid_managed_title_timeout_disables_model_request(timeout):
    assert _managed_conversation_title_destination(
        {
            "auxiliary": {
                "title_generation": {
                    "provider": "custom",
                    "model": "titles/v1",
                    "timeout": timeout,
                }
            }
        }
    ) is False


@pytest.mark.parametrize(
    ("provider", "model", "base_url"),
    [
        ("moa", "titles/v1", "https://titles.invalid/v1"),
        ("custom", "auto", "https://titles.invalid/v1"),
        ("custom", "titles/v1", ""),
        ("custom", "titles/v1", "titles.invalid/v1"),
        ("custom", "titles/v1", "https://[broken"),
        ("custom", "titles/v1", "https://titles.invalid:bad/v1"),
        ("custom", "titles/v1", "https://user:key@titles.invalid/v1"),
        ("custom", "titles/v1", "https://titles.invalid/v1?route=other"),
        ("custom", "titles/v1", "https://titles.invalid/v1#other"),
    ],
)
def test_managed_title_requires_nonvirtual_fixed_origin(provider, model, base_url):
    assert _managed_conversation_title_destination(
        {
            "auxiliary": {
                "title_generation": {
                    "provider": provider,
                    "model": model,
                    "base_url": base_url,
                }
            }
        }
    ) is False


def config_for(source, models=None):
    result = {"enabled": True, "concierge_id": "synthetic", "host": "127.0.0.1", "port": 0,
            "allowed_hosts": ["native.test"], "bindings": [{"issuer": OWNER.issuer,
            "subject": OWNER.subject, "default_source_id": "telegram-owner", "sources": [{
                "source_id": "telegram-owner", "platform": source.platform.value,
                "chat_id": source.chat_id, "chat_type": source.chat_type,
                "user_id": source.user_id, "thread_id": source.thread_id}]}]}
    if models is not None:
        result["models"] = models
    return result


def model_config():
    capabilities = {
        "text_input": "supported",
        "image_input": "unknown",
        "tools": "supported",
        "reasoning_controls": "unknown",
    }
    return {
        "default_model_id": "daily",
        "entries": [
            {
                "model_id": "daily",
                "display_name": "Daily",
                "provider": "provider-a",
                "model": "upstream/daily",
                "capabilities": capabilities,
            },
            {
                "model_id": "deep",
                "display_name": "Deep",
                "provider": "provider-b",
                "model": "upstream/deep",
                "capabilities": capabilities,
            },
        ],
    }


def headers(principal=OWNER, concierge="synthetic"):
    raw = json.dumps({"issuer": principal.issuer, "subject": principal.subject,
                      "concierge_id": concierge}).encode()
    return {"Host": "native.test", "Authorization": "Bearer " + TOKEN,
            "X-Hermes-PWA-Principal": base64.urlsafe_b64encode(raw).decode().rstrip("=")}


@asynccontextmanager
async def service(monkeypatch, tmp_path, *, models=None, resolver=None):
    native = setup(monkeypatch, tmp_path)
    runner, old, db, store, entry, source, adapter = native
    old.stop_command_recovery()
    del runner.conversation_ingress
    runner.config.pwa_http = config_for(source, models)
    if resolver is not None:
        runner._resolve_managed_model_provider = resolver
    runner._running = True
    monkeypatch.setenv("HERMES_PWA_FACADE_TOKEN", TOKEN)
    server = NativePwaHttp(runner)
    await server.start()
    async with aiohttp.ClientSession(base_url=f"http://127.0.0.1:{server.port}",
                                     headers=headers(), timeout=aiohttp.ClientTimeout(total=20)) as client:
        try:
            yield server, client, native
        finally:
            await server.close()
            await finish(server.ingress)
            db.close()


@pytest.mark.asyncio
async def test_real_http_history_discovery_create_retry_and_restart(monkeypatch, tmp_path):
    async with service(monkeypatch, tmp_path) as (server, client, native):
        _, _, db, store, entry, source, _ = native
        db.append_message(entry.session_id, "user", "authorized history")
        foreign = "foreign-root"
        db.create_session(foreign, source.platform.value, user_id="foreign", chat_id=source.chat_id, chat_type=source.chat_type)
        db.append_message(foreign, "user", "FOREIGN SECRET")
        page = await (await client.get("/v1/pwa/conversations")).json()
        assert [c["conversation_id"] for c in page["conversations"]] == [entry.session_id]
        assert "FOREIGN" not in json.dumps(page)
        history = await (await client.get(f"/v1/pwa/conversations/{entry.session_id}/history")).json()
        assert history["messages"][0]["content"] == "authorized history"
        before = store.list_sessions()
        requests = [client.post("/v1/pwa/conversations", json={"schema_version": "1.0", "create_id": "new-1"}) for _ in range(6)]
        responses = await asyncio.gather(*requests)
        bodies = [await response.json() for response in responses]
        assert sorted(r.status for r in responses) == [200]*5 + [201]
        root = bodies[0]["conversation_id"]
        assert {b["conversation_id"] for b in bodies} == {root}
        assert db.get_session(entry.session_id)["ended_at"] is None
        assert store.lookup_by_session_key(before[0].session_key).session_id == entry.session_id
        await server.close()
        del server.runner.conversation_ingress
        replacement = NativePwaHttp(server.runner)
        try:
            repeated, created = await replacement.ownership.create(replacement.ingress, OWNER, "new-1")
            assert repeated["conversation_id"] == root and not created
        finally:
            replacement.ingress.stop_command_recovery()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["conversations/foreign-root", "conversations/foreign-root/history", "commands/foreign-command"])
async def test_foreign_and_missing_resources_are_indistinguishable(monkeypatch, tmp_path, path):
    async with service(monkeypatch, tmp_path) as (_, client, native):
        db, source = native[2], native[5]
        db.create_session("foreign-root", source.platform.value, user_id="foreign", chat_id=source.chat_id, chat_type=source.chat_type)
        response = await client.get("/v1/pwa/" + path)
        assert response.status == 404
        assert (await response.json())["error"]["code"] == "authorization_denied"
        assert "foreign" not in await response.text()


@pytest.mark.asyncio
@pytest.mark.parametrize("change,status", [({"Authorization": "Bearer wrong"},401),
    ({"Host":"foreign.test"},400), ({"Origin":"https://agent.invalid"},400),
    ({"Cookie":"browser=secret"},400), ({"X-Hermes-PWA-Principal":"%%%"},400),
    (headers(Principal(OWNER.issuer,"foreign")),403), (headers(concierge="foreign"),403)])
async def test_service_auth_and_trusted_principal_boundaries(monkeypatch, tmp_path, change, status):
    async with service(monkeypatch, tmp_path) as (_, client, _):
        response = await client.get("/v1/pwa/health", headers={**headers(), **change})
        assert response.status == status
        assert TOKEN not in await response.text()


@pytest.mark.asyncio
@pytest.mark.parametrize("path,body", [("conversations", '{"schema_version":"1.0","create_id":"x","user_id":"forged"}'),
    ("conversations", '{"schema_version":"1.0","create_id":"x","create_id":"y"}'),
    ("commands", '{"schema_version":"1.0","user_id":"forged"}'), ("conversations",'{"schema_version":"1.0","create_id":NaN}')])
async def test_closed_request_objects_and_duplicate_json(monkeypatch, tmp_path, path, body):
    async with service(monkeypatch, tmp_path) as (_, client, _):
        response = await client.post("/v1/pwa/"+path, data=body, headers={"Content-Type":"application/json"})
        assert response.status == 400


@pytest.mark.asyncio
async def test_accepted_http_disconnect_and_listener_close_do_not_cancel_native_work(monkeypatch, tmp_path):
    async with service(monkeypatch, tmp_path) as (server, client, native):
        root = native[4].session_id
        accepted = asyncio.Event()
        response_gate = asyncio.Event()
        original = server.ingress.submit
        async def gated(*args):
            result = await original(*args)
            accepted.set()
            await response_gate.wait()
            return result
        monkeypatch.setattr(server.ingress, "submit", gated)
        request = asyncio.create_task(client.post("/v1/pwa/commands", json=command(root)))
        await asyncio.wait_for(accepted.wait(),10)
        await started()
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        response_gate.set()
        await server.close()
        assert JournalAgent.effects == ["first"]
        receipt = await original(OWNER, command(root))
        assert receipt["durability"] == "durable"
        assert receipt["application_state"] == "applied"
        await finish(server.ingress)
        assert JournalAgent.effects == ["first"]
        assert sum(m["role"] == "user" for m in native[2].get_messages(root)) == 1


@pytest.mark.asyncio
async def test_postcommit_alias_failure_retries_the_reserved_root(monkeypatch, tmp_path):
    async with service(monkeypatch, tmp_path) as (server, client, native):
        bind = server.runner.async_session_store.bind_conversation_alias
        async def fail(*args):
            raise OSError("synthetic alias unavailable")
        monkeypatch.setattr(server.runner.async_session_store, "bind_conversation_alias", fail)
        body = {"schema_version":"1.0","create_id":"commit-before-alias"}
        assert (await client.post("/v1/pwa/conversations",json=body)).status == 503
        reserved = native[2].native_pwa_assignment(server.ownership.scope(OWNER),create_id=body["create_id"])
        assert reserved is not None
        monkeypatch.setattr(server.runner.async_session_store, "bind_conversation_alias", bind)
        response = await client.post("/v1/pwa/conversations",json=body)
        assert response.status == 200
        assert (await response.json())["conversation_id"] == reserved["conversation_id"]


@pytest.mark.asyncio
async def test_cursor_pagination_revocation_and_compaction_invalidation(monkeypatch, tmp_path):
    async with service(monkeypatch, tmp_path) as (server, client, native):
        db, root = native[2], native[4].session_id
        for i in range(5):
            db.append_message(root,"user",f"message {i}")
        path=f"/v1/pwa/conversations/{root}/history"
        first=await (await client.get(path,params={"limit":"2"})).json()
        assert [m["content"] for m in first["messages"]] == ["message 0","message 1"]
        token=first["next_cursor"]
        assert token and root not in token and OWNER.subject not in token
        second=await (await client.get(path,params={"limit":"2","cursor":token})).json()
        assert [m["content"] for m in second["messages"]] == ["message 2","message 3"]
        assert (await client.get(path,params={"limit":"3","cursor":token})).status == 409
        assert (await client.get(path,params={"limit":"2","cursor":token+"x"})).status == 409
        db.append_message(root,"assistant","new write")
        assert (await client.get(path,params={"limit":"2","cursor":token})).status == 409
        server.ownership.config=replace(server.config,bindings=())
        assert (await client.get(path)).status == 403
        assert (await client.post("/v1/pwa/conversations",json={"schema_version":"1.0","create_id":"x"})).status == 403


@pytest.mark.asyncio
async def test_history_cursor_rejects_equal_size_content_rewrite(monkeypatch, tmp_path):
    async with service(monkeypatch, tmp_path) as (_, client, native):
        db, root = native[2], native[4].session_id
        for index in range(3):
            db.append_message(root, "user", f"message {index}")
        path = f"/v1/pwa/conversations/{root}/history"
        first = await (await client.get(path, params={"limit": "1"})).json()
        assert first["next_cursor"]
        db._execute_write(lambda conn: conn.execute(
            "UPDATE messages SET content='changed 1' WHERE session_id=? AND content='message 1'", (root,)))
        response = await client.get(path, params={"limit": "1", "cursor": first["next_cursor"]})
        assert response.status == 409
        assert "changed" not in await response.text()


@pytest.mark.asyncio
async def test_history_cursor_survives_independent_roots_but_not_native_compression(monkeypatch, tmp_path):
    async with service(monkeypatch, tmp_path) as (_, client, native):
        db, root, source = native[2], native[4].session_id, native[5]
        for index in range(3):
            db.append_message(root, "user", f"retained {index}")
        path = f"/v1/pwa/conversations/{root}/history"
        page = await (await client.get(path, params={"limit": "1"})).json()
        query = {"limit": "1", "cursor": page["next_cursor"]}
        native_row(db, source, "independent")
        native_row(db, source, "branch", parent_session_id=root, model_config={"_branched_from": root})
        assert (await client.get(path, params=query)).status == 200
        db.publish_compression_child(parent_session_id=root, child_session_id="tip",
            source=source.platform.value, messages=[{"role": "user", "content": "native handoff"}],
            require_compression_lease=False)
        assert (await client.get(path, params=query)).status == 409
        current = await (await client.get(path)).json()
        assert current["native_session_ids"] == [root, "tip"]


@pytest.mark.asyncio
async def test_history_cursor_fences_first_continuation_of_already_compressed_parent(monkeypatch, tmp_path):
    async with service(monkeypatch, tmp_path) as (_, client, native):
        db, root, source = native[2], native[4].session_id, native[5]
        for index in range(3):
            db.append_message(root, "user", f"retained {index}")
        db.end_session(root, "compression")
        path = f"/v1/pwa/conversations/{root}/history"
        page = await (await client.get(path, params={"limit": "1"})).json()
        native_row(db, source, "late-tip", parent_session_id=root)
        response = await client.get(path, params={"limit": "1", "cursor": page["next_cursor"]})
        assert response.status == 409


@pytest.mark.asyncio
async def test_native_source_revocation_blocks_history_and_receipts(monkeypatch, tmp_path):
    async with service(monkeypatch, tmp_path) as (server, client, native):
        root=native[4].session_id
        response=await client.post("/v1/pwa/commands",json=command(root))
        assert response.status == 200
        await started()
        monkeypatch.setattr(server.runner,"_is_user_authorized_for_source",lambda _:False)
        for path in (f"conversations/{root}", f"conversations/{root}/history", "commands/c1"):
            assert (await client.get("/v1/pwa/"+path)).status == 404


@pytest.mark.asyncio
async def test_history_omission_credential_redaction_and_no_access_log(monkeypatch,tmp_path,caplog):
    caplog.set_level(logging.INFO)
    async with service(monkeypatch,tmp_path) as (server,client,native):
        db,root=native[2],native[4].session_id
        db.set_session_title(root,TOKEN)
        db.append_message(root,"system","hidden system "+TOKEN)
        db.append_message(root,"user","secret "+TOKEN)
        db.append_message(root,"assistant","x"*65537)
        result=await (await client.get(f"/v1/pwa/conversations/{root}/history")).json()
        assert result["history_state"] == "partial"
        assert result["messages"][0]["content"] is None
        assert result["messages"][0]["omission_reason"] == "system_content"
        assert TOKEN not in json.dumps(result)
        assert result["messages"][2]["omission_reason"] == "oversized"
        conversation=await (await client.get(f"/v1/pwa/conversations/{root}")).json()
        assert TOKEN not in json.dumps(conversation)
        await client.get("/v1/pwa/health?code=synthetic-callback-secret&state=synthetic-state")
        assert TOKEN not in caplog.text and "synthetic-callback-secret" not in caplog.text
        caps=await (await client.get("/v1/pwa/capabilities")).json()
        for entry in caps["capabilities"]:
            if entry["capability_id"] == "remote_cancellation":
                assert entry["availability"] == "available"


@pytest.mark.asyncio
async def test_query_bounds_and_storage_failure_are_explicit(monkeypatch,tmp_path):
    async with service(monkeypatch,tmp_path) as (server,client,native):
        for suffix in ("?limit=0","?limit=101","?offset=1","?limit=1&limit=2"):
            assert (await client.get("/v1/pwa/conversations"+suffix)).status == 400
        assert (await client.get("/v1/pwa/health",data="body")).status == 400
        response=await client.post("/v1/pwa/conversations",data="x"*524289,headers={"Content-Type":"application/json"})
        assert response.status == 413
        def broken(*args):
            raise OSError("private failure "+TOKEN)
        monkeypatch.setattr(native[2],"native_pwa_history_signature",broken)
        response=await client.get(f"/v1/pwa/conversations/{native[4].session_id}/history")
        assert response.status == 503 and TOKEN not in await response.text()


@pytest.mark.parametrize("change", [{"enabled":"true"},{"private_network":"yes"},{"host":"0.0.0.0"},
    {"host":"0.0.0.0","private_network":True},{"allowed_hosts":["*"]},{"port":True},
    {"max_requests":0},{"cursor_ttl":float("nan")},{"token_env":"PATH$"},{"unknown":True}])
def test_invalid_config_is_closed(change):
    from tests.gateway.test_42039_duplicate_user_message import _source
    raw=config_for(_source()); raw.update(change)
    with pytest.raises(ValueError):
        PwaHttpConfig.from_dict(raw)


def test_shared_source_has_distinct_principals_and_unknown_fields_rejected():
    from tests.gateway.test_42039_duplicate_user_message import _source
    raw=config_for(_source())
    duplicate=copy.deepcopy(raw["bindings"][0]); duplicate["subject"]="foreign"
    raw["bindings"].append(duplicate)
    assert len(PwaHttpConfig.from_dict(raw).bindings) == 2
    raw=config_for(_source()); raw["bindings"][0]["sources"][0]["role_authorized"]=True
    with pytest.raises(ValueError): PwaHttpConfig.from_dict(raw)


def native_row(db, source, root, **kwargs):
    return db.create_session(root, source.platform.value, user_id=source.user_id,
        chat_id=source.chat_id, chat_type=source.chat_type, thread_id=source.thread_id,
        origin_json=json.dumps(source.to_dict()), **kwargs)


@pytest.mark.asyncio
async def test_discovery_includes_new_native_and_user_branches_excludes_delegation(monkeypatch,tmp_path):
    async with service(monkeypatch,tmp_path) as (server,client,native):
        db, source, root = native[2],native[5],native[4].session_id
        native_row(db,source,"new-native")
        native_row(db,source,"user-branch",parent_session_id=root,model_config={"_branched_from":root})
        native_row(db,source,"delegate",parent_session_id=root,model_config={"_delegate_from":root})
        db.create_session("ambiguous",source.platform.value,user_id=source.user_id)
        page=await (await client.get("/v1/pwa/conversations")).json()
        ids={c["conversation_id"] for c in page["conversations"]}
        assert ids == {root,"new-native","user-branch"}
        assert page["coverage"]["state"] == "partial"
        assert "legacy_ownership_unproven" in page["coverage"]["reasons"]
        assert server.ownership.event_authorized(source,"new-native")
        for denied in ("ambiguous","delegate"):
            assert (await client.get("/v1/pwa/conversations/"+denied)).status == 404
        raw=copy.deepcopy(server.runner.config.pwa_http)
        raw["bindings"][0]["sources"][0]["session_ids"]=["ambiguous"]
        server.ownership.config=PwaHttpConfig.from_dict(raw)
        assert (await client.get("/v1/pwa/conversations/ambiguous")).status == 200
        server.ownership.config=server.config
        assert (await client.get("/v1/pwa/conversations/ambiguous")).status == 404


@pytest.mark.asyncio
async def test_compression_order_and_foreign_lineage_contradictions(monkeypatch,tmp_path):
    async with service(monkeypatch,tmp_path) as (server,client,native):
        db, source, root = native[2],native[5],native[4].session_id
        db.append_message(root,"user","first segment")
        db.end_session(root,"compression")
        native_row(db,source,"tip",parent_session_id=root)
        db.append_message("tip","assistant","second segment")
        page=await (await client.get(f"/v1/pwa/conversations/{root}/history")).json()
        assert page["native_session_ids"] == [root,"tip"]
        assert [m["native_session_id"] for m in page["messages"]] == [root,"tip"]
        raw=copy.deepcopy(server.runner.config.pwa_http)
        raw["bindings"][0]["sources"][0]["session_ids"]=[root]
        server.ownership.config=PwaHttpConfig.from_dict(raw)
        db._execute_write(lambda conn:conn.execute("UPDATE sessions SET user_id='foreign' WHERE id='tip'"))
        for alias in (root,"tip"):
            assert (await client.get(f"/v1/pwa/conversations/{alias}/history")).status == 404


@pytest.mark.asyncio
async def test_compacted_display_copies_deduplicate_and_rewind_rows_are_excluded(monkeypatch,tmp_path):
    async with service(monkeypatch,tmp_path) as (_,client,native):
        db,root=native[2],native[4].session_id
        db.append_message(root,"user","retained",timestamp=1.0)
        db._execute_write(lambda conn:conn.execute("UPDATE messages SET active=0,compacted=1 WHERE session_id=?",(root,)))
        db.append_message(root,"user","retained",timestamp=1.0)
        db.append_message(root,"assistant","rewound",timestamp=2.0)
        db._execute_write(lambda conn:conn.execute("UPDATE messages SET active=0,compacted=0 WHERE content='rewound'"))
        page=await (await client.get(f"/v1/pwa/conversations/{root}/history")).json()
        assert [m["content"] for m in page["messages"]] == ["retained"]


@pytest.mark.asyncio
async def test_lineage_expansion_during_page_construction_conflicts(monkeypatch,tmp_path):
    async with service(monkeypatch,tmp_path) as (server,client,native):
        db,source,root=native[2],native[5],native[4].session_id
        db.append_message(root,"user","before compression")
        original=db.native_pwa_history_rows
        def compress(*args):
            rows=original(*args)
            db.end_session(root,"compression")
            native_row(db,source,"new-tip",parent_session_id=root)
            return rows
        monkeypatch.setattr(db,"native_pwa_history_rows",compress)
        response=await client.get(f"/v1/pwa/conversations/{root}/history")
        assert response.status == 409
        assert "before compression" not in await response.text()


@pytest.mark.asyncio
async def test_authorization_revocation_during_conversation_projection_is_rechecked(monkeypatch,tmp_path):
    async with service(monkeypatch,tmp_path) as (server,client,native):
        root=native[4].session_id
        original=server.ownership.safe_text
        def revoke(*args):
            server.runner._is_user_authorized_for_source=lambda _:False
            return original(*args)
        monkeypatch.setattr(server.ownership,"safe_text",revoke)
        assert (await client.get(f"/v1/pwa/conversations/{root}")).status == 404


@pytest.mark.asyncio
async def test_authentication_duplicate_headers_and_forged_principal_fields(monkeypatch,tmp_path):
    async with service(monkeypatch,tmp_path) as (_,client,_):
        duplicate=list(headers().items())+[("Authorization","Bearer "+TOKEN)]
        assert (await client.get("/v1/pwa/health",headers=duplicate)).status == 401
        raw=json.dumps({"issuer":OWNER.issuer,"subject":OWNER.subject,"concierge_id":"synthetic","user_id":"forged"}).encode()
        forged=headers(); forged["X-Hermes-PWA-Principal"]=base64.urlsafe_b64encode(raw).decode().rstrip("=")
        assert (await client.get("/v1/pwa/health",headers=forged)).status == 400


@pytest.mark.asyncio
async def test_real_gateway_config_loader_start_stop_and_fail_closed(monkeypatch,tmp_path):
    import yaml
    from gateway.config import load_gateway_config
    from gateway.run import GatewayRunner
    from tests.gateway.test_42039_duplicate_user_message import _source
    monkeypatch.setenv("HERMES_HOME",str(tmp_path))
    raw=config_for(_source())
    (tmp_path/"config.yaml").write_text(yaml.safe_dump({"gateway":{"pwa_http":raw},"security":{"tirith_enabled":False}}))
    config=load_gateway_config()
    assert config.pwa_http == raw
    monkeypatch.setenv("HERMES_PWA_FACADE_TOKEN",TOKEN)
    runner=GatewayRunner(config)
    monkeypatch.setattr(runner,"_spawn_supervised",lambda *args,**kwargs:None)
    monkeypatch.setattr(runner,"_spawn_reconnect_watcher",lambda:None)
    try:
        assert await runner.start()
        server=runner._pwa_http
        assert server.db is runner.session_store._db
        assert server.ingress.runner is runner
        async with aiohttp.ClientSession(headers=headers()) as client:
            response=await client.get(f"http://127.0.0.1:{server.port}/v1/pwa/health")
            assert response.status == 200
        await runner.stop()
        assert server._http_runner is None
        with pytest.raises(aiohttp.ClientConnectorError):
            async with aiohttp.ClientSession(headers=headers()) as client:
                await client.get(f"http://127.0.0.1:{server.port}/v1/pwa/health")
    finally:
        await runner.stop()
    monkeypatch.delenv("HERMES_PWA_FACADE_TOKEN")
    rejected=GatewayRunner(config)
    with pytest.raises(RuntimeError,match="Configured native PWA listener failed"):
        await rejected.start()
    await rejected.stop()


@pytest.mark.asyncio
async def test_model_capabilities_are_unavailable_when_catalog_is_omitted(
    monkeypatch, tmp_path
):
    async with service(monkeypatch, tmp_path) as (_, client, _):
        capabilities = (await (await client.get(
            "/v1/pwa/capabilities"
        )).json())["capabilities"]
        by_id = {row["capability_id"]: row for row in capabilities}
        assert by_id["model_catalog"]["availability"] == "unavailable"
        assert by_id["conversation_model_selection"]["availability"] == "unavailable"
        response = await client.get("/v1/pwa/models")
        assert response.status == 404
        assert (await response.json())["error"]["code"] == "capability_unavailable"


@pytest.mark.asyncio
async def test_managed_http_title_uses_explicit_route_without_model_catalog(
    monkeypatch, tmp_path
):
    import threading
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import yaml

    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "auxiliary": {
                    "title_generation": {
                        "enabled": True,
                        "provider": "custom",
                        "model": "titles/v1",
                        "base_url": "https://titles.invalid/v1",
                        "api_key": "title-only-key",
                        "timeout": 7,
                        "extra_body": {"seed": 3},
                    }
                }
            }
        )
    )
    outbound = threading.Event()
    captured = {}
    title_client = MagicMock()
    title_client.base_url = "https://titles.invalid/v1"

    def complete(**kwargs):
        captured["request"] = kwargs
        outbound.set()
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"title":"Bounded route"}')
                )
            ]
        )

    title_client.chat.completions.create.side_effect = complete

    def cached(provider, model, **kwargs):
        captured["destination"] = (provider, model, kwargs)
        return title_client, model

    monkeypatch.setattr("agent.auxiliary_client._get_cached_client", cached)
    async with service(monkeypatch, tmp_path) as (_, client, native):
        from agent import conversation_loop, turn_context

        provider_loop = conversation_loop.run_conversation

        def titled_loop(agent, text, system, history, *args, **kwargs):
            turn_context._maybe_title_session_at_turn_start(
                agent,
                [*list(history or []), {"role": "user", "content": text}],
            )
            return provider_loop(agent, text, system, history, *args, **kwargs)

        monkeypatch.setattr(conversation_loop, "run_conversation", titled_loop)
        root = native[4].session_id
        prompt = "z" * 1500
        response = await client.post(
            "/v1/pwa/commands", json=command(root, prompt, id="title-route")
        )
        assert response.status == 200, await response.text()
        agent = await started()
        assert agent._managed_pwa_title_destination["model"] == "titles/v1"
        assert await asyncio.to_thread(outbound.wait, 10)
        JournalAgent.gates[0].set()

    provider, model, route = captured["destination"]
    assert (provider, model) == ("custom", "titles/v1")
    assert route["base_url"] == "https://titles.invalid/v1"
    assert route["api_key"] == "title-only-key"
    assert route["main_runtime"] == {}
    request = captured["request"]
    assert request["model"] == "titles/v1"
    assert request["timeout"] == 7
    assert request["extra_body"]["seed"] == 3
    assert len(request["messages"][1]["content"].encode()) == 1000
    assert set(request["messages"][1]) == {"role", "content"}
    assert "title-only-key" not in json.dumps(request["messages"])


@pytest.mark.asyncio
async def test_model_catalog_selection_cas_and_historical_replay(
    monkeypatch, tmp_path
):
    unavailable = set()

    def resolve(provider):
        if provider in unavailable:
            raise RuntimeError("synthetic outage")
        return {"provider": provider, "api_key": "secret"}

    async with service(
        monkeypatch, tmp_path, models=model_config(), resolver=resolve
    ) as (_, client, native):
        root = native[4].session_id
        catalog = await (await client.get("/v1/pwa/models")).json()
        assert catalog["default_model_id"] == "daily"
        assert [row["availability"] for row in catalog["models"]] == [
            "available",
            "available",
        ]
        selected = await (await client.get(
            f"/v1/pwa/conversations/{root}/model"
        )).json()
        assert selected == {
            "schema_version": "1.0",
            "model_id": "daily",
            "model_version": 1,
        }

        request = {
            "schema_version": "1.0",
            "mutation_id": "mutation-1",
            "model_id": "deep",
            "expected_model_version": 1,
        }
        applied = await (await client.put(
            f"/v1/pwa/conversations/{root}/model", json=request
        )).json()
        assert applied["outcome"] == "applied"
        assert applied["model"] == {"model_id": "deep", "model_version": 2}

        unavailable.add("provider-b")
        replay = await (await client.put(
            f"/v1/pwa/conversations/{root}/model", json=request
        )).json()
        assert replay == {**applied, "outcome": "replayed"}

        changed = {**request, "model_id": "daily"}
        assert (await client.put(
            f"/v1/pwa/conversations/{root}/model", json=changed
        )).status == 409
        stale = {**request, "mutation_id": "mutation-2", "model_id": "daily"}
        assert (await client.put(
            f"/v1/pwa/conversations/{root}/model", json=stale
        )).status == 409


@pytest.mark.asyncio
async def test_model_route_outage_does_not_block_other_selection(monkeypatch, tmp_path):
    def resolve(provider):
        if provider == "provider-a":
            raise RuntimeError("default down")
        return {"provider": provider, "api_key": "secret"}

    async with service(
        monkeypatch, tmp_path, models=model_config(), resolver=resolve
    ) as (_, client, native):
        root = native[4].session_id
        catalog = await (await client.get("/v1/pwa/models")).json()
        assert [row["availability"] for row in catalog["models"]] == [
            "unavailable",
            "available",
        ]
        assert (await client.get(
            f"/v1/pwa/conversations/{root}/model"
        )).status == 503
        response = await client.put(
            f"/v1/pwa/conversations/{root}/model",
            json={
                "schema_version": "1.0",
                "mutation_id": "choose-deep",
                "model_id": "deep",
                "expected_model_version": 0,
            },
        )
        assert response.status == 200
        assert (await response.json())["model"] == {
            "model_id": "deep",
            "model_version": 1,
        }


@pytest.mark.asyncio
async def test_command_model_precondition_uses_current_selection(monkeypatch, tmp_path):
    async with service(
        monkeypatch,
        tmp_path,
        models=model_config(),
        resolver=lambda provider: {"provider": provider, "api_key": "secret"},
    ) as (_, client, native):
        root = native[4].session_id
        await client.put(
            f"/v1/pwa/conversations/{root}/model",
            json={
                "schema_version": "1.0",
                "mutation_id": "choose",
                "model_id": "daily",
                "expected_model_version": 0,
            },
        )
        body = command(root, "hello", id="versioned", kind="send")
        body["expected_model_version"] = 1
        assert (await client.post("/v1/pwa/commands", json=body)).status == 200
        body = command(root, "later", id="stale-version", kind="queue")
        body["expected_model_version"] = 2
        assert (await client.post("/v1/pwa/commands", json=body)).status == 409


@pytest.mark.asyncio
async def test_queued_execution_captures_latest_model_at_actual_start(
    monkeypatch, tmp_path
):
    async with service(
        monkeypatch,
        tmp_path,
        models=model_config(),
        resolver=lambda provider: {"provider": provider, "api_key": "secret"},
    ) as (server, client, native):
        db, root = native[2], native[4].session_id
        await client.put(
            f"/v1/pwa/conversations/{root}/model",
            json={
                "schema_version": "1.0",
                "mutation_id": "initial",
                "model_id": "daily",
                "expected_model_version": 0,
            },
        )
        first = command(root, "first", id="first", expected_model_version=1)
        assert (await client.post("/v1/pwa/commands", json=first)).status == 200
        await started()
        first_row = db.native_command_lookup(server.ingress._command_scope(OWNER), "first")
        assert db.native_execution_model(first_row["resulting_execution_id"]) == {
            "model_id": "daily",
            "model_version": 1,
        }

        await client.put(
            f"/v1/pwa/conversations/{root}/model",
            json={
                "schema_version": "1.0",
                "mutation_id": "switch",
                "model_id": "deep",
                "expected_model_version": 1,
            },
        )
        second = command(
            root,
            "second",
            id="second",
            kind="queue",
            expected_model_version=2,
        )
        assert (await client.post("/v1/pwa/commands", json=second)).status == 200
        assert db.native_execution_model(first_row["resulting_execution_id"]) == {
            "model_id": "daily",
            "model_version": 1,
        }

        JournalAgent.gates[0].set()
        await started()
        second_row = db.native_command_lookup(
            server.ingress._command_scope(OWNER), "second"
        )
        assert db.native_execution_model(second_row["resulting_execution_id"]) == {
            "model_id": "deep",
            "model_version": 2,
        }
