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
from tests.gateway.test_native_commands import JournalAgent, command, finish, setup, started

TOKEN = "synthetic-facade-secret-" + "s" * 40
OWNER = Principal("https://issuer.invalid", "owner")


def config_for(source):
    return {"enabled": True, "concierge_id": "synthetic", "host": "127.0.0.1", "port": 0,
            "allowed_hosts": ["native.test"], "bindings": [{"issuer": OWNER.issuer,
            "subject": OWNER.subject, "default_source_id": "telegram-owner", "sources": [{
                "source_id": "telegram-owner", "platform": source.platform.value,
                "chat_id": source.chat_id, "chat_type": source.chat_type,
                "user_id": source.user_id, "thread_id": source.thread_id}]}]}


def headers(principal=OWNER, concierge="synthetic"):
    raw = json.dumps({"issuer": principal.issuer, "subject": principal.subject,
                      "concierge_id": concierge}).encode()
    return {"Host": "native.test", "Authorization": "Bearer " + TOKEN,
            "X-Hermes-PWA-Principal": base64.urlsafe_b64encode(raw).decode().rstrip("=")}


@asynccontextmanager
async def service(monkeypatch, tmp_path):
    native = setup(monkeypatch, tmp_path)
    runner, old, db, store, entry, source, adapter = native
    old.stop_command_recovery()
    del runner.conversation_ingress
    runner.config.pwa_http = config_for(source)
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
                assert entry["availability"] == "unavailable"


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
