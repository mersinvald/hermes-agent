"""Real native HTTP wrapper with bounded projected files and authorization changes."""

from contextlib import asynccontextmanager
import copy
from dataclasses import replace
from datetime import datetime
import os

import aiohttp
import pytest

from gateway.conversation_control import Principal
from gateway.pwa_config import PwaHttpConfig
from gateway.pwa_http import NativePwaHttp
from gateway import pwa_workload
from tests.gateway.test_native_commands import finish, setup
from tests.gateway.test_pwa_http import OWNER, TOKEN, config_for, headers


IDENTITY = {"namespace": "ai", "pod-name": "hermes-108157884",
            "pod-uid": "6ef273cf-f186-4aea-a237-39b717f3adf8"}
PATH = "/v1/pwa/workload-context"
FOREIGN = Principal("https://issuer.invalid", "foreign")


@asynccontextmanager
async def service(monkeypatch, tmp_path, *, enabled=True, change_files=None,
                  second_owner=False, second_authorized=False, actual_auth=False):
    directory = tmp_path / "downward"
    directory.mkdir()
    # Exercise the real Kubernetes projection layout, including its symlinks.
    projection = directory / "..version"
    projection.mkdir()
    (directory / "..data").symlink_to(projection.name)
    for name, value in IDENTITY.items():
        (projection / name).write_text(value + "\n", encoding="ascii")
        (directory / name).symlink_to("..data/" + name)
    if change_files:
        change_files(directory)
    monkeypatch.setattr(pwa_workload, "DIRECTORY", directory)
    native = setup(monkeypatch, tmp_path)
    runner, old, db, _, _, source, _ = native
    old.stop_command_recovery()
    del runner.conversation_ingress
    config = config_for(source)
    if enabled:
        config["workload_context"] = {"source": "kubernetes_downward_api_v1"}
    if second_owner:
        other = copy.deepcopy(config["bindings"][0])
        other["subject"] = FOREIGN.subject
        other["sources"][0]["user_id"] = "foreign"
        other["sources"][0]["chat_id"] = "foreign-chat"
        config["bindings"].append(other)
    if second_authorized:
        runner._is_user_authorized_for_source = lambda _: True
    if actual_auth:
        del runner._is_user_authorized_for_source
        del runner._is_user_authorized
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", source.user_id)
    runner.config.pwa_http = config
    runner._running = True
    monkeypatch.setenv("HERMES_PWA_FACADE_TOKEN", TOKEN)
    server = NativePwaHttp(runner)
    await server.start()
    async with aiohttp.ClientSession(base_url=f"http://127.0.0.1:{server.port}",
                                     headers=headers(), timeout=aiohttp.ClientTimeout(total=10)) as client:
        try:
            yield server, client, native, directory
        finally:
            await server.close()
            await finish(server.ingress)
            db.close()


@pytest.mark.asyncio
async def test_owner_http_only_projection_and_stable_epoch(monkeypatch, tmp_path):
    async with service(monkeypatch, tmp_path) as (server, client, native, _):
        native[2].append_message(native[4].session_id, "user", "PRIVATE HISTORY")
        def no_history(*_):
            raise AssertionError("workload projection must not traverse native history")
        monkeypatch.setattr(native[2], "get_session", no_history)
        first = await client.get(PATH)
        body = await first.json()
        assert first.status == 200
        assert first.headers["Cache-Control"] == "no-store"
        assert set(body) == {"schema_version", "namespace", "pod_name", "pod_uid",
                             "owner_scope_hash", "context_revision", "epoch_started_at", "observed_at"}
        assert body["namespace"] == "ai" and body["pod_uid"] == IDENTITY["pod-uid"]
        second = await (await client.get(PATH)).json()
        assert first.status == 200
        assert body["context_revision"] == second["context_revision"]
        assert body["owner_scope_hash"] == second["owner_scope_hash"]
        assert datetime.fromisoformat(body["epoch_started_at"]) <= datetime.fromisoformat(body["observed_at"])
        assert datetime.fromisoformat(body["observed_at"]) <= datetime.fromisoformat(second["observed_at"])
        assert "PRIVATE" not in str(body) and TOKEN not in str(body)
        assert OWNER.issuer not in str(body) and str(tmp_path) not in str(body)


@pytest.mark.asyncio
async def test_foreign_principal_and_real_native_auth(monkeypatch, tmp_path):
    async with service(monkeypatch, tmp_path, second_owner=True) as (_, client, _, _):
        assert (await client.get(PATH, headers=headers(FOREIGN))).status == 404
        assert (await client.get(PATH, headers={"Authorization": "Bearer wrong"})).status == 401
        assert (await client.get(PATH, headers={"Origin": "https://browser.invalid"})).status == 400
        assert (await client.get(PATH, headers=headers(concierge="foreign"))).status == 403
        # Failed reads must not poison the legitimate owner's context.
        assert (await client.get(PATH)).status == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("method,suffix,kwargs", [
    ("GET", "?pod_uid=forged", {}), ("GET", "?owner=foreign", {}),
    ("GET", "?url=http://metadata.invalid", {}), ("GET", "", {"json": {"pod": "forged"}}),
    ("POST", "", {}), ("PUT", "", {}), ("DELETE", "", {}), ("HEAD", "", {}),
])
async def test_closed_get_only_wire(monkeypatch, tmp_path, method, suffix, kwargs):
    async with service(monkeypatch, tmp_path) as (_, client, _, _):
        response = await client.request(method, PATH + suffix, **kwargs)
        assert response.status in (400, 404)
        assert (await client.get(PATH)).status == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("name,value", [
    ("pod-uid", "not-a-uid"), ("pod-uid", IDENTITY["pod-uid"].upper()),
    ("namespace", "ai/other"), ("namespace", "a" * 64), ("namespace", "Ai"),
    ("pod-name", "../private"), ("pod-name", "x" * 254), ("pod-name", "bad..name"),
    ("pod-name", "bad\x00name"), ("pod-name", "pod\n\n"), ("pod-name", "é"),
])
async def test_malformed_projection_is_optional_and_latches(monkeypatch, tmp_path, name, value):
    def change(directory):
        (directory / name).write_text(value, encoding="utf-8")
    async with service(monkeypatch, tmp_path, change_files=change) as (_, client, _, directory):
        assert (await client.get(PATH)).status == 503
        (directory / name).write_text(IDENTITY[name], encoding="ascii")
        assert (await client.get(PATH)).status == 503
        assert (await client.get("/v1/pwa/health")).status == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["disabled", "missing", "fifo", "directory"])
async def test_no_identity_fallback_or_blocking_special_file(monkeypatch, tmp_path, kind):
    def change(directory):
        path = directory / "pod-uid"
        path.unlink()
        if kind == "fifo":
            os.mkfifo(path)
        elif kind == "directory":
            path.mkdir()
    async with service(monkeypatch, tmp_path, enabled=kind != "disabled",
                       change_files=None if kind == "disabled" else change) as (_, client, _, _):
        assert (await client.get(PATH)).status == 503
        assert (await client.get("/v1/pwa/health")).status == 200


@pytest.mark.asyncio
async def test_multiple_effective_owners_denied(monkeypatch, tmp_path):
    async with service(monkeypatch, tmp_path, second_owner=True, second_authorized=True) as (server, client, _, _):
        for principal in (OWNER, FOREIGN):
            assert (await client.get(PATH, headers=headers(principal))).status == 503
        server.runner._is_user_authorized_for_source = lambda s: s.user_id != "foreign"
        assert (await client.get(PATH)).status == 503


@pytest.mark.asyncio
async def test_actual_telegram_authorization_revocation_fences_boot(monkeypatch, tmp_path):
    async with service(monkeypatch, tmp_path, actual_auth=True) as (_, client, native, _):
        assert (await client.get(PATH)).status == 200
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "another-user")
        assert (await client.get(PATH)).status == 503
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", native[5].user_id)
        assert (await client.get(PATH)).status == 503


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["identity", "binding", "authorization"])
async def test_change_after_dispatch_cannot_return_stale_context(monkeypatch, tmp_path, change):
    async with service(monkeypatch, tmp_path) as (server, client, _, directory):
        initial = await (await client.get(PATH)).json()
        original = server._dispatch
        original_config = server.ownership.config
        original_auth = server.runner._is_user_authorized_for_source
        async def change_after_read(request, principal):
            result = await original(request, principal)
            if change == "identity":
                (directory / "pod-uid").write_text("7ef273cf-f186-4aea-a237-39b717f3adf8", encoding="ascii")
            elif change == "binding":
                binding = original_config.bindings[0]
                source = replace(binding.sources[0], session_ids=("new-authorized-root",))
                server.ownership.config = replace(original_config, bindings=(replace(binding, sources=(source,)),))
            else:
                server.runner._is_user_authorized_for_source = lambda _: False
            return result
        monkeypatch.setattr(server, "_dispatch", change_after_read)
        response = await client.get(PATH)
        assert response.status == 503
        assert initial["context_revision"] not in await response.text()
        monkeypatch.setattr(server, "_dispatch", original)
        (directory / "pod-uid").write_text(IDENTITY["pod-uid"], encoding="ascii")
        server.ownership.config = original_config
        server.runner._is_user_authorized_for_source = original_auth
        assert (await client.get(PATH)).status == 503


@pytest.mark.asyncio
async def test_restart_nonce_even_with_same_clock_and_config(monkeypatch, tmp_path):
    monkeypatch.setattr(pwa_workload, "utc_now", lambda: "2026-09-10T00:00:00Z")
    async with service(monkeypatch, tmp_path) as (server, client, _, _):
        first = await (await client.get(PATH)).json()
        await server.close()
        del server.runner.conversation_ingress
        replacement = NativePwaHttp(server.runner)
        try:
            await replacement.start()
            async with aiohttp.ClientSession(headers=headers()) as other:
                response = await other.get(f"http://127.0.0.1:{replacement.port}" + PATH)
                second = await response.json()
            assert response.status == 200
            assert first["epoch_started_at"] == second["epoch_started_at"]
            assert first["owner_scope_hash"] == second["owner_scope_hash"]
            assert first["context_revision"] != second["context_revision"]
        finally:
            await replacement.close()
            replacement.ingress.stop_command_recovery()


@pytest.mark.asyncio
async def test_clock_regression_latches_unavailable(monkeypatch, tmp_path):
    async with service(monkeypatch, tmp_path) as (_, client, _, _):
        assert (await client.get(PATH)).status == 200
        monkeypatch.setattr(pwa_workload, "utc_now", lambda: "2000-01-01T00:00:00Z")
        assert (await client.get(PATH)).status == 503
        monkeypatch.setattr(pwa_workload, "utc_now", lambda: "2099-01-01T00:00:00Z")
        assert (await client.get(PATH)).status == 503


@pytest.mark.parametrize("value", [{}, {"source": "auto"}, {"source": "kubernetes_downward_api_v1", "path": "/private"},
                                  True, [], "kubernetes_downward_api_v1"])
def test_config_does_not_allow_discovery_paths_or_untyped_enable(value):
    from tests.gateway.test_conversation_control import _source
    raw = config_for(_source())
    raw["workload_context"] = value
    with pytest.raises(ValueError):
        PwaHttpConfig.from_dict(raw)
