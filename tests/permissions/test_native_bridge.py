"""Native queue/Telegram callbacks against the current, independent service API.

Only the Telegram SDK send method is mocked. No bot starts or polls Telegram.
The optional contract fixture imports the explicitly selected current infra
service, rather than a stale copy of its ledger/API inside the Hermes fork.
"""

import asyncio
import contextvars
import datetime
import ipaddress
import json
import ssl
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import permission_bridge as bridge
from gateway.config import PlatformConfig
from gateway.session_context import set_session_vars, clear_session_vars
from plugins.platforms.telegram.adapter import TelegramAdapter
from tools import approval


REF = "a" * 32
TOKEN = "synthetic-human-credential-not-for-model-0000"


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr(approval, "_gateway_queues", {})
    monkeypatch.setattr(approval, "_gateway_notify_cbs", {})
    monkeypatch.setattr(approval, "_fire_approval_hook", lambda *a, **kw: None)
    monkeypatch.setattr(approval, "_get_approval_timeout", lambda: 5)
    tokens = set_session_vars(platform="telegram", user_id="101", chat_id="101", session_key="s", cron_session="")
    yield
    approval.unregister_gateway_notify("s")
    clear_session_vars(tokens)


def remote(ref=REF, actor="groundskeeper108"):
    return bridge.RemotePermission(ref, actor, "serviceUser", "101", "101", "https://permissions.test")


def details(ref=REF):
    return {"request_id": ref, "digest": "d" * 64, "expires": time.time() + 60,
            "state": "pending", "action": {"actor": "groundskeeper108", "owner": "serviceUser",
                "backend": "fixture", "tool": "fixture_set_value", "arguments": {"resource_id": "fixture-a", "value": 3},
                "resources": {"/resource_id": "fixture-a"}, "schema_version": "1", "contract_digest": "e" * 64},
            "choices": [{"id": "once", "choice_scope_label": "Once exact action"},
                        {"id": "deny", "choice_scope_label": "Deny exact action"}]}


def adapter():
    result = TelegramAdapter(PlatformConfig(enabled=True, token="synthetic-telegram-token"))
    result._bot = SimpleNamespace(send_message=AsyncMock(return_value=SimpleNamespace(message_id=42)))
    result._is_callback_user_authorized = lambda *a, **kw: True
    return result


def query_for(r, index=0, **changes):
    values = {"data": f"hp:{r.nonce}:{index}", "from_user": SimpleNamespace(id=101, first_name="Human"),
              "message": SimpleNamespace(chat_id=101, message_id=42, chat=SimpleNamespace(type="private"), message_thread_id=None),
              "answer": AsyncMock(), "edit_message_text": AsyncMock()}
    values.update(changes)
    return SimpleNamespace(**values)


def enqueue(r, request_id="native-entry"):
    entry = approval._ApprovalEntry({"request_id": request_id, "remote_permission": r})
    approval._gateway_queues.setdefault("s", []).append(entry)
    return entry


@pytest.mark.parametrize("text,expected", [(f"error [mkl.hitl.request:{REF}]", REF),
    (f"[mkl.hitl.request:{REF}][mkl.hitl.request:{REF}]", None),
    ("[mkl.hitl.request:" + "a" * 31 + "]", None),
    ("[mkl.hitl.request:https://attacker.test]", None),
    ("x" * 262145 + f"[mkl.hitl.request:{REF}]", None)])
def test_bounded_marker_is_only_lookup_hint(text, expected):
    assert bridge.request_reference(text) == expected


@pytest.mark.parametrize("mutation", [
    lambda d: d["action"].update(actor="finance108"),
    lambda d: d["action"].update(owner="anotherUser"),
    lambda d: d.update(request_id="b" * 32),
    lambda d: d.update(expires=time.time() - 1),
    lambda d: d.update(expires=float("nan")),
    lambda d: d.update(state="consumed"),
    lambda d: d.update(state="approved"),
    lambda d: d.update(digest="changed"),
    lambda d: d["choices"].append(d["choices"][0]),
    lambda d: d["choices"].append({"id": "always", "choice_scope_label": "Trust me"}),
    lambda d: d["action"]["arguments"].update(huge="x" * 4000),
])
def test_inspection_rejects_wrong_scope_or_unseen_action(monkeypatch, mutation):
    value = details()
    mutation(value)
    monkeypatch.setattr(bridge, "_http", lambda *a: value)
    with pytest.raises(bridge.BridgeError):
        bridge.inspect_prompt(remote())


def test_render_full_action_scopes_and_escape(monkeypatch):
    value = details()
    value["action"]["arguments"]["resource_id"] = "<b>\u202eevil</b>"
    value["choices"].append({"id": "bounded", "choice_scope_label": "Standing 0 to 10",
                             "scope": {**{key: value["action"][key] for key in
                                 ("actor", "owner", "backend", "tool", "contract_digest", "resources")},
                                 "template": {"id": "bounded", "constraints": {"maximum": 10}, "ttl_seconds": 60}}})
    monkeypatch.setattr(bridge, "_http", lambda *a: value)
    r = remote()
    text = bridge.inspect_prompt(r)
    assert "&lt;b&gt;" in text and "\\u202e" in text and "maximum" in text
    assert "d" * 64 in text and r.choices == ("once", "deny", "bounded")


@pytest.mark.parametrize("configuration", [{}, {"mcp_permissions": {"enabled": False}},
    {"mcp_permissions": {"enabled": True, "url": "https://permissions.test", "owner": "serviceUser"}}])
def test_disabled_or_unknown_peer_never_fetches_or_approves(monkeypatch, configuration):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: configuration)
    monkeypatch.setattr(bridge, "_http", lambda *a: pytest.fail("no human HTTP expected"))
    assert bridge.REQUIRED in bridge.request_permission(f"[mkl.hitl.request:{REF}]", "a2a_agents", "unknown")


def test_environment_identity_is_not_current_human(monkeypatch):
    cfg = {"mcp_permissions": {"enabled": True, "url": "https://permissions.test", "owner": "serviceUser"},
           "a2a_agents": {"groundskeeper": {"expected_permission_actor": "groundskeeper108"}}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setenv("HERMES_SESSION_USER_ID", "101")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "101")
    monkeypatch.setenv("HERMES_SESSION_KEY", "s")
    approval.register_gateway_notify("s", lambda _: pytest.fail("no bound context"))
    text = contextvars.Context().run(bridge.request_permission, f"[mkl.hitl.request:{REF}]", "a2a_agents", "groundskeeper")
    assert bridge.REQUIRED in text


def test_legacy_resolvers_cannot_resolve_remote_even_by_id():
    r = remote()
    entry = enqueue(r)
    for choice in ("once", "session", "always", "deny"):
        assert approval.resolve_gateway_approval("s", choice, resolve_all=True) == 0
        assert approval.resolve_gateway_approval("s", choice, request_id="native-entry") == 0
    assert entry.result is None and not entry.event.is_set()
    legacy = approval._ApprovalEntry({"command": "test"})
    approval._gateway_queues["s"].append(legacy)
    assert approval.resolve_gateway_approval("s", "once") == 1
    assert legacy.result == "once" and entry.result is None
    snapshot = approval.list_gateway_approvals("s")[0]
    assert json.dumps(snapshot) and r.nonce not in json.dumps(snapshot)
    assert approval.resolve_remote_gateway_approval("s", "native-entry", snapshot["remote_permission"], lambda: pytest.fail("snapshot is not callback authority")).decision is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", ["user", "chat", "message", "index", "nonce", "native_auth"])
async def test_callback_exact_binding_before_http(monkeypatch, mismatch):
    r = remote()
    monkeypatch.setattr(bridge, "_http", lambda *a: details())
    a = adapter()
    enqueue(r)
    assert (await a.send_exec_approval("101", "", "s", remote_permission=r, approval_request_id="native-entry")).success
    q = query_for(r)
    if mismatch == "user": q.from_user.id = 202
    if mismatch == "chat": q.message.chat_id = 202
    if mismatch == "message": q.message.message_id = 43
    if mismatch == "index": q.data = f"hp:{r.nonce}:8"
    if mismatch == "nonce": q.data = "hp:" + "z" * 32 + ":0"
    if mismatch == "native_auth": a._is_callback_user_authorized = lambda *a, **kw: False
    monkeypatch.setattr(bridge, "_http", lambda *a: pytest.fail("unauthorized callback reached HTTP"))
    await a._handle_callback_query(SimpleNamespace(callback_query=q), None)
    assert approval._gateway_queues["s"][0].result is None


@pytest.mark.asyncio
async def test_reverse_order_callbacks_duplicate_and_timeout(monkeypatch):
    monkeypatch.setattr(bridge, "_http", lambda r, *a: details(r.reference))
    a = adapter()
    first, second = remote(), remote("b" * 32)
    e1, e2 = enqueue(first, "one"), enqueue(second, "two")
    for r, ident in ((first, "one"), (second, "two")):
        assert (await a.send_exec_approval("101", "", "s", remote_permission=r, approval_request_id=ident)).success
    calls = []
    def decision(r, user, chat, body=None):
        calls.append((r.reference, user, chat, body))
        return {"state": "approved"}
    monkeypatch.setattr(bridge, "_http", decision)
    q = query_for(second)
    await asyncio.gather(*(a._handle_callback_query(SimpleNamespace(callback_query=q), None) for _ in range(3)))
    assert e2.result == "once" and e1.result is None and len(calls) == 1
    assert calls[0] == (second.reference, "101", "101", {"digest": "d" * 64, "choice": "once"})
    approval.unregister_gateway_notify("s")
    await a._handle_callback_query(SimpleNamespace(callback_query=query_for(first)), None)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_unknown_http_outcome_cannot_be_retried(monkeypatch):
    r = remote()
    e = enqueue(r)
    a = adapter()
    monkeypatch.setattr(bridge, "_http", lambda *a: details())
    await a.send_exec_approval("101", "", "s", remote_permission=r, approval_request_id="native-entry")
    calls = []
    def uncertain(*args):
        calls.append(args)
        raise TimeoutError(TOKEN)
    monkeypatch.setattr(bridge, "_http", uncertain)
    q = query_for(r)
    await a._handle_callback_query(SimpleNamespace(callback_query=q), None)
    await a._handle_callback_query(SimpleNamespace(callback_query=q), None)
    assert len(calls) == 1 and e.result == "deny"
    assert TOKEN not in repr(q.answer.call_args_list)


@pytest.fixture
def service(pytestconfig, tmp_path, monkeypatch):
    root = pytestconfig.getoption("--permissions-source")
    if not root:
        pytest.skip("Supply current infra permission-service source for HTTP contract acceptance")
    root = Path(root)
    monkeypatch.syspath_prepend(str(root / "src"))
    from mcp_permissions.ledger import Ledger
    from mcp_permissions.server import HumanServer, human_handler
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    cfg = json.loads((root / "fixture-config.json").read_text())
    cfg["owners"] = {"serviceUser": {"telegram_user_id": "101", "telegram_chat_id": "101"}}
    cfg["actors"] = {"groundskeeper108": "serviceUser", "finance108": "serviceUser"}
    ledger = Ledger(tmp_path / "ledger.sqlite", cfg)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(minutes=1))
            .not_valid_after(now + datetime.timedelta(hours=1))
            .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
            .sign(key, hashes.SHA256()))
    certfile, keyfile = tmp_path / "cert.pem", tmp_path / "key.pem"
    certfile.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    keyfile.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile, keyfile)
    http = HumanServer(("127.0.0.1", 0), human_handler(ledger, TOKEN))
    http.socket = ctx.wrap_socket(http.socket, server_side=True)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    original = urllib.request.build_opener
    trust = ssl.create_default_context(cafile=str(certfile))
    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: original(*a, urllib.request.HTTPSHandler(context=trust)))
    monkeypatch.setenv("MCP_PERMISSIONS_HUMAN_TOKEN", TOKEN)
    config = {"mcp_permissions": {"enabled": True, "url": f"https://127.0.0.1:{http.server_port}", "owner": "serviceUser"},
              "a2a_agents": {"groundskeeper": {"expected_permission_actor": "groundskeeper108"}}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    yield ledger, cfg, config
    http.shutdown()
    http.server_close()
    thread.join(timeout=3)


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", ["once", "deny", "standing"])
async def test_actual_service_native_queue_callback_contract(service, choice):
    ledger, cfg, config = service
    contract = cfg["contracts"][0]
    params = json.dumps({"name": contract["tool"], "arguments": {"resource_id": "fixture-a", "value": 3, "operation_id": "test-op"}}).encode()
    pending = ledger.check("groundskeeper108", [contract["backend"]], "tools/call", params)
    ref = pending["request_id"]
    a = adapter()
    loop = asyncio.get_running_loop()
    ready = asyncio.Event()
    captured = []
    async def send(data):
        captured.append(data)
        result = await a.send_exec_approval("101", "", "s", remote_permission=data["remote_permission"], approval_request_id=data["request_id"])
        assert result.success, result.error
        ready.set()
    def notify(data):
        asyncio.run_coroutine_threadsafe(send(data), loop).result(timeout=5)
    approval.register_gateway_notify("s", notify)
    waiter = asyncio.create_task(asyncio.to_thread(bridge.request_permission, f"[mkl.hitl.request:{ref}]", "a2a_agents", "groundskeeper"))
    await asyncio.wait_for(ready.wait(), 5)
    r = captured[0]["remote_permission"]
    selected = next(c for c in r.choices if c not in ("once", "deny")) if choice == "standing" else choice
    q = query_for(r, r.choices.index(selected))
    await a._handle_callback_query(SimpleNamespace(callback_query=q), None)
    result = await asyncio.wait_for(waiter, 5)
    assert TOKEN not in result and TOKEN not in repr(a._bot.send_message.call_args)
    state = ledger.inspect("serviceUser", ref)
    assert state["state"] == ("rejected" if choice == "deny" else "approved")
    assert not approval._gateway_queues
    if choice != "deny":
        assert "decision recorded" in result
        with ThreadPoolExecutor(max_workers=8) as pool:
            checks = list(pool.map(lambda _: ledger.check("groundskeeper108", [contract["backend"]], "tools/call", params), range(8)))
        assert sum("attempt_id" in c for c in checks) == 1
        assert ledger.inspect("serviceUser", ref)["outcome"] == "unknown"
        await a._handle_callback_query(SimpleNamespace(callback_query=q), None)
        assert ledger.check("groundskeeper108", [contract["backend"]], "tools/call", params)["type"] == "mkl.hitl.replay"


def test_actual_service_rejects_cross_agent_marker_before_display(service):
    ledger, cfg, config = service
    contract = cfg["contracts"][0]
    params = json.dumps({"name": contract["tool"], "arguments": {"resource_id": "fixture-a", "value": 3}}).encode()
    ref = ledger.check("finance108", [contract["backend"]], "tools/call", params)["request_id"]
    r = remote(ref)
    r.url = config["mcp_permissions"]["url"]
    with pytest.raises(bridge.BridgeError):
        bridge.inspect_prompt(r)
    assert not r.digest


@pytest.mark.parametrize("fault", ["digest", "sender", "chat", "credential", "idempotent"])
def test_actual_http_decision_rejects_changed_binding(service, monkeypatch, fault):
    ledger, cfg, config = service
    contract = cfg["contracts"][0]
    params = json.dumps({"name": contract["tool"], "arguments": {"resource_id": "fixture-a", "value": 3}}).encode()
    ref = ledger.check("groundskeeper108", [contract["backend"]], "tools/call", params)["request_id"]
    r = remote(ref)
    r.url = config["mcp_permissions"]["url"]
    bridge.inspect_prompt(r)
    user, chat = r.user, r.chat
    if fault == "digest": r.digest = "f" * 64
    if fault == "sender": user = "202"
    if fault == "chat": chat = "202"
    if fault == "credential": monkeypatch.setenv("MCP_PERMISSIONS_HUMAN_TOKEN", "wrong-credential-" * 3)
    if fault == "idempotent": bridge.decide(r, user, chat, 0)
    with pytest.raises(bridge.BridgeError) as error:
        bridge.decide(r, user, chat, 0)
    assert TOKEN not in str(error.value)
    assert ledger.inspect("serviceUser", ref)["state"] == ("approved" if fault == "idempotent" else "pending")


def test_actual_standing_choice_bounds_and_revocation(service):
    ledger, cfg, config = service
    contract = cfg["contracts"][0]
    def call(op, resource="fixture-a", value=3):
        params = json.dumps({"name": contract["tool"], "arguments": {"resource_id": resource, "value": value, "operation_id": op}}).encode()
        return ledger.check("groundskeeper108", [contract["backend"]], "tools/call", params)
    ref = call("op")['request_id']
    r = remote(ref)
    r.url = config["mcp_permissions"]["url"]
    bridge.inspect_prompt(r)
    assert bridge.decide(r, "101", "101", 2) == "once"
    grant = ledger.inspect("serviceUser", ref)["grant_id"]
    assert "attempt_id" in call("op")
    assert "attempt_id" in call("other-op")
    assert call("other-resource", resource="fixture-b")["type"] == "mkl.hitl.required"
    assert call("outside-bounds", value=11)["type"] == "mkl.hitl.required"
    ledger.revoke("serviceUser", "grants", grant)
    assert call("after-revoke")["type"] == "mkl.hitl.required"
    assert call("op")["type"] == "mkl.hitl.replay"


def test_http_transport_drops_exception_details_and_refuses_redirects(monkeypatch):
    monkeypatch.setenv("MCP_PERMISSIONS_HUMAN_TOKEN", TOKEN)
    captured = []
    def opener(*handlers):
        assert any(isinstance(h, urllib.request.ProxyHandler) and h.proxies == {} for h in handlers)
        captured.extend(handlers)
        class Failed:
            def open(self, request, **kwargs):
                assert request.get_header("Authorization") == "Bearer " + TOKEN
                raise OSError(TOKEN)
        return Failed()
    monkeypatch.setattr(urllib.request, "build_opener", opener)
    with pytest.raises(bridge.BridgeError) as error:
        bridge._http(remote(), "101", "101")
    assert TOKEN not in str(error.value)
    redirect = next(h for h in captured if isinstance(h, bridge._NoRedirect))
    with pytest.raises(bridge.BridgeError):
        redirect.redirect_request(None, None, 302, "", {}, "https://attacker.test")


@pytest.mark.asyncio
async def test_native_a2a_stream_to_current_service_callback_no_resend(service):
    from plugins.platforms.a2a.tools import a2a_call
    ledger, cfg, config = service
    contract = cfg["contracts"][0]
    params = json.dumps({"name": contract["tool"], "arguments": {"resource_id": "fixture-a", "value": 3}}).encode()
    ref = ledger.check("groundskeeper108", [contract["backend"]], "tools/call", params)["request_id"]
    calls = []
    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"protocolVersion": "0.3", "capabilities": {"streaming": True}}).encode())
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(body)
            ctx = body["params"]["message"]["contextId"]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for state, parts in (("working", [{"metadata": {"adk_type": "function_response"}, "data": {
                "name": "fixture_set_value", "id": "call", "response": {"error": f"Human approval required [mkl.hitl.request:{ref}]"}}}]),
                ("completed", [{"text": "No permission yet"}])):
                event = {"jsonrpc": "2.0", "id": body["id"], "result": {"kind": "status-update", "taskId": "task", "contextId": ctx,
                    "status": {"state": state, "message": {"parts": parts}}}}
                self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
    http = ThreadingHTTPServer(("127.0.0.1", 0), Peer)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    config["a2a_agents"]["groundskeeper"].update(url=f"http://127.0.0.1:{http.server_port}", streaming=True)
    a = adapter()
    loop, ready, captured = asyncio.get_running_loop(), asyncio.Event(), []
    async def send(data):
        captured.append(data["remote_permission"])
        result = await a.send_exec_approval("101", "", "s", remote_permission=captured[-1], approval_request_id=data["request_id"])
        assert result.success
        ready.set()
    approval.register_gateway_notify("s", lambda data: asyncio.run_coroutine_threadsafe(send(data), loop).result(timeout=5))
    try:
        task = asyncio.create_task(asyncio.to_thread(a2a_call, {"agent": "groundskeeper", "message": "inert request"}))
        await asyncio.wait_for(ready.wait(), 5)
        assert ledger.inspect("serviceUser", ref)["state"] == "pending"
        await a._handle_callback_query(SimpleNamespace(callback_query=query_for(captured[0])), None)
        result = await asyncio.wait_for(task, 5)
        assert "decision recorded" in result
        assert len(calls) == 1 and calls[0]["method"] == "message/stream"
        assert ledger.inspect("serviceUser", ref)["state"] == "approved"
        assert TOKEN not in result and TOKEN not in json.dumps(calls)
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=3)


@pytest.mark.parametrize("error", [True, False])
def test_native_mcp_marker_never_retries_or_autodecides(monkeypatch, error):
    from tools import mcp_tool
    from unittest.mock import MagicMock
    server = MagicMock()
    server.is_connected = True
    monkeypatch.setattr(mcp_tool, "_servers", {"fixture": server})
    calls = []
    def run(*args, **kw):
        calls.append(1)
        if error:
            raise RuntimeError(f"Human approval required [mkl.hitl.request:{REF}]")
        return json.dumps({"error": f"Human approval required [mkl.hitl.request:{REF}]"})
    monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", run)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setattr(bridge, "_http", lambda *args: pytest.fail("disabled bridge cannot decide"))
    result = mcp_tool._make_tool_handler("fixture", "setter", 10)({})
    assert bridge.REQUIRED in result and len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", ["once", "standing"])
@pytest.mark.parametrize("removal", ["timeout", "interrupt", "handover", "ack_failure", "client"])
async def test_confirmed_tls_decision_survives_waiter_removal(service, monkeypatch, choice, removal):
    ledger, cfg, config = service
    contract = cfg["contracts"][0]
    params = json.dumps({"name": contract["tool"], "arguments": {
        "resource_id": "fixture-a", "value": 3, "operation_id": "stable-race-op"}}).encode()
    ref = ledger.check("groundskeeper108", [contract["backend"]], "tools/call", params)["request_id"]
    r = remote(ref)
    r.url = config["mcp_permissions"]["url"]
    a = adapter()
    assert (await a.send_exec_approval("101", "", "s", remote_permission=r, approval_request_id="old")).success
    index = 0 if choice == "once" else 2
    committed, release, interrupted = threading.Event(), threading.Event(), threading.Event()
    ready = asyncio.Event()
    captured, posts = [], []
    execute_count = 0
    original = ledger.decide
    def delayed(owner, request, body):
        result = original(owner, request, body)
        posts.append(result)
        committed.set()
        assert release.wait(5)
        return result
    monkeypatch.setattr(ledger, "decide", delayed)
    monkeypatch.setattr(approval, "_get_approval_timeout", lambda: 0.5 if removal in ("timeout", "ack_failure", "client") else 5)
    monkeypatch.setattr(approval, "is_interrupted", interrupted.is_set)
    loop = asyncio.get_running_loop()
    def notify(data):
        captured.append(approval._gateway_queues["s"][0])
        loop.call_soon_threadsafe(ready.set)
    waiter = asyncio.create_task(asyncio.to_thread(approval._await_gateway_decision, "s", notify,
        {"request_id": "old", "remote_permission": r}))
    await asyncio.wait_for(ready.wait(), 3)
    q = query_for(r, index)
    if removal == "ack_failure":
        q.answer.side_effect = RuntimeError("synthetic Telegram acknowledgment failure")
    if removal == "client":
        callback = asyncio.create_task(asyncio.to_thread(approval.resolve_remote_gateway_approval,
            "s", "old", r, lambda: bridge.decide(r, "101", "101", index)))
    else:
        callback = asyncio.create_task(a._handle_callback_query(SimpleNamespace(callback_query=q), None))
    try:
        assert await asyncio.to_thread(committed.wait, 3)
        recorded = ledger.inspect("serviceUser", ref)
        assert recorded["state"] == "approved"
        assert execute_count == 0
        if removal == "interrupt": interrupted.set()
        if removal == "handover": approval.unregister_gateway_notify("s")
        await asyncio.wait_for(waiter, 3)
        old = captured[0]
        before = (old.event.is_set(), old.result)
        if removal in ("timeout", "ack_failure", "client"):
            assert before == (False, None)
        replacement = enqueue(remote("b" * 32), "new")
        snapshot = approval.list_gateway_approvals("s")
        release.set()
        resolution = await asyncio.wait_for(callback, 3)
        if removal == "client":
            assert resolution == approval.RemoteApprovalResolution("once", waiter_active=False)
        else:
            text = q.edit_message_text.call_args.kwargs["text"]
            assert text == "Permission recorded; request no longer waiting; no action automatically executed"
            assert TOKEN not in text and ref not in text and r.digest not in text
            q.answer.side_effect = None
            await a._handle_callback_query(SimpleNamespace(callback_query=query_for(r, index)), None)
        assert len(posts) == 1
        assert (old.event.is_set(), old.result) == before
        assert approval.list_gateway_approvals("s") == snapshot
        assert not replacement.event.is_set() and replacement.result is None
        inspected = bridge._http(r, "101", "101")
        assert inspected["request_id"] == ref and inspected["grant_id"] == recorded["grant_id"]
        assert inspected["state"] == "approved" and inspected["outcome"] == recorded["outcome"]
        if choice == "standing": assert inspected["grant_id"]
        # Inert execution is deliberately separate from the bridge. Only an
        # explicit, identical reattempt can consume authority and invoke it.
        assert execute_count == 0
        for _ in range(2):
            result = ledger.check("groundskeeper108", [contract["backend"]], "tools/call", params)
            if "attempt_id" in result:
                execute_count += 1
        assert execute_count == 1
    finally:
        release.set()
        await asyncio.wait_for(callback, 5)
        await asyncio.wait_for(waiter, 5)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["expired", "wrong_actor", "unavailable"])
async def test_tls_refusal_after_queue_removal_never_claims_grant(service, monkeypatch, failure):
    ledger, cfg, config = service
    contract = cfg["contracts"][0]
    params = json.dumps({"name": contract["tool"], "arguments": {"resource_id": "fixture-a", "value": 3}}).encode()
    ref = ledger.check("groundskeeper108", [contract["backend"]], "tools/call", params)["request_id"]
    r = remote(ref)
    r.url = config["mcp_permissions"]["url"]
    a, entered, release, posts = adapter(), threading.Event(), threading.Event(), []
    assert (await a.send_exec_approval("101", "", "s", remote_permission=r, approval_request_id="old")).success
    old = enqueue(r, "old")
    original = ledger.decide
    def delayed(owner, request, body):
        posts.append(request)
        entered.set()
        assert release.wait(5)
        if failure == "unavailable": raise OSError(TOKEN)
        return original(owner, request, body)
    monkeypatch.setattr(ledger, "decide", delayed)
    q = query_for(r)
    callback = asyncio.create_task(a._handle_callback_query(SimpleNamespace(callback_query=q), None))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        with approval._lock:
            approval._gateway_queues["s"].remove(old)
        new = enqueue(remote("b" * 32), "new")
        if failure == "expired": monkeypatch.setattr(ledger, "clock", lambda: r.expires + 1)
        if failure == "wrong_actor": ledger.config["actors"]["groundskeeper108"] = "changed-owner"
        release.set()
        await asyncio.wait_for(callback, 3)
        text = q.edit_message_text.call_args.kwargs["text"]
        assert ("rejected" if failure != "unavailable" else "unconfirmed") in text
        assert "recorded" not in text and TOKEN not in text
        assert not old.event.is_set() and old.result is None
        assert approval._gateway_queues["s"] == [new] and not new.event.is_set()
        await a._handle_callback_query(SimpleNamespace(callback_query=query_for(r)), None)
        assert posts == [ref]
        if failure == "wrong_actor": ledger.config["actors"]["groundskeeper108"] = "serviceUser"
        assert ledger.inspect("serviceUser", ref)["state"] == "pending"
    finally:
        release.set()
        await asyncio.wait_for(callback, 5)
