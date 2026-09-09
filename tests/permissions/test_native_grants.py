"""Real native Telegram dispatch against the independent loopback HTTPS service.

Only Telegram delivery is fake. Synthetic grants belong to a temp fixture owner;
no real bot, human update, workload tool, or production permission state is used.
"""

import asyncio
import copy
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import permission_grants as grants
from gateway.permission_bridge import BridgeError
from plugins.platforms.telegram.grants import native_grants
from test_native_bridge import adapter, service  # Shared isolated HTTPS fixture.


def message(**changes):
    fields = dict(text="/grants", from_user=SimpleNamespace(id=101, first_name="Fixture", is_bot=False),
                  chat=SimpleNamespace(id=101, type="private"), message_thread_id=None,
                  reply_text=AsyncMock())
    fields.update(changes)
    return SimpleNamespace(**fields)


async def command(a, msg=None):
    a._should_process_message = lambda *args, **kwargs: True
    a._is_user_authorized_from_message = lambda *args: True
    a.handle_message = AsyncMock(side_effect=AssertionError("Grant management must not enter the agent loop"))
    await a._handle_command(SimpleNamespace(effective_message=msg or message()), SimpleNamespace())


def button(a, action):
    view = next(iter(native_grants(a).views.values()))
    return SimpleNamespace(data=f"hg:{view.nonce}:{action}",
                           from_user=SimpleNamespace(id=101, first_name="Fixture", is_bot=False),
                           message=SimpleNamespace(chat_id=101, chat=SimpleNamespace(type="private"),
                                                   message_id=42, message_thread_id=None),
                           answer=AsyncMock(), edit_message_text=AsyncMock())


async def click(a, query):
    await a._handle_callback_query(SimpleNamespace(callback_query=query), SimpleNamespace())


def make_grant(service, operation="synthetic-grant", value=3):
    ledger, cfg, _ = service
    contract = cfg["contracts"][0]
    params = json.dumps({"name": contract["tool"], "arguments": {
        "resource_id": "fixture-a", "value": value, "operation_id": operation}}).encode()
    pending = ledger.check("groundskeeper108", [contract["backend"]], "tools/call", params)
    details = ledger.inspect("serviceUser", pending["request_id"])
    choice = next(c["id"] for c in details["choices"] if c["id"] not in ("once", "deny"))
    return ledger.decide("serviceUser", pending["request_id"], {
        "digest": details["digest"], "choice": choice})["grant_id"]


@pytest.mark.asyncio
async def test_native_command_inspect_revoke_service_and_replay(service):
    identity = make_grant(service)
    a = adapter()
    await command(a)
    a.handle_message.assert_not_awaited()
    assert identity in a._bot.send_message.call_args.kwargs["text"]
    inspect = button(a, "i0")
    await click(a, inspect)
    text = inspect.edit_message_text.call_args.kwargs["text"]
    for part in ("groundskeeper108", "fixture_set_value", "fixture-a", "maximum", "10", "expires", "Display fingerprint"):
        assert part in text
    revoke = button(a, "r")
    await click(a, revoke)
    assert "Grant revoked for future use" in revoke.edit_message_text.call_args.kwargs["text"]
    assert service[0].listing("serviceUser", "grants")[0]["revoked"] == 1
    await click(a, revoke)
    audits = service[0].listing("serviceUser", "audit")
    assert sum(e["event"] == "grant_revoked" for e in audits) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", ["user", "chat", "message", "thread", "nonce", "bot", "native_auth", "expired", "restart", "replacement", "action"])
async def test_callback_must_match_native_view_before_http(service, monkeypatch, mismatch):
    make_grant(service)
    a = adapter()
    await command(a)
    await click(a, button(a, "i0"))
    query = button(a, "r")
    if mismatch == "user": query.from_user.id = 202
    elif mismatch == "chat": query.message.chat_id = 202
    elif mismatch == "message": query.message.message_id = 43
    elif mismatch == "thread": query.message.message_thread_id = 12
    elif mismatch == "nonce": query.data = "hg:" + "x" * 32 + ":r"
    elif mismatch == "bot": query.from_user.is_bot = True
    elif mismatch == "native_auth": a._is_callback_user_authorized = lambda *args, **kwargs: False
    elif mismatch == "expired": next(iter(native_grants(a).views.values())).deadline = 0
    elif mismatch == "restart": a = adapter()
    elif mismatch == "replacement": await command(a)
    elif mismatch == "action": query.data += "extra"
    monkeypatch.setattr(grants, "human_http", lambda *args, **kwargs: pytest.fail("No request permitted"))
    await click(a, query)
    query.edit_message_text.assert_not_awaited()
    assert service[0].listing("serviceUser", "grants")[0]["revoked"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_sender", ["native_auth", "bot", "absent", "different_owner"])
async def test_command_requires_actual_authorized_sender(service, monkeypatch, bad_sender):
    identity = make_grant(service)
    a, msg = adapter(), message()
    if bad_sender == "native_auth": a._is_callback_user_authorized = lambda *args, **kwargs: False
    elif bad_sender == "bot": msg.from_user.is_bot = True
    elif bad_sender == "absent": msg.from_user = None
    else: msg.from_user.id = 202
    if bad_sender != "different_owner":
        monkeypatch.setattr(grants, "human_http", lambda *args, **kwargs: pytest.fail("No request permitted"))
    await command(a, msg)
    assert not native_grants(a).views
    assert not any(identity in str(c) for c in a._bot.send_message.call_args_list)


def row(cursor=1):
    return {"cursor": cursor, "grant_id": f"{cursor:032d}", "expires": None, "revoked": 0,
            "scope": {"actor": "fixture-actor", "owner": "fixture-owner", "backend": "fixture-backend",
                      "tool": "fixture_set_value", "contract_digest": "a" * 64,
                      "resources": {"/resource_id": "fixture-a"},
                      "template": {"id": "bounded", "constraints": {"maximum": 10}, "ttl_seconds": None}}}


OWNER = grants.GrantOwner("101", "101", "fixture-owner", "https://fixture.test")


@pytest.mark.parametrize("bad", ["owner", "cursor", "order", "duplicate_id", "scope", "revoked", "expiry", "template", "tail", "count", "oversize"])
def test_entire_owner_page_validated_before_display(monkeypatch, bad):
    rows = [row(i) for i in range(1, 8)]
    if bad == "owner": rows[0]["scope"]["owner"] = "another-owner"
    elif bad == "cursor": rows[0]["cursor"] = True
    elif bad == "order": rows.reverse()
    elif bad == "duplicate_id": rows[1]["grant_id"] = rows[0]["grant_id"]
    elif bad == "scope": rows[0]["scope"].pop("resources")
    elif bad == "revoked": rows[0]["revoked"] = "false"
    elif bad == "expiry": rows[0]["expires"] = float("nan")
    elif bad == "template": rows[0]["scope"]["template"].pop("constraints")
    elif bad == "tail": rows[6]["scope"] = {}  # Beyond the five visible rows.
    elif bad == "count": rows = [row(i) for i in range(1, 102)]
    elif bad == "oversize": rows[6]["scope"]["template"]["constraints"] = {"x": "a" * grants.MAX_PAGE_BYTES}
    monkeypatch.setattr(grants, "human_http", lambda *args, **kwargs: {"items": rows})
    with pytest.raises(BridgeError): grants.list_grants(OWNER)


@pytest.mark.parametrize("mutation", ["scope", "cursor", "identity", "expires"])
def test_refresh_binds_immutable_scope_id_cursor_and_expiry(monkeypatch, mutation):
    original = row()
    monkeypatch.setattr(grants, "human_http", lambda *args, **kwargs: {"items": [original]})
    displayed = grants.list_grants(OWNER)[0]
    if mutation == "scope": original["scope"]["template"]["constraints"]["maximum"] = 20
    elif mutation == "cursor": original["cursor"] = 2
    elif mutation == "identity": original["grant_id"] = "z" * 32
    else: original["expires"] = 1234
    with pytest.raises(BridgeError): grants.refresh_grant(OWNER, displayed)


@pytest.mark.asyncio
async def test_pagination_never_skips_owner_rows_and_buttons_are_bounded(monkeypatch):
    rows = [row(i * 3) for i in range(1, 108)]
    calls = []
    def http(url, user, chat, path, **kwargs):
        assert (user, chat) == ("101", "101")
        after = int(path.split("=")[1])
        calls.append(after)
        return {"items": [r for r in rows if r["cursor"] > after][:100]}
    monkeypatch.setattr(grants, "configured_owner", lambda *args: OWNER)
    monkeypatch.setattr(grants, "human_http", http)
    a = adapter()
    await command(a)
    seen = []
    while True:
        view = next(iter(native_grants(a).views.values()))
        seen.extend(g.grant_id for g in view.rows)
        assert len(view.rows) <= 5 and len(native_grants(a).views) == 1
        if not view.more: break
        query = button(a, "n")
        await click(a, query)
        markup = query.edit_message_text.call_args.kwargs["reply_markup"]
        assert all(len(b.callback_data.encode()) <= 64 for line in markup.inline_keyboard for b in line)
    assert seen == [r["grant_id"] for r in rows]
    assert calls[0] == 0 and calls[1] == rows[4]["cursor"]
    await click(a, button(a, "p"))
    assert next(iter(native_grants(a).views.values())).rows[0].cursor == rows[100]["cursor"]


@pytest.mark.asyncio
async def test_oversized_inspection_never_offers_revoke(monkeypatch):
    original = row()
    original["scope"]["template"]["constraints"] = {"description": "x" * 4000}
    monkeypatch.setattr(grants, "configured_owner", lambda *args: OWNER)
    monkeypatch.setattr(grants, "human_http", lambda *args, **kwargs: {"items": [original]})
    a = adapter()
    await command(a)
    query = button(a, "i0")
    await click(a, query)
    assert not native_grants(a).views
    assert query.edit_message_text.call_args.kwargs["reply_markup"] is None


def test_scope_escape_and_terminal_status(monkeypatch):
    original = row()
    original["scope"]["resources"]["/resource_id"] = "<b>\u202eevil</b>"
    monkeypatch.setattr(grants, "human_http", lambda *args, **kwargs: {"items": [original]})
    text = grants.grant_prompt(grants.list_grants(OWNER)[0])
    assert "&lt;b&gt;" in text and "\\u202e" in text
    original["expires"] = 1
    assert grants.list_grants(OWNER)[0].status == "expired"
    original["revoked"] = 1
    assert grants.list_grants(OWNER)[0].status == "revoked"


@pytest.mark.asyncio
@pytest.mark.parametrize("ack", ["lost_committed", "lost_not_committed", "telegram_ack", "telegram_edit"])
async def test_lost_ack_reconciles_read_only_and_duplicate_never_posts(service, monkeypatch, ack):
    make_grant(service)
    a = adapter()
    await command(a)
    await click(a, button(a, "i0"))
    query = button(a, "r")
    real, posts, paths = grants.human_http, [], []
    def http(url, user, chat, path, body=None, **kwargs):
        paths.append(path)
        if body is not None:
            posts.append(body)
            if ack == "lost_not_committed": raise BridgeError()
        result = real(url, user, chat, path, body, **kwargs)
        if body is not None and ack == "lost_committed": raise BridgeError()
        return result
    monkeypatch.setattr(grants, "human_http", http)
    if ack == "telegram_ack": query.answer.side_effect = RuntimeError("sensitive error")
    if ack == "telegram_edit": query.edit_message_text.side_effect = RuntimeError("sensitive error")
    await click(a, query)
    await click(a, query)
    assert posts == [{}]
    assert service[0].listing("serviceUser", "grants")[0]["revoked"] == (0 if ack == "lost_not_committed" else 1)
    if ack.startswith("lost"):
        assert paths[-1].startswith("/grants?after=")
        text = query.edit_message_text.call_args.kwargs["text"]
        assert ("original response was lost" if ack == "lost_committed" else "unconfirmed") in text


@pytest.mark.asyncio
@pytest.mark.parametrize("when", ["read", "post"])
async def test_replacement_during_network_wait_never_updates_new_view(service, monkeypatch, when):
    make_grant(service)
    a = adapter()
    await command(a)
    await click(a, button(a, "i0"))
    query = button(a, "r")
    started, release = threading.Event(), threading.Event()
    real, posts, delayed = grants.human_http, [], False
    def http(url, user, chat, path, body=None, **kwargs):
        nonlocal delayed
        result = real(url, user, chat, path, body, **kwargs)
        if body is not None: posts.append(body)
        if not delayed and ((when == "read" and body is None) or (when == "post" and body is not None)):
            delayed = True
            started.set()
            assert release.wait(10)
        return result
    monkeypatch.setattr(grants, "human_http", http)
    pending = asyncio.create_task(click(a, query))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        await command(a)
        replacement = next(iter(native_grants(a).views.values()))
    finally:
        release.set()
    await pending
    assert next(iter(native_grants(a).views.values())) is replacement
    query.edit_message_text.assert_not_awaited()
    assert len(posts) == (0 if when == "read" else 1)


@pytest.mark.asyncio
async def test_nonce_claimed_before_telegram_ack_and_remote_revoked_is_not_reposted(service, monkeypatch):
    identity = make_grant(service)
    a = adapter()
    await command(a)
    await click(a, button(a, "i0"))
    query = button(a, "r")
    service[0].revoke("serviceUser", "grants", identity)
    entered, release = asyncio.Event(), asyncio.Event()
    async def ack(**kwargs):
        entered.set()
        await release.wait()
    query.answer.side_effect = ack
    pending = asyncio.create_task(click(a, query))
    await asyncio.wait_for(entered.wait(), 5)
    duplicate = copy.copy(query)
    duplicate.answer = AsyncMock()
    await click(a, duplicate)
    release.set()
    await pending
    assert "already revoked" in query.edit_message_text.call_args.kwargs["text"]
    assert sum(e["event"] == "grant_revoked" for e in service[0].listing("serviceUser", "audit")) == 1


@pytest.mark.asyncio
async def test_view_capacity_expiry_and_failed_delivery_never_keep_authority(monkeypatch):
    from plugins.platforms.telegram import grants as ui_module

    monkeypatch.setattr(ui_module, "MAX_VIEWS", 2)
    monkeypatch.setattr(grants, "configured_owner", lambda user, chat: grants.GrantOwner(user, chat, OWNER.owner, OWNER.url))
    monkeypatch.setattr(grants, "human_http", lambda *args, **kwargs: {"items": []})
    a = adapter()
    for user in (101, 102, 103):
        await command(a, message(from_user=SimpleNamespace(id=user, first_name="Fixture", is_bot=False)))
    ui = native_grants(a)
    assert len(ui.views) == len(ui.generations) == 2
    assert a._bot.send_message.await_count == 2
    for view in ui.views.values():
        view.deadline = 0
        ui.generations[view.key] = view.generation, 0
    ui.prune()
    assert not ui.views and not ui.generations
    a._bot.send_message.side_effect = RuntimeError("sensitive delivery error")
    await command(a)
    assert not ui.views


@pytest.mark.asyncio
async def test_actual_service_pages_immutable_ids_owner_isolation_and_ledger_reopen(service):
    ledger, cfg, _ = service
    expected = []
    for i in range(103):
        identity = make_grant(service, f"synthetic-page-{i}")
        expected.append(identity)
        ledger.revoke("serviceUser", "grants", identity)
    # A separately owned synthetic row must never leak, including after gaps
    # in the global cursor sequence. All data remains in the fixture database.
    with ledger.transaction() as db:
        db.execute("INSERT INTO grants(id,actor,owner,scope,expires) VALUES(?,?,?,?,?)",
                   ("x" * 32, "other-actor", "other-owner", "{}", None))
    owner = grants.configured_owner("101", "101")
    first = grants.list_grants(owner)
    assert len(first) == 100
    tail = grants.list_grants(owner, first[-1].cursor)
    assert [g.grant_id for g in (*first, *tail)] == expected
    assert grants.list_grants(owner, tail[-1].cursor) == ()
    a = adapter()
    await command(a)
    old = button(a, "i0")
    replacement = adapter()
    await click(replacement, old)
    assert not native_grants(replacement).views
    await command(replacement)
    await click(replacement, button(replacement, "i0"))
    assert next(iter(native_grants(replacement).views.values())).inspected.revoked
    reopened = type(ledger)(ledger.path, cfg)
    assert len(reopened.listing("serviceUser", "grants")) == 100
    assert reopened.listing("serviceUser", "grants")[0]["revoked"] == 1


@pytest.mark.asyncio
async def test_config_change_or_expiry_during_read_prevents_revoke(service, monkeypatch):
    make_grant(service)
    a = adapter()
    await command(a)
    await click(a, button(a, "i0"))
    query = button(a, "r")
    real = grants.human_http
    def http(url, user, chat, path, body=None, **kwargs):
        assert body is None
        result = real(url, user, chat, path, body, **kwargs)
        service[2]["mcp_permissions"]["enabled"] = False
        return result
    monkeypatch.setattr(grants, "human_http", http)
    await click(a, query)
    assert not native_grants(a).views
    assert service[0].listing("serviceUser", "grants")[0]["revoked"] == 0
