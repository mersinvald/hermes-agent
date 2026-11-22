"""Native admission with real ownership and SQLite, without provider calls."""

import asyncio
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agent.content_policy import NativeContentPolicyDenied, guard_inbound, ingress_policy
from gateway.conversation_control import NativeConversationIngress, Principal
from gateway.pwa_config import NativeContentPolicyConfig, PwaHttpConfig
from gateway.pwa_models import NativeModelCatalog
from gateway.pwa_ownership import NativeContentPolicyAdmission, NativePwaOwnership
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner, TurnRunner
from gateway.session import SessionStore
from hermes_state import SessionDB
from tests.gateway.test_42039_duplicate_user_message import _bootstrap, _source
from tests.gateway.test_pwa_models import pwa_config

OWNER = Principal("https://issuer.invalid", "owner")


def setup_policy(monkeypatch, tmp_path, profiles=()):
    runner = _bootstrap(monkeypatch, tmp_path)
    source = _source()
    raw = pwa_config()
    configured = raw["bindings"][0]["sources"][0]
    configured.update(platform=source.platform.value, chat_id=source.chat_id,
                      chat_type=source.chat_type, user_id=source.user_id,
                      thread_id=source.thread_id)
    raw["content_policy"] = {"enabled": True, "auth_epoch": 7, "policy_revision": 9,
                             "assignments": {"telegram-owner": list(profiles)}}
    runner.config.pwa_http = raw
    parsed = PwaHttpConfig.from_dict(raw)
    db = SessionDB(tmp_path / "native-policy.db")
    store = SessionStore(tmp_path / "sessions", runner.config)
    store._db = db
    runner.session_store = store
    runner._is_user_authorized_for_source = lambda candidate: candidate.user_id == source.user_id
    entry = store.get_or_create_session(source)
    ownership = NativePwaOwnership(runner, db, parsed, secret="invented-secret")
    models = NativeModelCatalog(parsed.models, lambda _: {"provider": "custom", "base_url": "http://invented.invalid/v1"})
    ingress = NativeConversationIngress(runner, db, {}, concierge_id="synthetic",
                                      grant_provider=ownership, model_catalog=models)
    db.record_gateway_session_peer(entry.session_id, source=source.platform.value,
        user_id=source.user_id, session_key=entry.session_key, chat_id=source.chat_id,
        chat_type=source.chat_type, thread_id=source.thread_id)
    ingress.content_policy = NativeContentPolicyAdmission(ingress, ownership)
    return runner, ingress, db, entry, source


def command(root, command_id="invented-command"):
    return {"schema_version": "1.0", "command_id": command_id, "conversation_id": root,
            "type": "send", "payload": {"text": "invented document"}}


@pytest.mark.parametrize("enabled", [None, 0, 1, "true", [], {}])
def test_enabled_requires_an_exact_boolean(enabled):
    with pytest.raises(ValueError):
        NativeContentPolicyConfig.from_dict({"enabled": enabled})


@pytest.mark.parametrize("field,value", [("auth_epoch", True), ("auth_epoch", 0),
    ("policy_revision", False), ("policy_revision", -1), ("assignments", {}),
    ("assignments", {"source": None}), ("assignments", {"source": ["x", "x"]})])
def test_invalid_operator_assignments_and_revisions_reject(field, value):
    raw = {"enabled": True, "auth_epoch": 1, "policy_revision": 1,
           "assignments": {"source": []}}
    raw[field] = value
    with pytest.raises(ValueError):
        NativeContentPolicyConfig.from_dict(raw)


@pytest.mark.asyncio
async def test_sensitive_submit_denies_before_ledger_or_preprocessing(monkeypatch, tmp_path):
    runner, ingress, db, entry, _ = setup_policy(monkeypatch, tmp_path, ("restricted-documents",))
    monkeypatch.setattr(runner, "_handle_message", lambda _: pytest.fail("preprocessing reached"))
    try:
        with pytest.raises(NativeContentPolicyDenied, match="sensitive_native"):
            await ingress.submit(OWNER, command(entry.session_id))
        assert db.native_command_rows(entry.session_id) == []
    finally:
        ingress.stop_command_recovery()
        db.close()


@pytest.mark.asyncio
async def test_owned_receipt_replay_never_reprocesses_images(monkeypatch, tmp_path):
    runner, ingress, db, entry, _ = setup_policy(monkeypatch, tmp_path, ("restricted-documents",))
    request = command(entry.session_id)
    request["payload"]["image_ids"] = ["im_" + "a" * 32 + "_" + "b" * 64]
    request["expected_model_version"] = 1
    db.native_model_initialize(entry.session_id, "daily")
    row, _ = db.native_command_admit(ingress._command_scope(OWNER), request, effect="start")
    monkeypatch.setattr(ingress, "_arm_command_recovery", lambda: None)
    monkeypatch.setattr(ingress, "_commands_finished", lambda _: None)
    try:
        result = await ingress.submit(OWNER, request)
        assert result["command_id"] == row["command_id"]
        assert db.native_command_lookup(row["scope"], row["command_id"])["phase"] == "queued"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_real_execution_capture_keeps_independent_identity_and_selection(monkeypatch, tmp_path):
    runner, ingress, db, entry, source = setup_policy(monkeypatch, tmp_path)
    event = MessageEvent(text="invented", source=source)
    try:
        root, _ = await ingress.prepare_admission(event, source, entry.session_key)
        ingress.reserve(root, source, entry.session_key, 1, event)
        execution = runner._peek_session_state(entry.session_key).turn.conversation_execution
        assert db.acquire_session_turn_lease(root, "invented-lease", ttl_seconds=30)
        captured = db.native_model_capture(execution.execution_id, root, ingress._command_owner,
                                          "invented-lease", "daily")
        context = ingress.content_policy.context(execution, captured)
        facts = context.check("main")
        assert facts.execution_ref == execution.execution_id
        assert (facts.auth_epoch, facts.policy_revision, facts.selection_version) == (7, 9, 1)
        assert facts.owner_ref != ingress._command_owner
        db.native_execution_close(execution.execution_id, ingress._command_owner, outcome="failed")
        with pytest.raises(NativeContentPolicyDenied):
            context.check("memory")
    finally:
        ingress.stop_command_recovery()
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_adapter", [False, True])
async def test_recovered_enabled_command_retains_failed_ledger_row(monkeypatch, tmp_path, missing_adapter):
    _, ingress, db, entry, _ = setup_policy(monkeypatch, tmp_path)
    row, _ = db.native_command_admit(ingress._command_scope(OWNER), command(entry.session_id), effect="start")
    if missing_adapter:
        del ingress.content_policy
    try:
        with pytest.raises(NativeContentPolicyDenied, match="adapter_missing|restoration_required"):
            await ingress._dispatch_command(row)
        restored = db.native_command_lookup(row["scope"], row["command_id"])
        assert restored["phase"] == "not_applied"
        execution = db.get_session(entry.session_id)
        assert execution is not None
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("function", ["_prepare_inbound_message_text", "_prepare_profile_scoped_inbound_message_text"])
async def test_direct_preprocessing_source_value_has_no_admission(monkeypatch, tmp_path, function):
    runner, ingress, db, entry, source = setup_policy(monkeypatch, tmp_path)
    event = MessageEvent(text="invented", source=source)
    try:
        with pytest.raises(NativeContentPolicyDenied, match="event_missing"):
            await getattr(GatewayRunner, function)(runner, event=event, source=source,
                                                  history=[], session_key=entry.session_key)
    finally:
        ingress.stop_command_recovery()
        db.close()


@pytest.mark.asyncio
async def test_authenticated_event_and_current_configuration_are_required(monkeypatch, tmp_path):
    runner, ingress, db, entry, source = setup_policy(monkeypatch, tmp_path)
    event = MessageEvent(text="invented", source=source)
    try:
        root, control = await ingress.prepare_admission(event, source, entry.session_key)
        assert root == entry.session_id and control is None
        guard_inbound(runner, event, source)
        with pytest.raises(NativeContentPolicyDenied):
            guard_inbound(runner, copy.copy(event), source)
        runner.config.pwa_http = copy.deepcopy(runner.config.pwa_http)
        with pytest.raises(NativeContentPolicyDenied, match="configuration_changed"):
            guard_inbound(runner, event, source)
    finally:
        db.close()


@pytest.mark.asyncio
async def test_missing_adapter_denies_and_absent_section_retains_legacy(monkeypatch, tmp_path):
    runner, ingress, db, _, _ = setup_policy(monkeypatch, tmp_path)
    del ingress.content_policy
    try:
        with pytest.raises(NativeContentPolicyDenied, match="adapter_missing"):
            ingress_policy(ingress)
        del runner.config.pwa_http["content_policy"]
        assert ingress_policy(ingress) is None
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["assignment", "owner", "authorization"])
async def test_current_trusted_source_faults_deny_before_preprocessing(monkeypatch, tmp_path, fault):
    runner, ingress, db, entry, source = setup_policy(monkeypatch, tmp_path)
    raw = runner.config.pwa_http
    if fault == "assignment":
        raw["content_policy"]["assignments"] = {"other-source": []}
    elif fault == "owner":
        other = copy.deepcopy(raw["bindings"][0])
        other["subject"] = "another-owner"
        raw["bindings"].append(other)
    else:
        runner._is_user_authorized_for_source = lambda _: False
    ownership = NativePwaOwnership(runner, db, PwaHttpConfig.from_dict(raw), secret="invented")
    ingress._grant_provider = ownership
    ingress.content_policy = NativeContentPolicyAdmission(ingress, ownership)
    try:
        with pytest.raises((NativeContentPolicyDenied, PermissionError)):
            await ingress.prepare_admission(MessageEvent(text="invented", source=source), source, entry.session_key)
        assert db.native_command_rows(entry.session_id) == []
    finally:
        db.close()


@pytest.mark.asyncio
async def test_turn_runner_denies_missing_execution_before_agent_construction(monkeypatch, tmp_path):
    runner, ingress, db, entry, source = setup_policy(monkeypatch, tmp_path)
    runner._get_system_prompt_for_channel = lambda *args, **kwargs: ""
    runner._peek_session_state = lambda _: None
    context = SimpleNamespace(source=source, context_prompt="", channel_prompt="", history=[],
        session_key=entry.session_key, AIAgent=lambda **kwargs: pytest.fail("agent construction reached"))
    try:
        result = TurnRunner(runner, context).run_sync()
        assert result["failed"] is True and result["api_calls"] == 0
        assert result["error"] == "content_policy_restoration_required"
    finally:
        db.close()
