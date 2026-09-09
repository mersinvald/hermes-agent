"""Real prologue commits command input before later preparation/provider work."""

from types import SimpleNamespace


from gateway.native_commands import NativeCommandContext
from hermes_state import SessionDB
from hermes_state_commands import process_owner
from tests.agent.test_turn_context import _FakeAgent, _build


def test_native_prologue_input_and_api_sidecar_are_durable(monkeypatch, tmp_path):
    monkeypatch.setattr("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None)
    db = SessionDB(tmp_path / "state.db")
    db.create_session("sess-1", "telegram")
    owner = process_owner()
    command = {
        "schema_version": "1.0",
        "command_id": "c1",
        "conversation_id": "sess-1",
        "type": "send",
        "payload": {"text": "hello"},
    }
    db.native_command_admit("owner", command, effect="start")
    db.native_execution_open("sess-1", "e1", owner, command=("owner", "c1"))
    assert db.try_acquire_session_turn_lease("sess-1", "holder")
    agent = _FakeAgent()
    agent._session_db = db
    agent._active_session_turn_lease_holder = "holder"
    ingress = SimpleNamespace(db=db, _command_owner=owner)
    context = NativeCommandContext(
        ingress,
        SimpleNamespace(execution_id="e1", conversation_id="sess-1"),
        ("owner", "c1"),
        None,
    )
    agent._native_command_context = context
    ctx = _build(agent)
    row = db.native_command_lookup("owner", "c1")
    assert row["phase"] == "applied"
    assert ctx.messages[-1]["_row_id"] == row["input_row_id"]
    assert len(db.get_messages("sess-1")) == 1
    ctx.messages[-1]["api_content"] = "hello\nsynthetic prefetch context"
    context.refresh_input(agent, ctx.messages)
    saved = db.get_messages("sess-1")[0]
    assert saved["content"] == "hello"
    assert saved["api_content"] == "hello\nsynthetic prefetch context"
    db.close()
