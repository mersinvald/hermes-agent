"""Actual native dispatch functions with invented requests and offline providers."""

from types import SimpleNamespace
from dataclasses import replace

import pytest

from agent import chat_completion_helpers as helpers
from agent.content_policy import NativeContentPolicyDenied
from agent.conversation_loop import run_conversation
from agent.turn_context import build_turn_context, _maybe_title_session_at_turn_start
from run_agent import AIAgent
from tests.agent.test_content_policy import enabled_agent, make_context


@pytest.mark.parametrize("mode,provider", [
    ("chat_completions", "custom"), ("codex_responses", "custom"),
    ("anthropic_messages", "anthropic"), ("bedrock_converse", "bedrock"),
    ("chat_completions", "moa"),
])
@pytest.mark.parametrize("operation", ["nonstream", "stream", "summary", "fallback", "loop", "forwarder"])
def test_native_paths_deny_sensitive_content_before_client_or_callback(mode, provider, operation):
    context, _ = make_context(profiles=("restricted-documents",))
    agent = enabled_agent(context)
    agent.api_mode, agent.provider = mode, provider
    calls = []
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: calls.append(kwargs))))
    agent.client = client
    make_client = lambda *args, **kwargs: calls.append("client construction") or client
    callbacks = {
        "nonstream": lambda: helpers._dispatch_nonstreaming_api_request(agent, {"model": "invented"}, make_client=make_client),
        "stream": lambda: helpers.interruptible_streaming_api_call(agent, {"model": "invented"}),
        "summary": lambda: helpers.handle_max_iterations(agent, [], 1),
        "fallback": lambda: helpers.try_activate_fallback(agent),
        "loop": lambda: run_conversation(agent, "invented document"),
        "forwarder": lambda: AIAgent.run_conversation(agent, "invented document"),
    }
    with pytest.raises(NativeContentPolicyDenied, match="sensitive_native"):
        callbacks[operation]()
    assert calls == []


def test_ordinary_request_uses_actual_request_local_client_and_unchanged_arguments():
    context, _ = make_context()
    agent = enabled_agent(context)
    agent.api_mode, agent.provider = "chat_completions", "custom"
    calls = []
    reply = object()
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: calls.append(kwargs) or reply)))
    request = {"model": "invented", "messages": [{"role": "user", "content": "invented document"}]}
    assert helpers._dispatch_nonstreaming_api_request(agent, request, make_client=lambda *_: client) is reply
    assert calls == [request]


@pytest.mark.parametrize("change_at", ["client", "response"])
def test_current_binding_changes_at_provider_boundaries_do_not_release_results(change_at):
    context, state = make_context()
    agent = enabled_agent(context)
    agent.api_mode, agent.provider = "chat_completions", "custom"
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if change_at == "response":
            state["binding"] = replace(state["binding"], auth_epoch=8)
        return object()

    def make_client(*args):
        if change_at == "client":
            state["binding"] = replace(state["binding"], profiles=("restricted-documents",))
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    with pytest.raises(NativeContentPolicyDenied):
        helpers._dispatch_nonstreaming_api_request(agent, {"model": "invented"}, make_client=make_client)
    assert len(calls) == (1 if change_at == "response" else 0)


def test_turn_prologue_and_title_launch_refuse_before_callbacks():
    context, _ = make_context(profiles=("restricted-documents",))
    agent = enabled_agent(context)
    callback = lambda *args, **kwargs: pytest.fail("prologue callback ran")
    with pytest.raises(NativeContentPolicyDenied):
        build_turn_context(agent, "invented", None, [], "task", None, None,
            restore_or_build_system_prompt=callback, install_safe_stdio=callback,
            sanitize_surrogates=callback, summarize_user_message_for_log=callback,
            set_session_context=callback, set_current_write_origin=callback, ra=object())
    with pytest.raises(NativeContentPolicyDenied):
        _maybe_title_session_at_turn_start(agent, [])
