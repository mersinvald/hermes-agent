"""Actual auxiliary and title entrypoints with offline HTTP boundaries."""

from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from agent import auxiliary_client as auxiliary, title_generator as titles
from agent.content_policy import NativeContentPolicyDenied, content_policy_scope
from agent.context_compressor import ContextCompressor
from tests.agent.test_content_policy import enabled_agent, make_context


def response(request):
    return httpx.Response(200, request=request, json={"id": "invented", "object": "chat.completion",
        "model": "invented-model", "choices": [{"index": 0, "finish_reason": "stop",
        "message": {"role": "assistant", "content": '{"title":"Invented document review"}'}}]})


@pytest.mark.parametrize("function", ["call_llm", "_call_llm_impl", "generate_title", "maybe_auto_title", "auto_title_session"])
def test_sensitive_auxiliary_and_title_refuse_before_snippets_or_workers(function, monkeypatch):
    context, state = make_context()
    calls = []
    monkeypatch.setattr(httpx.Client, "send", lambda *args, **kwargs: calls.append("provider"))
    callbacks = {
        "call_llm": lambda: auxiliary.call_llm(messages=[]),
        "_call_llm_impl": lambda: auxiliary._call_llm_impl(messages=[]),
        "generate_title": lambda: titles.generate_title("invented", runtime_validator=lambda: calls.append("validator")),
        "maybe_auto_title": lambda: titles.maybe_auto_title(object(), "session", "invented"),
        "auto_title_session": lambda: titles.auto_title_session(object(), "session", "invented"),
    }
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(context):
            state["binding"] = replace(state["binding"], profiles=("restricted-documents",))
            callbacks[function]()
            pytest.fail("native entrypoint returned")
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("function", ["async_call_llm", "_async_call_llm_impl"])
async def test_async_auxiliary_refuses_before_provider_await(function, monkeypatch):
    context, state = make_context()
    calls = []

    async def send(*args, **kwargs):
        calls.append("provider")

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(context):
            state["binding"] = replace(state["binding"], profiles=("restricted-documents",))
            await getattr(auxiliary, function)(messages=[])
            pytest.fail("native async entrypoint returned")
    assert calls == []


@pytest.mark.parametrize("function,args", [("_generate_summary", ([],)), ("_micro_summarize_one", ("invented",)), ("compress", ([],))])
def test_compression_refuses_before_summary_conversion(function, args):
    context, _ = make_context(profiles=("restricted-documents",))
    compressor = enabled_agent(context)
    with pytest.raises(NativeContentPolicyDenied):
        getattr(ContextCompressor, function)(compressor, *args)


def test_managed_ordinary_title_uses_real_auxiliary_resolution_and_fake_http(monkeypatch):
    context, _ = make_context()
    calls = []

    def send(self, request, **kwargs):
        calls.append((str(request.url), request.content))
        return response(request)

    monkeypatch.setattr(httpx.Client, "send", send)
    destination = {"provider": "custom", "model": "invented-model",
                   "base_url": "http://invented.invalid/v1", "api_key": "invented-key"}
    with content_policy_scope(context):
        title = titles.generate_title("Review the invented document", destination=destination)
    assert title == "Invented document review"
    assert len(calls) == 1 and calls[0][0] == "http://invented.invalid/v1/chat/completions"


def test_broken_title_validator_cannot_override_policy_denial():
    context, state = make_context()
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(context):
            state["binding"] = replace(state["binding"], auth_epoch=8)
            titles.generate_title("invented", runtime_validator=lambda: (_ for _ in ()).throw(RuntimeError()))
            pytest.fail("title entrypoint returned")


from tests.agent.test_native_controlled_conversation import (
    auxiliary_world, bind_title, turn,
    excluded as excluded,
    caller_boundary as caller_boundary,
    invented_offline_configuration as invented_offline_configuration,
    ReservedProviderConversions, ReservedProviderEquality, ReservedProviderString,
)


@pytest.mark.parametrize("operation", ["private", "relay", "background", "private_background", "maybe", "instant", "stream"])
def test_retained_auxiliary_siblings_refuse_with_empty_profiles(excluded, caller_boundary, monkeypatch, operation):
    from agent import skill_commands
    monkeypatch.setattr(skill_commands, "describe_skill_invocation", excluded)
    monkeypatch.setattr(titles.threading, "Thread", excluded)
    monkeypatch.setattr(auxiliary, "_get_cached_client", excluded)
    monkeypatch.setattr(auxiliary, "_relay_auxiliary_metadata", excluded)
    world = auxiliary_world()
    result = turn(world)
    with content_policy_scope(world.context):
        bind_title(world)
    world.binding = replace(world.binding, profiles=())
    world.title_binding = replace(world.title_binding, profiles=())
    storage = SimpleNamespace(get_session_title=excluded, set_session_title=excluded)
    operations = {
        "private": lambda: auxiliary._call_llm_impl(messages=[]),
        "stream": lambda: auxiliary._relay_sync_stream(None, {}),
        "relay": lambda: auxiliary._relay_sync_completion(None, {}, create=excluded),
        "background": lambda: titles.auto_title_session(object(), "invented", result["final_response"]),
        "private_background": lambda: titles._auto_title_session(object(), "invented", result["final_response"]),
        "maybe": lambda: titles.maybe_auto_title(object(), "invented", result["final_response"]),
        "instant": lambda: titles.apply_instant_title(storage, "invented", result["final_response"]),
    }
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            operations[operation]()
    assert world.adapter.auxiliary_send_count == 0 and world.adapter.send_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["async_call_llm", "_async_call_llm_impl", "_relay_async_completion"])
async def test_retained_auxiliary_async_siblings_refuse(excluded, caller_boundary, monkeypatch, operation):
    monkeypatch.setattr(auxiliary, "_acquire_async_aux_semaphore", excluded)
    world = auxiliary_world()
    turn(world)
    with content_policy_scope(world.context):
        bind_title(world)
    world.binding = replace(world.binding, profiles=())
    world.title_binding = replace(world.title_binding, profiles=())
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            if operation == "_relay_async_completion":
                await auxiliary._relay_async_completion(None, {})
            else:
                await getattr(auxiliary, operation)(messages=[])
    assert world.adapter.auxiliary_send_count == 0


@pytest.mark.parametrize("provider_class", [str, ReservedProviderString,
    ReservedProviderEquality, ReservedProviderConversions])
def test_reserved_auxiliary_provider_refuses_before_relay_and_resolution(monkeypatch, provider_class):
    def forbidden(*args, **kwargs):
        pytest.fail("Ordinary auxiliary callback ran")
    monkeypatch.setattr(auxiliary.uuid, "uuid4", forbidden)
    monkeypatch.setattr(auxiliary, "_acquire_sync_aux_semaphore", forbidden)
    monkeypatch.setattr(auxiliary, "_get_cached_client", forbidden)
    with pytest.raises(NativeContentPolicyDenied):
        auxiliary.call_llm(provider=provider_class("synthetic_native"), messages=[])


@pytest.mark.parametrize("provider_class", [str, ReservedProviderEquality, ReservedProviderConversions])
def test_reserved_private_auxiliary_provider_refuses_without_custom_operations(monkeypatch, provider_class):
    def forbidden(*args, **kwargs):
        pytest.fail("Private auxiliary resolution ran")
    monkeypatch.setattr(auxiliary, "_get_cached_client", forbidden)
    with pytest.raises(NativeContentPolicyDenied):
        auxiliary._call_llm_impl(provider=provider_class("synthetic_native"), messages=[])


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["async_call_llm", "_async_call_llm_impl", "_relay_async_completion"])
async def test_reserved_async_auxiliary_provider_refuses_before_resolution(monkeypatch, operation):
    def forbidden(*args, **kwargs):
        pytest.fail("Async auxiliary resolution ran")
    monkeypatch.setattr(auxiliary, "_acquire_async_aux_semaphore", forbidden)
    monkeypatch.setattr(auxiliary, "_relay_auxiliary_metadata", forbidden)
    with pytest.raises(NativeContentPolicyDenied):
        if operation == "_relay_async_completion":
            await auxiliary._relay_async_completion(None, {}, provider=ReservedProviderConversions("synthetic_native"))
        else:
            await getattr(auxiliary, operation)(provider=ReservedProviderConversions("synthetic_native"), messages=[])


@pytest.mark.parametrize("marker", [None, 0, False, ReservedProviderConversions("invented_marker")])
def test_retained_auxiliary_missing_token_with_malformed_marker_refuses(excluded, caller_boundary, marker):
    world = auxiliary_world()
    turn(world)
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            world.context._controlled_main_bound = marker
            titles.generate_title("Invented source")
    assert world.context._denied and world.adapter.auxiliary_send_count == 0
