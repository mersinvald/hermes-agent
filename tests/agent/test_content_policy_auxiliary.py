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
