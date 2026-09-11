"""Real native agent loops must finish SDK observations without manual cleanup.

The model reply is synthetic; the actual conversation loop, lifecycle dispatcher,
plugin helpers and SDK run unchanged. Default HTTP export is intercepted only at
requests.Session.post, before any network request can leave the test process.
"""

import importlib
import json
from dataclasses import replace
from unittest.mock import patch

import pytest

pytest.importorskip("langfuse", minversion="4.15.2")

from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from agent.conversation_loop import run_conversation as _real_loop
from hermes_cli import plugins as plugin_manager
from tests.gateway.test_pwa_http import service, model_config
from tests.gateway.test_pwa_inspector import get
from tests.gateway.test_pwa_inspector_facts import scope
from tests.plugins.test_langfuse_native_sdk import CaptureOTLP
from tests.run_agent.test_run_agent import (
    agent as agent,  # Re-export the shared pytest fixture.
    TestRunConversation as _ConversationHarness,
    _mock_response,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["pwa", "telegram"])
@pytest.mark.parametrize("transport", ["capture", "default_http"])
async def test_actual_runner_finishes_and_exports_native_sdk(
    monkeypatch,
    tmp_path,
    agent,  # noqa: F811 - shared pytest fixture
    origin,
    transport,
):
    plugin = importlib.import_module("plugins.observability.langfuse")
    wire = []
    # SDK resource managers are cached by public key, including after shutdown.
    # Each case must own its exporter instead of reusing another case's client.
    public_key = f"pk-lf-native-runner-{transport}-{origin}"
    secret_key = f"sk-lf-native-runner-{transport}-{origin}"
    if transport == "default_http":
        import requests

        def post(session, url, data, **kwargs):
            assert url == "https://offline.invalid/api/public/otel/v1/traces"
            assert kwargs["verify"] is True
            assert session.headers["Content-Type"] == "application/x-protobuf"
            assert session.headers["x-langfuse-sdk-name"] == "python"
            assert session.headers["x-langfuse-sdk-version"] == "4.15.2"
            wire.append(data)
            response = requests.Response()
            response.status_code = 200
            response._content = b""
            return response

        monkeypatch.setattr(requests.Session, "post", post)
        monkeypatch.setenv("HERMES_LANGFUSE_PUBLIC_KEY", public_key)
        monkeypatch.setenv("HERMES_LANGFUSE_SECRET_KEY", secret_key)
        monkeypatch.setenv("HERMES_LANGFUSE_BASE_URL", "https://offline.invalid")
        monkeypatch.setattr(plugin, "_LANGFUSE_CLIENT", None)
        client = plugin._get_langfuse()
        assert client is not None
    else:
        capture, provider = CaptureOTLP(), TracerProvider()
        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url="http://127.0.0.1:1",
            tracer_provider=provider,
            span_exporter=capture,
            should_export_span=lambda span: (
                span.instrumentation_scope.name == "langfuse-sdk"
            ),
        )
        wire = capture.wire
        monkeypatch.setattr(plugin, "_LANGFUSE_CLIENT", client)
    manager = plugin_manager.PluginManager()
    manager._discovered = True
    plugin.register(
        plugin_manager.PluginContext(
            plugin_manager.PluginManifest(name="langfuse"), manager
        )
    )
    monkeypatch.setattr(plugin_manager, "get_plugin_manager", lambda: manager)
    monkeypatch.setenv("HERMES_LANGFUSE_CAPTURE", "full")
    try:
        async with service(monkeypatch, tmp_path, models=model_config()) as (
            server,
            http,
            native,
        ):
            db, root = native[2], native[4].session_id
            config = server.ownership.config
            server.ownership.config = replace(
                config,
                bindings=(
                    replace(config.bindings[0], observation_actor="synthetic-owner"),
                ),
                inspector_model_names=("Qwen/Qwen3.8-27B-262k",),
                inspector_provider_names=("custom",),
            )
            db.native_execution_open(
                root, "normal-sdk-turn", server.ingress._command_owner, origin=origin
            )
            _ConversationHarness()._setup_agent(agent)
            agent.session_id, agent.platform = root, origin
            agent.model, agent.provider = "upstream/daily", "custom"
            agent.base_url = "http://127.0.0.1:1/v1"
            agent.tool_use_enforcement = "off"
            response = _mock_response(
                content="PRIVATE ANSWER",
                usage={"prompt_tokens": 19, "completion_tokens": 5, "total_tokens": 24},
            )
            response.model = "Qwen/Qwen3.8-27B-262k"
            agent.client.chat.completions.create.return_value = response
            with (
                scope(server, root, "normal-sdk-turn"),
                patch("agent.conversation_loop.run_conversation", _real_loop),
                patch.object(agent, "_persist_session"),
                patch.object(agent, "_save_trajectory"),
                patch.object(agent, "_cleanup_task_resources"),
            ):
                result = agent.run_conversation("PRIVATE USER MESSAGE")
            assert result["completed"] is True
            # Check BEFORE shutdown, explicit SDK flush, session finalization,
            # or manual span ending: the ordinary post_api_request must do it.
            assert not plugin._TRACE_STATE and not plugin._NATIVE_PARENTS
            spans = []
            for data in wire:
                envelope = ExportTraceServiceRequest.FromString(data)
                spans.extend(
                    span
                    for resource in envelope.resource_spans
                    for captured_scope in resource.scope_spans
                    for span in captured_scope.spans
                )
            assert len(spans) == 2
            roots = [span for span in spans if not span.parent_span_id]
            assert len(roots) == 1
            assert len({span.trace_id for span in spans}) == 1
            generations = [span for span in spans if span.parent_span_id]
            assert generations[0].parent_span_id == roots[0].span_id
            assert b"PRIVATE" not in b"".join(wire)
            attributes = {
                item.key: item.value.string_value for item in generations[0].attributes
            }
            assert attributes["langfuse.observation.type"] == "generation"
            assert attributes["user.id"] == "synthetic-owner"
            usage = json.loads(attributes["langfuse.observation.usage_details"])
            assert usage["input"] == 19 and usage["output"] == 5
            task = (await get(http, "/v1/pwa/inspector/tasks"))["rows"][0]["task_ref"]
            facts = await get(http, f"/v1/pwa/inspector/tasks/{task}/facts")
            call = facts["model_calls"][0]
            assert call["state"] == "completed"
            with db._read_ctx() as connection:
                retained = connection.execute(
                    "SELECT body FROM native_inspector_facts WHERE execution_id=? AND kind='model'",
                    ("normal-sdk-turn",),
                ).fetchall()
            assert len(retained) == 1
            captured_call = json.loads(retained[0][0])
            assert captured_call["trace_id"] == generations[0].trace_id.hex()
            assert captured_call["observation_id"] == generations[0].span_id.hex()
            assert call["trace_ref"].startswith("d_")
            assert call["observation_ref"].startswith("d_")
            assert call["actual_model"] == "Qwen/Qwen3.8-27B-262k"
    finally:
        client.shutdown()
