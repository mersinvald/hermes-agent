"""Real v4 SDK export, intercepted as serialized OTLP; no external service."""

import importlib
import os
import sys
import time
from dataclasses import replace

import pytest

# Release verification can supply a reviewed isolated wheel target. Normal
# development installs the pinned `langfuse` extra instead.
if os.environ.get("HERMES_TEST_LANGFUSE_SDK"):
    sys.path.insert(0, os.environ["HERMES_TEST_LANGFUSE_SDK"])

pytest.importorskip("langfuse", minversion="4.15.2")

from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from hermes_cli.lifecycle import invoke_hook
from hermes_cli import plugins as plugin_manager
from tests.gateway.test_pwa_http import service, model_config
from tests.gateway.test_pwa_inspector_facts import scope


class CaptureOTLP(SpanExporter):
    def __init__(self):
        self.spans = []
        self.wire = []

    def export(self, spans):
        self.spans.extend(spans)
        self.wire.append(encode_spans(spans).SerializeToString())
        return SpanExportResult.SUCCESS


@pytest.mark.asyncio
async def test_native_real_sdk_exports_actor_parent_usage_and_no_content(
    monkeypatch, tmp_path
):
    plugin = importlib.import_module("plugins.observability.langfuse")
    provider, exporter = TracerProvider(), CaptureOTLP()
    client = Langfuse(
        public_key="pk-lf-native-sdk-test",
        secret_key="sk-lf-native-sdk-test",
        base_url="http://127.0.0.1:1",
        tracer_provider=provider,
        span_exporter=exporter,
        should_export_span=lambda span: (
            span.instrumentation_scope.name == "langfuse-sdk"
        ),
    )
    monkeypatch.setattr(plugin, "_LANGFUSE_CLIENT", client)
    manager = plugin_manager.PluginManager()
    manager._discovered = True
    plugin.register(
        plugin_manager.PluginContext(
            plugin_manager.PluginManifest(name="langfuse"), manager
        )
    )
    monkeypatch.setattr(plugin_manager, "get_plugin_manager", lambda: manager)
    monkeypatch.setenv(
        "HERMES_LANGFUSE_CAPTURE", "full"
    )  # Native scope still withholds payloads.
    async with service(monkeypatch, tmp_path, models=model_config()) as (
        server,
        _,
        native,
    ):
        db, root = native[2], native[4].session_id
        cfg = server.ownership.config
        server.ownership.config = replace(
            cfg,
            bindings=(replace(cfg.bindings[0], observation_actor="synthetic-owner"),),
        )
        db.native_execution_open(
            root, "sdk-execution", server.ingress._command_owner, origin="telegram"
        )
        with scope(server, root, "sdk-execution"):
            values = dict(
                session_id=root,
                task_id="native-task",
                turn_id="native-turn",
                api_request_id="one",
                api_call_count=1,
                model="upstream/daily",
                provider="provider-a",
                started_at=time.time(),
                messages=[{"role": "user", "content": "PRIVATE PROMPT"}],
            )
            invoke_hook("pre_api_request", **values)
            invoke_hook(
                "pre_tool_call",
                session_id=root,
                task_id="native-task",
                turn_id="native-turn",
                api_request_id="one",
                tool_name="read_file",
                tool_call_id="tool",
                args={"PRIVATE KEY": "PRIVATE ARG"},
            )
            invoke_hook(
                "post_tool_call",
                session_id=root,
                task_id="native-task",
                turn_id="native-turn",
                api_request_id="one",
                tool_name="read_file",
                tool_call_id="tool",
                args={},
                result="PRIVATE FILE",
            )
            child = dict(
                parent_session_id=root,
                parent_turn_id="native-turn",
                child_session_id="child",
                child_role="leaf",
                child_goal="PRIVATE GOAL",
            )
            invoke_hook("subagent_start", **child)
            child_values = dict(
                values,
                session_id="child",
                task_id="child-task",
                turn_id="child-turn",
                api_request_id="child-one",
                started_at=time.time(),
            )
            invoke_hook("pre_api_request", **child_values)
            error = dict(
                child_values,
                ended_at=time.time(),
                error={"type": "APITimeoutError", "message": "PRIVATE ERROR"},
                status_code=504,
                retryable=False,
            )
            invoke_hook("api_request_error", **error)
            child.update(
                child_status="failed", child_summary="PRIVATE SUMMARY", duration_ms=5
            )
            invoke_hook("subagent_stop", **child)
            completed = dict(
                values,
                ended_at=time.time(),
                response_model="upstream/daily",
                usage={"input_tokens": 11, "output_tokens": 7},
                accounting_cost={
                    "amount": 0.01,
                    "currency": "USD",
                    "basis": "estimated",
                    "source": "native_accounting",
                },
                assistant_content_chars=17,
                assistant_response="PRIVATE RESPONSE",
                api_duration=0.02,
            )
            invoke_hook("post_api_request", **completed)
        db.native_execution_close(
            "sdk-execution", server.ingress._command_owner, outcome="completed"
        )
        with provider.get_tracer("unrelated-library").start_as_current_span(
            "PRIVATE UNRELATED"
        ):
            pass
        assert not plugin._TRACE_STATE and not plugin._NATIVE_PARENTS
        client.flush()
        assert len(exporter.spans) == 6
        assert len({span.context.trace_id for span in exporter.spans}) == 1
        assert sum(span.parent is None for span in exporter.spans) == 1
        for span in exporter.spans:
            attrs = dict(span.attributes)
            assert attrs["user.id"] == "synthetic-owner"
            assert attrs.get("session.id", "").startswith("d_")
        wire = b"".join(exporter.wire)
        assert b"PRIVATE" not in wire
        assert b"hermes_native_correlation_ref" in wire
        assert b"hermes_native_parent_agent_ref" in wire
        generations = [
            span
            for span in exporter.spans
            if span.attributes.get("langfuse.observation.type") == "generation"
        ]
        assert len(generations) == 2
        assert any(
            span.attributes.get("langfuse.observation.level") == "ERROR"
            for span in generations
        )
        assert any(
            "11" in str(span.attributes.get("langfuse.observation.usage_details"))
            for span in generations
        )
        assert not plugin._TRACE_STATE and not plugin._NATIVE_PARENTS
        fact = db.native_inspector_fact("sdk-execution", "model", root + "\0one")
        assert fact["trace_id"] and fact["observation_id"]
    client.shutdown()


@pytest.mark.asyncio
async def test_real_dispatch_retry_tool_cycle_empty_finish_and_exact_cancellation(
    monkeypatch, tmp_path
):
    plugin = importlib.import_module("plugins.observability.langfuse")
    exporter, provider = CaptureOTLP(), TracerProvider()
    client = Langfuse(
        public_key="pk-lf-native-lifecycle-test",
        secret_key="sk-lf-native-lifecycle-test",
        base_url="http://127.0.0.1:1",
        tracer_provider=provider,
        span_exporter=exporter,
        should_export_span=lambda span: (
            span.instrumentation_scope.name == "langfuse-sdk"
        ),
    )
    monkeypatch.setattr(plugin, "_LANGFUSE_CLIENT", client)
    manager = plugin_manager.PluginManager()
    manager._discovered = True
    plugin.register(
        plugin_manager.PluginContext(
            plugin_manager.PluginManifest(name="langfuse"), manager
        )
    )
    monkeypatch.setattr(plugin_manager, "get_plugin_manager", lambda: manager)
    async with service(monkeypatch, tmp_path, models=model_config()) as (
        server,
        _,
        native,
    ):
        db, root = native[2], native[4].session_id
        cfg = server.ownership.config
        server.ownership.config = replace(
            cfg,
            bindings=(replace(cfg.bindings[0], observation_actor="synthetic-owner"),),
        )
        owner = server.ingress._command_owner
        db.native_execution_open(root, "retry-execution", owner, origin="pwa")
        with scope(server, root, "retry-execution"):
            first = dict(
                session_id=root,
                task_id="retry",
                turn_id="retry-turn",
                api_request_id="retry-one",
                api_call_count=1,
                model="upstream/daily",
                provider="provider-a",
                started_at=time.time(),
            )
            invoke_hook("pre_api_request", **first, retry_count=0)
            invoke_hook(
                "api_request_error",
                **first,
                ended_at=time.time(),
                status_code=429,
                retry_count=0,
                max_retries=3,
                retryable=True,
                error={"type": "RateLimitError", "message": "PRIVATE RATE LIMIT"},
            )
            invoke_hook("pre_api_request", **first, retry_count=1)
            invoke_hook(
                "post_api_request",
                **first,
                ended_at=time.time(),
                response_model="upstream/daily",
                usage={"input_tokens": 9, "output_tokens": 3},
                assistant_tool_call_count=1,
            )
            assert plugin._TRACE_STATE
            invoke_hook(
                "pre_tool_call",
                session_id=root,
                task_id="retry",
                turn_id="retry-turn",
                api_request_id="retry-one",
                tool_call_id="read",
                tool_name="read_file",
                args={"path": "PRIVATE"},
            )
            invoke_hook(
                "post_tool_call",
                session_id=root,
                task_id="retry",
                turn_id="retry-turn",
                api_request_id="retry-one",
                tool_call_id="read",
                tool_name="read_file",
                result="PRIVATE",
            )
            second = dict(
                first,
                api_request_id="retry-two",
                api_call_count=2,
                started_at=time.time(),
            )
            invoke_hook("pre_api_request", **second)
            invoke_hook(
                "post_api_request",
                **second,
                ended_at=time.time(),
                usage={"input_tokens": 10, "output_tokens": 0},
                assistant_content_chars=0,
                assistant_tool_call_count=0,
            )
            assert not plugin._TRACE_STATE  # Empty final response is still final.
        fact = db.native_inspector_fact(
            "retry-execution", "model", root + "\0retry-one"
        )
        assert fact["attempt_count"] == 2 and fact["failed_attempt_count"] == 1
        db.native_execution_close("retry-execution", owner, outcome="completed")
        source = cfg.bindings[0].sources[0].source
        neighbor = root + "-neighbor"
        db.create_session(
            neighbor,
            source.platform.value,
            user_id=source.user_id,
            chat_id=source.chat_id,
            chat_type=source.chat_type,
            thread_id=source.thread_id,
        )
        for session, execution in ((root, "cancel"), (neighbor, "neighbor")):
            db.native_execution_open(session, execution, owner, origin="pwa")
            with scope(server, session, execution):
                invoke_hook(
                    "pre_api_request",
                    session_id=session,
                    task_id=execution,
                    turn_id=execution,
                    api_request_id="pending",
                    api_call_count=1,
                    model="upstream/daily",
                    provider="provider-a",
                    started_at=time.time(),
                )
        assert len(plugin._TRACE_STATE) == 2
        from hermes_cli.lifecycle import finalize_session

        finalize_session(session_id=root, reason="cancelled")
        assert len(plugin._TRACE_STATE) == 1
        assert next(iter(plugin._TRACE_STATE.values())).native_session_id == neighbor
        finalize_session(session_id=neighbor, reason="cancelled")
        assert not plugin._TRACE_STATE and not plugin._NATIVE_PARENTS
        client.flush()
        assert len(exporter.spans) == 9
        assert b"PRIVATE" not in b"".join(exporter.wire)
        attempts = [
            dict(s.attributes) for s in exporter.spans if s.name == "LLM call 1"
        ]
        assert len(attempts) == 4
        assert b"hermes_native_attempt_count" in b"".join(exporter.wire)
    client.shutdown()
