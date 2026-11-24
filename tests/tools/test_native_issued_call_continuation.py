"""Consume one genuine pair through an independently admitted native completion."""

import asyncio
import copy
import hashlib
import json
import threading
import time
from contextvars import copy_context
from dataclasses import replace
from types import SimpleNamespace

import pytest
from hermes_content_policy import ContentPolicyRuntime, Destination, PATHS, ProfilePolicy, RouteBinding

from agent import native_main_payload as native
from agent.content_policy import NativeContentContext, NativeContentPolicyDenied, content_policy_scope
from agent.transports.types import NormalizedResponse, Usage
from tests.agent.test_native_main_payload import ControlledPort, canonical
from tests.tools.test_native_issued_call_execution import ARGUMENTS
from tests.tools.test_native_issued_call_pair import measured
from tests.tools.test_native_issued_call_format import (
    CompactFormatWorld,
    excluded as excluded,
    invented_offline_configuration as invented_offline_configuration,
    no_delegation_effects as no_delegation_effects,
    record_case_start as record_case_start,
    retain_case_durations as retain_case_durations,
    retain_failure_contexts as retain_failure_contexts,
    retain_incremental_case_receipts as retain_incremental_case_receipts,
)
from tests.tools.test_native_issued_call_wire import record_property as record_property


class ContinuationWorld(CompactFormatWorld):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.requests += (object(),)
        holder = replace(self.bindings[4])
        self.bindings += (holder,)
        destination = Destination("invented_5", "synthetic_native", "invented_model_5", "invented_endpoint_5", 1)
        port = ControlledPort(destination)
        port.output = b'{"accepted":true}'
        for path in PATHS:
            self.ports[5, path] = port
        self.routes += (RouteBinding("route_5_main", holder.selection_id, holder.selection_version,
            holder.consumer_ref, "main", destination, 50 if mode == "selected_expiry" else 100,
            "synthetic_5_main", "synthetic_only"),)
        if mode == "coverage_expiry":
            self.routes = tuple(replace(route, expires_at=50) if route.path == "fallback" else route
                for route in self.routes)
        self.profiles = (ProfilePolicy("restricted_documents", tuple(route.route_ref for route in self.routes), PATHS),)
        self.transports += ((destination.transport_ref, port),)
        self.selections.append(native.NativeMainSelection(holder.selection_id, holder.selection_version,
            "chat_completions", destination))
        self.selection_ports += (SimpleNamespace(current=self.selection),)
        self.policies += (native.NativeResultPolicy("producer_5", 4, ("restricted_documents",)),)
        self.consumer_context = NativeContentContext(self.admission, self.requests[5])
        self.timings = {}

    def prepare(self, record):
        from run_agent import AIAgent
        with measured(self, record, "setup_qualifications"):
            self.qualifier = ContentPolicyRuntime(admission=self.admission, profiles=self.profiles,
                routes=self.routes, transports=self.transports, clock=self.clock, policy_revision=9)
            qualifications = [(self.requests[int(route.route_ref.split("_")[1])], route) for route in self.routes]
            qualifications += [(self.requests[5], next(route for route in self.routes if route.path == path
                and (route.consumer_ref, route.selection_id, route.selection_version)
                    == (self.bindings[5].consumer_ref, self.bindings[5].selection_id, self.bindings[5].selection_version)))
                for path in PATHS if path != "main"]
            for request, route in qualifications:
                port = self.ports[int(route.route_ref.split("_")[1]), route.path]
                content = self.qualifier.admit(request, b"Invented qualification")
                result = self.qualifier.dispatch(request, content, path=route.path, route_ref=route.route_ref)
                self.qualifications.append((request, content, result, port.sent[-1]))
                port.sent.clear()
            self.ports[0, "main"].output = canonical(dict(id="invented_response", object="chat.completion", created=1,
                model="invented_model_0", choices=[dict(index=0, finish_reason="tool_calls", message=dict(role="assistant",
                    content=None, tool_calls=[dict(id="invented-call-1", type="function",
                        function=dict(name="delegate_task", arguments=ARGUMENTS))]))],
                usage=dict(prompt_tokens=5, completion_tokens=3, total_tokens=8)))
            self.ports[3, "main"].output = canonical(dict(id="invented_child_reply", object="chat.completion", created=12,
                model="invented_model_3", choices=[dict(index=0, finish_reason="stop",
                    message=dict(role="assistant", content="Three blue squares."))],
                usage=dict(prompt_tokens=3, completion_tokens=2, total_tokens=5)))
            self.ports[5, "main"].output = canonical(dict(id="invented_continuation_reply", object="chat.completion", created=21,
                model="invented_model_5", choices=[dict(index=0, finish_reason="stop",
                    message=dict(role="assistant", content="The leaf counted three blue squares."))],
                usage=dict(prompt_tokens=11, completion_tokens=7, total_tokens=18)))
            self.adapter = native.bind_controlled_main(self.context, selection=self.selection_ports[0],
                result_policy=self.result_port, producer_ref="producer_0", main_route_ref="route_0_main",
                profiles=self.profiles, routes=self.routes, transports=self.transports, clock=self.clock)
            with content_policy_scope(self.context):
                self.agent = AIAgent(controlled_main=self.adapter, provider="synthetic_native", api_mode="chat_completions",
                    model="invented_model_0", session_id="invented_source_session", enabled_toolsets=[], quiet_mode=True,
                    skip_memory=True, skip_context_files=True, skip_background_review=True)
        with measured(self, record, "bind_continuation"):
            self.authorization = native.bind_issued_call_continuation(self.adapter,
                consumer_context=self.consumer_context, consumer_selection=self.selection_ports[5],
                consumer_producer_ref="producer_5", consumer_main_route_ref="route_5_main",
                target_model=self.selections[5].destination.model_ref, session_id="invented_consumer_session")
        with content_policy_scope(self.context):
            with measured(self, record, "bind_enable"):
                self.probe_token = self.adapter.bind_tool_call_probe(self.agent, system_message="Request one leaf call.",
                    user_message="Count blue squares.", task_id="execution", expected_arguments_json=ARGUMENTS)
                self.adapter.enable_tool_call_assistant_source(self.agent, self.probe_token)
            with measured(self, record, "source_capture_close"):
                self.observation = self.adapter.run_tool_call_probe(self.agent, self.probe_token)
                self.agent.close()
                self.source = self.adapter.tool_call_assistant_sources[0]
            with measured(self, record, "issued_child_return"):
                self.bind()
                self.source_result = self.run()
            with measured(self, record, "formatter"):
                self.format_token = self.format_bind()
                self.formatted = self.adapter.run_issued_call_format(self.format_token)
            with measured(self, record, "Wf"):
                self.tool = self.adapter.retain_issued_call_wire(self.formatted)
            with measured(self, record, "Wa"):
                self.assistant = self.adapter.retain_tool_call_assistant_wire(self.source)
            with measured(self, record, "J"):
                self.pair = self.adapter.retain_issued_call_pair(self.assistant, self.tool)
        return self


def retain_continuation(world, record):
    runtime = world.adapter._runtime
    owner = native._continuation_owner(world.authorization)
    indices = {row.handle: i for i, row in enumerate(runtime._contents.values())}
    record("continuation_actual", json.dumps(dict(mode=world.mode, phase=owner.phase, windows=owner.windows,
        timings=owner.timings, source_timings=world.timings, qualification_sends=len(world.qualifications),
        data_sends={str(i): len(world.ports[i, "main"].sent) for i in range(6)},
        counts=(len(runtime._contents), len(runtime._envelopes), runtime._retained_bytes),
        before=owner.charge_before, after=owner.charge_after, poisoned=runtime._poisoned,
        raw=None if owner.raw_slot[2] is None else owner.raw_slot[2].decode(),
        request=None if owner.request_bytes is None else owner.request_bytes.decode(),
        wire=None if owner.wire_bytes is None else owner.wire_bytes.decode(),
        normalized=None if owner.normalized_bytes is None else owner.normalized_bytes.decode(),
        completion=None if owner.completion is None else owner.completion.completion_bytes.decode(),
        policy_observations=owner.policy_observations,
        graph=[dict(index=indices[row.handle], handle_identity=id(row.handle), payload=row.payload.decode(),
            hex=row.payload.hex(), sha256=hashlib.sha256(row.payload).hexdigest(), parents=[indices[h] for h in row.parents],
            binding=row.binding, profiles=row.profiles, path=row.path) for row in runtime._contents.values()])))


@pytest.mark.parametrize("mode", ["positive", "coverage_expiry", "original_control", "selected_expiry"])
def test_genuine_continuation(no_delegation_effects, record_property, mode, monkeypatch):
    world = ContinuationWorld(mode)
    try:
        world.prepare(record_property)
        owner = native._continuation_owner(world.authorization)
        runtime = owner.runtime
        before = (len(runtime._contents), len(runtime._envelopes), runtime._retained_bytes)
        original = tuple(vars(world.adapter).items())
        control = asyncio.CancelledError("Invented continuation policy cancellation")
        refusals = []
        continuation_started = time.monotonic()
        last_boundary = []
        last_stages = []

        def after_output():
            stages = tuple(value is not None for value in (owner.history_content, owner.request_content,
                owner.wire_content, owner.raw_slot[2], owner.response_content, owner.normalized_content, owner.completion))
            boundary = (owner.windows, owner.phase, stages)
            if not last_boundary or last_boundary[0] != boundary:
                last_boundary[:] = [boundary]
                record_property("C_boundary", json.dumps(dict(window=owner.windows, phase=owner.phase,
                    elapsed=time.monotonic() - continuation_started, completed_stages=stages,
                    charges=(len(runtime._contents), len(runtime._envelopes), runtime._retained_bytes),
                    C_sends=owner.adapter._send_count, callback_depth=owner.depth,
                    raw_bytes=None if owner.raw_slot[2] is None else len(owner.raw_slot[2]))))
            if not last_stages or last_stages[0] != stages:
                last_stages[:] = [stages]
                if any(stages):
                    retain_continuation(world, record_property)
            if mode == "positive" and owner.windows > 0 and not refusals:
                with pytest.raises(NativeContentPolicyDenied, match="reentry_or_thread"):
                    native.issued_call_continuation_results(world.authorization)
                refusals.append(True)
            if owner.raw_slot[2] is not None:
                if mode in ("coverage_expiry", "selected_expiry"):
                    world.now = 50
                elif mode == "original_control":
                    raise control

        world.hooks["clock"] = after_output
        if mode != "positive":
            error = asyncio.CancelledError if mode == "original_control" else NativeContentPolicyDenied
            with pytest.raises(error) as caught:
                with content_policy_scope(world.consumer_context):
                    native.run_issued_call_continuation(world.authorization, world.pair)
            assert owner.raw_slot[2] is world.ports[5, "main"].output and len(world.ports[5, "main"].sent) == 1
            assert owner.records == () and owner.phase == "denied"
            if mode == "original_control":
                assert caught.value is owner.control_exception is control
            else:
                assert owner.policy_observations[-1][-1] == world.now == 50
                if mode == "coverage_expiry":
                    assert all(route.expires_at > world.now for route in world.routes if route.path == "main")
                else:
                    assert owner.route.expires_at == 50
                    assert any(route.path == "main" and route is not owner.route
                        and route.selection_id == owner.route.selection_id and route.expires_at > world.now
                        for route in world.routes)
            return
        with measured(world, record_property, "continuation"):
            with content_policy_scope(world.consumer_context):
                with pytest.raises(TypeError):
                    copy.copy(world.authorization)
                copied_token = object.__new__(native.NativeIssuedCallContinuationInput)
                with pytest.raises(NativeContentPolicyDenied, match="token_unregistered"):
                    native.run_issued_call_continuation(copied_token, world.pair)
                with pytest.raises(NativeContentPolicyDenied, match="pair_unregistered"):
                    native.run_issued_call_continuation(world.authorization, replace(world.pair))
                assert before == (len(runtime._contents), len(runtime._envelopes), runtime._retained_bytes)
                result = native.run_issued_call_continuation(world.authorization, world.pair)
                assert native.issued_call_continuation_results(world.authorization)[0] is result
                with pytest.raises(NativeContentPolicyDenied, match="attempt_spent"):
                    native.run_issued_call_continuation(world.authorization, world.pair)
                context = copy_context()
                def foreign():
                    with pytest.raises(NativeContentPolicyDenied, match="reentry_or_thread"):
                        native.issued_call_continuation_results(world.authorization)
                thread = threading.Thread(target=lambda: context.run(foreign))
                thread.start()
                thread.join(timeout=5)
                assert not thread.is_alive()
                replacement_calls = []

                def replaced_pair_getter(pair_owner):
                    replacement_calls.append(pair_owner)
                    return ()

                with monkeypatch.context() as patch:
                    patch.setattr(native._IssuedCallPair, "proof_access", replaced_pair_getter)
                    with pytest.raises(NativeContentPolicyDenied, match="pair_code_changed"):
                        native.issued_call_continuation_results(world.authorization)
                    assert replacement_calls == []
        assert refusals == [True] and result.pair is world.pair and result.binding is owner.binding
        assert result.request["model"] is world.selections[5].destination.model_ref
        assert result.request["messages"][:2] == result.prefix_messages
        assert result.request["messages"][2] is world.pair.messages[0]
        assert result.request["messages"][3] is world.pair.messages[1]
        assert result.tools is native._tool_call_probes[world.adapter].request["tools"]
        assert set(result.request) == {"model", "messages", "tools", "stream", "tool_choice"}
        assert result.invocation.native_response is result.completion.response is owner.response
        assert result.invocation.response_bytes is owner.raw_slot[2] is world.ports[5, "main"].output
        assert type(result.normalized) is NormalizedResponse and type(result.normalized.usage) is Usage
        assert result.normalized.tool_calls is result.normalized.reasoning is result.normalized.provider_data is None
        assert result.completion.normalized is result.normalized and result.completion.usage is result.normalized.usage
        assert result.completion.final_response == "The leaf counted three blue squares."
        assert vars(result.normalized.usage) == dict(prompt_tokens=11, completion_tokens=7, total_tokens=18, cached_tokens=0)
        assert json.loads(result.completion.completion_bytes)["completed"] is True
        assert result.agent.tools == result.agent.enabled_toolsets == result.agent._session_messages == []
        assert all(vars(world.adapter)[key] is value for key, value in original)
        assert tuple(vars(world.adapter)) == tuple(key for key, _ in original)
        assert owner.windows == 6 and owner.phase == "released"
        assert len(runtime._contents) == before[0] + 6 and len(runtime._envelopes) == before[1] + 1
    finally:
        if hasattr(world, "authorization"):
            retain_continuation(world, record_property)


@pytest.mark.parametrize("change", ["stock", "replacement", "code_object_drift"])
def test_continuation_pair_code_guard(monkeypatch, change):
    pair_class, name, original, code = next(entry for entry in native._ISSUED_PAIR_CODE
        if entry[0] is native._IssuedCallPair and entry[1] == "proof_access")
    assert native._verify_continuation_pair_code() is None

    def replacement(pair_owner):
        raise AssertionError("Replacement getter must not run")

    with monkeypatch.context() as patch:
        if change == "replacement":
            patch.setattr(pair_class, name, replacement)
        elif change == "code_object_drift":
            patch.setattr(original, "__code__", replacement.__code__)
        if change == "stock":
            assert native._verify_continuation_pair_code() is None
        else:
            with pytest.raises(NativeContentPolicyDenied, match="pair_code_changed"):
                native._verify_continuation_pair_code()
    assert getattr(pair_class, name) is original and original.__code__ is code
    assert native._verify_continuation_pair_code() is None
