"""Retain a genuine same-call transcript without another native send."""

import asyncio
import hashlib
import json
import threading
import time
from contextvars import copy_context
from dataclasses import FrozenInstanceError, replace

import pytest
from hermes_content_policy import ContentPolicyRuntime

from agent import native_main_payload as native
from agent.content_policy import NativeContentPolicyDenied, content_policy_scope
from tests.agent.test_native_main_payload import canonical
from tests.tools.test_native_issued_call_execution import ARGUMENTS
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


class PairWorld(CompactFormatWorld):
    def prepare(self, record):
        from run_agent import AIAgent

        self.timings = {}
        with measured(self, record, "setup_qualifications"):
            self.qualifier = ContentPolicyRuntime(admission=self.admission, profiles=self.profiles,
                routes=self.routes, transports=self.transports, clock=self.clock, policy_revision=9)
            for route in self.routes:
                i = int(route.route_ref.split("_")[1])
                request, port = self.requests[i], self.ports[i, route.path]
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
            self.adapter = native.bind_controlled_main(self.context, selection=self.selection_ports[0],
                result_policy=self.result_port, producer_ref="producer_0", main_route_ref="route_0_main",
                profiles=self.profiles, routes=self.routes, transports=self.transports, clock=self.clock)
            with content_policy_scope(self.context):
                self.agent = AIAgent(controlled_main=self.adapter, provider="synthetic_native", api_mode="chat_completions",
                    model="invented_model_0", session_id="invented_source_session", enabled_toolsets=[], quiet_mode=True,
                    skip_memory=True, skip_context_files=True, skip_background_review=True)
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
        return self


class measured:
    def __init__(self, world, record, phase):
        self.world, self.record, self.phase = world, record, phase

    def __enter__(self):
        self.start = time.monotonic()
        self.record("phase_started", self.phase)

    def __exit__(self, cls, error, traceback):
        self.world.timings[self.phase] = time.monotonic() - self.start
        self.record("phase_" + self.phase, json.dumps(dict(seconds=self.world.timings[self.phase],
            outcome="passed" if error is None else "failed", error_type=None if cls is None else cls.__name__)))
        if hasattr(self.world, "adapter"):
            retain(self.world, self.record, self.phase)


def retain(world, record, phase):
    runtime = world.adapter._runtime
    indices = {row.handle: i for i, row in enumerate(runtime._contents.values())}
    owner = native._issued_pair_owner(world.adapter)
    record("actual_" + phase, json.dumps(dict(timings=world.timings,
        qualifications=len(world.qualifications), sends={str(i): len(world.ports[i, "main"].sent) for i in range(5)},
        data_sends=sum(len(p.sent) for p in dict.fromkeys(world.ports.values())),
        content_slots=len(runtime._contents), envelopes=len(runtime._envelopes), retained_bytes=runtime._retained_bytes,
        poisoned=runtime._poisoned, now=world.now,
        graph=[dict(index=indices[row.handle], identity=id(row.handle), payload=row.payload.decode(),
            bytes=len(row.payload), sha256=hashlib.sha256(row.payload).hexdigest(),
            parents=[indices[p] for p in row.parents], path=row.path, binding=row.binding, profiles=row.profiles)
            for row in runtime._contents.values()],
        pair=None if owner is None else dict(phase=owner.phase, windows=owner.windows,
            raw_messages=owner.raw_slot[0], raw_bytes=None if owner.raw_slot[2] is None else owner.raw_slot[2].decode(),
            raw_hex=None if owner.raw_slot[2] is None else owner.raw_slot[2].hex(),
            handle=None if owner.message_content is None else indices[owner.message_content],
            before=owner.charge_before, after=owner.charge_after, results=len(owner.records),
            policy_observations=owner.policy_observations,
            control_type=None if owner.control_exception is None else type(owner.control_exception).__name__)),
        ensure_ascii=False))


def charges(runtime):
    return len(runtime._contents), len(runtime._envelopes), runtime._retained_bytes


def test_genuine_same_call_pair(no_delegation_effects, record_property):
    world = PairWorld().prepare(record_property)
    adapter, runtime = world.adapter, world.adapter._runtime
    before = charges(runtime)
    namespace = tuple(vars(adapter).items())
    owners = len(native._issued_pair_owners)
    with measured(world, record_property, "pair"):
        with content_policy_scope(world.context):
            world.hooks["admission"] = lambda: pytest.fail("Copied pair reached admission")
            for a, t in ((replace(world.assistant), world.tool), (world.assistant, replace(world.tool))):
                with pytest.raises(NativeContentPolicyDenied, match="records_unregistered"):
                    adapter.retain_issued_call_pair(a, t)
                assert charges(runtime) == before and len(native._issued_pair_owners) == owners
            world.hooks.clear()
            reentries = []

            def reenter():
                world.hooks.pop("admission")
                with pytest.raises(NativeContentPolicyDenied, match="reentry_or_thread"):
                    _ = adapter.issued_call_pair_results
                reentries.append(True)

            world.hooks["admission"] = reenter
            result = adapter.retain_issued_call_pair(world.assistant, world.tool)
    with measured(world, record_property, "assertions"):
        assert reentries == [True]
        assert type(result) is native.NativeIssuedCallPairResult
        assert result.assistant_result is world.assistant and result.tool_result is world.tool
        assert result.model is world.formatted.binding.selection.destination.model_ref
        assert result.assistant_result.model is world.observation.binding.selection.destination.model_ref
        assert result.assistant_result.model is not result.model
        assert result.messages[0] is world.assistant.messages[0] and result.messages[1] is world.tool.messages[0]
        wa, wf, j = world.assistant.message_bytes, world.tool.message_bytes, result.message_bytes
        assert len(world.assistant.messages) == len(world.tool.messages) == 1
        assert j == b"[" + wa[1:-1] + b"," + wf[1:-1] + b"]"
        assert len(j) == len(wa) + len(wf) - 1
        assert j == json.dumps(result.messages, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
        retained = runtime._content(result.message_content)
        assert retained.payload is j and retained.parents[0] is world.assistant.message_content
        assert retained.parents[1] is world.tool.message_content and retained.path == "main"
        assert retained.profiles == ("restricted_documents",)
        assert charges(runtime) == (before[0] + 1, before[1], before[2] + len(j))
        owner = native._issued_pair_owner(adapter)
        assert owner.windows == 3 and owner.phase == "released"
        assert all(tuple(vars(obj)) == tuple(key for key, _ in snapshot)
            and all(vars(obj)[key] is value for key, value in snapshot) for obj, snapshot in owner.snapshots)
        assert all(vars(adapter)[name] is value for name, value in namespace) and len(vars(adapter)) == len(namespace)
        assert adapter.issued_call_pair_results == (result,)
        assert owner.assistant_owner.probe.stage_index == 8 and owner.assistant_owner.producer.windows == 3
        assert world.agent._native_controlled_binding.closed
        assert sum(len(p.sent) for p in dict.fromkeys(world.ports.values())) == 3
        assert adapter.send_count == 1 and owner.formatter.execution.adapter.send_count == 0
        assert owner.formatter.execution.frame.child_adapter.send_count == 1
        with pytest.raises(FrozenInstanceError):
            result.model = "different"
        with content_policy_scope(world.context):
            with pytest.raises(NativeContentPolicyDenied, match="attempt_spent"):
                adapter.retain_issued_call_pair(world.assistant, world.tool)
        refused = []
        context = copy_context()

        def foreign():
            with pytest.raises(NativeContentPolicyDenied, match="reentry_or_thread"):
                _ = adapter.issued_call_pair_results
            refused.append(True)

        thread = threading.Thread(target=lambda: context.run(foreign))
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive() and refused == [True]
        assert adapter.issued_call_pair_results[0] is result
        assert charges(runtime) == owner.charge_after and len(native._issued_pair_owners) == owners + 1


def test_final_non_main_coverage_after_J(no_delegation_effects, record_property):
    world = PairWorld()
    world.routes = tuple(replace(route, expires_at=50 if route.path == "fallback" else 100) for route in world.routes)
    world.prepare(record_property)
    clocks = []

    def expire():
        owner = native._issued_pair_owner(world.adapter)
        if owner is not None and owner.windows == 3:
            clocks.append(world.now)
            if len(clocks) == 8:
                world.now = 50

    world.hooks["clock"] = expire
    with measured(world, record_property, "pair_expiry"):
        with pytest.raises(NativeContentPolicyDenied, match="coverage_missing"):
            with content_policy_scope(world.context):
                world.adapter.retain_issued_call_pair(world.assistant, world.tool)
    owner = native._issued_pair_owner(world.adapter)
    assert owner.message_content is not None and owner.records == () and len(clocks) == 8
    assert owner.policy_observations[-1][-1] == world.now == 50
    assert all(world.now < route.expires_at for route in owner.formatter.selected_routes)
    assert owner.charge_after == (owner.charge_before[0] + 1, owner.charge_before[1],
        owner.charge_before[2] + len(owner.message_bytes))
    assert world.adapter.issued_call_pair_results == ()


def test_original_control_after_actual_sanitizer(no_delegation_effects, record_property):
    world = PairWorld().prepare(record_property)
    error = asyncio.CancelledError("Invented pair policy cancellation")

    def cancel():
        owner = native._issued_pair_owner(world.adapter)
        if owner is not None and owner.raw_slot[0] is not None:
            raise error

    world.hooks["clock"] = cancel
    with measured(world, record_property, "pair_control"):
        with pytest.raises(asyncio.CancelledError) as caught:
            with content_policy_scope(world.context):
                world.adapter.retain_issued_call_pair(world.assistant, world.tool)
    owner = native._issued_pair_owner(world.adapter)
    assert caught.value is owner.control_exception is error
    assert owner.raw_slot[0] is not None and owner.message_bytes is owner.raw_slot[2]
    assert owner.message_content is None and owner.records == () and owner.charge_after == owner.charge_before
    assert world.adapter.issued_call_pair_results == ()
