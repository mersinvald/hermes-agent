"""Run one public native history against an independent owned receiver."""

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import pytest
from hermes_content_policy import ContentPolicyRuntime, PATHS

from agent import issued_call_episode as episode, native_main_payload as native
from agent.content_policy import NativeContentPolicyDenied, content_policy_scope, current_context
from agent.transports.owned_loopback_http import OwnedLoopbackEndpoint, OwnedLoopbackHTTPTransport
from run_agent import AIAgent
from tests.agent.owned_loopback_receiver import OwnedReceiver, ResponseStep, response_bytes
from tests.tools.test_native_issued_call_episode import ARGUMENTS, canonical
from tests.tools.test_native_issued_call_continuation import ContinuationWorld
from tests.agent.test_native_controlled_conversation import excluded as excluded
from tests.tools.test_native_controlled_delegation import no_delegation_effects as no_delegation_effects


@pytest.fixture
def record(record_property, request):
    def retain(name, value):
        encoded = json.dumps(dict(nodeid=request.node.nodeid, property=name, value=value),
            default=lambda item: dict(bytes_hex=item.hex()) if type(item) is bytes else str(item))
        record_property(name, json.dumps(value, default=str))
        path = Path(request.config.option.xmlpath).with_suffix(".receipts.jsonl")
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, "a") as output:
            output.write(encoded + "\n")
            output.flush()
            os.fsync(output.fileno())
    return retain


def episode_outputs():
    return {
        0: canonical(dict(id="invented_response", object="chat.completion", created=1,
            model="invented_model_0", choices=[dict(index=0, finish_reason="tool_calls", message=dict(role="assistant",
                content=None, tool_calls=[dict(id="invented-call-1", type="function",
                    function=dict(name="delegate_task", arguments=ARGUMENTS))]))],
            usage=dict(prompt_tokens=5, completion_tokens=3, total_tokens=8))),
        2: b'{"accepted":true}',
        3: canonical(dict(id="invented_child_reply", object="chat.completion", created=12,
            model="invented_model_3", choices=[dict(index=0, finish_reason="stop",
                message=dict(role="assistant", content="Three blue squares."))],
            usage=dict(prompt_tokens=3, completion_tokens=2, total_tokens=5))),
        5: canonical(dict(id="invented_continuation_reply", object="chat.completion", created=21,
            model="invented_model_5", choices=[dict(index=0, finish_reason="stop",
                message=dict(role="assistant", content="The leaf counted three blue squares."))],
            usage=dict(prompt_tokens=11, completion_tokens=7, total_tokens=18))),
    }


class LoopbackWorld(ContinuationWorld):
    def __init__(self, receiver):
        super().__init__("positive")
        self.receiver = receiver
        self.retain = None
        self.outputs = episode_outputs()
        roles = ((0, "p"), (2, "receiver"), (3, "child"), (5, "c"))
        endpoints, destinations = {}, {}
        for index, role in roles:
            original = self.ports[index, "main"].destination
            destination = replace(original, endpoint_ref="synthetic_loopback_" + role)
            endpoint = OwnedLoopbackEndpoint(destination.endpoint_ref, receiver.bound[0], receiver.bound[1],
                "/owned-accountant-proof/v1/" + role, destination.provider_ref, destination.model_ref, destination.revision)
            port = OwnedLoopbackHTTPTransport(destination, endpoint=endpoint)
            for path in PATHS:
                self.ports[index, path] = port
            destinations[index], endpoints[index] = destination, endpoint
            self.selections[index] = replace(self.selections[index], destination=destination)
        self.routes = tuple(replace(route, destination=destinations[int(route.route_ref.split("_")[1])])
            if int(route.route_ref.split("_")[1]) in destinations else route for route in self.routes)
        self.transports = tuple((port.resolve().transport_ref, port) for port in dict.fromkeys(self.ports.values()))
        qualifications = [(self.requests[int(route.route_ref.split("_")[1])], route) for route in self.routes]
        qualifications += [(self.requests[5], next(route for route in self.routes if route.path == path
            and (route.consumer_ref, route.selection_id, route.selection_version)
                == (self.bindings[5].consumer_ref, self.bindings[5].selection_id, self.bindings[5].selection_version)))
            for path in PATHS if path != "main"]
        self.qualification_plan = tuple(qualifications)
        self.endpoint_table = tuple((index, tuple(asdict(endpoint).items())) for index, endpoint in endpoints.items())
        self.alias_table = tuple((index, path, self.ports[index, path].resolve().transport_ref)
                                 for index in range(6) for path in PATHS)
        self.qualification_table = tuple((route.route_ref, route.path, self.requests.index(request),
            route.destination.transport_ref, type(self.ports[int(route.route_ref.split("_")[1]), route.path])
                is OwnedLoopbackHTTPTransport) for request, route in qualifications)
        self.expected_qualification_http = sum(row[-1] for row in self.qualification_table)
        steps = []
        for _, route in qualifications:
            index = int(route.route_ref.split("_")[1])
            if index in endpoints:
                steps.append(ResponseStep(endpoints[index].request_path.encode(), response_bytes(b'{"accepted":true}'),
                    b"Invented qualification"))
        for index, _ in roles:
            steps.append(ResponseStep(endpoints[index].request_path.encode(), response_bytes(self.outputs[index])))
        self.response_steps = tuple(steps)

    def data_counts(self):
        rows = self.receiver.rows[self.expected_qualification_http:]
        return [sum(row.get("path") == f"/owned-accountant-proof/v1/{role}".encode() and row["complete"]
                    for row in rows) if role else len(self.ports[index, "main"].sent)
                for index, role in enumerate(("p", None, "receiver", "child", None, "c"))]

    def prepare_source(self):
        self.qualifier = ContentPolicyRuntime(admission=self.admission, profiles=self.profiles,
            routes=self.routes, transports=self.transports, clock=self.clock, policy_revision=9)
        qualifications = self.qualification_plan
        for request, route in qualifications:
            port = self.ports[int(route.route_ref.split("_")[1]), route.path]
            content = self.qualifier.admit(request, b"Invented qualification")
            result = self.qualifier.dispatch(request, content, path=route.path, route_ref=route.route_ref)
            envelope = tuple(self.qualifier._envelopes.values())[-1].envelope
            self.qualifications.append((request, content, result, envelope))
            if self.retain is not None:
                self.retain("qualification", dict(route=route.route_ref, path=route.path,
                    envelope=asdict(envelope), result_sha256=hashlib.sha256(self.qualifier._content(result).payload).hexdigest()))
            if type(port) is not OwnedLoopbackHTTPTransport:
                port.sent.clear()
        self.receiver.wait(self.expected_qualification_http)
        assert len(self.qualifications) == len(self.qualification_plan)
        self.adapter = native.bind_controlled_main(self.context, selection=self.selection_ports[0],
            result_policy=self.result_port, producer_ref="producer_0", main_route_ref="route_0_main",
            profiles=self.profiles, routes=self.routes, transports=self.transports, clock=self.clock)
        with content_policy_scope(self.context):
            self.agent = AIAgent(controlled_main=self.adapter, provider="synthetic_native", api_mode="chat_completions",
                model="invented_model_0", session_id="invented_source_session", enabled_toolsets=[], quiet_mode=True,
                skip_memory=True, skip_context_files=True, skip_background_review=True)
        self.probe_input = episode.NativeIssuedCallEpisodeProbe(
            "Request one leaf call.", "Count blue squares.", "execution", ARGUMENTS)
        self.execution_input = episode.NativeIssuedCallEpisodeExecution(
            self.requests[1], self.selection_ports[1], "producer_1", "route_1_main",
            self.requests[2], self.selection_ports[2], "producer_2",
            self.requests[3], self.selection_ports[3], "producer_3", "route_2_delegate",
            "route_3_main", "invented_child_session", "child_execution")
        self.formatter_input = episode.NativeIssuedCallEpisodeFormatter(
            self.requests[4], self.selection_ports[4], "producer_4", "route_4_main")
        self.consumer_input = episode.NativeIssuedCallEpisodeConsumer(
            self.consumer_context, self.selection_ports[5], "producer_5", "route_5_main",
            self.selections[5].destination.model_ref, "invented_consumer_session")
        return self



def test_wire_plan_and_original_json_scripts():
    with OwnedReceiver() as receiver:
        world = LoopbackWorld(receiver)
        assert world.expected_qualification_http == len(world.response_steps) - 4
        assert len(world.qualification_table) == 27
        assert {route.path for route in world.routes} == set(PATHS)
        assert all(route.qualification_scope == "synthetic_only" and route.qualification_ref.startswith("synthetic_")
                   for route in world.routes)
        for index in (0, 3, 5):
            output = world.outputs[index]
            decoded = (native._tool_probe_response(output, world.selections[index].destination.model_ref, ARGUMENTS)
                if index == 0 else native._decode_response(output, world.selections[index].destination.model_ref))
            assert decoded["model"] == world.selections[index].destination.model_ref
        assert json.loads(world.outputs[2]) == {"accepted": True}
        assert all(len(step.response) <= 8192 + 65536 for step in world.response_steps)
        assert all(port.resolve() is world.selections[i].destination for (i, _), port in world.ports.items())
        assert receiver.rows == [] and receiver.thread is None


def test_genuine_public_loopback_http(tmp_path, monkeypatch, record, no_delegation_effects):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("model: invented_model\nmemory:\n  enabled: false\n")
    root = Path(__file__).resolve().parents[2]
    paths = {name: str(Path(sys.modules[name].__file__).resolve())
             for name in ("agent.native_main_payload", "run_agent", "agent.issued_call_episode",
                          "agent.conversation_loop", "agent.content_policy", "hermes_content_policy",
                          "agent.transports.owned_loopback_http", "tests.agent.owned_loopback_receiver",
                          "tests.agent.test_native_public_loopback_http")}
    assert all(Path(path).is_relative_to(root) for name, path in paths.items()
               if name != "hermes_content_policy")
    assert Path(sys.prefix).resolve() == Path(
        "/home/mersinvald/.local/share/orca/runtime/accountant/hpwa-p10.47/venv").resolve()
    record("loaded_paths", dict(paths=paths, interpreter=sys.executable, prefix=sys.prefix,
        sha256={name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in paths.items()}))
    record("phase", "source_preparation")
    receiver = OwnedReceiver()
    world = None
    try:
        world = LoopbackWorld(receiver)
        world.retain = record
        record("frozen_plan", dict(endpoints=world.endpoint_table, aliases=world.alias_table,
            qualifications=world.qualification_table, expected_http_qualifications=world.expected_qualification_http,
            expected_http_data=4, expected_http_total=len(world.response_steps)))
        receiver.start(world.response_steps)
        world.prepare_source()
        record("qualification_results", [dict(route=route[0], envelope=asdict(item[3]),
            result_sha256=hashlib.sha256(world.qualifier._content(item[2]).payload).hexdigest())
            for route, item in zip(world.qualification_table, world.qualifications)])
        record("receiver_qualification", receiver.wait(world.expected_qualification_http))
        record("phase", "source_prepared")
        source_session = world.agent.session_id
        plan = episode.NativeIssuedCallConversationPlan(world.adapter, world.probe_input,
            world.execution_input, world.formatter_input, world.consumer_input)
        started, previous, continuation_before = time.monotonic(), [], []

        def observe():
            owners = [owner for owner in native._issued_continuation_owners if owner.parent is world.adapter]
            owner = owners[0] if owners else None
            runtime = world.adapter._runtime
            if owner is not None and owner.phase == "paired" and not continuation_before:
                continuation_before.append((len(runtime._contents), len(runtime._envelopes), runtime._retained_bytes))
            boundary = (len(runtime._contents), len(runtime._envelopes),
                        None if owner is None else owner.phase, None if owner is None else owner.windows,
                        *(getattr(rows.get(world.adapter), "phase", None) for rows in (
                            native._tool_call_probes, native._issued_call_frames, native._issued_call_formats)))
            if previous == [boundary]:
                return
            previous[:] = [boundary]
            indices = {row.handle: i for i, row in enumerate(runtime._contents.values())}
            record("boundary", dict(elapsed=time.monotonic()-started, boundary=boundary,
                retained_bytes=runtime._retained_bytes,
                sends=world.data_counts(),
                graph=[dict(index=indices[row.handle], bytes=len(row.payload), hex=row.payload.hex(),
                    sha256=hashlib.sha256(row.payload).hexdigest(), parents=[indices[p] for p in row.parents],
                    binding=row.binding, profiles=row.profiles, path=row.path)
                    for row in runtime._contents.values()]))

        world.hooks["clock"] = observe
        record("phase", "public_entry")
        public = world.agent.run_conversation(plan.probe.user_message,
            system_message=plan.probe.system_message, task_id=plan.probe.task_id, controlled_issued_call=plan)
        owner, = [owner for owner in native._issued_continuation_owners if owner.parent is world.adapter]
        result = owner.records[0][0]
        completion = result.completion
        runtime = owner.runtime
        assert completion is result.completion is owner.completion
        assert current_context() is None and owner.phase == "released" and owner.windows == 6
        assert result.binding is owner.binding and result.pair is owner.pair
        assert result.request["model"] is world.selections[5].destination.model_ref
        probe = native._tool_call_probes[world.adapter]
        issued = native._issued_call_frames[world.adapter]
        formatter = native._issued_call_formats[world.adapter]
        assert result.tools is probe.request["tools"] is result.request["tools"]
        assert all(result.request["messages"][i] is probe.request["messages"][i] for i in range(2))
        assert all(result.request["messages"][i+2] is result.pair.messages[i] for i in range(2))
        assert set(result.request) == {"model", "messages", "tools", "stream", "tool_choice"}
        assert result.request["stream"] is False and result.request["tool_choice"] == "none"
        assert result.invocation.native_response is completion.response is owner.response
        assert result.invocation.response_bytes == world.outputs[5]
        assert completion.normalized is result.normalized and completion.usage is result.normalized.usage
        assert completion.final_response == "The leaf counted three blue squares."
        assert probe.closed and probe.constructed.closed
        assert issued.phase == formatter.phase == "released"
        assert issued.frame.child_adapter._constructed.closed
        source = result.pair.assistant_result.source_result
        formatted = result.pair.tool_result.source_result
        issued_result = formatted.source_result
        assert source.source_observation is issued_result.observation is issued_result.binding.observation
        assert source.source_observation is probe.records[0]
        assert source.response is native._assistant_source_owner(world.adapter).response
        assert issued_result is issued.result_anchors[0][0]
        assert formatted is formatter.records[0][0]
        assert issued_result.invocation is issued.frame.record
        assert issued_result.invocation.child_completion is not None
        assert issued_result.return_facts is issued_result.invocation.return_facts
        assert formatted.message["content"] is issued_result.aggregate
        assert all(row.profiles == ("restricted_documents",) for row in runtime._contents.values())
        assert runtime._poisoned is False and owner.control_exception is None
        assert owner.charge_after == (len(runtime._contents), len(runtime._envelopes), runtime._retained_bytes)
        assert len(owner.policy_observations) > 0
        assert all(route.expires_at > world.now for route in world.routes)
        assert len(world.routes) <= 128 and len(world.profiles) <= 32 and len(world.transports) <= 128
        assert all(len(row.payload) <= 262144 for row in runtime._contents.values())
        assert len(result.invocation.response_bytes) <= 65536
        data_rows = receiver.wait(len(world.response_steps))[world.expected_qualification_http:]
        envelopes = tuple(row.envelope for row in runtime._envelopes.values())
        assert len(data_rows) == len(envelopes) == 4
        for row, envelope, index in zip(data_rows, envelopes, (0, 2, 3, 5)):
            assert row["body"] == envelope.payload
            assert envelope.destination == world.selections[index].destination
            assert row["path"] == dict(world.endpoint_table)[index][3][1].encode()
            assert row["local"] == row["bound"] == receiver.bound
            assert row["peer"][0] == "127.0.0.1" and row["peer"][1] != receiver.bound[1]
            assert row["host"] == row["client_reported_peer"] == f"127.0.0.1:{receiver.bound[1]}".encode()
            assert row["complete"] and row["failure"] is None and row["response_attempts"] == 1
        assert all(row["response_accepted"] == len(step.response) for row, step in zip(receiver.rows, world.response_steps))
        assert all(not world.ports[i, "main"].sent for i in (1, 4))
        assert all(not world.ports[i, "main"]._busy for i in (0, 2, 3, 5))
        record("receiver_data", data_rows)
        record("dispatch_envelopes", [asdict(value) for value in envelopes])
        assert owner.selection.destination is owner.route.destination is world.selections[5].destination
        assert world.data_counts() == [1, 0, 1, 1, 0, 1]
        assert len(world.qualifications) == 27 and len(runtime._envelopes) == 4
        h, k, w, r, n = (owner.history_content, owner.request_content, owner.wire_content,
                          owner.response_content, owner.normalized_content)
        j, a = result.pair.message_content, result.binding.authorization_content
        for handle, parents in ((h, (j, a)), (k, (h,)), (w, (h, k)), (r, (w,)),
                                (n, (h, k, w, r)), (completion.completion_content, (j, a, h, k, w, r, n))):
            assert runtime._content(handle).parents == parents
        assert len(continuation_before) == 1
        assert len(runtime._contents) == continuation_before[0][0] + 6
        assert len(runtime._envelopes) == continuation_before[0][1] + 1
        assert runtime._retained_bytes == sum(len(row.payload) for row in runtime._contents.values())
        assert type(public) is dict and public["completed"] is True
        assert public["final_response"] is completion.final_response
        assert public["messages"] == json.loads(result.request_bytes)["messages"] + [
            dict(role="assistant", content=completion.final_response)]
        assert len(public["messages"]) == 5
        assert [row["role"] for row in public["messages"]] == ["system", "user", "assistant", "tool", "assistant"]
        assert all(public["messages"][i] is not result.request["messages"][i] for i in range(4))
        assert public["model"] is result.binding.selection.destination.model_ref
        assert public["model"] == "invented_model_5" and world.agent.model == "invented_model_0"
        assert public["session_id"] is source_session and source_session == "invented_source_session"
        assert result.agent.session_id == "invented_consumer_session" != source_session
        accepted_keys = json.loads((root / "docs/evidence/accountant-native-public-episode-20261124/pins.json").read_text())
        assert set(public) == set(accepted_keys["preservation"]["controlled_text_result_keys_exact"])
        assert all(public[key] is False for key in ("failed", "partial", "interrupted", "response_transformed", "response_previewed"))
        assert all(public[key] is None for key in ("last_reasoning", "pre_transform_response", "estimated_cost_usd", "cost_source", "service_tier"))
        assert all(public[key] == 0 for key in ("cache_read_tokens", "cache_write_tokens", "reasoning_tokens"))
        assert (public["provider"], public["base_url"], public["cost_status"], public["turn_exit_reason"]) == (
            "synthetic_native", "", "unknown", "text_response(finish_reason=stop)")
        assert public["api_calls"] == 2
        assert (public["prompt_tokens"], public["completion_tokens"], public["total_tokens"]) == (16, 10, 26)
        assert (public["input_tokens"], public["output_tokens"], public["last_prompt_tokens"]) == (16, 10, 11)
        assert len(json.dumps(public, sort_keys=True, ensure_ascii=True, allow_nan=False,
            separators=(",", ":")).encode()) <= native._MAX_NATIVE_RESULT
        assert world.agent.client is world.agent._primary_runtime is None
        assert result.agent.client is result.agent._primary_runtime is None
        record("public_result", public)
        observe()
        before = (len(runtime._contents), len(runtime._envelopes), runtime._retained_bytes)
        with pytest.raises(NativeContentPolicyDenied):
            world.agent.run_conversation(plan.probe.user_message, system_message=plan.probe.system_message,
                task_id=plan.probe.task_id, controlled_issued_call=plan)
        assert before == (len(runtime._contents), len(runtime._envelopes), runtime._retained_bytes)
        record("final", dict(charges=before, owner_phase=owner.phase, windows=owner.windows,
            elapsed=time.monotonic()-started, completion=completion.completion_bytes.decode(),
            request=result.request_bytes.decode(), policy_observations=owner.policy_observations))
    finally:
        record("receiver_before_cleanup", tuple(receiver.rows))
        try:
            receiver.close()
        finally:
            record("cleanup", dict(thread_alive=receiver.thread is not None and receiver.thread.is_alive(),
                listener_closed=receiver.listener.fileno() == -1, accepted_socket_live=receiver.active is not None,
                receiver_errors=[type(error).__name__ for error in receiver.errors]))
