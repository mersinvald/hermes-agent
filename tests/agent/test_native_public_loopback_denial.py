"""Prove one native delegation denial with an independent owned observer."""

from dataclasses import asdict, fields, replace
import hashlib
from pathlib import Path
import socket
import sys
import time

import pytest
from hermes_content_policy import ContentPolicyRuntime, Destination, PATHS, PolicyError

from agent import issued_call_episode as episode, native_main_payload as native
from agent.content_policy import NativeContentPolicyDenied, current_context
from agent.transports.owned_loopback_http import OwnedLoopbackEndpoint, OwnedLoopbackHTTPTransport
from tests.agent.owned_loopback_observer import ForbiddenReceiver
from tests.agent.owned_loopback_receiver import OwnedReceiver
from tests.agent.test_native_public_loopback_http import LoopbackWorld, _public_failure_metadata, record as record
from tests.agent.test_native_controlled_conversation import excluded as excluded
from tests.tools.test_native_controlled_delegation import no_delegation_effects as no_delegation_effects
from tests.tools.test_native_issued_call_execution import IssuedWorld


FORBIDDEN_ROUTE = "route_2_delegate_forbidden"
DENIAL = "controlled_native_issued_call_selected_expired_or_denied"


def build_forbidden_delegate_world(allowed_receiver: OwnedReceiver,
                                   forbidden_receiver: ForbiddenReceiver) -> LoopbackWorld:
    world = LoopbackWorld(allowed_receiver)
    originals = (world.requests, world.selection_ports, world.admission.current, world.result_port.current,
                 world.clock, world.alias_table, world.qualification_plan, world.profiles)
    baseline_routes = world.routes
    destination = Destination("invented_2_forbidden", "synthetic_native", "invented_model_2",
                              "synthetic_loopback_receiver", 1)
    endpoint = OwnedLoopbackEndpoint(destination.endpoint_ref, forbidden_receiver.bound[0],
        forbidden_receiver.bound[1], "/owned-accountant-proof/v1/receiver",
        destination.provider_ref, destination.model_ref, destination.revision)
    port = OwnedLoopbackHTTPTransport(destination, endpoint=endpoint)
    allowed = next(route for route in baseline_routes if route.route_ref == "route_2_delegate")
    forbidden = replace(allowed, route_ref=FORBIDDEN_ROUTE, destination=destination,
                        qualification_ref="synthetic_2_delegate_forbidden")
    world.ports[2, "forbidden_delegate"] = port
    world.routes = (*baseline_routes, forbidden)
    world.transports = tuple((item.resolve().transport_ref, item) for item in dict.fromkeys(world.ports.values()))
    world.forbidden_metadata = (forbidden, endpoint, port)
    assert (world.requests, world.selection_ports, world.admission.current, world.result_port.current,
            world.clock, world.alias_table, world.qualification_plan, world.profiles) == originals
    assert all(actual is original for actual, original in zip(world.routes, baseline_routes))
    return world


def _assert_plan(world, allowed_receiver, forbidden_receiver):
    forbidden, endpoint, port = world.forbidden_metadata
    baseline = world.routes[:-1]
    assert type(world) is LoopbackWorld and len(world.requests) == 6
    assert len({id(request) for request in world.requests}) == 6
    assert world.admission.current.__func__ is IssuedWorld.current
    assert world.result_port.current.__func__ is IssuedWorld.produced
    assert world.clock.__func__ is IssuedWorld.time
    assert all(selection.current.__func__ is IssuedWorld.selection for selection in world.selection_ports)
    assert world.now == 10 and all(route.expires_at == 100 for route in world.routes)
    assert len(baseline) == 20 and len(world.profiles) == 1
    profile, = world.profiles
    assert profile.profile_ref == "restricted_documents" and profile.required_paths == PATHS
    assert profile.allowed_routes == tuple(route.route_ref for route in baseline)
    assert FORBIDDEN_ROUTE not in profile.allowed_routes
    assert all(holder.profiles == ("restricted_documents",) for holder in world.bindings)
    assert all(policy.profiles == ("restricted_documents",) for policy in world.policies)
    for holder in world.bindings:
        for path in PATHS:
            assert any(route.path == path and route.route_ref in profile.allowed_routes
                and (route.consumer_ref, route.selection_id, route.selection_version)
                    == (holder.consumer_ref, holder.selection_id, holder.selection_version) for route in baseline)
    assert all(world.ports[index, path].resolve() is world.selections[index].destination
               for index in range(6) for path in PATHS)
    assert world.alias_table == tuple((index, path, world.ports[index, path].resolve().transport_ref)
                                      for index in range(6) for path in PATHS)
    assert len(world.qualification_plan) == 27 and world.expected_qualification_http == 25
    assert sum(not row[-1] for row in world.qualification_table) == 2
    assert all(route is not forbidden for _, route in world.qualification_plan)
    assert world.ports[2, "forbidden_delegate"] is port
    assert world.admission.current(port) is world.bindings[2]
    assert world.selection_ports[2].current(world.requests[2]) is world.selections[2]
    assert world.selections[2].destination is world.ports[2, "main"].resolve()
    allowed = next(route for route in baseline if route.route_ref == "route_2_delegate")
    assert allowed.destination is world.selections[2].destination
    assert forbidden == replace(allowed, route_ref=FORBIDDEN_ROUTE, destination=port.resolve(),
                                qualification_ref="synthetic_2_delegate_forbidden")
    assert forbidden.destination is port.resolve() is port.destination
    assert forbidden.destination == Destination("invented_2_forbidden", "synthetic_native", "invented_model_2",
                                               "synthetic_loopback_receiver", 1)
    assert endpoint is port._endpoint and endpoint.port == forbidden_receiver.bound[1]
    assert len(type(port).__slots__) == 6 and type(port) is OwnedLoopbackHTTPTransport
    assert dict(world.transports)[forbidden.destination.transport_ref] is port
    assert allowed_receiver.listener.fileno() != forbidden_receiver.listener.fileno()
    assert allowed_receiver.bound[0] == forbidden_receiver.bound[0] == "127.0.0.1"
    assert 0 < allowed_receiver.bound[1] < 65536 and 0 < forbidden_receiver.bound[1] < 65536
    assert allowed_receiver.bound[1] != forbidden_receiver.bound[1]


def _observer_receipt(receiver, observation):
    return dict(observation=asdict(observation), finish_requested_at=receiver.finish_requested_at,
        drain_at=receiver.drain_at, cutoff_at=receiver.cutoff_at,
        cleanup_started_at=receiver.cleanup_started_at, cleanup_finished_at=receiver.cleanup_finished_at)


def _assert_observer(receiver, observation):
    assert observation.errors == ()
    assert observation.thread_stopped and observation.listener_closed and observation.sockets_closed
    assert observation.polls > 0
    assert (observation.started_at <= observation.ready_at <= observation.last_poll_at
            <= receiver.drain_at <= receiver.cutoff_at < observation.deadline_at)
    assert receiver.cutoff_at >= receiver.finish_requested_at + 0.5
    assert (receiver.cutoff_at <= receiver.cleanup_started_at <= receiver.cleanup_finished_at
            <= observation.finished_at)
    assert receiver.cleanup_finished_at - receiver.cleanup_started_at <= 5.0
    if receiver.thread_errors:
        raise receiver.thread_errors[0]


@pytest.mark.parametrize("kind", ("empty_connection", "partial_header"))
def test_observer_records_empty_and_partial_connection(kind, record):
    receiver = ForbiddenReceiver()
    try:
        receiver.start()
        assert receiver.ready.is_set() and receiver.thread.is_alive()
        raw = b"" if kind == "empty_connection" else b"POST /owned-accountant-proof/v1/receiver HTTP/1.1\r\nHost:"
        with socket.create_connection(receiver.bound, timeout=2.0) as connection:
            if raw:
                connection.sendall(raw)
        observation = receiver.finish()
        record("observer_control", dict(kind=kind, **_observer_receipt(receiver, observation)))
        _assert_observer(receiver, observation)
        row, = observation.connections
        assert row.order == 0 and row.raw == raw and not row.headers_complete and not row.request_complete
        assert row.local == receiver.bound and row.peer[0] == "127.0.0.1"
        assert row.error_code == ("eof_empty" if not raw else "eof_partial_headers")
        assert observation.ready_at <= row.accepted_at <= row.closed_at <= receiver.cutoff_at
        receiver.close()
        assert receiver.finish() is observation
    finally:
        receiver.close()


def test_forbidden_route_plan_preserves_coverage(record):
    allowed, forbidden = OwnedReceiver(), ForbiddenReceiver()
    try:
        world = build_forbidden_delegate_world(allowed, forbidden)
        _assert_plan(world, allowed, forbidden)
        assert not hasattr(world, "adapter") and not hasattr(world, "qualifier")
        assert not world.qualifications and allowed.thread is None and forbidden.thread is None
        record("inert_plan", dict(baseline_routes=len(world.routes)-1, qualifications=len(world.qualification_plan),
            aliases=world.alias_table, forbidden=asdict(world.forbidden_metadata[0]),
            endpoint=asdict(world.forbidden_metadata[1]), profiles=[asdict(row) for row in world.profiles],
            genuine_native_history=False))
    finally:
        try:
            allowed.close()
        finally:
            forbidden.close()


def test_shared_forbidden_delegate_pre_send(record):
    allowed, forbidden = OwnedReceiver(), ForbiddenReceiver()
    try:
        world = build_forbidden_delegate_world(allowed, forbidden)
        _assert_plan(world, allowed, forbidden)
        runtime = ContentPolicyRuntime(admission=world.admission, profiles=world.profiles, routes=world.routes,
            transports=world.transports, clock=world.clock, policy_revision=9)
        content = runtime.admit(world.requests[1], b"Invented forbidden delegate control")
        before = (len(runtime._contents), len(runtime._envelopes), runtime._retained_bytes)
        result = None
        with pytest.raises(PolicyError) as caught:
            result = runtime.dispatch(world.requests[1], content, path="delegate", route_ref=FORBIDDEN_ROUTE)
        assert type(caught.value) is PolicyError and caught.value.args == ("route_forbidden",)
        after = (len(runtime._contents), len(runtime._envelopes), runtime._retained_bytes)
        assert before == after and not runtime._envelopes and result is None
        assert len(runtime._contents) == 1 and runtime._content(content).profiles == ("restricted_documents",)
        assert not world.qualifications and not hasattr(world, "adapter")
        record("shared_control", dict(error_class="PolicyError", reason="route_forbidden", before=before,
            after=after, no_dispatch_envelope=True, no_result=True,
            admission_payload_sha256=hashlib.sha256(runtime._content(content).payload).hexdigest(),
            receiver_observation="UNKNOWN: listeners did not start"))
    finally:
        try:
            allowed.close()
        finally:
            forbidden.close()


def _charges(runtime):
    return len(runtime._contents), len(runtime._envelopes), runtime._retained_bytes


def _graph(runtime):
    indices = {row.handle: index for index, row in enumerate(runtime._contents.values())}
    return [dict(index=indices[row.handle], raw=row.payload, bytes=len(row.payload),
        sha256=hashlib.sha256(row.payload).hexdigest(), parents=[indices[parent] for parent in row.parents],
        binding=row.binding, profiles=row.profiles, path=row.path) for row in runtime._contents.values()]


def test_genuine_public_forbidden_delegate(tmp_path, monkeypatch, record, no_delegation_effects):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("model: invented_model\nmemory:\n  enabled: false\n")
    assert Path(sys.prefix).resolve() == Path(
        "/home/mersinvald/.local/share/orca/runtime/accountant/hpwa-p10.47/venv").resolve()
    allowed, forbidden = OwnedReceiver(), ForbiddenReceiver()
    world, observation = None, None
    public = None
    bind_before = []
    last_boundary = []
    started = time.monotonic()
    try:
        world = build_forbidden_delegate_world(allowed, forbidden)
        _assert_plan(world, allowed, forbidden)
        world.retain = record
        record("operator_freeze", dict(endpoints=world.endpoint_table, aliases=world.alias_table,
            qualifications=world.qualification_table, profiles=[asdict(row) for row in world.profiles],
            routes=[asdict(row) for row in world.routes], forbidden_endpoint=asdict(world.forbidden_metadata[1]),
            bindings=[asdict(row) for row in world.bindings], policies=[asdict(row) for row in world.policies],
            clock=world.now, expected_a_http=26, expected_f_connections=0))

        def observe():
            if not hasattr(world, "adapter"):
                return
            runtime = world.adapter._runtime
            issued = native._issued_call_frames.get(world.adapter)
            probe = native._tool_call_probes.get(world.adapter)
            boundary = (_charges(runtime), None if issued is None else issued.phase,
                        None if probe is None else probe.closed)
            if issued is not None and issued.phase == "binding" and not bind_before:
                bind_before.append(_charges(runtime))
                record("bind_before", dict(charges=bind_before[0], stage_index=issued.stage_index,
                    p_closed=probe.closed, forbidden_selected=issued.routes[2] is world.forbidden_metadata[0]))
            if last_boundary == [boundary]:
                return
            last_boundary[:] = [boundary]
            record("boundary", dict(elapsed=time.monotonic()-started, charges=_charges(runtime),
                issued_phase=None if issued is None else issued.phase,
                p_closed=None if probe is None else probe.closed, graph=_graph(runtime)))

        world.hooks["clock"] = observe
        allowed.start(world.response_steps[:world.expected_qualification_http + 1])
        forbidden.start()
        record("observer_ready", dict(bound=forbidden.bound, started_at=forbidden.started_at,
            ready_at=forbidden.ready_at, deadline_at=forbidden.deadline_at, polls=forbidden.polls,
            thread_alive=forbidden.thread.is_alive()))
        record("phase", "source_preparation")
        source_started = time.monotonic()
        world.prepare_source()
        record("source_preparation_seconds", time.monotonic() - source_started)
        record("phase", "source_prepared")
        assert len(world.qualifications) == 27
        assert len(allowed.wait(25)) == 25
        assert len(world.qualifier._envelopes) == 27
        record("qualification_results", dict(count=len(world.qualifications), http=25, non_http=2,
            envelopes=[asdict(row.envelope) for row in world.qualifier._envelopes.values()],
            charges=_charges(world.qualifier)))
        execution = replace(world.execution_input, delegate_route_ref=FORBIDDEN_ROUTE)
        assert all(getattr(execution, field.name) is getattr(world.execution_input, field.name)
                   for field in fields(execution) if field.name != "delegate_route_ref")
        plan = episode.NativeIssuedCallConversationPlan(world.adapter, world.probe_input,
            execution, world.formatter_input, world.consumer_input)
        record("public_plan", dict(delegate_route_ref=execution.delegate_route_ref,
            only_delegate_route_changed=True, execution_is_original_type=type(execution) is type(world.execution_input)))
        record("phase", "public_entry")
        try:
            public = world.agent.run_conversation(plan.probe.user_message,
                system_message=plan.probe.system_message, task_id=plan.probe.task_id, controlled_issued_call=plan)
        except BaseException as error:
            record("public_exception", _public_failure_metadata(error))
            if type(error) is not NativeContentPolicyDenied or error.args != (DENIAL,):
                raise
            checkpoint, preserved = False, False
            trace = error.__traceback__
            while trace is not None:
                if trace.tb_frame.f_code is native._IssuedCallExecution.checkpoint.__code__:
                    checkpoint = trace.tb_lineno == 7463
                if trace.tb_frame.f_code is native._issued_failure.__code__:
                    preserved = trace.tb_frame.f_locals["error"] is error
                trace = trace.tb_next
            record("selected_denial", dict(error_class=type(error).__name__, reason=error.args[0],
                checkpoint_line=7463 if checkpoint else None, original_exception_identity=preserved))
            assert checkpoint and preserved
        else:
            pytest.fail("Expected native delegation denial")
        observe()
        runtime = world.adapter._runtime
        issued = native._issued_call_frames[world.adapter]
        probe = native._tool_call_probes[world.adapter]
        continuation, = [owner for owner in native._issued_continuation_owners if owner.parent is world.adapter]
        record("terminal", dict(charges=_charges(runtime), graph=_graph(runtime),
            envelopes=[asdict(row.envelope) for row in runtime._envelopes.values()],
            owner_phase=issued.phase, stage_index=issued.stage_index, p_closed=probe.closed,
            continuation_phase=continuation.phase, continuation_windows=continuation.windows,
            public_result=public, sends=world.data_counts(), p_send_count=world.adapter.send_count,
            profile_union=issued.profile_union, clock=world.now, bind_before=bind_before,
            no_e_adapter=issued.adapter is None, no_e_frame=issued.frame is None,
            authorization_content_absent=issued.authorization_content is None,
            issued_content_absent=issued.issued_content is None,
            continuation_records_empty=not continuation.records,
            continuation_completion_absent=continuation.completion is None))
        assert current_context() is None and issued.phase == "denied"
        assert issued.adapter is issued.agent is issued.context is issued.frame is None
        assert issued.binding is issued.registered_frame is None and not issued.result_anchors
        assert issued.authorization_content is issued.authorization_bytes is issued.issued_content is issued.issued_bytes is None
        assert issued.result_content is issued.result_bytes is issued.aggregate is None
        assert issued.control_exception is None
        assert issued.routes[2] is world.forbidden_metadata[0]
        assert issued.receiver_port is world.ports[2, "forbidden_delegate"]
        assert issued.originals[2][0] is world.bindings[2] and issued.originals[2][1] is world.selections[2]
        assert issued.profile_union == ("restricted_documents",)
        assert bind_before and bind_before[0] == _charges(runtime)
        assert probe.closed and probe.constructed.closed and world.adapter.send_count == 1
        assert world.adapter not in native._issued_call_formats
        assert not any(frame.parent is world.adapter for frame in native._delegate_frames.values())
        assert not continuation.records and continuation.completion is None and public is None
        assert continuation.phase == "armed" and not continuation.attempted
        assert continuation.request_content is continuation.wire_content is continuation.invocation is None
        assert continuation.adapter.send_count == 0
        assert all(row.profiles == ("restricted_documents",) for row in runtime._contents.values())
        assert world.now == 10 and all(route.expires_at == 100 for route in world.routes)
        rows = allowed.wait(26)
        envelopes = tuple(row.envelope for row in runtime._envelopes.values())
        envelope, = envelopes
        row, = rows[25:]
        assert envelope.path == "main" and envelope.destination == world.selections[0].destination
        assert row["body"] == envelope.payload and row["path"] == b"/owned-accountant-proof/v1/p"
        assert len(rows) == 26 and all(item["complete"] and item["failure"] is None for item in rows)
        assert all(item["response_accepted"] == len(step.response) for item, step in zip(rows, allowed.steps))
        assert all(not world.ports[index, "main"].sent for index in (1, 4))
        assert world.data_counts() == [1, 0, 0, 0, 0, 0]
        record("p_wire_match", dict(envelope=asdict(envelope), receiver=row, body_equal=True, actual_a_total=len(rows)))
        observation = forbidden.finish()
        record("forbidden_final", _observer_receipt(forbidden, observation))
        _assert_observer(forbidden, observation)
        assert observation.connections == ()
        record("phase", "denial_verified")
    finally:
        record("receiver_before_cleanup", tuple(allowed.rows))
        try:
            if observation is None:
                observation = forbidden.finish()
                record("forbidden_partial_final", _observer_receipt(forbidden, observation))
        finally:
            try:
                forbidden.close()
                if forbidden.thread_errors:
                    raise forbidden.thread_errors[0]
            finally:
                try:
                    allowed.close()
                finally:
                    record("cleanup", dict(a_thread_alive=allowed.thread is not None and allowed.thread.is_alive(),
                        a_listener_closed=allowed.listener.fileno() == -1, a_socket_live=allowed.active is not None,
                        a_errors=[type(error).__name__ for error in allowed.errors],
                        f_thread_alive=forbidden.thread is not None and forbidden.thread.is_alive(),
                        f_listener_closed=forbidden.listener.fileno() == -1,
                        elapsed=time.monotonic()-started))
