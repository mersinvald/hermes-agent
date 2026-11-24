"""Check current fallback expiry after an actual owned HTTP main failure."""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace

import pytest
from hermes_content_policy import ContentPolicyRuntime, FailedDispatch, PATHS, PolicyError

from agent import native_main_payload as native
from agent.content_policy import NativeContentPolicyDenied
from agent.transports.owned_loopback_http import OwnedLoopbackEndpoint, OwnedLoopbackHTTPTransport
from tests.agent.owned_loopback_receiver import OwnedReceiver, ResponseStep, response_bytes
from tests.agent.test_native_controlled_conversation import (
    compression_excluded as compression_excluded,
    construct,
    excluded as excluded,
)
from tests.agent.test_native_controlled_fallback import FallbackWorld, bind_fallback, fixed_turn
from tests.agent.test_native_main_payload import (
    ControlledPort,
    World,
    caller_boundary as caller_boundary,
)


ACK = b'{"accepted":true}'
QUALIFICATION = b"Invented qualification"


@dataclass(frozen=True, slots=True)
class MainSendObservation:
    envelope: object
    output: bytes | None
    error: BaseException | None
    transport_receipt: tuple | None
    started_at: float
    finished_at: float


class RetainedMainPort:
    __slots__ = ("core", "destination", "_resolve", "_send", "observations")

    def __init__(self, core: OwnedLoopbackHTTPTransport):
        object.__setattr__(self, "core", core)
        object.__setattr__(self, "destination", core.destination)
        object.__setattr__(self, "_resolve", core.resolve)
        object.__setattr__(self, "_send", core.send)
        object.__setattr__(self, "observations", ())

    def __setattr__(self, name, value):
        raise AttributeError("Fixed main port")

    def __delattr__(self, name):
        raise AttributeError("Fixed main port")

    def resolve(self):
        return self._resolve()

    def send(self, envelope) -> bytes:
        assert len(self.observations) < 2
        started = time.monotonic()
        try:
            output = self._send(envelope)
        except BaseException as error:
            observation = MainSendObservation(envelope, None, error,
                getattr(error, "owned_loopback_receipt", None), started, time.monotonic())
            object.__setattr__(self, "observations", (*self.observations, observation))
            raise
        observation = MainSendObservation(envelope, output, None, None, started, time.monotonic())
        object.__setattr__(self, "observations", (*self.observations, observation))
        return output


@pytest.fixture
def record(request, record_property):
    def encode(value):
        if type(value) is bytes:
            return {"bytes_hex": value.hex()}
        raise TypeError("Unsupported receipt field")

    def retain(name, value):
        encoded = json.dumps(dict(nodeid=request.node.nodeid, property=name, value=value), default=encode)
        record_property(name, json.dumps(value, default=encode))
        path = Path(request.config.option.xmlpath).with_suffix(".receipts.jsonl")
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, "a") as output:
            output.write(encoded + "\n")
            output.flush()
            os.fsync(output.fileno())
    return retain


@pytest.fixture
def ordinary_effects(compression_excluded, caller_boundary, monkeypatch):
    from agent import auxiliary_client as auxiliary, chat_completion_helpers as chat
    from concurrent import futures
    import httpx

    for module, names in (
        (auxiliary, ("resolve_provider_client", "_get_cached_client", "_acquire_sync_aux_semaphore", "_call_llm_impl")),
        (chat, ("_reset_stale_streak", "_record_interrupted_provider_wait", "rewrite_prompt_model_identity")),
        (futures, ("ThreadPoolExecutor", "ProcessPoolExecutor")),
        (httpx.Client, ("send",)),
        (httpx.AsyncClient, ("send",)),
    ):
        for name in names:
            monkeypatch.setattr(module, name, compression_excluded)
    monkeypatch.setattr(chat, "time", SimpleNamespace(monotonic=compression_excluded, sleep=compression_excluded))


def build_world(a, b):
    world = FallbackWorld()
    world.initial_originals = (world.request, world.fallback_request, world.binding, world.fallback_binding,
        world.policy, world.fallback_policy, world.profiles, world.admission.current,
        world.selection_port.current, world.fallback_selection_port.current, world.result_port.current)
    replacements = {}
    endpoints = []
    for key, role, model, receiver in (
        ("main", "p", "invented_model_0", a),
        ("fallback_capability_fallback", "c", "invented_model_5", b),
    ):
        old = world.ports[key]
        destination = replace(old.destination, endpoint_ref="synthetic_loopback_" + role, model_ref=model)
        endpoint = OwnedLoopbackEndpoint(destination.endpoint_ref, receiver.bound[0], receiver.bound[1],
            "/owned-accountant-proof/v1/" + role, destination.provider_ref, destination.model_ref, destination.revision)
        core = OwnedLoopbackHTTPTransport(destination, endpoint=endpoint)
        port = RetainedMainPort(core) if role == "p" else core
        world.ports[key] = port
        _, admission = world.receiver_bindings.pop(id(old))
        world.receiver_bindings[id(port)] = (port, admission)
        replacements["route_main" if role == "p" else "fallback_fallback"] = destination
        if role == "p":
            world.selection = replace(world.selection, destination=destination)
        else:
            world.fallback_ports["fallback"] = port
            world.fallback_selection = replace(world.fallback_selection, destination=destination)
        endpoints.append(endpoint)
    world.routes = tuple(replace(route, expires_at=100 if route.route_ref == "fallback_fallback" else 1000,
        **(dict(destination=replacements[route.route_ref], qualification_ref="synthetic_loopback_" +
            ("main" if route.route_ref == "route_main" else "fallback"))
            if route.route_ref in replacements else {})) for route in world.routes)
    world.transports = tuple((port.destination.transport_ref, port) for port in world.ports.values())
    world.endpoints = tuple(endpoints)
    world.qualifications = []
    return world


def assert_plan(world, a, b):
    assert type(world) is FallbackWorld and world.now == 10
    assert len(world.routes) == 2 * len(PATHS) == 16
    assert len(world.ports) == 16 and sum(type(port) is ControlledPort for port in world.ports.values()) == 14
    assert world.profiles is world.initial_originals[6]
    assert (world.request, world.fallback_request, world.binding, world.fallback_binding,
        world.policy, world.fallback_policy) == world.initial_originals[:6]
    assert world.admission.current.__func__ is FallbackWorld.current
    assert world.selection_port.current.__func__ is World.selected
    assert world.fallback_selection_port.current.__func__ is FallbackWorld.fallback_selected
    assert world.result_port.current.__func__ is FallbackWorld.produced
    assert (world.admission.current, world.selection_port.current, world.fallback_selection_port.current,
            world.result_port.current) == world.initial_originals[7:]
    assert world.profiles[0].profile_ref == "restricted_documents"
    assert world.profiles[0].required_paths == PATHS
    assert world.profiles[0].allowed_routes == tuple(route.route_ref for route in world.routes)
    assert world.binding.profiles == world.fallback_binding.profiles == ("restricted_documents",)
    assert world.policy.profiles == world.fallback_policy.profiles == ("restricted_documents",)
    for binding in (world.binding, world.fallback_binding):
        routes = tuple(route for route in world.routes if route.selection_id == binding.selection_id)
        assert len(routes) == len(PATHS) and {route.path for route in routes} == set(PATHS)
        assert all((route.selection_version, route.consumer_ref) == (binding.selection_version, binding.consumer_ref)
                   for route in routes)
    assert [route.route_ref for route in world.routes if route.expires_at == 100] == ["fallback_fallback"]
    assert all(route.expires_at == 1000 for route in world.routes if route.route_ref != "fallback_fallback")
    assert all(route.qualification_scope == "synthetic_only" for route in world.routes)
    assert a.bound[0] == b.bound[0] == "127.0.0.1" and a.bound[1] != b.bound[1]
    for receiver, endpoint, port, selected in zip((a, b), world.endpoints,
            (world.ports["main"], world.fallback_ports["fallback"]), (world.selection, world.fallback_selection)):
        assert port.resolve() is port.destination is selected.destination
        assert endpoint.port == receiver.bound[1] and endpoint.address == receiver.bound[0]
        assert dict(world.transports)[selected.destination.transport_ref] is port
    assert not hasattr(world.ports["main"], "sent")


def qualify(world, record):
    runtime = ContentPolicyRuntime(admission=world.admission, profiles=world.profiles,
        routes=world.routes, transports=world.transports, clock=lambda: world.now, policy_revision=9)
    world.qualifier = runtime
    for route in world.routes:
        request = world.request if route.selection_id == world.binding.selection_id else world.fallback_request
        source = runtime.admit(request, QUALIFICATION)
        result = runtime.dispatch(request, source, path=route.path, route_ref=route.route_ref)
        envelope = tuple(runtime._envelopes.values())[-1].envelope
        world.qualifications.append((request, source, result, envelope))
        record("qualification", dict(route=route.route_ref, path=route.path, envelope=asdict(envelope),
            result=runtime._contents[id(result)].payload, charges=charges(runtime)))
    for port in world.ports.values():
        if type(port) is ControlledPort:
            assert len(port.sent) == 1
            port.sent.clear()
    assert len(world.qualifications) == len(runtime._envelopes) == 16
    return runtime


def charges(runtime):
    return len(runtime._contents), len(runtime._envelopes), runtime._retained_bytes


def graph(runtime):
    indices = {row.handle: index for index, row in enumerate(runtime._contents.values())}
    return [dict(index=indices[row.handle], raw=row.payload, bytes=len(row.payload),
        sha256=hashlib.sha256(row.payload).hexdigest(), parents=[indices[parent] for parent in row.parents],
        binding=row.binding, profiles=row.profiles, path=row.path) for row in runtime._contents.values()]


def make_trigger(world, b, record):
    retained = {}

    def trigger():
        adapter = world.adapter
        if retained or adapter is None or adapter._fallback_phase != "authenticate" or adapter._fallback_frame is None:
            return
        failed, = adapter.failed_main_attempts
        frame = adapter._fallback_frame
        assert frame.failed_main is failed and frame.binding is adapter._fallback_binding
        assert adapter._fallback_phase == "authenticate" and frame.activated is False
        assert frame.source_content is None and frame.invocation is None
        entry = adapter._runtime._failures[id(failed.shared_failure)]
        assert entry.token is failed.shared_failure and entry.facts is failed.failure_facts
        assert entry.request_handle is world.request
        assert failed.send_observation.exception_type is TimeoutError
        assert type(failed.shared_failure) is FailedDispatch and type(frame.token) is native.NativeFallbackInput
        assert frame.token is not failed.shared_failure and world.now == 10
        mark = b.mark_window()
        retained.update(failed=failed, frame=frame, entry=entry, mark=mark,
                        graph=graph(adapter._runtime), charges=charges(adapter._runtime), phase=adapter._fallback_phase)
        record("trigger", dict(mark=asdict(mark), graph=retained["graph"], charges=retained["charges"],
            phase=retained["phase"], before=world.now, next_time=100, shared_token_is_entry=True,
            native_token_distinct=True, facts_is_entry=True, request_is_original=True,
            frame_is_original=True, source_content=None, invocation=None, activated=False))
        world.now = 100
    return trigger, retained


def wire_qualification(receiver, role):
    destination = native.shared_policy().Destination("invented_" + role, "synthetic_native",
        "invented_model_0" if role == "p" else "invented_model_5", "synthetic_loopback_" + role, 1)
    endpoint = OwnedLoopbackEndpoint(destination.endpoint_ref, receiver.bound[0], receiver.bound[1],
        "/owned-accountant-proof/v1/" + role, destination.provider_ref, destination.model_ref, destination.revision)
    from hermes_content_policy import DispatchEnvelope, ExecutionBinding
    binding = ExecutionBinding("owner", "execution", "lineage", "document_specialist", 7, 9, "selection", 3, ())
    envelope = DispatchEnvelope(QUALIFICATION, binding, (), "document_specialist", "main", destination)
    port = OwnedLoopbackHTTPTransport(destination, endpoint=endpoint)
    receiver.start((ResponseStep(endpoint.request_path.encode(), response_bytes(ACK), QUALIFICATION),))
    assert port.send(envelope) == ACK
    receiver.wait(1)
    return envelope


def assert_receipt(receipt, *, errors=False):
    assert receipt.thread_stopped and receipt.listener_closed and receipt.sockets_closed
    assert receipt.ready_at < receipt.finish_requested_at <= receipt.drain_at <= receipt.cutoff_at <= receipt.finished_at
    assert receipt.cutoff_at >= receipt.finish_requested_at + 0.5
    assert receipt.cutoff_at < receipt.deadline_at and receipt.polls > 0
    assert all(row.settled and row.accepted_at <= row.closed_at <= receipt.cutoff_at for row in receipt.connections)
    assert sum(row.raw_length for row in receipt.connections) <= 2113544
    assert all(row.raw_length == len(row.raw) <= 264193 for row in receipt.connections)
    if not errors:
        assert receipt.errors == ()
    if receipt.mark is not None:
        assert receipt.mark.listener_id == receipt.listener_id and receipt.mark.bound == receipt.bound
        assert receipt.mark.active_count == 0 and receipt.polls > receipt.mark.polls


def test_observation_qualification_mark_drain(record):
    receiver = OwnedReceiver(observation_seconds=590.0)
    try:
        envelope = wire_qualification(receiver, "c")
        listener, descriptor, bound = receiver.listener, receiver.listener.fileno(), receiver.bound
        mark = receiver.mark_window()
        receipt = receiver.finish()
        record("control", asdict(receipt))
        assert_receipt(receipt)
        assert mark.row_count == 1 and receipt.connections[0].body == envelope.payload
        assert receipt.connections[0].response == response_bytes(ACK)
        assert receipt.connections[0].listener_fd == descriptor
        assert receiver.listener is listener and receipt.bound == bound and receipt.listener_id == id(listener)
        assert receipt.connections[mark.row_count:] == ()
        receiver.close()
        assert receiver.finish() is receipt
    finally:
        receiver.close()


@pytest.mark.parametrize("kind", ("empty", "partial"))
def test_marked_window_connection(kind, record):
    receiver = OwnedReceiver(observation_seconds=590.0)
    try:
        wire_qualification(receiver, "c")
        mark = receiver.mark_window()
        raw = b"" if kind == "empty" else b"POST /owned-accountant-proof/v1/c HTTP/1.1\r\nHost:"
        with socket.create_connection(receiver.bound, timeout=2.0) as connection:
            if raw:
                connection.sendall(raw)
        receipt = receiver.finish()
        record("control", dict(kind=kind, receipt=asdict(receipt)))
        assert_receipt(receipt, errors=True)
        assert receipt.errors
        prior, observed = receipt.connections
        assert prior.complete and prior.body == QUALIFICATION
        assert mark.row_count == 1 and observed.accepted_at >= mark.started_at
        assert observed.raw == raw and not observed.complete and observed.status == "error"
        assert observed.failure == ("eof_empty" if kind == "empty" else "eof_partial_request")
        assert observed.local == receipt.bound and observed.peer[0] == "127.0.0.1"
    finally:
        receiver.close()


def test_inert_initial_plan_and_trigger_guard(record):
    a, b = OwnedReceiver(observation_seconds=590.0), OwnedReceiver(observation_seconds=590.0)
    try:
        world = build_world(a, b)
        assert_plan(world, a, b)
        hook, trigger = make_trigger(world, b, record)
        world.fallback_selection_hook = hook
        for _ in range(3):
            world.fallback_selected(world.fallback_request)
        assert not trigger and world.now == 10 and b._mark is None
        assert world.adapter is None and not world.qualifications and a.thread is b.thread is None
        record("plan", dict(routes=[asdict(route) for route in world.routes],
            profiles=[asdict(profile) for profile in world.profiles], endpoints=[asdict(ep) for ep in world.endpoints],
            aliases=[(key, port.destination.transport_ref) for key, port in world.ports.items()],
            genuine_native_history=False, trigger_inert=True))
    finally:
        a.close()
        b.close()


def test_shared_expired_fallback_control(record):
    world = FallbackWorld()
    world.routes = tuple(replace(route, expires_at=100 if route.route_ref == "fallback_fallback" else 1000)
                         for route in world.routes)
    runtime = ContentPolicyRuntime(admission=world.admission, profiles=world.profiles, routes=world.routes,
        transports=tuple((port.destination.transport_ref, port) for port in world.ports.values()),
        clock=lambda: world.now, policy_revision=9)
    source = runtime.admit(world.fallback_request, QUALIFICATION)
    before = charges(runtime)
    world.now = 100
    result = None
    with pytest.raises(PolicyError) as caught:
        result = runtime.dispatch(world.fallback_request, source, path="fallback", route_ref="fallback_fallback")
    after = charges(runtime)
    record("shared_control", dict(before=before, after=after, reason=caught.value.args[0],
        graph=graph(runtime), no_envelope=not runtime._envelopes, no_result=result is None,
        native_post_failure_claim=False, receiver_observation="UNKNOWN: no listeners"))
    assert type(caught.value) is PolicyError and caught.value.args == ("path_coverage_missing",)
    assert before == after and len(runtime._contents) == 1 and not runtime._envelopes and result is None
    assert runtime._contents[id(source)].payload == QUALIFICATION
    assert not any(port.sent for port in world.ports.values())


def exception_chain(error):
    rows, seen = [], set()
    while error is not None and id(error) not in seen and len(rows) < 12:
        seen.add(id(error))
        frames = []
        traceback = error.__traceback__
        while traceback is not None:
            frames.append((Path(traceback.tb_frame.f_code.co_filename).name,
                           traceback.tb_frame.f_code.co_name, traceback.tb_lineno))
            traceback = traceback.tb_next
        reason = error.args[0] if error.args and type(error.args[0]) is str else None
        rows.append(dict(error_class=type(error).__name__, reason=reason, frames=frames,
                         suppress_context=error.__suppress_context__))
        error = error.__context__
    return rows


def test_genuine_current_fallback_expiry(tmp_path, monkeypatch, record, ordinary_effects, caller_boundary):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("model: invented_model\nmemory:\n  enabled: false\n")
    assert Path(sys.prefix).resolve() == Path(
        "/home/mersinvald/.local/share/orca/runtime/accountant/hpwa-p10.47/venv").resolve()
    root = Path(__file__).resolve().parents[2]
    modules = ("agent.native_main_payload", "agent.content_policy", "agent.transports.owned_loopback_http",
               "tests.agent.owned_loopback_receiver", __name__, "hermes_content_policy")
    loaded = {name: Path(sys.modules[name].__file__).resolve() for name in modules}
    assert all(path.is_relative_to(root) for name, path in loaded.items() if name != "hermes_content_policy")
    record("source", dict(interpreter=sys.executable,
        modules={name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                 for name, path in loaded.items()}))
    source_lines = Path(native.__file__).read_text().splitlines()
    snapshot_start = native.NativeMainPayloadAdapter._fallback_snapshot.__code__.co_firstlineno
    expiry_line = next(index for index, line in enumerate(source_lines, 1)
                       if index >= snapshot_start and line == "            _require(now < route.expires_at)")
    record("predicate", dict(file="agent/native_main_payload.py", line=expiry_line,
        expression="_require(now < route.expires_at)"))
    a, b = OwnedReceiver(observation_seconds=590.0), OwnedReceiver(observation_seconds=590.0)
    world, terminal_error, result = None, None, None
    receipts = {}
    try:
        world = build_world(a, b)
        assert_plan(world, a, b)
        hook, trigger = make_trigger(world, b, record)
        world.fallback_selection_hook = hook
        a.start((ResponseStep(world.endpoints[0].request_path.encode(), response_bytes(ACK), QUALIFICATION),
                 ResponseStep(world.endpoints[0].request_path.encode(), b"", hold_seconds=6.0, allow_disconnect=True)))
        b.start((ResponseStep(world.endpoints[1].request_path.encode(), response_bytes(ACK), QUALIFICATION),))
        record("initial", dict(routes=[asdict(route) for route in world.routes],
            profiles=[asdict(profile) for profile in world.profiles], endpoints=[asdict(ep) for ep in world.endpoints],
            a_listener=id(a.listener), b_listener=id(b.listener), a_fd=a.listener.fileno(), b_fd=b.listener.fileno()))
        qualifier = qualify(world, record)
        a.wait(1)
        b.wait(1)
        record("qualification_complete", dict(charges=charges(qualifier), graph=graph(qualifier), count=16,
            http=2, in_process=14, rows_a=tuple(a.rows), rows_b=tuple(b.rows)))
        World.bind(world, qualify=False)
        construct(world, model="invented_model_0", max_tokens=None)
        bind_fallback(world)
        record("phase", "native_bound")
        assert not trigger and world.now == 10
        try:
            result = fixed_turn(world)
        except BaseException as error:
            terminal_error = error
            record("exception", exception_chain(error))
        runtime, adapter = world.adapter._runtime, world.adapter
        record("terminal", dict(charges=charges(runtime), graph=graph(runtime), now=world.now,
            phase=adapter._fallback_phase, denied=world.context._denied, result_none=result is None,
            failed_count=len(adapter._failed_main_attempts), fallback_invocations=len(adapter._fallback_records),
            main_sends=adapter._send_count, fallback_sends=adapter._fallback_send_count,
            records=len(adapter._records), completions=len(adapter._completions)))
        observations = world.ports["main"].observations
        record("main_observations", [dict(envelope=asdict(row.envelope), output=row.output,
            error_class=None if row.error is None else type(row.error).__name__,
            receipt=row.transport_receipt, started_at=row.started_at, finished_at=row.finished_at)
            for row in observations])
        a.wait(2)
        receipts["a"] = a.finish()
        record("a_final", asdict(receipts["a"]))
        receipts["b"] = b.finish()
        record("b_final", asdict(receipts["b"]))
        assert type(terminal_error) is NativeContentPolicyDenied
        assert terminal_error.args == ("controlled_native_payload_denied",)
        inner = terminal_error.__context__
        assert type(inner) is NativeContentPolicyDenied and inner.args == ("controlled_native_payload_invalid",)
        assert ("native_main_payload.py", "_fallback_snapshot", expiry_line) in exception_chain(inner)[0]["frames"]
        assert trigger and world.now == 100
        failed, = adapter._failed_main_attempts
        frame = adapter._fallback_frame
        assert failed is trigger["failed"] and frame is trigger["frame"]
        assert frame.failed_main is failed and frame.binding is adapter._fallback_binding
        assert trigger["entry"] is runtime._failures[id(failed.shared_failure)]
        assert trigger["entry"].token is failed.shared_failure and trigger["entry"].facts is failed.failure_facts
        assert type(failed.shared_failure) is FailedDispatch and type(frame.token) is native.NativeFallbackInput
        assert frame.token is not failed.shared_failure
        assert trigger["entry"].request_handle is failed.request_handle is world.request
        assert adapter._fallback_phase == "authenticate" and not frame.activated
        assert frame.source_content is frame.invocation is frame.result is frame.result_content is None
        assert not adapter._fallback_records and adapter._fallback_send_count == 0
        assert not adapter._records and not adapter._completions and result is None and world.context._denied
        assert charges(runtime) == trigger["charges"] and graph(runtime) == trigger["graph"]
        qualification, attempt = observations
        assert qualification.error is None and qualification.output == ACK
        assert type(attempt.error) is TimeoutError and attempt.output is None
        assert attempt.transport_receipt is attempt.error.owned_loopback_receipt
        receipt = attempt.transport_receipt
        assert receipt[:2] == ("owned_loopback_http_v1", "response")
        assert receipt[2] == receipt[3] == a.bound and receipt[4] == 1
        assert receipt[5] > len(attempt.envelope.payload) and receipt[6] == b""
        assert attempt.envelope is failed.send_observation.envelope is failed.failure_facts.envelope
        assert failed.send_observation.exception_type is TimeoutError
        assert (failed.send_observation.send_count_before, failed.send_observation.send_count_after) == (0, 1)
        assert failed.request is caller_boundary.dispatches[-1][2]
        assert failed.request_bytes == attempt.envelope.payload
        assert failed.failure_facts.failure_content is failed.failure_content
        assert failed.failure_facts.request_content is failed.request_content
        assert failed.failure_facts.response_bytes is None and failed.failure_facts.failure_kind == "timeout_error"
        assert runtime._contents[id(failed.failure_content)].parents == (failed.request_content,)
        assert failed.failure_content in frame.parents and failed.request_content in frame.parents
        assert all(row.profiles == ("restricted_documents",) for row in runtime._contents.values())
        assert len(runtime._envelopes) == 1 and adapter._send_count == 1
        for route in world.routes:
            assert (100 < route.expires_at) == (route.route_ref != "fallback_fallback")
        assert world.binding is world.initial_originals[2] and world.policy is world.initial_originals[4]
        assert world.ports["main"].resolve() is world.selection.destination
        for name, final in receipts.items():
            assert_receipt(final)
            (a if name == "a" else b).check()
        assert receipts["a"].mark is None and len(receipts["a"].connections) == 2
        assert len(receipts["b"].connections) == receipts["b"].mark.row_count == 1
        assert receipts["b"].mark is trigger["mark"]
        assert receipts["b"].connections[receipts["b"].mark.row_count:] == ()
        assert receipts["b"].connections[0].body == QUALIFICATION
        assert receipts["a"].connections[0].body == QUALIFICATION
        main = receipts["a"].connections[1]
        assert main.complete and main.body == attempt.envelope.payload and main.response_accepted == 0
        assert main.status == "planned_disconnect" and main.failure == "eof_after_request"
        assert receipt[5] == main.raw_length
        assert not world.sdk_calls
        record("phase", "genuine_verified")
    finally:
        for name, receiver in (("a", a), ("b", b)):
            try:
                if name not in receipts and receiver.thread is not None:
                    receipts[name] = receiver.finish()
                    record(name + "_partial_final", asdict(receipts[name]))
            finally:
                receiver.close()
        record("cleanup", dict(a_closed=a.listener.fileno() == -1, b_closed=b.listener.fileno() == -1,
            a_stopped=a.thread is None or not a.thread.is_alive(), b_stopped=b.thread is None or not b.thread.is_alive(),
            a_errors=[type(error).__name__ for error in a.errors], b_errors=[type(error).__name__ for error in b.errors],
            later_state="UNKNOWN" if terminal_error is None and result is None else "retained"))
