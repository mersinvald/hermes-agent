"""Check the owned wire boundary and unchanged shared dispatch behavior."""

import asyncio
import builtins
from dataclasses import fields, replace
import importlib.util
from pathlib import Path
import socket
import sys
import threading
from types import SimpleNamespace

import pytest
from hermes_content_policy import (
    ContentPolicyRuntime, Destination, DispatchEnvelope, ExecutionBinding, FailedDispatch,
    PATHS, PolicyError, ProfilePolicy, RouteBinding,
)

from agent.transports import owned_loopback_http as wire
from tests.agent.owned_loopback_receiver import OwnedReceiver, ResponseStep, response_bytes


def configured(port_number, module=wire):
    destination = Destination("invented_0", "synthetic_native", "invented_model_0", "synthetic_loopback_p", 1)
    endpoint = module.OwnedLoopbackEndpoint(destination.endpoint_ref, "127.0.0.1", port_number,
        "/owned-accountant-proof/v1/p", destination.provider_ref, destination.model_ref, 1)
    port = module.OwnedLoopbackHTTPTransport(destination, endpoint=endpoint)
    return destination, endpoint, port


def envelope(destination, payload=b"Invented request"):
    binding = ExecutionBinding("owner", "execution", "lineage", "consumer", 7, 9,
                               "selection", 3, ("restricted_documents",))
    return DispatchEnvelope(payload, binding, binding.profiles, binding.consumer_ref, "main", replace(destination))


def runtime_for(port):
    request = object()
    binding = envelope(port.resolve()).source
    state = SimpleNamespace(binding=binding, now=10)
    admission = SimpleNamespace(current=lambda handle: state.binding)
    routes = tuple(RouteBinding("route_" + path, binding.selection_id, binding.selection_version,
        binding.consumer_ref, path, port.resolve(), 100, "synthetic_owned_" + path, "synthetic_only") for path in PATHS)
    profiles = (ProfilePolicy("restricted_documents", tuple(row.route_ref for row in routes), PATHS),)
    runtime = ContentPolicyRuntime(admission=admission, profiles=profiles, routes=routes,
        transports=((port.resolve().transport_ref, port),), clock=lambda: state.now, policy_revision=9)
    content = runtime.admit(request, b"Invented request")
    return runtime, request, content, state


def test_positive_wire_and_fresh_destination():
    with OwnedReceiver() as receiver:
        destination, endpoint, port = configured(receiver.bound[1])
        receiver.start((ResponseStep(endpoint.request_path.encode(), response_bytes(b"Invented response")),))
        request = envelope(destination)
        assert request.destination is not destination
        assert port.resolve() is destination
        assert port.send(request) == b"Invented response"
        row, = receiver.wait(1)
        assert row["body"] is not request.payload and row["body"] == request.payload
        assert row["local"] == row["bound"] == receiver.bound
        assert row["peer"][0] == "127.0.0.1" and row["peer"][1] != receiver.bound[1]
        assert row["host"] == row["client_reported_peer"] == f"127.0.0.1:{receiver.bound[1]}".encode()
        assert row["response_attempts"] == 1 and row["response_accepted"] == len(receiver.steps[0].response)
        assert row["complete"] and row["failure"] is None
        assert not port._busy and not hasattr(port, "__dict__") and not hasattr(port, "sent")
        with pytest.raises(AttributeError):
            port._busy = True
        assert [field.name for field in fields(endpoint)] == list(wire._ENDPOINT_FIELDS)


@pytest.mark.parametrize("field,value", [
    ("address", "localhost"), ("address", "http://127.0.0.1"), ("address", "::1"),
    ("address", "127.0.0.2"), ("port", True), ("port", 0), ("port", 65536),
    ("request_path", "/owned-accountant-proof/v1/p?x=1"),
    ("request_path", "/owned-accountant-proof/v1/p#x"),
    ("endpoint_ref", "invented_endpoint"), ("model_ref", "invented_model_2"),
    ("revision", True), ("revision", 0), ("revision", 1 << 53),
    ("provider_ref", type("StringSubclass", (str,), {})("synthetic_native")),
])
def test_descriptor_refusal(field, value):
    with OwnedReceiver() as receiver:
        destination, endpoint, _ = configured(receiver.bound[1])
        with pytest.raises(ValueError):
            wire.OwnedLoopbackHTTPTransport(destination, endpoint=replace(endpoint, **{field: value}))
        assert receiver.rows == []
        with pytest.raises(TypeError):
            wire.OwnedLoopbackEndpoint(**{**dict((f.name, getattr(endpoint, f.name)) for f in fields(endpoint)), "unknown": 1})


@pytest.mark.parametrize("change", ["endpoint", "reference", "destination", "method", "helper_code", "defaults",
                                    "socket", "clock", "constant", "layout", "busy"])
def test_drift_before_traffic(monkeypatch, change):
    with OwnedReceiver() as receiver:
        destination, endpoint, port = configured(receiver.bound[1])
        with monkeypatch.context() as patch:
            if change == "endpoint":
                object.__setattr__(endpoint, "port", receiver.bound[1] + 1)
            elif change == "reference":
                object.__setattr__(port, "_endpoint", replace(endpoint))
            elif change == "destination":
                object.__setattr__(destination, "model_ref", "invented_model_2")
            elif change == "method":
                patch.setattr(wire.OwnedLoopbackHTTPTransport, "resolve", lambda self: destination)
            elif change == "helper_code":
                patch.setattr(wire._endpoint, "__code__", (lambda *args: None).__code__)
            elif change == "defaults":
                patch.setattr(wire._remaining, "__defaults__", (1,))
            elif change == "socket":
                patch.setattr(socket.socket, "connect", lambda *args: None)
            elif change == "clock":
                patch.setattr(wire.time, "monotonic", lambda: 0)
            elif change == "constant":
                patch.setattr(wire, "_REQUEST_BODY", 999)
            elif change == "layout":
                patch.setattr(wire.OwnedLoopbackHTTPTransport, "__slots__", ())
            else:
                object.__setattr__(port, "_busy", 1)
            with pytest.raises(ValueError):
                wire.OwnedLoopbackHTTPTransport.send(port, envelope(destination))
        assert receiver.rows == []


def test_destination_body_thread_and_reentry_denials():
    with OwnedReceiver() as receiver:
        destination, _, port = configured(receiver.bound[1])
        for request in (envelope(replace(destination, transport_ref="other")),
                        envelope(destination, b"x" * 262145), envelope(destination, bytearray(b"x"))):
            with pytest.raises(ValueError):
                port.send(request)
        failures = []
        def foreign():
            try:
                port.send(envelope(destination))
            except BaseException as error:
                failures.append(error)
        thread = threading.Thread(target=foreign)
        thread.start()
        thread.join(5)
        assert not thread.is_alive() and len(failures) == 1
        assert str(failures[0]) == "owned_loopback_thread"
        object.__setattr__(port, "_busy", True)
        with pytest.raises(ValueError, match="owned_loopback_busy"):
            port.send(envelope(destination))
        object.__setattr__(port, "_busy", False)
        assert receiver.rows == []


FAULTS = (
    b"HTTP/1.0 200 OK\r\nContent-Length: 0\r\n\r\n",
    b"HTTP/1.1 100 Continue\r\nContent-Length: 0\r\n\r\n",
    b"HTTP/1.1 302 Found\r\nLocation: /owned-accountant-proof/v1/c\r\nContent-Length: 0\r\n\r\n",
    b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\nContent-Length: 1\r\n\r\nx",
    b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Length: 0\r\n\r\n",
    b"HTTP/1.1 200 OK\r\nContent-Encoding: gzip\r\nContent-Length: 0\r\n\r\n",
    b"HTTP/1.1 200 OK\r\nContent-Length: +1\r\n\r\nx",
    b"HTTP/1.1 200 OK\r\nBad header\r\nContent-Length: 0\r\n\r\n",
    b"HTTP/1.1 200 OK\r\nContent-Length: 65537\r\n\r\n",
    b"HTTP/1.1 200 OK\r\nX: " + b"x" * 8192 + b"\r\nContent-Length: 0\r\n\r\n",
    b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nx",
    b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\nxy",
    b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n",
)


@pytest.mark.parametrize("response", FAULTS, ids=("version", "informational", "redirect", "duplicate",
    "transfer", "encoding", "decimal", "header", "body_bound", "header_bound", "truncated", "extra", "unfinished"))
def test_actual_response_fault(response):
    with OwnedReceiver() as receiver:
        destination, endpoint, port = configured(receiver.bound[1])
        receiver.start((ResponseStep(endpoint.request_path.encode(), response, allow_disconnect=True),))
        with pytest.raises(ValueError) as caught:
            port.send(envelope(destination))
        receipt = caught.value.owned_loopback_receipt
        row, = receiver.wait(1)
        assert row["complete"] and row["body"] == b"Invented request"
        assert receipt[1] == "response" and receipt[3] == receiver.bound and receipt[4] == 1
        assert receipt[5] == len(row["raw"]) and response.startswith(receipt[6])
        assert len(receipt[6]) <= 8192 + 65536 + 1
        assert row["response_attempts"] == 1 and len(receiver.rows) == 1 and not port._busy


def test_actual_bounds():
    with OwnedReceiver() as receiver:
        destination, endpoint, port = configured(receiver.bound[1])
        body = b"r" * 65536
        receiver.start((ResponseStep(endpoint.request_path.encode(), response_bytes(body)),))
        payload = b"q" * 262144
        assert port.send(envelope(destination, payload)) == body
        row, = receiver.wait(1)
        assert row["body"] == payload and len(row["header"]) <= 2048


def test_refusal_preserves_oserror_subclass():
    reservation = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        reservation.bind(("127.0.0.1", 0))
        destination, _, port = configured(reservation.getsockname()[1])
        with pytest.raises(ConnectionRefusedError) as caught:
            port.send(envelope(destination))
        assert type(caught.value) is ConnectionRefusedError
        assert caught.value.owned_loopback_receipt[1] == "connect"
        assert caught.value.owned_loopback_receipt[4:6] == (1, 0)
        assert caught.value.owned_loopback_receipt[3] is None and not port._busy
    finally:
        reservation.close()


def test_actual_deadline_and_failure_token():
    with OwnedReceiver() as receiver:
        _, endpoint, port = configured(receiver.bound[1])
        receiver.start((ResponseStep(endpoint.request_path.encode(), b"", hold_seconds=5.25),))
        runtime, request, content, _ = runtime_for(port)
        failure = runtime.dispatch_attempt(request, content, path="main", route_ref="route_main")
        assert type(failure) is FailedDispatch
        facts = runtime.failure_facts(request, failure)
        assert facts.failure_kind == "timeout_error" and facts.send_count == 1 and facts.response_bytes is None
        assert facts.envelope.destination is not port.resolve()
        row, = receiver.wait(1)
        assert row["complete"] and len(runtime._envelopes) == 1
        assert not runtime._poisoned and len(receiver.rows) == 1 and not port._busy
        with pytest.raises(PolicyError):
            runtime.dispatch(request, content, path="fallback", route_ref="unqualified_fallback")
        assert len(receiver.rows) == 1


@pytest.mark.parametrize("denial", ["binding", "route", "profile"])
def test_shared_current_denial_before_send(denial):
    with OwnedReceiver() as receiver:
        _, _, port = configured(receiver.bound[1])
        runtime, request, content, state = runtime_for(port)
        if denial == "binding":
            state.binding = replace(state.binding, auth_epoch=8)
        elif denial == "route":
            state.now = 101
        else:
            state.binding = replace(state.binding, profiles=("unknown_profile",))
        with pytest.raises(PolicyError):
            runtime.dispatch(request, content, path="main", route_ref="route_main")
        assert len(runtime._envelopes) == 0 and receiver.rows == []


@pytest.mark.parametrize("operation", ["dispatch", "dispatch_attempt"])
def test_shared_post_send_fault_has_no_retry(operation):
    with OwnedReceiver() as receiver:
        _, endpoint, port = configured(receiver.bound[1])
        receiver.start((ResponseStep(endpoint.request_path.encode(), FAULTS[2]),))
        runtime, request, content, _ = runtime_for(port)
        with pytest.raises(PolicyError if operation == "dispatch" else ValueError) as caught:
            getattr(runtime, operation)(request, content, path="main", route_ref="route_main")
        if operation == "dispatch":
            assert str(caught.value) == "post_send_rejected"
        else:
            assert caught.value.owned_loopback_receipt[1] == "response"
        receiver.wait(1)
        assert runtime._poisoned and len(runtime._envelopes) == 1 and len(runtime._failures) == 0
        with pytest.raises(PolicyError):
            runtime.dispatch(request, content, path="main", route_ref="route_main")
        assert len(receiver.rows) == 1


def inert_module(monkeypatch, *, error=None, peer=None, reenter=False, deadline=False, drift=None):
    """Run original source with inert sockets, without network or native authority."""
    state = SimpleNamespace(sockets=[], clock=0.0, port=None, reentered=[], calls=[], accepted=0)
    class InertSocket:
        def __init__(self, family, kind):
            assert family == socket.AF_INET and kind == socket.SOCK_STREAM
            state.sockets.append(self)
            self.closed = False
            self.target = None
            self.reply = response_bytes(b"ok")
        def connect(self, target):
            self.target = target
            state.calls.append("connect")
        def getpeername(self):
            if drift == "prewrite":
                object.__setattr__(state.port._endpoint, "revision", 2)
            return peer or self.target
        def settimeout(self, seconds):
            assert 0 < seconds <= 5
        def send(self, data):
            state.calls.append("send")
            if reenter and not state.reentered:
                state.reentered.append(True)
                with pytest.raises(ValueError, match="owned_loopback_busy"):
                    state.port.send(envelope(state.port._destination))
            if error is not None and state.accepted >= 7:
                raise error
            count = min(7, len(data))
            state.accepted += count
            if deadline:
                state.clock += 1.0
            return count
        def recv(self, count):
            if drift == "postread":
                object.__setattr__(state.port._endpoint, "revision", 2)
            result, self.reply = self.reply[:count], self.reply[count:]
            return result
        def close(self):
            self.closed = True
    def monotonic():
        return state.clock
    fake_socket = SimpleNamespace(socket=InertSocket, AF_INET=socket.AF_INET, SOCK_STREAM=socket.SOCK_STREAM)
    def imported(name, *args, **kwargs):
        if name == "socket":
            return fake_socket
        if name == "time":
            return SimpleNamespace(monotonic=monotonic)
        return builtins.__import__(name, *args, **kwargs)
    name = "_owned_loopback_inert_control"
    spec = importlib.util.spec_from_file_location(name, wire.__file__)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__builtins__"] = dict(vars(builtins), __import__=imported)
    monkeypatch.setitem(sys.modules, name, module)
    exec(compile(Path(wire.__file__).read_bytes(), wire.__file__, "exec"), module.__dict__)
    destination, _, state.port = configured(12345, module)
    return state, module, destination


def test_inert_partial_write_reentry_and_peer_mismatch(monkeypatch):
    state, _, destination = inert_module(monkeypatch, reenter=True)
    assert state.port.send(envelope(destination)) == b"ok"
    assert len(state.sockets) == 1 and state.sockets[0].closed and len(state.calls) > 3
    assert state.reentered == [True] and state.port._busy is False
    state, _, destination = inert_module(monkeypatch, peer=("127.0.0.1", 12346))
    with pytest.raises(ValueError, match="owned_loopback_peer") as caught:
        state.port.send(envelope(destination))
    assert state.calls == ["connect"] and state.sockets[0].closed
    assert caught.value.owned_loopback_receipt[3:6] == (("127.0.0.1", 12346), 1, 0)


@pytest.mark.parametrize("error", [ConnectionError("inert"), ConnectionRefusedError("inert"),
    TimeoutError("inert"), OSError("inert"), asyncio.CancelledError("inert"), KeyboardInterrupt("inert")],
    ids=("exact_connection", "refused_subclass", "exact_timeout", "oserror", "cancel", "interrupt"))
def test_inert_exception_identity_and_shared_boundary(monkeypatch, error):
    state, _, destination = inert_module(monkeypatch, error=error)
    with pytest.raises(type(error)) as caught:
        state.port.send(envelope(destination))
    assert caught.value is error and error.owned_loopback_receipt[4:6] == (1, 7)
    assert all(s.closed for s in state.sockets) and not state.port._busy
    state, _, _ = inert_module(monkeypatch, error=error)
    runtime, request, content, _ = runtime_for(state.port)
    if type(error) in (ConnectionError, TimeoutError):
        failure = runtime.dispatch_attempt(request, content, path="main", route_ref="route_main")
        assert type(failure) is FailedDispatch and runtime.failure_facts(request, failure).response_bytes is None
        assert not runtime._poisoned
    else:
        with pytest.raises(type(error)) as caught:
            runtime.dispatch_attempt(request, content, path="main", route_ref="route_main")
        assert caught.value is error and runtime._poisoned and not runtime._failures
    assert len(runtime._envelopes) == 1 and len(state.sockets) == 1 and state.sockets[0].closed


def test_inert_absolute_write_deadline(monkeypatch):
    state, _, destination = inert_module(monkeypatch, deadline=True)
    with pytest.raises(TimeoutError) as caught:
        state.port.send(envelope(destination))
    assert type(caught.value) is TimeoutError and state.clock == 5
    assert caught.value.owned_loopback_receipt[5] == 35
    assert len(state.sockets) == 1 and state.sockets[0].closed and not state.port._busy


@pytest.mark.parametrize("drift", ["prewrite", "postread"])
def test_inert_drift_at_wire_checkpoints(monkeypatch, drift):
    state, _, destination = inert_module(monkeypatch, drift=drift)
    with pytest.raises(ValueError) as caught:
        state.port.send(envelope(destination))
    receipt = caught.value.owned_loopback_receipt
    assert receipt[1] == ("peer" if drift == "prewrite" else "postflight")
    assert receipt[4] == 1 and len(state.sockets) == 1 and state.sockets[0].closed
    assert receipt[5] == (0 if drift == "prewrite" else state.accepted)
    assert not state.port._busy
