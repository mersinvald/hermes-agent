"""Send invented bytes to one operator-selected numeric loopback endpoint."""

from dataclasses import dataclass
import re
import socket
import threading
import time
import types

from hermes_content_policy import Destination, DispatchEnvelope


@dataclass(frozen=True, slots=True)
class OwnedLoopbackEndpoint:
    endpoint_ref: str
    address: str
    port: int
    request_path: str
    provider_ref: str
    model_ref: str
    revision: int


_ADDRESS = "127.0.0.1"
_ENDPOINT_FIELDS = ("endpoint_ref", "address", "port", "request_path", "provider_ref", "model_ref", "revision")
_DESTINATION_FIELDS = ("transport_ref", "provider_ref", "model_ref", "endpoint_ref", "revision")
_ENDPOINTS = (
    ("synthetic_loopback_p", "/owned-accountant-proof/v1/p", "invented_model_0"),
    ("synthetic_loopback_receiver", "/owned-accountant-proof/v1/receiver", "invented_model_2"),
    ("synthetic_loopback_child", "/owned-accountant-proof/v1/child", "invented_model_3"),
    ("synthetic_loopback_c", "/owned-accountant-proof/v1/c", "invented_model_5"),
)
_REQUEST_HEADERS = 2048
_REQUEST_BODY = 262144
_RESPONSE_HEADERS = 8192
_RESPONSE_BODY = 65536
_CONNECT_SECONDS = 2.0
_TOTAL_SECONDS = 5.0
_MAX_REVISION = (1 << 53) - 1
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z", re.ASCII)
_HEADER_NAME = re.compile(rb"[!#$%&'*+.^_`|~0-9A-Za-z-]+\Z")
_STATUS = re.compile(rb"HTTP/1\.1 200 [\x20-\x7e]*\Z")
_FAMILY = socket.AF_INET
_KIND = socket.SOCK_STREAM
_SOCKET = socket.socket
_CONNECT = socket.socket.connect
_PEER = socket.socket.getpeername
_SEND = socket.socket.send
_RECV = socket.socket.recv
_TIMEOUT = socket.socket.settimeout
_CLOSE = socket.socket.close
_CLOCK = time.monotonic
_THREAD = threading.get_ident


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def _facts(value, cls, names):
    _require(type(value) is cls, "owned_loopback_type")
    return tuple(object.__getattribute__(value, name) for name in names)


def _destination(value):
    facts = _facts(value, Destination, _DESTINATION_FIELDS)
    _require(all(type(item) is str and _IDENTIFIER.fullmatch(item) for item in facts[:4]),
             "owned_loopback_destination")
    _require(type(facts[4]) is int and 1 <= facts[4] <= _MAX_REVISION, "owned_loopback_revision")
    return facts


def _endpoint(value, destination):
    facts = _facts(value, OwnedLoopbackEndpoint, _ENDPOINT_FIELDS)
    _require(all(type(facts[i]) is str for i in (0, 1, 3, 4, 5)), "owned_loopback_endpoint")
    _require(facts[1] == _ADDRESS and type(facts[2]) is int and 1 <= facts[2] <= 65535,
             "owned_loopback_address")
    _require((facts[0], facts[3], facts[5]) in _ENDPOINTS, "owned_loopback_path")
    _require(facts[4] == "synthetic_native" and type(facts[6]) is int
             and 1 <= facts[6] <= _MAX_REVISION, "owned_loopback_revision")
    _require((facts[4], facts[5], facts[0], facts[6]) == destination[1:], "owned_loopback_destination")
    return facts


def _code_check(anchors):
    _require(anchors is _INTEGRITY, "owned_loopback_integrity")
    functions, callables, constants, layouts, classes = anchors
    for owner, name, function, code, namespace, defaults, kwdefaults, items in functions:
        actual = owner[name] if type(owner) is dict else getattr(owner, name)
        _require(actual is function and type(actual) is types.FunctionType
                 and actual.__code__ is code and actual.__globals__ is namespace
                 and actual.__defaults__ is defaults and actual.__kwdefaults__ is kwdefaults,
                 "owned_loopback_code")
        current_items = tuple((actual.__kwdefaults__ or {}).items())
        _require(len(current_items) == len(items) and all(a[0] == b[0] and a[1] is b[1]
                 for a, b in zip(current_items, items)), "owned_loopback_defaults")
    for owner, name, original, original_type, metadata in callables:
        actual = getattr(owner, name)
        _require(actual is original and type(actual) is original_type, "owned_loopback_callable")
        if metadata is not None:
            _require(actual.__code__ is metadata[0] and actual.__globals__ is metadata[1]
                     and actual.__defaults__ is metadata[2] and actual.__kwdefaults__ is metadata[3],
                     "owned_loopback_callable_code")
            current = tuple((actual.__kwdefaults__ or {}).items())
            _require(len(current) == len(metadata[4]) and all(a[0] == b[0] and a[1] is b[1]
                     for a, b in zip(current, metadata[4])), "owned_loopback_callable_defaults")
    for name, original, original_type in constants:
        _require(globals()[name] is original and type(globals()[name]) is original_type,
                 "owned_loopback_constant")
    for cls, slots, descriptors, namespace, fields in layouts:
        _require(cls.__slots__ is slots and tuple(sorted(name for name, item in vars(cls).items()
                 if type(item) is types.MemberDescriptorType)) == tuple(sorted(slots)), "owned_loopback_layout")
        _require(all(getattr(cls, name) is descriptor for name, descriptor in descriptors),
                 "owned_loopback_layout")
        actual = tuple(vars(cls).items())
        _require(len(actual) == len(namespace) and all(a[0] == b[0] and a[1] is b[1]
                 for a, b in zip(actual, namespace)), "owned_loopback_class_layout")
        actual_fields = tuple(getattr(cls, "__dataclass_fields__", {}).items())
        _require(len(actual_fields) == len(fields) and all(a[0] == b[0] and a[1] is b[1]
                 for a, b in zip(actual_fields, fields)), "owned_loopback_fields")
    for name, cls in classes:
        _require(globals()[name] is cls, "owned_loopback_class")


def _guard(port, busy):
    _ORIGINAL_CODE_CHECK(port._integrity)
    _require(type(port) is OwnedLoopbackHTTPTransport, "owned_loopback_type")
    _require(type(port._busy) is bool and port._busy is busy, "owned_loopback_busy")
    _require(type(port._owner_thread) is int and _THREAD() == port._owner_thread,
             "owned_loopback_thread")
    anchors = port._anchors
    _require(type(anchors) is tuple and len(anchors) == 5, "owned_loopback_anchors")
    _require(port._destination is anchors[0] and port._endpoint is anchors[1]
             and port._owner_thread is anchors[4], "owned_loopback_identity")
    destination = _destination(port._destination)
    endpoint = _endpoint(port._endpoint, destination)
    _require(all(a is b for a, b in zip(destination, anchors[2]))
             and all(a is b for a, b in zip(endpoint, anchors[3])), "owned_loopback_facts")
    return endpoint


def _remaining(deadline, cap=None):
    remaining = deadline - _CLOCK()
    if remaining <= 0:
        raise TimeoutError("owned_loopback_deadline")
    return remaining if cap is None else min(remaining, cap)


def _receive(connection, deadline, count):
    _TIMEOUT(connection, _remaining(deadline))
    return _RECV(connection, count)


def _response(connection, deadline, observed):
    prefix = b""
    while b"\r\n\r\n" not in prefix:
        chunk = _receive(connection, deadline, min(4096, _RESPONSE_HEADERS + 1 - len(prefix)))
        observed.extend(chunk)
        prefix += chunk
        _require(bool(chunk), "owned_loopback_truncated_headers")
        _require(len(prefix) <= _RESPONSE_HEADERS or b"\r\n\r\n" in prefix,
                 "owned_loopback_response_headers")
    header, body = prefix.split(b"\r\n\r\n", 1)
    _require(len(header) + 4 <= _RESPONSE_HEADERS, "owned_loopback_response_headers")
    rows = header.split(b"\r\n")
    _require(_STATUS.fullmatch(rows[0]) is not None, "owned_loopback_status")
    headers = {}
    for row in rows[1:]:
        name, separator, value = row.partition(b":")
        _require(separator and _HEADER_NAME.fullmatch(name) is not None
                 and all(byte == 9 or 32 <= byte <= 126 for byte in value), "owned_loopback_header")
        name = name.lower()
        _require(name not in headers, "owned_loopback_duplicate_header")
        headers[name] = value.strip(b" \t")
    _require(b"transfer-encoding" not in headers and b"content-encoding" not in headers,
             "owned_loopback_encoding")
    length = headers.get(b"content-length", b"")
    _require(1 <= len(length) <= 10 and all(48 <= byte <= 57 for byte in length),
             "owned_loopback_length")
    size = int(length)
    _require(size <= _RESPONSE_BODY, "owned_loopback_response_body")
    _require(len(body) <= size, "owned_loopback_extra_bytes")
    while len(body) < size:
        chunk = _receive(connection, deadline, min(4096, size + 1 - len(body)))
        observed.extend(chunk)
        body += chunk
        _require(bool(chunk), "owned_loopback_truncated_body")
        _require(len(body) <= size, "owned_loopback_extra_bytes")
    extra = _receive(connection, deadline, 1)
    observed.extend(extra)
    _require(not extra, "owned_loopback_extra_bytes")
    return body


class OwnedLoopbackHTTPTransport:
    __slots__ = ("_destination", "_endpoint", "_owner_thread", "_anchors", "_integrity", "_busy")

    def __init__(self, destination, *, endpoint):
        _ORIGINAL_CODE_CHECK(_INTEGRITY)
        destination_facts = _destination(destination)
        endpoint_facts = _endpoint(endpoint, destination_facts)
        owner = _THREAD()
        object.__setattr__(self, "_destination", destination)
        object.__setattr__(self, "_endpoint", endpoint)
        object.__setattr__(self, "_owner_thread", owner)
        object.__setattr__(self, "_anchors", (destination, endpoint, destination_facts, endpoint_facts, owner))
        object.__setattr__(self, "_integrity", _INTEGRITY)
        object.__setattr__(self, "_busy", False)
        _ORIGINAL_GUARD(self, False)

    def __setattr__(self, name, value):
        raise AttributeError("owned_loopback_immutable")

    def __delattr__(self, name):
        raise AttributeError("owned_loopback_immutable")

    def resolve(self):
        _ORIGINAL_GUARD(self, False)
        return self._destination

    def send(self, envelope) -> bytes:
        connection, peer, selected = None, None, None
        accepted, attempts, phase = 0, 0, "preflight"
        observed = bytearray()
        entered, pending = False, None
        close = _CLOSE
        try:
            endpoint = _ORIGINAL_GUARD(self, False)
            _require(type(envelope) is DispatchEnvelope, "owned_loopback_envelope")
            _require(_destination(envelope.destination) == self._anchors[2], "owned_loopback_destination")
            _require(type(envelope.payload) is bytes and len(envelope.payload) <= _REQUEST_BODY,
                     "owned_loopback_request_body")
            selected = (_ADDRESS, endpoint[2])
            payload = envelope.payload
            object.__setattr__(self, "_busy", True)
            entered = True
            deadline = _CLOCK() + _TOTAL_SECONDS
            try:
                _ORIGINAL_GUARD(self, True)
                phase = "socket"
                connection = _SOCKET(_FAMILY, _KIND)
                _TIMEOUT(connection, _remaining(deadline, _CONNECT_SECONDS))
                phase, attempts = "connect", 1
                _CONNECT(connection, selected)
                phase = "peer"
                peer = _PEER(connection)
                _require(type(peer) is tuple and len(peer) == 2 and type(peer[0]) is str
                         and type(peer[1]) is int and peer == selected, "owned_loopback_peer")
                _ORIGINAL_GUARD(self, True)
                headers = (f"POST {endpoint[3]} HTTP/1.1\r\nHost: {selected[0]}:{selected[1]}\r\n"
                    f"X-Owned-Peer: {peer[0]}:{peer[1]}\r\nContent-Length: {len(payload)}\r\n"
                    "Connection: close\r\nContent-Type: application/octet-stream\r\n\r\n").encode("ascii")
                _require(len(headers) <= _REQUEST_HEADERS, "owned_loopback_request_headers")
                request = headers + payload
                phase = "write"
                while accepted < len(request):
                    _TIMEOUT(connection, _remaining(deadline))
                    count = _SEND(connection, request[accepted:])
                    _require(type(count) is int and 0 < count <= len(request) - accepted,
                             "owned_loopback_send_count")
                    accepted += count
                phase = "response"
                output = _response(connection, deadline, observed)
                phase = "postflight"
                _ORIGINAL_GUARD(self, True)
                _remaining(deadline)
                return output
            except BaseException as error:
                pending = error
                raise
            finally:
                try:
                    if connection is not None:
                        close(connection)
                except BaseException:
                    if pending is None:
                        phase = "close"
                        raise
                finally:
                    if entered:
                        active = self._busy
                        object.__setattr__(self, "_busy", False)
                        if pending is None:
                            _require(type(active) is bool and active is True, "owned_loopback_busy")
        except BaseException as error:
            connected = peer if (type(peer) is tuple and len(peer) == 2 and type(peer[0]) is str
                and len(peer[0]) <= 64 and type(peer[1]) is int and 0 < peer[1] <= 65535) else None
            receipt = ("owned_loopback_http_v1", phase, selected, connected, attempts, accepted,
                       bytes(observed[:_RESPONSE_HEADERS + _RESPONSE_BODY + 1]))
            try:
                error.owned_loopback_receipt = receipt
            except BaseException:
                pass
            raise


def _capture():
    namespace = globals()
    classes = (("OwnedLoopbackEndpoint", OwnedLoopbackEndpoint),
               ("OwnedLoopbackHTTPTransport", OwnedLoopbackHTTPTransport),
               ("Destination", Destination), ("DispatchEnvelope", DispatchEnvelope))
    functions = []
    for owner in (namespace, OwnedLoopbackEndpoint, OwnedLoopbackHTTPTransport):
        for name, function in tuple((owner if type(owner) is dict else vars(owner)).items()):
            if type(function) is types.FunctionType:
                functions.append((owner, name, function, function.__code__, function.__globals__,
                    function.__defaults__, function.__kwdefaults__, tuple((function.__kwdefaults__ or {}).items())))
    callables = tuple((owner, name, getattr(owner, name), type(getattr(owner, name)),
        (getattr(owner, name).__code__, getattr(owner, name).__globals__, getattr(owner, name).__defaults__,
         getattr(owner, name).__kwdefaults__, tuple((getattr(owner, name).__kwdefaults__ or {}).items()))
        if type(getattr(owner, name)) is types.FunctionType else None) for owner, name in (
        (socket, "AF_INET"), (socket, "SOCK_STREAM"), (socket, "socket"), (socket.socket, "__new__"),
        (socket.socket, "__init__"), (socket.socket, "connect"), (socket.socket, "getpeername"),
        (socket.socket, "send"), (socket.socket, "recv"), (socket.socket, "settimeout"),
        (socket.socket, "close"), (time, "monotonic"), (threading, "get_ident")))
    constants = tuple((name, value, type(value)) for name, value in tuple(namespace.items())
        if name.startswith("_") and not name.startswith("__") and name != "_INTEGRITY")
    layouts = tuple((cls, cls.__slots__, tuple((name, getattr(cls, name)) for name in cls.__slots__),
                     tuple(vars(cls).items()), tuple(getattr(cls, "__dataclass_fields__", {}).items()))
                    for _, cls in classes)
    return tuple(functions), callables, constants, layouts, classes


_ORIGINAL_CODE_CHECK = _code_check
_ORIGINAL_GUARD = _guard
_INTEGRITY = _capture()
