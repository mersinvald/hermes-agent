"""Record owned TCP bytes without client parser or policy imports."""

from dataclasses import dataclass
import hashlib
import socket
import selectors
import threading
import time


@dataclass(frozen=True, slots=True)
class ResponseStep:
    path: bytes
    response: bytes
    body: bytes | None = None
    hold_seconds: float = 0.0
    allow_disconnect: bool = False


def response_bytes(body):
    return b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode("ascii") + b"\r\nConnection: close\r\n\r\n" + body


@dataclass(frozen=True, slots=True)
class ReceiverWindowMark:
    bound: tuple
    listener_id: int
    started_at: float
    row_count: int
    active_count: int
    polls: int


@dataclass(frozen=True, slots=True)
class _ReceiverConnection:
    order: int
    bound: tuple
    local: tuple
    peer: tuple
    listener_fd: int
    accepted_at: float
    closed_at: float | None
    raw: bytes
    raw_sha256: str
    raw_length: int
    header: bytes
    body: bytes
    body_sha256: str
    body_length: int
    path: bytes | None
    host: bytes | None
    client_reported_peer: bytes | None
    complete: bool
    settled: bool
    response_attempts: int
    response_accepted: int
    response: bytes
    status: str
    failure: str | None


@dataclass(frozen=True, slots=True)
class ReceiverWindowReceipt:
    bound: tuple
    listener_id: int
    ready_at: float | None
    deadline_at: float | None
    mark: ReceiverWindowMark | None
    finish_requested_at: float | None
    drain_at: float | None
    cutoff_at: float | None
    finished_at: float
    polls: int
    connections: tuple
    errors: tuple
    thread_stopped: bool
    listener_closed: bool
    sockets_closed: bool


class OwnedReceiver:
    def __init__(self, *, observation_seconds: float | None = None):
        if observation_seconds is not None:
            assert type(observation_seconds) is float and 0 < observation_seconds <= 590.0
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.listener.bind(("127.0.0.1", 0))
            self.bound = self.listener.getsockname()
            assert type(self.bound[0]) is str and self.bound[0] == "127.0.0.1"
            assert type(self.bound[1]) is int and 0 < self.bound[1] < 65536
            self.listener.listen(32)
            self.listener.settimeout(0.1)
        except BaseException:
            self.listener.close()
            raise
        self.steps = None
        self.rows, self.errors = [], []
        self.ready, self.stop = threading.Event(), threading.Event()
        self.condition = threading.Condition()
        self.thread = None
        self.active = None
        self.observation_seconds = observation_seconds
        self._connections = {}
        self._selector = None
        self._listener_id = id(self.listener)
        self._listener_fd = self.listener.fileno()
        self._ready_at = self._deadline_at = None
        self._polls = 0
        self._mark = None
        self._finish_requested_at = self._drain_at = self._cutoff_at = None
        self._last_activity = None
        self._settle_seconds = 0.5
        self._receipt = None

    def start(self, steps):
        assert self.thread is None and type(steps) is tuple
        assert all(type(step) is ResponseStep and type(step.response) is bytes
                   and type(step.path) is bytes and (step.body is None or type(step.body) is bytes)
                   and type(step.hold_seconds) is float and 0 <= step.hold_seconds <= 6.0
                   and type(step.allow_disconnect) is bool for step in steps)
        self.steps = steps
        self.thread = threading.Thread(target=self._serve, name="owned-loopback-receiver", daemon=False)
        self.thread.start()
        assert self.ready.wait(5.0), "Receiver start timeout"
        self.check()
        return self

    def _serve(self):
        if self.observation_seconds is not None:
            return self._serve_observed()
        self.ready.set()
        try:
            while not self.stop.is_set():
                try:
                    connection, peer = self.listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self.stop.is_set():
                        break
                    raise
                self.active = connection
                step = self.steps[len(self.rows)] if len(self.rows) < len(self.steps) else None
                row = dict(order=len(self.rows), peer=peer, local=connection.getsockname(),
                    bound=self.bound, header=b"", body=b"", raw=b"", complete=False,
                    response_attempts=0, response_accepted=0, failure=None, settled=False)
                with self.condition:
                    self.rows.append(row)
                    self.condition.notify_all()
                try:
                    assert row["local"] == self.bound
                    connection.settimeout(5.0)
                    incoming = b""
                    while b"\r\n\r\n" not in incoming:
                        chunk = connection.recv(min(4096, 2049 - len(incoming)))
                        incoming += chunk
                        row["raw"] = incoming
                        if not chunk:
                            break
                        assert len(incoming) <= 2048 or b"\r\n\r\n" in incoming
                    if b"\r\n\r\n" not in incoming:
                        row["failure"] = "incomplete_headers"
                        continue
                    head, body = incoming.split(b"\r\n\r\n", 1)
                    row["header"], row["body"] = head + b"\r\n\r\n", body
                    lines = head.split(b"\r\n")
                    fields = dict(line.split(b": ", 1) for line in lines[1:])
                    assert lines[0].startswith(b"POST ") and lines[0].endswith(b" HTTP/1.1")
                    assert set(fields) == {b"Host", b"X-Owned-Peer", b"Content-Length", b"Connection", b"Content-Type"}
                    assert fields[b"Host"] == ("127.0.0.1:" + str(self.bound[1])).encode("ascii")
                    assert fields[b"Connection"] == b"close" and fields[b"Content-Type"] == b"application/octet-stream"
                    length = int(fields[b"Content-Length"])
                    assert 0 <= length <= 262144
                    row["host"] = fields[b"Host"]
                    row["client_reported_peer"] = fields[b"X-Owned-Peer"]
                    row["path"] = lines[0].split(b" ")[1]
                    while len(body) < length:
                        chunk = connection.recv(min(4096, length + 1 - len(body)))
                        body += chunk
                        row["body"] = body
                        row["raw"] = row["header"] + body
                        if not chunk:
                            break
                    if len(body) != length:
                        row["failure"] = "incomplete_body"
                        continue
                    row["complete"] = True
                    row["body_sha256"] = hashlib.sha256(body).hexdigest()
                    row["body_length"] = len(body)
                    assert row["order"] < len(self.steps), "Unexpected request"
                    step = self.steps[row["order"]]
                    assert row["path"] == step.path, "Unexpected request path"
                    if step.body is not None:
                        assert body == step.body, "Unexpected request body"
                    if step.hold_seconds:
                        self.stop.wait(step.hold_seconds)
                    row["response_attempts"] = 1
                    offset = 0
                    while offset < len(step.response):
                        count = connection.send(step.response[offset:])
                        assert count > 0
                        offset += count
                        row["response_accepted"] = offset
                except (BrokenPipeError, ConnectionResetError, socket.timeout) as error:
                    row["failure"] = type(error).__name__
                    if step is None or not step.allow_disconnect:
                        self.errors.append(error)
                except BaseException as error:
                    row["failure"] = type(error).__name__
                    self.errors.append(error)
                finally:
                    row["raw_sha256"] = hashlib.sha256(row["raw"]).hexdigest()
                    row["raw_length"] = len(row["raw"])
                    with self.condition:
                        row["settled"] = True
                        self.condition.notify_all()
                    connection.close()
                    self.active = None
        except BaseException as error:
            self.errors.append(error)
        finally:
            self.listener.close()

    def check(self):
        if self.errors:
            raise self.errors[0]

    def wait(self, count):
        deadline = time.monotonic() + 5.0
        with self.condition:
            while len(self.rows) < count or not all(row["settled"] for row in self.rows[:count]):
                self.check()
                remaining = deadline - time.monotonic()
                assert remaining > 0, "Receiver receipt timeout"
                self.condition.wait(min(remaining, 0.1))
        self.check()
        return tuple(self.rows)

    def close(self):
        if self.observation_seconds is not None:
            return self._close_observed()
        self.stop.set()
        self.listener.close()
        if self.active is not None:
            try:
                self.active.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self.thread is not None:
            self.thread.join(5.0)
            assert not self.thread.is_alive(), "Receiver cleanup timeout"
        assert self.active is None
        assert self.listener.fileno() == -1
        self.check()

    def mark_window(self) -> ReceiverWindowMark:
        assert self.observation_seconds is not None
        with self.condition:
            self.check()
            assert self._mark is None and self._receipt is None
            assert self.ready.is_set() and self.thread.is_alive()
            assert len(self.rows) >= len(self.steps)
            assert all(row["settled"] for row in self.rows) and not self._connections
            assert not self._selector.select(0), "Receiver backlog at mark"
            self._polls += 1
            self._mark = ReceiverWindowMark(self.bound, self._listener_id, time.monotonic(),
                len(self.rows), len(self._connections), self._polls)
            return self._mark

    def finish(self, *, settle_seconds: float = 0.5) -> ReceiverWindowReceipt:
        assert self.observation_seconds is not None
        assert type(settle_seconds) is float and 0.5 <= settle_seconds <= 1.0
        with self.condition:
            if self._receipt is not None:
                return self._receipt
            if self._finish_requested_at is None:
                self._finish_requested_at = time.monotonic()
                self._settle_seconds = settle_seconds
            self.condition.notify_all()
            deadline = time.monotonic() + 5.0
            while self._cutoff_at is None and self.thread is not None and self.thread.is_alive():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.errors.append(TimeoutError("Receiver drain timeout"))
                    break
                self.condition.wait(min(remaining, 0.1))
        self._close_observed()
        with self.condition:
            rows = tuple(_ReceiverConnection(**{name: row[name] for name in _ReceiverConnection.__dataclass_fields__})
                         for row in self.rows)
            self._receipt = ReceiverWindowReceipt(self.bound, self._listener_id, self._ready_at,
                self._deadline_at, self._mark, self._finish_requested_at, self._drain_at, self._cutoff_at,
                time.monotonic(), self._polls, rows,
                tuple((type(error).__name__, str(error)) for error in self.errors),
                self.thread is None or not self.thread.is_alive(), self.listener.fileno() == -1,
                not self._connections)
            return self._receipt

    def _close_observed(self):
        self.stop.set()
        if self.thread is None:
            self.listener.close()
            return
        self.thread.join(5.0)
        if self.thread.is_alive():
            with self.condition:
                self.errors.append(TimeoutError("Receiver cleanup timeout"))
            raise self.errors[-1]

    def _observed_settle(self, connection, status, failure=None):
        row = self._connections.pop(connection)
        self._selector.unregister(connection)
        connection.close()
        row.update(status=status, failure=failure, settled=True, closed_at=time.monotonic())
        row.update(raw_length=len(row["raw"]), raw_sha256=hashlib.sha256(row["raw"]).hexdigest(),
                   body_length=len(row["body"]), body_sha256=hashlib.sha256(row["body"]).hexdigest())
        self._last_activity = row["closed_at"]
        self.condition.notify_all()

    def _observed_read(self, connection, row):
        step = self.steps[row["order"]] if row["order"] < len(self.steps) else None
        chunk = connection.recv(min(4096, 264193 - len(row["raw"])))
        if not chunk:
            if row["complete"] and step is not None and step.allow_disconnect:
                self._observed_settle(connection, "planned_disconnect", "eof_after_request")
            else:
                reason = "eof_empty" if not row["raw"] else "eof_partial_request"
                raise ValueError(reason)
            return
        row["raw"] += chunk
        row["progress_at"] = self._last_activity = time.monotonic()
        assert len(row["raw"]) < 264193, "Receiver connection byte cap"
        assert sum(len(item["raw"]) for item in self.rows) <= 2113544, "Receiver aggregate byte cap"
        if row["complete"]:
            raise ValueError("Receiver bytes after complete request")
        if not row["header"]:
            if b"\r\n\r\n" not in row["raw"]:
                assert len(row["raw"]) <= 2048, "Receiver header cap"
                return
            head, body = row["raw"].split(b"\r\n\r\n", 1)
            row["header"], row["body"] = head + b"\r\n\r\n", body
            assert len(row["header"]) <= 2048, "Receiver header cap"
            lines = head.split(b"\r\n")
            fields = dict(line.split(b": ", 1) for line in lines[1:])
            assert lines[0].startswith(b"POST ") and lines[0].endswith(b" HTTP/1.1")
            assert set(fields) == {b"Host", b"X-Owned-Peer", b"Content-Length", b"Connection", b"Content-Type"}
            assert fields[b"Host"] == ("127.0.0.1:" + str(self.bound[1])).encode("ascii")
            assert fields[b"Connection"] == b"close" and fields[b"Content-Type"] == b"application/octet-stream"
            row.update(host=fields[b"Host"], client_reported_peer=fields[b"X-Owned-Peer"],
                       path=lines[0].split(b" ")[1], expected_length=int(fields[b"Content-Length"]))
            assert 0 <= row["expected_length"] <= 262144
        else:
            row["body"] = row["raw"][len(row["header"]):]
        assert len(row["body"]) <= row["expected_length"], "Receiver excess body"
        if len(row["body"]) == row["expected_length"]:
            row["complete"] = True
            assert step is not None, "Unexpected request"
            assert row["path"] == step.path, "Unexpected request path"
            if step.body is not None:
                assert row["body"] == step.body, "Unexpected request body"
            row["hold_until"] = time.monotonic() + step.hold_seconds
            row["status"] = "silent_hold" if step.hold_seconds else "response"

    def _serve_observed(self):
        try:
            with selectors.DefaultSelector() as selector:
                with self.condition:
                    self._selector = selector
                    self.listener.setblocking(False)
                    selector.register(self.listener, selectors.EVENT_READ)
                    started = self._last_activity = time.monotonic()
                    self._deadline_at = started + self.observation_seconds
                while not self.stop.is_set():
                    with self.condition:
                        events = selector.select(0.05)
                        self._polls += 1
                        now = time.monotonic()
                        if self._ready_at is None:
                            self._ready_at = now
                            self.ready.set()
                            self.condition.notify_all()
                        assert now < self._deadline_at, "Receiver observation deadline"
                        for key, mask in events:
                            connection = key.fileobj
                            if connection is self.listener:
                                assert len(self.rows) < 8, "Receiver extra accept pending at connection cap"
                                connection, peer = self.listener.accept()
                                connection.setblocking(False)
                                row = dict(order=len(self.rows), bound=self.bound, local=connection.getsockname(),
                                    peer=peer, listener_fd=self._listener_fd, accepted_at=time.monotonic(), closed_at=None,
                                    raw=b"", raw_sha256=hashlib.sha256(b"").hexdigest(), raw_length=0,
                                    header=b"", body=b"", body_sha256=hashlib.sha256(b"").hexdigest(), body_length=0,
                                    path=None, host=None, client_reported_peer=None, complete=False, settled=False,
                                    response_attempts=0, response_accepted=0, response=b"", status="read", failure=None,
                                    progress_at=time.monotonic(), hold_until=None)
                                self.rows.append(row)
                                self._connections[connection] = row
                                selector.register(connection, selectors.EVENT_READ)
                                self._last_activity = row["accepted_at"]
                                self.condition.notify_all()
                                assert row["local"] == self.bound
                                continue
                            row = self._connections[connection]
                            try:
                                if mask & selectors.EVENT_READ:
                                    self._observed_read(connection, row)
                                if connection in self._connections and mask & selectors.EVENT_WRITE:
                                    step = self.steps[row["order"]]
                                    row["response_attempts"] = 1
                                    count = connection.send(step.response[row["response_accepted"]:])
                                    assert count > 0
                                    row["response_accepted"] += count
                                    row["response"] = step.response[:row["response_accepted"]]
                                    row["progress_at"] = self._last_activity = time.monotonic()
                                    if row["response_accepted"] == len(step.response):
                                        self._observed_settle(connection, "response_sent")
                            except BlockingIOError:
                                pass
                            except BaseException as error:
                                step = self.steps[row["order"]] if row["order"] < len(self.steps) else None
                                planned = row["complete"] and step is not None and step.allow_disconnect and isinstance(
                                    error, (BrokenPipeError, ConnectionResetError))
                                if not planned:
                                    self.errors.append(error)
                                self._observed_settle(connection, "planned_disconnect" if planned else "error", str(error))
                        now = time.monotonic()
                        for connection, row in tuple(self._connections.items()):
                            if row["status"] == "silent_hold":
                                if now < row["hold_until"]:
                                    continue
                                row["status"], row["progress_at"] = "response", now
                            if row["status"] == "response":
                                step = self.steps[row["order"]]
                                if not step.response:
                                    row["response_attempts"] = 1
                                    self._observed_settle(connection, "empty_response")
                                    continue
                                selector.modify(connection, selectors.EVENT_READ | selectors.EVENT_WRITE)
                            if now - row["progress_at"] >= 5.0:
                                self.errors.append(TimeoutError("Receiver progress timeout"))
                                self._observed_settle(connection, "error", "progress_timeout")
                        if (self._finish_requested_at is not None and not self._connections and not events
                                and now >= max(self._finish_requested_at, self._last_activity) + self._settle_seconds):
                            final = selector.select(0)
                            self._polls += 1
                            if not final:
                                self._drain_at = time.monotonic()
                                self._cutoff_at = time.monotonic()
                                self.condition.notify_all()
                                break
                        self.condition.wait(0.001)
        except BaseException as error:
            with self.condition:
                self.errors.append(error)
        finally:
            with self.condition:
                for connection, row in tuple(self._connections.items()):
                    connection.close()
                    row.update(settled=True, closed_at=time.monotonic(), status="cleanup_incomplete",
                        failure="incomplete_cutoff", raw_length=len(row["raw"]),
                        raw_sha256=hashlib.sha256(row["raw"]).hexdigest(), body_length=len(row["body"]),
                        body_sha256=hashlib.sha256(row["body"]).hexdigest())
                    self.errors.append(RuntimeError("Receiver pending socket at cleanup"))
                self._connections.clear()
                self.listener.close()
                if self._cutoff_at is None:
                    self.errors.append(RuntimeError("Receiver incomplete cutoff"))
                self.condition.notify_all()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
