"""Observe owned TCP connections without policy or model callbacks."""

from dataclasses import dataclass
import selectors
import socket
import threading
import time


@dataclass(frozen=True, slots=True)
class ConnectionObservation:
    order: int
    accepted_at: float
    peer: tuple[str, int]
    local: tuple[str, int]
    raw: bytes
    headers_complete: bool
    request_complete: bool
    closed_at: float | None
    error_code: str | None


@dataclass(frozen=True, slots=True)
class ForbiddenObservation:
    bound: tuple[str, int]
    started_at: float | None
    ready_at: float | None
    last_poll_at: float | None
    deadline_at: float | None
    finished_at: float
    polls: int
    connections: tuple[ConnectionObservation, ...]
    errors: tuple[str, ...]
    thread_stopped: bool
    listener_closed: bool
    sockets_closed: bool


def _request_state(raw):
    if b"\r\n\r\n" not in raw:
        return False, False, None
    head, body = raw.split(b"\r\n\r\n", 1)
    lines = head.split(b"\r\n")
    lengths = [line.split(b":", 1)[1].strip() for line in lines[1:]
               if line.lower().startswith(b"content-length:")]
    if len(lengths) != 1 or not lengths[0].isdigit():
        return True, False, "invalid_headers"
    length = int(lengths[0]) if len(lengths[0]) <= 9 else -1
    if length < 0:
        return True, False, "invalid_length"
    if len(body) > length:
        return True, False, "extra_bytes"
    return True, len(body) == length, None


class ForbiddenReceiver:
    def __init__(self, *, window_seconds: float = 590.0, max_connections: int = 8,
                 max_bytes: int = 264192):
        if not (0 < window_seconds <= 590.0 and type(max_connections) is int
                and 0 < max_connections <= 8 and type(max_bytes) is int and 0 < max_bytes <= 264192):
            raise ValueError("observer_bounds")
        self.window_seconds, self.max_connections, self.max_bytes = window_seconds, max_connections, max_bytes
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.listener.bind(("127.0.0.1", 0))
            self.bound = self.listener.getsockname()
            self.listener.listen(32)
            self.listener.setblocking(False)
        except BaseException:
            self.listener.close()
            raise
        self.thread = None
        self.ready, self.stop, self.drain = threading.Event(), threading.Event(), threading.Event()
        self.lock = threading.Lock()
        self.finish_requested_at = self.settle_seconds = None
        self.started_at = self.ready_at = self.last_poll_at = self.deadline_at = None
        self.cutoff_at = self.drain_at = None
        self.cleanup_started_at = self.cleanup_finished_at = None
        self.polls = 0
        self.rows, self.errors, self.thread_errors = [], [], []
        self.active = {}
        self.sockets = []
        self.final = None
        self.last_activity_at = None

    def start(self) -> "ForbiddenReceiver":
        if self.thread is not None or self.listener.fileno() == -1:
            raise RuntimeError("observer_start_state")
        self.started_at = time.monotonic()
        self.last_activity_at = self.started_at
        self.deadline_at = self.started_at + self.window_seconds
        self.thread = threading.Thread(target=self._serve, name="forbidden-loopback-observer", daemon=False)
        self.thread.start()
        if not self.ready.wait(5.0) or not self.thread.is_alive():
            self.errors.append("missing_readiness")
            self.close()
            raise RuntimeError("missing_readiness")
        return self

    def _close_connection(self, selector, connection, code):
        row = self.active.pop(connection)
        if code is not None:
            row["error_code"] = code
        try:
            selector.unregister(connection)
        finally:
            connection.close()
            row["closed_at"] = time.monotonic()

    def _accept(self, selector):
        while True:
            try:
                connection, peer = self.listener.accept()
            except BlockingIOError:
                return
            accepted_at = time.monotonic()
            self.sockets.append(connection)
            if len(self.rows) >= self.max_connections:
                connection.close()
                self.errors.append("connection_cap_extra_accept")
                self.stop.set()
                return
            row = dict(order=len(self.rows), accepted_at=accepted_at, peer=peer,
                       local=connection.getsockname(), raw=b"", headers_complete=False,
                       request_complete=False, closed_at=None, error_code=None)
            self.rows.append(row)
            self.active[connection] = row
            self.last_activity_at = accepted_at
            connection.setblocking(False)
            selector.register(connection, selectors.EVENT_READ)

    def _read(self, selector, connection):
        row = self.active[connection]
        while True:
            try:
                chunk = connection.recv(min(4096, self.max_bytes + 1 - len(row["raw"])))
            except BlockingIOError:
                return
            row["raw"] += chunk
            if chunk:
                self.last_activity_at = time.monotonic()
            header, complete, code = _request_state(row["raw"])
            row["headers_complete"], row["request_complete"] = header, complete
            if len(row["raw"]) > self.max_bytes:
                self.errors.append("byte_cap")
                self._close_connection(selector, connection, "byte_cap")
                return
            if code is not None:
                self._close_connection(selector, connection, code)
                return
            if complete:
                self._close_connection(selector, connection, None)
                return
            if not chunk:
                code = "eof_partial_body" if header else "eof_partial_headers" if row["raw"] else "eof_empty"
                self._close_connection(selector, connection, code)
                return

    def _serve(self):
        selector = selectors.DefaultSelector()
        try:
            selector.register(self.listener, selectors.EVENT_READ)
            while not self.stop.is_set():
                events = selector.select(0.05)
                self.last_poll_at = time.monotonic()
                self.polls += 1
                if self.ready_at is None:
                    self.ready_at = self.last_poll_at
                    self.ready.set()
                for key, _ in events:
                    if key.fileobj is self.listener:
                        self._accept(selector)
                    elif key.fileobj in self.active:
                        self._read(selector, key.fileobj)
                now = time.monotonic()
                for connection, row in tuple(self.active.items()):
                    if now >= min(row["accepted_at"] + 5.0, self.deadline_at):
                        self.errors.append("connection_timeout")
                        self._close_connection(selector, connection, "timeout")
                if now >= self.deadline_at:
                    self.errors.append("window_timeout")
                    break
                with self.lock:
                    requested, settle = self.finish_requested_at, self.settle_seconds
                if requested is not None and now >= max(requested, self.last_activity_at) + settle:
                    self._accept(selector)
                    for connection in tuple(self.active):
                        self._read(selector, connection)
                    now = time.monotonic()
                    if now < max(requested, self.last_activity_at) + settle:
                        continue
                    if self.active:
                        self.errors.append("pending_at_drain")
                        for row in self.active.values():
                            row["error_code"] = "unknown_pending_at_cutoff"
                    if self.stop.is_set() or now >= self.deadline_at:
                        self.errors.append("drain_incomplete")
                        break
                    self.drain_at = now
                    self.cutoff_at = time.monotonic()
                    self.drain.set()
                    break
        except BaseException as error:
            self.errors.append("observer_error")
            self.thread_errors.append(error)
        finally:
            self.cleanup_started_at = time.monotonic()
            for connection in tuple(self.active):
                try:
                    row = self.active[connection]
                    self._close_connection(selector, connection, row["error_code"] or "unknown_cleanup")
                except BaseException as error:
                    self.errors.append("socket_cleanup_error")
                    self.thread_errors.append(error)
            try:
                selector.close()
                self.listener.close()
            except BaseException as error:
                self.errors.append("listener_cleanup_error")
                self.thread_errors.append(error)
            self.cleanup_finished_at = time.monotonic()

    def finish(self, *, settle_seconds: float = 0.5) -> ForbiddenObservation:
        if self.final is not None:
            return self.final
        if not (0.5 <= settle_seconds <= 1.0):
            raise ValueError("observer_settle_bounds")
        with self.lock:
            self.finish_requested_at = time.monotonic()
            self.settle_seconds = settle_seconds
        limit = time.monotonic() + 5.0
        while not self.drain.wait(0.05):
            if self.thread is None or not self.thread.is_alive() or time.monotonic() >= limit:
                self.errors.append("missing_drain_ack")
                break
        self.close()
        if self.ready_at is None:
            self.errors.append("missing_readiness")
        if self.cutoff_at is None or self.drain_at is None or self.cutoff_at >= self.deadline_at:
            self.errors.append("unknown_cutoff")
        self.final = ForbiddenObservation(self.bound, self.started_at, self.ready_at, self.last_poll_at,
            self.deadline_at, time.monotonic(), self.polls,
            tuple(ConnectionObservation(**row) for row in self.rows), tuple(self.errors),
            self.thread is not None and not self.thread.is_alive(), self.listener.fileno() == -1,
            all(connection.fileno() == -1 for connection in self.sockets))
        return self.final

    def close(self) -> None:
        if self.final is not None:
            return
        started = time.monotonic()
        self.stop.set()
        if self.thread is not None:
            self.thread.join(max(0.0, 5.0 - (time.monotonic() - started)))
        else:
            self.cleanup_started_at = time.monotonic()
            self.listener.close()
            self.cleanup_finished_at = time.monotonic()
        if self.thread is not None and self.thread.is_alive():
            self.errors.append("cleanup_timeout")
            self.listener.close()
            for connection in self.sockets:
                connection.close()
            raise RuntimeError("observer_cleanup_timeout")
        if self.listener.fileno() != -1 or any(connection.fileno() != -1 for connection in self.sockets):
            self.errors.append("cleanup_incomplete")
            raise RuntimeError("observer_cleanup_incomplete")
