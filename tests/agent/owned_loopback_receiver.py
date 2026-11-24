"""Record owned TCP bytes without client parser or policy imports."""

from dataclasses import dataclass
import hashlib
import socket
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


class OwnedReceiver:
    def __init__(self):
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

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
