"""Observe one source-pinned stack window when pytest explicitly loads this plugin."""

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import sysconfig
import threading
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "agent/native_main_payload.py"
NATIVE_SHA256 = "d89bf29d21deb65e5dd717bb33bf9cdd633ec47409a554e1c7b1f0076cd4c3dc"
TARGET = "tests/tools/test_native_issued_call_episode.py::test_genuine_issued_call_episode"
PLACEMENT = "tests/tools/test_native_anchor_sampler.py::test_call_hook_placement"
OUTPUT = ROOT / "docs/evidence/accountant-native-anchor-sampling-20261124/sample.json"
TRIGGERS = (("_IssuedCallContinuation.verify", 10963),
            ("_IssuedCallContinuation.pair_proof", 11166))
TRIGGER_HASHES = ("1f0c9df548bab3ef96f8d0ba94afa548f607ddef72ef4f448375b9eea4cb26b6",
                  "80db50f792f40c37aa1b061778c21ee7fe14373007546f4e58767bda04c65f2d")
STDLIB = Path(sysconfig.get_path("stdlib")).resolve()
PUBLIC_STDLIB = frozenset(("threading.py", "contextlib.py", "dataclasses.py", "weakref.py",
                          "inspect.py", "json/encoder.py", "collections/__init__.py"))
_CALL_THREAD = None
_CALL_NODE = None


def pinned_codes():
    """Check the loaded source and return immutable code and class references."""
    data = NATIVE.read_bytes()
    if hashlib.sha256(data).hexdigest() != NATIVE_SHA256:
        raise ValueError("native_source_pin")
    module = sys.modules["agent.native_main_payload"]
    if Path(module.__file__).resolve() != NATIVE:
        raise ValueError("native_source_path")
    tree = ast.parse(data)
    nodes = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == "_IssuedCallContinuation")
    codes = (module._IssuedCallContinuation.verify.__code__,
             module._IssuedCallContinuation.pair_proof.__code__)
    for code, (name, line), digest in zip(codes, TRIGGERS, TRIGGER_HASHES):
        node = next(node for node in nodes.body
                    if isinstance(node, ast.FunctionDef) and node.name == name.split(".")[-1])
        source = ast.get_source_segment(data.decode(), node)
        if (code.co_qualname, code.co_firstlineno, Path(code.co_filename).resolve()) != (name, line, NATIVE):
            raise ValueError("trigger_descriptor")
        if hashlib.sha256(source.encode()).hexdigest() != digest:
            raise ValueError("trigger_source_pin")
    classes = tuple((module.__dict__[node.name], node.name) for node in tree.body
                    if isinstance(node, ast.ClassDef))
    return codes, classes


def anchor_sizes(result, value):
    """Read lengths only from exact builtin containers."""
    rows = []
    if type(result) is list:
        rows.append(("anchors_result_length", len(result)))
    kind = type(value)
    if kind is dict:
        rows.append(("anchors_value_dict_length", len(value)))
    elif kind is list:
        rows.append(("anchors_value_list_length", len(value)))
    elif kind is tuple:
        rows.append(("anchors_value_tuple_length", len(value)))
    return tuple(rows)


def snapshot_sizes(snapshot, classes):
    """Read tuple sizes and compare class references by exact identity."""
    if type(snapshot) is not tuple or len(snapshot) != 3:
        return ()
    obj, pairs, anchors = snapshot
    if type(pairs) is not tuple or type(anchors) is not tuple:
        return ()
    label = "other"
    for cls, name in classes:
        if type(obj) is cls:
            label = name
            break
    return (("snapshot_sizes", label, len(pairs), len(anchors)),)


class Sampler:
    """Collect code positions from one thread without changing its execution."""

    def __init__(self, target_thread, trigger_codes, classes=(), *, window=2.0,
                 interval=0.01, poll_interval=0.05, sample_cap=200, frame_cap=512):
        if not (0 < window <= 2.0 and interval >= 0.01 and poll_interval >= 0.05
                and 0 < sample_cap <= 200 and 0 < frame_cap <= 512):
            raise ValueError("sampler_bounds")
        self.target_thread = target_thread
        self.trigger_codes = trigger_codes
        self.classes = classes
        self.window = window
        self.interval = interval
        self.poll_interval = poll_interval
        self.sample_cap = sample_cap
        self.frame_cap = frame_cap
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, name="native-anchor-observer", daemon=True)
        self.descriptor_cache = {}
        self.descriptors = []
        self.samples = []
        self.top = Counter()
        self.inclusive = Counter()
        self.occurrences = Counter()
        self.depths = Counter()
        self.positions = Counter()
        self.metadata = Counter()
        self.examples = []
        self.errors = []
        self.polls = 0
        self.truncated = 0
        self.start_ns = None
        self.end_ns = None
        self.start_utc_ns = None
        self.end_utc_ns = None
        self.cpu_seconds = 0.0
        self.observer_wall_seconds = 0.0
        self.active_wall_seconds = 0.0
        self.cleanup = False
        self.stop_reason = "not_started"

    def descriptor(self, code):
        if code in self.descriptor_cache:
            return self.descriptor_cache[code]
        filename = Path(code.co_filename)
        if filename.is_absolute() and filename.is_relative_to(ROOT):
            name = filename.relative_to(ROOT).as_posix()
        elif filename.is_absolute() and filename.is_relative_to(STDLIB):
            relative = filename.relative_to(STDLIB).as_posix()
            name = "stdlib/" + relative if relative in PUBLIC_STDLIB else "<external>"
        else:
            name = "<external>"
        index = len(self.descriptors)
        self.descriptor_cache[code] = index
        self.descriptors.append(dict(code=index, filename=name,
                                     qualname=code.co_qualname if name != "<external>" else "<external>",
                                     first_line=code.co_firstlineno if name != "<external>" else None))
        return index

    def safe_metadata(self, frame):
        code = frame.f_code
        if code.co_filename != str(NATIVE):
            return ()
        if code.co_qualname == "_nested_anchors":
            result = frame.f_locals.get("result")
            value = frame.f_locals.get("value")
            try:
                return anchor_sizes(result, value)
            finally:
                del result, value
        if code.co_qualname == "_issued_wire_snapshot_check":
            snapshot = frame.f_locals.get("snapshot")
            try:
                return snapshot_sizes(snapshot, self.classes)
            finally:
                del snapshot
        return ()

    def inspect_target(self, collect):
        """Release all live frames before this method returns."""
        frames = frame = None
        rows, metadata = [], []
        found = set()
        try:
            frames = sys._current_frames()
            frame = frames.get(self.target_thread)
            frames.clear()
            frames = None
            depth = 0
            while frame is not None and depth < self.frame_cap:
                code = frame.f_code
                for index, trigger in enumerate(self.trigger_codes):
                    if code is trigger:
                        found.add(index)
                if collect:
                    index = self.descriptor(code)
                    line = frame.f_lineno if self.descriptors[index]["filename"] != "<external>" else None
                    rows.append((index, line, depth))
                    metadata.extend(self.safe_metadata(frame))
                frame = frame.f_back
                depth += 1
            return len(found) == len(self.trigger_codes), frame is not None, rows, metadata
        finally:
            if frames is not None:
                frames.clear()
            del frame, frames

    def record(self, stamp, truncated, rows, metadata):
        self.samples.append(stamp)
        self.truncated += int(truncated)
        if rows:
            self.top[rows[0][:2]] += 1
        self.inclusive.update({row[0] for row in rows})
        self.occurrences.update(row[0] for row in rows)
        self.depths.update((row[0], row[2]) for row in rows)
        self.positions.update((row[0], row[1]) for row in rows)
        self.metadata.update(metadata)
        if len(self.examples) < 3:
            self.examples.append(dict(timestamp_ns=stamp, truncated=truncated, frames=rows))

    def run(self):
        cpu_start, wall_start = time.thread_time(), time.monotonic()
        try:
            while not self.stop_event.is_set():
                self.polls += 1
                triggered, _, _, _ = self.inspect_target(False)
                if triggered:
                    self.start_ns = time.monotonic_ns()
                    self.start_utc_ns = time.time_ns()
                    break
                self.stop_event.wait(self.poll_interval)
            if self.start_ns is None:
                self.stop_reason = "test_completed_before_trigger"
                return
            deadline = self.start_ns + int(self.window * 1_000_000_000)
            while not self.stop_event.is_set() and len(self.samples) < self.sample_cap:
                stamp = time.monotonic_ns()
                if stamp >= deadline:
                    self.stop_reason = "deadline"
                    break
                active_start = time.monotonic()
                _, truncated, rows, metadata = self.inspect_target(True)
                self.record(stamp, truncated, rows, metadata)
                self.active_wall_seconds += time.monotonic() - active_start
                delay = min(self.interval, max(0, (deadline - time.monotonic_ns()) / 1_000_000_000))
                self.stop_event.wait(delay)
            else:
                self.stop_reason = "sample_cap" if len(self.samples) >= self.sample_cap else "test_completed"
        except BaseException:
            self.errors.append("frame_observation_unavailable_or_denied")
            self.stop_reason = "diagnostic_error"
        finally:
            if self.start_ns is not None:
                self.end_ns = time.monotonic_ns()
                self.end_utc_ns = time.time_ns()
            self.cpu_seconds = time.thread_time() - cpu_start
            self.observer_wall_seconds = time.monotonic() - wall_start
            self.stop_event.set()

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=1.0)
        self.cleanup = not self.thread.is_alive()
        if not self.cleanup:
            self.errors.append("observer_join_timeout")

    def result(self):
        duration = None if self.start_ns is None else (self.end_ns - self.start_ns) / 1_000_000_000
        return dict(schema=1, nodeid=TARGET, plugin_source=str(Path(__file__).resolve()),
                    native_sha256=NATIVE_SHA256, target_thread=self.target_thread,
                    observer_thread=self.thread.ident, python=sys.version, interpreter=sys.executable,
                    worktree=str(ROOT), bounds=dict(window_seconds=self.window, sample_cap=self.sample_cap,
                    frame_cap=self.frame_cap, interval_seconds=self.interval, poll_seconds=self.poll_interval),
                    trigger_descriptors=[dict(qualname=n, first_line=l, source_sha256=h)
                                         for (n, l), h in zip(TRIGGERS, TRIGGER_HASHES)],
                    start_monotonic_ns=self.start_ns, end_monotonic_ns=self.end_ns,
                    start_utc_ns=self.start_utc_ns, end_utc_ns=self.end_utc_ns,
                    actual_window_seconds=duration,
                    overrun_seconds=None if duration is None else max(0, duration-self.window),
                    timestamps_monotonic_ns=self.samples, observations=len(self.samples),
                    pretrigger_poll_count=self.polls, truncated_stacks=self.truncated,
                    code_descriptors=self.descriptors,
                    top_positions=[(*key, count) for key, count in self.top.items()],
                    inclusive_presence=list(self.inclusive.items()),
                    recursive_frame_occurrences=list(self.occurrences.items()),
                    frame_depths=[(*key, count) for key, count in self.depths.items()],
                    all_positions=[(*key, count) for key, count in self.positions.items()],
                    structural_sizes=[(*key, count) for key, count in self.metadata.items()],
                    example_stacks=self.examples, errors=self.errors, cleanup=self.cleanup,
                    stop_reason=self.stop_reason, observer_cpu_seconds=self.cpu_seconds,
                    observer_wall_seconds=self.observer_wall_seconds,
                    observation_active_wall_seconds=self.active_wall_seconds,
                    usable=bool(duration is not None and duration <= 2.25 and len(self.samples) >= 50
                                and not self.errors and self.cleanup))


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_call(item):
    """Start observation inside the call hook and preserve the original outcome."""
    global _CALL_THREAD, _CALL_NODE
    _CALL_THREAD, _CALL_NODE = threading.get_ident(), item.nodeid
    sampler = None
    diagnostic_error = None
    if item.nodeid == TARGET:
        try:
            codes, classes = pinned_codes()
            sampler = Sampler(_CALL_THREAD, codes, classes)
            sampler.start()
        except BaseException:
            diagnostic_error = "sampler_setup_error"
    try:
        yield
    finally:
        if item.nodeid == TARGET:
            try:
                if sampler is not None:
                    sampler.stop()
                    result = sampler.result()
                else:
                    result = dict(nodeid=TARGET, usable=False, errors=[diagnostic_error])
                OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
            except BaseException:
                sys.stderr.write("native anchor diagnostic output failed\n")
        _CALL_THREAD = _CALL_NODE = None


@pytest.hookimpl
def pytest_collection_finish(session):
    if session.config.option.collectonly:
        path = session.config.getoption("--native-anchor-selection")
        if path:
            if [item.nodeid for item in session.items] == [TARGET]:
                pinned_codes()
            Path(path).write_text(json.dumps(dict(plugin_source=str(Path(__file__).resolve()),
                selected_ids=[item.nodeid for item in session.items],
                interpreter=sys.executable, python=sys.version), indent=2) + "\n")


def pytest_addoption(parser):
    parser.addoption("--native-anchor-selection", default=None,
                     help="Write the selected test IDs for this diagnostic.")
