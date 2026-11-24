"""Check observer limits with local fixture threads and no native policy calls."""

from contextlib import contextmanager
import json
from pathlib import Path
import sys
import threading
import time

import pytest

from tests import native_anchor_sampler_plugin as plugin


@contextmanager
def fixture_thread():
    ready = threading.Event()
    release = threading.Event()

    def inner():
        ready.set()
        release.wait(2)

    def outer():
        inner()

    thread = threading.Thread(target=outer)
    thread.start()
    assert ready.wait(1)
    try:
        yield thread, (outer.__code__, inner.__code__)
    finally:
        release.set()
        thread.join(timeout=1)
        assert not thread.is_alive()


def test_call_hook_placement():
    if plugin._CALL_NODE is None:
        pytest.skip("Explicit plugin loading enables the placement check.")
    assert plugin._CALL_NODE == plugin.PLACEMENT
    assert plugin._CALL_THREAD == threading.get_ident()
    assert Path(plugin.__file__).resolve() == plugin.ROOT / "tests/native_anchor_sampler_plugin.py"


def test_target_thread_and_both_trigger_codes():
    with fixture_thread() as (thread, codes):
        sampler = plugin.Sampler(thread.ident, codes)
        matched, truncated, rows, metadata = sampler.inspect_target(False)
        assert matched and not truncated and rows == metadata == []
        assert not sampler.descriptors and not sampler.samples
        wrong_thread = plugin.Sampler(threading.get_ident(), codes)
        assert not wrong_thread.inspect_target(False)[0]
        wrong_code = plugin.Sampler(thread.ident, (codes[0], test_target_thread_and_both_trigger_codes.__code__))
        assert not wrong_code.inspect_target(False)[0]


def test_frame_cap_and_code_only_external_filter():
    with fixture_thread() as (thread, codes):
        sampler = plugin.Sampler(thread.ident, codes, frame_cap=1)
        _, truncated, rows, metadata = sampler.inspect_target(True)
        assert truncated and len(rows) == 1 and metadata == []
        assert rows[0][2] == 0
        source = compile("def external(): pass", "/private/no-profile.py", "exec")
        code = next(value for value in source.co_consts if type(value) is type(source))
        descriptor = sampler.descriptors[sampler.descriptor(code)]
        assert descriptor["filename"] == descriptor["qualname"] == "<external>"
        assert descriptor["first_line"] is None
        assert "/private/" not in json.dumps(sampler.descriptors)
        assert sampler.safe_metadata(sys._getframe()) == ()


def test_deadline_sample_cap_and_cleanup(tmp_path):
    with fixture_thread() as (thread, codes):
        sampler = plugin.Sampler(thread.ident, codes, window=0.055,
                                 output_path=tmp_path / "partial.json")
        sampler.start()
        sampler.thread.join(timeout=1)
        sampler.stop()
        partial = json.loads((tmp_path / "partial.json").read_text())
        assert partial["observations"] == len(sampler.samples) and not partial["cleanup"]
        result = sampler.result()
        assert result["cleanup"] and not result["errors"]
        assert result["stop_reason"] == "deadline"
        assert 1 <= result["observations"] <= 6
        assert result["actual_window_seconds"] < 0.5
        stamps = result["timestamps_monotonic_ns"]
        assert all(b-a >= 10_000_000 for a, b in zip(stamps, stamps[1:]))
        capped = plugin.Sampler(thread.ident, codes, window=1, sample_cap=2)
        capped.start()
        capped.thread.join(timeout=1)
        capped.stop()
        assert len(capped.samples) == 2 and capped.stop_reason == "sample_cap" and capped.cleanup


def test_stop_before_trigger_and_observer_error():
    with fixture_thread() as (thread, codes):
        sampler = plugin.Sampler(thread.ident, (test_stop_before_trigger_and_observer_error.__code__,))
        sampler.start()
        time.sleep(0.015)
        sampler.stop()
        assert sampler.cleanup and sampler.start_ns is None and sampler.polls == 1
        assert not sampler.samples and not sampler.descriptors

    class DeniedObserver(plugin.Sampler):
        def inspect_target(self, collect):
            raise PermissionError("Do not retain this value.")

    denied = DeniedObserver(threading.get_ident(), codes)
    denied.start()
    denied.thread.join(timeout=1)
    denied.stop()
    assert denied.cleanup and denied.stop_event.is_set()
    assert denied.errors == ["frame_observation_unavailable_or_denied"]
    assert "Do not retain" not in json.dumps(denied.result())


def test_exact_builtin_metadata_without_custom_protocols():
    class Hostile(list):
        def __len__(self):
            raise AssertionError("custom length")

        def __iter__(self):
            raise AssertionError("custom iteration")

        def __repr__(self):
            raise AssertionError("custom representation")

    class HostileTuple(tuple):
        def __len__(self):
            raise AssertionError("custom length")

    class NativeLabel:
        pass

    value = Hostile()
    assert plugin.anchor_sizes(value, value) == ()
    assert plugin.anchor_sizes([value], {"unread": value}) == (
        ("anchors_result_length", 1), ("anchors_value_dict_length", 1))
    assert plugin.anchor_sizes(None, (value, value)) == (("anchors_value_tuple_length", 2),)
    assert plugin.anchor_sizes(None, [value]) == (("anchors_value_list_length", 1),)
    labels = ((NativeLabel, "NativeLabel"),)
    assert plugin.snapshot_sizes((NativeLabel(), (value,), (value, value)), labels) == (
        ("snapshot_sizes", "NativeLabel", 1, 2),)
    assert plugin.snapshot_sizes(HostileTuple(), labels) == ()
    assert plugin.snapshot_sizes((value, value, ()), labels) == ()
    assert plugin.snapshot_sizes((value, (), ()), labels) == (("snapshot_sizes", "other", 0, 0),)
    assert plugin.snapshot_sizes((value, ()), labels) == ()


def test_recursive_occurrences_are_separate_from_presence():
    sampler = plugin.Sampler(threading.get_ident(), ())
    sampler.record(1, True, [(0, 10, 0), (0, 11, 1), (1, 20, 2)], [])
    assert sampler.inclusive == {0: 1, 1: 1}
    assert sampler.occurrences == {0: 2, 1: 1}
    assert sampler.depths == {(0, 0): 1, (0, 1): 1, (1, 2): 1}
    assert sampler.top == {(0, 10): 1} and sampler.truncated == 1


@pytest.mark.parametrize("kwargs", [{"window": 2.1}, {"interval": 0.001},
                         {"poll_interval": 0.001}, {"sample_cap": 201}, {"frame_cap": 513}])
def test_invalid_bounds(kwargs):
    with pytest.raises(ValueError, match="sampler_bounds"):
        plugin.Sampler(threading.get_ident(), (), **kwargs)


def test_call_hook_preserves_original_exception():
    class Item:
        nodeid = "fixture_only::outcome"

    error = RuntimeError("original outcome")
    hook = plugin.pytest_runtest_call(Item())
    next(hook)
    with pytest.raises(RuntimeError) as raised:
        hook.throw(error)
    assert raised.value is error
    assert plugin._CALL_THREAD is plugin._CALL_NODE is None


@pytest.mark.parametrize("scenario,expected", [
    ("project", "pkg/observed.py"),
    ("selected_stdlib", "stdlib/dataclasses.py"),
    ("unselected_stdlib", "<external>"),
    ("symlink_escape", "<external>"),
    ("nested_escape", "<external>"),
    ("prefix_collision", "<external>"),
    ("relative", "<external>"),
    ("resolution_loop", "<external>"),
    ("missing_file", "<external>"),
    ("in_root_alias", "pkg/observed.py"),
    ("root_alias", "pkg/observed.py"),
    ("stdlib_root_alias", "stdlib/dataclasses.py"),
    ("selected_stdlib_escape", "<external>"),
])
def test_canonical_descriptor_scope(tmp_path, monkeypatch, scenario, expected):
    worktree, stdlib, outside = (tmp_path / name for name in ("tree", "stdlib", "outside"))
    for directory in (worktree / "pkg", stdlib, outside, tmp_path / "tree-neighbor"):
        directory.mkdir(parents=True)
    project = worktree / "pkg/observed.py"
    selected = stdlib / "dataclasses.py"
    unselected = stdlib / "unselected.py"
    external = outside / "dependency.py"
    neighbor = tmp_path / "tree-neighbor/observed.py"
    source = "def observed():\n    return None\n"
    for path in (project, selected, unselected, external, neighbor):
        path.write_text(source)
    filename = project
    if scenario == "selected_stdlib":
        filename = selected
    elif scenario == "unselected_stdlib":
        filename = unselected
    elif scenario == "symlink_escape":
        (worktree / "escape").symlink_to(outside, target_is_directory=True)
        filename = worktree / "escape/dependency.py"
    elif scenario == "nested_escape":
        (worktree / "inner").symlink_to(outside, target_is_directory=True)
        (worktree / "outer").symlink_to(worktree / "inner", target_is_directory=True)
        filename = worktree / "outer/dependency.py"
    elif scenario == "prefix_collision":
        filename = neighbor
    elif scenario == "relative":
        filename = Path("pkg/observed.py")
    elif scenario == "resolution_loop":
        filename = worktree / "loop.py"
        filename.symlink_to(filename)
    elif scenario == "missing_file":
        filename = worktree / "missing.py"
    elif scenario == "in_root_alias":
        filename = worktree / "alias.py"
        filename.symlink_to(project)
    elif scenario == "root_alias":
        alias = tmp_path / "tree-alias"
        alias.symlink_to(worktree, target_is_directory=True)
        worktree = alias
        filename = alias / "pkg/observed.py"
    elif scenario == "stdlib_root_alias":
        alias = tmp_path / "stdlib-alias"
        alias.symlink_to(stdlib, target_is_directory=True)
        stdlib = alias
        filename = alias / "dataclasses.py"
    elif scenario == "selected_stdlib_escape":
        selected.unlink()
        selected.symlink_to(external)
        filename = selected
    monkeypatch.setattr(plugin, "ROOT", worktree)
    monkeypatch.setattr(plugin, "STDLIB", stdlib)
    module = compile(source, str(filename), "exec")
    code = next(value for value in module.co_consts if type(value) is type(module))
    sampler = plugin.Sampler(threading.get_ident(), ())
    index = sampler.descriptor(code)
    descriptor = sampler.descriptors[index]
    assert descriptor["filename"] == expected
    if expected == "<external>":
        assert descriptor["qualname"] == "<external>" and descriptor["first_line"] is None
    else:
        assert descriptor["qualname"] == "observed" and descriptor["first_line"] == 1
    assert sampler.descriptor(code) == index and len(sampler.descriptors) == 1
