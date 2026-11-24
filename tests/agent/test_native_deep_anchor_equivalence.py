"""Compare complete anchor walks and snapshot checks by identity."""

import runpy
from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import native_main_payload as native

EVIDENCE = Path(__file__).resolve().parents[2] / "docs/evidence/accountant-native-deep-anchor-index-20261124"
SUPPORT = runpy.run_path(str(EVIDENCE / "fixtures.py"))


@pytest.fixture
def candidate():
    return SUPPORT["load_prototype"]()


def assert_identical(actual, expected):
    assert type(actual) is type(expected) is tuple
    assert len(actual) == len(expected)
    assert all(a is b for a, b in zip(actual, expected))


def outcome(call, value):
    try:
        return call(value)
    except BaseException as error:
        return type(error), error.args


@pytest.mark.parametrize("shape", ["empty", "small", "nested", "aliases"])
def test_preorder(candidate, shape):
    leaf = object()
    shared = [leaf, {"key": (leaf, None, b"bytes")}]
    value = {"empty": (), "small": [leaf], "nested": {"a": [(), {"b": (leaf,)}]},
             "aliases": (shared, shared, (shared,))}[shape]
    expected = native._nested_anchors(value)
    actual = candidate["_nested_anchors_deep"](value)
    assert_identical(actual, expected)
    if shape == "aliases":
        assert sum(item is shared for item in actual) == 3
        assert sum(item is leaf for item in actual) == 6


@pytest.mark.parametrize("family", range(27))
def test_every_record_family(candidate, family):
    cls = SUPPORT["record_types"]()[family]
    leaf = object()
    row = cls(**{field.name: (leaf, {"field": leaf}) for field in fields(cls)})
    value = (row, row)
    expected = native._nested_anchors(value)
    actual = candidate["_nested_anchors_deep"](value)
    assert_identical(actual, expected)
    assert sum(item is row for item in actual) == 2
    assert sum(item is leaf for item in actual) == 4 * len(fields(cls))


@pytest.mark.parametrize("shape", ["cycle", "depth"])
def test_recursion_failure(candidate, shape):
    value = []
    if shape == "cycle":
        value.append(value)
    else:
        for _ in range(1200):
            value = [value]
    for walk in (native._nested_anchors, candidate["_nested_anchors_deep"]):
        with pytest.raises(RecursionError) as error:
            walk(value)
        assert type(error.value) is RecursionError


def test_hostile_protocols_untouched(candidate):
    def forbidden(*args, **kwargs):
        raise AssertionError("custom protocol called")

    class Hostile:
        __hash__ = __eq__ = __iter__ = __getattribute__ = forbidden

    class List(list):
        __iter__ = __getitem__ = forbidden

    class Tuple(tuple):
        __iter__ = __getitem__ = forbidden

    class Dict(dict):
        items = __iter__ = __getitem__ = forbidden

    values = (Hostile(), List([object()]), Tuple((object(),)), Dict(a=object()))
    for value in values:
        assert_identical(candidate["_nested_anchors_deep"](value), (value,))
        assert_identical(native._nested_anchors(value), (value,))


def test_fresh_policy_fields_and_raw_reads(candidate, monkeypatch):
    walk = candidate["_nested_anchors_deep"]
    events = []
    policy = native.shared_policy()
    original_fields = native.fields

    class Guard:
        def __getattribute__(self, name):
            if name == "__dataclass_fields__":
                return object.__getattribute__(self, name)
            raise AssertionError("dynamic record read")

    @dataclass
    class Record(Guard):
        child: object

    row = Record(object())
    def policy_read():
        events.append("policy")
        return policy

    def field_read(item):
        events.append(item)
        return original_fields(item)

    for namespace in (vars(native), walk.__globals__):
        monkeypatch.setitem(namespace, "NativeMainMessage", Record)
        monkeypatch.setitem(namespace, "shared_policy", policy_read)
        monkeypatch.setitem(namespace, "fields", field_read)
    metadata = Record.__dataclass_fields__.copy()
    for method in (native._nested_anchors, walk):
        Record.__dataclass_fields__.clear()
        Record.__dataclass_fields__.update(metadata)
        events.clear()
        actual = method((row, row))
        assert events[0] == "policy" and events[1] is row and events[2] is row
        assert_identical(actual, (actual[0], row, object.__getattribute__(row, "child"),
                                  row, object.__getattribute__(row, "child")))
        del Record.__dataclass_fields__["child"]
        assert_identical(method(row), (row,))
        Record.__dataclass_fields__.update(metadata)


@pytest.mark.parametrize("custom_record", [False, True])
def test_custom_metaclass_fallback(candidate, monkeypatch, custom_record):
    events = []
    sentinel = RuntimeError("metaclass equality")

    class Meta(type):
        def __eq__(cls, other):
            events.append((cls, other))
            if other is bytes:
                raise sentinel
            return False

        def __hash__(cls):
            raise AssertionError("custom metaclass hash")

    @dataclass
    class Record(metaclass=Meta):
        child: object

    class Item(metaclass=Meta):
        pass

    walk = candidate["_nested_anchors_deep"]
    if custom_record:
        for namespace in (vars(native), walk.__globals__):
            monkeypatch.setitem(namespace, "NativeMainMessage", Record)
        value = (object(), b"data")
    else:
        value = Item()
    traces = []
    results = []
    for method in (native._nested_anchors, walk):
        events.clear()
        try:
            results.append(method(value))
        except RuntimeError as error:
            assert error is sentinel
            results.append(error)
        traces.append(tuple(events))
    assert len(traces[0]) == len(traces[1]) > 0
    assert all(a is c and b is d for (a, b), (c, d) in zip(*traces))
    if custom_record:
        assert results[0] is results[1] is sentinel
    else:
        assert_identical(results[0], results[1])


@pytest.mark.parametrize("mutation", ["equal", "add", "remove", "nested", "reorder", "invalid"])
def test_snapshot_mutations(candidate, mutation):
    obj = SimpleNamespace(first=[{"key": "value"}], second=object(), padding=tuple(object() for _ in range(4096)))
    snap = SUPPORT["snapshot"](obj)
    if mutation == "equal":
        obj.first = [{"key": "value"}]
    elif mutation == "add":
        obj.third = object()
    elif mutation == "remove":
        del obj.second
    elif mutation == "nested":
        obj.first[0]["key"] = object()
    elif mutation == "reorder":
        first = obj.first
        del obj.first
        obj.first = first
    else:
        obj.first[0]["key"] = object()
        obj.first.append({1: object()})
    expected = outcome(native._issued_wire_snapshot_check, snap)
    actual = outcome(candidate["candidate_snapshot_check"], snap)
    assert actual == expected
    assert actual[0] is native.NativeContentPolicyDenied
    if mutation != "invalid":
        assert actual[1] == ("controlled_native_issued_wire_terminal_state_changed",)
    else:
        assert actual[1] == ("controlled_native_payload_invalid",)


@pytest.mark.parametrize("size", [4095, 4096])
def test_threshold(candidate, monkeypatch, size):
    obj = SimpleNamespace(value=[object() for _ in range(size - 1)])
    snap = SUPPORT["snapshot"](obj)
    assert len(snap[2]) == size
    namespace = candidate["candidate_snapshot_check"].__globals__
    calls = []
    for name in ("_nested_anchors", "_nested_anchors_deep"):
        original = namespace[name]
        def traced(value, original=original, name=name):
            calls.append(name)
            return original(value)
        monkeypatch.setitem(namespace, name, traced)
    candidate["candidate_snapshot_check"](snap)
    assert calls == ["_nested_anchors_deep" if size >= 4096 else "_nested_anchors"]


def test_snapshot_short_circuit(candidate, monkeypatch):
    obj = SimpleNamespace(first=[object()], second=object())
    snap = SUPPORT["snapshot"](obj)
    obj.first = [{1: object()}]
    for method in (native._issued_wire_snapshot_check, candidate["candidate_snapshot_check"]):
        with pytest.raises(native.NativeContentPolicyDenied) as error:
            method(snap)
        assert error.value.args == ("controlled_native_issued_wire_terminal_state_changed",)


@pytest.mark.parametrize("kind,minimum", [("_DelegateFrame", 41416), ("_IssuedCallExecution", 31931)])
def test_deep_snapshot_wrapper(candidate, kind, minimum):
    snap = SUPPORT["build_fixture"](kind)
    assert len(snap[2]) >= minimum
    assert native._issued_wire_snapshot_check(snap) is None
    assert candidate["candidate_snapshot_check"](snap) is None
    name = "catalog_anchor" if kind == "_DelegateFrame" else "catalog_anchors"
    setattr(snap[0], name, tuple(list(getattr(snap[0], name))))
    assert outcome(native._issued_wire_snapshot_check, snap) == outcome(candidate["candidate_snapshot_check"], snap)


@pytest.mark.parametrize("size", [4095, 4096])
def test_runtime_threshold_representative(monkeypatch, size):
    if not hasattr(native, "_nested_anchors_deep"):
        pytest.skip("Candidate has not passed the benefit gate")
    obj = SimpleNamespace(value=[object() for _ in range(size - 1)])
    snap = SUPPORT["snapshot"](obj)
    calls = []
    for name in ("_nested_anchors", "_nested_anchors_deep"):
        original = getattr(native, name)
        def traced(value, original=original, name=name):
            calls.append(name)
            return original(value)
        monkeypatch.setattr(native, name, traced)
    native._issued_wire_snapshot_check(snap)
    assert calls == ["_nested_anchors_deep" if size >= 4096 else "_nested_anchors"]


def test_runtime_snapshot_representative():
    if not hasattr(native, "_nested_anchors_deep"):
        pytest.skip("Candidate has not passed the benefit gate")
    snap = SUPPORT["build_fixture"]("_IssuedCallExecution")
    native._issued_wire_snapshot_check(snap)
    snap[0].decoded["goal"] = object()
    with pytest.raises(native.NativeContentPolicyDenied) as error:
        native._issued_wire_snapshot_check(snap)
    assert error.value.args == ("controlled_native_issued_wire_terminal_state_changed",)


def test_runtime_fallback_representative(monkeypatch):
    if not hasattr(native, "_nested_anchors_deep"):
        pytest.skip("Candidate has not passed the benefit gate")
    sentinel = RuntimeError("fallback")
    class Meta(type):
        def __eq__(cls, other):
            raise sentinel
        def __hash__(cls):
            raise AssertionError("custom hash")
    class Item(metaclass=Meta):
        pass
    with pytest.raises(RuntimeError) as error:
        native._nested_anchors_deep(Item())
    assert error.value is sentinel
