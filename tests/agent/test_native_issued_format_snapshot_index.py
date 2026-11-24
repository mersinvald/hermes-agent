"""Check branch-only snapshots and the actual formatter verifier prefix.

Inert owners supply no native admission or authority qualification.
The genuine episode retains the original authority and callback checks.
"""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import native_main_payload as native


class PrefixReached(Exception):
    pass


class InertOwner:
    def __init__(self):
        self.access_calls = 0

    def access(self):
        self.access_calls += 1

    @property
    def methods(self):
        raise PrefixReached


def test_registered_helper_replacement_refused(monkeypatch):
    owner, calls = InertOwner(), []

    def replacement(value):
        calls.append(value)
        raise AssertionError("replacement used")

    original = native._nested_anchors_deep
    with monkeypatch.context() as patch:
        patch.setattr(native, "_nested_anchors_deep", replacement)
        with pytest.raises(native.NativeContentPolicyDenied) as error:
            native._IssuedCallFormat.verify(owner)
        assert error.value.args == ("controlled_native_issued_format_code_changed",)
        assert calls == [] and owner.access_calls == 1
    assert native._nested_anchors_deep is original


def test_registered_helper_code_drift_refused():
    owner, calls = InertOwner(), []
    function = native._nested_anchors_deep
    original_code = function.__code__

    def replacement(value):
        raise AssertionError("replacement used")

    try:
        function.__code__ = replacement.__code__
        with pytest.raises(native.NativeContentPolicyDenied) as error:
            native._IssuedCallFormat.verify(owner)
        assert error.value.args == ("controlled_native_issued_format_code_changed",)
        assert calls == [] and owner.access_calls == 1
    finally:
        function.__code__ = original_code
    assert native._nested_anchors_deep is function and function.__code__ is original_code


def test_stock_verifier_prefix():
    owner = InertOwner()
    with pytest.raises(PrefixReached):
        native._IssuedCallFormat.verify(owner)
    assert owner.access_calls == 1


@pytest.fixture(scope="module")
def snapshot_loop():
    """Compile the exact production loop for branch-only unit checks."""
    source = Path(native.__file__).read_text()
    module = ast.parse(source)
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "_IssuedCallFormat")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "verify")
    loop = next(node for node in ast.walk(method) if isinstance(node, ast.For)
                and isinstance(node.iter, ast.Attribute) and node.iter.attr == "source_snapshots")
    function = ast.parse("def check(self):\n    pass\n").body[0]
    function.body = [copy.deepcopy(loop)]
    return compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                   "<exact formatter snapshot loop: branch-only unit scope>", "exec")


def capture(obj):
    pairs = tuple(vars(obj).items())
    return obj, pairs, native._nested_anchors(tuple(value for _, value in pairs))[1:]


def execute(code, snapshot, *, reference=False):
    namespace, calls = dict(vars(native)), []
    for name in ("_nested_anchors", "_nested_anchors_deep"):
        walk = native._nested_anchors if reference else namespace[name]

        def observed(value, walk=walk, name=name):
            anchors = walk(value)
            calls.append((name, anchors))
            return anchors

        namespace[name] = observed
    exec(code, namespace)
    try:
        namespace["check"](SimpleNamespace(source_snapshots=(snapshot,)))
    except native.NativeContentPolicyDenied as error:
        return (type(error), error.args), calls
    return None, calls


def check_reference(code, snapshot):
    expected, reference_calls = execute(code, snapshot, reference=True)
    actual, calls = execute(code, snapshot)
    assert actual == expected
    assert len(calls) == len(reference_calls)
    for (name, anchors), (reference_name, original) in zip(calls, reference_calls):
        assert name == reference_name and native._same_anchors(anchors[1:], original[1:])
    return actual, calls


@pytest.mark.parametrize("size", [4095, 4096])
def test_branch_threshold(snapshot_loop, size):
    obj = SimpleNamespace(value=[object() for _ in range(size - 1)])
    snapshot = capture(obj)
    assert len(snapshot[2]) == size
    actual, calls = check_reference(snapshot_loop, snapshot)
    assert actual is None
    assert [name for name, _ in calls] == ["_nested_anchors_deep" if size == 4096 else "_nested_anchors"]
    assert native._same_anchors(calls[0][1][1:], snapshot[2])


@pytest.mark.parametrize("mutation", ["add", "remove", "rename", "reorder", "shallow"])
def test_branch_namespace_and_identity(snapshot_loop, mutation):
    obj = SimpleNamespace(first=[{"key": object()}], second=object(), padding=[object()] * 4096)
    snapshot = capture(obj)
    if mutation == "add":
        obj.third = object()
    elif mutation == "remove":
        del obj.second
    elif mutation == "rename":
        obj.renamed = obj.second
        del obj.second
    elif mutation == "reorder":
        first = obj.first
        del obj.first
        obj.first = first
    else:
        obj.first = list(obj.first)
    obj.padding.append({1: object()})
    actual, calls = check_reference(snapshot_loop, snapshot)
    assert actual == (native.NativeContentPolicyDenied, ("controlled_native_issued_format_terminal_state_changed",))
    assert calls == []


@pytest.mark.parametrize("invalid", [False, True])
def test_branch_nested_mutation_failure_order(snapshot_loop, invalid):
    obj = SimpleNamespace(first=[{"key": object()}], padding=[object()] * 4096)
    snapshot = capture(obj)
    obj.first[0]["key"] = object()
    if invalid:
        obj.first.append({1: object()})
    actual, calls = check_reference(snapshot_loop, snapshot)
    prefix = "controlled_native_payload_invalid" if invalid else "controlled_native_issued_format_terminal_state_changed"
    assert actual == (native.NativeContentPolicyDenied, (prefix,))
    assert len(calls) == (0 if invalid else 1)


def test_branch_repeated_aliases(snapshot_loop):
    leaf = object()
    shared = [{"key": leaf}]
    obj = SimpleNamespace(value=[shared] * 1400)
    snapshot = capture(obj)
    actual, calls = check_reference(snapshot_loop, snapshot)
    assert actual is None and calls[0][0] == "_nested_anchors_deep"
    assert sum(item is shared for item in calls[0][1]) == 1400
    assert sum(item is leaf for item in calls[0][1]) == 1400


def test_branch_custom_metaclass_fallback(snapshot_loop):
    events = []

    class Meta(type):
        def __eq__(cls, other):
            events.append((cls, other))
            return False

        def __hash__(cls):
            raise AssertionError("custom metaclass hash")

    class Item(metaclass=Meta):
        pass

    obj = SimpleNamespace(value=[object()] * 4096 + [Item()])
    snapshot = capture(obj)
    traces = []
    for reference in (True, False):
        events.clear()
        actual, calls = execute(snapshot_loop, snapshot, reference=reference)
        assert actual is None and calls[0][0] == "_nested_anchors_deep"
        assert native._same_anchors(calls[0][1][1:], snapshot[2])
        traces.append(tuple(events))
    assert len(traces[0]) == len(traces[1]) > 0
    assert all(a is c and b is d for (a, b), (c, d) in zip(*traces))
