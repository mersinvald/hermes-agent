"""Check fresh type membership against the accepted pure walker."""

import ast
import hashlib
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace

from agent import native_main_payload as native
from agent.transports.types import NormalizedResponse, Usage
from tests.tools.test_native_issued_call_continuation import ContinuationWorld


EVIDENCE = Path("docs/evidence/accountant-native-anchor-type-membership-20261124")
BASE = "0847ff73eadfabcc92b62504674eae670f0fcc41"
RECEIPTS = "docs/evidence/accountant-native-issued-call-episode-20261124/genuine-qualification/genuine.receipts.jsonl"
BINDINGS = (
    "shared_policy", "fields", "_require", "NativeMainResponse", "NativeMainChoice",
    "NativeMainMessage", "NativeMainUsage", "NativeMainSelection", "NativeResultPolicy",
    "NativeAuxiliaryInvocation", "NativeAuxiliaryLocalResult", "_AuxiliaryTitlePreparation",
    "_CompressionBinding", "_AuxiliaryBinding", "_CompressionPreparation", "_FallbackBinding",
    "_FailedSendObservation", "NativeFailedMainAttempt", "_FallbackFrame", "NativeFallbackInvocation",
    "_TurnFrame", "_PendingOperation", "NativeDelegateBinding", "NativeDelegateInvocation",
)
source = subprocess.check_output(["git", "show", BASE + ":agent/native_main_payload.py"], text=True)
source_tree = ast.parse(source)
reference_node = next(node for node in source_tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == "_nested_anchors")
reference_source = ast.get_source_segment(source, reference_node)
prototype = (EVIDENCE / "prototype.py").read_text()
reference_namespace = {name: getattr(native, name) for name in BINDINGS}
candidate_namespace = {name: getattr(native, name) for name in BINDINGS}
exec(compile(reference_source, "accepted-pure-walker", "exec"), reference_namespace)
exec(compile(prototype, str(EVIDENCE / "prototype.py"), "exec"), candidate_namespace)
reference_walk = reference_namespace["_nested_anchors"]
candidate_walk = candidate_namespace["_nested_anchors"]
assert Path(native.__file__).resolve() == Path.cwd() / "agent/native_main_payload.py"
assert Path(sys.prefix).resolve() == Path("/home/mersinvald/.local/share/orca/runtime/accountant/hpwa-p10.47/venv")


def capture(roots, skip=False, walker=reference_walk):
    actual = ()
    for root in roots:
        walked = walker(root)
        actual += walked[1:] if skip else walked
    return actual


def reference(roots, anchors, *, skip_temporary_root=False):
    return native._same_anchors(capture(roots, skip_temporary_root, reference_walk), anchors)


def candidate(roots, anchors, *, skip_temporary_root=False):
    return native._same_anchors(capture(roots, skip_temporary_root, candidate_walk), anchors)


def differential():
    cases = []
    events = []

    def check(name, value, bindings=None, expected=None, error_type=None):
        bindings = bindings or {}
        outcomes = []
        traces = []
        for namespace, walker in ((reference_namespace, reference_walk),
                                  (candidate_namespace, candidate_walk)):
            saved = {key: namespace[key] for key in bindings}
            namespace.update(bindings)
            events.clear()
            try:
                try:
                    result = walker(value)
                    outcomes.append(("return", result))
                except Exception as error:
                    outcomes.append(("error", type(error), error.args))
                traces.append(tuple(events))
            finally:
                namespace.update(saved)
        before, after = outcomes
        assert before[0] == after[0], (name, before, after)
        if before[0] == "return":
            assert type(before[1]) is type(after[1]) is tuple, name
            assert len(before[1]) == len(after[1]), name
            assert all(a is b for a, b in zip(before[1], after[1])), name
            assert native._same_anchors(before[1], after[1]), name
            if expected is not None:
                assert native._same_anchors(before[1], expected), name
            summary = dict(kind="return", length=len(before[1]))
        else:
            assert before[1:] == after[1:], (name, before, after)
            if error_type is not None:
                assert before[1] is error_type, (name, before)
            summary = dict(kind="error", type=before[1].__name__, args=list(before[2]))
        assert traces[0] == traces[1], (name, traces)
        cases.append(dict(id=name, outcome=summary, protocol_trace=[list(row) for row in traces[0]]))
        return outcomes[0]

    token = object()
    shared = [token]
    key_a, key_b = "first", "last"
    root = {key_a: [shared, shared], key_b: (None, False, 1)}
    expected = (root, key_a, root[key_a], shared, token, shared, token,
                key_b, root[key_b], None, False, root[key_b][2])
    check("roots_keys_containers_aliases", root, expected=expected)
    for name, value in (("opaque", token), ("empty_dict", {}), ("empty_list", []),
                        ("empty_tuple", ()), ("bytes", b"bytes"),
                        ("many_opaque_after_threshold", [object() for _ in range(32)])):
        check(name, value)
    before = reference_walk(root)
    root["new"] = token
    changed = check("fresh_container_mutation", root)
    assert not native._same_anchors(before, changed[1])
    del root["new"]
    check("container_restoration", root, expected=before)

    record_expression = next(node.value for node in reference_node.body
                             if isinstance(node, ast.Assign) and node.targets[0].id == "records")
    record_namespace = {name: getattr(native, name) for name in BINDINGS}
    record_namespace["policy"] = native.shared_policy()
    records = eval(compile(ast.Expression(record_expression), "record-catalog", "eval"), record_namespace)
    assert len(records) == 27
    for index, cls in enumerate(records):
        instance = object.__new__(cls)
        values = [object() for _ in fields(cls)]
        for field, value in zip(fields(cls), values):
            object.__setattr__(instance, field.name, value)
        check("record_" + str(index) + "_" + cls.__name__, instance,
              expected=(instance, *values))
        check("record_alias_after_threshold_" + str(index), [*range(8), instance, instance])

    @dataclass
    class LocalRecord:
        first: object
        second: object

    local = LocalRecord(object(), object())
    binding = {"NativeMainUsage": LocalRecord}
    check("new_recognized_alias", local, binding, (local, local.first, local.second))
    old = check("record_before_mutation", local, binding)[1]
    saved_first = local.first
    local.first = object()
    changed = check("fresh_record_mutation", local, binding)[1]
    assert not native._same_anchors(old, changed)
    local.first = saved_first
    check("record_restoration", local, binding, old)
    metadata = LocalRecord.__dataclass_fields__
    try:
        LocalRecord.__dataclass_fields__ = {"second": metadata["second"]}
        check("recognized_metadata_removed_field", [*range(9), local], binding)
        LocalRecord.__dataclass_fields__ = {"second": metadata["second"], "first": metadata["first"]}
        check("recognized_metadata_reordered_fields", local, binding, (local, local.second, local.first))
        LocalRecord.__dataclass_fields__ = {"first": metadata["first"]}
        check("recognized_metadata_added_field_between_calls", local, binding, (local, local.first))
    finally:
        LocalRecord.__dataclass_fields__ = metadata
    check("recognized_metadata_restoration", local, binding, old)
    check("alias_binding_removed_between_calls", local, expected=(local,))

    @dataclass
    class SecondAlias:
        child: object

    second = SecondAlias(local)
    check("new_alias_binding_between_calls", second,
          {"NativeMainUsage": LocalRecord, "NativeMainChoice": SecondAlias},
          (second, local, local.first, local.second))
    check("alias_binding_restored_between_calls", second, expected=(second,))

    class ParentA:
        pass

    class ParentB:
        pass

    @dataclass
    class MutableClass(ParentA):
        child: object

    mutable = MutableClass(token)
    try:
        MutableClass.__bases__ = (ParentB,)
        check("plain_class_bases_mutation", [*range(8), mutable], {"NativeMainUsage": MutableClass})
    finally:
        MutableClass.__bases__ = (ParentA,)
    check("plain_class_bases_restoration", mutable, {"NativeMainUsage": MutableClass}, (mutable, token))

    changing = LocalRecord(token, token)
    changing.child = token

    def reset_instance_policy():
        changing.__class__ = LocalRecord
        return native.shared_policy()

    def changing_instance_fields(item):
        result = fields(item)
        events.append(("instance_fields", type(item).__name__))
        changing.__class__ = SecondAlias
        return result

    try:
        check("instance_class_changes_between_alias_visits", [*range(8), changing, changing],
              {"NativeMainUsage": LocalRecord, "NativeMainChoice": SecondAlias,
               "shared_policy": reset_instance_policy, "fields": changing_instance_fields})
    finally:
        changing.__class__ = LocalRecord

    def observed_fields(item):
        result = fields(item)
        events.append(("fields", id(item), tuple(field.name for field in result)))
        return result

    check("fresh_fields_for_every_alias", [*range(8), local, local],
          dict(binding, fields=observed_fields))

    def reset_metadata_policy():
        LocalRecord.__dataclass_fields__ = metadata
        return native.shared_policy()

    def changing_fields(item):
        result = fields(item)
        events.append(("changing_fields", tuple(field.name for field in result)))
        LocalRecord.__dataclass_fields__ = {"second": metadata["second"]}
        return result

    try:
        check("metadata_changes_between_alias_visits", [*range(8), local, local],
              dict(binding, fields=changing_fields, shared_policy=reset_metadata_policy))
    finally:
        LocalRecord.__dataclass_fields__ = metadata

    @dataclass
    class GetterRecord:
        child: object

        def __getattribute__(self, name):
            if name == "child":
                raise AssertionError("field access must use object")
            return object.__getattribute__(self, name)

    getter = GetterRecord(token)
    check("exact_object_getattribute", [*range(9), getter], {"NativeMainUsage": GetterRecord})
    missing = object.__new__(LocalRecord)
    check("missing_field_after_eighth_visit", [*range(8), missing], binding, error_type=AttributeError)
    check("invalid_key", {1: token}, error_type=native.NativeContentPolicyDenied)
    check("invalid_key_after_eighth_visit", [*range(8), {1: token}], error_type=native.NativeContentPolicyDenied)
    recursive = []
    recursive.append(recursive)
    check("recursion", recursive, error_type=RecursionError)
    check("recursion_after_eighth_visit", [*range(8), recursive], error_type=RecursionError)

    class ObservedMeta(type):
        def __eq__(cls, other):
            events.append(("eq", cls.__name__, getattr(other, "__name__", type(other).__name__)))
            return False

        def __hash__(cls):
            raise AssertionError("custom class hash must not run")

    class CustomItem(metaclass=ObservedMeta):
        pass

    custom = CustomItem()
    check("custom_item_after_index_activation", [*range(8), custom, token])
    check("custom_catalog_retains_tuple_path", [*range(9), token], {"NativeMainUsage": CustomItem})

    class SequenceMeta(ObservedMeta):
        def __eq__(cls, other):
            events.append(("sequence_eq", getattr(other, "__name__", type(other).__name__)))
            return other is list

        __hash__ = ObservedMeta.__hash__

    class SequenceItem(metaclass=SequenceMeta):
        def __iter__(self):
            events.append(("sequence_iter",))
            return iter((token, token))

    check("custom_item_list_protocol", [*range(8), SequenceItem()])

    class BrokenSequence(metaclass=SequenceMeta):
        def __iter__(self):
            raise RuntimeError("later iteration failure", 8)

    check("custom_item_list_protocol_error", [*range(8), BrokenSequence()], error_type=RuntimeError)

    class Truth:
        def __bool__(self):
            events.append(("truth_bool",))
            return False

    class TruthMeta(ObservedMeta):
        def __eq__(cls, other):
            events.append(("truth_eq", getattr(other, "__name__", type(other).__name__)))
            return Truth()

        __hash__ = ObservedMeta.__hash__

    class TruthItem(metaclass=TruthMeta):
        pass

    check("custom_item_equality_truth_protocol", [*range(8), TruthItem()])
    check("custom_catalog_equality_truth_protocol", [*range(9), token], {"NativeMainUsage": TruthItem})

    class ReflectedMeta(ObservedMeta):
        def __eq__(cls, other):
            events.append(("not_implemented_eq", getattr(other, "__name__", type(other).__name__)))
            return NotImplemented

        __hash__ = ObservedMeta.__hash__

    class ReflectedItem(metaclass=ReflectedMeta):
        pass

    check("custom_item_not_implemented_equality", [*range(8), ReflectedItem()])
    check("custom_catalog_not_implemented_equality", [*range(9), token], {"NativeMainUsage": ReflectedItem})

    class UnhashableMeta(ObservedMeta):
        __hash__ = None

    class UnhashableItem(metaclass=UnhashableMeta):
        pass

    check("unhashable_item_after_index_activation", [*range(8), UnhashableItem()])
    check("unhashable_catalog", [*range(9), token], {"NativeMainUsage": UnhashableItem})

    class EqualAlias:
        def __eq__(self, other):
            events.append(("alias_eq", getattr(other, "__name__", type(other).__name__)))
            return False

        def __hash__(self):
            raise AssertionError("nonclass alias hash must not run")

    check("nonclass_catalog_alias", [*range(9), custom, token], {"NativeMainUsage": EqualAlias()})

    class MatchingMeta(ObservedMeta):
        def __eq__(cls, other):
            events.append(("match_eq", getattr(other, "__name__", type(other).__name__)))
            return other is LocalRecord

        __hash__ = ObservedMeta.__hash__

    class MatchingAlias(metaclass=MatchingMeta):
        pass

    check("observable_equality_recognizes_distinct_class", [*range(8), local],
          {"NativeMainUsage": MatchingAlias})

    class RaisingMeta(ObservedMeta):
        def __eq__(cls, other):
            events.append(("raising_eq", getattr(other, "__name__", type(other).__name__)))
            if other is LocalRecord:
                raise ValueError("later membership failure", 8)
            return False

        __hash__ = ObservedMeta.__hash__

    class RaisingAlias(metaclass=RaisingMeta):
        pass

    check("membership_failure_after_eighth_visit", [*range(8), local],
          {"NativeMainUsage": RaisingAlias}, error_type=ValueError)

    def failing_fields(item):
        events.append(("failing_fields", id(item)))
        raise LookupError("later fields failure", 8)

    check("fields_failure_after_eighth_visit", [*range(8), local],
          dict(binding, fields=failing_fields), error_type=LookupError)
    opaque = SimpleNamespace(__dataclass_fields__=metadata)
    check("opaque_metadata", [*range(8), opaque])
    opaque.__dataclass_fields__ = {}
    check("opaque_metadata_changed", opaque, expected=(opaque,))

    def observed_policy():
        events.append(("shared_policy",))
        return native.shared_policy()

    check("shared_policy_each_call", [*range(9), token], {"shared_policy": observed_policy})
    check("shared_policy_next_call", token, {"shared_policy": observed_policy})
    try:
        LocalRecord.__class__ = ObservedMeta
    except TypeError as error:
        cases.append(dict(id="plain_class_metaclass_mutation_rejected", outcome=dict(
            kind="error", type=type(error).__name__, args=list(error.args))))
    else:
        raise AssertionError("plain class metaclass changed on the supported interpreter")
    return cases


def inputs():
    raw = subprocess.check_output(["git", "show", BASE + ":" + RECEIPTS])
    rows = [json.loads(line) for line in raw.splitlines()]
    final = next(row["value"] for row in rows if row["property"] == "final")
    graph = [row["value"]["graph"] for row in rows if row["property"] == "boundary"][-1]
    request = json.loads(final["request"])
    request = {key: request[key] for key in ("model", "messages", "tools", "stream", "tool_choice")}
    response_data = json.loads(bytes.fromhex(graph[35]["hex"]))
    normalized_data = json.loads(bytes.fromhex(graph[36]["hex"]))
    world = ContinuationWorld("positive")
    source, selection, producer = world.bindings[5], world.selections[5], world.policies[5]
    request["model"] = selection.destination.model_ref
    response = native.NativeMainResponse(response_data["id"], response_data["object"], response_data["created"],
        request["model"], tuple(native.NativeMainChoice(row["index"],
            native.NativeMainMessage(**row["message"]), row["finish_reason"])
            for row in response_data["choices"]), native.NativeMainUsage(**response_data["usage"]))
    normalized_data["content"] = response.choices[0].message.content
    normalized_data["finish_reason"] = response.choices[0].finish_reason
    normalized_data["usage"] = Usage(**normalized_data["usage"])
    normalized = NormalizedResponse(**normalized_data)
    wrappers = tuple((ref, SimpleNamespace(port=port)) for ref, port in world.transports)
    catalog = (world.routes, world.profiles, world.transports, wrappers)
    first_function = native._IssuedCallContinuation.verify
    transport_function = type(world.ports[5, "main"]).send
    shape_values = [
        ("callable_defaults", ((first_function.__defaults__, first_function.__kwdefaults__),), True),
        ("catalog", (catalog,), False),
        ("authority", ((source, selection, producer),), True),
        ("request", (request,), False),
        ("response", (response,), False),
        ("normalized_usage", (vars(normalized), vars(normalized.usage)), False),
        ("transport_defaults", ((transport_function.__defaults__, transport_function.__kwdefaults__),), True),
    ]
    shapes = [dict(name=name, roots=roots, skip=skip, anchors=capture(roots, skip))
              for name, roots, skip in shape_values]
    assert request["model"] is selection.destination.model_ref is response.model
    assert normalized.content is response.choices[0].message.content
    assert world.routes[-1].destination is selection.destination
    for shape in shapes:
        assert reference(shape["roots"], shape["anchors"], skip_temporary_root=shape["skip"])
        assert candidate(shape["roots"], shape["anchors"], skip_temporary_root=shape["skip"])
    data_pins = dict(receipts_commit=BASE, receipts_path=RECEIPTS, receipts_sha256=hashlib.sha256(raw).hexdigest(),
        graph_indices=dict(request=33, response=35, normalized=36),
        payloads={str(i): dict(bytes=graph[i]["bytes"], sha256=graph[i]["sha256"]) for i in (33, 35, 36)},
        defaults=[first_function.__qualname__, transport_function.__qualname__],
        request_order=list(request), normalized_order=list(vars(normalized)),
        usage_order=list(vars(normalized.usage)), catalog_sizes=[len(part) for part in catalog],
        fixture="ContinuationWorld('positive') constructor only", binding_names=list(BINDINGS),
        imports=dict(native=str(Path(native.__file__).resolve()), interpreter=sys.executable, prefix=sys.prefix))
    return shapes, data_pins


def measure(shapes):
    batches = []
    repetitions = 1200
    for batch in range(5):
        order = ("reference", "candidate") if batch % 2 == 0 else ("candidate", "reference")
        durations = {}
        for name in order:
            function = reference if name == "reference" else candidate
            durations[name] = {}
            for shape in shapes:
                started = time.perf_counter_ns()
                for _ in range(repetitions):
                    assert function(shape["roots"], shape["anchors"], skip_temporary_root=shape["skip"])
                durations[name][shape["name"]] = (time.perf_counter_ns() - started) / 1e9
        batches.append(dict(batch=batch + 1, order=list(order), repetitions=repetitions, seconds=durations))
    summaries = []
    for shape in shapes:
        name = shape["name"]
        ref = [batch["seconds"]["reference"][name] / repetitions for batch in batches]
        cand = [batch["seconds"]["candidate"][name] / repetitions for batch in batches]
        r, c = statistics.median(ref), statistics.median(cand)
        noise = max(statistics.median(abs(x-r) for x in ref), statistics.median(abs(x-c) for x in cand))
        summaries.append(dict(name=name, anchor_count=len(shape["anchors"]), input_roots=len(shape["roots"]),
            skip_temporary_root=shape["skip"], reference_median_seconds=r, candidate_median_seconds=c,
            median_absolute_deviation_seconds=noise, candidate_over_reference=c/r,
            regression_beyond_noise=c > r * 1.10 + 3 * noise))
    total_ref = sum(row["reference_median_seconds"] for row in summaries)
    total_cand = sum(row["candidate_median_seconds"] for row in summaries)
    total_noise = sum(row["median_absolute_deviation_seconds"] for row in summaries)
    improvement = 1 - total_cand / total_ref
    confident = total_ref - total_cand > 3 * total_noise
    passed = improvement >= .20 and confident and not any(row["regression_beyond_noise"] for row in summaries)
    return dict(batches=batches, shapes=summaries, reference_sum_medians_seconds=total_ref,
        candidate_sum_medians_seconds=total_cand, aggregate_improvement=improvement,
        summed_median_absolute_deviation_seconds=total_noise, confident=confident, gate_passed=passed,
        noise_rule="Three times the larger per-shape median absolute deviation", paired_batches=5)


def main():
    started = time.monotonic()
    mode = sys.argv[1]
    if mode == "correctness":
        cases = differential()
        report = dict(differential_passed=True, cases=cases, interpreter=sys.version,
                      elapsed_after_import_seconds=time.monotonic() - started)
        (EVIDENCE / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(dict(differential_passed=True, cases=len(cases))))
    elif mode == "timing":
        report = json.loads((EVIDENCE / "result.json").read_text())
        assert report["differential_passed"]
        shapes, data_pins = inputs()
        report.update(timing=measure(shapes), data=data_pins,
                      prototype_sha256=hashlib.sha256(prototype.encode()).hexdigest(),
                      prototype_ast_sha256=hashlib.sha256(ast.dump(ast.parse(prototype), include_attributes=False).encode()).hexdigest(),
                      reference_sha256=hashlib.sha256(reference_source.encode()).hexdigest(),
                      timing_elapsed_after_import_seconds=time.monotonic() - started)
        (EVIDENCE / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(dict(timing=report["timing"]), indent=2))
    else:
        raise ValueError(mode)


if __name__ == "__main__":
    main()
