"""Compare live traversal with the unchanged native anchor reference."""

import ast
import hashlib
import json
import statistics
import subprocess
import sys
import time
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

from agent import native_main_payload as native
from agent.transports.types import NormalizedResponse, Usage
from tests.tools.test_native_issued_call_continuation import ContinuationWorld


EVIDENCE = Path("docs/evidence/accountant-native-continuation-anchor-comparison-20261124")
BASE = "5ceb15437ec9318cbd62903a550d20f8743a3c0f"
RECEIPTS = "docs/evidence/accountant-native-issued-call-episode-20261124/genuine-qualification/genuine.receipts.jsonl"
BINDINGS = (
    "shared_policy", "fields", "_require", "NativeMainResponse", "NativeMainChoice",
    "NativeMainMessage", "NativeMainUsage", "NativeMainSelection", "NativeResultPolicy",
    "NativeAuxiliaryInvocation", "NativeAuxiliaryLocalResult", "_AuxiliaryTitlePreparation",
    "_CompressionBinding", "_AuxiliaryBinding", "_CompressionPreparation", "_FallbackBinding",
    "_FailedSendObservation", "NativeFailedMainAttempt", "_FallbackFrame", "NativeFallbackInvocation",
    "_TurnFrame", "_PendingOperation", "NativeDelegateBinding", "NativeDelegateInvocation",
)
namespace = {name: getattr(native, name) for name in BINDINGS}
prototype = (EVIDENCE / "prototype.py").read_text()
exec(compile(prototype, str(EVIDENCE / "prototype.py"), "exec"), namespace)
candidate = namespace["_continuation_matches_anchors"]
assert Path(native.__file__).resolve() == Path.cwd() / "agent/native_main_payload.py"
assert Path(sys.prefix).resolve() == Path("/home/mersinvald/.local/share/orca/runtime/accountant/hpwa-p10.47/venv")


def reference(roots, anchors, *, skip_temporary_root=False):
    actual = ()
    for root in roots:
        walked = native._nested_anchors(root)
        actual += walked[1:] if skip_temporary_root else walked
    return native._same_anchors(actual, anchors)


def capture(roots, skip=False):
    actual = ()
    for root in roots:
        walked = native._nested_anchors(root)
        actual += walked[1:] if skip else walked
    return actual


def outcome(function, roots, anchors, skip):
    try:
        return ("return", function(roots, anchors, skip_temporary_root=skip))
    except Exception as error:
        return ("error", type(error).__module__, type(error).__qualname__, str(error))


def differential():
    cases = []

    def check(name, roots, anchors=None, skip=False, expected=None):
        if anchors is None:
            anchors = capture(roots, skip)
        before = outcome(reference, roots, anchors, skip)
        after = outcome(candidate, roots, anchors, skip)
        assert before == after, (name, before, after)
        if expected is not None:
            assert before == ("return", expected), (name, before)
        cases.append(dict(id=name, result=list(before), anchors=len(anchors)))

    scalar = object()
    check("scalar_identity", (scalar,), expected=True)
    check("scalar_distinct", (scalar,), (object(),), expected=False)
    check("equal_value_distinct_identity", (bytes(bytearray(b"same")),),
          (bytes(bytearray(b"same")),), expected=False)
    shared = [scalar]
    root = {"first": [shared, shared], "last": (None, False, 1)}
    anchors = capture((root,))
    check("containers_and_alias_occurrences", (root,), anchors, expected=True)
    changed = list(anchors)
    changed[4] = object()
    check("alias_occurrence_mismatch", (root,), tuple(changed), expected=False)
    check("short_anchors", (root,), anchors[:-1], expected=False)
    check("long_anchors", (root,), anchors + (scalar,), expected=False)
    check("empty_anchors", (root,), (), expected=False)
    check("multiple_roots", (root, shared, scalar), expected=True)
    check("empty_roots", (), (), expected=True)
    check("temporary_tuple_root", ((root, shared),), skip=True, expected=True)
    check("temporary_root_identity_is_skipped", ((root, shared),),
          capture(((root, shared),), True), skip=True, expected=True)
    check("empty_temporary_root", ((),), (), skip=True, expected=True)
    check("ordinary_root_identity", (dict(root),), anchors, expected=False)
    ordered = {"first": scalar, "second": scalar}
    order_anchors = capture((ordered,))
    ordered["first"] = ordered.pop("first")
    check("ordered_keys", (ordered,), order_anchors, expected=False)
    usage = native.NativeMainUsage(11, 7, 18)
    check("recognized_fields", (usage,), expected=True)
    usage_anchors = capture((usage,))
    object.__setattr__(usage, "total_tokens", 19)
    check("fresh_field_mutation", (usage,), usage_anchors, expected=False)
    object.__setattr__(usage, "total_tokens", 18)
    check("field_restoration", (usage,), usage_anchors, expected=True)
    root["new"] = object()
    check("fresh_container_mutation", (root,), anchors, expected=False)
    del root["new"]
    check("container_restoration", (root,), anchors, expected=True)
    check("invalid_key", ({1: scalar},), ())
    check("mismatch_then_invalid_key", ([object(), {1: scalar}],), (scalar,))
    recursive = []
    recursive.append(recursive)
    check("recursive_input", (recursive,), ())
    check("mismatch_then_recursion", ([object(), recursive],), (scalar,))
    check("later_root_invalid_key", (scalar, {1: scalar}), (object(),))
    opaque = SimpleNamespace(__dataclass_fields__={"first": fields(usage)[0]})
    opaque_anchors = capture((opaque,))
    check("unrecognized_metadata", (opaque,), opaque_anchors, expected=True)
    opaque.__dataclass_fields__ = {"second": fields(usage)[1]}
    check("unrecognized_metadata_changed", (opaque,), opaque_anchors, expected=True)
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
    passed = improvement >= .10 and confident and not any(row["regression_beyond_noise"] for row in summaries)
    return dict(batches=batches, shapes=summaries, reference_sum_medians_seconds=total_ref,
        candidate_sum_medians_seconds=total_cand, aggregate_improvement=improvement,
        summed_median_absolute_deviation_seconds=total_noise, confident=confident, gate_passed=passed,
        noise_rule="Three times the larger per-shape median absolute deviation", paired_batches=5)


def main():
    started = time.monotonic()
    cases = differential()
    shapes, data_pins = inputs()
    timing = measure(shapes)
    tree = ast.parse(prototype)
    report = dict(cases=cases, differential_passed=True, timing=timing, data=data_pins,
        prototype_sha256=hashlib.sha256(prototype.encode()).hexdigest(),
        prototype_ast_sha256=hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
        experiment_seconds_after_import=time.monotonic()-started,
        scope="Pure comparison only. No prepare, qualification, runtime admission, storage, model, service, or send call.")
    (EVIDENCE / "experiment-result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(differential_cases=len(cases), timing=timing), indent=2))


main()
