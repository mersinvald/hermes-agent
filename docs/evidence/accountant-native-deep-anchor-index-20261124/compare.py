"""Run the fixed pure comparison once and retain each duration."""

import hashlib
import json
import platform
import runpy
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    plan = json.loads((HERE / "comparison-plan.json").read_text())
    support = runpy.run_path(str(HERE / "fixtures.py"))
    native = support["native"]
    candidate = support["load_prototype"]()["candidate_snapshot_check"]
    fixtures = {kind: support["build_fixture"](kind) for kind in plan["fixtures"]}
    counts = {kind: dict(pairs=len(snap[1]), anchors=len(snap[2]),
                        unique_identities=len({id(item) for item in snap[2]}))
              for kind, snap in fixtures.items()}
    for kind, minimum in zip(plan["fixtures"], (41416, 31931)):
        assert counts[kind]["anchors"] >= minimum
    pins = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in plan["pin_paths"]}
    result = dict(plan=plan, pins=pins, counts=counts, python=sys.version,
                  executable=sys.executable, prefix=sys.prefix, platform=platform.platform(),
                  qualification="Synthetic size and shape models only", pairs=[], measured_seconds=0)
    (HERE / "comparison-freeze.json").write_text(json.dumps(result, indent=2) + "\n")
    methods = dict(reference=native._issued_wire_snapshot_check, candidate=candidate)
    started = time.perf_counter()
    for kind, snap in fixtures.items():
        for pair in range(5):
            row = dict(fixture=kind, pair=pair, order=["reference", "candidate"] if pair % 2 == 0
                       else ["candidate", "reference"], raw_seconds={})
            for name in row["order"]:
                durations = []
                for _ in range(10):
                    before = time.perf_counter()
                    assert methods[name](snap) is None
                    elapsed = time.perf_counter() - before
                    durations.append(elapsed)
                    result["measured_seconds"] += elapsed
                    assert time.perf_counter() - started < 20, "Comparison exceeded the fixed cap"
                row["raw_seconds"][name] = durations
            row["medians"] = {name: statistics.median(values) for name, values in row["raw_seconds"].items()}
            result["pairs"].append(row)
    result["batch_wall_seconds"] = time.perf_counter() - started
    sums = {name: sum(row["medians"][name] for row in result["pairs"]) for name in methods}
    result["median_sums"] = sums
    result["combined_gain"] = 1 - sums["candidate"] / sums["reference"]
    result["separation"] = {}
    for kind in fixtures:
        rows = [row for row in result["pairs"] if row["fixture"] == kind]
        slowest = max(row["medians"]["candidate"] for row in rows)
        fastest = min(row["medians"]["reference"] for row in rows)
        result["separation"][kind] = dict(slowest_candidate=slowest, fastest_reference=fastest, passes=slowest < fastest)
    result["gate_passes"] = result["combined_gain"] >= .20 and all(row["passes"] for row in result["separation"].values())
    assert pins == {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in pins}
    (HERE / "comparison-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("counts", "measured_seconds", "median_sums", "combined_gain", "separation", "gate_passes")}))


if __name__ == "__main__":
    main()
