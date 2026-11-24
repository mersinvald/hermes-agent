"""Read the two retained continuation receipts without native execution."""

import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


RAW = Path(sys.argv[1])
OUT = Path(__file__).parent
SOURCE = "4305a1edb5227bb66acdf66e716a53aebb10b2e8b9f93d3512323083f5e586fe"
TEST = "1cb07f12079b4a4d508e345aa71caa2a5fdb398d420484b4cacebf39ab6648a0"
FILE = "tests/tools/test_native_issued_call_continuation.py"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def payload(value):
    if value is None:
        return None
    data = value.encode()
    return {"bytes": len(data), "sha256": digest(data)}


def graph(rows):
    result = []
    for row in rows:
        data = row["payload"].encode()
        assert data == bytes.fromhex(row["hex"])
        assert digest(data) == row["sha256"]
        result.append({k: row[k] for k in
                       ("index", "sha256", "parents", "binding", "profiles", "path")}
                      | {"bytes": len(data)})
    return result


def runs(values):
    result = []
    for value in values:
        if result and result[-1]["value"] == value:
            result[-1]["count"] += 1
        else:
            result.append({"count": 1, "value": value})
    return result


results = {}
for case in ("positive", "coverage_expiry"):
    process = json.loads((RAW / (case + ".process.json")).read_bytes())
    log = (RAW / (case + ".log")).read_text(encoding="utf-8")
    assert process["exit_status"] == 0 and not process["outer_timeout"]
    assert "Summary: 1 files, 1 tests passed, 0 failed" in log
    assert "595s exceeded" not in log and "file timeout exceeded" not in log
    for marker in ("PytestUnhandledThreadExceptionWarning", "PytestUnraisableExceptionWarning",
                   "Exception in thread", "Exception ignored in:"):
        assert marker not in log
    rows = [json.loads(line) for line in
            (RAW / (case + ".properties.jsonl")).read_text(encoding="utf-8").splitlines()]
    cases = [json.loads(line) for line in
             (RAW / (case + ".cases.jsonl")).read_text(encoding="utf-8").splitlines()]
    nodeid = FILE + "::test_genuine_continuation[" + case + "]"
    assert all(row["nodeid"] == nodeid for row in rows + cases)
    assert all(row["source_sha256"] == SOURCE and row["test_sha256"] == TEST for row in cases)
    assert [(row["phase"], row["outcome"]) for row in cases] == [
        ("start", "pending"), ("setup", "passed"), ("call", "passed")]
    xml = ET.parse(RAW / (case + ".xml")).getroot()
    tests = xml.findall(".//testcase")
    assert len(tests) == 1 and tests[0].get("name") == "test_genuine_continuation[" + case + "]"
    assert not any(tests[0].find(tag) is not None for tag in ("failure", "error", "skipped"))
    xml_properties = [(p.get("name"), p.get("value")) for p in tests[0].findall("./properties/property")]
    assert all((row["property"], str(row["value"])) in xml_properties for row in rows)
    actuals = [json.loads(row["value"]) for row in rows if row["property"] == "continuation_actual"]
    final = actuals[-1]
    before = json.loads(next(row["value"] for row in rows if row["property"] == "actual_J"))
    final_graph = graph(final["graph"])
    assert len(final_graph) == final["counts"][0]
    assert sum(row["bytes"] for row in final_graph) == final["counts"][2]
    observed = {key: final[key] for key in ("mode", "phase", "windows", "timings", "source_timings",
        "qualification_sends", "data_sends", "counts", "before", "after", "poisoned")}
    observed["payloads"] = {key: payload(final[key]) for key in
                            ("raw", "request", "wire", "normalized", "completion")}
    observed["before_continuation"] = [before["content_slots"], before["envelopes"], before["retained_bytes"]]
    observed["continuation_charge_delta"] = [a - b for a, b in zip(final["counts"], observed["before_continuation"])]
    observed["policy_observation_count"] = len(final["policy_observations"])
    observed["policy_observation_runs"] = runs(final["policy_observations"])
    observed["final_time"] = final["policy_observations"][-1][-1]
    observed["boundaries"] = [json.loads(row["value"]) for row in rows if row["property"] == "C_boundary"]
    observed["stage_receipts"] = [{"phase": item["phase"], "windows": item["windows"],
        "counts": item["counts"], "poisoned": item["poisoned"],
        "payloads": {key: payload(item[key]) for key in ("raw", "request", "wire", "normalized", "completion")}}
        for item in actuals]
    observed["graph"] = final_graph
    write(case + ".json", observed)
    command = [arg if not arg.startswith("--junitxml=") else "--junitxml=<private-" + case + ".xml>"
               for arg in process["command"]]
    results[case] = {"requested_id": nodeid, "actual_id": nodeid, "outcome": "passed",
        "runtime_sha256": SOURCE, "test_sha256": TEST, "command": command,
        "private_xml_path_redacted": True, "pid": process["pid"],
        "command_seconds": process["command_seconds"], "start_utc": process["start_utc"],
        "end_utc": process["end_utc"], "exit_status": process["exit_status"],
        "outer_timeout": process["outer_timeout"], "inner_timeout": False,
        "floor_required_seconds": process["floor_required_seconds"],
        "floor_remaining_seconds": process["floor_remaining_seconds"],
        "outer_cap_seconds": 600, "file_cap_seconds": 595, "retries": 0,
        "xml_suites": [suite.attrib for suite in xml.findall(".//testsuite")],
        "xml_testcase": tests[0].attrib, "original_case_receipts": cases,
        "property_rows": len(rows), "xml_property_rows": len(xml_properties),
        "visible_exception_markers": [], "view": case + ".json"}
    assert final["phase"] == ("released" if case == "positive" else "denied")
    assert final["windows"] == (6 if case == "positive" else 3)
    assert final["counts"] == ([38, 4, 32851] if case == "positive" else [35, 4, 32125])
    assert final["policy_observations"][-1][-1] == (10 if case == "positive" else 50)
write("results.json", results)
write("raw-manifest.json", {p.name: {"bytes": p.stat().st_size,
    "mode": format(p.stat().st_mode & 0o777, "04o"), "sha256": digest(p.read_bytes())}
    for p in sorted(RAW.iterdir()) if p.is_file()})
