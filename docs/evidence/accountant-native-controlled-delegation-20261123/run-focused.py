"""Record one exact native selection through the required test runner."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
name, selection = sys.argv[1:3]
files = ("agent/native_main_payload.py", "tools/delegate_tool.py",
         "tests/tools/test_native_controlled_delegation.py")


def hashes():
    return {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in files}


xml = out / f"{name}.xml"
command = ["scripts/run_tests.sh", "--file-retries", "0", "--file-timeout", sys.argv[3] if len(sys.argv) > 3 else "120",
           "-j", "1", sys.argv[4] if len(sys.argv) > 4 else "tests/tools/test_native_controlled_delegation.py", "-k", selection,
           "--", "--durations=0", "--tb=short", f"--junitxml={xml}"]
receipt = dict(name=name, command=command, source_sha256=hashes(),
               native_source_scope=str(root), isolated_home=str(root / ".task-home"), retries=0)
environment = os.environ.copy()
environment["HOME"] = str(root / ".task-home")
start = time.monotonic()
with (out / f"{name}.log").open("w") as log:
    result = subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
receipt.update(exit_code=result.returncode, wall_seconds=round(time.monotonic() - start, 3),
               source_unchanged=hashes() == receipt["source_sha256"])
if xml.exists():
    receipt["cases"] = [dict(classname=case.get("classname"), name=case.get("name"),
        seconds=case.get("time"), properties={p.get("name"): p.get("value") for p in case.findall("properties/property")},
        status="failed" if case.find("failure") is not None else
        "error" if case.find("error") is not None else "skipped" if case.find("skipped") is not None else "passed")
        for case in ET.parse(xml).iter("testcase")]
(out / f"{name}.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({key: receipt[key] for key in ("name", "exit_code", "wall_seconds", "source_unchanged")}))
sys.exit(result.returncode)
