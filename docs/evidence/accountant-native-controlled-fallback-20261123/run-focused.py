"""Retain each focused selection with its source hashes and timings."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[3]
evidence = Path(__file__).resolve().parent
name, selection, *paths = sys.argv[1:]
assert not (evidence / (name + ".json")).exists()
source = ["agent/native_main_payload.py", "agent/chat_completion_helpers.py", "agent/conversation_loop.py", "run_agent.py", "tests/agent/test_native_controlled_fallback.py"]
def hashes():
    return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in source}
before = hashes()
command = ["scripts/run_tests.sh", "--file-retries", "0", "--file-timeout", "1800", "-j", "4", *paths, "-k", selection, "--", "--durations=10", "--tb=short"]
if len(paths) == 1:
    command += ["--junitxml=" + str(evidence / (name + ".xml"))]
env = dict(os.environ)
env["HOME"] = str(root / ".task-home")
Path(env["HOME"]).mkdir(exist_ok=True)
(evidence / (name + ".patch")).write_bytes(subprocess.check_output(["git", "diff", "HEAD", "--", *source]))
(evidence / (name + "-tests.py")).write_bytes((root / source[-1]).read_bytes())
start = time.monotonic()
with (evidence / (name + ".log")).open("w") as log:
    result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
receipt = dict(command=command, exit_code=result.returncode, wall_seconds=round(time.monotonic()-start, 3), source_sha256=before, source_unchanged=before==hashes(), file_retries=0, isolated_home=env["HOME"])
(evidence / (name + ".json")).write_text(json.dumps(receipt, indent=2)+"\n")
print(json.dumps(receipt))
raise SystemExit(result.returncode)
