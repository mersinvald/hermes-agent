"""Check the correction without another cumulative selection."""

import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parents[3]
evidence = Path(__file__).resolve().parent
pins = json.loads((evidence / "native-api.json").read_text())["source_sha256"]
selection = (
    "compression_complete_retained_first_summary or compression_ordered_interleaving "
    "or (compression_cancellation and (before_publication or post_publication)) "
    "or compression_complete_source_cannot_bypass_request_text_bound"
)
command = ["scripts/run_tests.sh", "--file-retries", "0", "--file-timeout", "900", "-j", "1",
           "tests/agent/test_native_controlled_conversation.py", "-k", selection, "--", "--durations=8"]
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
receipt = dict(command=command, started_utc=started, source_sha256=pins,
               api_sha256=hashlib.sha256((evidence / "native-api.json").read_bytes()).hexdigest(),
               consumer_sha256=hashlib.sha256((evidence / "consumer-contract.md").read_bytes()).hexdigest())
(evidence / "frozen-correction.json").write_text(json.dumps(receipt, indent=2) + "\n")
start = time.monotonic()
with (evidence / "correction-focused.log").open("w") as log:
    result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
receipt.update(exit_code=result.returncode, wall_seconds=round(time.monotonic() - start, 3),
               file_retries=0, source_hashes_unchanged=all(
                   hashlib.sha256((root / path).read_bytes()).hexdigest() == value for path, value in pins.items()),
               scope="Five current-source correction cases. The earlier 98 passes and two failures remain separate.")
(evidence / "correction-run.json").write_text(json.dumps(receipt, indent=2) + "\n")
raise SystemExit(result.returncode)
