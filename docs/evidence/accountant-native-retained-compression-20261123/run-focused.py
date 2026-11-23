"""Run the selected checks and retain the exact source hashes."""

import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parents[3]
evidence = Path(__file__).resolve().parent
paths = json.loads((evidence / "native-api.json").read_text())["source_sha256"]
def hashes():
    return {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}
selection = (
    "(test_compression_ and not (shared_mixed_capacity or binding_exact_facts or constructor_exact_options or invalid_output)) "
    "or history_compression_host_refuses or summary_shared_helpers or default_compressor_keeps_optional "
    "or real_constructor_first_and_repeated_entrypoints "
    "or retained_title_empty_snippet_completes_local_without_send "
    "or retained_title_semantic_none_has_complete_output_proof "
    "or TestGenerateSummaryNoneContent or TestNonStringContent or TestSummaryFallbackToMainModel "
    "or TestSummaryTargetRatio or TestSummaryPromptBounding or TestSummaryPrefixNormalization "
    "or focus_topic_injected_into_summary_prompt or no_focus_topic_no_injection "
    "or TestResponseFinishReason or length_stop_is_rejected_and_aborts "
    "or length_stop_falls_back_to_main_model_once or stop_finish_reason_still_succeeds"
)
command = ["scripts/run_tests.sh", "--file-retries", "0", "--file-timeout", "900", "-j", "5",
           "tests/agent/test_native_controlled_conversation.py", "tests/agent/test_content_policy_auxiliary.py",
           "tests/agent/test_context_compressor.py", "tests/agent/test_compress_focus.py",
           "tests/agent/test_compressor_truncated_summary_guard.py", "-k", selection, "--", "--durations=12"]
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
before = hashes()
assert before == paths
(evidence / "frozen-source.json").write_text(json.dumps(dict(
    source_sha256=before,
    consumer_sha256=hashlib.sha256((evidence / "consumer-contract.md").read_bytes()).hexdigest(),
    api_sha256=hashlib.sha256((evidence / "native-api.json").read_bytes()).hexdigest(),
    frozen_utc=started,
), indent=2) + "\n")
start = time.monotonic()
with (evidence / "final-focused.log").open("w") as log:
    result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
receipt = dict(command=command, exit_code=result.returncode,
               started_utc=started, wall_seconds=round(time.monotonic() - start, 3),
               source_sha256=before, source_hashes_unchanged=before == hashes(),
               file_retries=0, file_timeout_seconds=900,
               scope="Focused final selection. The separate current-source 26-case feedback-08 scope is not a fresh combined suite.")
(evidence / "final-run.json").write_text(json.dumps(receipt, indent=2) + "\n")
raise SystemExit(result.returncode)
