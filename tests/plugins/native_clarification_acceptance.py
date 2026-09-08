#!/usr/bin/env python3
"""Native Go acceptance using the read-only infra fixture's test-only Go overlay.

Requires --fixture-source and the pinned --kagent-source; only the LLM is fake.
No runtime code is replaced. No production services or credentials are used.
"""
import argparse
import json
import os
from pathlib import Path
import re
import runpy
import socket
import subprocess
import sys
import tempfile
import urllib.request
import uuid


def client(args):
    original_connect = socket.socket.connect

    def connect(sock, address):
        if not isinstance(address, tuple) or address[0] != "127.0.0.1":
            raise RuntimeError("fixture permits IPv4 loopback only")
        return original_connect(sock, address)

    socket.socket.connect = connect
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    sys.path.insert(0, args.hermes_source)
    from plugins.platforms.a2a import tools, protocol

    def rpc(method, params):
        response = tools._http_post_json(args.backend, {"jsonrpc": "2.0", "id": str(uuid.uuid4()),
            "method": method, "params": params}, {}, 10)
        assert "error" not in response, response
        return response["result"]

    def returned_ids(output):
        match = re.search(r"context ([^ ]+) \u00b7 task ([^ ]+)", output)
        assert match, output
        return match[2], match[1]

    if args.restore_context:
        history = tools.a2a_history({"context_id": args.restore_context})
        tasks = [json.loads(line[6:]) for line in history.splitlines() if line.startswith("Task: ")]
        pending_task, = [task for task in tasks if task["state"] == "input-required"]
        assert "Which synthetic room?" in history, history
        decision = {"decision_type": "approve", "ask_user_answers": [
            {"answer": ["fixture-room"]}, {"answer": ["fixture-time"]}]}
        output = tools.a2a_call({"agent": pending_task["peer"], "message": "",
            "task_id": pending_task["task_id"], "context_id": pending_task["context_id"], "data": [decision]})
        assert not output.startswith("Error:"), output
        returned_task, returned_context = returned_ids(output)
        assert returned_task == pending_task["task_id"] and returned_context == pending_task["context_id"]
        result = rpc("tasks/get", {"id": returned_task})
        assert result["id"] == pending_task["task_id"]
        assert result["contextId"] == pending_task["context_id"]
        assert result["status"]["state"] == "completed"
        assert "native ask_user result contains fixture-room" in output, output
        history = tools.a2a_history({"context_id": args.restore_context})
        latest_tasks = [json.loads(line[6:]) for line in history.splitlines() if line.startswith("Task: ")]
        assert all(task["state"] == "completed" for task in latest_tasks)
        print(json.dumps({"check": "holdauth_fresh_process_resume", "task_id": result["id"],
                          "context_id": result["contextId"], "state": "completed"}), flush=True)
        return

    # Real config loading in both processes; no runtime substitution for recovery.
    (Path(os.environ["HERMES_HOME"]) / "config.yaml").write_text(json.dumps({
        "a2a_agents": {"holdauth-fixture": {"url": args.url}}}))
    pending = set()
    try:
        output = tools.a2a_call({"agent": "holdauth-fixture", "message": "ASK_FIXTURE"})
        task, context = returned_ids(output)
        initial = rpc("tasks/get", {"id": task})
        assert initial["id"] == task and initial["contextId"] == context
        pending.add(task)
        assert task in output and context in output and "Which synthetic room?" in output, output
        # Context-only calls keep their original new-task semantics.
        output = tools.a2a_call({"agent": "holdauth-fixture", "message": "fixture-room", "context_id": context})
        other_task, other_context = returned_ids(output)
        assert other_task != task and other_context == context
        assert rpc("tasks/get", {"id": other_task})["status"]["state"] == "completed"
        assert rpc("tasks/get", {"id": task})["status"]["state"] == "input-required"
        decision = {"decision_type": "approve", "ask_user_answers": [
            {"answer": ["fixture-room"]}, {"answer": ["fixture-time"]}]}
        # This explicit test response concerns the sole installed tool, ask_user.
        # It is not evidence that model-controlled approvals are safe for mutation.
        restored = subprocess.check_output([sys.executable, str(Path(__file__).resolve()),
            "--client", "--hermes-source", args.hermes_source, "--url", args.url,
            "--backend", args.backend, "--restore-context", context], text=True)
        resumed = json.loads(restored)
        assert resumed["task_id"] == task and resumed["context_id"] == context, resumed
        assert resumed["state"] == "completed", resumed
        print(restored.strip(), flush=True)
        assert rpc("tasks/get", {"id": task})["status"]["state"] == "completed"
        pending.remove(task)
        print(json.dumps({"check": "native_hermes_resume", "task_id": task, "context_id": context,
            "same_task": True, "actual_ask_user_answer": True, "state": "completed"}), flush=True)
        initial["artifacts"] = [{"parts": [{"text": "Prior partial output"}]}]
        for state in ("TASK_STATE_INPUT_REQUIRED", "input-required"):
            initial["status"]["state"] = state
            assert "Which synthetic room?" in tools._reply_text_from_result(initial)
            assert "Prior partial output" not in tools._reply_text_from_result(initial)
        for bad_task, bad_context in ((task, context), ("unknown", context), (task, "wrong")):
            result = tools.a2a_call({"agent": "holdauth-fixture", "message": "", "task_id": bad_task,
                "context_id": bad_context, "data": [decision]})
            assert result.startswith("Error:"), result
        print(json.dumps({"check": "projection_acceptance", "status": "PASS", "unresolved": 0}), flush=True)
    finally:
        for task in pending:
            rpc("tasks/cancel", {"id": task})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hermes-source", required=True)
    parser.add_argument("--kagent-source")
    parser.add_argument("--fixture-source")
    parser.add_argument("--require-supported", action="store_true")
    parser.add_argument("--compile-dir", type=Path, help="Cross-compile static linux arm64/amd64 native test binaries here instead of running locally")
    parser.add_argument("--client", action="store_true")
    parser.add_argument("--url")
    parser.add_argument("--backend")
    parser.add_argument("--restore-context")
    args = parser.parse_args()
    if args.client:
        client(args)
        return
    if not args.kagent_source or not args.fixture_source:
        parser.error("--kagent-source and --fixture-source are required")
    fixture = runpy.run_path(args.fixture_source)
    root = Path(args.kagent_source).resolve()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    assert revision == fixture["KAGENT_PIN"]
    assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root)
    with tempfile.TemporaryDirectory(prefix="hermes-clarification-") as temporary:
        directory = Path(temporary)
        test = directory / "native_clarification_test.go"
        go_test = fixture["GO_TEST"].replace(
            'strings.Contains(result,"fixture-room")',
            'strings.Contains(result,"fixture-room") && strings.Contains(result,"fixture-time")')
        go_test = go_test.replace(
            '"choices":[]any{"fixture-room","other-room"}}}',
            '"choices":[]any{"fixture-room","other-room"}},map[string]any{"question":"Which synthetic time?","choices":[]any{"fixture-time","other-time"}}}')
        assert go_test != fixture["GO_TEST"] and '"Which synthetic time?"' in go_test
        test.write_text(go_test)
        overlay = directory / "overlay.json"
        overlay.write_text(json.dumps({"Replace": {str(root / "go/core/internal/a2a/native_clarification_test.go"): str(test)}}))
        env = {"PATH": os.environ["PATH"], "HOME": temporary, "HERMES_HOME": temporary,
            "PYTHONDONTWRITEBYTECODE": "1", "GOMAXPROCS": "2", "GOPROXY": "off", "GOSUMDB": "off", "GOTOOLCHAIN": "local",
            "GOPATH": subprocess.check_output(["go", "env", "GOPATH"], text=True).strip(),
            "GOCACHE": subprocess.check_output(["go", "env", "GOCACHE"], text=True).strip(),
            "CLARIFICATION_PYTHON": sys.executable, "CLARIFICATION_SCRIPT": str(Path(__file__).resolve()),
            "CLARIFICATION_HERMES": str(Path(args.hermes_source).resolve())}
        goroot = subprocess.check_output(["go", "env", "GOROOT"], cwd=root / "go", text=True).strip()
        if args.compile_dir:
            args.compile_dir.mkdir(parents=True, exist_ok=True)
            for arch in ("arm64", "amd64"):
                print("Compiling native loopback fixture for linux/" + arch, flush=True)
                subprocess.run([str(Path(goroot) / "bin/go"), "test", "-mod=readonly", "-overlay", str(overlay),
                    "-c", "-ldflags=-s -w", "-o", str(args.compile_dir.resolve() / ("native-clarification-" + arch)),
                    "./core/internal/a2a"], cwd=root / "go", env={**env, "GOOS": "linux", "GOARCH": arch,
                    "CGO_ENABLED": "0"}, check=True, timeout=480)
            return
        subprocess.run([str(Path(goroot) / "bin/go"), "test", "-mod=readonly", "-overlay", str(overlay),
            "./core/internal/a2a", "-run", "^TestNativeClarification$", "-count=1", "-v", "-timeout=100s"],
            cwd=root / "go", env=env, check=True, timeout=240)


if __name__ == "__main__":
    main()
