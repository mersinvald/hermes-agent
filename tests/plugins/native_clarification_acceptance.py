#!/usr/bin/env python3
"""Native Go acceptance using the read-only infra fixture's test-only Go overlay.

Requires --fixture-source and the pinned --kagent-source; only the LLM is fake.
No runtime code is replaced. No production services or credentials are used.
"""
import argparse
import json
import os
from pathlib import Path
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
    original_post = tools._http_post_json
    sent = []

    def observe(url, body, headers, timeout):
        result = original_post(url, body, headers, timeout)
        if body["method"] in ("SendMessage", "message/send"):
            sent.append((body, result))
        return result

    tools._http_post_json = observe

    def rpc(method, params):
        response = original_post(args.backend, {"jsonrpc": "2.0", "id": str(uuid.uuid4()),
            "method": method, "params": params}, {}, 10)
        assert "error" not in response, response
        return response["result"]

    def latest():
        return protocol.unwrap_send_message_response(sent[-1][1]["result"])

    pending = set()
    try:
        output = tools.a2a_call({"agent": args.url, "message": "ASK_FIXTURE"})
        initial = latest()
        task, context = initial["id"], initial["contextId"]
        pending.add(task)
        assert task in output and context in output and "Which synthetic room?" in output, output
        # Context-only calls keep their original new-task semantics.
        tools.a2a_call({"agent": args.url, "message": "fixture-room", "context_id": context})
        assert latest()["id"] != task
        assert rpc("tasks/get", {"id": task})["status"]["state"] == "input-required"
        decision = {"decision_type": "approve", "ask_user_answers": [
            {"answer": ["fixture-room"]}, {"answer": ["fixture-time"]}]}
        # This explicit test response concerns the sole installed tool, ask_user.
        # It is not evidence that model-controlled approvals are safe for mutation.
        output = tools.a2a_call({"agent": args.url, "message": "", "task_id": task,
            "context_id": context, "data": [decision]})
        assert not output.startswith("Error:"), output
        resumed = latest()
        assert resumed["id"] == task and resumed["contextId"] == context, resumed
        assert resumed["status"]["state"] == "TASK_STATE_COMPLETED", resumed
        assert "native ask_user result contains fixture-room" in output, output
        wire = sent[-1][0]["params"]["message"]
        assert wire["taskId"] == task and wire["parts"][0]["data"] == decision, wire
        assert rpc("tasks/get", {"id": task})["status"]["state"] == "completed"
        pending.remove(task)
        print(json.dumps({"check": "native_hermes_resume", "task_id": task, "context_id": context,
            "same_task": True, "actual_ask_user_answer": True, "state": "completed"}), flush=True)
        initial["artifacts"] = [{"parts": [{"text": "Prior partial output"}]}]
        for state in ("TASK_STATE_INPUT_REQUIRED", "input-required"):
            initial["status"]["state"] = state
            assert "Which synthetic room?" in tools._reply_text_from_result(initial)
            assert "Prior partial output" not in tools._reply_text_from_result(initial)
        before = len(sent)
        for bad_task, bad_context in ((task, context), ("unknown", context), (task, "wrong")):
            result = tools.a2a_call({"agent": args.url, "message": "", "task_id": bad_task,
                "context_id": bad_context, "data": [decision]})
            assert result.startswith("Error:"), result
        assert len(sent) == before
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
    parser.add_argument("--client", action="store_true")
    parser.add_argument("--url")
    parser.add_argument("--backend")
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
        subprocess.run([str(Path(goroot) / "bin/go"), "test", "-mod=readonly", "-overlay", str(overlay),
            "./core/internal/a2a", "-run", "^TestNativeClarification$", "-count=1", "-v", "-timeout=100s"],
            cwd=root / "go", env=env, check=True, timeout=240)


if __name__ == "__main__":
    main()
