#!/usr/bin/env python3
"""Explicitly scoped actual-controller acceptance, not a passthrough substitute.

Read configuration on stdin: url, token, task_id, context_id, and resume (bool).
Only configuration with resume=true sends the explicit
synthetic blue answer. No tasks are created, canceled or retried. Run in an
isolated test Pod with access only to the designated public gateway.
"""
import json
import os
from pathlib import Path
import sys
import tempfile


def main():
    cfg = json.load(sys.stdin)
    assert cfg["task_id"] == "01a08220-774d-75d7-8096-77998baf5aa1"
    assert cfg["context_id"] == "ctx-a209dd2644ae42b3"
    assert cfg["url"] == "http://pilot-gateway.kagent-system.svc.cluster.local:8083/api/a2a/kagent-system/groundskeeper-108157884/"
    with tempfile.TemporaryDirectory(prefix="controller-lookup-", dir="/tmp") as home:
        os.environ["HOME"] = os.environ["HERMES_HOME"] = home
        os.environ["SYNTHETIC_CALLER_KEY"] = cfg["token"]
        (Path(home) / "config.yaml").write_text(json.dumps({"a2a_agents": {"controller-fixture": {
            "url": cfg["url"], "timeout": 120, "auth": {"type": "bearer", "token": "${env:SYNTHETIC_CALLER_KEY}"}}}}))
        from plugins.platforms.a2a import tools, protocol
        peer = tools._resolve_peer("controller-fixture")
        headers = tools._auth_header(peer["auth"])
        card = tools._fetch_card(peer["url"], headers, 15)
        endpoint = tools._rpc_url(peer["url"], card)
        assert endpoint.rstrip("/") == peer["url"].rstrip("/"), "unexpected canonical peer endpoint"
        iface = tools._select_jsonrpc_interface(card)
        assert iface["protocolVersion"] == "1.0", "unexpected advertised protocol"
        original = tools._http_post_json
        matrix = []
        pending = None
        for method, version in (("GetTask", "1.0"), ("tasks/get", "1.0"), ("tasks/get", "0.3")):
            request_id = protocol.new_task_id()
            response = original(endpoint, {"jsonrpc": "2.0", "id": request_id, "method": method,
                "params": {"id": cfg["task_id"]}}, {**headers, "A2A-Version": version}, 20)
            assert response.get("jsonrpc") == "2.0" and response.get("id") == request_id
            task = protocol.unwrap_send_message_response(response.get("result", {}))
            matrix.append({"method": method, "version": version, "error_code": (response.get("error") or {}).get("code"),
                "state": (task.get("status") or {}).get("state"), "matching_envelope": True})
            if method == "tasks/get" and version == "0.3":
                assert task.get("id") == cfg["task_id"] and task.get("contextId") == cfg["context_id"]
                assert tools._short_state(task["status"]["state"]) == "input-required"
                question = protocol.extract_text(task["status"].get("message") or {}).lower()
                assert "blue" in question and "red" in question, "not the authorized synthetic color question"
                pending = task
        print(json.dumps({"check": "actual_controller_method_matrix", "matrix": matrix}), flush=True)
        assert pending is not None
        if not cfg.get("resume", False):
            return
        observed = []

        def observe(url, body, request_headers, timeout):
            response = original(url, body, request_headers, timeout)
            assert url == endpoint
            assert request_headers["Authorization"] == headers["Authorization"]
            observed.append({"method": body["method"], "version": request_headers["A2A-Version"],
                "error_code": (response.get("error") or {}).get("code"),
                "matching_envelope": response.get("id") == body["id"] and response.get("jsonrpc") == "2.0"})
            return response

        tools._http_post_json = observe  # observation only, no request/response changes
        try:
            reply = tools.a2a_call({"agent": "controller-fixture", "message": "", "task_id": cfg["task_id"],
                "context_id": cfg["context_id"], "data": [{"decision_type": "approve", "ask_user_answers": [{"answer": ["blue"]}]}]})
        finally:
            tools._http_post_json = original
        assert cfg["token"] not in reply
        assert not reply.startswith("Error:"), "native resume failed; no send retry performed"
        methods = [entry["method"] for entry in observed]
        assert methods in (["GetTask", "SendMessage"], ["GetTask", "tasks/get", "SendMessage"]), methods
        assert all(entry["matching_envelope"] for entry in observed)
        assert observed[-1]["version"] == "1.0"
        assert cfg["task_id"] in reply and cfg["context_id"] in reply
        request_id = protocol.new_task_id()
        response = original(endpoint, {"jsonrpc": "2.0", "id": request_id, "method": "tasks/get",
            "params": {"id": cfg["task_id"]}}, {**headers, "A2A-Version": "0.3"}, 20)
        assert response.get("id") == request_id and "error" not in response
        result = protocol.unwrap_send_message_response(response["result"])
        assert result["id"] == cfg["task_id"] and result["contextId"] == cfg["context_id"]
        assert tools._short_state(result["status"]["state"]) == "completed"
        text = tools._reply_text_from_result(result)
        assert "blue" in text.lower(), "completed artifact did not acknowledge the explicit color answer"
        print(json.dumps({"check": "actual_controller_packaged_resume", "task_id": result["id"],
            "context_id": result["contextId"], "state": "completed", "blue_acknowledged": True,
            "calls": observed, "fallback_observed": "tasks/get" in methods,
            "explicit_sends": 1, "write_retries": 0}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"check": "actual_controller_acceptance_failed", "exception_type": type(error).__name__,
            "note": "diagnostics suppressed; do not retry a possibly accepted SendMessage"}), flush=True)
        raise SystemExit(1) from None
