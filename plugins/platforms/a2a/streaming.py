"""Bounded A2A SSE transport. Never replay a send after opening a stream."""

import json
import time
import urllib.request

from . import protocol

MAX_EVENT_BYTES = 262144
MAX_STREAM_BYTES = 8 * 1024 * 1024
MAX_EVENTS = 4096


def send_stream(url, body, headers, timeout, *, context_id, task_id,
                notify, auth_values, peer, origin):
    from .tools import _PeerError, _PeerRedirectHandler, _redact_auth, _short_state, _validate_headers, _recoverable_peer

    def invalid():
        return _PeerError("Error: invalid or interrupted peer stream; execution may have started. Do not replay the request.")

    headers = {"Content-Type": "application/json", "Accept": "text/event-stream", **headers}
    _validate_headers(headers)
    encoded = json.dumps(body).encode("utf-8")
    if len(encoded) > MAX_EVENT_BYTES:
        raise _PeerError("Error: streaming request exceeds the size limit.")
    request = urllib.request.Request(url, data=encoded, headers=headers, method="POST")
    deadline = time.monotonic() + timeout
    task = None
    total = count = 0
    terminal = {"completed", "failed", "canceled", "rejected", "input-required", "auth-required"}

    def clean_parts(container, pending=False):
        # Only final text (or explicit pending-question data) is model input.
        # Tool data, reasoning, metadata and partial text never enter progress.
        kept = []
        for part in container.get("parts", []):
            if not isinstance(part, dict):
                raise invalid()
            meta = part.get("metadata") or {}
            if any(meta.get(key) for key in ("adk_thought", "kagent_thought", "thought")):
                continue
            if isinstance(part.get("text"), str):
                kept.append({"text": part["text"]})
            elif pending and isinstance(part.get("data"), dict):
                kept.append({"data": part["data"]})
        return {"parts": kept}

    def persist(state):
        metadata = {"version": 1, "peer": peer, "origin": origin,
                    "task_id": task["id"], "context_id": task["contextId"], "state": state}
        if _recoverable_peer(peer) and _redact_auth(json.dumps(metadata), auth_values) == json.dumps(metadata):
            protocol.persist_message(task["contextId"], "agent", "", task["id"],
                                     request_id=body["id"], peer_task=metadata)

    def consume(envelope):
        nonlocal task, context_id, task_id, count
        count += 1
        if (count > MAX_EVENTS or not isinstance(envelope, dict)
                or envelope.get("jsonrpc") != "2.0" or envelope.get("id") != body["id"]
                or "error" in envelope or not isinstance(envelope.get("result"), dict)):
            raise invalid()
        result = envelope["result"]
        event = result
        for key in ("task", "message", "statusUpdate", "artifactUpdate"):
            if key in result:
                if len(result) != 1 or not isinstance(result[key], dict):
                    raise invalid()
                event = result[key]
                break
        is_task = "id" in event and "status" in event
        incoming_task = event.get("id") if is_task else event.get("taskId")
        incoming_context = event.get("contextId")
        if "parts" in event and not incoming_task:
            if task is not None or task_id or (context_id and incoming_context != context_id):
                raise invalid()
            return {"contextId": incoming_context, **clean_parts(event)}
        for value in (incoming_task, incoming_context):
            if (not isinstance(value, str) or not value or len(value) > 1024
                    or any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in value)
                    or _redact_auth(value, auth_values) != value):
                raise invalid()
        if (task_id and incoming_task != task_id) or (context_id and incoming_context != context_id):
            raise invalid()
        task_id, context_id = incoming_task, incoming_context
        if task is None:
            task = {"id": task_id, "contextId": context_id, "status": {"state": "unknown"}, "artifacts": []}
            persist("unknown")
        if "status" in event:
            status = event["status"]
            state = _short_state(status.get("state", ""))
            if state not in terminal | {"submitted", "working", "unknown"}:
                raise invalid()
            message = status.get("message") or {}
            if state == "working":
                for part in message.get("parts", []):
                    meta = part.get("metadata") or {}
                    if meta.get("adk_thought") or meta.get("kagent_thought"):
                        continue
                    kind = meta.get("adk_type", meta.get("kagent_type"))
                    data = part.get("data") or {}
                    if kind == "function_call" and isinstance(data, dict):
                        name = data.get("name")
                        notify("tool:" + name if isinstance(name, str) and len(name) <= 128 else "tool")
                    elif kind == "function_response":
                        response = data.get("response") if isinstance(data, dict) else None
                        is_error = isinstance(response, dict) and (response.get("isError") is True or bool(response.get("error")))
                        notify("tool_error" if is_error else "tool_result")
            task["status"] = {"state": status["state"]}
            if state in terminal:
                task["status"]["message"] = clean_parts(message, state == "input-required")
            persist(state)
            if is_task:
                task["artifacts"] = [clean_parts(a) for a in event.get("artifacts", [])]
            if state in terminal:
                return task
        if "artifact" in event:
            artifact = event["artifact"]
            artifact_id = artifact.get("artifactId")
            if not isinstance(artifact_id, str) or not artifact_id or len(artifact_id) > 1024:
                raise invalid()
            existing = next((a for a in task["artifacts"] if a.get("artifactId") == artifact_id), None)
            if event.get("append") is True:
                if existing is None or existing.get("_closed"):
                    raise invalid()
                existing["parts"].extend(clean_parts(artifact)["parts"])
                existing["_closed"] = event.get("lastChunk") is True
            else:
                if existing is not None:
                    raise invalid()
                task["artifacts"].append({"artifactId": artifact_id, **clean_parts(artifact),
                                          "_closed": event.get("lastChunk") is True})
        return None

    try:
        with urllib.request.build_opener(_PeerRedirectHandler()).open(request, timeout=timeout) as response:
            if response.headers.get_content_type() != "text/event-stream":
                raise invalid()
            lines = []
            size = 0
            while True:
                if time.monotonic() >= deadline:
                    raise invalid()
                line = response.readline(MAX_EVENT_BYTES + 1)
                total += len(line)
                size += len(line)
                if size > MAX_EVENT_BYTES or total > MAX_STREAM_BYTES:
                    raise invalid()
                if not line:
                    raise invalid()
                if line in (b"\n", b"\r\n"):
                    data = b"\n".join(lines)
                    if data.strip():
                        final = consume(json.loads(data.decode("utf-8")))
                        if final is not None:
                            return {"result": final}
                    lines, size = [], 0
                elif line.startswith(b"data:"):
                    lines.append(line[5:].removeprefix(b" ").rstrip(b"\r\n"))
    except Exception:
        if task is not None:
            persist("unknown")
        raise invalid() from None
