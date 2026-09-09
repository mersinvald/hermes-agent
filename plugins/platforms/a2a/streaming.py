"""Bounded A2A SSE transport. Never replay a send after opening a stream."""

import json
import hashlib
import time
import urllib.request

from . import protocol

MAX_EVENT_BYTES = 262144
MAX_STREAM_BYTES = 8 * 1024 * 1024
MAX_EVENTS = 4096


def send_stream(url, body, headers, timeout, *, context_id, task_id,
                notify, auth_values, peer, origin, initial_task=None):
    from .tools import _PeerError, _PeerRedirectHandler, _redact_auth, _short_state, _validate_headers, _recoverable_peer

    def invalid():
        return _PeerError("Error: invalid, unsupported or interrupted peer stream. Do not replay the request. Use explicit action=follow with persisted task/context to observe existing work; subscription availability does not imply gapless replay.")

    headers = {"Content-Type": "application/json", "Accept": "text/event-stream", **headers}
    _validate_headers(headers)
    encoded = json.dumps(body).encode("utf-8")
    if len(encoded) > MAX_EVENT_BYTES:
        raise _PeerError("Error: streaming request exceeds the size limit.")
    request = urllib.request.Request(url, data=encoded, headers=headers, method="POST")
    deadline = time.monotonic() + timeout
    task = None
    total = count = 0
    last_event_id = None
    seen_ids = {}
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
        if last_event_id:
            metadata["last_event_id"] = last_event_id
        if _recoverable_peer(peer) and _redact_auth(json.dumps(metadata), auth_values) == json.dumps(metadata):
            protocol.persist_message(task["contextId"], "agent", "", task["id"],
                                     request_id=body["id"], peer_task=metadata)

    if initial_task is not None:
        task = {"id": task_id, "contextId": context_id, "status": initial_task["status"],
                "artifacts": [{"artifactId": a.get("artifactId"), **clean_parts(a)}
                              for a in initial_task.get("artifacts", [])]}

    def consume(envelope, event_id=None):
        nonlocal task, context_id, task_id, count, last_event_id
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
        if event_id is not None:
            last_event_id = event_id
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
                    if not isinstance(part, dict) or not isinstance(part.get("metadata", {}), dict):
                        continue
                    meta = part.get("metadata") or {}
                    if meta.get("adk_thought") or meta.get("kagent_thought"):
                        continue
                    kind = meta.get("adk_type", meta.get("kagent_type"))
                    data = part.get("data") or {}
                    if not isinstance(data, dict):
                        continue
                    name, call_id = data.get("name"), data.get("id")
                    if (not isinstance(name, str) or not name or len(name) > 128
                            or not isinstance(call_id, str) or not call_id or len(call_id) > 1024):
                        continue
                    if kind == "function_call" and isinstance(data, dict):
                        # Go ADK omits args for an empty argument object.
                        if not isinstance(data.get("args", {}), dict):
                            continue
                        notify("tool:" + name if isinstance(name, str) and len(name) <= 128 else "tool")
                    elif kind == "function_response":
                        response = data.get("response") if isinstance(data, dict) else None
                        if not isinstance(response, dict):
                            continue
                        is_error = isinstance(response, dict) and (response.get("isError") is True or bool(response.get("error")))
                        notify("tool_error" if is_error else "tool_result")
            task["status"] = {"state": status["state"]}
            if state in terminal:
                task["status"]["message"] = clean_parts(message, state == "input-required")
            persist(state)
            if is_task:
                task["artifacts"] = [{"artifactId": a.get("artifactId"), **clean_parts(a)} for a in event.get("artifacts", [])]
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
                added = clean_parts(artifact)["parts"]
                if existing["parts"] and added and "text" in existing["parts"][-1] and "text" in added[0]:
                    existing["parts"][-1]["text"] += added.pop(0)["text"]
                existing["parts"].extend(added)
                existing["_closed"] = event.get("lastChunk") is True
            else:
                if existing is not None:
                    existing.update({**clean_parts(artifact), "_closed": event.get("lastChunk") is True})
                else:
                    task["artifacts"].append({"artifactId": artifact_id, **clean_parts(artifact),
                                              "_closed": event.get("lastChunk") is True})
        return None

    try:
        with urllib.request.build_opener(_PeerRedirectHandler()).open(request, timeout=timeout) as response:
            if response.headers.get_content_type() != "text/event-stream":
                raise invalid()
            lines = []
            event_id = None
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
                        digest = hashlib.sha256(data).digest()
                        if event_id in seen_ids:
                            if seen_ids[event_id] != digest:
                                raise invalid()
                            lines, size, event_id = [], 0, None
                            continue
                        if event_id is not None:
                            if len(seen_ids) >= MAX_EVENTS:
                                raise invalid()
                            seen_ids[event_id] = digest
                        final = consume(json.loads(data.decode("utf-8")), event_id)
                        if final is not None:
                            return {"result": final}
                    lines, size, event_id = [], 0, None
                elif line.startswith(b"data:"):
                    lines.append(line[5:].removeprefix(b" ").rstrip(b"\r\n"))
                elif line.startswith(b"id:"):
                    event_id = line[3:].removeprefix(b" ").rstrip(b"\r\n").decode("utf-8")
                    if (len(event_id) > 1024 or any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in event_id)
                            or _redact_auth(event_id, auth_values) != event_id):
                        raise invalid()
    except Exception:
        if task is not None:
            persist("unknown")
        raise invalid() from None
