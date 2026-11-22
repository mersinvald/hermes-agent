"""Real manager entrypoints deny before callbacks or background scheduling."""

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from agent.content_policy import NativeContentPolicyDenied, content_policy_scope
from agent.memory_manager import MemoryManager
from tests.agent.test_content_policy import make_context


@pytest.mark.parametrize("function,args,kwargs", [
    ("prefetch_all", ("invented",), {}), ("_prefetch_provider", (object(), "invented"), {}),
    ("queue_prefetch_all", ("invented",), {}), ("sync_all", ("invented", "reply"), {}),
    ("handle_tool_call", ("invented", {}), {}), ("on_turn_start", (1, "invented"), {}),
    ("on_session_end", ([],), {}), ("on_pre_compress", ([],), {}),
    ("on_memory_write", ("add", "memory", "invented"), {}),
    ("on_delegation", ("invented", "reply"), {}),
    ("commit_session_boundary_async", ([],), {"new_session_id": "next"}),
    ("on_session_switch", ("next",), {}), ("_submit_background", (lambda: pytest.fail("worker ran"),), {}),
])
def test_sensitive_memory_refuses_before_any_provider_or_worker(function, args, kwargs):
    context, state = make_context()
    manager = MemoryManager()
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(context):
            state["binding"] = replace(state["binding"], profiles=("restricted-documents",))
            getattr(manager, function)(*args, **kwargs)
            pytest.fail("memory entrypoint returned")


def test_ordinary_prefetch_without_providers_keeps_empty_result():
    context, _ = make_context()
    with content_policy_scope(context):
        assert MemoryManager().prefetch_all("invented") == ""


def test_queued_memory_rechecks_current_binding_before_worker_callback():
    context, state = make_context()
    manager = MemoryManager()
    gate = threading.Event()
    calls = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        blocker = executor.submit(gate.wait, 5)
        manager._sync_executor = executor
        try:
            with content_policy_scope(context):
                manager._submit_background(lambda: calls.append("invented write"))
            state["binding"] = replace(state["binding"], profiles=("restricted-documents",))
        finally:
            gate.set()
        blocker.result(timeout=5)
        manager.flush_pending(timeout=5)
    assert calls == [] and context._denied
