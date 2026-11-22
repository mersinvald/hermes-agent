"""Native context checks with invented bindings and the installed shared engine."""

import copy
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
from hermes_content_policy import ExecutionBinding, PATHS

from agent.content_policy import (
    NativeContentContext, NativeContentPolicyDenied, content_policy_scope,
    guard_native_content, guard_tool_result,
)


def make_context(*, profiles=(), result_profiles=None):
    request = object()
    state = {"binding": ExecutionBinding("owner", "execution", "lineage", "native",
                                        7, 9, "selection", 3, profiles)}

    class Admission:
        def current(self, handle):
            assert handle is request
            return state["binding"]

    context = NativeContentContext(Admission(), request, result_profiles=result_profiles)
    return context, state


def enabled_agent(context):
    return SimpleNamespace(_native_content_policy_required=True,
                           _native_content_policy_context=context)


@pytest.mark.parametrize("path", PATHS)
def test_explicit_ordinary_binding_uses_shared_admission_on_each_native_path(path):
    context, state = make_context()
    assert guard_native_content(path, enabled_agent(context)) is context
    assert context._binding == state["binding"]


@pytest.mark.parametrize("path", PATHS)
def test_specialist_sensitive_binding_has_no_native_transport(path):
    context, _ = make_context(profiles=("restricted-documents",))
    with pytest.raises(NativeContentPolicyDenied, match="sensitive_native"):
        guard_native_content(path, enabled_agent(context))
    assert context._denied


@pytest.mark.parametrize("field,value", [
    ("owner_ref", "other"), ("execution_ref", "other"), ("lineage_ref", "other"),
    ("consumer_ref", "other"), ("auth_epoch", 8), ("policy_revision", 10),
    ("selection_id", "other"), ("selection_version", 4),
])
def test_current_facts_cannot_reuse_an_earlier_context(field, value):
    context, state = make_context()
    context.check("main")
    state["binding"] = replace(state["binding"], **{field: value})
    with pytest.raises(NativeContentPolicyDenied):
        context.check("main")


def test_missing_copied_and_foreign_contexts_grant_no_authority():
    context, _ = make_context()
    for value in (None, object(), copy.copy(context)):
        with pytest.raises(NativeContentPolicyDenied):
            guard_native_content("main", enabled_agent(value))
    other, _ = make_context()
    with content_policy_scope(context):
        with pytest.raises(NativeContentPolicyDenied):
            guard_native_content("main", enabled_agent(other))


@pytest.mark.parametrize("result_profiles", [None, lambda _: None, lambda _: ("restricted-documents",)])
def test_unknown_or_sensitive_result_invalidates_earlier_ordinary_context(result_profiles):
    context, _ = make_context(result_profiles=result_profiles)
    agent = enabled_agent(context)
    guard_native_content("main", agent)
    with pytest.raises(NativeContentPolicyDenied):
        guard_tool_result(agent, "invented-producer")
    with pytest.raises(NativeContentPolicyDenied, match="stale"):
        guard_native_content("fallback", agent)


def test_authenticated_ordinary_result_keeps_ordinary_binding():
    calls = []
    context, _ = make_context(result_profiles=lambda producer: calls.append(producer) or ())
    guard_tool_result(enabled_agent(context), "document-review")
    assert calls == ["document-review"]
    context.check("main")


def test_ordinary_worker_constructs_its_own_shared_runtime():
    context, _ = make_context()
    context.check("main")
    errors = []

    def run():
        try:
            context.check("memory")
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(2)
    assert not thread.is_alive() and errors == []


def test_missing_canonical_dependency_denies_enabled_work(monkeypatch):
    import agent.content_policy as native
    context, _ = make_context()
    monkeypatch.setattr(native.importlib, "import_module", lambda _: (_ for _ in ()).throw(ImportError()))
    with pytest.raises(NativeContentPolicyDenied, match="dependency_missing"):
        context.check("main")


def test_context_reentry_invalidates_even_when_the_host_catches_the_inner_error():
    context, state = make_context()
    original = context.admission.current

    def reentrant(handle):
        try:
            context.check("main")
        except NativeContentPolicyDenied:
            pass
        return original(handle)

    context.admission.current = reentrant
    with pytest.raises(NativeContentPolicyDenied):
        context.check("main")
    assert context._denied and state["binding"].profiles == ()
