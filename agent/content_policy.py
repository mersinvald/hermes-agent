"""Native admission and refusal adapters for the shared content policy.

Generic native operations refuse contexts with a controlled main adapter.
Ordinary checks use fresh, thread-local shared runtimes without content transfer.
"""

from __future__ import annotations

import importlib
import asyncio
import threading
import time
import weakref
from contextlib import contextmanager
from contextvars import ContextVar


class NativeContentPolicyDenied(PermissionError):
    """Expose a fixed reason without content or provider diagnostics."""


_active = ContextVar("native_content_policy", default=None)
_contexts = weakref.WeakSet()
_registry_lock = threading.Lock()


def shared_policy():
    """Load only the canonical package delivered by the host."""
    try:
        return importlib.import_module("hermes_content_policy")
    except ImportError:
        raise NativeContentPolicyDenied("content_policy_dependency_missing") from None


def require_ordinary(profiles):
    """Reject unsupported content before any payload operation."""
    if type(profiles) is not tuple or any(type(p) is not str for p in profiles):
        raise NativeContentPolicyDenied("content_policy_binding_missing")
    if profiles:
        raise NativeContentPolicyDenied("sensitive_native_processing_unsupported")


class NativeContentContext:
    """Keep one live host binding and invalidate every use after rejection.

    The host supplies admission, its exact request capability, and optional
    authenticated result-profile lookup. Message fields cannot supply these ports.
    """

    def __init__(self, admission, request_handle, *, result_profiles=None):
        self.admission = admission
        self.request_handle = request_handle
        self.result_profiles = result_profiles
        self._binding = None
        self._denied = False
        self._local = threading.local()
        self._controlled_main_payload = None
        self._controlled_main_bound = False
        with _registry_lock:
            if len(_contexts) >= 128:
                raise NativeContentPolicyDenied("content_policy_capacity")
            _contexts.add(self)

    def invalidate(self):
        self._denied = True
        if self._controlled_main_bound:
            from agent.native_main_payload import invalidate_controlled_context
            invalidate_controlled_context(self)

    def _current(self, request_handle):
        if request_handle is not self.request_handle or self._denied:
            raise NativeContentPolicyDenied("content_policy_context_stale")
        policy = shared_policy()
        binding = self.admission.current(request_handle)
        if type(binding) is not policy.ExecutionBinding:
            raise NativeContentPolicyDenied("content_policy_binding_missing")
        require_ordinary(binding.profiles)
        for name in ("owner_ref", "execution_ref", "lineage_ref", "consumer_ref", "selection_id"):
            if type(getattr(binding, name)) is not str:
                raise NativeContentPolicyDenied("content_policy_binding_invalid")
        for name in ("auth_epoch", "policy_revision", "selection_version"):
            if type(getattr(binding, name)) is not int:
                raise NativeContentPolicyDenied("content_policy_binding_invalid")
        return binding

    def current(self, request_handle):
        binding = self._current(request_handle)
        if self._binding is not None and binding != self._binding:
            raise NativeContentPolicyDenied("content_policy_context_stale")
        return binding

    def check(self, path):
        if type(self) is not NativeContentContext or self not in _contexts:
            raise NativeContentPolicyDenied("content_policy_context_missing")
        if self._controlled_main_bound:
            self.invalidate()
            raise NativeContentPolicyDenied("controlled_native_operation_unsupported")
        if self._denied or getattr(self._local, "busy", False):
            self.invalidate()
            raise NativeContentPolicyDenied("content_policy_context_stale")
        self._local.busy = True
        try:
            policy = shared_policy()
            if type(path) is not str or path not in policy.PATHS:
                raise NativeContentPolicyDenied("content_policy_path_unsupported")
            binding = self.current(self.request_handle)
            self._binding = binding
            # Each check owns its runtime. No shared capability crosses threads.
            runtime = policy.ContentPolicyRuntime(
                admission=self, profiles=(), routes=(), transports=(),
                clock=lambda: int(time.monotonic()),
                policy_revision=binding.policy_revision,
            )
            runtime.admit(self.request_handle, b"")
            if self._denied:
                raise NativeContentPolicyDenied("content_policy_context_stale")
            return binding
        except Exception as exc:
            self.invalidate()
            if isinstance(exc, NativeContentPolicyDenied):
                raise
            raise NativeContentPolicyDenied("content_policy_binding_invalid") from None
        finally:
            self._local.busy = False

    def result(self, producer):
        self.check("tool_result")
        try:
            if not callable(self.result_profiles):
                raise NativeContentPolicyDenied("content_policy_result_unknown")
            require_ordinary(self.result_profiles(producer))
            self.check("tool_result")
        except Exception:
            self.invalidate()
            raise NativeContentPolicyDenied("content_policy_result_unqualified") from None


def current_context(agent=None):
    active = _active.get()
    namespace = vars(agent) if agent is not None and hasattr(agent, "__dict__") else {}
    attached = namespace.get("_native_content_policy_context")
    required = namespace.get("_native_content_policy_required") is True
    context = attached if attached is not None else active
    if context is None:
        if required:
            raise NativeContentPolicyDenied("content_policy_context_missing")
        return None
    if (type(context) is not NativeContentContext or context not in _contexts
            or (active is not None and attached is not None and active is not attached)):
        for candidate in (context, active):
            if type(candidate) is NativeContentContext and candidate._controlled_main_bound:
                candidate.invalidate()
        raise NativeContentPolicyDenied("content_policy_context_missing")
    return context


def guard_native_content(path, agent=None):
    context = current_context(agent)
    provider = getattr(agent, "provider", None) if agent is not None else None
    if type(provider) is str and provider == "synthetic_native" and (
            context is None or not context._controlled_main_bound):
        if context is not None:
            context.invalidate()
        raise NativeContentPolicyDenied("controlled_native_adapter_missing")
    if context is not None:
        context.check(path)
    return context


def guard_tool_result(agent, producer):
    context = current_context(agent)
    if context is not None:
        context.result(producer)


def checked_native_call(path, callback, agent=None):
    """Check current authority before dispatch and before result release."""
    context = guard_native_content(path, agent)
    result = callback()
    if context is not None:
        context.check(path)
    return result


@contextmanager
def content_policy_scope(context):
    if context is not None:
        if type(context) is not NativeContentContext or context not in _contexts:
            raise NativeContentPolicyDenied("content_policy_context_missing")
        validate_native_scope(context)
    token = _active.set(context)
    try:
        yield
        if context is not None:
            validate_native_scope(context)
    finally:
        _active.reset(token)


def validate_native_scope(context):
    """Validate scope facts without granting a generic controlled operation."""
    if type(context) is not NativeContentContext or context not in _contexts:
        raise NativeContentPolicyDenied("content_policy_context_missing")
    if context._controlled_main_bound:
        from agent.native_main_payload import validate_controlled_scope
        validate_controlled_scope(context)
    else:
        context.check("main")


def ingress_policy(ingress):
    """Use current trusted configuration, including when the adapter is absent."""
    from gateway.pwa_config import NativeContentPolicyConfig

    raw = getattr(getattr(ingress.runner, "config", None), "pwa_http", None)
    section = raw.get("content_policy") if isinstance(raw, dict) else None
    try:
        config = NativeContentPolicyConfig.from_dict(section)
    except ValueError:
        raise NativeContentPolicyDenied("content_policy_configuration_invalid") from None
    if config is None or not config.enabled:
        return None
    adapter = getattr(ingress, "content_policy", None)
    if adapter is None:
        raise NativeContentPolicyDenied("content_policy_adapter_missing")
    adapter.check_configuration()
    return adapter


def guard_inbound(runner, event, source):
    ingress = getattr(runner, "conversation_ingress", None)
    if ingress is None:
        raw = getattr(getattr(runner, "config", None), "pwa_http", None)
        if isinstance(raw, dict) and raw.get("content_policy") is not None:
            from gateway.pwa_config import NativeContentPolicyConfig
            config = NativeContentPolicyConfig.from_dict(raw["content_policy"])
            if config.enabled:
                raise NativeContentPolicyDenied("content_policy_adapter_missing")
        return
    adapter = ingress_policy(ingress)
    if adapter is not None:
        adapter.check_event(event, source)


def guard_completion(runner, source):
    """Restored data grants no authority, even without a policy flag."""
    ingress = getattr(runner, "conversation_ingress", None)
    if ingress is None:
        guard_inbound(runner, None, source)
        return
    adapter = ingress_policy(ingress)
    if adapter is not None:
        # This component does not restore durable policy for enabled sources.
        if source is None:
            raise NativeContentPolicyDenied("content_policy_restoration_required")
        adapter.source_owner(source)
        raise NativeContentPolicyDenied("content_policy_restoration_required")


async def guard_completion_event(runner, event):
    from gateway.pwa_config import NativeContentPolicyConfig

    raw = getattr(getattr(runner, "config", None), "pwa_http", None)
    section = raw.get("content_policy") if isinstance(raw, dict) else None
    try:
        config = NativeContentPolicyConfig.from_dict(section)
    except ValueError:
        raise NativeContentPolicyDenied("content_policy_configuration_invalid") from None
    if config is not None and config.enabled:
        source = await asyncio.to_thread(runner._build_process_event_source, event)
        guard_completion(runner, source)
