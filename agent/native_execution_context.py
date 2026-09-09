"""Actual native execution provenance, copied through native child/tool threads.

This scope is installed by TurnRunner, never inferred from a session route,
model argument, browser input, or global A2A transcript.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class NativeExecutionOrigin:
    conversation_id: str
    execution_id: str
    owner: str

    def record(self):
        return dict(
            conversation_id=self.conversation_id,
            execution_id=self.execution_id,
            owner=self.owner,
        )


_current = ContextVar("native_execution_origin", default=None)


@contextmanager
def native_execution_scope(origin, controller=None):
    token = _current.set((origin, controller))
    try:
        yield
    finally:
        _current.reset(token)


def current_native_execution():
    return _current.get()


def native_origin_record():
    context = current_native_execution()
    return context[0].record() if context else None


def native_execution_cancel_requested():
    context = current_native_execution()
    if not context or context[1] is None:
        return False
    origin, controller = context
    return controller.db.native_execution_cancel_requested(origin.execution_id)
