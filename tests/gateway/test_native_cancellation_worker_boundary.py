"""Execution binding races use the real controller and AIAgent interrupt methods.

The journal is a small recording stub here; actual durable/HTTP composition is
covered by the adjacent gateway tests and Q01's synthetic external peers.
"""

import threading
from types import SimpleNamespace

import pytest

from gateway.native_cancellation import NativeCancellationController
from run_agent import AIAgent
from tools.interrupt import get_interrupt_reason, is_interrupted, set_interrupt


@pytest.fixture
def bound_worker():
    agent = AIAgent.__new__(AIAgent)
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._tool_interrupt_reason = None
    agent._hard_interrupt_requested = threading.Event()
    agent._pending_redirect_lock = threading.Lock()
    agent._pending_redirect = None
    agent._pending_steer_lock = threading.Lock()
    agent._pending_steer = None
    agent._execution_thread_id = threading.get_ident()
    agent._interrupt_thread_signal_pending = False
    agent._active_children_lock = threading.Lock()
    agent._active_children = []
    agent._tool_worker_threads = set()
    agent._tool_worker_threads_lock = threading.Lock()
    agent.quiet_mode = True
    agent.api_mode = "test"
    execution = SimpleNamespace(execution_id="old", generation=1)
    row = dict(conversation_id="root", execution_id="old", owner="owner",
               native_request_state="pending", scope="scope", command_id="cancel")
    notes = []

    class Journal:
        input_started = True
        before_note = None

        def native_cancel_snapshot(self, *args, **kwargs):
            return {"execution": {"input_started": self.input_started}}

        def native_cancel_note_native(self, scope, command_id, state):
            notes.append(state)
            if self.before_note is not None:
                self.before_note(state)

    ingress = SimpleNamespace(
        db=Journal(), _owner=lambda root: ("key", None, execution),
        _command_owner="owner", _completion_lock=threading.Lock(),
        _completion_observations={},
    )
    controller = NativeCancellationController(ingress, client=object())
    controller.bind_worker(execution, agent)
    try:
        yield controller, ingress, execution, agent, row, notes
    finally:
        set_interrupt(False, agent._execution_thread_id)


def test_unbind_retires_own_signal_and_preserves_late_steer(bound_worker):
    controller, _, execution, agent, row, notes = bound_worker
    controller._request_native(row)
    assert agent._interrupt_requested and agent._hard_interrupt_requested.is_set()
    assert is_interrupted() and get_interrupt_reason() == "user_cancel"
    assert agent.steer("late authored steer")
    # A different worker/generation cannot clean up this execution's signal.
    controller.unbind_worker(SimpleNamespace(execution_id="old", generation=2), agent)
    assert agent._interrupt_requested
    controller.unbind_worker(execution, agent)
    assert not agent._interrupt_requested and agent._interrupt_message is None
    assert not agent._hard_interrupt_requested.is_set() and not is_interrupted()
    assert agent._pending_steer == "late authored steer"
    assert notes == ["unknown", "requested"]


def test_unbind_preserves_an_overwriting_user_interrupt(bound_worker):
    controller, _, execution, agent, row, _ = bound_worker
    controller._request_native(row)
    agent.interrupt("Please continue with this authored input")
    controller.unbind_worker(execution, agent)
    assert agent._interrupt_requested
    assert agent._interrupt_message == "Please continue with this authored input"
    assert is_interrupted() and get_interrupt_reason() == "user sent a new message"


def test_unbind_preserves_accepted_active_turn_redirect(bound_worker):
    controller, _, execution, agent, row, _ = bound_worker
    controller._request_native(row)
    agent.clear_interrupt()  # The actual finalizer's existing boundary.
    agent._model_request_active = threading.Event()
    agent._model_request_active.set()
    assert agent.redirect("Accepted authored correction")
    controller.unbind_worker(execution, agent)
    assert agent._interrupt_requested and agent._interrupt_message is None
    assert agent._pending_redirect == "Accepted authored correction"
    assert is_interrupted()


def test_pending_cancel_still_waits_for_input_and_signals_current_worker(bound_worker):
    controller, ingress, execution, agent, row, notes = bound_worker
    ingress.db.input_started = False
    controller._request_native(row)
    assert not agent._interrupt_requested and notes == []
    ingress.db.input_started = True
    controller._request_native(row)
    assert agent._interrupt_requested
    # A bind does not indiscriminately reset legitimate startup cancellation.
    controller.bind_worker(execution, agent)
    assert agent._interrupt_requested and agent._hard_interrupt_requested.is_set()


def test_unbind_keeps_outstanding_tool_signal_until_that_worker_cleans_up(bound_worker):
    controller, _, execution, agent, row, _ = bound_worker
    started, inspect = threading.Event(), threading.Event()
    observations = []

    def tool():
        tid = threading.get_ident()
        with agent._tool_worker_threads_lock:
            agent._tool_worker_threads.add(tid)
        started.set()
        try:
            if inspect.wait(5):
                observations.append(is_interrupted())
        finally:
            with agent._tool_worker_threads_lock:
                agent._tool_worker_threads.discard(tid)
            set_interrupt(False, tid)

    worker = threading.Thread(target=tool)
    worker.start()
    try:
        assert started.wait(5)
        controller._request_native(row)
        controller.unbind_worker(execution, agent)
        assert not agent._interrupt_requested and not is_interrupted()
    finally:
        inspect.set()
        worker.join(5)
    assert not worker.is_alive() and observations == [True]
    assert not agent._tool_worker_threads


def test_lookup_that_outlives_unbind_cannot_interrupt_reused_agent(bound_worker):
    controller, ingress, execution, agent, row, notes = bound_worker

    def returned(state):
        if state == "unknown":
            controller.unbind_worker(execution, agent)
            controller.bind_worker(SimpleNamespace(execution_id="new", generation=2), agent)
            agent.interrupt("New execution's authored input")

    ingress.db.before_note = returned
    controller._request_native(row)
    assert agent._interrupt_message == "New execution's authored input"
    assert not agent._hard_interrupt_requested.is_set()
    assert notes == ["unknown", "not_running"]


def test_unbind_waits_for_inflight_signal_then_retires_it(bound_worker):
    controller, _, execution, agent, row, notes = bound_worker
    signaling, unbind_attempted = threading.Event(), threading.Event()
    failures = []
    lock = controller._workers_lock

    class ObservedLock:
        def __enter__(self):
            if threading.current_thread().name == "q01-owned-unbind":
                unbind_attempted.set()
            return lock.__enter__()

        def __exit__(self, *args):
            return lock.__exit__(*args)

    controller._workers_lock = ObservedLock()
    actual_interrupt = agent.hard_interrupt

    def interrupt(*args, **kwargs):
        assert lock.locked(), "signal must finish before unbind can retire the binding"
        signaling.set()
        assert unbind_attempted.wait(5), "owned unbind did not reach the binding lock"
        actual_interrupt(*args, **kwargs)

    agent.hard_interrupt = interrupt

    def invoke(function):
        try:
            function()
        except BaseException as exc:
            failures.append(exc)

    signal = threading.Thread(target=invoke, args=(lambda: controller._request_native(row),))
    unbind = threading.Thread(name="q01-owned-unbind", target=invoke,
                              args=(lambda: controller.unbind_worker(execution, agent),))
    signal.start()
    try:
        assert signaling.wait(5), "owned signal did not begin"
        unbind.start()
        assert unbind_attempted.wait(5), "owned unbind did not attempt its lock"
    finally:
        unbind_attempted.set()
        signal.join(5)
        if unbind.ident is not None:
            unbind.join(5)
    assert not signal.is_alive() and not unbind.is_alive() and not failures
    assert not agent._interrupt_requested and not agent._hard_interrupt_requested.is_set()
    assert not is_interrupted() and notes == ["unknown", "requested"]
