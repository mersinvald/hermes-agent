"""Explicit owner-scoped native interrupt and known remote-task reconciliation.

This controller observes existing workers; it never starts a replacement agent,
invalidates a generation, releases a lease, deletes queued input or replays a
remote cancellation write after its durable attempt reservation.
"""

import asyncio
import hashlib
import hmac
import logging
import re
import secrets
import threading

from agent.interrupt_compat import request_hard_interrupt
from agent.native_execution_context import NativeExecutionOrigin
from gateway.native_events import execution_view
from hermes_state_commands import canonical, CommandConflict
from hermes_state_events import timestamp
from plugins.platforms.a2a.cancellation import (
    ControlObservation,
    ControlRequestRejected,
    RecordedTask,
    TaskControllerClient,
)

logger = logging.getLogger(__name__)


class NativeCancellationController:
    def __init__(
        self,
        ingress,
        *,
        client=None,
        max_control_jobs=4,
        poll_interval=5,
        control_turn_seconds=30,
    ):
        if type(max_control_jobs) is not int or not 1 <= max_control_jobs <= 64:
            raise ValueError("control concurrency must be between 1 and 64")
        if type(poll_interval) is not int or not 1 <= poll_interval <= 300:
            raise ValueError("control polling must be between 1 and 300 seconds")
        self.max_control_jobs = max_control_jobs
        if (
            type(control_turn_seconds) not in (int, float)
            or not 0 < control_turn_seconds <= 90
        ):
            raise ValueError(
                "control turn deadline must be positive and at most 90 seconds"
            )
        self.control_turn_seconds = control_turn_seconds
        self.poll_interval = poll_interval
        self._scan_after = 0
        self._reconciling = False
        self.ingress = ingress
        self.db = ingress.db
        self.client = client or TaskControllerClient(max_connections=max_control_jobs)
        self._jobs = {}
        self._timer = None
        self._cursor_key = secrets.token_bytes(32)
        self._loop = None
        self._stopped = False
        self._dirty = set()
        self._workers = {}
        self._workers_lock = threading.Lock()

    def bind_worker(self, execution, agent):
        with self._workers_lock:
            self._workers[execution.execution_id] = (execution, agent)

    def unbind_worker(self, execution, agent):
        with self._workers_lock:
            if self._workers.get(execution.execution_id) == (execution, agent):
                self._workers.pop(execution.execution_id, None)

    def stop_recovery(self):
        self._stopped = True
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if not self._jobs:
            asyncio.get_running_loop().create_task(self.client.close())

    def start_recovery(self):
        self._stopped = False
        self._loop = asyncio.get_running_loop()
        if self._timer is None:
            self._timer = self._loop.call_later(0, self._tick)

    def wake(self, row):
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        try:
            row = self.db.native_cancel_lookup(row["scope"], row["command_id"])
            try:
                self._request_native(row)
            except Exception:
                logger.warning(
                    "Native interrupt request pending; remote work remains durable"
                )
            row = self.db.native_cancel_lookup(row["scope"], row["command_id"])
            execution = row["execution_id"]
            if execution in self._jobs:
                self._dirty.add(execution)
            else:
                self.db.native_cancel_defer(
                    row["scope"], row["command_id"], 0, row["remote_after"]
                )
        except Exception:
            logger.warning(
                "Native cancellation wake storage unavailable; admission remains durable"
            )
        finally:
            # All wakes enter the admission-order pump, including idle gaps.
            self._pump_soon()

    def _pump_soon(self):
        if self._stopped:
            return
        if self._timer is not None:
            self._timer.cancel()
        self._timer = asyncio.get_running_loop().call_later(0, self._tick)

    def _launch(self, row):
        execution = row["execution_id"]
        if self._stopped or execution in self._jobs:
            return
        try:
            self._request_native(row)
        except Exception:
            logger.warning(
                "Native interrupt request pending; remote work remains durable"
            )
        task = asyncio.create_task(self._process(row))
        self._jobs[execution] = task
        task.add_done_callback(lambda done: self._finished(execution, done))

    def _finished(self, execution, task):
        self._jobs.pop(execution, None)
        if execution in self._dirty:
            self._dirty.discard(execution)
            if not self._stopped:
                try:
                    row = self.db.native_cancel_for_execution(execution)
                    if row:
                        self.db.native_cancel_defer(
                            row["scope"], row["command_id"], 0, row["remote_after"]
                        )
                except Exception:
                    logger.warning("Native cancellation follow-up storage unavailable")
        if self._stopped and not self._jobs:
            asyncio.get_running_loop().create_task(self.client.close())
        if not task.cancelled() and task.exception() is not None:
            logger.warning("Native cancellation observation pending reconciliation")
        self._pump_soon()

    def _arm(self):
        if (
            not self._stopped
            and self._timer is None
            and not getattr(self.ingress.runner, "_draining", False)
        ):
            self._timer = asyncio.get_running_loop().call_later(
                self.poll_interval, self._tick
            )

    def _tick(self):
        self._timer = None
        if not getattr(self.ingress.runner, "_draining", False):
            asyncio.create_task(self.reconcile())

    async def reconcile(self):
        if self._reconciling or self._stopped:
            return
        self._reconciling = True
        try:
            # One bounded page per pump, cycling the persisted admission order.
            # Completed jobs advance fair scanning rather than allowing the
            # first unresolved execution to monopolize a slot every poll.
            available = self.max_control_jobs - len(self._jobs)
            if available <= 0:
                return
            rows = await asyncio.to_thread(
                self.db.native_cancel_recovery_rows,
                after=self._scan_after,
                limit=available,
            )
            if not rows and self._scan_after:
                self._scan_after = 0
                rows = await asyncio.to_thread(
                    self.db.native_cancel_recovery_rows, after=0, limit=available
                )
            for row in rows:
                self._scan_after = row["ordinal"]
                self._launch(row)
        except Exception:
            logger.warning("Native cancellation recovery storage unavailable")
        finally:
            self._reconciling = False
            self._arm()

    def observed_dispatch(self, origin):
        """Actual late task identity wakes accepted control without a browser."""
        row = self.db.native_cancel_for_execution(origin.execution_id)
        if row and self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self.wake, row)

    def _request_native(self, row):
        origin = NativeExecutionOrigin(
            row["conversation_id"], row["execution_id"], row["owner"]
        )
        from tools.async_delegation import interrupt_for_native_execution

        clarifications = getattr(self.ingress, "clarifications", None)
        if clarifications is not None:
            clarifications.cancel_execution(origin)
        interrupt_for_native_execution(origin.record())
        if row["native_request_state"] != "pending":
            return
        owner = self.ingress._owner(row["conversation_id"])
        if (
            owner
            and owner[2].execution_id == row["execution_id"]
            and row["owner"] == self.ingress._command_owner
        ):
            with self._workers_lock:
                worker = self._workers.get(row["execution_id"])
            agent = worker[1] if worker and worker[0] == owner[2] else None
            execution = self.db.native_cancel_snapshot(
                row["scope"], row["command_id"], limit=1
            )["execution"]
            if agent is None or not execution or not execution["input_started"]:
                # run_conversation resets its interrupt flag in the prologue;
                # only signal once actual native input startup has happened.
                return
            with self.ingress._completion_lock:
                finished = row["execution_id"] in self.ingress._completion_observations
            if finished:
                self.db.native_cancel_note_native(
                    row["scope"], row["command_id"], "not_running"
                )
                return
            # Persist ambiguity first. A crash between these operations must not
            # infer that the worker was interrupted or automatically signal a
            # replacement after restart.
            self.db.native_cancel_note_native(
                row["scope"], row["command_id"], "unknown"
            )
            signaled = request_hard_interrupt(
                agent, "Explicit execution cancellation", tool_reason="user_cancel"
            )
            self.db.native_cancel_note_native(
                row["scope"], row["command_id"], "requested" if signaled else "failed"
            )
        else:
            execution = self.db.native_cancel_snapshot(
                row["scope"], row["command_id"], limit=1
            )["execution"]
            state = (
                "unknown"
                if execution and execution["state"] == "open"
                else "not_running"
            )
            self.db.native_cancel_note_native(row["scope"], row["command_id"], state)

    @staticmethod
    def _target(row):
        keys = (
            "dispatch_id",
            "peer_name",
            "configured_endpoint_fingerprint",
            "rpc_endpoint",
            "protocol_version",
            "tenant",
            "task_id",
            "context_id",
            "configured_tenant",
        )
        return RecordedTask(**{key: row[key] for key in keys})

    async def _remote(self, command, row):
        if not row["task_id"] or not row["binding_key"]:
            return  # Lost-ID or unconfigured peer stays explicit unknown.
        target = self._target(row)
        if row["cancel_request_state"] is not None:
            if not row["write_reserved"] and row["cancel_request_state"] in {
                "failed",
                "rejected",
                "not_needed",
            }:
                return
            observation = await self.client.observe(target, cancellation_requested=True)
            self.db.native_remote_cancel_note(
                row["dispatch_id"], observation, update_request=False
            )
            return

        def reserve():
            try:
                self.db.native_remote_cancel_reserve(
                    command["scope"], command["command_id"], row["dispatch_id"]
                )
            except CommandConflict:
                raise ControlRequestRejected("cancel_target_unavailable") from None

        observation = await self.client.cancel_once(target, before_send=reserve)
        if observation.write_attempted:
            self.db.native_remote_cancel_note(row["dispatch_id"], observation)
            # A response/timeout is followed by a read, never another write.
            observed = await self.client.observe(target, cancellation_requested=True)
            self.db.native_remote_cancel_note(
                row["dispatch_id"], observed, update_request=False
            )
        else:
            self.db.native_remote_cancel_no_write(
                command["scope"], command["command_id"], row["dispatch_id"], observation
            )

    async def _process(self, row):
        after = row["remote_after"]
        delay = self.poll_interval
        try:
            targets = await asyncio.to_thread(
                self.db.native_remote_targets,
                row["execution_id"],
                after=after,
                limit=2,
                scope=row["scope"],
                command_id=row["command_id"],
            )
            if targets:
                target = targets[0]
                try:
                    async with asyncio.timeout(self.control_turn_seconds):
                        await self._remote(row, target)
                except TimeoutError:
                    # No write reservation means this was a known-not-sent
                    # preflight timeout. A fresh explicit command may try again.
                    current = self.db.native_remote_targets(
                        row["execution_id"],
                        after=after,
                        limit=1,
                        scope=row["scope"],
                        command_id=row["command_id"],
                    )[0]
                    if not current["write_reserved"]:
                        self.db.native_remote_cancel_no_write(
                            row["scope"],
                            row["command_id"],
                            target["dispatch_id"],
                            ControlObservation("failed", "unknown", "cancel_not_sent"),
                        )
                after = target["ordinal"]
                if len(targets) > 1:
                    delay = 0
                else:
                    after = 0
            else:
                after = 0
        finally:
            await asyncio.to_thread(
                self.db.native_cancel_defer,
                row["scope"],
                row["command_id"],
                delay,
                after,
            )

    def _cursor(self, row, ordinal):
        context = canonical([
            row["scope"],
            row["command_id"],
            row["conversation_id"],
            row["execution_id"],
            ordinal,
        ])
        signature = hmac.new(
            self._cursor_key, context.encode(), hashlib.sha256
        ).hexdigest()
        return f"{ordinal:x}.{signature}"

    async def receipt(
        self, principal, command_id, *, remote_cursor=None, remote_limit=50
    ):
        if type(remote_limit) is not int or not 1 <= remote_limit <= 100:
            raise ValueError("invalid remote page limit")
        scope = self.ingress._command_scope(principal)
        row = await asyncio.to_thread(self.db.native_cancel_lookup, scope, command_id)
        if not row:
            raise LookupError("command unavailable; absence does not authorize resend")
        if self.db.native_control_lookup(scope, command_id) is not None:
            return await self.ingress.controls.receipt(principal, command_id,
                remote_cursor=remote_cursor, remote_limit=remote_limit)
        self.ingress._authorize(principal, row["conversation_id"])
        after = 0
        if remote_cursor is not None:
            if not isinstance(remote_cursor, str) or not re.fullmatch(
                r"[0-9a-f]{1,16}\.[0-9a-f]{64}", remote_cursor
            ):
                raise ValueError("invalid remote cursor")
            after = int(remote_cursor.split(".")[0], 16)
            if not hmac.compare_digest(self._cursor(row, after), remote_cursor):
                raise ValueError("invalid remote cursor")
        view = await asyncio.to_thread(
            self.db.native_cancel_snapshot,
            scope,
            command_id,
            after=after,
            limit=remote_limit,
        )
        _, grant = self.ingress._authorize(principal, row["conversation_id"])
        if not self.ingress.runner._is_user_authorized_for_source(grant.source):
            raise PermissionError("native source is not authorized")
        return self.receipt_view(view)

    def receipt_view(self, view):
        row = view["command"]
        command_id = row["command_id"]
        targets = []
        for target in view["targets"]:
            targets.append(
                dict(
                    dispatch_id=target["dispatch_id"],
                    request_origin=(
                        "this_command"
                        if (target["cancel_scope"], target["cancel_command_id"])
                        == (row["scope"], row["command_id"])
                        else "prior_command"
                        if target["cancel_request_state"] is not None
                        else "not_attempted"
                    ),
                    request_state=target["cancel_request_state"]
                    or (
                        "pending"
                        if target["task_id"] and target["binding_key"]
                        else "unknown"
                    ),
                    observed_task_state=target["observed_state"],
                    task_identity="known" if target["task_id"] else "unknown",
                    recorded_at=timestamp(target["updated_at"]),
                )
            )
        execution = view["execution"]
        native_state = execution_view(execution)["state"] if execution else "unknown"
        with self.ingress._completion_lock:
            completion_pending = (
                row["execution_id"] in self.ingress._completion_observations
            )
        if completion_pending or (
            execution
            and execution["state"] == "open"
            and execution["owner"] != self.ingress._command_owner
        ):
            native_state = "unknown"
        return dict(
            schema_version="1.0",
            receipt_kind="cancel",
            command_id=command_id,
            conversation_id=row["conversation_id"],
            target_execution_id=row["execution_id"],
            payload_fingerprint=row["fingerprint"],
            receipt_state="accepted",
            durability="durable",
            queue_scope="execution_only",
            queued_and_unconsumed_steers="retained",
            native=dict(
                request_state=view["command"]["native_request_state"],
                observed_state=native_state,
            ),
            remote_coverage=dict(
                known_count=view["known_count"],
                task_id_unknown_count=view["task_id_unknown_count"],
                descendants="unknown",
                has_more=view["has_more"],
            ),
            remote_targets=targets,
            next_remote_cursor=self._cursor(
                row, view["targets"][-1]["ordinal"] if view["targets"] else 0
            )
            if view["has_more"]
            else None,
            recorded_at=timestamp(row["recorded_at"]),
        )
