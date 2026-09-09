"""Opt-in native conversation ingress for a restricted authenticated facade.

This is an in-process GatewayRunner seam, not a second executor or a durable
command service. Provision grants on the trusted server; never build them from
request bodies. Transport authentication, receipts and replay are separate work.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Mapping
from uuid import uuid4

from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
from gateway.native_commands import DurableCommandIngressMixin


@dataclass(frozen=True)
class Principal:
    issuer: str
    subject: str


@dataclass(frozen=True)
class ConversationGrant:
    """Explicit owner assignment, including the native authorization identity."""
    session_id: str
    source: SessionSource


@dataclass(frozen=True)
class ConversationExecution:
    execution_id: str
    conversation_id: str
    generation: int
    source: SessionSource
    origin: str
    command: tuple | None = None


def _source_identity(source):
    return (source.platform, source.profile, source.chat_id, source.chat_type,
            source.user_id, source.thread_id, source.scope_id)


class NativeConversationIngress(DurableCommandIngressMixin):
    """One event-loop-local view over the runner's existing TurnState/FIFO.

    `send` awaits native handling for an idle conversation. A transport must
    detach that call from its connection lifetime; its return is NOT a durable
    receipt. Active sends return promptly after native steer/FIFO acceptance.
    Grants are fixed server configuration and may name any compression alias.
    Only one instance may be installed, before the runner starts admitting work.
    """

    def __init__(self, runner, db, grants: Mapping[Principal, tuple[ConversationGrant, ...]], *, concierge_id="default", grant_provider=None, event_limits=None):
        if getattr(runner.config, "multiplex_profiles", False) is True:
            raise ValueError("native conversation ingress requires a single profile scope")
        if getattr(runner, "conversation_ingress", None) is not None:
            raise RuntimeError("native conversation ingress already installed")
        if any(state.turn.agent is not None for state in runner._sessions_map().values()):
            raise RuntimeError("install native ingress before admitting turns")
        self.runner = runner
        self.db = db
        self._grant_provider = grant_provider
        self.grants = {principal: tuple(ConversationGrant(g.session_id, replace(g.source))
                                        for g in entries)
                       for principal, entries in grants.items()}
        self._init_commands(concierge_id)
        self.db.native_events_enable(event_limits)
        from gateway.native_events import NativeEventFeed
        self.events = NativeEventFeed(self)
        from gateway.native_cancellation import NativeCancellationController
        self.cancellations = NativeCancellationController(self)
        runner.conversation_ingress = self

    def trusted_channel_sources(self):
        """Configured native source identities, never transport-supplied grants."""
        provider = getattr(self, "_grant_provider", None)
        if provider is not None:
            return provider.trusted_channel_sources()
        return tuple(replace(grant.source) for entries in self.grants.values() for grant in entries)

    def _resolve(self, session_id):
        if self._grant_provider is not None:
            return self._grant_provider.resolve(session_id)
        lineage = self.db.get_compression_lineage(session_id)
        if not lineage:
            raise LookupError("conversation unavailable")
        return {"conversation_id": lineage[0], "native_root_session_id": lineage[0],
                "native_session_id": lineage[-1], "native_session_ids": lineage}

    def _authorize(self, principal, session_id):
        if self._grant_provider is not None:
            return self._grant_provider.authorize(principal, session_id)
        # Reject unknown principals before looking up arbitrary resource IDs.
        entries = self.grants.get(principal)
        if not entries:
            raise PermissionError("conversation unavailable")
        try:
            projection = self._resolve(session_id)
            for grant in entries:
                try:
                    granted = self._resolve(grant.session_id)
                except LookupError:
                    continue
                if granted["conversation_id"] == projection["conversation_id"]:
                    return projection, grant
        except LookupError:
            pass
        raise PermissionError("conversation unavailable")

    def _owner(self, root):
        for key, state in self.runner._sessions_map().items():
            execution = state.turn.conversation_execution
            if execution is not None and execution.conversation_id == root:
                return key, state, execution
        return None

    def _execution_view(self, owner):
        if owner is None:
            return None
        _, state, execution = owner
        agent = state.turn.agent
        return {"execution_id": execution.execution_id,
                "conversation_id": execution.conversation_id,
                "origin": execution.origin,
                "execution_state": ("running" if callable(getattr(agent, "steer", None))
                                    else "unknown" if agent is None else "starting")}

    async def inspect(self, principal, conversation_id):
        projection, _ = await asyncio.to_thread(self._authorize, principal, conversation_id)
        return {**projection, "active_execution": self._execution_view(self._owner(projection["conversation_id"]))}

    async def native_conversation(self, principal, conversation_id):
        """Authorized minimal native DTO; application metadata belongs to S02."""
        if self._grant_provider is None:
            raise RuntimeError("native PWA projection unavailable")
        return await self._grant_provider.native_conversation(self, principal, conversation_id)

    async def history(self, principal, conversation_id):
        projection, _ = await asyncio.to_thread(self._authorize, principal, conversation_id)
        # Retained native rows remain separated by segment. Do not concatenate
        # compressed summaries into a reconstructed model transcript.
        segments = []
        for session_id in projection["native_session_ids"]:
            messages = await asyncio.to_thread(self.db.get_messages, session_id, include_compacted=True)
            segments.append({"native_session_id": session_id, "messages": messages})
        return {**projection, "segments": segments}

    async def send(self, principal, conversation_id, text, *, mode="send"):
        if mode not in {"send", "queue"} or not isinstance(text, str) or not text.strip():
            raise ValueError("expected nonempty text and send or queue mode")
        projection, grant = await asyncio.to_thread(self._authorize, principal, conversation_id)
        source = replace(grant.source)
        # This identity is provisioned server-side. Ordinary gateway auth still
        # runs, and browser text cannot invoke slash/admin/approval commands.
        if not self.runner._is_user_authorized_for_source(source):
            raise PermissionError("native source is not authorized")
        source.native_conversation_route = projection["conversation_id"]
        entry = await self.runner.async_session_store.bind_conversation_alias(
            source, projection["native_session_id"],
        )
        key = entry.session_key
        event = MessageEvent(text=text, source=source, allow_gateway_control=False,
                             metadata={"gateway_session_strict": True,
                                       "gateway_session_key": key,
                                       "gateway_session_id": entry.session_id})
        # Wire-invisible marker; a caller-supplied metadata dict cannot opt in.
        event._native_browser_ingress = self
        event._native_send_mode = mode
        return await self.runner._handle_message(event)

    def _event_authorized(self, event, root):
        if self._grant_provider is not None:
            return self._grant_provider.event_authorized(event.source, root)
        identity = _source_identity(event.source)
        for entries in self.grants.values():
            for grant in entries:
                if _source_identity(grant.source) == identity:
                    try:
                        if self._resolve(grant.session_id)["conversation_id"] == root:
                            return True
                    except LookupError:
                        continue
        return False

    def _control(self, event, owner):
        key, state, execution = owner
        if not self._event_authorized(event, execution.conversation_id):
            raise PermissionError("conversation control unavailable")
        if getattr(event, "_native_command_key", None):
            self._enqueue_commands(owner)
            return {"disposition":"queued","execution":self._execution_view(owner)}
        mode = "queue" if event.internal else getattr(event, "_native_send_mode", "send")
        agent = state.turn.agent
        if mode != "queue" and callable(getattr(agent, "steer", None)):
            if agent.steer(event.text):
                return {"disposition": "steered", "execution": self._execution_view(owner)}
        adapter = self.runner._adapter_for_source(execution.source)
        if adapter is None or not isinstance(getattr(adapter, "_pending_messages", None), dict):
            raise RuntimeError("native queue unavailable")
        # The FIFO belongs to the actual owner route. Preserve input origin but
        # retain its native channel context/cache identity at the continuation.
        queued = replace(event, source=replace(execution.source))
        self.copy_event_context(event, queued)
        queued._native_origin = self._origin(event)
        self.runner._enqueue_fifo(key, queued, adapter)
        return {"disposition": "queued", "execution": self._execution_view(owner)}

    def copy_event_context(self, event, replacement):
        """Carry native-only trust/origin through trusted event rewrites."""
        if getattr(event, "_native_browser_ingress", None) is self:
            replacement._native_browser_ingress = self
            replacement._native_send_mode = event._native_send_mode
        for name in ("_native_command_key", "_native_execution_id"):
            if hasattr(event, name):
                setattr(replacement, name, getattr(event, name))
        if hasattr(event, "_native_origin"):
            replacement._native_origin = event._native_origin

    def _origin(self, event):
        if getattr(event, "_native_browser_ingress", None) is self:
            return "pwa"
        return getattr(event, "_native_origin", event.source.platform.value)

    async def control_existing(self, event, key):
        if event.internal or event.is_command() or event.media_urls or event.media_types:
            return None
        entry = await self.runner.async_session_store.lookup_by_session_key(key)
        if entry is None:
            return None
        try:
            projection = await asyncio.to_thread(self._resolve, entry.session_id)
        except LookupError:
            return None  # Fresh rows are created and verified at admission.
        owner = self._owner(projection["conversation_id"])
        if owner is not None:
            return self._control(event, owner)
        return None

    async def prepare_admission(self, event, source, key):
        metadata = event.metadata or {}
        if metadata.get("gateway_session_strict"):
            entry = await self.runner.async_session_store.lookup_by_session_key(key)
            if entry is None or entry.session_id != metadata.get("gateway_session_id"):
                raise LookupError("native conversation binding changed")
        else:
            entry = await self.runner.async_session_store.get_or_create_session(
                source, touch_activity=not event.internal,
            )
        projection = await asyncio.to_thread(self._ensure_projection, entry, source, key)
        if projection is None or not self._event_authorized(event, projection["conversation_id"]):
            return None, None
        owner = self._owner(projection["conversation_id"])
        if owner is not None:
            return None, self._control(event, owner)
        return projection["conversation_id"], None

    def reserve(self, root, source, key, generation, event):
        # Called in the same synchronous block as the runner's route claim,
        # immediately after prepare_admission returns: there is no await gap.
        if root is not None:
            execution_id = getattr(event, "_native_execution_id", None) or str(uuid4())
            command = getattr(event, "_native_command_key", None)
            try:
                self.db.native_execution_open(root, execution_id, self._command_owner, command=command, origin=self._origin(event))
            except BaseException:
                self.runner._release_running_agent_state(key)
                raise
            self.runner._session_state(key).turn.conversation_execution = ConversationExecution(
                execution_id, root, generation, replace(source), self._origin(event), command)

    def _ensure_projection(self, entry, source, key):
        row = self.db.get_session(entry.session_id)
        if row is None:
            provider_source = self._grant_provider is not None and any(
                _source_identity(candidate) == _source_identity(source)
                for candidate in self._grant_provider.trusted_channel_sources())
            if not provider_source and not any(grant.session_id == entry.session_id
                       and _source_identity(grant.source) == _source_identity(source)
                       for entries in self.grants.values() for grant in entries):
                return None
            self.db.record_gateway_session_peer(
                entry.session_id, source=source.platform.value,
                user_id=source.user_id, session_key=key,
                chat_id=source.chat_id, chat_type=source.chat_type,
                thread_id=source.thread_id)
        return self._resolve(entry.session_id)

    async def admit(self, event, source, entry, key, generation):
        # SessionStore historically tolerates failed SQLite creation. This
        # opt-in ingress cannot: the durable agent lease skips missing rows.
        projection = await asyncio.to_thread(self._ensure_projection, entry, source, key)
        if projection is None:
            return None
        state = self.runner._peek_session_state(key)
        reserved = state.turn.conversation_execution if state else None
        if (reserved is not None and reserved.generation == generation
                and reserved.conversation_id != projection["conversation_id"]):
            raise RuntimeError("native conversation changed during admission")
        if not self._event_authorized(event, projection["conversation_id"]):
            if reserved is not None and reserved.generation == generation:
                raise PermissionError("native source changed during admission")
            # Unprovisioned native conversations retain their existing behavior.
            # A colliding foreign source must still be denied below.
            owner = self._owner(projection["conversation_id"])
            if owner is not None:
                raise PermissionError("conversation control unavailable")
            return None
        # No await between owner lookup and publishing in TurnState. Racing
        # aliases therefore control the winner before either loads history.
        owner = self._owner(projection["conversation_id"])
        if owner is not None:
            if owner[0] == key and owner[2].generation == generation:
                return None
            return self._control(event, owner)
        self.reserve(projection["conversation_id"], source, key, generation, event)
        return None

    def release(self, key, generation):
        state = self.runner._peek_session_state(key)
        execution = state.turn.conversation_execution if state else None
        if execution is not None and execution.generation == generation:
            durable = self.db.native_execution(execution.conversation_id)
            if durable and durable["execution_id"] == execution.execution_id and (not durable["input_started"] or not self.db.native_execution_has_lease(execution.conversation_id)):
                with self._completion_lock:
                    known = execution.execution_id in self._completion_observations
                if known:
                    self._retry_completion(execution.execution_id)
                else:
                    self.db.native_execution_close(execution.execution_id, self._command_owner, crashed=True)
            state.turn.conversation_execution = None
            self._wake_commands(execution.conversation_id)

    def continue_execution(self, key, generation, event):
        state = self.runner._peek_session_state(key)
        execution = state.turn.conversation_execution if state else None
        if execution is not None and execution.generation == generation:
            if getattr(event, "_native_command_key", None) and execution.execution_id == getattr(event, "_native_execution_id", None):
                return  # Already claimed at the dequeue boundary, before awaits.
            with self._completion_lock:
                known = execution.execution_id in self._completion_observations
            if known:
                if not self._retry_completion(execution.execution_id):
                    raise RuntimeError("native completion persistence is pending")
            else:
                self.db.native_execution_close(execution.execution_id, self._command_owner)
            execution_id = getattr(event, "_native_execution_id", None) or str(uuid4())
            command = getattr(event, "_native_command_key", None)
            self.db.native_execution_open(execution.conversation_id, execution_id, self._command_owner, command=command, origin=self._origin(event) if event is not None else execution.origin)
            state.turn.conversation_execution = replace(
                execution, execution_id=execution_id, command=command,
                origin=self._origin(event) if event is not None else execution.origin)
