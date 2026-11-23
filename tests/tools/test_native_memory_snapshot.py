"""Read invented frozen memory through the original authenticated operator."""

import hashlib
import asyncio
import json
import sys
import threading
from contextlib import contextmanager
from contextvars import copy_context
from dataclasses import replace

import pytest
from hermes_content_policy import PATHS

from agent.content_policy import NativeContentPolicyDenied, content_policy_scope, guard_tool_result
from agent.native_main_payload import (
    NativeMemorySource, NativeMemorySnapshotInput, NativeResultPolicy, _memory_snapshots,
)
from tests.agent.test_native_main_payload import (
    World, canonical,
    invented_offline_configuration as invented_offline_configuration,
)
from tests.agent.test_native_controlled_conversation import excluded as excluded
from tests.tools.test_native_controlled_delegation import retain_case_durations as retain_case_durations
from tests.tools.test_native_parent_resumption import (
    retain_incremental_case_receipts as retain_incremental_case_receipts,
    record_case_start as record_case_start,
)
from tools.memory_tool import MemoryStore, get_memory_dir


class MemoryWorld(World):
    def __init__(self, *, empty=False, entry="Blue squares use two counts."):
        super().__init__()
        directory = get_memory_dir()
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "MEMORY.md").write_text("" if empty else entry + "\n", encoding="utf-8")
        (directory / "USER.md").write_text("Unrelated red triangles stay separate.\n", encoding="utf-8")
        self.store = MemoryStore()
        self.store.load_from_disk()
        self.memory_policy = NativeResultPolicy("invented_memory_producer", 6, ("restricted_documents",))
        self.source = NativeMemorySource(self.store, "memory", self.binding.owner_ref,
                                         "invented_host_profile", 1, self.memory_policy)
        self.result_port.memory_source = self.memory_source
        self.lookups = []
        self.on_lookup = self.on_clock = None

    @property
    def now(self):
        if getattr(self, "on_clock", None) is not None:
            self.on_clock()
        return self._now

    @now.setter
    def now(self, value):
        self._now = value

    def produced(self, ref):
        if ref == "invented_memory_producer":
            return self.memory_policy
        return super().produced(ref)

    def memory_source(self, handle, ref):
        assert handle is self.request and ref is self.memory_policy.producer_ref
        self.lookups.append((handle, ref))
        if self.on_lookup is not None:
            self.on_lookup()
        return self.source

    def construct(self):
        from run_agent import AIAgent
        self.bind()
        with content_policy_scope(self.context):
            self.agent = AIAgent(controlled_main=self.adapter, provider="synthetic_native",
                api_mode="chat_completions", model="invented_model", session_id="invented_memory_session",
                enabled_toolsets=[], quiet_mode=True, skip_memory=True, skip_context_files=True,
                skip_background_review=True)
        return self

    def capture(self):
        with content_policy_scope(self.context):
            self.token = self.adapter.bind_memory_snapshot(
                self.agent, store=self.store, producer_ref=self.memory_policy.producer_ref)
            return self.adapter.read_memory_snapshot(self.agent, self.token)


@contextmanager
def getter_observations(store):
    observations = []
    original = sys.gettrace()
    code = MemoryStore.format_for_system_prompt.__code__

    def observed(frame, event, value):
        if frame.f_code is code:
            assert frame.f_locals["self"] is store
            observations.append((event, frame, value))
            return observed
        return None

    sys.settrace(observed)
    try:
        yield observations
    finally:
        sys.settrace(original)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def test_actual_nonempty_snapshot(excluded, record_property):
    world = MemoryWorld().construct()
    runtime = world.adapter._runtime
    assert runtime._retained_bytes == 0 and not runtime._contents
    frozen = world.store._system_prompt_snapshot
    expected = frozen["memory"]
    with getter_observations(world.store) as observations:
        result = world.capture()
    owner = _memory_snapshots[world.adapter]
    binding = result.binding
    assert type(world.token) is NativeMemorySnapshotInput and world.token is binding.token
    assert result is owner.record and binding is owner.binding
    assert binding.agent is world.agent and binding.request_handle is world.request
    assert binding.store is world.store and binding.target is world.source.target
    assert binding.producer_policy is world.source.producer_policy is world.memory_policy
    assert binding.selection is world.selection
    assert binding.snapshots[0] is owner.source is world.source
    assert binding.request_bytes == b'{"target":"memory"}'
    assert digest(binding.request_bytes) == "c1115f389258bbd38c28b6d0754ebeaaa4e63fbf5c933e8c49dd68bf285d0b0c"
    assert result.block is expected is owner.getter_block
    assert len(expected.encode()) == 359
    assert digest(expected.encode()) == "ccf4ac249b0c29904c33330172bd9f4410bfeccfd3f0590d6bb0cef7abb36e67"
    assert result.result_bytes == canonical(dict(target="memory", block=expected))
    assert len(result.result_bytes) == 671
    assert digest(result.result_bytes) == "66decc8c2d60a7c82e086aa93eb2cc0f69ae41049d62fd63421528b97219c10a"
    assert result.parents == (binding.request_content, binding.source_content)
    for handle, payload, parents in (
        (binding.source_content, binding.source_bytes, ()),
        (binding.request_content, binding.request_bytes, ()),
        (result.result_content, result.result_bytes, result.parents),
    ):
        content = runtime._content(handle)
        assert content.payload is payload and content.parents == parents
        assert content.profiles == ("restricted_documents",)
    assert runtime._content(result.result_content).path == "memory"
    source = json.loads(binding.source_bytes)
    assert source["block"] == expected and source["source"]["profile_ref"] == "invented_host_profile"
    assert "Unrelated red triangles" not in binding.source_bytes.decode()
    assert len(runtime._contents) == 3 and not runtime._envelopes
    assert runtime._retained_bytes == sum(map(len, (binding.source_bytes, binding.request_bytes, result.result_bytes)))
    calls = [row for row in observations if row[0] == "call"]
    returns = [row for row in observations if row[0] == "return"]
    assert len(calls) == len(returns) == 1
    assert calls[0][1] is returns[0][1] and returns[0][2] is expected
    assert calls[0][1].f_locals["target"] is binding.target
    assert world.store._system_prompt_snapshot is frozen
    assert world.agent._memory_store is world.agent._memory_manager is None
    assert world.lookups and all(handle is world.request for handle, _ in world.lookups)
    assert owner.phase == "released" and owner.active_call is None
    assert world.adapter.send_count == world.adapter.auxiliary_send_count == world.adapter.fallback_send_count == 0
    assert not world.adapter.invocations and not world.adapter.auxiliary_invocations and not world.sdk_calls
    assert all(not port.sent for port in world.ports.values())
    assert tuple(row[0] for row in world.qualification_records) == PATHS
    assert len(world.qualification_records) == 8
    record_property("memory_proof", json.dumps(dict(
        source_bytes=binding.source_bytes.decode(), request_bytes=binding.request_bytes.decode(),
        getter_block=expected, result_bytes=result.result_bytes.decode(), retained_bytes=runtime._retained_bytes,
        slots=len(runtime._contents), qualification_sends=8, capture_sends=0, getter_calls=1,
        source_identity=True, producer_identity=True, result_identity=True, ordered_parents=True), sort_keys=True))


def bind(world, **options):
    arguments = dict(store=world.store, producer_ref=world.memory_policy.producer_ref)
    arguments.update(options)
    world.token = world.adapter.bind_memory_snapshot(world.agent, **arguments)
    return _memory_snapshots[world.adapter]


def assert_denied(world, *, slots=0, getter=False):
    assert world.context._denied and world.adapter._invalid
    owner = _memory_snapshots.get(world.adapter)
    if owner is not None:
        assert owner.phase == "denied" and owner.active_call is None
        assert owner.getter_entered is getter
    assert len(world.adapter._runtime._contents) == slots
    assert world.agent._memory_store is world.agent._memory_manager is None
    assert not world.adapter.invocations and world.adapter.send_count == 0 and not world.sdk_calls
    assert all(not port.sent for port in world.ports.values())


def test_actual_empty_snapshot(excluded, record_property):
    world = MemoryWorld(empty=True).construct()
    with getter_observations(world.store) as observations:
        result = world.capture()
    assert result.block is None
    assert result.result_bytes == b'{"block":null,"target":"memory"}'
    assert len(result.result_bytes) == 32
    assert digest(result.result_bytes) == "f87661743e6e96ece438d3a69d2ba49aebb1cc8e00661af59a64fd346faa5c2f"
    assert sum(row[0] == "call" for row in observations) == 1
    assert result.parents == (result.binding.request_content, result.binding.source_content)
    assert world.adapter._runtime._content(result.result_content).profiles == ("restricted_documents",)
    assert all(not port.sent for port in world.ports.values())
    record_property("empty_proof", json.dumps(dict(source_bytes=result.binding.source_bytes.decode(),
        result_bytes=result.result_bytes.decode(), block=None, getter_calls=1, qualification_sends=8, capture_sends=0)))


@pytest.mark.parametrize("explicit", [False, True])
def test_original_options_and_retained_profiles(excluded, explicit):
    world = MemoryWorld().construct()
    with content_policy_scope(world.context):
        arguments = dict(producer_ref=world.memory_policy.producer_ref)
        if explicit:
            arguments["target"] = world.source.target
        arguments["store"] = world.store
        token = world.adapter.bind_memory_snapshot(world.agent, **arguments)
        owner = _memory_snapshots[world.adapter]
        options = owner.binding.options
        assert options[1][0] is world.adapter and options[1][1] is world.agent
        assert [name for name, _ in options[2]] == list(arguments)
        assert all(value is arguments[name] for name, value in options[2])
        assert next(row for row in options[0] if row[0] == "target")[1] is explicit
        world.binding = replace(world.binding, profiles=())
        result = world.adapter.read_memory_snapshot(world.agent, token)
        assert result.binding.producer_policy is world.memory_policy
        for content in (result.binding.source_content, result.binding.request_content, result.result_content):
            assert world.adapter._runtime._content(content).profiles == ("restricted_documents",)
    world.adapter.validate_scope()
    world.agent.close()
    world.agent.close()
    world.adapter.validate_scope()
    assert owner.closed and world.adapter._constructed.closed and not world.context._denied


class Hostile:
    def __str__(self):
        pytest.fail("Converted an invalid option")

    def __eq__(self, other):
        pytest.fail("Compared an invalid option")

    def __bool__(self):
        pytest.fail("Converted an invalid option")


class TextSubclass(str):
    pass


@pytest.mark.parametrize("options", [
    dict(target="user"), dict(target=TextSubclass("memory")), dict(target=Hostile()),
    dict(producer_ref=""), dict(producer_ref=TextSubclass("invented_memory_producer")),
    dict(producer_ref=Hostile()), dict(store=Hostile()), dict(store=b"memory"), dict(store={}),
], ids=["user", "target-subclass", "target-hostile", "empty-producer", "producer-subclass",
        "producer-hostile", "store-hostile", "store-bytes", "store-record"])
def test_options_refuse_before_callbacks(excluded, options):
    world = MemoryWorld().construct()
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            def forbidden():
                pytest.fail("Observed an invalid option")
            world.admission_hook = world.selection_hook = world.producer_hook = forbidden
            bind(world, **options)
    assert_denied(world)
    assert not world.lookups


@pytest.mark.parametrize("change", [
    "missing-lookup", "missing-source", "json-source", "other-store", "equal-target", "wrong-owner",
    "zero-version", "bool-version", "producer-copy", "unauthorized-producer", "disabled", "bool-limit",
    "source-subclass", "profile-type", "user-flag-type", "snapshot-type",
], ids=str)
def test_source_admission_requires_original_association(excluded, change):
    world = MemoryWorld().construct()
    if change == "missing-lookup":
        del world.result_port.memory_source
    elif change == "missing-source":
        world.source = None
    elif change == "json-source":
        world.source = dict(store=world.store, target="memory")
    elif change == "other-store":
        other = MemoryStore()
        other.load_from_disk()
        assert other._system_prompt_snapshot == world.store._system_prompt_snapshot
        world.source = replace(world.source, store=other)
    elif change == "equal-target":
        target = "".join(("mem", "ory"))
        assert target == world.source.target and target is not world.source.target
        world.source = replace(world.source, target=target)
    elif change == "wrong-owner":
        world.source = replace(world.source, owner_ref="another_owner")
    elif change in ("zero-version", "bool-version"):
        world.source = replace(world.source, source_version=0 if change == "zero-version" else True)
    elif change == "producer-copy":
        world.source = replace(world.source, producer_policy=replace(world.memory_policy))
    elif change == "unauthorized-producer":
        world.memory_policy = replace(world.memory_policy, profiles=("extra_restriction",))
        world.source = replace(world.source, producer_policy=world.memory_policy)
    elif change == "disabled":
        world.store.memory_enabled = False
    elif change == "bool-limit":
        world.store.memory_char_limit = True
    elif change == "source-subclass":
        class SourceSubclass(NativeMemorySource):
            pass
        world.source = SourceSubclass(world.store, "memory", world.binding.owner_ref,
                                      "invented_host_profile", 1, world.memory_policy)
    elif change == "profile-type":
        world.source = replace(world.source, profile_ref=TextSubclass("invented_host_profile"))
    elif change == "user-flag-type":
        world.store.user_profile_enabled = 1
    elif change == "snapshot-type":
        world.store._system_prompt_snapshot["memory"] = TextSubclass(world.store._system_prompt_snapshot["memory"])
    with pytest.raises(NativeContentPolicyDenied, match="controlled_native_memory_"):
        with content_policy_scope(world.context):
            bind(world)
    assert_denied(world)


@pytest.mark.parametrize("change", [
    "source-copy", "source-version", "producer-copy", "producer-revision", "lookup", "current",
    "mapping-copy", "block", "user-block", "flags", "limits", "live-entry", "getter", "class",
    "holder", "selection-copy", "destination-copy", "main-route-copy", "health-method",
], ids=str)
def test_binding_rejects_later_substitution(excluded, monkeypatch, change):
    world = MemoryWorld().construct()
    with pytest.raises(NativeContentPolicyDenied, match="controlled_native_memory_"):
        with content_policy_scope(world.context):
            bind(world)
            if change == "source-copy":
                world.source = replace(world.source)
            elif change == "source-version":
                object.__setattr__(world.source, "source_version", 2)
            elif change == "producer-copy":
                world.memory_policy = replace(world.memory_policy)
            elif change == "producer-revision":
                object.__setattr__(world.memory_policy, "revision", 7)
            elif change == "lookup":
                world.result_port.memory_source = lambda *args: world.source
            elif change == "current":
                world.result_port.current = lambda ref: world.memory_policy
            elif change == "mapping-copy":
                world.store._system_prompt_snapshot = dict(world.store._system_prompt_snapshot)
            elif change == "block":
                world.store._system_prompt_snapshot["memory"] = world.store.memory_entries[0]
            elif change == "user-block":
                world.store._system_prompt_snapshot["user"] += "changed"
            elif change == "flags":
                world.store.user_profile_enabled = False
            elif change == "limits":
                world.store.user_char_limit += 1
            elif change == "live-entry":
                world.store.memory_entries.append("Changed live entry.")
            elif change == "getter":
                world.store.format_for_system_prompt = lambda target: world.store._system_prompt_snapshot[target]
            elif change == "class":
                from tools import memory_tool
                monkeypatch.setattr(memory_tool, "MemoryStore", type("OtherStore", (MemoryStore,), {}))
            elif change == "holder":
                world.binding = replace(world.binding, auth_epoch=8)
            elif change == "selection-copy":
                world.selection = replace(world.selection)
            elif change == "destination-copy":
                world.ports["memory"].destination = replace(world.ports["memory"].destination)
            elif change == "main-route-copy":
                world.adapter._main_route = replace(world.adapter._main_route)
            elif change == "health-method":
                monkeypatch.setattr(world.adapter._runtime, "_check_operator", lambda: None)
            world.adapter.read_memory_snapshot(world.agent, world.token)
    assert_denied(world, slots=2)


@pytest.mark.parametrize("kind", ["copied", "wrong-type", "wrong-agent"])
def test_original_token_and_agent(excluded, kind):
    world = MemoryWorld().construct()
    with pytest.raises(NativeContentPolicyDenied, match="token_invalid"):
        with content_policy_scope(world.context):
            bind(world)
            token = object.__new__(NativeMemorySnapshotInput) if kind == "copied" else object()
            agent = object() if kind == "wrong-agent" else world.agent
            if kind == "wrong-agent":
                token = world.token
            world.adapter.read_memory_snapshot(agent, token)
    assert_denied(world, slots=2)


def test_copied_context_keeps_original_thread(excluded):
    world = MemoryWorld().construct()
    failures = []
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            bind(world)
            context = copy_context()
            def wrong_thread():
                try:
                    context.run(world.adapter.read_memory_snapshot, world.agent, world.token)
                except BaseException as error:
                    failures.append(error)
            thread = threading.Thread(target=wrong_thread)
            thread.start()
            thread.join(timeout=10)
            assert not thread.is_alive()
    assert len(failures) == 1 and "thread_invalid" in str(failures[0])
    assert_denied(world, slots=2)


@pytest.mark.parametrize("operation", ["read", "bind", "model", "dispatch", "tool", "validation"])
def test_callback_reentry_never_uses_phase_or_equal_arguments(excluded, operation):
    world = MemoryWorld().construct()
    caught = []
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            bind(world)
            def nested():
                world.on_lookup = None
                try:
                    if operation == "read":
                        world.adapter.read_memory_snapshot(world.agent, world.token)
                    elif operation == "bind":
                        bind(world)
                    elif operation == "model":
                        world.agent.run_conversation("Invented work", system_message="Invented system")
                    elif operation == "dispatch":
                        world.adapter.dispatch(world.agent, dict(model="invented_model", messages=[]))
                    elif operation == "tool":
                        guard_tool_result(world.agent, "invented")
                    elif operation == "validation":
                        world.adapter.validate_scope()
                except BaseException as error:
                    caught.append(error)
            world.on_lookup = nested
            world.adapter.read_memory_snapshot(world.agent, world.token)
    assert len(caught) == 1 and isinstance(caught[0], NativeContentPolicyDenied)
    assert_denied(world, slots=2)


@pytest.mark.parametrize("exception", [InterruptedError, asyncio.CancelledError, KeyboardInterrupt, SystemExit])
def test_original_controls_survive_shared_normalization(excluded, exception):
    world = MemoryWorld().construct()
    original = exception("Invented control")
    observed = []
    def stop():
        if world.adapter._runtime._busy:
            observed.append(original)
            raise original
    world.on_clock = stop
    with pytest.raises(exception) as failure:
        with content_policy_scope(world.context):
            bind(world)
    assert failure.value is original
    assert observed == [original] and _memory_snapshots[world.adapter].control_exception is original
    assert_denied(world)


@pytest.mark.parametrize("boundary", ["getter", "derived", "release"])
def test_late_poison_cannot_release_from_graph_reads(excluded, boundary, record_property):
    world = MemoryWorld().construct()
    poisoned = []
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            owner = bind(world)
            def poison():
                ready = (owner.getter_returned if boundary == "getter" else
                         owner.result_content is not None if boundary == "derived" else owner.phase == "released")
                if ready:
                    world.on_clock = None
                    def wrong_thread():
                        try:
                            world.adapter._runtime.admit(world.request, b"Invented forbidden thread")
                        except BaseException as error:
                            poisoned.append(error)
                    thread = threading.Thread(target=wrong_thread)
                    thread.start()
                    thread.join(timeout=10)
                    assert not thread.is_alive()
            world.on_clock = poison
            world.adapter.read_memory_snapshot(world.agent, world.token)
    assert len(poisoned) == 1 and poisoned[0].args == ("reentrant_or_threaded_use",)
    assert owner.getter_returned and owner.getter_block is world.store._system_prompt_snapshot["memory"]
    assert owner.result_bytes == canonical(dict(target="memory", block=owner.getter_block))
    assert_denied(world, slots=2 if boundary == "getter" else 3, getter=True)
    record_property("partial_proof", json.dumps(dict(boundary=boundary, getter_returned=True,
        result_bytes=owner.result_bytes.decode(), result_derived=owner.result_content is not None,
        released_to_caller=False, slots=len(world.adapter._runtime._contents), capture_sends=0)))


@pytest.mark.parametrize("boundary", ["getter", "release"])
def test_selected_grant_expiry_with_independent_live_coverage(excluded, boundary):
    world = MemoryWorld()
    original = next(route for route in world.routes if route.path == "main")
    alternate = replace(original, route_ref="independent_main", expires_at=200)
    world.routes = (*(route if route is original else replace(route, expires_at=200)
                      for route in world.routes), alternate)
    world.profiles = tuple(replace(profile, allowed_routes=(*profile.allowed_routes, alternate.route_ref))
                           for profile in world.profiles)
    world.construct()
    with pytest.raises(NativeContentPolicyDenied, match="selection_expired"):
        with content_policy_scope(world.context):
            owner = bind(world)
            def expire():
                if (owner.getter_returned if boundary == "getter" else owner.phase == "released"):
                    world.now = original.expires_at
            world.on_clock = expire
            world.adapter.read_memory_snapshot(world.agent, world.token)
    assert world.now < alternate.expires_at and world.now == original.expires_at
    assert all(any(route.path == path and world.now < route.expires_at for route in world.routes) for path in PATHS)
    assert_denied(world, slots=2 if boundary == "getter" else 3, getter=True)


def test_one_genuine_byte_capacity_boundary(excluded, record_property):
    world = MemoryWorld().construct()
    fillers = []
    with pytest.raises(NativeContentPolicyDenied, match="capacity_denied"):
        with content_policy_scope(world.context):
            owner = bind(world)
            runtime = world.adapter._runtime
            result_bytes = canonical(dict(target="memory", block=world.store._system_prompt_snapshot["memory"]))
            remaining = 8 * 1024 * 1024 - runtime._retained_bytes - len(result_bytes) + 1
            while remaining:
                size = min(1024 * 1024, remaining)
                fillers.append(runtime.admit(world.request, b"x" * size))
                remaining -= size
            charged = runtime._retained_bytes
            world.adapter.read_memory_snapshot(world.agent, world.token)
    assert len(fillers) == 8
    assert owner.getter_returned and owner.result_bytes == result_bytes
    assert owner.result_content is owner.record is None
    assert runtime._retained_bytes == charged == 8 * 1024 * 1024 - len(result_bytes) + 1
    assert_denied(world, slots=10, getter=True)
    record_property("capacity_proof", json.dumps(dict(setup_admissions=8, capture_admissions=2,
        failed_derivations=1, completed_results=0, charged_bytes=charged,
        result_bytes=result_bytes.decode(), capture_sends=0)))


def test_binding_failure_retains_one_actual_admission(excluded, record_property):
    world = MemoryWorld().construct()
    original = InterruptedError("Invented failure after source admission")
    token = None
    def stop():
        owner = _memory_snapshots.get(world.adapter)
        if owner is not None and owner.source_content is not None:
            raise original
    world.on_clock = stop
    with pytest.raises(InterruptedError) as failure:
        with content_policy_scope(world.context):
            token = world.adapter.bind_memory_snapshot(
                world.agent, store=world.store, producer_ref=world.memory_policy.producer_ref)
    owner = _memory_snapshots[world.adapter]
    assert failure.value is original and token is None and owner.binding is None
    assert owner.request_content is None and owner.record is None
    assert world.adapter._runtime._content(owner.source_content).payload is owner.source_bytes
    assert world.adapter._runtime._retained_bytes == len(owner.source_bytes)
    assert_denied(world, slots=1)
    record_property("partial_admission", json.dumps(dict(source_bytes=owner.source_bytes.decode(),
        slots=1, retained_bytes=len(owner.source_bytes), token_issued=False, getter_calls=0, capture_sends=0)))


@pytest.mark.parametrize("state", ["pending", "auxiliary", "fallback", "closed"])
def test_only_unused_constructed_parent_can_bind(excluded, state):
    world = MemoryWorld().construct()
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            if state == "pending":
                world.adapter._pending = object()
            elif state == "auxiliary":
                world.adapter._aux_binding = object()
            elif state == "fallback":
                world.adapter._fallback_terminal = True
            elif state == "closed":
                world.agent.close()
            world.adapter.bind_memory_snapshot(world.agent, store=world.store,
                producer_ref=world.memory_policy.producer_ref)
    assert_denied(world)
    assert not world.lookups


def test_binding_callback_has_no_token_or_read_authority(excluded):
    world = MemoryWorld().construct()
    caught = []
    issued = None
    def nested():
        owner = _memory_snapshots[world.adapter]
        assert owner.binding is None and owner.phase == "bound"
        try:
            world.adapter.read_memory_snapshot(world.agent, object())
        except NativeContentPolicyDenied as error:
            caught.append(error)
    world.on_lookup = nested
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            issued = world.adapter.bind_memory_snapshot(
                world.agent, store=world.store, producer_ref=world.memory_policy.producer_ref)
    assert len(caught) == 1 and issued is None
    assert_denied(world)


@pytest.mark.parametrize("call", ["no-scope", "direct-owner"])
def test_original_scope_and_public_call_frame(excluded, call):
    world = MemoryWorld().construct()
    with content_policy_scope(world.context):
        owner = bind(world)
    with pytest.raises(NativeContentPolicyDenied):
        if call == "no-scope":
            world.adapter.read_memory_snapshot(world.agent, world.token)
        else:
            with content_policy_scope(world.context):
                owner.read(world.agent, world.token, sys._getframe())
    assert_denied(world, slots=2)


def test_source_bound_refuses_complete_oversize_without_truncation(excluded):
    world = MemoryWorld(entry="z" * 262144).construct()
    with pytest.raises(NativeContentPolicyDenied, match="source_size_invalid"):
        with content_policy_scope(world.context):
            bind(world)
    assert_denied(world)


@pytest.mark.parametrize("operation", ["read", "bind", "main", "tool", "delegation"])
def test_released_owner_is_terminal_after_profile_removal(excluded, operation):
    world = MemoryWorld().construct()
    result = world.capture()
    world.binding = replace(world.binding, profiles=())
    world.adapter.validate_scope()
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(world.context):
            if operation == "read":
                world.adapter.read_memory_snapshot(world.agent, world.token)
            elif operation == "bind":
                bind(world)
            elif operation == "main":
                world.agent.run_conversation("Invented work", system_message="Invented system")
            elif operation == "tool":
                guard_tool_result(world.agent, "invented")
            elif operation == "delegation":
                world.adapter.bind_delegation(world.agent, receiver_request=object(), receiver_selection=object(),
                    receiver_producer_ref="invented", child_request=object(), child_selection=object(),
                    child_producer_ref="invented", delegate_route_ref="invented", child_main_route_ref="invented",
                    child_session_id="invented", child_task_id="invented")
    assert _memory_snapshots[world.adapter].record is result
    assert_denied(world, slots=3, getter=True)


def test_phase_relabel_never_repeats_actual_getter(excluded):
    world = MemoryWorld().construct()
    original = world.capture()
    owner = _memory_snapshots[world.adapter]
    owner.phase = owner.phase_anchor = "bound"
    with getter_observations(world.store) as observations:
        with pytest.raises(NativeContentPolicyDenied):
            with content_policy_scope(world.context):
                world.adapter.read_memory_snapshot(world.agent, world.token)
    assert not observations
    assert owner.record is original
    assert_denied(world, slots=3, getter=True)
