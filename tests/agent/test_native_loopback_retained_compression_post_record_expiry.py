"""Check expiry after the adapter accepts the completed compression record."""

from dataclasses import asdict, replace, fields
from types import FrameType, SimpleNamespace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import pytest
from hermes_content_policy import PATHS

from agent import native_main_payload as native, context_compressor as compression
from agent.content_policy import NativeContentPolicyDenied, content_policy_scope
from tests.agent.owned_loopback_receiver import OwnedReceiver, ResponseStep, response_bytes
from tests.agent.test_native_controlled_conversation import (
    bind_compression, construct_compressor, summarize, construct, turn,
    compression_excluded as compression_excluded,
    excluded as excluded,
)
from tests.agent.test_native_loopback_fallback_denial import (
    ACK, QUALIFICATION, assert_receipt, charges, graph,
    ordinary_effects as ordinary_effects,
    record as _original_record,
)
from tests.agent.test_native_main_payload import (
    ControlledPort, World, canonical, reply,
    caller_boundary as caller_boundary,
)

SUMMARY_TEXT = "<think>scratch</think>## Goal\nKeep the blue folder."
MAIN_TEXT = "The blue folder stays."
USER_TEXT = "Keep the blue folder."


class _PhaseReceipts:
    def __init__(self, path):
        self.path = path

    def pytest_runtest_logreport(self, report):
        if report.nodeid.split("::", 1)[0] != "tests/agent/test_native_loopback_retained_compression_post_record_expiry.py":
            return
        row = dict(nodeid=report.nodeid, when=report.when, outcome=report.outcome,
            duration=report.duration, recorded_at=time.time())
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, "a") as output:
            output.write(json.dumps(row) + "\n")
            output.flush()
            os.fsync(output.fileno())


@pytest.fixture(scope="session", autouse=True)
def phase_receipts(pytestconfig):
    path = os.environ.get("HERMES_NATIVE_PHASE_RECEIPTS")
    if not path and pytestconfig.option.xmlpath:
        path = Path(pytestconfig.option.xmlpath).with_suffix(".phases.jsonl")
    if path:
        plugin = _PhaseReceipts(path)
        pytestconfig.pluginmanager.register(plugin)
        pytestconfig.add_cleanup(lambda: pytestconfig.pluginmanager.unregister(plugin))


from tests.agent.test_native_loopback_retained_compression import (
    compression_reply_bytes,
)


from tests.agent.test_native_loopback_retained_compression_expiry import (
    configure_expiry_world, qualify_expiry_world,
)


@pytest.fixture
def record(request, record_property):
    if request.config.option.xmlpath:
        return request.getfixturevalue("_original_record")
    return lambda name, value: record_property(name, value)


def typed_wire(response, model):
    assert type(response) is native.NativeMainResponse
    assert type(response.choices) is tuple and len(response.choices) == 1
    choice, = response.choices
    assert type(choice) is native.NativeMainChoice
    message, usage = choice.message, response.usage
    assert type(message) is native.NativeMainMessage
    assert type(usage) is native.NativeMainUsage
    assert message.tool_calls is None
    assert type(choice.index) is int and choice.index == 0
    assert all(type(value) is str for value in (response.id, response.object, response.model,
        choice.finish_reason, message.role, message.content))
    assert all(type(value) is int for value in (response.created, usage.prompt_tokens,
        usage.completion_tokens, usage.total_tokens))
    assert response.model == model
    value = asdict(response)
    del value["choices"][0]["message"]["tool_calls"]
    return canonical(value)


from tests.agent.test_native_loopback_retained_compression_decode_expiry import caught_exception_receipt

FIRST_STAGE = canonical(dict(content=SUMMARY_TEXT, finish_reason="stop", length_stop=False, nonempty=True))
FIRST_SHA256 = "6723f2258b6c86fe504e1faa94fddbe52cd715428a58d6dd4827e5eab20b8d32"


_CURRENT_FRAME = sys._getframe
_COMPLETION_CODE = native.NativeMainPayloadAdapter.complete_auxiliary.__code__


def row_facts(row):
    return (row.handle, row.payload, row.parents, row.binding, row.profiles, row.path,
        hashlib.sha256(row.payload).hexdigest(), len(row.payload))


def returned_output_facts(local, adapter, frame, record, binding, stages, output, rows):
    """Check projected facts. This predicate grants no original-frame authority."""
    assert len(rows) == 5 and len({id(row.handle) for row in rows}) == 5
    expected = dict(self=adapter, request=frame.sealed, response=record.native_response,
        frame=frame, record=record, binding=binding, stages=stages, output=output)
    assert all(local.get(key) is value for key, value in expected.items())
    assert local.get("path") == "compression" and local.get("task") == "compression"
    contents = local.get("contents")
    assert type(contents) is list and len(contents) == 4
    assert all(a is b.handle for a, b in zip(contents, rows[:4], strict=True))
    assert local.get("stage") is stages[3]
    base = (frame.source_content, record.request_content, record.result_content)
    for index, row in enumerate(rows):
        payload = stages[index] if index < 4 else local.get("payload")
        assert row.payload is payload and type(payload) is bytes
        assert row.binding == tuple(asdict(record.target).values())
        assert row.path == "compression" and row.profiles == ("restricted_documents",)
        assert all(a is b for a, b in zip(row.parents, (*base, *contents[:index]), strict=True))
    assert local.get("content") is rows[4].handle
    assert local["payload"] == canonical(output)
    assert all(a is b for a, b in zip(local.get("parents", ()), (*base, *contents), strict=True))
    return True


_OPERATOR_EXPECTED = '{"compression_binding":{"auth_epoch":7,"consumer_ref":"document_specialist","execution_ref":"execution","lineage_ref":"lineage","owner_ref":"owner","policy_revision":9,"profiles":["restricted_documents"],"selection_id":"compression_selection","selection_version":6},"compression_policy":{"producer_ref":"compression_producer","profiles":["restricted_documents"],"revision":3},"compression_selection":{"api_mode":"chat_completions","destination":{"endpoint_ref":"synthetic_loopback_c","model_ref":"invented_model_5","provider_ref":"synthetic_native","revision":1,"transport_ref":"compression_transport_compression"},"selection_id":"compression_selection","selection_version":6},"endpoints":[{"address":"127.0.0.1","endpoint_ref":"synthetic_loopback_p","model_ref":"invented_model_0","port":null,"provider_ref":"synthetic_native","request_path":"/owned-accountant-proof/v1/p","revision":1},{"address":"127.0.0.1","endpoint_ref":"synthetic_loopback_c","model_ref":"invented_model_5","port":null,"provider_ref":"synthetic_native","request_path":"/owned-accountant-proof/v1/c","revision":1}],"main_binding":{"auth_epoch":7,"consumer_ref":"document_specialist","execution_ref":"execution","lineage_ref":"lineage","owner_ref":"owner","policy_revision":9,"profiles":["restricted_documents"],"selection_id":"selection","selection_version":3},"main_policy":{"producer_ref":"invented_producer","profiles":["restricted_documents"],"revision":4},"main_selection":{"api_mode":"chat_completions","destination":{"endpoint_ref":"synthetic_loopback_p","model_ref":"invented_model_0","provider_ref":"synthetic_native","revision":1,"transport_ref":"invented_main"},"selection_id":"selection","selection_version":3},"now":10,"profiles":[{"allowed_routes":["route_main","route_fallback","route_title","route_compression","route_memory","route_delegate","route_a2a","route_tool_result","title_main","title_fallback","title_title","title_compression","title_memory","title_delegate","title_a2a","title_tool_result","compression_main","compression_fallback","compression_title","compression_compression","compression_memory","compression_delegate","compression_a2a","compression_tool_result","compression_compression_live_alternate"],"profile_ref":"restricted_documents","required_paths":["main","fallback","title","compression","memory","delegate","a2a","tool_result"]}],"routes":[{"consumer_ref":"document_specialist","destination":{"endpoint_ref":"synthetic_loopback_p","model_ref":"invented_model_0","provider_ref":"synthetic_native","revision":1,"transport_ref":"invented_main"},"expires_at":200,"path":"main","qualification_ref":"synthetic_loopback_main","qualification_scope":"synthetic_only","route_ref":"route_main","selection_id":"selection","selection_version":3},{"consumer_ref":"document_specialist","destination":{"endpoint_ref":"invented_endpoint","model_ref":"invented_model","provider_ref":"synthetic_native","revision":1,"transport_ref":"invented_fallback"},"expires_at":200,"path":"fallback","qualification_ref":"synthetic_inprocess_fallback","qualification_scope":"synthetic_only","route_ref":"route_fallback","selection_id":"selection","selection_version":3},{"consumer_ref":"document_specialist","destination":{"endpoint_ref":"invented_endpoint","model_ref":"invented_model","provider_ref":"synthetic_native","revision":1,"transport_ref":"invented_title"},"expires_at":200,"path":"title","qualification_ref":"synthetic_inprocess_title","qualification_scope":"synthetic_only","route_ref":"route_title","selection_id":"selection","selection_version":3},{"consumer_ref":"document_specialist","destination":{"endpoint_ref":"invented_endpoint","model_ref":"invented_model","provider_ref":"synthetic_native","revision":1,"transport_ref":"invented_compression"},"expires_at":200,"path":"compression","qualification_ref":"synthetic_inprocess_compression","qualification_scope":"synthetic_only","route_ref":"route_compression","selection_id":"selection","selection_version":3},{"consumer_ref":"document_specialist","destination":{"endpoint_ref":"invented_endpoint","model_ref":"invented_model","provider_ref":"synthetic_native","revision":1,"transport_ref":"invented_memory"},"expires_at":200,"path":"memory","qualification_ref":"synthetic_inprocess_memory","qualification_scope":"synthetic_only","route_ref":"route_memory","selection_id":"selection","selection_version":3},{"consumer_ref":"document_specialist","destination":{"endpoint_ref":"invented_endpoint","model_ref":"invented_model","provider_ref":"synthetic_native","revision":1,"transport_ref":"invented_delegate"},"expires_at":200,"path":"delegate","qualification_ref":"synthetic_inprocess_delegate","qualification_scope":"synthetic_only","route_ref":"route_delegate","selection_id":"selection","selection_version":3},{"consumer_ref":"document_specialist","destination":{"endpoint_ref":"invented_endpoint","model_ref":"invented_model","provider_ref":"synthetic_native","revision":1,"transport_ref":"invented_a2a"},"expires_at":200,"path":"a2a","qualification_ref":"synthetic_inprocess_a2a","qualification_scope":"synthetic_only","route_ref":"route_a2a","selection_id":"selection","selection_version":3},{"consumer_ref":"document_specialist","destination":{"endpoint_ref":"invented_endpoint","model_ref":"invented_model","provider_ref":"synthetic_native","revision":1,"transport_ref":"invented_tool_result"},"expires_at":200,"path":"tool_result","qualification_ref":"synthetic_inprocess_tool_result","qualification_scope":"synthetic_only","route_ref":"route_tool_result","selection_id":"selection","selection_version":3},{"consumer_ref":"document_specialist","destination":{"endpoint_ref":"invented_endpoint","model_ref":"invented_title_model","provider_ref":"synthetic_native","revision":1,"transport_ref":"title_transport_main"},"expires_at":200,"path":"main","qualification_ref":"synthetic_inprocess_main","qualification_scope":"synthetic_only","route_ref":"title_main","selection_id":"title_selection","selection_version":5},{"consumer_ref":"document_specialist","destination":{"endpoint_ref":"invented_endpoint","model_ref":"invented_title_model","provider_ref":"synthetic_native","revision":1,"transport_ref":"title_transport_fallback"},"expires_at":200,"path":"fallback","qualification_ref":"synthetic_inprocess_fallback","qualification_scope":"synthetic_only","route_ref":"title_fallback","selection_id":"title_selection","selection_version":5},{"consumer_ref":"document_specialist","destination":{"endpoint_ref":"invented_endpoint","model_ref":"invented_title_model","provider_ref":"synthetic_native","revision":1,"transport_ref":"title_transport_title"},"expires_at":200,"path":"title","qualification_ref":"synthetic_inprocess_title","qualification_scope":"synthetic_only","route_ref":"title_title","selection_id":"title_selection","selection_version":5},{"consumer_ref":"document_specialist","destination":{"endpoint_ref":"invented_endpoint","model_ref":"invented_title_model","provider_ref":"synthetic_native","revision":1,"transport_ref":"title_transport_compression"},"expires_at":200,"path":"compression","qualification_ref":"synthetic_inprocess_compression","qualification_scope":"synthetic_only","route_ref":"title_compression","selection_id":"title_selection","selection_version":5},{"consumer_ref":"document_specialist","destination":{"endpoint_ref":"invented_endpoint","model_ref":"invented_title_model","provider_ref":"synthetic_native","revision":1,"transport_ref":"title_transport_memory"},"expires_at":200,"path":"memory","qualification_ref":"synthetic_inprocess_memory","qualification_scope":"synthetic_only","route_ref":"title_memory","selection_id":"title_selection","selection_version":5},{"consumer_ref":"document_specialist","destination":{"endpoint_ref":"invented_endpoint","model_ref":"invented_title_model","provider_ref":"synthetic_native","revision":1,"transport_ref":"title_transport_delegate"},"expires_at":200,"path":"delegate","qualification_ref":"synthetic_inprocess_delegate","qualification_scope":"synthetic_only","route_ref":"title_delegate","selection_id":"title_selection","selection_version":5},{"consumer_ref":"document_specialist","destination":{"endpoint_ref":"invented_endpoint","model_ref":"invented_title_model","provider_ref":"synthetic_native","revision":1,"transport_ref":"title_transport_a2a"},"expires_at":200,"path":"a2a","qualification_ref":"synthetic_inprocess_a2a","qualification_scope":"synthetic_only","route_ref":"title_a2a","selection_id":"title_selection","selection_version":5},{"consumer_ref":"document_specialist","destination":{"endpoint_ref":"invented_endpoint","model_ref":"invented_title_model","provider_ref":"synthetic_native","revision":1,"transport_ref":"title_transport_tool_result"},"expires_at":200,"path":"tool_result","qualification_ref":"synthetic_inprocess_tool_result","qualification_scope":"synthetic_only","route_ref":"title_tool_result","selection_id":"title_selection","selection_version":5},{"consumer_ref":"document_specialist","destination":{"endpoint_ref":"invented_endpoint","model_ref":"invented_compression_model","provider_ref":"synthetic_native","revision":1,"transport_ref":"compression_transport_main"},"expires_at":200,"path":"main","qualification_ref":"synthetic_inprocess_main","qualification_scope":"synthetic_only","route_ref":"compression_main","selection_id":"compression_selection","selection_version":6},{"consumer_ref":"document_specialist","destination":{"endpoint_ref":"invented_endpoint","model_ref":"invented_compression_model","provider_ref":"synthetic_native","revision":1,"transport_ref":"compression_transport_fallback"},"expires_at":200,"path":"fallback","qualification_ref":"synthetic_inprocess_fallback","qualification_scope":"synthetic_only","route_ref":"compression_fallback","selection_id":"compression_selection","selection_version":6},{"consumer_ref":"document_specialist","destination":{"endpoint_ref":"invented_endpoint","model_ref":"invented_compression_model","provider_ref":"synthetic_native","revision":1,"transport_ref":"compression_transport_title"},"expires_at":200,"path":"title","qualification_ref":"synthetic_inprocess_title","qualification_scope":"synthetic_only","route_ref":"compression_title","selection_id":"compression_selection","selection_version":6},{"consumer_ref":"document_specialist","destination":{"endpoint_ref":"synthetic_loopback_c","model_ref":"invented_model_5","provider_ref":"synthetic_native","revision":1,"transport_ref":"compression_transport_compression"},"expires_at":100,"path":"compression","qualification_ref":"synthetic_loopback_compression","qualification_scope":"synthetic_only","route_ref":"compression_compression","selection_id":"compression_selection","selection_version":6},{"consumer_ref":"document_specialist","destination":{"endpoint_ref":"invented_endpoint","model_ref":"invented_compression_model","provider_ref":"synthetic_native","revision":1,"transport_ref":"compression_transport_memory"},"expires_at":200,"path":"memory","qualification_ref":"synthetic_inprocess_memory","qualification_scope":"synthetic_only","route_ref":"compression_memory","selection_id":"compression_selection","selection_version":6},{"consumer_ref":"document_specialist","destination":{"endpoint_ref":"invented_endpoint","model_ref":"invented_compression_model","provider_ref":"synthetic_native","revision":1,"transport_ref":"compression_transport_delegate"},"expires_at":200,"path":"delegate","qualification_ref":"synthetic_inprocess_delegate","qualification_scope":"synthetic_only","route_ref":"compression_delegate","selection_id":"compression_selection","selection_version":6},{"consumer_ref":"document_specialist","destination":{"endpoint_ref":"invented_endpoint","model_ref":"invented_compression_model","provider_ref":"synthetic_native","revision":1,"transport_ref":"compression_transport_a2a"},"expires_at":200,"path":"a2a","qualification_ref":"synthetic_inprocess_a2a","qualification_scope":"synthetic_only","route_ref":"compression_a2a","selection_id":"compression_selection","selection_version":6},{"consumer_ref":"document_specialist","destination":{"endpoint_ref":"invented_endpoint","model_ref":"invented_compression_model","provider_ref":"synthetic_native","revision":1,"transport_ref":"compression_transport_tool_result"},"expires_at":200,"path":"tool_result","qualification_ref":"synthetic_inprocess_tool_result","qualification_scope":"synthetic_only","route_ref":"compression_tool_result","selection_id":"compression_selection","selection_version":6},{"consumer_ref":"document_specialist","destination":{"endpoint_ref":"invented_endpoint","model_ref":"invented_compression_model","provider_ref":"synthetic_native","revision":1,"transport_ref":"compression_transport_main"},"expires_at":200,"path":"compression","qualification_ref":"synthetic_loopback_compression","qualification_scope":"synthetic_only","route_ref":"compression_compression_live_alternate","selection_id":"compression_selection","selection_version":6}],"title_binding":{"auth_epoch":7,"consumer_ref":"document_specialist","execution_ref":"execution","lineage_ref":"lineage","owner_ref":"owner","policy_revision":9,"profiles":["restricted_documents"],"selection_id":"title_selection","selection_version":5},"title_policy":{"producer_ref":"title_producer","profiles":["restricted_documents"],"revision":2}}'


def operator_projection(world):
    endpoints = [asdict(endpoint) for endpoint in world.endpoints]
    for endpoint in endpoints:
        endpoint["port"] = None
    return dict(routes=[asdict(route) for route in world.routes], profiles=[asdict(profile) for profile in world.profiles],
        main_binding=asdict(world.binding), compression_binding=asdict(world.compression_binding),
        title_binding=asdict(world.title_binding), title_policy=asdict(world.title_policy),
        main_policy=asdict(world.policy), compression_policy=asdict(world.compression_policy),
        main_selection=asdict(world.selection), compression_selection=asdict(world.compression_selection),
        endpoints=endpoints, now=10)


_REQUIRED_FACTS = (
    "front_guard", "python_frame", "original_code", "original_locals", "partial_frame",
    "partial_record", "updated_frame", "retained_frame", "accepted", "accepted_anchors",
    "unchanged_fields", "nested_anchors", "returned_handles", "ordered_handles", "payload_identity",
    "content_identity", "parents", "binding", "profiles", "path", "response", "request",
    "pre_graph", "graph_identity", "authorities", "callbacks", "methods", "clock",
    "runtime_copy", "map_cardinality", "native_busy_valid", "shared_healthy_idle",
    "alternate_live", "prepared_tuple", "fourth_stage", "no_local_result", "no_prior_output",
)


def association_ready(facts):
    """Check original associations using opaque projected references."""
    try:
        pairs = (("local_frame", "original_frame"), ("local_record", "partial"),
            ("pending", "retained"), ("payload", "output_bytes"), ("payload", "row_payload"),
            ("content", "output_content"), ("content", "row_content"),
            ("stages", "prepared_stages"), ("output", "prepared_output"))
        assert all(facts[a] is facts[b] for a, b in pairs)
        assert facts["pending"] is not facts["original_frame"]
        assert facts["completed"] is not facts["partial"]
        assert len(facts["accepted"]) == len(facts["accepted_anchors"]) == 1
        assert facts["accepted"][0] is facts["accepted_anchors"][0] is facts["completed"]
        assert len(facts["handles"]) == len(facts["recorded_handles"]) == 4
        assert all(a is b for a, b in zip(facts["handles"], facts["recorded_handles"], strict=True))
        assert len(facts["parents"]) == len(facts["expected_parents"]) == 7
        assert all(a is b for a, b in zip(facts["parents"], facts["expected_parents"], strict=True))
        return True
    except (AssertionError, KeyError, TypeError, ValueError):
        return False


def post_record_ready(facts):
    """Check projections without granting frame or graph authority."""
    return (association_ready(facts.get("association", {}))
        and all(facts.get(name) is True for name in _REQUIRED_FACTS)
        and facts.get("now") == 10 and facts.get("spent") is False
        and facts.get("stages") == 4 and facts.get("rows") == 5
        and facts.get("pre_rows") == 16 and facts.get("records") == facts.get("anchors") == 1)


def nested_facts(value):
    """Serialize existing nested associations without application callbacks."""
    import hermes_content_policy as shared
    names = ("NativeMainResponse", "NativeMainChoice", "NativeMainMessage", "NativeMainUsage",
        "NativeMainSelection", "NativeResultPolicy", "NativeAuxiliaryInvocation", "NativeAuxiliaryLocalResult",
        "_AuxiliaryTitlePreparation", "_CompressionBinding", "_AuxiliaryBinding", "_CompressionPreparation",
        "_FallbackBinding", "_FailedSendObservation", "NativeFailedMainAttempt", "_FallbackFrame",
        "NativeFallbackInvocation", "_TurnFrame", "_PendingOperation", "NativeDelegateBinding", "NativeDelegateInvocation")
    records = tuple(getattr(native, name) for name in names) + tuple(getattr(shared, name) for name in
        ("ExecutionBinding", "Destination", "DispatchEnvelope", "RouteBinding", "ProfilePolicy", "DispatchFailureFacts"))
    result = []
    def visit(item):
        result.append(item)
        if type(item) is dict:
            for key, child in item.items():
                assert type(key) is str
                visit(key)
                visit(child)
        elif type(item) in (list, tuple):
            for child in item:
                visit(child)
        elif type(item) in records:
            for field in fields(item):
                visit(object.__getattribute__(item, field.name))
    visit(value)
    return tuple(result)


def accepted_record_facts(adapter, original_frame, original_record, updated, local, stages, output, rows):
    assert type(updated) is native._CompressionFrame and updated is not original_frame
    assert original_frame.record is original_record
    assert original_record.output_stages == original_record.output_contents == ()
    assert original_record.output is original_record.output_bytes is original_record.output_content is None
    for field in fields(original_frame):
        if field.name != "record":
            assert getattr(updated, field.name) is getattr(original_frame, field.name)
    completed = updated.record
    assert type(completed) is native.NativeAuxiliaryInvocation and completed is not original_record
    changed = {"output_stages", "output_contents", "output_bytes", "output", "output_content"}
    for field in fields(original_record):
        if field.name not in changed:
            assert getattr(completed, field.name) is getattr(original_record, field.name)
    assert completed.output_stages is stages and completed.output is output
    assert completed.output_bytes is local["payload"] is rows[4].payload
    assert completed.output_content is local["content"] is rows[4].handle
    assert len(completed.output_contents) == 4
    assert all(a is b for a, b in zip(completed.output_contents, local["contents"], strict=True))
    assert len(adapter._aux_frames) == 1 and adapter._aux_frames[-1][0] is updated
    values = tuple(getattr(updated, field.name) for field in fields(updated))
    anchor = adapter._aux_frames[-1][1]
    assert all(a is b for a, b in zip(values, anchor[0], strict=True))
    assert all(a is b for a, b in zip(nested_facts(values)[1:], anchor[1], strict=True))
    assert len(adapter._aux_records) == len(adapter._aux_anchors) == 1
    assert adapter._aux_records[0] is completed and adapter._aux_anchors[0][0] is completed
    values = tuple(getattr(completed, field.name) for field in fields(completed))
    anchor = adapter._aux_anchors[0]
    assert all(a is b for a, b in zip(values, anchor[1], strict=True))
    assert all(a is b for a, b in zip(nested_facts(values)[1:], anchor[2], strict=True))
    assert not adapter._aux_local_results
    association = dict(local_frame=local["frame"], original_frame=original_frame,
        local_record=local["record"], partial=original_record, pending=updated,
        retained=adapter._aux_frames[-1][0], completed=completed, accepted=adapter._aux_records,
        accepted_anchors=tuple(anchor[0] for anchor in adapter._aux_anchors),
        handles=tuple(local["contents"]), recorded_handles=completed.output_contents,
        parents=local["parents"], expected_parents=(original_frame.source_content,
            original_record.request_content, original_record.result_content, *local["contents"]),
        payload=local["payload"], output_bytes=completed.output_bytes, row_payload=rows[4].payload,
        content=local["content"], output_content=completed.output_content, row_content=rows[4].handle,
        stages=completed.output_stages, prepared_stages=stages, output=completed.output, prepared_output=output)
    assert association_ready(association)
    return completed


def expire_after_output_record(world):
    retained, captured, prepared = {}, [], []
    expected_response = compression_reply_bytes()
    originals = (world.routes, world.profiles, world.transports, world.selection,
        world.compression_selection, world.admission.current, world.selection_port.current,
        world.title_selection_port.current, world.compression_selection_port.current, world.result_port.current)
    authorities = (world.request, world.title_request, world.compression_request,
        world.binding, world.title_binding, world.compression_binding,
        world.policy, world.title_policy, world.compression_policy)
    assert canonical(operator_projection(world)) == _OPERATOR_EXPECTED.encode("ascii")
    assert world.now == 10 and world.adapter is None and not world.qualifications
    receiver_facts = tuple((key, port, binding, asdict(binding))
        for key, (port, binding) in world.receiver_bindings.items())
    operator_facts = tuple(asdict(route) for route in world.routes)
    ports = tuple((port, port.resolve, port.send) for port in world.ports.values())
    callables = tuple((owner, name, getattr(owner, name)) for owner, names in (
        (native, ("_response", "_decode_compression_response", "_compression_response_bytes")),
        (native.NativeMainPayloadAdapter, ("_dispatch_retained_operation", "dispatch_auxiliary",
            "complete_auxiliary", "_compression_output_stages", "_compression_response_transform",
            "_interrupt_turn", "_checkpoint", "_snapshot", "_compression_snapshot", "_invalidate", "_derive_auxiliary_output",
            "_retain_aux_frame", "_retain_aux_record", "_accept_aux_record")),
        (compression.ContextCompressor, ("_generate_summary",)),
    ) for name in names)

    def trigger():
        assert sys._getframe is _CURRENT_FRAME
        assert native.NativeMainPayloadAdapter.complete_auxiliary.__code__ is _COMPLETION_CODE
        assert canonical(operator_projection(world)) == _OPERATOR_EXPECTED.encode("ascii")
        assert len(world.receiver_bindings) == len(receiver_facts)
        for key, port, binding, facts in receiver_facts:
            assert world.receiver_bindings[key][0] is port and world.receiver_bindings[key][1] is binding
            assert asdict(binding) == facts
        adapter = world.adapter
        if adapter is None:
            return
        runtime = adapter._runtime
        if not captured:
            assert adapter._aux_pending is None and adapter._pending is None and adapter._operation is None
            rows = runtime._route_rows
            selected = next(route for route in world.routes if route.route_ref == "compression_compression")
            shared = rows[selected.route_ref]
            assert type(rows) is dict and len(rows) == 25
            assert tuple(asdict(route) for route in world.routes) == operator_facts
            assert set(rows) == {route.route_ref for route in world.routes}
            for route in world.routes:
                copied = rows[route.route_ref]
                assert type(copied) is type(route) and type(copied.destination) is type(route.destination)
                assert copied is not route and copied.destination is not route.destination
                assert asdict(copied) == asdict(route)
            captured.append((adapter, runtime, rows, shared, shared.destination, selected, adapter._clock,
                tuple((key, row, row.destination, asdict(row)) for key, row in rows.items())))
        original_adapter, original_runtime, route_rows, shared_route, shared_destination, selected, clock, route_facts = captured[0]
        assert adapter is original_adapter and runtime is original_runtime
        assert adapter._clock is clock and runtime._route_rows is route_rows
        assert route_rows[selected.route_ref] is shared_route and shared_route.destination is shared_destination
        assert tuple(asdict(route) for route in world.routes) == operator_facts
        assert len(route_rows) == len(route_facts)
        for key, row, destination, facts in route_facts:
            assert route_rows[key] is row and row.destination is destination and asdict(row) == facts
        assert all(getattr(owner, name) is original for owner, name, original in callables)
        current = (world.routes, world.profiles, world.transports, world.selection,
            world.compression_selection, world.admission.current, world.selection_port.current,
            world.title_selection_port.current, world.compression_selection_port.current, world.result_port.current)
        assert all(a is b if not callable(a) else a == b for a, b in zip(current, originals, strict=True))
        current_authorities = (world.request, world.title_request, world.compression_request,
            world.binding, world.title_binding, world.compression_binding,
            world.policy, world.title_policy, world.compression_policy)
        assert all(a is b for a, b in zip(current_authorities, authorities, strict=True))
        assert all(port.resolve == resolve and port.send == send for port, resolve, send in ports)
        assert world.admission_hook is trigger
        if retained:
            assert retained["spent"] and world.now == 100
            assert adapter._aux_pending is retained["updated_frame"]
            assert adapter._aux_output is retained["prepared"]
            assert adapter._dispatch_response is retained["native_response"]
            return
        assert world.now == 10
        if runtime._busy or runtime._poisoned or runtime._failures:
            return
        pending = adapter._aux_pending
        frame = prepared[0][0] if prepared else pending
        output_tuple = getattr(adapter, "_aux_output", None)
        if frame is None or frame.record is None or output_tuple is None:
            return
        assert type(frame) is native._CompressionFrame
        assert type(frame.record) is native.NativeAuxiliaryInvocation
        record = frame.record
        assert type(output_tuple) is tuple and len(output_tuple) == 3
        output, stages, summary = output_tuple
        assert type(output) is dict and set(output) == {"summary"}
        assert type(summary) is str and output["summary"] is summary
        assert type(stages) is tuple and len(stages) == 4 and all(type(stage) is bytes for stage in stages)
        assert stages[0] == FIRST_STAGE and len(stages[0]) == 125
        assert hashlib.sha256(stages[0]).hexdigest() == FIRST_SHA256
        assert json.loads(stages[3]) == dict(prefixed=summary, output=output)
        stripped = "## Goal\nKeep the blue folder."
        assert json.loads(stages[1]) == dict(stripped=stripped, chosen=stripped, redacted=stripped)
        facts = json.loads(stages[2])
        assert set(facts) == {"restored", "grounded", "augmented", "has_user_turn"}
        assert facts["restored"] == stripped and facts["has_user_turn"] is True
        assert type(facts["grounded"]) is type(facts["augmented"]) is str
        assert summary.endswith(facts["augmented"])
        assert runtime._busy is runtime._poisoned is False and not runtime._failures
        assert adapter._busy and not adapter._invalid and not world.context._denied
        assert adapter._operation is adapter._checked_operation is adapter._pending is None
        assert selected.expires_at == 100 and world.routes[-1].expires_at == 200
        assert all(route.expires_at == 200 for route in world.routes if route is not selected)
        completion = frame.completion
        assert type(completion) is native._NativeCompletion and completion is adapter._completions[-1]
        source, invocation = completion.frame, completion.invocation
        assert type(source) is native._TurnFrame and type(invocation) is native.NativeMainInvocation
        assert record.source_completion is completion and record.source_frame is source
        assert record.source_invocation is invocation and frame.messages is completion.result["messages"]
        assert record.source_handle is world.request and record.request_handle is world.compression_request
        assert record.source is world.binding and record.target is world.compression_binding
        assert record.selection is world.compression_selection and record.producer_policy is world.compression_policy
        assert record.path == record.task == "compression"
        assert type(frame.preparation) is native._CompressionPreparation
        assert frame.sealed is not None and type(frame.sealed) is native.NativeAuxiliaryRequest
        assert frame.stages is frame.preparation.stages is record.input_stages
        assert frame.stage_contents is record.stage_contents and len(frame.stages) == len(frame.stage_contents) == 7
        assert frame.request is frame.preparation.value["request"]
        assert frame.request_bytes is record.request_bytes
        assert frame.source_bytes is record.source_bytes and frame.source_bytes == canonical(frame.messages)
        assert frame.parents is record.parents and frame.parent_requests is record.parent_requests
        main_parents = (source.admitted, source.input_content, source.output_content, *source.request_contents,
            invocation.request_content, invocation.result_content)
        assert source.input_content is source.admitted
        assert len(record.parents) == len(main_parents) + 1
        assert all(a is b for a, b in zip(record.parents, (*main_parents, completion.content), strict=True))
        assert all(handle is world.request for handle in record.parent_requests)
        assert len(record.parent_requests) == len(record.parents)
        assert record.output_stages == record.output_contents == ()
        assert record.output is record.output_bytes is record.output_content is None
        envelope, raw, response = record.envelope, record.response_bytes, record.native_response
        assert type(raw) is bytes and raw == expected_response
        assert response is adapter._dispatch_response and record.result_content is adapter._dispatch_result
        assert typed_wire(response, world.compression_selection.destination.model_ref) == raw
        assert envelope.payload is record.request_bytes and envelope.source == world.compression_binding
        assert envelope.path == "compression" and envelope.profiles == ("restricted_documents",)
        assert envelope.destination == record.observed_destination == shared_destination == selected.destination
        sealed = json.loads(record.request_bytes)
        assert sealed == dict(path="compression", task="compression", selection=["compression_selection", 6, "chat_completions"],
            destination=asdict(world.compression_selection.destination), request=frame.request)
        assert json.loads(frame.stages[-1]) == frame.preparation.value
        binding = tuple(asdict(world.compression_binding).values())
        def row_for(handle, payload, parents):
            row = runtime._contents[id(handle)]
            assert row.handle is handle and row.payload is payload and row.path == "compression"
            assert row.binding == binding and row.profiles == ("restricted_documents",)
            assert all(a is b for a, b in zip(row.parents, parents, strict=True))
            return row
        bridge = row_for(frame.source_content, frame.source_bytes, frame.parents)
        for index, (payload, handle) in enumerate(zip(frame.stages, frame.stage_contents, strict=True)):
            row_for(handle, payload, (frame.source_content, *frame.stage_contents[:index]))
        request_row = row_for(record.request_content, record.request_bytes, (frame.source_content, *frame.stage_contents))
        result_row = row_for(record.result_content, raw, (record.request_content,))
        issued = runtime._envelopes[id(envelope)]
        assert issued.envelope is envelope and issued.content is request_row and issued.route is shared_route
        assert issued.sender_handle is world.compression_request
        port = world.compression_ports["compression"]
        assert len(port.observations) == 2
        observation = port.observations[1]
        assert observation.envelope is envelope and observation.output is raw
        assert observation.error is observation.transport_receipt is None
        main_handles = (source.admitted, source.output_content, *source.request_contents,
            invocation.request_content, invocation.result_content, completion.content)
        pre_handles = (*main_handles, frame.source_content, *frame.stage_contents, record.request_content, record.result_content)
        assert len(pre_handles) == 16 and len({id(handle) for handle in pre_handles}) == 16
        assert all(runtime._contents[id(handle)].handle is handle for handle in pre_handles)
        extra = set(runtime._contents) - {id(handle) for handle in pre_handles}
        anchors = (frame, record, frame.preparation, frame.request, frame.sealed, completion, source, invocation,
            output_tuple, output, stages, summary, response, envelope, raw, request_row, result_row,
            tuple((id(handle), runtime._contents[id(handle)], row_facts(runtime._contents[id(handle)])) for handle in pre_handles))
        if not prepared:
            assert not extra
            prepared.append(anchors)
        else:
            assert all(a is b for a, b in zip(anchors[:-1], prepared[0][:-1], strict=True))
            assert all(runtime._contents[key] is row and row_facts(row) == facts
                and row.payload is facts[1] and all(a is b for a, b in zip(row.parents, facts[2], strict=True))
                for key, row, facts in prepared[0][-1])
        if not extra:
            return
        assert 1 <= len(extra) <= 5
        output_rows = []
        for index, payload in enumerate(stages[:min(len(extra), 4)]):
            matches = [runtime._contents[key] for key in extra if runtime._contents[key].payload is payload]
            assert len(matches) == 1
            row, = matches
            assert id(row.handle) in extra and runtime._contents[id(row.handle)] is row
            row_for(row.handle, payload, (frame.source_content, record.request_content,
                record.result_content, *(entry.handle for entry in output_rows)))
            output_rows.append(row)
        assert len({id(row.handle) for row in output_rows}) == len(output_rows)
        if len(extra) < 5:
            return
        final_key, = extra - {id(row.handle) for row in output_rows}
        final_row = runtime._contents[final_key]
        assert final_key == id(final_row.handle) and final_row.payload == canonical(output)
        row_for(final_row.handle, final_row.payload, (frame.source_content, record.request_content,
            record.result_content, *(row.handle for row in output_rows)))
        output_rows.append(final_row)
        assert not adapter._aux_local_results
        assert len(adapter._aux_frames) == 1
        if pending is frame:
            assert not adapter._aux_records and not adapter._aux_anchors
            return
        live = _CURRENT_FRAME()
        while live is not None and live.f_code is not _COMPLETION_CODE:
            live = live.f_back
        if live is None:
            return
        assert type(live) is FrameType and live.f_code is _COMPLETION_CODE
        local = live.f_locals
        assert local.get("self") is adapter and local.get("request") is frame.sealed
        assert local.get("response") is response and local.get("frame") is frame and local.get("record") is record
        assert local.get("binding") is adapter._compression_binding.auxiliary
        assert local.get("stages") is stages and local.get("output") is output
        assert local.get("stage") is stages[3] and local.get("content") is final_row.handle
        assert final_row.payload is local.get("payload")
        returned_output_facts(local, adapter, frame, record, adapter._compression_binding.auxiliary,
            stages, output, output_rows)
        assert live.f_lineno == 3049
        completed = accepted_record_facts(adapter, frame, record, pending, local, stages, output, output_rows)
        facts = json.loads(stages[2])
        stripped = "## Goal\nKeep the blue folder."
        assert summary == compression.SUMMARY_PREFIX + "\n" + facts["augmented"]
        expected_payloads = (FIRST_STAGE,
            canonical(dict(stripped=stripped, chosen=stripped, redacted=stripped)),
            canonical(dict(restored=stripped, grounded=facts["grounded"], augmented=facts["augmented"], has_user_turn=True)),
            canonical(dict(prefixed=summary, output=output)), canonical(output))
        assert stages == expected_payloads[:4] and local["payload"] == expected_payloads[4]
        assert len(runtime._contents) == 21 and len(runtime._envelopes) == 2
        assert sum(len(row.payload) for row in runtime._contents.values()) == runtime._retained_bytes
        assert sum(len(row.payload) for _, row, _ in prepared[0][-1]) == 30099
        assert runtime._retained_bytes == 30099 + sum(map(len, expected_payloads))
        prefix = output_rows[0]
        retained.update(spent=True, updated_frame=pending, completed_record=completed,
            accepted_records=adapter._aux_records, accepted_anchors=adapter._aux_anchors, retained_frames=adapter._aux_frames,
            live_frame=live, completion_code=_COMPLETION_CODE,
            output_rows=tuple(output_rows), output_handles=tuple(row.handle for row in output_rows),
            final_payload=local["payload"], local_parents=local["parents"], expected_payloads=expected_payloads, frame=frame, record=record, prepared=output_tuple,
            prepared_output=output, prepared_stages=stages, prepared_summary=summary, prefix=prefix,
            prefix_bytes=prefix.payload, pre_handles=tuple(id(handle) for handle in pre_handles),
            pre_rows={key:row for key,row,_ in prepared[0][-1]}, native_response=response, choices=response.choices,
            choice=response.choices[0], message=response.choices[0].message, usage=response.usage,
            result=record.result_content, row=result_row, request_row=request_row, envelope=envelope,
            output=raw, actual=record.observed_destination, observation=observation, issued=issued,
            route_rows=route_rows, shared_route=shared_route, shared_destination=shared_destination,
            operator_route=selected, operator_facts=asdict(selected), selected=selected, alternate=world.routes[-1],
            clock=clock, current=world.admission.current, hook=trigger, callables=callables,
            pre_facts=prepared[0][-1], frame_fields=tuple(getattr(frame, field.name) for field in fields(frame)),
            partial_fields=tuple(getattr(record, field.name) for field in fields(record)),
            graph=graph(runtime), envelopes=dict(runtime._envelopes))
        retained["receipt"] = dict(before=10, after=100, frame_id=id(frame), record_id=id(record),
            updated_frame_id=id(pending), accepted_record_id=id(completed), accepted_count=len(adapter._aux_records),
            accepted_anchors_id=id(adapter._aux_anchors), checkpoint=live.f_lineno,
            completion_frame_id=id(live), completion_code_id=id(_COMPLETION_CODE),
            returned_stage_ids=[id(handle) for handle in local["contents"]], final_content_id=id(local["content"]),
            output_handle_ids=[id(row.handle) for row in output_rows],
            output_payloads=[dict(payload=row.payload, sha256=hashlib.sha256(row.payload).hexdigest(),
                length=len(row.payload), parents=[id(parent) for parent in row.parents]) for row in output_rows],
            prepared_tuple_id=id(output_tuple), prepared_stage_ids=[id(stage) for stage in stages],
            prepared_output_id=id(output), prepared_summary_id=id(summary), prefix_handle_id=id(prefix.handle),
            prefix_payload_id=id(prefix.payload), prefix_payload=prefix.payload, prefix_sha256=FIRST_SHA256,
            prefix_parents=[id(parent) for parent in prefix.parents], pre_handles=retained["pre_handles"],
            route_rows_id=id(route_rows), shared_route_id=id(shared_route), shared_destination_id=id(shared_destination),
            operator_route_id=id(selected), operator_route=asdict(selected), shared_route=asdict(shared_route),
            response_id=id(response), raw_id=id(raw), request_content_id=id(record.request_content),
            result_content_id=id(record.result_content), charges=charges(runtime), graph=retained["graph"],
            local_partial_output_fields_empty=True, accepted_completed_output_fields=True, private_prepared_output_exists=True, shared_healthy_idle=True, spent=True)
        world.now = 100
    return trigger, retained


def test_post_record_expiry_gate_waits_for_accepted_record():
    facts = {name: True for name in _REQUIRED_FACTS}
    original_frame, partial, pending, completed = (object() for _ in range(4))
    handles = tuple(object() for _ in range(4))
    payload, content, stages, output = object(), object(), object(), object()
    parents = tuple(object() for _ in range(3)) + handles
    association = dict(local_frame=original_frame, original_frame=original_frame,
        local_record=partial, partial=partial, pending=pending, retained=pending,
        completed=completed, accepted=(completed,), accepted_anchors=(completed,),
        handles=handles, recorded_handles=tuple(list(handles)), parents=parents,
        expected_parents=tuple(list(parents)), payload=payload, output_bytes=payload,
        row_payload=payload, content=content, output_content=content, row_content=content,
        stages=stages, prepared_stages=stages, output=output, prepared_output=output)
    facts.update(now=10, spent=False, stages=4, rows=5, pre_rows=16, records=1, anchors=1,
        association=association)
    assert post_record_ready(facts)
    for key in association:
        assert not post_record_ready(dict(facts, association=dict(association, **{key: object()})))
    for key, wrong in (("pending", original_frame), ("completed", partial),
            ("accepted", ()), ("accepted", (completed, completed)),
            ("accepted_anchors", (partial,)), ("handles", handles[::-1]),
            ("parents", parents[::-1]), ("recorded_handles", (*handles, handles[0]))):
        assert not post_record_ready(dict(facts, association=dict(association, **{key: wrong})))
    assert not association_ready({})
    rejections = []
    for name in _REQUIRED_FACTS:
        assert not post_record_ready(dict(facts, **{name: False}))
        rejections.append(name)
    for name, wrong in (("now", 100), ("spent", True), ("stages", 3), ("rows", 4),
            ("rows", 6), ("pre_rows", 15), ("pre_rows", 17), ("records", 0),
            ("records", 2), ("anchors", 0), ("anchors", 2)):
        assert not post_record_ready(dict(facts, **{name: wrong}))
    now, spent, captures = 10, False, []
    for candidate in [dict(facts, accepted=False), facts, facts]:
        current = dict(candidate, now=now, spent=spent)
        if post_record_ready(current):
            spent = True
            captures.append(candidate)
            now = 100
    assert now == 100 and spent and captures == [facts]
    assert len(rejections) == len(_REQUIRED_FACTS)
    assert not post_record_ready({})
    world = configure_expiry_world(SimpleNamespace(bound=("127.0.0.1", 19001)),
        SimpleNamespace(bound=("127.0.0.1", 19002)))
    hook, retained = expire_after_output_record(world)
    world.admission_hook = hook
    hook()
    assert world.now == 10 and not retained and world.adapter is None
    assert not world.qualifications and not world.sdk_calls


def test_genuine_compression_expiry_after_output_record_acceptance(ordinary_effects, caller_boundary, record):
    started = time.monotonic()
    modules = (sys.modules[__name__], native, compression, sys.modules["agent.auxiliary_client"], sys.modules["agent.transports.owned_loopback_http"],
        sys.modules["hermes_content_policy"], sys.modules["tests.agent.test_native_controlled_conversation"])
    assert Path(sys.prefix).resolve() == Path(
        "/home/mersinvald/.local/share/orca/runtime/accountant/hpwa-p10.47/venv").resolve()
    root = Path(__file__).resolve().parents[2]
    for module in modules:
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(root) if module.__name__ != "hermes_content_policy" else path == Path(sys.prefix) / "lib/python3.11/site-packages/hermes_content_policy/__init__.py"
    record("source", dict(interpreter=sys.executable, prefix=sys.prefix,
        modules={module.__name__: dict(path=str(Path(module.__file__).resolve()),
            sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()) for module in modules}))
    a, b = OwnedReceiver(observation_seconds=140.0), OwnedReceiver(observation_seconds=140.0)
    world, result, summary, qualifier = None, None, None, None
    receipts = {}
    verified = False
    try:
        world = configure_expiry_world(a, b)
        hook, trigger = expire_after_output_record(world)
        world.admission_hook = hook
        hook()
        assert world.now == 10 and not trigger
        for receiver, endpoint, port in zip((a, b), world.endpoints,
                (world.ports["main"], world.compression_ports["compression"]), strict=True):
            assert (endpoint.address, endpoint.port) == receiver.bound
            assert port.destination is (world.selection.destination if receiver is a else world.compression_selection.destination)
            assert dict(world.transports)[port.destination.transport_ref] is port
            assert world.receiver_bindings[id(port)][0] is port
        assert a.bound != b.bound
        record("authentic_baseline", dict(routes=[asdict(row) for row in world.authentic_baseline[0]],
            profiles=[asdict(row) for row in world.authentic_baseline[1]]))
        record("authorized_delta", dict(expiries={r.route_ref:r.expires_at for r in world.routes},
            alternate=asdict(world.routes[-1]), profiles=[asdict(row) for row in world.profiles]))
        authority_originals = (*world.initial_originals[:9], world.profiles, *world.initial_originals[10:])
        originals = (world.routes, world.profiles, world.transports, world.selection, world.compression_selection,
            tuple((port, port.resolve, port.send) for port in world.ports.values()))
        listeners = (a.listener, b.listener)
        main_value = json.loads(reply(MAIN_TEXT))
        main_value["model"] = "invented_model_0"
        main_bytes, summary_bytes = canonical(main_value), compression_reply_bytes()
        record("operator", dict(routes=[asdict(route) for route in world.routes],
            profiles=[asdict(profile) for profile in world.profiles],
            main_binding=asdict(world.binding), compression_binding=asdict(world.compression_binding),
            main_policy=asdict(world.policy), compression_policy=asdict(world.compression_policy),
            main_selection=asdict(world.selection), compression_selection=asdict(world.compression_selection),
            endpoints=[asdict(endpoint) for endpoint in world.endpoints], now=world.now))
        for receiver, endpoint, payload in zip((a, b), world.endpoints, (main_bytes, summary_bytes), strict=True):
            receiver.start((ResponseStep(endpoint.request_path.encode(), response_bytes(ACK), QUALIFICATION),
                ResponseStep(endpoint.request_path.encode(), response_bytes(payload))))
        qualifier = qualify_expiry_world(world, record)
        a.wait(1)
        b.wait(1)
        record("qualification_complete", dict(graph=graph(qualifier), charges=charges(qualifier),
            envelopes=[asdict(row.envelope) for row in qualifier._envelopes.values()],
            hops=[lineage.hops for lineage in qualifier._lineages.values()], seconds=time.monotonic()-started))
        assert len(a.rows) == len(b.rows) == 1 and a.rows[0]["settled"] and b.rows[0]["settled"]
        assert a._mark is b._mark is None
        World.bind(world, qualify=False)
        construct(world, model="invented_model_0", max_tokens=None)
        from run_agent import AIAgent
        assert type(world.agent) is AIAgent
        assert world.agent.model == "invented_model_0" and world.agent.max_tokens is None
        assert world.agent.client is world.agent._primary_runtime is None
        assert "temperature" not in vars(world.agent)
        original_agent = dict(vars(world.agent))
        original_clock = adapter_clock = world.adapter._clock
        original_current = world.admission.current
        result = turn(world, USER_TEXT, system_message="Fixed system")
        adapter, runtime = world.adapter, world.adapter._runtime
        invocation, = adapter.invocations
        frame, = adapter._turn_frames
        completion, = adapter._completions
        assert completion.result is result and completion.invocation is invocation and completion.frame is frame
        assert result["final_response"] == MAIN_TEXT and result["final_response"] is invocation.native_response.choices[0].message.content
        assert result["completed"] and not result["failed"] and not result["interrupted"]
        assert completion.payload == canonical(result)
        assert type(invocation.native_response) is native.NativeMainResponse
        assert invocation.response_bytes == main_bytes and invocation.native_response.model == "invented_model_0"
        assert native._response_bytes(invocation.native_response, "invented_model_0") == main_bytes
        assert frame.arguments[0] is USER_TEXT and frame.arguments[2] is None
        main_parents = (frame.admitted, frame.input_content, frame.output_content, *frame.request_contents,
            invocation.request_content, invocation.result_content)
        assert frame.input_content is frame.admitted
        main_contents = (frame.admitted, frame.output_content, *frame.request_contents,
            invocation.request_content, invocation.result_content, completion.content)
        assert len({id(handle) for handle in main_contents}) == len(main_contents)
        assert {id(row.handle) for row in runtime._contents.values()} == {id(handle) for handle in main_contents}
        assert all(x is y for x, y in zip(runtime._contents[id(completion.content)].parents, main_parents, strict=True))
        assert adapter._turn_pending is adapter._operation is None and not world.context._denied
        assert len(main_contents) == 6
        record("main_released", dict(result=result, graph=graph(runtime), charges=charges(runtime),
            envelope=asdict(invocation.envelope), response=invocation.response_bytes,
            completion_id=id(completion), result_id=id(result), frame_id=id(frame), invocation_id=id(invocation),
            native_response_id=id(invocation.native_response), seconds=time.monotonic()-started))
        main_retained_bytes = runtime._retained_bytes
        expected_content_tokens = compression.estimate_messages_tokens_rough(result["messages"])
        original_result = canonical(result)
        prefix = world.agent._cached_system_prompt
        messages = result["messages"]
        message_rows = tuple(messages)
        with pytest.raises(NativeContentPolicyDenied) as refusal:
            with content_policy_scope(world.context):
                bind_compression(world)
                compressor = construct_compressor(world)
                compressor_state = dict(vars(compressor))
                summary = summarize(world, compressor, result)
        error = refusal.value
        record("caught_native_exception", caught_exception_receipt(error))
        record("trigger", trigger["receipt"])
        chain = [error]
        assert type(error) is NativeContentPolicyDenied and error.args == ("controlled_native_payload_invalid",)
        assert error.__cause__ is error.__context__ is None and error.__suppress_context__ is False
        original_traceback = traceback = error.__traceback__
        completion_frame = None
        locations = []
        while traceback is not None:
            if traceback.tb_frame.f_code is native.NativeMainPayloadAdapter.complete_auxiliary.__code__:
                completion_frame = traceback.tb_frame
            locations.append(dict(file=traceback.tb_frame.f_code.co_filename,
                function=traceback.tb_frame.f_code.co_name, line=traceback.tb_lineno))
            traceback = traceback.tb_next
        assert error.__traceback__ is original_traceback and completion_frame is not None
        local = completion_frame.f_locals
        assert local["frame"] is trigger["frame"] and local["record"] is trigger["record"]
        assert local["response"] is trigger["native_response"]
        assert local["stages"] is trigger["prepared"][1] and local["output"] is trigger["prepared"][0]
        assert completion_frame is trigger["live_frame"] and completion_frame.f_code is trigger["completion_code"]
        assert len(local["contents"]) == 4
        assert all(a is b for a, b in zip(local["contents"], trigger["output_handles"][:4], strict=True))
        assert local["content"] is trigger["output_handles"][4]
        assert local["payload"] is trigger["final_payload"]
        returned_output_facts(local, adapter, trigger["frame"], trigger["record"],
            adapter._compression_binding.auxiliary, trigger["prepared_stages"], trigger["prepared_output"], trigger["output_rows"])
        assert local["stage"] is trigger["prepared"][1][3]
        assert any(row["function"] == "complete_auxiliary" and row["line"] == 3049 for row in locations)
        assert any(row["function"] == "_compression_snapshot" and row["line"] == 3140 for row in locations)
        assert not any("site-packages/hermes_content_policy/" in row["file"] for row in locations)
        record("exception", dict(type=type(error).__name__, identity=id(error), args=error.args,
            cause=None, context=None, suppressed=False, locations=locations, traceback_id=id(original_traceback),
            completion_frame_id=id(completion_frame), prefix_handle_id=id(local["contents"][0]),
            returned_stage_ids=[id(handle) for handle in local["contents"]],
            final_content_id=id(local["content"]), final_payload_id=id(local["payload"]), predicate="now < route.expires_at", line=3140))
        assert summary is None and runtime._poisoned is False and world.context._denied is True
        assert trigger and trigger["clock"] is original_clock is adapter_clock
        assert trigger["current"] == original_current == world.admission.current
        assert trigger["hook"] is world.admission_hook is hook
        assert adapter._pending is adapter._aux_pending is adapter._operation is adapter._checked_operation is adapter._turn_pending is None
        auxiliary, = adapter.auxiliary_invocations
        aux_frame, = (entry[0] for entry in adapter._aux_frames)
        record("compression_accepted", dict(graph=graph(runtime), charges=charges(runtime),
            envelope=asdict(auxiliary.envelope), response=auxiliary.response_bytes,
            input_stages=auxiliary.input_stages, output_stages=auxiliary.output_stages,
            source_id=id(auxiliary.source_completion), frame_id=id(aux_frame), record_id=id(auxiliary),
            response_id=id(auxiliary.response_bytes), native_response_id=id(auxiliary.native_response),
            native_response=asdict(auxiliary.native_response),
            trigger_native_response_id=id(trigger["native_response"]), result_id=id(auxiliary.result_content),
            trigger_result_id=id(trigger["result"]), summary=None))
        assert completion_frame.f_locals["record"] is trigger["record"] and auxiliary is trigger["completed_record"]
        assert auxiliary is not trigger["record"] and aux_frame is trigger["updated_frame"]
        assert adapter._aux_records is trigger["accepted_records"]
        assert adapter._aux_anchors is trigger["accepted_anchors"]
        assert adapter._aux_frames is trigger["retained_frames"]
        assert accepted_record_facts(adapter, trigger["frame"], trigger["record"], aux_frame,
            local, trigger["prepared_stages"], trigger["prepared_output"], trigger["output_rows"]) is auxiliary
        assert auxiliary.source_completion is completion and auxiliary.source_completion.result is result
        assert auxiliary.source_frame is frame and auxiliary.source_invocation is invocation
        assert auxiliary.source_handle is world.request and auxiliary.request_handle is world.compression_request
        assert aux_frame.messages is messages is result["messages"]
        assert all(x is y for x, y in zip(messages, message_rows, strict=True))
        assert aux_frame.completion is completion and aux_frame.record is auxiliary
        expected_parents = (*main_parents, completion.content)
        assert all(x is y for x, y in zip(auxiliary.parents, expected_parents, strict=True))
        assert all(handle is world.request for handle in auxiliary.parent_requests)
        assert len(auxiliary.parent_requests) == len(expected_parents)
        assert auxiliary.source_bytes == canonical(messages)
        bridge = runtime._contents[id(aux_frame.source_content)]
        assert bridge.payload == auxiliary.source_bytes
        assert all(x is y for x, y in zip(bridge.parents, auxiliary.parents, strict=True))
        assert auxiliary.selection is world.compression_selection and auxiliary.producer_policy is world.compression_policy
        assert aux_frame.parents is auxiliary.parents and aux_frame.parent_requests is auxiliary.parent_requests
        assert auxiliary.path == auxiliary.task == "compression"
        assert (auxiliary.target.selection_id, auxiliary.target.selection_version) == ("compression_selection", 6)
        assert auxiliary.envelope.destination == auxiliary.observed_destination == world.compression_selection.destination
        assert auxiliary.envelope.path == "compression" and auxiliary.envelope.source == world.compression_binding
        assert auxiliary.request_bytes == auxiliary.envelope.payload
        sealed = json.loads(auxiliary.request_bytes)
        assert sealed["path"] == sealed["task"] == "compression"
        assert sealed["selection"] == ["compression_selection", 6, "chat_completions"]
        assert sealed["destination"] == asdict(world.compression_selection.destination)
        assert len(auxiliary.input_stages) == len(auxiliary.stage_contents) == 7
        facts = [json.loads(stage) for stage in auxiliary.input_stages]
        assert facts[0]["constructor"] == world.adapter._compression_options
        assert facts[0]["previous_summary"] is facts[0]["focus_topic"] is None
        assert facts[0]["source_field"] == "messages" and facts[0]["memory_context"] == ""
        assert facts[1] == dict(content_tokens=expected_content_tokens,
            ratio=0.2, context_length=65536, maximum=3276, summary_budget=2000)
        assert facts[2]["pruned_skill_names"] == [] and USER_TEXT in facts[2]["serialized"]
        assert facts[3] == dict(sampled=facts[2]["serialized"], lean=True, sampling=False,
            original_length=len(facts[2]["serialized"]), slice_count=8, aggregate_cap=160000)
        assert facts[4] == dict(sanitized_memory="", serialized_memory='""', memory_section="",
                               has_user_turn=True, summary_date="2026-11-23")
        assert "current date is 2026-11-23" in facts[5]["prompt"]
        assert "Keep the blue folder." in facts[5]["prompt"]
        assert compression._LEAN_SESSION_LOG_HEADING in facts[5]["prompt"]
        wire = dict(model="invented_model_5", messages=[dict(role="user", content=facts[5]["prompt"])], stream=False)
        request = json.loads(auxiliary.request_bytes)
        assert request["request"] == facts[6]["request"] == wire
        assert facts[6]["arguments"] == aux_frame.preparation.value["arguments"]
        assert aux_frame.request is aux_frame.preparation.value["request"]
        assert aux_frame.request["messages"] is aux_frame.preparation.value["arguments"]["messages"]
        assert aux_frame.request["messages"][0]["content"] == facts[5]["prompt"]
        assert aux_frame.preparation.value["arguments"]["temperature"] is aux_frame.preparation.value["arguments"]["max_tokens"] is None
        assert "temperature" not in wire and "max_tokens" not in wire
        assert auxiliary.result_content is trigger["result"]
        assert auxiliary.native_response is trigger["native_response"]
        assert auxiliary.native_response.choices is trigger["choices"]
        assert auxiliary.native_response.choices[0] is trigger["choice"]
        assert auxiliary.native_response.choices[0].message is trigger["message"]
        assert auxiliary.native_response.usage is trigger["usage"]
        assert native._compression_response_bytes(auxiliary.native_response, "invented_model_5") == auxiliary.response_bytes
        assert typed_wire(auxiliary.native_response, "invented_model_5") == auxiliary.response_bytes
        assert all(getattr(owner, name) is original for owner, name, original in trigger["callables"])
        assert runtime._contents[id(auxiliary.result_content)] is trigger["row"]
        assert trigger["row"].payload is auxiliary.response_bytes
        assert trigger["row"].parents == (auxiliary.request_content,)
        assert trigger["row"].parents[0] is auxiliary.request_content
        assert trigger["request_row"].handle is auxiliary.request_content
        assert trigger["updated_frame"] is aux_frame and trigger["frame"] is not aux_frame
        assert aux_frame.preparation is trigger["frame"].preparation
        assert aux_frame.request is trigger["frame"].request and aux_frame.sealed is trigger["frame"].sealed
        assert runtime._busy is runtime._poisoned is False and not runtime._failures
        assert adapter._invalid is True and adapter._busy is False
        with pytest.raises(NativeContentPolicyDenied) as scope_refusal:
            with content_policy_scope(world.context):
                pytest.fail("Denied scope entered")
        assert scope_refusal.value is not error and scope_refusal.value.args == ("controlled_native_payload_denied",)
        record("scope_refusal", dict(error_id=id(scope_refusal.value), reason=scope_refusal.value.args,
            original_id=id(error), distinct=True, shared_busy=runtime._busy, shared_poisoned=runtime._poisoned,
            native_busy=adapter._busy, native_invalid=adapter._invalid, context_denied=world.context._denied))
        assert auxiliary.output_stages is trigger["prepared_stages"]
        assert auxiliary.output is trigger["prepared_output"]
        assert auxiliary.output_bytes is trigger["final_payload"]
        assert auxiliary.output_content is trigger["output_handles"][-1]
        assert auxiliary.envelope is trigger["envelope"] and auxiliary.response_bytes is trigger["output"]
        assert auxiliary.request_bytes is trigger["request_row"].payload
        assert auxiliary.response_bytes == summary_bytes
        for index, (payload, content) in enumerate(zip(auxiliary.input_stages, auxiliary.stage_contents, strict=True)):
            row = runtime._contents[id(content)]
            assert row.payload == payload
            assert all(x is y for x, y in zip(row.parents, (aux_frame.source_content, *auxiliary.stage_contents[:index]), strict=True))
        row = runtime._contents[id(auxiliary.request_content)]
        assert row.payload == auxiliary.request_bytes
        assert all(x is y for x, y in zip(row.parents, (aux_frame.source_content, *auxiliary.stage_contents), strict=True))
        assert adapter._aux_output is trigger["prepared"]
        assert adapter._aux_output[0] is trigger["prepared_output"]
        assert adapter._aux_output[1] is trigger["prepared_stages"]
        assert adapter._aux_output[2] is trigger["prepared_summary"]
        for index, row in enumerate(trigger["output_rows"]):
            assert runtime._contents[id(row.handle)] is row
            expected_payload = trigger["expected_payloads"][index]
            assert row.payload == expected_payload
            assert row.payload is (trigger["prepared_stages"][index] if index < 4 else trigger["final_payload"])
            assert len(row.payload) == len(expected_payload)
            assert hashlib.sha256(row.payload).hexdigest() == hashlib.sha256(expected_payload).hexdigest()
            assert all(a is b for a, b in zip(row.parents,
                (aux_frame.source_content, auxiliary.request_content, auxiliary.result_content,
                    *trigger["output_handles"][:index]), strict=True))
            assert row.binding == tuple(asdict(world.compression_binding).values())
            assert row.profiles == ("restricted_documents",) and row.path == "compression"
        assert set(runtime._contents) == set(trigger["pre_handles"]) | {id(handle) for handle in trigger["output_handles"]}
        assert all(runtime._contents[key] is row for key, row in trigger["pre_rows"].items())
        for key, row, facts in trigger["pre_facts"]:
            assert runtime._contents[key] is row and row_facts(row) == facts and row.payload is facts[1]
            assert all(a is b for a, b in zip(row.parents, facts[2], strict=True))
        assert all(getattr(trigger["frame"], field.name) is value
            for field, value in zip(fields(trigger["frame"]), trigger["frame_fields"], strict=True))
        assert all(getattr(trigger["record"], field.name) is value
            for field, value in zip(fields(trigger["record"]), trigger["partial_fields"], strict=True))
        assert graph(runtime) == trigger["graph"]
        assert runtime._envelopes == trigger["envelopes"] and not runtime._failures
        assert adapter._runtime._route_rows is trigger["route_rows"]
        assert trigger["route_rows"]["compression_compression"] is trigger["shared_route"]
        assert trigger["shared_route"].destination is trigger["shared_destination"]
        assert asdict(trigger["shared_route"]) == trigger["operator_facts"] == asdict(trigger["operator_route"])
        assert vars(compressor) == compressor_state
        assert compressor._previous_summary is None and compressor.compression_count == 0
        assert canonical(result) == original_result and world.agent._cached_system_prompt is prefix
        assert adapter.auxiliary_local_results == ()
        assert adapter._aux_binding is adapter._fallback_binding is None
        assert not runtime._failures
        current_authorities = (world.request, world.title_request, world.compression_request,
            world.binding, world.title_binding, world.compression_binding, world.policy, world.title_policy,
            world.compression_policy, world.profiles, world.admission.current, world.selection_port.current,
            world.title_selection_port.current, world.compression_selection_port.current, world.result_port.current)
        assert all(x is y if not callable(x) else x == y for x, y in zip(current_authorities, authority_originals, strict=True))
        assert len(frame.request_contents) == 1
        assert len(wire["messages"]) <= 32
        assert all(len(row["content"].encode("utf-8")) <= 16384 for row in wire["messages"])
        assert all(len(payload) <= 262144 for payload in (auxiliary.source_bytes,))
        assert adapter.send_count == adapter.auxiliary_send_count == 1 and adapter.fallback_send_count == 0
        assert not adapter.failed_main_attempts and not adapter.fallback_invocations
        assert len(caller_boundary.dispatches) == 1 and not world.sdk_calls
        assert all(vars(world.agent)[key] is value for key, value in original_agent.items())
        assert all(x is y for x, y in zip(
            (world.routes, world.profiles, world.transports, world.selection, world.compression_selection),
            world.configured_baseline, strict=True))
        assert world.now == 100
        assert [route.route_ref for route in world.routes if route.expires_at <= world.now] == ["compression_compression"]
        assert world.routes[-1].selection_id == "compression_selection" and world.routes[-1].selection_version == 6
        assert world.routes[-1].expires_at > world.now and world.routes[-1].path == "compression"
        for port, resolve, send in originals[5]:
            assert port.resolve == resolve and port.send == send
            if type(port) is ControlledPort:
                assert not port.sent
        for current in (qualifier, runtime):
            rows = tuple(current._contents.values())
            assert sum(len(row.payload) for row in rows) == current._retained_bytes <= 8 * 1024 * 1024
            assert len(rows) + len(current._envelopes) + len(current._failures) <= 128
            assert all(len(row.payload) <= 1024 * 1024 and len(row.parents) <= 32 for row in rows)
            assert all(row.profiles == ("restricted_documents",) for row in rows)
            assert all(lineage.hops <= 128 for lineage in current._lineages.values())
        assert adapter._aux_pending is adapter._operation is adapter._turn_pending is None
        assert adapter._clock is original_clock
        assert len(runtime._envelopes) == 2
        assert len(runtime._contents) == len(main_contents) + 1 + len(auxiliary.stage_contents) + 1 + 1 + 5 == 21
        assert all(len(payload) <= 262144 for payload in (*auxiliary.input_stages, auxiliary.request_bytes, invocation.request_bytes))
        assert runtime._retained_bytes == sum(len(row.payload) for row in runtime._contents.values())
        assert runtime._retained_bytes == main_retained_bytes + len(auxiliary.source_bytes) + sum(map(len, auxiliary.input_stages)) + len(auxiliary.request_bytes) + len(auxiliary.response_bytes) + sum(len(row.payload) for row in trigger["output_rows"])
        assert charges(qualifier) == (50, 25, 7241)
        assert charges(runtime) == (21, 2, 30099 + sum(map(len, trigger["expected_payloads"])))
        assert len(auxiliary.response_bytes) == len(summary_bytes) == 312
        assert len(auxiliary.response_bytes) <= 65536 and len(invocation.response_bytes) <= 65536
        a.wait(2)
        b.wait(2)
        for receiver in (a, b):
            mark = receiver.mark_window()
            assert mark.row_count == 2 and mark.active_count == 0
        for name, receiver, listener, operation, payload in zip(("a", "b"), (a, b), listeners,
                (invocation, auxiliary), (main_bytes, summary_bytes), strict=True):
            receipt = receipts[name] = receiver.finish()
            record(name + "_final", asdict(receipt))
            assert_receipt(receipt)
            receiver.check()
            assert len(receipt.connections) == 2 and receipt.connections[receipt.mark.row_count:] == ()
            assert receipt.listener_id == id(listener) and receiver.listener is listener
            port = world.ports["main"] if name == "a" else world.compression_ports["compression"]
            assert len(port.observations) == 2
            for index, connection in enumerate(receipt.connections):
                observation = port.observations[index]
                assert connection.complete and connection.settled
                assert connection.header.endswith(b"\r\n\r\n")
                assert connection.raw == connection.header + connection.body
                assert connection.raw_length == len(connection.raw)
                assert connection.raw_sha256 == hashlib.sha256(connection.raw).hexdigest()
                assert connection.body_length == len(connection.body)
                assert b"Content-Length: " + str(len(connection.body)).encode() in connection.header.split(b"\r\n")
                assert connection.client_reported_peer is not None and connection.host is not None
                assert connection.peer != connection.local
                assert connection.local == connection.bound == receipt.bound and connection.peer[0] == "127.0.0.1"
                assert connection.body == observation.envelope.payload
                assert connection.body_sha256 == hashlib.sha256(observation.envelope.payload).hexdigest()
                header, separator, body = connection.response.partition(b"\r\n\r\n")
                expected = ACK if index == 0 else payload
                assert separator == b"\r\n\r\n" and body == expected
                assert b"Content-Length: " + str(len(body)).encode() in header.split(b"\r\n")
                assert observation.output == body and observation.error is observation.transport_receipt is None
                assert connection.response_accepted == len(connection.response)
                assert connection.response == response_bytes(body)
                assert (connection.body_length, len(body), len(connection.response)) == (
                    (22, 17, 75) if index == 0 else ((355, 270, 329) if name == "a" else (6664, 312, 371)))
            data = receipt.connections[1]
            assert port.observations[1].envelope is operation.envelope
            assert port.observations[1].output is operation.response_bytes
            assert data.body == operation.request_bytes
            assert data.response.partition(b"\r\n\r\n")[2] == operation.response_bytes
        assert receipts["a"].bound != receipts["b"].bound
        assert all(receipt.cutoff_at - receipt.mark.started_at >= 0.5 for receipt in receipts.values())
        record("verified", dict(result=result, summary=None, output=None, graph=graph(runtime),
            charges=charges(runtime), qualification_charges=charges(qualifier),
            envelopes=[asdict(row.envelope) for row in runtime._envelopes.values()],
            hops=[lineage.hops for lineage in runtime._lineages.values()],
            main_usage=dict(prompt_tokens=result["prompt_tokens"], completion_tokens=result["completion_tokens"], total_tokens=result["total_tokens"]),
            compression_usage=json.loads(auxiliary.response_bytes)["usage"], estimated_cost_usd=result["estimated_cost_usd"],
            cost_status=result["cost_status"], conversation_exit=True, auxiliary_scope_refused=True,
            agent_attributes_preserved=True, seconds=time.monotonic()-started))
        assert (result["prompt_tokens"], result["completion_tokens"], result["total_tokens"]) == (3, 2, 5)
        assert time.monotonic()-started < 145
        world.agent.close()
        world.agent.close()
        assert adapter.invocations == (invocation,) and adapter.auxiliary_invocations == (auxiliary,)
        with pytest.raises(NativeContentPolicyDenied) as caught:
            turn(world, USER_TEXT, system_message="Fixed system")
        assert type(caught.value) is NativeContentPolicyDenied and world.context._denied
        assert caught.value is not error and caught.value is not scope_refusal.value
        assert caught.value.args == ("controlled_native_payload_denied",)
        assert adapter._turn_pending is adapter._operation is None and adapter._constructed.closed
        assert adapter._send_count == adapter._aux_send_count == 1 and adapter._fallback_send_count == 0
        assert len(world.ports["main"].observations) == len(world.compression_ports["compression"].observations) == 2
        verified = True
        record("terminal_refusal", dict(error_type=type(caught.value).__name__, reason=caught.value.args,
            error_id=id(caught.value), closed=adapter._constructed.closed, sends=adapter._send_count,
            compression_sends=adapter._aux_send_count, seconds=time.monotonic()-started))
    finally:
        try:
            if world is not None:
                for runtime_name, current in (("qualification", qualifier), ("native", None if world.adapter is None else world.adapter._runtime)):
                    if current is not None:
                        record(runtime_name + "_retained_final", dict(graph=graph(current), charges=charges(current),
                            envelopes=[asdict(row.envelope) for row in current._envelopes.values()],
                            hops=[lineage.hops for lineage in current._lineages.values()], verified=verified,
                            main_release=result is not None, compression_release=summary is not None))
                for name, port in (("main", world.ports["main"]), ("compression", world.compression_ports["compression"])):
                    record(name + "_observations", [dict(envelope=asdict(row.envelope), output=row.output,
                        error_type=None if row.error is None else type(row.error).__name__, error_id=id(row.error),
                        transport_receipt=row.transport_receipt, started_at=row.started_at, finished_at=row.finished_at)
                        for row in port.observations])
        finally:
            cleanup_errors = []
            for name, receiver in (("a", a), ("b", b)):
                try:
                    if name not in receipts and receiver.thread is not None:
                        receipts[name] = receiver.finish()
                        record(name + "_partial_final", asdict(receipts[name]))
                except BaseException as error:
                    cleanup_errors.append(error)
                finally:
                    try:
                        receiver.close()
                    except BaseException as error:
                        cleanup_errors.append(error)
            record("cleanup", dict(a_closed=a.listener.fileno() == -1, b_closed=b.listener.fileno() == -1,
                a_stopped=a.thread is None or not a.thread.is_alive(), b_stopped=b.thread is None or not b.thread.is_alive(),
                a_sockets_closed=not a._connections, b_sockets_closed=not b._connections,
                a_errors=[type(error).__name__ for error in a.errors], b_errors=[type(error).__name__ for error in b.errors],
                cleanup_errors=[type(error).__name__ for error in cleanup_errors], seconds=time.monotonic()-started))
            assert not cleanup_errors
            assert not a.errors and not b.errors
            assert a.listener.fileno() == b.listener.fileno() == -1
            assert not a._connections and not b._connections
            assert a.thread is None or not a.thread.is_alive()
            assert b.thread is None or not b.thread.is_alive()
