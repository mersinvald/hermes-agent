"""Build synthetic namespace models without authority or runtime admission."""

import json
import runpy
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

from agent import native_main_payload as native

HERE = Path(__file__).resolve().parent


def load_prototype():
    return runpy.run_path(str(HERE / "prototype.py"), init_globals=vars(native).copy())


def record_types():
    policy = native.shared_policy()
    return (native.NativeMainResponse, native.NativeMainChoice, native.NativeMainMessage,
            native.NativeMainUsage, native.NativeMainSelection, native.NativeResultPolicy,
            policy.ExecutionBinding, policy.Destination, policy.DispatchEnvelope, policy.RouteBinding,
            native.NativeAuxiliaryInvocation, native.NativeAuxiliaryLocalResult, native._AuxiliaryTitlePreparation,
            native._CompressionBinding, native._AuxiliaryBinding, native._CompressionPreparation,
            native._FallbackBinding, native._FailedSendObservation, native.NativeFailedMainAttempt,
            native._FallbackFrame, native.NativeFallbackInvocation, native._TurnFrame, native._PendingOperation,
            native.NativeDelegateBinding, native.NativeDelegateInvocation, policy.ProfilePolicy,
            policy.DispatchFailureFacts)


def make_record(cls, **values):
    return cls(**{field.name: values.get(field.name, object()) for field in fields(cls)})


def snapshot(obj):
    pairs = tuple(vars(obj).items())
    return obj, pairs, native._nested_anchors(tuple(value for _, value in pairs))[1:]


def build_fixture(kind):
    names = json.loads((HERE / "namespace-mapping.json").read_text())["namespaces"][kind]
    policy = native.shared_policy()
    destination = policy.Destination("synthetic_transport", "synthetic_provider", "synthetic_model", "synthetic_endpoint", 1)
    selection = native.NativeMainSelection("synthetic_selection", 1, "chat_completions", destination)
    producer = native.NativeResultPolicy("synthetic_producer", 1, ("synthetic_profile",))
    holder = policy.ExecutionBinding("owner", "execution", "lineage", "consumer", 1, 1, "selection", 1, producer.profiles)
    routes = tuple(policy.RouteBinding(f"route_{i}", "selection", 1, "consumer", path, destination, 100,
                                      "synthetic_qualification", "synthetic_only")
                   for i in range(3) for path in ("main", "delegate", "title", "compression"))
    profiles = tuple(policy.ProfilePolicy(f"profile_{i}", tuple(row.route_ref for row in routes),
                                        ("main", "delegate", "title", "compression")) for i in range(3))
    port, method = object(), object()
    methods = tuple((port, f"method_{i}", (port, method)) for i in range(16))
    catalog = routes, profiles, (("synthetic_transport", port),)
    originals = ((holder, selection, producer),) * 3
    initial = tuple((native._nested_anchors(row), row) for row in originals)
    snapshots = originals, holder, catalog, methods, methods
    binding = make_record(native.NativeDelegateBinding, snapshots=snapshots,
                          anchors=native._nested_anchors(snapshots), delegate_route=routes[1],
                          child_main_route=routes[0], child_session_id="synthetic_session", child_task_id="synthetic_task")
    message = native.NativeMainMessage("assistant", "Synthetic reply")
    response = native.NativeMainResponse("synthetic_response", "chat.completion", 1, "synthetic_model",
                                        (native.NativeMainChoice(0, message, "stop"),), native.NativeMainUsage(3, 2, 5))
    envelope = policy.DispatchEnvelope(b"synthetic_request", holder, producer.profiles, "consumer", "main", destination)
    arguments = (("goal", True, "Synthetic goal"), ("context", True, None), ("role", True, "leaf")), (), ()
    invocation = make_record(native.NativeDelegateInvocation, binding=binding, arguments=arguments,
                             argument_anchors=native._nested_anchors(arguments), raw_bytes=b"synthetic_input",
                             envelope=envelope, child_invocation=response, return_facts=(holder, envelope, response))
    closed_proof = (binding, invocation, native._nested_anchors(invocation), methods, initial)
    values = dict(catalog=catalog, catalog_anchor=native._nested_anchors(catalog),
                  catalog_anchors=native._nested_anchors(catalog), methods=methods, extra_methods=methods,
                  methods_anchor=methods, execution_methods=methods, execution_method_anchor=methods,
                  initial=initial, originals=originals, initial_anchor=native._nested_anchors(initial),
                  initial_anchors=native._nested_anchors(initial), binding=binding, registered_binding=binding,
                  binding_anchor=tuple(getattr(binding, field.name) for field in fields(binding)),
                  snapshot_anchor=native._nested_anchors(snapshots), record=invocation, registered_record=invocation,
                  record_anchor=tuple(getattr(invocation, field.name) for field in fields(invocation)),
                  record_nested=native._nested_anchors(invocation), closed_proof=closed_proof,
                  result_anchors=((invocation, native._nested_anchors(invocation)),),
                  arguments=arguments, options=arguments, option_anchors=native._nested_anchors(arguments),
                  partial_operations=((envelope, holder, producer),),
                  partial_operation_anchors=native._nested_anchors((envelope, holder, producer)),
                  capabilities=(port, port, port), capability_anchor=(port, port, port),
                  selections=(selection, selection, selection), selection_anchor=(selection, selection, selection),
                  producers=(producer, producer, producer), routes=routes, profile_union=producer.profiles,
                  constructor_options=dict(provider="synthetic_native", model="synthetic_model"),
                  leaf=dict(goal="Synthetic goal", context=None, role="leaf"),
                  decoded=dict(goal="Synthetic goal", context=None, role="leaf"),
                  plan_anchor=native._nested_anchors((response, selection, producer)),
                  phase="released", phase_anchor="released", terminal=True, running=False, checking=False,
                  inputs=((b"synthetic_input", port),), outputs=((b"synthetic_reply", port),))
    obj = SimpleNamespace(**{name: values.get(name, object()) for name in names})
    return snapshot(obj)
