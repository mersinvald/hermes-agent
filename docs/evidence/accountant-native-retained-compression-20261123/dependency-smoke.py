"""Check the exact installed dependency and one invented shared derivation."""

import hashlib
import importlib.metadata
import inspect
import json
from dataclasses import replace
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root))

import hermes_content_policy as policy
from agent import native_main_payload
from tests.agent.test_native_controlled_conversation import CompressionWorld

wheel = Path("/home/mersinvald/.local/share/orca/runtime/accountant/content-policy-artifacts/hpwa-p10.13/hermes_content_policy-0.2.0-py3-none-any.whl")
engine = Path(policy.__file__).resolve()
wheel_hash = hashlib.sha256(wheel.read_bytes()).hexdigest()
engine_hash = hashlib.sha256(engine.read_bytes()).hexdigest()
assert wheel_hash == "b6dd72e900cea284b16f0aee0cb706824ce7e495176fb826c96fb508bfea5e6a"
assert engine_hash == "576e057d0b59053a254a5a766509526824e14640081fdb31ba58b1b03517a625"
assert importlib.metadata.version("hermes-content-policy") == "0.2.0"
assert Path(native_main_payload.__file__).resolve() == root / "agent/native_main_payload.py"
assert engine.is_relative_to(root / ".venv")
world = CompressionWorld()
runtime = policy.ContentPolicyRuntime(
    admission=world.admission, profiles=world.profiles, routes=world.routes,
    transports=tuple((port.destination.transport_ref, port) for port in world.ports.values()),
    clock=lambda: world.now, policy_revision=9,
)
source = runtime.admit(world.request, b"Invented shared smoke source")
bridge = runtime.derive_for_selection(
    world.compression_request, (source,), b"Invented shared smoke request",
    path="compression", parent_requests=(world.request,),
)
result = runtime.dispatch(world.compression_request, bridge, path="compression", route_ref="compression_compression")
world.binding = replace(world.binding, profiles=())
world.compression_binding = replace(world.compression_binding, profiles=())
returned = runtime.derive_for_selection(
    world.request, (result,), b"Invented shared return", path="main",
    parent_requests=(world.compression_request,),
)
assert returned is not bridge and result is not source
try:
    runtime.derive(world.request, (result,), b"Unqualified same-selection claim", path="main")
except policy.PolicyError:
    pass
else:
    raise AssertionError("Ordinary derivation accepted a foreign selection")
print(json.dumps(dict(
    python=sys.version.split()[0], version=importlib.metadata.version("hermes-content-policy"),
    wheel_sha256=wheel_hash, engine_sha256=engine_hash,
    installed_engine=str(engine), native_import=str(Path(native_main_payload.__file__).resolve()),
    derive_for_selection=str(inspect.signature(policy.ContentPolicyRuntime.derive_for_selection)),
    native_bind_compression=str(inspect.signature(native_main_payload.NativeMainPayloadAdapter.bind_compression)),
    native_compression_input=str(inspect.signature(native_main_payload.NativeMainPayloadAdapter.compression_input)),
    representative="shared bridge, dispatch, profile removal, return, and ordinary derivation refusal passed",
    scope="Shared API/import/representative smoke with native source imports. No native installed functional selection.",
), indent=2))
