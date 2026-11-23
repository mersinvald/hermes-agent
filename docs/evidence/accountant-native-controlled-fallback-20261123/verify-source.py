"""Verify the allowed source changes and freeze the consumer inputs."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
evidence = Path(__file__).resolve().parent
source = ["agent/native_main_payload.py", "agent/chat_completion_helpers.py",
          "agent/conversation_loop.py", "run_agent.py", "tests/agent/test_native_controlled_fallback.py"]
def digest(data):
    return hashlib.sha256(data).hexdigest()

def segment(text, node):
    return "".join(text.splitlines(keepends=True)[node.lineno-1:node.end_lineno])

def method(text, name, owner=None):
    tree = ast.parse(text)
    nodes = tree.body if owner is None else next(n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    return next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)

checks = {}
for path, name, owner in (("agent/chat_completion_helpers.py", "try_activate_fallback", None),
                          ("run_agent.py", "_try_activate_fallback", "AIAgent"),
                          ("agent/conversation_loop.py", "_try_controlled_text_turn", None)):
    old = subprocess.check_output(["git", "show", "a905202143476bdf2d0ed7d0a7115e373e8121d9:" + path], text=True)
    new = (root / path).read_text()
    before, after = segment(old, method(old, name, owner)), segment(new, method(new, name, owner))
    checks[path + ":only_selected_function"] = new.replace(after, before, 1) == old
    if name == "try_activate_fallback":
        marker = "    from agent.content_policy import guard_native_content\n"
        checks["ordinary_fallback_body_byte_identical"] = before[before.index(marker):] == after[after.index(marker):]
        checks["controlled_caller_byte_identical"] = segment(old, method(old, "_controlled_main_api_call")) == segment(new, method(new, "_controlled_main_api_call"))
    if owner:
        checks["native_constructor_byte_identical"] = segment(old, method(old, "__init__", owner)) == segment(new, method(new, "__init__", owner))
old = ast.parse(subprocess.check_output(["git", "show", "a905202143476bdf2d0ed7d0a7115e373e8121d9:agent/native_main_payload.py"], text=True))
new = ast.parse((root / source[0]).read_text())
def records(tree):
    return {n.name: [x.target.id for x in n.body if isinstance(x, ast.AnnAssign)] for n in tree.body if isinstance(n, ast.ClassDef)}
before, after = records(old), records(new)
checks["existing_record_fields_unchanged"] = all(after[k] == v for k, v in before.items())
allowed = set(source)
changed = subprocess.check_output(["git", "diff", "--name-only", "a905202143476bdf2d0ed7d0a7115e373e8121d9"], text=True).splitlines()
checks["tracked_allowlist"] = all(p in allowed or p.startswith("docs/evidence/accountant-native-controlled-fallback-20261123/") for p in changed)
inputs = json.loads((evidence / "input-verification.json").read_text())
for path, value in inputs["inputs"]["native_base_blobs"].items():
    if path not in source:
        checks[path + ":preserved"] = digest((root / path).read_bytes()) == value["sha256"]
for path, expected in inputs["protected_metadata"].items():
    stat = (root / path).stat()
    checks[path + ":metadata_preserved"] = expected == dict(inode=stat.st_ino, size=stat.st_size, mtime_ns=stat.st_mtime_ns, mode=stat.st_mode)
checks["wheel_preserved"] = digest(Path(inputs["inputs"]["artifact_handoff"]["wheel_path"]).read_bytes()) == inputs["inputs"]["artifact_handoff"]["wheel_sha256"]
api = dict(records={name: after[name] for name in ("_FallbackBinding", "_FailedSendObservation", "NativeFailedMainAttempt", "_FallbackFrame", "NativeFallbackInvocation")},
           tokens=["NativeFallbackInput"], methods={})
for name in ("bind_fallback", "_begin_fallback", "_activate_fallback", "_dispatch_fallback", "_complete_fallback_turn"):
    api["methods"][name] = ast.unparse(method((root / source[0]).read_text(), name, "NativeMainPayloadAdapter").args)
api["properties"] = {"failed_main_attempts": "tuple[NativeFailedMainAttempt, ...]", "fallback_invocations": "tuple[NativeFallbackInvocation, ...]", "fallback_send_count": "int"}
(evidence / "native-api.json").write_text(json.dumps(api, indent=2) + "\n")
(evidence / "source-verification.json").write_text(json.dumps(checks, indent=2) + "\n")
assert all(checks.values()), checks
frozen = dict(source_sha256={p: digest((root / p).read_bytes()) for p in source},
              api_sha256=digest((evidence / "native-api.json").read_bytes()),
              consumer_sha256=digest((evidence / "consumer-contract.md").read_bytes()))
(evidence / "frozen-source.json").write_text(json.dumps(frozen, indent=2) + "\n")
print(json.dumps(dict(checks=checks, frozen=frozen)))
