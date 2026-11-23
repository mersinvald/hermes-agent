"""Verify the permitted source changes and the preserved API."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
base = "a951a7156c6da9beadd0dff6f0eed85554afd116"
owned = ["agent/native_main_payload.py", "agent/context_compressor.py", "run_agent.py",
         "agent/conversation_compression.py", "tests/agent/test_native_controlled_conversation.py",
         "tests/agent/test_content_policy_auxiliary.py", "tests/agent/test_context_compressor.py"]

def original(path):
    return subprocess.check_output(["git", "show", f"{base}:{path}"], cwd=root)


def classes(tree):
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def signatures(cls):
    return {node.name: ast.dump(node.args) for node in cls.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


pins = {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in owned}
old = ast.parse(original(owned[0]))
new = ast.parse((root / owned[0]).read_bytes())
old_classes, new_classes = classes(old), classes(new)
preserved_fields = {}
for name, cls in old_classes.items():
    old_fields = [ast.dump(node) for node in cls.body if isinstance(node, ast.AnnAssign)]
    new_fields = [ast.dump(node) for node in new_classes[name].body if isinstance(node, ast.AnnAssign)]
    assert old_fields == new_fields, name
    preserved_fields[name] = [node.target.id for node in cls.body if isinstance(node, ast.AnnAssign)]
    for method, signature in signatures(cls).items():
        assert signatures(new_classes[name])[method] == signature, (name, method)

old_compressor = classes(ast.parse(original(owned[1])))["ContextCompressor"]
new_compressor = classes(ast.parse((root / owned[1]).read_bytes()))["ContextCompressor"]
occurrences = {}
for previous in (node for node in old_compressor.body if isinstance(node, ast.FunctionDef)):
    name, signature = previous.name, ast.dump(previous.args)
    index = occurrences.get(name, 0)
    occurrences[name] = index + 1
    actual = [node for node in new_compressor.body if isinstance(node, ast.FunctionDef) and node.name == name][index]
    args = actual.args
    if name in ("__init__", "_generate_summary"):
        expected = "native_main_adapter" if name == "__init__" else "controlled_aux"
        assert len(args.kwonlyargs) == 1 and args.kwonlyargs[0].arg == expected
        assert len(args.kw_defaults) == 1 and isinstance(args.kw_defaults[0], ast.Constant)
        assert args.kw_defaults[0].value is None
        args.kwonlyargs, args.kw_defaults = [], []
    assert ast.dump(args) == signature, name

for path, owner, name in (("run_agent.py", "AIAgent", "_compress_context"),
                           ("agent/conversation_compression.py", None, "compress_context")):
    before, after = ast.parse(original(path)), ast.parse((root / path).read_bytes())
    def find(tree):
        body = classes(tree)[owner].body if owner else tree.body
        return next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name)
    before_fn, after_fn = find(before), find(after)
    assert isinstance(after_fn.body[1], ast.ImportFrom)
    assert after_fn.body[1].module == "agent.content_policy"
    assert after_fn.body[2].value.func.id == "guard_native_content"
    del after_fn.body[1:3]
    assert ast.dump(before) == ast.dump(after), path

for path in owned[4:]:
    assert (root / path).read_bytes().startswith(original(path)), path
changed = subprocess.check_output(["git", "diff", "--name-only", base], cwd=root).decode().splitlines()
assert all(path in owned or path.startswith("docs/evidence/accountant-native-retained-compression-20261123/")
           for path in changed), changed
api = dict(
    base=base, source_sha256=pins, old_record_fields=preserved_fields,
    old_adapter_signatures_preserved=True, old_compressor_positional_signatures_preserved=True,
    old_tests_preserved_byte_for_byte=True, host_changes_only_early_generic_guards=True,
    bind_compression=signatures(new_classes["NativeMainPayloadAdapter"])["bind_compression"],
    compression_input=signatures(new_classes["NativeMainPayloadAdapter"])["compression_input"],
    closed_types={name: [node.target.id for node in new_classes[name].body if isinstance(node, ast.AnnAssign)]
                  for name in ("_CompressionBinding", "_CompressionFrame", "_CompressionPreparation")},
)
print(json.dumps(api, indent=2))
