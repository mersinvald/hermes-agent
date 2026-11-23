"""Compare prompt AST operations without importing native code."""

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
BASE = "a951a7156c6da9beadd0dff6f0eed85554afd116"
CANDIDATE = "1910b9bfb9eb7ef2144de9be96fcbe80060bcac2"
SOURCE = "agent/context_compressor.py"
ALIASES = ["summary_budget", "content_to_summarize", "_memory_section", "has_user_turn", "_today_str"]
TARGETS = {
    "_language_and_provenance_rule", "_historical_task_instructions", "_goal_instructions",
    "_constraints_instructions", "_resolved_questions_instructions", "_pending_asks_instructions",
    "_summarizer_preamble", "_temporal_anchoring_rule", "_session_log_section", "_template_sections",
    "prompt", "_memory_section", "_serialized_memory_context",
}
HELPERS = [
    "_serialize_for_summary", "_sample_summary_input", "_ground_historical_task_snapshot",
    "_augment_summary_lean", "_with_summary_prefix", "_bound_summary_input",
    "_validate_summary_user_provenance", "_transcript_has_real_user_turn",
    "_is_synthetic_compression_user_turn",
]


def source(commit):
    return subprocess.check_output(["git", "show", f"{commit}:{SOURCE}"], cwd=ROOT)


def compressor(data):
    return next(node for node in ast.parse(data).body
                if isinstance(node, ast.ClassDef) and node.name == "ContextCompressor")


def method(cls, name):
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)


def dump(node):
    return ast.dump(node, include_attributes=False)


def operations(node, kind):
    return Counter(dump(child) for child in ast.walk(node) if isinstance(child, kind))


def legacy_values(nodes):
    result = Counter()
    for node in nodes:
        for child in ast.walk(node):
            if isinstance(child, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in TARGETS for target in child.targets
            ):
                value = child.value
                if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name) and value.value.id == "facts":
                    continue
                result[dump(value)] += 1
            if isinstance(child, ast.AugAssign) and isinstance(child.target, ast.Name) and child.target.id == "prompt":
                result[dump(child.value)] += 1
    return result


before, after = source(BASE), source(CANDIDATE)
assert (ROOT / SOURCE).read_bytes() == after
old_cls, new_cls = compressor(before), compressor(after)
original = method(old_cls, "_generate_summary")
prompt, inputs = method(new_cls, "_summary_prompt"), method(new_cls, "_summary_source")
old_assign, new_assign = operations(original, ast.Assign), operations(prompt, ast.Assign)
old_aug, new_aug = operations(original, ast.AugAssign), operations(prompt, ast.AugAssign)
extra = new_assign - old_assign
aliases = [node.targets[0].id for node in ast.walk(prompt)
           if isinstance(node, ast.Assign) and dump(node) in extra]
assert aliases == ALIASES
assert sum(extra.values()) == len(ALIASES) == 5
assert sum((old_assign & new_assign).values()) == 21
assert new_aug == old_aug & new_aug and sum(new_aug.values()) == 1
assert new_assign - extra <= old_assign
legacy_before, legacy_after = legacy_values([original]), legacy_values([inputs, prompt])
assert legacy_before == legacy_after and sum(legacy_after.values()) == 24
legacy_inputs, legacy_prompt = legacy_values([inputs]), legacy_values([prompt])
helper_hashes = {}
for name in HELPERS:
    old, new = dump(method(old_cls, name)), dump(method(new_cls, name))
    assert old == new
    helper_hashes[name] = hashlib.sha256(new.encode()).hexdigest()

result = dict(
    python_version=sys.version.split()[0],
    base=BASE,
    source_candidate=CANDIDATE,
    source_path=SOURCE,
    base_source_sha256=hashlib.sha256(before).hexdigest(),
    candidate_source_sha256=hashlib.sha256(after).hexdigest(),
    original_Assign_operations_preserved=21,
    original_AugAssign_operations_preserved=1,
    new_fact_alias_assignments=5,
    new_fact_alias_names=ALIASES,
    all_non_alias_prompt_operations_identical=True,
    counting_definition=dict(
        original_method="ContextCompressor._generate_summary",
        current_method="ContextCompressor._summary_prompt",
        traversal="ast.walk on each complete FunctionDef",
        counter_key="ast.dump(node, include_attributes=False)",
        preserved="Sum of the Counter intersection for each node type separately",
        aliases="Current Assign Counter minus original Assign Counter",
    ),
    legacy_selected_value_expressions_preserved=24,
    legacy_counting_definition=dict(
        original_methods=["_generate_summary"],
        current_methods=["_summary_source", "_summary_prompt"],
        assign_targets=sorted(TARGETS),
        assign_key="ast.dump(node.value, include_attributes=False)",
        assign_exclusion="Exclude a value that is a Subscript of the Name facts",
        augassign_target="prompt",
        augassign_key="ast.dump(node.value, include_attributes=False)",
        equality="The original and current value-expression Counters must be equal",
        current_breakdown=dict(source_Assign_values=sum(legacy_inputs.values()),
                               prompt_Assign_values=sum(legacy_prompt.values()) - 1,
                               prompt_AugAssign_values=1),
        omitted_prompt_Assign_target="_bounded_previous_summary",
    ),
    unchanged_actual_helpers=HELPERS,
    helper_ast_sha256=helper_hashes,
    scope="Source/AST comparison only. Runtime compatibility checks remain separate.",
)
print(json.dumps(result, indent=2))
