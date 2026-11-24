"""Audit the existing filtered view without native imports or sample collection."""

import argparse
import ast
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import stat
import sysconfig


ORIGINAL_SHA256 = "67c02424f6d7685a03e8de15c4ba1ca62579f047bdc034f057b2d34e2f92f7c4"
PUBLIC_SHA256 = "407446fd5d531e3bc6767c93c22385f81f0486531f857a7cb2935d3153d166c6"
QUALIFICATION_KEYS = frozenset(("usable", "collector_numeric_usability", "post_collection_scope_validation"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def audit(private_original, worktree):
    """Compare ordered observations separately from historical qualification fields."""
    public_path = worktree / "docs/evidence/accountant-native-anchor-sampling-20261124/sample.json"
    plugin_path = worktree / "tests/native_anchor_sampler_plugin.py"
    assert stat.S_IMODE(private_original.stat().st_mode) == 0o600
    assert stat.S_IMODE(private_original.parent.stat().st_mode) == 0o700
    private_bytes, public_bytes = private_original.read_bytes(), public_path.read_bytes()
    assert digest(private_bytes) == ORIGINAL_SHA256
    assert digest(public_bytes) == PUBLIC_SHA256
    original, public = json.loads(private_bytes), json.loads(public_bytes)
    plugin_bytes = plugin_path.read_bytes()
    tree = ast.parse(plugin_bytes)
    helper = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "_descriptor_filename")
    allowlist = next(node.value.args[0] for node in tree.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "PUBLIC_STDLIB"
                             for target in node.targets))
    namespace = {"Path": Path, "PUBLIC_STDLIB": frozenset(ast.literal_eval(allowlist))}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(plugin_path), "exec"), namespace)
    classifier = namespace["_descriptor_filename"]
    stdlib = Path(sysconfig.get_path("stdlib")).resolve(strict=True)
    worktree = worktree.resolve(strict=True)
    expected = {key: copy.deepcopy(value) for key, value in original.items()
                if key not in QUALIFICATION_KEYS}
    redacted = set()
    for descriptor in expected["code_descriptors"]:
        filename = descriptor["filename"]
        if filename == "<external>":
            assert descriptor["qualname"] == "<external>" and descriptor["first_line"] is None
            continue
        if filename.startswith("stdlib/"):
            path = stdlib / filename.removeprefix("stdlib/")
        else:
            assert not Path(filename).is_absolute()
            path = worktree / filename
        canonical = path.resolve(strict=True)
        if canonical.is_relative_to(worktree):
            independently_expected = canonical.relative_to(worktree).as_posix()
        elif canonical.is_relative_to(stdlib) and canonical.relative_to(stdlib).as_posix() in namespace["PUBLIC_STDLIB"]:
            independently_expected = "stdlib/" + canonical.relative_to(stdlib).as_posix()
        else:
            independently_expected = "<external>"
        classified = classifier(str(path), worktree, stdlib)
        assert classified == independently_expected
        if classified == "<external>":
            redacted.add(descriptor["code"])
            descriptor.update(filename="<external>", qualname="<external>", first_line=None)
        else:
            assert classified == filename
    assert len(redacted) == 27
    coordinates = {}
    for key in ("top_positions", "all_positions"):
        changed = 0
        for row in expected[key]:
            if row[0] in redacted:
                changed += int(row[1] is not None)
                row[1] = None
        counts = Counter((row[0], row[1]) for row in expected[key])
        coordinates[key] = dict(rows=len(expected[key]), redacted_current_lines=changed,
                                duplicate_coordinates_preserved=sum(n-1 for n in counts.values()))
    example_changes = 0
    for example in expected["example_stacks"]:
        for row in example["frames"]:
            if row[0] in redacted:
                example_changes += int(row[1] is not None)
                row[1] = None
    public_observations = {key: value for key, value in public.items() if key not in QUALIFICATION_KEYS}
    assert expected == public_observations
    assert set(public)-set(original) == {"collector_numeric_usability", "post_collection_scope_validation"}
    assert original["usable"] is True and public["usable"] is False
    assert public["collector_numeric_usability"] is original["usable"]
    assert public["post_collection_scope_validation"]["original_profile_sha256"] == ORIGINAL_SHA256
    assert public["post_collection_scope_validation"]["outcome"] == "failed"
    assert digest(private_original.read_bytes()) == ORIGINAL_SHA256
    assert digest(public_path.read_bytes()) == PUBLIC_SHA256
    return dict(original_sha256=ORIGINAL_SHA256, public_sha256=PUBLIC_SHA256,
                plugin_sha256=digest(plugin_bytes), audit_sha256=digest(Path(__file__).read_bytes()),
                result="PASS", descriptor_redactions=len(redacted), redacted_code_ids=sorted(redacted),
                coordinates=coordinates, example_current_line_redactions=example_changes,
                observation_payload_exact=True, code_ids_counts_depths_timestamps_sizes_exact=True,
                native_positions_and_example_order_exact=True, historical_qualification_fields_unchanged=True,
                private_modes_verified=True, existing_public_view_reused_by_hash=True,
                native_runtime_imports=0, new_sample_windows=0,
                qualification=dict(original_capture="FAILED", filtered_observation_view="PASS",
                                   corrected_collector="unit qualification recorded separately",
                                   corrected_live_capture="UNEXECUTED",
                                   original_functional_history="PASSED once at the original pin",
                                   original_task_and_budget="FAILED and unchanged"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.private_original, Path(__file__).resolve().parents[4])
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(result=result["result"], descriptor_redactions=result["descriptor_redactions"])))


if __name__ == "__main__":
    main()
