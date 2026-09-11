"""Reject dependency drift and unreviewed SDK bytes before image publication."""

import importlib.util
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "native_observability_layer", ROOT / "docker/native_observability_layer.py",
)
layer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layer)
BASE = "bb40fc99e564e63f9435ce1df92fd166775892ad"


def manifests():
    before = lambda path: subprocess.check_output(["git", "show", BASE + ":" + path], cwd=ROOT)
    return [before("pyproject.toml"), (ROOT / "pyproject.toml").read_bytes(),
            before("uv.lock"), (ROOT / "uv.lock").read_bytes()]


def test_optional_sdk_is_the_only_dependency_change():
    layer.dependency_guard(*manifests())


@pytest.mark.parametrize("index,before,after", [
    (1, b"langfuse==4.15.2", b"langfuse==4.15.3"),
    (1, b"opentelemetry-sdk==1.39.1", b"opentelemetry-sdk==1.40.0"),
    (3, b'name = "backoff"\nversion = "2.2.1"', b'name = "backoff"\nversion = "2.3.0"'),
    (3, b"98c27a3c06e18c4497045f2d4decce2c716ef11cb215cf5b27bbea6ee0877115", b"0" * 64),
])
def test_unreviewed_dependency_or_wheel_change_rejected(index, before, after):
    values = manifests()
    assert before in values[index]
    values[index] = values[index].replace(before, after)
    with pytest.raises(AssertionError):
        layer.dependency_guard(*values)


def test_unverified_wheel_bytes_rejected(tmp_path):
    filename, _ = layer.WHEELS["langfuse"]
    (tmp_path / filename).write_bytes(b"not the reviewed wheel")
    with pytest.raises(AssertionError, match="wheel hash mismatch"):
        layer.wheel_entries(tmp_path, "amd64")


def test_wheel_path_symlink_rejected(tmp_path):
    filename, _ = layer.WHEELS["langfuse"]
    target = tmp_path / "different-file"
    target.write_bytes(b"not a wheel")
    (tmp_path / filename).symlink_to(target)
    with pytest.raises(AssertionError, match="regular reviewed wheel"):
        layer.wheel_entries(tmp_path, "arm64")


def test_sdk_guard_is_loaded_from_committed_bytes(monkeypatch):
    publisher_spec = importlib.util.spec_from_file_location(
        "native_patch_publisher", ROOT / "docker/publish-native-patch.py",
    )
    publisher = importlib.util.module_from_spec(publisher_spec)
    publisher_spec.loader.exec_module(publisher)
    committed = b'SDK = "committed-only"\n'
    calls = []

    def git(*args):
        calls.append(args)
        return committed

    monkeypatch.setattr(publisher, "run", git)
    support, digest = publisher.committed_sdk_support("a" * 40)
    assert support.SDK == "committed-only"
    assert digest == publisher.sha(committed)
    assert calls == [("git", "show", "a" * 40 + ":docker/native_observability_layer.py")]
