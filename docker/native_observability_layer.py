"""Closed, hash-pinned Langfuse dependency addition to the reviewed native base.

The normal publisher remains source-only. Its explicit observability mode uses
this guard to permit only the reviewed optional extra and these three packages.
Existing package versions, dependency metadata and runtime settings stay fixed.
"""

from __future__ import annotations

import hashlib
import io
from email.parser import BytesParser
from pathlib import PurePosixPath
import stat
import tarfile
import tomllib
import zipfile

SDK = "4.15.2"
SITE = "opt/hermes/.venv/lib/python3.13/site-packages/"
EXTRA = ["langfuse==4.15.2", "opentelemetry-sdk==1.39.1",
         "opentelemetry-exporter-otlp-proto-http==1.39.1"]
CUTOFF = "2026-09-10T00:00:00Z"
WHEELS = {
    "langfuse": (
        "langfuse-4.15.2-py3-none-any.whl",
        "98c27a3c06e18c4497045f2d4decce2c716ef11cb215cf5b27bbea6ee0877115",
    ),
    "backoff": (
        "backoff-2.2.1-py3-none-any.whl",
        "63579f9a0628e06278f7e47b7d7d5b6ce20dc65c5e96a6f3ca99a6adca0396e8",
    ),
    "wrapt-amd64": (
        "wrapt-1.17.3-cp313-cp313-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl",
        "6fd1ad24dc235e4ab88cda009e19bf347aabb975e44fd5c2fb22a3f6e4141277",
    ),
    "wrapt-arm64": (
        "wrapt-1.17.3-cp313-cp313-manylinux2014_aarch64.manylinux_2_17_aarch64.manylinux_2_28_aarch64.whl",
        "0ed61b7c2d49cee3c027372df5809a59d60cf1b6c2f81ee980a091f3afed6a2d",
    ),
}
DEPENDENCY_LAYERS = {
    "amd64": "sha256:abe9d4e60393e44fe6e9f79136d807895b7465f831d75fb39ecfd891d8d5c8f8",
    "arm64": "sha256:0d1e55605218384192a602d91e4ffb35155789a4642f93f15c80b3011bc54d83",
}
EXISTING = {
    "annotated-types": "0.7.0", "anyio": "4.12.1", "certifi": "2026.5.20",
    "charset-normalizer": "3.4.4", "googleapis-common-protos": "1.73.0",
    "h11": "0.16.0", "httpcore": "1.0.9", "httpx": "0.28.1", "idna": "3.18",
    "importlib-metadata": "8.7.1", "opentelemetry-api": "1.39.1",
    "opentelemetry-exporter-otlp-proto-common": "1.39.1",
    "opentelemetry-exporter-otlp-proto-http": "1.39.1", "opentelemetry-proto": "1.39.1",
    "opentelemetry-sdk": "1.39.1", "opentelemetry-semantic-conventions": "0.60b1",
    "packaging": "26.0", "protobuf": "6.33.5", "pydantic": "2.13.4",
    "pydantic-core": "2.46.4", "requests": "2.33.0", "typing-extensions": "4.15.0",
    "typing-inspection": "0.4.2", "urllib3": "2.7.0", "zipp": "3.23.0",
}


def verify_base_dependencies(manifest, architecture, read_blob, added_entries):
    """Read exact runtime METADATA and validate the active Linux dependency closure."""
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    layers = manifest["layers"]
    starts = [index for index, item in enumerate(layers)
              if item["digest"] == DEPENDENCY_LAYERS[architecture]]
    assert len(starts) == 1, "unreviewed base dependency installation"
    metadata = {}
    for descriptor in layers[starts[0]:]:
        with tarfile.open(fileobj=io.BytesIO(read_blob(descriptor["digest"])), mode="r:gz") as archive:
            for member in archive:
                path = member.name.removeprefix("./").rstrip("/")
                assert path not in added_entries, "SDK path already exists in base"
                assert not (".wh." in path and (
                    path.startswith(SITE) or SITE.startswith(str(PurePosixPath(path).parent) + "/")
                )), "dependency whiteout requires new review"
                if not path.startswith(SITE):
                    continue
                parts = path[len(SITE):].split("/")
                if len(parts) != 2 or not parts[0].endswith(".dist-info") or parts[1] != "METADATA":
                    continue
                assert member.isfile() and member.size < 1024 * 1024
                value = BytesParser().parsebytes(archive.extractfile(member).read())
                name = canonicalize_name(value["Name"])
                if name in EXISTING:
                    metadata[name] = value
    assert {name: value["Version"] for name, value in metadata.items()} == EXISTING
    for path, raw in added_entries.items():
        if path.endswith(".dist-info/METADATA"):
            value = BytesParser().parsebytes(raw)
            name = canonicalize_name(value["Name"])
            assert name not in metadata
            metadata[name] = value
    assert set(metadata) == set(EXISTING) | {"langfuse", "backoff", "wrapt"}
    environment = {**default_environment(), "python_version": "3.13", "python_full_version": "3.13.0",
                   "sys_platform": "linux", "os_name": "posix", "platform_system": "Linux",
                   "platform_machine": "x86_64" if architecture == "amd64" else "aarch64",
                   "platform_python_implementation": "CPython", "implementation_name": "cpython", "extra": ""}
    checked, pending = set(), ["langfuse"]
    while pending:
        name = pending.pop()
        if name in checked:
            continue
        checked.add(name)
        for text in metadata[name].get_all("Requires-Dist", []):
            requirement = Requirement(text)
            if requirement.marker and not requirement.marker.evaluate(environment):
                continue
            dependency = canonicalize_name(requirement.name)
            assert not requirement.extras and requirement.url is None
            assert dependency in metadata, "missing runtime dependency: " + dependency
            assert requirement.specifier.contains(metadata[dependency]["Version"])
            pending.append(dependency)
    return {name: metadata[name]["Version"] for name in sorted(checked)}


def dependency_guard(before_project, after_project, before_lock, after_lock):
    """Compare parsed manifests, permitting exactly one pinned opt-in extra."""
    old = tomllib.loads(before_project.decode())
    new = tomllib.loads(after_project.decode())
    assert new["project"]["optional-dependencies"].pop("langfuse") == EXTRA
    assert new["tool"]["uv"]["exclude-newer-package"].pop("langfuse") == CUTOFF
    assert new == old, "unreviewed project/dependency change"

    old = tomllib.loads(before_lock.decode())
    new = tomllib.loads(after_lock.decode())
    assert new["options"]["exclude-newer-package"].pop("langfuse") == CUTOFF
    added = [package for package in new["package"] if package["name"] == "langfuse"]
    assert len(added) == 1 and added[0]["version"] == SDK
    assert added[0]["source"] == {"registry": "https://pypi.org/simple"}
    assert len(added[0]["wheels"]) == 1
    assert added[0]["wheels"][0]["hash"] == "sha256:" + WHEELS["langfuse"][1]
    assert {row["name"] for row in added[0]["dependencies"]} == {
        "backoff", "httpx", "opentelemetry-api", "opentelemetry-exporter-otlp-proto-http",
        "opentelemetry-sdk", "packaging", "pydantic", "typing-extensions", "wrapt",
    }
    new["package"].remove(added[0])
    project = next(package for package in new["package"] if package["name"] == "hermes-agent")
    assert project["optional-dependencies"].pop("langfuse") == [
        {"name": "langfuse"}, {"name": "opentelemetry-exporter-otlp-proto-http"},
        {"name": "opentelemetry-sdk"},
    ]
    metadata = project["metadata"]
    requirements = [row for row in metadata["requires-dist"]
                    if row.get("marker") == "extra == 'langfuse'"]
    assert requirements == [
        {"name": "langfuse", "marker": "extra == 'langfuse'", "specifier": "==4.15.2"},
        {"name": "opentelemetry-exporter-otlp-proto-http", "marker": "extra == 'langfuse'", "specifier": "==1.39.1"},
        {"name": "opentelemetry-sdk", "marker": "extra == 'langfuse'", "specifier": "==1.39.1"},
    ]
    metadata["requires-dist"] = [row for row in metadata["requires-dist"] if row not in requirements]
    metadata["provides-extras"].remove("langfuse")
    assert new == old, "unreviewed existing locked dependency change"


def wheel_entries(directory, architecture):
    """Return image paths/bytes; never execute a wheel or install at startup."""
    assert architecture in {"amd64", "arm64"}
    entries, wheels = {}, {}
    for package in ("langfuse", "backoff", "wrapt-" + architecture):
        filename, digest = WHEELS[package]
        path = directory / filename
        assert path.is_file() and not path.is_symlink(), "regular reviewed wheel required"
        assert path.stat().st_size <= 4 * 1024 * 1024
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest, "wheel hash mismatch: " + filename
        name = package.split("-")[0]
        wheels[filename] = "sha256:" + digest
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = archive.infolist()
            assert len(members) <= 2000
            assert sum(member.file_size for member in members) <= 32 * 1024 * 1024
            assert len({member.filename for member in members}) == len(members)
            for member in members:
                relative = PurePosixPath(member.filename)
                assert not relative.is_absolute() and ".." not in relative.parts
                assert "\\" not in member.filename and relative.parts
                assert relative.parts[0] == name or (
                    relative.parts[0].startswith(name + "-")
                    and relative.parts[0].endswith(".dist-info")
                ), "wheel writes outside reviewed package"
                mode = member.external_attr >> 16
                assert not stat.S_ISLNK(mode)
                if member.is_dir():
                    continue
                target = SITE + str(relative)
                assert target not in entries
                entries[target] = archive.read(member)
    return entries, wheels
