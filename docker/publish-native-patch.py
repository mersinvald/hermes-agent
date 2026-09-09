#!/usr/bin/env python3
"""Publish a pinned native source-only patch without a Docker daemon.

Requires Python 3.11+, crane 0.21.3 and registry auth in an externally supplied
DOCKER_CONFIG. Credentials are never build inputs. Use --output outside the
checkout. Without --publish this verifies the base and generates layers only.
The reviewed runtime allowlist deliberately excludes dependency/build changes.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

BASE_REVISION = "29112bef099274229cadff79cdff7bf7b99c4b77"
BASE = "docker.io/nousresearch/hermes-agent@sha256:64923faeae267792bf9bf87fe3b4c4869e35004e360c7df01730ad801b74d524"
RUNTIME = ("plugins/platforms/a2a/tools.py", "plugins/platforms/a2a/protocol.py",
           "plugins/platforms/a2a/streaming.py", "agent/tool_executor.py",
           "agent/agent_runtime_helpers.py", "model_tools.py", "gateway/run.py",
           "gateway/session_context.py", "gateway/permission_bridge.py",
           "gateway/session_state.py", "gateway/session.py", "gateway/conversation_control.py",
           "gateway/native_commands.py", "gateway/shutdown_flush.py", "hermes_state_commands.py",
           "gateway/pwa_config.py", "gateway/pwa_ownership.py", "gateway/pwa_http.py", "hermes_state_pwa.py",
           "gateway/config.py", "hermes_cli/config_defaults.py",
           "run_agent.py", "agent/tool_dispatch_helpers.py",
           "hermes_state.py", "hermes_state_common.py", "agent/turn_context.py", "agent/conversation_loop.py",
           "tools/approval.py", "tools/mcp_tool.py", "plugins/platforms/telegram/adapter.py")
NEW_RUNTIME = ("plugins/platforms/a2a/streaming.py", "gateway/permission_bridge.py",
               "gateway/conversation_control.py", "gateway/native_commands.py", "hermes_state_commands.py",
               "gateway/pwa_config.py", "gateway/pwa_ownership.py", "gateway/pwa_http.py", "hermes_state_pwa.py")
ROOT = Path(__file__).resolve().parents[1]


def run(*args, timeout=300):
    return subprocess.check_output(args, cwd=ROOT, timeout=timeout)


def document(*args):
    return json.loads(run("crane", *args))


def sha(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def blob(repository, digest, cache):
    path = cache / digest.replace(":", "-")
    if not path.exists():
        path.write_bytes(run("crane", "blob", repository + "@" + digest))
    data = path.read_bytes()
    assert sha(data) == digest, "corrupt cached registry blob"
    return data


def base_files(repository, manifest, config, cache):
    history = [item for item in config["history"] if not item.get("empty_layer")]
    assert len(history) == len(manifest["layers"])
    starts = [i for i, item in enumerate(history) if item.get("created_by") == "COPY --chmod=a+rX,go-w . . # buildkit"]
    assert len(starts) == 1, "unrecognized native source layout"
    manifests = [i for i, item in enumerate(history) if item.get("created_by") == "COPY pyproject.toml uv.lock ./ # buildkit"]
    assert len(manifests) == 1 and manifests[0] < starts[0]
    wanted = {"opt/hermes/" + name for name in (*(name for name in RUNTIME if name not in NEW_RUNTIME),
        "plugins/platforms/a2a/security.py", "plugins/platforms/a2a/__init__.py", "pyproject.toml", "uv.lock")}
    wanted |= {"opt/hermes/.hermes_build_sha", "etc/hermes/image-provenance.json"}
    found = {}
    for layer in manifest["layers"][manifests[0]:]:
        data = blob(repository, layer["digest"], cache)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for member in archive:
                name = member.name.removeprefix("./").rstrip("/")
                assert name not in {"opt/hermes/" + path for path in NEW_RUNTIME}, "new runtime file exists in base"
                assert ".wh." not in name, "unexpected whiteout after native source copy"
                if any(path.startswith(name + "/") for path in wanted):
                    assert member.isdir(), "non-directory native source parent"
                if name in wanted:
                    assert member.isfile() and member.uid == 0 and member.gid == 0
                    assert member.mode & 0o022 == 0 and member.mode & 0o004
                    found[name] = (archive.extractfile(member).read(), member.mode)
    assert set(found) == wanted, "missing native source/provenance files: " + str(sorted(wanted - set(found)))
    for name in wanted - {"opt/hermes/.hermes_build_sha", "etc/hermes/image-provenance.json"}:
        assert found[name][0] == run("git", "show", BASE_REVISION + ":" + name.removeprefix("opt/hermes/")), name
    assert found["opt/hermes/.hermes_build_sha"][0].decode().strip() == BASE_REVISION
    assert json.loads(found["etc/hermes/image-provenance.json"][0])["revision"] == BASE_REVISION
    for name in NEW_RUNTIME:
        assert subprocess.run(["git", "cat-file", "-e", BASE_REVISION + ":" + name],
                              cwd=ROOT, capture_output=True).returncode != 0
        found["opt/hermes/" + name] = (b"", 0o644)
    return found


def runtime_sources(revision):
    changed = run("git", "diff", "--name-only", BASE_REVISION, revision).decode().splitlines()
    assert all(name in changed for name in RUNTIME)
    assert all(name in RUNTIME or name.startswith("tests/") or name in (
        "docker/publish-native-patch.py", "plugins/platforms/a2a/PROGRESS.md", "cli-config.yaml.example",
        "gateway/CONVERSATION_CONTROL.md") for name in changed), "not a source-only patch"
    # Always read committed blobs, never the dirty or case-colliding host checkout.
    runtime = {name: run("git", "show", revision + ":" + name) for name in changed if name in RUNTIME}
    return runtime


def write_layer(layer, entries, files, epoch):
    with tarfile.open(layer, "w", format=tarfile.USTAR_FORMAT) as archive:
        for name, data in sorted(entries.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), files[name][1], epoch
            info.uid = info.gid = 0
            archive.addfile(info, io.BytesIO(data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, help="Destination registry/repository, without a tag")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    assert run("crane", "version").decode().strip() == "0.21.3", "use pinned crane 0.21.3"
    revision = run("git", "rev-parse", "HEAD").decode().strip()
    runtime = runtime_sources(revision)
    epoch = int(run("git", "show", "-s", "--format=%ct", revision))
    output = args.output.resolve()
    assert not output.is_relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "blobs"
    cache.mkdir(exist_ok=True)
    base_repository = BASE.split("@")[0]
    index = document("manifest", BASE)
    children = {entry["platform"]["architecture"]: entry for entry in index["manifests"]
        if entry["platform"].get("os") == "linux"}
    assert set(children) == {"amd64", "arm64"}
    tag = "2026.8.31-mkl-" + revision[:12] + "-candidate"
    report = {"revision": revision, "base": BASE, "base_revision": BASE_REVISION,
        "source_sha256": {name: sha(data) for name, data in runtime.items()}, "tag": tag, "architectures": {}}
    targets = []
    for arch in ("arm64", "amd64"):
        print("Verifying native base and composing " + arch, flush=True)
        base_ref = base_repository + "@" + children[arch]["digest"]
        manifest = document("manifest", base_ref)
        config = document("config", base_ref)
        assert config["architecture"] == arch and config["os"] == "linux"
        assert config["config"]["Labels"]["org.opencontainers.image.revision"] == BASE_REVISION
        files = base_files(base_repository, manifest, config, cache)
        provenance = json.loads(files["etc/hermes/image-provenance.json"][0])
        provenance.update(image=args.repository, revision=revision, base_image=BASE,
            base_manifest=children[arch]["digest"], base_revision=BASE_REVISION,
            source_sha256=report["source_sha256"], distribution="native-source-patch")
        entries = {**{"opt/hermes/" + name: data for name, data in runtime.items()}, "opt/hermes/.hermes_build_sha": (revision + "\n").encode(),
            "etc/hermes/image-provenance.json": (json.dumps(provenance, sort_keys=True, separators=(",", ":")) + "\n").encode()}
        layer = output / (arch + ".tar")
        write_layer(layer, entries, files, epoch)
        layer_hash = sha(layer.read_bytes())
        target = args.repository + ":" + tag + "-" + arch
        result = {"base_manifest": children[arch]["digest"], "layer_diff_id": layer_hash,
            "files": {name: {"sha256": sha(data), "mode": oct(files[name][1]), "uid": 0, "gid": 0}
                      for name, data in entries.items()}, "reference": target}
        if args.publish:
            existing = subprocess.run(["crane", "digest", target], capture_output=True)
            assert existing.returncode != 0, "refusing to overwrite existing candidate tag"
            assert b"MANIFEST_UNKNOWN" in existing.stderr or b"404" in existing.stderr, "cannot establish tag absence"
            labels = {"org.opencontainers.image.revision": revision,
                "org.opencontainers.image.source": "https://github.com/mersinvald/hermes-agent",
                "org.opencontainers.image.base.name": BASE,
                "org.opencontainers.image.base.digest": children[arch]["digest"],
                "dev.mkl.patch.diff-id": layer_hash,
                "dev.mkl.patch.source-manifest-sha256": sha(json.dumps(report["source_sha256"], sort_keys=True).encode())}
            command = ["crane", "mutate", base_ref, "--append", str(layer), "--tag", target]
            for key, value in labels.items():
                command += ["--label", key + "=" + value]
            subprocess.run(command, check=True, timeout=600)
            digest = run("crane", "digest", target).decode().strip()
            pinned = args.repository + "@" + digest
            actual, actual_config = document("manifest", pinned), document("config", pinned)
            assert actual["layers"][:-1] == manifest["layers"], "upstream layers changed"
            assert actual_config["rootfs"]["diff_ids"] == config["rootfs"]["diff_ids"] + [layer_hash]
            assert actual_config["architecture"] == arch
            expected_settings = dict(config["config"])
            expected_settings["Labels"] = {**expected_settings.get("Labels", {}), **labels}
            assert actual_config["config"] == expected_settings, "runtime configuration changed"
            appended = blob(args.repository, actual["layers"][-1]["digest"], cache)
            with tarfile.open(fileobj=io.BytesIO(appended), mode="r:gz") as archive:
                assert set(archive.getnames()) == set(entries)
                for name, data in entries.items():
                    assert archive.extractfile(name).read() == data
            subprocess.run(["crane", "validate", "--fast", "--remote", pinned], check=True, timeout=60)
            result["digest"] = digest
            targets.append(pinned)
        report["architectures"][arch] = result
        (output / "publication.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"architecture": arch, **result}), flush=True)
    if args.publish:
        target = args.repository + ":" + tag
        existing = subprocess.run(["crane", "digest", target], capture_output=True)
        assert existing.returncode != 0 and (b"MANIFEST_UNKNOWN" in existing.stderr or b"404" in existing.stderr)
        subprocess.run(["crane", "index", "append", "-m", targets[0], "-m", targets[1], "-t", target], check=True, timeout=120)
        report["index_digest"] = run("crane", "digest", target).decode().strip()
        actual = document("manifest", args.repository + "@" + report["index_digest"])
        assert {m["platform"]["architecture"]: m["digest"] for m in actual["manifests"]} == {
            arch: result["digest"] for arch, result in report["architectures"].items()}
        (output / "publication.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"index": target, "digest": report["index_digest"]}), flush=True)


if __name__ == "__main__":
    main()
