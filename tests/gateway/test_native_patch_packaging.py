"""Exercise patch selection and archive composition without a registry."""
import importlib.util
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace


def test_base_layer_preserves_existing_title_modules(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "native_patch", Path(__file__).resolve().parents[2] / "docker/publish-native-patch.py")
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    # These modules already ship in the pinned upstream source layer. They must
    # be verified/replaced with their existing modes, never treated as additions.
    existing_titles = {"agent/auxiliary_client.py", "agent/title_generator.py"}
    existing = (set(patch.RUNTIME) - set(patch.NEW_RUNTIME)) | existing_titles
    existing |= {"plugins/platforms/a2a/security.py", "plugins/platforms/a2a/__init__.py", "pyproject.toml", "uv.lock"}
    entries = {"opt/hermes/" + name: ("base " + name).encode() for name in existing}
    entries["opt/hermes/.hermes_build_sha"] = patch.BASE_REVISION.encode()
    entries["etc/hermes/image-provenance.json"] = json.dumps({"revision": patch.BASE_REVISION}).encode()
    layer = io.BytesIO()
    with tarfile.open(fileobj=layer, mode="w:gz") as archive:
        for name, data in entries.items():
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(data), 0o755 if name.endswith("title_generator.py") else 0o644
            archive.addfile(member, io.BytesIO(data))
    monkeypatch.setattr(patch, "blob", lambda *args: layer.getvalue())
    monkeypatch.setattr(patch, "run", lambda *args: ("base " + args[2].split(":", 1)[1]).encode())
    monkeypatch.setattr(patch.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    config = {"history": [
        {"created_by": "COPY pyproject.toml uv.lock ./ # buildkit"},
        {"created_by": "COPY --chmod=a+rX,go-w . . # buildkit"},
    ]}
    files = patch.base_files("synthetic", {"layers": [{"digest": "a"}, {"digest": "b"}]}, config, tmp_path)
    for name in existing_titles:
        assert files["opt/hermes/" + name] == (entries["opt/hermes/" + name],
                                                0o755 if name.endswith("title_generator.py") else 0o644)


def test_conversation_runtime_is_selected_and_archived(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "native_patch", Path(__file__).resolve().parents[2] / "docker/publish-native-patch.py")
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    def git(*args, **kwargs):
        if args[:3] == ("git", "diff", "--name-only"):
            return "\n".join(patch.RUNTIME).encode()
        assert args[:2] == ("git", "show")
        return ("synthetic payload for " + args[2]).encode()
    monkeypatch.setattr(patch, "run", git)
    sources = patch.runtime_sources("revision")
    entries = {"opt/hermes/" + name: data for name, data in sources.items()}
    files = {name: (b"", 0o644) for name in entries}
    layer = tmp_path / "layer.tar"
    patch.write_layer(layer, entries, files, 1234)
    with tarfile.open(layer) as archive:
        for name in ("gateway/permission_grants.py", "plugins/platforms/telegram/grants.py", "hermes_cli/commands.py", "gateway/native_redirect.py", "gateway/native_clarification.py",
                     "hermes_state_controls.py", "hermes_state_clarifications.py", "tools/clarify_gateway.py",
                     "agent/native_execution_context.py", "agent/auxiliary_client.py", "agent/title_generator.py", "gateway/native_cancellation.py",
                     "hermes_state_cancellation.py", "plugins/platforms/a2a/cancellation.py",
                     "plugins/platforms/a2a/provenance.py", "tools/async_delegation.py", "tools/delegate_tool.py",
                     "gateway/conversation_control.py", "gateway/session_state.py", "gateway/session.py", "gateway/run.py",
                     "gateway/native_commands.py", "gateway/native_events.py", "hermes_state_events.py", "hermes_state_commands.py", "hermes_state.py",
                     "gateway/pwa_models.py", "hermes_state_models.py",
                     "gateway/pwa_images.py", "gateway/pwa_image_http.py",
                     "gateway/telegram_conversations.py", "hermes_state_delivery.py", "gateway/authz_mixin.py",
                     "gateway/platforms/base.py", "gateway/slash_commands.py",
                     "hermes_state_common.py", "agent/turn_context.py", "agent/conversation_loop.py",
                     "agent/tool_dispatch_helpers.py", "run_agent.py", "gateway/shutdown_flush.py"):
            member = archive.getmember("opt/hermes/" + name)
            assert archive.extractfile(member).read() == sources[name]
            assert member.mode == 0o644
            assert member.uid == member.gid == 0
