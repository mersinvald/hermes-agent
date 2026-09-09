"""Exercise patch selection and archive composition without a registry."""
import importlib.util
import tarfile
from pathlib import Path


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
        for name in ("gateway/conversation_control.py", "gateway/session_state.py", "gateway/session.py", "gateway/run.py",
                     "gateway/native_commands.py", "hermes_state_commands.py", "hermes_state.py",
                     "gateway/telegram_conversations.py", "hermes_state_delivery.py", "gateway/authz_mixin.py",
                     "gateway/platforms/base.py", "gateway/slash_commands.py",
                     "hermes_state_common.py", "agent/turn_context.py", "agent/conversation_loop.py",
                     "agent/tool_dispatch_helpers.py", "run_agent.py", "gateway/shutdown_flush.py"):
            member = archive.getmember("opt/hermes/" + name)
            assert archive.extractfile(member).read() == sources[name]
            assert member.mode == 0o644
            assert member.uid == member.gid == 0
