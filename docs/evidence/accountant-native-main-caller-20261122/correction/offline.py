"""Use invented configuration and block network callbacks for this selection."""

import os
from pathlib import Path
import socket
import tempfile

import httpx


def pytest_configure(config):
    home = tempfile.TemporaryDirectory(prefix="hpwa-p10.7-correction-")
    config.add_cleanup(home.cleanup)
    (Path(home.name) / "config.yaml").write_text(
        "model: invented_model\nmemory:\n  enabled: false\n"
    )
    os.environ["HERMES_HOME"] = home.name

    def blocked(*args, **kwargs):
        raise AssertionError("Network callback ran")

    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked
    httpx.Client.send = blocked
    httpx.AsyncClient.send = blocked
