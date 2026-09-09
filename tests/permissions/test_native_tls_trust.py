"""Real subprocess/default urllib TLS; no custom client context or fake opener."""
import datetime
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import ssl
import subprocess
import sys
import threading

import certifi
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from gateway import permission_bridge


@pytest.mark.parametrize("case", ["merged", "wrong_ca", "untrusted", "hostname", "http"])
def test_native_default_trust_subprocess(pytestconfig, tmp_path, monkeypatch, case):
    source = pytestconfig.getoption("--permissions-source")
    if not source:
        pytest.skip("Current permission service source required")
    monkeypatch.syspath_prepend(str(Path(source) / "src"))
    from mcp_permissions.ledger import Ledger
    from mcp_permissions.server import HumanServer, human_handler

    now = datetime.datetime.now(datetime.timezone.utc)
    root_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    root_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "inert-private-ca")])
    root = (x509.CertificateBuilder().subject_name(root_name).issuer_name(root_name)
            .public_key(root_key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=1)).not_valid_after(now + datetime.timedelta(hours=1))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(root_key.public_key()), critical=False)
            .add_extension(x509.KeyUsage(False, False, False, False, False, True, True, False, False), critical=True)
            .sign(root_key, hashes.SHA256()))
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "inert-service")])
    leaf = (x509.CertificateBuilder().subject_name(name).issuer_name(root_name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=1)).not_valid_after(now + datetime.timedelta(hours=1))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(root_key.public_key()), critical=False)
            .sign(root_key, hashes.SHA256()))
    cert_path, key_path = tmp_path / "server.pem", tmp_path / "key.pem"
    cert_path.write_bytes(leaf.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    public = Path(certifi.where()).read_bytes()
    defaults = ssl.create_default_context().get_ca_certs(binary_form=True)
    public += b"\n" + b"\n".join(ssl.DER_cert_to_PEM_cert(c).encode() for c in defaults)
    bundle = tmp_path / "merged.pem"
    bundle.write_bytes(public + b"\n" + root.public_bytes(serialization.Encoding.PEM))
    public_only = tmp_path / "public.pem"
    public_only.write_bytes(public)
    config = json.loads((Path(source) / "fixture-config.json").read_text())
    ledger = Ledger(tmp_path / "ledger.sqlite", config)
    contract = config["contracts"][0]
    actor, owner = next(iter(config["actors"].items()))
    human = config["owners"][owner]
    params = json.dumps({"name": contract["tool"], "arguments": {"resource_id": "fixture-a", "value": 3}}).encode()
    ref = ledger.check(actor, [contract["backend"]], "tools/call", params)["request_id"]
    token = "synthetic-human-only-tls-test-credential-0000"
    server = HumanServer(("127.0.0.1", 0), human_handler(ledger, token))
    if case != "http":
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert_path, key_path)
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    code = r'''
import hashlib, json, os, ssl, sys
from gateway.permission_bridge import RemotePermission, _http, BridgeError
from telegram.request import HTTPXRequest
cfg = json.loads(sys.argv[1])
def fingerprints(ctx):
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname
    return {hashlib.sha256(c).hexdigest() for c in ctx.get_ca_certs(binary_form=True)}
default = fingerprints(ssl.create_default_context())
assert set(cfg['public']) <= default
telegram = HTTPXRequest()
assert set(cfg['public']) <= fingerprints(telegram._client._transport._pool._ssl_context)
r = RemotePermission(cfg['ref'], cfg['actor'], cfg['owner'], cfg['user'], cfg['chat'], cfg['url'])
try:
    details = _http(r, cfg['user'], cfg['chat'])
except BridgeError:
    assert cfg['case'] != 'merged'
else:
    assert cfg['case'] == 'merged' and details['request_id'] == cfg['ref']
    result = _http(r, cfg['user'], cfg['chat'], {'digest': details['digest'], 'choice': 'once'})
    assert result['state'] == 'approved'
print('verified ' + cfg['case'])
'''
    public_certs = ssl.create_default_context(cafile=str(public_only)).get_ca_certs(binary_form=True)
    child = {"case": case, "ref": ref, "actor": actor, "owner": owner,
             "user": human["telegram_user_id"], "chat": human["telegram_chat_id"],
             "public": [hashlib.sha256(c).hexdigest() for c in public_certs],
             "url": f"{'http' if case == 'http' else 'https'}://{'localhost' if case == 'hostname' else '127.0.0.1'}:{server.server_port}"}
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "HERMES_HOME": str(tmp_path),
           "MCP_PERMISSIONS_HUMAN_TOKEN": token, "PYTHONDONTWRITEBYTECODE": "1",
           "SSL_CERT_FILE": str(public_only if case in ("wrong_ca", "untrusted") else bundle),
           "HTTPS_PROXY": "http://127.0.0.1:1", "HTTP_PROXY": "http://127.0.0.1:1", "NO_PROXY": ""}
    try:
        result = subprocess.run([sys.executable, "-c", code, json.dumps(child)], env=env,
                                cwd=Path(permission_bridge.__file__).resolve().parents[1], capture_output=True, text=True, timeout=15)
        assert token not in result.stdout + result.stderr
        assert result.returncode == 0, result.stderr
        assert ledger.inspect(owner, ref)["state"] == ("approved" if case == "merged" else "pending")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
