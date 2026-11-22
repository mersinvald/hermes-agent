"""Actual A2A send refuses sensitive work before discovery and transport."""

from dataclasses import replace

import pytest

from agent.content_policy import NativeContentPolicyDenied, content_policy_scope
from plugins.platforms.a2a import tools
from tests.agent.test_content_policy import make_context


@pytest.mark.parametrize("streaming,follow", [(False, False), (True, False), (False, True)])
def test_sensitive_peer_send_refuses_before_discovery_redaction_or_http(monkeypatch, streaming, follow):
    context, state = make_context()
    calls = []
    monkeypatch.setattr(tools, "_http_post_json", lambda *args, **kwargs: calls.append("post"))
    monkeypatch.setattr(tools, "_http_get_json", lambda *args, **kwargs: calls.append("discovery"))
    with pytest.raises(NativeContentPolicyDenied):
        with content_policy_scope(context):
            state["binding"] = replace(state["binding"], profiles=("restricted-documents",))
            tools._send_task("document-review", {"url": "http://invented.invalid", "streaming": streaming},
                             "invented document", "", follow=follow)
            pytest.fail("A2A send returned")
    assert calls == []
