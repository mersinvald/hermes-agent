"""Native tool-result admission before conversion and canonical append."""

from types import SimpleNamespace

import pytest

from agent.content_policy import NativeContentPolicyDenied, guard_native_content
from agent.tool_executor import execute_tool_calls_concurrent, execute_tool_calls_sequential
from run_agent import AIAgent
from tests.agent.test_content_policy import enabled_agent, make_context


@pytest.mark.parametrize("execute", [execute_tool_calls_concurrent, execute_tool_calls_sequential])
@pytest.mark.parametrize("profiles,result_profiles", [(("restricted-documents",), None), ((), None),
                                                       ((), lambda _: ("restricted-documents",))])
def test_executors_reject_unknown_or_sensitive_producers_before_dispatch_and_append(execute, profiles, result_profiles):
    context, _ = make_context(profiles=profiles, result_profiles=result_profiles)
    agent = enabled_agent(context)
    assistant = SimpleNamespace(tool_calls=[SimpleNamespace(function=SimpleNamespace(name="document-review"))])
    messages = []
    with pytest.raises(NativeContentPolicyDenied):
        execute(agent, assistant, messages, "invented-task")
    assert messages == []
    with pytest.raises(NativeContentPolicyDenied):
        guard_native_content("main", agent)


def test_unknown_result_denies_before_string_or_image_conversion():
    context, _ = make_context()
    agent = enabled_agent(context)

    class Result:
        def __str__(self):
            pytest.fail("result converted")

    with pytest.raises(NativeContentPolicyDenied):
        AIAgent._tool_result_content_for_active_model(agent, "document-review", Result())


def test_explicit_ordinary_producer_preserves_native_string_result():
    context, _ = make_context(result_profiles=lambda _: ())
    value = "invented ordinary document"
    assert AIAgent._tool_result_content_for_active_model(enabled_agent(context), "document-review", value) is value
