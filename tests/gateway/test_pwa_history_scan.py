"""Literal display search and opaque operation-state boundary behavior."""

import pytest

from gateway.pwa_history_scan import ScanHandles, literal_query, literal_snippet
from hermes_state_commands import CommandConflict


@pytest.mark.parametrize(
    "query", ["\x00word", "word\n", "\tword", "\ud800", "", "   ", "x" * 257]
)
def test_literal_query_rejects_invalid_bounded_input(query):
    with pytest.raises(ValueError, match="invalid history query"):
        literal_query(query)


@pytest.mark.parametrize(
    "query", ['"word"', "OR", "%_() *", "x' OR 1=1 --", "a\\b", "Straße"]
)
def test_search_metacharacters_are_literals(query):
    assert literal_query("  " + query + "  ") == query
    assert (
        literal_snippet("prefix " + query + " suffix", query)["text"]
        == "prefix " + query + " suffix"
    )
    assert literal_snippet("unrelated display", query) is None


@pytest.mark.parametrize(
    "text,query",
    [
        ("Straße", "STRASSE"),
        ("İstanbul", "i\u0307s"),
        ("ﬃ", "ffi"),
        ("ß", "s"),
        ("Σςσ", "σσσ"),
    ],
)
def test_casefold_expansion_returns_real_source_window(text, query):
    source = "a" * 700 + text + "z" * 700
    snippet = literal_snippet(source, query)
    assert len(snippet["text"]) == 500
    assert text in snippet["text"]
    assert snippet["text"] in source
    assert snippet["prefix_omitted"] and snippet["suffix_omitted"]


def test_snippet_at_boundaries_and_no_unicode_normalization():
    assert literal_snippet("ß" + "x" * 700, "ss") == {
        "text": "ß" + "x" * 499,
        "prefix_omitted": False,
        "suffix_omitted": True,
        "match_truncated": False,
    }
    assert literal_snippet("x" * 700 + "ß", "ss") == {
        "text": "x" * 499 + "ß",
        "prefix_omitted": True,
        "suffix_omitted": False,
        "match_truncated": False,
    }
    assert literal_snippet("café", "cafe\u0301") is None


@pytest.mark.parametrize(
    "prefix,suffix", [("", ""), ("", "end"), ("start", ""), ("a" * 700, "z" * 700)]
)
@pytest.mark.parametrize(
    "match,query",
    [("ffi" * 256, "ﬃ" * 256), ("ß" + "ffi" * 254 + "ﬃ", "s" + "ﬃ" * 254 + "f")],
)
def test_long_fold_expansion_has_explicit_partial_source_match(
    prefix, suffix, match, query
):
    assert len(query) == 256
    source = prefix + match + suffix
    snippet = literal_snippet(source, query)
    assert snippet["match_truncated"]
    assert 0 < len(snippet["text"]) <= 500
    assert snippet["text"] in source
    assert snippet["text"].casefold() in query.casefold()
    assert snippet["text"].casefold() != query.casefold()
    assert snippet["prefix_omitted"] or snippet["suffix_omitted"]


def test_handle_retry_scoping_immutability_and_absolute_expiry():
    now = [100]
    handles = ScanHandles(ttl=10, clock=lambda: now[0])
    scope = ["owner", "configured-source", "search", "literal", 20]
    state = {"position": ["root", 4], "generation": 0}
    token = handles.put(scope, state)
    assert handles.put(scope, state) == token
    state["position"][1] = 99
    returned = handles.get(token, scope)
    assert returned["position"] == ["root", 4]
    returned["position"][1] = 88
    assert handles.get(token, scope)["position"] == ["root", 4]
    for changed in (
        ["foreign", *scope[1:]],
        [*scope[:-1], 21],
        [*scope[:2], "sync"],
        [],
    ):
        with pytest.raises(CommandConflict):
            handles.get(token, changed)
    with pytest.raises(CommandConflict):
        handles.get(token[:-1] + ("a" if token[-1] != "a" else "b"), scope)
    now[0] = 109
    assert handles.put(scope, {"position": ["root", 4], "generation": 0}) == token
    now[0] = 110
    with pytest.raises(CommandConflict):
        handles.get(token, scope)
    replacement = handles.put(scope, {"position": ["root", 4], "generation": 0})
    assert replacement != token
    with pytest.raises(CommandConflict):
        handles.get(token, scope)


def test_handle_count_byte_eviction_restart_and_generation_privacy():
    handles = ScanHandles(max_entries=2, max_bytes=1024, max_state_bytes=1024)
    first = handles.put(["scope"], {"position": 1})
    second = handles.put(["scope"], {"position": 2})
    third = handles.put(["scope"], {"position": 3})
    with pytest.raises(CommandConflict):
        handles.get(first, ["scope"])
    handles.get(second, ["scope"])
    huge = handles.put(["scope"], {"payload": "x" * 1000})
    with pytest.raises(CommandConflict):
        handles.get(third, ["scope"])
    with pytest.raises(ValueError):
        handles.put(["scope"], {"payload": "x" * 1024})
    fresh = ScanHandles()
    with pytest.raises(CommandConflict):
        fresh.get(huge, ["scope"])
    assert handles.lineage_version(["root"], 1) != handles.lineage_version(["root"], 2)
    assert handles.lineage_version(["root"], 1) != fresh.lineage_version(["root"], 1)
    handles.close()
    with pytest.raises(CommandConflict):
        handles.get(huge, ["scope"])
