from __future__ import annotations

from maseval.validators import detect_format, to_spans
from maseval.validators.base import (
    MAX_SPAN_TEXT,
    _clean_pumpkin_agent,
    _trail_text,
    _ww_role_agent,
)


def test_detect_format_pumpkin():
    trace = {
        "projectId": "p1",
        "observations": [
            {"id": "o1", "name": "Planner run", "startTime": "2024-01-01T00:00:00Z"}
        ],
    }
    assert detect_format(trace) == "pumpkin"


def test_detect_format_pumpkin_via_observation_shape():
    trace = {
        "observations": [
            {"traceId": "t1", "startTime": "2024-01-01T00:00:00Z", "name": "run"}
        ]
    }
    assert detect_format(trace) == "pumpkin"


def test_detect_format_who_and_when_as_list():
    trace = [{"mistake_agent": "A", "mistake_step": 1, "role": "user", "content": "x"}]
    assert detect_format(trace) == "who_and_when"


def test_detect_format_unknown():
    assert detect_format({"foo": [{"a": 1}, {"b": 2}]}) == "unknown"
    assert detect_format(42) == "unknown"


def test_detect_format_trail_requires_span_markers():
    assert detect_format({"spans": [{"id": "1", "text": "hi"}]}) != "trail"


def test_trail_to_spans_walks_tree_with_parent_and_kind():
    trace = {
        "spans": [
            {
                "span_id": "root",
                "span_name": "CodeAgent",
                "span_kind": "CHAIN",
                "trace_state": "",
                "child_spans": [
                    {
                        "span_id": "child",
                        "span_name": "search",
                        "span_kind": "TOOL",
                        "child_spans": [],
                    }
                ],
            }
        ]
    }
    fmt, spans = to_spans(trace)
    assert fmt == "trail"
    by_id = {s["span_id"]: s for s in spans}
    assert set(by_id) == {"root", "child"}
    assert by_id["root"]["parent"] is None
    assert by_id["child"]["parent"] == "root"
    assert by_id["child"]["kind"] == "TOOL"


def test_trail_text_excludes_llm_output_but_keeps_tool_output():
    llm = {
        "span_name": "chat",
        "span_kind": "LLM",
        "span_attributes": {
            "openinference.span.kind": "LLM",
            "llm.output_messages.0.message.content": "RateLimitError: 429 too many requests",
            "input.value": "task prompt",
        },
    }
    tool = {
        "span_name": "lookup",
        "span_kind": "TOOL",
        "span_attributes": {
            "openinference.span.kind": "TOOL",
            "output.value": "KeyError: 'city'",
        },
    }
    assert "429" not in _trail_text(llm)  # generated prose is not evidence
    assert "task prompt" not in _trail_text(llm)  # re-fed input excluded
    assert "KeyError" in _trail_text(tool)  # real tool failure kept


def test_who_and_when_native_roles_attributed():
    fmt, spans = to_spans(
        {"history": [{"role": "WebSurfer", "content": "searching"}, {"role": "user", "content": "q"}]}
    )
    assert fmt == "who_and_when"
    assert spans[0]["agent"] == "WebSurfer"
    assert spans[1]["agent"] is None  # a chat role is not an agent name


def test_who_and_when_chat_only_gaia_has_no_agents():
    fmt, spans = to_spans(
        {"history": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
    )
    assert all(s["agent"] is None for s in spans)


def test_ww_role_agent_collapses_parenthetical_and_drops_chat_roles():
    assert _ww_role_agent("Orchestrator (thought)") == "Orchestrator"
    assert _ww_role_agent("Orchestrator (-> WebSurfer)") == "Orchestrator"
    assert _ww_role_agent("user") is None
    assert _ww_role_agent("human") is None
    assert _ww_role_agent(None) is None


def test_clean_pumpkin_agent_drops_util_spans():
    assert _clean_pumpkin_agent("Planner run") == "Planner"
    assert _clean_pumpkin_agent("Chat completion") is None
    assert _clean_pumpkin_agent("generation") is None
    assert _clean_pumpkin_agent("WebSurfer") == "WebSurfer"


def test_span_text_truncated_to_max():
    big = "x" * (MAX_SPAN_TEXT + 500)
    _, spans = to_spans({"history": [{"role": "user", "content": big}]})
    assert len(spans[0]["text"]) == MAX_SPAN_TEXT
