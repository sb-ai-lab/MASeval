from __future__ import annotations

import pytest

from maseval.validators import (
    ApiHttpValidator,
    EnvironmentSetupValidator,
    ProviderValidator,
    ToolSchemaValidator,
    detect_format,
    run_on_trace,
)


def _failure_types(findings) -> set[str]:
    return {f["failure_type"] for f in findings}


# format detection
def test_detect_format_who_and_when():
    trace = {
        "mistake_agent": "WebSurfer",
        "mistake_step": 3,
        "history": [
            {"role": "user", "content": "Find the capital of France."},
            {"role": "WebSurfer", "content": "Searching the web..."},
        ],
    }
    assert detect_format(trace) == "who_and_when"


def test_detect_format_trail():
    trace = {
        "spans": [
            {
                "span_id": "a1b2c3",
                "span_name": "CodeAgent.run",
                "span_kind": "CHAIN",
                "trace_state": "",
                "child_spans": [],
            }
        ]
    }
    assert detect_format(trace) == "trail"


# provider HTTP 429 rate limit
def test_provider_429_positive():
    span = {
        "idx": "s1",
        "text": (
            "openai.RateLimitError: Error code: 429 - "
            "{'error': {'message': 'Rate limit reached for gpt-4o; "
            "please try again in 20s', 'type': 'requests'}}"
        ),
        "agent": "Researcher",
    }
    findings = ProviderValidator().run([span])
    assert "provider_rate_limit" in _failure_types(findings)


def test_provider_429_not_from_prompt_negative():
    span = {
        "idx": "s1",
        "text": (
            "The marketing dashboard shows we served 429 customers today, "
            "up from 380 yesterday. Summarize the growth trend."
        ),
        "agent": "Analyst",
    }
    findings = ProviderValidator().run([span])
    assert "provider_rate_limit" not in _failure_types(findings)
    assert findings == []


# api "not found" as a plain agent phrase
@pytest.mark.xfail(
    reason=(
        "api_validators 'not found' regex is too broad: it matches plain prose "
        "like 'I have not found the answer' as an HTTP 404. Fix is to require a "
        "context cue (Traceback/Error/HTTP/status_code/404) — that lives in "
        "api_validators.py (Roma's file), which is intentionally not modified here. "
        "Flips to xpass once the cue is added."
    ),
    strict=False,
)
def test_api_not_found_negative_for_plain_text():
    span = {
        "idx": "s1",
        "text": "I searched every document but I have not found the answer to your question.",
        "agent": "Assistant",
    }
    findings = ApiHttpValidator().run([span])
    assert "api_not_found" not in _failure_types(findings)


# tool JSON-shaped output that does not parse
def test_tool_json_parse_positive():
    span = {
        "idx": "s1",
        "text": '{"status": "ok", "items": [1, 2, 3',
        "agent": "search_tool",
    }
    findings = ToolSchemaValidator().run([span])
    assert "tool_output_invalid_json" in _failure_types(findings)


# environment missing file
def test_environment_file_not_found_positive():
    span = {
        "idx": "s1",
        "text": (
            "Traceback (most recent call last):\n"
            "FileNotFoundError: [Errno 2] No such file or directory: "
            "'/data/config.yaml'"
        ),
        "agent": "loader",
    }
    findings = EnvironmentSetupValidator().run([span])
    assert "env_file_not_found" in _failure_types(findings)


# no false positives on an ordinary user task
def test_no_false_positive_on_user_task_text():
    task = (
        "You are a research assistant. Read the attached sales report, extract "
        "the quarterly revenue for each product line, and return the result as a "
        "JSON object. If a product is missing from the report, skip it. Then draft "
        "a short summary email to the finance team requesting their review."
    )
    trace = {
        "history": [
            {"role": "user", "content": task},
            {
                "role": "assistant",
                "content": "Understood. I will read the report and prepare the JSON.",
            },
        ]
    }
    result = run_on_trace(trace)
    total = sum(len(m["findings"]) for m in result["metrics"].values())
    assert total == 0, result["metrics"]
