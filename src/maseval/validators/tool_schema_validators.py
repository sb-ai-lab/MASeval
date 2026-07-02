"""Validators for tool calls, function schemas, and structured outputs.

Detects malformed tool call arguments, missing required parameters, calls to unknown tools,
wrong model output formats, and invalid JSON schema outputs from tools.
"""

from __future__ import annotations

import json
import re

from maseval.validators.base import (
    DESERIALIZE,
    MAX_SPAN_TEXT,
    BaseValidator,
    CheckFn,
    Finding,
    Span,
    runtime_exception_pattern,
)

_JSON_LIKE = re.compile(r'^\s*(\{\s*["}]|\[\s*(["\[{\]]|-?\d|true|false|null))')

# Error/failure context cue. Phrases like "expected json" or "missing required
# field" appear just as often in a *task/prompt* ("return the expected json",
# "each row is missing a required field") as in a real failure. Rules prone to
# that ambiguity only fire when the span also shows error framing nearby: an
# exception name (matched via the "error"/"exception" substring — TypeError,
# ValidationError, JSONDecodeError, ...), a traceback, an HTTP/status context, or
# an explicit failure verb. The strict, self-evidently-error alternatives of each
# rule are kept separate and stay unconditional.
_ERROR_CTX = (
    r"(?:error|exception|traceback|stack ?trace|"
    r"raise[sd]?|threw|thrown|status[_ ]?code|\bhttp\b|"
    r"\bfailed\b|\bfailure\b|\bcould not\b)"
)


class ToolSchemaValidator(BaseValidator):
    EXPLANATIONS = {
        # check_malformed_tool_call
        "malformed_tool_call_validator": "The span text reports a tool call that could not be parsed or executed.",
        "tool_invalid_arguments_json": "The tool/function arguments were not valid JSON or could not be parsed.",
        "tool_schema_validation_error": "The tool call failed schema validation (arguments do not match the tool schema).",
        "tool_argument_type_error": "A tool argument had the wrong type (one type expected, another supplied).",
        "tool_null_parameter": "A required structured parameter was null where a value was expected.",
        "tool_call_type_error": "The span text reports a TypeError/AttributeError raised during a tool call.",
        "tool_call_runtime_error": "The span text reports a runtime exception (NameError, KeyError, "
        "ValueError, IndexError, UnboundLocalError, ...) raised while executing a tool/code call.",
        "tool_unexpected_none": "The tool returned None where a concrete value/structure was expected.",
        "tool_returned_error": "A tool call returned a structured failure result (success=false, a "
        "non-empty error field, or failed results) instead of a usable value.",
        # check_missing_argument
        "missing_required_argument_validator": "The span text reports a tool call with a missing or unexpected parameter.",
        "tool_missing_required_argument": "The tool call is missing a required argument, so it cannot run as intended.",
        "tool_unexpected_argument": "The tool call passed a keyword argument the tool does not accept.",
        # check_unknown_tool
        "unknown_tool_validator": "The span text reports a call to a tool/function that does not exist.",
        "tool_not_found": "The agent called a tool/function that is not registered or does not exist.",
        "tool_not_callable": "The referenced tool/function exists but is not callable.",
        "tool_not_implemented": "The tool raised NotImplementedError / the feature is not implemented.",
        # check_wrong_output_format
        "wrong_output_format_validator": "The span text reports a model response in the wrong output format.",
        "output_expected_json": "JSON output was expected but the model produced text that failed to parse.",
        "output_format_violation": "The response did not match the expected/required output format.",
        "output_deserialization_error": "A tool output could not be deserialized into the expected shape.",
        # check_tool_output_schema
        "tool_output_schema_validator": "JSON-shaped tool output that does not parse into valid JSON.",
        "tool_output_invalid_json": "The tool output looks like JSON but does not parse, so its schema cannot be trusted.",
    }

    def get_checks(self) -> list[CheckFn]:
        return [
            self.check_malformed_tool_call,
            self.check_missing_argument,
            self.check_unknown_tool,
            self.check_wrong_output_format,
            self.check_tool_output_schema,
        ]

    def check_malformed_tool_call(self, spans: list[Span]) -> list[Finding]:
        rules = [
            (
                r"invalid (json|arguments)[^.\n]{0,30}(tool|function|arguments)|"
                r"failed to parse (tool|function) (call|arguments)",
                "tool_invalid_arguments_json",
            ),
            (
                r"schema validation (failed|error)|"
                r"(?<!raise )(?<!except )(?<!class )(?<!def )\bvalidationerror\b\s*[:(]|"
                # "N validation errors for X" — but NOT for a provider response
                # object (ChatCompletion...), which is a ProviderValidator concern.
                r"\d+ validation errors? for\b(?!\s*(?:chatcompletion|completion|chatcompletionchunk)\b)|"
                r"does not match (the )?schema",
                "tool_schema_validation_error",
            ),
            (
                r"argument[^.\n]{0,30}(?:type|expected)[^.\n]{0,30}(?:got|received)",
                "tool_argument_type_error",
            ),
            (
                r"null[^.\n]{0,25}(parameter|argument|value)[^.\n]{0,25}expected|"
                r"expected[^.\n]{0,25}(parameter|argument)[^.\n]{0,25}(got|received) null",
                "tool_null_parameter",
            ),
            (
                runtime_exception_pattern("TypeError|AttributeError"),
                "tool_call_type_error",
            ),
            (
                runtime_exception_pattern(
                    "NameError|UnboundLocalError|KeyError|ValueError|IndexError|"
                    "ZeroDivisionError|RecursionError|OverflowError|RuntimeError|"
                    "InterpreterError|AgentExecutionError|ToolExecutionError|Exception"
                ),
                "tool_call_runtime_error",
            ),
            (
                r"returned none instead of|got none[^.\n]{0,15}expected|"
                r"none[^.\n]{0,15}not subscriptable",
                "tool_unexpected_none",
            ),
            (
                # Structured tool-failure result: success=false, a failed count,
                # a non-empty failed_results array, or an error field whose value
                # starts with a failure word (catches errors reported as tool
                # output JSON rather than a raised exception).
                r'"success"\s*:\s*false|"failed_count"\s*:\s*[1-9]|'
                r'"failed_results"\s*:\s*\[\s*\{|'
                r'"error"\s*:\s*"(?:failed|invalid|could not|unable|cannot|no such|'
                r'not found|timed? ?out|denied|refused|error|exception|missing)[^"]*"',
                "tool_returned_error",
            ),
        ]
        return self._scan(spans, "malformed_tool_call_validator", rules)

    def check_missing_argument(self, spans: list[Span]) -> list[Finding]:
        rules = [
            # Self-evidently-error forms fire unconditionally...
            (
                r"'[^']+' is a required property|"
                r"missing \d+ required (positional |keyword-only )?argument",
                "tool_missing_required_argument",
            ),
            # ...the generic phrase needs error framing (else it is task prose like
            # "each record is missing a required field"). Same failure_type, so at
            # most one fires per span.
            (
                r"missing required (argument|parameter|field|property)",
                "tool_missing_required_argument",
                _ERROR_CTX,
            ),
            (
                r"got an unexpected keyword argument|unexpected keyword argument",
                "tool_unexpected_argument",
            ),
        ]
        return self._scan(spans, "missing_required_argument_validator", rules)

    def check_unknown_tool(self, spans: list[Span]) -> list[Finding]:
        rules = [
            (
                r"\b(tool|function)\b[^.\n]{0,25}(not found|not registered|does not exist|"
                r"unknown|not available|not defined|not recognized)|"
                r"no (such )?\b(tool|function)\b (named|called|registered|available|found|exists)",
                "tool_not_found",
            ),
            (r"\b(tool|function)\b[^.\n]{0,15}not callable", "tool_not_callable"),
            (runtime_exception_pattern("NotImplementedError"), "tool_not_implemented"),
        ]
        return self._scan(spans, "unknown_tool_validator", rules)

    def check_wrong_output_format(self, spans: list[Span]) -> list[Finding]:
        rules = [
            # Self-evidently-error forms fire unconditionally...
            (
                r"could not (parse|decode) (the )?(model )?(output|response)|"
                r"jsondecodeerror",
                "output_expected_json",
            ),
            # ...the bare "expected json" phrase needs error framing (else it is task
            # prose like "return the expected json"). Same failure_type as above.
            (
                r"expected (valid )?json",
                "output_expected_json",
                _ERROR_CTX,
            ),
            (
                r"(output|response) format (error|violation|invalid)|"
                r"does not match (the )?(expected|required) format",
                "output_format_violation",
            ),
            (
                DESERIALIZE + r"[^.\n]{0,10}(?:error|fail)[^.\n]{0,25}tool[ _]?(?:output|result|response)|"
                r"tool[ _]?(?:output|result|response)[^.\n]{0,25}" + DESERIALIZE + r"[^.\n]{0,10}(?:error|fail)|"
                r"failed to " + DESERIALIZE + r"[^.\n]{0,25}tool[ _]?(?:output|result)",
                "output_deserialization_error",
            ),
        ]
        return self._scan(spans, "wrong_output_format_validator", rules)

    def check_tool_output_schema(self, spans: list[Span]) -> list[Finding]:
        findings: list[Finding] = []
        for span in spans:
            text = span.get("text", "")
            if not text or len(text) >= MAX_SPAN_TEXT or not _JSON_LIKE.match(text):
                continue
            try:
                json.JSONDecoder().raw_decode(text.lstrip())
            except (json.JSONDecodeError, ValueError):
                findings.append(
                    self._make_finding(
                        span,
                        metric_name="tool_output_schema_validator",
                        failure_type="tool_output_invalid_json",
                        explanation=self._explanation(
                            "tool_output_schema_validator", "tool_output_invalid_json"
                        ),
                    )
                )
        return findings
