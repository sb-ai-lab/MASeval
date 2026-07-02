from __future__ import annotations

from maseval.validators import ToolSchemaValidator


def _types(text: str) -> set[str]:
    span = {"span_id": "s", "text": text, "agent": None}
    return {f["failure_type"] for f in ToolSchemaValidator().run([span])}


def test_tool_not_found():
    assert "tool_not_found" in _types("ToolException: tool `weather` is not registered")


def test_tool_not_implemented():
    assert "tool_not_implemented" in _types("NotImplementedError: subclasses must implement run()")


def test_unexpected_keyword_argument():
    assert "tool_unexpected_argument" in _types(
        "TypeError: search() got an unexpected keyword argument 'limit'"
    )


def test_missing_required_positional_argument():
    assert "tool_missing_required_argument" in _types(
        "TypeError: run() missing 1 required positional argument: 'query'"
    )


def test_is_a_required_property_fires_unconditionally():
    assert "tool_missing_required_argument" in _types("'name' is a required property")


def test_runtime_keyerror_flagged():
    assert "tool_call_runtime_error" in _types("Traceback ... KeyError: 'city' during tool run")


def test_type_error_flagged():
    assert "tool_call_type_error" in _types(
        "AttributeError: 'NoneType' object has no attribute 'get'"
    )


def test_source_raise_not_flagged_as_runtime():
    # Edited source that merely *raises* / *handles* an exception is not a crash.
    assert _types("raise ValueError('bad input')") == set()
    assert _types("except KeyError:\n    return None") == set()


def test_docstring_raises_not_flagged():
    assert _types("Raises KeyError if the key is absent from the mapping.") == set()


def test_valid_json_output_not_flagged():
    assert _types('{"ok": true, "items": [1, 2, 3]}') == set()


def test_broken_json_output_flagged():
    assert "tool_output_invalid_json" in _types('{"ok": true, "items": [1, 2, 3')


def test_non_json_text_not_flagged():
    assert _types("The measured temperature is 42 degrees.") == set()


def test_expected_json_prose_not_flagged_without_error_context():
    assert _types("Please return the expected JSON object with two keys.") == set()


def test_expected_json_flagged_with_error_context():
    assert "output_expected_json" in _types("Failed to parse the model output: expected valid JSON")
