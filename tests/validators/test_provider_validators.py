from __future__ import annotations

from maseval.validators import ProviderValidator


def _types(text: str) -> set[str]:
    span = {"span_id": "s", "text": text, "agent": None}
    return {f["failure_type"] for f in ProviderValidator().run([span])}


def test_unauthorized_401():
    assert "provider_unauthorized" in _types(
        "openai.AuthenticationError: Error code: 401 - invalid credentials"
    )


def test_invalid_api_key():
    assert "provider_invalid_api_key" in _types("Incorrect API key provided: sk-***")


def test_forbidden_403():
    assert "provider_forbidden" in _types(
        "PermissionDeniedError: Error code: 403 - you do not have access"
    )


def test_quota_exceeded():
    assert "provider_quota_exceeded" in _types("You exceeded your current quota, please check billing")


def test_tpm_exceeded():
    assert "provider_tpm_exceeded" in _types("Rate limit: tokens per minute exceeded for this org")


def test_payment_required_402():
    assert "provider_payment_required" in _types("Error code: 402 - payment required")


def test_insufficient_balance():
    assert "provider_insufficient_balance" in _types(
        "APIError: insufficient balance to complete the request"
    )


def test_model_not_found():
    assert "provider_model_not_found" in _types(
        "NotFoundError: The model `gpt-5-ultra` does not exist"
    )


def test_model_deprecated():
    assert "provider_model_deprecated" in _types(
        'The model "gpt-4-vision-preview" has been deprecated; use gpt-4o'
    )


def test_context_length_exceeded():
    assert "provider_context_length_exceeded" in _types(
        "BadRequestError: maximum context length is 8192 tokens, however you requested 9000"
    )


def test_output_truncated_finish_reason_length():
    assert "provider_output_truncated" in _types('response finished with finish_reason="length"')


def test_malformed_invalid_json():
    assert "provider_invalid_json" in _types(
        "JSONDecodeError: Expecting value: line 1 column 1 (char 0) while reading the response"
    )


def test_malformed_missing_choices():
    assert "provider_missing_choices" in _types("KeyError: 'choices' in the completion response")


def test_unexpected_model_behavior():
    assert "provider_unexpected_structure" in _types(
        "pydantic_ai.exceptions.UnexpectedModelBehavior: invalid tool call"
    )


def test_rate_limit_phrase_without_provider_cue_is_ignored():
    # "rate limit" in a product sentence, no provider/HTTP/429 context -> not a
    # provider failure (the rate-limit rule is gated on a provider cue).
    assert _types("We hit the signup rate limit for new users this month.") == set()


def test_plain_model_word_is_not_a_model_error():
    assert _types("The model predicted the sales trend for next quarter.") == set()
