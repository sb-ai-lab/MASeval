"""Validators for LLM API provider failures and operational constraints.

Detects authentication errors, quota limits, billing issues, model availability errors,
context length limit exceeded errors, and malformed provider responses.
"""

from __future__ import annotations

from maseval.validators.base import DESERIALIZE, BaseValidator, CheckFn, Finding, Span

_CODE = r"(?:status[_ ]?code|http[_ ]?status|error[_ ]?code)\W{0,4}"

_PROVIDER_CUE = (
    r"(?:litellm|openai|anthropic|cohere|mistral|together(?:ai)?|groq|bedrock|"
    r"vertex(?:ai)?|gemini|azure|huggingface|replicate|fireworks|"
    r"ratelimiterror|authenticationerror|permissiondeniederror|"
    r"apierror|apistatuserror|apiconnectionerror|apitimeouterror|"
    r"http[/ ]?\d|status[_ ]?code|http[_ ]?status|error[_ ]?code|"
    r"tokens? per minute|requests? per minute|\btpm\b|\brpm\b)"
)


class ProviderValidator(BaseValidator):
    EXPLANATIONS = {
        # check_quota
        "provider_quota_validator": "The span text reports a provider rate/quota limit being hit.",
        "provider_rate_limit": "The provider throttled the request with a rate limit / HTTP 429 "
        "(too many requests in the time window).",
        "provider_quota_exceeded": "The provider account's usage quota has been exhausted.",
        "provider_tpm_exceeded": "The provider's tokens-per-minute (TPM) limit was exceeded.",
        "provider_rpm_exceeded": "The provider's requests-per-minute (RPM) limit was exceeded.",
        # check_credential
        "provider_credential_validator": "The span text reports an API-key/auth problem on a provider call.",
        "provider_invalid_api_key": "The provider rejected the API key as invalid or incorrect.",
        "provider_expired_api_key": "The provider rejected the API key because it has expired.",
        "provider_revoked_api_key": "The provider rejected the API key because it was revoked.",
        "provider_unauthorized": "The provider returned 401 Unauthorized / an authentication error.",
        "provider_forbidden": "The provider returned 403 Forbidden / permission denied for the call.",
        "provider_invalid_token": "The authentication token is invalid or has expired.",
        "provider_missing_credential": "No API key / credential was supplied for the provider call.",
        # check_billing
        "provider_billing_validator": "The span text reports a provider billing problem.",
        "provider_insufficient_balance": "The call failed because the account balance/credit is insufficient.",
        "provider_billing_hard_limit": "A billing hard limit was reached, blocking further calls.",
        "provider_payment_required": "The provider returned 402 Payment Required.",
        "provider_credits_exhausted": "The account's prepaid credits are exhausted.",
        "provider_account_suspended": "The account was suspended/deactivated for billing reasons.",
        # check_model_availability
        "provider_model_availability_validator": "The span text reports the requested model is unavailable.",
        "provider_model_not_found": "The requested model does not exist / is unknown to the provider.",
        "provider_model_deprecated": "The requested model has been deprecated by the provider.",
        "provider_model_unavailable": "The requested model is unavailable in this region/account.",
        "provider_model_not_enabled": "The requested model is not enabled for this project/account.",
        "provider_unsupported_model": "The requested model name is not supported by the provider.",
        "provider_model_incompatible": "The provider/runtime lacks support for the requested model.",
        "provider_model_validation_failed": "Validation of the requested model failed at the provider.",
        # check_context_limit
        "provider_context_limit_validator": "The span text reports a context/token limit being exceeded.",
        "provider_context_length_exceeded": "The request exceeded the model's maximum context length.",
        "provider_input_too_long": "The input/prompt is longer than the model accepts.",
        "provider_output_truncated": "The output was truncated by the max-tokens limit (finish_reason=length).",
        "provider_max_completion_tokens_exceeded": "The requested max completion tokens exceeds the model limit.",
        "provider_context_overflow": "The combined input/output overflowed the model's context window.",
        "provider_token_limit_exceeded": "A token limit for the model/request was exceeded.",
        # check_malformed_response
        "provider_malformed_response_validator": "The span text reports a broken provider response.",
        "provider_invalid_json": "The provider response was not valid JSON (a decode error was reported).",
        "provider_missing_choices": "The provider response is missing expected choices/message/content fields.",
        "provider_unexpected_structure": "The provider response did not match the expected structure.",
        "provider_stream_interrupted": "The streaming response was interrupted / ended unexpectedly.",
        "provider_response_deserialization_error": "The provider response could not be deserialized into the expected shape.",
    }

    def get_checks(self) -> list[CheckFn]:
        return [
            self.check_credential,
            self.check_quota,
            self.check_billing,
            self.check_model_availability,
            self.check_context_limit,
            self.check_malformed_response,
        ]

    def check_quota(self, spans: list[Span]) -> list[Finding]:
        rules = [
            (
                r"RateLimitError|" + _CODE + r"429(?!\d)|"
                r"\b429\b[^.\n]{0,20}too many requests|too many requests[^.\n]{0,20}\b429\b|"
                r"(?:http|https|status|error)\W{0,8}too many requests|"
                r"rate[ _]?limit[^.\n]{0,25}(exceed|reach|hit|error|throttl)|"
                r"(exceed|hit|throttl)[^.\n]{0,15}rate[ _]?limit",
                "provider_rate_limit",
                _PROVIDER_CUE,
            ),
            (
                # Match both word orders — the canonical OpenAI message is
                # "You exceeded your current quota" (verb before "quota") — plus
                # the raw error code.
                r"quota[^.\n]{0,15}(exceeded|exhausted|reached)|"
                r"(exceeded|exhausted|reached)[^.\n]{0,20}quota|"
                r"insufficient[_ ]?quota",
                "provider_quota_exceeded",
            ),
            (
                r"tokens per minute[^.\n]{0,15}exceeded|\btpm\b[^.\n]{0,15}exceeded",
                "provider_tpm_exceeded",
            ),
            (
                r"requests per minute[^.\n]{0,15}exceeded|\brpm\b[^.\n]{0,15}exceeded",
                "provider_rpm_exceeded",
            ),
        ]
        return self._scan(spans, "provider_quota_validator", rules)

    def check_credential(self, spans: list[Span]) -> list[Finding]:
        rules = [
            (r"invalid[_ ]?api[_ ]?key|incorrect api key", "provider_invalid_api_key"),
            (r"api[_ ]?key[^.\n]{0,20}expired|expired api key", "provider_expired_api_key"),
            (r"api[_ ]?key[^.\n]{0,20}revoked|revoked[^.\n]{0,20}key", "provider_revoked_api_key"),
            (
                r"AuthenticationError|" + _CODE + r"401(?!\d)",
                "provider_unauthorized",
            ),
            (
                r"PermissionDeniedError|" + _CODE + r"403(?!\d)",
                "provider_forbidden",
            ),
            (r"invalid[ _]token|expired[ _]token|token[^.\n]{0,15}expired", "provider_invalid_token"),
            (
                r"missing[^.\n]{0,15}credential|credential[^.\n]{0,15}missing|"
                r"missing api key|api key not (found|set|provided)",
                "provider_missing_credential",
            ),
        ]
        return self._scan(spans, "provider_credential_validator", rules)

    def check_billing(self, spans: list[Span]) -> list[Finding]:
        rules = [
            (r"insufficient (balance|funds|credit)", "provider_insufficient_balance"),
            (r"billing hard limit|hard limit reached", "provider_billing_hard_limit"),
            (r"" + _CODE + r"402(?!\d)|\b402\s+payment required\b", "provider_payment_required"),
            (
                r"credits?[^.\n]{0,15}(exhausted|depleted)|out of credits",
                "provider_credits_exhausted",
            ),
            (
                r"account[^.\n]{0,20}(suspended|deactivated)[^.\n]{0,20}billing|"
                r"billing[^.\n]{0,20}suspended",
                "provider_account_suspended",
            ),
        ]
        return self._scan(spans, "provider_billing_validator", rules)

    def check_model_availability(self, spans: list[Span]) -> list[Finding]:
        rules = [
            (
                r"model[^.\n]{0,30}(does not exist|not found)|unknown model",
                "provider_model_not_found",
            ),
            (
                r"model\s+['\"][\w.\-/:@]+['\"][^.\n]{0,25}deprecat|"
                r"deprecat\w*[^.\n]{0,25}model\s+['\"][\w.\-/:@]+['\"]",
                "provider_model_deprecated",
            ),
            (
                r"model[^.\n]{0,30}(unavailable|not available)[^.\n]{0,30}(region|account)",
                "provider_model_unavailable",
            ),
            (r"model[^.\n]{0,20}not enabled", "provider_model_not_enabled"),
            (
                r"unsupported model|model\s+['\"][\w.\-/:]+['\"][^.\n]{0,15}not supported|"
                r"model[^.\n]{0,20}not supported",
                "provider_unsupported_model",
            ),
            (
                r"(lacks?|lacked|no) support for[^.\n]{0,30}model|model[^.\n]{0,20}not implemented",
                "provider_model_incompatible",
            ),
            (
                r"(?:failed|error)[^.\n]{0,20}validat\w*[^.\n]{0,20}"
                r"(?:model\s+['\"][\w.\-/:@]+['\"]|"
                r"\b(?:gpt|claude|llama|gemini|mistral|qwen)[-\d][\w.\-]*|\bo[1-4]-[\w.\-]+)|"
                r"model\s+['\"][\w.\-/:@]+['\"][^.\n]{0,15}validation (?:failed|error)",
                "provider_model_validation_failed",
            ),
        ]
        return self._scan(spans, "provider_model_availability_validator", rules)

    def check_context_limit(self, spans: list[Span]) -> list[Finding]:
        rules = [
            (
                r"context (length|window)[^.\n]{0,20}exceeded|maximum context length|"
                r"context_length_exceeded",
                "provider_context_length_exceeded",
            ),
            (r"input too long|prompt is too long", "provider_input_too_long"),
            (
                r"truncated[^.\n]{0,15}max[_ ]?tokens|finish_reason\W{0,5}['\"]?length",
                "provider_output_truncated",
            ),
            (
                r"max(imum)?[_ ]?completion[_ ]?tokens[^.\n]{0,15}exceeded",
                "provider_max_completion_tokens_exceeded",
            ),
            (r"context overflow", "provider_context_overflow"),
            (
                r"token limit (violation|exceeded|reached)|exceed(s|ed)?[^.\n]{0,15}token limit",
                "provider_token_limit_exceeded",
            ),
        ]
        return self._scan(spans, "provider_context_limit_validator", rules)

    def check_malformed_response(self, spans: list[Span]) -> list[Finding]:
        rules = [
            (
                r"invalid json[^.\n]{0,15}response|"
                r"json(?:decode)?error\b[^.\n]{0,40}"
                r"(?:expecting value|extra data|unterminated|invalid control|"
                r"delimiter|line \d+ column \d+|char \d+)",
                "provider_invalid_json",
            ),
            (
                r"(missing|no)[^.\n]{0,15}(choices|message|content)[^.\n]{0,15}(field|key|in response)|"
                r"keyerror\W{0,3}'?(choices|message|content)'?",
                "provider_missing_choices",
            ),
            (
                r"unexpected response (format|structure)|malformed response|"
                r"\bUnexpectedModelBehavior\b|"
                r"invalid response from[^.\n]{0,40}(?:chat completions?|completions?|endpoint)|"
                r"\d+ validation errors? for ChatCompletion",
                "provider_unexpected_structure",
            ),
            (
                r"stream(ing)? (interrupted|ended unexpectedly|aborted)",
                "provider_stream_interrupted",
            ),
            (
                DESERIALIZE + r"[^.\n]{0,10}(?:error|fail)[^.\n]{0,25}(?:response|completion|message)|"
                r"(?:response|completion)[^.\n]{0,25}" + DESERIALIZE + r"[^.\n]{0,10}(?:error|fail)|"
                r"could not (?:parse|decode)[^.\n]{0,15}response",
                "provider_response_deserialization_error",
            ),
        ]
        return self._scan(spans, "provider_malformed_response_validator", rules)
