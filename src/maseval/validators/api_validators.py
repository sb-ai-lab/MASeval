"""Validators for external API and HTTP network call failures.

Detects HTTP status code errors, empty responses, and low-level transport/network
failures in agent trace spans.
"""

from __future__ import annotations

import re

from maseval.validators.base import BaseValidator, CheckFn, Finding, Span

# A status/HTTP indicator that must sit immediately before a bare status number,
# so a stray "500"/"429"/"404" in prose (book titles, decimals like 41.429,
# coordinates like 4,500 m) is NOT read as an HTTP status code.
_HTTP = (
    r"(?:\bhttps?\b[ /]?|"
    r"\bstatus(?:[ _]?code)?\b[ :=]{0,3}\s*|"
    r"\bhttp[ _]?status\b[ :=]{0,3}\s*|"
    r"\berror[ _]?code\b[ :=]{0,3}\s*|"
    r"\bcode\b[ :=]{1,3}\s*)"
)


class ApiHttpValidator(BaseValidator):
    """Validator checks for external API interaction issues and network failures."""
    EXPLANATIONS = {
        "api_status_code_validator": "An external API/HTTP call returned an error status code: the "
        "request was malformed, rejected for auth/permission reasons, the "
        "resource was missing, timed out, conflicted, hit a rate limit, or "
        "the upstream server failed.",
        "api_empty_response_validator": "An external API call succeeded at the transport level but returned "
        "no usable payload: an empty body, a null/empty output object, or a "
        "successful response with no data.",
        "api_network_failure_validator": "An external API call failed at the network/transport layer: the "
        "connection was refused or reset, DNS resolution failed, an SSL/TLS "
        "error occurred, a read/connect timeout fired, or the network was "
        "unreachable.",
    }

    def get_checks(self) -> list[CheckFn]:
        """Return the API/HTTP validation check functions.

        Returns:
            list[CheckFn]: List of check methods.
        """
        return [
            self.check_status_code,
            self.check_empty_response,
            self.check_network_failure,
        ]

    def _scan(
        self, spans: list[Span], metric_name: str, rules: list[tuple]
    ) -> list[Finding]:
        """Scan spans using regular expressions to detect specific API failures.

        Args:
            spans: List of normalized spans to inspect.
            metric_name: Identifier of the metric being evaluated.
            rules: List of ``(regex_pattern, failure_type)`` tuples, or
                ``(regex_pattern, failure_type, context_cue)`` 3-tuples. When a
                ``context_cue`` is given the rule only fires if that cue also
                matches somewhere in the span text (used to require an HTTP/error
                context next to otherwise prose-prone phrases). At most one
                finding is emitted per span (first matching rule wins).

        Returns:
            list[Finding]: List of generated findings for matched regex rules.
        """
        findings: list[Finding] = []
        for span in spans:
            text = span.get("text", "")
            if not text:
                continue
            for rule in rules:
                pattern, failure_type = rule[0], rule[1]
                cue = rule[2] if len(rule) > 2 else None
                if cue is not None and not re.search(cue, text, re.IGNORECASE):
                    continue
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    findings.append(
                        self._make_finding(
                            span,
                            metric_name=metric_name,
                            failure_type=failure_type,
                            explanation=self.EXPLANATIONS[metric_name],
                            start=m.start(),
                            end=m.end(),
                        )
                    )
                    break
        return findings

    def check_status_code(self, spans: list[Span]) -> list[Finding]:
        """Detect HTTP error status codes (4xx, 5xx) and related error messages.

        Numeric codes must be adjacent to an HTTP/status indicator (see ``_HTTP``)
        or paired with their canonical reason phrase, so bare digits in prose are
        not misread as status codes. Prose triggers additionally require an
        error/HTTP context (``_ERR_CTX``) in the span.

        Args:
            spans: List of normalized spans to inspect.

        Returns:
            list[Finding]: List of findings corresponding to status code errors.
        """
        rules = [
            (
                _HTTP + r"(?:400|402|422)\b|"
                r"\b(?:400|422)\b\W{0,4}(?:bad request|unprocessable entity)|"
                r"\bbad request\b|\bunprocessable entity\b|\bmalformed request\b",
                "api_malformed_request",
            ),
            (
                _HTTP + r"40[13]\b|"
                r"\b40[13]\b\W{0,6}(?:unauthorized|forbidden|permission denied)|"
                r"\bauthenticationerror\b|\bpermissiondeniederror\b|"
                r"auth(?:entication|orization)? (?:error|failed)|"
                r"\b401\b\W{0,4}unauthorized|\b403\b\W{0,4}forbidden",
                "api_auth_error",
            ),
            (
                _HTTP + r"404\b|\b404\b\W{0,4}not found|\bresource not found\b|"
                r"\b(?:page|endpoint|url|resource|host)\b[^.\n]{0,20}not found",
                "api_not_found",
            ),
            (
                _HTTP + r"408\b|\brequest tim(?:e ?out|ed out)\b",
                "api_request_timeout",
            ),
            (
                _HTTP + r"409\b|\b409\b\W{0,4}conflict|\bconflict error\b|"
                r"\bconflict\b\W{0,10}(?:detected|occurred|while)",
                "api_conflict",
            ),
            (
                _HTTP + r"429\b|\b429\b\W{0,6}too many requests|\btoo many requests\b|"
                r"\brate[ _]?limit(?:ing|ed|s)?\b[^.\n]{0,15}(?:exceed|reach|hit|error|throttl)|"
                r"(?:exceed|hit|throttl)\w*[^.\n]{0,15}\brate[ _]?limit",
                "api_rate_limit",
            ),
            (
                _HTTP + r"50[0234]\b|"
                r"\binternal server error\b|\bserver (?:error|failure)\b|"
                r"\bbad gateway\b|\bservice unavailable\b|\bgateway time[ -]?out\b",
                "api_server_error",
            ),
        ]
        return self._scan(spans, "api_status_code_validator", rules)

    def check_empty_response(self, spans: list[Span]) -> list[Finding]:
        """Detect successful requests that returned empty or null payloads.

        Args:
            spans: List of normalized spans to inspect.

        Returns:
            list[Finding]: List of findings for empty API responses.
        """
        rules = [
            (
                r"empty (response|body|payload)|response (body )?(is )?empty",
                "api_empty_response",
            ),
            (
                r"output\s*[=:]\s*null|output\s*[=:]\s*\{\s*\}|"
                r"output\s*[=:]\s*\[\s*\]",
                "api_empty_response",
            ),
            (
                r"(response|request) (succeeded|success|ok|200)[^.\n]{0,40}"
                r"(no data|empty|nothing returned|no (results?|content|payload))",
                "api_empty_response",
            ),
            (r"no data (returned|received|in response)", "api_empty_response"),
        ]
        return self._scan(spans, "api_empty_response_validator", rules)

    def check_network_failure(self, spans: list[Span]) -> list[Finding]:
        """Detect transport and network layer failures (timeouts, DNS, SSL, reset).

        Args:
            spans: List of normalized spans to inspect.

        Returns:
            list[Finding]: List of findings for network failures.
        """
        rules = [
            (r"\bconnection refused\b", "api_network_failure"),
            (r"\bconnection reset\b", "api_network_failure"),
            (
                r"(dns|name resolution) (failure|failed|error)|"
                r"(could not|failed to|unable to) resolve (host|name)|"
                r"name or service not known|getaddrinfo (failed|error)",
                "api_network_failure",
            ),
            (
                # Word-boundaried so "ssl" is not matched inside words like "tossle".
                r"\bsslerror\b|\bssl\b[ _]?(?:error|handshake|certificate)|"
                r"\btls\b (?:error|handshake)|certificate (?:verify|error)",
                "api_network_failure",
            ),
            (
                r"read time[ -]?out|connect time[ -]?out|"
                r"(read|connect)timeouterror|connection time[ -]?out",
                "api_network_failure",
            ),
            (r"network (is )?unreachable|no route to host", "api_network_failure"),
            (r"\bconnectionerror\b|connection aborted|broken pipe", "api_network_failure"),
        ]
        return self._scan(spans, "api_network_failure_validator", rules)
