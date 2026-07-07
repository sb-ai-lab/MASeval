"""Base classes and utility functions for trace validation.

Provides the BaseValidator abstract class and format normalization functions
to convert various raw trace schemas into a uniform span representation.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, NotRequired, TypedDict

Finding = dict[str, Any]


class Span(TypedDict):
    """A normalized unit of a trace fed to the validators."""

    idx: str
    text: str
    agent: str | None
    kind: NotRequired[str | None]
    parent: NotRequired[str | None]


CheckFn = Callable[[list["Span"]], list[Finding]]

QUOTE_RADIUS = 80
QUOTE_FALLBACK = 300
MAX_SPAN_TEXT = 50000

DESERIALIZE = r"deserial(?:ise|ize|isation|ization)"

_ERROR_SIGNAL = re.compile(
    r"\b[A-Za-z_]*(?:error|exception)\b|traceback|"
    r"\b[45]\d\d\b|status[_ ]?code|"
    r"\b(?:not found|not registered|not implemented|not callable|"
    r"failed|failure|invalid|refused|denied|timed ?out|timeout|"
    r"unauthorized|forbidden|missing|deprecated|unavailable|"
    r"rate[ _]?limit|quota|exceeded|no such file)\b",
    re.IGNORECASE,
)

_WS_RUN = re.compile(r"(?:(?<!\\)\\[nrt]|\s)+")


def _quote(text: str, start: int, end: int) -> str:
    a = max(0, start - QUOTE_RADIUS)
    b = min(len(text), end + QUOTE_RADIUS)
    snippet = _WS_RUN.sub(" ", text[a:b]).strip()
    if a > 0:
        cut = snippet.find(" ")
        if 0 <= cut <= 15:
            snippet = snippet[cut + 1 :]
        snippet = "…" + snippet
    if b < len(text):
        cut = snippet.rfind(" ")
        if cut >= len(snippet) - 15:
            snippet = snippet[:cut]
        snippet = snippet + "…"
    return snippet


def runtime_exception_pattern(names: str) -> str:
    """Build a regex matching a *runtime* occurrence of the given exception name(s).

    Matches a traceback header (``TypeError: message``) or an explicit
    raised/thrown verb (``raised an AttributeError``, ``TypeError was raised``),
    while excluding edited source that merely defines or handles the exception
    (``raise X(...)``, ``except X:``, ``class X``, ``def X``) and docstrings
    ("raises X if ..."), so source code is not flagged as a live crash.

    Args:
        names: A regex alternation of exception names, e.g. ``"TypeError|AttributeError"``.

    Returns:
        str: A regex pattern (case-insensitive use assumed).
    """
    return (
        rf"(?<!except )(?<!raise )(?<!class )(?<!def )\b(?:{names})\b"
        rf"(?::\s\S|\s+(?:was|were|is)\s+(?:raised|thrown|encountered|caught))"
        rf"|(?:raised|threw|thrown|caught|hit)\s+(?:an?\s+)?(?:{names})\b"
    )

_MINOR_HINTS = (
    "rate_limit",
    "tpm",
    "rpm",
    "quota",
    "timeout",
    "network",
    "empty_response",
    "truncated",
    "overload",
    "stream",
)

_CHAT_ROLES = {
    "system",
    "user",
    "assistant",
    "tool",
    "tool-call",
    "tool-response",
    "tool_call",
    "tool_response",
    "human",
}

_TRAJECTORY_KEYS = (
    "trajectory",
    "steps",
    "history",
    "spans",
    "messages",
    "events",
    "observations",
    "turns",
    "conversation",
)


def severity_for(failure_type: str) -> str:
    """Map a failure_type to a coarse severity.

    Args:
        failure_type: The specific sub-type classification of a finding.

    Returns:
        str: "minor" for transient/retryable failures, otherwise "major".
    """
    ft = (failure_type or "").lower()
    # Match hints as whole underscore-delimited segments so e.g. a future
    # "upstream_error" is not misread as transient via the "stream" hint.
    padded = f"_{ft}_"
    return "minor" if any(f"_{hint}_" in padded for hint in _MINOR_HINTS) else "major"


class BaseValidator(ABC):
    """Abstract base class for all deterministic trace validators."""

    #: Optional mapping of explanation text, keyed by failure_type or by metric_name.
    EXPLANATIONS: dict[str, str] = {}

    @abstractmethod
    def get_checks(self) -> list[CheckFn]:
        """Return a list of check functions executed by this validator.

        Returns:
            list[CheckFn]: List of validation check functions.
        """
        ...

    def run(self, spans: list[Span]) -> list[Finding]:
        """Execute all validation checks against the provided spans.

        Args:
            spans: List of normalized span dictionaries, each containing
                'span_id', 'text', and 'agent' keys.

        Returns:
            list[Finding]: List of validation findings detected in the spans.
        """
        findings: list[Finding] = []
        for check in self.get_checks():
            findings.extend(check(spans))
        return findings

    def _explanation(self, metric_name: str, failure_type: str) -> str:
        """Resolve the explanation for a finding, preferring failure_type granularity.

        Args:
            metric_name: Identifier of the metric being evaluated.
            failure_type: Specific sub-type classification of the failure.

        Returns:
            str: The most specific explanation available (failure_type, then metric_name).
        """
        return self.EXPLANATIONS.get(failure_type) or self.EXPLANATIONS.get(metric_name) or ""

    def _scan(
        self, spans: list[Span], metric_name: str, rules: list[tuple]
    ) -> list[Finding]:
        """Scan spans against regex rules, emitting one finding per distinct failure_type.

        The match offset is preserved so evidence quotes point at the actual hit
        (not the head of the span), and each failure_type is reported at most once
        per span (so distinct sub-categories in the same span are all captured
        without flooding identical duplicates).

        Args:
            spans: List of normalized spans to inspect.
            metric_name: Identifier of the metric being evaluated.
            rules: List of ``(regex_pattern, failure_type)`` tuples, or
                ``(regex_pattern, failure_type, context_cue)`` 3-tuples. When a
                ``context_cue`` is given the rule only fires if that cue also matches
                somewhere in the span text (used to require a provider/HTTP context
                next to otherwise prose-prone phrases).

        Returns:
            list[Finding]: List of generated findings for matched regex rules.
        """
        findings: list[Finding] = []
        for span in spans:
            text = span.get("text", "")
            if not text:
                continue
            seen: set[str] = set()
            for rule in rules:
                pattern, failure_type = rule[0], rule[1]
                cue = rule[2] if len(rule) > 2 else None
                if failure_type in seen:
                    continue
                if cue is not None and not re.search(cue, text, re.IGNORECASE):
                    continue
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    seen.add(failure_type)
                    findings.append(
                        self._make_finding(
                            span,
                            metric_name=metric_name,
                            failure_type=failure_type,
                            explanation=self._explanation(metric_name, failure_type),
                            start=m.start(),
                            end=m.end(),
                        )
                    )
        return findings

    def _make_finding(
        self,
        span: Span,
        metric_name: str,
        failure_type: str,
        explanation: str,
        start: int | None = None,
        end: int | None = None,
    ) -> Finding:
        """Construct a standardized finding dictionary.

        Args:
            span: The span dictionary where the issue was detected.
            metric_name: Identifier for the validation metric.
            failure_type: Specific sub-type classification of the failure.
            explanation: Human-readable explanation of the issue.
            start: Optional start offset of the regex match in ``span['text']``.
            end: Optional end offset of the regex match in ``span['text']``.

        Returns:
            Finding: Formatted finding structure.
        """
        text = span.get("text", "")
        if start is not None and end is not None:
            quote = _quote(text, start, end)
        else:
            # Offset-less check: anchor on the first error signal rather than the
            # span head (which is typically the re-fed prompt). Fall back to a
            # cleaned head only when no signal is present.
            m = _ERROR_SIGNAL.search(text)
            if m:
                quote = _quote(text, m.start(), m.end())
            else:
                head = _WS_RUN.sub(" ", text[:QUOTE_FALLBACK]).strip()
                quote = head + "…" if len(text) > QUOTE_FALLBACK else head
        return {
            "metric_name": metric_name,
            "explanation": explanation,
            "culprit_agent": span.get("agent"),
            "failure_type": failure_type,
            "severity": severity_for(failure_type),
            "evidence": [
                {
                    "idx": span["idx"],
                    "agent": span.get("agent"),
                    "quote": quote,
                }
            ],
            # Internal-only (never emitted to output): (span_id, match_start, match_end)
            # used by the runner for cross-validator overlap dedup. Offsets are None for
            # offset-less checks (e.g. the JSON-parse check).
            "_match": (span["idx"], start, end),
        }


def detect_format(trace: Any) -> str:
    """Detect the schema format of an input trace structure.

    Args:
        trace: Raw trace object (dictionary, list, or other representation).

    Returns:
        str: Format identifier ('trail', 'who_and_when', 'pumpkin', or 'unknown').
    """
    if isinstance(trace, dict):
        keys = set(trace)

        spans = trace.get("spans")
        if isinstance(spans, list) and spans:
            first = next((s for s in spans if isinstance(s, dict)), None)
            if first is not None and ({"span_name", "trace_state", "span_kind"} & set(first)):
                return "trail"

        if {"mistake_agent", "mistake_step"} & keys and isinstance(
            trace.get("history"), list
        ):
            return "who_and_when"

        observations = trace.get("observations")
        if isinstance(observations, list) and (
            {"projectId", "htmlPath"} & keys
            or (
                observations
                and isinstance(observations[0], dict)
                and {"traceId", "startTime"} & set(observations[0])
            )
        ):
            return "pumpkin"

        if {"mistake_agent", "mistake_step"} & keys:
            return "who_and_when"
        if isinstance(trace.get("history"), list):
            return "who_and_when"
        if isinstance(observations, list) and observations:
            return "pumpkin"

    if isinstance(trace, list) and trace and isinstance(trace[0], dict):
        if {"mistake_agent", "mistake_step"} & set(trace[0]):
            return "who_and_when"
    return "unknown"


def _flatten(obj: Any) -> str:
    """Convert arbitrary object into a string representation suitable for regex search.

    Args:
        obj: Any Python object or primitive.

    Returns:
        str: String representation or JSON string serialized representation.
    """
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    return json.dumps(obj, ensure_ascii=False, default=str)


def _trail_agent(node: dict) -> str | None:
    """Extract the acting agent/tool name from a TRAIL span node.

    Args:
        node: A single TRAIL span dictionary.

    Returns:
        str | None: The agent or tool name, or None if unavailable.
    """
    attrs = node.get("span_attributes")
    if isinstance(attrs, dict):
        for key in (
            "smolagents.managed_agents.0.name",
            "agent.name",
            "tool.name",
        ):
            value = attrs.get(key)
            if value:
                return str(value)
    name = node.get("span_name")
    return str(name) if name else None


def _trail_text(node: dict) -> str:
    """Build the regex-scanned text of a TRAIL node from high-signal fields only.

    Includes the span name/kind, status, events and logs, plus output-side
    span_attributes for *non-LLM* spans (tool/chain results, where real failures
    surface — e.g. a tool raising a TypeError). Deliberately excludes:

    * the call inputs and re-fed conversation history (``input.value``,
      ``llm.*_messages.*``): they duplicate across every downstream span and
      carry task prose, flooding the matcher with false positives and dominating
      regex runtime;
    * the *generated output* of LLM spans (``output.value`` /
      ``llm.output_messages.*``): model prose/code that merely *mentions* an error
      (quoting a stack trace in its reasoning, echoing the task's bug report) is
      not an infrastructure failure — a genuine provider failure surfaces in the
      span's status/events instead.

    The first part is always the plain-text span name, so the text never *starts*
    with ``{`` (which would make a span look like a broken JSON tool output).

    Args:
        node: A single TRAIL span dictionary.

    Returns:
        str: Concatenated high-signal text for regex scanning.
    """
    attrs = node.get("span_attributes")
    attrs = attrs if isinstance(attrs, dict) else {}
    kind = str(attrs.get("openinference.span.kind") or node.get("span_kind") or "").upper()
    is_llm = kind == "LLM"

    parts: list[str] = []
    for key in ("span_name", "span_kind", "status_code", "status_message", "events", "logs"):
        value = node.get(key)
        if value:
            parts.append(_flatten(value))
    for key, value in attrs.items():
        key_lower = key.lower()
        if "input" in key_lower or "messages" in key_lower:
            continue
        if is_llm and "output" in key_lower:
            continue
        if value:
            parts.append(_flatten(value))
    return " ".join(parts)


def trail_to_spans(trace: dict) -> list[Span]:
    """Convert TRAIL format trace into normalized spans.

    TRAIL traces are recursive trees: ``trace['spans']`` holds the root span(s),
    and every node nests its descendants under ``child_spans``. This walks the
    whole tree depth-first (pre-order) and emits one Span per node, using the
    node's native hex ``span_id``. The node text is built from high-signal fields
    only (see :func:`_trail_text`) — the ``child_spans`` subtree and the re-fed
    input/message history are excluded, so each node carries only its own
    infrastructure-relevant content.

    Args:
        trace: TRAIL trace dictionary containing a 'spans' list.

    Returns:
        list[Span]: Normalized list of spans, one per tree node.
    """
    spans: list[Span] = []

    def visit(node: Any, parent_id: str | None) -> None:
        if not isinstance(node, dict):
            spans.append(
                {
                    "idx": str(len(spans)),
                    "text": _flatten(node),
                    "agent": None,
                    "kind": None,
                    "parent": parent_id,
                }
            )
            return
        span_id = str(node.get("span_id") or node.get("spanId") or node.get("id") or len(spans))
        agent = _trail_agent(node)
        attrs = node.get("span_attributes")
        kind = (
            str(
                (attrs.get("openinference.span.kind") if isinstance(attrs, dict) else None)
                or node.get("span_kind")
                or ""
            ).upper()
            or None
        )
        spans.append(
            {
                "idx": span_id,
                "text": _trail_text(node),
                "agent": agent,
                "kind": kind,
                "parent": parent_id,
            }
        )
        for child in node.get("child_spans") or []:
            visit(child, span_id)

    for sp in trace.get("spans", []):
        visit(sp, None)
    return spans


def _ww_role_agent(role: Any) -> str | None:
    """Normalize a Who&When ``role`` into an agent name, or None if it is not one.

    Generic chat roles (system/user/assistant/tool/human) are not agent names, so
    they yield None even in a natively-agentic trace (e.g. a ``human`` turn mixed
    in with WebSurfer/Orchestrator turns). Native roles may carry a parenthetical
    qualifier (``Orchestrator (thought)``, ``Orchestrator (-> WebSurfer)``); these
    collapse to the base name so one agent is not split into many distinct labels.

    Args:
        role: The raw ``step['role']`` value.

    Returns:
        str | None: The base agent name, or None when the role is not an agent.
    """
    if not role:
        return None
    base = re.split(r"\s*\(", str(role))[0].strip()
    if not base or base.lower() in _CHAT_ROLES:
        return None
    return base


def who_and_when_to_spans(trace: Any) -> list[Span]:
    """Convert Who&When format trace into normalized spans.

    Uses the sequence index as span identifier. In native Who&When the agent name
    lives in ``role`` (e.g. WebSurfer / Orchestrator), so it is used as the agent
    (normalized via :func:`_ww_role_agent`: chat roles are dropped and
    parenthetical qualifiers collapsed). For converted-GAIA transcripts (every
    step uses only generic chat roles), the role is *not* an agent name, so agent
    attribution is left empty.

    Args:
        trace: Who&When trace object containing history or trajectory steps.

    Returns:
        list[Span]: Normalized list of spans.
    """
    if isinstance(trace, dict):
        steps = trace.get("history")
        if steps is None:
            steps = trace.get("trajectory", [])
    else:
        steps = trace

    roles = [s.get("role") if isinstance(s, dict) else None for s in steps]
    chat_only = bool(roles) and all(
        (r is None) or (str(r).strip().lower() in _CHAT_ROLES) for r in roles
    )

    spans: list[Span] = []
    for i, st in enumerate(steps):
        if not isinstance(st, dict):
            spans.append({"idx": str(i), "text": _flatten(st), "agent": None})
            continue
        if chat_only:
            agent = st.get("name") or st.get("mistake_agent") or st.get("agent")
        else:
            agent = _ww_role_agent(st.get("role")) or st.get("mistake_agent") or st.get("agent")
        if "content" in st:
            # name first (when present) so the text does not *start* with a JSON token —
            # a serialized tool-call content like {"args": ...} would otherwise be mistaken
            # for a broken JSON tool output once the name is appended after it.
            parts = []
            name = st.get("name")
            if name:
                parts.append(_flatten(name))
            parts.append(_flatten(st.get("content")))
            text = " ".join(p for p in parts if p)
        else:
            text = _flatten(st)
        spans.append({"idx": str(i), "text": text, "agent": agent})
    return spans


def _clean_pumpkin_agent(name: Any) -> str | None:
    """Normalize a Pumpkin observation name into an agent label.

    Strips Langfuse decorations (e.g. trailing ' run') and drops generation/util
    spans (chat completions, tool runners) that are not real agents.

    Args:
        name: The raw ``observation['name']`` value.

    Returns:
        str | None: A cleaned agent name, or None when the span is not an agent.
    """
    if not name:
        return None
    cleaned = re.split(r"\s+run\b", str(name))[0].strip()
    low = cleaned.lower()
    if not cleaned or low.startswith(("chat ", "running ", "generation")):
        return None
    return cleaned


def pumpkin_to_spans(trace: dict) -> list[Span]:
    """Convert Pumpkin format trace into normalized spans.

    Observations are emitted in chronological order (by ``startTime``), with the
    native observation id as span id. Text concatenates level, status message,
    input, and output.

    Args:
        trace: Pumpkin trace dictionary containing an 'observations' list.

    Returns:
        list[Span]: Normalized list of spans.
    """
    obs_list = trace.get("observations", [])

    def _start_key(o: Any) -> str:
        return o.get("startTime") or "" if isinstance(o, dict) else ""

    try:
        obs_sorted = sorted(obs_list, key=_start_key)
    except TypeError:
        obs_sorted = obs_list

    spans: list[Span] = []
    for i, obs in enumerate(obs_sorted):
        if not isinstance(obs, dict):
            spans.append({"idx": str(i), "text": _flatten(obs), "agent": None})
            continue
        span_id = obs.get("id") or str(i)
        agent = _clean_pumpkin_agent(obs.get("name"))
        parts = [
            obs.get("level"),
            obs.get("statusMessage"),
            _flatten(obs.get("input")),
            _flatten(obs.get("output")),
        ]
        text = " ".join(p for p in parts if p)
        spans.append({"idx": str(span_id), "text": text, "agent": agent})
    return spans


def unknown_to_spans(trace: Any) -> list[Span]:
    """Convert unrecognized format trace into normalized spans.

    Picks the step list from a known trajectory key when present; otherwise falls
    back to the longest list of dicts (then longest list of anything). Assigns
    0-indexed span IDs and leaves agent attribution empty.

    Args:
        trace: Arbitrary trace structure.

    Returns:
        list[Span]: Normalized list of spans.
    """
    if isinstance(trace, list):
        objects: list[Any] = trace
    elif isinstance(trace, dict):
        objects = None  # type: ignore[assignment]
        for key in _TRAJECTORY_KEYS:
            value = trace.get(key)
            if isinstance(value, list) and value:
                objects = value
                break
        if objects is None:
            lists = [v for v in trace.values() if isinstance(v, list) and v]
            dict_lists = [lst for lst in lists if isinstance(lst[0], dict)]
            pool = dict_lists or lists
            objects = max(pool, key=len) if pool else [trace]
    else:
        objects = [trace]
    return [
        {"idx": str(i), "text": _flatten(o), "agent": None}
        for i, o in enumerate(objects)
    ]


def to_spans(trace: Any) -> tuple[str, list[Span]]:
    """Detect trace format and convert it into normalized spans.

    Span text is capped at ``MAX_SPAN_TEXT`` characters to bound regex runtime.

    Args:
        trace: Raw input trace structure.

    Returns:
        tuple[str, list[Span]]: Tuple containing the detected format string and the
            list of normalized spans.
    """
    fmt = detect_format(trace)
    if fmt == "trail":
        spans = trail_to_spans(trace)
    elif fmt == "who_and_when":
        spans = who_and_when_to_spans(trace)
    elif fmt == "pumpkin":
        spans = pumpkin_to_spans(trace)
    else:
        spans = unknown_to_spans(trace)

    for s in spans:
        if len(s["text"]) > MAX_SPAN_TEXT:
            s["text"] = s["text"][:MAX_SPAN_TEXT]
    return fmt, spans
