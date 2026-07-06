"""Deterministic evidence verification for LLM evaluator findings.

EvidenceVerifier does not decide whether an LLM finding is semantically correct.
It only checks whether the finding is grounded in the provided trace:

* cited span ids or zero-based message indices exist;
* quoted evidence can be found in the cited spans;
* culprit-agent candidates are at least compatible with the cited evidence;
* evidence roles use an expected vocabulary.

The output is intentionally close to the schema used in the MASQUE Studio design:
``verified | weak | invalid`` + check-level diagnostics.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from ..models import (
    Confidence,
    DialogueMessage,
    EvaluationInput,
    EvidenceChecks,
    EvidenceItemCheck,
    EvidenceStatus,
    EvidenceVerificationMetricResult,
    EvidenceVerificationResult,
    Finding,
    MetricResult,
    RawTraceInput,
)


@dataclass(frozen=True)
class _SpanRecord:
    """Internal normalized view of one trace item that can be cited as evidence."""

    span_id: str
    content: str
    agent: str | None = None
    span_type: str | None = None




@dataclass(frozen=True)
class _ResolvedEvidenceItem:
    """Internal resolution result for one cited evidence item."""

    check: EvidenceItemCheck
    record: _SpanRecord | None = None

class EvidenceVerifier:
    """Verify whether LLM evaluator findings are trace-grounded.

    Parameters:
        allowed_roles: Accepted evidence roles. This is intentionally small and
            explicit, but can be extended per project.
        allow_agent_name_as_span_id: If True, agent names are accepted as
            citeable ids. Defaults to False because MASQUE Studio should prefer
            concrete step/message indices or state/response ids over agent-level references.
    """

    DEFAULT_ALLOWED_ROLES = {
        "root_cause",
        "supporting",
        "contributing",
        "propagation",
        "final_effect",
        "context",
        "response",
        "primary",
        "secondary",
        "culprit",
        "action",
        "observation",
        "tool_call",
        "agent_output",
        "agent output",
        "output",
    }

    # Common ids emitted by the current prompts / parsers.
    _RAW_ID_PATTERNS = (
        re.compile(r'"(?:idx|span_id|state_id|response_id|id)"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"'),
        re.compile(r"\b(?:idx|span|state|response)_[A-Za-z0-9_.:-]+\b"),
        re.compile(r"\btw-\d+\b"),
        re.compile(r"\b[A-Za-z][A-Za-z0-9 .()/-]+_\d+\b"),
    )
    _INDEXED_TRACE_BLOCK_PATTERN = re.compile(
        r"(?ms)^\s*\[(\d+)\]\s*(.*?)(?=^\s*\[\d+\]\s*|\Z)"
    )
    _LOOSE_INDEXED_TRACE_BLOCK_PATTERN = re.compile(
        r"(?ms)^\s*(?:message|step|turn)\s*[#:]?\s*(\d+)\s*[:\-]\s*(.*?)(?=^\s*(?:message|step|turn)\s*[#:]?\s*\d+\s*[:\-]|\Z)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        allowed_roles: set[str] | None = None,
        allow_agent_name_as_span_id: bool = False,
    ):
        self.allowed_roles = {r.lower() for r in (allowed_roles or self.DEFAULT_ALLOWED_ROLES)}
        self.allow_agent_name_as_span_id = allow_agent_name_as_span_id

    def verify_metric_result(
        self,
        metric_result: MetricResult | Mapping[str, Any],
        eval_input: EvaluationInput | RawTraceInput,
    ) -> EvidenceVerificationMetricResult:
        """Verify all findings for one metric."""

        metric_result = self._coerce_metric_result(metric_result)
        span_index = self._build_span_index(eval_input)
        raw_trace_text = self._raw_trace_text(eval_input)

        verifications = [
            self.verify_finding(
                metric_name=metric_result.metric_name,
                finding=finding,
                finding_index=i,
                span_index=span_index,
                raw_trace_text=raw_trace_text,
            )
            for i, finding in enumerate(metric_result.findings)
        ]
        return EvidenceVerificationMetricResult(
            metric_name=metric_result.metric_name,
            verifications=verifications,
        )

    def verify_all(
        self,
        metric_results: Mapping[str, MetricResult | Mapping[str, Any]],
        eval_input: EvaluationInput | RawTraceInput,
    ) -> dict[str, EvidenceVerificationMetricResult]:
        """Verify a mapping ``{metric_name: MetricResult | dict}``."""

        verified: dict[str, EvidenceVerificationMetricResult] = {}
        for metric_name, result in metric_results.items():
            if not self._looks_like_metric_result(result):
                continue
            verified[metric_name] = self.verify_metric_result(result, eval_input)
        return verified

    def verify_finding(
        self,
        metric_name: str,
        finding: Finding,
        finding_index: int,
        span_index: Mapping[str, _SpanRecord],
        raw_trace_text: str | None = None,
    ) -> EvidenceVerificationResult:
        """Verify one LLM finding."""

        evidence_resolutions = self._resolve_evidence_items(
            finding=finding,
            span_index=span_index,
            raw_trace_text=raw_trace_text,
        )
        evidence_item_checks = [r.check for r in evidence_resolutions]
        resolved_records = [r.record for r in evidence_resolutions if r.record is not None]
        checks = EvidenceChecks(
            all_idxs_exist=bool(finding.evidence) and all(i.span_exists for i in evidence_item_checks),
            quotes_found_in_spans=bool(finding.evidence) and all(i.quote_found for i in evidence_item_checks),
            culprit_agent_matches_evidence=self._check_culprit_agent_matches(finding, resolved_records),
            span_roles_are_plausible=bool(finding.evidence) and all(i.role_plausible for i in evidence_item_checks),
        )
        evidence_status = self._status_from_items(finding, checks, evidence_item_checks)
        # In the soft verifier, WEAK means "grounded enough to inspect/use,"
        # not "discard". INVALID is reserved for gross grounding failures
        # such as missing evidence or quotes absent from the trace.
        usable_for_diagnosis = evidence_status != EvidenceStatus.INVALID
        explanation = self._make_explanation(evidence_status, checks, evidence_item_checks)

        return EvidenceVerificationResult(
            metric_name=metric_name,
            finding_index=finding_index,
            evidence_status=evidence_status,
            evidence_checks=checks,
            evidence_item_checks=evidence_item_checks,
            usable_for_diagnosis=usable_for_diagnosis,
            verifier_explanation=explanation,
        )

    def _status_from_items(
        self,
        finding: Finding,
        checks: EvidenceChecks,
        evidence_item_checks: list[EvidenceItemCheck],
    ) -> EvidenceStatus:
        """Assign a deliberately permissive grounding status.

        The verifier is meant to catch only gross grounding failures. A finding is
        INVALID only when it has no evidence or none of its quoted evidence can be
        found in the trace. If at least one evidence item is grounded, the finding
        is still useful for diagnosis and is marked WEAK unless all checks pass.
        This prevents one bad citation from discarding a mostly grounded finding.
        """
        if not finding.evidence or not evidence_item_checks:
            return EvidenceStatus.INVALID

        grounded_items = [item for item in evidence_item_checks if item.quote_found]
        if not grounded_items:
            return EvidenceStatus.INVALID

        # Some evidence items are grounded and some are not: keep the finding,
        # but mark it weak so downstream aggregation can review it.
        if len(grounded_items) < len(evidence_item_checks):
            return EvidenceStatus.WEAK

        if (
            not checks.all_idxs_exist
            or not checks.culprit_agent_matches_evidence
            or not checks.span_roles_are_plausible
            or finding.confidence_estimate == Confidence.LOW
            or finding.needs_human_review
        ):
            return EvidenceStatus.WEAK
        return EvidenceStatus.VERIFIED

    @staticmethod
    def _make_explanation(
        status: EvidenceStatus,
        checks: EvidenceChecks,
        evidence_item_checks: list[EvidenceItemCheck],
    ) -> str:
        if status == EvidenceStatus.VERIFIED:
            return "All cited spans exist, quoted evidence matches the trace, and the finding is usable for diagnosis."

        failed = [
            name
            for name, value in checks.model_dump().items()
            if value is False
        ]
        item_problems = [
            f"evidence[{item.evidence_index}] {item.idx}: {item.problem}"
            for item in evidence_item_checks
            if item.problem
        ]
        item_details = " Details: " + " | ".join(item_problems[:5]) if item_problems else ""

        if status == EvidenceStatus.INVALID:
            if "quotes_found_in_spans" in failed:
                return (
                    "None of the cited quotes were found in the trace, "
                    "so this LLM finding cannot be used as evidence." + item_details
                )
            return "The LLM finding is not sufficiently grounded in the trace and cannot be used as evidence." + item_details

        grounded_count = sum(1 for item in evidence_item_checks if item.quote_found)
        if 0 < grounded_count < len(evidence_item_checks):
            return (
                f"The LLM finding is partially grounded: {grounded_count}/{len(evidence_item_checks)} "
                "evidence quotes were found in the trace. Treat it as weak rather than invalid; "
                "the missing citations should be ignored or reviewed."
                + item_details
            )

        if "all_idxs_exist" in failed and "quotes_found_in_spans" not in failed:
            return (
                "The quoted evidence was found in the raw trace, but at least one cited idx "
                "is not resolvable in the normalized span index. Treat this finding as weak: "
                "the text is grounded, but the step id needs normalization or prompt correction."
                + item_details
            )

        return (
            "The LLM finding is partially grounded but needs review before it is used for diagnosis. "
            f"Failed or weak checks: {', '.join(failed) if failed else 'low confidence / human-review flag'}."
            + item_details
        )

    def _resolve_evidence_items(
        self,
        finding: Finding,
        span_index: Mapping[str, _SpanRecord],
        raw_trace_text: str | None,
    ) -> list[_ResolvedEvidenceItem]:
        """Resolve and verify each evidence item.

        Numeric ``span_id`` values are treated as message indices. The verifier
        first tries the exact index, then ``index - 1`` and ``index + 1``. This
        handles the common mismatch between zero-based and one-based numbering
        in LLM outputs while still recording the resolved id explicitly.
        """
        resolved_items: list[_ResolvedEvidenceItem] = []
        for evidence_index, evidence in enumerate(finding.evidence):
            span_id = str(evidence.idx).strip()
            role_plausible = (evidence.role or "").strip().lower() in self.allowed_roles
            quote = evidence.quote or ""

            record, resolved_span_id, resolution_strategy = self._resolve_span_record(
                span_id=span_id,
                quote=quote,
                span_index=span_index,
                raw_trace_text=raw_trace_text,
            )

            quote_found = False
            used_raw_trace_fallback = resolution_strategy.startswith("raw_")

            if record is not None and quote.strip() and self._contains_quote(record.content, quote):
                quote_found = True
            elif quote.strip() and raw_trace_text and self._contains_quote(raw_trace_text, quote):
                # Soft mode: the exact/neighbor step id may be wrong, but the quoted
                # evidence is present in the trace. Resolve the item to a raw-text
                # window around the quote, infer the nearby agent if possible, and
                # treat the span as softly resolved instead of failing the finding.
                quote_found = True
                used_raw_trace_fallback = True
                raw_record = self._record_from_raw_quote(
                    span_id=span_id,
                    quote=quote,
                    raw_trace_text=raw_trace_text,
                )
                if raw_record is not None:
                    record = raw_record
                    resolution_strategy = "raw_quote_numeric_soft" if self._is_int_like(span_id) else "raw_quote_soft"

            # In soft mode, a record obtained from raw-quote fallback is still a
            # resolved evidence location. We reserve span_exists=false for cases
            # where neither an indexed block nor a raw quote window could be found.
            span_exists = record is not None

            problems: list[str] = []
            if not span_exists:
                problems.append(
                    "idx could not be resolved and the quote was not found in the trace; "
                    "this is a gross grounding error"
                )
            elif used_raw_trace_fallback:
                problems.append(
                    "idx was resolved softly by locating the quote in raw trace text; "
                    "exact message index may be off, but the evidence text is grounded"
                )
            if not quote_found:
                if not quote.strip():
                    problems.append("quote is empty")
                else:
                    problems.append("quote was not found in the cited span, neighboring index, or raw trace")
            if not role_plausible:
                problems.append(f"unsupported evidence role: {evidence.role!r}")
            if span_exists and resolved_span_id != span_id:
                problems.append(
                    f"idx {span_id!r} was resolved as {resolved_span_id!r} using {resolution_strategy}"
                )

            check = EvidenceItemCheck(
                evidence_index=evidence_index,
                idx=span_id,
                span_exists=span_exists,
                quote_found=quote_found,
                role_plausible=role_plausible,
                used_raw_trace_fallback=used_raw_trace_fallback,
                resolved_idx=resolved_span_id if resolved_span_id != span_id else None,
                resolved_agent=record.agent if record is not None else None,
                resolution_strategy=resolution_strategy,
                problem="; ".join(problems) if problems else None,
            )
            resolved_items.append(_ResolvedEvidenceItem(check=check, record=record))
        return resolved_items

    def _resolve_span_record(
        self,
        span_id: str,
        quote: str,
        span_index: Mapping[str, _SpanRecord],
        raw_trace_text: str | None,
    ) -> tuple[_SpanRecord | None, str, str]:
        """Resolve a cited span id to a record.

        For numeric ids, try exact index and then +/-1. Prefer the candidate
        whose content contains the quote; otherwise fall back to exact index if
        it exists.
        """
        if self._is_int_like(span_id):
            idx = int(span_id)
            candidates = [idx, idx - 1, idx + 1]
            existing: list[tuple[str, _SpanRecord, str]] = []
            for candidate in candidates:
                if candidate < 0:
                    continue
                candidate_id = str(candidate)
                record = span_index.get(candidate_id)
                if record is None:
                    continue
                strategy = "numeric_exact" if candidate == idx else "numeric_pm1_fuzzy"
                existing.append((candidate_id, record, strategy))
                # Prefer the numeric candidate whose block actually contains the quote.
                # This makes the verifier tolerant to zero-based vs one-based index
                # confusion while still grounding the finding in a concrete message.
                if quote.strip() and self._contains_quote(record.content, quote):
                    return record, candidate_id, strategy
            if existing:
                # If no candidate contains the quote, keep the exact id if it exists;
                # otherwise use the first available neighbor. Quote verification will
                # decide whether the finding becomes weak/invalid.
                for candidate_id, record, strategy in existing:
                    if candidate_id == span_id:
                        return record, candidate_id, strategy
                candidate_id, record, strategy = existing[0]
                return record, candidate_id, strategy

        if span_id in span_index:
            return span_index[span_id], span_id, "exact"

        return None, span_id, "unresolved"

    @staticmethod
    def _is_int_like(value: str) -> bool:
        try:
            int(str(value).strip())
            return True
        except (TypeError, ValueError):
            return False

    def _record_from_raw_quote(
        self,
        span_id: str,
        quote: str,
        raw_trace_text: str | None,
    ) -> _SpanRecord | None:
        if not raw_trace_text or not quote.strip() or not self._contains_quote(raw_trace_text, quote):
            return None
        content = self._raw_indexed_block_containing_quote(raw_trace_text, quote)
        if content is None:
            content = self._raw_window_around_quote(raw_trace_text, quote)
        return _SpanRecord(
            span_id=span_id,
            content=content,
            agent=self._infer_agent_from_raw_step(content),
            span_type="raw_quote_soft_fallback",
        )

    @classmethod
    def _raw_indexed_block_containing_quote(cls, raw_trace_text: str, quote: str) -> str | None:
        """Return the indexed `[N] ...` block that contains the quote, if any.

        This catches cases where the judge cited the wrong numeric id but the
        quote clearly belongs to a neighboring or differently numbered block.
        """
        for pattern in (cls._INDEXED_TRACE_BLOCK_PATTERN, cls._LOOSE_INDEXED_TRACE_BLOCK_PATTERN):
            for match in pattern.finditer(raw_trace_text):
                content = match.group(2).strip()
                if content and cls._contains_quote(content, quote):
                    return content
        return None

    @staticmethod
    def _raw_window_around_quote(raw_trace_text: str, quote: str, window: int = 1800) -> str:
        quote_unescaped = quote.replace("\\n", "\n").replace('\\"', '"')
        pos = raw_trace_text.find(quote_unescaped)
        if pos < 0:
            pos = raw_trace_text.find(quote)
        if pos < 0:
            normalized_quote = EvidenceVerifier._normalize_for_matching(quote)
            # If exact offsets are unavailable after normalization, use the whole
            # trace only as a last resort. This still lets simple agent-name
            # checks find nearby explicit `agent:` labels in many traces.
            if normalized_quote in EvidenceVerifier._normalize_for_matching(raw_trace_text):
                return raw_trace_text[: min(len(raw_trace_text), window * 2)]
            return raw_trace_text[: min(len(raw_trace_text), window * 2)]
        start = max(0, pos - window)
        end = min(len(raw_trace_text), pos + len(quote_unescaped) + window)
        return raw_trace_text[start:end]

    def _check_culprit_agent_matches(
        self,
        finding: Finding,
        resolved_records: list[_SpanRecord],
    ) -> bool:
        """Check whether cited evidence is compatible with culprit candidates.

        This is intentionally soft. We only return False when the verifier can
        confidently resolve evidence to agent-owned records and none of those
        records belongs to any listed culprit candidate. If agent ownership cannot
        be inferred from the trace, we do not fail the finding solely because of
        missing metadata.
        """
        if not finding.culprit_agent_candidates:
            return True
        if not finding.evidence or not resolved_records:
            return False

        candidate_agents = {
            c.agent.strip().lower()
            for c in finding.culprit_agent_candidates
            if c.agent and c.agent.strip()
        }
        if not candidate_agents:
            return True

        saw_known_agent = False
        for record in resolved_records:
            content_l = record.content.lower()
            span_id_l = record.span_id.lower()
            record_agent = (record.agent or "").strip()
            record_agent_l = record_agent.lower()

            if record_agent_l:
                saw_known_agent = True
                for agent_l in candidate_agents:
                    if agent_l == record_agent_l or agent_l in record_agent_l or record_agent_l in agent_l:
                        return True

            # Fallbacks for raw traces that include agent labels in the text window.
            for agent_l in candidate_agents:
                if agent_l in span_id_l:
                    return True
                if re.search(rf"(?im)^\s*(?:agent|agent_name|name|sender|role)\s*:\s*{re.escape(agent_l)}\b", content_l):
                    return True

        # If the trace did not expose agent ownership, do not punish the finding.
        # The quote grounding is the primary signal in this MVP verifier.
        if not saw_known_agent:
            return True
        return False

    def _check_roles(self, finding: Finding) -> bool:
        return bool(finding.evidence) and all(
            (e.role or "").strip().lower() in self.allowed_roles for e in finding.evidence
        )

    @staticmethod
    def _contains_quote(text: str, quote: str) -> bool:
        if quote in text:
            return True
        normalized_text = EvidenceVerifier._normalize_for_matching(text)
        normalized_quote = EvidenceVerifier._normalize_for_matching(quote)
        return normalized_quote in normalized_text

    @staticmethod
    def _normalize_for_matching(value: str) -> str:
        value = value.replace("\\n", "\n")
        value = value.replace('\\"', '"')
        return re.sub(r"\s+", " ", value).strip()

    def _build_span_index(self, eval_input: EvaluationInput | RawTraceInput) -> dict[str, _SpanRecord]:
        span_index: dict[str, _SpanRecord] = {}

        if getattr(eval_input, "dialogue_history", None):
            for i, msg in enumerate(eval_input.dialogue_history or []):
                content = self._to_text(msg.content)
                agent = msg.role.value if isinstance(msg, DialogueMessage) else None
                span_id = f"message_{i}"
                span_index[span_id] = _SpanRecord(
                    span_id=span_id,
                    content=content,
                    agent=agent,
                    span_type="dialogue_message",
                )
                # Lightweight fallback: allow the LLM to cite message index 0, 1, ...
                span_index.setdefault(str(i), _SpanRecord(
                    span_id=str(i),
                    content=content,
                    agent=agent,
                    span_type="dialogue_message_index",
                ))

        if getattr(eval_input, "agent_responses", None):
            for i, response in enumerate(eval_input.agent_responses or []):
                content = self._to_text(response.content)
                agent = self._agent_from_metadata_or_id(response.metadata, response.response_id)
                span_index[response.response_id] = _SpanRecord(
                    span_id=response.response_id,
                    content=content,
                    agent=agent,
                    span_type="agent_response",
                )
                span_index.setdefault(str(i), _SpanRecord(
                    span_id=str(i),
                    content=content,
                    agent=agent,
                    span_type="agent_response_index",
                ))

        if getattr(eval_input, "agent_states", None):
            for i, state in enumerate(eval_input.agent_states or []):
                content_parts = [self._to_text(state.content)]
                agent = self._agent_from_metadata_or_id(state.metadata, state.state_id)
                if state.tool_call is not None:
                    content_parts.append(self._to_text(state.tool_call.model_dump(mode="json")))
                    if state.tool_call.id:
                        span_index[state.tool_call.id] = _SpanRecord(
                            span_id=state.tool_call.id,
                            content=self._to_text(state.tool_call.model_dump(mode="json")),
                            agent=agent,
                            span_type="tool_call",
                        )
                content = "\n".join(p for p in content_parts if p)
                span_index[state.state_id] = _SpanRecord(
                    span_id=state.state_id,
                    content=content,
                    agent=agent,
                    span_type=state.type.value,
                )
                span_index.setdefault(str(i), _SpanRecord(
                    span_id=str(i),
                    content=content,
                    agent=agent,
                    span_type="agent_state_index",
                ))

        if getattr(eval_input, "policies", None):
            for policy in eval_input.policies or []:
                span_index[policy.policy_id] = _SpanRecord(
                    span_id=policy.policy_id,
                    content=self._to_text(policy.model_dump(mode="json")),
                    span_type="policy",
                )

        if getattr(eval_input, "agents_tools_info", None):
            for tools_info in eval_input.agents_tools_info or []:
                if self.allow_agent_name_as_span_id:
                    span_index[tools_info.agent_name] = _SpanRecord(
                        span_id=tools_info.agent_name,
                        content=self._to_text(tools_info.model_dump(mode="json")),
                        agent=tools_info.agent_name,
                        span_type="agent_tools_info",
                    )
                for tool_call in tools_info.tools_called:
                    if tool_call.id:
                        span_index[tool_call.id] = _SpanRecord(
                            span_id=tool_call.id,
                            content=self._to_text(tool_call.model_dump(mode="json")),
                            agent=tools_info.agent_name,
                            span_type="tool_call",
                        )

        if getattr(eval_input, "agents_errors", None):
            for i, error in enumerate(eval_input.agents_errors or []):
                span_id = f"agent_error_{i}"
                span_index[span_id] = _SpanRecord(
                    span_id=span_id,
                    content=error.error_message or "",
                    agent=error.agent_name,
                    span_type="agent_error",
                )

        if self.allow_agent_name_as_span_id and getattr(eval_input, "agents_pool", None):
            for agent in eval_input.agents_pool.agents:
                span_index[agent.agent_name] = _SpanRecord(
                    span_id=agent.agent_name,
                    content=agent.instructions,
                    agent=agent.agent_name,
                    span_type="agent_description",
                )

        raw_trace = getattr(eval_input, "trace", None)
        if raw_trace:
            self._add_indexed_raw_trace_steps(span_index, raw_trace)
            self._add_raw_trace_ids(span_index, raw_trace)

        return span_index

    def _add_indexed_raw_trace_steps(self, span_index: dict[str, _SpanRecord], raw_trace: str) -> None:
        """Index raw trace blocks by numeric message id.

        Supports the preferred format produced by the launcher (`[0] ...`) and
        looser forms such as `Message 0: ...` or `Step 1 - ...`. Numeric ids are
        intended to be cited by LLM evaluators as `evidence[i].idx`.
        """
        for pattern in (self._INDEXED_TRACE_BLOCK_PATTERN, self._LOOSE_INDEXED_TRACE_BLOCK_PATTERN):
            for match in pattern.finditer(raw_trace):
                step_id = match.group(1)
                content = match.group(2).strip()
                if step_id and content:
                    # Raw formatted trace indices are the ids shown to the LLM,
                    # so they take precedence over unrelated internal list indices.
                    span_index[step_id] = _SpanRecord(
                        span_id=step_id,
                        content=content,
                        agent=self._infer_agent_from_raw_step(content),
                        span_type="raw_trace_step_index",
                    )

    def _add_raw_trace_ids(self, span_index: dict[str, _SpanRecord], raw_trace: str) -> None:
        for pattern in self._RAW_ID_PATTERNS:
            for match in pattern.finditer(raw_trace):
                span_id = match.group(1) if match.groups() else match.group(0)
                span_id = self._safe_unescape(span_id)
                if span_id and span_id not in span_index:
                    span_index[span_id] = _SpanRecord(
                        span_id=span_id,
                        content=raw_trace,
                        agent=self._infer_agent_from_id(span_id),
                        span_type="raw_trace_reference",
                    )

    @staticmethod
    def _raw_trace_text(eval_input: EvaluationInput | RawTraceInput) -> str | None:
        trace = getattr(eval_input, "trace", None)
        if trace:
            return trace
        return None

    @staticmethod
    def _infer_agent_from_raw_step(content: str) -> str | None:
        # Supports explicit prefixes produced by the example launcher, e.g.
        # `agent: Orchestrator`. Be conservative: do not treat arbitrary
        # heading-like lines such as `The visible text on the page is:` as agents.
        for pattern in (
            r"(?im)^\s*agent\s*:\s*([^\n]+)",
            r"(?im)^\s*agent_name\s*:\s*([^\n]+)",
            r"(?im)^\s*sender\s*:\s*([^\n]+)",
        ):
            match = re.search(pattern, content)
            if match:
                candidate = match.group(1).strip().strip('"\'')
                return EvidenceVerifier._clean_agent_candidate(candidate)

        # Also support compact block headers such as `WebSurfer:` or
        # `Orchestrator:` at the beginning of a raw indexed block. Require a
        # single identifier-like token to avoid false positives from OCR text.
        match = re.search(r"(?m)^\s*([A-Z][A-Za-z0-9_]{1,80})\s*:\s*", content[:300])
        if match:
            return EvidenceVerifier._clean_agent_candidate(match.group(1))
        return None

    @staticmethod
    def _clean_agent_candidate(candidate: str) -> str | None:
        candidate = candidate.strip().strip('"\'')
        if not candidate:
            return None
        # Reject sentence-like or page-heading values mistakenly captured from
        # raw OCR / browser text. Agent names should be short identifiers.
        if len(candidate) > 80 or "\n" in candidate:
            return None
        if len(candidate.split()) > 3:
            return None
        bad_prefixes = (
            "the visible text",
            "automatic ocr",
            "main content",
            "other content",
            "ui element",
            "metadata",
            "viewport",
        )
        if candidate.lower().startswith(bad_prefixes):
            return None
        return candidate

    @staticmethod
    def _agent_from_metadata_or_id(metadata: dict[str, Any] | None, span_id: str) -> str | None:
        if metadata:
            for key in ("agent", "agent_name", "name"):
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return EvidenceVerifier._infer_agent_from_id(span_id)

    @staticmethod
    def _infer_agent_from_id(span_id: str) -> str | None:
        if "_" not in span_id:
            return None
        prefix = span_id.rsplit("_", 1)[0]
        if prefix in {"span", "state", "response", "message", "agent_error"}:
            return None
        return prefix or None

    @staticmethod
    def _safe_unescape(value: str) -> str:
        try:
            return bytes(value, "utf-8").decode("unicode_escape")
        except Exception:
            return value

    @staticmethod
    def _to_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            return str(value)

    @staticmethod
    def _looks_like_metric_result(value: Any) -> bool:
        if isinstance(value, MetricResult):
            return True
        return isinstance(value, Mapping) and "metric_name" in value and "findings" in value

    @staticmethod
    def _coerce_metric_result(metric_result: MetricResult | Mapping[str, Any]) -> MetricResult:
        if isinstance(metric_result, MetricResult):
            return metric_result
        return MetricResult.model_validate(metric_result)
