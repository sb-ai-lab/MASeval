"""Deterministic evidence verification for LLM evaluator findings.

EvidenceVerifier does not decide whether an LLM finding is semantically correct.
It only checks whether the finding is grounded in the provided trace:

* cited idxs or zero-based message indices exist;
* quoted evidence can be found in the cited idxs;
* culprit-agent candidates are at least compatible with the cited evidence;
* evidence roles use an expected vocabulary.

The output is intentionally close to the schema used in the MASQUE Studio design:
``verified | weak | invalid`` + check-level diagnostics.

LLM mode
--------
When ``mode="llm"``, the verifier does not apply the deterministic grounding
rules. Instead it sends every finding of one metric (together with the trace
excerpts its cited evidence resolves to) to an LLM judge in a single batched
call. The judge returns one grounding verdict per finding. The batched call
keeps the cost comparable to a single evaluator run rather than one call per
finding. If the LLM call fails, the metric falls back to the deterministic
logic so the pipeline never breaks.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

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


class EvidenceLLMVerdict(BaseModel):
    """LLM grounding verdict for a single finding (batched LLM verifier)."""

    finding_index: int
    status: Literal["verified", "weak", "invalid"]
    explanation: str
    grounded_evidence_indices: list[int] = []


class EvidenceLLMBatchVerdict(BaseModel):
    """Batched LLM verdict: one per finding of a metric."""

    verdicts: list[EvidenceLLMVerdict]


EVIDENCE_VERIFIER_LLM_PROMPT = """\
You are a strict, deterministic-style evidence verifier for findings produced by \
other LLM evaluators of a multi-agent system (MAS) trace. Your ONLY job is to judge \
whether each finding is actually grounded in the cited trace excerpts. You do NOT \
decide whether the finding is semantically correct or important; only whether the \
cited evidence supports the claim.

For each finding you are given:
- the finding's problem description and the evaluator's claim,
- the cited evidence items, each with its `idx`, `role`, the evaluator's `claim`,
  the exact `quote`, and the `resolved_trace_excerpt` (the actual trace text the
  idx resolved to, or `UNRESOLVED` if the verifier could not locate it).

Return one verdict per finding. For each finding set:
- `status`:
  - "verified"  if the cited quotes/claims are clearly present in the resolved
    trace excerpts and the finding is well grounded;
  - "weak"      if at least some cited evidence is grounded but other citations are
    missing, vague, or only partially supported (the finding is still usable);
  - "invalid"   if none of the cited quotes can be found in the resolved excerpts,
    the idx is unresolved, or the finding has no usable evidence.
- `explanation`: one concise sentence explaining the verdict, citing which evidence
  was or was not found.
- `grounded_evidence_indices`: the `evidence_index` values (0-based within the
  finding's evidence list) whose quote/claim is actually present in its resolved
  trace excerpt. If you mark the finding "verified", this should usually include all
  of them. If "invalid", this is usually empty.

Be conservative: when in doubt about grounding, prefer "weak" over "verified", and
"invalid" only for gross grounding failures. Output STRICTLY the requested JSON."""


@dataclass(frozen=True)
class _IdxRecord:
    """Internal normalized view of one trace item that can be cited as evidence."""

    idx: str
    content: str
    agent: str | None = None
    idx_type: str | None = None




@dataclass(frozen=True)
class _ResolvedEvidenceItem:
    """Internal resolution result for one cited evidence item."""

    check: EvidenceItemCheck
    record: _IdxRecord | None = None

class EvidenceVerifier:
    """Verify whether LLM evaluator findings are trace-grounded.

    Parameters:
        allowed_roles: Accepted evidence roles. This is intentionally small and
            explicit, but can be extended per project.
        allow_agent_name_as_idx: If True, agent names are accepted as
            citeable ids. Defaults to False because MASQUE Studio should prefer
            concrete step/message indices or state/response ids over agent-level references.
        mode: Verification strategy. ``"deterministic"`` (default) applies the
            rule-based grounding checks; ``"llm"`` sends every finding of a metric
            to an LLM judge (batched per metric) and uses its verdicts. ``"llm"``
            requires ``model`` to be provided.
        model: An OpenAI-compatible chat model (``OpenAIChatModel`` or a model id
            string) used only when ``mode="llm"``.
    """

    MODES = ("deterministic", "llm")

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
        allow_agent_name_as_idx: bool = False,
        mode: str = "deterministic",
        model: OpenAIChatModel | str | None = None,
    ):
        if mode not in self.MODES:
            raise ValueError(f"Unknown EvidenceVerifier mode: {mode!r}; expected one of {self.MODES}")
        self.allowed_roles = {r.lower() for r in (allowed_roles or self.DEFAULT_ALLOWED_ROLES)}
        self.allow_agent_name_as_idx = allow_agent_name_as_idx
        self.mode = mode
        self.model = model
        self._llm_agent: Agent | None = None
        if self.mode == "llm":
            if self.model is None:
                raise ValueError("EvidenceVerifier(mode='llm') requires a `model`.")
            self._llm_agent = Agent(
                model=self.model,
                output_type=EvidenceLLMBatchVerdict,
                system_prompt=EVIDENCE_VERIFIER_LLM_PROMPT,
                retries=2,
            )

    def verify_metric_result(
        self,
        metric_result: MetricResult | Mapping[str, Any],
        eval_input: EvaluationInput | RawTraceInput,
    ) -> EvidenceVerificationMetricResult:
        """Verify all findings for one metric."""

        metric_result = self._coerce_metric_result(metric_result)
        idx_index = self._build_idx_index(eval_input)
        raw_trace_text = self._raw_trace_text(eval_input)

        verifications = [
            self.verify_finding(
                metric_name=metric_result.metric_name,
                finding=finding,
                finding_index=i,
                idx_index=idx_index,
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

    async def verify_all_async(
        self,
        metric_results: Mapping[str, MetricResult | Mapping[str, Any]],
        eval_input: EvaluationInput | RawTraceInput,
    ) -> dict[str, EvidenceVerificationMetricResult]:
        """Async variant of :meth:`verify_all`.

        In ``"llm"`` mode each metric is verified in a single batched LLM call;
        otherwise this is equivalent to :meth:`verify_all`.
        """

        verified: dict[str, EvidenceVerificationMetricResult] = {}
        for metric_name, result in metric_results.items():
            if not self._looks_like_metric_result(result):
                continue
            if self.mode == "llm":
                verified[metric_name] = await self.verify_metric_result_async(result, eval_input)
            else:
                verified[metric_name] = self.verify_metric_result(result, eval_input)
        return verified

    async def verify_metric_result_async(
        self,
        metric_result: MetricResult | Mapping[str, Any],
        eval_input: EvaluationInput | RawTraceInput,
    ) -> EvidenceVerificationMetricResult:
        """Async variant of :meth:`verify_metric_result` (LLM mode only)."""

        if self.mode != "llm":
            return self.verify_metric_result(metric_result, eval_input)

        metric_result = self._coerce_metric_result(metric_result)
        idx_index = self._build_idx_index(eval_input)
        raw_trace_text = self._raw_trace_text(eval_input)
        return await self._verify_metric_llm(
            metric_result=metric_result,
            idx_index=idx_index,
            raw_trace_text=raw_trace_text,
        )

    async def _verify_metric_llm(
        self,
        metric_result: MetricResult,
        idx_index: Mapping[str, _IdxRecord],
        raw_trace_text: str | None,
    ) -> EvidenceVerificationMetricResult:
        """Verify all findings of one metric with a single batched LLM call.

        Falls back to the deterministic verifier for the whole metric if the LLM
        call fails or the judge does not return one verdict per finding.
        """

        assert self._llm_agent is not None
        findings = list(metric_result.findings)

        if not findings:
            return EvidenceVerificationMetricResult(
                metric_name=metric_result.metric_name, verifications=[]
            )

        payload = self._build_llm_finding_payload(findings, idx_index, raw_trace_text)
        prompt = (
            "Verify the grounding of the following findings from metric "
            f"'{metric_result.metric_name}'. Return exactly one verdict per finding, "
            "using the finding_index values given.\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        )

        try:
            response = await self._llm_agent.run(prompt)
            verdicts = {v.finding_index: v for v in response.output.verdicts}
            if set(verdicts) != set(range(len(findings))):
                raise ValueError(
                    f"LLM returned verdicts for {sorted(verdicts)} but expected "
                    f"{list(range(len(findings)))}"
                )
        except Exception as exc:  # pragma: no cover - network/model dependent
            print(
                f"[EvidenceVerifier:llm] fell back to deterministic for "
                f"'{metric_result.metric_name}': {exc}"
            )
            deterministic = self._verify_metric_result_deterministic(
                metric_result, idx_index, raw_trace_text
            )
            return EvidenceVerificationMetricResult(
                metric_name=metric_result.metric_name,
                verifications=[
                    self._mark_deterministic_fallback(v) for v in deterministic.verifications
                ],
            )

        verifications = [
            self._verification_from_llm_verdict(
                metric_name=metric_result.metric_name,
                finding=finding,
                finding_index=i,
                verdict=verdicts[i],
            )
            for i, finding in enumerate(findings)
        ]
        return EvidenceVerificationMetricResult(
            metric_name=metric_result.metric_name, verifications=verifications
        )

    def _verify_metric_result_deterministic(
        self,
        metric_result: MetricResult,
        idx_index: Mapping[str, _IdxRecord],
        raw_trace_text: str | None,
    ) -> EvidenceVerificationMetricResult:
        """Synchronous deterministic verification (shared by fallback path)."""

        verifications = [
            self.verify_finding(
                metric_name=metric_result.metric_name,
                finding=finding,
                finding_index=i,
                idx_index=idx_index,
                raw_trace_text=raw_trace_text,
            )
            for i, finding in enumerate(metric_result.findings)
        ]
        return EvidenceVerificationMetricResult(
            metric_name=metric_result.metric_name, verifications=verifications
        )

    @staticmethod
    def _mark_deterministic_fallback(
        verification: EvidenceVerificationResult,
    ) -> EvidenceVerificationResult:
        return verification.model_copy(
            update={
                "verifier_method": "deterministic_fallback",
                "verifier_explanation": (
                    f"(LLM verifier unavailable; deterministic fallback) "
                    f"{verification.verifier_explanation}"
                ),
            }
        )

    def _build_llm_finding_payload(
        self,
        findings: list[Finding],
        idx_index: Mapping[str, _IdxRecord],
        raw_trace_text: str | None,
    ) -> list[dict[str, Any]]:
        """Serialize findings + resolved trace excerpts for the LLM judge."""

        payload: list[dict[str, Any]] = []
        for finding_index, finding in enumerate(findings):
            resolutions = self._resolve_evidence_items(
                finding=finding, idx_index=idx_index, raw_trace_text=raw_trace_text
            )
            evidence_items = []
            for item in resolutions:
                excerpt = item.record.content if item.record is not None else None
                evidence_items.append(
                    {
                        "evidence_index": item.check.evidence_index,
                        "idx": item.check.idx,
                        "resolved_idx": item.check.resolved_idx,
                        "role": (finding.evidence[item.check.evidence_index].role
                                 if item.check.evidence_index < len(finding.evidence) else None),
                        "claim": (finding.evidence[item.check.evidence_index].claim
                                  if item.check.evidence_index < len(finding.evidence) else None),
                        "quote": (finding.evidence[item.check.evidence_index].quote
                                  if item.check.evidence_index < len(finding.evidence) else None),
                        "resolved_trace_excerpt": excerpt if excerpt is not None else "UNRESOLVED",
                    }
                )
            payload.append(
                {
                    "finding_index": finding_index,
                    "severity_estimate": finding.severity_estimate.value,
                    "confidence_estimate": finding.confidence_estimate.value,
                    "problem_description": finding.problem_description,
                    "culprit_agent_candidates": [
                        c.agent for c in finding.culprit_agent_candidates
                    ],
                    "evidence": evidence_items,
                }
            )
        return payload

    def _verification_from_llm_verdict(
        self,
        metric_name: str,
        finding: Finding,
        finding_index: int,
        verdict: EvidenceLLMVerdict,
    ) -> EvidenceVerificationResult:
        """Map an LLM verdict into the standard EvidenceVerificationResult schema."""

        grounded = set(verdict.grounded_evidence_indices or [])
        if not finding.evidence:
            grounded = set()

        evidence_item_checks: list[EvidenceItemCheck] = []
        for evidence_index, evidence in enumerate(finding.evidence):
            is_grounded = evidence_index in grounded
            if not grounded and verdict.status in ("verified", "weak"):
                # LLM did not enumerate grounded indices but approved the finding:
                # treat all cited evidence as grounded.
                is_grounded = True
            evidence_item_checks.append(
                EvidenceItemCheck(
                    evidence_index=evidence_index,
                    idx=str(evidence.idx),
                    idx_exists=is_grounded,
                    quote_found=is_grounded,
                    role_plausible=True,
                    resolution_strategy="llm_verifier",
                    problem=None if is_grounded else "quote/claim not grounded per LLM verifier",
                )
            )

        checks = EvidenceChecks(
            all_idxs_exist=all(c.idx_exists for c in evidence_item_checks) if evidence_item_checks else False,
            quotes_found_in_idxs=all(c.quote_found for c in evidence_item_checks) if evidence_item_checks else False,
            culprit_agent_matches_evidence=True,
            idx_roles_are_plausible=True,
        )
        evidence_status = EvidenceStatus(verdict.status)
        usable_for_diagnosis = evidence_status != EvidenceStatus.INVALID
        explanation = f"(LLM verifier) {verdict.explanation}"

        return EvidenceVerificationResult(
            metric_name=metric_name,
            finding_index=finding_index,
            evidence_status=evidence_status,
            evidence_checks=checks,
            evidence_item_checks=evidence_item_checks,
            usable_for_diagnosis=usable_for_diagnosis,
            verifier_explanation=explanation,
            verifier_method="llm",
        )

    def verify_finding(
        self,
        metric_name: str,
        finding: Finding,
        finding_index: int,
        idx_index: Mapping[str, _IdxRecord],
        raw_trace_text: str | None = None,
    ) -> EvidenceVerificationResult:
        """Verify one LLM finding."""

        evidence_resolutions = self._resolve_evidence_items(
            finding=finding,
            idx_index=idx_index,
            raw_trace_text=raw_trace_text,
        )
        evidence_item_checks = [r.check for r in evidence_resolutions]
        resolved_records = [r.record for r in evidence_resolutions if r.record is not None]
        checks = EvidenceChecks(
            all_idxs_exist=bool(finding.evidence) and all(i.idx_exists for i in evidence_item_checks),
            quotes_found_in_idxs=bool(finding.evidence) and all(i.quote_found for i in evidence_item_checks),
            culprit_agent_matches_evidence=self._check_culprit_agent_matches(finding, resolved_records),
            idx_roles_are_plausible=bool(finding.evidence) and all(i.role_plausible for i in evidence_item_checks),
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
            or not checks.idx_roles_are_plausible
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
            return "All cited idxs exist, quoted evidence matches the trace, and the finding is usable for diagnosis."

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
            if "quotes_found_in_idxs" in failed:
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

        if "all_idxs_exist" in failed and "quotes_found_in_idxs" not in failed:
            return (
                "The quoted evidence was found in the raw trace, but at least one cited idx "
                "is not resolvable in the normalized idx index. Treat this finding as weak: "
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
        idx_index: Mapping[str, _IdxRecord],
        raw_trace_text: str | None,
    ) -> list[_ResolvedEvidenceItem]:
        """Resolve and verify each evidence item.

        Numeric ``idx`` values are treated as message indices. The verifier
        first tries the exact index, then ``index - 1`` and ``index + 1``. This
        handles the common mismatch between zero-based and one-based numbering
        in LLM outputs while still recording the resolved id explicitly.
        """
        resolved_items: list[_ResolvedEvidenceItem] = []
        for evidence_index, evidence in enumerate(finding.evidence):
            idx_val = str(evidence.idx).strip()
            role_plausible = (evidence.role or "").strip().lower() in self.allowed_roles
            quote = evidence.quote or ""

            record, resolved_idx, resolution_strategy = self._resolve_idx_record(
                idx_val=idx_val,
                quote=quote,
                idx_index=idx_index,
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
                # treat the idx as softly resolved instead of failing the finding.
                quote_found = True
                used_raw_trace_fallback = True
                raw_record = self._record_from_raw_quote(
                    idx_val=idx_val,
                    quote=quote,
                    raw_trace_text=raw_trace_text,
                )
                if raw_record is not None:
                    record = raw_record
                    resolution_strategy = "raw_quote_numeric_soft" if self._is_int_like(idx_val) else "raw_quote_soft"

            # In soft mode, a record obtained from raw-quote fallback is still a
            # resolved evidence location. We reserve idx_exists=false for cases
            # where neither an indexed block nor a raw quote window could be found.
            idx_exists = record is not None

            problems: list[str] = []
            if not idx_exists:
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
                    problems.append("quote was not found in the cited idx, neighboring index, or raw trace")
            if not role_plausible:
                problems.append(f"unsupported evidence role: {evidence.role!r}")
            if idx_exists and resolved_idx != idx_val:
                problems.append(
                    f"idx {idx_val!r} was resolved as {resolved_idx!r} using {resolution_strategy}"
                )

            check = EvidenceItemCheck(
                evidence_index=evidence_index,
                idx=idx_val,
                idx_exists=idx_exists,
                quote_found=quote_found,
                role_plausible=role_plausible,
                used_raw_trace_fallback=used_raw_trace_fallback,
                resolved_idx=resolved_idx if resolved_idx != idx_val else None,
                resolved_agent=record.agent if record is not None else None,
                resolution_strategy=resolution_strategy,
                problem="; ".join(problems) if problems else None,
            )
            resolved_items.append(_ResolvedEvidenceItem(check=check, record=record))
        return resolved_items

    def _resolve_idx_record(
        self,
        idx_val: str,
        quote: str,
        idx_index: Mapping[str, _IdxRecord],
        raw_trace_text: str | None,
    ) -> tuple[_IdxRecord | None, str, str]:
        """Resolve a cited idx to a record.

        For numeric ids, try exact index and then +/-1. Prefer the candidate
        whose content contains the quote; otherwise fall back to exact index if
        it exists.
        """
        if self._is_int_like(idx_val):
            idx = int(idx_val)
            candidates = [idx, idx - 1, idx + 1]
            existing: list[tuple[str, _IdxRecord, str]] = []
            for candidate in candidates:
                if candidate < 0:
                    continue
                candidate_id = str(candidate)
                record = idx_index.get(candidate_id)
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
                    if candidate_id == idx_val:
                        return record, candidate_id, strategy
                candidate_id, record, strategy = existing[0]
                return record, candidate_id, strategy

        if idx_val in idx_index:
            return idx_index[idx_val], idx_val, "exact"

        return None, idx_val, "unresolved"

    @staticmethod
    def _is_int_like(value: str) -> bool:
        try:
            int(str(value).strip())
            return True
        except (TypeError, ValueError):
            return False

    def _record_from_raw_quote(
        self,
        idx_val: str,
        quote: str,
        raw_trace_text: str | None,
    ) -> _IdxRecord | None:
        if not raw_trace_text or not quote.strip() or not self._contains_quote(raw_trace_text, quote):
            return None
        content = self._raw_indexed_block_containing_quote(raw_trace_text, quote)
        if content is None:
            content = self._raw_window_around_quote(raw_trace_text, quote)
        return _IdxRecord(
            idx=idx_val,
            content=content,
            agent=self._infer_agent_from_raw_step(content),
            idx_type="raw_quote_soft_fallback",
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
        resolved_records: list[_IdxRecord],
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
            idx_l = record.idx.lower()
            record_agent = (record.agent or "").strip()
            record_agent_l = record_agent.lower()

            if record_agent_l:
                saw_known_agent = True
                for agent_l in candidate_agents:
                    if agent_l == record_agent_l or agent_l in record_agent_l or record_agent_l in agent_l:
                        return True

            # Fallbacks for raw traces that include agent labels in the text window.
            for agent_l in candidate_agents:
                if agent_l in idx_l:
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

    def _build_idx_index(self, eval_input: EvaluationInput | RawTraceInput) -> dict[str, _IdxRecord]:
        idx_index: dict[str, _IdxRecord] = {}

        if getattr(eval_input, "dialogue_history", None):
            for i, msg in enumerate(eval_input.dialogue_history or []):
                content = self._to_text(msg.content)
                agent = msg.role.value if isinstance(msg, DialogueMessage) else None
                idx_val = f"message_{i}"
                idx_index[idx_val] = _IdxRecord(
                    idx=idx_val,
                    content=content,
                    agent=agent,
                    idx_type="dialogue_message",
                )
                # Lightweight fallback: allow the LLM to cite message index 0, 1, ...
                idx_index.setdefault(str(i), _IdxRecord(
                    idx=str(i),
                    content=content,
                    agent=agent,
                    idx_type="dialogue_message_index",
                ))

        if getattr(eval_input, "agent_responses", None):
            for i, response in enumerate(eval_input.agent_responses or []):
                content = self._to_text(response.content)
                agent = self._agent_from_metadata_or_id(response.metadata, response.response_id)
                idx_index[response.response_id] = _IdxRecord(
                    idx=response.response_id,
                    content=content,
                    agent=agent,
                    idx_type="agent_response",
                )
                idx_index.setdefault(str(i), _IdxRecord(
                    idx=str(i),
                    content=content,
                    agent=agent,
                    idx_type="agent_response_index",
                ))

        if getattr(eval_input, "agent_states", None):
            for i, state in enumerate(eval_input.agent_states or []):
                content_parts = [self._to_text(state.content)]
                agent = self._agent_from_metadata_or_id(state.metadata, state.state_id)
                if state.tool_call is not None:
                    content_parts.append(self._to_text(state.tool_call.model_dump(mode="json")))
                    if state.tool_call.id:
                        idx_index[state.tool_call.id] = _IdxRecord(
                            idx=state.tool_call.id,
                            content=self._to_text(state.tool_call.model_dump(mode="json")),
                            agent=agent,
                            idx_type="tool_call",
                        )
                content = "\n".join(p for p in content_parts if p)
                idx_index[state.state_id] = _IdxRecord(
                    idx=state.state_id,
                    content=content,
                    agent=agent,
                    idx_type=state.type.value,
                )
                idx_index.setdefault(str(i), _IdxRecord(
                    idx=str(i),
                    content=content,
                    agent=agent,
                    idx_type="agent_state_index",
                ))

        if getattr(eval_input, "policies", None):
            for policy in eval_input.policies or []:
                idx_index[policy.policy_id] = _IdxRecord(
                    idx=policy.policy_id,
                    content=self._to_text(policy.model_dump(mode="json")),
                    idx_type="policy",
                )

        if getattr(eval_input, "agents_tools_info", None):
            for tools_info in eval_input.agents_tools_info or []:
                if self.allow_agent_name_as_idx:
                    idx_index[tools_info.agent_name] = _IdxRecord(
                        idx=tools_info.agent_name,
                        content=self._to_text(tools_info.model_dump(mode="json")),
                        agent=tools_info.agent_name,
                        idx_type="agent_tools_info",
                    )
                for tool_call in tools_info.tools_called:
                    if tool_call.id:
                        idx_index[tool_call.id] = _IdxRecord(
                            idx=tool_call.id,
                            content=self._to_text(tool_call.model_dump(mode="json")),
                            agent=tools_info.agent_name,
                            idx_type="tool_call",
                        )

        if getattr(eval_input, "agents_errors", None):
            for i, error in enumerate(eval_input.agents_errors or []):
                idx_val = f"agent_error_{i}"
                idx_index[idx_val] = _IdxRecord(
                    idx=idx_val,
                    content=error.error_message or "",
                    agent=error.agent_name,
                    idx_type="agent_error",
                )

        if self.allow_agent_name_as_idx and getattr(eval_input, "agents_pool", None):
            for agent in eval_input.agents_pool.agents:
                idx_index[agent.agent_name] = _IdxRecord(
                    idx=agent.agent_name,
                    content=agent.instructions,
                    agent=agent.agent_name,
                    idx_type="agent_description",
                )

        raw_trace = getattr(eval_input, "trace", None)
        if raw_trace:
            self._add_indexed_raw_trace_steps(idx_index, raw_trace)
            self._add_raw_trace_ids(idx_index, raw_trace)

        return idx_index

    def _add_indexed_raw_trace_steps(self, idx_index: dict[str, _IdxRecord], raw_trace: str) -> None:
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
                    idx_index[step_id] = _IdxRecord(
                        idx=step_id,
                        content=content,
                        agent=self._infer_agent_from_raw_step(content),
                        idx_type="raw_trace_step_index",
                    )

    def _add_raw_trace_ids(self, idx_index: dict[str, _IdxRecord], raw_trace: str) -> None:
        for pattern in self._RAW_ID_PATTERNS:
            for match in pattern.finditer(raw_trace):
                idx_val = match.group(1) if match.groups() else match.group(0)
                idx_val = self._safe_unescape(idx_val)
                if idx_val and idx_val not in idx_index:
                    idx_index[idx_val] = _IdxRecord(
                        idx=idx_val,
                        content=raw_trace,
                        agent=self._infer_agent_from_id(idx_val),
                        idx_type="raw_trace_reference",
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
    def _agent_from_metadata_or_id(metadata: dict[str, Any] | None, idx_val: str) -> str | None:
        if metadata:
            for key in ("agent", "agent_name", "name"):
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return EvidenceVerifier._infer_agent_from_id(idx_val)

    @staticmethod
    def _infer_agent_from_id(idx_val: str) -> str | None:
        if "_" not in idx_val:
            return None
        prefix = idx_val.rsplit("_", 1)[0]
        if prefix in {"idx", "span", "state", "response", "message", "agent_error"}:
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
