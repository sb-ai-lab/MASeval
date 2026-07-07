"""Diagnostic report aggregation for evidence-grounded LLM evaluator outputs.

The module consumes the JSON object produced by the findings evaluators plus
``EvidenceVerifier`` and builds one compact report for downstream review/UI:

* answer status: whether the final answer matches a reference, when available;
* diagnostic status: severity counts, primary culprit, primary failure type;
* diagnostic report: issue list, problematic agents, problematic spans, review targets.

The aggregation intentionally uses ``EvidenceVerifier`` as a gate. Findings with
``usable_for_diagnosis=false`` are not used for the main diagnostic counts, but
are kept as review targets. ``weak`` findings are treated as usable, because in
this pipeline they mean "grounded enough, but approximate" rather than "discard".
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Mapping

SeverityName = str
MetricName = str

DEPRECATED_METRIC_NAMES = {
    "mas_task_completion",
    "MAS_TASK_COMPLETION",
}

NON_FINDING_KEYS = {
    "evidence_verification",
    "report",
    "status",
    "diagnostic_report",
    "gt",
    "label_answer",
    "predicted_answer",
    "reference_answer",
    "answer_status",
    "final_answer_verification",
}

SEVERITY_WEIGHT: dict[str, float] = {
    "critical": 3.0,
    "major": 2.0,
    "minor": 1.0,
}

EVIDENCE_STATUS_WEIGHT: dict[str, float] = {
    "verified": 1.0,
    "weak": 0.7,
    "invalid": 0.0,
}

METRIC_TO_FAILURE_TYPE: dict[str, str] = {
    "observation_alignment": "observation_alignment_error",
    "policy_alignment": "policy_alignment_error",
    "state_consistency": "state_inconsistency",
    "tool_selection": "wrong_tool_selection",
    "tool_parameter_extraction": "tool_parameter_error",
    "task_completeness": "task_incomplete",
    "mas_planning": "planning_failure",
    "mas_complexity": "system_complexity_failure",
    "mas_task_transfer": "false_handoff",
    "mas_roles_distribution": "role_distribution_failure",
    "tool_performance": "tool_execution_failure",
    "prompt_quality": "prompt_quality_issue",
}

ANSWER_KEY_CANDIDATES = (
    "predicted_answer",
    "model_answer",
    "final_answer",
    "answer",
)

REFERENCE_KEY_CANDIDATES = (
    "reference_answer",
    "ground_truth_answer",
    "correct_answer",
    "gold_answer",
    "ground_truth",
)


def build_evaluation_report(
    evaluation: Mapping[str, Any],
    *,
    predicted_answer: Any | None = None,
    reference_answer: Any | None = None,
    verification_mode: str | None = None,
    verifier_mode: str = "soft",
) -> dict[str, Any]:
    """Build a compact final report from metric findings and evidence checks.

    Args:
        evaluation: A full per-task JSON object. It may contain metric outputs
            under metric-name keys and an ``evidence_verification`` key produced
            by :class:`EvidenceVerifier`.
        predicted_answer: Optional final answer produced by the evaluated agent.
            If omitted, the aggregator tries to extract ``FINAL ANSWER: ...``
            from findings, then falls back to top-level answer-like keys.
        reference_answer: Optional reference / gold answer. If omitted, the
            aggregator tries common top-level reference keys.
        verification_mode: Optional explicit mode. If omitted, it is inferred as
            ``reference_based`` when a reference is available, ``trace_grounded``
            when only a predicted answer is available, otherwise ``unavailable``.
        verifier_mode: EvidenceVerifier gating for which findings count toward the
            diagnostic predictions -- ``"none"`` (all findings), ``"strict"``
            (only ``verified``), or ``"soft"`` (``verified``/``weak``; ``invalid``
            to review targets). Default ``"soft"`` matches prior behavior.

    Returns:
        A JSON-serializable dict with ``status`` and ``diagnostic_report``.
    """

    predicted_answer = _infer_predicted_answer(evaluation, predicted_answer)
    reference_answer = _infer_reference_answer(evaluation, reference_answer)
    answer_status = _answer_status_from_final_answer_verification(
        evaluation,
        predicted_answer=predicted_answer,
        reference_answer=reference_answer,
    )
    if answer_status is None:
        answer_status = _answer_status_from_existing_report(
            evaluation,
            predicted_answer=predicted_answer,
            reference_answer=reference_answer,
        )
    if answer_status is None:
        answer_status = _build_answer_status(
            predicted_answer=predicted_answer,
            reference_answer=reference_answer,
            verification_mode=verification_mode,
        )

    evidence_by_metric = _verification_index(evaluation.get("evidence_verification", {}))

    issues: list[dict[str, Any]] = []
    review_targets: list[dict[str, Any]] = []
    # Ranking weights are used internally only for ordering/primary choice.
    # They are intentionally not exposed in the public report schema.
    agent_scores: dict[str, float] = defaultdict(float)
    agent_counts: dict[str, int] = defaultdict(int)
    span_scores: dict[str, float] = defaultdict(float)
    span_counts: dict[str, int] = defaultdict(int)
    metric_scores: dict[str, float] = defaultdict(float)
    severity_counts: Counter[str] = Counter()

    for metric_name, metric_result in _iter_metric_results(evaluation):
        verifications = evidence_by_metric.get(metric_name, {})
        findings = metric_result.get("findings") or []
        for finding_index, finding in enumerate(findings):
            verification = verifications.get(finding_index)
            finding_summary = _summarize_finding(
                metric_name=metric_name,
                finding_index=finding_index,
                finding=finding,
                verification=verification,
                verifier_mode=verifier_mode,
            )

            if finding_summary["usable_for_diagnosis"]:
                issues.append(finding_summary)
                severity = finding_summary["severity_estimate"]
                severity_counts[severity] += 1
                base_score = _issue_score(finding_summary)
                metric_scores[metric_name] += base_score

                for agent in finding_summary["culprit_agents"]:
                    agent_scores[agent] += base_score
                    agent_counts[agent] += 1

                for span_id in finding_summary["problematic_spans"]:
                    span_scores[span_id] += base_score
                    span_counts[span_id] += 1
            else:
                review_targets.append(finding_summary)

    problematic_agents = _rank_agents(agent_scores, agent_counts)
    problematic_spans = _rank_spans(span_scores, span_counts)
    primary_metric = _top_key(metric_scores)
    primary_culprit_agent = problematic_agents[0]["agent"] if problematic_agents else None
    first_problem_span = _first_span(problematic_spans)

    diagnostic_status = {
        "verdict": _diagnostic_verdict(issues, review_targets),
        "critical_issues": int(severity_counts.get("critical", 0)),
        "major_issues": int(severity_counts.get("major", 0)),
        "minor_issues": int(severity_counts.get("minor", 0)),
        "problematic_agents_count": len(problematic_agents),
        "problematic_spans_count": len(problematic_spans),
        "primary_failure_type": METRIC_TO_FAILURE_TYPE.get(primary_metric, primary_metric),
        "primary_culprit_agent": primary_culprit_agent,
        "first_problem_span": first_problem_span,
    }

    review_required, review_reason = _review_status(answer_status, issues, review_targets)

    return {
        "status": {
            "answer_status": answer_status,
            "diagnostic_status": diagnostic_status,
            "review_status": {
                "required": review_required,
                "reason": review_reason,
            },
        },
        "diagnostic_report": {
            "problematic_agents": problematic_agents,
            "problematic_spans": problematic_spans,
            "issues": issues,
            "review_targets": review_targets,
        },
    }


def build_report_from_file(
    input_path: str,
    *,
    output_path: str | None = None,
    predicted_answer: Any | None = None,
    reference_answer: Any | None = None,
    verification_mode: str | None = None,
) -> dict[str, Any]:
    """Load a per-task JSON file, build report, and optionally save it.

    This helper is intentionally dependency-free, so it can be used from small
    scripts without constructing pydantic models.
    """

    import json
    from pathlib import Path

    path = Path(input_path)
    evaluation = json.loads(path.read_text())
    report = build_evaluation_report(
        evaluation,
        predicted_answer=predicted_answer,
        reference_answer=reference_answer,
        verification_mode=verification_mode,
    )
    if output_path:
        Path(output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _iter_metric_results(evaluation: Mapping[str, Any]):
    for metric_name, value in evaluation.items():
        if metric_name in NON_FINDING_KEYS:
            continue
        if metric_name in DEPRECATED_METRIC_NAMES:
            continue
        if not isinstance(value, Mapping):
            continue
        if "metric_name" not in value or "findings" not in value:
            continue
        actual_metric_name = str(value.get("metric_name") or metric_name)
        if actual_metric_name in DEPRECATED_METRIC_NAMES:
            continue
        yield metric_name, value


def _verification_index(evidence_verification: Any) -> dict[str, dict[int, Mapping[str, Any]]]:
    index: dict[str, dict[int, Mapping[str, Any]]] = {}
    if not isinstance(evidence_verification, Mapping):
        return index

    for metric_name, metric_verification in evidence_verification.items():
        if metric_name in DEPRECATED_METRIC_NAMES:
            continue
        if not isinstance(metric_verification, Mapping):
            continue
        verifications = metric_verification.get("verifications") or []
        metric_index: dict[int, Mapping[str, Any]] = {}
        for item in verifications:
            if not isinstance(item, Mapping):
                continue
            try:
                finding_index = int(item.get("finding_index"))
            except (TypeError, ValueError):
                continue
            metric_index[finding_index] = item
        index[metric_name] = metric_index
    return index


def _is_usable(
    verifier_mode: str,
    verification: Mapping[str, Any] | None,
    evidence_status: str,
) -> bool:
    """Decide whether a finding counts toward the main diagnostic predictions,
    under one of three EvidenceVerifier ablation settings:

    * ``"none"``   -- no verifier: every LLM finding is relevant.
    * ``"strict"`` -- only ``verified`` findings are relevant.
    * ``"soft"``   -- ``verified``/``weak`` are relevant; ``invalid`` (and any
      finding without verifier output) goes to review targets. This is the
      default and reproduces the pre-ablation behavior.
    """
    if verifier_mode == "none":
        return True
    # strict / soft both require verifier output to admit a finding.
    if verification is None:
        return False
    if verifier_mode == "strict":
        return evidence_status == "verified"
    # soft
    return evidence_status in ("verified", "weak")


def _summarize_finding(
    *,
    metric_name: str,
    finding_index: int,
    finding: Mapping[str, Any],
    verification: Mapping[str, Any] | None,
    verifier_mode: str = "soft",
) -> dict[str, Any]:
    severity = str(finding.get("severity_estimate") or "major").lower()
    confidence = str(finding.get("confidence_estimate") or "medium").lower()
    evidence_status = str((verification or {}).get("evidence_status") or "unverified").lower()

    evidence_item_checks = list((verification or {}).get("evidence_item_checks") or [])
    grounded_evidence_indices = {
        int(item.get("evidence_index"))
        for item in evidence_item_checks
        if isinstance(item, Mapping) and item.get("quote_found")
    }

    usable_for_diagnosis = _is_usable(verifier_mode, verification, evidence_status)

    culprit_agents = _extract_culprit_agents(finding)
    problematic_spans = _extract_problematic_spans(finding, evidence_item_checks)

    return {
        "metric_name": metric_name,
        "finding_index": finding_index,
        "severity_estimate": severity,
        "confidence_estimate": confidence,
        "evidence_status": evidence_status,
        "usable_for_diagnosis": usable_for_diagnosis,
        "culprit_agents": culprit_agents,
        "problematic_spans": problematic_spans,
        "grounded_evidence_count": len(grounded_evidence_indices),
        "total_evidence_count": len(finding.get("evidence") or []),
        "problem_description": finding.get("problem_description"),
        "suggested_fix": finding.get("suggested_fix"),
        "needs_human_review": bool(finding.get("needs_human_review", False)),
        "verifier_explanation": (verification or {}).get("verifier_explanation"),
    }


def _extract_culprit_agents(finding: Mapping[str, Any]) -> list[str]:
    agents: list[str] = []
    for candidate in finding.get("culprit_agent_candidates") or []:
        if not isinstance(candidate, Mapping):
            continue
        agent = candidate.get("agent")
        if isinstance(agent, str) and agent.strip():
            agents.append(agent.strip())
    return _unique_preserving_order(agents)


def _extract_problematic_spans(
    finding: Mapping[str, Any],
    evidence_item_checks: list[Any],
) -> list[str]:
    spans: list[str] = []
    evidence = finding.get("evidence") or []

    if evidence_item_checks:
        for item in evidence_item_checks:
            if not isinstance(item, Mapping):
                continue
            # Use grounded citations for primary span lists. Missing citations are
            # kept in review_targets through the full issue summary.
            if not (item.get("quote_found") or item.get("span_exists")):
                continue
            span_id = item.get("resolved_span_id") or item.get("span_id")
            if span_id is not None:
                spans.append(str(span_id))
        return _unique_preserving_order(spans)

    for item in evidence:
        if isinstance(item, Mapping) and item.get("span_id") is not None:
            spans.append(str(item.get("span_id")))
    return _unique_preserving_order(spans)


def _issue_score(issue: Mapping[str, Any]) -> float:
    severity_weight = SEVERITY_WEIGHT.get(str(issue.get("severity_estimate", "major")).lower(), 2.0)
    evidence_weight = EVIDENCE_STATUS_WEIGHT.get(str(issue.get("evidence_status", "weak")).lower(), 0.7)
    confidence = str(issue.get("confidence_estimate") or "medium").lower()
    confidence_weight = {"high": 1.0, "medium": 0.75, "low": 0.5}.get(confidence, 0.75)
    return severity_weight * evidence_weight * confidence_weight


def _rank_agents(agent_scores: Mapping[str, float], agent_counts: Mapping[str, int]) -> list[dict[str, Any]]:
    return [
        {"agent": agent, "findings_count": int(agent_counts.get(agent, 0))}
        for agent, _score in sorted(
            agent_scores.items(),
            key=lambda x: (-int(agent_counts.get(x[0], 0)), -x[1], x[0].lower()),
        )
    ]


def _rank_spans(span_scores: Mapping[str, float], span_counts: Mapping[str, int]) -> list[dict[str, Any]]:
    return [
        {"span_id": span_id, "findings_count": int(span_counts.get(span_id, 0))}
        for span_id, _score in sorted(
            span_scores.items(),
            key=lambda x: (-int(span_counts.get(x[0], 0)), -x[1], _span_sort_key(x[0])),
        )
    ]


def _top_key(scores: Mapping[str, float]) -> str | None:
    if not scores:
        return None
    return max(scores.items(), key=lambda x: (x[1], x[0]))[0]


def _first_span(problematic_spans: list[Mapping[str, Any]]) -> str | None:
    if not problematic_spans:
        return None
    return min((str(item["span_id"]) for item in problematic_spans), key=_span_sort_key)


def _span_sort_key(span_id: str) -> tuple[int, int | str]:
    text = str(span_id)
    if text.isdigit():
        return (0, int(text))
    match = re.search(r"(\d+)", text)
    if match:
        return (1, int(match.group(1)))
    return (2, text)


def _diagnostic_verdict(issues: list[dict[str, Any]], review_targets: list[dict[str, Any]]) -> str:
    if issues:
        return "issues_found"
    if review_targets:
        return "needs_review"
    return "clean"


def _review_status(
    answer_status: Mapping[str, Any],
    issues: list[dict[str, Any]],
    review_targets: list[dict[str, Any]],
) -> tuple[bool, str | None]:
    if answer_status.get("verdict") == "needs_review":
        return True, "Answer verification requires manual review."
    if any(issue.get("needs_human_review") for issue in issues):
        return True, "At least one usable diagnostic finding explicitly requested human review."
    if not issues and review_targets:
        return True, "Only ungrounded or unverified findings are available."
    return False, None


def _build_answer_status(
    *,
    predicted_answer: Any | None,
    reference_answer: Any | None,
    verification_mode: str | None,
) -> dict[str, Any]:
    if verification_mode is None:
        if reference_answer is not None:
            verification_mode = "reference_based"
        elif predicted_answer is not None:
            verification_mode = "trace_grounded"
        else:
            verification_mode = "unavailable"

    if predicted_answer is None and reference_answer is None:
        verdict = "unknown"
        reason = "No predicted answer or reference answer is available."
    elif reference_answer is None:
        verdict = "unknown"
        reason = "Predicted answer was found, but no reference answer is available."
    elif predicted_answer is None:
        verdict = "unknown"
        reason = "Reference answer is available, but predicted answer could not be inferred."
    elif _answers_match(predicted_answer, reference_answer):
        verdict = "correct"
        reason = "Predicted answer matches the reference answer."
    else:
        verdict = "incorrect"
        reason = "Predicted answer does not match the reference answer."

    return {
        "verdict": verdict,
        "verification_mode": verification_mode,
        "predicted_answer": None if predicted_answer is None else str(predicted_answer),
        "reference_answer": None if reference_answer is None else str(reference_answer),
        "reason": reason,
    }


def _answer_status_from_final_answer_verification(
    evaluation: Mapping[str, Any],
    *,
    predicted_answer: Any | None,
    reference_answer: Any | None,
) -> dict[str, Any] | None:
    final_answer_verification = evaluation.get("final_answer_verification")
    if not isinstance(final_answer_verification, Mapping):
        return None
    answer_status = dict(final_answer_verification)
    answer_status["predicted_answer"] = (
        None if predicted_answer is None else str(predicted_answer)
    )
    answer_status["reference_answer"] = (
        None if reference_answer is None else str(reference_answer)
    )
    return answer_status


def _answer_status_from_existing_report(
    evaluation: Mapping[str, Any],
    *,
    predicted_answer: Any | None,
    reference_answer: Any | None,
) -> dict[str, Any] | None:
    report = evaluation.get("report")
    if not isinstance(report, Mapping):
        return None
    status = report.get("status")
    if not isinstance(status, Mapping):
        return None
    answer_status = status.get("answer_status")
    if not isinstance(answer_status, Mapping):
        return None
    result = dict(answer_status)
    result["predicted_answer"] = None if predicted_answer is None else str(predicted_answer)
    result["reference_answer"] = None if reference_answer is None else str(reference_answer)
    return result


def _infer_predicted_answer(evaluation: Mapping[str, Any], explicit: Any | None) -> Any | None:
    if explicit is not None:
        return explicit
    for key in ANSWER_KEY_CANDIDATES:
        value = evaluation.get(key)
        if value is not None:
            return value
    extracted = _extract_final_answer_from_findings(evaluation)
    if extracted is not None:
        return extracted
    # `label_answer` appears in old experiment outputs. It is a last-resort
    # fallback because some datasets use it for references rather than predictions.
    return evaluation.get("label_answer")


def _infer_reference_answer(evaluation: Mapping[str, Any], explicit: Any | None) -> Any | None:
    if explicit is not None:
        return explicit
    for key in REFERENCE_KEY_CANDIDATES:
        value = evaluation.get(key)
        if value is not None:
            return value
    # In old who_and_when outputs `label_answer` is often the reference answer.
    return evaluation.get("label_answer")


def _extract_final_answer_from_findings(evaluation: Mapping[str, Any]) -> str | None:
    patterns = [
        re.compile(r"FINAL\s+ANSWER\s*:\s*([^\n\r]+)", re.IGNORECASE),
        re.compile(r"final_answer\s*[:=]\s*['\"]?([^'\"\n\r]+)", re.IGNORECASE),
    ]
    for _, metric_result in _iter_metric_results(evaluation):
        for finding in metric_result.get("findings") or []:
            if not isinstance(finding, Mapping):
                continue
            for evidence in finding.get("evidence") or []:
                if not isinstance(evidence, Mapping):
                    continue
                quote = str(evidence.get("quote") or "")
                for pattern in patterns:
                    match = pattern.search(quote)
                    if match:
                        return match.group(1).strip().strip('"\'`')
    return None


def _answers_match(left: Any, right: Any) -> bool:
    left_norm = _normalize_answer(left)
    right_norm = _normalize_answer(right)
    if left_norm == right_norm:
        return True
    try:
        return float(left_norm) == float(right_norm)
    except (TypeError, ValueError):
        return False


def _normalize_answer(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"^final\s+answer\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().strip('"\'`')
    text = re.sub(r"\s+", " ", text)
    return text


def _unique_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            unique.append(value)
            seen.add(value)
    return unique
