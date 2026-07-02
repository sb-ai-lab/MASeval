from maseval.metrics import EvidenceVerifier
from maseval.models import (
    AgentState,
    Confidence,
    CulpritAgentCandidate,
    Evidence,
    EvaluationInput,
    Finding,
    MetricResult,
    Severity,
    StateType,
)


def test_evidence_verifier_marks_verified_finding():
    eval_input = EvaluationInput(
        agent_states=[
            AgentState(
                state_id="WebSurfer_1",
                type=StateType.ASSISTANT,
                content="FINAL ANSWER: 31",
            )
        ]
    )
    result = MetricResult(
        metric_name="task_transfer",
        findings=[
            Finding(
                severity_estimate=Severity.CRITICAL,
                confidence_estimate=Confidence.HIGH,
                culprit_agent_candidates=[
                    CulpritAgentCandidate(
                        agent="WebSurfer",
                        reason="The cited span belongs to WebSurfer.",
                    )
                ],
                evidence=[
                    Evidence(
                        span_id="WebSurfer_1",
                        role="root_cause",
                        claim="The agent produced the final answer.",
                        quote="FINAL ANSWER: 31",
                    )
                ],
                problem_description="Unsupported final answer.",
            )
        ],
    )

    verified = EvidenceVerifier().verify_metric_result(result, eval_input)

    assert verified.verifications[0].evidence_status == "verified"
    assert verified.verifications[0].usable_for_diagnosis is True
    assert verified.verifications[0].evidence_checks.all_span_ids_exist is True
    assert verified.verifications[0].evidence_item_checks[0].span_exists is True
    assert verified.verifications[0].evidence_item_checks[0].quote_found is True


def test_evidence_verifier_marks_invalid_missing_span():
    eval_input = EvaluationInput(
        agent_states=[
            AgentState(
                state_id="WebSurfer_1",
                type=StateType.ASSISTANT,
                content="FINAL ANSWER: 31",
            )
        ]
    )
    result = MetricResult(
        metric_name="task_transfer",
        findings=[
            Finding(
                severity_estimate=Severity.CRITICAL,
                confidence_estimate=Confidence.HIGH,
                culprit_agent_candidates=[],
                evidence=[
                    Evidence(
                        span_id="missing_span",
                        role="root_cause",
                        claim="The agent produced the final answer.",
                        quote="FINAL ANSWER: 31",
                    )
                ],
                problem_description="Unsupported final answer.",
            )
        ],
    )

    verified = EvidenceVerifier().verify_metric_result(result, eval_input)

    assert verified.verifications[0].evidence_status == "invalid"
    assert verified.verifications[0].usable_for_diagnosis is False
    assert verified.verifications[0].evidence_checks.all_span_ids_exist is False
    assert verified.verifications[0].evidence_item_checks[0].span_exists is False
    assert "not present in the normalized trace" in verified.verifications[0].evidence_item_checks[0].problem


def test_evidence_verifier_accepts_culprit_role_and_any_matching_candidate():
    eval_input = EvaluationInput(
        agent_states=[
            AgentState(
                state_id="state_1",
                type=StateType.ASSISTANT,
                content="WebSurfer produced FINAL ANSWER: 31",
            )
        ]
    )
    result = MetricResult(
        metric_name="task_completeness",
        findings=[
            Finding(
                severity_estimate=Severity.CRITICAL,
                confidence_estimate=Confidence.HIGH,
                culprit_agent_candidates=[
                    CulpritAgentCandidate(agent="UnknownAgent", reason="Alternative candidate."),
                    CulpritAgentCandidate(agent="WebSurfer", reason="Appears in the cited evidence."),
                ],
                evidence=[
                    Evidence(
                        span_id="state_1",
                        role="culprit",
                        claim="The agent produced the final answer.",
                        quote="FINAL ANSWER: 31",
                    )
                ],
                problem_description="Unsupported final answer.",
            )
        ],
    )

    verified = EvidenceVerifier().verify_metric_result(result, eval_input)

    assert verified.verifications[0].evidence_status == "verified"
    assert verified.verifications[0].evidence_checks.culprit_agent_matches_evidence is True
    assert verified.verifications[0].evidence_checks.span_roles_are_plausible is True
