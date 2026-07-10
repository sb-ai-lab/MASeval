from maseval.metrics import EvidenceVerifier
from maseval.metrics.evidence_verifier import EvidenceLLMBatchVerdict, EvidenceLLMVerdict
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

import pytest


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
                        reason="The cited idx belongs to WebSurfer.",
                    )
                ],
                evidence=[
                    Evidence(
                        idx="WebSurfer_1",
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
    assert verified.verifications[0].evidence_checks.all_idxs_exist is True
    assert verified.verifications[0].evidence_item_checks[0].idx_exists is True
    assert verified.verifications[0].evidence_item_checks[0].quote_found is True


def test_evidence_verifier_marks_invalid_missing_idx():
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
                        idx="missing_idx",
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
    assert verified.verifications[0].evidence_checks.all_idxs_exist is False
    assert verified.verifications[0].evidence_item_checks[0].idx_exists is False
    assert (
        "could not be resolved"
        in verified.verifications[0].evidence_item_checks[0].problem
    )


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
                    CulpritAgentCandidate(
                        agent="UnknownAgent", reason="Alternative candidate."
                    ),
                    CulpritAgentCandidate(
                        agent="WebSurfer", reason="Appears in the cited evidence."
                    ),
                ],
                evidence=[
                    Evidence(
                        idx="state_1",
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
    assert (
        verified.verifications[0].evidence_checks.culprit_agent_matches_evidence is True
    )
    assert verified.verifications[0].evidence_checks.idx_roles_are_plausible is True


@pytest.mark.asyncio
async def test_evidence_verifier_llm_mode_uses_judge_verdicts(monkeypatch):
    """In mode='llm' the verifier batches findings and trusts the LLM verdicts."""

    class _FakeOutput:
        verdicts = [
            EvidenceLLMVerdict(
                finding_index=0,
                status="weak",
                explanation="quote partially grounded",
                grounded_evidence_indices=[0],
            ),
            EvidenceLLMVerdict(
                finding_index=1,
                status="invalid",
                explanation="no quote found in trace",
                grounded_evidence_indices=[],
            ),
        ]

    class _FakeResponse:
        output = _FakeOutput()

    class _FakeAgent:
        async def run(self, prompt):
            return _FakeResponse()

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
                evidence=[
                    Evidence(idx="WebSurfer_1", role="root_cause", claim="c", quote="FINAL ANSWER: 31"),
                ],
                problem_description="p1",
            ),
            Finding(
                severity_estimate=Severity.MAJOR,
                confidence_estimate=Confidence.MEDIUM,
                evidence=[
                    Evidence(idx="missing", role="root_cause", claim="c", quote="nope"),
                ],
                problem_description="p2",
            ),
        ],
    )

    verifier = EvidenceVerifier(model="test", mode="llm")
    verifier._llm_agent = _FakeAgent()

    verified = await verifier.verify_metric_result_async(result, eval_input)

    assert verified.verifications[0].evidence_status == "weak"
    assert verified.verifications[0].verifier_method == "llm"
    assert verified.verifications[0].usable_for_diagnosis is True
    assert verified.verifications[1].evidence_status == "invalid"
    assert verified.verifications[1].usable_for_diagnosis is False
    # grounded evidence should be flagged for the weak finding's first item only.
    assert verified.verifications[0].evidence_item_checks[0].quote_found is True
