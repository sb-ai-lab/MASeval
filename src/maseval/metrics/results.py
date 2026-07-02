"""Result models for metric evaluations.

All LLM evaluators share a findings-based result container. EvidenceVerifier
adds a deterministic grounding check for every finding.
"""

from ..models import (
    EvidenceChecks,
    EvidenceItemCheck,
    EvidenceStatus,
    EvidenceVerificationMetricResult,
    EvidenceVerificationResult,
    FindingsResult,
)

__all__ = [
    "FindingsResult",
    "EvidenceChecks",
    "EvidenceItemCheck",
    "EvidenceStatus",
    "EvidenceVerificationMetricResult",
    "EvidenceVerificationResult",
]
