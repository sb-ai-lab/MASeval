"""MASeval - A library for automated Multi-Agent System evaluation using pydantic AI."""

import os
from langfuse import Langfuse

from . import prompts
from .metrics import LLMMetric, MetricType, EvidenceVerifier, create_metric, get_langfuse_judge_client
from .models import (
    AgentResponse,
    AgentState,
    DialogueMessage,
    EvaluationInput,
    EvidenceChecks,
    EvidenceItemCheck,
    EvidenceStatus,
    EvidenceVerificationMetricResult,
    EvidenceVerificationResult,
    Finding,
    FindingsResult,
    MetricResult,
    ToolCall,
)
from .reporting import build_evaluation_report, build_report_from_file
from .diagnostic_accuracy import evaluate_agent_step_accuracy, read_prediction_file, read_annotations


def get_langfuse_download_client() -> Langfuse:
    """Get a Langfuse client for downloading traces from your evaluation project.

    This client uses LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment
    variables. Use this in examples/scripts to retrieve traces for evaluation.

    IMPORTANT: Initialize this client BEFORE calling get_langfuse_judge_client()
    to ensure evaluations are traced to the judge project, not the download project.

    Returns:
        Langfuse: A Langfuse client configured for downloading traces (tracing disabled)
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST")

    # Disable tracing to prevent this client from setting up OpenTelemetry
    # and interfering with the judge client's tracer provider
    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
        tracing_enabled=False,
        timeout=120,
    )


__all__ = [
    "DialogueMessage",
    "AgentResponse",
    "AgentState",
    "ToolCall",
    "EvaluationInput",
    "Finding",
    "FindingsResult",
    "MetricResult",
    "EvidenceVerifier",
    "EvidenceChecks",
    "EvidenceItemCheck",
    "EvidenceStatus",
    "EvidenceVerificationMetricResult",
    "EvidenceVerificationResult",
    "LLMMetric",
    "MetricType",
    "create_metric",
    "prompts",
    "get_langfuse_judge_client",
    "get_langfuse_download_client",
    "build_evaluation_report",
    "build_report_from_file",
    "evaluate_agent_step_accuracy",
    "read_prediction_file",
    "read_annotations",
]

__version__ = "0.1.0"
