"""Pydantic models for trace-format-agnostic evaluation input."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class MessageRole(str, Enum):
    """Role of a message in the dialogue."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class DialogueMessage(BaseModel):
    """A single message in the dialogue history."""

    role: MessageRole
    content: str
    metadata: dict[str, Any] | None = None


class AgentResponse(BaseModel):
    """An agent's response in the conversation."""

    response_id: str
    content: str
    timestamp: datetime | None = None
    metadata: dict[str, Any] | None = None


class StateType(str, Enum):
    """Type of agent state."""

    SYSTEM = "system.text"
    ASSISTANT = "assistant.text"
    TOOL_CALL_RESPONSE = "user.tool_call_response"
    TOOL_CALL = "assistant.tool_call"
    INVALID_TOOL_CALL = "assistant.invalid_tool_call"
    USER = "user.text"


class ToolsInfo(BaseModel):
    """Tool calls made by an agent and tools that were available."""

    agent_name: str
    tools_called: list[ToolCall | InvalidToolCall] = Field(default_factory=list)
    available_tools: list[ToolDefinition] = Field(default_factory=list)


class ToolCall(BaseModel):
    """A tool call made by the agent."""

    id: str | None = None
    name: str
    parameters: dict[str, Any] | str
    result: str | None = None


class InvalidToolCall(BaseModel):
    """A tool call made by the agent."""

    id: str | None = None
    name: str
    parameters: dict[str, Any] | str
    status_message: str
    result: str | None = None


class AgentState(BaseModel):
    """Intermediate agent state (thought or action)."""

    state_id: str
    type: StateType
    content: str | dict[str, Any] | list[str]
    tool_call: ToolCall | InvalidToolCall | None = None
    timestamp: datetime | None = None
    metadata: dict[str, Any] | None = None


class Policy(BaseModel):
    """A policy that the agent should follow."""

    policy_id: str
    description: str
    requirements: list[str] = Field(default_factory=list)


class ToolDefinition(BaseModel):
    """Definition of an available tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    required_parameters: list[str] = Field(default_factory=list)


class TokensInfo(BaseModel):
    """Token usage information for an agent."""

    agent_name: str
    input_tokens: int
    output_tokens: int
    total_tokens: int


class Latency(BaseModel):
    """Latency information for an agent's response generation."""

    agent_name: str
    latency: float


class AgentError(BaseModel):
    """Errors in agents."""

    agent_name: str
    error_message: str | None = None


class AgentDescription(BaseModel):
    """Description of an agent."""

    agent_name: str
    instructions: str


class AgentsPool(BaseModel):
    """Agents pool for a task."""

    agents: list[AgentDescription]
    graph: str


class EvaluationInput(BaseModel):
    """Complete input for metric evaluation - trace-format-agnostic."""

    user_query: str | None = None

    # Core dialogue data
    dialogue_history: list[DialogueMessage] | None = None
    agent_responses: list[AgentResponse] | None = None  # = Field(default_factory=list)
    agent_states: list[AgentState] | None = None  # = Field(default_factory=list)

    # Context for evaluation
    policies: list[Policy] | None = None  # = Field(default_factory=list)
    agents_tools_info: list[ToolsInfo] | None = None  # = Field(default_factory=list)

    # Agents info (Tokens, Latency)
    agents_tokens_info: list[TokensInfo] | None = None  # = Field(default_factory=list)
    agents_latency_info: list[Latency] | None = None  # = Field(default_factory=list)

    # Metadata
    trace_id: str | None = None
    session_id: str | None = None
    environment: str | None = None
    metadata: dict[str, Any] | None = None

    # Errors handling
    agents_errors: list[AgentError] | None = None  # = Field(default_factory=list)
    agents_pool: AgentsPool | None = None


class RawTraceInput(EvaluationInput):
    trace: str


class Severity(str, Enum):
    """Estimated severity of a finding identified by an LLM evaluator."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class Confidence(str, Enum):
    """Estimated confidence of an LLM evaluator in its own finding."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CulpritAgentCandidate(BaseModel):
    """A candidate agent considered responsible for a finding."""

    agent: str
    reason: str


class Evidence(BaseModel):
    """A piece of evidence supporting a finding.

    ``idx`` should point to a concrete trace item. Preferred format for raw
    traces is a zero-based message/step index, e.g. ``"0"``, ``"1"``, ``"2"``.
    If the trace already exposes stable ids, ``idx`` may also be a ``state_id``,
    ``response_id``, ``policy_id``, or a tool-call id. It should not be a plain
    agent name like ``WebSurfer`` or a descriptive pseudo-id like
    ``Orchestrator (thought)``.
    """

    idx: str
    role: str  # free-form evidence role, e.g. root_cause / propagation / context
    claim: str
    quote: str

    @field_validator("idx", mode="before")
    @classmethod
    def _coerce_idx_to_string(cls, value: Any) -> str:
        """Accept JSON numbers from LLM outputs and normalize them to strings."""
        return str(value)


class Finding(BaseModel):
    """A single problem identified by an LLM evaluator for one metric."""

    severity_estimate: Severity
    confidence_estimate: Confidence
    culprit_agent_candidates: list[CulpritAgentCandidate] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    problem_description: str
    suggested_fix: str | None = None
    needs_human_review: bool = False


class FindingsResult(BaseModel):
    """Universal findings container returned by LLM evaluators."""

    metric_name: str
    findings: list[Finding] = Field(default_factory=list)


class MetricResult(BaseModel):
    """Complete result of a metric evaluation: a list of findings."""

    metric_name: str
    findings: list[Finding] = Field(default_factory=list)


class EvidenceStatus(str, Enum):
    """Grounding status assigned by EvidenceVerifier to an LLM finding."""

    VERIFIED = "verified"
    WEAK = "weak"
    INVALID = "invalid"


class EvidenceChecks(BaseModel):
    """Deterministic checks applied to evidence cited by an LLM evaluator."""

    all_idxs_exist: bool
    quotes_found_in_spans: bool
    culprit_agent_matches_evidence: bool
    span_roles_are_plausible: bool




class EvidenceItemCheck(BaseModel):
    """Per-evidence-item diagnostic produced by EvidenceVerifier.

    These checks make it clear which concrete evidence citation failed, instead
    of only returning aggregate booleans for the whole finding.
    """

    evidence_index: int
    idx: str
    span_exists: bool
    quote_found: bool
    role_plausible: bool
    used_raw_trace_fallback: bool = False
    resolved_idx: str | None = None
    resolved_agent: str | None = None
    resolution_strategy: str | None = None
    problem: str | None = None


class EvidenceVerificationResult(BaseModel):
    """EvidenceVerifier output for one finding of one metric."""

    metric_name: str
    finding_index: int
    evidence_status: EvidenceStatus
    evidence_checks: EvidenceChecks
    evidence_item_checks: list[EvidenceItemCheck] = Field(default_factory=list)
    usable_for_diagnosis: bool
    verifier_explanation: str


class EvidenceVerificationMetricResult(BaseModel):
    """EvidenceVerifier output for all findings of one metric."""

    metric_name: str
    verifications: list[EvidenceVerificationResult] = Field(default_factory=list)

