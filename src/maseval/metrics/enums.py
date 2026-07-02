"""Metric type enumerations."""

from enum import Enum


class MetricType(str, Enum):
    """Types of available metrics."""

    OBSERVATION_ALIGNMENT = "observation_alignment"
    POLICY_ALIGNMENT = "policy_alignment"
    STATE_CONSISTENCY = "state_consistency"
    TOOL_SELECTION = "tool_selection"
    TOOL_PARAMETER_EXTRACTION = "tool_parameter_extraction"
    TASK_COMPLETENESS = "task_completeness"
    TOOL_EFFICIENCY = "tool_efficiency"
    MAS_TIME = "mas_time"
    MAS_TOKENS = "mas_tokens"
    AGENT_TIME = "agent_time"
    AGENT_TOKENS = "agent_tokens"
    MAS_PLANNING = "mas_planning"
    MAS_COMPLEXITY = "mas_complexity"
    MAS_TASK_TRANSFER = "mas_task_transfer"
    MAS_ROLES_DISTRIBUTION = "mas_roles_distribution"
    AGENT_REFLECTION_MAS_PLANNING = "agent_reflection_mas_planning"
    MAS_TASK_COMPLETION = "mas_task_completion"
    PROMPT_QUALITY = "prompt_quality"
    TOOL_PERFORMANCE = "tool_performance"
