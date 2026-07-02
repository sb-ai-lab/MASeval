"""Factory function for creating metric instances."""

from pydantic_ai.models.openai import OpenAIChatModel

from .base import LLMMetric
from .enums import MetricType
from .llm_metrics import (
    MASComplexityMetric,
    MASPlanningMetric,
    MASRolesDistributionMetric,
    MASTaskTransferMetric,
    ObservationAlignmentMetric,
    PolicyAlignmentMetric,
    StateConsistencyMetric,
    SystemTaskCompletionMetric,
    TaskCompletenessMetric,
    ToolParameterExtractionMetric,
    ToolSelectionMetric,
    PromptQualityMetric,
    ToolPerformanceMetric,
)
from .non_llm_metrics import (
    AgentTimeMetric,
    AgentTokensMetric,
    MASTimeMetric,
    MASTokensMetric,
    ToolEfficiencyMetric,
)


def create_metric(
    metric_type: MetricType, model: OpenAIChatModel | None = None
) -> (
    LLMMetric
    | ToolEfficiencyMetric
    | MASTimeMetric
    | MASTokensMetric
    | AgentTimeMetric
    | AgentTokensMetric
):
    """Create a metric instance based on type."""
    metric_classes = {
        MetricType.OBSERVATION_ALIGNMENT: ObservationAlignmentMetric,
        MetricType.POLICY_ALIGNMENT: PolicyAlignmentMetric,
        MetricType.STATE_CONSISTENCY: StateConsistencyMetric,
        MetricType.TOOL_SELECTION: ToolSelectionMetric,
        MetricType.TOOL_PARAMETER_EXTRACTION: ToolParameterExtractionMetric,
        MetricType.TASK_COMPLETENESS: TaskCompletenessMetric,
        MetricType.TOOL_EFFICIENCY: ToolEfficiencyMetric,
        MetricType.MAS_TIME: MASTimeMetric,
        MetricType.MAS_TOKENS: MASTokensMetric,
        MetricType.AGENT_TIME: AgentTimeMetric,
        MetricType.AGENT_TOKENS: AgentTokensMetric,
        MetricType.MAS_PLANNING: MASPlanningMetric,
        MetricType.MAS_COMPLEXITY: MASComplexityMetric,
        MetricType.MAS_TASK_TRANSFER: MASTaskTransferMetric,
        MetricType.MAS_ROLES_DISTRIBUTION: MASRolesDistributionMetric,
        MetricType.MAS_TASK_COMPLETION: SystemTaskCompletionMetric,
        MetricType.PROMPT_QUALITY: PromptQualityMetric,
        MetricType.TOOL_PERFORMANCE: ToolPerformanceMetric,
    }

    if metric_type not in metric_classes:
        raise ValueError(f"Unknown metric type: {metric_type}")

    return metric_classes[metric_type](model)
