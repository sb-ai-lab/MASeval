"""LLM-based evaluation metrics."""

from __future__ import annotations

import os

from pydantic_ai.models.openai import OpenAIChatModel

from ..models import (
    EvaluationInput,
    FindingsResult,
    MetricResult,
    RawTraceInput,
)
from .base import LLMMetric

PROMPTS_LANG = os.environ.get("PROMPTS_LANG", "ENG")
if PROMPTS_LANG == 'ENG':
    from ..prompts import (
        MAS_COMPLEXITY_PROMPT_BASE_TEMPLATE,
        MAS_PLANNING_PROMPT_BASE_TEMPLATE,
        MAS_ROLES_DISTRIBUTION_PROMPT_BASE_TEMPLATE,
        MAS_TASK_COMPLETION_PROMPT,
        MAS_TASK_TRANSFER_PROMPT_BASE_TEMPLATE,
        OBSERVATION_ALIGNMENT_PROMPT,
        POLICY_ALIGNMENT_PROMPT,
        PROMPT_QUALITY_PROMPT_BASE_TEMPLATE,
        STATE_CONSISTENCY_PROMPT,
        TASK_COMPLETENESS_PROMPT,
        TOOL_PARAMETER_EXTRACTION_PROMPT,
        TOOL_PERFORMANCE_PROMPT_BASE_TEMPLATE,
        TOOL_SELECTION_PROMPT,
    )
else:
    from ..prompts.prompts_ru_no_fs import (
        MAS_COMPLEXITY_PROMPT_BASE_TEMPLATE,
        MAS_PLANNING_PROMPT_BASE_TEMPLATE,
        MAS_ROLES_DISTRIBUTION_PROMPT_BASE_TEMPLATE,
        MAS_TASK_COMPLETION_PROMPT,
        MAS_TASK_TRANSFER_PROMPT_BASE_TEMPLATE,
        OBSERVATION_ALIGNMENT_PROMPT,
        POLICY_ALIGNMENT_PROMPT,
        STATE_CONSISTENCY_PROMPT,
        TASK_COMPLETENESS_PROMPT,
        TOOL_PARAMETER_EXTRACTION_PROMPT,
        TOOL_SELECTION_PROMPT,
    )


class SystemTaskCompletionMetric(LLMMetric):
    """Metric for evaluating whether the system as a whole successfully completed the user's task."""

    def __init__(self, model: OpenAIChatModel):
        super().__init__(
            model=model,
            metric_name="MAS_TASK_COMPLETION",
            prompt_template=MAS_TASK_COMPLETION_PROMPT,
            result_type=FindingsResult,
        )
        self.agent.name = "SystemTaskCompletionEvaluator"

    def _get_relevant_data(self, eval_input: EvaluationInput | RawTraceInput) -> dict:
        """Extract only fields relevant for metric."""
        if eval_input.user_query:
            return {
                "user_query": eval_input.user_query,
                "agent_states": eval_input.agent_states,
            }
        elif hasattr(eval_input, "trace"):
            return {"raw_trace": eval_input.trace}
        else:
            return eval_input.model_dump_json()


class MASTaskTransferMetric(LLMMetric):
    """Metric for evaluating task transfer quality between agents."""

    def __init__(self, model: OpenAIChatModel):
        super().__init__(
            model=model,
            metric_name="mas_task_transfer",
            prompt_template=MAS_TASK_TRANSFER_PROMPT_BASE_TEMPLATE,
            result_type=FindingsResult,
        )
        self.agent.name = "MASTaskTransferEvaluator"

    def _get_relevant_data(self, eval_input: EvaluationInput | RawTraceInput) -> dict:
        """Extract only fields relevant for metric."""
        if eval_input.user_query:
            return {
                "user_query": eval_input.user_query,
                "agent_states": eval_input.agent_states,
            }
        elif hasattr(eval_input, "trace"):
            return {"raw_trace": eval_input.trace}
        else:
            return eval_input.model_dump_json()


class MASComplexityMetric(LLMMetric):
    """Metric for evaluating multi-agent system complexity and interconnectedness."""

    def __init__(self, model: OpenAIChatModel):
        super().__init__(
            model=model,
            metric_name="mas_complexity",
            prompt_template=MAS_COMPLEXITY_PROMPT_BASE_TEMPLATE,
            result_type=FindingsResult,
        )
        self.agent.name = "MASComplexityEvaluator"

    def _get_relevant_data(self, eval_input: EvaluationInput | RawTraceInput) -> dict:
        """Extract only fields relevant for metric."""
        if eval_input.user_query:
            return {
                "user_query": eval_input.user_query,
                "agent_states": eval_input.agent_states,
            }
        elif hasattr(eval_input, "trace"):
            return {"raw_trace": eval_input.trace}
        else:
            return eval_input.model_dump_json()


class MASRolesDistributionMetric(LLMMetric):
    """Metric for evaluating balance and distribution of roles among agents."""

    def __init__(self, model: OpenAIChatModel):
        super().__init__(
            model=model,
            metric_name="mas_roles_distribution",
            prompt_template=MAS_ROLES_DISTRIBUTION_PROMPT_BASE_TEMPLATE,
            result_type=FindingsResult,
        )
        self.agent.name = "MASRolesDistributionEvaluator"

    def _get_relevant_data(self, eval_input: EvaluationInput | RawTraceInput) -> dict:
        """Extract only fields relevant for metric."""
        if eval_input.user_query:
            return {
                "user_query": eval_input.user_query,
                "agent_states": eval_input.agent_states,
            }
        elif hasattr(eval_input, "trace"):
            return {"raw_trace": eval_input.trace}
        else:
            return eval_input.model_dump_json()


class MASPlanningMetric(LLMMetric):
    """Metric for evaluating multi-agent system planning quality."""

    def __init__(self, model: OpenAIChatModel):
        super().__init__(
            model=model,
            metric_name="mas_planning",
            prompt_template=MAS_PLANNING_PROMPT_BASE_TEMPLATE,
            result_type=FindingsResult,
        )
        self.agent.name = "MASPlanningEvaluator"

    def _get_relevant_data(self, eval_input: EvaluationInput | RawTraceInput) -> dict:
        """Extract only fields relevant for metric."""
        if eval_input.user_query:
            return {
                "user_query": eval_input.user_query,
                "agent_states": eval_input.agent_states,
            }
        elif hasattr(eval_input, "trace"):
            return {"raw_trace": eval_input.trace}
        else:
            return eval_input.model_dump_json()


class ObservationAlignmentMetric(LLMMetric):
    """Metric for evaluating observation alignment."""

    def __init__(self, model: OpenAIChatModel):
        super().__init__(
            model=model,
            metric_name="observation_alignment",
            prompt_template=OBSERVATION_ALIGNMENT_PROMPT,
            result_type=FindingsResult,
        )
        self.agent.name = "ObservationAlignmentEvaluator"

    def _get_relevant_data(self, eval_input: EvaluationInput | RawTraceInput) -> dict:
        """Extract only fields relevant for metric."""
        if eval_input.user_query:
            return {
                "user_query": eval_input.user_query,
                "dialogue_history": eval_input.dialogue_history,
                "agent_responses": eval_input.agent_responses,
            }
        elif hasattr(eval_input, "trace"):
            return {"raw_trace": eval_input.trace}
        else:
            return eval_input.model_dump_json()


class PolicyAlignmentMetric(LLMMetric):
    """Metric for evaluating policy alignment."""

    def __init__(self, model: OpenAIChatModel):
        super().__init__(
            model=model,
            metric_name="policy_alignment",
            prompt_template=POLICY_ALIGNMENT_PROMPT,
            result_type=FindingsResult,
        )
        self.agent.name = "PolicyAlignmentEvaluator"

    def _get_relevant_data(self, eval_input: EvaluationInput | RawTraceInput) -> dict:
        """Extract only fields relevant for metric."""
        if eval_input.user_query:
            return {
                "user_query": eval_input.user_query,
                "agent_states": eval_input.agent_states,
            }
        elif hasattr(eval_input, "trace"):
            return {"raw_trace": eval_input.trace}
        else:
            return eval_input.model_dump_json()


class StateConsistencyMetric(LLMMetric):
    """Metric for evaluating state consistency."""

    def __init__(self, model: OpenAIChatModel):
        super().__init__(
            model=model,
            metric_name="state_consistency",
            prompt_template=STATE_CONSISTENCY_PROMPT,
            result_type=FindingsResult,
        )
        self.agent.name = "StateConsistencyEvaluator"

    def _get_relevant_data(self, eval_input: EvaluationInput | RawTraceInput) -> dict:
        """Extract only fields relevant for metric."""
        if eval_input.user_query:
            return {
                "user_query": eval_input.user_query,
                "agent_states": eval_input.agent_states,
                "dialogue_history": eval_input.dialogue_history,
            }
        elif hasattr(eval_input, "trace"):
            return {"raw_trace": eval_input.trace}
        else:
            return eval_input.model_dump_json()


class ToolSelectionMetric(LLMMetric):
    """Metric for evaluating tool selection."""

    def __init__(self, model: OpenAIChatModel):
        super().__init__(
            model=model,
            metric_name="tool_selection",
            prompt_template=TOOL_SELECTION_PROMPT,
            result_type=FindingsResult,
        )
        self.agent.name = "ToolSelectionEvaluator"

    def _get_relevant_data(self, eval_input: EvaluationInput | RawTraceInput) -> dict:
        """Extract only fields relevant for metric."""
        if eval_input.user_query:
            return {
                "user_query": eval_input.user_query,
                "agent_states": eval_input.agent_states,
                "agents_tools_info": eval_input.agents_tools_info,
            }
        elif hasattr(eval_input, "trace"):
            return {"raw_trace": eval_input.trace}
        else:
            return eval_input.model_dump_json()


class ToolParameterExtractionMetric(LLMMetric):
    """Metric for evaluating tool parameter extraction."""

    def __init__(self, model: OpenAIChatModel):
        super().__init__(
            model=model,
            metric_name="tool_parameter_extraction",
            prompt_template=TOOL_PARAMETER_EXTRACTION_PROMPT,
            result_type=FindingsResult,
        )
        self.agent.name = "ToolParameterExtractionEvaluator"

    def _get_relevant_data(self, eval_input: EvaluationInput | RawTraceInput) -> dict:
        """Extract only fields relevant for metric."""
        if eval_input.user_query:
            return {
                "user_query": eval_input.user_query,
                "agent_states": eval_input.agent_states,
                "agents_tools_info": eval_input.agents_tools_info,
            }
        elif hasattr(eval_input, "trace"):
            return {"raw_trace": eval_input.trace}
        else:
            return eval_input.model_dump_json()


class TaskCompletenessMetric(LLMMetric):
    """Metric for evaluating task completeness."""

    def __init__(self, model: OpenAIChatModel):
        super().__init__(
            model=model,
            metric_name="task_completeness",
            prompt_template=TASK_COMPLETENESS_PROMPT,
            result_type=FindingsResult,
        )
        self.agent.name = "TaskCompletenessEvaluator"

    def _get_relevant_data(self, eval_input: EvaluationInput | RawTraceInput) -> dict:
        """Extract only fields relevant for metric."""
        if eval_input.user_query:
            return {
                "user_query": eval_input.user_query,
                "agent_states": eval_input.agent_states,
            }
        elif hasattr(eval_input, "trace"):
            return {"raw_trace": eval_input.trace}
        else:
            return eval_input.model_dump_json()


class ToolPerformanceMetric(LLMMetric):
    """Metric for evaluating tool performance."""

    def __init__(self, model: OpenAIChatModel):
        super().__init__(
            model=model,
            metric_name="tool_performance",
            prompt_template=TOOL_PERFORMANCE_PROMPT_BASE_TEMPLATE,
            result_type=FindingsResult,
        )
        self.agent.name = "ToolPerformanceEvaluator"

    def _get_relevant_data(self, eval_input: EvaluationInput | RawTraceInput) -> dict:
        """Extract only fields relevant for metric."""
        if eval_input.user_query:
            return {
                "user_query": eval_input.user_query,
                "agent_states": eval_input.agent_states,
                "agents_tools_info": eval_input.agents_tools_info,
            }
        elif hasattr(eval_input, "trace"):
            return {"raw_trace": eval_input.trace}
        else:
            return eval_input.model_dump_json()


class PromptQualityMetric(LLMMetric):
    """Metric for evaluating prompt quality."""

    def __init__(self, model: OpenAIChatModel):
        super().__init__(
            model=model,
            metric_name="prompt_quality",
            prompt_template=PROMPT_QUALITY_PROMPT_BASE_TEMPLATE,
            result_type=FindingsResult,
        )
        self.agent.name = "PromptQualityEvaluator"

    def _get_relevant_data(self, eval_input: EvaluationInput | RawTraceInput) -> dict:
        """Extract only fields relevant for metric."""
        if eval_input.user_query:
            return {
                "agents_info": eval_input.agents_pool.agents,
            }
        elif hasattr(eval_input, "trace"):
            return {"raw_trace": eval_input.trace}
        else:
            return eval_input.model_dump_json()