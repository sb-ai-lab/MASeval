"""Non-LLM based metrics that compute values directly from data."""

from pydantic_ai.models.openai import OpenAIChatModel

from ..models import EvaluationInput, StateType


class ToolEfficiencyMetric:
    """Metric for evaluating tool efficiency."""

    def __init__(self, model: None = None):
        self.model = model

    async def evaluate(self, eval_input: EvaluationInput) -> float:
        C_t = 0
        C_f = 0

        for state in eval_input.agent_states:
            if state.type == StateType.TOOL_CALL:
                C_t += 1
            elif state.type == StateType.INVALID_TOOL_CALL:
                C_f += 1
                C_t += 1

        if (C_t + C_f) == 0:
            tool_ef = 0
        else:
            tool_ef = (C_t - C_f) / (C_t + C_f)

        return round(tool_ef, 3)


class MASTimeMetric:
    """Metric for evaluating the time taken by the Multi-Agent System to complete the task."""

    def __init__(self, model: OpenAIChatModel):
        self.model = model

    async def evaluate(self, eval_input: EvaluationInput) -> float:
        total_time = 0

        for agent in eval_input.agents_latency_info:
            total_time += agent.latency
        return round(total_time, 3)


class MASTokensMetric:
    """Metric for evaluating the number of tokens used by the Multi-Agent System to complete the task."""

    def __init__(self, model: OpenAIChatModel):
        self.model = model

    async def evaluate(self, eval_input: EvaluationInput) -> dict[str, int]:
        mas_tokens = {}
        total_input_tokens = 0
        total_output_tokens = 0

        for agent in eval_input.agents_tokens_info:
            total_input_tokens += agent.input_tokens
            total_output_tokens += agent.output_tokens

        mas_tokens["input"] = total_input_tokens
        mas_tokens["output"] = total_output_tokens

        return mas_tokens


class AgentTimeMetric:
    """Metric for evaluating the time taken by an agent to complete the task."""

    def __init__(self, model: OpenAIChatModel):
        self.model = model

    async def evaluate(self, eval_input: EvaluationInput) -> dict[str, float]:
        agent_time = {}

        for agent in eval_input.agents_latency_info:
            agent_time[agent.agent_name] = agent.latency

        return agent_time


class AgentTokensMetric:
    """Metric for evaluating the number of tokens used by an agent to complete the task."""

    def __init__(self, model: OpenAIChatModel):
        self.model = model

    async def evaluate(self, eval_input: EvaluationInput) -> dict[str, dict[str, int]]:
        agent_tokens = {}

        for agent in eval_input.agents_tokens_info:
            agent_tokens[agent.agent_name] = {
                "input": agent.input_tokens,
                "output": agent.output_tokens,
            }

        return agent_tokens
