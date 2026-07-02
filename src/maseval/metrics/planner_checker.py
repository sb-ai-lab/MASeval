from __future__ import annotations

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from ..models import EvaluationInput
from typing import Type

from ..prompts import PLANNER_IDENTIFICATION_PROMPT_BASE_TEMPLATE


class PlannerCheckerResult(BaseModel):
    """Result for planner checker."""

    agent_name: str
    is_planner: bool
    justification: str


class PlannerCheckerResults(BaseModel):
    """Results for planner checker."""

    agents: list[PlannerCheckerResult]


class PlannerChecker:
    """Planner checker class."""

    def __init__(
        self,
        model: OpenAIChatModel | str | None,
        prompt_template: PLANNER_IDENTIFICATION_PROMPT_BASE_TEMPLATE,
        result_type: PlannerCheckerResults,
    ):
        """Initialize the metric.

        Args:
            model: OpenAI model instance for evaluation
            prompt_template: Prompt template for evaluation
            result_type: Pydantic model for the expected result
        """
        self.model = model
        self.prompt_template = prompt_template
        self.result_type = result_type

        # Create the pydantic AI agent with dependency injection
        self.agent = Agent(
            model=self.model,
            output_type=self.result_type,
            deps_type=EvaluationInput,
        )

        # Add system prompt that includes the evaluation data
        @self.agent.system_prompt
        def get_system_prompt(ctx: RunContext[EvaluationInput]) -> str:
            """Generate system prompt with evaluation data."""
            eval_input = ctx.deps

            # Combine the prompt template with the actual data
            if eval_input.agents_pool:
                return f"""{self.prompt_template}
                            Use this agents pool to perform your assessment.
                            
                            **AGENTS POOL:**
                            {eval_input.agents_pool.agents}

                            Use this graph to determine the position of the planners.

                            **GRAPH:**
                            {eval_input.agents_pool.graph}


                            """
            else:
                return f"""{self.prompt_template}
                    Use this raw trace to perform your assessment.

                    **RAW TRACE:**
                    {eval_input.trace}

                    """

    async def evaluate(self, eval_input: EvaluationInput) -> PlannerCheckerResults:
        """Evaluate the metric on the given input."""
        # Run the agent with dependency injection
        result = await self.agent.run(
            "Please evaluate the provided data according to the metric criteria.",
            deps=eval_input,
        )

        return result.output
