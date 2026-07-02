from __future__ import annotations

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, NativeOutput
from pydantic_ai.models.openai import OpenAIChatModel

from ..models import EvaluationInput
from typing import Type, Any
import json

from ..prompts import MAS_UNIFIED_COMPREHENSIVE_EVALUATION_PROMPT


class SingleJudgeResult(BaseModel):
    """Result for single judge."""

    score: str
    justification: str

    
class FinalSingleJudgeResult(BaseModel):
    """Full single judge result."""

    final_score: SingleJudgeResult


class SingleJudge:
    """Summarizer class."""

    def __init__(
        self,
        model: OpenAIChatModel | str | None,
        prompt_template: MAS_UNIFIED_COMPREHENSIVE_EVALUATION_PROMPT,
        result_type: FinalSingleJudgeResult,
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
            name="SingleJudge",
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
                            Use this evaluation data to perform your assessment.
                            
                            **USER_QUERY**
                            {eval_input.user_query}

                            **AGENT_STATES**
                            {eval_input.agent_states}

                            **AGENTS_TOOLS_INFO**
                            {eval_input.agents_tools_info}

                            """
            else:
                return f"""{self.prompt_template}
                    Use this raw trace to perform your assessment.

                    **RAW TRACE:**
                    {eval_input.trace}

                    """

    async def evaluate(self, eval_input: EvaluationInput) -> SingleJudgeResult:
        """Evaluate the metric on the given input."""
        # Run the agent with dependency injection
        result = await self.agent.run(
            "Please evaluate the provided data according to the metric criteria.",
            deps=eval_input,
        )

        return result.output
