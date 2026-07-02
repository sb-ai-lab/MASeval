from __future__ import annotations

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, NativeOutput
from pydantic_ai.models.openai import OpenAIChatModel
from ..models import EvaluationInput
from typing import Type, Any
import json

from ..prompts import MAS_SUMMARIZER_PROMPT_BASE_TEMPLATE


class SummarizerResult(BaseModel):
    """Result for summarizer."""

    score: str
    justification: str
    
    
class SummarizerConfidenceResult(BaseModel):
    """Result for summarizer."""
    score: str
    justification: str
    confidence : float 

class FinalSummarizerConfidenceResult(BaseModel):
    """Full summarizer result."""

    final_score: SummarizerConfidenceResult
    
class FinalSummarizerResult(BaseModel):
    """Full summarizer result."""

    final_score: SummarizerResult


class Summarizer:
    """Summarizer class."""

    def __init__(
        self,
        model: OpenAIChatModel | str | None,
        prompt_template: MAS_SUMMARIZER_PROMPT_BASE_TEMPLATE,
        result_type: FinalSummarizerResult,
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
            name="MetricsSummarizer",
            output_type=self.result_type,
            deps_type=dict[str, Any],
        )

        # Add system prompt that includes the evaluation data
        @self.agent.system_prompt
        def get_system_prompt(ctx: RunContext[EvaluationInput]) -> str:
            metrics_results = ctx.deps
            metrics_results_dump = json.dumps(metrics_results, indent=2, default=str)
            import re
            import codecs
            from pathlib import Path
            text = metrics_results_dump.replace("\\\\u", "\\u")

            pattern = re.compile(r'(?:\\u[0-9a-fA-F]{4})+')

            def decode_unicode_escape_block(match: re.Match) -> str:
                frag = match.group(0)           

                decoded = codecs.decode(frag, "unicode_escape")
                return decoded

            decoded = pattern.sub(decode_unicode_escape_block, text)
            metrics_results_dump = re.sub(r'[\ud800-\udfff]', '', decoded)

        # def get_system_prompt(ctx: RunContext[dict[str, Any]]) -> str:
        #     """Generate system prompt with evaluation data."""
        #     metrics_results = ctx.deps
        #     metrics_results_dump = json.dumps(metrics_results, default=str, indent=2)

            # Combine the prompt template with the actual data
            return f"""{self.prompt_template}
                        Use this metrics results to perform your assessment.
                        
                        **METRICS RESULTS:**
                        {metrics_results_dump}
                        """

    async def evaluate(self, metrics_results: dict[str, Any]) -> SummarizerResult:
        """Evaluate the metric on the given input."""
        # Run the agent with dependency injection
        result = await self.agent.run(
            "Please evaluate the provided data according to the metric criteria.",
            deps=metrics_results,
        )

        return result.output
