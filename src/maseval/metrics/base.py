"""Base classes for LLM-based evaluation metrics."""

from __future__ import annotations

import json
from abc import ABC
from typing import Any, Type

from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput, PromptedOutput, RunContext
from pydantic_ai.models.openai import OpenAIChatModel

from ..models import (
    EvaluationInput,
    FindingsResult,
    MetricResult,
    RawTraceInput,
)


class LLMMetric(ABC):
    """Base class for LLM-based evaluation metrics."""

    def __init__(
        self,
        model: OpenAIChatModel | str | None,
        metric_name: str,
        prompt_template: str,
        result_type: Type[BaseModel] = FindingsResult,
        deps_type: Type[EvaluationInput | RawTraceInput] = EvaluationInput,
    ):
        """Initialize the metric."""
        self.model = model
        self.metric_name = metric_name
        self.prompt_template = prompt_template
        self.result_type = result_type
        self.deps_type = deps_type

        # Create the pydantic AI agent with dependency injection
        self.agent = Agent(
            model=self.model,
            output_type=self.result_type,
            deps_type=self.deps_type,
            retries=3,
        )
        self._setup_system_prompt()

    def _setup_system_prompt(self):
        """Setup system prompt with personal input fields"""

        @self.agent.system_prompt
        def get_system_prompt(ctx: RunContext[EvaluationInput]) -> str:
            eval_input = ctx.deps
            relevant_data = self._get_relevant_data(eval_input)
            eval_data = json.dumps(relevant_data, indent=2, default=str)
            import re
            import codecs
            from pathlib import Path
            text = eval_data.replace("\\\\u", "\\u")

            pattern = re.compile(r'(?:\\u[0-9a-fA-F]{4})+')

            def decode_unicode_escape_block(match: re.Match) -> str:
                frag = match.group(0)           

                decoded = codecs.decode(frag, "unicode_escape")
                return decoded

            decoded = pattern.sub(decode_unicode_escape_block, text)
            eval_data = re.sub(r'[\ud800-\udfff]', '', decoded)

            return f"""{self.prompt_template}
        
                    **EVALUATION DATA:**
                    {eval_data}

                    Use this evaluation data to perform your assessment."""

    def _get_relevant_data(self, eval_input: EvaluationInput | RawTraceInput) -> dict:
        """Handle both input types"""
        if hasattr(eval_input, "agents_pool"):
            return {
                "agents_pool": eval_input.agents_pool,
                "graph": eval_input.agents_pool.graph,
            }
        else:
            return eval_input.model_dump_json()

    async def evaluate(self, eval_input: EvaluationInput | RawTraceInput) -> MetricResult:
        try:
            result = await self.agent.run(
                "Please evaluate the provided data according to the metric criteria.",
                deps=eval_input,
            )
            return self._convert_result(result.output)
        except Exception as e:
            if "finish_reason" in str(e) and self.result_type:
                agent_text = Agent(self.model, deps_type=self.deps_type)
                agent_text.system_prompt = self.agent._system_prompts[0]

                response = await agent_text.run("... + Return JSON", deps=eval_input)
                parsed = json.loads(response.output)
                return self._convert_result(self.result_type(**parsed))
            raise

    def _convert_result(self, result: FindingsResult) -> MetricResult:
        """Convert findings result to MetricResult.

        Shared by all LLM metrics: the evaluator output is already a
        ``FindingsResult`` (``metric_name`` + ``findings``); we carry it
        into a ``MetricResult`` tagged with the metric's own name.
        """
        return MetricResult(
            metric_name=self.metric_name,
            findings=list(result.findings) if result.findings else [],
        )
