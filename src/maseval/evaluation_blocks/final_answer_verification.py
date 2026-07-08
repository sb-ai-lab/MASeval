"""Final-answer verification helpers for benchmark traces."""

import asyncio
import json
import os
from typing import Any, Literal

from dotenv import load_dotenv
import pandas as pd
from pydantic import BaseModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

from maseval.metrics.enums import MetricType
from maseval.metrics.factory import create_metric
from maseval.models import Confidence, MetricResult, RawTraceInput


Verdict = Literal["ideal", "poor"]
VerificationMethod = Literal["direct_comp", "mtc_judge", "equivalence_judge"]


class FinalAnswerVerificationResult(BaseModel):
    """Normalized result returned by final-answer verification."""

    metric_name: Literal["final_answer_verification"] = "final_answer_verification"
    verdict: Verdict
    confidence: Confidence
    method: VerificationMethod


class FinalAnswerVerifier:
    """Verify a final answer using direct comparison, equivalence judging, or MTC."""

    def __init__(self, model: str):
        self.model = model

    async def verify_final_answer(
        self,
        trace: Any,
        gt: str | None = None,
        df: str | None = None,
    ) -> FinalAnswerVerificationResult | None:
        """Return a normalized final-answer verification result for a trace."""

        if df is None:
            print("No df_name provided, skipping final answer verification...")
            return

        if gt is None:
            return await self._call_mas_task_completion_judge(trace)

        final_answer = self._extract_final_answer(trace, df)
        if final_answer is None:
            return await self._call_mas_task_completion_judge(trace)

        if final_answer != gt:
            return await self._call_equivalence_judge(final_answer, gt)

        return FinalAnswerVerificationResult(
            verdict="ideal",
            confidence=Confidence.HIGH,
            method="direct_comp",
        )

    async def _call_mas_task_completion_judge(self, trace: Any) -> FinalAnswerVerificationResult:
        model = OpenAIChatModel(
            self.model,
            provider="openrouter",
            settings={
                "temperature": 0.0,
            },
        )

        metric = create_metric(MetricType.MAS_TASK_COMPLETION, model)
        result = await metric.evaluate(RawTraceInput(trace=self._dump_trace(trace)))
        return self._convert_mtc_result(result)

    async def _call_equivalence_judge(
        self,
        final_answer: str,
        gt: str,
    ) -> FinalAnswerVerificationResult:
        dataset = Dataset(
            name="comparative_eval",
            cases=[
                Case(
                    inputs=final_answer,
                    expected_output=gt,
                )
            ],
            evaluators=[
                LLMJudge(
                    model=self.model,
                    rubric="Response is semantically equivalent to the expected output",
                    include_input=True,
                    include_expected_output=True,
                    score={"evaluation_name": "semantic_similarity"},
                    assertion={"evaluation_name": "correct_meaning"},
                )
            ],
        )

        report = await dataset.evaluate(lambda answer: answer, progress=False)
        if report.failures:
            raise RuntimeError(f"Equivalence judge failed: {report.failures[0]}")
        if not report.cases:
            raise RuntimeError("Equivalence judge returned no cases")

        case = report.cases[0]
        assertion = case.assertions.get("correct_meaning")
        score = case.scores.get("semantic_similarity")
        similarity = self._clamp_score(float(score.value)) if score is not None else None
        is_equivalent = (
            bool(assertion.value)
            if assertion is not None
            else bool(similarity is not None and similarity >= 0.5)
        )

        if similarity is None:
            confidence = Confidence.HIGH
        elif is_equivalent:
            confidence = self._confidence_from_score(similarity)
        else:
            confidence = self._confidence_from_score(1.0 - similarity)

        return FinalAnswerVerificationResult(
            verdict="ideal" if is_equivalent else "poor",
            confidence=confidence,
            method="equivalence_judge",
        )

    def _convert_mtc_result(self, result: MetricResult) -> FinalAnswerVerificationResult:
        if not result.findings:
            verdict = "ideal"
            confidence = Confidence.HIGH
        else:
            verdict = "poor"
            confidence = result.findings[0].confidence_estimate

        return FinalAnswerVerificationResult(
            verdict=verdict,
            confidence=confidence,
            method="mtc_judge",
        )

    @staticmethod
    def _confidence_from_score(score: float) -> Confidence:
        if score >= 0.8:
            return Confidence.HIGH
        if score >= 0.5:
            return Confidence.MEDIUM
        return Confidence.LOW

    @staticmethod
    def _clamp_score(value: float) -> float:
        return max(0.0, min(1.0, value))

    @staticmethod
    def _dump_trace(trace: Any) -> str:
        return json.dumps(trace, ensure_ascii=False, default=FinalAnswerVerifier._json_default)

    @staticmethod
    def _json_default(value: Any) -> Any:
        if hasattr(value, "tolist"):
            return value.tolist()
        return str(value)

    def _extract_final_answer(self, trace: Any, df: str) -> str | None:
        extractors = {
            "trail_gaia": self._extract_final_answer_trail_gaia,
            "trail_swe": self._extract_final_answer_trail_swe,
            "ghost": self._extract_final_answer_ghost,
            "who_and_when": self._extract_final_answer_who_and_when,
            "aegis": self._extract_final_answer_aegis,
        }

        try:
            extractor = extractors[df]
        except KeyError as exc:
            raise ValueError(f"Invalid df_name: {df}") from exc

        return extractor(trace)

    def _extract_final_answer_trail_gaia(self, trace: dict) -> str:
        final_span = trace["spans"][0]["child_spans"][1]["child_spans"][1]
        span_attributes = final_span["span_attributes"]

        if span_attributes["smolagents.managed_agents.0.name"] == "search_agent":
            return span_attributes["output.value"]

        raise ValueError("No response in trace")

    def _extract_final_answer_ghost(self, trace: dict) -> str | None:
        if hasattr(trace, "output") and trace.output:
            if "response" in trace.output:
                return trace.output["response"]

            raise ValueError("No response in trace")

        return None

    def _extract_final_answer_trail_swe(self, trace: dict) -> str:
        if trace["spans"][0]["logs"][0]["body"]["function.output"] is not None:
            return trace["spans"][0]["logs"][0]["body"]["function.output"]

        raise ValueError("No response in trace")

    def _extract_final_answer_aegis(self, trace: Any) -> str | None:
        rec = trace if isinstance(trace, dict) else {}
        inp = rec.get("input") or {}

        final_output = inp.get("final_output")
        if final_output:
            return str(final_output)

        history = inp.get("conversation_history") or []
        for state in reversed(history):
            content = state.get("content") if isinstance(state, dict) else None
            if content:
                return str(content)

        return None

    def _extract_final_answer_who_and_when(self, trace: Any) -> str | None:
        history = trace["history"] if isinstance(trace, dict) else trace

        for state in reversed(history):
            content = state.get("content") if isinstance(state, dict) else str(state)
            if not content or "FINAL ANSWER:" not in content:
                continue

            final_answer = content.split("FINAL ANSWER:", 1)[1].splitlines()[0].strip()
            return final_answer or None

        return None


if __name__ == "__main__":
    print("Starting final answer verification...")
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key

    final_answer_verifier = FinalAnswerVerifier(model="google/gemini-2.5-flash")

    # trail_gaia
    with open("/home/user/Desktop/AutoMAS/maseval-research-1/trail_benchmark/benchmarking/data/GAIA/0adc4f3b99d9564d32811e913cc9d248.json", "r") as f:
        trace = json.load(f)

    trail_gaia_gt = "100"
    result = asyncio.run(final_answer_verifier.verify_final_answer(trace, trail_gaia_gt, df="trail_gaia"))
    print(result)

    # trail_swe
    with open("/home/user/Desktop/AutoMAS/maseval-research-1/trail_benchmark/benchmarking/data/SWE Bench/0e6f7928953ab5a568bae640ce915cc3.json", "r") as f:
        trace = json.load(f)

    result = asyncio.run(final_answer_verifier.verify_final_answer(trace, df="trail_swe"))
    print(result)

    # who_and_when
    df_handcrafted = pd.read_parquet("hf://datasets/Kevin355/Who_and_When/Hand-Crafted.parquet")
    first_trace = df_handcrafted.iloc[4]
    trace_history = first_trace["history"]
    who_and_when_gt = first_trace["groundtruth"]

    result = asyncio.run(final_answer_verifier.verify_final_answer(trace_history, who_and_when_gt, df="who_and_when"))
    print(result)