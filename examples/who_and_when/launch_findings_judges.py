"""Launch all LLM findings-evaluators on the Who & When (Hand-Crafted) dataset.

Each task is evaluated by every LLM evaluator (metric). Raw findings are:
  1. Saved to a per-task JSON file.
  2. Traced to Langfuse under a parent span, with the full findings body as
     the span's `output`.

Nothing is run after the LLM evaluators (no Summarizer / SingleJudge /
non-LLM metrics).
"""

import ast
import asyncio
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env")

# Disable the default OTEL tracer provider BEFORE importing maseval, so that
# the Langfuse judge client can install its own provider later without conflict.
os.environ["OTEL_TRACES_EXPORTER"] = "none"
from opentelemetry import trace

trace.set_tracer_provider(None)

from pydantic_ai.models.openai import OpenAIChatModel
from rich import print

from maseval import get_langfuse_judge_client
from maseval.evaluation_blocks.final_answer_verification import (
    FinalAnswerVerificationResult,
    FinalAnswerVerifier,
)
from maseval.metrics import EvidenceVerifier, MetricType, create_metric
from maseval.models import RawTraceInput
from maseval.reporting import build_evaluation_report
from maseval.validators import run_on_trace

# All active LLM evaluators (the regex-based mas_api_issues / mas_environment_setup_errors
# were removed; non-LLM metrics are intentionally excluded).
LLM_METRICS_TO_TEST = [
    MetricType.OBSERVATION_ALIGNMENT,
    MetricType.POLICY_ALIGNMENT,
    MetricType.STATE_CONSISTENCY,
    MetricType.TOOL_SELECTION,
    MetricType.TOOL_PARAMETER_EXTRACTION,
    MetricType.MAS_PLANNING,
    MetricType.MAS_COMPLEXITY,
    MetricType.MAS_TASK_TRANSFER,
    MetricType.MAS_ROLES_DISTRIBUTION,
    MetricType.TOOL_PERFORMANCE,
    MetricType.PROMPT_QUALITY,
]



def _coerce_history_to_steps(history) -> list:
    """Best-effort conversion of dataset history into ordered trace steps."""
    if isinstance(history, list):
        return history
    if isinstance(history, tuple):
        return list(history)
    if hasattr(history, "tolist"):  # numpy ndarray / pandas array from read_parquet
        return list(history.tolist())
    if isinstance(history, str):
        text = history.strip()
        for parser in (ast.literal_eval, json.loads):
            try:
                parsed = parser(text)
                if isinstance(parsed, list):
                    return parsed
                if isinstance(parsed, dict):
                    # Some exports wrap messages under a common key.
                    for key in ("history", "messages", "trace", "steps"):
                        value = parsed.get(key)
                        if isinstance(value, list):
                            return value
                    return [parsed]
            except Exception:
                pass
        return [history]
    return [history]


def _format_trace_step(step) -> str:
    """Human-readable message block shown to LLM evaluators."""
    if isinstance(step, dict):
        agent = (
            step.get("agent")
            or step.get("agent_name")
            or step.get("name")
            or step.get("role")
            or step.get("sender")
        )
        content = (
            step.get("content")
            or step.get("message")
            or step.get("text")
            or step.get("response")
            or step.get("value")
        )
        if content is None:
            content = json.dumps(step, ensure_ascii=False, default=str)
        prefix = f"agent: {agent}\n" if agent else ""
        return prefix + str(content)
    return str(step)


def _format_indexed_raw_trace(history, question) -> str:
    """Format raw traces with stable zero-based message indices.

    LLM evaluators can cite these indices in `evidence[i].idx` when no
    explicit state_id/response_id is available. EvidenceVerifier then maps
    `[0]`, `[1]`, ... blocks back to the quoted text.
    """
    steps = _coerce_history_to_steps(history)
    lines = [
        "USER QUESTION:",
        str(question),
        "",
        "TRACE MESSAGES (zero-based indices; cite these numbers in evidence[i].idx):",
    ]
    for i, step in enumerate(steps):
        lines.append(f"[{i}] {_format_trace_step(step)}")
    return "\n".join(lines)


async def main(
    model_name: str,
    enable_tracing: bool,
    df: pd.DataFrame,
    result_file_name: str = "findings_",
    folder_name: str = "who&when_hand_findings",
    from_idx: int = 0,
    verifier: str = "deterministic",
):
    """Run all LLM findings-evaluators on each row of the dataset.

    Args:
        model_name: OpenRouter model id (e.g. "google/gemini-2.5-flash").
        enable_tracing: If True, trace each evaluation to the Langfuse judge project.
        df: DataFrame with columns history / question / question_ID / ground_truth(groundtruth) / is_correct(is_corrected).
        result_file_name: Prefix for per-task JSON result files.
        folder_name: Subfolder (next to this script) where JSON files are written.
        from_idx: Skip rows before this index (resume support).
    """

    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key

    model = OpenAIChatModel(
        model_name,
        provider="openrouter",
        settings={"temperature": 0.0},
    )

    if enable_tracing:
        print("\n=== Setting up judge client for evaluation traces ===")
        judge_client = get_langfuse_judge_client()
        print("Judge client initialized - evaluations will be traced to judge project")
    else:
        print("Tracing is disabled. Set enable_tracing=True to trace to Langfuse.")
        judge_client = None

    final_answer_verifier = FinalAnswerVerifier(model=model_name)

    evidence_verifier = (
        EvidenceVerifier(model=model, mode=verifier)
        if verifier == "llm"
        else EvidenceVerifier()
    )

    cnt = 0
    for task in range(len(df)):
        cnt += 1
        if cnt <= from_idx:
            continue

        row = df.iloc[task]
        eval_input = RawTraceInput(
            trace=_format_indexed_raw_trace(row["history"], row["question"])
        )
        print(f"\n=== Evaluating Task {task} ===")

        # Ground-truth metadata (the dataset uses two column naming variants)
        trace_metadata = {
            "task_id": str(row["question_ID"]),
            "trace_id": task,
        }
        if "groundtruth" in row.index:
            trace_metadata["ground_truth"] = row["groundtruth"]
            trace_metadata["correct_answer"] = row["is_corrected"]
        else:
            trace_metadata["ground_truth"] = row["ground_truth"]
            trace_metadata["correct_answer"] = row["is_correct"]

        ground_truth_value = trace_metadata["correct_answer"]
        try:
            ground_truth_value = int(ground_truth_value)
        except (TypeError, ValueError):
            ground_truth_value = None

        if judge_client is not None:
            await _evaluate_task_traced(
                judge_client,
                model,
                eval_input,
                row["history"],
                task,
                trace_metadata,
                ground_truth_value,
                final_answer_verifier,
                result_file_name,
                folder_name,
                evidence_verifier,
                verifier,
            )
        else:
            await _evaluate_task(
                model,
                eval_input,
                row["history"],
                task,
                trace_metadata,
                ground_truth_value,
                final_answer_verifier,
                result_file_name,
                folder_name,
                evidence_verifier,
                verifier,
            )


async def _run_all_metrics(model, eval_input: RawTraceInput) -> dict:
    """Run every LLM evaluator on one task and return {metric_name: MetricResult}."""
    findings_results: dict = {}
    for metric_type in LLM_METRICS_TO_TEST:
        print(f"\n--- Evaluating LLM metric: {metric_type.value} ---")
        try:
            metric = create_metric(metric_type, model)
            result = await metric.evaluate(eval_input)
            findings_results[metric_type.value] = result
            print(
                f"Metric: {result.metric_name} | findings: {len(result.findings)}"
            )
            for finding in result.findings:
                print(
                    f"  - [{finding.severity_estimate.value}/{finding.confidence_estimate.value}] "
                    f"{finding.problem_description[:120]}"
                )
        except Exception as e:
            print(f"Error evaluating {metric_type.value}: {e}")
            continue
    return findings_results


def _serialize_findings(findings_results: dict) -> dict:
    """Serialize {metric_name: MetricResult} into a JSON-friendly dict."""
    return {
        metric_name: {
            "metric_name": result.metric_name,
            "findings": [f.model_dump(mode="json") for f in result.findings],
        }
        for metric_name, result in findings_results.items()
    }


def _serialize_evidence_verification(evidence_results: dict) -> dict:
    """Serialize EvidenceVerifier outputs into a JSON-friendly dict."""
    return {
        metric_name: verification_result.model_dump(mode="json")
        for metric_name, verification_result in evidence_results.items()
    }


async def _run_final_answer_verification(
    final_answer_verifier: FinalAnswerVerifier,
    history,
    ground_truth,
) -> FinalAnswerVerificationResult | None:
    """Run final-answer verification for one Who & When trace."""

    try:
        result = await final_answer_verifier.verify_final_answer(
            history,
            gt=str(ground_truth),
            df="who_and_when",
        )
    except Exception as e:
        print(f"Error verifying final answer: {e}")
        return None

    if result is not None:
        print("\n--- Evaluating: final_answer_verification ---")
        print(
            f"Metric: {result.metric_name} | "
            f"score: {result.verdict} | "
            f"confidence: {result.confidence.value} | "
            f"method: {result.method}"
        )
    return result


def _save_results(
    serializable_results: dict,
    trace_metadata: dict,
    ground_truth_value,
    task: int,
    result_file_name: str,
    folder_name: str,
    verifier: str = "deterministic",
) -> Path:
    """Write the per-task JSON file. Returns the file path."""
    output_dir = Path(__file__).parent / folder_name
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / f"{result_file_name}{task}.json"

    payload = dict(serializable_results)
    payload["gt"] = ground_truth_value
    payload["label_answer"] = trace_metadata["ground_truth"]
    payload["reference_answer"] = trace_metadata["ground_truth"]
    payload["report"] = build_evaluation_report(
        payload,
        reference_answer=trace_metadata["ground_truth"],
        verifier_mode="llm" if verifier == "llm" else "soft",
    )
    # Keep final-answer verification in the report status only, not as a
    # separate top-level block in the saved per-task JSON.
    payload.pop("final_answer_verification", None)

    with open(output_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    return output_file


async def _evaluate_task(
    model,
    eval_input: RawTraceInput,
    history,
    task: int,
    trace_metadata: dict,
    ground_truth_value,
    final_answer_verifier: FinalAnswerVerifier,
    result_file_name: str,
    folder_name: str,
    evidence_verifier: EvidenceVerifier,
    verifier: str = "deterministic",
):
    """Run evaluators without Langfuse tracing."""
    findings_results = await _run_all_metrics(model, eval_input)
    if verifier == "llm":
        evidence_results = await evidence_verifier.verify_all_async(findings_results, eval_input)
    else:
        evidence_results = evidence_verifier.verify_all(findings_results, eval_input)
    final_answer_result = await _run_final_answer_verification(
        final_answer_verifier,
        history,
        trace_metadata["ground_truth"],
    )
    serializable_results = _serialize_findings(findings_results)
    serializable_results["non_llm_validators"] = run_on_trace({"history": _coerce_history_to_steps(history)})
    serializable_results["evidence_verification"] = _serialize_evidence_verification(evidence_results)
    if final_answer_result is not None:
        serializable_results["final_answer_verification"] = final_answer_result.model_dump(mode="json")
    _save_results(
        serializable_results,
        trace_metadata,
        ground_truth_value,
        task,
        result_file_name,
        folder_name,
        verifier=verifier,
    )


async def _evaluate_task_traced(
    judge_client,
    model,
    eval_input: RawTraceInput,
    history,
    task: int,
    trace_metadata: dict,
    ground_truth_value,
    final_answer_verifier: FinalAnswerVerifier,
    result_file_name: str,
    folder_name: str,
    evidence_verifier: EvidenceVerifier,
    verifier: str = "deterministic",
):
    """Run evaluators inside a Langfuse parent span; full findings go to span.output."""
    with judge_client.start_as_current_span(
        name=f"evaluate_task_{task}",
        input={"task_id": task, "trace_id": trace_metadata["task_id"]},
        metadata=trace_metadata,
    ) as span:
        judge_client.update_current_trace(
            tags=["maseval", "findings", f"task_id:{task}"]
        )

        findings_results = await _run_all_metrics(model, eval_input)
        if verifier == "llm":
            evidence_results = await evidence_verifier.verify_all_async(findings_results, eval_input)
        else:
            evidence_results = evidence_verifier.verify_all(findings_results, eval_input)
        final_answer_result = await _run_final_answer_verification(
            final_answer_verifier,
            history,
            trace_metadata["ground_truth"],
        )
        serializable_results = _serialize_findings(findings_results)
        serializable_results["non_llm_validators"] = run_on_trace({"history": _coerce_history_to_steps(history)})
        serializable_results["evidence_verification"] = _serialize_evidence_verification(evidence_results)
        if final_answer_result is not None:
            serializable_results["final_answer_verification"] = final_answer_result.model_dump(mode="json")
        _save_results(
            serializable_results,
            trace_metadata,
            ground_truth_value,
            task,
            result_file_name,
            folder_name,
            verifier=verifier,
        )

        # Full findings body as the span output (per design choice)
        span_output = {
            "findings": {
                m_name: [f.model_dump(mode="json") for f in result.findings]
                for m_name, result in findings_results.items()
            },
            "evidence_verification": _serialize_evidence_verification(evidence_results),
            "ground_truth_comparison": {
                "ground_truth": trace_metadata["ground_truth"],
                "match": trace_metadata["correct_answer"],
            },
        }
        if final_answer_result is not None:
            span_output["final_answer_verification"] = final_answer_result.model_dump(mode="json")
        span.update(output=span_output)


if __name__ == "__main__":
    import argparse

    df_hc = pd.read_parquet(
        "hf://datasets/Kevin355/Who_and_When/Hand-Crafted.parquet"
    )
    df_algo = pd.read_parquet(
        "hf://datasets/Kevin355/Who_and_When/Algorithm-Generated.parquet"
    )

    # Resume support: read FROM_IDX from env (default 0). Useful when the
    # script is interrupted and you want to continue from a specific task.
    try:
        from_idx = int(os.environ.get("FROM_IDX", "0"))
    except ValueError:
        from_idx = 0

    parser = argparse.ArgumentParser(description="Run Who&When findings judges.")
    parser.add_argument(
        "--run",
        choices=("hc", "algo", "both"),
        default="both",
        help="Which dataset(s) to evaluate.",
    )
    parser.add_argument(
        "--verifier",
        choices=("deterministic", "llm"),
        default="deterministic",
        help="EvidenceVerifier strategy: deterministic (default) or LLM-judged grounding.",
    )
    args = parser.parse_args()

    async def _run_selected(selected_run: str):
        if selected_run in ("algo", "both"):
            await main(
                model_name="google/gemini-2.5-flash",
                enable_tracing=True,
                df=df_algo,
                result_file_name="gemini_findings_",
                folder_name="who&when_algo_gemini_idx_msg_v2",
                from_idx=from_idx,
                verifier=args.verifier,
            )
        if selected_run in ("hc", "both"):
            await main(
                model_name="google/gemini-2.5-flash:google-ai-studio",
                enable_tracing=True,
                df=df_hc,
                result_file_name="gemini_findings_",
                folder_name="who&when_hand_gemini_idx_msg_studio",
                from_idx=from_idx,
                verifier=args.verifier,
            )

    asyncio.run(_run_selected(args.run))
