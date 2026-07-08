"""Run the LLM findings-evaluators on microsoft/AgentRx trajectories.

Mirrors the Who&When findings runner, but for AgentRx:
  * spans are keyed by the native 1-based step ``index`` (== gold ``step_number``);
  * FinalAnswerVerifier runs in its no-ground-truth mode (the MAS Task
    Completion judge), which needs no gold final answer -- AgentRx has none;
  * no deterministic ``non_llm_validators`` (AgentRx is not a validator format;
    running them would emit misaligned spans).

Each trajectory is evaluated by every LLM metric; raw findings + EvidenceVerifier
output + a compact ``report`` are written to one per-trajectory JSON file.

Usage:
    python launch_findings_judges.py --config magentic --model google/gemini-2.5-flash
    python launch_findings_judges.py --config tau --from-idx 0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")

# Disable the default OTEL tracer provider BEFORE importing maseval.
os.environ["OTEL_TRACES_EXPORTER"] = "none"
from opentelemetry import trace  # noqa: E402

trace.set_tracer_provider(None)

from pydantic_ai.models.openai import OpenAIChatModel  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
for p in (ROOT / "src", THIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from maseval.evaluation_blocks.final_answer_verification import (  # noqa: E402
    FinalAnswerVerificationResult,
    FinalAnswerVerifier,
)
from maseval.metrics import EvidenceVerifier, MetricType, create_metric  # noqa: E402
from maseval.models import RawTraceInput  # noqa: E402
from maseval.reporting import build_evaluation_report  # noqa: E402

import agentrx_data  # noqa: E402

# Same LLM evaluator set as the Who&When findings run.
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

METRIC_TIMEOUT = float(os.environ.get("METRIC_TIMEOUT", "600"))
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "4096"))


async def _run_all_metrics(model, eval_input: RawTraceInput) -> dict:
    """Run every LLM evaluator on one trajectory; return {metric_name: result}."""
    findings_results: dict = {}
    for metric_type in LLM_METRICS_TO_TEST:
        print(f"  --- {metric_type.value} ---")
        try:
            metric = create_metric(metric_type, model)
            result = await asyncio.wait_for(
                metric.evaluate(eval_input), timeout=METRIC_TIMEOUT
            )
            findings_results[metric_type.value] = result
            print(f"      findings: {len(result.findings)}")
        except Exception as exc:  # noqa: BLE001 - isolate one metric's failure
            print(f"      error: {type(exc).__name__}: {exc}")
    return findings_results


def _serialize_findings(findings_results: dict) -> dict:
    return {
        metric_name: {
            "metric_name": result.metric_name,
            "findings": [f.model_dump(mode="json") for f in result.findings],
        }
        for metric_name, result in findings_results.items()
    }


def _serialize_evidence_verification(evidence_results: dict) -> dict:
    return {
        metric_name: verification_result.model_dump(mode="json")
        for metric_name, verification_result in evidence_results.items()
    }


async def _run_final_answer_verification(
    final_answer_verifier: FinalAnswerVerifier, ex
) -> FinalAnswerVerificationResult | None:
    """Run FinalAnswerVerifier in no-ground-truth mode for one trajectory.

    With ``gt=None`` the verifier routes to the MAS Task Completion judge, which
    assesses whether the task was completed from the trace alone (no gold answer
    needed). ``df`` only needs to be non-None here. The trace is passed
    structured (instruction + steps) so the judge knows what "complete" means.
    """
    try:
        return await final_answer_verifier.verify_final_answer(
            {"instruction": ex.instruction, "steps": ex.steps},
            gt=None,
            df="agentrx",
        )
    except Exception as exc:  # noqa: BLE001 - one judge failure must not abort a trajectory
        print(f"      final_answer_verification error: {type(exc).__name__}: {exc}")
        return None


async def _evaluate_example(
    model, ex, folder: Path, result_file_name: str, final_answer_verifier
) -> None:
    """Evaluate one AgentRx trajectory and write its per-trajectory JSON."""
    eval_input = RawTraceInput(
        trace=agentrx_data.format_trace(ex.steps, ex.instruction)
    )
    findings_results = await _run_all_metrics(model, eval_input)
    evidence_results = EvidenceVerifier().verify_all(findings_results, eval_input)
    final_answer_result = await _run_final_answer_verification(final_answer_verifier, ex)

    payload = _serialize_findings(findings_results)
    # Top-level id so the scorer can match by trajectory_id as well as by
    # filename index (trajectory_id is one of maseval's metadata id keys).
    payload["trajectory_id"] = ex.trajectory_id
    payload["evidence_verification"] = _serialize_evidence_verification(evidence_results)
    # No-GT MAS task-completion verdict; feeds the report's answer_status.
    if final_answer_result is not None:
        payload["final_answer_verification"] = final_answer_result.model_dump(mode="json")
    # AgentRx gold (for downstream inspection; scoring uses the gold table).
    payload["agentrx_meta"] = {
        "trajectory_id": ex.trajectory_id,
        "num_steps": len(ex.steps),
        "gold_failed_agents": list(ex.all_failed_agents),
        "gold_failure_steps": list(ex.all_failure_steps),
        "root_cause_agent": ex.root_cause_agent,
        "root_cause_step": ex.root_cause_step,
    }
    # AgentRx has no gold final answer -> reference_based answer status is
    # unavailable; answer_status comes from the no-GT task-completion verdict.
    payload["report"] = build_evaluation_report(payload, reference_answer=None)

    folder.mkdir(parents=True, exist_ok=True)
    out_file = folder / f"{result_file_name}{ex.row_index}.json"
    out_file.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"  saved -> {out_file}")


async def main(
    model_name: str,
    config: str,
    folder_name: str | None = None,
    result_file_name: str = "findings_",
    from_idx: int = 0,
    resume: bool = True,
) -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (check examples/agentrx/.env).")
    os.environ["OPENROUTER_API_KEY"] = api_key

    model = OpenAIChatModel(
        model_name,
        provider="openrouter",
        settings={"temperature": 0.0, "max_tokens": MAX_OUTPUT_TOKENS},
    )
    final_answer_verifier = FinalAnswerVerifier(model=model_name)

    examples = agentrx_data.load_examples(config)
    folder = THIS_DIR / (folder_name or f"agentrx_{config}_findings")
    print(f"AgentRx/{config}: {len(examples)} trajectories -> {folder}")

    for ex in examples:
        if ex.row_index < from_idx:
            continue
        out_file = folder / f"{result_file_name}{ex.row_index}.json"
        if resume and out_file.exists():
            print(f"[{ex.row_index}] exists, skipping")
            continue
        print(f"\n=== [{ex.row_index}] {ex.trajectory_id} ({len(ex.steps)} steps) ===")
        await _evaluate_example(model, ex, folder, result_file_name, final_answer_verifier)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM findings judges on AgentRx.")
    parser.add_argument("--config", choices=list(agentrx_data.CONFIGS), default="magentic")
    parser.add_argument("--model", default="google/gemini-2.5-flash")
    parser.add_argument("--folder", default=None, help="Output subfolder name.")
    parser.add_argument("--from-idx", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true", help="Re-run existing files.")
    args = parser.parse_args()

    asyncio.run(
        main(
            model_name=args.model,
            config=args.config,
            folder_name=args.folder,
            from_idx=args.from_idx,
            resume=not args.no_resume,
        )
    )
