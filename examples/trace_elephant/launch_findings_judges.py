"""Run the LLM findings-evaluators on TraceElephant failure traces.

Mirrors the AgentRx findings runner, but for TraceElephant:
  * spans are keyed by the 1-based step position (== gold ``mistake_step``);
  * FinalAnswerVerifier runs in its no-ground-truth mode (the MAS Task
    Completion judge) -- with ``gt=None`` it never touches a ``df`` extractor, so
    ``df="trace_elephant"`` is fine and no new extractor is needed;
  * no deterministic ``non_llm_validators`` (TraceElephant is not a validator
    format; running them would emit misaligned spans).

Each trace is evaluated by every LLM metric; raw findings + EvidenceVerifier
output + a compact ``report`` are written to one per-trace JSON file.

Usage:
    python launch_findings_judges.py --model google/gemini-2.5-flash
    python launch_findings_judges.py --system captain --from-idx 0
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

import trace_elephant_data as ted  # noqa: E402

# Same LLM evaluator set as the Who&When / AgentRx findings runs.
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
# Retries per evaluator. OpenRouter returns finish_reason='error' when the
# upstream (gemini) call fails transiently under load; pydantic_ai then raises
# (surfacing as UnexpectedModelBehavior / an IndexError in maseval's error path).
# These are retryable, so give each evaluator a few attempts with backoff.
METRIC_MAX_ATTEMPTS = int(os.environ.get("METRIC_MAX_ATTEMPTS", "4"))

SYSTEMS = ("captain", "magentic", "swe")


async def _run_all_metrics(model, eval_input: RawTraceInput) -> dict:
    """Run every LLM evaluator on one trace; return {metric_name: result}."""
    findings_results: dict = {}
    for metric_type in LLM_METRICS_TO_TEST:
        print(f"  --- {metric_type.value} ---")
        for attempt in range(1, METRIC_MAX_ATTEMPTS + 1):
            try:
                metric = create_metric(metric_type, model)
                result = await asyncio.wait_for(
                    metric.evaluate(eval_input), timeout=METRIC_TIMEOUT
                )
                findings_results[metric_type.value] = result
                print(f"      findings: {len(result.findings)}")
                break
            except Exception as exc:  # noqa: BLE001 - isolate one metric's failure
                if attempt < METRIC_MAX_ATTEMPTS:
                    await asyncio.sleep(2 * attempt)
                    print(f"      retry {attempt}/{METRIC_MAX_ATTEMPTS - 1} after {type(exc).__name__}")
                else:
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
    """Run FinalAnswerVerifier in no-ground-truth mode for one trace.

    With ``gt=None`` the verifier routes to the MAS Task Completion judge, which
    assesses whether the task was completed from the trace alone (no gold answer
    needed). ``df`` only needs to be non-None here. The trace is passed
    structured (instruction + steps) so the judge knows what "complete" means.
    """
    # Cap the history the MTC judge sees so oversized SWE traces don't blow the
    # model context (the judge only needs enough to assess task completion).
    n = max(1, len(ex.history))
    cap = max(ted._MIN_STEP_CHARS, ted.TRACE_CHAR_BUDGET // n)
    steps = [
        {"name": s["name"],
         "content": s["content"] if len(s["content"]) <= cap else s["content"][:cap] + "…[truncated]"}
        for s in ex.history
    ]
    try:
        return await final_answer_verifier.verify_final_answer(
            {"instruction": ex.question, "steps": steps},
            gt=None,
            df="trace_elephant",
        )
    except Exception as exc:  # noqa: BLE001 - one judge failure must not abort a trace
        print(f"      final_answer_verification error: {type(exc).__name__}: {exc}")
        return None


async def _evaluate_example(
    model, ex, folder: Path, result_file_name: str, final_answer_verifier
) -> None:
    """Evaluate one TraceElephant trace and write its per-trace JSON."""
    eval_input = RawTraceInput(trace=ted.format_trace(ex))
    findings_results = await _run_all_metrics(model, eval_input)
    evidence_results = EvidenceVerifier().verify_all(findings_results, eval_input)
    final_answer_result = await _run_final_answer_verification(final_answer_verifier, ex)

    payload = _serialize_findings(findings_results)
    # Top-level id so the scorer can match by task_name as well as by filename index.
    payload["task_name"] = ex.task_name
    payload["evidence_verification"] = _serialize_evidence_verification(evidence_results)
    # No-GT MAS task-completion verdict; feeds the report's answer_status.
    if final_answer_result is not None:
        payload["final_answer_verification"] = final_answer_result.model_dump(mode="json")
    # TraceElephant gold (for inspection; scoring uses the gold table).
    payload["trace_elephant_meta"] = {
        "task_name": ex.task_name,
        "system_category": ex.system_category,
        "system_name": ex.system_name,
        "num_steps": len(ex.history),
        "mistake_agent": ex.mistake_agent,
        "mistake_step": ex.mistake_step,
    }
    # TraceElephant has no gold final answer for the judge path -> answer_status
    # comes from the no-GT task-completion verdict.
    payload["report"] = build_evaluation_report(payload, reference_answer=None)

    folder.mkdir(parents=True, exist_ok=True)
    out_file = folder / f"{result_file_name}{ex.row_index}.json"
    out_file.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"  saved -> {out_file}")


async def main(
    model_name: str,
    system: str = "all",
    data_dir: str | None = None,
    folder_name: str | None = None,
    result_file_name: str = "findings_",
    from_idx: int = 0,
    resume: bool = True,
) -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (check examples/trace_elephant/.env).")
    os.environ["OPENROUTER_API_KEY"] = api_key

    model = OpenAIChatModel(
        model_name,
        provider="openrouter",
        settings={"temperature": 0.0, "max_tokens": MAX_OUTPUT_TOKENS},
    )
    final_answer_verifier = FinalAnswerVerifier(model=model_name)

    data_dir = data_dir or str(THIS_DIR / "data")
    examples = ted.load_examples(data_dir)
    if system != "all":
        examples = [e for e in examples if e.system_category == system]
        # Re-index the filtered subset so row_index is contiguous per run.
        for i, e in enumerate(examples):
            e.row_index = i
    folder = THIS_DIR / (folder_name or f"trace_elephant_{system}_findings")
    print(f"TraceElephant/{system}: {len(examples)} traces -> {folder}")

    for ex in examples:
        if ex.row_index < from_idx:
            continue
        out_file = folder / f"{result_file_name}{ex.row_index}.json"
        if resume and out_file.exists():
            print(f"[{ex.row_index}] exists, skipping")
            continue
        print(f"\n=== [{ex.row_index}] {ex.task_name} ({len(ex.history)} steps) ===")
        await _evaluate_example(model, ex, folder, result_file_name, final_answer_verifier)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM findings judges on TraceElephant.")
    parser.add_argument("--system", choices=("all", *SYSTEMS), default="all")
    parser.add_argument("--model", default="google/gemini-2.5-flash")
    parser.add_argument("--data-dir", default=None, help="Path to extracted data/ dir.")
    parser.add_argument("--folder", default=None, help="Output subfolder name.")
    parser.add_argument("--from-idx", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true", help="Re-run existing files.")
    args = parser.parse_args()

    asyncio.run(
        main(
            model_name=args.model,
            system=args.system,
            data_dir=args.data_dir,
            folder_name=args.folder,
            from_idx=args.from_idx,
            resume=not args.no_resume,
        )
    )
