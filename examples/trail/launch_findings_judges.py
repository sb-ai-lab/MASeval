"""Launch all LLM findings-evaluators + non-LLM validators on the TRAIL (GAIA) dataset.

Each of the 117 trace files is evaluated by every LLM evaluator (metric) and by
the deterministic (non-LLM) validators. Raw findings are:

1. Saved to a per-task JSON file under ``trail_gemini_findings_v1/``.
2. Traced to Langfuse under a parent span, with the full findings body as the
   span's ``output``.

After all metrics run, a ``FinalAnswerVerifier`` is invoked without a ground
truth (``gt=None``), so the LLM judge itself decides whether the trace's final
answer is correct/ideal. This verdict is stored in ``report.status.answer_status``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich import print
from tqdm import tqdm

load_dotenv(".env")

# Disable the default OTEL tracer provider BEFORE importing maseval, so that
# the Langfuse judge client can install its own provider later without conflict.
os.environ["OTEL_TRACES_EXPORTER"] = "none"

from opentelemetry import trace  # noqa: E402

trace.set_tracer_provider(None)

from pydantic_ai.models.openai import OpenAIChatModel  # noqa: E402

from maseval import get_langfuse_judge_client  # noqa: E402
from maseval.evaluation_blocks.final_answer_verification import (  # noqa: E402
    FinalAnswerVerifier,
)
from maseval.metrics import EvidenceVerifier, MetricType, create_metric  # noqa: E402
from maseval.models import RawTraceInput  # noqa: E402
from maseval.reporting import build_evaluation_report  # noqa: E402
from maseval.run_stats import aggregate_task_stats, extract_usage  # noqa: E402
from maseval.validators import run_on_trace  # noqa: E402
from maseval.validators.base import trail_to_spans  # noqa: E402


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


# Override per machine with $TRAIL_GAIA_DIR / $TRAIL_ANNO_DIR.
TRACES_DIR = Path(os.environ.get(
    "TRAIL_GAIA_DIR",
    "/Users/alina/trail-benchmark/benchmarking/data/GAIA",
))
ANNOTATIONS_DIR = Path(os.environ.get(
    "TRAIL_ANNO_DIR",
    "/Users/alina/trail-benchmark/benchmarking/processed_annotations_gaia",
))
OUTPUT_DIR = Path(__file__).parent / "trail_gemini_findings_v1"

# Binarization threshold for the gold annotation ``scores[0].overall``:
# >= 4 -> trace is "correct" (gt=1), < 4 -> problematic (gt=0).
GT_OVERALL_THRESHOLD = 4.0


def _iter_trail_traces(traces_dir: Path):
    """Yield ``(trace_path, raw_trace)`` for every JSON in ``traces_dir``."""

    for path in sorted(traces_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                yield path, json.load(f)
        except Exception as exc:  # noqa: BLE001
            print(f"[red]Skipping broken trace file {path}: {exc}[/red]")


def _load_annotation(trace_id: str) -> dict[str, Any] | None:
    """Load gold annotation ``{errors, scores}`` matching ``trace_id``."""

    path = ANNOTATIONS_DIR / f"{trace_id}.json"
    if not path.is_file():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"[red]Bad annotation {path}: {exc}[/red]")
        return None


def _binarize_annotation(annotation: dict[str, Any] | None) -> tuple[int | None, float | None]:
    """Return ``(gt, overall_score)`` following ``exp_calc_metrics_TRAIL``.

    If the annotation has ``scores[0].overall``, use threshold >= 4 -> 1, else 0.
    Otherwise fall back to ``errors``: no errors -> 1, all errors -> 0, mixed -> None.
    """

    if annotation is None:
        return None, None

    scores = annotation.get("scores") or []
    if scores and isinstance(scores[0], dict) and "overall" in scores[0]:
        overall = float(scores[0]["overall"])
        return (1 if overall >= GT_OVERALL_THRESHOLD else 0), overall

    errors = annotation.get("errors") or []
    if not errors:
        return 1, None

    return None, None


def _format_indexed_trail_trace(raw_trace: dict) -> str:
    """Format a TRAIL raw trace dict into indexed, LLM-judge-friendly text.

    Uses ``trail_to_spans`` to flatten the recursive OTel span tree into a list
    of normalized spans, then re-numbers them with sequential zero-based indices
    ``[0]``, ``[1]``, ... so that ``EvidenceVerifier`` can map LLM-judge
    citations back to span content.

    The original hex ``idx``/``span_id`` of each span is still noted in the
    header when available.
    """

    spans = trail_to_spans(raw_trace)
    lines = [
        "TRACE SPANS (zero-based indices; cite these numbers in evidence[i].idx):",
    ]

    for i, span in enumerate(spans):
        agent = span.get("agent")
        kind = span.get("kind")
        parent = span.get("parent")
        orig_idx = span.get("idx") or span.get("span_id")

        header_parts = [f"[{i}]"]
        if orig_idx is not None and str(orig_idx) != str(i):
            header_parts.append(f"span_id={orig_idx}")
        if kind:
            header_parts.append(f"kind={kind}")
        if parent is not None:
            header_parts.append(f"parent={parent}")
        if agent:
            header_parts.append(f"agent={agent}")

        lines.append(" ".join(header_parts))

        text = span.get("text") or ""
        if text:
            lines.append(text)
        lines.append("")

    return "\n".join(lines)


def _serialize_findings(findings_results: dict) -> dict:
    """Serialize ``{metric_name: MetricResult}`` into a JSON-friendly dict."""

    return {
        metric_name: {
            "metric_name": result.metric_name,
            "findings": [finding.model_dump(mode="json") for finding in result.findings],
        }
        for metric_name, result in findings_results.items()
    }


def _serialize_evidence_verification(evidence_results: dict) -> dict:
    """Serialize EvidenceVerifier outputs into a JSON-friendly dict."""

    return {
        metric_name: verification_result.model_dump(mode="json")
        for metric_name, verification_result in evidence_results.items()
    }


async def _run_all_metrics(model, eval_input: RawTraceInput) -> tuple[dict, dict]:
    """Run every LLM evaluator on one task.

    Returns ``(findings_results, metric_status)`` — the latter carries per-metric
    timing + token usage for the ``task_stats`` rollup.
    """

    findings_results: dict = {}
    metric_status: dict = {}

    for metric_type in LLM_METRICS_TO_TEST:
        name = metric_type.value
        print(f"\n--- Evaluating LLM metric: {name} ---")
        metric = None
        t0 = time.perf_counter()
        try:
            metric = create_metric(metric_type, model)
            result = await metric.evaluate(eval_input)
            inp, out, tot = extract_usage(getattr(metric, "last_usage", None))
            findings_results[name] = result
            metric_status[name] = {
                "status": "ok", "duration_s": round(time.perf_counter() - t0, 3),
                "input_tokens": inp, "output_tokens": out, "total_tokens": tot,
            }

            print(f"Metric: {result.metric_name} | findings: {len(result.findings)} "
                  f"| in={inp} out={out}")
            for finding in result.findings:
                print(
                    f"  - [{finding.severity_estimate.value}/"
                    f"{finding.confidence_estimate.value}] "
                    f"{finding.problem_description[:120]}"
                )
        except Exception as exc:  # noqa: BLE001
            inp, out, tot = extract_usage(getattr(metric, "last_usage", None))
            metric_status[name] = {
                "status": "failed", "detail": f"{type(exc).__name__}: {exc}"[:300],
                "duration_s": round(time.perf_counter() - t0, 3),
                "input_tokens": inp, "output_tokens": out, "total_tokens": tot,
            }
            print(f"Error evaluating {name}: {exc}")
            continue

    return findings_results, metric_status


async def _run_final_answer_verification_no_gt(
    final_answer_verifier: FinalAnswerVerifier,
    raw_trace: dict,
) -> Any | None:
    """Run the final-answer judge without ground truth.

    With ``gt=None`` the verifier falls back to a ``MAS_TASK_COMPLETION`` LLM judge
    that inspects the dumped trace and emits ``verdict=ideal|poor``.
    """

    try:
        return await final_answer_verifier.verify_final_answer(
            raw_trace,
            gt=None,
            df="trail_gaia",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Final-answer verifier failed: {exc}")
        return None


def _save_results(
    serializable_results: dict,
    trace_id: str,
    trace_path: Path,
    annotation: dict[str, Any] | None,
    gt: int | None,
    gt_overall_score: float | None,
    output_dir: Path,
) -> Path:
    """Write the per-task JSON file. Returns the file path."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{trace_id}.json"

    payload = dict(serializable_results)
    payload["trace_id"] = trace_id
    payload["benchmark"] = "trail"
    payload["split"] = "gaia"
    payload["source_file"] = str(trace_path)
    payload["gt"] = gt
    payload["gt_overall_score"] = gt_overall_score

    if annotation is not None:
        payload["labels"] = annotation

    # IMPORTANT:
    # build_evaluation_report must see final_answer_verification, otherwise it
    # cannot put the no-GT final-answer check into report.status.answer_status.
    payload["report"] = build_evaluation_report(
        payload,
        reference_answer=None,
    )

    # Keep final-answer verification only inside report.status.answer_status,
    # not as a duplicated top-level block in the saved JSON.
    payload.pop("final_answer_verification", None)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
    return output_file


async def _evaluate_task(
    model,
    final_answer_verifier: FinalAnswerVerifier,
    raw_trace: dict,
    trace_id: str,
    trace_path: Path,
    annotation: dict[str, Any] | None,
    gt: int | None,
    gt_overall_score: float | None,
    output_dir: Path,
):
    """Run evaluators without Langfuse tracing."""

    eval_input = RawTraceInput(trace=_format_indexed_trail_trace(raw_trace))

    _t0 = time.perf_counter()
    findings_results, metric_status = await _run_all_metrics(model, eval_input)
    task_stats = aggregate_task_stats(metric_status, time.perf_counter() - _t0)
    evidence_results = EvidenceVerifier().verify_all(findings_results, eval_input)
    final_answer_result = await _run_final_answer_verification_no_gt(
        final_answer_verifier,
        raw_trace,
    )
    print("=" * 80)
    print("FINAL ANSWER RESULT:")
    print(final_answer_result)
    print("=" * 80)

    serializable_results = _serialize_findings(findings_results)
    serializable_results["task_stats"] = task_stats
    serializable_results["non_llm_validators"] = run_on_trace(raw_trace)
    serializable_results["evidence_verification"] = _serialize_evidence_verification(
        evidence_results
    )

    if final_answer_result is not None:
        serializable_results["final_answer_verification"] = final_answer_result.model_dump(
            mode="json"
        )

    _save_results(
        serializable_results=serializable_results,
        trace_id=trace_id,
        trace_path=trace_path,
        annotation=annotation,
        gt=gt,
        gt_overall_score=gt_overall_score,
        output_dir=output_dir,
    )


async def _evaluate_task_traced(
    judge_client,
    model,
    final_answer_verifier: FinalAnswerVerifier,
    raw_trace: dict,
    trace_id: str,
    trace_path: Path,
    annotation: dict[str, Any] | None,
    gt: int | None,
    gt_overall_score: float | None,
    output_dir: Path,
):
    """Run evaluators inside a Langfuse parent span."""

    trace_metadata = {
        "trace_id": trace_id,
        "benchmark": "trail",
        "split": "gaia",
        "gt": gt,
        "gt_overall_score": gt_overall_score,
    }

    with judge_client.start_as_current_span(
        name=f"evaluate_task_{trace_id}",
        input={"trace_id": trace_id, "annotation_source": str(trace_path)},
        metadata=trace_metadata,
    ) as span:
        judge_client.update_current_trace(
            tags=["maseval", "findings", f"task_id:{trace_id}", "trail", "gaia"]
        )

        eval_input = RawTraceInput(trace=_format_indexed_trail_trace(raw_trace))

        _t0 = time.perf_counter()
        findings_results, metric_status = await _run_all_metrics(model, eval_input)
        task_stats = aggregate_task_stats(metric_status, time.perf_counter() - _t0)
        evidence_results = EvidenceVerifier().verify_all(findings_results, eval_input)
        final_answer_result = await _run_final_answer_verification_no_gt(
            final_answer_verifier,
            raw_trace,
        )

        serializable_results = _serialize_findings(findings_results)
        serializable_results["task_stats"] = task_stats
        serializable_results["non_llm_validators"] = run_on_trace(raw_trace)
        serializable_results["evidence_verification"] = _serialize_evidence_verification(
            evidence_results
        )

        if final_answer_result is not None:
            serializable_results["final_answer_verification"] = (
                final_answer_result.model_dump(mode="json")
            )

        _save_results(
            serializable_results=serializable_results,
            trace_id=trace_id,
            trace_path=trace_path,
            annotation=annotation,
            gt=gt,
            gt_overall_score=gt_overall_score,
            output_dir=output_dir,
        )

        span_output = {
            "findings": {
                metric_name: [
                    finding.model_dump(mode="json") for finding in result.findings
                ]
                for metric_name, result in findings_results.items()
            },
            "evidence_verification": _serialize_evidence_verification(evidence_results),
            "final_answer_verification": (
                final_answer_result.model_dump(mode="json")
                if final_answer_result is not None
                else None
            ),
            "gold": {
                "gt": gt,
                "gt_overall_score": gt_overall_score,
            },
        }
        span.update(output=span_output)


async def main(
    model_name: str = "google/gemini-2.5-flash",
    enable_tracing: bool = True,
    traces_dir: Path = TRACES_DIR,
    output_dir: Path = OUTPUT_DIR,
    from_idx: int = 0,
):
    """Run all evaluators over TRAIL GAIA traces.

    Args:
        model_name: OpenRouter model id.
        enable_tracing: If True, trace each evaluation to Langfuse.
        traces_dir: Directory with TRAIL trace JSON files.
        output_dir: Directory where per-task JSON files are written.
        from_idx: Skip traces before this index.
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

    trace_items = list(_iter_trail_traces(traces_dir))
    if not trace_items:
        raise FileNotFoundError(f"No trace JSON files in {traces_dir}")

    print(f"\nFound {len(trace_items)} trace files in {traces_dir}")

    cnt = 0
    skipped = 0

    for trace_path, raw_trace in tqdm(trace_items, desc="Evaluating TRAIL traces"):
        cnt += 1
        if cnt <= from_idx:
            continue

        trace_id = str(raw_trace.get("trace_id") or trace_path.stem)

        # Resume support: skip already-evaluated traces.
        if (output_dir / f"{trace_id}.json").is_file():
            skipped += 1
            continue

        annotation = _load_annotation(trace_id)
        gt, gt_overall_score = _binarize_annotation(annotation)

        print(f"\n=== Evaluating Task {trace_id} (idx={cnt}) ===")
        print(f"trace={trace_path.name} | gt={gt} | overall_score={gt_overall_score}")

        if judge_client is not None:
            await _evaluate_task_traced(
                judge_client,
                model,
                final_answer_verifier,
                raw_trace,
                trace_id,
                trace_path,
                annotation,
                gt,
                gt_overall_score,
                output_dir,
            )
        else:
            await _evaluate_task(
                model,
                final_answer_verifier,
                raw_trace,
                trace_id,
                trace_path,
                annotation,
                gt,
                gt_overall_score,
                output_dir,
            )

    print(f"\nDone. Evaluated {cnt - skipped} traces. Skipped {skipped} already evaluated.")
    print(f"Results in: {output_dir}")


if __name__ == "__main__":
    try:
        from_idx = int(os.environ.get("FROM_IDX", "0"))
    except ValueError:
        from_idx = 0

    import asyncio

    asyncio.run(
        main(
            model_name="google/gemini-2.5-flash",
            enable_tracing=True,
            traces_dir=TRACES_DIR,
            output_dir=OUTPUT_DIR,
            from_idx=from_idx,
        )
    )
