"""Run the maseval findings pipeline on the AEGIS benchmark (cross-dataset eval, issue #149).

AEGIS records (Aegis-Bench/test.jsonl) have the shape:
    {id, metadata, input:{query, conversation_history:[{step, agent_name, content, phase}]},
     output:{faulty_agents:[{agent_name, error_type, injection_strategy}]},
     ground_truth:{correct_answer, ...}}

For each record we run every LLM evaluator + the deterministic non-LLM validators,
verify evidence, build the diagnostic report, and save one WW-shaped JSON per record
so the boary scorer (maseval.diagnostic_accuracy) can consume it unchanged.

The AEGIS `id` is NOT unique (different injected-error samples share an id), so the
stable global row index is used as the matching key (`sample_id`); the original id is
kept only as informational `aegis_id`. AEGIS ground truth (`output.faulty_agents`) is
copied into each result file for scoring.

Usage (from this directory, with a .env holding OPENROUTER_API_KEY):
    python launch_aegis.py --limit 5                 # smoke test on first 5
    python launch_aegis.py                           # full run (all records)
    python launch_aegis.py --from-idx 100 --limit 50 # resume / shard slice
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")

# Disable the default OTEL tracer provider BEFORE importing maseval.
os.environ["OTEL_TRACES_EXPORTER"] = "none"
from opentelemetry import trace

trace.set_tracer_provider(None)

from pydantic_ai.models.openai import OpenAIChatModel
from rich import print

from maseval.evaluation_blocks.final_answer_verification import FinalAnswerVerifier
from maseval.metrics import EvidenceVerifier, MetricType, create_metric
from maseval.models import RawTraceInput
from maseval.reporting import build_evaluation_report
from maseval.validators import run_on_trace

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


def _aegis_steps(rec: dict) -> list:
    """Extract the conversation_history step list from an AEGIS record."""
    inp = rec.get("input") or {}
    steps = inp.get("conversation_history")
    return steps if isinstance(steps, list) else []


def _aegis_predicted_answer(rec: dict) -> str | None:
    """Same extraction FinalAnswerVerifier uses internally, surfaced for the report.

    build_evaluation_report's generic predicted-answer inference only recognizes a
    literal ``FINAL ANSWER:`` marker in the trace text (the WW convention); AEGIS
    traces carry the answer in `input.final_output` instead, so it must be passed
    in explicitly or `report.status.answer_status.predicted_answer` stays null.
    """
    inp = rec.get("input") or {}
    final_output = inp.get("final_output")
    if final_output:
        return str(final_output)
    for state in reversed(inp.get("conversation_history") or []):
        content = state.get("content") if isinstance(state, dict) else None
        if content:
            return str(content)
    return None


def _format_step(step) -> str:
    if isinstance(step, dict):
        agent = step.get("agent_name") or step.get("agent") or step.get("role")
        content = step.get("content")
        if content is None:
            content = json.dumps(step, ensure_ascii=False, default=str)
        prefix = f"agent: {agent}\n" if agent else ""
        return prefix + str(content)
    return str(step)


def _format_indexed_raw_trace(steps: list, query: str) -> str:
    """Zero-based indexed trace text; LLM evaluators cite these indices as evidence[i].idx."""
    lines = [
        "USER QUESTION:",
        str(query),
        "",
        "TRACE MESSAGES (zero-based indices; cite these numbers in evidence[i].idx):",
    ]
    for i, step in enumerate(steps):
        lines.append(f"[{i}] {_format_step(step)}")
    return "\n".join(lines)


async def _run_all_metrics(model, eval_input: RawTraceInput) -> dict:
    """Run all LLM evaluators for one trace sequentially (mirrors the WW launcher;
    avoids bursting OpenRouter rate limits with concurrent calls)."""
    findings_results: dict = {}
    for metric_type in LLM_METRICS_TO_TEST:
        print(f"  -- LLM metric: {metric_type.value}")
        try:
            metric = create_metric(metric_type, model)
            result = await metric.evaluate(eval_input)
            findings_results[metric_type.value] = result
            print(f"     findings: {len(result.findings)}")
        except Exception as e:  # noqa: BLE001 - isolate a single metric failure
            print(f"     [skip] {metric_type.value}: {e}")
            continue
    return findings_results


def _serialize_findings(findings_results: dict) -> dict:
    return {
        name: {
            "metric_name": result.metric_name,
            "findings": [f.model_dump(mode="json") for f in result.findings],
        }
        for name, result in findings_results.items()
    }


def _serialize_evidence(evidence_results: dict) -> dict:
    return {name: r.model_dump(mode="json") for name, r in evidence_results.items()}


async def _run_final_answer_verification(verifier: FinalAnswerVerifier, rec: dict, gt_answer):
    try:
        return await verifier.verify_final_answer(rec, gt=gt_answer, df="aegis")
    except Exception as e:  # noqa: BLE001 - final-answer verification is best-effort
        print(f"  [skip] final_answer_verification: {e}")
        return None


async def main(input_path: str, out_dir: str, model_name: str, from_idx: int, limit: int | None):
    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} AEGIS records from {input_path}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = OpenAIChatModel(model_name, provider="openrouter", settings={"temperature": 0.0})
    final_answer_verifier = FinalAnswerVerifier(model=model_name)

    end = len(records) if limit is None else min(len(records), from_idx + limit)
    for idx in range(from_idx, end):
        rec = records[idx]
        rec_id = rec.get("id", str(idx))
        steps = _aegis_steps(rec)
        query = (rec.get("input") or {}).get("query", "")
        print(f"\n=== [{idx}] {rec_id}  ({len(steps)} steps) ===")

        eval_input = RawTraceInput(trace=_format_indexed_raw_trace(steps, query))

        findings_results = await _run_all_metrics(model, eval_input)
        evidence_results = EvidenceVerifier().verify_all(findings_results, eval_input)

        gt_answer = (rec.get("ground_truth") or {}).get("correct_answer")
        final_answer_result = await _run_final_answer_verification(
            final_answer_verifier, rec, gt_answer
        )

        # WW-shaped payload: metric findings live at the TOP LEVEL (that is what
        # reporting._iter_metric_results / diagnostic_accuracy expect), alongside
        # evidence_verification, non_llm_validators and a reference answer. Then
        # build the diagnostic report from that same payload so the boary scorer
        # can consume the file identically to the WW pipeline.
        payload: dict = {}
        payload.update(_serialize_findings(findings_results))
        payload["evidence_verification"] = _serialize_evidence(evidence_results)
        payload["non_llm_validators"] = run_on_trace(rec.get("input") or rec)
        payload["reference_answer"] = gt_answer
        if final_answer_result is not None:
            payload["final_answer_verification"] = final_answer_result.model_dump(mode="json")
        payload["report"] = build_evaluation_report(
            payload,
            predicted_answer=_aegis_predicted_answer(rec),
            reference_answer=gt_answer,
        )

        # AEGIS-specific extras — ignored by the report builder (they lack the
        # metric_name/findings shape), used later for scoring against ground truth.
        # `sample_id` (stable global row index) is the matching key; the AEGIS `id`
        # is NOT unique, so it is kept only as informational `aegis_id`.
        payload["sample_id"] = f"{idx:05d}"
        payload["aegis_id"] = rec_id
        payload["metadata"] = rec.get("metadata")
        payload["ground_truth_faulty_agents"] = (rec.get("output") or {}).get("faulty_agents")
        payload["ground_truth_answer"] = gt_answer

        out_path = out / f"aegis_findings_{idx:05d}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        nlv = payload["non_llm_validators"]
        print(
            f"  saved -> {out_path.name}  "
            f"[nlv_format={nlv['detected_format']}, attribution={nlv['agent_attribution_available']}, "
            f"llm_metrics={len(findings_results)}]"
        )


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Run maseval on the AEGIS benchmark")
    p.add_argument("--input", default=str(here.parent.parent / "data" / "aegis" / "test.jsonl"))
    p.add_argument("--out", default=str(here / "aegis_findings"))
    p.add_argument("--model", default="google/gemini-2.5-flash")
    p.add_argument("--from-idx", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="Process at most N records from --from-idx")
    args = p.parse_args()

    asyncio.run(main(args.input, args.out, args.model, args.from_idx, args.limit))
