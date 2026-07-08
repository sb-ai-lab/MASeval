"""Backfill FinalAnswerVerifier (no-GT) into existing AgentRx findings files.

The findings runner originally skipped FinalAnswerVerifier. This adds it after
the fact WITHOUT re-running the 11 LLM metrics: for each existing
``findings_{i}.json`` it runs only the no-ground-truth MAS Task Completion judge,
injects ``final_answer_verification``, and rebuilds the stored ``report`` so its
``answer_status`` reflects the verdict.

Accuracy metrics are unaffected (answer_status does not feed the agent/step
ranking) -- this is purely additive. Re-run
``build_agent_step_accuracy_report.py`` afterwards only if you want the verdict
surfaced in the reports.

Usage:
    python backfill_final_answer.py --config magentic --model google/gemini-2.5-flash
    python backfill_final_answer.py --config tau
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

os.environ["OTEL_TRACES_EXPORTER"] = "none"
from opentelemetry import trace  # noqa: E402

trace.set_tracer_provider(None)

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
for p in (ROOT / "src", THIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from maseval.evaluation_blocks.final_answer_verification import (  # noqa: E402
    FinalAnswerVerifier,
)
from maseval.reporting import build_evaluation_report  # noqa: E402

import agentrx_data  # noqa: E402
from launch_findings_judges import _run_final_answer_verification  # noqa: E402


async def main(
    config: str,
    model_name: str = "google/gemini-2.5-flash",
    folder_name: str | None = None,
    overwrite: bool = False,
) -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (check examples/agentrx/.env).")
    os.environ["OPENROUTER_API_KEY"] = api_key

    verifier = FinalAnswerVerifier(model=model_name)
    examples = {ex.row_index: ex for ex in agentrx_data.load_examples(config)}
    folder = THIS_DIR / (folder_name or f"agentrx_{config}_findings")
    files = sorted(folder.glob("findings_*.json"))
    if not files:
        raise FileNotFoundError(f"No findings files in {folder}")

    print(f"AgentRx/{config}: backfilling {len(files)} files in {folder}")
    done = 0
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("final_answer_verification") is not None and not overwrite:
            print(f"  {path.name}: already has final_answer_verification, skipping")
            continue

        row_index = int(path.stem.split("_")[-1])
        ex = examples.get(row_index)
        if ex is None:
            print(f"  {path.name}: no example for row_index={row_index}, skipping")
            continue

        result = await _run_final_answer_verification(verifier, ex)
        if result is None:
            print(f"  {path.name}: verifier returned None, leaving as-is")
            continue

        payload["final_answer_verification"] = result.model_dump(mode="json")
        payload["report"] = build_evaluation_report(payload, reference_answer=None)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        done += 1
        print(f"  {path.name}: verdict={result.verdict} method={result.method}")

    print(f"Backfilled {done}/{len(files)} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill AgentRx final-answer verification.")
    parser.add_argument("--config", choices=list(agentrx_data.CONFIGS), default="magentic")
    parser.add_argument("--model", default="google/gemini-2.5-flash")
    parser.add_argument("--folder", default=None, help="Findings subfolder name.")
    parser.add_argument("--overwrite", action="store_true", help="Redo files that already have it.")
    args = parser.parse_args()

    asyncio.run(
        main(
            config=args.config,
            model_name=args.model,
            folder_name=args.folder,
            overwrite=args.overwrite,
        )
    )
