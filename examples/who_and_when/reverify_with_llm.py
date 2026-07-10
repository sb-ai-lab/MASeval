"""Re-run only the EvidenceVerifier in LLM mode over stored findings.

This avoids re-running the LLM judges / evaluators. It reads every per-task
JSON produced by ``launch_findings_judges.py`` (the files must contain the
per-metric ``findings`` blocks -- the current launch script stores them; older
runs that only saved ``report``/``evidence_verification`` cannot be re-verified
because the raw findings are missing), rebuilds the indexed raw trace from the
original dataset, runs ``EvidenceVerifier(mode="llm")`` over the stored findings,
replaces ``evidence_verification`` and rebuilds ``report`` under
``verifier_mode="llm"``.

Plain-Python usage (edit the CONFIG block at the bottom, then run):
    python reverify_with_llm.py
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env")

# Disable the default OTEL tracer provider before importing maseval, matching
# launch_findings_judges.py.
os.environ["OTEL_TRACES_EXPORTER"] = "none"
from opentelemetry import trace

trace.set_tracer_provider(None)

from pydantic_ai.models.openai import OpenAIChatModel

from maseval.metrics import EvidenceVerifier
from maseval.reporting import build_evaluation_report
from maseval.models import RawTraceInput

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from launch_findings_judges import (  # noqa: E402
    _format_indexed_raw_trace,
    _serialize_evidence_verification,
)

NON_FINDING_KEYS = {
    "evidence_verification",
    "report",
    "status",
    "diagnostic_report",
    "gt",
    "label_answer",
    "predicted_answer",
    "reference_answer",
    "answer_status",
    "final_answer_verification",
    "non_llm_validators",
}

HF_DATASETS = {
    "hc": "hf://datasets/Kevin355/Who_and_When/Hand-Crafted.parquet",
    "algo": "hf://datasets/Kevin355/Who_and_When/Algorithm-Generated.parquet",
}


def _looks_like_metric_result(value: object) -> bool:
    return isinstance(value, dict) and "metric_name" in value and "findings" in value


def _load_annotations(split: str) -> pd.DataFrame:
    return pd.read_parquet(HF_DATASETS[split]).reset_index(drop=True)


def _task_index_from_filename(path: Path) -> int | None:
    match = re.search(r"(\d+)(?=\.json$)", path.name)
    return int(match.group(1)) if match else None


def _extract_findings(payload: dict) -> dict[str, dict]:
    return {
        key: value
        for key, value in payload.items()
        if key not in NON_FINDING_KEYS and _looks_like_metric_result(value)
    }


async def reverify_file(
    path: Path,
    df: pd.DataFrame,
    verifier: EvidenceVerifier,
    output_path: Path,
    verifier_mode: str,
) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    findings = _extract_findings(payload)

    if not findings:
        return f"SKIP   {path.name}: no per-metric findings stored (cannot re-verify)"

    task = _task_index_from_filename(path)
    if task is None or task >= len(df):
        return f"SKIP   {path.name}: cannot map to dataset row {task}"

    row = df.iloc[task]
    history = row.get("history")
    question = row.get("question", "")
    if history is None:
        return f"SKIP   {path.name}: dataset row has no history"
    eval_input = RawTraceInput(
        trace=_format_indexed_raw_trace(history, question)
    )

    evidence_results = await verifier.verify_all_async(findings, eval_input)
    payload["evidence_verification"] = _serialize_evidence_verification(evidence_results)
    payload["report"] = build_evaluation_report(
        payload,
        reference_answer=payload.get("reference_answer"),
        verifier_mode=verifier_mode,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    n = len(findings)
    return f"OK     {path.name}: re-verified {n} metric(s) -> {output_path.name}"


async def main_async(config: dict) -> int:
    input_folder = Path(config["input_folder"])
    if not input_folder.is_dir():
        print(f"Input folder not found: {input_folder}")
        return 2

    output_folder = (
        Path(config["output_folder"])
        if config.get("output_folder")
        else (input_folder if config.get("in_place") else Path(f"{input_folder}_llm"))
    )

    df = _load_annotations(config["split"])
    model = OpenAIChatModel(
        config["model"],
        provider="openrouter",
        settings={"temperature": 0.0},
    )
    verifier = EvidenceVerifier(model=model, mode="llm")
    verifier_mode = "llm" if config.get("verifier_mode", "llm") == "llm" else "soft"

    paths = sorted(Path(p) for p in glob.glob(str(input_folder / "*.json")))
    if not paths:
        print(f"No JSON files in {input_folder}")
        return 1

    semaphore = asyncio.Semaphore(config.get("concurrency", 8))

    async def _run(path: Path) -> str:
        out = path if config.get("in_place") else (output_folder / path.name)
        async with semaphore:
            return await reverify_file(path, df, verifier, out, verifier_mode)

    results = await asyncio.gather(*(_run(p) for p in paths))
    for line in results:
        print(line)

    skipped = sum(1 for r in results if r.startswith("SKIP"))
    print(
        f"\nDone. {len(results) - skipped} re-verified, {skipped} skipped, "
        f"output -> {output_folder}"
    )
    return 0


if __name__ == "__main__":
    # ---- Edit this block to configure a run -------------------------------
    CONFIG = {
        # Folder of per-task JSON files from launch_findings_judges.py.
        "input_folder": "/home/alina/Desktop/MASeval/examples/who_and_when/who&when_hand_gemini_idx_msg_v2",
        # Which Who&When dataset the files came from (to rebuild the trace): "hc" or "algo".
        "split": "hc",
        # OpenRouter judge model used by the LLM verifier.
        "model": "google/gemini-2.5-flash",
        # Gate used when rebuilding `report`: "llm" or "soft".
        "verifier_mode": "llm",
        # Where to write re-verified files. If None, writes to "<input_folder>_llm".
        "output_folder": None,
        # If True, overwrite the original files instead of writing to a new folder.
        "in_place": False,
        # Max concurrent files.
        "concurrency": 4,
    }
    # ----------------------------------------------------------------------
    raise SystemExit(asyncio.run(main_async(CONFIG)))
