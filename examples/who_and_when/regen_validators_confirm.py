"""Regenerate the deterministic ``non_llm_validators`` (+ LLM confirmation) on
properly-expanded Who&When traces, reusing the already-good LLM findings.

Why this exists: HF parquet returns the ``history`` column as a numpy ndarray of
message dicts. The launcher's ``_coerce_history_to_steps`` only expands it after
the ndarray fix; runs made before that collapsed every message into a single
step (idx 0, agent None), so the stored ``non_llm_validators`` are degenerate and
appointing is dead. The 11 LLM evaluators in those same files were produced on a
24-step trace and are fine, so we do NOT re-run them (expensive). We only:

  1. re-expand the trace correctly,
  2. re-run ``run_on_trace`` (regex, zero API) to get real per-step findings,
  3. re-confirm + appoint with the LLM (~1 call per finding-bearing trace),

and write the augmented files to a sibling ``*_fixed`` folder (non-destructive).

Usage:
    python regen_validators_confirm.py \
        --src who&when_hand_gemini_llmconfirm_scratch \
        --parquet Hand-Crafted.parquet
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
for p in (ROOT / "src", THIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Load the root .env (canonical creds) then a local one if present.
load_dotenv(ROOT / ".env")
load_dotenv(THIS_DIR / ".env")

from maseval.validators import run_on_trace  # noqa: E402
from maseval.validators.llm_confirm import confirm_trace_async  # noqa: E402

from apply_llm_confirm import _coerce_history_to_steps, _load_parquet  # noqa: E402


def _n_findings(nlv: dict) -> int:
    return sum(len(m.get("findings", [])) for m in (nlv.get("metrics") or {}).values())


async def main(src: str, parquet: str, model: str, from_idx: int = 0) -> None:
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set (source the root .env).")

    src_dir = THIS_DIR / src
    out_dir = THIS_DIR / f"{src}_fixed"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_parquet(parquet)
    files = sorted(
        glob.glob(str(src_dir / "gemini_findings_*.json")),
        key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]),
    )
    print(f"{src}: {len(files)} findings files, {len(df)} rows -> {out_dir.name}")

    totals = {"confirmed": 0, "benign": 0, "uncertain": 0, "appointed": 0, "traces_with_findings": 0}
    for fp in files:
        i = int(fp.rsplit("_", 1)[1].split(".")[0])
        if i < from_idx:
            continue
        data = json.loads(Path(fp).read_text(encoding="utf-8"))

        # Re-expand the trace correctly and regenerate the deterministic layer.
        trace = {"history": _coerce_history_to_steps(df.iloc[i]["history"])}
        nlv = run_on_trace(trace)
        n = _n_findings(nlv)
        if n:
            totals["traces_with_findings"] += 1
            try:
                await confirm_trace_async(trace, nlv, model)
            except Exception as exc:  # noqa: BLE001 - one trace must not abort the run
                print(f"  [{i}] confirm error: {type(exc).__name__}: {exc}")
            summary = nlv.get("llm_confirmation_summary", {})
            for k in ("confirmed", "benign", "uncertain", "appointed"):
                totals[k] += int(summary.get(k, 0) or 0)
            print(f"  [{i}] {n} det-findings -> {summary}")

        data["non_llm_validators"] = nlv
        (out_dir / Path(fp).name).write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )

    print("\n=== regenerated validators + confirm ===")
    print(f"finding-bearing traces: {totals['traces_with_findings']}")
    print(f"per-finding verdicts: confirmed={totals['confirmed']} benign={totals['benign']} "
          f"uncertain={totals['uncertain']} | appointed={totals['appointed']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate validators+confirm on expanded traces.")
    parser.add_argument("--src", default="who&when_hand_gemini_llmconfirm_scratch")
    parser.add_argument("--parquet", default="Hand-Crafted.parquet")
    parser.add_argument("--model", default="google/gemini-2.5-flash")
    parser.add_argument("--from-idx", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(main(src=args.src, parquet=args.parquet, model=args.model, from_idx=args.from_idx))
