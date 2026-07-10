"""Apply the opt-in LLM confirmation layer to existing Who&When findings.

``maseval.validators.llm_confirm`` is a thin, opt-in pass over the DETERMINISTIC
``non_llm_validators`` findings (it confirms genuine-vs-benign and appoints the
causal agent turn). It does NOT touch the 11 LLM evaluators, so there is no need
to re-run them: this script loads the already-generated per-trace findings, runs
``confirm_trace_async`` on their ``non_llm_validators`` block, and writes the
augmented findings to a sibling ``*_llmconfirm`` folder (non-destructive).

Only traces that carry deterministic findings incur an LLM call (confirm is a
no-op otherwise), so this is cheap (~1 call per finding-bearing trace).

The launcher's ``--llm-confirm`` flag does the same thing inline during a fresh
findings run; this script is the way to apply it to findings you already have.

Usage:
    python apply_llm_confirm.py --run hc      # Hand-Crafted split
    python apply_llm_confirm.py --run algo --model google/gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import ast
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

load_dotenv(THIS_DIR / ".env")

from maseval.validators.llm_confirm import confirm_trace_async  # noqa: E402

SPLITS = {
    "hc": ("who&when_hand_gemini_idx_msg_v2", "Hand-Crafted.parquet"),
    "algo": ("who&when_algo_gemini_idx_msg_v2", "Algorithm-Generated.parquet"),
}


def _load_parquet(filename: str):
    """Robustly fetch a Who&When split (cached hf_hub_download, not flaky streaming)."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    last = None
    for _ in range(5):
        try:
            path = hf_hub_download("Kevin355/Who_and_When", filename=filename, repo_type="dataset")
            return pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001 - transient network
            last = exc
    raise RuntimeError(f"Could not download {filename}: {last}")


def _coerce_history_to_steps(history) -> list:
    """Best-effort conversion of dataset history into ordered trace steps
    (mirrors launch_findings_judges._coerce_history_to_steps)."""
    if isinstance(history, list):
        return history
    if isinstance(history, tuple):
        return list(history)
    if hasattr(history, "tolist"):  # numpy ndarray / pandas array from read_parquet
        # Expand the per-message list; otherwise the whole array collapses into a
        # single step (idx 0, no agent), breaking the validators + appointing.
        return list(history.tolist())
    if isinstance(history, str):
        text = history.strip()
        for parser in (ast.literal_eval, json.loads):
            try:
                parsed = parser(text)
                if isinstance(parsed, list):
                    return parsed
                if isinstance(parsed, dict):
                    for key in ("history", "messages", "trace", "steps"):
                        value = parsed.get(key)
                        if isinstance(value, list):
                            return value
                    return [parsed]
            except Exception:
                pass
        return [history]
    return [history]


def _n_findings(nlv: dict) -> int:
    return sum(len(m.get("findings", [])) for m in (nlv.get("metrics") or {}).values())


async def main(run: str, model: str, from_idx: int = 0) -> None:
    import pandas as pd

    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set (source a .env with the key).")

    folder_name, parquet = SPLITS[run]
    src_dir = THIS_DIR / folder_name
    out_dir = THIS_DIR / f"{folder_name}_llmconfirm"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_parquet(parquet)
    files = sorted(glob.glob(str(src_dir / "gemini_findings_*.json")),
                   key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    print(f"{run}: {len(files)} findings files, {len(df)} dataset rows -> {out_dir.name}")

    totals = {"confirmed": 0, "benign": 0, "uncertain": 0, "appointed": 0, "traces_confirmed": 0}
    for fp in files:
        i = int(fp.rsplit("_", 1)[1].split(".")[0])
        if i < from_idx:
            continue
        findings = json.loads(Path(fp).read_text(encoding="utf-8"))
        nlv = findings.get("non_llm_validators") or {}
        n = _n_findings(nlv)
        if n == 0:
            # confirm is a no-op; copy through unchanged.
            (out_dir / Path(fp).name).write_text(json.dumps(findings, indent=2, default=str), encoding="utf-8")
            continue

        history = df.iloc[i]["history"]
        trace = {"history": _coerce_history_to_steps(history)}
        try:
            await confirm_trace_async(trace, nlv, model)
        except Exception as exc:  # noqa: BLE001 - one trace must not abort the run
            print(f"  [{i}] confirm error: {type(exc).__name__}: {exc}")
            (out_dir / Path(fp).name).write_text(json.dumps(findings, indent=2, default=str), encoding="utf-8")
            continue

        summary = nlv.get("llm_confirmation_summary", {})
        totals["traces_confirmed"] += 1
        for k in ("confirmed", "benign", "uncertain", "appointed"):
            totals[k] += int(summary.get(k, 0) or 0)
        findings["non_llm_validators"] = nlv
        (out_dir / Path(fp).name).write_text(json.dumps(findings, indent=2, default=str), encoding="utf-8")
        print(f"  [{i}] {n} det-findings -> {summary}")

    print("\n=== LLM-confirm summary (" + run + ") ===")
    print(f"finding-bearing traces confirmed: {totals['traces_confirmed']}")
    print(f"per-finding verdicts: confirmed={totals['confirmed']} "
          f"benign={totals['benign']} uncertain={totals['uncertain']} | appointed={totals['appointed']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply LLM confirmation to existing Who&When findings.")
    parser.add_argument("--run", choices=("hc", "algo"), default="hc")
    parser.add_argument("--model", default="google/gemini-2.5-flash")
    parser.add_argument("--from-idx", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(main(run=args.run, model=args.model, from_idx=args.from_idx))
