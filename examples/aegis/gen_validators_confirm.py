"""Add the LLM confirmation layer to AEGIS's already-stored deterministic
``non_llm_validators`` and write augmented result files to ``aegis_findings_confirm/``.

AEGIS's launcher already runs the validators (each finding carries ``culprit_agent``
+ per-idx ``evidence`` agents), so — unlike TraceElephant — we do NOT re-run
``run_on_trace``; we only attach ``llm_confirmation`` (~1 call per finding-bearing
trace). The confirmer needs the trace history, which the result files do not store,
so we reload it from the source JSONL, matched on ``sample_id`` (the stable global
row index — the AEGIS ``id`` is NOT unique). We hand the confirmer the exact object
the validators ran on (``rec["input"]``) so ``build_raw_spans`` re-derives the same
span idxs the findings cite.

AEGIS gold is agent-only (no step index), so the confirmer's only useful job here is
benign-pruning; appointing (``corrected_idx``) is irrelevant for scoring.

Resumable: existing output files are skipped. All 600 files are written through
(only the finding-bearing ones incur an LLM call) so the scorer reads one folder.

Usage:
    python gen_validators_confirm.py
    python gen_validators_confirm.py --limit 1     # smoke test on the first bearing trace
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

load_dotenv(ROOT / ".env")
load_dotenv(THIS_DIR / ".env")

from maseval.validators.llm_confirm import confirm_trace_async  # noqa: E402

# Source AEGIS JSONL (conversation_history). Override per machine with $AEGIS_SRC.
DEFAULT_SRC = os.environ.get("AEGIS_SRC", "your/path/test.jsonl")


def _n(nlv: dict) -> int:
    return sum(len(m.get("findings", [])) for m in (nlv.get("metrics") or {}).values())


async def main(model: str, src: str, limit: int | None) -> None:
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set (source the root .env).")

    records = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    print(f"Loaded {len(records)} AEGIS source records from {src}")

    files = sorted(glob.glob(str(THIS_DIR / "aegis_findings" / "*.json")))
    out = THIS_DIR / "aegis_findings_confirm"
    out.mkdir(parents=True, exist_ok=True)
    totals = {"bearing": 0, "confirmed": 0, "benign": 0, "uncertain": 0, "appointed": 0}
    done = 0
    for fp in files:
        name = Path(fp).name
        out_f = out / name
        if out_f.exists():
            continue
        data = json.loads(Path(fp).read_text(encoding="utf-8"))
        nlv = data.get("non_llm_validators") or {}
        if _n(nlv):
            sid = int(data["sample_id"])
            trace = (records[sid].get("input") or records[sid])
            try:
                await confirm_trace_async(trace, nlv, model)  # full reading view
            except Exception as exc:  # noqa: BLE001
                print(f"  [{name}] confirm error: {type(exc).__name__}: {exc}")
            s = nlv.get("llm_confirmation_summary", {})
            totals["bearing"] += 1
            for k in ("confirmed", "benign", "uncertain", "appointed"):
                totals[k] += int(s.get(k, 0) or 0)
            print(f"  [{name}] sid={sid} {_n(nlv)} det-findings -> {s}")
            done += 1
        data["non_llm_validators"] = nlv
        out_f.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        if limit is not None and done >= limit:
            print(f"(stopped after {done} confirmed traces per --limit)")
            break
    print(f"totals: {totals}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add LLM confirmation to AEGIS validators.")
    parser.add_argument("--model", default="google/gemini-2.5-flash")
    parser.add_argument("--src", default=DEFAULT_SRC, help="AEGIS source JSONL (conversation_history).")
    parser.add_argument("--limit", type=int, default=None, help="Confirm at most N bearing traces (smoke test).")
    args = parser.parse_args()
    asyncio.run(main(model=args.model, src=args.src, limit=args.limit))
