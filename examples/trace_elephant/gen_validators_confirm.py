"""Run the deterministic validators + LLM confirmer on TraceElephant traces and
attach them to the existing judge findings (the launcher deliberately skipped the
validators; see note below), writing to ``trace_elephant_{system}_valconfirm/``.

Why the launcher skipped them: TraceElephant's judge/gold idx space is 1-based
(``format_trace`` uses ``enumerate(..., start=1)``) while ``who_and_when_to_spans``
(the validators' span builder) is 0-based. That 1-vs-0 offset is the whole
"misaligned spans" reason — it is a narrow bookkeeping issue, not a deeper
incompatibility. We run the validators here and let the *scorer* shift validator
idxs into the 1-based space; the raw findings are stored in their native 0-based
form, exactly as ``run_on_trace`` emits them.

Reuses the stored 11-judge findings untouched; only adds ``non_llm_validators``
(regex, free) + ``llm_confirmation`` (~1 call per finding-bearing trace, full
reading view). Resumable: existing output files are skipped.

Usage:
    python gen_validators_confirm.py                 # all systems
    python gen_validators_confirm.py --system swe
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
for p in (ROOT / "src", THIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

load_dotenv(ROOT / ".env")
load_dotenv(THIS_DIR / ".env")

from maseval.validators import run_on_trace  # noqa: E402
from maseval.validators.llm_confirm import confirm_trace_async  # noqa: E402

import trace_elephant_data as ted  # noqa: E402

SYSTEMS = ("captain", "magentic", "swe")


def _n(nlv: dict) -> int:
    return sum(len(m.get("findings", [])) for m in (nlv.get("metrics") or {}).values())


async def main(system: str, model: str, data_dir: str | None = None,
               confirm: bool = True) -> None:
    if confirm and not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set (source the root .env).")

    data_dir = data_dir or str(THIS_DIR / "data")
    examples = ted.load_examples(data_dir)
    bysys: dict[str, list] = defaultdict(list)
    for e in examples:
        bysys[e.system_category].append(e)
    systems = SYSTEMS if system == "all" else (system,)

    for sysname in systems:
        exs = bysys.get(sysname, [])
        for i, e in enumerate(exs):  # per-system reindex, matches the judge folder
            e.row_index = i
        src = THIS_DIR / f"trace_elephant_{sysname}_findings"
        out = THIS_DIR / f"trace_elephant_{sysname}_valconfirm"
        out.mkdir(parents=True, exist_ok=True)
        totals = {"confirmed": 0, "benign": 0, "uncertain": 0, "appointed": 0, "bearing": 0}
        print(f"\n=== {sysname}: {len(exs)} traces -> {out.name} ===")
        for e in exs:
            out_f = out / f"findings_{e.row_index}.json"
            if out_f.exists():
                continue
            src_f = src / f"findings_{e.row_index}.json"
            data = json.loads(src_f.read_text(encoding="utf-8")) if src_f.exists() else {}
            trace = {"history": e.history}
            nlv = run_on_trace(trace)
            if _n(nlv):
                totals["bearing"] += 1
                if confirm:
                    try:
                        await confirm_trace_async(trace, nlv, model)  # full reading view
                    except Exception as exc:  # noqa: BLE001
                        print(f"  [{e.row_index}] confirm error: {type(exc).__name__}: {exc}")
                    s = nlv.get("llm_confirmation_summary", {})
                    for k in ("confirmed", "benign", "uncertain", "appointed"):
                        totals[k] += int(s.get(k, 0) or 0)
                    print(f"  [{e.row_index}] {_n(nlv)} det-findings -> {s}")
            data["non_llm_validators"] = nlv
            out_f.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        print(f"  {sysname} totals: {totals}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validators + confirmer on TraceElephant.")
    parser.add_argument("--system", choices=("all", *SYSTEMS), default="all")
    parser.add_argument("--model", default="google/gemini-2.5-flash")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--no-confirm", action="store_true",
                        help="Run the free deterministic validators only (no LLM calls).")
    args = parser.parse_args()
    asyncio.run(main(system=args.system, model=args.model, data_dir=args.data_dir,
                     confirm=not args.no_confirm))
