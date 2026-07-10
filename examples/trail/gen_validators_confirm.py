"""Add the LLM confirmation layer to TRAIL's already-stored deterministic
``non_llm_validators`` and write augmented result files to
``trail_gemini_findings_v1_confirm/``.

Like AEGIS, TRAIL's launcher already ran the validators, so we do NOT re-run
``run_on_trace``; we only attach ``llm_confirmation`` (~1 call per finding-bearing
trace). The confirmer needs the raw trace, which the result files do not store, so
we reload it from the GAIA trace directory matched on ``trace_id``. We hand the
confirmer the raw trace dict so ``build_raw_spans`` (→ ``trail_to_spans``) re-derives
the native hex ``span_id`` idxs the findings cite.

TRAIL localization gold is a **span-hash** (``labels.errors[].location``); the
validators already speak that namespace (evidence idx = span_id), while the LLM
judges cite numeric span *positions* — the scorer bridges that. The confirmer's
appointing (``corrected_idx``) is therefore meaningful here (unlike agent-only AEGIS).

Resumable: existing output files are skipped. All 117 files are written through
(only finding-bearing ones incur an LLM call) so the scorer reads one folder.

Usage:
    python gen_validators_confirm.py
    python gen_validators_confirm.py --limit 1     # smoke test
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

# GAIA raw-trace dir. Override per machine with $TRAIL_GAIA_DIR.
DEFAULT_GAIA = os.environ.get(
    "TRAIL_GAIA_DIR",
    "/mnt/c/Users/barak/Downloads/trail-benchmark-main/benchmarking/data/GAIA",
)


def _n(nlv: dict) -> int:
    return sum(len(m.get("findings", [])) for m in (nlv.get("metrics") or {}).values())


async def main(model: str, gaia_dir: str, limit: int | None) -> None:
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set (source the root .env).")

    files = sorted(glob.glob(str(THIS_DIR / "trail_gemini_findings_v1" / "*.json")))
    out = THIS_DIR / "trail_gemini_findings_v1_confirm"
    out.mkdir(parents=True, exist_ok=True)
    totals = {"bearing": 0, "confirmed": 0, "benign": 0, "uncertain": 0, "appointed": 0, "no_gaia": 0}
    done = 0
    for fp in files:
        name = Path(fp).name
        out_f = out / name
        if out_f.exists():
            continue
        data = json.loads(Path(fp).read_text(encoding="utf-8"))
        nlv = data.get("non_llm_validators") or {}
        if _n(nlv):
            tid = data["trace_id"]
            gpath = Path(gaia_dir) / f"{tid}.json"
            if not gpath.exists():
                print(f"  [{name}] MISSING GAIA trace {tid}")
                totals["no_gaia"] += 1
            else:
                raw = json.loads(gpath.read_text(encoding="utf-8"))
                try:
                    await confirm_trace_async(raw, nlv, model)  # full reading view
                except Exception as exc:  # noqa: BLE001
                    print(f"  [{name}] confirm error: {type(exc).__name__}: {exc}")
                s = nlv.get("llm_confirmation_summary", {})
                totals["bearing"] += 1
                for k in ("confirmed", "benign", "uncertain", "appointed"):
                    totals[k] += int(s.get(k, 0) or 0)
                print(f"  [{name}] tid={tid[:8]} {_n(nlv)} det-findings -> {s}")
                done += 1
        data["non_llm_validators"] = nlv
        out_f.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        if limit is not None and done >= limit:
            print(f"(stopped after {done} confirmed traces per --limit)")
            break
    print(f"totals: {totals}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add LLM confirmation to TRAIL validators.")
    parser.add_argument("--model", default="google/gemini-2.5-flash")
    parser.add_argument("--gaia-dir", default=DEFAULT_GAIA, help="Dir of GAIA raw trace JSONs.")
    parser.add_argument("--limit", type=int, default=None, help="Confirm at most N bearing traces.")
    args = parser.parse_args()
    asyncio.run(main(model=args.model, gaia_dir=args.gaia_dir, limit=args.limit))
