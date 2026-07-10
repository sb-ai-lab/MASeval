"""EvidenceVerifier ablation for TRAIL error-location localization.

Scores the LLM judge predictions under three verifier gating settings and writes
one JSON report per setting plus a printed table:

* ``none``   -- no verifier: every LLM finding counts;
* ``strict`` -- only ``verified`` findings count;
* ``soft``   -- ``verified``/``weak`` count, ``invalid`` -> review.

``non_llm_validators`` are NOT counted in any setting (LLM findings only), matching
the AEGIS/TraceElephant ablations. TRAIL gold is a span-hash ``location`` (no agent),
so only location Step Top-1 / Step Hit are populated. Judge idxs are numeric span
positions, mapped to span-hashes via ``trail_to_spans`` on the reloaded GAIA trace
(see ``score_none_plus_validators`` for the namespace bridge).

Usage:
    python verifier_ablation.py
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
for p in (ROOT / "src", THIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from maseval.diagnostic_accuracy import read_prediction_file  # noqa: E402
from maseval.validators.base import trail_to_spans  # noqa: E402

from score_none_plus_validators import DEFAULT_GAIA, _gold_locations  # noqa: E402

MODES = ("none", "strict", "soft")


def score(pred_dir: str, gaia_dir: str, mode: str) -> dict:
    pred_dir_p = THIS_DIR / pred_dir if not Path(pred_dir).is_absolute() else Path(pred_dir)
    files = sorted(glob.glob(str(pred_dir_p / "*.json")))
    if not files:
        raise FileNotFoundError(f"No prediction files in {pred_dir_p}")
    gaia = Path(gaia_dir)

    top1, hit = [], []
    for fp in files:
        data = json.loads(Path(fp).read_text(encoding="utf-8"))
        gold = _gold_locations(data)
        if not gold:
            continue
        gpath = gaia / f"{data['trace_id']}.json"
        if not gpath.exists():
            continue
        spans = trail_to_spans(json.loads(gpath.read_text(encoding="utf-8")))
        pos2hash = {str(i): s["idx"] for i, s in enumerate(spans)}
        pred = read_prediction_file(fp, verifier_mode=mode)
        idxs = [pos2hash[i] for i in pred.idxs if pos2hash.get(i)]
        first = pos2hash.get(str(pred.first_idx)) if pred.first_idx else None
        top1.append(bool(first and first in gold))
        hit.append(any(h in gold for h in idxs))

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    return {"matched": len(top1), "step_top1": mean(top1), "step_hit": mean(hit)}


def _pct(v):
    return f"{v * 100:5.1f}%" if isinstance(v, (int, float)) else "    —"


def _render(summaries: dict) -> str:
    matched = next(iter(summaries.values()))["matched"]
    lines = [
        f"=== Verifier ablation — TRAIL (matched {matched}) ===",
        "LLM findings only; non_llm_validators not counted. Location-level (TRAIL gold has no agent).",
        "",
        f"{'Metric':<12}{'none':>9}{'strict':>9}{'soft':>9}",
        "-" * 39,
    ]
    for label, key in (("Step Top-1", "step_top1"), ("Step Hit", "step_hit")):
        lines.append(f"{label:<12}" + "".join(_pct(summaries[m][key]).rjust(9) for m in MODES))
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRAIL EvidenceVerifier ablation (location).")
    parser.add_argument("--pred-dir", default="trail_gemini_findings_v1")
    parser.add_argument("--gaia-dir", default=DEFAULT_GAIA)
    args = parser.parse_args()

    summaries = {m: score(args.pred_dir, args.gaia_dir, m) for m in MODES}
    print(_render(summaries))

    reports_dir = THIS_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    for m in MODES:
        (reports_dir / f"verifier_ablation_trail_{m}.json").write_text(
            json.dumps(summaries[m], indent=2), encoding="utf-8")
    print(f"\nSaved: {reports_dir / 'verifier_ablation_trail_{none,strict,soft}.json'}")
