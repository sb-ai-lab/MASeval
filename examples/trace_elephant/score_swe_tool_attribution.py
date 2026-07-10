"""SWE-Agent culprit attribution scored in TraceElephant's own namespace.

The default agent scorer compares the pipeline's named culprit *agent*
(``SWE-Agent`` / ``assistant``) against TraceElephant's SWE gold ``mistake_agent``
— but for SWE that gold field is a **tool** name (``bash``,
``str_replace_editor``, ``str_replace_based_edit_tool``, ...), not an agent. The
namespaces never intersect, so agent accuracy comes out 0 even though step
localization works. (Captain/Magentic annotate real sub-agents, which the judges
name directly, so they are scored as-is and do not need this.)

TraceElephant labels every SWE step by the tool it invoked, and the gold
``mistake_agent`` equals the label of the gold ``mistake_step`` (verified: 100%).
So the correct, comparable culprit prediction for SWE is **the tool at the step
we localize to**:

* Agent Top-1 = tool at the top-1 predicted step (``first_problem_idx``) equals
  the gold tool;
* Agent Hit  = the gold tool is the label of any predicted problematic step.

Step metrics are namespace-agnostic and identical to the default scorer; they are
recomputed here only so one table carries the full picture under each verifier
gating.

Usage:
    python score_swe_tool_attribution.py                 # none/strict/soft
    python score_swe_tool_attribution.py --verifier-mode none
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

from maseval.diagnostic_accuracy import (  # noqa: E402
    _idx_matches_any,
    _normalize_agent,
    read_prediction_file,
)

import calculate_agent_step_accuracy as C  # noqa: E402

MODES = ("none", "strict", "soft")


def _label_map(example) -> dict[str, str]:
    """1-based step idx -> step label (the invoked tool for SWE)."""
    return {str(k + 1): s["name"] for k, s in enumerate(example.history)}


def score(
    data_dir: str | None = None,
    pred_glob: str | None = None,
    verifier_mode: str | None = None,
    step_tolerance: int = 1,
) -> dict:
    data_dir = data_dir or str(THIS_DIR / "data")
    examples = C._subset(data_dir, "swe")
    by_row = {e.row_index: e for e in examples}
    pred_glob = pred_glob or str(THIS_DIR / "trace_elephant_swe_findings")
    files = sorted(
        glob.glob(str(Path(pred_glob) / "*.json")) if Path(pred_glob).is_dir()
        else glob.glob(pred_glob),
        key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]),
    )
    if not files:
        raise FileNotFoundError(f"No SWE prediction files: {pred_glob!r}")

    acc = {k: [] for k in ("agent_top1", "agent_hit", "step_top1", "step_hit", "step_pm1")}
    for fp in files:
        i = int(fp.rsplit("_", 1)[1].split(".")[0])
        ex = by_row.get(i)
        if ex is None:
            continue
        name_at = _label_map(ex)
        gold_tool = _normalize_agent(ex.mistake_agent)
        gold_steps = [str(ex.mistake_step)] if ex.mistake_step else []

        pred = read_prediction_file(fp, verifier_mode=verifier_mode)

        if gold_tool:
            top1_tool = _normalize_agent(name_at.get(str(pred.first_idx))) if pred.first_idx else ""
            pred_tools = {_normalize_agent(name_at.get(str(ix))) for ix in pred.idxs if name_at.get(str(ix))}
            acc["agent_top1"].append(top1_tool == gold_tool)
            acc["agent_hit"].append(gold_tool in pred_tools)
        if gold_steps:
            acc["step_top1"].append(
                bool(pred.first_idx and _idx_matches_any(pred.first_idx, gold_steps, tolerance=0)))
            acc["step_hit"].append(
                any(_idx_matches_any(ix, gold_steps, tolerance=0) for ix in pred.idxs))
            acc["step_pm1"].append(
                any(_idx_matches_any(ix, gold_steps, tolerance=step_tolerance) for ix in pred.idxs))

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    return {"matched": len(files), "summary": {k: mean(v) for k, v in acc.items()}}


def _pct(v):
    return f"{v * 100:5.1f}%" if isinstance(v, (int, float)) else "    —"


ROWS = [
    ("Agent Top-1 (tool)", "agent_top1"),
    ("Agent Hit (tool)", "agent_hit"),
    ("Step Top-1", "step_top1"),
    ("Step Hit", "step_hit"),
    ("Step Hit ±1", "step_pm1"),
]


def _render(results: dict[str, dict]) -> str:
    matched = next(iter(results.values()))["matched"]
    lines = [
        f"=== SWE tool-attribution (step-derived) — TraceElephant (matched {matched}) ===",
        "Culprit = tool label at the localized step (SWE gold mistake_agent is a tool).",
        "",
        f"{'Metric':<20}" + "".join(m.rjust(9) for m in MODES),
        "-" * (20 + 9 * len(MODES)),
    ]
    for label, key in ROWS:
        lines.append(f"{label:<20}" + "".join(_pct(results[m]["summary"][key]).rjust(9) for m in MODES))
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SWE tool-namespace culprit attribution.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--pred-glob", default=None)
    parser.add_argument("--verifier-mode", choices=MODES, default=None,
                        help="Score a single mode; omit to run all three.")
    parser.add_argument("--step-tolerance", type=int, default=1)
    args = parser.parse_args()

    modes = (args.verifier_mode,) if args.verifier_mode else MODES
    results = {
        m: score(args.data_dir, args.pred_glob, verifier_mode=m, step_tolerance=args.step_tolerance)
        for m in modes
    }
    if len(modes) == len(MODES):
        print(_render(results))
    else:
        print(json.dumps(results[modes[0]]["summary"], indent=2))

    reports_dir = THIS_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "swe_tool_attribution.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {reports_dir / 'swe_tool_attribution.json'}")
