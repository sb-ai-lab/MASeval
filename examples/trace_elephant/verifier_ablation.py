"""EvidenceVerifier ablation for TraceElephant agent/step localization.

Scores the same TraceElephant findings under three verifier gating settings and
writes one JSON report per setting, plus a printed comparison table:

* ``none``   -- no verifier: every LLM finding counts;
* ``strict`` -- only ``verified`` findings count (weak + invalid -> review);
* ``soft``   -- ``verified``/``weak`` count, ``invalid`` -> review (prior default).

``non_llm_validators`` are NOT counted in any setting (LLM findings only). Reports
are rebuilt from the raw LLM findings under each gating via the shared
``maseval.diagnostic_accuracy`` scorer (``verifier_mode``).

Run ``launch_findings_judges.py`` first to produce the findings this scores.

Debugger-friendly: call ``run(...)`` directly, or run as a script.

    run(system="all")
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
for p in (ROOT / "src", THIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import calculate_agent_step_accuracy as scorer  # noqa: E402

MODES = ("none", "strict", "soft")

# TraceElephant gold is one agent + one step, so both levels are meaningful.
METRIC_KEYS = [
    ("agent_top1_acc", "Agent Top-1"),
    ("agent_hit_acc", "Agent Hit"),
    ("step_top1_acc", "Step Top-1"),
    ("step_top1_pm1_acc", "Step Top-1 ±1"),
    ("step_hit_acc", "Step Hit"),
]


def run(
    system: str = "all",
    *,
    pred_glob=None,
    data_dir: str | None = None,
    step_tolerance: int = 1,
    reports_dir: str | Path = THIS_DIR / "reports",
) -> dict[str, dict]:
    """Score the 3 verifier modes, write per-mode reports, return the summaries."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict] = {}
    for mode in MODES:
        result = scorer.main(
            system=system,
            pred_glob=pred_glob,
            data_dir=data_dir,
            step_tolerance=step_tolerance,
            verifier_mode=mode,
            first_idx_mode="top_ranked",
            output_path=str(reports_dir / f"verifier_ablation_{system}_{mode}.json"),
            print_summary=False,
        )
        summaries[mode] = result["summary"]

    print(_render(system, summaries))
    return summaries


def _pct(v):
    return f"{v * 100:5.1f}%" if isinstance(v, (int, float)) else "    —"


def _render(system: str, summaries: dict[str, dict]) -> str:
    matched = summaries["soft"].get("matched_annotations")
    lines = [
        f"=== Verifier ablation — TraceElephant / {system} (matched {matched}) ===",
        "LLM findings only; non_llm_validators not counted.",
        "",
        f"{'Metric':<16}{'none':>9}{'strict':>9}{'soft':>9}",
        "-" * 43,
    ]
    for key, label in METRIC_KEYS:
        row = "".join(_pct(summaries[m].get(key)).rjust(9) for m in MODES)
        lines.append(f"{label:<16}{row}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TraceElephant EvidenceVerifier ablation.")
    parser.add_argument("--system", choices=("all", "captain", "magentic", "swe"), default="all")
    parser.add_argument("--pred-glob", default=None, help="Override findings dir/glob.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--step-tolerance", type=int, default=1)
    args = parser.parse_args()

    run(
        system=args.system,
        pred_glob=args.pred_glob,
        data_dir=args.data_dir,
        step_tolerance=args.step_tolerance,
    )
