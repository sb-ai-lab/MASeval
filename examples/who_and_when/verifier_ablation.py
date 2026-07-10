"""EvidenceVerifier ablation for Who&When agent/step localization.

Scores the same prediction files under three verifier gating settings and writes
one Markdown + JSON report per setting, plus a printed comparison table:

* ``none``   -- no verifier: every LLM finding counts;
* ``strict`` -- only ``verified`` findings count (weak + invalid -> review);
* ``soft``   -- ``verified``/``weak`` count, ``invalid`` -> review (prior default).

``non_llm_validators`` are NOT counted in any setting (LLM findings only).

Debugger-friendly: call ``run(...)`` directly, or run as a script.

    run(pred_glob="who&when_hc_gemini_findings_v9_report_v2/*.json",
        annotations="hf://datasets/Kevin355/Who_and_When/Hand-Crafted.parquet",
        split="hc")
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
for p in (ROOT / "src", THIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from build_agent_step_accuracy_report import main as build_report  # noqa: E402

MODES = ("none", "strict", "soft", "llm")

METRIC_KEYS = [
    ("agent_top1_acc", "Agent Top-1"),
    ("agent_hit_acc", "Agent Hit"),
    ("step_top1_acc", "Step Top-1"),
    ("step_hit_acc", "Step Hit"),
    ("step_hit_pm1_acc", "Step Hit ±1"),
]


def run(
    pred_glob: str | Path | Sequence[str | Path],
    annotations: str,
    *,
    split: str = "hc",
    dataset_name: str = "Who&When / Hand-Crafted",
    reports_dir: str | Path = THIS_DIR / "reports",
    step_tolerance: int = 1,
) -> dict[str, dict]:
    """Score the 3 verifier modes, write per-mode reports, return the summaries."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict] = {}
    for mode in MODES:
        result = build_report(
            pred_glob=pred_glob,
            output_json_path=str(reports_dir / f"verifier_ablation_{split}_{mode}.json"),
            output_md_path=str(reports_dir / f"verifier_ablation_{split}_{mode}.md"),
            hf_annotations=annotations,
            experiment_name=f"Who&When {split.upper()} — verifier={mode}",
            dataset_name=dataset_name,
            run_label="verifier ablation (LLM findings only, no validators)",
            step_tolerance=step_tolerance,
            verifier_mode=mode,
            notes=(
                f"EvidenceVerifier ablation, mode='{mode}'. non_llm_validators are "
                "NOT counted. Reports rebuilt from raw LLM findings under this gating."
            ),
            print_summary=False,
        )
        summaries[mode] = result["summary"]

    print(_render(split, annotations, summaries))
    return summaries


def _pct(v):
    return f"{v * 100:5.1f}%" if isinstance(v, (int, float)) else "    —"


def _render(split: str, annotations: str, summaries: dict[str, dict]) -> str:
    matched = summaries["soft"].get("matched_annotations")
    lines = [
        f"=== Verifier ablation — {split.upper()} (matched {matched}) ===",
        f"Annotations: {annotations}",
        "LLM findings only; non_llm_validators not counted.",
        "",
        f"{'Metric':<14}{'none':>9}{'strict':>9}{'soft':>9}",
        "-" * 41,
    ]
    for key, label in METRIC_KEYS:
        row = "".join(_pct(summaries[m].get(key)).rjust(9) for m in MODES)
        lines.append(f"{label:<14}{row}")
    return "\n".join(lines)


if __name__ == "__main__":
    """Score the same prediction files under every EvidenceVerifier gate.

    Modes: none / strict / soft / llm. ``non_llm_validators`` are NOT counted.

    Example 1 -- deterministic evidence (the standard launch output):
        python verifier_ablation.py --split hc

    Example 2 -- LLM-judged evidence. First rebuild the verifier output of an
    existing folder with the LLM verifier (no judges re-run), then score it:
        python reverify_with_llm.py \
            --input-folder "who&when_hand_gemini_idx_msg_v2" \
            --split hc --model "google/gemini-2.5-flash"
        python verifier_ablation.py \
            --split hc \
            --pred-glob "who&when_hand_gemini_idx_msg_v2_llm/*.json"

    Comparing Example 1 vs Example 2 shows the effect of swapping the
    deterministic verifier for the LLM verifier (the ``llm`` column in each
    report matches ``soft`` gating, but the underlying ``evidence_status``
    values were produced by the LLM judge).
    """
    import argparse

    def _cached(name: str) -> str:
        import glob

        cache_roots: list[Path] = []
        for env_var in ("HUGGINGFACE_HUB_CACHE", "HF_HOME"):
            value = os.environ.get(env_var)
            if value:
                root = Path(value)
                cache_roots.append(root if root.name == "hub" else root / "hub")

        xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache_home:
            cache_roots.append(Path(xdg_cache_home) / "huggingface" / "hub")

        cache_roots.append(Path.home() / ".cache" / "huggingface" / "hub")

        hits: list[str] = []
        for root in cache_roots:
            hits = glob.glob(str(root / "datasets--Kevin355--Who_and_When" / "snapshots" / "*" / name))
            if hits:
                break
        return hits[0] if hits else f"hf://datasets/Kevin355/Who_and_When/{name}"

    parser = argparse.ArgumentParser(description="Who&When EvidenceVerifier ablation.")
    parser.add_argument("--split", choices=("hc", "algo", "both"), default="both")
    parser.add_argument("--pred-glob", default=None, help="Override prediction glob.")
    parser.add_argument("--annotations", default=None, help="Override annotation path/URL.")
    parser.add_argument("--step-tolerance", type=int, default=1)
    args = parser.parse_args()

    V2 = str(THIS_DIR)
    SPECS = {
        "hc": (f"{V2}/who&when_hand_gemini_idx_msg_v4/*.json",
               _cached("Hand-Crafted.parquet"), "Who&When / Hand-Crafted"),
        "algo": (f"{V2}/who&when_algo_gemini_idx_msg_v2/*.json",
                 _cached("Algorithm-Generated.parquet"), "Who&When / Algorithm-Generated"),
    }
    splits = ("hc", "algo") if args.split == "both" else (args.split,)
    for sp in splits:
        pg, ann, ds = SPECS[sp]
        run(
            pred_glob=args.pred_glob or pg,
            annotations=args.annotations or ann,
            split=sp,
            dataset_name=ds,
            step_tolerance=args.step_tolerance,
        )
        print()
