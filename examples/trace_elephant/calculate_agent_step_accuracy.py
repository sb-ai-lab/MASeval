"""Agent / step localization accuracy for TraceElephant findings runs.

Compares per-trace findings JSON (``report.diagnostic_report``) against the
TraceElephant gold (``mistake_agent`` / ``mistake_step``, one decisive failure
per trace), using the shared ``maseval.diagnostic_accuracy`` scorer.

TraceElephant gold is a single (agent, step), so this is Who&When-parity
scoring: agent Top-1 = predicted primary is the gold agent; step Top-1 = the
top-ranked predicted span is the gold step (``--step-tolerance`` adds ±k). We
also report Hit (gold in the predicted set) for reference.

Prediction files (``findings_{i}.json``) match gold row ``i`` by filename index
(and by ``task_name`` when present). ``--system`` must match the launch run's
``--system`` so the gold subset is filtered and re-indexed identically.

Usage:
    python calculate_agent_step_accuracy.py --system all
    python calculate_agent_step_accuracy.py --system captain --step-tolerance 1
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Sequence

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
for p in (ROOT / "src", THIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from maseval.diagnostic_accuracy import evaluate_agent_step_accuracy  # noqa: E402

import trace_elephant_data as ted  # noqa: E402

SYSTEMS = ("captain", "magentic", "swe")


def _subset(data_dir: str, system: str):
    """The examples for ``system``, re-indexed exactly as launch_findings_judges does."""
    examples = ted.load_examples(data_dir)
    if system != "all":
        examples = [e for e in examples if e.system_category == system]
        for i, e in enumerate(examples):
            e.row_index = i
    return examples


def _resolve_pred_paths(pred_glob) -> list[Path]:
    import glob

    specs = pred_glob if isinstance(pred_glob, (list, tuple)) else [pred_glob]
    files: list[Path] = []
    for spec in specs:
        p = Path(spec)
        if p.is_dir():
            files.extend(sorted(p.glob("*.json")))
        elif any(ch in str(spec) for ch in "*?["):
            files.extend(sorted(Path(x) for x in glob.glob(str(spec))))
        elif p.exists():
            files.append(p)
    return files


def main(
    system: str = "all",
    pred_glob: str | Path | Sequence[str | Path] | None = None,
    *,
    data_dir: str | None = None,
    step_tolerance: int = 1,
    verifier_mode: str | None = None,
    first_idx_mode: str = "top_ranked",
    output_path: str | Path | None = None,
    print_summary: bool = True,
) -> dict:
    data_dir = data_dir or str(THIS_DIR / "data")
    if pred_glob is None:
        pred_glob = THIS_DIR / f"trace_elephant_{system}_findings"
    pred_paths = _resolve_pred_paths(pred_glob)
    if not pred_paths:
        raise FileNotFoundError(f"No prediction JSON files matched: {pred_glob!r}")

    examples = _subset(data_dir, system)

    with tempfile.TemporaryDirectory(prefix="trace_elephant_gold_") as tmp:
        gold_path = ted.write_gold_jsonl(examples, str(Path(tmp) / "gold.jsonl"))
        result = evaluate_agent_step_accuracy(
            pred_paths,
            gold_path,
            agent_columns=["mistake_agents"],
            step_columns=["mistake_steps"],
            step_tolerance=step_tolerance,
            verifier_mode=verifier_mode,
            first_idx_mode=first_idx_mode,
        )

    result["run_config"] = {
        "system": system,
        "pred_glob": str(pred_glob),
        "prediction_files_count": len(pred_paths),
        "step_tolerance": step_tolerance,
        "verifier_mode": verifier_mode,
        "first_idx_mode": first_idx_mode,
    }
    result["annotation_source"] = {
        "type": "trace_elephant",
        "system": system,
        "rows": len(examples),
    }

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if print_summary:
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return result


calculate_agent_step_accuracy = main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TraceElephant agent/step accuracy.")
    parser.add_argument("--system", choices=("all", *SYSTEMS), default="all")
    parser.add_argument("--pred-glob", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--step-tolerance", type=int, default=1)
    parser.add_argument("--verifier-mode", choices=("none", "strict", "soft"), default=None)
    parser.add_argument(
        "--first-idx-mode",
        choices=("top_ranked", "min_index"),
        default="top_ranked",
        help="How the top-1 predicted span is chosen (default top_ranked).",
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()
    main(
        system=args.system,
        pred_glob=args.pred_glob,
        data_dir=args.data_dir,
        step_tolerance=args.step_tolerance,
        verifier_mode=args.verifier_mode,
        first_idx_mode=args.first_idx_mode,
        output_path=args.output_json,
    )
