"""Agent / step localization accuracy for AgentRx findings runs.

Compares per-trajectory findings JSON (``report.diagnostic_report``) against the
AgentRx gold, using the shared ``maseval.diagnostic_accuracy`` scorer.

Gold scope (``gold_scope``):
  * ``"all"`` (default) -- every annotated failure counts (set semantics):
    agent Top-1 = predicted primary is ANY failed agent; step Hit = any
    predicted span is any failure step.
  * ``"root_cause"`` -- only the single decisive failure counts (strict,
    Who&When-parity): predicted primary must be the root-cause agent/step.

Prediction files (``findings_{i}.json``) match gold row ``i`` by filename index
(and by ``trajectory_id`` when present).

Usage:
    python calculate_agent_step_accuracy.py --config magentic --gold-scope all
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

import agentrx_data  # noqa: E402

GOLD_COLUMNS = {
    "all": (["mistake_agents"], ["mistake_steps"]),
    "root_cause": (["root_cause_agent"], ["root_cause_step"]),
}


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
    config: str = "magentic",
    pred_glob: str | Path | Sequence[str | Path] | None = None,
    *,
    gold_scope: str = "all",
    step_tolerance: int = 1,
    verifier_mode: str | None = None,
    first_idx_mode: str = "top_ranked",
    output_path: str | Path | None = None,
    print_summary: bool = True,
) -> dict:
    if gold_scope not in GOLD_COLUMNS:
        raise ValueError(f"gold_scope must be one of {list(GOLD_COLUMNS)}")

    if pred_glob is None:
        pred_glob = THIS_DIR / f"agentrx_{config}_findings"
    pred_paths = _resolve_pred_paths(pred_glob)
    if not pred_paths:
        raise FileNotFoundError(f"No prediction JSON files matched: {pred_glob!r}")

    agent_columns, step_columns = GOLD_COLUMNS[gold_scope]

    with tempfile.TemporaryDirectory(prefix="agentrx_gold_") as tmp:
        gold_path = agentrx_data.write_gold_jsonl(config, str(Path(tmp) / "gold.jsonl"))
        result = evaluate_agent_step_accuracy(
            pred_paths,
            gold_path,
            agent_columns=agent_columns,
            step_columns=step_columns,
            step_tolerance=step_tolerance,
            verifier_mode=verifier_mode,
            first_idx_mode=first_idx_mode,
        )

    result["run_config"] = {
        "config": config,
        "pred_glob": str(pred_glob),
        "prediction_files_count": len(pred_paths),
        "gold_scope": gold_scope,
        "agent_columns": agent_columns,
        "step_columns": step_columns,
        "step_tolerance": step_tolerance,
        "verifier_mode": verifier_mode,
        "first_idx_mode": first_idx_mode,
    }
    result["annotation_source"] = {
        "type": "agentrx",
        "config": config,
        "gold_scope": gold_scope,
        "rows": len(agentrx_data.load_examples(config)),
    }

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if print_summary:
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return result


# Backward-compatible alias.
calculate_agent_step_accuracy = main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgentRx agent/step accuracy.")
    parser.add_argument("--config", choices=list(agentrx_data.CONFIGS), default="magentic")
    parser.add_argument("--pred-glob", default=None)
    parser.add_argument("--gold-scope", choices=list(GOLD_COLUMNS), default="all")
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
        config=args.config,
        pred_glob=args.pred_glob,
        gold_scope=args.gold_scope,
        step_tolerance=args.step_tolerance,
        verifier_mode=args.verifier_mode,
        first_idx_mode=args.first_idx_mode,
        output_path=args.output_json,
    )
