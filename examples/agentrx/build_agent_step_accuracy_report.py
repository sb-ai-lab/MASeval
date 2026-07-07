"""Build a Markdown + JSON accuracy report for an AgentRx findings run.

Thin wrapper: computes metrics with ``calculate_agent_step_accuracy`` (AgentRx
gold) and renders them with the shared Who&When Markdown builder, so AgentRx
reports have the same layout as the Who&When ones.

Usage:
    python build_agent_step_accuracy_report.py --config magentic --gold-scope all
    python build_agent_step_accuracy_report.py --config tau --gold-scope root_cause
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
WW_DIR = ROOT / "examples" / "who_and_when"
for p in (ROOT / "src", THIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import agentrx_data  # noqa: E402
import calculate_agent_step_accuracy as calc  # noqa: E402


def _ww_build_markdown_report():
    """Load the Who&When Markdown renderer by explicit path (the module shares
    this file's name, so import-by-name would resolve to us)."""
    import importlib.util

    path = WW_DIR / "build_agent_step_accuracy_report.py"
    spec = importlib.util.spec_from_file_location("ww_report_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_markdown_report


build_markdown_report = _ww_build_markdown_report()

REPORTS_DIR = THIS_DIR / "reports"

# Paper Table 6 (arXiv:2602.02475), their trained "AGENTRX" root-cause step
# localization: Step Acc / Acc@±1 / Acc@±3 / Acc@±5. Used only as a reference
# column; our judge is untrained and not directly comparable.
PAPER_TABLE6 = {
    "tau": {"label": "τ-Bench", "acc": (54.0, 59.8, 72.4, 83.9)},
    "magentic": {"label": "Magentic-One", "acc": (31.8, 40.9, 47.7, 50.8),
                 "label2": "Magentic-One*", "acc2": (46.9, 61.7, 72.8, 79.0)},
    "flash": {"label": "Flash (not released)", "acc": (83.3, 98.4, 100.0, 100.0)},
}


def _table6_section(config: str, summary: dict, gold_scope: str) -> str:
    """Append a top-1 tolerance sweep next to paper Table 6 (root-cause step)."""
    ours = tuple(
        (summary.get(k) or 0.0) * 100
        for k in ("step_top1_acc", "step_top1_pm1_acc",
                  "step_top1_pm3_acc", "step_top1_pm5_acc")
    )
    ref = PAPER_TABLE6.get(config)
    lines = [
        "",
        "## Paper comparison — Table 6 (root-cause Step Acc, top-1)",
        "",
        "Top-1 span (first_idx_mode=top_ranked) vs gold at ±0/±1/±3/±5. The "
        "paper reports a single predicted root-cause step, so `gold_scope="
        "root_cause` is the closest analog. Their AGENTRX is a trained method; "
        "ours is an untrained LLM judge — treat this as orientation, not parity.",
        "",
        "| Source | Step Acc | Acc@±1 | Acc@±3 | Acc@±5 |",
        "|---|---|---|---|---|",
        f"| Ours ({config} / {gold_scope}) | "
        + " | ".join(f"{v:.1f}%" for v in ours) + " |",
    ]
    if ref:
        lines.append(
            f"| Paper AGENTRX — {ref['label']} | "
            + " | ".join(f"{v:.1f}%" for v in ref["acc"]) + " |"
        )
        if "acc2" in ref:
            lines.append(
                f"| Paper AGENTRX — {ref['label2']} | "
                + " | ".join(f"{v:.1f}%" for v in ref["acc2"]) + " |"
            )
    return "\n".join(lines) + "\n"


def main(
    config: str = "magentic",
    *,
    pred_glob=None,
    gold_scope: str = "all",
    step_tolerance: int = 1,
    verifier_mode: str | None = None,
    first_idx_mode: str = "top_ranked",
    model_name: str = "unknown",
    output_json_path: str | Path | None = None,
    output_md_path: str | Path | None = None,
    top_error_examples: int = 20,
    print_summary: bool = True,
) -> dict:
    tag = f"{config}_{gold_scope}" + (f"_{verifier_mode}" if verifier_mode else "")
    output_json_path = output_json_path or (REPORTS_DIR / f"agentrx_{tag}.json")
    output_md_path = output_md_path or (REPORTS_DIR / f"agentrx_{tag}.md")

    result = calc.main(
        config=config,
        pred_glob=pred_glob,
        gold_scope=gold_scope,
        step_tolerance=step_tolerance,
        verifier_mode=verifier_mode,
        first_idx_mode=first_idx_mode,
        output_path=output_json_path,
        print_summary=False,
    )

    dataset_name = agentrx_data.CONFIGS[config]["dataset_name"]
    report_args = {
        "experiment_name": f"AgentRx {config} — gold={gold_scope}"
        + (f", verifier={verifier_mode}" if verifier_mode else ""),
        "model_name": model_name,
        "dataset_name": dataset_name,
        "run_label": f"AgentRx findings ({gold_scope} gold)",
        "pred_glob": str(result["run_config"]["pred_glob"]),
        "hf_annotations": f"microsoft/AgentRx :: {config} ({gold_scope})",
        "output_json_path": str(output_json_path),
        "output_md_path": str(output_md_path),
        "id_column": "trajectory_id",
        "agent_columns": result["run_config"]["agent_columns"],
        "step_columns": result["run_config"]["step_columns"],
        "step_tolerance": step_tolerance,
        "build_missing_report": True,
        "top_error_examples": top_error_examples,
        "notes": (
            f"AgentRx / {config}, gold_scope='{gold_scope}'. Spans keyed by the "
            "native 1-based step index (== gold step_number). non_llm_validators "
            f"are not used. Step Top-1 uses first_idx_mode='{first_idx_mode}' "
            "(top_ranked = the model's #1-ranked span), comparable to the paper's "
            "single-root-cause Step Acc; step_top1_pm1/pm3/pm5 add ±1/±3/±5 "
            "tolerance to match Table 6's Acc@±k."
        ),
    }

    markdown = build_markdown_report(result, report_args)
    markdown += _table6_section(config, result["summary"], gold_scope)
    Path(output_md_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md_path).write_text(markdown, encoding="utf-8")
    result["markdown_report_path"] = str(output_md_path)

    if print_summary:
        import json

        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        print(f"Saved Markdown report to: {output_md_path}")
        print(f"Saved metrics JSON to:    {output_json_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build AgentRx accuracy report.")
    parser.add_argument("--config", choices=list(agentrx_data.CONFIGS), default="magentic")
    parser.add_argument("--pred-glob", default=None)
    parser.add_argument("--gold-scope", choices=("all", "root_cause"), default="all")
    parser.add_argument("--step-tolerance", type=int, default=1)
    parser.add_argument("--verifier-mode", choices=("none", "strict", "soft"), default=None)
    parser.add_argument(
        "--first-idx-mode",
        choices=("top_ranked", "min_index"),
        default="top_ranked",
        help="How the top-1 predicted span is chosen (default top_ranked).",
    )
    parser.add_argument("--model", default="unknown", help="Judge model name for the report.")
    args = parser.parse_args()
    main(
        config=args.config,
        pred_glob=args.pred_glob,
        gold_scope=args.gold_scope,
        step_tolerance=args.step_tolerance,
        verifier_mode=args.verifier_mode,
        first_idx_mode=args.first_idx_mode,
        model_name=args.model,
    )
