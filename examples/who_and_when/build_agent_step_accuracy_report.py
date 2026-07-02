"""Build a Markdown report for Agent Acc and Step-level Acc.

Debugger-friendly script: no argparse, no CLI parsing. Open this file in a
Python debugger and call ``main(...)`` directly.

Example:

    result = main(
        pred_glob="/home/alina/Desktop/maseval-research/examples/who_and_when/who&when_hand_gemini_findings_v9_report/*.json",
        output_json_path="agent_step_metrics.json",
        output_md_path="agent_step_metrics.md",
        experiment_name="Who&When / Gemini / v9 report",
    )

The script computes metrics with ``calculate_agent_step_accuracy.main`` and then
writes a human-readable Markdown report with experiment settings, metric tables,
and a compact error-analysis table.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

# Allow running this file from a checkout without installing the package.
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
THIS_DIR = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    from calculate_agent_step_accuracy import (  # noqa: E402
        DEFAULT_HF_ANNOTATIONS,
        DEFAULT_PRED_GLOB,
        main as calculate_agent_step_accuracy,
    )
except Exception:  # pragma: no cover - lets debugger load the file directly.
    module_path = THIS_DIR / "calculate_agent_step_accuracy.py"
    spec = importlib.util.spec_from_file_location("calculate_agent_step_accuracy", module_path)
    if spec is None or spec.loader is None:
        raise
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    DEFAULT_HF_ANNOTATIONS = module.DEFAULT_HF_ANNOTATIONS
    DEFAULT_PRED_GLOB = module.DEFAULT_PRED_GLOB
    calculate_agent_step_accuracy = module.main


DEFAULT_EXPERIMENT_NAME = "Who&When Agent/Step Localization"
DEFAULT_OUTPUT_JSON_PATH = "agent_step_metrics.json"
DEFAULT_OUTPUT_MD_PATH = "agent_step_metrics.md"


def main(
    pred_glob: str | Path | Sequence[str | Path] = DEFAULT_PRED_GLOB,
    output_json_path: str | Path | None = DEFAULT_OUTPUT_JSON_PATH,
    output_md_path: str | Path = DEFAULT_OUTPUT_MD_PATH,
    hf_annotations: str = DEFAULT_HF_ANNOTATIONS,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    model_name: str | None = "Gemini",
    dataset_name: str = "Who&When / Hand-Crafted",
    run_label: str | None = "v9 report",
    id_column: str | None = None,
    agent_columns: Sequence[str] | None = None,
    step_columns: Sequence[str] | None = None,
    step_tolerance: int = 1,
    build_missing_report: bool = True,
    top_error_examples: int = 20,
    notes: str | None = None,
    print_summary: bool = True,
) -> dict:
    """Calculate metrics and write a Markdown experiment report.

    Args:
        pred_glob: Prediction JSON glob/directory/file/list.
        output_json_path: Optional full metrics JSON output path.
        output_md_path: Markdown report output path.
        hf_annotations: HuggingFace parquet URL with human annotations.
        experiment_name: Human-readable experiment title.
        model_name: Model/judge name shown in the report.
        dataset_name: Dataset/split shown in the report.
        run_label: Optional run label, e.g. ``"v9 report"``.
        id_column: Optional annotation id column.
        agent_columns: Optional annotation columns with gold culprit agent(s).
        step_columns: Optional annotation columns with gold problematic step/span id(s).
        step_tolerance: Numeric tolerance for step-level relaxed matching.
        build_missing_report: Build ``report`` for JSON files that do not have it.
        top_error_examples: Number of mismatch examples to include in the MD table.
        notes: Optional free-form notes shown near the top of the report.
        print_summary: Print summary and output path.

    Returns:
        The same dict returned by ``calculate_agent_step_accuracy.main`` with one
        extra key: ``markdown_report_path``.
    """

    result = calculate_agent_step_accuracy(
        pred_glob=pred_glob,
        output_path=output_json_path,
        hf_annotations=hf_annotations,
        id_column=id_column,
        agent_columns=agent_columns,
        step_columns=step_columns,
        step_tolerance=step_tolerance,
        build_missing_report=build_missing_report,
        print_summary=False,
    )

    report_args = {
        "experiment_name": experiment_name,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "run_label": run_label,
        "pred_glob": str(pred_glob),
        "hf_annotations": hf_annotations,
        "output_json_path": str(output_json_path) if output_json_path is not None else None,
        "output_md_path": str(output_md_path),
        "id_column": id_column,
        "agent_columns": list(agent_columns) if agent_columns is not None else None,
        "step_columns": list(step_columns) if step_columns is not None else None,
        "step_tolerance": step_tolerance,
        "build_missing_report": build_missing_report,
        "top_error_examples": top_error_examples,
        "notes": notes,
    }

    markdown = build_markdown_report(result, report_args)
    output_md = Path(output_md_path)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown, encoding="utf-8")

    result["markdown_report_path"] = str(output_md)

    if print_summary:
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        print(f"Saved Markdown report to: {output_md}")
        if output_json_path is not None:
            print(f"Saved full metrics JSON to: {output_json_path}")

    return result


def build_markdown_report(result: dict[str, Any], args: dict[str, Any]) -> str:
    """Build a readable Markdown report from metric results and run args."""

    summary = result.get("summary") or {}
    annotation_source = result.get("annotation_source") or {}
    run_config = result.get("run_config") or {}
    per_example = result.get("per_example") or []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    title = args.get("experiment_name") or DEFAULT_EXPERIMENT_NAME

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"Generated: `{now}`")
    lines.append("")

    notes = args.get("notes")
    if notes:
        lines.append("## Notes")
        lines.append("")
        lines.append(str(notes).strip())
        lines.append("")

    lines.append("## Experiment setup")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    setup_rows = [
        ("Experiment", args.get("experiment_name")),
        ("Model / judge", args.get("model_name")),
        ("Dataset", args.get("dataset_name")),
        ("Run label", args.get("run_label")),
        ("Prediction glob", args.get("pred_glob")),
        ("Prediction files", summary.get("total_prediction_files") or run_config.get("prediction_files_count")),
        ("HF annotations", args.get("hf_annotations") or annotation_source.get("path")),
        ("Annotation rows", annotation_source.get("rows")),
        ("Matched annotations", summary.get("matched_annotations")),
        ("Unmatched predictions", summary.get("unmatched_predictions")),
        ("ID column", args.get("id_column") or "auto"),
        ("Agent annotation columns", _fmt_list(args.get("agent_columns")) or "auto"),
        ("Step annotation columns", _fmt_list(args.get("step_columns")) or "auto"),
        ("Step tolerance", f"±{args.get('step_tolerance', 1)}"),
        ("Build missing report", args.get("build_missing_report")),
        ("EvidenceVerifier policy", "verified/weak are counted; invalid findings are excluded and left for review"),
    ]
    for key, value in setup_rows:
        lines.append(f"| {key} | {_md(value)} |")
    lines.append("")

    lines.append("## Main results")
    lines.append("")
    lines.append("| Group | Metric | Value | Count | Meaning |")
    lines.append("|---|---:|---:|---:|---|")
    metric_rows = [
        ("Agent", "Top-1 Acc", summary.get("agent_top1_acc"), summary.get("agent_examples"), "Primary culprit agent matches the annotation."),
        ("Agent", "Hit Acc", summary.get("agent_hit_acc"), summary.get("agent_examples"), "At least one predicted culprit agent matches the annotation."),
        ("Agent", "Exact Set Acc", summary.get("agent_exact_set_acc"), summary.get("agent_examples"), "Predicted agent set exactly equals the gold agent set."),
        ("Step", "Top-1 Acc", summary.get("step_top1_acc"), summary.get("step_examples"), "First predicted problematic span equals the gold span."),
        ("Step", "Hit Acc", summary.get("step_hit_acc"), summary.get("step_examples"), "At least one predicted span exactly matches a gold span."),
        ("Step", "Hit Acc ±1", summary.get("step_hit_pm1_acc"), summary.get("step_examples"), "At least one predicted numeric span is within ±1 of a gold span."),
    ]
    for group, metric, value, count, meaning in metric_rows:
        lines.append(f"| {group} | {metric} | {_pct(value)} | {_md(count)} | {meaning} |")
    lines.append("")

    lines.append("## Diagnostic quality summary")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Invalid findings excluded from main predictions | {_md(summary.get('invalid_findings_total'))} |")
    lines.append(f"| Agent examples | {_md(summary.get('agent_examples'))} |")
    lines.append(f"| Step examples | {_md(summary.get('step_examples'))} |")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.extend(_build_interpretation(summary))
    lines.append("")

    error_rows = _select_error_examples(per_example, limit=int(args.get("top_error_examples") or 0))
    lines.append(f"## Example mismatches / review targets (top {len(error_rows)})")
    lines.append("")
    if error_rows:
        lines.append("| # | File | Gold agents | Predicted agents | Agent hit | Gold spans | Predicted spans | Step hit ±1 | Invalid findings |")
        lines.append("|---:|---|---|---|---:|---|---|---:|---:|")
        for i, row in enumerate(error_rows, start=1):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(i),
                        _md(Path(str(row.get("file", ""))).name),
                        _md(_fmt_list(row.get("gold_agents"))),
                        _md(_fmt_list(row.get("predicted_agents"))),
                        _check(row.get("agent_hit_correct")),
                        _md(_fmt_list(row.get("gold_spans"))),
                        _md(_fmt_list(row.get("predicted_spans"))),
                        _check(row.get("step_hit_pm1_correct")),
                        _md(row.get("invalid_findings_count")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No mismatches selected.")
    lines.append("")

    lines.append("## Reproducibility")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(args, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def _build_interpretation(summary: dict[str, Any]) -> list[str]:
    rows: list[str] = []
    agent_hit = summary.get("agent_hit_acc")
    agent_top1 = summary.get("agent_top1_acc")
    step_hit = summary.get("step_hit_acc")
    step_pm1 = summary.get("step_hit_pm1_acc")
    step_top1 = summary.get("step_top1_acc")

    if isinstance(agent_hit, float) and isinstance(agent_top1, float):
        gap = agent_hit - agent_top1
        rows.append(
            f"- Agent localization: Hit Acc is {_pct(agent_hit)}, Top-1 Acc is {_pct(agent_top1)} "
            f"(gap {_pct(gap)}). A large gap means the correct agent is often present but not ranked first."
        )
    if isinstance(step_hit, float) and isinstance(step_pm1, float):
        gap = step_pm1 - step_hit
        rows.append(
            f"- Step localization: exact Hit Acc is {_pct(step_hit)}, relaxed Hit Acc ±1 is {_pct(step_pm1)} "
            f"(gap {_pct(gap)}). A large gap suggests off-by-one or indexing mismatch."
        )
    if isinstance(step_top1, float) and isinstance(step_hit, float):
        rows.append(
            f"- First-span ranking: Step Top-1 Acc is {_pct(step_top1)}. If this is much lower than Step Hit Acc, "
            "use the full problematic_spans list rather than only first_problem_span."
        )
    if not rows:
        rows.append("- Not enough comparable examples to generate an automatic interpretation.")
    return rows


def _select_error_examples(per_example: list[Any], *, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    rows = [row for row in per_example if isinstance(row, dict)]

    def is_problem(row: dict[str, Any]) -> bool:
        checks = [
            row.get("agent_hit_correct"),
            row.get("step_hit_pm1_correct"),
            row.get("step_hit_correct"),
        ]
        return any(value is False for value in checks) or int(row.get("invalid_findings_count") or 0) > 0

    def severity(row: dict[str, Any]) -> tuple[int, int, str]:
        misses = sum(1 for key in ("agent_hit_correct", "step_hit_pm1_correct", "step_hit_correct") if row.get(key) is False)
        invalid = int(row.get("invalid_findings_count") or 0)
        return (-misses, -invalid, str(row.get("file") or ""))

    return sorted([row for row in rows if is_problem(row)], key=severity)[:limit]


def _pct(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "100.0%" if value else "0.0%"
    if isinstance(value, (int, float)):
        return f"{value * 100:.1f}%"
    return _md(value)


def _check(value: Any) -> str:
    if value is True:
        return "✅"
    if value is False:
        return "❌"
    return "—"


def _fmt_list(value: Any, *, max_items: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if not isinstance(value, (list, tuple, set)):
        return str(value)
    items = [str(item) for item in value if item is not None and str(item).strip()]
    if not items:
        return ""
    if len(items) > max_items:
        return ", ".join(items[:max_items]) + f", … (+{len(items) - max_items})"
    return ", ".join(items)


def _md(value: Any) -> str:
    if value is None:
        return "—"
    text = str(value)
    text = text.replace("\n", "<br>")
    text = text.replace("|", "\\|")
    return text


if __name__ == "__main__":
    result = main(
        pred_glob=DEFAULT_PRED_GLOB,
        output_json_path=DEFAULT_OUTPUT_JSON_PATH,
        output_md_path=DEFAULT_OUTPUT_MD_PATH,
        hf_annotations=DEFAULT_HF_ANNOTATIONS,
        experiment_name=DEFAULT_EXPERIMENT_NAME,
        model_name="Gemini",
        dataset_name="Who&When / Hand-Crafted",
        run_label="v9 report",
        id_column=None,
        agent_columns=None,
        step_columns=None,
        step_tolerance=1,
        build_missing_report=True,
        top_error_examples=20,
        notes=None,
        print_summary=True,
    )
