"""Debugger-friendly experiment: non-LLM validator weight ablation.

This script reads per-task JSON files produced by ``launch_findings_judges.py``.
Those JSONs are expected to contain:

- LLM metric findings;
- ``evidence_verification``;
- optionally ``non_llm_validators`` produced by ``maseval.validators.run_on_trace``.

For every value of ``non_llm_validator_weight`` it rebuilds ``report`` using the
same raw findings, writes temporary weighted JSONs, and computes Agent Acc /
Step-level Acc against Who&When annotations.

No argparse. Open in debugger and call ``main(...)``.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from maseval.diagnostic_accuracy import evaluate_agent_step_accuracy  # noqa: E402
    from maseval.reporting_weighted import build_weighted_evaluation_report  # noqa: E402
except Exception:  # pragma: no cover - debugger-friendly fallback.
    diag_path = SRC / "maseval" / "diagnostic_accuracy.py"
    diag_spec = importlib.util.spec_from_file_location("maseval_diagnostic_accuracy", diag_path)
    if diag_spec is None or diag_spec.loader is None:
        raise
    diag_module = importlib.util.module_from_spec(diag_spec)
    sys.modules[diag_spec.name] = diag_module
    diag_spec.loader.exec_module(diag_module)
    evaluate_agent_step_accuracy = diag_module.evaluate_agent_step_accuracy

    rep_path = SRC / "maseval" / "reporting_weighted.py"
    rep_spec = importlib.util.spec_from_file_location("maseval_reporting_weighted", rep_path)
    if rep_spec is None or rep_spec.loader is None:
        raise
    rep_module = importlib.util.module_from_spec(rep_spec)
    sys.modules[rep_spec.name] = rep_module
    rep_spec.loader.exec_module(rep_module)
    build_weighted_evaluation_report = rep_module.build_weighted_evaluation_report


DEFAULT_PRED_GLOB = (
    "/home/alina/Desktop/maseval-research/examples/who_and_when/"
    "who&when_algo_gemini_idx_msg_v2/*.json"
)

HF_ANNOTATIONS = {
    "hand": "hf://datasets/Kevin355/Who_and_When/Hand-Crafted.parquet",
    "algo": "hf://datasets/Kevin355/Who_and_When/Algorithm-Generated.parquet",
}

DEFAULT_WEIGHTS = (0.0, 1.0)


def main(
    pred_glob: str | Path | Sequence[str | Path] = DEFAULT_PRED_GLOB,
    *,
    dataset_split: str = "algo",
    weights: Sequence[float] = DEFAULT_WEIGHTS,
    output_dir: str | Path = "non_llm_weight_ablation",
    id_column: str | None = None,
    agent_columns: Sequence[str] | None = None,
    step_columns: Sequence[str] | None = None,
    step_tolerance: int = 1,
    keep_weighted_jsons: bool = True,
    print_summary: bool = True,
) -> dict[str, Any]:
    """Run weight ablation and return all metrics.

    Args:
        pred_glob: Prediction JSON glob/directory/list.
        dataset_split: ``"hand"`` or ``"algo"``. Used only to select the HF
            annotation table.
        weights: Non-LLM validator weights to test.
        output_dir: Directory for weighted JSONs, summary JSON/CSV/MD.
        id_column: Optional annotation id column.
        agent_columns: Optional gold-agent columns.
        step_columns: Optional gold-step/span columns.
        step_tolerance: Numeric tolerance for step matching.
        keep_weighted_jsons: If true, store weighted per-task JSONs for inspection.
        print_summary: If true, print compact table to stdout.

    Returns:
        Dict with ``summary_rows`` and per-weight results.
    """

    prediction_paths = _resolve_prediction_paths(pred_glob)
    if not prediction_paths:
        raise FileNotFoundError(f"No prediction JSON files matched: {pred_glob}")

    dataset_split = dataset_split.lower().strip()
    if dataset_split not in HF_ANNOTATIONS:
        raise ValueError(f"dataset_split must be one of {sorted(HF_ANNOTATIONS)}, got {dataset_split!r}")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    annotations_df = _load_hf_annotations(HF_ANNOTATIONS[dataset_split])
    annotations_path = output_root / f"annotations_{dataset_split}.jsonl"
    _write_annotations_jsonl(annotations_df, annotations_path)

    all_results: dict[str, Any] = {
        "experiment": {
            "name": "non_llm_validator_weight_ablation",
            "dataset_split": dataset_split,
            "annotation_source": HF_ANNOTATIONS[dataset_split],
            "prediction_files": len(prediction_paths),
            "weights": [float(w) for w in weights],
            "step_tolerance": step_tolerance,
        },
        "summary_rows": [],
        "results_by_weight": {},
    }

    for weight in weights:
        weight = float(weight)
        weight_label = _weight_label(weight)
        weighted_dir = output_root / f"weighted_{weight_label}"
        if weighted_dir.exists():
            shutil.rmtree(weighted_dir)
        weighted_dir.mkdir(parents=True, exist_ok=True)

        weighted_paths = _build_weighted_prediction_files(
            prediction_paths=prediction_paths,
            output_dir=weighted_dir,
            non_llm_validator_weight=weight,
        )

        result = evaluate_agent_step_accuracy(
            weighted_paths,
            annotations_path,
            id_column=id_column,
            agent_columns=agent_columns,
            step_columns=step_columns,
            step_tolerance=step_tolerance,
            build_missing_report=False,
        )

        summary = dict(result["summary"])
        summary["non_llm_validator_weight"] = weight
        summary["weighted_json_dir"] = str(weighted_dir)
        summary["non_llm_issues_total"] = _sum_non_llm_issues(weighted_paths)
        summary["llm_issues_total"] = _sum_llm_issues(weighted_paths)
        all_results["summary_rows"].append(summary)
        all_results["results_by_weight"][str(weight)] = result

        if not keep_weighted_jsons:
            shutil.rmtree(weighted_dir, ignore_errors=True)

    summary_json = output_root / "non_llm_weight_ablation_results.json"
    summary_json.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_csv = output_root / "non_llm_weight_ablation_summary.csv"
    _write_summary_csv(all_results["summary_rows"], summary_csv)

    summary_md = output_root / "non_llm_weight_ablation_summary.md"
    _write_summary_md(all_results, summary_md)

    if print_summary:
        print(_format_console_table(all_results["summary_rows"]))
        print(f"\nSaved:\n- {summary_json}\n- {summary_csv}\n- {summary_md}")

    return all_results


def _build_weighted_prediction_files(
    *,
    prediction_paths: Sequence[Path],
    output_dir: Path,
    non_llm_validator_weight: float,
) -> list[Path]:
    weighted_paths: list[Path] = []
    for path in prediction_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["report"] = build_weighted_evaluation_report(
            payload,
            non_llm_validator_weight=non_llm_validator_weight,
            reference_answer=payload.get("reference_answer") or payload.get("label_answer"),
        )
        output_path = output_dir / path.name
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        weighted_paths.append(output_path)
    return weighted_paths


def _sum_non_llm_issues(paths: Sequence[Path]) -> int:
    total = 0
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        aggregation = (payload.get("report") or {}).get("aggregation") or {}
        total += int(aggregation.get("non_llm_issues_used") or 0)
    return total


def _sum_llm_issues(paths: Sequence[Path]) -> int:
    total = 0
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        aggregation = (payload.get("report") or {}).get("aggregation") or {}
        total += int(aggregation.get("llm_issues_used") or 0)
    return total


def _load_hf_annotations(hf_annotations: str) -> pd.DataFrame:
    df = pd.read_parquet(hf_annotations)
    df = df.reset_index(drop=True)
    df["row_index"] = df.index.astype(str)
    return df


def _write_annotations_jsonl(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            cleaned = {str(key): _json_safe(value) for key, value in record.items()}
            f.write(json.dumps(cleaned, ensure_ascii=False) + "\n")


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return str(value)


def _resolve_prediction_paths(pred_glob: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(pred_glob, str):
        matched = glob.glob(pred_glob)
        if matched:
            return sorted(Path(path) for path in matched)
        path = Path(pred_glob)
        if path.is_dir():
            return sorted(path.glob("*.json"))
        return [path]
    if isinstance(pred_glob, Path):
        if pred_glob.is_dir():
            return sorted(pred_glob.glob("*.json"))
        return [pred_glob]
    paths: list[Path] = []
    for item in pred_glob:
        if isinstance(item, Path):
            paths.extend(sorted(item.glob("*.json")) if item.is_dir() else [item])
            continue
        matched = glob.glob(str(item))
        if matched:
            paths.extend(Path(path) for path in matched)
        else:
            path = Path(item)
            paths.extend(sorted(path.glob("*.json")) if path.is_dir() else [path])
    return sorted(paths)


def _weight_label(weight: float) -> str:
    return str(weight).replace(".", "p")


def _write_summary_csv(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    import csv

    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    keys = _summary_keys(rows)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def _write_summary_md(result: Mapping[str, Any], output_path: Path) -> None:
    rows = list(result.get("summary_rows") or [])
    exp = result.get("experiment") or {}
    lines = [
        "# Non-LLM Validator Weight Ablation",
        "",
        "## Experiment setup",
        "",
        f"- Dataset split: `{exp.get('dataset_split')}`",
        f"- Annotation source: `{exp.get('annotation_source')}`",
        f"- Prediction files: `{exp.get('prediction_files')}`",
        f"- Step tolerance: `±{exp.get('step_tolerance')}`",
        f"- Weights: `{exp.get('weights')}`",
        "",
        "LLM judge findings use weight `1.0`. Deterministic validator findings use `λ`.",
        "`λ=0.0` disables deterministic validators. `λ=1.0` makes them equal to LLM findings for ranking.",
        "",
        "## Results",
        "",
        _markdown_table(rows),
        "",
        "## Metric notes",
        "",
        "- **Agent Top-1 Acc**: primary predicted culprit agent matches gold.",
        "- **Agent Hit Acc**: at least one predicted culprit agent matches gold.",
        "- **Step Hit Acc**: at least one predicted problematic step exactly matches gold.",
        "- **Step Hit@±1**: at least one predicted step is within ±1 of gold.",
        "- **Non-LLM issues total**: deterministic validator findings used in reports for this λ.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "No rows."
    cols = [
        "non_llm_validator_weight",
        "agent_top1_acc",
        "agent_hit_acc",
        "agent_exact_set_acc",
        "step_top1_acc",
        "step_hit_acc",
        "step_hit_pm1_acc",
        "llm_issues_total",
        "non_llm_issues_total",
        "invalid_findings_total",
    ]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---:" if c.endswith("acc") or c.endswith("total") or c.endswith("weight") else "---" for c in cols) + " |"
    body = []
    for row in rows:
        values = []
        for col in cols:
            value = row.get(col)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *body])


def _summary_keys(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    preferred = [
        "non_llm_validator_weight",
        "agent_examples",
        "agent_top1_acc",
        "agent_hit_acc",
        "agent_exact_set_acc",
        "step_examples",
        "step_top1_acc",
        "step_hit_acc",
        "step_hit_pm1_acc",
        "llm_issues_total",
        "non_llm_issues_total",
        "invalid_findings_total",
    ]
    rest = sorted({key for row in rows for key in row.keys()} - set(preferred))
    return [key for key in preferred if any(key in row for row in rows)] + rest


def _format_console_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "λ | Agent@1 | Agent Hit | Step Hit | Step Hit±1 | LLM issues | non-LLM issues",
        "--|---------|-----------|----------|------------|------------|---------------",
    ]
    for row in rows:
        lines.append(
            f"{row.get('non_llm_validator_weight')} | "
            f"{_fmt(row.get('agent_top1_acc'))} | "
            f"{_fmt(row.get('agent_hit_acc'))} | "
            f"{_fmt(row.get('step_hit_acc'))} | "
            f"{_fmt(row.get('step_hit_pm1_acc'))} | "
            f"{row.get('llm_issues_total')} | "
            f"{row.get('non_llm_issues_total')}"
        )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


if __name__ == "__main__":
    result = main(
        pred_glob="/home/alina/Desktop/MASeval/examples/who_and_when/who&when_hand_gemini_idx_msg_v2/*.json",
        dataset_split="hand",
        weights=(0.0, 1.0),
        output_dir="non_llm_weight_ablation_hand_gemini",
        step_tolerance=1,
    )