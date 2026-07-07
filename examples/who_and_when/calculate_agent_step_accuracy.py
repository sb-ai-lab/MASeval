"""Debugger-friendly runner for Agent Acc and Step-level Acc.

Open this file in a debugger and call ``main(...)`` directly.
The module does not read command-line arguments and does not require a local
annotation file: annotations are loaded directly from HuggingFace.

Example:

    result = main(
        pred_glob="/home/alina/Desktop/maseval-research/examples/who_and_when/who&when_hand_gemini_findings_v9_report/*.json",
        output_path="agent_step_metrics.json",
    )

The function reads JSON outputs produced by the MASQUE findings/evidence/report
pipeline and compares ``report.diagnostic_report`` against human annotations.
Findings rejected by ``EvidenceVerifier`` are not counted in the main predictions;
they stay in ``review_targets`` and contribute only to ``invalid_findings_total``.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

# Allow running this file from a checkout without installing the package.
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from maseval.diagnostic_accuracy import evaluate_agent_step_accuracy  # noqa: E402
except Exception:  # pragma: no cover - allows running without optional package deps.
    module_path = SRC / "maseval" / "diagnostic_accuracy.py"
    spec = importlib.util.spec_from_file_location("maseval_diagnostic_accuracy", module_path)
    if spec is None or spec.loader is None:
        raise
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    evaluate_agent_step_accuracy = module.evaluate_agent_step_accuracy


DEFAULT_PRED_GLOB = (
    "/home/alina/Desktop/maseval-research/examples/who_and_when/"
    "who&when_hand_gemini_findings_v9_report/*.json"
)

DEFAULT_HF_ANNOTATIONS = "hf://datasets/Kevin355/Who_and_When/Hand-Crafted.parquet"
# DEFAULT_HF_ANNOTATIONS = "hf://datasets/Kevin355/Who_and_When/Algorithm-Generated.parquet"


def main(
    pred_glob: str | Path | Sequence[str | Path] = DEFAULT_PRED_GLOB,
    output_path: str | Path | None = "agent_step_metrics.json",
    hf_annotations: str = DEFAULT_HF_ANNOTATIONS,
    id_column: str | None = None,
    agent_columns: Sequence[str] | None = None,
    step_columns: Sequence[str] | None = None,
    step_tolerance: int = 1,
    build_missing_report: bool = True,
    verifier_mode: str | None = None,
    print_summary: bool = True,
) -> dict:
    """Calculate Agent Acc and Step-level Acc.

    Args:
        pred_glob: Glob string like ``"outputs/*.json"``, a directory path, a
            single JSON path, or an explicit list of JSON paths.
        output_path: Optional path where the full result JSON will be saved.
        hf_annotations: HuggingFace parquet URL for annotation table.
        id_column: Optional annotation id column. If omitted, common id columns
            and filename suffix indices are tried.
        agent_columns: Optional annotation columns with gold culprit agent(s).
            If omitted, ``maseval.diagnostic_accuracy`` auto-detects common names,
            including ``mistake_agent``.
        step_columns: Optional annotation columns with gold problematic step/span id(s).
            If omitted, ``maseval.diagnostic_accuracy`` auto-detects common names,
            including ``mistake_step`` / ``mistake_span`` / ``span_id``.
        step_tolerance: Numeric tolerance for step matching. Default 1 handles
            zero-based vs one-based off-by-one.
        build_missing_report: If true, build ``report`` when a JSON file does not
            already contain one.
        verifier_mode: If not ``None``, rebuild reports under this EvidenceVerifier
            setting (``none``/``strict``/``soft``) for the verifier ablation.
            ``None`` scores the stored report as-is.
        print_summary: If true, print the summary block to stdout.

    Returns:
        Full JSON-serializable result with ``summary`` and ``per_example``.
    """

    prediction_paths = _resolve_prediction_paths(pred_glob)
    if not prediction_paths:
        raise FileNotFoundError(f"No prediction JSON files matched: {pred_glob}")

    annotations_df = _load_hf_annotations(hf_annotations)

    with tempfile.TemporaryDirectory(prefix="maseval_annotations_") as tmp_dir:
        annotations_path = Path(tmp_dir) / "annotations.jsonl"
        _write_annotations_jsonl(annotations_df, annotations_path)

        result = evaluate_agent_step_accuracy(
            prediction_paths,
            annotations_path,
            id_column=id_column,
            agent_columns=agent_columns,
            step_columns=step_columns,
            step_tolerance=step_tolerance,
            build_missing_report=build_missing_report,
            verifier_mode=verifier_mode,
        )

    result["run_config"] = {
        "pred_glob": str(pred_glob),
        "prediction_files": [str(path) for path in prediction_paths],
        "prediction_files_count": len(prediction_paths),
        "hf_annotations": hf_annotations,
        "id_column": id_column,
        "agent_columns": list(agent_columns) if agent_columns is not None else None,
        "step_columns": list(step_columns) if step_columns is not None else None,
        "step_tolerance": step_tolerance,
        "build_missing_report": build_missing_report,
        "verifier_mode": verifier_mode,
    }
    result["annotation_source"] = {
        "type": "huggingface_parquet",
        "path": hf_annotations,
        "rows": int(len(annotations_df)),
        "columns": list(map(str, annotations_df.columns)),
    }

    if print_summary:
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        if print_summary:
            print(f"Saved full metrics to: {output}")

    return result


# Backward-compatible alias, in case older local code imports this name.
calculate_agent_step_accuracy = main


def _load_hf_annotations(hf_annotations: str) -> pd.DataFrame:
    """Load annotation table directly from HuggingFace parquet."""

    df = pd.read_parquet(hf_annotations)

    # Keep row order stable and expose row index for matching gemini_findings_0.json -> row 0.
    df = df.reset_index(drop=True)
    df["row_index"] = df.index.astype(str)

    return df


def _write_annotations_jsonl(df: pd.DataFrame, output_path: Path) -> None:
    """Write a dataframe to JSONL in a format readable by diagnostic_accuracy."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            cleaned = {str(key): _json_safe(value) for key, value in record.items()}
            f.write(json.dumps(cleaned, ensure_ascii=False) + "\n")


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-safe Python values."""

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
    """Resolve glob/list input into sorted JSON paths."""

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


if __name__ == "__main__":
    result = main(
        pred_glob=DEFAULT_PRED_GLOB,
        output_path="agent_step_metrics.json",
        hf_annotations=DEFAULT_HF_ANNOTATIONS,
        id_column=None,
        agent_columns=None,
        step_columns=None,
        step_tolerance=1,
        build_missing_report=True,
        print_summary=True,
    )
