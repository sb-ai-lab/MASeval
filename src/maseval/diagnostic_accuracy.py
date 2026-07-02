"""Accuracy utilities for evidence-grounded diagnostic reports.

The module compares per-trace diagnostic JSON files with human annotations and
computes two simple localization metrics:

* Agent accuracy: did the predicted culprit agent match the annotated agent?
* Step-level accuracy: did the predicted problematic span/step match the
  annotated step/span?

The implementation intentionally reads the compact ``report`` produced by
``maseval.reporting`` when it is present. This means findings rejected by
``EvidenceVerifier`` (``usable_for_diagnosis=false``) do not affect the main
accuracy numbers. If a JSON file does not contain ``report``, the module can
build one on the fly from the metric outputs.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # Optional: only needed for parquet/xlsx annotation files.
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas is optional for library use.
    pd = None  # type: ignore

try:
    from .reporting import build_evaluation_report
except Exception:  # pragma: no cover - keeps the script usable in partial installs.
    # When this module is loaded directly from a script, relative imports are not
    # available. Fall back to loading reporting.py from the same directory.
    try:
        import importlib.util

        _reporting_path = Path(__file__).with_name("reporting.py")
        _spec = importlib.util.spec_from_file_location("maseval_reporting", _reporting_path)
        if _spec is None or _spec.loader is None:
            raise ImportError(f"Could not load {_reporting_path}")
        _module = importlib.util.module_from_spec(_spec)
        import sys

        sys.modules[_spec.name] = _module
        _spec.loader.exec_module(_module)
        build_evaluation_report = _module.build_evaluation_report  # type: ignore[attr-defined]
    except Exception:
        build_evaluation_report = None  # type: ignore

AGENT_COLUMNS = (
    "mistake_agent",
    "mistake_agents",
    "culprit_agent",
    "culprit_agents",
    "problematic_agent",
    "problematic_agents",
    "gold_agent",
    "gold_agents",
    "annotated_agent",
    "annotated_agents",
    "agent",
    "agents",
)

STEP_COLUMNS = (
    "mistake_step",
    "mistake_steps",
    "mistake_span",
    "mistake_spans",
    "problematic_step",
    "problematic_steps",
    "problematic_span",
    "problematic_spans",
    "gold_step",
    "gold_steps",
    "gold_span",
    "gold_spans",
    "annotated_step",
    "annotated_steps",
    "annotated_span",
    "annotated_spans",
    "step_id",
    "step_ids",
    "span_id",
    "span_ids",
)

ID_COLUMNS = (
    "trace_id",
    "sample_id",
    "task_id",
    "trajectory_id",
    "conversation_id",
    "name",
    "id",
    "index",
    "row_id",
)

METADATA_ID_KEYS = (
    "trace_id",
    "sample_id",
    "task_id",
    "trajectory_id",
    "conversation_id",
    "name",
    "id",
    "index",
    "row_id",
)

NULL_STRINGS = {"", "none", "null", "nan", "n/a", "na", "-"}


@dataclass(slots=True)
class PredictionRecord:
    """Predicted diagnostic localization for one JSON file."""

    file: str
    example_id: str | None
    agents: list[str]
    primary_agent: str | None
    spans: list[str]
    first_span: str | None
    invalid_findings_count: int


@dataclass(slots=True)
class AnnotationRecord:
    """Gold diagnostic localization for one annotated example."""

    example_id: str | None
    row_index: int
    agents: list[str]
    spans: list[str]
    raw: dict[str, Any]


@dataclass(slots=True)
class ExampleComparison:
    """Per-example comparison result."""

    file: str
    example_id: str | None
    annotation_row_index: int | None
    gold_agents: list[str]
    predicted_agents: list[str]
    primary_agent: str | None
    agent_top1_correct: bool | None
    agent_hit_correct: bool | None
    agent_exact_set_correct: bool | None
    gold_spans: list[str]
    predicted_spans: list[str]
    first_span: str | None
    step_top1_correct: bool | None
    step_hit_correct: bool | None
    step_hit_pm1_correct: bool | None
    invalid_findings_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "example_id": self.example_id,
            "annotation_row_index": self.annotation_row_index,
            "gold_agents": self.gold_agents,
            "predicted_agents": self.predicted_agents,
            "primary_agent": self.primary_agent,
            "agent_top1_correct": self.agent_top1_correct,
            "agent_hit_correct": self.agent_hit_correct,
            "agent_exact_set_correct": self.agent_exact_set_correct,
            "gold_spans": self.gold_spans,
            "predicted_spans": self.predicted_spans,
            "first_span": self.first_span,
            "step_top1_correct": self.step_top1_correct,
            "step_hit_correct": self.step_hit_correct,
            "step_hit_pm1_correct": self.step_hit_pm1_correct,
            "invalid_findings_count": self.invalid_findings_count,
        }


def evaluate_agent_step_accuracy(
    prediction_paths: Sequence[str | Path],
    annotation_path: str | Path,
    *,
    id_column: str | None = None,
    agent_columns: Sequence[str] | None = None,
    step_columns: Sequence[str] | None = None,
    step_tolerance: int = 1,
    build_missing_report: bool = True,
) -> dict[str, Any]:
    """Compare prediction JSON files with annotations and compute accuracies.

    Args:
        prediction_paths: JSON files produced by the findings/evidence/report
            pipeline.
        annotation_path: CSV/JSON/JSONL/parquet annotation file.
        id_column: Optional annotation id column used to match JSON files.
            If omitted, common id columns are tried; filename suffix indices are
            also matched to annotation row numbers.
        agent_columns: Optional annotation columns containing gold culprit agents.
        step_columns: Optional annotation columns containing gold step/span ids.
        step_tolerance: Numeric tolerance for step matching. ``1`` allows
            zero-based vs one-based off-by-one matches.
        build_missing_report: If true, build ``report`` from metric outputs when
            a JSON file does not already contain one.

    Returns:
        JSON-serializable dict with summary metrics and per-example results.
    """

    annotations = read_annotations(
        annotation_path,
        id_column=id_column,
        agent_columns=agent_columns,
        step_columns=step_columns,
    )
    annotation_index = _build_annotation_index(annotations, id_column=id_column)

    predictions = [
        read_prediction_file(path, build_missing_report=build_missing_report)
        for path in sorted(map(Path, prediction_paths), key=lambda p: str(p))
    ]

    comparisons: list[ExampleComparison] = []
    unmatched_predictions: list[dict[str, Any]] = []

    for prediction in predictions:
        annotation = _match_annotation(prediction, annotation_index)
        if annotation is None:
            unmatched_predictions.append(
                {"file": prediction.file, "example_id": prediction.example_id}
            )
            continue
        comparisons.append(
            compare_prediction_to_annotation(
                prediction,
                annotation,
                step_tolerance=step_tolerance,
            )
        )

    return {
        "summary": _summarize_comparisons(
            comparisons,
            total_prediction_files=len(predictions),
            unmatched_predictions=len(unmatched_predictions),
        ),
        "unmatched_predictions": unmatched_predictions,
        "per_example": [comparison.to_dict() for comparison in comparisons],
    }


def read_prediction_file(
    path: str | Path,
    *,
    build_missing_report: bool = True,
) -> PredictionRecord:
    """Extract predicted problematic agents and spans from one JSON file."""

    file_path = Path(path)
    data = json.loads(file_path.read_text(encoding="utf-8"))

    report = data.get("report")
    if not isinstance(report, Mapping) and build_missing_report:
        if build_evaluation_report is None:
            raise RuntimeError(
                "JSON file has no report and maseval.reporting could not be imported."
            )
        report = build_evaluation_report(data)

    report = report if isinstance(report, Mapping) else {}
    diagnostic_report = report.get("diagnostic_report") or {}
    status = report.get("status") or {}
    diagnostic_status = status.get("diagnostic_status") or {}

    problematic_agents = diagnostic_report.get("problematic_agents") or []
    agents = _unique_preserve_order(
        _extract_named_values(problematic_agents, key="agent")
    )
    primary_agent = _clean_scalar(diagnostic_status.get("primary_culprit_agent"))
    if primary_agent and primary_agent not in agents:
        agents.insert(0, primary_agent)

    problematic_spans = diagnostic_report.get("problematic_spans") or []
    spans = _unique_preserve_order(
        _extract_named_values(problematic_spans, key="span_id")
    )
    first_span = _clean_scalar(diagnostic_status.get("first_problem_span"))
    if first_span and first_span not in spans:
        spans.insert(0, first_span)

    # Fallback to issues if the report omits pre-aggregated lists. These are
    # already EvidenceVerifier-gated by report construction, so invalid findings
    # are still excluded from main predictions.
    issues = diagnostic_report.get("issues") or []
    if not agents:
        agents = _unique_preserve_order(
            value
            for issue in issues
            for value in _as_list(issue.get("culprit_agents"))
        )
    if not spans:
        spans = _unique_preserve_order(
            value
            for issue in issues
            for value in _as_list(issue.get("problematic_spans"))
        )

    if primary_agent is None and agents:
        primary_agent = agents[0]
    if first_span is None and spans:
        first_span = spans[0]

    review_targets = diagnostic_report.get("review_targets") or []
    invalid_findings_count = sum(
        1
        for item in review_targets
        if isinstance(item, Mapping)
        and str(item.get("evidence_status", "")).lower() == "invalid"
    )

    return PredictionRecord(
        file=str(file_path),
        example_id=_infer_prediction_id(data, file_path),
        agents=agents,
        primary_agent=primary_agent,
        spans=spans,
        first_span=first_span,
        invalid_findings_count=invalid_findings_count,
    )


def read_annotations(
    annotation_path: str | Path,
    *,
    id_column: str | None = None,
    agent_columns: Sequence[str] | None = None,
    step_columns: Sequence[str] | None = None,
) -> list[AnnotationRecord]:
    """Read annotation file and normalize gold agents/spans.

    Supported formats: CSV/TSV, JSON list, JSONL/NDJSON, parquet. Excel is
    supported when pandas has the relevant optional dependencies installed.
    """

    rows = _read_annotation_rows(Path(annotation_path))
    agent_columns = tuple(agent_columns or AGENT_COLUMNS)
    step_columns = tuple(step_columns or STEP_COLUMNS)

    records: list[AnnotationRecord] = []
    for row_index, row in enumerate(rows):
        normalized_row = {str(k): v for k, v in row.items()}
        example_id = _infer_annotation_id(normalized_row, row_index, id_column=id_column)
        agents = _extract_annotation_values(normalized_row, agent_columns)
        spans = _extract_annotation_values(normalized_row, step_columns)
        records.append(
            AnnotationRecord(
                example_id=example_id,
                row_index=row_index,
                agents=agents,
                spans=spans,
                raw=normalized_row,
            )
        )
    return records


def compare_prediction_to_annotation(
    prediction: PredictionRecord,
    annotation: AnnotationRecord,
    *,
    step_tolerance: int = 1,
) -> ExampleComparison:
    """Compare one prediction with one annotation."""

    gold_agents_norm = {_normalize_agent(agent) for agent in annotation.agents}
    pred_agents_norm = [_normalize_agent(agent) for agent in prediction.agents]
    primary_agent_norm = (
        _normalize_agent(prediction.primary_agent) if prediction.primary_agent else None
    )

    has_gold_agents = bool(gold_agents_norm)
    agent_top1_correct = None
    agent_hit_correct = None
    agent_exact_set_correct = None
    if has_gold_agents:
        agent_top1_correct = bool(primary_agent_norm and primary_agent_norm in gold_agents_norm)
        agent_hit_correct = any(agent in gold_agents_norm for agent in pred_agents_norm)
        agent_exact_set_correct = set(pred_agents_norm) == gold_agents_norm

    has_gold_spans = bool(annotation.spans)
    step_top1_correct = None
    step_hit_correct = None
    step_hit_pm1_correct = None
    if has_gold_spans:
        step_top1_correct = bool(
            prediction.first_span
            and _span_matches_any(prediction.first_span, annotation.spans, tolerance=0)
        )
        step_hit_correct = any(
            _span_matches_any(span, annotation.spans, tolerance=0)
            for span in prediction.spans
        )
        step_hit_pm1_correct = any(
            _span_matches_any(span, annotation.spans, tolerance=step_tolerance)
            for span in prediction.spans
        )

    return ExampleComparison(
        file=prediction.file,
        example_id=prediction.example_id or annotation.example_id,
        annotation_row_index=annotation.row_index,
        gold_agents=annotation.agents,
        predicted_agents=prediction.agents,
        primary_agent=prediction.primary_agent,
        agent_top1_correct=agent_top1_correct,
        agent_hit_correct=agent_hit_correct,
        agent_exact_set_correct=agent_exact_set_correct,
        gold_spans=annotation.spans,
        predicted_spans=prediction.spans,
        first_span=prediction.first_span,
        step_top1_correct=step_top1_correct,
        step_hit_correct=step_hit_correct,
        step_hit_pm1_correct=step_hit_pm1_correct,
        invalid_findings_count=prediction.invalid_findings_count,
    )


def _summarize_comparisons(
    comparisons: Sequence[ExampleComparison],
    *,
    total_prediction_files: int,
    unmatched_predictions: int,
) -> dict[str, Any]:
    agent_rows = [c for c in comparisons if c.agent_hit_correct is not None]
    step_rows = [c for c in comparisons if c.step_hit_correct is not None]

    return {
        "total_prediction_files": total_prediction_files,
        "matched_annotations": len(comparisons),
        "unmatched_predictions": unmatched_predictions,
        "agent_examples": len(agent_rows),
        "agent_top1_acc": _mean_bool(c.agent_top1_correct for c in agent_rows),
        "agent_hit_acc": _mean_bool(c.agent_hit_correct for c in agent_rows),
        "agent_exact_set_acc": _mean_bool(c.agent_exact_set_correct for c in agent_rows),
        "step_examples": len(step_rows),
        "step_top1_acc": _mean_bool(c.step_top1_correct for c in step_rows),
        "step_hit_acc": _mean_bool(c.step_hit_correct for c in step_rows),
        "step_hit_pm1_acc": _mean_bool(c.step_hit_pm1_correct for c in step_rows),
        "invalid_findings_total": sum(c.invalid_findings_count for c in comparisons),
    }


def _read_annotation_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if isinstance(item, Mapping):
                    rows.append(dict(item))
        return rows

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [dict(item) for item in data if isinstance(item, Mapping)]
        if isinstance(data, Mapping):
            for key in ("annotations", "data", "rows", "examples"):
                value = data.get(key)
                if isinstance(value, list):
                    return [dict(item) for item in value if isinstance(item, Mapping)]
            return [dict(data)]
        raise ValueError(f"Unsupported JSON annotation structure: {path}")

    if suffix in {".parquet", ".pq", ".xlsx", ".xls"}:
        if pd is None:
            raise RuntimeError(f"Reading {suffix} annotations requires pandas.")
        if suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(path)
        else:
            df = pd.read_excel(path)
        return _records_from_dataframe(df)

    # CSV/TSV fallback. Try UTF-8 first, then common Windows encodings because
    # some existing annotation exports are cp1251/cp1252.
    delimiter = "\t" if suffix == ".tsv" else None
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "cp1251", "cp1252"):
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                sample = f.read(4096)
                f.seek(0)
                dialect = csv.Sniffer().sniff(sample) if delimiter is None else None
                reader = csv.DictReader(f, dialect=dialect) if dialect else csv.DictReader(f, delimiter=delimiter or ",")
                return [dict(row) for row in reader]
        except Exception as exc:  # noqa: PERF203 - fallback encodings are intentional.
            last_error = exc
    raise RuntimeError(f"Could not read annotation file {path}: {last_error}")


def _records_from_dataframe(df: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        cleaned = {}
        for key, value in record.items():
            if isinstance(value, float) and math.isnan(value):
                value = None
            cleaned[str(key)] = value
        records.append(cleaned)
    return records


def _build_annotation_index(
    annotations: Sequence[AnnotationRecord],
    *,
    id_column: str | None = None,
) -> dict[str, AnnotationRecord]:
    index: dict[str, AnnotationRecord] = {}
    for annotation in annotations:
        keys = {str(annotation.row_index)}
        if annotation.example_id is not None:
            keys.add(str(annotation.example_id))
        if id_column and annotation.raw.get(id_column) is not None:
            keys.add(str(annotation.raw[id_column]))
        for key in ID_COLUMNS:
            value = annotation.raw.get(key)
            if value is not None and _clean_scalar(value) is not None:
                keys.add(str(value))
        for key in keys:
            index.setdefault(key, annotation)
    return index


def _match_annotation(
    prediction: PredictionRecord,
    annotation_index: Mapping[str, AnnotationRecord],
) -> AnnotationRecord | None:
    candidates = []
    if prediction.example_id is not None:
        candidates.append(str(prediction.example_id))
    file_index = _filename_last_int(Path(prediction.file))
    if file_index is not None:
        candidates.append(str(file_index))
    for candidate in candidates:
        if candidate in annotation_index:
            return annotation_index[candidate]
    return None


def _infer_prediction_id(data: Mapping[str, Any], path: Path) -> str | None:
    for key in METADATA_ID_KEYS:
        value = data.get(key)
        cleaned = _clean_scalar(value)
        if cleaned is not None:
            return cleaned

    metadata = data.get("metadata") or data.get("trace_metadata") or {}
    if isinstance(metadata, Mapping):
        for key in METADATA_ID_KEYS:
            cleaned = _clean_scalar(metadata.get(key))
            if cleaned is not None:
                return cleaned

    file_index = _filename_last_int(path)
    return str(file_index) if file_index is not None else None


def _infer_annotation_id(
    row: Mapping[str, Any],
    row_index: int,
    *,
    id_column: str | None = None,
) -> str | None:
    if id_column:
        return _clean_scalar(row.get(id_column)) or str(row_index)
    for key in ID_COLUMNS:
        cleaned = _clean_scalar(row.get(key))
        if cleaned is not None:
            return cleaned
    return str(row_index)


def _filename_last_int(path: Path) -> int | None:
    matches = re.findall(r"\d+", path.stem)
    return int(matches[-1]) if matches else None


def _extract_named_values(items: Any, *, key: str) -> list[str]:
    values: list[str] = []
    for item in _as_list(items):
        if isinstance(item, Mapping):
            cleaned = _clean_scalar(item.get(key))
            if cleaned is not None:
                values.append(cleaned)
        else:
            cleaned = _clean_scalar(item)
            if cleaned is not None:
                values.append(cleaned)
    return values


def _extract_annotation_values(
    row: Mapping[str, Any],
    columns: Sequence[str],
) -> list[str]:
    values: list[str] = []
    lower_to_actual = {str(key).lower(): str(key) for key in row.keys()}
    for column in columns:
        actual = lower_to_actual.get(column.lower())
        if actual is None:
            continue
        values.extend(_as_list(row.get(actual)))
    return _unique_preserve_order(_clean_scalar(value) for value in values)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, list | tuple | set):
        return list(value)
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in NULL_STRINGS:
            return []
        # Try JSON-ish list first: ["A", "B"] or ['A', 'B'].
        if (text.startswith("[") and text.endswith("]")) or (
            text.startswith("{") and text.endswith("}")
        ):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return parsed
                if isinstance(parsed, Mapping):
                    return [parsed]
            except Exception:
                # Annotation CSVs sometimes contain Python repr lists.
                import ast

                try:
                    parsed = ast.literal_eval(text)
                    if isinstance(parsed, list | tuple | set):
                        return list(parsed)
                    if isinstance(parsed, Mapping):
                        return [parsed]
                except Exception:
                    pass
        # Common multi-label separators.
        for sep in (";", "|", ","):
            if sep in text:
                return [part.strip() for part in text.split(sep) if part.strip()]
        return [text]
    return [value]


def _clean_scalar(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if text.lower() in NULL_STRINGS:
        return None
    return text


def _normalize_agent(agent: Any) -> str:
    text = _clean_scalar(agent) or ""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _normalize_span(span: Any) -> str:
    text = _clean_scalar(span) or ""
    return re.sub(r"\s+", "", text.lower())


def _span_number(span: Any) -> int | None:
    text = _clean_scalar(span)
    if text is None:
        return None
    # Handles "17", "span_0017", "message[17]". For non-numeric ids, exact
    # normalized string matching is used instead.
    matches = re.findall(r"\d+", text)
    return int(matches[-1]) if matches else None


def _span_matches_any(pred_span: Any, gold_spans: Iterable[Any], *, tolerance: int) -> bool:
    pred_norm = _normalize_span(pred_span)
    pred_num = _span_number(pred_span)
    for gold in gold_spans:
        gold_norm = _normalize_span(gold)
        if pred_norm and pred_norm == gold_norm:
            return True
        gold_num = _span_number(gold)
        if pred_num is not None and gold_num is not None:
            if abs(pred_num - gold_num) <= tolerance:
                return True
    return False


def _unique_preserve_order(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = _clean_scalar(value)
        if cleaned is None:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def _mean_bool(values: Iterable[bool | None]) -> float | None:
    concrete = [bool(v) for v in values if v is not None]
    if not concrete:
        return None
    return sum(concrete) / len(concrete)
