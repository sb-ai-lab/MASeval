"""Verifier-mode ablation for Who&When WITH non-LLM validators included.

Mirrors ``examples/who_and_when/verifier_ablation.py`` (Agent/Step localization
accuracy under EvidenceVerifier gating none/strict/soft) but the predictions
used for scoring also count the deterministic, non-LLM validators
(``non_llm_validators``) alongside the LLM evaluators.

Why a custom builder instead of ``run_non_llm_weight_ablation``:
``maseval.reporting_weighted.build_weighted_evaluation_report`` merges the
validators into the predicted problematic *agents* (and spans) but only emits
``problematic_spans`` -- never ``problematic_idxs``. ``evaluate_agent_step_accuracy``
reads ``problematic_idxs`` for step accuracy, so step accuracy collapses to 0 on
the weighted path. Here we keep the weighted builder for the agent side, but we
recompute ``problematic_idxs`` by merging (a) the LLM-gated idxs from the standard
report and (b) the numeric ``evidence.idx`` exposed by validator findings. This
yields a faithful "with validators" variant where both agent and step accuracy
include the validators.

Validators are included with weight 1.0 (equal to LLM findings). No weight sweep:
the deliverable is the with-validators (lambda=1.0) variant across verifier
modes, plus an LLM-only (lambda=0.0) baseline for contrast.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
THIS_DIR = Path(__file__).resolve().parent
for p in (str(SRC), str(THIS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from maseval.reporting import build_evaluation_report  # noqa: E402
from maseval.reporting_weighted import build_weighted_evaluation_report  # noqa: E402
from maseval.diagnostic_accuracy import evaluate_agent_step_accuracy  # noqa: E402
import run_non_llm_weight_ablation as wab  # annotations loading helpers  # noqa: E402

MODES = ("none", "strict", "soft")
METRIC_KEYS = [
    ("agent_top1_acc", "Agent Top-1"),
    ("agent_hit_acc", "Agent Hit"),
    ("agent_exact_set_acc", "Agent Exact-Set"),
    ("step_top1_acc", "Step Top-1"),
    ("step_hit_acc", "Step Hit"),
    ("step_hit_pm1_acc", "Step Hit ±1"),
]

HF_ANNOTATIONS = {
    "hand": "hf://datasets/Kevin355/Who_and_When/Hand-Crafted.parquet",
    "algo": "hf://datasets/Kevin355/Who_and_When/Algorithm-Generated.parquet",
}
SPLIT_DIRS = {
    "hand": THIS_DIR / "09_07_ww" / "who&when_hand_gemini_findings_09_07_old",
    "algo": THIS_DIR / "09_07_ww" / "who&when_algo_gemini_findings_09_07_old",
}
VALIDATOR_WEIGHT = 1.0


def _set_usable(verification: dict, mode: str) -> None:
    """Mirror reporting._is_usable so the weighted builder gates LLM findings."""
    status = str(verification.get("evidence_status", "")).lower()
    if mode == "none":
        usable = True
    elif mode == "strict":
        usable = status == "verified"
    else:  # soft
        usable = status != "invalid"
    verification["usable_for_diagnosis"] = usable


def _validator_idxs(payload: dict) -> list[dict]:
    """Numeric step idxs from deterministic validators (only failing ones)."""
    out: list[dict] = []
    nv = payload.get("non_llm_validators", {})
    for blk in (nv.get("metrics") or {}).values():
        if not isinstance(blk, dict):
            continue
        for f in blk.get("findings") or []:
            if str(f.get("verdict", "")).lower() != "fail":
                continue
            count = int(f.get("occurrences", 1) or 1)
            for ev in f.get("evidence") or []:
                idx = str(ev.get("idx", "")).strip()
                if idx.lstrip("-").isdigit():
                    out.append({"idx": idx, "count": count})
    return out


def _merge_idxs(a: list[dict], b: list[dict]) -> list[dict]:
    merged: dict[str, int] = {}
    for item in (a or []):
        merged[str(item["idx"])] = merged.get(str(item["idx"]), 0) + int(item.get("count", 1))
    for item in (b or []):
        merged[str(item["idx"])] = merged.get(str(item["idx"]), 0) + int(item.get("count", 1))
    return [{"idx": k, "count": v} for k, v in merged.items()]


def _idx_sort_key(s: str):
    s = str(s)
    return (0, int(s)) if s.lstrip("-").isdigit() else (1, s)


def rebuild_report_with_validators(payload: dict, mode: str, weight: float) -> dict:
    """Return a copy of payload whose ``report`` includes LLM + validators.

    Agent side via the weighted builder (validators merged). Step side via a
    merged ``problematic_idxs`` (LLM-gated idxs + validator numeric idxs).
    """
    payload2 = copy.deepcopy(payload)
    ev = payload2.get("evidence_verification")
    if isinstance(ev, dict):
        for blk in ev.values():
            if isinstance(blk, dict):
                for v in blk.get("verifications", []):
                    _set_usable(v, mode)

    weighted = build_weighted_evaluation_report(payload2, non_llm_validator_weight=weight)

    if weight > 0:
        llm_rep = build_evaluation_report(payload2, verifier_mode=mode, first_idx_mode="min_index")
        llm_idxs = (llm_rep.get("diagnostic_report") or {}).get("problematic_idxs") or []
        val_idxs = _validator_idxs(payload2)
        combined = _merge_idxs(llm_idxs, val_idxs)
        dr = weighted.setdefault("diagnostic_report", {})
        dr["problematic_idxs"] = combined
        if combined:
            first = min((str(i["idx"]) for i in combined), key=_idx_sort_key)
            weighted.setdefault("status", {}).setdefault("diagnostic_status", {})["first_problem_idx"] = first

    payload2["report"] = weighted
    return payload2


def rebuild_report_llm_only(payload: dict, mode: str) -> dict:
    """LLM-only baseline (validators excluded) -- exactly verifier_ablation.py."""
    payload2 = copy.deepcopy(payload)
    report = build_evaluation_report(payload2, verifier_mode=mode, first_idx_mode="min_index")
    payload2["report"] = report
    return payload2


def _score_split(split: str, step_tolerance: int, output_dir: Path) -> dict:
    pred_glob = str(SPLIT_DIRS[split] / "*.json")
    prediction_paths = wab._resolve_prediction_paths(pred_glob)
    if not prediction_paths:
        raise FileNotFoundError(f"No prediction files for split {split}: {pred_glob}")

    annotations_df = wab._load_hf_annotations(HF_ANNOTATIONS[split])
    annotations_path = output_dir / "annotations.jsonl"
    wab._write_annotations_jsonl(annotations_df, annotations_path)

    results: dict[str, dict] = {}

    def run_variant(label: str, builder, weight: float):
        variant_dir = output_dir / label
        variant_dir.mkdir(parents=True, exist_ok=True)
        weighted_paths = []
        for p in prediction_paths:
            data = json.loads(Path(p).read_text(encoding="utf-8"))
            rebuilt = builder(data, weight) if weight is not None else builder(data)
            out_path = variant_dir / Path(p).name
            out_path.write_text(json.dumps(rebuilt, ensure_ascii=False, indent=2), encoding="utf-8")
            weighted_paths.append(out_path)
        result = evaluate_agent_step_accuracy(
            weighted_paths,
            str(annotations_path),
            step_tolerance=step_tolerance,
            build_missing_report=False,
        )
        results[label] = result["summary"]

    # With validators (lambda = 1.0) across verifier modes.
    for mode in MODES:
        run_variant(f"with_validators_{mode}", lambda d, _w=weight: rebuild_report_with_validators(d, mode, VALIDATOR_WEIGHT), VALIDATOR_WEIGHT)
    # LLM-only baseline (lambda = 0.0) across verifier modes.
    for mode in MODES:
        run_variant(f"llm_only_{mode}", lambda d: rebuild_report_llm_only(d, mode), None)

    return results


def _pct(v):
    return f"{v * 100:5.1f}%" if isinstance(v, (int, float)) else "    —"


def _render(split: str, results: dict) -> str:
    lines = [f"=== Who&When {split.upper()} — verifier ablation WITH validators ===", ""]
    # With validators table.
    lines.append(f"With validators (lambda={VALIDATOR_WEIGHT}):")
    lines.append(f"{'Metric':<16}" + "".join(f"{m:>10}" for m in MODES))
    lines.append("-" * 46)
    for key, label in METRIC_KEYS:
        row = "".join(_pct(results[f"with_validators_{m}"].get(key)).rjust(10) for m in MODES)
        lines.append(f"{label:<16}{row}")
    lines.append("")

    # LLM-only baseline table.
    lines.append("Baseline LLM-only (lambda=0.0, validators excluded):")
    lines.append(f"{'Metric':<16}" + "".join(f"{m:>10}" for m in MODES))
    lines.append("-" * 46)
    for key, label in METRIC_KEYS:
        row = "".join(_pct(results[f"llm_only_{m}"].get(key)).rjust(10) for m in MODES)
        lines.append(f"{label:<16}{row}")
    lines.append("")
    return "\n".join(lines)


def main(step_tolerance: int = 1, output_root: Path | None = None):
    output_root = Path(output_root or (THIS_DIR / "09_07_ww" / "validator_ablation"))
    output_root.mkdir(parents=True, exist_ok=True)

    all_text = []
    for split in ("hand", "algo"):
        out_dir = output_root / split
        results = _score_split(split, step_tolerance, out_dir)
        text = _render(split, results)
        all_text.append(text)
        # Per-split summary JSON.
        summary = {k: v for k, v in results.items()}
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    full = "\n".join(all_text)
    print(full)
    (output_root / "validator_ablation_with_validators.md").write_text(full + "\n", encoding="utf-8")
    print(f"\nWritten: {output_root / 'validator_ablation_with_validators.md'}")


if __name__ == "__main__":
    main(step_tolerance=1)
