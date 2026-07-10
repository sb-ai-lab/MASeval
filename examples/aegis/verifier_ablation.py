"""EvidenceVerifier ablation for AEGIS agent localization.

Scores the same AEGIS result files under three verifier gating settings and
writes one JSON report per setting, plus a printed comparison table:

* ``none``   -- no verifier: every LLM finding counts;
* ``strict`` -- only ``verified`` findings count (weak + invalid -> review);
* ``soft``   -- ``verified``/``weak`` count, ``invalid`` -> review (prior default).

``non_llm_validators`` are NOT counted in any setting (LLM findings only).

Unlike the Who&When ablation (which reads gold from an HF parquet), AEGIS gold
culprit agents are embedded in each result file (``ground_truth_faulty_agents``);
we build the annotation table from them, keyed on ``sample_id`` (the stable global
row index — the AEGIS ``id`` is NOT unique). AEGIS gold has no step index, so only
the AGENT-level metrics are populated.

Debugger-friendly: call ``run(...)`` directly, or run as a script.

    run(pred_glob="aegis_findings")
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
for p in (ROOT / "src", THIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from maseval.diagnostic_accuracy import (  # noqa: E402
    evaluate_agent_step_accuracy,
    _normalize_agent,
)

MODES = ("none", "strict", "soft")


def calculate_metrics(true_labels_list, pred_labels_list):
    """Set-based micro/macro precision/recall/F1 — verbatim port of the AutoJudge
    AEGIS ``calculate.py`` reference math (kept inline so this ablation is
    self-contained on ``main``, where ``calculate.py`` lives only on the aegis branch).
    """
    all_classes = set()
    for labels in true_labels_list:
        all_classes.update(labels)
    for labels in pred_labels_list:
        all_classes.update(labels)
    all_classes = list(all_classes)

    micro_tp = micro_fp = micro_fn = 0
    per_class_stats = {cls: {"tp": 0, "fp": 0, "fn": 0} for cls in all_classes}

    for true_set, pred_set in zip(true_labels_list, pred_labels_list):
        tp_set = true_set.intersection(pred_set)
        fp_set = pred_set.difference(true_set)
        fn_set = true_set.difference(pred_set)
        micro_tp += len(tp_set)
        micro_fp += len(fp_set)
        micro_fn += len(fn_set)
        for cls in tp_set:
            per_class_stats[cls]["tp"] += 1
        for cls in fp_set:
            per_class_stats[cls]["fp"] += 1
        for cls in fn_set:
            per_class_stats[cls]["fn"] += 1

    def _prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f

    micro_p, micro_r, micro_f1 = _prf(micro_tp, micro_fp, micro_fn)

    macro_p_list, macro_r_list, macro_f1_list = [], [], []
    for cls in all_classes:
        s = per_class_stats[cls]
        p, r, f = _prf(s["tp"], s["fp"], s["fn"])
        macro_p_list.append(p)
        macro_r_list.append(r)
        macro_f1_list.append(f)

    _avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return {
        "precision_micro": micro_p,
        "recall_micro": micro_r,
        "f1_micro": micro_f1,
        "precision_macro": _avg(macro_p_list),
        "recall_macro": _avg(macro_r_list),
        "f1_macro": _avg(macro_f1_list),
    }


# AEGIS is agent-level only (gold has no step index). The *_acc metrics come from
# the WW code path; the P/R/F1 keys are the set-based micro/macro scores computed
# by ``calculate_metrics`` (the AutoJudge AEGIS reference math), folded in per mode.
METRIC_KEYS = [
    ("agent_top1_acc", "Agent Top-1"),
    ("agent_hit_acc", "Agent Hit"),
    ("agent_exact_set_acc", "Agent Exact-Set"),
    ("precision_micro", "Precision (micro)"),
    ("recall_micro", "Recall (micro)"),
    ("f1_micro", "F1 (micro)"),
    ("f1_macro", "F1 (macro)"),
]


def _agent_f1(per_example: list[dict]) -> dict:
    """Set-based micro/macro P/R/F1 over normalized agent names, per calculate.py."""
    true_list, pred_list = [], []
    for row in per_example:
        true_list.append({_normalize_agent(a) for a in row.get("gold_agents") or []})
        pred_list.append({_normalize_agent(a) for a in row.get("predicted_agents") or []})
    return calculate_metrics(true_list, pred_list)


def _gt_agent_names(rec: dict) -> list[str]:
    out = []
    for a in rec.get("ground_truth_faulty_agents") or []:
        name = a.get("agent_name") if isinstance(a, dict) else a
        if name:
            out.append(str(name))
    return out


def run(
    pred_glob: str | Path = THIS_DIR / "aegis_findings",
    *,
    split: str = "aegis",
    dataset_name: str = "AEGIS",
    reports_dir: str | Path = THIS_DIR / "reports",
) -> dict[str, dict]:
    """Score the 3 verifier modes, write per-mode reports, return the summaries."""
    files = sorted(Path(pred_glob).glob("*.json"))
    if not files:
        raise SystemExit(f"No result files in {pred_glob}")

    # Build the annotation table from the ground truth embedded in each result file.
    ann_rows = []
    for fp in files:
        rec = json.load(open(fp, encoding="utf-8"))
        ann_rows.append({"sample_id": rec.get("sample_id"), "faulty_agents": _gt_agent_names(rec)})

    ann_path = Path(pred_glob).parent / "aegis_annotations.jsonl"
    with open(ann_path, "w", encoding="utf-8") as f:
        for row in ann_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    file_strs = [str(fp) for fp in files]
    summaries: dict[str, dict] = {}
    for mode in MODES:
        result = evaluate_agent_step_accuracy(
            prediction_paths=file_strs,
            annotation_path=str(ann_path),
            id_column="sample_id",
            agent_columns=["faulty_agents"],
            build_missing_report=True,
            verifier_mode=mode,
        )
        summary = dict(result["summary"])
        summary.update(_agent_f1(result["per_example"]))
        summaries[mode] = summary
        result["summary"] = summary
        with open(reports_dir / f"verifier_ablation_{split}_{mode}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    print(_render(split, dataset_name, summaries))
    return summaries


def _pct(v):
    return f"{v * 100:5.1f}%" if isinstance(v, (int, float)) else "    —"


def _render(split: str, dataset_name: str, summaries: dict[str, dict]) -> str:
    matched = summaries["soft"].get("matched_annotations")
    lines = [
        f"=== Verifier ablation — {split.upper()} (matched {matched}) ===",
        f"Dataset: {dataset_name}",
        "LLM findings only; non_llm_validators not counted. Agent-level (AEGIS gold has no step).",
        "",
        f"{'Metric':<18}{'none':>9}{'strict':>9}{'soft':>9}",
        "-" * 45,
    ]
    for key, label in METRIC_KEYS:
        row = "".join(_pct(summaries[m].get(key)).rjust(9) for m in MODES)
        lines.append(f"{label:<18}{row}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="AEGIS EvidenceVerifier ablation.")
    parser.add_argument("--pred-glob", default=str(here / "aegis_findings"),
                        help="Directory of AEGIS result JSON files.")
    args = parser.parse_args()

    run(pred_glob=args.pred_glob)
