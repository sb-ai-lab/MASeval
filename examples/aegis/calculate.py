"""AEGIS agent-attribution metrics, in the style of the AutoJudge AEGIS
``calculate.py`` reference (micro/macro precision/recall/F1 + exact-set accuracy).

This mirrors the reference metric math, but reads the MASeval-shaped result files
written by ``launch_aegis.py`` instead of the AutoJudge ``model_detection`` shape:

  * Predicted culprit agents come from the EvidenceVerifier-gated
    ``report.diagnostic_report.problematic_agents[].agent``.
  * Gold culprit agents come from ``ground_truth_faulty_agents[].agent_name``.

Only the AGENT level is computed. AEGIS labels ``error_type`` with the MAST
failure-mode taxonomy (FM-1.1 … FM-3.3), which the MASeval pipeline does not
predict, so the (agent, error_type) "global" and "error type" levels of the
reference are intentionally omitted.

The ``calculate_metrics`` set-based micro/macro math is a verbatim port of the
reference so the numbers are directly comparable.

``score_aegis.py`` scores the same files through the shared Who&When code path
(``maseval.diagnostic_accuracy``); this file is the standalone reference-style
scorer.

Usage:
    python calculate.py                          # scores ./aegis_findings/*.json
    python calculate.py --file_path aegis_findings
    python calculate.py --file_path results.jsonl
"""

import argparse
import json
import re
from pathlib import Path


def _normalize_agent(name):
    """Match maseval.diagnostic_accuracy._normalize_agent (the repo convention):
    strip non-alphanumerics and lowercase. The MASeval LLM findings re-capitalize
    agent names (e.g. gold ``assistant1`` -> predicted ``Assistant1``), so exact
    matching would silently miss real hits."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def calculate_metrics(true_labels_list, pred_labels_list):
    all_classes = set()
    for labels in true_labels_list:
        all_classes.update(labels)
    for labels in pred_labels_list:
        all_classes.update(labels)
    all_classes = list(all_classes)

    micro_tp = 0
    micro_fp = 0
    micro_fn = 0

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

    micro_precision = (
        micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    )
    micro_recall = (
        micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    )
    micro_f1 = (
        2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    macro_precision_list, macro_recall_list, macro_f1_list = [], [], []
    for cls in all_classes:
        tp = per_class_stats[cls]["tp"]
        fp = per_class_stats[cls]["fp"]
        fn = per_class_stats[cls]["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        macro_precision_list.append(precision)
        macro_recall_list.append(recall)
        macro_f1_list.append(f1)

    macro_precision = (
        sum(macro_precision_list) / len(macro_precision_list)
        if len(macro_precision_list) > 0
        else 0.0
    )
    macro_recall = (
        sum(macro_recall_list) / len(macro_recall_list)
        if len(macro_recall_list) > 0
        else 0.0
    )
    macro_f1 = (
        sum(macro_f1_list) / len(macro_f1_list) if len(macro_f1_list) > 0 else 0.0
    )

    return {
        "precision_micro": micro_precision,
        "recall_micro": micro_recall,
        "f1_micro": micro_f1,
        "precision_macro": macro_precision,
        "recall_macro": macro_recall,
        "f1_macro": macro_f1,
    }


def _gold_agents(data):
    """Gold culprit agent names from AEGIS ground truth."""
    out = []
    for a in data.get("ground_truth_faulty_agents") or []:
        name = a.get("agent_name") if isinstance(a, dict) else a
        if name:
            out.append(str(name))
    return out


def _pred_agents(data):
    """Predicted culprit agents: the EvidenceVerifier-gated ranked list from the
    MASeval report. Falls back to the primary culprit if the list is absent."""
    report = data.get("report") or {}
    diagnostic_report = report.get("diagnostic_report") or {}
    agents = [
        entry.get("agent")
        for entry in diagnostic_report.get("problematic_agents") or []
        if isinstance(entry, dict) and entry.get("agent")
    ]
    if not agents:
        primary = (
            report.get("status", {})
            .get("diagnostic_status", {})
            .get("primary_culprit_agent")
        )
        if primary:
            agents = [primary]
    return [str(a) for a in agents]


def _load_records(filepath):
    filepath = Path(filepath)
    records = []
    if filepath.is_dir():
        for fp in sorted(filepath.glob("*.json")):
            try:
                records.append(json.loads(fp.read_text(encoding="utf-8")))
            except Exception as e:
                print(f"Error reading {fp}: {e}")
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
    return records


def evaluate_results(filepath):
    total_samples = 0
    valid_samples = 0
    agent_exact_matches = 0

    all_true_agents, all_pred_agents = [], []

    for data in _load_records(filepath):
        total_samples += 1

        gold = _gold_agents(data)
        pred = _pred_agents(data)

        valid_samples += 1

        true_agent_set = set(_normalize_agent(a) for a in gold)
        pred_agent_set = set(_normalize_agent(a) for a in pred)

        all_true_agents.append(true_agent_set)
        all_pred_agents.append(pred_agent_set)

        if true_agent_set == pred_agent_set:
            agent_exact_matches += 1

    agent_level_accuracy = (
        agent_exact_matches / valid_samples if valid_samples > 0 else 0.0
    )

    agent_metrics = calculate_metrics(all_true_agents, all_pred_agents)

    results = {
        "Overall": {
            "Total Samples": total_samples,
            "Agent Level Accuracy": agent_level_accuracy,
        },
        "Agent Level": {
            **agent_metrics,
            "accuracy": agent_level_accuracy,
        },
    }

    print("--- AEGIS Agent-Attribution Evaluation ---")
    print(
        "Note: agent level only; AEGIS error_type uses the MAST taxonomy "
        "(FM-*), which the MASeval pipeline does not predict."
    )
    print("Agent-name matching: normalized (lowercased, alnum-only).")
    for section_name, metrics in results.items():
        print(f"\n[{section_name}]")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                print(f"- {metric_name}: {value:.4f}")
            else:
                print(f"- {metric_name}: {value}")

    return results


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Evaluate AEGIS agent attribution")
    parser.add_argument(
        "--file_path",
        type=str,
        default=str(here / "aegis_findings"),
        help="Path to results directory or JSONL file",
    )
    args = parser.parse_args()

    evaluate_results(args.file_path)
