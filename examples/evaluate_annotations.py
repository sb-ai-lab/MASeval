"""Evaluate LLM annotation quality against human ground truth from manifest.csv.

Computes per-category Precision, Recall, F1 (binary classification) at both
L1 and L2 levels, and produces a heatmap of the L2 confusion matrix.

Usage:
    python evaluate_annotations.py \\
        --manifest balanced_traces_1000_after/manifest.csv \\
        --annotations gemini_flash_annotations/ \\
        --output evaluation_results/
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

L2_TO_L1 = {
    "reasoning_planning": "cognitive",
    "hallucination": "cognitive",
    "instruction_following": "cognitive",
    "tool_calling": "interaction",
    "mas_coordination": "interaction",
    "context_state": "procedural",
    "verification_termination": "procedural",
    "environmental": "infrastructure",
    "api_system": "infrastructure",
    "ideal": "ideal",
}

L2_CATEGORIES = [
    "reasoning_planning",
    "hallucination",
    "instruction_following",
    "tool_calling",
    "mas_coordination",
    "context_state",
    "verification_termination",
    "environmental",
    "api_system",
    "ideal",
]

L1_CATEGORIES = ["cognitive", "interaction", "procedural", "infrastructure", "ideal"]

SKIP_CATEGORIES = {"meta"}


def load_manifest(manifest_path: Path) -> dict:
    ground_truth = {}
    with manifest_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fname = os.path.basename(row["dest_file"])
            l1 = {c.strip() for c in row["l1_categories"].split(";") if c.strip()} - SKIP_CATEGORIES
            l2 = {c.strip() for c in row["l2_categories"].split(";") if c.strip()} - SKIP_CATEGORIES
            ground_truth[fname] = {"l1": l1, "l2": l2}
    return ground_truth


def load_predictions(anno_dir: Path) -> dict:
    predictions = {}
    for fn in sorted(anno_dir.glob("*.json")):
        with fn.open("r", encoding="utf-8") as f:
            data = json.load(f)
        error_keys = {e["error_key"] for e in data.get("errors", [])} - SKIP_CATEGORIES
        l1_keys = {L2_TO_L1[k] for k in error_keys if k in L2_TO_L1}
        predictions[fn.name] = {"l2": error_keys, "l1": l1_keys}
    return predictions


def compute_metrics(gt: dict, pred: dict, categories: list) -> dict:
    tp = {c: 0 for c in categories}
    fp = {c: 0 for c in categories}
    fn = {c: 0 for c in categories}
    tn = {c: 0 for c in categories}

    for fname in sorted(gt.keys()):
        truth = gt[fname]
        pred_set = pred.get(fname, set())
        for c in categories:
            in_truth = c in truth
            in_pred = c in pred_set
            if in_truth and in_pred:
                tp[c] += 1
            elif not in_truth and in_pred:
                fp[c] += 1
            elif in_truth and not in_pred:
                fn[c] += 1
            else:
                tn[c] += 1

    results = {}
    for c in categories:
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        recall = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results[c] = {
            "TP": tp[c], "FP": fp[c], "FN": fn[c], "TN": tn[c],
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
        }
    return results


def compute_confusion_matrix(gt: dict, pred: dict, categories: list) -> np.ndarray:
    n = len(categories)
    cat_idx = {c: i for i, c in enumerate(categories)}
    matrix = np.zeros((n, n), dtype=int)
    for fname in sorted(gt.keys()):
        truth = gt[fname]
        pred_set = pred.get(fname, set())
        for tc in truth & set(categories):
            for pc in pred_set & set(categories):
                matrix[cat_idx[tc], cat_idx[pc]] += 1
    return matrix


def macro_avg(results: dict) -> dict:
    metrics = ["Precision", "Recall", "F1"]
    avg = {}
    for m in metrics:
        vals = [v[m] for v in results.values()]
        avg[m] = round(sum(vals) / len(vals), 4) if vals else 0.0
    return avg


def micro_avg(results: dict) -> dict:
    tp_sum = sum(v["TP"] for v in results.values())
    fp_sum = sum(v["FP"] for v in results.values())
    fn_sum = sum(v["FN"] for v in results.values())
    precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"Precision": round(precision, 4), "Recall": round(recall, 4), "F1": round(f1, 4)}


def print_table(label: str, categories: list, results: dict):
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")
    header = f"{'Category':<25} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}  {'Prec':>6} {'Rec':>6} {'F1':>6}"
    print(header)
    print("-" * 80)
    for c in categories:
        r = results[c]
        print(f"{c:<25} {r['TP']:>5} {r['FP']:>5} {r['FN']:>5} {r['TN']:>5}  {r['Precision']:>6.4f} {r['Recall']:>6.4f} {r['F1']:>6.4f}")
    print("-" * 80)
    ma = macro_avg(results)
    mi = micro_avg(results)
    print(f"{'Macro avg':<25} {'':>5} {'':>5} {'':>5} {'':>5}  {ma['Precision']:>6.4f} {ma['Recall']:>6.4f} {ma['F1']:>6.4f}")
    print(f"{'Micro avg':<25} {'':>5} {'':>5} {'':>5} {'':>5}  {mi['Precision']:>6.4f} {mi['Recall']:>6.4f} {mi['F1']:>6.4f}")


def plot_heatmap(matrix: np.ndarray, categories: list, output_path: Path):
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm = matrix.astype(float) / row_sums

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(categories)))
    ax.set_yticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(categories, fontsize=9)
    ax.set_xlabel("Predicted (LLM)", fontsize=11)
    ax.set_ylabel("True (Human)", fontsize=11)
    ax.set_title("L2 Category Confusion Matrix (row-normalized)", fontsize=13)

    for i in range(len(categories)):
        for j in range(len(categories)):
            val = matrix[i, j]
            pct = norm[i, j]
            color = "white" if pct > 0.5 else "black"
            ax.text(j, i, f"{val}\n({pct:.0%})", ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Heatmap saved to {output_path}")


def analyze_false_positives_on_ideals(
    manifest_path: Path, anno_dir: Path, output_dir: Path
):
    ground_truth = load_manifest(manifest_path)
    predictions = load_predictions(anno_dir)

    ideal_traces = {fn for fn, cats in ground_truth.items() if cats["l2"] == {"ideal"}}

    total_ideal = len(ideal_traces)
    annotated_ideal = 0
    annotated_with_errors = 0
    fp_error_counts = Counter()
    fp_error_keys = Counter()
    by_dataset = defaultdict(lambda: {"total": 0, "correct_ideal": 0, "false_positive": 0})

    with manifest_path.open("r", encoding="utf-8") as f:
        rows_by_fname = {}
        for row in csv.DictReader(f):
            rows_by_fname[os.path.basename(row["dest_file"])] = row

    for fn in sorted(ideal_traces):
        ds = rows_by_fname.get(fn, {}).get("dataset", "unknown")
        by_dataset[ds]["total"] += 1

        if fn not in predictions:
            continue

        keys = predictions[fn]["l2"]
        n_errors = len(keys - {"ideal"})

        if len(keys) == 1 and "ideal" in keys:
            annotated_ideal += 1
            by_dataset[ds]["correct_ideal"] += 1
        else:
            annotated_with_errors += 1
            by_dataset[ds]["false_positive"] += 1
            for k in keys:
                if k != "ideal":
                    fp_error_keys[k] += 1

        fp_error_counts[n_errors] += 1

    print(f"\n{'='*80}")
    print(f"  False-Positive Analysis: LLM on Ideal (error-free) Traces")
    print(f"{'='*80}")
    print(f"Total ideal traces (human label = 'ideal' only): {total_ideal}")
    print(f"Annotated by LLM:")
    print(f"  Correctly as 'ideal':                                          {annotated_ideal}")
    print(f"  Incorrectly with errors (false-positive):                      {annotated_with_errors}")
    available = annotated_ideal + annotated_with_errors
    if available > 0:
        print(f"  False-positive rate on ideal traces:                           {annotated_with_errors/available:.1%}")
    print(f"\nFalse-positive error keys (what the LLM sees where nothing is wrong):")
    for k, c in fp_error_keys.most_common():
        print(f"  {k:<30s} {c:>5d}")

    print(f"\nNumber of LLM-predicted errors per ideal trace:")
    for n in sorted(fp_error_counts.keys()):
        label = "0 (predicted ideal)" if n == 0 else str(n)
        print(f"  {label:>30s}: {fp_error_counts[n]:>5d} traces")

    print(f"\nFalse-positive rate by dataset:")
    for ds in sorted(by_dataset.keys()):
        d = by_dataset[ds]
        total = d["total"]
        correct = d["correct_ideal"]
        fp = d["false_positive"]
        matched = correct + fp
        rate = fp / matched if matched > 0 else 0
        print(f"  {ds:<20s}: {correct:>3d} correct / {fp:>3d} false-positive / {total:>3d} total  (FP rate: {rate:.0%})")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    labels = list(k for k, _ in fp_error_keys.most_common())
    values = list(v for _, v in fp_error_keys.most_common())
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    axes[0].barh(labels[::-1], values[::-1], color=colors[::-1])
    axes[0].set_xlabel("Count")
    axes[0].set_title("False-Positive Error Keys\non Ideal Traces")

    x_labels = [str(n) if n > 0 else "0 (ideal)" for n in sorted(fp_error_counts.keys())]
    x_values = [fp_error_counts[n] for n in sorted(fp_error_counts.keys())]
    axes[1].bar(x_labels, x_values, color="salmon")
    axes[1].set_xlabel("Number of Errors Predicted")
    axes[1].set_ylabel("Number of Ideal Traces")
    axes[1].set_title("Errors Predicted per Ideal Trace")

    plt.tight_layout()
    fig.savefig(output_dir / "ideal_fp_analysis.png", dpi=150)
    plt.close(fig)

    result = {
        "total_ideal": total_ideal,
        "correct_ideal": annotated_ideal,
        "false_positive": annotated_with_errors,
        "false_positive_rate": round(annotated_with_errors / available, 4) if available > 0 else None,
        "fp_error_keys": dict(fp_error_keys.most_common()),
        "fp_error_counts": {str(k): v for k, v in sorted(fp_error_counts.items())},
        "by_dataset": {ds: dict(d) for ds, d in by_dataset.items()},
    }
    with (output_dir / "ideal_fp_analysis.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nChart saved to {output_dir / 'ideal_fp_analysis.png'}")
    print(f"JSON saved to {output_dir / 'ideal_fp_analysis.json'}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM annotations against human ground truth")
    parser.add_argument("--manifest", default="balanced_traces_1000_after/manifest.csv")
    parser.add_argument("--annotations", default="gemini_flash_annotations")
    parser.add_argument("--output", default="evaluation_results")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    anno_dir = Path(args.annotations)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = load_manifest(manifest_path)
    predictions = load_predictions(anno_dir)

    matched = set(ground_truth.keys()) & set(predictions.keys())
    print(f"Manifest traces: {len(ground_truth)}")
    print(f"Annotated traces: {len(predictions)}")
    print(f"Matched: {len(matched)}")

    missing = set(ground_truth.keys()) - set(predictions.keys())
    if missing:
        print(f"Missing annotations: {len(missing)}")

    gt_l2 = {k: ground_truth[k]["l2"] for k in matched}
    pr_l2 = {k: predictions[k]["l2"] for k in matched}
    gt_l1 = {k: ground_truth[k]["l1"] for k in matched}
    pr_l1 = {k: predictions[k]["l1"] for k in matched}

    l2_results = compute_metrics(gt_l2, pr_l2, L2_CATEGORIES)
    l1_results = compute_metrics(gt_l1, pr_l1, L1_CATEGORIES)

    print_table("L2 Category Metrics (fine-grained error types)", L2_CATEGORIES, l2_results)
    print_table("L1 Category Metrics (high-level groups)", L1_CATEGORIES, l1_results)

    matrix = compute_confusion_matrix(gt_l2, pr_l2, L2_CATEGORIES)
    plot_heatmap(matrix, L2_CATEGORIES, output_dir / "l2_confusion_heatmap.png")

    analyze_false_positives_on_ideals(manifest_path, anno_dir, output_dir)

    with (output_dir / "l2_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(l2_results, f, indent=2, ensure_ascii=False)
    with (output_dir / "l1_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(l1_results, f, indent=2, ensure_ascii=False)

    print(f"\nJSON results saved to {output_dir}/")


if __name__ == "__main__":
    main()