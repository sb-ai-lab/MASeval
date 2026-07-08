"""Score AEGIS results with the SAME code path as WW (maseval.diagnostic_accuracy).

Each result file written by launch_aegis.py carries:
  * a WW-shaped `report.diagnostic_report` (EvidenceVerifier-gated predictions), and
  * `ground_truth_faulty_agents` (the AEGIS gold culprit agents).

We build an annotation table from the embedded ground truth (keyed on `sample_id`,
the stable global row index — the AEGIS `id` is NOT unique), then call the boary
`evaluate_agent_step_accuracy` so the numbers are directly comparable to WW:
  agent_top1_acc / agent_hit_acc / agent_exact_set_acc.
(AEGIS ground truth has no step index, so step-level metrics come out empty.)

Usage:
    python score_aegis.py                       # scores ./aegis_findings/*.json
    python score_aegis.py --pred aegis_findings --out aegis_score.json
"""

import argparse
import glob
import json
import sys
from pathlib import Path

# Allow running from a checkout without installing the package.
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from maseval.diagnostic_accuracy import evaluate_agent_step_accuracy


def _gt_agent_names(rec: dict) -> list[str]:
    out = []
    for a in rec.get("ground_truth_faulty_agents") or []:
        name = a.get("agent_name") if isinstance(a, dict) else a
        if name:
            out.append(str(name))
    return out


def main(pred_glob: str, out_path: str | None):
    files = sorted(glob.glob(str(Path(pred_glob) / "*.json")))
    if not files:
        raise SystemExit(f"No result files in {pred_glob}")

    # Build an annotation table from the ground truth embedded in each result file.
    # Key on `sample_id` (the stable global row index) — the AEGIS `id` is NOT
    # unique, so matching on it would pair predictions with the wrong ground truth.
    ann_rows = []
    for fp in files:
        rec = json.load(open(fp, encoding="utf-8"))
        ann_rows.append({"sample_id": rec.get("sample_id"), "faulty_agents": _gt_agent_names(rec)})

    ann_path = Path(pred_glob).parent / "aegis_annotations.jsonl"
    with open(ann_path, "w", encoding="utf-8") as f:
        for row in ann_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    result = evaluate_agent_step_accuracy(
        prediction_paths=files,
        annotation_path=str(ann_path),
        id_column="sample_id",
        agent_columns=["faulty_agents"],
        build_missing_report=True,
    )

    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nWrote details -> {out_path}")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Score AEGIS attribution (WW-identical metric)")
    p.add_argument("--pred", default=str(here / "aegis_findings"))
    p.add_argument("--out", default=str(here / "aegis_score.json"))
    args = p.parse_args()
    main(args.pred, args.out)
