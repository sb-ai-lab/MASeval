"""Score AEGIS agent localization for verifier=``none``, folding in the
deterministic ``non_llm_validators`` (+ LLM confirmation) — the AEGIS analogue of
the Who&When / TraceElephant ``score_none_plus_validators``.

AEGIS gold is **agent-only** (``ground_truth_faulty_agents``; no step index), so:

* the reported metrics are agent Top-1 / Hit / Exact-Set **and** the set-based
  micro/macro precision/recall/F1 (the AutoJudge AEGIS reference math, reused from
  ``verifier_ablation.calculate_metrics``);
* the confirmer's only useful job is **benign-pruning** — appointing
  (``corrected_idx``) is irrelevant with no step gold, so there is no locus choice.

Each validator finding already names its ``culprit_agent`` (no idx→agent map
needed, unlike TraceElephant). We OR that agent into the verifier=``none`` agent
set under three verdict filters:

* ``all``       -- every regex finding (no confirmer gating);
* ``conf+unc``  -- confirmer verdict confirmed OR uncertain;
* ``confirmed`` -- confirmed only (**the honest column** — folding in guesses
                   raises recall but costs precision; the confirmed filter is the
                   precision protection, so F1 is the honest arbiter here).

Only 49/600 traces are finding-bearing, so the fold-in is invisible in the 600-trace
overall; we therefore also report the **bearing-only** subset, where the signal lives.

Usage:
    python score_none_plus_validators.py
    python score_none_plus_validators.py --pred-dir aegis_findings_confirm
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
for p in (ROOT / "src", THIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from maseval.diagnostic_accuracy import _normalize_agent, read_prediction_file  # noqa: E402

from verifier_ablation import calculate_metrics  # noqa: E402  (AEGIS reference P/R/F1)

FILTERS = ("none_only", "all", "conf_unc", "confirmed")
FILTER_LABELS = {
    "none_only": "none (LLM only)",
    "all": "none + val (all)",
    "conf_unc": "none + val (conf+unc)",
    "confirmed": "none + val (confirmed)",
}


def _gt_agents(rec: dict) -> list[str]:
    out = []
    for a in rec.get("ground_truth_faulty_agents") or []:
        name = a.get("agent_name") if isinstance(a, dict) else a
        if name:
            out.append(str(name))
    return out


def _iter_val_findings(nlv: dict):
    for m in (nlv.get("metrics") or {}).values():
        for f in m.get("findings", []):
            yield f


def _val_agents(finding: dict) -> list[str]:
    """The agent(s) a validator finding blames: its ``culprit_agent`` plus any
    per-idx evidence agents (deduped, order preserved)."""
    out = []
    c = finding.get("culprit_agent")
    if c:
        out.append(str(c))
    for ev in finding.get("evidence") or []:
        a = ev.get("agent")
        if a and str(a) not in out:
            out.append(str(a))
    return out


def _passes(finding: dict, filt: str) -> bool:
    if filt == "all":
        return True
    verdict = (finding.get("llm_confirmation") or {}).get("verdict")
    if filt == "conf_unc":
        return verdict in ("confirmed", "uncertain")
    if filt == "confirmed":
        return verdict == "confirmed"
    return False


def score(pred_dir: str) -> dict:
    pred_dir_p = THIS_DIR / pred_dir if not Path(pred_dir).is_absolute() else Path(pred_dir)
    files = sorted(glob.glob(str(pred_dir_p / "*.json")))
    if not files:
        raise FileNotFoundError(f"No prediction files in {pred_dir_p}")

    # Per-filter accumulators: agent bool lists + (gold,pred) sets for P/R/F1.
    acc = {f: {"top1": [], "hit": [], "exact": [], "gold_sets": [], "pred_sets": []}
           for f in FILTERS}
    acc_bearing = {f: {"top1": [], "hit": [], "exact": [], "gold_sets": [], "pred_sets": []}
                   for f in FILTERS}
    matched = 0
    bearing = 0
    counts = {"confirmed": 0, "benign": 0, "uncertain": 0}

    for fp in files:
        data = json.loads(Path(fp).read_text(encoding="utf-8"))
        gold = {_normalize_agent(a) for a in _gt_agents(data)}
        if not gold:
            continue
        matched += 1

        pred = read_prediction_file(fp, verifier_mode="none")
        base_agents = list(pred.agents)
        base_primary = pred.primary_agent

        nlv = data.get("non_llm_validators") or {}
        val_hits = list(_iter_val_findings(nlv))
        is_bearing = bool(val_hits)
        if is_bearing:
            bearing += 1
            s = nlv.get("llm_confirmation_summary") or {}
            for k in ("confirmed", "benign", "uncertain"):
                counts[k] += int(s.get(k, 0) or 0)

        for filt in FILTERS:
            agents = list(base_agents)
            primary = base_primary
            if filt != "none_only":
                for finding in val_hits:
                    if not _passes(finding, filt):
                        continue
                    for a in _val_agents(finding):
                        if a not in agents:
                            agents.append(a)
                        if primary is None:
                            primary = a

            pred_set = {_normalize_agent(a) for a in agents}
            primary_norm = _normalize_agent(primary) if primary else None
            top1 = bool(primary_norm and primary_norm in gold)
            hit = any(a in gold for a in pred_set)
            exact = pred_set == gold
            for store, active in ((acc, True), (acc_bearing, is_bearing)):
                if not active:
                    continue
                store[filt]["top1"].append(top1)
                store[filt]["hit"].append(hit)
                store[filt]["exact"].append(exact)
                store[filt]["gold_sets"].append(gold)
                store[filt]["pred_sets"].append(pred_set)

    def summarize(store: dict) -> dict:
        out = {}
        for f in FILTERS:
            d = store[f]
            n = len(d["top1"])
            prf = calculate_metrics(d["gold_sets"], d["pred_sets"]) if n else {}
            out[f] = {
                "agent_top1": (sum(d["top1"]) / n) if n else None,
                "agent_hit": (sum(d["hit"]) / n) if n else None,
                "agent_exact": (sum(d["exact"]) / n) if n else None,
                "precision_micro": prf.get("precision_micro"),
                "recall_micro": prf.get("recall_micro"),
                "f1_micro": prf.get("f1_micro"),
                "f1_macro": prf.get("f1_macro"),
            }
        return out

    return {
        "matched": matched,
        "bearing": bearing,
        "counts": counts,
        "overall": summarize(acc),
        "bearing_only": summarize(acc_bearing),
    }


def _pct(v):
    return f"{v * 100:5.1f}%" if isinstance(v, (int, float)) else "    —"


ROWS = [
    ("Agent Top-1", "agent_top1"), ("Agent Hit", "agent_hit"), ("Agent Exact-Set", "agent_exact"),
    ("Precision (micro)", "precision_micro"), ("Recall (micro)", "recall_micro"),
    ("F1 (micro)", "f1_micro"), ("F1 (macro)", "f1_macro"),
]


def _render(title: str, summ: dict, n: int) -> str:
    header = f"{'Metric':<18}" + "".join(FILTER_LABELS[f].rjust(22) for f in FILTERS)
    lines = [f"=== {title} (n={n}) ===", "", header, "-" * len(header)]
    for label, key in ROWS:
        lines.append(f"{label:<18}" + "".join(_pct(summ[f][key]).rjust(22) for f in FILTERS))
    return "\n".join(lines)


def _markdown(res: dict, pred_dir: str) -> str:
    md = [
        "# AEGIS — none + non_llm_validators (LLM-confirm) fold-in",
        "",
        f"- Predictions: `{pred_dir}`",
        f"- Matched (has gold): {res['matched']}; finding-bearing: {res['bearing']}",
        f"- Confirmer: confirmed={res['counts']['confirmed']}, "
        f"uncertain={res['counts']['uncertain']}, benign={res['counts']['benign']}",
        "- Baseline: verifier=`none` over the 11 LLM evaluators.",
        "- Fold-in agent = validator `culprit_agent` (+ per-idx evidence agents). "
        "AEGIS gold is agent-only, so no step metrics and no locus/appointing.",
        "- Filters: `all` = every regex finding; `conf+unc` = confirmed|uncertain; "
        "`confirmed` = confirmed only. **F1 is the honest arbiter** (fold-in trades "
        "precision for recall; the confirmed filter protects precision).",
        "",
    ]
    for title, key in (("Overall (all traces with gold)", "overall"),
                       ("Bearing-only (traces with ≥1 validator finding)", "bearing_only")):
        n = res["matched"] if key == "overall" else res["bearing"]
        md.append(f"## {title} (n={n})")
        md.append("")
        md.append("| Metric | " + " | ".join(FILTER_LABELS[f] for f in FILTERS) + " |")
        md.append("|---|" + "---:|" * len(FILTERS))
        s = res[key]
        for label, mk in ROWS:
            md.append("| " + label + " | " + " | ".join(_pct(s[f][mk]).strip() for f in FILTERS) + " |")
        md.append("")
    return "\n".join(md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AEGIS none + validators (llm-confirm) fold-in.")
    parser.add_argument("--pred-dir", default="aegis_findings_confirm")
    args = parser.parse_args()
    res = score(args.pred_dir)

    print(_render("AEGIS — Overall (all traces with gold)", res["overall"], res["matched"]))
    print()
    print(f"confirmer: confirmed={res['counts']['confirmed']} "
          f"uncertain={res['counts']['uncertain']} benign={res['counts']['benign']}")
    print()
    print(_render("AEGIS — Bearing-only", res["bearing_only"], res["bearing"]))

    reports_dir = THIS_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "none_plus_validators.json").write_text(
        json.dumps({"pred_dir": args.pred_dir, "filter_labels": FILTER_LABELS, **res},
                   indent=2), encoding="utf-8")
    (reports_dir / "none_plus_validators.md").write_text(
        _markdown(res, args.pred_dir), encoding="utf-8")
    print(f"\nSaved: {reports_dir / 'none_plus_validators.md'}")
