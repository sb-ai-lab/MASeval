"""Score Who&When agent/step localization for verifier=none, optionally folding
in the deterministic ``non_llm_validators`` (with the LLM confirmation layer).

Baseline is the verifier=``none`` prediction over the 11 LLM evaluators
(``diagnostic_accuracy.read_prediction_file(verifier_mode="none")``). On top of
that we OR-in the deterministic validator findings, under three verdict filters:

* ``all``       -- every regex-detected finding counts (no confirmer gating);
* ``conf+unc``  -- findings the confirmer marked confirmed OR uncertain;
* ``confirmed`` -- only findings the confirmer marked confirmed.

For each counted validator finding the predicted locus is the **appointed causal
turn** (``llm_confirmation.corrected_idx``), falling back to the finding's
evidence/surface idx when appointing returned null. The predicted AGENT is the
agent of the span at that idx, resolved from the trace history via
``who_and_when_to_spans`` (the finding itself carries no agent name). The extra
agent/idx are appended to the LLM ``none`` prediction (dedup, order preserved),
so this can only ADD recall, never remove an LLM prediction.

Prints a comparison table: none (LLM only) vs none+validators under each filter.

Usage:
    python score_none_plus_validators.py \
        --pred-dir who&when_hand_gemini_llmconfirm_scratch_fixed \
        --parquet Hand-Crafted.parquet
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

from maseval.diagnostic_accuracy import (  # noqa: E402
    _idx_matches_any,
    _normalize_agent,
    read_prediction_file,
)
from maseval.validators.base import who_and_when_to_spans  # noqa: E402

from apply_llm_confirm import _coerce_history_to_steps, _load_parquet  # noqa: E402

FILTERS = ("none_only", "all", "conf_unc", "confirmed")
FILTER_LABELS = {
    "none_only": "none (LLM only)",
    "all": "none + val (all)",
    "conf_unc": "none + val (conf+unc)",
    "confirmed": "none + val (confirmed)",
}


def _iter_val_findings(nlv: dict):
    for m in (nlv.get("metrics") or {}).values():
        for f in m.get("findings", []):
            yield f


def _val_locus_idx(finding: dict) -> str | None:
    """Appointed causal turn if present, else the evidence/surface idx."""
    conf = finding.get("llm_confirmation") or {}
    corrected = conf.get("corrected_idx")
    if corrected is not None and str(corrected).strip():
        return str(corrected)
    ev = finding.get("evidence") or []
    if ev and ev[0].get("idx") is not None:
        return str(ev[0]["idx"])
    return None


def _passes(finding: dict, filt: str) -> bool:
    if filt == "all":
        return True
    verdict = (finding.get("llm_confirmation") or {}).get("verdict")
    if filt == "conf_unc":
        return verdict in ("confirmed", "uncertain")
    if filt == "confirmed":
        return verdict == "confirmed"
    return False


def _extend(agents: list[str], idxs: list[str], a: str | None, ix: str | None):
    if ix is not None and ix not in idxs:
        idxs.append(ix)
    if a and a not in agents:
        agents.append(a)


def score(pred_dir: str, parquet: str, step_tolerance: int = 1) -> dict:
    pred_dir_p = THIS_DIR / pred_dir if not Path(pred_dir).is_absolute() else Path(pred_dir)
    files = sorted(
        glob.glob(str(pred_dir_p / "gemini_findings_*.json")),
        key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]),
    )
    if not files:
        raise FileNotFoundError(f"No prediction files in {pred_dir_p}")
    df = _load_parquet(parquet)

    # Accumulators per filter.
    acc = {f: {"agent_top1": [], "agent_hit": [], "step_top1": [], "step_hit": [], "step_pm1": []}
           for f in FILTERS}
    matched = 0

    for fp in files:
        i = int(fp.rsplit("_", 1)[1].split(".")[0])
        row = df.iloc[i]
        gold_agents = {_normalize_agent(a) for a in _as_list(row.get("mistake_agent"))}
        gold_idxs = [str(x) for x in _as_list(row.get("mistake_step"))]
        if not gold_agents and not gold_idxs:
            continue
        matched += 1

        # Baseline: verifier=none over the LLM evaluators.
        pred = read_prediction_file(fp, verifier_mode="none")
        base_agents = list(pred.agents)
        base_idxs = list(pred.idxs)
        base_primary = pred.primary_agent
        base_first = pred.first_idx

        # idx -> agent map from the (properly expanded) trace history.
        steps = _coerce_history_to_steps(row["history"])
        spans = who_and_when_to_spans({"history": steps})
        idx_to_agent = {s["idx"]: s.get("agent") for s in spans}

        data = json.loads(Path(fp).read_text(encoding="utf-8"))
        nlv = data.get("non_llm_validators") or {}
        val_hits = list(_iter_val_findings(nlv))

        for filt in FILTERS:
            agents = list(base_agents)
            idxs = list(base_idxs)
            primary = base_primary
            first = base_first
            if filt != "none_only":
                for finding in val_hits:
                    if not _passes(finding, filt):
                        continue
                    ix = _val_locus_idx(finding)
                    a = idx_to_agent.get(ix) if ix is not None else None
                    _extend(agents, idxs, a, ix)
                    if primary is None and a:
                        primary = a
                    if first is None and ix is not None:
                        first = ix

            pred_agents_norm = [_normalize_agent(a) for a in agents]
            primary_norm = _normalize_agent(primary) if primary else None
            if gold_agents:
                acc[filt]["agent_top1"].append(bool(primary_norm and primary_norm in gold_agents))
                acc[filt]["agent_hit"].append(any(a in gold_agents for a in pred_agents_norm))
            if gold_idxs:
                acc[filt]["step_top1"].append(
                    bool(first and _idx_matches_any(first, gold_idxs, tolerance=0)))
                acc[filt]["step_hit"].append(
                    any(_idx_matches_any(ix, gold_idxs, tolerance=0) for ix in idxs))
                acc[filt]["step_pm1"].append(
                    any(_idx_matches_any(ix, gold_idxs, tolerance=step_tolerance) for ix in idxs))

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    summary = {f: {k: mean(v) for k, v in acc[f].items()} for f in FILTERS}
    return {"matched": matched, "summary": summary}


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [x for x in value if x is not None and str(x).strip()]
    if hasattr(value, "tolist"):
        return [x for x in value.tolist() if x is not None and str(x).strip()]
    text = str(value).strip()
    return [text] if text and text.lower() not in {"none", "nan", "null", ""} else []


def _pct(v):
    return f"{v * 100:5.1f}%" if isinstance(v, (int, float)) else "    —"


def _render(res: dict, split: str = "") -> str:
    s = res["summary"]
    rows = [
        ("Agent Top-1", "agent_top1"),
        ("Agent Hit", "agent_hit"),
        ("Step Top-1", "step_top1"),
        ("Step Hit", "step_hit"),
        ("Step Hit ±1", "step_pm1"),
    ]
    header = f"{'Metric':<14}" + "".join(FILTER_LABELS[f].rjust(22) for f in FILTERS)
    lines = [
        f"=== none + non_llm_validators (llm-confirm), {split.upper() or '?'} (matched {res['matched']}) ===",
        "Locus = appointed causal turn (corrected_idx), fallback surface idx; "
        "agent = agent at that idx.",
        "",
        header,
        "-" * len(header),
    ]
    for label, key in rows:
        lines.append(f"{label:<14}" + "".join(_pct(s[f][key]).rjust(22) for f in FILTERS))
    return "\n".join(lines)


def _markdown(res: dict, pred_dir: str, parquet: str) -> str:
    s = res["summary"]
    rows = [
        ("Agent Top-1", "agent_top1"), ("Agent Hit", "agent_hit"),
        ("Step Top-1", "step_top1"), ("Step Hit", "step_hit"), ("Step Hit ±1", "step_pm1"),
    ]
    md = [
        "# Who&When — none + non_llm_validators (LLM-confirm) fold-in",
        "",
        f"- Split / annotations: `{parquet}`",
        f"- Predictions: `{pred_dir}`",
        f"- Matched traces: {res['matched']}",
        "- Baseline: verifier=`none` over the 11 LLM evaluators.",
        "- Fold-in locus: appointed causal turn (`corrected_idx`), fallback surface idx; "
        "agent = agent at that idx (via `who_and_when_to_spans`).",
        "- Filters: `all` = every regex finding; `conf+unc` = confirmed|uncertain; "
        "`confirmed` = confirmed only.",
        "",
        "| Metric | " + " | ".join(FILTER_LABELS[f] for f in FILTERS) + " |",
        "|---|" + "---:|" * len(FILTERS),
    ]
    for label, key in rows:
        md.append("| " + label + " | " + " | ".join(_pct(s[f][key]).strip() for f in FILTERS) + " |")
    md.append("")
    return "\n".join(md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score none + validators (llm-confirm) fold-in.")
    parser.add_argument("--pred-dir", default="who&when_hand_gemini_llmconfirm_scratch_fixed")
    parser.add_argument("--parquet", default="Hand-Crafted.parquet")
    parser.add_argument("--step-tolerance", type=int, default=1)
    parser.add_argument("--split", default="hc", help="Label for the output report filename.")
    args = parser.parse_args()
    res = score(args.pred_dir, args.parquet, step_tolerance=args.step_tolerance)
    print(_render(res, args.split))

    reports_dir = THIS_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"none_plus_validators_{args.split}.json"
    md_path = reports_dir / f"none_plus_validators_{args.split}.md"
    json_path.write_text(json.dumps(
        {"pred_dir": args.pred_dir, "parquet": args.parquet,
         "filter_labels": FILTER_LABELS, **res}, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(res, args.pred_dir, args.parquet), encoding="utf-8")
    print(f"\nSaved: {md_path}")
    print(f"Saved: {json_path}")
