"""Score TRAIL error-location localization for verifier=``none``, folding in the
deterministic ``non_llm_validators`` (+ LLM confirmation) — the TRAIL analogue of
the Who&When / TraceElephant / AEGIS ``score_none_plus_validators``.

TRAIL gold is a **span-hash location** (``labels.errors[].location``), with no
agent field (GAIA is effectively single-agent; the "agent" on a finding is a step
label). So this scores **location** localization only — the mirror image of AEGIS
(agent-only).

**The namespace bridge (the crux).** ``trail_to_spans`` gives every span a native
hex ``span_id``; gold locations and the *validators*' evidence idx both live in that
hash space. But the LLM *judges* cite numeric span **positions** (``[0]``, ``[1]``,
… from ``_format_indexed_trail_trace``, which enumerates the same span list). So we
build ``pos2hash = {str(i): span["idx"]}`` from the trace and map every judge idx
(and ``first_idx``) through it; validator idxs need no mapping. Everything is then
compared in the one hash space. The GAIA raw trace is reloaded by ``trace_id`` for
this map (and it is the same trace the confirmer read).

Validator locus, two variants:

* ``surface``   -- the finding's evidence idx (a span_id);
* ``appointed`` -- the confirmer's ``corrected_idx`` (a span_id), fallback surface.

Filters: ``all`` / ``conf+unc`` / ``confirmed`` (confirmed = the honest column).
Reports overall (all traces with gold) + bearing-only (traces with ≥1 validator
finding), where the signal lives.

Usage:
    python score_none_plus_validators.py
    python score_none_plus_validators.py --locus surface
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

from maseval.diagnostic_accuracy import read_prediction_file  # noqa: E402
from maseval.validators.base import trail_to_spans  # noqa: E402

DEFAULT_GAIA = "/mnt/c/Users/barak/Downloads/trail-benchmark-main/benchmarking/data/GAIA"

FILTERS = ("none_only", "all", "conf_unc", "confirmed")
FILTER_LABELS = {
    "none_only": "none (LLM only)",
    "all": "none + val (all)",
    "conf_unc": "none + val (conf+unc)",
    "confirmed": "none + val (confirmed)",
}
METRICS = ("step_top1", "step_hit")


def _gold_locations(data: dict) -> set[str]:
    return {e["location"] for e in (data.get("labels") or {}).get("errors", []) if e.get("location")}


def _iter_val_findings(nlv: dict):
    for m in (nlv.get("metrics") or {}).values():
        for f in m.get("findings", []):
            yield f


def _val_locus_hash(finding: dict, locus_mode: str) -> str | None:
    """Span-hash locus. ``appointed`` = corrected_idx (fallback surface); ``surface`` = evidence idx."""
    if locus_mode == "appointed":
        c = (finding.get("llm_confirmation") or {}).get("corrected_idx")
        if c is not None and str(c).strip():
            return str(c)
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


def score(pred_dir: str, gaia_dir: str, locus_mode: str) -> dict:
    pred_dir_p = THIS_DIR / pred_dir if not Path(pred_dir).is_absolute() else Path(pred_dir)
    files = sorted(glob.glob(str(pred_dir_p / "*.json")))
    if not files:
        raise FileNotFoundError(f"No prediction files in {pred_dir_p}")
    gaia = Path(gaia_dir)

    acc = {f: {k: [] for k in METRICS} for f in FILTERS}
    acc_bearing = {f: {k: [] for k in METRICS} for f in FILTERS}
    matched = bearing = 0
    counts = {"confirmed": 0, "benign": 0, "uncertain": 0, "appointed": 0}

    for fp in files:
        data = json.loads(Path(fp).read_text(encoding="utf-8"))
        gold = _gold_locations(data)
        if not gold:
            continue
        gpath = gaia / f"{data['trace_id']}.json"
        if not gpath.exists():
            continue
        matched += 1
        spans = trail_to_spans(json.loads(gpath.read_text(encoding="utf-8")))
        pos2hash = {str(i): s["idx"] for i, s in enumerate(spans)}

        pred = read_prediction_file(fp, verifier_mode="none")
        base_idxs = [pos2hash[i] for i in pred.idxs if pos2hash.get(i)]      # judge pos -> hash
        base_first = pos2hash.get(str(pred.first_idx)) if pred.first_idx else None

        nlv = data.get("non_llm_validators") or {}
        val_hits = list(_iter_val_findings(nlv))
        is_bearing = bool(val_hits)
        if is_bearing:
            bearing += 1
            s = nlv.get("llm_confirmation_summary") or {}
            for k in counts:
                counts[k] += int(s.get(k, 0) or 0)

        for filt in FILTERS:
            idxs = list(base_idxs)
            first = base_first
            if filt != "none_only":
                for finding in val_hits:
                    if not _passes(finding, filt):
                        continue
                    h = _val_locus_hash(finding, locus_mode)
                    if h is not None and h not in idxs:
                        idxs.append(h)
                    if first is None and h is not None:
                        first = h

            top1 = bool(first and first in gold)
            hit = any(h in gold for h in idxs)
            for store, active in ((acc, True), (acc_bearing, is_bearing)):
                if active:
                    store[filt]["step_top1"].append(top1)
                    store[filt]["step_hit"].append(hit)

    def summarize(store):
        return {f: {k: (sum(v) / len(v) if v else None) for k, v in store[f].items()} for f in FILTERS}

    return {
        "matched": matched, "bearing": bearing, "counts": counts,
        "overall": summarize(acc), "bearing_only": summarize(acc_bearing),
    }


def _pct(v):
    return f"{v * 100:5.1f}%" if isinstance(v, (int, float)) else "    —"


ROWS = [("Step Top-1", "step_top1"), ("Step Hit", "step_hit")]


def _render(title: str, summ: dict, n: int, locus: str) -> str:
    header = f"{'Metric':<12}" + "".join(FILTER_LABELS[f].rjust(22) for f in FILTERS)
    lines = [f"=== {title} — locus={locus} (n={n}) ===", "", header, "-" * len(header)]
    for label, key in ROWS:
        lines.append(f"{label:<12}" + "".join(_pct(summ[f][key]).rjust(22) for f in FILTERS))
    return "\n".join(lines)


def _markdown(all_res: dict, locus_modes: list[str], pred_dir: str) -> str:
    any_res = all_res[locus_modes[0]]
    md = [
        "# TRAIL — none + non_llm_validators (LLM-confirm) fold-in",
        "",
        f"- Predictions: `{pred_dir}`",
        f"- Matched (gold + GAIA trace): {any_res['matched']}; finding-bearing: {any_res['bearing']}",
        f"- Confirmer: confirmed={any_res['counts']['confirmed']}, "
        f"uncertain={any_res['counts']['uncertain']}, benign={any_res['counts']['benign']}, "
        f"appointed={any_res['counts']['appointed']}",
        "- Gold = span-hash `location`; no agent gold (location-only, mirror of AEGIS).",
        "- Judge idxs are numeric span positions, mapped to span-hashes via "
        "`trail_to_spans`; validator/gold idxs are span-hashes directly.",
        "- Filters: `all` = every regex finding; `conf+unc` = confirmed|uncertain; "
        "`confirmed` = confirmed only (**the honest column**).",
        "",
    ]
    for locus in locus_modes:
        res = all_res[locus]
        md.append(f"## Locus = {locus}")
        md.append("")
        for title, key in (("Overall (traces with gold)", "overall"),
                           ("Bearing-only (≥1 validator finding)", "bearing_only")):
            n = res["matched"] if key == "overall" else res["bearing"]
            md.append(f"### {title} (n={n})")
            md.append("")
            md.append("| Metric | " + " | ".join(FILTER_LABELS[f] for f in FILTERS) + " |")
            md.append("|---|" + "---:|" * len(FILTERS))
            s = res[key]
            for label, mk in ROWS:
                md.append("| " + label + " | " + " | ".join(_pct(s[f][mk]).strip() for f in FILTERS) + " |")
            md.append("")
    return "\n".join(md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRAIL none + validators (llm-confirm) fold-in.")
    parser.add_argument("--pred-dir", default="trail_gemini_findings_v1_confirm")
    parser.add_argument("--gaia-dir", default=DEFAULT_GAIA)
    parser.add_argument("--locus", choices=("appointed", "surface", "both"), default="both")
    args = parser.parse_args()

    locus_modes = ["appointed", "surface"] if args.locus == "both" else [args.locus]
    all_res = {lm: score(args.pred_dir, args.gaia_dir, lm) for lm in locus_modes}

    for lm in locus_modes:
        res = all_res[lm]
        print(_render("TRAIL — Overall", res["overall"], res["matched"], lm))
        print()
        print(_render("TRAIL — Bearing-only", res["bearing_only"], res["bearing"], lm))
        print(f"  confirmer: confirmed={res['counts']['confirmed']} "
              f"uncertain={res['counts']['uncertain']} benign={res['counts']['benign']} "
              f"appointed={res['counts']['appointed']}")
        print()

    reports_dir = THIS_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "none_plus_validators.json").write_text(
        json.dumps({"pred_dir": args.pred_dir, "filter_labels": FILTER_LABELS, "results": all_res},
                   indent=2), encoding="utf-8")
    (reports_dir / "none_plus_validators.md").write_text(
        _markdown(all_res, locus_modes, args.pred_dir), encoding="utf-8")
    print(f"Saved: {reports_dir / 'none_plus_validators.md'}")
