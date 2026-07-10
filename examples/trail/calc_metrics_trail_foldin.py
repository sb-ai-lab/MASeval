"""TRAIL LoCc fold-in: judges vs judges + non_llm_validators (LLM-confirmed).

Extends ``calc_metrics_trail_findings.py`` (same LoCc definition, same
``resolve_location`` logic incl. ``IDX_SHIFT`` and the span-hash resolution) to
add the deterministic validator findings into the predicted-location set, so the
Who&When/AEGIS-style fold-in is measured on TRAIL's *canonical* location metric
rather than a bespoke step-hit.

LoCc = mean over scored traces of ``|gold ∩ pred| / |gold|`` (per-trace location
recall); ±1/±3 are the neighbor-tolerant variants (span-index distance).

Reads the CONFIRM folder (``trail_gemini_findings_v1_confirm``), which carries the
judges + validators + ``llm_confirmation``. Validator evidence idxs are already
span-hashes, so they resolve directly; the appointed locus uses the confirmer's
``corrected_idx``.

Verdict filters: none (judges only, reproduces the 18.3% baseline) / all /
conf+unc / confirmed. Locus for folded validators: surface (evidence idx) or
appointed (corrected_idx, fallback surface). Reported overall + bearing-only.

Usage:
    python calc_metrics_trail_foldin.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(THIS.parents[1] / "src"))

from calc_metrics_trail_findings import (  # noqa: E402
    GAIA_DIR,
    IDX_SHIFT,
    LOC_TOLERANCES,
    MINOR_CONFIDENCE_CONFIGS,
    _is_qualifying_finding,
    resolve_location,
)
from maseval.validators.base import trail_to_spans  # noqa: E402

CONFIRM_DIR = THIS / "trail_gemini_findings_v1_confirm"
FILTERS = ("none_only", "all", "conf_unc", "confirmed")
FILTER_LABELS = {
    "none_only": "none (judges)", "all": "+val all",
    "conf_unc": "+val conf+unc", "confirmed": "+val confirmed",
}
DEFAULT_MINOR = "all"


def _maps(trace_id):
    p = GAIA_DIR / f"{trace_id}.json"
    if not p.is_file():
        return {}, {}
    spans = trail_to_spans(json.loads(p.read_text(encoding="utf-8")))
    idx_to_sid = {i: str(s.get("idx") or s.get("span_id"))
                  for i, s in enumerate(spans) if (s.get("idx") or s.get("span_id")) is not None}
    return idx_to_sid, {sid: i for i, sid in idx_to_sid.items()}


def _resolve_idx(v, idx_to_sid):
    """Single-idx version of resolve_location's inner logic (hash or +1-shifted numeric)."""
    v = str(v).strip()
    if not v:
        return None
    if v in set(idx_to_sid.values()):
        return v
    if v.lstrip("-").isdigit():
        base = int(v) + IDX_SHIFT
        for cand in (base, int(v), base - 1, base + 1):
            if cand in idx_to_sid:
                return idx_to_sid[cand]
    return None


def judge_locs(data, idx_to_sid, keep_minor):
    out = []
    for block in data.values():
        if not isinstance(block, dict) or "findings" not in block:
            continue  # skips non_llm_validators (nested under 'metrics')
        for f in block["findings"]:
            if not _is_qualifying_finding(f, keep_minor):
                continue
            loc = resolve_location(f, idx_to_sid)
            if loc:
                out.append(loc)
    return out


def _iter_val(data):
    nlv = data.get("non_llm_validators") or {}
    for m in (nlv.get("metrics") or {}).values():
        for f in m.get("findings", []):
            yield f


def _passes(f, filt):
    if filt == "all":
        return True
    verdict = (f.get("llm_confirmation") or {}).get("verdict")
    if filt == "conf_unc":
        return verdict in ("confirmed", "uncertain")
    if filt == "confirmed":
        return verdict == "confirmed"
    return False


def val_locs(data, idx_to_sid, filt, locus):
    out = []
    for f in _iter_val(data):
        if not _passes(f, filt):
            continue
        chosen = None
        if locus == "appointed":
            c = (f.get("llm_confirmation") or {}).get("corrected_idx")
            if c is not None and str(c).strip():
                chosen = c
        if chosen is None:
            ev = f.get("evidence") or []
            if ev:
                chosen = ev[0].get("idx")
        loc = _resolve_idx(chosen, idx_to_sid) if chosen is not None else None
        if loc:
            out.append(loc)
    return out


def locc(gold_locs, pred_locs, sid_to_idx):
    exact = len(set(gold_locs) & set(pred_locs)) / len(set(gold_locs))
    gt = [sid_to_idx[l] for l in gold_locs if l in sid_to_idx]
    pr = [sid_to_idx[l] for l in pred_locs if l in sid_to_idx]
    tol = {}
    for t in LOC_TOLERANCES:
        if t == 0:
            continue
        hit = sum(1 for pi in pr if any(abs(pi - gi) <= t for gi in gt))
        tol[t] = (hit / len(gt)) if gt else 0.0
    return exact, tol


def run(locus, keep_minor):
    acc = {f: {"exact": 0.0, 1: 0.0, 3: 0.0, "n": 0} for f in FILTERS}
    acc_b = {f: {"exact": 0.0, 1: 0.0, 3: 0.0, "n": 0} for f in FILTERS}
    for fp in sorted(CONFIRM_DIR.glob("*.json")):
        data = json.loads(fp.read_text(encoding="utf-8"))
        labels = data.get("labels")
        if not labels or "errors" not in labels:
            continue
        # Denominator matches calc_metrics_trail_findings: every trace with an
        # 'errors' key is scored; traces with no gold location or no span map
        # contribute exact=0 (they still count in n).
        gold = [e["location"] for e in labels["errors"] if e.get("location")]
        tid = data.get("trace_id")
        idx_to_sid, sid_to_idx = _maps(tid)
        computable = bool(gold and sid_to_idx)
        bearing = any(True for _ in _iter_val(data))
        base = judge_locs(data, idx_to_sid, keep_minor) if computable else []
        for filt in FILTERS:
            if computable:
                preds = list(base)
                if filt != "none_only":
                    preds += val_locs(data, idx_to_sid, filt, locus)
                ex, tol = locc(gold, preds, sid_to_idx)
            else:
                ex, tol = 0.0, {}
            for store, active in ((acc, True), (acc_b, bearing)):
                if active:
                    store[filt]["exact"] += ex
                    store[filt][1] += tol.get(1, 0.0)
                    store[filt][3] += tol.get(3, 0.0)
                    store[filt]["n"] += 1
    return acc, acc_b


def _tab(title, acc):
    n = acc["none_only"]["n"]
    hdr = f"{'LoCc':<10}" + "".join(FILTER_LABELS[f].rjust(16) for f in FILTERS)
    lines = [f"--- {title} (n={n}) ---", hdr, "-" * len(hdr)]
    for key, lab in (("exact", "exact"), (1, "within ±1"), (3, "within ±3")):
        row = f"{lab:<10}"
        for f in FILTERS:
            m = acc[f]["n"]
            row += (f"{acc[f][key]/m*100:6.1f}%" if m else "     —").rjust(16)
        lines.append(row)
    return "\n".join(lines)


if __name__ == "__main__":
    keep = MINOR_CONFIDENCE_CONFIGS[DEFAULT_MINOR]
    out = [f"TRAIL LoCc fold-in (minor policy: {DEFAULT_MINOR}; confirm dir)\n"]
    for locus in ("surface", "appointed"):
        acc, acc_b = run(locus, keep)
        out.append(f"\n==================== LOCUS = {locus} ====================")
        out.append(_tab("Overall", acc))
        out.append("")
        out.append(_tab("Bearing-only", acc_b))
    text = "\n".join(out)
    print(text)
    (THIS / "trail_foldin_locc.txt").write_text(text + "\n", encoding="utf-8")
    print(f"\nWritten: {THIS / 'trail_foldin_locc.txt'}")
