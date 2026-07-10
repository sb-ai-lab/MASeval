"""Mapping-free TRAIL metrics for our MASeval pipeline run (minor-filter sweep).

Scores the per-task findings produced by ``launch_findings_judges.py``
(``examples/trail/trail_gemini_findings_v1/``) against the gold TRAIL
annotations embedded in each file under ``labels`` -- WITHOUT translating our
11 maseval metrics into TRAIL's 21-category taxonomy (the two taxonomies are
incommensurable, so category-level comparison is not attempted here).

Metrics computed (all criterion-independent):

1. LoCc (location accuracy) -- TRAIL's location metric, which by definition
   uses ONLY span locations, not categories. We resolve each finding's cited
   evidence ``idx`` back to the original TRAIL ``span_id`` and compare the set
   of flagged spans to the set of gold error locations. Reported as exact
   match plus within +/-1 / +/-3 span steps.

2. Volume correlation -- Spearman correlation between the number of findings
   our pipeline emits per trace and the number of gold errors, plus the
   correlation of our finding count with the gold overall score.

3. Descriptive summary -- distribution of our finding volume, gold error
   volume, severity mix, and evidence-grounding status across the run.

A constant ``IDX_SHIFT = +1`` corrects a version drift between the trace
formatter the LLM judge saw at evaluation time and the current
``trail_to_spans`` ordering (validated: exact gold-span match rises from 0.25%
to ~40% under the shift).

The script sweeps ``MINOR_CONFIDENCE_CONFIGS`` -- how strictly "minor" findings
are filtered -- to show sensitivity of the metrics to that choice.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# Make the repo root importable (src/ layout).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from maseval.validators.base import trail_to_spans  # noqa: E402

FINDINGS_DIR = Path(__file__).parent / "trail_gemini_findings_v1"
# GAIA raw-trace dir. Override per machine with $TRAIL_GAIA_DIR.
GAIA_DIR = Path(os.environ.get(
    "TRAIL_GAIA_DIR",
    "/mnt/c/Users/barak/Downloads/trail-benchmark-main/benchmarking/data/GAIA",
))

# Version-drift correction for the zero-based span index (see module docstring).
IDX_SHIFT = 1

# Tolerance windows (in span-index steps) reported alongside exact LoCc.
LOC_TOLERANCES = [0, 1, 3]

# How "minor" findings are treated. Config name -> set of confidence levels at
# which a minor finding is still kept (None = keep all minors regardless).
MINOR_CONFIDENCE_CONFIGS = {
    "minor=high": {"high"},
    "minor>=medium": {"high", "medium"},
    "all": None,
}


def _is_qualifying_finding(finding: dict, keep_minor) -> bool:
    severity = str(finding.get("severity_estimate", "major")).lower()
    confidence = str(finding.get("confidence_estimate", "medium")).lower()
    if severity == "minor":
        if keep_minor is None:
            return True
        return confidence in keep_minor
    return True


def build_idx_to_span_id(trace_id: str) -> dict[int, str]:
    """Map zero-based span index -> original TRAIL span_id for one trace."""
    trace_path = GAIA_DIR / f"{trace_id}.json"
    if not trace_path.is_file():
        return {}
    try:
        raw = json.loads(trace_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    spans = trail_to_spans(raw)
    mapping: dict[int, str] = {}
    for i, span in enumerate(spans):
        sid = span.get("idx") or span.get("span_id")
        if sid is not None:
            mapping[i] = str(sid)
    return mapping


def build_spanid_to_index(trace_id: str) -> dict[str, int]:
    """Inverse of :func:`build_idx_to_span_id`: TRAIL span_id -> index."""
    return {sid: i for i, sid in build_idx_to_span_id(trace_id).items()}


def resolve_location(finding: dict, idx_to_span_id: dict[int, str]) -> str | None:
    """Resolve a finding's evidence into a gold span_id (or None if unresolvable)."""
    evidence = finding.get("evidence") or []
    if not evidence:
        return None
    for ev in evidence:
        v = str(ev.get("idx", "")).strip()
        if v in set(idx_to_span_id.values()):
            return v
        if v.lstrip("-").isdigit():
            base = int(v) + IDX_SHIFT
            for cand in (base, int(v), base - 1, base + 1):
                if cand in idx_to_span_id:
                    return idx_to_span_id[cand]
    return None


def extract_predicted_locations(data: dict, trace_id: str, keep_minor) -> list[str]:
    """Collect the gold span_ids our findings point at (mapping-free)."""
    idx_to_span_id = build_idx_to_span_id(trace_id)
    locations: list[str] = []
    for block in data.values():
        if not isinstance(block, dict) or "findings" not in block:
            continue
        for finding in block["findings"]:
            if not _is_qualifying_finding(finding, keep_minor):
                continue
            loc = resolve_location(finding, idx_to_span_id)
            if loc:
                locations.append(loc)
    return locations


def count_qualifying_findings(data: dict, keep_minor) -> int:
    n = 0
    for block in data.values():
        if not isinstance(block, dict) or "findings" not in block:
            continue
        n += sum(1 for f in block["findings"] if _is_qualifying_finding(f, keep_minor))
    return n


def _ranks(a):
    order = sorted(range(len(a)), key=lambda i: a[i])
    r = [0.0] * len(a)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = _ranks(x), _ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = (sum((v - mx) ** 2 for v in rx)) ** 0.5
    vy = (sum((v - my) ** 2 for v in ry)) ** 0.5
    return cov / (vx * vy) if vx and vy else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def run_config(keep_minor):
    """Compute all mapping-free metrics for one minor-filter configuration."""
    locc_exact_sum = 0.0
    tol_acc_sum: dict[int, float] = {t: 0.0 for t in LOC_TOLERANCES}
    tol_denom = 0

    finding_counts: list[int] = []
    gold_error_counts: list[int] = []
    gold_overall: list[float] = []
    severity_counter: Counter = Counter()
    evidence_status_counter: Counter = Counter()

    scored = 0
    traces_with_pred = 0

    for fp in sorted(FINDINGS_DIR.glob("*.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue

        labels = data.get("labels")
        if not labels or "errors" not in labels:
            continue
        scored += 1
        trace_id = data.get("trace_id")

        gold_errors = labels.get("errors", [])
        pred_locs = extract_predicted_locations(data, trace_id, keep_minor)
        if pred_locs:
            traces_with_pred += 1

        # --- LoCc (location only) ---
        gold_locs = [e["location"] for e in gold_errors if e["location"]]
        sid_to_idx = build_spanid_to_index(trace_id)
        if gold_locs and sid_to_idx:
            common = set(gold_locs).intersection(set(pred_locs))
            locc_exact_sum += len(common) / len(set(gold_locs))

            gt_idxs = [sid_to_idx[l] for l in gold_locs if l in sid_to_idx]
            pred_idxs = [sid_to_idx[l] for l in pred_locs if l in sid_to_idx]
            if gt_idxs:
                tol_denom += 1
                for tol in LOC_TOLERANCES:
                    hit = sum(1 for pi in pred_idxs if any(abs(pi - gi) <= tol for gi in gt_idxs))
                    tol_acc_sum[tol] += hit / len(gt_idxs)

        # --- Summary accumulators ---
        n_find = count_qualifying_findings(data, keep_minor)
        finding_counts.append(n_find)
        gold_error_counts.append(len(gold_errors))
        ov = labels.get("scores", [{}])[0].get("overall")
        if ov is not None:
            gold_overall.append(float(ov))

        for block in data.values():
            if not isinstance(block, dict) or "findings" not in block:
                continue
            for finding in block["findings"]:
                if not _is_qualifying_finding(finding, keep_minor):
                    continue
                severity_counter[str(finding.get("severity_estimate", "major")).lower()] += 1

        for ev_block in data.get("evidence_verification", {}).values():
            if not isinstance(ev_block, dict):
                continue
            for ver in ev_block.get("verifications", []):
                evidence_status_counter[str(ver.get("evidence_status", "unknown")).lower()] += 1

    if scored == 0:
        return None

    result = {
        "scored": scored,
        "traces_with_pred": traces_with_pred,
        "locc_exact": locc_exact_sum / scored,
        "locc_tol": {
            t: (tol_acc_sum[t] / tol_denom if tol_denom else 0.0)
            for t in LOC_TOLERANCES
            if t != 0
        },
        "rho_gold_err": spearman(finding_counts, gold_error_counts) if len(finding_counts) > 2 else 0.0,
        "rho_overall": spearman(finding_counts, gold_overall) if len(finding_counts) > 2 else 0.0,
        "find_mean": float(np.mean(finding_counts)),
        "find_median": _median(finding_counts),
        "sev": dict(severity_counter),
        "ev": dict(evidence_status_counter),
    }
    return result


# Default minor-filter policy: keep ALL findings (no minor dropping).
DEFAULT_CONFIG = "all"


def main():
    r = run_config(MINOR_CONFIDENCE_CONFIGS[DEFAULT_CONFIG])
    if r is None:
        print("No scorable traces found.")
        return

    lines = []
    lines.append("=" * 80)
    lines.append("Mapping-free TRAIL metrics for MASeval pipeline run")
    lines.append(f"(examples/trail/trail_gemini_findings_v1 | minor policy: {DEFAULT_CONFIG})")
    lines.append("=" * 80)
    lines.append(f"Scored traces           : {r['scored']}")
    lines.append(f"Traces with >=1 finding : {r['traces_with_pred']}")
    lines.append("")
    lines.append("--- LoCc (location accuracy, category-free) ---")
    lines.append(f"exact match             : {r['locc_exact']:.4f}")
    for tol in (1, 3):
        lines.append(f"within +/-{tol} spans       : {r['locc_tol'][tol]:.4f}")
    lines.append("")
    lines.append("--- Volume correlation (criterion-independent) ---")
    lines.append(f"Spearman(#our findings, #gold errors) : {r['rho_gold_err']:.4f}")
    lines.append(f"Spearman(#our findings, gold overall) : {r['rho_overall']:.4f}")
    lines.append("")
    lines.append("--- Descriptive summary ---")
    lines.append(
        f"our findings / trace    : mean={r['find_mean']:.2f} "
        f"median={r['find_median']:.1f}"
    )
    lines.append(
        f"severity mix (our findings): critical={r['sev'].get('critical',0)} "
        f"major={r['sev'].get('major',0)} minor={r['sev'].get('minor',0)}"
    )
    lines.append(
        f"evidence grounding      : " + ", ".join(f"{k}={v}" for k, v in r["ev"].items())
    )

    out = "\n".join(lines)
    print(out)

    out_file = FINDINGS_DIR.parent / "trail_gemini_findings_v1-metrics_mappingfree.txt"
    out_file.write_text(out + "\n", encoding="utf-8")
    print(f"\nWritten summary to: {out_file}")


if __name__ == "__main__":
    main()
