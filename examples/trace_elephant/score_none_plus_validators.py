"""Score TraceElephant agent/step localization for verifier=``none``, folding in
the deterministic ``non_llm_validators`` (+ LLM confirmation) — the TraceElephant
analogue of Who&When's ``score_none_plus_validators.py``.

Baseline is the verifier=``none`` prediction over the 11 LLM evaluators. On top we
OR-in the deterministic validator findings under three verdict filters:

* ``all``       -- every regex-detected finding (no confirmer gating);
* ``conf+unc``  -- confirmer verdict confirmed OR uncertain;
* ``confirmed`` -- confirmed only (the honest column — captain fires on 100% of
                   traces, so ``all`` is a recall ceiling artifact, not a result).

**Index spaces.** TraceElephant's judges *and* gold are 1-based
(``format_trace`` uses ``enumerate(..., start=1)``); only the validators
(``who_and_when_to_spans``) are 0-based. That 0-vs-1 offset was the entire reason
the launcher skipped these validators — a narrow bookkeeping issue, not a deeper
incompatibility. We shift every validator locus (surface evidence idx **and**
appointed ``corrected_idx``) by **+1** here, so baseline, gold, judges and
validators all live in one 1-based space.

**Agent namespace.** The fold-in agent is always the step *label* at the (1-based)
validator locus — a sub-agent for captain/magentic, the invoked tool for swe.
For the baseline, captain/magentic use the judge-named ``problematic_agents``;
swe uses the tool at the judge locus (its gold ``mistake_agent`` is a tool, so
the judge names never match — same fix as ``score_swe_tool_attribution.py``).

**Locus variants.** Reported twice: ``appointed`` (confirmer ``corrected_idx``,
fallback surface) and ``surface`` (raw evidence idx). On TraceElephant the
validators fire *before* the (later) gold step and appointing moves *earlier*, so
which locus wins is data-dependent — we don't assume appointed wins as it did on
Who&When.

Usage:
    python score_none_plus_validators.py                    # all systems + overall
    python score_none_plus_validators.py --system swe
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

import calculate_agent_step_accuracy as C  # noqa: E402

SYSTEMS = ("captain", "magentic", "swe")
SWE_TOOL_SYSTEMS = ("swe",)  # baseline agent derived from the tool at the locus

FILTERS = ("none_only", "all", "conf_unc", "confirmed")
FILTER_LABELS = {
    "none_only": "none (LLM only)",
    "all": "none + val (all)",
    "conf_unc": "none + val (conf+unc)",
    "confirmed": "none + val (confirmed)",
}
METRICS = ("agent_top1", "agent_hit", "step_top1", "step_hit", "step_pm1")


def _label_map(example) -> dict[str, str]:
    """1-based step idx -> step label (sub-agent for captain/magentic, tool for swe)."""
    return {str(k + 1): s["name"] for k, s in enumerate(example.history)}


def _iter_val_findings(nlv: dict):
    for m in (nlv.get("metrics") or {}).values():
        for f in m.get("findings", []):
            yield f


def _shift(idx) -> str | None:
    """Validator 0-based idx -> 1-based string (None if unparseable)."""
    if idx is None:
        return None
    s = str(idx).strip()
    if not s:
        return None
    try:
        return str(int(s) + 1)
    except ValueError:
        return None


def _val_locus_idx(finding: dict, locus_mode: str) -> str | None:
    """1-based locus. ``appointed`` = corrected_idx (fallback surface); ``surface`` = evidence idx."""
    conf = finding.get("llm_confirmation") or {}
    if locus_mode == "appointed":
        c = _shift(conf.get("corrected_idx"))
        if c is not None:
            return c
    ev = finding.get("evidence") or []
    if ev and ev[0].get("idx") is not None:
        return _shift(ev[0]["idx"])
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


def score(system: str, locus_mode: str, data_dir: str | None = None,
          pred_glob: str | None = None, step_tolerance: int = 1) -> dict:
    data_dir = data_dir or str(THIS_DIR / "data")
    examples = C._subset(data_dir, system)
    by_row = {e.row_index: e for e in examples}
    swe_tool = system in SWE_TOOL_SYSTEMS

    pred_glob = pred_glob or str(THIS_DIR / f"trace_elephant_{system}_valconfirm")
    files = sorted(
        glob.glob(str(Path(pred_glob) / "*.json")) if Path(pred_glob).is_dir()
        else glob.glob(pred_glob),
        key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]),
    )
    if not files:
        raise FileNotFoundError(f"No prediction files: {pred_glob!r}")

    acc = {f: {k: [] for k in METRICS} for f in FILTERS}
    counts = {"bearing": 0, "confirmed": 0, "benign": 0, "uncertain": 0, "appointed": 0}
    matched = 0

    for fp in files:
        i = int(fp.rsplit("_", 1)[1].split(".")[0])
        ex = by_row.get(i)
        if ex is None:
            continue
        gold_agents = {_normalize_agent(ex.mistake_agent)} if ex.mistake_agent else set()
        gold_idxs = [str(ex.mistake_step)] if ex.mistake_step else []
        if not gold_agents and not gold_idxs:
            continue
        matched += 1
        name_at = _label_map(ex)

        pred = read_prediction_file(fp, verifier_mode="none")
        base_idxs = list(pred.idxs)          # 1-based judge idxs
        base_first = pred.first_idx
        if swe_tool:
            base_agents = [name_at[str(ix)] for ix in pred.idxs if name_at.get(str(ix))]
            base_primary = name_at.get(str(pred.first_idx)) if pred.first_idx else None
        else:
            base_agents = list(pred.agents)
            base_primary = pred.primary_agent

        data = json.loads(Path(fp).read_text(encoding="utf-8"))
        nlv = data.get("non_llm_validators") or {}
        val_hits = list(_iter_val_findings(nlv))
        s = nlv.get("llm_confirmation_summary") or {}
        if val_hits:
            counts["bearing"] += 1
            for k in ("confirmed", "benign", "uncertain", "appointed"):
                counts[k] += int(s.get(k, 0) or 0)

        for filt in FILTERS:
            agents = list(base_agents)
            idxs = list(base_idxs)
            primary = base_primary
            first = base_first
            if filt != "none_only":
                for finding in val_hits:
                    if not _passes(finding, filt):
                        continue
                    ix = _val_locus_idx(finding, locus_mode)
                    a = name_at.get(ix) if ix is not None else None
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
    return {"matched": matched, "summary": summary, "counts": counts, "raw": acc}


def _combine(per_system: dict[str, dict]) -> dict:
    """True overall by concatenating per-trace booleans across systems."""
    acc = {f: {k: [] for k in METRICS} for f in FILTERS}
    counts = {"bearing": 0, "confirmed": 0, "benign": 0, "uncertain": 0, "appointed": 0}
    matched = 0
    for res in per_system.values():
        matched += res["matched"]
        for k in counts:
            counts[k] += res["counts"][k]
        for f in FILTERS:
            for k in METRICS:
                acc[f][k].extend(res["raw"][f][k])

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    summary = {f: {k: mean(v) for k, v in acc[f].items()} for f in FILTERS}
    return {"matched": matched, "summary": summary, "counts": counts}


def _pct(v):
    return f"{v * 100:5.1f}%" if isinstance(v, (int, float)) else "    —"


ROWS = [
    ("Agent Top-1", "agent_top1"), ("Agent Hit", "agent_hit"),
    ("Step Top-1", "step_top1"), ("Step Hit", "step_hit"), ("Step Hit ±1", "step_pm1"),
]


def _render(name: str, res: dict, locus_mode: str) -> str:
    s = res["summary"]
    c = res["counts"]
    header = f"{'Metric':<14}" + "".join(FILTER_LABELS[f].rjust(22) for f in FILTERS)
    lines = [
        f"=== {name} — none + validators (llm-confirm), locus={locus_mode} (matched {res['matched']}) ===",
        f"bearing={c['bearing']} confirmed={c['confirmed']} uncertain={c['uncertain']} "
        f"benign={c['benign']} appointed={c['appointed']}",
        "",
        header,
        "-" * len(header),
    ]
    for label, key in ROWS:
        lines.append(f"{label:<14}" + "".join(_pct(s[f][key]).rjust(22) for f in FILTERS))
    return "\n".join(lines)


def _markdown(all_res: dict, locus_modes: list[str]) -> str:
    md = [
        "# TraceElephant — none + non_llm_validators (LLM-confirm) fold-in",
        "",
        "- Baseline: verifier=`none` over the 11 LLM evaluators.",
        "- Fold-in idx shifted **+1** (validators 0-based → judge/gold 1-based space).",
        "- Fold-in agent = step label at the locus (sub-agent for captain/magentic, "
        "tool for swe); swe baseline agent = tool at the judge locus.",
        "- Filters: `all` = every regex finding (captain fires on 100% of traces → "
        "recall ceiling, not a result); `conf+unc` = confirmed|uncertain; "
        "`confirmed` = confirmed only (**the honest column**).",
        "",
    ]
    for locus_mode in locus_modes:
        md.append(f"## Locus = {locus_mode}")
        md.append("")
        for name in (*SYSTEMS, "OVERALL"):
            res = all_res[locus_mode][name]
            c = res["counts"]
            md.append(f"### {name} (matched {res['matched']})")
            md.append("")
            md.append(f"_bearing={c['bearing']}, confirmed={c['confirmed']}, "
                      f"uncertain={c['uncertain']}, benign={c['benign']}, "
                      f"appointed={c['appointed']}_")
            md.append("")
            md.append("| Metric | " + " | ".join(FILTER_LABELS[f] for f in FILTERS) + " |")
            md.append("|---|" + "---:|" * len(FILTERS))
            s = res["summary"]
            for label, key in ROWS:
                md.append("| " + label + " | "
                          + " | ".join(_pct(s[f][key]).strip() for f in FILTERS) + " |")
            md.append("")
    return "\n".join(md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TraceElephant none + validators fold-in.")
    parser.add_argument("--system", choices=("all", *SYSTEMS), default="all")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--step-tolerance", type=int, default=1)
    parser.add_argument("--locus", choices=("appointed", "surface", "both"), default="both")
    args = parser.parse_args()

    systems = SYSTEMS if args.system == "all" else (args.system,)
    locus_modes = ["appointed", "surface"] if args.locus == "both" else [args.locus]

    all_res: dict[str, dict] = {}
    for locus_mode in locus_modes:
        per_system = {
            name: score(name, locus_mode, data_dir=args.data_dir,
                        step_tolerance=args.step_tolerance)
            for name in systems
        }
        blocks = dict(per_system)
        if args.system == "all":
            blocks["OVERALL"] = _combine(per_system)
        all_res[locus_mode] = blocks
        print()
        for name, res in blocks.items():
            print(_render(name, res, locus_mode))
            print()

    reports_dir = THIS_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    # Strip raw boolean lists before serializing.
    serializable = {
        lm: {name: {k: v for k, v in res.items() if k != "raw"} for name, res in blocks.items()}
        for lm, blocks in all_res.items()
    }
    json_path = reports_dir / "none_plus_validators.json"
    json_path.write_text(json.dumps(
        {"filter_labels": FILTER_LABELS, "results": serializable}, indent=2), encoding="utf-8")
    if args.system == "all":
        md_path = reports_dir / "none_plus_validators.md"
        md_path.write_text(_markdown(all_res, locus_modes), encoding="utf-8")
        print(f"Saved: {md_path}")
    print(f"Saved: {json_path}")
