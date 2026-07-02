from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from maseval.validators.api_validators import ApiHttpValidator
from maseval.validators.base import to_spans
from maseval.validators.environment_validators import EnvironmentSetupValidator
from maseval.validators.provider_validators import ProviderValidator
from maseval.validators.tool_schema_validators import ToolSchemaValidator

ALL_VALIDATORS = [
    ProviderValidator,
    ToolSchemaValidator,
    ApiHttpValidator,
    EnvironmentSetupValidator,
]

_MAX_EVIDENCE = 5


def _norm_quote(quote: str) -> str:
    return re.sub(r"\s+", " ", quote).strip().lower()[:120]


def run_on_trace(trace: Any) -> dict[str, Any]:
    fmt, spans = to_spans(trace)
    span_meta = {s["span_id"]: (s.get("kind"), s.get("parent")) for s in spans}
    raw: list[tuple[type, dict[str, Any]]] = []
    failed: list[str] = []
    for validator_cls in ALL_VALIDATORS:
        try:
            results = validator_cls().run(spans)
        except Exception as exc:  # noqa: BLE001 - isolate one validator's failure
            failed.append(f"{validator_cls.__name__}: {exc}")
            continue
        for finding in results:
            raw.append((validator_cls, finding))

    claimed: list[tuple[Any, int, int, type]] = []
    kept: list[tuple[type, dict[str, Any]]] = []
    for validator_cls, finding in raw:
        span_id, start, end = finding.get("_match", (None, None, None))
        if start is not None:
            if any(
                s == span_id and vc is not validator_cls and start < ce and cs < end
                for (s, cs, ce, vc) in claimed
            ):
                continue
            claimed.append((span_id, start, end, validator_cls))
        kept.append((validator_cls, finding))

    metrics: dict[str, Any] = {}
    dedup: dict[tuple[str, str, Any, str], dict[str, Any]] = {}
    indexed: list[tuple[type, str, str, Any, str, dict[str, Any]]] = []
    for validator_cls, finding in kept:
        name = finding["metric_name"]
        metric = metrics.setdefault(
            name,
            {
                "metric_name": name,
                "explanation": validator_cls.EXPLANATIONS.get(name, finding["explanation"]),
                "findings": [],
            },
        )
        evidence = finding["evidence"]
        quote = evidence[0].get("quote", "") if evidence else ""
        agent = finding["culprit_agent"]
        key = (name, finding["failure_type"], agent, _norm_quote(quote))
        existing = dedup.get(key)
        if existing is not None:
            existing["occurrences"] += 1
            remaining = _MAX_EVIDENCE - len(existing["evidence"])
            if remaining > 0:
                existing["evidence"].extend(evidence[:remaining])
            continue
        out = {
            "verdict": "fail",
            "severity": finding.get("severity"),
            "culprit_agent": agent,
            "failure_type": finding["failure_type"],
            "explanation": finding["explanation"],
            "occurrences": 1,
            "evidence": list(evidence),
        }
        dedup[key] = out
        metric["findings"].append(out)
        primary = finding.get("_match", (None,))[0] or (evidence[0]["span_id"] if evidence else None)
        indexed.append((validator_cls, name, finding["failure_type"], primary, _norm_quote(quote), out))

    _collapse_chain_into_tool(metrics, indexed, span_meta)

    result: dict[str, Any] = {
        "detected_format": fmt,
        "agent_attribution_available": any(s.get("agent") for s in spans),
        "metrics": metrics,
    }
    if failed:  # only present when a validator failed, so the normal shape is unchanged
        result["errors"] = failed
    return result


_COLLAPSE_SCOPE = (ProviderValidator, ToolSchemaValidator)


def _shares_substring(a: str, b: str, n: int = 30) -> bool:
    if not a or not b:
        return False
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) < n:
        return short in long
    return any(short[i : i + n] in long for i in range(len(short) - n + 1))


def _collapse_chain_into_tool(
    metrics: dict[str, Any],
    indexed: list[tuple[type, str, str, Any, str, dict[str, Any]]],
    span_meta: dict[Any, tuple[Any, Any]],
) -> None:
    

    def is_ancestor(ancestor: Any, node: Any) -> bool:
        seen: set[Any] = set()
        cur = span_meta.get(node, (None, None))[1]
        while cur is not None and cur not in seen:
            if cur == ancestor:
                return True
            seen.add(cur)
            cur = span_meta.get(cur, (None, None))[1]
        return False

    drop: set[int] = set()
    for vc1, name1, ft1, sp1, q1, f1 in indexed:
        if vc1 not in _COLLAPSE_SCOPE or not sp1:
            continue
        if (span_meta.get(sp1, (None, None))[0] or "") == "TOOL":
            continue
        for vc2, name2, ft2, sp2, q2, f2 in indexed:
            if f1 is f2 or vc2 not in _COLLAPSE_SCOPE or not sp2:
                continue
            if name1 != name2 or ft1 != ft2:
                continue
            if (span_meta.get(sp2, (None, None))[0] or "") != "TOOL":
                continue
            if is_ancestor(sp1, sp2) and _shares_substring(q1, q2):
                drop.add(id(f1))
                break

    if not drop:
        return
    for metric in metrics.values():
        metric["findings"] = [f for f in metric["findings"] if id(f) not in drop]
    empty = [name for name, metric in metrics.items() if not metric["findings"]]
    for name in empty:
        del metrics[name]


def _findings_count(result: dict[str, Any]) -> int:
    return sum(len(m["findings"]) for m in result.get("metrics", {}).values())


def run_on_file(trace_file: str, out_dir: str) -> dict[str, Any] | None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tf = Path(trace_file)
    try:
        with open(tf, encoding="utf-8") as f:
            trace = json.load(f)
    except Exception as exc:  # noqa: BLE001 - isolate any load failure
        print(f"[skip] {tf.name}: load error: {exc}")
        return None
    try:
        res = run_on_trace(trace)
    except Exception as exc:  # noqa: BLE001 - isolate any validation failure
        print(f"[skip] {tf.name}: validation error: {exc}")
        return None
    out_path = out / f"validation_results_{tf.stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(
        f"{tf.name} -> {out_path.name}  "
        f"[format={res['detected_format']}, findings={_findings_count(res)}]"
    )
    return res


def run_on_dir(trace_dir: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trace_files = sorted(Path(trace_dir).glob("*.json"))
    if not trace_files:
        raise SystemExit(f"No .json traces found in {trace_dir}")

    skipped: list[str] = []
    for tf in trace_files:
        res = run_on_file(str(tf), out_dir)
        if res is None:
            skipped.append(tf.name)

    ok = len(trace_files) - len(skipped)
    print(f"\nDone. {ok}/{len(trace_files)} ok, {len(skipped)} skipped.")
    if skipped:
        print("Skipped files:")
        for name in skipped:
            print(f"- {name}")
