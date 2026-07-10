"""Per-task token/latency metering helpers for the example launchers.

A launcher times each LLM metric (``time.perf_counter``) and reads the metric's
``last_usage`` (set by ``metrics.base.LLMMetric.evaluate``) into a per-metric
record, then rolls those up with :func:`aggregate_task_stats` into a ``task_stats``
block it stores on each result file — so runs are cost/latency-auditable after the
fact (see ``who&when_*_v9`` runs).

Per-metric record shape:
    {"status": "ok"|"failed", "reason"?, "detail"?,
     "duration_s", "input_tokens", "output_tokens", "total_tokens"}
"""

from __future__ import annotations

from typing import Any


def extract_usage(usage: Any) -> tuple[int, int, int]:
    """Pull ``(input, output, total)`` tokens from a pydantic_ai usage object,
    tolerating naming differences across versions. Returns zeros when unavailable."""
    if usage is None:
        return 0, 0, 0

    def pick(*names: str) -> int:
        for n in names:
            v = getattr(usage, n, None)
            if v is not None:
                try:
                    return int(v)
                except (TypeError, ValueError):
                    continue
        return 0

    inp = pick("input_tokens", "request_tokens", "prompt_tokens")
    out = pick("output_tokens", "response_tokens", "completion_tokens")
    tot = pick("total_tokens") or (inp + out)
    return inp, out, tot


def aggregate_task_stats(metric_status: dict, wall_s: float) -> dict:
    """Roll up per-metric timing/tokens for one task into a ``task_stats`` block."""
    vals = list(metric_status.values())
    return {
        "wall_s": round(wall_s, 3),
        "sum_metric_s": round(sum(v.get("duration_s") or 0 for v in vals), 3),
        "input_tokens": sum(v.get("input_tokens") or 0 for v in vals),
        "output_tokens": sum(v.get("output_tokens") or 0 for v in vals),
        "total_tokens": sum(v.get("total_tokens") or 0 for v in vals),
        "metrics_ok": sum(1 for v in vals if v.get("status") == "ok"),
        "metrics_failed": sum(1 for v in vals if v.get("status") == "failed"),
    }
