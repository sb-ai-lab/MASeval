"""Shared data access for the TraceElephant findings pipeline.

TraceElephant (https://huggingface.co/datasets/TraceElephant/TraceElephant, ACL
2026, "Seeing the Whole Elephant") ships 220 annotated *failure* traces (of 380
total executions) from three multi-agent systems: **Captain-Agent**,
**Magentic-One**, and **SWE-Agent**. Each failed trace is annotated with the
responsible agent (``mistake_agent``) and the decisive step (``mistake_step``,
the earliest inevitable error, 1-based over the execution history).

On disk (after unzipping ``data.zip``) a trace lives in a per-task directory
under a ``{captain,magentic,swe-agent}-runs-*`` parent, in one of two shapes:

* **new** -- ``trace_metadata.json`` (task_instruction, ground_truth, mistake_agent,
  mistake_step, agent_system_intro) + ``step_records.json`` (a list of
  ``{agent_name, input, output}`` steps, where ``output`` is a ChatCompletion repr);
* **old** -- ``summary.json`` (question, ground_truth, mistake_agent, mistake_step) +
  ``history.json`` (a list of ``{agent_name, request, response}`` steps).

A flat ``{task}.json`` with ``mistake_agent``/``mistake_step``/``history`` is also
supported. We normalize all three into a ``history`` of ``{name, content}`` steps.

The gold ``mistake_step`` is the **1-based position in the (failure-relevant)
history**, so spans MUST be keyed by that position (see :func:`format_trace` /
:func:`idxs`); evaluators cite the step number in ``evidence[i].idx``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# runs-dir prefix -> coarse system category (matches TraceElephant's evaluate.py).
_RUNS_PREFIXES = {
    "captain-runs-": "captain",
    "magentic-runs-": "magentic",
    "swe-agent-runs-": "swe",
}


def _extract_completion_content(output: Any) -> str | None:
    """Pull the assistant message text out of a ChatCompletion repr string.

    ``step_records.json`` stores ``output`` as ``"ChatCompletion(... content='...' ...)"``.
    Mirror TraceElephant's own regex so the rendered step shows the real content.
    """
    if not isinstance(output, str) or "content=" not in output:
        return None
    m = re.search(r"content='(.*?)'(?:,|\))", output.replace("\\'", "'"))
    if not m:
        return None
    return m.group(1).replace("\\n", "\n").replace("\\t", "\t")


def _step_content(name: str, request: Any, response: Any, raw_output: Any) -> str:
    """A compact, attribution-relevant rendering of one agent turn.

    Prefer the assistant's response text (what the agent actually produced);
    fall back to the raw request/response payloads so nothing is silently lost.
    """
    content = _extract_completion_content(raw_output)
    if content:
        return content
    if isinstance(response, dict) and response.get("content"):
        return str(response["content"])
    if response:
        return json.dumps(response, ensure_ascii=False, default=str)
    if request:
        return json.dumps(request, ensure_ascii=False, default=str)
    return ""


@dataclass
class TraceElephantExample:
    """One annotated failure trace: normalized history + gold attribution."""

    row_index: int
    task_name: str
    system_category: str  # captain | magentic | swe | other
    system_name: str | None
    question: str
    ground_truth: str
    agent_system_intro: str
    # history[i] = {"name": agent, "content": text}; step number = i + 1.
    history: list[dict[str, str]] = field(default_factory=list)
    mistake_agent: str = ""
    mistake_step: str = ""


def _normalize_task(task_dir: Path) -> dict[str, Any] | None:
    """Load one task dir into the normalized shape, or None if unrecognized."""
    meta_f = task_dir / "trace_metadata.json"
    steps_f = task_dir / "step_records.json"
    summary_f = task_dir / "summary.json"
    history_f = task_dir / "history.json"

    if meta_f.exists() and steps_f.exists():
        meta = json.loads(meta_f.read_text(encoding="utf-8"))
        step_records = json.loads(steps_f.read_text(encoding="utf-8"))
        history = [
            {
                "name": s.get("agent_name", "Unknown"),
                "content": _step_content(
                    s.get("agent_name", "Unknown"), s.get("input", {}), None, s.get("output", "")
                ),
            }
            for s in step_records
        ]
        return {
            "question": meta.get("task_instruction", ""),
            "ground_truth": meta.get("ground_truth", ""),
            "agent_system_intro": meta.get("agent_system_intro", ""),
            "system_name": meta.get("system_name"),
            "history": history,
            "mistake_agent": str(meta.get("mistake_agent", "") or ""),
            "mistake_step": str(meta.get("mistake_step", "") or ""),
        }

    if summary_f.exists() and history_f.exists():
        summary = json.loads(summary_f.read_text(encoding="utf-8"))
        history_steps = json.loads(history_f.read_text(encoding="utf-8"))
        history = [
            {
                "name": s.get("agent_name", "Unknown"),
                "content": _step_content(
                    s.get("agent_name", "Unknown"), s.get("request", {}), s.get("response", {}), None
                ),
            }
            for s in history_steps
        ]
        return {
            "question": summary.get("question", ""),
            "ground_truth": summary.get("ground_truth", ""),
            "agent_system_intro": summary.get("agent_system_intro", ""),
            "system_name": summary.get("system_name"),
            "history": history,
            "mistake_agent": str(summary.get("mistake_agent", "") or ""),
            "mistake_step": str(summary.get("mistake_step", "") or ""),
        }

    return None


def _flat_task(json_file: Path) -> dict[str, Any] | None:
    """Load a flat ``{task}.json`` that already carries history + gold."""
    try:
        data = json.loads(json_file.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict) or "mistake_agent" not in data:
        return None
    history = []
    for s in data.get("history") or []:
        name = s.get("name") or s.get("agent_name", "Unknown")
        content = s.get("content")
        if content is None:
            content = _step_content(name, s.get("request", {}), s.get("response", {}), s.get("output", ""))
        history.append({"name": name, "content": str(content)})
    return {
        "question": data.get("question", ""),
        "ground_truth": data.get("ground_truth", ""),
        "agent_system_intro": data.get("agent_system_intro", ""),
        "system_name": data.get("system_name"),
        "history": history,
        "mistake_agent": str(data.get("mistake_agent", "") or ""),
        "mistake_step": str(data.get("mistake_step", "") or ""),
    }


def _iter_task_dirs(data_dir: Path):
    """Yield (task_name, task_dir, category) for every task under data_dir.

    Handles the ``{system}-runs-*`` nested layout and a flat directory of task
    subdirectories / task JSON files.
    """
    runs_dirs = [
        d for d in data_dir.iterdir()
        if d.is_dir() and any(d.name.startswith(p) for p in _RUNS_PREFIXES)
    ]
    if runs_dirs:
        for runs_dir in sorted(runs_dirs):
            category = next(
                (c for p, c in _RUNS_PREFIXES.items() if runs_dir.name.startswith(p)), "other"
            )
            for task_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
                yield task_dir.name, task_dir, category
        return
    # Flat: task subdirectories, then loose task JSON files.
    for task_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        yield task_dir.name, task_dir, "other"
    for jf in sorted(data_dir.glob("*.json")):
        yield jf.stem, jf, "other"


def load_examples(data_dir: str | Path, *, failures_only: bool = True) -> list[TraceElephantExample]:
    """Load TraceElephant tasks under ``data_dir`` into ordered examples.

    ``failures_only`` keeps the 220 annotated failure traces (those carrying both
    a ``mistake_agent`` and a ``mistake_step``) -- the attribution benchmark.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"TraceElephant data dir not found: {data_dir}")

    examples: list[TraceElephantExample] = []
    for task_name, path, category in _iter_task_dirs(data_dir):
        norm = _normalize_task(path) if path.is_dir() else _flat_task(path)
        if norm is None:
            continue
        if failures_only and not (norm["mistake_agent"] and norm["mistake_step"]):
            continue
        if not norm["history"]:
            continue
        examples.append(
            TraceElephantExample(
                row_index=len(examples),
                task_name=task_name,
                system_category=category,
                system_name=norm.get("system_name"),
                question=norm["question"],
                ground_truth=norm["ground_truth"],
                agent_system_intro=norm["agent_system_intro"],
                history=norm["history"],
                mistake_agent=norm["mistake_agent"],
                mistake_step=norm["mistake_step"],
            )
        )
    return examples


def idxs(history: list[dict[str, str]]) -> list[str]:
    """The idx (span-id) space: each step's 1-based position (== gold mistake_step)."""
    return [str(i + 1) for i in range(len(history))]


def format_trace(example: TraceElephantExample) -> str:
    """Render a failure trace for an LLM evaluator.

    Each step is prefixed with its **1-based step number** (== gold
    ``mistake_step``); evaluators cite that number in ``evidence[i].idx``.
    """
    lines = ["USER INSTRUCTION:", str(example.question), ""]
    if example.agent_system_intro:
        lines += ["AGENT SYSTEM:", str(example.agent_system_intro), ""]
    lines.append(
        "TRACE STEPS (each step is prefixed with its 1-based step number in "
        "square brackets; cite that number in evidence[i].idx):"
    )
    for i, step in enumerate(example.history, start=1):
        lines.append(f"[{i}] {step['name']}: {step['content']}")
    return "\n".join(lines)


def gold_records(examples: list[TraceElephantExample]) -> list[dict[str, Any]]:
    """Annotation rows for the scorer, one per failure trace.

    TraceElephant gold is a single decisive (agent, step) per trace; we expose it
    as one-element lists so the shared set-based accuracy scorer applies directly.
    """
    return [
        {
            "row_index": ex.row_index,
            "task_name": ex.task_name,
            "system_category": ex.system_category,
            "mistake_agents": [ex.mistake_agent] if ex.mistake_agent else [],
            "mistake_steps": [ex.mistake_step] if ex.mistake_step else [],
            "mistake_agent": ex.mistake_agent,
            "mistake_step": ex.mistake_step,
        }
        for ex in examples
    ]


def write_gold_jsonl(examples: list[TraceElephantExample], output_path: str) -> str:
    """Write the gold table to JSONL (used by the accuracy scorer)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in gold_records(examples):
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    return str(out)
