"""Shared data access for the microsoft/AgentRx findings pipeline.

AgentRx (https://huggingface.co/datasets/microsoft/AgentRx) ships, per config,
a *trajectory* file and an *annotation* file:

* ``{config}_dataset.jsonl`` -- ``{trajectory_id, instruction, steps}`` where
  ``steps`` is a list of ``{index, substeps:[{sub_index, role, content}]}`` and
  ``index`` is the **1-based** step number.
* ``{config}.jsonl`` -- ``{trajectory_id, failure_summary, failures, root_cause,
  root_cause_failure_id, ...}`` where each failure carries ``step_number``
  (== the trajectory step ``index``), ``failed_agent`` and ``failure_category``.

Gold ``step_number`` **is** the step ``index``, so spans MUST be keyed by
``step["index"]`` (see :func:`format_trace` / :func:`idxs`), never by an enumerate
position -- otherwise findings land in a different index space than the gold and
localization silently misaligns. (Evaluators cite the step index in
``evidence[i].idx``.)

Two configs:

* ``magentic`` -- Magentic-One multi-agent web/file trajectories; annotations
  join to trajectories by ``trajectory_id`` (UUID). 44 annotated of 58.
* ``tau`` -- tau-bench retail; the annotation ``trajectory_id`` is a bare number
  ``"N"`` that maps to the dataset id ``"tau_retail_N"``. 29 of 29.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable

CONFIGS: dict[str, dict[str, Any]] = {
    "magentic": {
        "dataset_file": "magentic_dataset.jsonl",
        "annotation_file": "magentic_one.jsonl",
        # annotation trajectory_id == dataset trajectory_id
        "to_dataset_id": lambda tid: str(tid),
        "dataset_name": "AgentRx / Magentic-One",
    },
    "tau": {
        "dataset_file": "tau_retail_dataset.jsonl",
        "annotation_file": "tau_retail.jsonl",
        # annotation "N" -> dataset "tau_retail_N"
        "to_dataset_id": lambda tid: f"tau_retail_{tid}",
        "dataset_name": "AgentRx / tau-bench retail",
    },
}

_REPO_ID = "microsoft/AgentRx"


def _download(filename: str) -> str:
    """Resolve an AgentRx file to a local path (HuggingFace cache)."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(_REPO_ID, filename=filename, repo_type="dataset")


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class AgentRxExample:
    """One annotated trajectory: trace + gold failure localization."""

    row_index: int
    trajectory_id: str
    instruction: str
    steps: list[dict[str, Any]]
    failures: list[dict[str, Any]]
    root_cause_failure_id: str | None
    # Gold, de-duplicated, preserving first-seen order.
    all_failed_agents: list[str] = field(default_factory=list)
    all_failure_steps: list[int] = field(default_factory=list)
    root_cause_agent: str | None = None
    root_cause_step: int | None = None
    failure_categories: list[str] = field(default_factory=list)


def _dedup(seq):
    seen, out = set(), []
    for x in seq:
        if x is None:
            continue
        key = str(x)
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out


@lru_cache(maxsize=None)
def load_examples(config: str) -> tuple[AgentRxExample, ...]:
    """Load and join one config into ordered annotated examples.

    Order follows the annotation file. Only trajectories that join to a
    trajectory record are kept (all of them, verified for both configs).
    """
    if config not in CONFIGS:
        raise ValueError(f"Unknown AgentRx config {config!r}; choose from {list(CONFIGS)}")
    cfg = CONFIGS[config]
    to_ds_id: Callable[[Any], str] = cfg["to_dataset_id"]

    trajectories = {r["trajectory_id"]: r for r in _read_jsonl(_download(cfg["dataset_file"]))}
    annotations = _read_jsonl(_download(cfg["annotation_file"]))

    examples: list[AgentRxExample] = []
    missing: list[str] = []
    for annotation in annotations:
        ds_id = to_ds_id(annotation["trajectory_id"])
        traj = trajectories.get(ds_id)
        if traj is None:
            missing.append(str(annotation["trajectory_id"]))
            continue
        failures = annotation.get("failures") or []
        rc_id = annotation.get("root_cause_failure_id")
        rc = next((f for f in failures if str(f.get("failure_id")) == str(rc_id)), None)
        examples.append(
            AgentRxExample(
                row_index=len(examples),
                trajectory_id=ds_id,
                instruction=traj.get("instruction", ""),
                steps=traj.get("steps") or [],
                failures=failures,
                root_cause_failure_id=str(rc_id) if rc_id is not None else None,
                all_failed_agents=_dedup(f.get("failed_agent") for f in failures),
                all_failure_steps=_dedup(f.get("step_number") for f in failures),
                root_cause_agent=(rc or {}).get("failed_agent"),
                root_cause_step=(rc or {}).get("step_number"),
                failure_categories=_dedup(f.get("failure_category") for f in failures),
            )
        )
    if missing:
        raise RuntimeError(
            f"{len(missing)} {config} annotations did not join to a trajectory "
            f"(e.g. {missing[:5]}). Check CONFIGS[{config!r}]['to_dataset_id']."
        )
    return tuple(examples)


def coerce_substeps(substeps: Any) -> list[Any]:
    """Return substeps as a list. Some AgentRx substeps are serialized as a
    Python-repr string of a list rather than a real list."""
    if isinstance(substeps, list):
        return substeps
    if isinstance(substeps, str):
        text = substeps.strip()
        for parser in (ast.literal_eval, json.loads):
            try:
                parsed = parser(text)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
    return []


def step_text(step: dict[str, Any]) -> str:
    """Flatten one step's substeps into ``role: content`` lines."""
    parts: list[str] = []
    for sub in coerce_substeps(step.get("substeps")):
        if isinstance(sub, dict):
            role = sub.get("role")
            content = sub.get("content")
            if content is None:
                content = json.dumps(sub, ensure_ascii=False, default=str)
            parts.append(f"{role}: {content}" if role else str(content))
        else:
            parts.append(str(sub))
    return "\n".join(p for p in parts if p)


def idxs(steps: list[dict[str, Any]]) -> list[str]:
    """The idx (span-id) space for a trajectory: each step's native 1-based index."""
    return [str(s.get("index")) for s in steps]


def format_trace(steps: list[dict[str, Any]], instruction: str) -> str:
    """Render a trajectory for an LLM evaluator.

    Each step is prefixed with its **native 1-based step index** (== gold
    ``step_number``); evaluators cite that number in ``evidence[i].idx``.
    """
    lines = [
        "USER INSTRUCTION:",
        str(instruction),
        "",
        "TRACE STEPS (each step is prefixed with its 1-based step number in "
        "square brackets; cite that number in evidence[i].idx):",
    ]
    for step in steps:
        lines.append(f"[{step.get('index')}] {step_text(step)}")
    return "\n".join(lines)


def gold_records(config: str) -> list[dict[str, Any]]:
    """Annotation rows for the scorer, one per annotated trajectory.

    Column intent:
      * ``mistake_agents`` / ``mistake_steps`` -- ALL failures (set semantics;
        the lenient default that the accuracy scorer is built for).
      * ``root_cause_agent`` / ``root_cause_step`` -- the single decisive failure
        (strict, Who&When-parity; pass these as explicit agent/step columns).
    """
    records = []
    for ex in load_examples(config):
        records.append(
            {
                "row_index": ex.row_index,
                "trajectory_id": ex.trajectory_id,
                "mistake_agents": list(ex.all_failed_agents),
                "mistake_steps": list(ex.all_failure_steps),
                "root_cause_agent": ex.root_cause_agent,
                "root_cause_step": ex.root_cause_step,
                "failure_categories": list(ex.failure_categories),
                "num_failures": len(ex.failures),
            }
        )
    return records


def write_gold_jsonl(config: str, output_path: str) -> str:
    """Write the gold table to JSONL (used by the accuracy scorer)."""
    from pathlib import Path

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in gold_records(config):
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    return str(out)


if __name__ == "__main__":
    # Quick sanity dump.
    for cfg in CONFIGS:
        exs = load_examples(cfg)
        print(f"{cfg}: {len(exs)} annotated trajectories")
        e = exs[0]
        print(f"  ex0 trajectory_id={e.trajectory_id} steps={len(e.steps)} "
              f"idxs[:5]={idxs(e.steps)[:5]}")
        print(f"  ex0 gold: agents={e.all_failed_agents} steps={e.all_failure_steps} "
              f"root_cause=({e.root_cause_agent}, {e.root_cause_step})")
