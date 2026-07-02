"""Annotate errors in MAS traces using multiple LLM judges via OpenRouter.

Usage:
    python annotate_errors.py --traces <trace_dir> --output <output_dir>
    python annotate_errors.py --download exgentic_agent_llm_traces

Arguments:
    --traces    Path to directory with raw trace JSON files
    --output    Path to directory for annotation output (default: ./error_annotations)
    --download  Download a dataset by name and save traces locally

Available datasets for --download:
    exgentic_agent_llm_traces  - Exgentic/agent-llm-traces from HuggingFace

Output: one JSON per model per trace in <output_dir>/
"""

import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from maseval.openrouter_model import OpenRouterChatModel
from pydantic_ai import Agent

LANGFUSE_ANNO_SK = os.getenv("LANGFUSE_SECRET_KEY_ANNO", "")
LANGFUSE_ANNO_PK = os.getenv("LANGFUSE_PUBLIC_KEY_ANNO", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_BASE_URL_ANNO", "")

_langfuse_client = None

if LANGFUSE_ANNO_SK and LANGFUSE_ANNO_PK:
    from langfuse import Langfuse

    _langfuse_client = Langfuse(
        public_key=LANGFUSE_ANNO_PK,
        secret_key=LANGFUSE_ANNO_SK,
        host=LANGFUSE_HOST,
    )


ERROR_CATEGORIES = [
    "reasoning_planning",
    "hallucination",
    "instruction_following",
    "tool_calling",
    "mas_coordination",
    "context_state",
    "verification_termination",
    "environmental",
    "api_system",
    "ideal",
]

CATEGORY_DEFINITIONS = """reasoning_planning — Failures inside the agent's own inference and goal-directed thinking: flawed or inefficient plans, incorrect problem identification, mismatches between stated reasoning and chosen action, goal drift, oversimplification, and ignored constraints. The agent thinks wrongly even when its tools, inputs, and collaborators are fine.
hallucination — Generation of content not grounded in any input, context, or tool output. Spans both confident confabulation of false facts and deliberate invention of values, records, or details that appear nowhere in the trace. Isolated as its own category because it demands grounding-based mitigation rather than better planning.
instruction_following — The agent ignores, misreads, or violates explicit directives from the user, system prompt, or orchestrator. Covers under-execution (skipping required steps), over-execution (taking unsanctioned actions), and acting outside its assigned task or role.
tool_calling — Failures at the agent–tool interface: malformed or invalid calls (schema violations, missing arguments, formatting errors), selecting the wrong tool, misreading what a tool returned, and requesting intent that no available tool can satisfy.
mas_coordination — Failures specific to multi-agent settings: breakdowns in communication, handoff, and shared state across agents — context resets, withheld information, ignored peer outputs, task derailment during handoff, poor role distribution, and inconsistent beliefs about task state.
context_state — Failures of memory and state tracking within a single agent's trajectory: losing prior context, repeating already-completed steps, and holding an incorrect model of where it currently is in the task.
verification_termination — Failures in checking correctness and deciding when to stop: terminating too early, failing to recognize completion, verifying incompletely or incorrectly, taking redundant actions, and exhausting turn or token budgets.
environmental — Failures originating outside the agent's cognition, in the surrounding environment: triggered guardrails, missing resources, misconfigured or erroring environments, and authentication or permission failures.
api_system — Externally-caused, reproducible failures at the API and infrastructure layer: timeouts, rate limits, downstream service errors, and system failures that reflect environment state rather than agent capability.
ideal — Trajectories that are error-free across every evaluated dimension; the no-error reference class."""

SYSTEM_PROMPT = f"""You are an expert error analyst for multi-agent systems (MAS).
Analyze the MAS execution trace below. For each error you find, output:
- **error_key**: one of the allowed categories listed below
- **step**: integer step number where the error occurs (0-indexed)
- **justification**: 1-2 sentences explaining why

Allowed categories (with definitions):
{CATEGORY_DEFINITIONS}

IMPORTANT RULES:
- If no errors are found, output exactly one entry with error_key="ideal", step=-1, and a brief justification.
- "ideal" and any other error category are MUTUALLY EXCLUSIVE. Never mix "ideal" with other errors.
- Do NOT include any "task_success" or "task_success_reasoning" fields.
- Each error category may appear AT MOST ONCE. If the same error category occurs at multiple steps, report it only once with step set to the most critical occurrence, and list ALL relevant steps in the justification.

Be thorough. Only report real errors."""

MODELS = [
    "google/gemini-2.5-flash",
]

MODELS_WITHOUT_TOOL_CHOICE = set()

TASK_KEY_MAP = {
    "pumpkin": "trace",
    "aeb": "full_trajectory",
    "aegis": "input",
    "aftraj": "turns",
    "agentracer": "history",
    "agentrx": "content",
    "custom_split": "steps",
    "exgentic": None,
    "new_traces": "trace",
    "nlile": "trace",
    "swebench": "spans",
    "trace_elephant": "step_records",
    "trail": "trace",
    "who_and_when": "history",
}


class ErrorAnnotation(BaseModel):
    error_key: str
    step: int
    justification: str


class TraceAnnotationResult(BaseModel):
    errors: List[ErrorAnnotation]


def load_trace(trace_file: Path, task: str) -> str:
    with trace_file.open("r", encoding="utf-8") as f:
        trace_data = json.load(f)

    key = TASK_KEY_MAP.get(task)
    if key is not None:
        if isinstance(trace_data, dict):
            trace_data = trace_data[key]
        else:
            raise ValueError(
                f"Expected dict with key '{key}' for task '{task}', got {type(trace_data).__name__}"
            )

    return json.dumps(trace_data, indent=2, ensure_ascii=False)


def _parse_json_from_text(text: str) -> dict:
    """Extract and normalize JSON from LLM text output."""
    import re
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()

    blobs: list = []
    i = 0
    while i < len(text):
        if text[i] in ("{", "["):
            open_ch = text[i]
            close_ch = "}" if open_ch == "{" else "]"
            depth = 0
            j = i
            while j < len(text):
                if text[j] == open_ch:
                    depth += 1
                elif text[j] == close_ch:
                    depth -= 1
                    if depth == 0:
                        try:
                            blobs.append(json.loads(text[i : j + 1]))
                        except json.JSONDecodeError:
                            pass
                        i = j + 1
                        break
                j += 1
            else:
                i += 1
        else:
            i += 1

    if not blobs:
        raise ValueError(f"No JSON found in output: {text[:300]}")

    errors = []
    for blob in blobs:
        if isinstance(blob, dict):
            if "errors" in blob and isinstance(blob["errors"], list):
                errors.extend(blob["errors"])
            elif "error_key" in blob:
                errors.append(blob)
        elif isinstance(blob, list):
            for item in blob:
                if isinstance(item, dict) and "error_key" in item:
                    errors.append(item)

    has_ideal = any(e.get("error_key") == "ideal" for e in errors)
    has_other = any(e.get("error_key") != "ideal" for e in errors)
    if has_ideal and has_other:
        errors = [e for e in errors if e.get("error_key") != "ideal"]
    if not errors and not has_ideal:
        errors = [{"error_key": "ideal", "step": -1, "justification": "No errors found in the trace."}]

    return {"errors": errors}


async def annotate_and_upload(
    trace_file: Path,
    output_dir: Path,
    model_name: str,
    trace_content: str,
    tags: list[str] | None = None,
):
    """Annotate a single trace with one model and upload to Langfuse."""
    output_file = output_dir / trace_file.name

    if output_file.exists():
        print(f"    {model_name}: already annotated, skipping.")
        return

    source_folder = trace_file.parent.name
    span_tags = list(tags) if tags else []

    if _langfuse_client is not None:
        with _langfuse_client.start_as_current_span(
            name=trace_file.stem,
            input={
                "system_prompt": SYSTEM_PROMPT,
                "trace_content": trace_content,
                "trace_file": str(trace_file),
                "trace_id": trace_file.stem,
                "model": model_name,
            },
            metadata={
                "trace_id": trace_file.stem,
                "model": model_name,
                "source_folder": source_folder,
            },
        ) as span:
            _langfuse_client.update_current_trace(tags=span_tags)
            try:
                annotation = await annotate_with_model(model_name, trace_content)
                annotation["model"] = model_name
                annotation["trace_file"] = str(trace_file)
                annotation["timestamp"] = datetime.now(timezone.utc).isoformat()

                with open(output_file, "w") as f:
                    json.dump(annotation, f, indent=2, ensure_ascii=False)

                span.update(output=annotation)

                n = len(annotation.get("errors", []))
                keys = [e.get("error_key", "?") for e in annotation.get("errors", [])]
                print(f"    {model_name}: {n} errors — {keys}")
            except Exception as e:
                span.update(output={"error": str(e)})
                print(f"    {model_name}: ERROR — {e}")
        _langfuse_client.flush()
    else:
        try:
            annotation = await annotate_with_model(model_name, trace_content)
            annotation["model"] = model_name
            annotation["trace_file"] = str(trace_file)
            annotation["timestamp"] = datetime.now(timezone.utc).isoformat()

            with open(output_file, "w") as f:
                json.dump(annotation, f, indent=2, ensure_ascii=False)

            n = len(annotation.get("errors", []))
            keys = [e.get("error_key", "?") for e in annotation.get("errors", [])]
            print(f"    {model_name}: {n} errors — {keys}")
        except Exception as e:
            print(f"    {model_name}: ERROR — {e}")


async def annotate_with_model(model_name: str, trace_content: str) -> dict:
    model = OpenRouterChatModel(
        model_name,
        provider="openrouter",
        settings={"temperature": 0.0},
    )

    if model_name in MODELS_WITHOUT_TOOL_CHOICE:
        agent: Agent = Agent(
            model=model,
            output_type=str,
            retries=3,
        )

        @agent.system_prompt
        def get_system_prompt() -> str:
            return SYSTEM_PROMPT + "\n\nIMPORTANT: Output ONLY a valid JSON object, no markdown fences, no extra text."

        result = await agent.run(
            f"Analyze the following MAS trace and identify all errors.\n\n"
            f"```json\n{trace_content}\n```"
        )
        parsed = _parse_json_from_text(result.output)
        return TraceAnnotationResult(**parsed).model_dump()
    else:
        agent: Agent[None, TraceAnnotationResult] = Agent(
            model=model,
            output_type=TraceAnnotationResult,
            retries=3,
        )

        @agent.system_prompt
        def get_system_prompt() -> str:
            return SYSTEM_PROMPT

        result = await agent.run(
            f"Analyze the following MAS trace and identify all errors.\n\n"
            f"```json\n{trace_content}\n```"
        )
        return result.output.model_dump()


async def process_trace(trace_file: Path, output_dir: Path, task: str, tags: list[str] | None = None):
    """Process a single trace file with all models."""
    try:
        trace_content = load_trace(trace_file, task)
    except Exception as e:
        print(f"  ERROR loading {trace_file.name}: {e}")
        return
    print(f"  Trace: {trace_file.name} ({len(trace_content):,} chars) [task={task}]")

    for model_name in MODELS:
        await annotate_and_upload(
            trace_file=trace_file,
            output_dir=output_dir,
            model_name=model_name,
            trace_content=trace_content,
            tags=tags,
        )


async def main(trace_dir: str, output_dir: str = "./gemini_flash_annotations", tags: list[str] | None = None):
    trace_dir = Path(trace_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not trace_dir.exists():
        print(f"Error: trace directory not found: {trace_dir}")
        return

    trace_files = sorted(trace_dir.rglob("*.json"))
    if not trace_files:
        print(f"No JSON files found in {trace_dir}")
        return

    print(f"Found {len(trace_files)} trace(s) in {trace_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Models: {MODELS}")
    print(f"Tags: {tags}")
    langfuse_status = "ON" if _langfuse_client else "OFF (keys not set)"
    print(f"Langfuse tracing: {langfuse_status}")
    print()

    for trace_file in trace_files:
        print(f"{'='*60}")
        rel_path = trace_file.relative_to(trace_dir)
        task = rel_path.parts[0]
        try:
            await process_trace(trace_file, output_dir, task=task, tags=tags)
        except Exception as e:
            print(f"  ERROR processing {trace_file}: {e}")
        print()

    if _langfuse_client:
        _langfuse_client.flush()

    print("Done.")


def _download_parquet_dataset(ds_info, output_dir):
    from datasets import load_dataset as hf_load_dataset

    print(f"Downloading {ds_info['hf_repo']} (split={ds_info['split']})...")
    ds = hf_load_dataset(ds_info["hf_repo"], split=ds_info["split"])
    print(f"Loaded {len(ds)} traces")

    written = 0
    skipped = 0
    for row in ds:
        session_id = row.get("session_id", "")
        if not session_id:
            skipped += 1
            continue
        filepath = output_dir / f"{session_id}.json"
        if filepath.exists():
            skipped += 1
            continue
        trace_data = {
            "harness": row.get("harness", ""),
            "benchmark": row.get("benchmark", ""),
            "models": row.get("models", []),
            "max_tokens": row.get("max_tokens", 0),
            "total_tokens": row.get("total_tokens", 0),
            "session_id": session_id,
            "spans": row.get("spans", []),
            "collected_at": row.get("collected_at", ""),
        }
        with open(filepath, "w") as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False)
        written += 1

    print(f"Done: {written} files written, {skipped} skipped (no session_id or already existed)")
    print(f"Total files in {output_dir}: {len(list(output_dir.glob('*.json')))}")


def _download_jsonl_grouped(ds_info, output_dir):
    from datasets import load_dataset as hf_load_dataset

    print(f"Downloading {ds_info['hf_repo']} (split={ds_info['split']})...")
    ds = hf_load_dataset(ds_info["hf_repo"], split=ds_info["split"])
    print(f"Loaded {len(ds)} rows, grouping by thread_id/session...")

    sessions = {}
    for row in ds:
        task_meta = row.get("task_metadata", "")
        if isinstance(task_meta, str):
            try:
                task_meta = json.loads(task_meta)
            except (json.JSONDecodeError, TypeError):
                task_meta = {}
        if not isinstance(task_meta, dict):
            task_meta = {}

        session_name = task_meta.get("session_name", "")
        task_id = task_meta.get("task_id", "")
        source_dataset = task_meta.get("source_dataset", "")

        if not session_name and not task_id:
            session_name = f"unknown_{row.get('thread_id', 'nothread')}"

        key = session_name or task_id
        if key not in sessions:
            sessions[key] = {
                "session_name": key,
                "task_id": task_id,
                "source_dataset": source_dataset,
                "spans": [],
            }

        span_entry = {
            "type": row.get("type", ""),
            "request_id": row.get("request_id", ""),
            "timestamp_rel_s": row.get("timestamp_rel_s"),
            "timestamp_utc": row.get("timestamp_utc", ""),
            "method": row.get("method", ""),
            "path": row.get("path", ""),
            "status_code": row.get("status_code"),
            "body": row.get("body", ""),
            "thread_id": row.get("thread_id"),
        }
        sessions[key]["spans"].append(span_entry)

    MAX_CHARS = 2_000_000
    written = 0
    too_large = 0
    for key, session_data in sessions.items():
        content = json.dumps(session_data, ensure_ascii=False)
        if len(content) > MAX_CHARS:
            too_large += 1
            continue
        safe_key = key.replace("/", "_").replace("\\", "_")
        filepath = output_dir / f"{safe_key}.json"
        if filepath.exists():
            continue
        with open(filepath, "w") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        written += 1

    print(f"Sessions: {len(sessions)}")
    print(f"Written: {written}")
    print(f"Skipped (too large >{MAX_CHARS} chars): {too_large}")
    print(f"Total files in {output_dir}: {len(list(output_dir.glob('*.json')))}")


if __name__ == "__main__":
    import argparse

    DOWNLOAD_DATASETS = {
        "exgentic_agent_llm_traces": {
            "hf_repo": "Exgentic/agent-llm-traces",
            "split": "train",
            "output_dir": "data/exgentic_agent_llm_traces",
        },
        "swebench_minimax_traces": {
            "hf_repo": "sammshen/swebench-minimax-traces",
            "split": "train",
            "output_dir": "data/swebench_minimax_traces",
            "format": "jsonl_grouped",
        },
    }

    parser = argparse.ArgumentParser(description="Annotate errors in MAS traces")
    parser.add_argument("--traces", default=None, help="Path to directory with raw trace JSON files (default: balanced_traces_1000_after/traces)")
    parser.add_argument("--output", default="./gemini_flash_annotations", help="Path to directory for annotation output")
    parser.add_argument("--tags", default=None, help="Comma-separated tags for Langfuse (e.g. v1)")
    parser.add_argument("--download", choices=list(DOWNLOAD_DATASETS.keys()), help="Download a dataset by name")
    args = parser.parse_args()

    tags = args.tags.split(",") if args.tags else None

    if args.download:
        ds_info = DOWNLOAD_DATASETS[args.download]
        output_dir = Path(__file__).resolve().parent.parent / ds_info["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)

        if ds_info.get("format") == "jsonl_grouped":
            _download_jsonl_grouped(ds_info, output_dir)
        else:
            _download_parquet_dataset(ds_info, output_dir)
    else:
        trace_dir = args.traces
        if not trace_dir:
            trace_dir = str(Path(__file__).resolve().parent.parent / "balanced_traces_1000_after" / "traces")
        asyncio.run(main(trace_dir, args.output, tags=tags))