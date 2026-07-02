import argparse
import ast
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from rich import print
from tqdm import tqdm


load_dotenv(".env")


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIL_RUN_EVAL = REPO_ROOT / "trail_benchmark" / "benchmarking" / "run_eval.py"
DEFAULT_TRACES_DIR = REPO_ROOT / "balanced_traces_1000_after" / "traces"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "trail_balanced_traces_1000_judge_res"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

TRACE_FIELD_BY_BENCHMARK = {
    "aeb": "full_trajectory",
    "aegis": "input",
    "aftraj": "turns",
    "agentracer": "history",
    "agentrx": "content",
    "custom_split": "steps",
    "nlile": "trace",
    "pumpkin": "trace",
    "swebench": "spans",
    "trace_elephant": "step_records",
    "trail": "trace",
    "who_and_when": "history",
}


def iter_trace_files(traces_dir: Path):
    for trace_path in sorted(traces_dir.rglob("*.json")):
        relative_path = trace_path.relative_to(traces_dir)
        benchmark = relative_path.parts[0]
        yield benchmark, relative_path, trace_path


def extract_trace_payload(trace_data, benchmark: str):
    field_name = TRACE_FIELD_BY_BENCHMARK.get(benchmark)

    if benchmark in ["exgentic", "new_traces"]:
        return trace_data

    if field_name is None:
        raise KeyError(f"No trace payload field configured for benchmark: {benchmark}")

    return trace_data[field_name]


def parse_model_json(content: str):
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("\n```", 1)[0]
    return json.loads(content)


def load_trail_prompt_template() -> str:
    source = TRAIL_RUN_EVAL.read_text(encoding="utf-8")
    module = ast.parse(source)

    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "get_prompt":
            for statement in node.body:
                if (
                    isinstance(statement, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id == "prompt"
                        for target in statement.targets
                    )
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                ):
                    return statement.value.value

    raise RuntimeError(f"Could not load TRAIL prompt template from {TRAIL_RUN_EVAL}")


def call_openrouter(
    prompt_template: str,
    trace: str,
    model: str,
    max_tokens: int,
    timeout_seconds: float,
):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    base_url = os.getenv("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL).rstrip("/")
    prompt = prompt_template.format(trace=trace)
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter request failed ({e.code}): {error_body}") from e

    response_json = json.loads(response_body)
    return response_json["choices"][0]["message"]["content"]


def evaluate_balanced_traces(
    traces_dir: Path,
    output_dir: Path,
    model: str,
    sleep_seconds: float,
    max_tokens: int,
    timeout_seconds: float,
    limit: int | None = None,
):
    trace_files = list(iter_trace_files(traces_dir))
    if limit is not None:
        trace_files = trace_files[:limit]

    print(f"Found {len(trace_files)} trace files in {traces_dir}")
    prompt_template = load_trail_prompt_template()

    for benchmark, relative_path, trace_path in tqdm(trace_files, desc="Evaluating traces: "):
        output_file = output_dir / relative_path
        trace_id = str(relative_path.with_suffix(""))

        if output_file.is_file():
            print(f"\n=== Skipping already evaluated trace {relative_path} ===")
            continue

        print(f"\n=== Evaluating Task {relative_path} ===")

        try:
            with trace_path.open("r", encoding="utf-8") as f:
                raw_trace = json.load(f)

            trace_payload = extract_trace_payload(raw_trace, benchmark)
            content = call_openrouter(
                prompt_template,
                json.dumps(trace_payload, ensure_ascii=False, default=str),
                model=model,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
            )
            result_dict = parse_model_json(content)

            serializable_result = {
                "trace_id": raw_trace.get("trace_id", trace_id)
                if isinstance(raw_trace, dict)
                else trace_id,
                "benchmark": benchmark,
                "source_file": str(relative_path),
                "scores": result_dict["scores"][0],
                "errors": result_dict["errors"],
            }

            output_file.parent.mkdir(parents=True, exist_ok=True)
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(serializable_result, f, indent=2, ensure_ascii=False)

            print(f"Results saved to: {output_file}")
            if sleep_seconds:
                time.sleep(sleep_seconds)
        except Exception as e:
            print(f"Something went wrong for {relative_path}: {e}")
            continue


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the TRAIL judge over balanced_traces_1000/traces."
    )
    parser.add_argument("--traces-dir", type=Path, default=DEFAULT_TRACES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="google/gemini-2.5-flash")
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_balanced_traces(
        traces_dir=args.traces_dir,
        output_dir=args.output_dir,
        model=args.model,
        sleep_seconds=args.sleep_seconds,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        limit=args.limit,
    )
