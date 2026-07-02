"""Example usage of the MASeval library with balanced traces."""

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")

from pydantic_ai.models.openai import OpenAIChatModel

from maseval import get_langfuse_judge_client
from maseval.metrics import MetricType, create_metric
from maseval.metrics.judge import Summarizer, SummarizerResult
from maseval.models import RawTraceInput

from rich import print
from tqdm import tqdm

from maseval.prompts import (
    MAS_SUMMARIZER_PROMPT_BASE_TEMPLATE_BIN
)

sys.path.insert(0, str(Path(__file__).parent))


async def main(enable_tracing, result_folder_name: str = "judge_res"):
    """Main example demonstrating the library usage with Langfuse traces."""

    # Initialize the OpenRouter model
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key

    model = OpenAIChatModel(
        "google/gemini-2.5-flash",
        provider="openrouter",
        settings={
            "temperature": 0.0,
        },
    )

    # Step 1: Get trace files
    traces_dir = Path(__file__).resolve().parents[2] / "balanced_traces_1000_after" / "traces"
    trace_files = []

    for task_dir in sorted(path for path in traces_dir.iterdir() if path.is_dir()):
        if task_dir.name == "custom_split":
            for subtask_dir in sorted(path for path in task_dir.iterdir() if path.is_dir()):
                for trace_path in sorted(subtask_dir.glob("*.json")):
                    trace_files.append((task_dir.name, trace_path))
        else:
            for trace_path in sorted(task_dir.glob("*.json")):
                trace_files.append((task_dir.name, trace_path))

    # Step 2: Initialize judge client for uploading evaluation traces
    # IMPORTANT: Do this AFTER downloading traces and BEFORE running evaluations
    # This sets up the OpenTelemetry tracer to upload to the judge project
    if enable_tracing:
        print("\n=== Setting up judge client for evaluation traces ===")
        judge_client = get_langfuse_judge_client()
        print("Judge client initialized - evaluations will be traced to judge project")
    else:
        judge_client = None

    output_root = Path(__file__).parent / result_folder_name

    # Run evaluation for each trace
    for task, trace_path in tqdm(trace_files, desc="Evaluating traces: "):
        trace_id = str(trace_path.relative_to(traces_dir))
        output_file = output_root / trace_path.relative_to(traces_dir)

        if output_file.is_file():
            print(f"\n=== Skipping already evaluated trace {trace_id} ===")
            print(f"Results already exist at: {output_file}")
            continue

        print(f"\n=== Evaluating Task {trace_id} ===")
        with trace_path.open("r", encoding="utf-8") as f:
            trace_data = json.load(f)

        if task == "pumpkin":
            trace_data = trace_data["trace"]
        elif task == "aeb":
            trace_data = trace_data["full_trajectory"]
        elif task == "aegis":
            trace_data = trace_data["input"]
        elif task == "aftraj":
            trace_data = trace_data["turns"]
        elif task == "agentracer":
            trace_data = trace_data["history"]
        elif task == "agentrx":
            trace_data = trace_data["content"]
        elif task == "custom_split":
            trace_data = trace_data["steps"]
        elif task == "exgentic":
            trace_data = trace_data
        elif task == "new_traces":
            trace_data = trace_data
        elif task == "nlile":
            trace_data = trace_data["trace"]
        elif task == "swebench":
            trace_data = trace_data["spans"]
        elif task == "trace_elephant":
            trace_data = trace_data["step_records"]
        elif task == "trail":
            trace_data = trace_data["trace"]
        elif task == "who_and_when":
            trace_data = trace_data["history"]

        eval_input = RawTraceInput(
            trace=json.dumps(trace_data, ensure_ascii=False, default=str)
        )

        print("\n=== Running Metric Evaluations ===")

        # Choose metrics you want to run
        llm_metrics_to_test = [
            MetricType.OBSERVATION_ALIGNMENT,
            MetricType.POLICY_ALIGNMENT,
            MetricType.STATE_CONSISTENCY,
            MetricType.TASK_COMPLETENESS,
            MetricType.TOOL_SELECTION,
            MetricType.TOOL_PARAMETER_EXTRACTION,
            MetricType.MAS_COMPLEXITY,
            MetricType.MAS_TASK_TRANSFER,
            MetricType.MAS_ROLES_DISTRIBUTION,
            MetricType.MAS_TASK_COMPLETION,
            MetricType.MAS_API_ISSUES,
            MetricType.MAS_ENVIRONMENT_SETUP_ERRORS,
            MetricType.PROMPT_QUALITY,
            MetricType.TOOL_PERFORMANCE,
        ]

        non_llm_metrics_to_test = []

        llm_results = {}
        non_llm_results = {}

        # Prepare metadata for the trace
        trace_metadata = {
            "task_id": task,
        }

        # Create a parent span for all evaluations of this task
        # All metric evaluations will be grouped under this span
        if enable_tracing:
            with judge_client.start_as_current_span(
                name=f"evaluate_task_{task}",
                input={"task_id": task, "trace_id": trace_id},
                metadata=trace_metadata,
            ) as span:
                # Update the trace with tags (tags are set at trace level, not span level)
                judge_client.update_current_trace(
                    tags=["maseval", f"task_id:{task}"]
                )
                for metric_type in llm_metrics_to_test:
                    print(f"\n--- Evaluating LLM metric:{metric_type.value} ---")
                    try:
                        metric = create_metric(metric_type, model)
                        result = await metric.evaluate(eval_input)
                        llm_results[metric_type.value] = result

                        print(f"Metric: {result.metric_name}")
                        print(f"Number of scores: {len(result.scores)}")

                        for score in result.scores:
                            print(f"  - Item {score.item_id}: {score.score.value}")
                            print(
                                f"    Justification: {score.justification[:100]}..."
                            )
                    except Exception as e:
                        print(f"Error evaluating {metric_type.value}: {e}")
                        continue

                for metric_type in non_llm_metrics_to_test:
                    print(
                        f"\n--- Evaluating Non-LLM metric:{metric_type.value} ---"
                    )
                    try:
                        metric = create_metric(metric_type)
                        result = await metric.evaluate(eval_input)
                        non_llm_results[metric_type.value] = result
                    except Exception as e:
                        print(f"Error evaluating {metric_type.value}: {e}")
                        continue

                # Update span with results summary
                span_output = {
                    "results_summary": {
                        metric_name: {
                            "total_scores": len(result.scores),
                            "ideal": sum(
                                1 for s in result.scores if s.score.value == "ideal"
                            ),
                            "fair": sum(
                                1 for s in result.scores if s.score.value == "fair"
                            ),
                            "poor": sum(
                                1 for s in result.scores if s.score.value == "poor"
                            ),
                        }
                        for metric_name, result in llm_results.items()
                    }
                }

                serializable_results = {
                    metric_name: {
                        "metric_name": result.metric_name,
                        "scores": [
                            {
                                "item_id": score.item_id,
                                "score": score.score.value,
                                "justification": score.justification,
                            }
                            for score in result.scores
                        ],
                    }
                    for metric_name, result in llm_results.items()
                }

                for metric_name, result in non_llm_results.items():
                    serializable_results[metric_name] = {
                        "metric_name": metric_name,
                        "scores": [
                            {
                                "item_id": "overall_score",
                                "score": result,
                                "justification": None,
                            }
                        ],
                    }

                print(f"\n--- Evaluating Summarizer ---")
                try:
                    summarizer = Summarizer(
                        model,
                        MAS_SUMMARIZER_PROMPT_BASE_TEMPLATE_BIN,
                        SummarizerResult,
                    )

                    summarizer_results = await summarizer.evaluate(serializable_results)

                    serializable_results["summarizer_score"] = {
                        "metric_name": "summarizer_score",
                        "scores": [
                            {
                                "item_id": "overall_score",
                                "score": summarizer_results.score,
                                "justification": summarizer_results.justification,
                            }
                        ],
                    }

                    print(
                        f"Summary:\nMAS evaluation score: {summarizer_results.score}\nJustification: {summarizer_results.justification}"
                    )

                except Exception as e:
                    print(f"Error evaluating summarizer: {e}")
                    continue

                # Save results using the same folder structure as the input traces.
                output_file.parent.mkdir(parents=True, exist_ok=True)

                with output_file.open("w", encoding="utf-8") as f:
                    json.dump(serializable_results, f, indent=2)

                print(f"\nResults saved to: {output_file}")

                span.update(output=span_output)
        else:
            print("Please, use langfuse tracing!")


if __name__ == "__main__":
    asyncio.run(
        main(
            enable_tracing=True,
    ))
