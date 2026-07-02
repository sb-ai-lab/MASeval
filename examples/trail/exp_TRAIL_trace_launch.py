import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")
import time

from pydantic_ai.models.openai import OpenAIChatModel

from maseval import get_langfuse_judge_client
from maseval.metrics import MetricType, create_metric
from maseval.metrics.judge import Summarizer, SummarizerResult
from maseval.models import RawTraceInput
from rich import print
from tqdm import tqdm

from maseval import get_langfuse_judge_client
from maseval.metrics import MetricType, create_metric
from maseval.prompts import \
    MAS_SUMMARIZER_PROMPT_BASE_TEMPLATE_BIN  # uncommit if need; MAS_SUMMARIZER_PROMPT_BASE_TEMPLATE,
from maseval.metrics.judge import Summarizer, SummarizerResult

sys.path.insert(0, str(Path(__file__).parent))


async def main(
    enable_tracing, trace_dir: Path, anno_dir: Path, dir_to_save=Path("mad_results")
):
    """Run metrics on TRAIL datadset"""

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

    # Step 1: Get Langfuse client for downloading traces
    # This uses LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY
    print("=== Downloading traces from evaluation project ===")

    data_dir = Path(trace_dir)
    all_traces = [json.load(open(f)) for f in data_dir.glob("*.json")]

    def safe_load(f):
        try:
            with open(f, "r", encoding="utf-8") as file:
                return json.load(file)
        except:
            return {}

    anno = [safe_load(f) for f in Path(anno_dir).glob("*.json")]

    # Step 2: Initialize judge client for uploading evaluation traces
    # IMPORTANT: Do this AFTER downloading traces and BEFORE running evaluations
    # This sets up the OpenTelemetry tracer to upload to the judge project
    if enable_tracing:
        print("\n=== Setting up judge client for evaluation traces ===")
        judge_client = get_langfuse_judge_client()
        print("Judge client initialized - evaluations will be traced to judge project")
    else:
        judge_client = None

    cnt = 0

    # Run evaluation for each task
    for task in tqdm(all_traces, desc="Evaluating tasks: "):

        cnt += 1
        if cnt < 81:
            continue
        trace_data = str(task["spans"])
        annotation = [
            i for i in anno if i.get("trace_id", "") == task["spans"][0]["trace_id"]
        ]

        if annotation == []:
            continue
        annotation = annotation[0]

        print(f"\n=== Evaluating Task {task['trace_id']} ===")
        eval_input = RawTraceInput(trace=trace_data)

        if eval_input.agents_errors != None:
            print("Task has agents errors")
            print(f"Agents errors: {eval_input.agents_errors}")
        else:
            print("\n=== Running Metric Evaluations ===")

            # Choose metrics you want to run
            llm_metrics_to_test = [
                MetricType.TOOL_SELECTION,
                MetricType.MAS_COMPLEXITY,
                MetricType.MAS_TASK_COMPLETION,
            ]

            non_llm_metrics_to_test = [
                MetricType.TOOL_EFFICIENCY,
            ]

            llm_results = {}
            non_llm_results = {}

            # Prepare metadata for the trace
            trace_metadata = {
                "trace_id": task["trace_id"],
                "ЕTRAIL_trace": task["spans"],
            }

            # Create a parent span for all evaluations of this task
            # All metric evaluations will be grouped under this span
            with judge_client.start_as_current_span(
                name=f"evaluate_task_{task['trace_id']}",
                input={"trace_id": task["trace_id"]},
                metadata=trace_metadata,
            ) as span:
                # Update the trace with tags (tags are set at trace level, not span level)
                judge_client.update_current_trace(
                    tags=["maseval", f"task_id:{task['trace_id']}"]
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
                            print(f"    Justification: {score.justification[:100]}...")
                    except Exception as e:
                        print(f"Error evaluating {metric_type.value}: {e}")
                        continue

                for metric_type in non_llm_metrics_to_test:
                    print(f"\n--- Evaluating Non-LLM metric:{metric_type.value} ---")
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

                summarizer = Summarizer(
                    model,
                    MAS_SUMMARIZER_PROMPT_BASE_TEMPLATE_BIN,
                    SummarizerResult,
                )

                modified_results = serializable_results.copy()
                keys_to_remove = [
                    "gt",
                    "label_answer",
                    "mas_answer",
                    "mas_time",
                    "mas_tokens",
                    "agent_tokens",
                    "agent_time",
                    "tool_efficiency",
                ]

                for key in keys_to_remove:
                    modified_results.pop(key, None)

                with judge_client.start_as_current_span(
                    name=f"evaluate_summarizer_{task['trace_id']}",
                    input={"metric_results": list(modified_results.keys())},
                ) as summarizer_span:
                    try:
                        summarizer_results = await summarizer.evaluate(modified_results)

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

                        summarizer_span.update(
                            output={
                                "score": summarizer_results.score,
                                "justification": summarizer_results.justification,
                            }
                        )
                        time.sleep(3)
                    except Exception as e:
                        print(f"Error evaluating summarizer: {e}")
                        import traceback

                        traceback.print_exc()

                        summarizer_span.update(output={"error": str(e)})
                        time.sleep(3)

                    print(
                        f"Summary:\nMAS evaluation score: {summarizer_results.score}\nJustification: {summarizer_results.justification}"
                    )

                serializable_results["annotation"] = annotation

                # Save results to file per task
                output_dir = dir_to_save
                output_dir.mkdir(exist_ok=True)
                output_file = (
                    output_dir / f"evaluation_results_{task['trace_id']}_{cnt}.json"
                )

                with open(output_file, "w") as f:
                    json.dump(serializable_results, f, indent=2)

                print(f"\nResults saved to: {output_file}")

                span.update(output=span_output)
                time.sleep(3)

            for metric_name, result in non_llm_results.items():
                print(f"{metric_name}: {result}")

            judge_client.flush()


if __name__ == "__main__":
    asyncio.run(
        main(
            enable_tracing=True,
            trace_dir="/home/alina/Desktop/maseval-research/trail_benchmark/benchmarking copy/data/GAIA",
            anno_dir="/home/alina/Desktop/maseval-research/trail_benchmark/benchmarking copy/processed_annotations_gaia",
            dir_to_save=Path(__file__).parent / "trail_all_3score_30_10_25",
        )
    )
