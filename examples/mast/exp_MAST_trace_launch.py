"""Example usage of the MASeval library with Langfuse traces."""

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")

from pydantic_ai.models.openai import OpenAIChatModel

from maseval import get_langfuse_download_client, get_langfuse_judge_client
from maseval.metrics import MetricType, create_metric
from maseval.models import Policy
from maseval.metrics.planner_checker import PlannerChecker, PlannerCheckerResults
from maseval.metrics.judge import Summarizer, SummarizerResult
from maseval.models import RawTraceInput
from rich import print
from tqdm import tqdm

from maseval import get_langfuse_download_client, get_langfuse_judge_client
from maseval.metrics import MetricType, create_metric
from maseval.models import Policy, RowTraceInput
from maseval.metrics.planner_checker import PlannerChecker, PlannerCheckerResults
from maseval.prompts import (MAS_SUMMARIZER_PROMPT_BASE_TEMPLATE,
                             PLANNER_IDENTIFICATION_PROMPT_BASE_TEMPLATE)
from maseval.metrics.judge import Summarizer, SummarizerResult

sys.path.insert(0, str(Path(__file__).parent))


async def main(
    mad_traces, enable_tracing, is_check_ideal=True, dir_to_save=Path("mad_results")
):
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
    lf = get_langfuse_download_client()

    with open(mad_traces, "r", encoding="utf-8") as f:
        all_traces = json.load(f)

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
        if is_check_ideal:
            if task["mast_annotation"] != {
                "1.1": 0,
                "1.2": 0,
                "1.3": 0,
                "1.4": 0,
                "1.5": 0,
                "2.1": 0,
                "2.2": 0,
                "2.3": 0,
                "2.4": 0,
                "2.5": 0,
                "2.6": 0,
                "3.1": 0,
                "3.2": 0,
                "3.3": 0,
            }:
                continue
        else:
            if sum(task["mast_annotation"].values()) < 6:
                continue
        cnt += 1
        trace_data = task["trace"]["trajectory"]
        annotation = task["mast_annotation"]

        print(f"\n=== Evaluating Task {task['trace_id']} ===")
        eval_input = RawTraceInput(trace=trace_data)

        if eval_input.agents_errors != None:
            print("Task has agents errors")
            print(f"Agents errors: {eval_input.agents_errors}")
        else:
            planner_checker = PlannerChecker(
                model,
                PLANNER_IDENTIFICATION_PROMPT_BASE_TEMPLATE,
                PlannerCheckerResults,
            )
            planner_results = await planner_checker.evaluate(eval_input)

            # Add some example policies for policy alignment testing
            eval_input.policies = [
                Policy(
                    policy_id="helpful_policy",
                    description="Agent should be helpful and provide useful information",
                    requirements=[
                        "Provide accurate information",
                        "Be responsive to user needs",
                    ],
                ),
                Policy(
                    policy_id="safety_policy",
                    description="Agent should not provide harmful information",
                    requirements=["Avoid harmful content", "Maintain user safety"],
                ),
            ]

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
            ]

            non_llm_metrics_to_test = [
                MetricType.TOOL_EFFICIENCY,
            ]

            print(f"Checking for planners...")
            for result in planner_results.agents:
                if result.is_planner:
                    llm_metrics_to_test.append(MetricType.MAS_PLANNING)
                    print(f"Planner found: {result.agent_name}")
                    break

            if MetricType.MAS_PLANNING not in llm_metrics_to_test:
                print("No planner found")

            llm_results = {}
            non_llm_results = {}

            # Prepare metadata for the trace
            trace_metadata = {
                "trace_id": task["trace_id"],
                "MAST_trace": task["trace"]["trajectory"],
            }

            # Create a parent span for all evaluations of this task
            # All metric evaluations will be grouped under this span
            if enable_tracing:
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

                    summarizer = Summarizer(
                        model,
                        MAS_SUMMARIZER_PROMPT_BASE_TEMPLATE,
                        SummarizerResult,
                    )

                    modified_results = serializable_results.copy()

                    keys_to_remove = [
                        "mas_time",
                        "mas_tokens",
                        "agent_tokens",
                        "agent_time",
                    ]

                    for key in keys_to_remove:
                        modified_results.pop(key, None)

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
                    except:
                        pass

                    serializable_results["annotation"] = annotation

                    # Save results to file per task
                    output_dir = dir_to_save
                    output_dir.mkdir(exist_ok=True)
                    output_file = (
                        output_dir
                        / f"evaluation_results_{task['trace_id']}_{task['trace']['index']}_{cnt}.json"
                    )

                    with open(output_file, "w") as f:
                        json.dump(serializable_results, f, indent=2)

                    print(f"\nResults saved to: {output_file}")

                    span.update(output=span_output)
            else:
                for metric_type in llm_metrics_to_test:
                    print(f"\n--- Evaluating LLM metric:{metric_type.value} ---")
                    try:
                        metric = create_metric(metric_type, model)
                        result = await metric.evaluate(eval_input)
                        llm_results[metric_type.value] = result
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
                    MAS_SUMMARIZER_PROMPT_BASE_TEMPLATE,
                    SummarizerResult,
                )

                modified_results = serializable_results.copy()

                keys_to_remove = [
                    "mas_time",
                    "mas_tokens",
                    "agent_tokens",
                    "agent_time",
                ]

                for key in keys_to_remove:
                    modified_results.pop(key, None)

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

                serializable_results["annotation"] = annotation

                # Save results to file per task
                output_dir = dir_to_save
                output_dir.mkdir(exist_ok=True)
                output_file = (
                    output_dir
                    / f"evaluation_results_{task['trace_id']}_{task['trace']['index']}_{cnt}.json"
                )

                with open(output_file, "w") as f:
                    json.dump(serializable_results, f, indent=2)

                print(f"\nResults saved to: {output_file}")

            print("\n=== Evaluation Summary ===")
            for metric_name, result in llm_results.items():
                total_scores = len(result.scores)
                ideal_scores = sum(1 for s in result.scores if s.score.value == "ideal")
                fair_scores = sum(1 for s in result.scores if s.score.value == "fair")
                poor_scores = sum(1 for s in result.scores if s.score.value == "poor")

                print(f"{metric_name}:")
                print(
                    f"  Total: {total_scores}, Ideal: {ideal_scores}, fair: {fair_scores}, Poor: {poor_scores}"
                )

            for metric_name, result in non_llm_results.items():
                print(f"{metric_name}: {result}")


if __name__ == "__main__":
    asyncio.run(
        main(
            mad_traces="data/mad_dataset.json",
            enable_tracing=True,
            is_check_ideal=False,
            dir_to_save=Path(__file__).parent / "mast_results_bad_anno",
        )
    )
