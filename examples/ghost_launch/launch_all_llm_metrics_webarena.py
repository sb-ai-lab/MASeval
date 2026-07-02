"""Launch llm-metric, judge with WebArena traces"""

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")

from pydantic_ai.models.openai import OpenAIChatModel

from maseval.models import RawTraceInput

from maseval import get_langfuse_judge_client
from maseval.metrics import MetricType, create_metric
from maseval.models import Policy
from maseval.metrics.judge import Summarizer, SummarizerResult

from rich import print
from tqdm import tqdm

from maseval.prompts import (
    MAS_SUMMARIZER_PROMPT_BASE_TEMPLATE_BIN
)
import os
import json
from typing import Any, List

def load_all_traces(
    root_dir: str = "data/webarena_traces_reddit",
) -> List[Any]:
    data: List[Any] = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.lower().endswith(".json"):
                continue
            full_path = os.path.join(dirpath, filename)
            with open(full_path, "r", encoding="utf-8") as f:
                data.append(json.load(f))

    return data


async def main(gaia_eval, enable_tracing, result_file_name: str = "judge_res"):
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

    # Initialize judge client for uploading evaluation traces
    # IMPORTANT: Do this AFTER downloading traces and BEFORE running evaluations
    # This sets up the OpenTelemetry tracer to upload to the judge project
    if enable_tracing:
        print("\n=== Setting up judge client for evaluation traces ===")
        judge_client = get_langfuse_judge_client()
        print("Judge client initialized - evaluations will be traced to judge project")
    else:
        judge_client = None

    # Run evaluation for each task
    for t in tqdm(load_all_traces(), desc="Evaluating tasks: "):
        trace_data = t
        task = trace_data['id']
        
        print(f"\n=== Evaluating Task {task} ===")
        trace_data = str(trace_data["trajectory"])

        print(f"\n=== Evaluating Task {task} ===")
        eval_input = RawTraceInput(trace=trace_data)

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
            MetricType.MAS_TIME,
            MetricType.MAS_TOKENS,
            MetricType.AGENT_TIME,
            MetricType.AGENT_TOKENS,
        ]

        if MetricType.MAS_PLANNING not in llm_metrics_to_test:
            print("No planner found")

        llm_results = {}
        non_llm_results = {}

        # Prepare metadata for the trace
        trace_metadata = {
            "task_id": task,
        }

        # Add ground truth info if doing GAIA evaluation
        if gaia_eval:
                trace_metadata["ground_truth"] = None
                trace_metadata["mas_response"] = None
                trace_metadata["correct_answer"] = None

        # Create a parent span for all evaluations of this task
        # All metric evaluations will be grouped under this span
        if enable_tracing:
            with judge_client.start_as_current_span(
                name=f"evaluate_task_{task}",
                input={"task_id": task},
                metadata=trace_metadata,
            ) as span:
                # Update the trace with tags (tags are set at trace level, not span level)
                judge_client.update_current_trace(
                    tags=["maseval", f"task_id:{task}"]
                )
                for metric_type in llm_metrics_to_test:
                    if gaia_eval:
                        ground_truth_value = (
                            None
                        )

                    print(f"\n--- Evaluating LLM metric:{metric_type.value} ---")
                    try:
                        metric = create_metric(metric_type, model)
                        result = await metric.evaluate(eval_input)
                        llm_results[metric_type.value] = result

                        print(f"Metric: {result.metric_name}")
                        
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

                if gaia_eval:
                    # if 1: mas response matches gt;
                    # if 0: mas response does not match gt
                    serializable_results["gt"] = None
                    serializable_results["mas_answer"] = None
                    serializable_results["label_answer"] = None
                    serializable_results["id"] = task

                print(f"\n--- Evaluating Summarizer ---")
                try:
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
                        "observation_alignment",
                        "policy_alignment",
                        "state_consistency",
                        "task_completeness",
                        "tool_parameter_extraction",
                        "mas_task_transfer",
                        "mas_roles_distribution",
                        "tool_efficiency",                            
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

                except Exception as e:
                    print(f"Error evaluating summarizer: {e}")
                    continue

                # Save results to file per task
                output_dir = Path(__file__).parent / "scores"
                output_dir.mkdir(exist_ok=True)
                output_file = output_dir / (result_file_name + f"{task}.json")

                with open(output_file, "w") as f:
                    json.dump(serializable_results, f, indent=2)

                print(f"\nResults saved to: {output_file}")

                # Add ground truth comparison to output if available
                if (
                    gaia_eval
                    and hasattr(trace_data, "output")):
                    if (
                        "ground_truth" in trace_data.output
                        and "response" in trace_data.output
                    ):
                        span_output["ground_truth_comparison"] = {
                            "ground_truth": trace_data.output["ground_truth"],
                            "mas_response": trace_data.output["response"],
                            "match": trace_data.output["response"]
                            == trace_data.output["ground_truth"],
                        }

                span.update(output=span_output)
        else:
            print('Please, use langfuse tracing!')


if __name__ == "__main__":
    asyncio.run(
        main(
            gaia_eval=True,
            enable_tracing=True,
            # name="gaia_task_db0c3ed0-a4af-4442-bb6f-884d6da055cb",  # "gaia_task_07aac7b1-ffc3-4787-8e4c-7fb522156097"
            result_file_name = "web_arena_bench"
    ))
