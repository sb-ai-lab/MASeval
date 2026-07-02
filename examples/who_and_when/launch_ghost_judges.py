import asyncio
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env")

from maseval.models import RawTraceInput

from pydantic_ai.models.openai import OpenAIChatModel
from rich import print

from maseval import get_langfuse_judge_client
from maseval.metrics import MetricType, create_metric
from maseval.metrics.judge import Summarizer, SummarizerResult
from maseval.prompts import MAS_SUMMARIZER_PROMPT_BASE_TEMPLATE_BIN
import os
os.environ["OTEL_TRACES_EXPORTER"] = "none"

from opentelemetry import trace
trace.set_tracer_provider(None) 

async def main(
    model_name,
    gaia_eval,
    enable_tracing,
    df,
    keys_to_remove,
    result_file_name: str = "judge_res",
    folder_name: str = "score",
    from_idx: int = 0
):
    """Launch ghost judges for Who and When dataset using specified model and metrics"""

    # Initialize the OpenRouter model
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key

    model = OpenAIChatModel(
        model_name,
        provider="openrouter",
        settings={
            "temperature": 0.0,
        },
    )

    # Initialize judge client for uploading evaluation traces
    if enable_tracing:
        print("\n=== Setting up judge client for evaluation traces ===")
        judge_client = get_langfuse_judge_client()
        print("Judge client initialized - evaluations will be traced to judge project")
    else:
        judge_client = None

    cnt = 0

    # Run evaluation for each task
    for task in range(len(df)):
        cnt += 1
        if cnt < from_idx:
            continue
        eval_input = {
            "history": df.iloc[task]["history"],
            "question": df.iloc[task]["question"],
        }
        eval_input = RawTraceInput(trace=str(eval_input))
        print(f"\n=== Evaluating Task {task} ===")

        # Choose metrics you want to run
        llm_metrics_to_test = [
            # MetricType.OBSERVATION_ALIGNMENT,
            # MetricType.POLICY_ALIGNMENT,
            # MetricType.STATE_CONSISTENCY,
            # MetricType.TASK_COMPLETENESS,
            MetricType.TOOL_SELECTION,
            # MetricType.TOOL_PARAMETER_EXTRACTION,
            MetricType.MAS_COMPLEXITY,
            # MetricType.MAS_TASK_TRANSFER,
            # MetricType.MAS_ROLES_DISTRIBUTION,
            MetricType.MAS_TASK_COMPLETION,
            MetricType.MAS_API_ISSUES,
            MetricType.MAS_ENVIRONMENT_SETUP_ERRORS,
            # MetricType.PROMPT_QUALITY,
            MetricType.TOOL_PERFORMANCE
        ]

        non_llm_metrics_to_test = [
            MetricType.TOOL_EFFICIENCY,
        ]

        llm_results = {}
        non_llm_results = {}

        # Prepare metadata for the trace
        trace_metadata = {
            "task_id": df.iloc[task]["question_ID"],
            "trace_id": task,
        }

        if 'groundtruth' in df.iloc[task].keys():
            trace_metadata["ground_truth"] = df.iloc[task]["groundtruth"]
            trace_metadata["correct_answer"] = df.iloc[task]["is_corrected"]
        else:
            trace_metadata["ground_truth"] = df.iloc[task]["ground_truth"]
            trace_metadata["correct_answer"] = df.iloc[task]["is_correct"]

        # Create a parent span for all evaluations of this task
        # All metric evaluations will be grouped under this span
        if enable_tracing:
            with judge_client.start_as_current_span(
                name=f"evaluate_task_{task}",
                input={"task_id": task, "trace_id": df.iloc[task]["question_ID"]},
                metadata=trace_metadata,
            ) as span:
                # Update the trace with tags (tags are set at trace level, not span level)
                judge_client.update_current_trace(tags=["maseval", f"task_id:{task}"])
                for metric_type in llm_metrics_to_test:
                    if gaia_eval:
                        ground_truth_value = trace_metadata["correct_answer"]
                        try:
                            ground_truth_value = int(ground_truth_value)
                        except:
                            continue

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

                if gaia_eval:
                    # if 1: mas response matches gt;
                    # if 0: mas response does not match gt
                    serializable_results["gt"] = ground_truth_value
                    serializable_results["label_answer"] = trace_metadata["ground_truth"]

                print(f"\n--- Evaluating Summarizer ---")
                try:
                    summarizer = Summarizer(
                        model,
                        MAS_SUMMARIZER_PROMPT_BASE_TEMPLATE_BIN,
                        SummarizerResult,
                    )

                    modified_results = serializable_results.copy()

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
                output_dir = Path(__file__).parent / folder_name
                output_dir.mkdir(exist_ok=True)
                output_file = output_dir / (result_file_name + f"{task}.json")

                with open(output_file, "w") as f:
                    json.dump(serializable_results, f, indent=2)

                print(f"\nResults saved to: {output_file}")

                span_output["ground_truth_comparison"] = {
                    "ground_truth": trace_metadata["ground_truth"],
                    "match": trace_metadata["correct_answer"],
                }
                span.update(output=span_output)
        else:
            print("Please, use langfuse tracing!")


if __name__ == "__main__":
    df_handcrafted = pd.read_parquet("hf://datasets/Kevin355/Who_and_When/Hand-Crafted.parquet")
    # df_algorithm = pd.read_parquet("hf://datasets/Kevin355/Who_and_When/Algorithm-Generated.parquet")

    asyncio.run(
        main(
            model_name="openai/gpt-4o",
            # model_name="google/gemini-2.5-flash",
            gaia_eval=True,
            enable_tracing=True,
            result_file_name="gpt4o_",
            folder_name="who&when_hand_gpt4o_6ens",
            df = df_handcrafted,
            keys_to_remove = [
                "gt",
                "label_answer",
                "mas_answer",
                "mas_time",
                "mas_tokens",
                "agent_tokens",
                "agent_time",
                # #
                "observation_alignment",
                "policy_alignment",
                "state_consistency",
                "task_completeness",
                "tool_parameter_extraction",
                "mas_task_transfer",
                "mas_roles_distribution",
                "tool_efficiency",
                "prompt_quality"
            ],
            from_idx=0
        )
    )
