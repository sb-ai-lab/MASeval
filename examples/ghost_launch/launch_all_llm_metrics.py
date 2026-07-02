"""Main example usage of the MASeval library with Langfuse traces."""

import asyncio
import json
import os
from pathlib import Path
import time
from dotenv import load_dotenv

load_dotenv(".env")

from pydantic_ai.models.openai import OpenAIChatModel

from maseval import get_langfuse_download_client, get_langfuse_judge_client
from maseval.metrics import MetricType, create_metric
from maseval.models import Policy
from maseval.metrics.planner_checker import PlannerChecker, PlannerCheckerResults
from maseval.metrics.judge import Summarizer, SummarizerResult

from rich import print
from tqdm import tqdm

from maseval.prompts import (
    PLANNER_IDENTIFICATION_PROMPT_BASE_TEMPLATE,
    MAS_SUMMARIZER_PROMPT_BASE_TEMPLATE_BIN
)

from maseval.parsers.langfuse_parser_v3 import parse_langfuse_task


async def main(gaia_eval, enable_tracing, name, name_v2 = None, result_file_name: str = "judge_res", folder_name: str = "score"):
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

    # Step 1: Get Langfuse client for downloading traces
    # This uses LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY
    print("=== Downloading traces from evaluation project ===")
    lf = get_langfuse_download_client()

    traces_page1 = lf.api.trace.list(name=name, limit=100, page=1)
    traces_page2 = lf.api.trace.list(name=name, limit=100, page=2)
    if name_v2:
        trace_v2_page1 = lf.api.trace.list(name=name_v2, limit=50, page=1)
        trace_v2_page2 = lf.api.trace.list(name=name_v2, limit=50, page=2)

        all_traces = traces_page1.data + traces_page2.data + trace_v2_page1.data + trace_v2_page2.data
    else:
        all_traces = traces_page1.data + traces_page2.data
        
    task_ids = [item.id for item in all_traces]

    if not task_ids:
        raise ValueError(f"No tasks found in trace {name}")

    print(f"Found {len(task_ids)} tasks in trace {name}")

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
    for task in tqdm(task_ids, desc="Evaluating tasks: "):
        succ = False
        while not(succ):
            try:
                trace_data = lf.api.trace.get(task)
                succ = True
            except:
                time.sleep(10)
                pass
            
        print(f"\n=== Evaluating Task {task} ===")
        eval_input = parse_langfuse_task(trace_data)
        
        time.sleep(10)

        if eval_input.agents_errors != None:
            print("Task has agents errors")
            print(f"Agents errors: {eval_input.agents_errors}")
        else:
            if eval_input.agents_pool:
                planner_checker = PlannerChecker(
                    model,
                    PLANNER_IDENTIFICATION_PROMPT_BASE_TEMPLATE,
                    PlannerCheckerResults,
                )
                planner_results = await planner_checker.evaluate(eval_input)

            print("=== Parsed Task Information ===")
            print(f"Trace ID: {eval_input.trace_id}")
            print(f"Dialogue messages: {len(eval_input.dialogue_history)}")
            print(f"Agent responses: {len(eval_input.agent_responses)}")
            print(f"Agent states: {len(eval_input.agent_states)}")

            print("\n=== Dialogue History ===")
            for i, msg in enumerate(eval_input.dialogue_history):
                print(f"{i+1}. {msg.role}: {msg.content[:100]}...")

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
                "task_id": task,
                "trace_id": eval_input.trace_id,
                "dialogue_messages": len(eval_input.dialogue_history),
                "agent_responses": len(eval_input.agent_responses),
                "agent_states": len(eval_input.agent_states),
            }

            # Add ground truth info if doing GAIA evaluation
            if gaia_eval and hasattr(trace_data, "output") and trace_data.output:
                if (
                    "ground_truth" in trace_data.output
                    and "response" in trace_data.output
                ):
                    trace_metadata["ground_truth"] = trace_data.output["ground_truth"]
                    trace_metadata["mas_response"] = trace_data.output["response"]
                    trace_metadata["correct_answer"] = (
                        trace_data.output["response"]
                        == trace_data.output["ground_truth"]
                    )

            # Create a parent span for all evaluations of this task
            # All metric evaluations will be grouped under this span
            if enable_tracing:
                with judge_client.start_as_current_span(
                    name=f"evaluate_task_{task}",
                    input={"task_id": task, "trace_id": eval_input.trace_id},
                    metadata=trace_metadata,
                ) as span:
                    # Update the trace with tags (tags are set at trace level, not span level)
                    judge_client.update_current_trace(
                        tags=["maseval", f"task_id:{task}"]
                    )
                    for metric_type in llm_metrics_to_test:
                        if gaia_eval:
                            ground_truth_value = (
                                trace_data.output["response"]
                                == trace_data.output["ground_truth"]
                            )

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

                    if gaia_eval:
                        # if 1: mas response matches gt;
                        # if 0: mas response does not match gt
                        serializable_results["gt"] = int(ground_truth_value)
                        serializable_results["mas_answer"] = trace_data.output[
                            "response"
                        ]
                        serializable_results["label_answer"] = trace_data.output[
                            "ground_truth"
                        ]

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
                    output_dir = Path(__file__).parent / folder_name
                    output_dir.mkdir(exist_ok=True)
                    output_file = output_dir / (result_file_name + f"{task}.json")

                    with open(output_file, "w") as f:
                        json.dump(serializable_results, f, indent=2)

                    print(f"\nResults saved to: {output_file}")

                    # Add ground truth comparison to output if available
                    if (
                        gaia_eval
                        and hasattr(trace_data, "output")
                        and trace_data.output
                    ):
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
            
            # Main dataset: large MAS (gr.1)
            # name="gaia_task_db0c3ed0-a4af-4442-bb6f-884d6da055cb",  
            
            # Main dataset: small MAS (gr. 2)
            # name= "gaia_task_07aac7b1-ffc3-4787-8e4c-7fb522156097"
            
            # AutoMAS ru gaia, gemini launch
            name = "gaia_task_f240dd56-80c6-4571-b831-0bc4cf9508a6",
            
            result_file_name = "ru_gaia_mas_gemini_meta_sonnet",
            folder_name="ru_gaia"
    ))
