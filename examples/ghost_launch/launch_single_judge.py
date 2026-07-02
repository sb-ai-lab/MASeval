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
from maseval.metrics.single_judge import SingleJudge, SingleJudgeResult

from rich import print
from tqdm import tqdm
import time

from maseval.prompts import (
    MAS_UNIFIED_COMPREHENSIVE_EVALUATION_PROMPT
)

sys.path.insert(0, str(Path(__file__).parent))
from maseval.parsers.langfuse_parser_v3 import parse_langfuse_task


async def main(gaia_eval, enable_tracing, name, name_v2 = None, result_file_name: str = "judge_res"):
    """Main example demonstrating the library usage with Langfuse traces."""

    # Initialize the OpenRouter model
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key

    # Step 1: Get Langfuse client for downloading traces
    # This uses LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY
    print("=== Downloading traces from evaluation project ===")
    lf = get_langfuse_download_client()

    time.sleep(3)
    traces_page1 = lf.api.trace.list(name=name, limit=50, page=1)
    time.sleep(3)
    traces_page2 = lf.api.trace.list(name=name, limit=50, page=2)
    time.sleep(3)
    traces_page3 = lf.api.trace.list(name=name, limit=50, page=3) 
    time.sleep(3)
    traces_page4 = lf.api.trace.list(name=name, limit=50, page=4)
    time.sleep(3)

    if name_v2:
        trace_v2_page1 = lf.api.trace.list(name=name_v2, limit=100, page=1)
        trace_v2_page2 = lf.api.trace.list(name=name_v2, limit=100, page=2)

        all_traces = traces_page1.data + traces_page2.data + trace_v2_page1.data + trace_v2_page2.data
    else:
        all_traces = traces_page1.data + traces_page2.data + traces_page3.data + traces_page4.data
        
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
        # Initializing base model
        model = OpenAIChatModel(
            "google/gemini-2.5-flash", #"qwen/qwen3-235b-a22b-2507",
            provider="openrouter",
            settings={
                "temperature": 0.0,
            },
        )

        succ = False
        time.sleep(3)
        cnt = 0
        while not(succ):
            try:
                trace_data = lf.api.trace.get(task)
                succ = True
            except:
                if cnt > 4:
                    print('Cant process id', task)
                    continue
                time.sleep(3)
                cnt += 1
                pass

        # trace_data = lf.api.trace.get(task)
        print(f"\n=== Evaluating Task {task} ===")
        eval_input = parse_langfuse_task(trace_data)

        time.sleep(10)

        if eval_input.agents_errors != None:
            print("Task has agents errors")
            print(f"Agents errors: {eval_input.agents_errors}")
        else:
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
                        tags=["test_single_judge(big_mas)", f"task_id:{task}"]
                    )
                    ground_truth_value = (
                                trace_data.output["response"]
                                == trace_data.output["ground_truth"]
                            )

                    serializable_results = {}
                    span_output={}

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

                    print(f"\n--- Evaluating Single Judge ---")
                    try:
                        single_judge = SingleJudge(
                            model,
                            MAS_UNIFIED_COMPREHENSIVE_EVALUATION_PROMPT,
                            SingleJudgeResult,
                        )

                        single_judge_results = await single_judge.evaluate(eval_input)

                    except Exception as e:
                        print(f"Error evaluating summarizer: {e}")
                        continue


                    serializable_results["single_judge_score"] = {
                        "metric_name": "single_judge_score",
                        "scores": [
                            {
                                "item_id": "overall_score",
                                "score": single_judge_results.score,
                                "justification": single_judge_results.justification,
                            }
                        ],
                    }

                    print(
                        f"Summary:\nMAS evaluation score: {single_judge_results.score}\nJustification: {single_judge_results.justification}"
                    )

                    # Save results to file per task
                    output_dir = Path(__file__).parent / "test_single_judge(14.01)(big_mas)"
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
            # name="gaia_task_batched_mas_3404666a-21e6-49f3-9ada-f6b05112935c", # single_mas
            # name="gaia_task_2fee57e0-2ccc-4f13-93c9-29d866aa5b27", # ru_gaia,
            name="gaia_task_db0c3ed0-a4af-4442-bb6f-884d6da055cb", # big_mas
            # name="gaia_task_07aac7b1-ffc3-4787-8e4c-7fb522156097" # small_mas
        )
    )
