"""Example usage of the MASeval library."""

import asyncio
import json
import os
import sys
from pathlib import Path

from pydantic_ai.models.openai import OpenAIChatModel

from maseval.metrics import MetricType, create_metric
from maseval.models import Policy

# Add examples directory to path so we can import langfuse_parser
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

from maseval.parsers.langfuse_parser_langgraph import parse_langfuse_trace

load_dotenv()


async def main():
    """Main example demonstrating the library usage."""

    # Initialize the OpenRouter model
    # Note: You'll need to set your OpenRouter API key
    api_key = os.getenv("OPENROUTER_API_KEY")

    # Set the API key as environment variable for OpenRouter
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key

    model = OpenAIChatModel(
        "google/gemini-2.5-flash-lite",  # Using a known available OpenRouter model
        provider="openrouter",
    )

    # Load and parse the trace example
    trace_file = Path(__file__).parent.parent / "data" / "trace_example.json"
    with open(trace_file, "r") as f:
        trace_data = json.load(f)

    eval_input = parse_langfuse_trace(trace_data)

    print("=== Parsed Trace Information ===")
    print(f"Trace ID: {eval_input.trace_id}")
    print(f"Dialogue messages: {len(eval_input.dialogue_history)}")
    print(f"Agent responses: {len(eval_input.agent_responses)}")
    print(f"Agent states: {len(eval_input.agent_states)}")
    print(f"Available tools: {len(eval_input.available_tools)}")

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

    # Test different metrics
    metrics_to_test = [
        MetricType.OBSERVATION_ALIGNMENT,
        MetricType.POLICY_ALIGNMENT,
        MetricType.STATE_CONSISTENCY,
        MetricType.TASK_COMPLETENESS,
    ]

    results = {}

    for metric_type in metrics_to_test:
        print(f"\n--- Evaluating {metric_type.value} ---")

        try:
            # Create the metric
            metric = create_metric(metric_type, model)

            # Run evaluation
            result = await metric.evaluate(eval_input)
            results[metric_type.value] = result

            print(f"Metric: {result.metric_name}")
            print(f"Number of scores: {len(result.scores)}")

            for score in result.scores:
                print(f"  - Item {score.item_id}: {score.score.value}")
                print(f"    Justification: {score.justification[:100]}...")

        except Exception as e:
            print(f"Error evaluating {metric_type.value}: {e}")
            continue

    print("\n=== Evaluation Summary ===")
    for metric_name, result in results.items():
        total_scores = len(result.scores)
        ideal_scores = sum(1 for s in result.scores if s.score.value == "ideal")
        good_scores = sum(1 for s in result.scores if s.score.value == "good")
        poor_scores = sum(1 for s in result.scores if s.score.value == "poor")

        print(f"{metric_name}:")
        print(
            f"  Total: {total_scores}, Ideal: {ideal_scores}, Good: {good_scores}, Poor: {poor_scores}"
        )

    # Save results to file
    output_file = Path(__file__).parent / "evaluation_results.json"
    with open(output_file, "w") as f:
        serializable_results = {}
        for metric_name, result in results.items():
            serializable_results[metric_name] = {
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
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
