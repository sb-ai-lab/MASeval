# MASeval - Multi-Agent System Evaluation Library

A Python library for automated evaluation of Multi-Agent Systems using pydantic AI. This library provides trace-format-agnostic evaluation metrics to assess various aspects of agent performance.

## Features

- **Trace-format-agnostic**: Works with different trace formats (Langfuse, custom formats, etc.)
- **LLM-based evaluation**: Uses pydantic AI for intelligent metric evaluation
- **Multiple metrics**: Built-in support for common evaluation criteria
- **OpenRouter integration**: Easy integration with various LLM models via OpenRouter
- **Extensible**: Simple to add custom metrics
- **Type-safe**: Full type safety with pydantic models

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd maseval-research

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

## Quick Start

```python
import asyncio
from pydantic_ai.models.openai import OpenAIChatModel
from maseval.metrics import MetricType, create_metric
from maseval.models import EvaluationInput, DialogueMessage, MessageRole

# Initialize OpenRouter model
model = OpenAIChatModel(
    "anthropic/claude-3.5-sonnet",
    base_url="https://openrouter.ai/api/v1",
    api_key="sk-or-your-api-key",
)

# Create evaluation input
eval_input = EvaluationInput(
    dialogue_history=[
        DialogueMessage(role=MessageRole.USER, content="Hello"),
        DialogueMessage(role=MessageRole.ASSISTANT, content="Hi there!")
    ]
)

# Create and run metric
async def evaluate():
    metric = create_metric(MetricType.OBSERVATION_ALIGNMENT, model)
    result = await metric.evaluate(eval_input)
    return result

result = asyncio.run(evaluate())
```

## Available Metrics

Each metric has its own specific score result model for better type safety:

### 1. Observation Alignment (`ObservationAlignmentResult`)
Evaluates whether agent responses align with conversational history and user requirements.

**Criteria:**
- Consistency with user requests
- Completeness relative to the request
- Accuracy and truthfulness
- Consistency with previous details

### 2. Policy Alignment (`PolicyAlignmentResult`)
Checks if the agent follows predefined policies throughout the session.

**Criteria:**
- Full compliance with all policies
- Identification of policy violations
- Turn-by-turn policy adherence

### 3. State Consistency (`StateConsistencyResult`)
Evaluates agent's intermediate states (thoughts, actions) for logical flow.

**Criteria:**
- Consistency with user requests
- Consistency with previous states
- Accuracy and truthfulness

### 4. Tool Selection (`ToolSelectionResult`)
Assesses whether the agent selects appropriate tools for given tasks.

**Criteria:**
- Tool relevance to the question
- Best fit selection among available tools
- Justification based on question content

### 5. Tool Parameter Extraction (`ToolParameterExtractionResult`)
Evaluates how well the agent extracts parameters for tool calls.

**Criteria:**
- Parameter completeness
- Value justification
- Parameter accuracy

### 6. Task Completeness (`TaskCompletenessResult`)
Evaluates how completely the agent addresses user tasks.

**Criteria:**
- Task relevance
- Response completeness
- Actionability and usefulness
- Scope appropriateness

## Data Models

### Core Models

```python
from maseval.models import (
    EvaluationInput,    # Complete input for evaluation
    DialogueMessage,    # Single message in dialogue
    AgentResponse,      # Agent's response
    AgentState,         # Intermediate agent state
    MetricResult,       # Evaluation result
    MetricScore,        # Individual score
    ScoreValue,         # Score values: POOR, GOOD, IDEAL
)
```

### EvaluationInput Structure

```python
EvaluationInput(
    dialogue_history=[...],      # List of DialogueMessage
    agent_responses=[...],       # List of AgentResponse
    agent_states=[...],          # List of AgentState
    policies=[...],              # List of Policy (optional)
    available_tools=[...],       # List of ToolDefinition (optional)
    trace_id="...",              # Optional trace identifier
    metadata={...}               # Optional metadata
)
```

## Trace Parsing

The library includes example parsers for different trace formats:

### Langfuse Parser

```python
from examples.langfuse_parser import parse_langfuse_trace
import json

with open("trace.json", "r") as f:
    trace_data = json.load(f)

eval_input = parse_langfuse_trace(trace_data)
```

### Custom Parser

Create your own parser by implementing functions that convert your trace format to `EvaluationInput`:

```python
def parse_custom_trace(trace_data) -> EvaluationInput:
    # Parse dialogue history
    dialogue_history = [...]
    
    # Parse agent responses
    agent_responses = [...]
    
    # Parse agent states
    agent_states = [...]
    
    return EvaluationInput(
        dialogue_history=dialogue_history,
        agent_responses=agent_responses,
        agent_states=agent_states
    )
```

## Custom Metrics

Extend the library with custom metrics:

```python
from maseval.metrics import LLMMetric, MultiScoreResult

class CustomMetric(LLMMetric):
    def __init__(self, model):
        prompt = "Your custom evaluation prompt..."
        super().__init__(
            model=model,
            metric_name="custom_metric",
            prompt_template=prompt,
            result_type=MultiScoreResult,
            dependencies=["dialogue_history"]
        )
    
    def _prepare_context(self, eval_input):
        return {
            "dialogue_history": [msg.model_dump_json() for msg in eval_input.dialogue_history]
        }
    
    def _convert_result(self, result):
        # Convert LLM result to MetricResult
        ...
```

## Environment Setup

### Required Environment Variables

```bash
# For OpenRouter
export OPENROUTER_API_KEY="sk-or-your-api-key"

# Alternative: OpenAI API
export OPENAI_API_KEY="sk-your-openai-key"

# For Langfuse tracing (optional but recommended)
# Keys for downloading traces from your evaluation project:
export LANGFUSE_PUBLIC_KEY="pk-lf-your-public-key"
export LANGFUSE_SECRET_KEY="sk-lf-your-secret-key"

# Keys for uploading maseval judge/evaluation traces:
export LANGFUSE_PUBLIC_KEY_JUDGE="pk-lf-your-judge-public-key"
export LANGFUSE_SECRET_KEY_JUDGE="sk-lf-your-judge-secret-key"

# Optional: Langfuse host (defaults to https://cloud.langfuse.com)
export LANGFUSE_HOST="https://cloud.langfuse.com"
```

### Langfuse Integration

The library uses **two separate Langfuse clients** to avoid confusion between trace sources:

1. **Download Client** (`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`):
   - Used in examples and scripts to retrieve traces from your evaluation project
   - Only used for downloading/reading traces
   - Has `tracing_enabled=False` to prevent interfering with judge client

2. **Judge Client** (`LANGFUSE_PUBLIC_KEY_JUDGE` / `LANGFUSE_SECRET_KEY_JUDGE`):
   - Used by the framework to upload evaluation traces
   - Sets the global OpenTelemetry tracer provider for pydantic AI instrumentation
   - All metric evaluations are traced to the judge project

**Critical Usage Pattern:**

```python
from maseval import get_langfuse_download_client, get_langfuse_judge_client
from maseval.metrics import create_metric, MetricType

# Step 1: Download traces FIRST
lf_download = get_langfuse_download_client()
trace = lf_download.api.trace.get("trace-id")

# Step 2: Initialize judge client AFTER downloading, BEFORE evaluating
# This sets up OpenTelemetry to trace evaluations to the judge project
get_langfuse_judge_client()

# Step 3: Run evaluations - these will be traced to judge project
metric = create_metric(MetricType.TASK_COMPLETENESS, model)
result = await metric.evaluate(eval_input)  # ✓ Traced to judge project
```

**Important Notes:**
- Always call `get_langfuse_judge_client()` **after** downloading traces
- Always call it **before** running evaluations
- The download client has tracing disabled, so it won't interfere
- The judge client sets the global OpenTelemetry tracer, so all subsequent pydantic AI traces go to the judge project

### Trace Grouping and Tagging

To organize evaluations in Langfuse, you can group all metric evaluations for a single task under one parent trace:

```python
from maseval import get_langfuse_judge_client

# Initialize judge client
judge_client = get_langfuse_judge_client()

# Group all evaluations for one task under a single span
for task_id in task_ids:
    with judge_client.start_as_current_span(
        name=f"evaluate_task_{task_id}",
        input={"task_id": task_id},
        metadata={"task_id": task_id, "dialogue_messages": 10}
    ) as span:
        # Update the trace with tags (tags are at trace level)
        judge_client.update_current_trace(
            tags=["maseval", f"task_id:{task_id}"]  # ✅ Add task_id as tag
        )
        
        # All evaluations inside this block are grouped together
        for metric_type in [MetricType.TASK_COMPLETENESS, MetricType.MAS_TASK_COMPLETION]:
            metric = create_metric(metric_type, model)
            result = await metric.evaluate(eval_input)
        
        # Update span with results summary
        span.update(output={"results_summary": {...}})
```

**Benefits:**
- **Organization**: One trace per task with all evaluations as nested spans
- **Filtering**: Use tags like `task_id:abc123` to find specific evaluations
- **Metadata**: Track task information, ground truth, and results
- **Hierarchy**: Navigate through parent-child relationships in Langfuse UI

### Model Configuration

```python
# OpenRouter (recommended for model variety)
from pydantic_ai.models.openai import OpenAIChatModel

model = OpenAIChatModel(
    "anthropic/claude-3.5-sonnet",  # or any OpenRouter model
    provider="openrouter"
    api_key="",
)
```

### Dependency Injection

The library uses pydantic AI's dependency injection system to pass evaluation data to metrics:

```python
# The evaluation input is automatically injected into the agent
async def evaluate(self, eval_input: EvaluationInput) -> MetricResult:
    result = await self.agent.run(
        "Please evaluate the provided data according to the metric criteria.",
        deps=eval_input  # Dependency injection
    )
    return self._convert_result(result.data)
```

This provides clean separation between the evaluation logic and data formatting.

## Examples

See the `examples/` directory for complete usage examples:

- `simple_example.py`: Basic library usage with mock data
- `example_usage.py`: Full evaluation pipeline with real API calls
- `langfuse_parser.py`: Trace parser for Langfuse format

## Running Examples

```bash
# Simple demo (no API calls)
PYTHONPATH=src uv run python -m examples.simple_example

# Full example (requires API key)
export OPENROUTER_API_KEY="sk-or-your-key"
PYTHONPATH=src uv run python -m examples.example_usage
```

## Development

### Project Structure

```
src/maseval/
├── __init__.py          # Public API
├── models.py            # Pydantic models
├── metrics.py           # Metric implementations
└── prompts.py           # System prompts for metrics

examples/
├── simple_example.py    # Basic demo
├── example_usage.py     # Full example
└── langfuse_parser.py   # Trace parser

data/
└── trace_example.json   # Example trace data
```

### Testing

```bash
# Run the simple example to test basic functionality
PYTHONPATH=src uv run python -m examples.simple_example

# Test with real API calls (requires API key)
export OPENROUTER_API_KEY="your-key"
PYTHONPATH=src uv run python -m examples.example_usage
```

## API Reference

### Metric Types

```python
from maseval.metrics import MetricType

MetricType.OBSERVATION_ALIGNMENT
MetricType.POLICY_ALIGNMENT
MetricType.STATE_CONSISTENCY
MetricType.TOOL_SELECTION
MetricType.TASK_COMPLETENESS
```

### Score Values

```python
from maseval.models import ScoreValue

ScoreValue.POOR   # Significant issues
ScoreValue.GOOD   # Partially correct
ScoreValue.IDEAL  # Perfect performance
```

### Factory Function

```python
from maseval.metrics import create_metric

metric = create_metric(MetricType.OBSERVATION_ALIGNMENT, model)
result = await metric.evaluate(eval_input)
```

### Accessing Prompts

All system prompts are available for inspection and customization:

```python
from maseval import prompts

# View a specific prompt
print(prompts.OBSERVATION_ALIGNMENT_PROMPT)

# All available prompts
prompt_names = [name for name in dir(prompts) if name.endswith('_PROMPT')]
print(prompt_names)
```
