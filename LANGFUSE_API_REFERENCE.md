# Langfuse API Reference for MASeval

## Correct Method Names

### Creating Parent Context for Evaluations

✅ **CORRECT:**
```python
with judge_client.start_as_current_span(
    name="my_span",
    input={...},
    metadata={...}
) as span:
    # Update trace with tags (tags are at trace level, not span level)
    judge_client.update_current_trace(tags=["tag1", "tag2"])
    # Your code here
```

❌ **INCORRECT:**
```python
# Method doesn't exist!
with judge_client.start_as_current_trace(...) as trace:
    # Your code here

# Tags parameter not supported in start_as_current_span!
with judge_client.start_as_current_span(tags=["tag1"]) as span:
    # Your code here
```

### Important: Tags Must Be Set at Trace Level

- `start_as_current_span()` does **NOT** accept a `tags` parameter
- Tags must be set using `update_current_trace(tags=[...])`
- Call `update_current_trace()` inside the span context to tag the trace

## Available Langfuse Methods

### Creating Contexts

- `start_as_current_span()` - Create a span and set it as current (use this for grouping)
  - **Accepted parameters**: `name`, `input`, `output`, `metadata`, `version`, `level`, `status_message`, `trace_context`, `end_on_exit`
  - **NOT accepted**: `tags` (must use `update_current_trace(tags=...)` instead)
- `start_as_current_observation()` - Create an observation and set it as current
- `start_as_current_generation()` - Create a generation and set it as current
- `start_span()` - Create a span without setting it as current
- `start_observation()` - Create an observation without setting it as current
- `start_generation()` - Create a generation without setting it as current

### Updating Contexts

- `update_current_span()` - Update the current span
- `update_current_trace()` - Update the current trace
- `update_current_generation()` - Update the current generation

### Getting Information

- `get_current_trace_id()` - Get the ID of the current trace
- `get_current_observation_id()` - Get the ID of the current observation
- `get_trace_url()` - Get the Langfuse UI URL for a trace
- `create_trace_id()` - Generate a new trace ID

### Scoring

- `score_current_span()` - Add a score to the current span
- `score_current_trace()` - Add a score to the current trace

## How Tracing Works in MASeval

### The Setup

1. **Judge Client Initialization**: Sets global OpenTelemetry tracer provider
2. **Pydantic AI Instrumentation**: `Agent.instrument_all()` enables automatic tracing
3. **Parent Span Creation**: `start_as_current_span()` creates a grouping context
4. **Child Traces**: Pydantic AI agents automatically create child traces under the parent span

### The Hierarchy

```
OpenTelemetry Trace (created automatically by pydantic AI)
└── Parent Span (created by start_as_current_span)
    ├── Agent Run 1 (TaskCompletenessEvaluator)
    │   ├── System prompt
    │   ├── User message
    │   └── Model response
    └── Agent Run 2 (SystemTaskCompletionEvaluator)
        ├── System prompt
        ├── User message
        └── Model response
```

## Usage in MASeval

### Creating a Parent Span for Task Evaluation

```python
from maseval import get_langfuse_judge_client

judge_client = get_langfuse_judge_client()

for task_id in task_ids:
    # Create parent span for this task's evaluations
    with judge_client.start_as_current_span(
        name=f"evaluate_task_{task_id}",
        input={"task_id": task_id, "trace_id": eval_input.trace_id},
        metadata={
            "task_id": task_id,
            "dialogue_messages": len(eval_input.dialogue_history),
            "ground_truth": trace_data.output.get("ground_truth"),
        }
    ) as span:
        # Update the trace with tags (tags are at trace level, not span level)
        judge_client.update_current_trace(
            tags=["maseval", f"task_id:{task_id}"]
        )
        
        # All metric evaluations here will be grouped under this span
        for metric_type in metrics_to_test:
            metric = create_metric(metric_type, model)
            result = await metric.evaluate(eval_input)
        
        # Update span with results
        span.update(
            output={
                "results_summary": {...},
                "ground_truth_comparison": {...}
            }
        )
```

## Why start_as_current_span?

The `start_as_current_span` method:
1. Creates a span in Langfuse
2. Sets it as the **current OpenTelemetry context**
3. Any pydantic AI agent runs inside this context become **child traces**
4. This creates the grouping/hierarchy we want

## Important Notes

- The parent span serves as a **context manager** for grouping
- Pydantic AI's automatic instrumentation creates the actual **traces**
- The span becomes the **parent** of those traces
- This gives us the hierarchy: Span → Traces → Generations
- Tags and metadata are attached to the span for easy filtering

## Common Patterns

### Pattern 1: Single Task Evaluation
```python
with judge_client.start_as_current_span(name="evaluate_task") as span:
    result = await metric.evaluate(eval_input)
    span.update(output={"result": result})
```

### Pattern 2: Multiple Metrics for One Task
```python
with judge_client.start_as_current_span(name=f"evaluate_task_{task_id}") as span:
    results = {}
    for metric_type in metrics:
        metric = create_metric(metric_type, model)
        results[metric_type] = await metric.evaluate(eval_input)
    span.update(output={"results": results})
```

### Pattern 3: Batch Evaluation with Tagging
```python
for i, task_id in enumerate(task_ids):
    with judge_client.start_as_current_span(
        name=f"evaluate_task_{task_id}",
        tags=["batch", f"task_id:{task_id}", f"batch_index:{i}"]
    ) as span:
        # Evaluations here
        pass
```

