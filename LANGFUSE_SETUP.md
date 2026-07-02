# Langfuse Dual-Client Setup Guide

## Overview

This library uses **two separate Langfuse clients** to prevent trace confusion:

1. **Download Client**: For retrieving traces from your evaluation project
2. **Judge Client**: For uploading evaluation/judge traces from maseval

## Environment Variables

Add these to your `.env` file:

```bash
# For downloading traces from evaluation project
LANGFUSE_PUBLIC_KEY="pk-lf-your-eval-public-key"
LANGFUSE_SECRET_KEY="sk-lf-your-eval-secret-key"

# For uploading maseval judge traces
LANGFUSE_PUBLIC_KEY_JUDGE="pk-lf-your-judge-public-key"
LANGFUSE_SECRET_KEY_JUDGE="sk-lf-your-judge-secret-key"

# Optional (defaults to https://cloud.langfuse.com)
LANGFUSE_HOST="https://cloud.langfuse.com"
```

## Critical Usage Pattern

**⚠️ IMPORTANT: Order matters!**

```python
from maseval import get_langfuse_download_client, get_langfuse_judge_client
from maseval.metrics import create_metric, MetricType

# ✅ Step 1: Download traces FIRST
lf = get_langfuse_download_client()
traces = lf.api.trace.list(name="my-task")
trace_data = lf.api.trace.get("trace-id")

# ✅ Step 2: Initialize judge client AFTER downloading
get_langfuse_judge_client()

# ✅ Step 3: Run evaluations (traced to judge project)
metric = create_metric(MetricType.TASK_COMPLETENESS, model)
result = await metric.evaluate(eval_input)
```

## Why This Order?

### The Problem
- Langfuse sets a **global OpenTelemetry tracer provider** when initialized
- If you create judge client first, then download client, the download client would override it
- All traces would go to the wrong project!

### The Solution
1. **Download client**: Has `tracing_enabled=False` to avoid setting up OpenTelemetry
2. **Judge client**: Sets up OpenTelemetry for pydantic AI instrumentation
3. By initializing judge client **after** download client, we ensure evaluations trace to judge project

## Example Implementation

See `examples/example_pydanticAI_usage.py` for a complete example:

```python
async def main(gaia_eval, name):
    # Step 1: Download traces
    print("=== Downloading traces ===")
    lf = get_langfuse_download_client()
    traces = lf.api.trace.list(name=name, limit=100)
    
    # Step 2: Initialize judge client
    print("=== Setting up judge client ===")
    judge_client = get_langfuse_judge_client()
    print("Judge client ready - evaluations will trace to judge project")
    
    # Step 3: Run evaluations with trace grouping
    for task_id in task_ids:
        trace_data = lf.api.trace.get(task_id)
        eval_input = parse_langfuse_task(trace_data)
        
        # Group all evaluations for this task under one span
        with judge_client.start_as_current_span(
            name=f"evaluate_task_{task_id}",
            input={"task_id": task_id, "trace_id": eval_input.trace_id},
            metadata={"task_id": task_id, "trace_id": eval_input.trace_id}
        ) as span:
            # Update the trace with tags (tags are at trace level)
            judge_client.update_current_trace(
                tags=["maseval", f"task_id:{task_id}"]
            )
            
            # All metric evaluations run inside this context
            # They will appear as nested spans in Langfuse! ✅
            metric = create_metric(MetricType.TASK_COMPLETENESS, model)
            result = await metric.evaluate(eval_input)
            
            # Update span with results
            span.update(output={"results": ...})
```

### Trace Grouping

The example above shows how to **group all metric evaluations for a single task** under one parent trace in Langfuse. This provides:

- **Better organization**: Each task has one trace with all evaluations as children
- **Easy filtering**: Use tags like `task_id:abc123` to find specific task evaluations
- **Metadata tracking**: Store task info, ground truth, and results in the trace
- **Cleaner UI**: Navigate through hierarchical traces in Langfuse

## Verification

You can verify the setup works by checking your Langfuse projects:
- **Evaluation project** (using `LANGFUSE_PUBLIC_KEY`): Should have your original traces
- **Judge project** (using `LANGFUSE_PUBLIC_KEY_JUDGE`): Should have evaluation agent traces

## Troubleshooting

**Problem**: Traces appearing in wrong project

**Solution**: Make sure you:
1. Call `get_langfuse_download_client()` first
2. Call `get_langfuse_judge_client()` after downloading
3. Call `get_langfuse_judge_client()` before running evaluations

**Problem**: No traces in judge project

**Solution**: Ensure:
1. `LANGFUSE_PUBLIC_KEY_JUDGE` and `LANGFUSE_SECRET_KEY_JUDGE` are set correctly
2. You called `get_langfuse_judge_client()` before running evaluations
3. The keys have write permissions in the judge project

