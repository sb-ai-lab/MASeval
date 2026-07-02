# Trace Grouping in Langfuse

## Overview

All metric evaluations for a single task are now grouped under **one parent trace** in Langfuse, making it easy to:
- Navigate and analyze evaluations
- Filter by task_id
- Track metadata and results
- See hierarchical relationships

## Implementation

### Code Structure

```python
from maseval import get_langfuse_judge_client

# Initialize judge client
judge_client = get_langfuse_judge_client()

# Process each task
for task_id in task_ids:
    # Create parent span for this task
    with judge_client.start_as_current_span(
        name=f"evaluate_task_{task_id}",
        input={"task_id": task_id, "trace_id": eval_input.trace_id},
        metadata={
            "task_id": task_id,
            "trace_id": eval_input.trace_id,
            "dialogue_messages": len(eval_input.dialogue_history),
            "agent_responses": len(eval_input.agent_responses),
            # Add ground truth if available
            "ground_truth": trace_data.output.get("ground_truth"),
            "correct_answer": is_correct
        }
    ) as span:
        # Update the trace with tags (tags are at trace level, not span level)
        judge_client.update_current_trace(
            tags=["maseval", f"task_id:{task_id}"]
        )
        
        # All metric evaluations run inside this context
        for metric_type in metrics_to_test:
            metric = create_metric(metric_type, model)
            result = await metric.evaluate(eval_input)
            # ✅ This evaluation is a child span of the parent span
        
        # Update span with results summary
        span.update(output={
            "results_summary": {...},
            "ground_truth_comparison": {...}
        })
```

## What Gets Tracked

### 1. Trace Name
```
evaluate_task_{task_id}
```
Example: `evaluate_task_abc123`

### 2. Tags
```python
tags=["maseval", f"task_id:{task_id}"]
```
- `maseval`: Identifies all maseval evaluation traces
- `task_id:{task_id}`: Allows filtering by specific task

### 3. Input
```python
input={
    "task_id": task_id,
    "trace_id": eval_input.trace_id
}
```

### 4. Metadata
```python
metadata={
    "task_id": task_id,
    "trace_id": eval_input.trace_id,
    "dialogue_messages": 10,
    "agent_responses": 5,
    "agent_states": 8,
    # If GAIA evaluation:
    "ground_truth": "42",
    "mas_response": "42",
    "correct_answer": True
}
```

### 5. Output
```python
output={
    "results_summary": {
        "task_completeness": {
            "total_scores": 5,
            "ideal": 3,
            "good": 2,
            "poor": 0
        },
        "mas_task_completion": {
            "total_scores": 1,
            "ideal": 1,
            "good": 0,
            "poor": 0
        }
    },
    # If GAIA evaluation:
    "ground_truth_comparison": {
        "ground_truth": "42",
        "mas_response": "42",
        "match": True
    }
}
```

## Langfuse UI Hierarchy

```
📊 evaluate_task_abc123  (Parent Trace)
   ├── 📝 Input: {task_id, trace_id}
   ├── 🏷️  Tags: [maseval, task_id:abc123]
   ├── 📋 Metadata: {task info, ground truth, ...}
   │
   ├── 🔍 TaskCompletenessEvaluator run  (Child Span 1)
   │   ├── System prompt
   │   ├── User message
   │   ├── Model request
   │   └── Model response
   │
   ├── 🔍 SystemTaskCompletionEvaluator run  (Child Span 2)
   │   ├── System prompt
   │   ├── User message
   │   ├── Model request
   │   └── Model response
   │
   └── ✅ Output: {results_summary, ground_truth_comparison}
```

## Benefits

### 1. Organization
- **Before**: 10 separate traces for 10 metrics = 10 top-level items
- **After**: 1 parent trace with 10 nested child spans = 1 top-level item

### 2. Filtering
Search in Langfuse:
```
tag:task_id:abc123
```
Returns all evaluations for that specific task.

### 3. Analysis
- View all metrics for a task at once
- Compare metric results side-by-side
- Track evaluation latency per task
- See which metrics passed/failed together

### 4. Ground Truth Tracking
For GAIA or other benchmarks:
- Metadata shows if MAS got the correct answer
- Output includes ground truth comparison
- Easy to filter for correct vs incorrect tasks

## Usage in Examples

See `examples/example_pydanticAI_usage.py` for the complete implementation:

```python
async def main(gaia_eval, name):
    # Download traces
    lf = get_langfuse_download_client()
    traces = lf.api.trace.list(name=name)
    
    # Initialize judge client
    judge_client = get_langfuse_judge_client()
    
    # Process each task with grouping
    for task_id in task_ids:
        trace_data = lf.api.trace.get(task_id)
        eval_input = parse_langfuse_task(trace_data)
        
        # Group all evaluations under one span ✅
        with judge_client.start_as_current_span(...) as trace:
            for metric_type in metrics_to_test:
                result = await metric.evaluate(eval_input)
            
            trace.update(output=results_summary)
```

## Querying in Langfuse

### Find all maseval evaluations
```
tag:maseval
```

### Find evaluations for specific task
```
tag:task_id:abc123
```

### Find evaluations with correct answers (GAIA)
Filter by metadata: `correct_answer = true`

### Find failed evaluations
Filter by metadata: `correct_answer = false`

## Future Enhancements

Possible additions:
- Session ID grouping (group multiple tasks from one session)
- Batch evaluation traces (group multiple tasks evaluated together)
- Comparison traces (compare different models on same task)
- Time-based grouping (group by evaluation run timestamp)

