# TraceInspector (MASeval)

**TraceInspector** is a modular framework for comprehensive diagnosis of failures in LLM-based multi-agent systems.

Instead of relying on a single general-purpose LLM-as-a-Judge prompt, TraceInspector combines:

- specialized LLM judges for semantic failures;
- deterministic validators for objective execution failures;
- an optional LLM Confirmer for filtering unsupported deterministic findings;
- optional rule-based evidence verification for LLM findings;
- a rule-based report builder that produces a unified diagnostic report.

The framework is implemented in the **MASeval** repository and is designed to work across heterogeneous trace formats and evaluation benchmarks.
---

## Why TraceInspector?

A final success/failure score is not enough for debugging a multi-agent system. Developers usually need to know:

- which agent introduced the failure;
- where the first problematic step occurred;
- whether the failure originated from reasoning, coordination, tool use, an API, or the execution environment;
- what evidence in the trace supports the diagnosis;
- how the system can be fixed.

TraceInspector addresses these questions by generating structured, evidence-linked findings instead of only returning a scalar score.

---

## System Overview

TraceInspector contains three main diagnostic branches.

### 1. Final Answer Verification

The final-answer module determines whether the system successfully solved the original task.

TraceInspector supports:

- **reference-based mode**, when a ground-truth answer is available;
- **no-GT mode**, where the final answer is evaluated using only the execution trace.

The experiments in the accompanying paper use the **no-GT setting**: the diagnostic components do not see the benchmark ground truth, which is used only for offline metric computation.

### 2. Deterministic Validation

Rule-based validators detect failures that can be inferred directly from execution logs.

Validator groups currently cover:

- **API / HTTP failures**
- **Provider / LLM service failures**
- **Environment / setup failures**
- **Tool / schema failures**

Typical examples include failed HTTP requests, empty API responses, missing credentials, context-limit errors, missing dependencies, invalid tool arguments, malformed outputs, missing files, and permission errors.

Deterministic findings can optionally be passed to an **LLM Confirmer**. The Confirmer sees both the execution trace and the validator outputs, removes unsupported detections, resolves ambiguous cases, and converts retained findings into the common structured format used by the rest of the pipeline.

### 3. Specialized LLM Judges

TraceInspector decomposes semantic diagnosis into eleven specialized LLM judges:

- Observation Alignment
- Policy Alignment
- State Consistency
- Tool Selection
- Tool Parameter Extraction
- Multi-Agent Planning
- Multi-Agent Complexity
- Multi-Agent Task Transfer
- Multi-Agent Role Distribution
- Tool Performance
- Prompt Quality

Each judge analyzes a narrow diagnostic dimension, which makes the output easier to inspect and compare than a single monolithic judge response.

---

## Structured Findings

Each LLM judge returns one or more structured findings.

```json
{
  "metric_name": "mas_task_transfer",
  "findings": [
    {
      "severity_estimate": "critical",
      "confidence_estimate": "high",
      "culprit_agent_candidates": [
        {
          "agent": "ExhibitionContentAnalyzer",
          "reason": "This agent introduced the unsupported claim that was passed downstream."
        }
      ],
      "evidence": [
        {
          "span_id": "span_0017",
          "role": "root_cause",
          "claim": "The agent introduced an unsupported claim.",
          "quote": "No exhibited zodiac animals have visible hands."
        },
        {
          "span_id": "span_0024",
          "role": "propagation",
          "claim": "A downstream agent relied on the unsupported claim.",
          "quote": "Since no animals have visible hands, image retrieval is unnecessary."
        }
      ],
      "problem_description": "An unsupported factual handoff caused a downstream agent to skip a required step.",
      "suggested_fix": "Require downstream agents to validate factual handoffs before skipping required tools.",
      "needs_human_review": false
    }
  ]
}
```

The rule-based report builder merges findings from all enabled components into a single diagnostic report containing responsible agents, problematic steps, failure descriptions, severity estimates, supporting evidence, and suggested fixes.

---

## Evidence Verification

LLM findings can optionally be processed by a lightweight rule-based evidence verifier.

The verifier checks:

- whether referenced message or span identifiers exist;
- whether quoted evidence appears in the trace;
- whether culprit-agent attribution is structurally consistent;
- whether the cited role and evidence location are plausible.

Each finding receives one of three labels:

- **valid**
- **weak**
- **invalid**

The evaluation scripts support three policies:

- **none** — keep all LLM findings;
- **soft** — remove only invalid findings;
- **strict** — remove invalid and weak findings.

In the experiments reported in the paper, evidence filtering is disabled for the main results because it did not consistently improve diagnostic metrics with Gemini Flash 2.5.

---

## Supported Evaluation Benchmarks

The repository contains benchmark-specific adapters and evaluation scripts for:

- **Who&When**
  - Agent Accuracy
  - Step Accuracy
- **TRAIL**
  - Localization Accuracy
- **AEGIS**
  - Agent MF1
  - Agent mF1
- **TraceElephant**
  - Agent Accuracy
  - Step Accuracy

Benchmark ground truth is used only for offline metric computation. It is not included in the input shown to the diagnostic judges.

---

## Installation

### Using `uv` (recommended)

```bash
git clone https://github.com/sb-ai-lab/MASeval.git
cd MASeval
uv sync
```

### Using `pip`

```bash
git clone https://github.com/sb-ai-lab/MASeval.git
cd MASeval
pip install -e .
```

---

## Environment Variables

The exact provider variables depend on the selected model configuration.

Typical setup:

```bash
export OPENROUTER_API_KEY="..."
# or
export OPENAI_API_KEY="..."
```

Optional Langfuse configuration:

```bash
# Trace source
export LANGFUSE_PUBLIC_KEY="..."
export LANGFUSE_SECRET_KEY="..."

# Judge/evaluation project
export LANGFUSE_PUBLIC_KEY_JUDGE="..."
export LANGFUSE_SECRET_KEY_JUDGE="..."

export LANGFUSE_HOST="https://cloud.langfuse.com"
```

Trace downloading and judge tracing use separate Langfuse clients so that source traces and evaluation traces remain isolated.

---

## Running the Pipeline

The repository contains complete benchmark-specific examples in `examples/`.

For Who&When, the main scripts include:

```text
examples/who_and_when/launch_findings_judges.py
examples/who_and_when/calculate_agent_step_accuracy.py
examples/who_and_when/verifier_ablation.py
```

A typical workflow is:

1. convert a benchmark trace into the MASeval input format;
2. run the specialized LLM judges;
3. run deterministic validators;
4. optionally confirm deterministic findings;
5. optionally apply evidence filtering to LLM findings;
6. build the final diagnostic report;
7. compute benchmark-specific metrics.

Open the corresponding example script and edit its `main(...)` call or configuration block. The experiment scripts are intentionally debugger-friendly and do not require command-line argument parsing.

---

## Available LLM Metrics

```python
from maseval.metrics import MetricType

LLM_METRICS_TO_TEST = [
    MetricType.OBSERVATION_ALIGNMENT,
    MetricType.POLICY_ALIGNMENT,
    MetricType.STATE_CONSISTENCY,
    MetricType.TOOL_SELECTION,
    MetricType.TOOL_PARAMETER_EXTRACTION,
    MetricType.MAS_PLANNING,
    MetricType.MAS_COMPLEXITY,
    MetricType.MAS_TASK_TRANSFER,
    MetricType.MAS_ROLES_DISTRIBUTION,
    MetricType.TOOL_PERFORMANCE,
    MetricType.PROMPT_QUALITY,
]
```

Metrics are instantiated through the MASeval metric factory and return typed Pydantic outputs.

---

## Trace Adapters

TraceInspector is trace-format-agnostic at the diagnostic level. A source trace must first be converted into the MASeval input representation.

The repository includes benchmark- and platform-specific adapters in `examples/`. Custom integrations can be implemented by converting a source trace into `EvaluationInput` and preserving:

- message order;
- agent names;
- message or span identifiers;
- tool calls and tool outputs;
- trace metadata required by the selected diagnostic components.

Explicit, stable message identifiers are strongly recommended because they improve evidence localization and benchmark step matching.

---

## Reproducibility

The reported experiments use temperature `0.0` for all LLM-based components.

The complete Gemini Flash 2.5 evaluation was repeated five times and produced identical diagnostic outputs and evaluation metrics. Therefore, the paper reports deterministic single-run results without confidence intervals or standard deviations.

---

## Project Structure

```text
MASeval/
├── src/maseval/            # Core models, metrics, prompts, validators, reporting
├── examples/               # Benchmark adapters and experiment runners
│   ├── who_and_when/
│   ├── trail/
│   └── ...
├── tests/                  # Unit and integration tests
├── data/                   # Example and benchmark data
├── test_results/           # Stored experiment outputs
├── LANGFUSE_SETUP.md
├── LANGFUSE_API_REFERENCE.md
└── TRACE_GROUPING.md
```

---

## Extending TraceInspector

The framework is modular by design.

You can add:

- a new specialized LLM judge;
- a deterministic validator;
- a trace adapter;
- a benchmark-specific metric converter;
- a report post-processing component.

New LLM judges should return the common structured finding schema so their outputs can be aggregated by the existing report builder.


---

## Contributing

Issues and pull requests are welcome. Contributions are especially useful for:

- new deterministic validators;
- adapters for additional multi-agent frameworks;
- new diagnostic benchmarks;
- additional specialized LLM judges;
- improvements to evidence localization and report generation.