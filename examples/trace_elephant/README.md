# TraceElephant — running the MASeval judge on their benchmark

Runs the MASeval LLM findings pipeline on the **TraceElephant** failure-attribution
benchmark ([HF dataset](https://huggingface.co/datasets/TraceElephant/TraceElephant),
[paper](https://arxiv.org/abs/2604.22708), *"Seeing the Whole Elephant"*, ACL 2026).

This mirrors the `agentrx` example: our multi-evaluator judge + `EvidenceVerifier`
+ report aggregation produce a per-trace agent/step localization, which we score
against TraceElephant's gold `mistake_agent` / `mistake_step`.

## Benchmark

220 annotated **failure** traces (of 380 total executions) from three multi-agent
systems, each fully observable (agent inputs *and* outputs, inter-agent messages,
tool calls):

| System | Runs dirs | Traces |
|---|---|--:|
| Captain-Agent | `captain-runs-{assistantbench,gaia}` | 85 |
| Magentic-One | `magentic-runs-{assistant-bench,gaia}` | 91 |
| SWE-Agent | `swe-agent-runs-swe-bench` | 44 |

Each trace is annotated with one decisive failure: the **responsible agent** and
the **decisive step** (earliest inevitable error, 1-based over the history).

## Data

The dataset ships as a single `data.zip` on HuggingFace. Download + extract into
`./data` (git-ignored):

```python
from huggingface_hub import hf_hub_download
import zipfile
z = hf_hub_download("TraceElephant/TraceElephant", "data.zip", repo_type="dataset")
zipfile.ZipFile(z).extractall("examples/trace_elephant")  # -> examples/trace_elephant/data/
```

`trace_elephant_data.py` walks `data/{system}-runs-*/{task}/` and normalizes both
on-disk shapes (`trace_metadata.json`+`step_records.json`, or `summary.json`+
`history.json`) into a `history` of `{name, content}` steps. The gold `mistake_step`
is the 1-based history position, so spans are keyed by that index (evaluators cite
it in `evidence[i].idx`).

## Run

```bash
# 1) generate findings (11 LLM evaluators + EvidenceVerifier + report + FinalAnswerVerifier)
python launch_findings_judges.py --system all --model google/gemini-2.5-flash
#    or per system: --system captain | magentic | swe

# 2) score agent/step localization vs TraceElephant gold
python calculate_agent_step_accuracy.py --system all --step-tolerance 1
```

`FinalAnswerVerifier` runs in **no-ground-truth mode**: with `gt=None` it routes to
the MAS Task Completion judge (assesses completion from the trace alone) and feeds
the report's `answer_status`. TraceElephant has no gold final answer for the judge,
so this is the right mode — same as `agentrx`.

## Metric

Who&When-parity single-target scoring (TraceElephant gold is one agent + one step):

- **Agent Top-1** — the report's primary culprit is the gold `mistake_agent`
  (names matched via the repo's `_normalize_agent` convention).
- **Step Top-1** (`first_idx_mode=top_ranked`) — the top-ranked predicted span is
  the gold `mistake_step`; `--step-tolerance k` allows ±k.
- **Agent/Step Hit** — gold appears anywhere in the predicted set (reference).

> Note: TraceElephant's own `evaluate.py` uses a lenient `str(gold_step) in
> predicted_step` membership check, not exact/±k — keep our numbers in their own
> column rather than lining them up against the Who&When ±k tables.
