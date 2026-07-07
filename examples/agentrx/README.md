# AgentRx findings evaluation

Runs the Who&When-style LLM findings judges on the
[microsoft/AgentRx](https://huggingface.co/datasets/microsoft/AgentRx) dataset and
scores agent/step localization against the AgentRx gold.

## Dataset

Two configs, loaded straight from the Hub (cached via `huggingface_hub`):

| config | trajectories (annotated) | join key |
|---|---|---|
| `magentic` | 44 of 58 | `trajectory_id` (UUID) |
| `tau` | 29 of 29 | annotation `"N"` → dataset `tau_retail_N` |

Each trajectory is `{trajectory_id, instruction, steps}` where `steps` is a list
of `{index, substeps:[{sub_index, role, content}]}` and `index` is the **1-based**
step number. Gold lives in the annotation file: `failures:[{step_number,
failed_agent, failure_category}]` plus a `root_cause` pointing at the single
decisive failure.

**Span alignment (the load-bearing invariant):** gold `step_number` *is* the step
`index`, so spans are keyed by `str(step["index"])` and the LLM cites that number
in `evidence[i].idx`. Never enumerate positions — that would offset spans out
of the gold's index space. See `agentrx_data.format_trace` / `idxs`.

## Files

- `agentrx_data.py` — loads + joins both configs, builds the trace text
  (index-keyed) and the gold table. Run it directly for a sanity dump.
- `launch_findings_judges.py` — runs the 11 LLM findings metrics per trajectory,
  writes `findings_{i}.json` (findings + evidence verification + report). No
  `FinalAnswerVerifier` (no gold answer) and no deterministic validators
  (AgentRx is not a validator format).
- `calculate_agent_step_accuracy.py` — scores `report.diagnostic_report` vs gold.
- `build_agent_step_accuracy_report.py` — same, plus a Markdown report (reuses the
  Who&When renderer). Writes to `reports/`.

## Usage

```bash
# 0. Auth: the runner needs an OpenRouter key (via a local .env or exported env)
echo "OPENROUTER_API_KEY=sk-or-..." > examples/agentrx/.env

# 1. Run the findings judges (one LLM call per metric per trajectory)
python examples/agentrx/launch_findings_judges.py --config magentic --model google/gemini-2.5-flash
python examples/agentrx/launch_findings_judges.py --config tau

# 2. Build the accuracy report (Markdown + JSON in reports/)
python examples/agentrx/build_agent_step_accuracy_report.py --config magentic --gold-scope all
python examples/agentrx/build_agent_step_accuracy_report.py --config magentic --gold-scope root_cause
```

Run scripts with the project venv (`maseval` installed, plus `pandas`,
`huggingface_hub`).

## Gold scope

- `--gold-scope all` (default): every annotated failure counts (set semantics) —
  agent Top-1 = predicted primary is *any* failed agent; step Hit = any predicted
  span is *any* failure step. Lenient recall.
- `--gold-scope root_cause`: only the single decisive failure counts (strict,
  Who&When parity) — predicted primary must be the root-cause agent/step.

## Verifier ablation

`--verifier-mode {none,strict,soft}` rebuilds the report under a given
EvidenceVerifier gating (same semantics as the Who&When ablation): `none` = all
findings count, `strict` = only `verified`, `soft` = `verified`+`weak` (default).

## Note on Step Top-1

`first_problem_span` is the *lowest-indexed* flagged span (usually the
system/human step), so **Step Top-1 is ~0 by construction** — read **Step Hit** /
**Step Hit ±1** for step localization. This matches the Who&When behavior and is
not a bug in this pipeline.
