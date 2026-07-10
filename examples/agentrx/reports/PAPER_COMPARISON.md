# AgentRx paper vs. our MASeval judge — what is reliably comparable

Paper: *AgentRx: Diagnosing AI Agent Failures from Execution Trajectories* (Barke et al.,
arXiv:2602.02475). Benchmark = **115 failed trajectories**: τ-bench (29) + Flash (42) +
Magentic (44, randomly sampled from Who&When).

Our numbers: gemini-2.5-flash, 11 LLM evaluators + EvidenceVerifier + report aggregation,
`first_idx_mode=top_ranked`. The paper's attribution metric is a **single critical (root-cause)
failure** target, so our comparable numbers are the **`root_cause` gold scope** (our `all` scope
counts hitting *any* failure step and is NOT comparable to the paper).

## The N problem (why most cells are not comparable)

| Paper table | Metric | τ-bench N | Magentic N | Flash N |
|---|---|---|---|---|
| Table 2 | Agent acc + Step acc (single target) | 29 | **16** | — |
| Table 6 | Step Acc + Acc@±1/±3/±5 (single step, tolerance) | 29 | **44** | 42 |
| Table 6 | Magentic∗ = filtered subset ≤50 steps | — | **27** | — |

- HF `magentic_dataset.jsonl` has **58** raw traces; only **44** are annotated
  (`magentic_one.jsonl`) — those 44 are the paper's Magentic domain, and what we scored.
- **Table 2's Magentic uses only 16** trajectories (a further subset the paper does not
  enumerate) — so Table 2's Magentic cells are **not reproducible** from the released data.
- **Flash (42) is not on HF** — we have no Flash data at all.
- **Magentic∗ (27)** would require us to re-filter our 44 to ≤50 steps.

## ✅ Reliably comparable (matched N + matched metric definition)

### τ-bench — N = 29 (ours) = 29 (paper), both Table 2 and Table 6

| Metric | **Ours** (gemini-flash) | Paper W&W∗ (GPT-5) | Paper Baseline (GPT-5) | Paper AgentRx (GPT-5) |
|---|--:|--:|--:|--:|
| Agent acc (root cause) | **41.4** | 62 | 75.9 | — |
| Step acc (exact, ±0) | **0.0** | 17.2 | 32.2 | 54.0 |
| Step Acc@±1 | **10.3** | — | 36.8 | 59.8 |
| Step Acc@±3 | **20.7** | — | 50.6 | 72.4 |
| Step Acc@±5 | **27.6** | — | 66.7 | 83.9 |

(For reference, our τ-bench `step_hit` exact = 34.5% — the root-cause step is often *in* our
predicted set, but it is essentially never our single top-ranked span, so top-1 step acc ≈ 0.)

### Magentic — N = 44 (ours) = 44 (paper), Table 6 STEP accuracy only

| Metric | **Ours** (gemini-flash) | Paper Baseline (GPT-5) | Paper AgentRx (GPT-5) |
|---|--:|--:|--:|
| Step acc (exact, ±0) | **15.9** | 31.8 | 31.8 |
| Step Acc@±1 | **20.5** | 40.9 | 40.9 |
| Step Acc@±3 | **31.8** | 50.0 | 47.7 |
| Step Acc@±5 | **34.1** | 53.3 | 50.8 |

## ❌ Not reliably comparable

- **Magentic agent accuracy** (Table 2, N=16): different, unenumerated subset. Their numbers:
  W&W∗ 6.2%, Baseline 81.2% — on 16 traj, not our 44.
- **Flash** (any metric): not released on HF.
- **Magentic∗** (Table 6, N=27): filtered subset we did not construct.
- Our `all`-scope step numbers (magentic all: step-top1 43.2%) — different metric (any failure
  step, not the single root cause); do not line these up against the paper.

## Caveats on the ✅ cells

1. **Model gap**: paper is **GPT-5**; we are **gemini-2.5-flash**. Not an apples-to-apples model
   comparison.
2. **Different system**: their W&W∗/Baseline/AgentRx are targeted root-cause predictors; ours is a
   multi-evaluator findings aggregator. W&W∗ is the closest in spirit (also Who&When-derived).
3. **Single-target metric**: paper scores one predicted root-cause step/agent vs the gold critical
   failure — matched by our `root_cause` scope + `top_ranked` top-1 with ±k tolerance.
