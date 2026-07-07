# AgentRx tau — gold=all

Generated: `2026-07-07 19:33:26`

## Notes

AgentRx / tau, gold_scope='all'. Spans keyed by the native 1-based step index (== gold step_number). non_llm_validators are not used. Step Top-1 uses first_idx_mode='top_ranked' (top_ranked = the model's #1-ranked span), comparable to the paper's single-root-cause Step Acc; step_top1_pm1/pm3/pm5 add ±1/±3/±5 tolerance to match Table 6's Acc@±k.

## Experiment setup

| Field | Value |
|---|---|
| Experiment | AgentRx tau — gold=all |
| Model / judge | google/gemini-2.5-flash |
| Dataset | AgentRx / tau-bench retail |
| Run label | AgentRx findings (all gold) |
| Prediction glob | /mnt/c/Users/barak/MASeval/examples/agentrx/agentrx_tau_findings |
| Prediction files | 29 |
| HF annotations | microsoft/AgentRx :: tau (all) |
| Annotation rows | 29 |
| Matched annotations | 29 |
| Unmatched predictions | 0 |
| ID column | trajectory_id |
| Agent annotation columns | mistake_agents |
| Step annotation columns | mistake_steps |
| Step tolerance | ±1 |
| Build missing report | True |
| Verifier mode (ablation) | soft (stored) |
| EvidenceVerifier policy | verified/weak are counted; invalid findings are excluded and left for review |

## Main results

| Group | Metric | Value | Count | Meaning |
|---|---:|---:|---:|---|
| Agent | Top-1 Acc | 48.3% | 29 | Primary culprit agent matches the annotation. |
| Agent | Hit Acc | 48.3% | 29 | At least one predicted culprit agent matches the annotation. |
| Agent | Exact Set Acc | 0.0% | 29 | Predicted agent set exactly equals the gold agent set. |
| Step | Top-1 Acc | 6.9% | 29 | First predicted problematic idx equals the gold idx. |
| Step | Hit Acc | 44.8% | 29 | At least one predicted idx exactly matches a gold idx. |
| Step | Hit Acc ±1 | 44.8% | 29 | At least one predicted numeric idx is within ±1 of a gold idx. |

## Diagnostic quality summary

| Field | Value |
|---|---:|
| Invalid findings excluded from main predictions | 3 |
| Agent examples | 29 |
| Step examples | 29 |

## Interpretation

- Agent localization: Hit Acc is 48.3%, Top-1 Acc is 48.3% (gap 0.0%). A large gap means the correct agent is often present but not ranked first.
- Step localization: exact Hit Acc is 44.8%, relaxed Hit Acc ±1 is 44.8% (gap 0.0%). A large gap suggests off-by-one or indexing mismatch.
- First-idx ranking: Step Top-1 Acc is 6.9%. If this is much lower than Step Hit Acc, use the full problematic_idxs list rather than only first_problem_idx.

## Example mismatches / review targets (top 19)

| # | File | Gold agents | Predicted agents | Agent hit | Gold idxs | Predicted idxs | Step hit ±1 | Invalid findings |
|---:|---|---|---|---:|---|---|---:|---:|
| 1 | findings_22.json | User |  | ❌ | 20 |  | ❌ | 1 |
| 2 | findings_0.json | Assistant |  | ❌ | 3, 7 |  | ❌ | 0 |
| 3 | findings_1.json | Assistant |  | ❌ | 3, 7 |  | ❌ | 0 |
| 4 | findings_11.json | Assistant |  | ❌ | 17 |  | ❌ | 0 |
| 5 | findings_13.json | Assistant |  | ❌ | 19 |  | ❌ | 0 |
| 6 | findings_14.json | User |  | ❌ | 32 |  | ❌ | 0 |
| 7 | findings_15.json | User |  | ❌ | 14 |  | ❌ | 0 |
| 8 | findings_2.json | Assistant |  | ❌ | 15 |  | ❌ | 0 |
| 9 | findings_20.json | User |  | ❌ | 20 |  | ❌ | 0 |
| 10 | findings_21.json | Assistant |  | ❌ | 26 |  | ❌ | 0 |
| 11 | findings_23.json | Assistant |  | ❌ | 37 |  | ❌ | 0 |
| 12 | findings_24.json | User | assistant | ❌ | 34 | 21, 22, user, 2, 29 | ❌ | 0 |
| 13 | findings_27.json | Assistant |  | ❌ | 57 |  | ❌ | 0 |
| 14 | findings_8.json | Assistant |  | ❌ | 11 |  | ❌ | 0 |
| 15 | findings_10.json | Assistant | assistant, Retail agent | ✅ | 37 | 21, 15, 1, 19, 22 | ❌ | 0 |
| 16 | findings_26.json | Assistant | assistant, Retail agent | ✅ | 35 | 45, 42, 43, 26, 41, 44 | ❌ | 0 |
| 17 | findings_16.json | User | assistant, Retail agent policy, Retail agent | ❌ | 32 | 23, 33, 1, 18, 32 | ✅ | 0 |
| 18 | findings_3.json | Assistant | assistant, Retail agent | ✅ | 19 | 23, 22, 19, 1, 12, 15 | ✅ | 1 |
| 19 | findings_7.json | Assistant | assistant, Retail agent | ✅ | 31 | 26, 32, 31, 33, 1, 28, … (+1) | ✅ | 1 |

## Reproducibility

```json
{
  "experiment_name": "AgentRx tau — gold=all",
  "model_name": "google/gemini-2.5-flash",
  "dataset_name": "AgentRx / tau-bench retail",
  "run_label": "AgentRx findings (all gold)",
  "pred_glob": "/mnt/c/Users/barak/MASeval/examples/agentrx/agentrx_tau_findings",
  "hf_annotations": "microsoft/AgentRx :: tau (all)",
  "output_json_path": "/mnt/c/Users/barak/MASeval/examples/agentrx/reports/agentrx_tau_all.json",
  "output_md_path": "/mnt/c/Users/barak/MASeval/examples/agentrx/reports/agentrx_tau_all.md",
  "id_column": "trajectory_id",
  "agent_columns": [
    "mistake_agents"
  ],
  "step_columns": [
    "mistake_steps"
  ],
  "step_tolerance": 1,
  "build_missing_report": true,
  "top_error_examples": 20,
  "notes": "AgentRx / tau, gold_scope='all'. Spans keyed by the native 1-based step index (== gold step_number). non_llm_validators are not used. Step Top-1 uses first_idx_mode='top_ranked' (top_ranked = the model's #1-ranked span), comparable to the paper's single-root-cause Step Acc; step_top1_pm1/pm3/pm5 add ±1/±3/±5 tolerance to match Table 6's Acc@±k."
}
```

## Paper comparison — Table 6 (root-cause Step Acc, top-1)

Top-1 span (first_idx_mode=top_ranked) vs gold at ±0/±1/±3/±5. The paper reports a single predicted root-cause step, so `gold_scope=root_cause` is the closest analog. Their AGENTRX is a trained method; ours is an untrained LLM judge — treat this as orientation, not parity.

| Source | Step Acc | Acc@±1 | Acc@±3 | Acc@±5 |
|---|---|---|---|---|
| Ours (tau / all) | 6.9% | 17.2% | 27.6% | 34.5% |
| Paper AGENTRX — τ-Bench | 54.0% | 59.8% | 72.4% | 83.9% |
