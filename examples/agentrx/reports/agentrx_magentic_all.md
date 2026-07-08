# AgentRx magentic — gold=all

Generated: `2026-07-07 19:30:22`

## Notes

AgentRx / magentic, gold_scope='all'. Spans keyed by the native 1-based step index (== gold step_number). non_llm_validators are not used. Step Top-1 uses first_idx_mode='top_ranked' (top_ranked = the model's #1-ranked span), comparable to the paper's single-root-cause Step Acc; step_top1_pm1/pm3/pm5 add ±1/±3/±5 tolerance to match Table 6's Acc@±k.

## Experiment setup

| Field | Value |
|---|---|
| Experiment | AgentRx magentic — gold=all |
| Model / judge | google/gemini-2.5-flash |
| Dataset | AgentRx / Magentic-One |
| Run label | AgentRx findings (all gold) |
| Prediction glob | /mnt/c/Users/barak/MASeval/examples/agentrx/agentrx_magentic_findings |
| Prediction files | 44 |
| HF annotations | microsoft/AgentRx :: magentic (all) |
| Annotation rows | 44 |
| Matched annotations | 44 |
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
| Agent | Top-1 Acc | 72.7% | 44 | Primary culprit agent matches the annotation. |
| Agent | Hit Acc | 97.7% | 44 | At least one predicted culprit agent matches the annotation. |
| Agent | Exact Set Acc | 34.1% | 44 | Predicted agent set exactly equals the gold agent set. |
| Step | Top-1 Acc | 43.2% | 44 | First predicted problematic idx equals the gold idx. |
| Step | Hit Acc | 88.6% | 44 | At least one predicted idx exactly matches a gold idx. |
| Step | Hit Acc ±1 | 93.2% | 44 | At least one predicted numeric idx is within ±1 of a gold idx. |

## Diagnostic quality summary

| Field | Value |
|---|---:|
| Invalid findings excluded from main predictions | 10 |
| Agent examples | 44 |
| Step examples | 44 |

## Interpretation

- Agent localization: Hit Acc is 97.7%, Top-1 Acc is 72.7% (gap 25.0%). A large gap means the correct agent is often present but not ranked first.
- Step localization: exact Hit Acc is 88.6%, relaxed Hit Acc ±1 is 93.2% (gap 4.5%). A large gap suggests off-by-one or indexing mismatch.
- First-idx ranking: Step Top-1 Acc is 43.2%. If this is much lower than Step Hit Acc, use the full problematic_idxs list rather than only first_problem_idx.

## Example mismatches / review targets (top 13)

| # | File | Gold agents | Predicted agents | Agent hit | Gold idxs | Predicted idxs | Step hit ±1 | Invalid findings |
|---:|---|---|---|---:|---|---|---:|---:|
| 1 | findings_25.json | Assistant |  | ❌ | 17 |  | ❌ | 0 |
| 2 | findings_2.json | Websurfer | WebSurfer, Orchestrator | ✅ | 17 | 59, 63, 67, 75, 79, 83, … (+12) | ❌ | 0 |
| 3 | findings_4.json | Orchestrator | Orchestrator, WebSurfer | ✅ | 4 | 17, 2, 10, 13, 15, 1, … (+2) | ❌ | 0 |
| 4 | findings_29.json | Assistant | Assistant | ✅ | 12 | 13, 16, 1, 11 | ✅ | 0 |
| 5 | findings_8.json | Orchestrator, WebSurfer | Orchestrator, WebSurfer | ✅ | 7, 9, 10 | 25, 37, 23, 21, 11, 15, … (+3) | ✅ | 0 |
| 6 | findings_12.json | WebSurfer, Orchestrator | Orchestrator, WebSurfer | ✅ | 9, 10, 13, 14, 17, 18, … (+1) | 13, 11, 15, 14, 26, 32, … (+12) | ✅ | 2 |
| 7 | findings_21.json | WebSurfer, Orchestrator | WebSurfer, Orchestrator, Assistant | ✅ | 5, 13, 17, 25, 29, 33, … (+16) | 93, 115, 100, 126, 42, 53, … (+40) | ✅ | 2 |
| 8 | findings_16.json | WebSurfer, Orchestrator | Orchestrator, WebSurfer, Assistant | ✅ | 9, 10 | 30, 9, 25, 29, 1, 21, … (+5) | ✅ | 1 |
| 9 | findings_19.json | WebSurfer, Orchestrator | Orchestrator, WebSurfer, Assistant | ✅ | 6, 7, 10, 14, 18, 22, … (+20) | 113, 23, 27, 15, 104, 26, … (+24) | ✅ | 1 |
| 10 | findings_22.json | WebSurfer | Orchestrator, WebSurfer | ✅ | 5, 12, 16, 20, 24, 28, … (+13) | 25, 37, 56, 14, 64, 13, … (+20) | ✅ | 1 |
| 11 | findings_24.json | WebSurfer, Assistant | Orchestrator, WebSurfer | ✅ | 9, 21, 25 | 9, 2, 21, 22, 5 | ✅ | 1 |
| 12 | findings_28.json | Orchestrator | Orchestrator | ✅ | 2 | 2, 1, 5 | ✅ | 1 |
| 13 | findings_41.json | WebSurfer, Orchestrator | Orchestrator, WebSurfer, Assistant | ✅ | 17, 21, 31 | 13, 10, 21, 22, 31, 14, … (+11) | ✅ | 1 |

## Reproducibility

```json
{
  "experiment_name": "AgentRx magentic — gold=all",
  "model_name": "google/gemini-2.5-flash",
  "dataset_name": "AgentRx / Magentic-One",
  "run_label": "AgentRx findings (all gold)",
  "pred_glob": "/mnt/c/Users/barak/MASeval/examples/agentrx/agentrx_magentic_findings",
  "hf_annotations": "microsoft/AgentRx :: magentic (all)",
  "output_json_path": "/mnt/c/Users/barak/MASeval/examples/agentrx/reports/agentrx_magentic_all.json",
  "output_md_path": "/mnt/c/Users/barak/MASeval/examples/agentrx/reports/agentrx_magentic_all.md",
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
  "notes": "AgentRx / magentic, gold_scope='all'. Spans keyed by the native 1-based step index (== gold step_number). non_llm_validators are not used. Step Top-1 uses first_idx_mode='top_ranked' (top_ranked = the model's #1-ranked span), comparable to the paper's single-root-cause Step Acc; step_top1_pm1/pm3/pm5 add ±1/±3/±5 tolerance to match Table 6's Acc@±k."
}
```

## Paper comparison — Table 6 (root-cause Step Acc, top-1)

Top-1 span (first_idx_mode=top_ranked) vs gold at ±0/±1/±3/±5. The paper reports a single predicted root-cause step, so `gold_scope=root_cause` is the closest analog. Their AGENTRX is a trained method; ours is an untrained LLM judge — treat this as orientation, not parity.

| Source | Step Acc | Acc@±1 | Acc@±3 | Acc@±5 |
|---|---|---|---|---|
| Ours (magentic / all) | 43.2% | 52.3% | 65.9% | 72.7% |
| Paper AGENTRX — Magentic-One | 31.8% | 40.9% | 47.7% | 50.8% |
| Paper AGENTRX — Magentic-One* | 46.9% | 61.7% | 72.8% | 79.0% |
