# AgentRx magentic — gold=root_cause

Generated: `2026-07-07 19:31:55`

## Notes

AgentRx / magentic, gold_scope='root_cause'. Spans keyed by the native 1-based step index (== gold step_number). non_llm_validators are not used. Step Top-1 uses first_idx_mode='top_ranked' (top_ranked = the model's #1-ranked span), comparable to the paper's single-root-cause Step Acc; step_top1_pm1/pm3/pm5 add ±1/±3/±5 tolerance to match Table 6's Acc@±k.

## Experiment setup

| Field | Value |
|---|---|
| Experiment | AgentRx magentic — gold=root_cause |
| Model / judge | google/gemini-2.5-flash |
| Dataset | AgentRx / Magentic-One |
| Run label | AgentRx findings (root_cause gold) |
| Prediction glob | /mnt/c/Users/barak/MASeval/examples/agentrx/agentrx_magentic_findings |
| Prediction files | 44 |
| HF annotations | microsoft/AgentRx :: magentic (root_cause) |
| Annotation rows | 44 |
| Matched annotations | 44 |
| Unmatched predictions | 0 |
| ID column | trajectory_id |
| Agent annotation columns | root_cause_agent |
| Step annotation columns | root_cause_step |
| Step tolerance | ±1 |
| Build missing report | True |
| Verifier mode (ablation) | soft (stored) |
| EvidenceVerifier policy | verified/weak are counted; invalid findings are excluded and left for review |

## Main results

| Group | Metric | Value | Count | Meaning |
|---|---:|---:|---:|---|
| Agent | Top-1 Acc | 63.6% | 44 | Primary culprit agent matches the annotation. |
| Agent | Hit Acc | 97.7% | 44 | At least one predicted culprit agent matches the annotation. |
| Agent | Exact Set Acc | 4.5% | 44 | Predicted agent set exactly equals the gold agent set. |
| Step | Top-1 Acc | 15.9% | 44 | First predicted problematic idx equals the gold idx. |
| Step | Hit Acc | 56.8% | 44 | At least one predicted idx exactly matches a gold idx. |
| Step | Hit Acc ±1 | 77.3% | 44 | At least one predicted numeric idx is within ±1 of a gold idx. |

## Diagnostic quality summary

| Field | Value |
|---|---:|
| Invalid findings excluded from main predictions | 10 |
| Agent examples | 44 |
| Step examples | 44 |

## Interpretation

- Agent localization: Hit Acc is 97.7%, Top-1 Acc is 63.6% (gap 34.1%). A large gap means the correct agent is often present but not ranked first.
- Step localization: exact Hit Acc is 56.8%, relaxed Hit Acc ±1 is 77.3% (gap 20.5%). A large gap suggests off-by-one or indexing mismatch.
- First-idx ranking: Step Top-1 Acc is 15.9%. If this is much lower than Step Hit Acc, use the full problematic_idxs list rather than only first_problem_idx.

## Example mismatches / review targets (top 20)

| # | File | Gold agents | Predicted agents | Agent hit | Gold idxs | Predicted idxs | Step hit ±1 | Invalid findings |
|---:|---|---|---|---:|---|---|---:|---:|
| 1 | findings_25.json | Assistant |  | ❌ | 17 |  | ❌ | 0 |
| 2 | findings_21.json | WebSurfer | WebSurfer, Orchestrator, Assistant | ✅ | 5 | 93, 115, 100, 126, 42, 53, … (+40) | ❌ | 2 |
| 3 | findings_15.json | WebSurfer | WebSurfer, Orchestrator, FileSurfer | ✅ | 13 | 106, 110, 114, 118, 98, 102, … (+24) | ❌ | 0 |
| 4 | findings_17.json | Orchestrator | FileSurfer, Orchestrator, WebSurfer | ✅ | 15 | 21, 36, 51, 29, 18, 22, … (+6) | ❌ | 0 |
| 5 | findings_18.json | Orchestrator | WebSurfer, Orchestrator | ✅ | 10 | 13, 21, 25, 22, 14, 26, … (+1) | ❌ | 0 |
| 6 | findings_2.json | Websurfer | WebSurfer, Orchestrator | ✅ | 17 | 59, 63, 67, 75, 79, 83, … (+12) | ❌ | 0 |
| 7 | findings_27.json | Orchestrator | Orchestrator, WebSurfer | ✅ | 7 | 10, 12, 1, 4, 5, 9 | ❌ | 0 |
| 8 | findings_4.json | Orchestrator | Orchestrator, WebSurfer | ✅ | 4 | 17, 2, 10, 13, 15, 1, … (+2) | ❌ | 0 |
| 9 | findings_42.json | Orchestrator | Orchestrator, WebSurfer | ✅ | 4 | 21, 25, 26, 52, 9, 13, … (+12) | ❌ | 0 |
| 10 | findings_7.json | WebSurfer | Orchestrator, WebSurfer | ✅ | 5 | 29, 33, 55, 67, 90, 113, … (+22) | ❌ | 0 |
| 11 | findings_12.json | Orchestrator | Orchestrator, WebSurfer | ✅ | 10 | 13, 11, 15, 14, 26, 32, … (+12) | ✅ | 2 |
| 12 | findings_19.json | Orchestrator | Orchestrator, WebSurfer, Assistant | ✅ | 7 | 113, 23, 27, 15, 104, 26, … (+24) | ✅ | 1 |
| 13 | findings_22.json | WebSurfer | Orchestrator, WebSurfer | ✅ | 5 | 25, 37, 56, 14, 64, 13, … (+20) | ✅ | 1 |
| 14 | findings_1.json | Orchestrator | Orchestrator, WebSurfer | ✅ | 10 | 89, 11, 13, 15, 17, 19, … (+5) | ✅ | 0 |
| 15 | findings_13.json | Orchestrator | Orchestrator, FileSurfer, WebSurfer, Assistant | ✅ | 4 | 19, 91, 6, 14, 18, 62, … (+44) | ✅ | 0 |
| 16 | findings_20.json | Orchestrator | Orchestrator, WebSurfer, Assistant, ComputerTerminal | ✅ | 27 | 102, 68, 72, 106, 107, 14, … (+34) | ✅ | 0 |
| 17 | findings_29.json | Assistant | Assistant | ✅ | 12 | 13, 16, 1, 11 | ✅ | 0 |
| 18 | findings_5.json | Orchestrator | Orchestrator, WebSurfer | ✅ | 7 | 5, 6, 8 | ✅ | 0 |
| 19 | findings_8.json | Orchestrator | Orchestrator, WebSurfer | ✅ | 10 | 25, 37, 23, 21, 11, 15, … (+3) | ✅ | 0 |
| 20 | findings_16.json | Orchestrator | Orchestrator, WebSurfer, Assistant | ✅ | 10 | 30, 9, 25, 29, 1, 21, … (+5) | ✅ | 1 |

## Reproducibility

```json
{
  "experiment_name": "AgentRx magentic — gold=root_cause",
  "model_name": "google/gemini-2.5-flash",
  "dataset_name": "AgentRx / Magentic-One",
  "run_label": "AgentRx findings (root_cause gold)",
  "pred_glob": "/mnt/c/Users/barak/MASeval/examples/agentrx/agentrx_magentic_findings",
  "hf_annotations": "microsoft/AgentRx :: magentic (root_cause)",
  "output_json_path": "/mnt/c/Users/barak/MASeval/examples/agentrx/reports/agentrx_magentic_root_cause.json",
  "output_md_path": "/mnt/c/Users/barak/MASeval/examples/agentrx/reports/agentrx_magentic_root_cause.md",
  "id_column": "trajectory_id",
  "agent_columns": [
    "root_cause_agent"
  ],
  "step_columns": [
    "root_cause_step"
  ],
  "step_tolerance": 1,
  "build_missing_report": true,
  "top_error_examples": 20,
  "notes": "AgentRx / magentic, gold_scope='root_cause'. Spans keyed by the native 1-based step index (== gold step_number). non_llm_validators are not used. Step Top-1 uses first_idx_mode='top_ranked' (top_ranked = the model's #1-ranked span), comparable to the paper's single-root-cause Step Acc; step_top1_pm1/pm3/pm5 add ±1/±3/±5 tolerance to match Table 6's Acc@±k."
}
```

## Paper comparison — Table 6 (root-cause Step Acc, top-1)

Top-1 span (first_idx_mode=top_ranked) vs gold at ±0/±1/±3/±5. The paper reports a single predicted root-cause step, so `gold_scope=root_cause` is the closest analog. Their AGENTRX is a trained method; ours is an untrained LLM judge — treat this as orientation, not parity.

| Source | Step Acc | Acc@±1 | Acc@±3 | Acc@±5 |
|---|---|---|---|---|
| Ours (magentic / root_cause) | 15.9% | 20.5% | 31.8% | 34.1% |
| Paper AGENTRX — Magentic-One | 31.8% | 40.9% | 47.7% | 50.8% |
| Paper AGENTRX — Magentic-One* | 46.9% | 61.7% | 72.8% | 79.0% |
