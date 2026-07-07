# Who&When Agent/Step Localization

Generated: `2026-06-27 17:46:22`

## Experiment setup

| Field | Value |
|---|---|
| Experiment | Who&When Agent/Step Localization |
| Model / judge | Gemini |
| Dataset | Who&When / Hand-Crafted |
| Run label | v9 report |
| Prediction glob | /home/alina/Desktop/maseval-research/examples/who_and_when/who&when_hand_gemini_findings_v9_report/*.json |
| Prediction files | 58 |
| HF annotations | hf://datasets/Kevin355/Who_and_When/Hand-Crafted.parquet |
| Annotation rows | 58 |
| Matched annotations | 58 |
| Unmatched predictions | 0 |
| ID column | auto |
| Agent annotation columns | auto |
| Step annotation columns | auto |
| Step tolerance | ±1 |
| Build missing report | True |
| EvidenceVerifier policy | verified/weak are counted; invalid findings are excluded and left for review |

## Main results

| Group | Metric | Value | Count | Meaning |
|---|---:|---:|---:|---|
| Agent | Top-1 Acc | 53.4% | 58 | Primary culprit agent matches the annotation. |
| Agent | Hit Acc | 91.4% | 58 | At least one predicted culprit agent matches the annotation. |
| Agent | Exact Set Acc | 8.6% | 58 | Predicted agent set exactly equals the gold agent set. |
| Step | Top-1 Acc | 0.0% | 58 | First predicted problematic idx equals the gold idx. |
| Step | Hit Acc | 43.1% | 58 | At least one predicted idx exactly matches a gold idx. |
| Step | Hit Acc ±1 | 63.8% | 58 | At least one predicted numeric idx is within ±1 of a gold idx. |

## Diagnostic quality summary

| Field | Value |
|---|---:|
| Invalid findings excluded from main predictions | 116 |
| Agent examples | 58 |
| Step examples | 58 |

## Interpretation

- Agent localization: Hit Acc is 91.4%, Top-1 Acc is 53.4% (gap 37.9%). A large gap means the correct agent is often present but not ranked first.
- Step localization: exact Hit Acc is 43.1%, relaxed Hit Acc ±1 is 63.8% (gap 20.7%). A large gap suggests off-by-one or indexing mismatch.
- First-idx ranking: Step Top-1 Acc is 0.0%. If this is much lower than Step Hit Acc, use the full problematic_idxs list rather than only first_problem_idx.

## Example mismatches / review targets (top 20)

| # | File | Gold agents | Predicted agents | Agent hit | Gold idxs | Predicted idxs | Step hit ±1 | Invalid findings |
|---:|---|---|---|---:|---|---|---:|---:|
| 1 | gemini_findings_25.json | Assistant | Orchestrator, WebSurfer | ❌ | 51 | 0, 7, 13 | ❌ | 2 |
| 2 | gemini_findings_16.json | Assistant | WebSurfer, Orchestrator, FileSurfer | ❌ | 82 | 52, 50, 48, 54, 40, 41, … (+21) | ❌ | 0 |
| 3 | gemini_findings_26.json | Assistant |  | ❌ | 16 |  | ❌ | 0 |
| 4 | gemini_findings_50.json | WebSurfer | Orchestrator | ❌ | 24 | 13, 11, 10, 12, 14, 17, … (+1) | ❌ | 0 |
| 5 | gemini_findings_51.json | WebSurfer |  | ❌ | 12 |  | ❌ | 0 |
| 6 | gemini_findings_11.json | WebSurfer | Orchestrator, WebSurfer | ✅ | 4 | 12, 13, 11, 15, 10, 14, … (+3) | ❌ | 7 |
| 7 | gemini_findings_9.json | WebSurfer | WebSurfer | ✅ | 16 | 7, 13, 12 | ❌ | 7 |
| 8 | gemini_findings_37.json | WebSurfer | WebSurfer, Orchestrator | ✅ | 12 | 10, 4, 9 | ❌ | 5 |
| 9 | gemini_findings_39.json | WebSurfer | WebSurfer, Orchestrator | ✅ | 8 | 4, 0 | ❌ | 5 |
| 10 | gemini_findings_27.json | Orchestrator | Orchestrator, WebSurfer, Assistant | ✅ | 20 | 14, 10, 12, 13, 17, 11, … (+1) | ❌ | 4 |
| 11 | gemini_findings_48.json | Orchestrator | Orchestrator | ✅ | 29 | 21, 18, 17, 19, 20, 22, … (+1) | ❌ | 4 |
| 12 | gemini_findings_15.json | FileSurfer | Orchestrator, FileSurfer, WebSurfer | ✅ | 32 | 13, 17, 11, 15, 14, 19, … (+12) | ❌ | 2 |
| 13 | gemini_findings_18.json | WebSurfer | Orchestrator, FileSurfer, WebSurfer | ✅ | 4 | 12, 16, 11, 13, 17, 29, … (+15) | ❌ | 2 |
| 14 | gemini_findings_40.json | WebSurfer | Orchestrator, WebSurfer, FileSurfer | ✅ | 8 | 23, 17, 36, 21, 15, 40, … (+49) | ❌ | 2 |
| 15 | gemini_findings_41.json | FileSurfer | Orchestrator, FileSurfer | ✅ | 4 | 12, 14, 15, 11, 13, 16, … (+3) | ❌ | 2 |
| 16 | gemini_findings_0.json | WebSurfer | Orchestrator, WebSurfer, FileSurfer | ✅ | 3 | 15, 11, 12, 23, 17, 21, … (+24) | ❌ | 1 |
| 17 | gemini_findings_2.json | WebSurfer | WebSurfer, Orchestrator | ✅ | 8 | 30, 32, 31, 33, 29, 34, … (+4) | ❌ | 1 |
| 18 | gemini_findings_53.json | Orchestrator | WebSurfer, Orchestrator | ✅ | 50 | 27, 17, 30, 20, 29, 21, … (+21) | ❌ | 1 |
| 19 | gemini_findings_12.json | Orchestrator | Orchestrator, FileSurfer, Assistant, ComputerTerminal | ✅ | 51 | 11, 12, 14, 15, 17, 13, … (+24) | ❌ | 0 |
| 20 | gemini_findings_32.json | Orchestrator | Orchestrator | ✅ | 18 | 13, 10, 11, 12, 0, 9, … (+3) | ❌ | 0 |

## Reproducibility

```json
{
  "experiment_name": "Who&When Agent/Step Localization",
  "model_name": "Gemini",
  "dataset_name": "Who&When / Hand-Crafted",
  "run_label": "v9 report",
  "pred_glob": "/home/alina/Desktop/maseval-research/examples/who_and_when/who&when_hand_gemini_findings_v9_report/*.json",
  "hf_annotations": "hf://datasets/Kevin355/Who_and_When/Hand-Crafted.parquet",
  "output_json_path": "agent_step_metrics.json",
  "output_md_path": "agent_step_metrics.md",
  "id_column": null,
  "agent_columns": null,
  "step_columns": null,
  "step_tolerance": 1,
  "build_missing_report": true,
  "top_error_examples": 20,
  "notes": null
}
```
