# Who&When HC — verifier=none

Generated: `2026-07-09 20:51:49`

## Notes

EvidenceVerifier ablation, mode='none'. non_llm_validators are NOT counted. Reports rebuilt from raw LLM findings under this gating.

## Experiment setup

| Field | Value |
|---|---|
| Experiment | Who&When HC — verifier=none |
| Model / judge | Gemini |
| Dataset | Who&When / Hand-Crafted |
| Run label | verifier ablation (LLM findings only, no validators) |
| Prediction glob | examples/who_and_when/who&when_hand_gemini_llmconfirm_scratch/*.json |
| Prediction files | 58 |
| HF annotations | /home/barak/.cache/huggingface/hub/datasets--Kevin355--Who_and_When/snapshots/59b9fcba1aaed7bbf206b5f4d3c68b8face2f49c/Hand-Crafted.parquet |
| Annotation rows | 58 |
| Matched annotations | 58 |
| Unmatched predictions | 0 |
| ID column | auto |
| Agent annotation columns | auto |
| Step annotation columns | auto |
| Step tolerance | ±1 |
| Build missing report | True |
| Verifier mode (ablation) | none |
| EvidenceVerifier policy | no verifier: every LLM finding counts |

## Main results

| Group | Metric | Value | Count | Meaning |
|---|---:|---:|---:|---|
| Agent | Top-1 Acc | 55.2% | 58 | Primary culprit agent matches the annotation. |
| Agent | Hit Acc | 91.4% | 58 | At least one predicted culprit agent matches the annotation. |
| Agent | Exact Set Acc | 8.6% | 58 | Predicted agent set exactly equals the gold agent set. |
| Step | Top-1 Acc | 1.7% | 58 | First predicted problematic idx equals the gold idx. |
| Step | Hit Acc | 37.9% | 58 | At least one predicted idx exactly matches a gold idx. |
| Step | Hit Acc ±1 | 65.5% | 58 | At least one predicted numeric idx is within ±1 of a gold idx. |

## Diagnostic quality summary

| Field | Value |
|---|---:|
| Invalid findings excluded from main predictions | 0 |
| Agent examples | 58 |
| Step examples | 58 |

## Interpretation

- Agent localization: Hit Acc is 91.4%, Top-1 Acc is 55.2% (gap 36.2%). A large gap means the correct agent is often present but not ranked first.
- Step localization: exact Hit Acc is 37.9%, relaxed Hit Acc ±1 is 65.5% (gap 27.6%). A large gap suggests off-by-one or indexing mismatch.
- First-idx ranking: Step Top-1 Acc is 1.7%. If this is much lower than Step Hit Acc, use the full problematic_idxs list rather than only first_problem_idx.

## Example mismatches / review targets (top 20)

| # | File | Gold agents | Predicted agents | Agent hit | Gold idxs | Predicted idxs | Step hit ±1 | Invalid findings |
|---:|---|---|---|---:|---|---|---:|---:|
| 1 | gemini_findings_16.json | Assistant | WebSurfer, Orchestrator, FileSurfer | ❌ | 82 | 50, 60, 64, 48, 52, 20, … (+33) | ❌ | 0 |
| 2 | gemini_findings_26.json | Assistant | WebSurfer | ❌ | 16 | 10 | ❌ | 0 |
| 3 | gemini_findings_45.json | Orchestrator | Assistant | ❌ | 18 | 14, 20, 13, 25, 27 | ❌ | 0 |
| 4 | gemini_findings_50.json | WebSurfer | Orchestrator, Assistant | ❌ | 24 | 13, 14, 16, 15, 12, 11, … (+2) | ❌ | 0 |
| 5 | gemini_findings_11.json | WebSurfer | WebSurfer, Orchestrator | ✅ | 4 | 15, 11, 12, 14, 17, 13, … (+5) | ❌ | 0 |
| 6 | gemini_findings_12.json | Orchestrator | Orchestrator, FileSurfer, Assistant, ComputerTerminal | ✅ | 51 | 26, 20, 30, 12, 14, 16, … (+27) | ❌ | 0 |
| 7 | gemini_findings_15.json | FileSurfer | FileSurfer, Orchestrator, WebSurfer | ✅ | 32 | 19, 17, 15, 13, 14, 23, … (+10) | ❌ | 0 |
| 8 | gemini_findings_18.json | WebSurfer | Orchestrator, FileSurfer, WebSurfer | ✅ | 4 | 20, 11, 14, 15, 21, 24, … (+9) | ❌ | 0 |
| 9 | gemini_findings_2.json | WebSurfer | WebSurfer, Orchestrator | ✅ | 8 | 30, 32, 34, 36, 31, 35, … (+14) | ❌ | 0 |
| 10 | gemini_findings_20.json | WebSurfer | Orchestrator, WebSurfer, Assistant | ✅ | 5 | 40, 20, 13, 39, 34, 16, … (+40) | ❌ | 0 |
| 11 | gemini_findings_24.json | WebSurfer | WebSurfer, Orchestrator | ✅ | 8 | 20, 10, 22, 12, 14, 16, … (+2) | ❌ | 0 |
| 12 | gemini_findings_25.json | Assistant | Orchestrator, Assistant, WebSurfer | ✅ | 51 | 0, 13, 19, 7, 14, 17, … (+4) | ❌ | 0 |
| 13 | gemini_findings_32.json | Orchestrator | Orchestrator, WebSurfer | ✅ | 18 | 15, 14, 13, 10, 16, 12, … (+1) | ❌ | 0 |
| 14 | gemini_findings_39.json | WebSurfer | WebSurfer | ✅ | 8 | 4, 0 | ❌ | 0 |
| 15 | gemini_findings_41.json | FileSurfer | FileSurfer, Orchestrator, WebSurfer | ✅ | 4 | 13, 16, 11, 12, 14, 15, … (+3) | ❌ | 0 |
| 16 | gemini_findings_51.json | WebSurfer | WebSurfer | ✅ | 12 |  | ❌ | 0 |
| 17 | gemini_findings_53.json | Orchestrator | WebSurfer, Orchestrator | ✅ | 50 | 20, 30, 18, 38, 26, 12, … (+17) | ❌ | 0 |
| 18 | gemini_findings_55.json | WebSurfer | Orchestrator, WebSurfer | ✅ | 8 | 16, 15, 10, 48, 39, 27, … (+36) | ❌ | 0 |
| 19 | gemini_findings_8.json | WebSurfer | WebSurfer, Orchestrator | ✅ | 8 | 23, 20, 16, 17, 15, 22, … (+2) | ❌ | 0 |
| 20 | gemini_findings_9.json | WebSurfer | WebSurfer, Orchestrator | ✅ | 16 | 0, 6, 12, 13, 2, 10 | ❌ | 0 |

## Reproducibility

```json
{
  "experiment_name": "Who&When HC — verifier=none",
  "model_name": "Gemini",
  "dataset_name": "Who&When / Hand-Crafted",
  "run_label": "verifier ablation (LLM findings only, no validators)",
  "pred_glob": "examples/who_and_when/who&when_hand_gemini_llmconfirm_scratch/*.json",
  "hf_annotations": "/home/barak/.cache/huggingface/hub/datasets--Kevin355--Who_and_When/snapshots/59b9fcba1aaed7bbf206b5f4d3c68b8face2f49c/Hand-Crafted.parquet",
  "output_json_path": "/mnt/c/Users/barak/MASeval/examples/who_and_when/reports/verifier_ablation_hc_none.json",
  "output_md_path": "/mnt/c/Users/barak/MASeval/examples/who_and_when/reports/verifier_ablation_hc_none.md",
  "id_column": null,
  "agent_columns": null,
  "step_columns": null,
  "step_tolerance": 1,
  "build_missing_report": true,
  "verifier_mode": "none",
  "top_error_examples": 20,
  "notes": "EvidenceVerifier ablation, mode='none'. non_llm_validators are NOT counted. Reports rebuilt from raw LLM findings under this gating."
}
```
