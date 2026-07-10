# Who&When HC — verifier=strict

Generated: `2026-07-09 20:51:51`

## Notes

EvidenceVerifier ablation, mode='strict'. non_llm_validators are NOT counted. Reports rebuilt from raw LLM findings under this gating.

## Experiment setup

| Field | Value |
|---|---|
| Experiment | Who&When HC — verifier=strict |
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
| Verifier mode (ablation) | strict |
| EvidenceVerifier policy | only 'verified' findings count; weak + invalid go to review |

## Main results

| Group | Metric | Value | Count | Meaning |
|---|---:|---:|---:|---|
| Agent | Top-1 Acc | 46.6% | 58 | Primary culprit agent matches the annotation. |
| Agent | Hit Acc | 77.6% | 58 | At least one predicted culprit agent matches the annotation. |
| Agent | Exact Set Acc | 20.7% | 58 | Predicted agent set exactly equals the gold agent set. |
| Step | Top-1 Acc | 3.4% | 58 | First predicted problematic idx equals the gold idx. |
| Step | Hit Acc | 17.2% | 58 | At least one predicted idx exactly matches a gold idx. |
| Step | Hit Acc ±1 | 37.9% | 58 | At least one predicted numeric idx is within ±1 of a gold idx. |

## Diagnostic quality summary

| Field | Value |
|---|---:|
| Invalid findings excluded from main predictions | 92 |
| Agent examples | 58 |
| Step examples | 58 |

## Interpretation

- Agent localization: Hit Acc is 77.6%, Top-1 Acc is 46.6% (gap 31.0%). A large gap means the correct agent is often present but not ranked first.
- Step localization: exact Hit Acc is 17.2%, relaxed Hit Acc ±1 is 37.9% (gap 20.7%). A large gap suggests off-by-one or indexing mismatch.
- First-idx ranking: Step Top-1 Acc is 3.4%. If this is much lower than Step Hit Acc, use the full problematic_idxs list rather than only first_problem_idx.

## Example mismatches / review targets (top 20)

| # | File | Gold agents | Predicted agents | Agent hit | Gold idxs | Predicted idxs | Step hit ±1 | Invalid findings |
|---:|---|---|---|---:|---|---|---:|---:|
| 1 | gemini_findings_31.json | WebSurfer |  | ❌ | 4 |  | ❌ | 7 |
| 2 | gemini_findings_33.json | WebSurfer |  | ❌ | 8 |  | ❌ | 5 |
| 3 | gemini_findings_10.json | WebSurfer | Orchestrator | ❌ | 4 | 0, 13, 10, 12, 18, 24 | ❌ | 3 |
| 4 | gemini_findings_50.json | WebSurfer | Orchestrator, Assistant | ❌ | 24 | 14, 15, 16, 12, 13 | ❌ | 2 |
| 5 | gemini_findings_42.json | WebSurfer | Orchestrator, Assistant | ❌ | 16 | 29, 20, 0, 4, 19, 21, … (+5) | ❌ | 1 |
| 6 | gemini_findings_51.json | WebSurfer |  | ❌ | 12 |  | ❌ | 1 |
| 7 | gemini_findings_16.json | Assistant | WebSurfer, Orchestrator, FileSurfer | ❌ | 82 | 50, 48, 52, 20, 22, 28, … (+16) | ❌ | 0 |
| 8 | gemini_findings_26.json | Assistant | WebSurfer | ❌ | 16 | 10 | ❌ | 0 |
| 9 | gemini_findings_45.json | Orchestrator | Assistant | ❌ | 18 | 14, 20 | ❌ | 0 |
| 10 | gemini_findings_6.json | WebSurfer |  | ❌ | 8 |  | ❌ | 0 |
| 11 | gemini_findings_38.json | WebSurfer | Orchestrator, WebSurfer | ✅ | 32 | 7, 8, 9, 11, 13, 38 | ❌ | 6 |
| 12 | gemini_findings_9.json | WebSurfer | WebSurfer, Orchestrator | ✅ | 16 | 0, 6, 12, 2, 10 | ❌ | 6 |
| 13 | gemini_findings_39.json | WebSurfer | WebSurfer | ✅ | 8 | 0, 4 | ❌ | 5 |
| 14 | gemini_findings_41.json | FileSurfer | FileSurfer, Orchestrator, WebSurfer | ✅ | 4 | 13, 16, 11, 12, 14, 10, … (+3) | ❌ | 5 |
| 15 | gemini_findings_11.json | WebSurfer | Orchestrator, WebSurfer | ✅ | 4 | 15, 11, 12, 14, 17, 20, … (+5) | ❌ | 4 |
| 16 | gemini_findings_37.json | WebSurfer | WebSurfer | ✅ | 12 | 10, 4 | ❌ | 4 |
| 17 | gemini_findings_18.json | WebSurfer | Orchestrator, FileSurfer, WebSurfer | ✅ | 4 | 20, 11, 12, 14, 15, 21, … (+8) | ❌ | 3 |
| 18 | gemini_findings_24.json | WebSurfer | WebSurfer | ✅ | 8 | 20 | ❌ | 3 |
| 19 | gemini_findings_53.json | Orchestrator | WebSurfer, Orchestrator | ✅ | 50 | 30, 18, 20, 29, 33, 10, … (+8) | ❌ | 3 |
| 20 | gemini_findings_57.json | WebSurfer | WebSurfer | ✅ | 12 | 29, 6, 8, 23, 27 | ❌ | 2 |

## Reproducibility

```json
{
  "experiment_name": "Who&When HC — verifier=strict",
  "model_name": "Gemini",
  "dataset_name": "Who&When / Hand-Crafted",
  "run_label": "verifier ablation (LLM findings only, no validators)",
  "pred_glob": "examples/who_and_when/who&when_hand_gemini_llmconfirm_scratch/*.json",
  "hf_annotations": "/home/barak/.cache/huggingface/hub/datasets--Kevin355--Who_and_When/snapshots/59b9fcba1aaed7bbf206b5f4d3c68b8face2f49c/Hand-Crafted.parquet",
  "output_json_path": "/mnt/c/Users/barak/MASeval/examples/who_and_when/reports/verifier_ablation_hc_strict.json",
  "output_md_path": "/mnt/c/Users/barak/MASeval/examples/who_and_when/reports/verifier_ablation_hc_strict.md",
  "id_column": null,
  "agent_columns": null,
  "step_columns": null,
  "step_tolerance": 1,
  "build_missing_report": true,
  "verifier_mode": "strict",
  "top_error_examples": 20,
  "notes": "EvidenceVerifier ablation, mode='strict'. non_llm_validators are NOT counted. Reports rebuilt from raw LLM findings under this gating."
}
```
