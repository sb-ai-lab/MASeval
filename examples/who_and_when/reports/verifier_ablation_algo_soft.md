# Who&When ALGO — verifier=soft

Generated: `2026-07-08 16:08:30`

## Notes

EvidenceVerifier ablation, mode='soft'. non_llm_validators are NOT counted. Reports rebuilt from raw LLM findings under this gating.

## Experiment setup

| Field | Value |
|---|---|
| Experiment | Who&When ALGO — verifier=soft |
| Model / judge | Gemini |
| Dataset | Who&When / Algorithm-Generated |
| Run label | verifier ablation (LLM findings only, no validators) |
| Prediction glob | C:\Users\barak\MASeval\examples\who_and_when/who&when_algo_gemini_idx_msg_v3/*.json |
| Prediction files | 126 |
| HF annotations | C:\Users\barak\.cache\huggingface\hub\datasets--Kevin355--Who_and_When\snapshots\59b9fcba1aaed7bbf206b5f4d3c68b8face2f49c\Algorithm-Generated.parquet |
| Annotation rows | 126 |
| Matched annotations | 126 |
| Unmatched predictions | 0 |
| ID column | auto |
| Agent annotation columns | auto |
| Step annotation columns | auto |
| Step tolerance | ±1 |
| Build missing report | True |
| Verifier mode (ablation) | soft |
| EvidenceVerifier policy | verified/weak are counted; invalid findings are excluded and left for review |

## Main results

| Group | Metric | Value | Count | Meaning |
|---|---:|---:|---:|---|
| Agent | Top-1 Acc | 47.6% | 126 | Primary culprit agent matches the annotation. |
| Agent | Hit Acc | 66.7% | 126 | At least one predicted culprit agent matches the annotation. |
| Agent | Exact Set Acc | 12.7% | 126 | Predicted agent set exactly equals the gold agent set. |
| Step | Top-1 Acc | 23.8% | 126 | First predicted problematic idx equals the gold idx. |
| Step | Hit Acc | 59.5% | 126 | At least one predicted idx exactly matches a gold idx. |
| Step | Hit Acc ±1 | 68.3% | 126 | At least one predicted numeric idx is within ±1 of a gold idx. |

## Diagnostic quality summary

| Field | Value |
|---|---:|
| Invalid findings excluded from main predictions | 5 |
| Agent examples | 126 |
| Step examples | 126 |

## Interpretation

- Agent localization: Hit Acc is 66.7%, Top-1 Acc is 47.6% (gap 19.0%). A large gap means the correct agent is often present but not ranked first.
- Step localization: exact Hit Acc is 59.5%, relaxed Hit Acc ±1 is 68.3% (gap 8.7%). A large gap suggests off-by-one or indexing mismatch.
- First-idx ranking: Step Top-1 Acc is 23.8%. If this is much lower than Step Hit Acc, use the full problematic_idxs list rather than only first_problem_idx.

## Example mismatches / review targets (top 20)

| # | File | Gold agents | Predicted agents | Agent hit | Gold idxs | Predicted idxs | Step hit ±1 | Invalid findings |
|---:|---|---|---|---:|---|---|---:|---:|
| 1 | gemini_findings_101.json | PythonDebugging_Expert |  | ❌ | 0 |  | ❌ | 0 |
| 2 | gemini_findings_106.json | Culinary_Awards_Expert |  | ❌ | 2 |  | ❌ | 0 |
| 3 | gemini_findings_107.json | DataAnalysis_Expert |  | ❌ | 0 |  | ❌ | 0 |
| 4 | gemini_findings_110.json | Clinical_Trial_Data_Analysis_Expert |  | ❌ | 5 |  | ❌ | 0 |
| 5 | gemini_findings_114.json | DataVerification_Expert |  | ❌ | 5 |  | ❌ | 0 |
| 6 | gemini_findings_123.json | Research_Expert |  | ❌ | 2 |  | ❌ | 0 |
| 7 | gemini_findings_13.json | Marathon_Expert |  | ❌ | 4 |  | ❌ | 0 |
| 8 | gemini_findings_21.json | Statistics_Expert |  | ❌ | 1 |  | ❌ | 0 |
| 9 | gemini_findings_22.json | Verification_Expert |  | ❌ | 3 |  | ❌ | 0 |
| 10 | gemini_findings_25.json | Verification_Expert |  | ❌ | 2 |  | ❌ | 0 |
| 11 | gemini_findings_27.json | Lyrics_Expert |  | ❌ | 0 |  | ❌ | 0 |
| 12 | gemini_findings_28.json | Filmography_Expert |  | ❌ | 1 |  | ❌ | 0 |
| 13 | gemini_findings_29.json | Verification_Expert |  | ❌ | 1 |  | ❌ | 0 |
| 14 | gemini_findings_32.json | Verification_Expert | Geography_Expert, Grocery_Expert | ❌ | 1 | 6, 4, 5, 3, 8, 7 | ❌ | 0 |
| 15 | gemini_findings_35.json | MartialArts_Expert |  | ❌ | 5 |  | ❌ | 0 |
| 16 | gemini_findings_38.json | Verification_Expert | Local_Knowledge_Expert, Fitness_Expert | ❌ | 6 | 1, 2, 3 | ❌ | 0 |
| 17 | gemini_findings_43.json | Culinary_Expert |  | ❌ | 3 |  | ❌ | 0 |
| 18 | gemini_findings_45.json | Validation_Expert | DataAnalysis_Expert | ❌ | 8 | 3, 4, 6, 5 | ❌ | 0 |
| 19 | gemini_findings_46.json | BiblicalScholar_Expert |  | ❌ | 1 |  | ❌ | 0 |
| 20 | gemini_findings_51.json | Excel_Expert |  | ❌ | 0 |  | ❌ | 0 |

## Reproducibility

```json
{
  "experiment_name": "Who&When ALGO — verifier=soft",
  "model_name": "Gemini",
  "dataset_name": "Who&When / Algorithm-Generated",
  "run_label": "verifier ablation (LLM findings only, no validators)",
  "pred_glob": "C:\\Users\\barak\\MASeval\\examples\\who_and_when/who&when_algo_gemini_idx_msg_v3/*.json",
  "hf_annotations": "C:\\Users\\barak\\.cache\\huggingface\\hub\\datasets--Kevin355--Who_and_When\\snapshots\\59b9fcba1aaed7bbf206b5f4d3c68b8face2f49c\\Algorithm-Generated.parquet",
  "output_json_path": "C:\\Users\\barak\\MASeval\\examples\\who_and_when\\reports\\verifier_ablation_algo_soft.json",
  "output_md_path": "C:\\Users\\barak\\MASeval\\examples\\who_and_when\\reports\\verifier_ablation_algo_soft.md",
  "id_column": null,
  "agent_columns": null,
  "step_columns": null,
  "step_tolerance": 1,
  "build_missing_report": true,
  "verifier_mode": "soft",
  "top_error_examples": 20,
  "notes": "EvidenceVerifier ablation, mode='soft'. non_llm_validators are NOT counted. Reports rebuilt from raw LLM findings under this gating."
}
```
