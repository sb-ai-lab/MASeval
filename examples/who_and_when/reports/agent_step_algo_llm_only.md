# Who&When ALGO — llm_only

Generated: `2026-07-06 23:43:48`

## Experiment setup

| Field | Value |
|---|---|
| Experiment | Who&When ALGO — llm_only |
| Model / judge | Gemini |
| Dataset | Who&When / Algorithm-Generated |
| Run label | v9 report (v2 run) |
| Prediction glob | /tmp/claude-1000/-mnt-c-Users-barak-research/54f50c8a-e3ce-4026-8be1-c0bcd0d69949/scratchpad/algo_v2_fresh_validators/*.json |
| Prediction files | 126 |
| HF annotations | /home/barak/.cache/huggingface/hub/datasets--Kevin355--Who_and_When/snapshots/59b9fcba1aaed7bbf206b5f4d3c68b8face2f49c/Algorithm-Generated.parquet |
| Annotation rows | 126 |
| Matched annotations | 126 |
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
| Agent | Top-1 Acc | 47.6% | 126 | Primary culprit agent matches the annotation. |
| Agent | Hit Acc | 56.3% | 126 | At least one predicted culprit agent matches the annotation. |
| Agent | Exact Set Acc | 21.4% | 126 | Predicted agent set exactly equals the gold agent set. |
| Step | Top-1 Acc | 16.7% | 126 | First predicted problematic span equals the gold span. |
| Step | Hit Acc | 29.4% | 126 | At least one predicted span exactly matches a gold span. |
| Step | Hit Acc ±1 | 55.6% | 126 | At least one predicted numeric span is within ±1 of a gold span. |

## Diagnostic quality summary

| Field | Value |
|---|---:|
| Invalid findings excluded from main predictions | 76 |
| Agent examples | 126 |
| Step examples | 126 |

## Interpretation

- Agent localization: Hit Acc is 56.3%, Top-1 Acc is 47.6% (gap 8.7%). A large gap means the correct agent is often present but not ranked first.
- Step localization: exact Hit Acc is 29.4%, relaxed Hit Acc ±1 is 55.6% (gap 26.2%). A large gap suggests off-by-one or indexing mismatch.
- First-span ranking: Step Top-1 Acc is 16.7%. If this is much lower than Step Hit Acc, use the full problematic_spans list rather than only first_problem_span.

## Example mismatches / review targets (top 20)

| # | File | Gold agents | Predicted agents | Agent hit | Gold spans | Predicted spans | Step hit ±1 | Invalid findings |
|---:|---|---|---|---:|---|---|---:|---:|
| 1 | gemini_findings_82.json | ArtHistory_Expert | Verification_Expert, Chinese_Zodiac_Expert | ❌ | 4 | 7, 8, 9, 0, 2 | ❌ | 4 |
| 2 | gemini_findings_4.json | YouTubeDownload_Expert | ResultVerification_Expert | ❌ | 0 | 6 | ❌ | 2 |
| 3 | gemini_findings_17.json | AcademicPublication_Expert | Verification_Expert | ❌ | 5 | 7, 8, 9 | ❌ | 1 |
| 4 | gemini_findings_32.json | Verification_Expert | Grocery_Expert, Geography_Expert | ❌ | 1 | 9, 3, 5 | ❌ | 1 |
| 5 | gemini_findings_45.json | Validation_Expert | DataAnalysis_Expert | ❌ | 8 | 6, 3 | ❌ | 1 |
| 6 | gemini_findings_55.json | Corporate_Governance_Expert | DataVerification_Expert, Computer_terminal | ❌ | 8 | 5, 4, 6 | ❌ | 1 |
| 7 | gemini_findings_67.json | SpeciesSightingsData_Expert |  | ❌ | 1 |  | ❌ | 1 |
| 8 | gemini_findings_73.json | Verification_Expert | DataAnalysis_Expert | ❌ | 8 | 0 | ❌ | 1 |
| 9 | gemini_findings_95.json | DataExtraction_Expert | DataAnalysis_Expert, DataVerification_Expert | ❌ | 1 | 7, 6, 8 | ❌ | 1 |
| 10 | gemini_findings_0.json | Verification_Expert |  | ❌ | 1 |  | ❌ | 0 |
| 11 | gemini_findings_100.json | HawaiiRealEstate_Expert |  | ❌ | 2 |  | ❌ | 0 |
| 12 | gemini_findings_102.json | WaybackMachine_Expert |  | ❌ | 1 |  | ❌ | 0 |
| 13 | gemini_findings_107.json | DataAnalysis_Expert |  | ❌ | 0 |  | ❌ | 0 |
| 14 | gemini_findings_11.json | AnimalBehavior_Expert |  | ❌ | 8 |  | ❌ | 0 |
| 15 | gemini_findings_110.json | Clinical_Trial_Data_Analysis_Expert |  | ❌ | 5 |  | ❌ | 0 |
| 16 | gemini_findings_113.json | WikipediaHistory_Expert |  | ❌ | 2 |  | ❌ | 0 |
| 17 | gemini_findings_114.json | DataVerification_Expert |  | ❌ | 5 |  | ❌ | 0 |
| 18 | gemini_findings_115.json | DataVerification_Expert | Gaming_Awards_Expert, WebServing_Expert | ❌ | 3 | 0 | ❌ | 0 |
| 19 | gemini_findings_116.json | GIS_DataAnalysis_Expert |  | ❌ | 1 |  | ❌ | 0 |
| 20 | gemini_findings_12.json | Paintball_Expert |  | ❌ | 3 |  | ❌ | 0 |

## Reproducibility

```json
{
  "experiment_name": "Who&When ALGO — llm_only",
  "model_name": "Gemini",
  "dataset_name": "Who&When / Algorithm-Generated",
  "run_label": "v9 report (v2 run)",
  "pred_glob": "/tmp/claude-1000/-mnt-c-Users-barak-research/54f50c8a-e3ce-4026-8be1-c0bcd0d69949/scratchpad/algo_v2_fresh_validators/*.json",
  "hf_annotations": "/home/barak/.cache/huggingface/hub/datasets--Kevin355--Who_and_When/snapshots/59b9fcba1aaed7bbf206b5f4d3c68b8face2f49c/Algorithm-Generated.parquet",
  "output_json_path": "reports/agent_step_algo_llm_only.json",
  "output_md_path": "reports/agent_step_algo_llm_only.md",
  "id_column": null,
  "agent_columns": null,
  "step_columns": null,
  "step_tolerance": 1,
  "build_missing_report": true,
  "include_validators": false,
  "top_error_examples": 20,
  "notes": null
}
```
