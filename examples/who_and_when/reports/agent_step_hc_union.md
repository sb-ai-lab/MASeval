# Who&When HC — union

Generated: `2026-07-06 23:43:48`

## Experiment setup

| Field | Value |
|---|---|
| Experiment | Who&When HC — union |
| Model / judge | Gemini |
| Dataset | Who&When / Hand-Crafted |
| Run label | v9 report (v2 run) |
| Prediction glob | /tmp/claude-1000/-mnt-c-Users-barak-research/54f50c8a-e3ce-4026-8be1-c0bcd0d69949/scratchpad/hc_v2_fresh_validators/*.json |
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
| EvidenceVerifier policy | verified/weak are counted; invalid findings are excluded and left for review |

## Main results

| Group | Metric | Value | Count | Meaning |
|---|---:|---:|---:|---|
| Agent | Top-1 Acc | 55.2% | 58 | Primary culprit agent matches the annotation. |
| Agent | Hit Acc | 89.7% | 58 | At least one predicted culprit agent matches the annotation. |
| Agent | Exact Set Acc | 13.8% | 58 | Predicted agent set exactly equals the gold agent set. |
| Step | Top-1 Acc | 0.0% | 58 | First predicted problematic span equals the gold span. |
| Step | Hit Acc | 43.1% | 58 | At least one predicted span exactly matches a gold span. |
| Step | Hit Acc ±1 | 62.1% | 58 | At least one predicted numeric span is within ±1 of a gold span. |

## Diagnostic quality summary

| Field | Value |
|---|---:|
| Invalid findings excluded from main predictions | 115 |
| Agent examples | 58 |
| Step examples | 58 |

## Interpretation

- Agent localization: Hit Acc is 89.7%, Top-1 Acc is 55.2% (gap 34.5%). A large gap means the correct agent is often present but not ranked first.
- Step localization: exact Hit Acc is 43.1%, relaxed Hit Acc ±1 is 62.1% (gap 19.0%). A large gap suggests off-by-one or indexing mismatch.
- First-span ranking: Step Top-1 Acc is 0.0%. If this is much lower than Step Hit Acc, use the full problematic_spans list rather than only first_problem_span.

## Example mismatches / review targets (top 20)

| # | File | Gold agents | Predicted agents | Agent hit | Gold spans | Predicted spans | Step hit ±1 | Invalid findings |
|---:|---|---|---|---:|---|---|---:|---:|
| 1 | gemini_findings_50.json | WebSurfer | Orchestrator | ❌ | 24 | 9, 10, 12, 11, 13, 15, … (+5) | ❌ | 3 |
| 2 | gemini_findings_16.json | Assistant | WebSurfer, Orchestrator, FileSurfer | ❌ | 82 | 52, 50, 48, 55, 54, 58, … (+46) | ❌ | 0 |
| 3 | gemini_findings_26.json | Assistant |  | ❌ | 16 |  | ❌ | 0 |
| 4 | gemini_findings_51.json | WebSurfer |  | ❌ | 12 |  | ❌ | 0 |
| 5 | gemini_findings_11.json | WebSurfer | WebSurfer, Orchestrator | ✅ | 4 | 11, 15, 13, 10, 16, 12, … (+3) | ❌ | 5 |
| 6 | gemini_findings_23.json | WebSurfer | WebSurfer, Orchestrator | ✅ | 4 | 38, 10, 14, 22, 0, 35, … (+49) | ❌ | 4 |
| 7 | gemini_findings_39.json | WebSurfer | WebSurfer | ✅ | 8 | 4 | ❌ | 4 |
| 8 | gemini_findings_0.json | WebSurfer | FileSurfer, WebSurfer, Orchestrator | ✅ | 3 | 15, 19, 37, 21, 35, 11, … (+38) | ❌ | 3 |
| 9 | gemini_findings_18.json | WebSurfer | FileSurfer, Orchestrator, WebSurfer | ✅ | 4 | 13, 19, 11, 17, 18, 21, … (+18) | ❌ | 3 |
| 10 | gemini_findings_24.json | WebSurfer | WebSurfer, Orchestrator | ✅ | 8 | 20, 19, 10, 14, 16, 17, … (+6) | ❌ | 3 |
| 11 | gemini_findings_36.json | WebSurfer | WebSurfer, Orchestrator | ✅ | 4 | 13, 15, 17, 0, 6, 8, … (+4) | ❌ | 3 |
| 12 | gemini_findings_53.json | Orchestrator | Orchestrator, WebSurfer | ✅ | 50 | 10, 20, 30, 22, 26, 11, … (+12) | ❌ | 3 |
| 13 | gemini_findings_41.json | FileSurfer | FileSurfer, Orchestrator, WebSurfer | ✅ | 4 | 12, 13, 14, 16, 15, 11, … (+3) | ❌ | 2 |
| 14 | gemini_findings_2.json | WebSurfer | WebSurfer, Orchestrator | ✅ | 8 | 30, 36, 31, 33, 35, 29, … (+13) | ❌ | 1 |
| 15 | gemini_findings_48.json | Orchestrator | Orchestrator, WebSurfer | ✅ | 29 | 22, 0, 20, 16, 19, 21, … (+1) | ❌ | 1 |
| 16 | gemini_findings_8.json | WebSurfer | WebSurfer, Orchestrator | ✅ | 8 | 19, 15, 24, 18, 13, 17, … (+2) | ❌ | 1 |
| 17 | gemini_findings_12.json | Orchestrator | FileSurfer, Orchestrator, Assistant, ComputerTerminal | ✅ | 51 | 15, 25, 11, 27, 17, 19, … (+25) | ❌ | 0 |
| 18 | gemini_findings_21.json | Websurfer | WebSurfer, Orchestrator | ✅ | 24 | 0, 11, 13, 49, 30, 10, … (+42) | ❌ | 0 |
| 19 | gemini_findings_25.json | Assistant | Orchestrator, WebSurfer, Assistant | ✅ | 51 | 0, 16, 13, 17 | ❌ | 0 |
| 20 | gemini_findings_30.json | WebSurfer | Assistant | ❌ | 12 | 10, 11 | ✅ | 0 |

## Reproducibility

```json
{
  "experiment_name": "Who&When HC — union",
  "model_name": "Gemini",
  "dataset_name": "Who&When / Hand-Crafted",
  "run_label": "v9 report (v2 run)",
  "pred_glob": "/tmp/claude-1000/-mnt-c-Users-barak-research/54f50c8a-e3ce-4026-8be1-c0bcd0d69949/scratchpad/hc_v2_fresh_validators/*.json",
  "hf_annotations": "/home/barak/.cache/huggingface/hub/datasets--Kevin355--Who_and_When/snapshots/59b9fcba1aaed7bbf206b5f4d3c68b8face2f49c/Hand-Crafted.parquet",
  "output_json_path": "reports/agent_step_hc_union.json",
  "output_md_path": "reports/agent_step_hc_union.md",
  "id_column": null,
  "agent_columns": null,
  "step_columns": null,
  "step_tolerance": 1,
  "build_missing_report": true,
  "include_validators": true,
  "top_error_examples": 20,
  "notes": null
}
```
