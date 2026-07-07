# Non-LLM Validator Weight Ablation

## Experiment setup

- Dataset split: `algo`
- Annotation source: `hf://datasets/Kevin355/Who_and_When/Algorithm-Generated.parquet`
- Prediction files: `126`
- Step tolerance: `±1`
- Weights: `[0.0, 1.0]`

LLM judge findings use weight `1.0`. Deterministic validator findings use `λ`.
`λ=0.0` disables deterministic validators. `λ=1.0` makes them equal to LLM findings for ranking.

## Results

| non_llm_validator_weight | agent_top1_acc | agent_hit_acc | agent_exact_set_acc | step_top1_acc | step_hit_acc | step_hit_pm1_acc | llm_issues_total | non_llm_issues_total | invalid_findings_total |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0000 | 0.4444 | 0.5635 | 0.1746 | 0.1429 | 0.3492 | 0.5873 | 475 | 0 | 92 |
| 1.0000 | 0.4444 | 0.5635 | 0.1746 | 0.1508 | 0.3571 | 0.6270 | 475 | 38 | 92 |

## Metric notes

- **Agent Top-1 Acc**: primary predicted culprit agent matches gold.
- **Agent Hit Acc**: at least one predicted culprit agent matches gold.
- **Step Hit Acc**: at least one predicted problematic step exactly matches gold.
- **Step Hit@±1**: at least one predicted step is within ±1 of gold.
- **Non-LLM issues total**: deterministic validator findings used in reports for this λ.