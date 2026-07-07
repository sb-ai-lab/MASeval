# Non-LLM Validator Weight Ablation

## Experiment setup

- Dataset split: `hand`
- Annotation source: `hf://datasets/Kevin355/Who_and_When/Hand-Crafted.parquet`
- Prediction files: `58`
- Step tolerance: `±1`
- Weights: `[0.0, 1.0]`

LLM judge findings use weight `1.0`. Deterministic validator findings use `λ`.
`λ=0.0` disables deterministic validators. `λ=1.0` makes them equal to LLM findings for ranking.

## Results

| non_llm_validator_weight | agent_top1_acc | agent_hit_acc | agent_exact_set_acc | step_top1_acc | step_hit_acc | step_hit_pm1_acc | llm_issues_total | non_llm_issues_total | invalid_findings_total |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0000 | 0.5690 | 0.8966 | 0.0690 | 0.0345 | 0.3621 | 0.6379 | 568 | 0 | 95 |
| 1.0000 | 0.5690 | 0.8966 | 0.0690 | 0.0172 | 0.3621 | 0.6379 | 568 | 19 | 95 |

## Metric notes

- **Agent Top-1 Acc**: primary predicted culprit agent matches gold.
- **Agent Hit Acc**: at least one predicted culprit agent matches gold.
- **Step Hit Acc**: at least one predicted problematic step exactly matches gold.
- **Step Hit@±1**: at least one predicted step is within ±1 of gold.
- **Non-LLM issues total**: deterministic validator findings used in reports for this λ.