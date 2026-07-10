# Who&When — none + non_llm_validators (LLM-confirm) fold-in

- Split / annotations: `Hand-Crafted.parquet`
- Predictions: `who&when_hand_gemini_idx_msg_v2_fixed`
- Matched traces: 58
- Baseline: verifier=`none` over the 11 LLM evaluators.
- Fold-in locus: appointed causal turn (`corrected_idx`), fallback surface idx; agent = agent at that idx (via `who_and_when_to_spans`).
- Filters: `all` = every regex finding; `conf+unc` = confirmed|uncertain; `confirmed` = confirmed only.

| Metric | none (LLM only) | none + val (all) | none + val (conf+unc) | none + val (confirmed) |
|---|---:|---:|---:|---:|
| Agent Top-1 | 60.3% | 60.3% | 60.3% | 60.3% |
| Agent Hit | 89.7% | 89.7% | 89.7% | 89.7% |
| Step Top-1 | 1.7% | 1.7% | 1.7% | 1.7% |
| Step Hit | 36.2% | 39.7% | 39.7% | 39.7% |
| Step Hit ±1 | 63.8% | 69.0% | 69.0% | 69.0% |
