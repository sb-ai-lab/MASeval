# Who&When — none + non_llm_validators (LLM-confirm) fold-in

- Split / annotations: `Algorithm-Generated.parquet`
- Predictions: `who&when_algo_gemini_idx_msg_v2_fixed`
- Matched traces: 126
- Baseline: verifier=`none` over the 11 LLM evaluators.
- Fold-in locus: appointed causal turn (`corrected_idx`), fallback surface idx; agent = agent at that idx (via `who_and_when_to_spans`).
- Filters: `all` = every regex finding; `conf+unc` = confirmed|uncertain; `confirmed` = confirmed only.

| Metric | none (LLM only) | none + val (all) | none + val (conf+unc) | none + val (confirmed) |
|---|---:|---:|---:|---:|
| Agent Top-1 | 46.0% | 49.2% | 49.2% | 49.2% |
| Agent Hit | 60.3% | 63.5% | 63.5% | 63.5% |
| Step Top-1 | 13.5% | 15.9% | 15.9% | 15.9% |
| Step Hit | 34.9% | 47.6% | 46.8% | 46.8% |
| Step Hit ±1 | 61.9% | 65.9% | 65.9% | 65.9% |
