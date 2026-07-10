# AEGIS — none + non_llm_validators (LLM-confirm) fold-in

- Predictions: `aegis_findings_confirm`
- Matched (has gold): 600; finding-bearing: 49
- Confirmer: confirmed=85, uncertain=0, benign=24
- Baseline: verifier=`none` over the 11 LLM evaluators.
- Fold-in agent = validator `culprit_agent` (+ per-idx evidence agents). AEGIS gold is agent-only, so no step metrics and no locus/appointing.
- Filters: `all` = every regex finding; `conf+unc` = confirmed|uncertain; `confirmed` = confirmed only. **F1 is the honest arbiter** (fold-in trades precision for recall; the confirmed filter protects precision).

## Overall (all traces with gold) (n=600)

| Metric | none (LLM only) | none + val (all) | none + val (conf+unc) | none + val (confirmed) |
|---|---:|---:|---:|---:|
| Agent Top-1 | 78.5% | 78.8% | 78.7% | 78.7% |
| Agent Hit | 86.7% | 88.5% | 88.0% | 88.0% |
| Agent Exact-Set | 14.3% | 13.5% | 13.7% | 13.7% |
| Precision (micro) | 61.1% | 60.9% | 61.0% | 61.0% |
| Recall (micro) | 75.7% | 76.6% | 76.4% | 76.4% |
| F1 (micro) | 67.6% | 67.9% | 67.8% | 67.8% |
| F1 (macro) | 30.0% | 30.4% | 30.3% | 30.3% |

## Bearing-only (traces with ≥1 validator finding) (n=49)

| Metric | none (LLM only) | none + val (all) | none + val (conf+unc) | none + val (confirmed) |
|---|---:|---:|---:|---:|
| Agent Top-1 | 46.9% | 51.0% | 49.0% | 49.0% |
| Agent Hit | 63.3% | 85.7% | 79.6% | 79.6% |
| Agent Exact-Set | 12.2% | 2.0% | 4.1% | 4.1% |
| Precision (micro) | 38.8% | 40.6% | 40.2% | 40.2% |
| Recall (micro) | 63.5% | 82.5% | 77.8% | 77.8% |
| F1 (micro) | 48.2% | 54.5% | 53.0% | 53.0% |
| F1 (macro) | 36.9% | 40.9% | 38.4% | 38.4% |
