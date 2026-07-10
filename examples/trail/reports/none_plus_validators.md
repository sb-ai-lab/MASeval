# TRAIL — none + non_llm_validators (LLM-confirm) fold-in

- Predictions: `trail_gemini_findings_v1_confirm`
- Matched (gold + GAIA trace): 113; finding-bearing: 46
- Confirmer: confirmed=481, uncertain=0, benign=7, appointed=254
- Gold = span-hash `location`; no agent gold (location-only, mirror of AEGIS).
- Judge idxs are numeric span positions, mapped to span-hashes via `trail_to_spans`; validator/gold idxs are span-hashes directly.
- Filters: `all` = every regex finding; `conf+unc` = confirmed|uncertain; `confirmed` = confirmed only (**the honest column**).

## Locus = appointed

### Overall (traces with gold) (n=113)

| Metric | none (LLM only) | none + val (all) | none + val (conf+unc) | none + val (confirmed) |
|---|---:|---:|---:|---:|
| Step Top-1 | 0.0% | 0.0% | 0.0% | 0.0% |
| Step Hit | 16.8% | 30.1% | 30.1% | 30.1% |

### Bearing-only (≥1 validator finding) (n=46)

| Metric | none (LLM only) | none + val (all) | none + val (conf+unc) | none + val (confirmed) |
|---|---:|---:|---:|---:|
| Step Top-1 | 0.0% | 0.0% | 0.0% | 0.0% |
| Step Hit | 34.8% | 67.4% | 67.4% | 67.4% |

## Locus = surface

### Overall (traces with gold) (n=113)

| Metric | none (LLM only) | none + val (all) | none + val (conf+unc) | none + val (confirmed) |
|---|---:|---:|---:|---:|
| Step Top-1 | 0.0% | 0.0% | 0.0% | 0.0% |
| Step Hit | 16.8% | 16.8% | 16.8% | 16.8% |

### Bearing-only (≥1 validator finding) (n=46)

| Metric | none (LLM only) | none + val (all) | none + val (conf+unc) | none + val (confirmed) |
|---|---:|---:|---:|---:|
| Step Top-1 | 0.0% | 0.0% | 0.0% | 0.0% |
| Step Hit | 34.8% | 34.8% | 34.8% | 34.8% |
