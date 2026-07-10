# TraceElephant — none + non_llm_validators (LLM-confirm) fold-in

- Baseline: verifier=`none` over the 11 LLM evaluators.
- Fold-in idx shifted **+1** (validators 0-based → judge/gold 1-based space).
- Fold-in agent = step label at the locus (sub-agent for captain/magentic, tool for swe); swe baseline agent = tool at the judge locus.
- Filters: `all` = every regex finding (captain fires on 100% of traces → recall ceiling, not a result); `conf+unc` = confirmed|uncertain; `confirmed` = confirmed only (**the honest column**).

## Locus = appointed

### captain (matched 85)

_bearing=85, confirmed=204, uncertain=0, benign=33, appointed=99_

| Metric | none (LLM only) | none + val (all) | none + val (conf+unc) | none + val (confirmed) |
|---|---:|---:|---:|---:|
| Agent Top-1 | 63.5% | 70.6% | 68.2% | 68.2% |
| Agent Hit | 74.1% | 90.6% | 88.2% | 88.2% |
| Step Top-1 | 2.4% | 4.7% | 3.5% | 3.5% |
| Step Hit | 30.6% | 43.5% | 40.0% | 40.0% |
| Step Hit ±1 | 78.8% | 85.9% | 84.7% | 84.7% |

### magentic (matched 91)

_bearing=14, confirmed=30, uncertain=0, benign=0, appointed=13_

| Metric | none (LLM only) | none + val (all) | none + val (conf+unc) | none + val (confirmed) |
|---|---:|---:|---:|---:|
| Agent Top-1 | 45.1% | 46.2% | 46.2% | 46.2% |
| Agent Hit | 75.8% | 78.0% | 78.0% | 78.0% |
| Step Top-1 | 5.5% | 5.5% | 5.5% | 5.5% |
| Step Hit | 58.2% | 58.2% | 58.2% | 58.2% |
| Step Hit ±1 | 76.9% | 76.9% | 76.9% | 76.9% |

### swe (matched 44)

_bearing=18, confirmed=60, uncertain=0, benign=10, appointed=32_

| Metric | none (LLM only) | none + val (all) | none + val (conf+unc) | none + val (confirmed) |
|---|---:|---:|---:|---:|
| Agent Top-1 | 61.4% | 68.2% | 68.2% | 68.2% |
| Agent Hit | 75.0% | 88.6% | 88.6% | 88.6% |
| Step Top-1 | 2.3% | 2.3% | 2.3% | 2.3% |
| Step Hit | 43.2% | 45.5% | 45.5% | 45.5% |
| Step Hit ±1 | 59.1% | 61.4% | 61.4% | 61.4% |

### OVERALL (matched 220)

_bearing=117, confirmed=294, uncertain=0, benign=43, appointed=144_

| Metric | none (LLM only) | none + val (all) | none + val (conf+unc) | none + val (confirmed) |
|---|---:|---:|---:|---:|
| Agent Top-1 | 55.5% | 60.0% | 59.1% | 59.1% |
| Agent Hit | 75.0% | 85.0% | 84.1% | 84.1% |
| Step Top-1 | 3.6% | 4.5% | 4.1% | 4.1% |
| Step Hit | 44.5% | 50.0% | 48.6% | 48.6% |
| Step Hit ±1 | 74.1% | 77.3% | 76.8% | 76.8% |

## Locus = surface

### captain (matched 85)

_bearing=85, confirmed=204, uncertain=0, benign=33, appointed=99_

| Metric | none (LLM only) | none + val (all) | none + val (conf+unc) | none + val (confirmed) |
|---|---:|---:|---:|---:|
| Agent Top-1 | 63.5% | 69.4% | 67.1% | 67.1% |
| Agent Hit | 74.1% | 89.4% | 87.1% | 87.1% |
| Step Top-1 | 2.4% | 4.7% | 3.5% | 3.5% |
| Step Hit | 30.6% | 48.2% | 45.9% | 45.9% |
| Step Hit ±1 | 78.8% | 87.1% | 85.9% | 85.9% |

### magentic (matched 91)

_bearing=14, confirmed=30, uncertain=0, benign=0, appointed=13_

| Metric | none (LLM only) | none + val (all) | none + val (conf+unc) | none + val (confirmed) |
|---|---:|---:|---:|---:|
| Agent Top-1 | 45.1% | 46.2% | 46.2% | 46.2% |
| Agent Hit | 75.8% | 78.0% | 78.0% | 78.0% |
| Step Top-1 | 5.5% | 5.5% | 5.5% | 5.5% |
| Step Hit | 58.2% | 58.2% | 58.2% | 58.2% |
| Step Hit ±1 | 76.9% | 76.9% | 76.9% | 76.9% |

### swe (matched 44)

_bearing=18, confirmed=60, uncertain=0, benign=10, appointed=32_

| Metric | none (LLM only) | none + val (all) | none + val (conf+unc) | none + val (confirmed) |
|---|---:|---:|---:|---:|
| Agent Top-1 | 61.4% | 70.5% | 70.5% | 70.5% |
| Agent Hit | 75.0% | 90.9% | 90.9% | 90.9% |
| Step Top-1 | 2.3% | 2.3% | 2.3% | 2.3% |
| Step Hit | 43.2% | 45.5% | 45.5% | 45.5% |
| Step Hit ±1 | 59.1% | 63.6% | 63.6% | 63.6% |

### OVERALL (matched 220)

_bearing=117, confirmed=294, uncertain=0, benign=43, appointed=144_

| Metric | none (LLM only) | none + val (all) | none + val (conf+unc) | none + val (confirmed) |
|---|---:|---:|---:|---:|
| Agent Top-1 | 55.5% | 60.0% | 59.1% | 59.1% |
| Agent Hit | 75.0% | 85.0% | 84.1% | 84.1% |
| Step Top-1 | 3.6% | 4.5% | 4.1% | 4.1% |
| Step Hit | 44.5% | 51.8% | 50.9% | 50.9% |
| Step Hit ±1 | 74.1% | 78.2% | 77.7% | 77.7% |
