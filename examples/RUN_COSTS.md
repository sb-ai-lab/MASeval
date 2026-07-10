# Per-run API cost (gemini-2.5-flash actual + GPT-4o comparator)

**Model:** all runs used `google/gemini-2.5-flash` — **$0.30/M input, $2.50/M output**
(Google native = OpenRouter passthrough). GPT-4o comparator column uses **$2.50/M in,
$10.00/M out** on the *same token counts*.

**Method.** No token usage is stored in the finding files, so tokens are counted with
`tiktoken` (`o200k_base`). A judge run is **11 findings judges + 1 FinalAnswerVerifier
= 12 LLM calls/trace**, each re-sending the formatted trace (EvidenceVerifier +
non-LLM validators are deterministic → free). A confirmer run is **1 call/bearing
trace**. Input = `12·trace_tokens + (11 prompts=12,009 + FAV=1,287) + 12·schema(~300)`;
output = tiktoken of the stored model outputs.

**Validation (against the two runs with real `task_stats`):** tiktoken **output** vs
recorded — `who&when_algo_v9` 207,335 vs 205,712 (**1%**), `who&when_hc_v9` 502,593 vs
483,163 (**4%**). The input model reproduces their recorded per-trace tokens. So the
counts below are trustworthy to a few %.

## Measured runs (trace sources in hand)

| Run | traces | input tok | output tok | **gemini $** | gpt-4o $ |
|---|--:|--:|--:|--:|--:|
| TE captain judges | 85 | 63.2M | 586K | **20.42** | 163.82 |
| TE captain confirm | 85 | 5.3M | 18K | **1.63** | 13.42 |
| TE magentic judges | 91 | 101.0M | 796K | **32.30** | 260.53 |
| TE magentic confirm | 14 | 1.6M | 2K | **0.49** | 4.07 |
| TE swe judges | 44 | 68.0M | 221K | **20.95** | 172.16 |
| TE swe confirm | 18 | 2.3M | 5K | **0.72** | 5.90 |
| AEGIS judges | 600 | 58.1M | 5.05M | **30.05** | 195.70 |
| AEGIS confirm | 49 | 0.71M | 9K | **0.24** | 1.87 |
| TRAIL judges | 117 | 31.0M | 427K | **10.37** | 81.83 |
| TRAIL confirm | 47 | 2.1M | 41K | **0.73** | 5.66 |
| **Measured total** | | **~333M** | **~7.2M** | **≈ 117.9** | **≈ 905** |

Notes: TE magentic/swe traces are enormous (~90K trace-tokens/call) → they dominate.
AEGIS output is high (5M) because 600 traces each carry 11 verbose findings blocks.

## Who&When — actual recorded cost (real `task_stats`, 15 metrics)

| Run | traces | input tok | output tok | **gemini $** | gpt-4o $ |
|---|--:|--:|--:|--:|--:|
| WW algo v9_report_v2 | 126 | 10.40M | 206K | **3.63** | 26.06 |
| WW hc v9_report_v2 | 58 | 19.72M | 483K | **7.12** | 54.13 |

Ground truth from `task_stats` (algo 54.6 min, hc 74.2 min wall). **Hand-crafted
traces are ~4× larger than algorithm-generated** (340K vs 83K input tok/trace), so hc
costs more despite fewer traces. Our older WW `idx_msg` runs use 11 judges + FAV
(12 calls) instead of 15, so on the same traces they'd be ~12/15 of these:
algo ≈ $2.9, hand ≈ $5.7.

## Still unmeasured (no local trace source)

| Run | traces | note |
|---|--:|---|
| AgentRx magentic | 44 | traces ~300K chars (≈ TE-magentic rate → ~$16 gemini / ~$130 gpt-4o) |
| AgentRx tau | 29 | ~$2–4 gemini |

Point me at the AgentRx trace source and I'll run the same tiktoken pass for exact numbers.

## Bottom line
- **This session's fold-in work (all confirmers): ~$4.4 total** on gemini — cheap.
- **The judge runs are the cost:** TE ~$74, AEGIS ~$30, WW (algo+hc v9) ~$10.8, TRAIL ~$11.
- **All-in gemini spend across measured + recorded runs ≈ $129.**
- On GPT-4o the same token volume would be **~8×** more — driven by the 8×/4× higher
  input/output prices.
