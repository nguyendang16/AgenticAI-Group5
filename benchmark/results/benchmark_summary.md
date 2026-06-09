# KG Benchmark Summary

## Executive summary

- Valid paired papers: **7**
- Excluded / invalid pairs: **1**
- Pair completion rate: **70.0%**
- Median Δ rubric_alignment (KG_ON − KG_OFF): **-0.200**
- Median Δ factual_correctness: **-0.100**
- Median Δ faithfulness_mean: **-0.069**

## Analysis subset

- Included papers: **7**
- Subset rationale: Exclude KG_OFF overall wins where rubric_alignment favored KG_ON (B2a)

## Pairwise judge (KG_ON vs KG_OFF)

- KG_ON wins: **4** / 7 (57.1%)
- KG_OFF wins: **2** / 7 (28.6%)
- Ties: **1** / 7 (14.3%)

## Three-way medians

Median rubric_alignment by condition:
- TRAD_LLM: **0.400**
- KG_OFF: **0.900**
- KG_ON: **0.600**

Median faithfulness_mean by condition:
- TRAD_LLM: **0.062**
- KG_OFF: **0.075**
- KG_ON: **0.000**

## Trad pairwise win rates

- **TRAD_VS_KG_OFF**: KG_OFF: **7**
- **TRAD_VS_KG_ON**: KG_ON: **7**

## Per-venue rubric alignment

| Venue | n_pairs | median Δ rubric_alignment | 95% CI | Wilcoxon p |
| --- | ---: | ---: | --- | ---: |
| ACL | 1 | -0.200 | [-0.200, -0.200] | n/a |
| ETS | 1 | -0.300 | [-0.300, -0.300] | n/a |
| ICLR | 2 | -0.500 | [-0.800, -0.200] | n/a |
| ICML | 1 | -0.100 | [-0.100, -0.100] | n/a |
| NeurIPS | 2 | -0.200 | [-0.400, 0.000] | n/a |

## Decision metrics (sklearn, when labels exist)

No labeled decision metrics available.

## Top venues where KG helps / hurts (rubric_alignment)

**KG helps most:**
- ICML: median Δ = -0.100
- ACL: median Δ = -0.200
- NeurIPS: median Δ = -0.200

**KG hurts most:**
- ICLR: median Δ = -0.500
- ETS: median Δ = -0.300
- NeurIPS: median Δ = -0.200

## Statistical test

Wilcoxon signed-rank p-value on rubric_alignment deltas (n ≥ 8): **n/a**

## Data quality

- Invalid or incomplete pairs excluded from comparison: **1**

## Appendix: Engineering metrics

- Median Δ total_tokens: **674364.000**
- Median Δ runtime_seconds: **35.574**
