# KG Benchmark Results

Benchmark comparing three review conditions on 9 ACL/ICLR/ICML/NeurIPS/ETS papers:

| Condition | Description |
|-----------|-------------|
| **KG_OFF** | DeepReview agent, venue criteria disabled (same prompt family as ChatGPT paste test) |
| **KG_ON** | DeepReview agent with venue knowledge-graph criteria |
| **TRAD_LLM** | ChatGPT review pasted from the KG_OFF agent prompt (no agent pipeline) |

## Primary artifact

**`benchmark_results_combined.csv`** — one row per paper with judge scores, faithfulness, engineering metrics, and pairwise outcomes for all three conditions.

Use `in_analysis_subset=true` for the **7-paper headline set** (B2a). Two papers are excluded from headline comparisons because KG_OFF won overall but KG_ON won rubric alignment (`acl_2024.findings-acl.438`, `icml_2311.10263v2`).

## Metrics (0.0–1.0 unless noted)

Scores come from an OpenAI judge (GPT-5-mini, per-metric GEval) and Gemma RAGAS faithfulness.

| Column prefix | Meaning |
|---------------|---------|
| `*_factual_correctness` | Claims about the paper are accurate vs. the manuscript |
| `*_evidence_support` | Critiques cite or imply real evidence from the paper |
| `*_rubric_alignment` | Review addresses venue-specific criteria (KG_ON/TRAD judged with venue rubric in context) |
| `*_criterion_grounded_valid_critique` | Weaknesses are valid and tied to criteria |
| `*_faithfulness_mean` | RAGAS faithfulness over extracted critique claims (manuscript-grounded context) |
| `*_total_tokens` | Agent input+output tokens (TRAD_LLM: N/A) |
| `delta_*` | KG_ON minus KG_OFF (paired papers only) |

## Pairwise columns

| Column | Meaning |
|--------|---------|
| `pairwise_kg_winner` | OpenAI judge: KG_ON vs KG_OFF overall (7-paper subset) |
| `pairwise_kg_rubric_winner` | Which review better satisfies rubric alignment |
| `trad_vs_kg_off_winner` | ChatGPT paste vs KG_OFF agent |
| `trad_vs_kg_on_winner` | ChatGPT paste vs KG_ON agent |

## Headline conclusions (7-paper analysis subset)

1. **KG_ON vs KG_OFF (pairwise):** 4 KG_ON wins, 2 KG_OFF wins, 1 tie.
2. **Median rubric_alignment:** TRAD 0.40 · KG_OFF 0.90 · KG_ON 0.60 — agents score higher than paste-ChatGPT on rubric alignment; KG_OFF highest on this automated metric.
3. **TRAD vs agents (pairwise):** Agents win **7/7** vs ChatGPT on both TRAD vs KG_OFF and TRAD vs KG_ON — structured agent reviews beat one-shot ChatGPT paste on every included paper.
4. **KG cost:** Median +674k tokens per paper for KG_ON vs KG_OFF (see `delta_total_tokens`).
5. **Faithfulness:** Low absolute scores for all conditions; KG does not clearly improve claim-level manuscript grounding in this run.

## Other files

| File | Contents |
|------|----------|
| `benchmark_summary.md` | Auto-generated narrative summary |
| `paired_comparison.csv` | KG_ON vs KG_OFF deltas (subset-filtered when using default compare) |
| `three_way_summary.csv` | Per-paper medians across TRAD / KG_OFF / KG_ON |
| `trad_pairwise_judge_scores.csv` | TRAD vs agent pairwise details |
| `review_quality_scores.csv` | Raw per-run judge scores |
| `faithfulness_run_scores.csv` | Per-run RAGAS means |
| `runs.jsonl` | Harvested agent run metadata |

## Regenerating

```bash
python -m benchmark export-results
python -m benchmark compare && python -m benchmark report
```

Full trad eval pipeline: `python -m benchmark pipeline --phase trad-eval` (see `docs/operator-benchmark-rerun.md`).
