# KG Benchmark Results

Benchmark comparing three review conditions on 9 ACL/ICLR/ICML/NeurIPS/ETS papers:

| Condition | Description |
|-----------|-------------|
| **KG_OFF** | DeepReview agent, venue criteria disabled (same prompt family as ChatGPT paste test) |
| **KG_ON** | DeepReview agent with venue knowledge-graph criteria |
| **TRAD_LLM** | ChatGPT review pasted from the KG_OFF agent prompt (no agent pipeline) |

## Primary artifacts

| File | Use |
|------|-----|
| **`benchmark_results.xlsx`** | Multi-sheet workbook for side-by-side comparison (start here) |
| **`benchmark_results_combined.csv`** | One row per paper — all metrics and pairwise outcomes |

### Excel workbook sheets

| Sheet | Contents |
|-------|----------|
| **Overview** | Headline counts and medians (7-paper subset) |
| **Metric Matrix** | Judge + faithfulness medians as TRAD / KG_OFF / KG_ON columns |
| **By Paper (subset)** | 7 analysis papers with grouped metric columns |
| **By Paper (all)** | All 9 papers |
| **Judge (long)** | Long format: paper × condition × metric — easy to pivot or chart |
| **Faithfulness** | Per-paper RAGAS means and claim counts |
| **KG Pairwise** | KG_ON vs KG_OFF paired deltas |
| **Trad Pairwise** | ChatGPT vs agent pairwise judge results |
| **Three-Way Detail** | Per-paper medians across all three conditions |

## Metrics (0.0–1.0 unless noted)

Scores come from an OpenAI judge (GPT-5-mini, per-metric GEval) and Gemma RAGAS faithfulness.

| Column prefix | Meaning |
|---------------|---------|
| `*_factual_correctness` | Claims about the paper are accurate vs. the manuscript |
| `*_evidence_support` | Critiques cite or imply real evidence from the paper |
| `*_rubric_alignment` | Review addresses venue-specific criteria (KG_ON/TRAD judged with venue rubric in context) |
| `*_criterion_grounded_valid_critique` | Weaknesses are valid and tied to criteria |
| `*_faithfulness_mean` | RAGAS faithfulness over extracted critique claims (Weaknesses + Key Issues bullets) |
| `*_faithfulness_n` | Number of claims scored (max 10 per review) |
| `*_total_tokens` | Agent input+output tokens (TRAD_LLM: N/A) |
| `delta_*` | KG_ON minus KG_OFF (paired papers only) |

**TRAD faithfulness note:** TRAD reviews use numbered section headers (`3. Weaknesses`) and `•` bullets rather than agent markdown (`## Weaknesses`, `- `). Claim extraction supports both formats; earlier runs with `faithfulness_n=0` were a parsing gap, not missing RAGAS scores.

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
4. **Median faithfulness_mean:** TRAD 0.06 · KG_OFF 0.08 · KG_ON 0.00 — all conditions score low; differences are small in this run.
5. **KG cost:** Median +674k tokens per paper for KG_ON vs KG_OFF (see `delta_total_tokens`).

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
# After fixing claim extraction or re-scoring TRAD faithfulness:
python -m benchmark faithfulness --source trad

python -m benchmark three-way-compare
python -m benchmark compare && python -m benchmark report
python -m benchmark export-results
```

Full trad eval pipeline: `python -m benchmark pipeline --phase trad-eval` (see `docs/operator-benchmark-rerun.md`).
