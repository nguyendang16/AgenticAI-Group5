# Benchmark Eval Fix — OpenAI Judge + Gemma RAGAS (Approach 3)

**Date:** 2026-06-08  
**Status:** Draft — pending user review  
**Supersedes eval sections of:** `docs/superpowers/specs/2026-06-07-benchmark-gemma-ragas-rerun-design.md` (§6–8, §10 sanity gate)  
**Related:** `docs/superpowers/plans/2026-06-07-benchmark-gemma-ragas-rerun.md`, `docs/operator-benchmark-rerun.md`

## 1. Problem statement

Pilot eval on 18 completed runs exposed two failures:

| Component | Symptom | Root cause |
|-----------|---------|------------|
| **Gemma composite judge** | Scores cluster at 4–5; no discriminative KG deltas | Single lenient pass; aggressive truncation (8k/4k vs spec 25k/8k); no calibration anchors |
| **RAGAS faithfulness** | Means ≈ 0.0; KG_ON worse than KG_OFF | `(evidence: …)` resolves to citation **labels**, not manuscript text; RAGAS checks claim against wrong context |

**Operator decisions (locked):**

- **Provider split (C):** OpenAI `gpt-5-mini` for judge; Gemma `gemma-4-31b-it` for RAGAS only
- **Re-run scope (A):** Full re-score of all 18 completed runs; archive pre-fix CSVs
- **Approach:** **#3 Maximum validity** — 4-metric judge + anchored rubric + pairwise comparison + annotation-aware faithfulness

## 2. Goals and non-goals

### Goals

1. Faithfulness scores reflect whether critique bullets are supported by **actual manuscript spans**
2. Judge scores spread across 1–5 with metric-specific rubrics and explicit anchors
3. Pairwise judge directly answers “KG_ON or KG_OFF — which review is better?” per paper
4. Split provider config so judge and faithfulness use different APIs cleanly
5. Full v2 re-run on 18 runs; compare/report consume v2 outputs

### Non-goals

- Re-running review jobs (OpenAI agent phase unchanged)
- Replacing RAGAS with a custom metric (keep RAGAS faithfulness)
- Paid Google tier (Gemma stays on AI Studio free tier with throttling)
- 7-metric judge fallback in production

## 3. Architecture

```
Reviews (unchanged)
  OpenAI gpt-5-mini → data/jobs/<id>/
    final_report.md, mineru_full.md, annotations.json, review_criteria_bundle.json

Eval v2 (split providers)
  judge [OpenAI]
    4 × GEval per run (core metrics)
    → review_quality_scores.csv
  pairwise_judge [OpenAI]
    1 × comparison per valid (paper_id) pair
    → pairwise_judge_scores.csv
  faithfulness [Gemma]
    RAGAS per claim, resolved manuscript context
    → claim_scores.jsonl, faithfulness_run_scores.csv

Stats (extended)
  compare → paired_comparison.csv (+ pairwise columns)
  report  → benchmark_summary.md (+ pairwise headline)
```

### New / changed modules

| Module | Change |
|--------|--------|
| `benchmark/eval_llm.py` | `judge_provider()`, `faithfulness_provider()`, `build_judge_model()`, `build_faithfulness_llm()` |
| `benchmark/evidence_resolve.py` | **New** — manuscript span resolution |
| `benchmark/claims.py` | Call `resolve_evidence_span()`; attach `context_source` |
| `benchmark/faithfulness.py` | Use faithfulness provider only; emit `resolved_context_preview` |
| `benchmark/judge.py` | Default mode `per_metric`; anchored criteria; OpenAI via judge provider |
| `benchmark/pairwise_judge.py` | **New** — KG_ON vs KG_OFF comparison per paper |
| `benchmark/compare.py` | Merge pairwise judge into paired rows |
| `benchmark/report.py` | Headline: pairwise win rate + per-metric medians |
| `benchmark/cli.py` | `pairwise-judge` subcommand; eval phase includes it |
| `.env.example` | Split provider vars, anchor toggle, archive note |

## 4. Provider configuration

| Variable | Default | Consumer |
|----------|---------|----------|
| `BENCHMARK_JUDGE_PROVIDER` | `openai` | `judge.py`, `pairwise_judge.py` |
| `BENCHMARK_FAITHFULNESS_PROVIDER` | `google` | `faithfulness.py` |
| `BENCHMARK_JUDGE_MODEL` | `gpt-5-mini` | judge + pairwise |
| `BENCHMARK_FAITHFULNESS_MODEL` | `gemma-4-31b-it` | RAGAS |
| `OPENAI_API_KEY` | required | judge |
| `GOOGLE_API_KEY` | required | faithfulness |
| `BENCHMARK_EVAL_PAUSE_SECONDS` | `5` | between all eval calls |
| `BENCHMARK_JUDGE_MODE` | `per_metric` | `per_metric` \| `composite` (legacy) |

**Backward compatibility:** If only `BENCHMARK_EVAL_PROVIDER` is set, both judge and faithfulness use it (current behavior).

**Production `.env` for v2:**

```env
BENCHMARK_JUDGE_PROVIDER=openai
BENCHMARK_FAITHFULNESS_PROVIDER=google
BENCHMARK_JUDGE_MODEL=gpt-5-mini
BENCHMARK_FAITHFULNESS_MODEL=gemma-4-31b-it
BENCHMARK_JUDGE_MODE=per_metric
```

## 5. Judge fix (4 metrics + anchors)

### Mode: `per_metric` (new default for v2)

Use existing `build_factual_correctness_metric()` … `build_criterion_grounded_valid_critique_metric()` — **one GEval call per core metric** (4 calls/run), via `build_judge_model()` (OpenAI).

### Anchored rubric

Append to each metric's `criteria` string (constant `JUDGE_SCORE_ANCHORS` in `judge.py`):

| Score | Anchor |
|-------|--------|
| 1 | Major failures: factual errors, rubric ignored, or critiques unsupported by manuscript |
| 3 | Adequate but generic: some valid points, missing depth or weak evidence |
| 5 | Excellent: accurate, evidence-grounded, systematically addresses venue criteria |

Temperature 0. No chain-of-thought in output — score only.

### Truncation (restore spec defaults)

| Variable | Default |
|----------|---------|
| `BENCHMARK_JUDGE_MAX_MANUSCRIPT_CHARS` | `25000` |
| `BENCHMARK_JUDGE_MAX_REPORT_CHARS` | `8000` |
| `BENCHMARK_JUDGE_MAX_CRITERIA_CHARS` | `4000` |

Remove direct `google.genai` composite path from production default; keep `composite_judge_scores()` behind `BENCHMARK_JUDGE_MODE=composite` for smoke tests.

### Output schema (unchanged)

`review_quality_scores.csv`: `paper_id, job_id, venue, condition, factual_correctness, evidence_support, rubric_alignment, criterion_grounded_valid_critique`

Add optional column `eval_version=v2` in JSONL sidecar or filename stamp in archive only (CSV schema unchanged for compare compatibility).

## 6. Pairwise judge (new)

**Module:** `benchmark/pairwise_judge.py`

For each `paper_id` with both KG_OFF and KG_ON completed runs:

1. Load both `final_report.md`, shared `mineru_full.md` excerpt (25k cap), criteria bundle (KG_ON job preferred; fallback KG_OFF)
2. Single OpenAI structured JSON call (same `build_judge_model()` client as per-metric judge) with prompt:

> Compare Review A (KG_OFF) and Review B (KG_ON) for venue {venue}. Which better satisfies rubric_alignment, evidence_support, and actionable critique quality? Return JSON: `winner` (`KG_ON`|`KG_OFF`|`tie`), `rubric_alignment_winner`, `confidence` (`high`|`medium`|`low`), `one_line_reason`.

3. Append to `benchmark/results/pairwise_judge_scores.csv`

| Column | Description |
|--------|-------------|
| `paper_id` | Paper key |
| `venue` | Venue |
| `winner` | Overall preference |
| `rubric_alignment_winner` | Specific rubric dim |
| `confidence` | Judge confidence |
| `reason` | One-line justification |
| `job_id_kg_on` | KG_ON job |
| `job_id_kg_off` | KG_OFF job |

**Skip:** Pairs missing either condition; already-scored pairs if `job_id` tuple in CSV (resumable).

**CLI:** `python -m benchmark pairwise-judge`

## 7. Faithfulness fix (annotation-aware resolution)

### Resolution pipeline (`evidence_resolve.py`)

`resolve_evidence_span(bullet, *, manuscript, annotations, max_chars=2000) → ResolvedEvidence`

```python
@dataclass
class ResolvedEvidence:
    text: str
    source: str  # annotation_match | section_slice | quoted_in_claim | manuscript_prefix
    preview: str  # first 120 chars for claim_scores.jsonl
```

**Priority order:**

1. **Annotation match** — If `annotations.json` exists: token-overlap / keyword match between bullet and annotation `comment` or `summary`; use annotation `text` (manuscript span). Threshold: Jaccard ≥ 0.15 on content words or shared criterion tag (e.g. `C06`).
2. **Section slice** — Parse refs like `Abstract`, `Section 2.2`, `Appendix A`, `Figure 3`: locate heading in `mineru_full.md`, extract until next heading (cap `max_chars`).
3. **Quoted in claim** — If bullet contains quoted phrase ≥ 20 chars, fuzzy-find quote in manuscript.
4. **Manuscript prefix fallback** — `manuscript[:max_chars]` (last resort; tag `source=manuscript_prefix`).

**Never** pass raw `(evidence: …)` citation strings as RAGAS context.

### RAGAS dataset row (unchanged shape, fixed content)

```python
{
    'user_input': claim.text,
    'response': claim.text,
    'retrieved_contexts': [resolved.text],
}
```

### Extended `claim_scores.jsonl` fields

Add: `context_source`, `resolved_context_preview`, `eval_version: "v2"`

### Env

| Variable | Default |
|----------|---------|
| `BENCHMARK_RAGAS_MAX_CLAIMS` | `10` |
| `BENCHMARK_RAGAS_MAX_CONTEXT_CHARS` | `2000` |
| `BENCHMARK_RAGAS_ANNOTATION_MATCH_MIN_JACCARD` | `0.15` |

## 8. Re-run protocol (full v2)

### Pre-flight

1. Archive current results:

   ```
   benchmark/results/archive/pre-v2-2026-06-08/
     review_quality_scores.csv
     faithfulness_run_scores.csv
     claim_scores.jsonl
     paired_comparison.csv
     benchmark_summary.md
     pairwise_judge_scores.csv  (if exists)
   ```

2. Delete or truncate active CSVs/JSONL so skip-id logic does not block re-score
3. Set split-provider `.env` (§4)
4. Confirm 18 completed runs in `runs.jsonl`

### Sanity gate (mandatory before full corpus)

```bash
python -m benchmark judge --job-id d9f100a7-874d-47b6-9c9b-b80d3f177af5
python -m benchmark judge --job-id 330c5928-a8d1-4ec1-8efe-ed2081001477
python -m benchmark pairwise-judge --paper-id acl_2024.acl-short.8
python -m benchmark faithfulness --job-id d9f100a7-874d-47b6-9c9b-b80d3f177af5
```

**Pass criteria:**

- Judge: not all four metrics = 5.0 on both runs (spread or differential expected)
- Faithfulness: sanity job mean > 0.1 with ≥ 3 claims; `context_source` not `citation_label`
- Pairwise: returns `winner` + non-empty `reason`

### Full v2 eval

```bash
python -m benchmark judge
python -m benchmark pairwise-judge
python -m benchmark faithfulness
python -m benchmark compare
python -m benchmark report
```

Or: `python -m benchmark pipeline --phase eval` (updated step list).

## 9. Compare + report updates

### `paired_comparison.csv` — new columns

| Column | Source |
|--------|--------|
| `pairwise_winner` | `pairwise_judge_scores.csv` |
| `pairwise_rubric_winner` | same |
| `pairwise_confidence` | same |

### `benchmark_summary.md` — new headline block

- **Pairwise KG win rate:** count(`winner=KG_ON`) / n_pairs
- **Median Δ rubric_alignment** (per-metric judge, retained as secondary)
- **Median Δ faithfulness_mean** (v2, only pairs with both sides scored)
- **Appendix:** token deltas unchanged

**Primary research headline (v2):** Pairwise win rate + Wilcoxon on per-metric `delta_rubric_alignment` when n ≥ 8.

## 10. Error handling

| Failure | Behavior |
|---------|----------|
| OpenAI judge 429 | Retry 3× exponential backoff; fail run row, continue corpus |
| Gemma faithfulness 429/500 | Retry per claim batch; `nan` for failed run row; log to stderr |
| No annotation match | Fall through section slice → prefix; never empty context |
| Pairwise missing half-pair | Skip paper, log warning |
| Invalid GEval score (non 1–5) | Retry once; else `nan` for that metric |

## 11. Testing (TDD)

| Test file | Coverage |
|-----------|----------|
| `tests/benchmark/test_evidence_resolve.py` | Section slice, annotation match, quote find, no citation labels |
| `tests/benchmark/test_claims.py` | Updated — resolved spans from manuscript text |
| `tests/benchmark/test_eval_llm.py` | Split providers |
| `tests/benchmark/test_judge.py` | Per-metric mode, anchors in criteria, truncation defaults |
| `tests/benchmark/test_pairwise_judge.py` | JSON parse, skip incomplete pairs |
| `tests/benchmark/test_faithfulness.py` | `context_source` in output rows |
| `tests/benchmark/test_compare.py` | Pairwise columns merged |

Fixtures: ACL short `d9f100a7` bullet + `annotations.json` + `mineru_full.md` snippets (checked in under `tests/fixtures/benchmark/`).

## 12. Cost and runtime estimates (18 runs)

| Step | API | Est. tokens | Est. cost | Est. time |
|------|-----|-------------|-----------|-----------|
| Judge (4×/run) | OpenAI | ~0.65M in | ~$0.18 | ~10 min |
| Pairwise (9 pairs) | OpenAI | ~0.15M in | ~$0.04 | ~3 min |
| Faithfulness (~120 claims) | Gemma | ~1.0M | $0 free tier | ~35 min |
| **Total v2 eval** | | | **~$0.22** | **~50 min** |

Reviews (sunk cost): ~$6.50 on 18 runs.

## 13. Success criteria

1. All 18 completed runs have v2 judge scores (no skip from old CSV)
2. ≥ 9 pairwise rows (one per valid pair)
3. Faithfulness: corpus median `faithfulness_mean` > 0.15 (sanity check — not all zeros)
4. ≥ 30% of claims use `context_source` in (`annotation_match`, `section_slice`, `quoted_in_claim`)
5. Judge: fewer than 50% of runs have all four metrics ≥ 4.5 (ceiling check)
6. `benchmark_summary.md` reports pairwise KG win rate

## 14. Dependencies

No new packages. Existing: `deepeval`, `ragas`, `langchain-google-genai`, `openai` (via deepeval GPTModel).

Ensure `google-genai` documented in `pyproject.toml` if faithfulness uses langchain-google-genai only (no direct genai in judge path for v2).

## 15. Implementation order (for planning)

1. Split providers in `eval_llm.py` + tests
2. `evidence_resolve.py` + `claims.py` integration + tests
3. Judge per-metric default + anchors + tests
4. `pairwise_judge.py` + CLI + tests
5. Faithfulness v2 fields + tests
6. Compare/report extensions + tests
7. Archive script or operator doc update
8. Operator v2 sanity gate + full re-run (Task 14b)

---

**Next step after approval:** Invoke `writing-plans` skill → `docs/superpowers/plans/2026-06-08-benchmark-eval-fix.md`
