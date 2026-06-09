# Traditional LLM Eval + Analysis Subset (Approach 3)

**Date:** 2026-06-09  
**Status:** Approved  
**Related:** `docs/superpowers/specs/2026-06-08-benchmark-eval-fix-design.md`, `trad_LLM/`, `benchmark/exports/kg_off_agent_prompts/`

## 1. Problem statement

Two deliverables:

1. **Analysis subset (B2a):** Current KG_ON vs KG_OFF pairwise headline is 4 / 4 / 1 on 9 papers. Operator wants **4 / 2 / 1 on 7 papers** by excluding papers where KG_OFF won overall but KG_ON won rubric_alignment.
2. **Traditional LLM baseline:** Nine ChatGPT reviews (pasted from KG_OFF agent prompts) live in `trad_LLM/` as PDFs. Need the same eval v2 stack (judge + faithfulness + pairwise) and a **three-way** comparison (TRAD vs KG_OFF vs KG_ON) on the same 7-paper analysis set.

**Operator decisions (locked):**

| Decision | Choice |
|----------|--------|
| Exclusion rule | **B2a** — drop `acl_2024.findings-acl.438` and `icml_2311.10263v2` |
| Three-way scope | **7 papers** — same subset as reshaped KG pairwise headline |
| Architecture | **Approach 3** — adapter + shared eval; no synthetic `data/jobs/` entries |
| Agent re-eval | **No** — reuse existing KG_ON/KG_OFF scores for all 9 papers |

## 2. Goals and non-goals

### Goals

1. Filter compare/report to 7-paper analysis subset by default; headline KG pairwise becomes 4 / 2 / 1
2. Ingest `trad_LLM/*.pdf` → markdown reviews with explicit `paper_id` mapping
3. Score TRAD with same v2 judge metrics and RAGAS faithfulness as agent runs
4. Pairwise: TRAD vs KG_ON and TRAD vs KG_OFF per paper (7-paper subset)
5. Report three-way per-metric medians and pairwise win rates
6. Preserve full 9-paper raw data; excluded papers labeled in appendix

### Non-goals

- Re-running OpenAI review jobs (KG_ON / KG_OFF)
- Re-scoring agent arms for excluded papers
- Synthetic `data/jobs/` dirs or `checks.py` validation for TRAD
- Fourth condition beyond TRAD_LLM / KG_ON / KG_OFF

## 3. Architecture

```
trad_LLM/*.pdf
  trad-ingest [pymupdf]
    → benchmark/trad_reviews/<paper_id>.md
    → benchmark/trad_registry.jsonl
    → benchmark/results/trad_runs.jsonl

Eval v2 (extended)
  judge --source trad     → append TRAD_LLM rows to review_quality_scores.csv
  faithfulness --source trad → append to faithfulness_run_scores.csv, claim_scores.jsonl
  trad-pairwise-judge     → trad_pairwise_judge_scores.csv
  compare (filtered)      → paired_comparison.csv (7 papers)
  three-way-compare       → three_way_summary.csv
  report                  → benchmark_summary.md (+ three-way section)

Analysis filter
  benchmark/analysis_subset.json → compare, report, trad-pairwise (default)
  --all-papers CLI flag → full 9-paper tables (appendix)
```

### New / changed modules

| Module | Change |
|--------|--------|
| `benchmark/analysis_subset.py` | **New** — load/filter `included_paper_ids` |
| `benchmark/analysis_subset.json` | **New** — B2a 7-paper list |
| `benchmark/trad_registry.jsonl` | **New** — PDF → paper_id, manuscript_job_id, criteria_job_id |
| `benchmark/trad_ingest.py` | **New** — PDF extraction, registry validation |
| `benchmark/review_sources.py` | **New** — `load_review_artifacts(job_id \| trad_row)` |
| `benchmark/judge.py` | Use `review_sources`; `judge_trad_all()` |
| `benchmark/faithfulness.py` | Use `review_sources`; `faithfulness_trad_all()` |
| `benchmark/trad_pairwise_judge.py` | **New** — TRAD vs KG_ON / TRAD vs KG_OFF |
| `benchmark/compare.py` | Analysis-subset filter; optional `paper_id` param |
| `benchmark/three_way_compare.py` | **New** — median scores across three conditions |
| `benchmark/report.py` | Subset headline + three-way section |
| `benchmark/paths.py` | Trad paths constants |
| `benchmark/cli.py` | `trad-ingest`, `judge --source trad`, `trad-pairwise-judge`, `three-way-compare` |

## 4. Analysis subset (B2a)

**File:** `benchmark/analysis_subset.json`

```json
{
  "version": "b2a-v1",
  "reason": "Exclude KG_OFF overall wins where rubric_alignment favored KG_ON (B2a)",
  "included_paper_ids": [
    "acl_2024.acl-short.8",
    "ets_liu-usingaibasedobject-2023",
    "iclr_1412.6980v9",
    "iclr_9447_tabular_insights_visual_i",
    "icml_2402.01869v2",
    "neurips_1706.03762v7",
    "neurips_2402.05602v2"
  ],
  "excluded_paper_ids": [
    "acl_2024.findings-acl.438",
    "icml_2311.10263v2"
  ]
}
```

**Behavior:**

- `compare`, `report`, `trad-pairwise-judge`, `three-way-compare` filter to `included_paper_ids` by default
- `--all-papers` includes all 9; report labels excluded rows
- Raw `runs.jsonl`, score CSVs, and full pairwise rows are never deleted
- Expected KG_ON vs KG_OFF headline on 7 papers: **4 wins / 2 losses / 1 tie**

## 5. TRAD ingestion

### Registry (`benchmark/trad_registry.jsonl`)

One JSON object per line:

| Field | Description |
|-------|-------------|
| `paper_id` | Manifest paper_id |
| `venue` | Venue string |
| `source_pdf` | Path relative to repo root under `trad_LLM/` |
| `manuscript_job_id` | KG_OFF job (same prompt input as ChatGPT paste) |
| `criteria_job_id` | KG_ON job (venue `review_criteria_bundle.json` for judge context) |

Synthetic `job_id` for TRAD rows: `trad:<paper_id>` (e.g. `trad:acl_2024.acl-short.8`).

### PDF → markdown

- Use `pymupdf` (already in `pyproject.toml`)
- Write `benchmark/trad_reviews/<paper_id>.md`
- Emit `benchmark/results/trad_runs.jsonl` row per registry entry:

```json
{
  "job_id": "trad:acl_2024.acl-short.8",
  "paper_id": "acl_2024.acl-short.8",
  "venue": "ACL",
  "condition": "TRAD_LLM",
  "status": "completed",
  "manuscript_job_id": "330c5928-a8d1-4ec1-8efe-ed2081001477",
  "criteria_job_id": "d9f100a7-874d-47b6-9c9b-b80d3f177af5",
  "review_path": "benchmark/trad_reviews/acl_2024.acl-short.8.md"
}
```

### Faithfulness

TRAD has no `annotations.json`. Claims extracted from markdown sections only; `evidence_resolve` v2 fallbacks (quote → manuscript prefix).

## 6. Eval + three-way compare

### Judge / faithfulness

- Reuse eval v2 GEval per-metric judge and Gemma RAGAS faithfulness
- `load_review_artifacts()` resolves:
  - `final_markdown` ← `trad_reviews/<paper_id>.md`
  - `manuscript_excerpt` ← `data/jobs/<manuscript_job_id>/mineru_full.md`
  - `criteria_json` ← `data/jobs/<criteria_job_id>/review_criteria_bundle.json`
- Skip `validate_run_completion()` for `TRAD_LLM`; require review file ≥ 2048 bytes
- Append TRAD rows to existing score CSVs (do not overwrite agent rows)

### Pairwise extensions

| `pair_type` | Comparison |
|-------------|------------|
| `TRAD_VS_KG_OFF` | ChatGPT paste vs agent (no KG) |
| `TRAD_VS_KG_ON` | ChatGPT paste vs KG-augmented agent |
| `KG_ON_VS_KG_OFF` | Existing (filtered to 7) |

Output: `benchmark/results/trad_pairwise_judge_scores.csv`

### Three-way summary

`three_way_compare` produces `benchmark/results/three_way_summary.csv`:

- Per `paper_id` (7 rows): median judge metrics for TRAD_LLM, KG_OFF, KG_ON
- Overall row: cross-paper medians per condition

### Report additions

- Analysis subset note (7 papers, 2 excluded with reason)
- KG_ON vs KG_OFF pairwise: 4 / 2 / 1
- Three-way per-metric median table
- TRAD pairwise win rates (vs KG_OFF, vs KG_ON)
- Appendix: full 9-paper scores when `--all-papers` used

## 7. CLI

```bash
python -m benchmark trad-ingest
python -m benchmark judge --source trad
python -m benchmark faithfulness --source trad
python -m benchmark trad-pairwise-judge
python -m benchmark compare                    # 7-paper default
python -m benchmark three-way-compare
python -m benchmark report
python -m benchmark compare --all-papers       # appendix tables
```

Eval phase extension (optional `pipeline --phase trad-eval`):

```
trad-ingest → judge --source trad → faithfulness --source trad → trad-pairwise-judge → compare → three-way-compare → report
```

## 8. Testing

| Test file | Coverage |
|-----------|----------|
| `tests/benchmark/test_analysis_subset.py` | Load/filter B2a list |
| `tests/benchmark/test_trad_ingest.py` | PDF extract fixture; registry validation |
| `tests/benchmark/test_review_sources.py` | Trad vs job artifact loading |
| `tests/benchmark/test_compare_subset.py` | 7-paper filter; 4/2/1 headline fixture |
| `tests/benchmark/test_three_way_compare.py` | Three-condition median table |
| `tests/benchmark/test_trad_pairwise_judge.py` | Prompt parsing (mock LLM) |

## 9. Operator runbook

1. Ensure v2 eval complete on agent runs (`benchmark/results/review_quality_scores.csv` exists)
2. Place PDFs in `trad_LLM/` (already done)
3. `python -m benchmark trad-ingest` — verify 9 `.md` files under `benchmark/trad_reviews/`
4. `python -m benchmark judge --source trad` (~7 × 4 GEval calls; uses OpenAI)
5. `python -m benchmark faithfulness --source trad` (~7 papers × claims; uses Gemma)
6. `python -m benchmark trad-pairwise-judge` (~14 pairwise calls on 7 papers)
7. `python -m benchmark compare && python -m benchmark three-way-compare && python -m benchmark report`
8. Confirm `benchmark_summary.md` shows 4 / 2 / 1 KG pairwise and three-way section

**Worktree:** Implement on `feat/benchmark-gemma-ragas-rerun` in `.worktrees/gemma-ragas-rerun` (v2 eval code base).
