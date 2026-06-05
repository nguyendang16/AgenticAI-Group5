# KG vs No-KG Benchmark Design

**Date:** 2026-06-06  
**Status:** Approved  
**Related:** `docs/2026-06-01-tiered-system-evaluation-design.md` (general reliability; complementary)

## 1. Goal

Build an automated, paired benchmark that measures whether venue-specific knowledge graph (KG) criteria improve peer reviews versus a generic fast-mode baseline on the same papers.

Primary output: `paired_comparison.csv` with deterministic, DeepEval judge, and (where labels exist) sklearn decision-metric deltas per paper.

## 2. Non-Goals

- Human expert rating in the core loop
- Reimplementing evaluation frameworks (DeepEval, RAGAS, sklearn, scipy, statsmodels, pandas) — **import and wrap only**
- Rewriting the review pipeline (harness wraps existing `main.py` CLI)
- Full 100-run benchmark before smoke validation (40 runs first)
- RAGAS claim-level faithfulness in v1 (deferred to v2; requires claim extractor)

## 3. Frozen Benchmark Profile

Both KG_ON and KG_OFF use identical settings except the KG toggle:

| Setting | Value |
|---------|-------|
| `REVIEW_FAST_MODE` | `true` |
| `PAPER_SEARCH_ENABLED` | `false` |
| `REVIEW_INFER_VENUE_FROM_PAPER` | `false` |
| `REVIEW_VENUE` | from manifest per paper |
| `ENABLE_FINAL_GATES` | `false` |
| `AGENT_MODEL` / `AGENT_REASONING_EFFORT` | pinned; recorded per run |

| Condition | `REVIEW_CRITERIA_ENABLED` |
|-----------|---------------------------|
| KG_ON | `true` |
| KG_OFF | `false` |

## 4. System Fix — Generic Fast-Mode Baseline (KG_OFF)

When `criteria_bundle is None` in fast mode:

- Prompt omits criteria block, Claim-Level Audit instructions, and `criterion_id` on `pdf_annotate`
- Final report sections: Summary, Strengths, Weaknesses, Key Issues, Actionable Suggestions, Scores (6 sections)
- KG_ON unchanged: 7 sections including Claim-Level Audit + Criterion Legend prepended on save

**Files:** `deepreview/prompts/review_agent_prompt.py`, `deepreview/tools/review_tools.py`

## 5. Dataset — All Automation

### 5.1 Local seed: `KG_testPapers/` (11 PDFs)

| File | Manifest venue |
|------|----------------|
| `1412.6980v9.pdf` | ICLR |
| `9447_Tabular_Insights_Visual_I.pdf` | ICLR |
| `1706.03762v7.pdf` | NeurIPS |
| `2402.05602v2.pdf` | NeurIPS |
| `2311.10263v2.pdf` | ICML |
| `2402.01869v2.pdf` | ICML |
| `2404.01847v3.pdf` | AAAI |
| `2024.acl-short.8.pdf` | ACL |
| `2024.findings-acl.438.pdf` | ACL |
| `1-s2.0-S0360131524002380-main.pdf` | Computers And Education |
| `Liu-UsingAIBasedObject-2023.pdf` | ETS |

### 5.2 Gap fill via automation

| Venue | Local count | Gap |
|-------|-------------|-----|
| ACL | 2 | 0 |
| ICML | 2 | 0 |
| ICLR | 2 | 0 |
| NeurIPS | 2 | 0 |
| AAAI | 1 | +1 |
| Computers And Education | 1 | +1 |
| ETS | 1 | +1 |
| CHI | 0 | +2 |
| TWELF | 0 | +2 |
| ETRD | 0 | +2 |

**Total smoke corpus:** 20 papers → 40 paired runs.

- `ingest_local.py` — scan `KG_testPapers/`, detect venue, apply override table
- `fetch_openreview.py` — CHI, AAAI, optional decision labels
- `fetch_journals.py` — TWELF, ETRD, C&E #2, ETS #2 (DOI/curated; log `venue_gap` on failure)

Canonical PDF path: `benchmark/papers/<paper_id>.pdf` (copy or symlink from `KG_testPapers` on ingest).

## 6. Harness Architecture

```
benchmark/
  ingest_local.py      # KG_testPapers → manifest rows
  fetch_openreview.py  # OpenReview gap fill
  fetch_journals.py    # non-OpenReview gap fill
  build_manifest.py    # merge → manifest.jsonl
  registry.py          # SQLite run index
  env.py               # KG_ON / KG_OFF env dicts
  runner.py            # paired submit + poll
  collect.py           # harvest job artifacts
  checks.py            # deterministic validators
  judge.py             # DeepEval report-level metrics (thin wrapper)
  decision_metrics.py  # sklearn accept/reject metrics when labels exist
  compare.py           # paired deltas; scipy + statsmodels stats
  report.py            # benchmark_summary.md
  cli.py               # python -m benchmark
  # v2 only:
  claims.py            # atomic claim extraction from reports
  faithfulness.py      # RAGAS claim+evidence scoring (thin wrapper)
```

Wraps `python main.py submit` and `python main.py watch`; sets env vars before each worker spawn.

## 7. External Evaluation Libraries (Import, Don't Rebuild)

All scoring logic lives in maintained open-source packages. Our code provides prompts, I/O glue, and aggregation only.

| Library | GitHub | Role in benchmark | Phase |
|---------|--------|-------------------|-------|
| **DeepEval** | https://github.com/confident-ai/deepeval | Report-level LLM-as-judge: rubric alignment, specificity, actionability, factual correctness (`GEval`, `LLMTestCase`) | **v1** |
| **RAGAS** | https://github.com/explodinggradients/ragas | Claim-level faithfulness: `faithfulness`, `answer_relevancy` on (claim, evidence, manuscript) tuples | **v2** |
| **scikit-learn** | https://github.com/scikit-learn/scikit-learn | Decision metrics when `expected_decision` in manifest: `accuracy_score`, `balanced_accuracy_score`, `f1_score`, `roc_auc_score` | **v1** (conditional) |
| **SciPy** | https://github.com/scipy/scipy | Paired non-parametric tests: `scipy.stats.wilcoxon` on KG_ON − KG_OFF deltas | **v1** |
| **statsmodels** | https://github.com/statsmodels/statsmodels | Formal paired comparison CIs: bootstrap / paired t-test helpers | **v1** |
| **pandas** | https://github.com/pandas-dev/pandas | Join runs, aggregate venue/overall summaries, write CSV | **v1** |
| **openreview-py** | https://github.com/openreview/openreview-py | Paper acquisition + `expected_decision` labels (not scoring) | **v1** fetch only |

**Principle:** Never reimplement metric definitions that these libraries already provide. Custom code is limited to: parsing our job artifacts, building `LLMTestCase` / RAGAS dataset rows, and calling library APIs.

## 8. Deterministic Checks

### Per-run (both conditions)

- `status == completed`
- `final_report.md` exists, size > 2 KB
- `events.jsonl` exists; runtime and token usage recorded
- `paper_search` calls == 0

### KG_ON

- `review_criteria_resolved.criteria_count > 0`
- `review_criteria_bundle.json` exists
- `## Criterion Legend` present
- `## Claim-Level Audit` present
- Venue-prefixed criterion IDs present

### KG_OFF

- `criteria_count == 0`
- No Criterion Legend, no Claim-Level Audit, no venue-prefixed IDs
- No `criterion_id` in `annotations.json`

**Invalid KG_ON** (`enabled=true` but `no_criteria_found`): exclude pair from scored comparison.

## 9. Report-Level LLM Evaluation (v1 — DeepEval)

**Module:** `benchmark/judge.py`  
**Library:** `deepeval` — https://github.com/confident-ai/deepeval

Judge model: `BENCHMARK_JUDGE_MODEL` (default `gpt-5-mini`), temperature 0.

Use DeepEval primitives directly:

- `from deepeval.metrics import GEval` for custom rubric metrics
- `from deepeval.test_case import LLMTestCase` for structured inputs
- `metric.measure(test_case)` — do not write a parallel judge framework

Metrics (score 1–5 + binary flags where noted):

| Metric | Description |
|--------|-------------|
| `factual_correctness` | Review claims vs manuscript |
| `evidence_support` | Critiques backed by text |
| `rubric_alignment` | Venue criteria coverage (criteria passed for KG_ON) |
| `specificity` | Concrete methods/experiments/results |
| `actionability` | Actionable suggestions |
| `unsupported_critique_rate` | Negative claims without evidence (lower is better) |
| `criterion_grounded_valid_critique` | Composite headline for KG_ON |

Inputs: `mineru_full.md` (truncated), `final_report.md`, venue, optional `review_criteria_bundle.json`.

## 10. Decision Metrics (v1 — scikit-learn, conditional)

**Module:** `benchmark/decision_metrics.py`  
**Library:** `scikit-learn` — https://github.com/scikit-learn/scikit-learn

Runs only for papers where manifest has `expected_decision` (populated by `fetch_openreview.py` when available).

1. Parse predicted recommendation from `Scores` section of `final_report.md` (accept / reject / borderline heuristics).
2. Compare to `expected_decision` ground truth.
3. Call sklearn metrics — do not reimplement:

```python
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
```

Outputs per run: `predicted_decision`, `decision_correct`, plus corpus-level `balanced_accuracy`, `macro_f1` in `decision_metrics.csv`.

Paired comparison adds `delta_balanced_accuracy`, `delta_macro_f1` where n ≥ 5 labeled papers per venue.

## 11. Statistical Comparison (v1 — SciPy + statsmodels + pandas)

**Module:** `benchmark/compare.py`  
**Libraries:**
- pandas — https://github.com/pandas-dev/pandas
- SciPy — https://github.com/scipy/scipy
- statsmodels — https://github.com/statsmodels/statsmodels

| Analysis | Library | API |
|----------|---------|-----|
| Join KG_ON / KG_OFF rows | pandas | `merge`, `groupby` |
| Paired delta | pandas | `kg_on_col - kg_off_col` |
| Non-parametric paired test | SciPy | `scipy.stats.wilcoxon` |
| Bootstrap 95% CI on median delta | statsmodels / numpy | `statsmodels.stats.weightstats` or manual bootstrap |
| Venue/overall summaries | pandas | `groupby.agg` → CSV |

## 12. Claim-Level Faithfulness (v2 — RAGAS)

**Modules:** `benchmark/claims.py`, `benchmark/faithfulness.py`  
**Library:** `ragas` — https://github.com/explodinggradients/ragas

Deferred until v1 smoke passes. Unit of evaluation: **atomic review claim + cited manuscript evidence**, not whole report.

Use RAGAS evaluation API directly:

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
```

Do not reimplement faithfulness scoring logic. Our code only extracts claim–evidence pairs from `final_report.md` / `annotations.json` and builds the RAGAS dataset schema.

## 13. Outputs

```
benchmark/results/
  runs.jsonl
  deterministic_scores.csv
  review_quality_scores.csv      # DeepEval
  decision_metrics.csv           # sklearn (rows only where labels exist)
  paired_comparison.csv
  venue_summary.csv
  overall_summary.csv
  benchmark_summary.md
  # v2:
  claim_scores.jsonl             # RAGAS per-claim
```

## 14. Success Criteria (Smoke)

- ≥ 90% of pairs complete with valid condition checks
- Corpus median `delta_rubric_alignment > 0`
- End-to-end automation: ingest → run → judge → report without manual steps
- Where OpenReview labels exist: `decision_metrics.csv` produced via sklearn (no custom metric code)

## 15. Deferred (v2)

- RAGAS `faithfulness` / `answer_relevancy` on atomic claim + evidence pairs (`claims.py` + `faithfulness.py`)
- Full 100-run benchmark (10 venues × 5 papers)
- Reuse MinerU parse across paired conditions
