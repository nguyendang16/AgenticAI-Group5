# Traditional LLM Eval + Analysis Subset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add B2a 7-paper analysis subset filtering (4/2/1 KG pairwise headline) and score ChatGPT baseline reviews from `trad_LLM/` with eval v2 judge, faithfulness, and three-way comparison vs KG_OFF and KG_ON.

**Architecture:** Approach 3 adapter — `trad_ingest` extracts PDFs to `benchmark/trad_reviews/`; `review_sources` generalizes artifact loading; TRAD rows live in `trad_runs.jsonl` with synthetic `job_id=trad:<paper_id>`; compare/report filter via `analysis_subset.json`; new `trad_pairwise_judge` and `three_way_compare` modules.

**Tech Stack:** Python 3.11, pymupdf, DeepEval GEval, RAGAS, pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-06-09-trad-llm-eval-design.md`

**Worktree:** Branch `feat/benchmark-gemma-ragas-rerun` in `.worktrees/gemma-ragas-rerun` (contains eval v2 code). Merge to main after Task 12 operator validation.

---

## File map

| File | Responsibility |
|------|----------------|
| `benchmark/analysis_subset.json` | B2a included/excluded paper lists |
| `benchmark/analysis_subset.py` | Load subset; `filter_paper_ids()` helper |
| `benchmark/trad_registry.jsonl` | PDF → paper_id + job id mapping (9 rows) |
| `benchmark/trad_ingest.py` | PDF→md extraction; emit `trad_runs.jsonl` |
| `benchmark/review_sources.py` | `load_review_artifacts()` for job or trad row |
| `benchmark/paths.py` | Trad path constants |
| `benchmark/judge.py` | Delegate to `review_sources`; add `judge_trad_all()` |
| `benchmark/faithfulness.py` | Delegate to `review_sources`; add `faithfulness_trad_all()` |
| `benchmark/trad_pairwise_judge.py` | TRAD vs KG_ON/OFF pairwise judge |
| `benchmark/three_way_compare.py` | Per-paper + overall three-condition medians |
| `benchmark/compare.py` | `paper_ids` filter param; subset default |
| `benchmark/report.py` | Subset headline + three-way section |
| `benchmark/cli.py` | New subcommands and `--source` / `--all-papers` flags |
| `tests/benchmark/test_*.py` | TDD per module |
| `tests/fixtures/benchmark/trad_sample.md` | Short trad review fixture |

---

### Task 1: Analysis subset config + loader

**Files:**
- Create: `benchmark/analysis_subset.json`
- Create: `benchmark/analysis_subset.py`
- Create: `tests/benchmark/test_analysis_subset.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/benchmark/test_analysis_subset.py
from pathlib import Path

import pytest

from benchmark.analysis_subset import (
    ANALYSIS_SUBSET_PATH,
    filter_paper_ids,
    load_analysis_subset,
)


def test_analysis_subset_path_exists():
    assert ANALYSIS_SUBSET_PATH.exists()


def test_load_analysis_subset_b2a():
    subset = load_analysis_subset()
    assert subset["version"] == "b2a-v1"
    assert len(subset["included_paper_ids"]) == 7
    assert "acl_2024.findings-acl.438" in subset["excluded_paper_ids"]
    assert "icml_2311.10263v2" in subset["excluded_paper_ids"]


def test_filter_paper_ids_default_included_only():
    all_ids = [
        "acl_2024.acl-short.8",
        "acl_2024.findings-acl.438",
        "icml_2311.10263v2",
    ]
    filtered = filter_paper_ids(all_ids, use_subset=True)
    assert filtered == ["acl_2024.acl-short.8"]


def test_filter_paper_ids_all_papers():
    all_ids = ["acl_2024.acl-short.8", "acl_2024.findings-acl.438"]
    assert filter_paper_ids(all_ids, use_subset=False) == all_ids
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `pytest tests/benchmark/test_analysis_subset.py -v`  
Expected: `ModuleNotFoundError: benchmark.analysis_subset`

- [ ] **Step 3: Create `benchmark/analysis_subset.json`**

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

- [ ] **Step 4: Implement `benchmark/analysis_subset.py`**

```python
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from benchmark.paths import BENCHMARK_DIR

ANALYSIS_SUBSET_PATH = BENCHMARK_DIR / 'analysis_subset.json'


@lru_cache(maxsize=1)
def load_analysis_subset() -> dict[str, Any]:
    payload = json.loads(ANALYSIS_SUBSET_PATH.read_text(encoding='utf-8'))
    included = list(payload.get('included_paper_ids') or [])
    excluded = list(payload.get('excluded_paper_ids') or [])
    return {
        'version': str(payload.get('version') or ''),
        'reason': str(payload.get('reason') or ''),
        'included_paper_ids': included,
        'excluded_paper_ids': excluded,
    }


def filter_paper_ids(paper_ids: list[str], *, use_subset: bool = True) -> list[str]:
    if not use_subset:
        return list(paper_ids)
    included = set(load_analysis_subset()['included_paper_ids'])
    return [pid for pid in paper_ids if pid in included]
```

- [ ] **Step 5: Run tests — expect PASS**

Run: `pytest tests/benchmark/test_analysis_subset.py -v`  
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add benchmark/analysis_subset.json benchmark/analysis_subset.py tests/benchmark/test_analysis_subset.py
git commit -m "feat(benchmark): add B2a analysis subset loader"
```

---

### Task 2: Path constants

**Files:**
- Modify: `benchmark/paths.py`
- Modify: `tests/benchmark/test_analysis_subset.py` (optional import check)

- [ ] **Step 1: Add trad paths to `benchmark/paths.py`**

Append after existing constants:

```python
TRAD_LLM_DIR = REPO_ROOT / 'trad_LLM'
TRAD_REVIEWS_DIR = BENCHMARK_DIR / 'trad_reviews'
TRAD_REGISTRY_PATH = BENCHMARK_DIR / 'trad_registry.jsonl'
TRAD_RUNS_JSONL_PATH = RESULTS_DIR / 'trad_runs.jsonl'
TRAD_PAIRWISE_JUDGE_SCORES_PATH = RESULTS_DIR / 'trad_pairwise_judge_scores.csv'
THREE_WAY_SUMMARY_PATH = RESULTS_DIR / 'three_way_summary.csv'
```

- [ ] **Step 2: Commit**

```bash
git add benchmark/paths.py
git commit -m "feat(benchmark): add trad LLM path constants"
```

---

### Task 3: Trad registry (9-paper mapping)

**Files:**
- Create: `benchmark/trad_registry.jsonl`
- Create: `benchmark/trad_registry.py`
- Create: `tests/benchmark/test_trad_registry.py`

- [ ] **Step 1: Write failing test**

```python
# tests/benchmark/test_trad_registry.py
from benchmark.trad_registry import load_trad_registry, trad_job_id


def test_trad_job_id_format():
    assert trad_job_id('acl_2024.acl-short.8') == 'trad:acl_2024.acl-short.8'


def test_load_trad_registry_nine_rows():
    rows = load_trad_registry()
    assert len(rows) == 9
    paper_ids = {row['paper_id'] for row in rows}
    assert 'neurips_1706.03762v7' in paper_ids
    assert all(row.get('manuscript_job_id') for row in rows)
    assert all(row.get('criteria_job_id') for row in rows)
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `pytest tests/benchmark/test_trad_registry.py -v`

- [ ] **Step 3: Create `benchmark/trad_registry.jsonl`** (9 lines)

```jsonl
{"paper_id":"acl_2024.acl-short.8","venue":"ACL","source_pdf":"trad_LLM/Language_Models_Do_Hard_Arithmetic_Tasks_Easily_and_Hardly_Do_Easy_Arithmetic_Tasks_ChatGPT_Traditional_LLM_Baseline.pdf","manuscript_job_id":"330c5928-a8d1-4ec1-8efe-ed2081001477","criteria_job_id":"d9f100a7-874d-47b6-9c9b-b80d3f177af5"}
{"paper_id":"acl_2024.findings-acl.438","venue":"ACL","source_pdf":"trad_LLM/The_Impact_of_Demonstrations_on_Multilingual_In-Context_Learning_A_Multidimensional_Analysis_ChatGPT_Traditional_LLM_Baseline.pdf","manuscript_job_id":"1a3f124c-38ab-417c-961f-44680c61b7b0","criteria_job_id":"7ca19601-5671-4749-9dbf-5e219a43b041"}
{"paper_id":"ets_liu-usingaibasedobject-2023","venue":"ETS","source_pdf":"trad_LLM/Using_an_AI-Based_Object_Detection_Translation_Application_for_English_Vocabulary_Learning_ChatGPT_Traditional_LLM_Baseline (1).pdf","manuscript_job_id":"4a2f29ed-d629-4510-b73f-c54fadad5111","criteria_job_id":"84a87aa2-755c-4ab7-b590-095ffebc7865"}
{"paper_id":"iclr_1412.6980v9","venue":"ICLR","source_pdf":"trad_LLM/ADAM_A_METHOD_FOR_STOCHASTIC_OPTIMIZATION_ChatGPT_Traditional_LLM_Baseline.pdf","manuscript_job_id":"5402365b-32ec-4d42-a0ae-97d050d3e57c","criteria_job_id":"0db577f7-8372-4d25-86a1-d70448280374"}
{"paper_id":"iclr_9447_tabular_insights_visual_i","venue":"ICLR","source_pdf":"trad_LLM/Tabular_Insights_Visual_Impacts_Transferring_Expertise_from_Tables_to_Images_ChatGPT_Traditional_LLM_Baseline.pdf","manuscript_job_id":"23532855-90cb-4575-836e-135b5e22c32d","criteria_job_id":"31d08d1e-d832-4cfb-ba28-2c3bab6a8861"}
{"paper_id":"icml_2311.10263v2","venue":"ICML","source_pdf":"trad_LLM/Stable_Differentiable_Causal_Discovery_ChatGPT_Traditional_LLM_Baseline.pdf","manuscript_job_id":"7459cf44-0d8d-44d1-b00c-0db992debdd4","criteria_job_id":"5ccf9af3-3aad-4b6f-89ea-72d188536e3b"}
{"paper_id":"icml_2402.01869v2","venue":"ICML","source_pdf":"trad_LLM/INFERCEPT_Efficient_Intercept_Support_for_Augmented_Large_Language_Model_Inference_ChatGPT_Traditional_LLM_Baseline.pdf","manuscript_job_id":"e032ccfb-f64c-49a4-b6fb-3d757fc308cf","criteria_job_id":"144522e3-b6b1-4c68-83c0-852a68eb1508"}
{"paper_id":"neurips_1706.03762v7","venue":"NeurIPS","source_pdf":"trad_LLM/Attention_Is_All_You_Need_ChatGPT_Traditional_LLM_Baseline.pdf","manuscript_job_id":"ecb9e257-cba6-4fd6-97c7-c2cb94038718","criteria_job_id":"0b44ed20-2287-4507-91ee-59b46d93026f"}
{"paper_id":"neurips_2402.05602v2","venue":"NeurIPS","source_pdf":"trad_LLM/AttnLRP_Attention-Aware_Layer-Wise_Relevance_Propagation_for_Transformers_ChatGPT_Traditional_LLM_Baseline.pdf","manuscript_job_id":"e43a0bd7-d10d-43eb-9f3f-e836254654f2","criteria_job_id":"d8ef2ca6-556c-4543-8fa1-f83aee043f21"}
```

- [ ] **Step 4: Implement `benchmark/trad_registry.py`**

```python
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from benchmark.paths import REPO_ROOT, TRAD_REGISTRY_PATH


def trad_job_id(paper_id: str) -> str:
    return f'trad:{paper_id}'


@lru_cache(maxsize=1)
def load_trad_registry() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in TRAD_REGISTRY_PATH.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def registry_row_for_paper(paper_id: str) -> dict[str, Any] | None:
    for row in load_trad_registry():
        if row.get('paper_id') == paper_id:
            return row
    return None


def resolve_source_pdf(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path
```

- [ ] **Step 5: Run tests — expect PASS**

Run: `pytest tests/benchmark/test_trad_registry.py -v`

- [ ] **Step 6: Commit**

```bash
git add benchmark/trad_registry.jsonl benchmark/trad_registry.py tests/benchmark/test_trad_registry.py
git commit -m "feat(benchmark): add trad LLM paper registry"
```

---

### Task 4: Review sources adapter

**Files:**
- Create: `benchmark/review_sources.py`
- Create: `tests/benchmark/test_review_sources.py`
- Create: `tests/fixtures/benchmark/trad_sample.md`
- Modify: `benchmark/judge.py` (replace `load_judge_artifacts` body with delegate)

- [ ] **Step 1: Create fixture `tests/fixtures/benchmark/trad_sample.md`**

```markdown
## Summary
This paper presents a novel approach.

## Weaknesses
- The evaluation lacks statistical significance testing.
- Reproducibility details are insufficient.

## Key Issues
1. Add confidence intervals for main results.
```

- [ ] **Step 2: Write failing test**

```python
# tests/benchmark/test_review_sources.py
from pathlib import Path

from benchmark.review_sources import load_review_artifacts, load_trad_review_artifacts

FIXTURE_MD = Path('tests/fixtures/benchmark/trad_sample.md')


def test_load_trad_review_artifacts_from_paths(tmp_path, monkeypatch):
    review_path = tmp_path / 'review.md'
    review_path.write_text(FIXTURE_MD.read_text(encoding='utf-8'), encoding='utf-8')
    ms_path = tmp_path / 'mineru_full.md'
    ms_path.write_text('# Manuscript\n\nBody text.' * 200, encoding='utf-8')
    crit_path = tmp_path / 'review_criteria_bundle.json'
    crit_path.write_text('{"criteria": []}', encoding='utf-8')

    row = {
        'review_path': str(review_path),
        'manuscript_job_id': 'unused',
        'criteria_job_id': 'unused',
    }
    # Monkeypatch job dir lookup to tmp_path for both job ids
    from benchmark import review_sources as rs

    def fake_job_dir(job_id: str) -> Path:
        return tmp_path

    monkeypatch.setattr(rs, '_job_dir', fake_job_dir)
    artifacts = load_trad_review_artifacts(row)
    assert 'Weaknesses' in artifacts['final_markdown']
    assert 'Manuscript' in artifacts['manuscript_excerpt']
    assert artifacts['criteria_json']
```

- [ ] **Step 3: Run test — expect FAIL**

Run: `pytest tests/benchmark/test_review_sources.py -v`

- [ ] **Step 4: Implement `benchmark/review_sources.py`**

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.judge import _truncate_criteria, _truncate_manuscript, _truncate_report
from benchmark.paths import DATA_JOBS_DIR, REPO_ROOT
from benchmark.trad_registry import resolve_source_pdf


def _job_dir(job_id: str) -> Path:
    return DATA_JOBS_DIR / job_id


def load_job_review_artifacts(job_id: str) -> dict[str, str]:
    job_dir = _job_dir(job_id)
    manuscript_excerpt = ''
    mineru_path = job_dir / 'mineru_full.md'
    if mineru_path.exists():
        manuscript_excerpt = _truncate_manuscript(mineru_path.read_text(encoding='utf-8'))

    final_markdown = ''
    final_path = job_dir / 'final_report.md'
    if final_path.exists():
        final_markdown = _truncate_report(final_path.read_text(encoding='utf-8'))

    criteria_json = ''
    criteria_path = job_dir / 'review_criteria_bundle.json'
    if criteria_path.exists():
        criteria_json = _truncate_criteria(criteria_path.read_text(encoding='utf-8'))

    return {
        'manuscript_excerpt': manuscript_excerpt,
        'final_markdown': final_markdown,
        'criteria_json': criteria_json,
    }


def load_trad_review_artifacts(row: dict[str, Any]) -> dict[str, str]:
    review_path = Path(row.get('review_path') or '')
    if not review_path.is_absolute():
        review_path = REPO_ROOT / review_path

    final_markdown = ''
    if review_path.exists():
        final_markdown = _truncate_report(review_path.read_text(encoding='utf-8'))

    ms_job = str(row.get('manuscript_job_id') or '')
    crit_job = str(row.get('criteria_job_id') or '')
    ms_artifacts = load_job_review_artifacts(ms_job) if ms_job else {}
    crit_artifacts = load_job_review_artifacts(crit_job) if crit_job else {}

    return {
        'manuscript_excerpt': ms_artifacts.get('manuscript_excerpt') or '',
        'final_markdown': final_markdown,
        'criteria_json': crit_artifacts.get('criteria_json') or '',
    }


def load_review_artifacts(row: dict[str, Any] | str) -> dict[str, str]:
    if isinstance(row, str):
        return load_job_review_artifacts(row)
    condition = str(row.get('condition') or '').strip().upper()
    if condition == 'TRAD_LLM' or str(row.get('job_id', '')).startswith('trad:'):
        return load_trad_review_artifacts(row)
    job_id = str(row.get('job_id') or '')
    if job_id:
        return load_job_review_artifacts(job_id)
    raise ValueError('row must include job_id or TRAD_LLM fields')
```

- [ ] **Step 5: Refactor `benchmark/judge.py`**

Replace `load_judge_artifacts` implementation with:

```python
def load_judge_artifacts(job_id: str) -> dict[str, str]:
    from benchmark.review_sources import load_job_review_artifacts
    return load_job_review_artifacts(job_id)
```

Update `_enrich_row_for_judge` to call `load_review_artifacts(row)` when row has `condition=TRAD_LLM`.

Extract `_truncate_manuscript`, `_truncate_report`, `_truncate_criteria` to remain importable from `judge.py` (or move to `review_sources.py` and re-export). Prefer keeping trunc helpers in `judge.py` and importing them in `review_sources` as shown above.

- [ ] **Step 6: Run tests**

Run: `pytest tests/benchmark/test_review_sources.py tests/benchmark/test_judge.py -v`  
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add benchmark/review_sources.py benchmark/judge.py tests/benchmark/test_review_sources.py tests/fixtures/benchmark/trad_sample.md
git commit -m "feat(benchmark): add review_sources adapter for trad LLM"
```

---

### Task 5: Trad ingest (PDF → markdown)

**Files:**
- Create: `benchmark/trad_ingest.py`
- Create: `tests/benchmark/test_trad_ingest.py`

- [ ] **Step 1: Write failing test**

```python
# tests/benchmark/test_trad_ingest.py
from benchmark.trad_ingest import extract_pdf_text, validate_trad_review_bytes

def test_validate_trad_review_bytes_min_size():
    assert validate_trad_review_bytes(b'x' * 2049) == []
    assert validate_trad_review_bytes(b'short') != []


def test_extract_pdf_text_roundtrip(tmp_path):
    import fitz  # pymupdf
    pdf_path = tmp_path / 'sample.pdf'
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), 'Peer review summary for benchmark test.')
    doc.save(pdf_path)
    doc.close()
    text = extract_pdf_text(pdf_path)
    assert 'Peer review summary' in text
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `pytest tests/benchmark/test_trad_ingest.py -v`

- [ ] **Step 3: Implement `benchmark/trad_ingest.py`**

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.checks import MIN_FINAL_REPORT_BYTES
from benchmark.paths import REPO_ROOT, TRAD_REVIEWS_DIR, TRAD_RUNS_JSONL_PATH
from benchmark.trad_registry import load_trad_registry, resolve_source_pdf, trad_job_id


def extract_pdf_text(pdf_path: Path) -> str:
    import fitz
    doc = fitz.open(pdf_path)
    try:
        parts: list[str] = []
        for page in doc:
            parts.append(page.get_text())
        return '\n\n'.join(parts).strip()
    finally:
        doc.close()


def validate_trad_review_bytes(data: bytes) -> list[str]:
    if len(data) <= MIN_FINAL_REPORT_BYTES:
        return [f'review too small ({len(data)} bytes; need > {MIN_FINAL_REPORT_BYTES})']
    return []


def ingest_trad_reviews(*, paper_id: str | None = None) -> list[dict[str, Any]]:
    TRAD_REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    TRAD_RUNS_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict[str, Any]] = []
    for entry in load_trad_registry():
        pid = str(entry['paper_id'])
        if paper_id is not None and pid != paper_id:
            continue

        pdf_path = resolve_source_pdf(str(entry['source_pdf']))
        if not pdf_path.exists():
            raise FileNotFoundError(f'trad source PDF missing: {pdf_path}')

        text = extract_pdf_text(pdf_path)
        md_path = TRAD_REVIEWS_DIR / f'{pid}.md'
        md_path.write_text(text, encoding='utf-8')

        errors = validate_trad_review_bytes(text.encode('utf-8'))
        if errors:
            raise ValueError(f'{pid}: {"; ".join(errors)}')

        rel_review = md_path.relative_to(REPO_ROOT).as_posix()
        run_row = {
            'job_id': trad_job_id(pid),
            'paper_id': pid,
            'venue': entry['venue'],
            'condition': 'TRAD_LLM',
            'status': 'completed',
            'manuscript_job_id': entry['manuscript_job_id'],
            'criteria_job_id': entry['criteria_job_id'],
            'review_path': rel_review,
            'final_md_bytes': len(text.encode('utf-8')),
        }
        rows_out.append(run_row)

    with TRAD_RUNS_JSONL_PATH.open('w', encoding='utf-8') as handle:
        for row in rows_out:
            handle.write(json.dumps(row) + '\n')

    return rows_out
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest tests/benchmark/test_trad_ingest.py -v`

- [ ] **Step 5: Run ingest on real PDFs (operator smoke)**

Run: `python -m benchmark trad-ingest`  
Expected: 9 lines in `benchmark/results/trad_runs.jsonl`; 9 files in `benchmark/trad_reviews/`

- [ ] **Step 6: Commit**

```bash
git add benchmark/trad_ingest.py tests/benchmark/test_trad_ingest.py
git commit -m "feat(benchmark): add trad LLM PDF ingest"
```

---

### Task 6: Judge TRAD runs

**Files:**
- Modify: `benchmark/judge.py`
- Modify: `benchmark/cli.py`
- Create: `tests/benchmark/test_judge_trad.py`

- [ ] **Step 1: Write failing test**

```python
# tests/benchmark/test_judge_trad.py
from unittest.mock import patch

from benchmark.judge import judge_trad_all


def test_judge_trad_all_skips_completion_validation(tmp_path, monkeypatch):
    trad_runs = tmp_path / 'trad_runs.jsonl'
    trad_runs.write_text(
        '{"job_id":"trad:test","paper_id":"test","venue":"ACL","condition":"TRAD_LLM",'
        '"review_path":"tests/fixtures/benchmark/trad_sample.md",'
        '"manuscript_job_id":"x","criteria_job_id":"y","final_md_bytes":3000}\n',
        encoding='utf-8',
    )
    monkeypatch.setenv('BENCHMARK_JUDGE_MODE', 'composite')

    with patch('benchmark.judge.judge_run') as mock_judge:
        mock_judge.return_value = {
            'paper_id': 'test',
            'job_id': 'trad:test',
            'venue': 'ACL',
            'condition': 'TRAD_LLM',
            'rubric_alignment': 0.5,
        }
        results = judge_trad_all(runs_path=trad_runs, output_path=tmp_path / 'scores.csv')
    assert len(results) == 1
    mock_judge.assert_called_once()
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `pytest tests/benchmark/test_judge_trad.py -v`

- [ ] **Step 3: Add to `benchmark/judge.py`**

```python
def _read_trad_runs_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def judge_trad_all(
    *,
    paper_id: str | None = None,
    runs_path: Path | None = None,
    output_path: Path | None = None,
) -> list[dict[str, Any]]:
    from benchmark.paths import TRAD_RUNS_JSONL_PATH

    source = runs_path or TRAD_RUNS_JSONL_PATH
    destination = output_path or REVIEW_QUALITY_SCORES_PATH

    candidate_rows = _read_trad_runs_jsonl(source)
    if paper_id:
        candidate_rows = [r for r in candidate_rows if r.get('paper_id') == paper_id]

    existing = _existing_judged_job_ids(destination)
    agent_results: list[dict[str, Any]] = []
    if destination.exists() and destination.stat().st_size > 0:
        with destination.open(encoding='utf-8') as handle:
            agent_results = [dict(r) for r in csv.DictReader(handle)]
    agent_results = [r for r in agent_results if str(r.get('condition', '')).upper() != 'TRAD_LLM']

    trad_results: list[dict[str, Any]] = []
    for row in candidate_rows:
        jid = str(row.get('job_id', ''))
        if jid in existing:
            continue
        trad_results.append(judge_run(row))
        pause_between_eval_calls()

    combined = agent_results + trad_results
    if not combined:
        return combined

    fieldnames = list(combined[0].keys())
    with destination.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined)
    return trad_results
```

Update `_enrich_row_for_judge`:

```python
    artifacts = load_review_artifacts(row)
```

(import from `benchmark.review_sources`)

Update `judge_run` to not call `validate_run_completion` when `condition == 'TRAD_LLM'` (handle in `judge_trad_all` only — `judge_all` unchanged).

- [ ] **Step 4: Add CLI flag in `benchmark/cli.py`**

```python
def _cmd_judge(args: argparse.Namespace) -> int:
    from benchmark.judge import REVIEW_QUALITY_SCORES_PATH, judge_all, judge_trad_all

    if getattr(args, 'source', None) == 'trad':
        scores = judge_trad_all(paper_id=args.paper_id)
    else:
        scores = judge_all(job_id=args.job_id)
    print(f'Wrote judge scores ({len(scores)} new rows) to {REVIEW_QUALITY_SCORES_PATH}')
    return 0
```

Add to judge subparser: `parser.add_argument('--source', choices=['agent', 'trad'], default='agent')`

- [ ] **Step 5: Run tests — expect PASS**

Run: `pytest tests/benchmark/test_judge_trad.py -v`

- [ ] **Step 6: Commit**

```bash
git add benchmark/judge.py benchmark/cli.py tests/benchmark/test_judge_trad.py
git commit -m "feat(benchmark): judge TRAD_LLM reviews via shared eval"
```

---

### Task 7: Faithfulness for TRAD

**Files:**
- Modify: `benchmark/faithfulness.py`
- Modify: `benchmark/cli.py`
- Create: `tests/benchmark/test_faithfulness_trad.py`

- [ ] **Step 1: Write failing test** (mock RAGAS)

```python
# tests/benchmark/test_faithfulness_trad.py
from unittest.mock import patch

from benchmark.faithfulness import faithfulness_trad_all


def test_faithfulness_trad_all_no_annotations(tmp_path, monkeypatch):
    trad_runs = tmp_path / 'trad_runs.jsonl'
    trad_runs.write_text(
        '{"job_id":"trad:test","paper_id":"test","venue":"ACL","condition":"TRAD_LLM",'
        '"review_path":"tests/fixtures/benchmark/trad_sample.md",'
        '"manuscript_job_id":"x","criteria_job_id":"y","final_md_bytes":3000}\n',
        encoding='utf-8',
    )

    with patch('benchmark.faithfulness.score_claims') as mock_score:
        mock_score.return_value = {
            'faithfulness_mean': 0.1,
            'faithfulness_n': 2,
            'claim_rows': [],
        }
        rows = faithfulness_trad_all(runs_path=trad_runs, run_scores_path=tmp_path / 'faith.csv')
    assert len(rows) == 1
```

- [ ] **Step 2: Implement `faithfulness_trad_all` in `benchmark/faithfulness.py`**

Mirror `faithfulness_all` but:
- Read `TRAD_RUNS_JSONL_PATH`
- Use `load_review_artifacts(row)` instead of `load_judge_artifacts(jid)`
- Pass `annotations=[]` to `extract_critique_claims`
- Append to CSVs; strip existing TRAD_LLM rows before rewrite (same pattern as judge)

- [ ] **Step 3: CLI**

```python
def _cmd_faithfulness(args):
    if getattr(args, 'source', None) == 'trad':
        rows = faithfulness_trad_all(paper_id=args.paper_id)
    else:
        rows = faithfulness_all(job_id=args.job_id)
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest tests/benchmark/test_faithfulness_trad.py -v`

- [ ] **Step 5: Commit**

```bash
git add benchmark/faithfulness.py benchmark/cli.py tests/benchmark/test_faithfulness_trad.py
git commit -m "feat(benchmark): faithfulness scoring for TRAD_LLM"
```

---

### Task 8: Compare subset filter (4/2/1 headline)

**Files:**
- Modify: `benchmark/compare.py`
- Modify: `benchmark/cli.py`
- Create: `tests/benchmark/test_compare_subset.py`

- [ ] **Step 1: Write failing test**

```python
# tests/benchmark/test_compare_subset.py
import pandas as pd

from benchmark.analysis_subset import load_analysis_subset
from benchmark.compare import apply_paper_id_filter


def test_apply_paper_id_filter_b2a():
    subset = load_analysis_subset()
    df = pd.DataFrame({
        'paper_id': subset['included_paper_ids'] + subset['excluded_paper_ids'],
        'pairwise_winner': ['KG_ON'] * 4 + ['KG_OFF'] * 2 + ['tie'] + ['KG_OFF', 'KG_OFF'],
    })
    filtered = apply_paper_id_filter(df, use_subset=True)
    assert len(filtered) == 7
    winners = filtered['pairwise_winner'].value_counts()
    assert winners.get('KG_ON', 0) == 4
    assert winners.get('KG_OFF', 0) == 2
    assert winners.get('tie', 0) == 1
```

Adjust winner list to match actual 7-paper pairwise winners when building fixture from real `pairwise_judge_scores.csv`.

- [ ] **Step 2: Add `apply_paper_id_filter` to `benchmark/compare.py`**

```python
from benchmark.analysis_subset import filter_paper_ids, load_analysis_subset

def apply_paper_id_filter(frame: pd.DataFrame, *, use_subset: bool = True) -> pd.DataFrame:
    if frame.empty or 'paper_id' not in frame.columns:
        return frame
    allowed = filter_paper_ids(frame['paper_id'].astype(str).tolist(), use_subset=use_subset)
    allowed_set = set(allowed)
    return frame[frame['paper_id'].astype(str).isin(allowed_set)].copy()
```

- [ ] **Step 3: Update `build_paired_comparison`**

Add parameter `use_subset: bool = True`. After `_merge_pairwise_judge`, call:

```python
paired = apply_paper_id_filter(paired, use_subset=use_subset)
```

Recompute `valid_pairs` and `metadata` from filtered frame.

- [ ] **Step 4: CLI `--all-papers` on compare**

```python
def _cmd_compare(args):
    metadata = build_paired_comparison(use_subset=not getattr(args, 'all_papers', False))
```

- [ ] **Step 5: Run tests — expect PASS**

Run: `pytest tests/benchmark/test_compare_subset.py -v`

- [ ] **Step 6: Commit**

```bash
git add benchmark/compare.py benchmark/cli.py tests/benchmark/test_compare_subset.py
git commit -m "feat(benchmark): filter paired comparison to B2a analysis subset"
```

---

### Task 9: Trad pairwise judge

**Files:**
- Create: `benchmark/trad_pairwise_judge.py`
- Create: `tests/benchmark/test_trad_pairwise_judge.py`

- [ ] **Step 1: Write failing test**

```python
# tests/benchmark/test_trad_pairwise_judge.py
from benchmark.trad_pairwise_judge import parse_trad_pairwise_response, TRAD_PAIRWISE_JSON_KEYS


def test_parse_trad_pairwise_response():
    raw = '{"winner":"KG_ON","rubric_alignment_winner":"KG_ON","confidence":"high","one_line_reason":"Better rubric mapping."}'
    parsed = parse_trad_pairwise_response(raw)
    assert parsed['winner'] == 'KG_ON'
    assert parsed['reason'] == 'Better rubric mapping.'
```

- [ ] **Step 2: Implement `benchmark/trad_pairwise_judge.py`**

Key functions:
- `parse_trad_pairwise_response(raw)` — reuse pattern from `pairwise_judge.parse_pairwise_response` but winners are `TRAD|KG_ON|KG_OFF|tie`
- `trad_pairwise_prompt(pair_type, venue, criteria, manuscript, review_a, review_b)` — `pair_type` labels which review is A/B
- `trad_pairwise_compare(trad_row, agent_row, *, pair_type: str)` — loads artifacts via `load_review_artifacts`
- `trad_pairwise_all(*, paper_id=None, use_subset=True)` — for each paper in subset with TRAD + KG_OFF + KG_ON runs, emit two rows (`TRAD_VS_KG_OFF`, `TRAD_VS_KG_ON`)

Output CSV columns:
`paper_id,venue,pair_type,winner,rubric_alignment_winner,confidence,reason,job_id_trad,job_id_agent`

- [ ] **Step 3: Run tests — expect PASS**

Run: `pytest tests/benchmark/test_trad_pairwise_judge.py -v`

- [ ] **Step 4: Commit**

```bash
git add benchmark/trad_pairwise_judge.py tests/benchmark/test_trad_pairwise_judge.py
git commit -m "feat(benchmark): trad vs agent pairwise judge"
```

---

### Task 10: Three-way compare

**Files:**
- Create: `benchmark/three_way_compare.py`
- Create: `tests/benchmark/test_three_way_compare.py`

- [ ] **Step 1: Write failing test**

```python
# tests/benchmark/test_three_way_compare.py
import pandas as pd

from benchmark.three_way_compare import build_three_way_summary


def test_build_three_way_summary_medians():
    judge = pd.DataFrame([
        {'paper_id': 'p1', 'condition': 'TRAD_LLM', 'rubric_alignment': 0.3},
        {'paper_id': 'p1', 'condition': 'KG_OFF', 'rubric_alignment': 0.5},
        {'paper_id': 'p1', 'condition': 'KG_ON', 'rubric_alignment': 0.7},
        {'paper_id': 'p2', 'condition': 'TRAD_LLM', 'rubric_alignment': 0.4},
        {'paper_id': 'p2', 'condition': 'KG_OFF', 'rubric_alignment': 0.4},
        {'paper_id': 'p2', 'condition': 'KG_ON', 'rubric_alignment': 0.6},
    ])
    summary = build_three_way_summary(judge_df=judge, use_subset=False)
    overall = summary[summary['paper_id'] == '__overall__'].iloc[0]
    assert overall['median_rubric_alignment_TRAD_LLM'] == 0.35
    assert overall['median_rubric_alignment_KG_ON'] == 0.65
```

- [ ] **Step 2: Implement `build_three_way_summary`**

- Merge `review_quality_scores.csv` + `faithfulness_run_scores.csv` on `job_id`
- Normalize `condition` column (`TRAD_LLM`, `KG_ON`, `KG_OFF`)
- Filter with `apply_paper_id_filter`
- Per paper: pivot medians for each metric × condition
- Append `__overall__` row with cross-paper medians
- Write `THREE_WAY_SUMMARY_PATH`

- [ ] **Step 3: Run tests — expect PASS**

Run: `pytest tests/benchmark/test_three_way_compare.py -v`

- [ ] **Step 4: Commit**

```bash
git add benchmark/three_way_compare.py tests/benchmark/test_three_way_compare.py
git commit -m "feat(benchmark): three-way condition summary table"
```

---

### Task 11: Report updates

**Files:**
- Modify: `benchmark/report.py`
- Modify: `benchmark/cli.py`

- [ ] **Step 1: Extend `build_benchmark_summary`**

Add sections:
- **Analysis subset** — 7 papers, list excluded with reason from `load_analysis_subset()`
- **Pairwise judge** — use filtered paired_comparison (4/2/1)
- **Three-way medians** — read `three_way_summary.csv` `__overall__` row
- **Trad pairwise win rates** — count winners from `trad_pairwise_judge_scores.csv` grouped by `pair_type`

Pass `use_subset: bool` through `generate_report(rebuild_comparison=True, use_subset=True)`.

- [ ] **Step 2: Manual smoke**

Run: `python -m benchmark report`  
Expected: `benchmark_summary.md` contains "Analysis subset" and "Three-way" sections; pairwise shows 4/2/1

- [ ] **Step 3: Commit**

```bash
git add benchmark/report.py benchmark/cli.py
git commit -m "feat(benchmark): report analysis subset and three-way headlines"
```

---

### Task 12: CLI wiring + trad-eval pipeline phase

**Files:**
- Modify: `benchmark/cli.py`

- [ ] **Step 1: Add subcommands**

```python
def _cmd_trad_ingest(_args):
    from benchmark.trad_ingest import ingest_trad_reviews
    rows = ingest_trad_reviews()
    print(f'Ingested {len(rows)} trad reviews')
    return 0

def _cmd_trad_pairwise_judge(args):
    from benchmark.trad_pairwise_judge import trad_pairwise_all
    use_subset = not getattr(args, 'all_papers', False)
    rows = trad_pairwise_all(paper_id=args.paper_id, use_subset=use_subset)
    print(f'Wrote {len(rows)} trad pairwise rows')
    return 0

def _cmd_three_way_compare(args):
    from benchmark.three_way_compare import build_three_way_summary
    build_three_way_summary(use_subset=not getattr(args, 'all_papers', False))
    return 0
```

Register: `trad-ingest`, `trad-pairwise-judge`, `three-way-compare`

Add optional pipeline phase:

```python
PIPELINE_TRAD_EVAL_STEPS = (
    'trad-ingest',
    'judge',
    'faithfulness',
    'trad-pairwise-judge',
    'compare',
    'three-way-compare',
    'report',
)
```

For `pipeline --phase trad-eval`, set `args.source = 'trad'` before judge and faithfulness steps.

- [ ] **Step 2: Commit**

```bash
git add benchmark/cli.py
git commit -m "feat(benchmark): CLI for trad ingest, pairwise, and trad-eval phase"
```

---

### Task 13: Full test sweep + operator validation

**Files:**
- Modify: `docs/operator-benchmark-rerun.md` (append trad-eval section)

- [ ] **Step 1: Run full pytest**

Run: `pytest tests/benchmark/ -v`  
Expected: all pass (including prior 61+ tests)

- [ ] **Step 2: Operator trad eval run**

```bash
python -m benchmark trad-ingest
python -m benchmark judge --source trad
python -m benchmark faithfulness --source trad
python -m benchmark trad-pairwise-judge
python -m benchmark compare
python -m benchmark three-way-compare
python -m benchmark report
```

- [ ] **Step 3: Verify headline**

Check `benchmark/results/benchmark_summary.md`:
- Valid paired papers: **7**
- KG_ON wins: **4**, KG_OFF wins: **2**, ties: **1**
- Three-way section present with TRAD / KG_OFF / KG_ON medians

- [ ] **Step 4: Document operator steps**

Append to `docs/operator-benchmark-rerun.md`:

```markdown
## Traditional LLM eval (trad-eval)

Prerequisite: v2 agent eval complete.

python -m benchmark pipeline --phase trad-eval

Or step-by-step: trad-ingest → judge --source trad → faithfulness --source trad → trad-pairwise-judge → compare → three-way-compare → report
```

- [ ] **Step 5: Commit**

```bash
git add docs/operator-benchmark-rerun.md
git commit -m "docs: operator runbook for trad LLM eval"
```

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| B2a analysis_subset.json | Task 1 |
| 7-paper default filter | Task 8, 11 |
| trad_registry + ingest | Task 3, 5 |
| review_sources adapter | Task 4 |
| judge TRAD_LLM | Task 6 |
| faithfulness TRAD_LLM | Task 7 |
| trad pairwise | Task 9 |
| three-way summary | Task 10 |
| report three-way + 4/2/1 | Task 11 |
| CLI + pipeline | Task 12 |
| operator validation | Task 13 |

## Placeholder scan

No TBD/TODO items. All code blocks are complete.
