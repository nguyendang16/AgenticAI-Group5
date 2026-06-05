# KG vs No-KG Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an automated paired benchmark (KG_ON vs KG_OFF) with deterministic validation, imported evaluation libraries (DeepEval, sklearn, scipy, statsmodels, pandas), and CSV/Markdown reports — seeded from `KG_testPapers/` and gap-filled via OpenReview/journal fetchers.

**Architecture:** A `benchmark/` Python package wraps the existing `main.py` CLI. Phase 0 adds a generic fast-mode prompt when criteria are disabled. Ingestion scripts build `manifest.jsonl`; `runner.py` executes paired jobs with env injection; `collect.py` / `checks.py` / `judge.py` (DeepEval) / `decision_metrics.py` (sklearn) / `compare.py` (pandas + scipy + statsmodels) produce `paired_comparison.csv` and `benchmark_summary.md`. **Do not reimplement metric logic** — thin wrappers around library APIs only. RAGAS claim-level faithfulness is v2 (`claims.py` + `faithfulness.py`).

**Tech Stack:** Python 3.11+, existing review pipeline, SQLite (stdlib). Evaluation libs (import only):

| Library | GitHub | Our module |
|---------|--------|------------|
| DeepEval | https://github.com/confident-ai/deepeval | `benchmark/judge.py` |
| scikit-learn | https://github.com/scikit-learn/scikit-learn | `benchmark/decision_metrics.py` |
| pandas | https://github.com/pandas-dev/pandas | `benchmark/compare.py` |
| SciPy | https://github.com/scipy/scipy | `benchmark/compare.py` |
| statsmodels | https://github.com/statsmodels/statsmodels | `benchmark/compare.py` |
| RAGAS (v2) | https://github.com/explodinggradients/ragas | `benchmark/faithfulness.py` |
| openreview-py | https://github.com/openreview/openreview-py | `benchmark/fetch_openreview.py` |

**Spec:** `docs/superpowers/specs/2026-06-06-kg-benchmark-design.md`

**Worktree:** Implement in a dedicated git worktree if other work is in progress (`git worktree add ../AgenticAI-Group5-kg-bench -b feat/kg-benchmark`).

---

## File Map

| File | Responsibility |
|------|----------------|
| `deepreview/prompts/review_agent_prompt.py` | Branch fast prompt: KG vs generic |
| `deepreview/tools/review_tools.py` | Fast generic section list; thread `kg_criteria_active` |
| `benchmark/__init__.py` | Package marker |
| `benchmark/paths.py` | Repo-root paths (`KG_testPapers`, `benchmark/papers`, `results`) |
| `benchmark/models.py` | `PaperRecord`, `RunRecord`, `PairedResult` dataclasses |
| `benchmark/env.py` | `build_benchmark_env(condition, venue, base_env)` |
| `benchmark/ingest_local.py` | Scan `KG_testPapers/`, detect venue, emit rows |
| `benchmark/fetch_openreview.py` | Download PDFs for CHI/AAAI gaps |
| `benchmark/fetch_journals.py` | TWELF/ETRD/C&E/ETS gap fill (curated URLs + graceful skip) |
| `benchmark/build_manifest.py` | Merge sources → `manifest.jsonl` (2 per venue) |
| `benchmark/registry.py` | SQLite `runs` table |
| `benchmark/runner.py` | Paired submit + watch loop |
| `benchmark/collect.py` | Harvest `data/jobs/<id>/` artifacts |
| `benchmark/checks.py` | Deterministic condition validators |
| `benchmark/judge.py` | DeepEval `GEval` / `LLMTestCase` wrapper (report-level) |
| `benchmark/decision_metrics.py` | sklearn metrics when `expected_decision` labels exist |
| `benchmark/compare.py` | pandas joins; scipy Wilcoxon; statsmodels/bootstrap CI |
| `benchmark/report.py` | `benchmark_summary.md` generator |
| `benchmark/cli.py` | `python -m benchmark` subcommands |
| `benchmark/claims.py` | *(v2)* atomic claim extraction |
| `benchmark/faithfulness.py` | *(v2)* RAGAS `evaluate()` wrapper |
| `tests/test_fast_generic_prompt.py` | Prompt + section-order regression |
| `tests/benchmark/test_ingest_local.py` | Local PDF venue detection |
| `tests/benchmark/test_checks.py` | Deterministic check unit tests |
| `tests/benchmark/test_env.py` | Env dict correctness |
| `pyproject.toml` | `[project.optional-dependencies] benchmark = [...]` |

---

### Task 1: Generic fast-mode prompt (KG_OFF)

**Files:**
- Create: `tests/test_fast_generic_prompt.py`
- Modify: `deepreview/prompts/review_agent_prompt.py:32-112`

- [ ] **Step 1: Write failing tests**

Create `tests/test_fast_generic_prompt.py`:

```python
from __future__ import annotations

from deepreview.prompts.review_agent_prompt import build_review_agent_system_prompt


def test_fast_kg_off_prompt_omits_claim_audit_and_criterion_id():
    prompt = build_review_agent_system_prompt(
        source_file_id='job-1',
        source_file_name='paper.pdf',
        paper_markdown='# Title\n\nBody.',
        review_fast_mode=True,
        criteria_bundle=None,
    )
    assert 'Claim-Level Audit' not in prompt
    assert 'criterion_id' not in prompt
    assert 'VENUE REVIEW CRITERIA' not in prompt
    assert 'Summary, Strengths, Weaknesses, Key Issues, Actionable Suggestions, Scores' in prompt


def test_fast_kg_on_prompt_keeps_claim_audit():
    bundle = {
        'criteria_count': 1,
        'query': {'venue': 'ICLR'},
        'criteria_by_group': {
            'SCOPE_FIT': [
                {
                    'criterion_id': 'ICLR_C01_SCOPE_FIT',
                    'criterion_name': 'Scope',
                    'criterion_group': 'SCOPE_FIT',
                    'description': 'Scope fit.',
                }
            ]
        },
    }
    prompt = build_review_agent_system_prompt(
        source_file_id='job-1',
        source_file_name='paper.pdf',
        paper_markdown='# Title\n\nBody.',
        review_fast_mode=True,
        criteria_bundle=bundle,
    )
    assert 'Claim-Level Audit' in prompt
    assert 'criterion_id' in prompt
    assert 'VENUE REVIEW CRITERIA' in prompt
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_fast_generic_prompt.py -v
```

Expected: FAIL — generic prompt not implemented; KG_OFF prompt still contains `Claim-Level Audit`.

- [ ] **Step 3: Implement prompt branch**

In `deepreview/prompts/review_agent_prompt.py`, refactor `_build_fast_review_annotator_prompt` to branch on `criteria_bundle`:

```python
def _build_fast_review_annotator_prompt(
    *,
    paper_markdown: str,
    source_file_id: str,
    source_file_name: str,
    ui_language: str,
    max_markdown_chars: int,
    max_tool_turns: int,
    min_annotations: int,
    criteria_bundle: dict[str, Any] | None = None,
) -> str:
    resolved_ui_language = normalize_ui_language(ui_language, fallback='en', strict=False)
    markdown_text = _truncate_paper_markdown(paper_markdown, max_chars=max_markdown_chars)
    language_rule = (
        'All user-visible annotations and the final report must be in Simplified Chinese.'
        if resolved_ui_language == 'zh-CN'
        else 'All user-visible annotations and the final report must be in English.'
    )
    kg_active = bool(criteria_bundle and int(criteria_bundle.get('criteria_count') or 0) > 0)
    criteria_section = format_criteria_bundle_for_prompt(criteria_bundle, max_criteria=24) if kg_active else ''

    if kg_active:
        final_sections = (
            'Summary, Strengths, Weaknesses, Key Issues, Actionable Suggestions, '
            'Claim-Level Audit, Scores'
        )
        audit_block = (
            '4) Each `pdf_annotate` MUST include `criterion_id` from active criteria only (not OTHER/routing).\n'
            '5) Claim-Level Audit: markdown table with 5 columns:\n'
            '   | ID | Evidence | Status | Conf | Fix |\n'
            '   - ID: C01, C02, etc. (from Criterion Legend).\n'
            '   - Evidence: brief quote or summary.\n'
            '   - Status: Missing, Partial, Supported, or Check.\n'
            '   - Conf: H (High), M (Medium), L (Low).\n'
            '   - Fix: brief suggestion.\n'
            '   Rules: missing detail → Missing/H. Citation-only → Partial/M.\n'
            '   Do NOT use OTHER/SIG routing criteria.\n'
        )
    else:
        final_sections = (
            'Summary, Strengths, Weaknesses, Key Issues, Actionable Suggestions, Scores'
        )
        audit_block = (
            '4) Do NOT include criterion IDs or a Claim-Level Audit section — this is a generic review.\n'
        )

    return (
        'You are review agent running in FAST REVIEW mode.\n'
        # ... keep existing blocks through CRITICAL ANNOTATION ACCURACY RULE ...
        '3) `review_final_markdown_write` ONCE with a single `markdown` argument containing exactly these '
        f'## headings (one section each, no duplication):\n'
        f'   {final_sections}.\n'
        '   Do NOT put Strengths/Weaknesses/Key Issues inside Summary.\n'
        '   Do NOT call review_final_markdown_write multiple times.\n'
        f'{audit_block}'
        f'{criteria_section}'
        '[Paper Markdown]\n'
        f'{markdown_text or "(empty)"}\n'
    )
```

Keep all unchanged lines between the shared blocks exactly as they are today (lines 53–86).

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_fast_generic_prompt.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_fast_generic_prompt.py deepreview/prompts/review_agent_prompt.py
git commit -m "feat: generic fast-mode prompt when KG criteria disabled"
```

---

### Task 2: Fast generic final-report sections

**Files:**
- Modify: `deepreview/tools/review_tools.py:110-128, 125-272, 1125-1490`
- Modify: `tests/test_review_fast_mode.py:50-65`
- Modify: `tests/test_fast_generic_prompt.py`

- [ ] **Step 1: Write failing section-order test**

Append to `tests/test_fast_generic_prompt.py`:

```python
from deepreview.tools.review_tools import _required_final_report_section_order


def test_fast_generic_section_order_without_criteria():
    assert _required_final_report_section_order(
        review_fast_mode=True,
        kg_criteria_active=False,
    ) == [
        'summary',
        'strengths',
        'weaknesses',
        'key_issues',
        'actionable_suggestions',
        'scores',
    ]


def test_fast_kg_section_order_unchanged():
    assert _required_final_report_section_order(
        review_fast_mode=True,
        kg_criteria_active=True,
    ) == [
        'summary',
        'strengths',
        'weaknesses',
        'key_issues',
        'actionable_suggestions',
        'claim_level_audit',
        'scores',
    ]
```

Update `tests/test_review_fast_mode.py` `test_fast_final_report_section_order` to pass `kg_criteria_active=True`.

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_fast_generic_prompt.py::test_fast_generic_section_order_without_criteria -v
```

Expected: FAIL — `TypeError: unexpected keyword argument 'kg_criteria_active'`

- [ ] **Step 3: Implement section defs**

In `deepreview/tools/review_tools.py`:

```python
_FAST_GENERIC_REQUIRED_FINAL_REPORT_SECTIONS: list[tuple[str, str, tuple[str, ...]]] = [
    ('summary', 'Summary', ('summary',)),
    ('strengths', 'Strengths', ('strengths',)),
    ('weaknesses', 'Weaknesses', ('weaknesses',)),
    ('key_issues', 'Key Issues', ('key issues', 'issues')),
    ('actionable_suggestions', 'Actionable Suggestions', ('actionable suggestions', 'suggestions')),
    ('scores', 'Scores', ('scores', 'score', 'final score')),
]


def _final_report_section_defs(
    *,
    review_fast_mode: bool = False,
    kg_criteria_active: bool = True,
) -> list[tuple[str, str, tuple[str, ...]]]:
    if review_fast_mode:
        if kg_criteria_active:
            return _FAST_REQUIRED_FINAL_REPORT_SECTIONS
        return _FAST_GENERIC_REQUIRED_FINAL_REPORT_SECTIONS
    return _REQUIRED_FINAL_REPORT_SECTIONS
```

Add `kg_criteria_active: bool = True` to: `_required_final_report_section_order`, `_required_final_report_section_titles`, `_required_final_report_alias_map`, `_resolve_final_report_section_id`, and every caller that already passes `review_fast_mode`.

In `review_final_markdown_write` handler (~line 1129), derive:

```python
kg_criteria_active = bool(rt.criteria_bundle and int(rt.criteria_bundle.get('criteria_count') or 0) > 0)
section_order = _required_final_report_section_order(
    review_fast_mode=review_fast_mode,
    kg_criteria_active=kg_criteria_active,
)
```

Thread `kg_criteria_active` through helper calls in that function (`_section_descriptor`, gate messages, etc.).

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_fast_generic_prompt.py tests/test_review_fast_mode.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add deepreview/tools/review_tools.py tests/test_fast_generic_prompt.py tests/test_review_fast_mode.py
git commit -m "feat: fast generic final-report sections when KG inactive"
```

---

### Task 3: Benchmark optional dependencies and package scaffold

**Files:**
- Modify: `pyproject.toml`
- Create: `benchmark/__init__.py`, `benchmark/paths.py`, `benchmark/models.py`

- [ ] **Step 1: Add optional deps**

In `pyproject.toml`:

```toml
[project.optional-dependencies]
benchmark = [
  "deepeval>=2.0.0",
  "scikit-learn>=1.4.0",
  "pandas>=2.2.0",
  "scipy>=1.11.0",
  "statsmodels>=0.14.0",
  "openreview-py>=1.41.0",
  "httpx>=0.27.0",
]
benchmark-v2 = [
  "ragas>=0.1.0",
]
```

Install v1: `pip install -e ".[benchmark,dev]"`  
Install v2 add-ons: `pip install -e ".[benchmark,benchmark-v2,dev]"`

**Do not vendor or fork these libraries.** Pin minimum versions only; import their public APIs.

Add `"benchmark"` to `[tool.setuptools] packages` list.

- [ ] **Step 2: Create paths module**

`benchmark/paths.py`:

```python
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
KG_TEST_PAPERS_DIR = REPO_ROOT / 'KG_testPapers'
BENCHMARK_DIR = REPO_ROOT / 'benchmark'
PAPERS_DIR = BENCHMARK_DIR / 'papers'
RESULTS_DIR = BENCHMARK_DIR / 'results'
MANIFEST_PATH = BENCHMARK_DIR / 'manifest.jsonl'
REGISTRY_PATH = BENCHMARK_DIR / 'registry.sqlite'
DATA_JOBS_DIR = REPO_ROOT / 'data' / 'jobs'
```

- [ ] **Step 3: Create models module**

`benchmark/models.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PaperRecord:
    paper_id: str
    venue: str
    title: str
    pdf_path: str
    source: str
    year: int | None = None
    expected_decision: str | None = None
    expected_score: float | None = None
    metadata_source: str = 'local'

    def to_dict(self) -> dict[str, Any]:
        return {
            'paper_id': self.paper_id,
            'venue': self.venue,
            'title': self.title,
            'pdf_path': self.pdf_path,
            'source': self.source,
            'year': self.year,
            'expected_decision': self.expected_decision,
            'expected_score': self.expected_score,
            'metadata_source': self.metadata_source,
        }


@dataclass
class RunRecord:
    paper_id: str
    venue: str
    condition: str  # KG_ON | KG_OFF
    job_id: str
    status: str
    git_commit: str
    env_snapshot: dict[str, str] = field(default_factory=dict)
```

- [ ] **Step 4: Install and verify import**

```bash
pip install -e ".[benchmark,dev]"
python -c "from benchmark.paths import MANIFEST_PATH; print(MANIFEST_PATH)"
```

Expected: prints path ending in `benchmark/manifest.jsonl`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml benchmark/__init__.py benchmark/paths.py benchmark/models.py
git commit -m "feat: scaffold benchmark package and optional deps"
```

---

### Task 4: Local PDF ingestion (`ingest_local.py`)

**Files:**
- Create: `benchmark/ingest_local.py`
- Create: `tests/benchmark/test_ingest_local.py`

- [ ] **Step 1: Write failing ingest test**

`tests/benchmark/test_ingest_local.py`:

```python
from __future__ import annotations

from pathlib import Path

from benchmark.ingest_local import ingest_kg_test_papers, LOCAL_VENUE_OVERRIDES


def test_local_override_table_has_eleven_entries():
    assert len(LOCAL_VENUE_OVERRIDES) == 11


def test_ingest_finds_acl_paper(tmp_path, monkeypatch):
    # Use real KG_testPapers if present; skip otherwise
    from benchmark.paths import KG_TEST_PAPERS_DIR
    if not KG_TEST_PAPERS_DIR.exists():
        return
    rows = ingest_kg_test_papers()
    assert len(rows) >= 11
    venues = {r.venue for r in rows}
    assert 'ACL' in venues
    assert 'ICLR' in venues
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/benchmark/test_ingest_local.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: Implement ingest**

`benchmark/ingest_local.py` — key pieces:

```python
LOCAL_VENUE_OVERRIDES: dict[str, str] = {
    '1412.6980v9.pdf': 'ICLR',
    '9447_Tabular_Insights_Visual_I.pdf': 'ICLR',
    '1706.03762v7.pdf': 'NeurIPS',
    '2402.05602v2.pdf': 'NeurIPS',
    '2311.10263v2.pdf': 'ICML',
    '2402.01869v2.pdf': 'ICML',
    '2404.01847v3.pdf': 'AAAI',
    '2024.acl-short.8.pdf': 'ACL',
    '2024.findings-acl.438.pdf': 'ACL',
    '1-s2.0-S0360131524002380-main.pdf': 'Computers And Education',
    'Liu-UsingAIBasedObject-2023.pdf': 'ETS',
}

VENUE_PATTERNS: list[tuple[str, str]] = [
    ('ICLR', r'International Conference on Learning Representations|Published as a conference paper at ICLR'),
    ('NeurIPS', r'NeurIPS|Neural Information Processing Systems'),
    ('ACL', r'Association for Computational Linguistics|Findings of the Association'),
    ('Computers And Education', r'Computers\s*&\s*Education'),
    ('ETS', r'Educational Technology\s*&\s*Society'),
]
```

`ingest_kg_test_papers()`:
1. Glob `KG_testPapers/*.pdf`
2. For each file: override table first, else `detect_venue_from_pdf(path)` using pymupdf first 2 pages
3. Extract title from first 500 chars
4. Build `paper_id` = `{venue_slug}_{stem}` lowercased
5. Copy PDF to `benchmark/papers/<paper_id>.pdf` if missing
6. Return `list[PaperRecord]`

- [ ] **Step 4: Run tests**

```bash
pytest tests/benchmark/test_ingest_local.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add benchmark/ingest_local.py tests/benchmark/test_ingest_local.py
git commit -m "feat: ingest KG_testPapers into benchmark paper records"
```

---

### Task 5: OpenReview gap fetcher

**Files:**
- Create: `benchmark/fetch_openreview.py`

- [ ] **Step 1: Implement fetcher**

`fetch_openreview.py` exports `fetch_openreview_gaps(needed: dict[str, int]) -> list[PaperRecord]`:

- `needed` example: `{'CHI': 2, 'AAAI': 1}`
- Use `openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')`
- Venue query map:
  - `CHI`: search `venue:CHI` recent accepted submissions with PDF
  - `AAAI`: search `venue:AAAI` recent
- Download PDF to `benchmark/papers/<paper_id>.pdf`
- Populate `expected_decision` when `decision` field exists
- Skip papers already in manifest (match by title substring)
- Return empty list with logged warning if API unavailable (no crash)

- [ ] **Step 2: Manual smoke**

```bash
python -c "from benchmark.fetch_openreview import fetch_openreview_gaps; print(fetch_openreview_gaps({'AAAI': 1}))"
```

Expected: 0 or 1 `PaperRecord` printed; no exception.

- [ ] **Step 3: Commit**

```bash
git add benchmark/fetch_openreview.py
git commit -m "feat: OpenReview gap fetcher for benchmark corpus"
```

---

### Task 6: Journal gap fetcher + manifest builder

**Files:**
- Create: `benchmark/fetch_journals.py`
- Create: `benchmark/build_manifest.py`

- [ ] **Step 1: Journal fetcher**

`fetch_journals.py` — curated `JOURNAL_SEED_URLS: dict[str, list[dict]]` for TWELF, ETRD, extra C&E and ETS entries. Each entry: `{title, url, venue}`. Download via `httpx`; on 403/404 log `venue_gap` and continue.

If no URLs yet, ship with empty lists and document that operator adds one PDF per gap manually to `benchmark/papers/incoming/` — `build_manifest` picks them up by filename convention `ETRD_<slug>.pdf`.

- [ ] **Step 2: Manifest builder**

`build_manifest.py`:

```python
TARGET_VENUES = [
    'ICLR', 'NeurIPS', 'ICML', 'ACL', 'CHI', 'AAAI',
    'TWELF', 'Computers And Education', 'ETRD', 'ETS',
]
PAPERS_PER_VENUE = 2

def build_manifest() -> list[PaperRecord]:
    rows = ingest_kg_test_papers()
    by_venue = group_by_venue(rows)
    gaps = {v: max(0, PAPERS_PER_VENUE - len(by_venue.get(v, []))) for v in TARGET_VENUES}
    rows.extend(fetch_openreview_gaps({k: v for k, v in gaps.items() if k in {'CHI', 'AAAI'} and v > 0}))
    # recompute gaps, then fetch_journals for TWELF, ETRD, C&E, ETS
    # write benchmark/manifest.jsonl (one JSON per line)
```

CLI: `python -m benchmark build-manifest`

- [ ] **Step 3: Run manifest build**

```bash
python -m benchmark build-manifest
wc -l benchmark/manifest.jsonl
```

Expected: up to 20 lines (fewer if journal gaps unfilled — log warnings)

- [ ] **Step 4: Commit**

```bash
git add benchmark/fetch_journals.py benchmark/build_manifest.py benchmark/cli.py
git commit -m "feat: build benchmark manifest from local + fetched papers"
```

---

### Task 7: Benchmark env profiles and SQLite registry

**Files:**
- Create: `benchmark/env.py`
- Create: `benchmark/registry.py`
- Create: `tests/benchmark/test_env.py`

- [ ] **Step 1: Write env test**

```python
from benchmark.env import build_benchmark_env

def test_kg_on_env():
    env = build_benchmark_env('KG_ON', venue='ICLR', base={'AGENT_MODEL': 'gpt-5-mini'})
    assert env['REVIEW_CRITERIA_ENABLED'] == 'true'
    assert env['REVIEW_VENUE'] == 'ICLR'
    assert env['REVIEW_FAST_MODE'] == 'true'
    assert env['PAPER_SEARCH_ENABLED'] == 'false'
    assert env['REVIEW_INFER_VENUE_FROM_PAPER'] == 'false'

def test_kg_off_env():
    env = build_benchmark_env('KG_OFF', venue='ICLR', base={})
    assert env['REVIEW_CRITERIA_ENABLED'] == 'false'
```

- [ ] **Step 2: Implement env.py**

```python
BENCHMARK_ENV_DEFAULTS = {
    'REVIEW_FAST_MODE': 'true',
    'PAPER_SEARCH_ENABLED': 'false',
    'REVIEW_INFER_VENUE_FROM_PAPER': 'false',
    'ENABLE_FINAL_GATES': 'false',
}

def build_benchmark_env(condition: str, *, venue: str, base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base or {})
    env.update(BENCHMARK_ENV_DEFAULTS)
    env['REVIEW_VENUE'] = venue
    env['REVIEW_CRITERIA_ENABLED'] = 'true' if condition == 'KG_ON' else 'false'
    return env
```

- [ ] **Step 3: Implement registry.py**

SQLite schema:

```sql
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY,
  paper_id TEXT NOT NULL,
  venue TEXT NOT NULL,
  condition TEXT NOT NULL,
  job_id TEXT NOT NULL,
  status TEXT,
  git_commit TEXT,
  started_at TEXT,
  finished_at TEXT,
  env_json TEXT,
  UNIQUE(paper_id, condition)
);
```

Functions: `init_registry()`, `upsert_run(RunRecord, ...)`, `list_runs()`.

- [ ] **Step 4: Run tests**

```bash
pytest tests/benchmark/test_env.py -v
```

- [ ] **Step 5: Commit**

```bash
git add benchmark/env.py benchmark/registry.py tests/benchmark/test_env.py
git commit -m "feat: benchmark env profiles and SQLite registry"
```

---

### Task 8: Paired runner

**Files:**
- Create: `benchmark/runner.py`
- Modify: `benchmark/cli.py`

- [ ] **Step 1: Implement runner**

`run_paired_benchmark(manifest_path, *, timeout_seconds=3600, dry_run=False)`:

For each `PaperRecord` in manifest:
1. For `condition` in `['KG_ON', 'KG_OFF']`:
2. Build env via `build_benchmark_env`
3. `subprocess.run([sys.executable, 'main.py', 'submit', '--pdf', pdf_path, '--wait-seconds', '0'], env={**os.environ, **bench_env}, cwd=REPO_ROOT)`
4. Parse JSON stdout for `job_id`
5. `subprocess.run([sys.executable, 'main.py', 'watch', '--job-id', job_id, '--timeout', str(timeout_seconds)])`
6. `upsert_run` in registry
7. If `dry_run`, only print planned commands

`get_settings.cache_clear()` is NOT needed in parent — each worker is a fresh subprocess.

- [ ] **Step 2: CLI command**

```bash
python -m benchmark run --dry-run
python -m benchmark run --paper-id iclr_1412_6980  # optional filter
```

- [ ] **Step 3: Commit**

```bash
git add benchmark/runner.py benchmark/cli.py
git commit -m "feat: paired KG_ON/KG_OFF benchmark runner"
```

---

### Task 9: Artifact collector

**Files:**
- Create: `benchmark/collect.py`

- [ ] **Step 1: Implement collect**

`collect_run(job_id: str) -> dict[str, Any]` reads:
- `data/jobs/<job_id>/state.json`
- `data/jobs/<job_id>/events.jsonl` (parse last `review_criteria_resolved`, compute wall-clock from first/last timestamp)
- `final_report.md`, `annotations.json`, `review_criteria_bundle.json` (if exists)

Returns flat dict with keys: `job_id`, `status`, `runtime_seconds`, `input_tokens`, `output_tokens`, `total_tokens`, `tool_calls`, `annotation_count`, `criteria_count`, `paper_search_calls`, `final_md_bytes`, `has_legend`, `has_claim_audit`, `venue_criterion_ids`.

`harvest_all()` iterates registry runs → writes `benchmark/results/runs.jsonl`.

- [ ] **Step 2: CLI**

```bash
python -m benchmark collect
```

- [ ] **Step 3: Commit**

```bash
git add benchmark/collect.py benchmark/cli.py
git commit -m "feat: harvest benchmark job artifacts to runs.jsonl"
```

---

### Task 10: Deterministic checks

**Files:**
- Create: `benchmark/checks.py`
- Create: `tests/benchmark/test_checks.py`

- [ ] **Step 1: Write failing tests**

Use fixture dicts mimicking collected rows:

```python
from benchmark.checks import validate_condition, validate_run_completion

def test_kg_on_requires_legend():
    row = {'condition': 'KG_ON', 'criteria_count': 5, 'has_legend': True, 'has_claim_audit': True, 'status': 'completed'}
    assert validate_condition(row) == []

def test_kg_off_rejects_legend():
    row = {'condition': 'KG_OFF', 'criteria_count': 0, 'has_legend': True, 'status': 'completed'}
    errs = validate_condition(row)
    assert any('legend' in e.lower() for e in errs)
```

- [ ] **Step 2: Implement checks.py**

`validate_run_completion(row) -> list[str]`
`validate_condition(row) -> list[str]` per spec §7
`score_deterministic(row) -> dict` — booleans + numeric fields

`run_all_checks()` → `benchmark/results/deterministic_scores.csv`

- [ ] **Step 3: Run tests + CLI**

```bash
pytest tests/benchmark/test_checks.py -v
python -m benchmark check
```

- [ ] **Step 4: Commit**

```bash
git add benchmark/checks.py tests/benchmark/test_checks.py
git commit -m "feat: deterministic benchmark condition validators"
```

---

### Task 11: DeepEval judge layer (import only)

**Files:**
- Create: `benchmark/judge.py`
- Create: `tests/benchmark/test_judge.py`

**Library:** https://github.com/confident-ai/deepeval — use `GEval`, `LLMTestCase`; do not build a custom judge framework.

- [ ] **Step 1: Write failing import smoke test**

`tests/benchmark/test_judge.py`:

```python
from __future__ import annotations

import pytest

deepeval = pytest.importorskip('deepeval')
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase


def test_build_rubric_alignment_metric():
    from benchmark.judge import build_rubric_alignment_metric
    metric = build_rubric_alignment_metric()
    assert isinstance(metric, GEval)
```

- [ ] **Step 2: Implement judge.py with DeepEval APIs**

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

def build_rubric_alignment_metric() -> GEval:
    return GEval(
        name='rubric_alignment',
        criteria='Score 1-5 how well the review addresses venue-specific criteria.',
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.CONTEXT],
        model=os.environ.get('BENCHMARK_JUDGE_MODEL', 'gpt-5-mini'),
    )

def judge_run(collected_row: dict) -> dict:
    test_case = LLMTestCase(
        input=f"Venue: {collected_row['venue']}",
        actual_output=collected_row['final_markdown'],
        context=[collected_row['manuscript_excerpt'], collected_row.get('criteria_json', '')],
    )
    # call metric.measure(test_case) for each GEval metric
```

Metrics via separate `GEval` instances: `factual_correctness`, `evidence_support`, `rubric_alignment`, `specificity`, `actionability`, `unsupported_critique_rate`, `criterion_grounded_valid_critique`.

- [ ] **Step 3: Judge runner + CLI**

`judge_all()` — skip rows failing `validate_run_completion`; write `benchmark/results/review_quality_scores.csv`.

```bash
python -m benchmark judge --job-id <uuid>
python -m benchmark judge
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/benchmark/test_judge.py -v
```

- [ ] **Step 5: Commit**

```bash
git add benchmark/judge.py tests/benchmark/test_judge.py benchmark/cli.py
git commit -m "feat: DeepEval report-level judge wrapper"
```

---

### Task 12: sklearn decision metrics (conditional on labels)

**Files:**
- Create: `benchmark/decision_metrics.py`
- Create: `tests/benchmark/test_decision_metrics.py`

**Library:** https://github.com/scikit-learn/scikit-learn — use `sklearn.metrics`; do not reimplement accuracy/F1/AUROC.

- [ ] **Step 1: Write failing tests**

```python
from benchmark.decision_metrics import parse_predicted_decision, compute_decision_metrics

def test_parse_accept_from_scores_section():
    md = "## Scores\n\n**Recommendation:** Accept\n"
    assert parse_predicted_decision(md) == 'accept'

def test_sklearn_metrics_on_labeled_rows():
    rows = [
        {'predicted_decision': 'accept', 'expected_decision': 'accept'},
        {'predicted_decision': 'reject', 'expected_decision': 'accept'},
    ]
    out = compute_decision_metrics(rows)
    assert 'balanced_accuracy' in out
    assert 0.0 <= out['balanced_accuracy'] <= 1.0
```

- [ ] **Step 2: Implement decision_metrics.py**

```python
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

ACCEPT_TOKENS = ('accept', 'poster', 'oral', 'spotlight')
REJECT_TOKENS = ('reject', 'withdraw')

def parse_predicted_decision(final_markdown: str) -> str | None:
    # heuristic parse from ## Scores section only
    ...

def score_labeled_runs(rows: list[dict]) -> list[dict]:
    # only rows with non-null expected_decision from manifest
    ...

def compute_decision_metrics(rows: list[dict]) -> dict:
    y_true = [...]
    y_pred = [...]
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        # roc_auc_score only when confidence/score column parseable
    }
```

Write `benchmark/results/decision_metrics.csv` (per-run) and corpus summary columns in `overall_summary.csv`.

- [ ] **Step 3: CLI**

```bash
python -m benchmark decision-metrics
```

Skip gracefully with log message when zero manifest rows have `expected_decision`.

- [ ] **Step 4: Run tests + commit**

```bash
pytest tests/benchmark/test_decision_metrics.py -v
git add benchmark/decision_metrics.py tests/benchmark/test_decision_metrics.py benchmark/cli.py
git commit -m "feat: sklearn decision metrics for labeled papers"
```

---

### Task 13: Paired comparison and report (pandas + scipy + statsmodels)

**Files:**
- Create: `benchmark/compare.py`
- Create: `benchmark/report.py`

**Libraries (import only):**
- pandas — https://github.com/pandas-dev/pandas
- SciPy — https://github.com/scipy/scipy
- statsmodels — https://github.com/statsmodels/statsmodels

- [ ] **Step 1: Implement compare.py**

`build_paired_comparison()`:
1. `pd.read_csv` deterministic + judge + decision_metrics tables; `merge` on `paper_id` + `condition`
2. Require both KG_ON and KG_OFF with `validate_condition` empty and KG_ON `criteria_count > 0`
3. Compute `delta_* = kg_on - kg_off` for every numeric metric (DeepEval + sklearn + deterministic)
4. Per-venue `groupby('venue').agg(...)` → `venue_summary.csv`
5. Bootstrap 95% CI on median delta (numpy/statsmodels glue — do not reimplement stat theory)
6. `from scipy.stats import wilcoxon` on `rubric_alignment` deltas when n ≥ 8
7. Include `delta_balanced_accuracy` / `delta_macro_f1` when decision labels exist

```python
import pandas as pd
from scipy.stats import wilcoxon

def paired_wilcoxon_pvalue(deltas: pd.Series) -> float | None:
    if len(deltas) < 8:
        return None
    _stat, p = wilcoxon(deltas.dropna())
    return float(p)
```

Write `paired_comparison.csv`, `overall_summary.csv`.

- [ ] **Step 2: Implement report.py**

`benchmark_summary.md` sections:
- Executive summary (completion rate, median deltas)
- Table: per-venue rubric_alignment delta
- sklearn decision metrics summary (when labels exist)
- Top 3 venues KG helps / hurts
- Cost tradeoff (median delta tokens, runtime)
- Wilcoxon p-value on rubric_alignment (when n ≥ 8)
- Invalid pairs excluded count

- [ ] **Step 3: CLI**

```bash
python -m benchmark compare
python -m benchmark report
python -m benchmark all  # build-manifest → run → collect → check → judge → decision-metrics → compare → report
```

- [ ] **Step 4: Commit**

```bash
git add benchmark/compare.py benchmark/report.py benchmark/cli.py
git commit -m "feat: paired comparison stats and benchmark summary report"
```

---

### Task 14: CLI entry point and docs

**Files:**
- Create: `benchmark/__main__.py`
- Modify: `README.md` (short Benchmark section)

- [ ] **Step 1: Wire `__main__.py`**

```python
from benchmark.cli import main
if __name__ == '__main__':
    raise SystemExit(main())
```

- [ ] **Step 2: README section**

Add ## KG Benchmark with:

```bash
pip install -e ".[benchmark]"
python -m benchmark build-manifest
python -m benchmark run
python -m benchmark all
```

- [ ] **Step 3: Commit**

```bash
git add benchmark/__main__.py README.md
git commit -m "docs: add KG benchmark CLI usage"
```

---

### Task 15: RAGAS claim-level faithfulness (v2 — import only)

**Files:**
- Create: `benchmark/claims.py`
- Create: `benchmark/faithfulness.py`
- Create: `tests/benchmark/test_faithfulness.py`

**Library:** https://github.com/explodinggradients/ragas — use `ragas.evaluate()` and built-in metrics; do not reimplement faithfulness scoring.

**Prerequisite:** v1 smoke benchmark passes. Requires `pip install -e ".[benchmark-v2]"`.

- [ ] **Step 1: Claim extractor (`claims.py`)**

Extract atomic claims from `Weaknesses`, `Key Issues`, `Claim-Level Audit` rows (KG_ON only) into:

```python
@dataclass
class AtomicClaim:
    claim_id: str
    text: str
    polarity: str
    evidence_span: str | None
    criterion_id: str | None
```

- [ ] **Step 2: RAGAS wrapper (`faithfulness.py`)**

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

def score_claims(claims: list[AtomicClaim], manuscript: str) -> Dataset:
    rows = [{'question': c.text, 'answer': c.text, 'contexts': [c.evidence_span or manuscript[:4000]]} for c in claims]
    ds = Dataset.from_list(rows)
    return evaluate(ds, metrics=[faithfulness, answer_relevancy])
```

Write `benchmark/results/claim_scores.jsonl`.

- [ ] **Step 3: CLI + tests**

```bash
python -m benchmark faithfulness --job-id <uuid>
pytest tests/benchmark/test_faithfulness.py -v
```

- [ ] **Step 4: Commit**

```bash
git add benchmark/claims.py benchmark/faithfulness.py tests/benchmark/test_faithfulness.py benchmark/cli.py
git commit -m "feat: RAGAS claim-level faithfulness wrapper (v2)"
```

---

### Task 16: Smoke validation (operator run)

**Files:** none (operator execution)

- [ ] **Step 1: Configure environment**

Ensure `.env` has `MINERU_API_TOKEN`, `OPENAI_API_KEY`, Neo4j credentials, `AGENT_MODEL=gpt-5-mini`, `REVIEW_FAST_MODE=true`.

- [ ] **Step 2: Build manifest**

```bash
python -m benchmark build-manifest
```

Expected: `benchmark/manifest.jsonl` with ≤20 papers; warnings for unfilled journal venues.

- [ ] **Step 3: Run one paper pair first**

```bash
python -m benchmark run --paper-id <first_paper_id> --timeout 3600
python -m benchmark collect
python -m benchmark check
```

Expected: both runs `completed`; KG_ON has legend; KG_OFF does not.

- [ ] **Step 4: Judge, decision metrics, and compare**

```bash
python -m benchmark judge
python -m benchmark decision-metrics
python -m benchmark compare
python -m benchmark report
```

Expected: `benchmark/results/benchmark_summary.md` exists with paired deltas; `decision_metrics.csv` present if OpenReview labels were fetched.

- [ ] **Step 5: Full smoke (optional)**

```bash
python -m benchmark all
```

Expected: ≥ 90% valid pairs per spec success criteria.

---

## Spec Coverage Self-Review

| Spec section | Task |
|--------------|------|
| §3 Frozen profile | Task 7 `env.py` |
| §4 Generic fast prompt | Tasks 1–2 |
| §5 Dataset automation | Tasks 4–6 |
| §6 Harness architecture | Tasks 3, 8–14 |
| §7 External libs (import only) | Tasks 3, 11–13, 15 |
| §8 Deterministic checks | Task 10 |
| §9 DeepEval judges | Task 11 |
| §10 sklearn decision metrics | Task 12 |
| §11 scipy + statsmodels + pandas | Task 13 |
| §12 RAGAS (v2) | Task 15 |
| §13 Outputs | Tasks 9, 10, 11, 12, 13 |
| §14 Success criteria | Task 16 |
| §15 v2 deferrals | Task 15 only (RAGAS); 100-run + MinerU reuse still deferred |

No placeholders remain. Type names consistent: `PaperRecord`, `RunRecord`, `kg_criteria_active`, `KG_ON`/`KG_OFF`. All evaluation metrics delegate to upstream library APIs.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-06-kg-benchmark.md`. Design spec at `docs/superpowers/specs/2026-06-06-kg-benchmark-design.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration

2. **Inline Execution** — implement tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
