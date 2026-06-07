# Benchmark Eval v2 Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix judge ceiling and RAGAS faithfulness bugs; re-score all 18 completed runs with OpenAI per-metric judge + pairwise comparison + Gemma RAGAS using real manuscript evidence spans.

**Architecture:** Split `BENCHMARK_JUDGE_PROVIDER` (OpenAI) from `BENCHMARK_FAITHFULNESS_PROVIDER` (Gemma). New `evidence_resolve.py` resolves critique context via annotations → section slices → quotes. Judge defaults to `per_metric` with anchored rubrics. New `pairwise_judge.py` compares KG_ON vs KG_OFF per paper. Compare/report add pairwise headline.

**Tech Stack:** Python 3.11, DeepEval GEval, RAGAS, `langchain-google-genai`, pytest.

**Spec:** `docs/superpowers/specs/2026-06-08-benchmark-eval-fix-design.md`

**Worktree:** Implement on branch `feat/benchmark-gemma-ragas-rerun` in `.worktrees/gemma-ragas-rerun` (or merge branch to main first).

---

## File map

| File | Responsibility |
|------|----------------|
| `benchmark/eval_llm.py` | Split `judge_provider` / `faithfulness_provider` factories |
| `benchmark/evidence_resolve.py` | **New** — manuscript span resolution |
| `benchmark/claims.py` | Use `resolve_evidence_span`; extend `AtomicClaim` |
| `benchmark/judge.py` | `per_metric` default, anchors, truncation defaults, judge model |
| `benchmark/pairwise_judge.py` | **New** — KG_ON vs KG_OFF JSON judge |
| `benchmark/faithfulness.py` | Faithfulness-only LLM; v2 claim fields |
| `benchmark/compare.py` | Merge pairwise columns |
| `benchmark/report.py` | Pairwise KG win rate headline |
| `benchmark/cli.py` | `pairwise-judge` subcommand; eval pipeline step |
| `benchmark/paths.py` | `PAIRWISE_JUDGE_SCORES_PATH` |
| `.env.example` | Split provider vars |
| `docs/operator-benchmark-rerun.md` | v2 eval section |
| `tests/fixtures/benchmark/` | ACL short manuscript + annotations snippets |
| `tests/benchmark/test_*.py` | TDD for each module |

---

### Task 1: Split eval providers

**Files:**
- Modify: `benchmark/eval_llm.py`
- Modify: `tests/benchmark/test_eval_llm.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/benchmark/test_eval_llm.py — append

def test_judge_provider_defaults_openai(monkeypatch):
    monkeypatch.delenv('BENCHMARK_EVAL_PROVIDER', raising=False)
    monkeypatch.delenv('BENCHMARK_JUDGE_PROVIDER', raising=False)
    from benchmark.eval_llm import judge_provider
    assert judge_provider() == 'openai'


def test_faithfulness_provider_defaults_google(monkeypatch):
    monkeypatch.delenv('BENCHMARK_EVAL_PROVIDER', raising=False)
    monkeypatch.delenv('BENCHMARK_FAITHFULNESS_PROVIDER', raising=False)
    from benchmark.eval_llm import faithfulness_provider
    assert faithfulness_provider() == 'google'


def test_legacy_eval_provider_fallback(monkeypatch):
    monkeypatch.setenv('BENCHMARK_EVAL_PROVIDER', 'openai')
    monkeypatch.delenv('BENCHMARK_JUDGE_PROVIDER', raising=False)
    monkeypatch.delenv('BENCHMARK_FAITHFULNESS_PROVIDER', raising=False)
    from benchmark.eval_llm import judge_provider, faithfulness_provider
    assert judge_provider() == 'openai'
    assert faithfulness_provider() == 'openai'


def test_build_judge_model_openai(monkeypatch):
    monkeypatch.setenv('BENCHMARK_JUDGE_PROVIDER', 'openai')
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')
    monkeypatch.setenv('BENCHMARK_JUDGE_MODEL', 'gpt-5-mini')
    from benchmark.eval_llm import build_judge_model
    from deepeval.models import GPTModel
    assert isinstance(build_judge_model(), GPTModel)


def test_faithfulness_model_name(monkeypatch):
    monkeypatch.setenv('BENCHMARK_FAITHFULNESS_MODEL', 'gemma-4-31b-it')
    from benchmark.eval_llm import faithfulness_model_name
    assert faithfulness_model_name() == 'gemma-4-31b-it'
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `pytest tests/benchmark/test_eval_llm.py::test_judge_provider_defaults_openai -v`  
Expected: `ImportError` or `AttributeError`

- [ ] **Step 3: Implement split providers**

Add to `benchmark/eval_llm.py`:

```python
DEFAULT_JUDGE_PROVIDER = 'openai'
DEFAULT_FAITHFULNESS_PROVIDER = 'google'
DEFAULT_FAITHFULNESS_MODEL = 'gemma-4-31b-it'


def _legacy_provider() -> str | None:
    raw = os.environ.get('BENCHMARK_EVAL_PROVIDER', '').strip().lower()
    return raw or None


def judge_provider() -> str:
    explicit = os.environ.get('BENCHMARK_JUDGE_PROVIDER', '').strip().lower()
    if explicit:
        return explicit
    return _legacy_provider() or DEFAULT_JUDGE_PROVIDER


def faithfulness_provider() -> str:
    explicit = os.environ.get('BENCHMARK_FAITHFULNESS_PROVIDER', '').strip().lower()
    if explicit:
        return explicit
    return _legacy_provider() or DEFAULT_FAITHFULNESS_PROVIDER


def judge_model_name() -> str:
    return os.environ.get('BENCHMARK_JUDGE_MODEL', 'gpt-5-mini').strip()


def faithfulness_model_name() -> str:
    return os.environ.get('BENCHMARK_FAITHFULNESS_MODEL', DEFAULT_FAITHFULNESS_MODEL).strip()


def build_judge_model() -> Any:
    provider = judge_provider()
    if provider == 'openai':
        from deepeval.models import GPTModel
        api_key = os.environ.get('OPENAI_API_KEY') or os.environ.get('API_KEY')
        base_url = os.environ.get('OPENAI_BASE_URL') or os.environ.get('BASE_URL') or None
        return GPTModel(model=judge_model_name(), api_key=api_key, base_url=base_url, temperature=0)
    if provider == 'google':
        from deepeval.models import GeminiModel
        api_key = os.environ.get('GOOGLE_API_KEY', '').strip()
        if not api_key:
            raise RuntimeError('GOOGLE_API_KEY required for google judge provider')
        return GeminiModel(model=judge_model_name(), api_key=api_key, temperature=0)
    raise ValueError(f'Unsupported judge provider: {provider}')


def build_faithfulness_llm() -> Any:
    __import__('ragas')
    provider = faithfulness_provider()
    if provider == 'google':
        from langchain_google_genai import ChatGoogleGenerativeAI
        from ragas.llms import LangchainLLMWrapper
        api_key = os.environ.get('GOOGLE_API_KEY', '').strip()
        if not api_key:
            raise RuntimeError('GOOGLE_API_KEY required for google faithfulness provider')
        return LangchainLLMWrapper(
            ChatGoogleGenerativeAI(
                model=faithfulness_model_name(),
                google_api_key=api_key,
                temperature=0,
            )
        )
    if provider == 'openai':
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper
        return LangchainLLMWrapper(ChatOpenAI(model=faithfulness_model_name(), temperature=0))
    raise ValueError(f'Unsupported faithfulness provider: {provider}')
```

Keep `build_deepeval_model()` as alias: `return build_judge_model()` for backward compat. Update `build_ragas_llm()` to call `build_faithfulness_llm()`.

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest tests/benchmark/test_eval_llm.py -v`  
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add benchmark/eval_llm.py tests/benchmark/test_eval_llm.py
git commit -m "feat(benchmark): split judge and faithfulness eval providers"
```

---

### Task 2: Evidence resolution module

**Files:**
- Create: `benchmark/evidence_resolve.py`
- Create: `tests/fixtures/benchmark/acl_short_manuscript.md`
- Create: `tests/fixtures/benchmark/acl_short_annotations.json`
- Create: `tests/benchmark/test_evidence_resolve.py`

- [ ] **Step 1: Add fixtures**

`tests/fixtures/benchmark/acl_short_manuscript.md` — copy first ~80 lines from `data/jobs/d9f100a7-874d-47b6-9c9b-b80d3f177af5/mineru_full.md` (Abstract + Section 2.2 header).

`tests/fixtures/benchmark/acl_short_annotations.json` — one annotation object with `text` containing `230%` and `comment` about statistical reporting.

- [ ] **Step 2: Write failing tests**

```python
# tests/benchmark/test_evidence_resolve.py
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures' / 'benchmark'


@pytest.fixture
def manuscript() -> str:
    return (FIXTURES / 'acl_short_manuscript.md').read_text(encoding='utf-8')


@pytest.fixture
def annotations() -> list[dict]:
    payload = json.loads((FIXTURES / 'acl_short_annotations.json').read_text(encoding='utf-8'))
    return payload['annotations']


def test_section_slice_abstract(manuscript):
    from benchmark.evidence_resolve import resolve_evidence_span

    bullet = (
        'Statistical reporting is weak (evidence: Abstract and Section 2.2/2.3; C06).'
    )
    resolved = resolve_evidence_span(bullet, manuscript=manuscript, annotations=[])
    assert resolved.source == 'section_slice'
    assert '230%' in resolved.text or 'Abstract' in resolved.text
    assert 'Abstract and Section 2.2' not in resolved.text


def test_annotation_match(manuscript, annotations):
    from benchmark.evidence_resolve import resolve_evidence_span

    bullet = (
        'Statistical reporting is weak: quantitative claims lack confidence intervals '
        '(evidence: Abstract; C06).'
    )
    resolved = resolve_evidence_span(
        bullet, manuscript=manuscript, annotations=annotations
    )
    assert resolved.source == 'annotation_match'
    assert '230%' in resolved.text


def test_never_returns_citation_label(manuscript):
    from benchmark.evidence_resolve import resolve_evidence_span

    bullet = 'Gap (evidence: Appendix C; C06).'
    resolved = resolve_evidence_span(bullet, manuscript=manuscript, annotations=[])
    assert resolved.source != 'citation_label'
    assert 'Appendix C; C06' not in resolved.text
```

- [ ] **Step 3: Run tests — expect FAIL**

Run: `pytest tests/benchmark/test_evidence_resolve.py -v`  
Expected: `ModuleNotFoundError`

- [ ] **Step 4: Implement `benchmark/evidence_resolve.py`**

```python
from __future__ import annotations

import os
import re
from dataclasses import dataclass

_STOPWORDS = frozenset({'the', 'a', 'an', 'and', 'or', 'to', 'of', 'in', 'on', 'for', 'is', 'are'})

_SECTION_REF_RE = re.compile(
    r'\b(abstract|section\s+[\d.]+|appendix\s+[a-z0-9]+|figure\s+\d+|table\s+\d+)\b',
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE)
_QUOTE_RE = re.compile(r'["\']([^"\']{20,})["\']')
_CRITERION_TAG_RE = re.compile(r'\bC0?\d\b', re.IGNORECASE)


@dataclass
class ResolvedEvidence:
    text: str
    source: str
    preview: str


def _content_tokens(text: str) -> set[str]:
    return {
        t.lower()
        for t in re.findall(r'[a-zA-Z]{3,}', text)
        if t.lower() not in _STOPWORDS
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _min_jaccard() -> float:
    return float(os.environ.get('BENCHMARK_RAGAS_ANNOTATION_MATCH_MIN_JACCARD', '0.15'))


def _match_annotation(bullet: str, annotations: list[dict]) -> ResolvedEvidence | None:
    bullet_tokens = _content_tokens(bullet)
    bullet_criteria = {m.group(0).upper() for m in _CRITERION_TAG_RE.finditer(bullet)}
    best_score = 0.0
    best_text = ''
    for item in annotations:
        comment = str(item.get('comment') or item.get('summary') or '')
        criterion = str(item.get('criterion_id') or '')
        ann_tokens = _content_tokens(comment)
        score = _jaccard(bullet_tokens, ann_tokens)
        if bullet_criteria and criterion:
            crit_short = criterion.split('_')[-1][:3] if '_' in criterion else ''
            if any(tag in criterion.upper() for tag in bullet_criteria):
                score = max(score, 0.25)
        if score > best_score:
            best_score = score
            best_text = str(item.get('text') or '')
    if best_score >= _min_jaccard() and best_text.strip():
        text = best_text.strip()
        return ResolvedEvidence(text=text[: _max_chars()], source='annotation_match', preview=text[:120])
    return None


def _max_chars() -> int:
    return int(os.environ.get('BENCHMARK_RAGAS_MAX_CONTEXT_CHARS', '2000'))


def _slice_section(manuscript: str, ref: str) -> str:
    ref_lower = ref.lower().strip()
    headings = list(_HEADING_RE.finditer(manuscript))
    start_idx = None
    for match in headings:
        heading = match.group(1).lower()
        if ref_lower in heading or heading.startswith(ref_lower):
            start_idx = match.start()
            break
    if start_idx is None:
        if ref_lower == 'abstract':
            for match in headings:
                if 'abstract' in match.group(1).lower():
                    start_idx = match.start()
                    break
    if start_idx is None:
        return ''
    end_idx = len(manuscript)
    for match in headings:
        if match.start() > start_idx:
            end_idx = match.start()
            break
    return manuscript[start_idx:end_idx].strip()[: _max_chars()]


def _section_slice(bullet: str, manuscript: str) -> ResolvedEvidence | None:
    refs = _SECTION_REF_RE.findall(bullet)
    for ref in refs:
        chunk = _slice_section(manuscript, ref)
        if len(chunk) >= 80:
            return ResolvedEvidence(text=chunk, source='section_slice', preview=chunk[:120])
    return None


def _quoted_in_claim(bullet: str, manuscript: str) -> ResolvedEvidence | None:
    for match in _QUOTE_RE.finditer(bullet):
        quote = match.group(1).strip()
        if quote in manuscript:
            return ResolvedEvidence(
                text=quote[: _max_chars()],
                source='quoted_in_claim',
                preview=quote[:120],
            )
    return None


def resolve_evidence_span(
    bullet: str,
    *,
    manuscript: str,
    annotations: list[dict] | None = None,
    max_chars: int | None = None,
) -> ResolvedEvidence:
    cap = max_chars if max_chars is not None else _max_chars()
    ann = annotations or []

    matched = _match_annotation(bullet, ann)
    if matched:
        matched.text = matched.text[:cap]
        return matched

    sliced = _section_slice(bullet, manuscript)
    if sliced:
        sliced.text = sliced.text[:cap]
        return sliced

    quoted = _quoted_in_claim(bullet, manuscript)
    if quoted:
        quoted.text = quoted.text[:cap]
        return quoted

    prefix = manuscript[:cap]
    return ResolvedEvidence(text=prefix, source='manuscript_prefix', preview=prefix[:120])
```

- [ ] **Step 5: Run tests — expect PASS**

Run: `pytest tests/benchmark/test_evidence_resolve.py -v`  
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add benchmark/evidence_resolve.py tests/benchmark/test_evidence_resolve.py tests/fixtures/benchmark/
git commit -m "feat(benchmark): annotation-aware evidence span resolution"
```

---

### Task 3: Wire claims + faithfulness to resolved evidence

**Files:**
- Modify: `benchmark/claims.py`
- Modify: `benchmark/faithfulness.py`
- Modify: `tests/benchmark/test_claims.py`
- Modify: `tests/benchmark/test_faithfulness.py`

- [ ] **Step 1: Extend AtomicClaim**

```python
# benchmark/claims.py — AtomicClaim fields
@dataclass
class AtomicClaim:
    claim_id: str
    text: str
    section: str
    evidence_span: str | None
    context_source: str = 'legacy'
    resolved_context_preview: str = ''
```

Update `extract_critique_claims` signature:

```python
def extract_critique_claims(
    report_md: str,
    *,
    manuscript: str,
    annotations: list[dict] | None = None,
    max_claims: int | None = None,
) -> list[AtomicClaim]:
```

Replace `_evidence_for_bullet` body with:

```python
from benchmark.evidence_resolve import resolve_evidence_span

resolved = resolve_evidence_span(bullet, manuscript=manuscript, annotations=annotations or [])
return resolved.text  # store in evidence_span; also set context_source on AtomicClaim
```

- [ ] **Step 2: Update failing claim test**

```python
def test_extract_uses_manuscript_not_citation_label():
    from benchmark.claims import extract_critique_claims

    ms = '## Abstract\n\nWe report 230% increase in confidence.\n\n## 1 Introduction\n'
    report = '## Weaknesses\n- Stats weak (evidence: Abstract; C06).\n'
    claims = extract_critique_claims(report, manuscript=ms, max_claims=5)
    assert len(claims) == 1
    assert '230%' in (claims[0].evidence_span or '')
    assert claims[0].context_source in ('section_slice', 'manuscript_prefix', 'annotation_match')
```

Remove assertion `claims[0].evidence_span == 'Table 1'` — replace with resolved span test using manuscript containing table text.

- [ ] **Step 3: Update faithfulness.py**

In `faithfulness_all`, load annotations:

```python
def _load_annotations(job_id: str) -> list[dict]:
    path = DATA_JOBS_DIR / job_id / 'annotations.json'
    if not path.exists():
        return []
    import json
    payload = json.loads(path.read_text(encoding='utf-8'))
    return list(payload.get('annotations') or [])

# in loop:
annotations = _load_annotations(jid)
claims = extract_critique_claims(
    artifacts.get('final_markdown') or '',
    manuscript=manuscript,
    annotations=annotations,
)
```

In `score_claims` claim_rows append:

```python
'context_source': claim.context_source,
'resolved_context_preview': claim.resolved_context_preview,
'eval_version': 'v2',
```

Switch `build_ragas_llm()` → `build_faithfulness_llm()`.

- [ ] **Step 4: Add faithfulness test (mock RAGAS)**

```python
def test_faithfulness_emits_context_source(monkeypatch, tmp_path):
    # mock evaluate() returning [0.5]; assert claim row has context_source != legacy
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/benchmark/test_claims.py tests/benchmark/test_faithfulness.py -v`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git commit -am "feat(benchmark): faithfulness uses resolved manuscript evidence"
```

---

### Task 4: Judge per-metric mode + anchors

**Files:**
- Modify: `benchmark/judge.py`
- Modify: `tests/benchmark/test_judge.py`
- Modify: `tests/benchmark/test_composite_judge.py`

- [ ] **Step 1: Write failing tests**

```python
def test_judge_mode_defaults_per_metric(monkeypatch):
    monkeypatch.delenv('BENCHMARK_JUDGE_MODE', raising=False)
    from benchmark.judge import _judge_mode
    assert _judge_mode() == 'per_metric'


def test_manuscript_max_chars_default_25k(monkeypatch):
    monkeypatch.delenv('BENCHMARK_JUDGE_MAX_MANUSCRIPT_CHARS', raising=False)
    from benchmark.judge import _manuscript_max_chars
    assert _manuscript_max_chars() == 25000


def test_core_metrics_include_anchors():
    from benchmark.judge import build_factual_correctness_metric, JUDGE_SCORE_ANCHORS
    metric = build_factual_correctness_metric()
    assert 'Score 1' in metric.criteria or '1:' in metric.criteria
    assert JUDGE_SCORE_ANCHORS in metric.criteria
```

- [ ] **Step 2: Implement judge changes**

```python
JUDGE_SCORE_ANCHORS = (
    'Anchors: 1=major failures; 3=adequate but generic; 5=excellent and evidence-grounded. '
    'Return only a single number 1-5.'
)

DEFAULT_MANUSCRIPT_MAX_CHARS = 25_000

def _judge_mode() -> str:
    return os.environ.get('BENCHMARK_JUDGE_MODE', 'per_metric').strip().lower()

def _anchored(criteria: str) -> str:
    return f'{criteria} {JUDGE_SCORE_ANCHORS}'

# In each build_*_metric for CORE_JUDGE_METRICS, wrap criteria with _anchored()
# In build_composite_metric / _build_geval_metric:
#   model=build_judge_model()  (not build_deepeval_model)

def judge_run(collected_row):
    ...
    if _judge_mode() == 'composite':
        scores.update(composite_judge_scores(row))
        return scores
    if _judge_mode() == 'per_metric':
        metrics = {k: build_all_metrics()[k] for k in CORE_JUDGE_METRICS}
        for name, metric in metrics.items():
            scores[name] = metric.measure(test_case)
        return scores
    ...
```

Update `_truncate_report` default to `8000`, `_truncate_criteria` default to `4000`.

- [ ] **Step 3: Run tests**

Run: `pytest tests/benchmark/test_judge.py tests/benchmark/test_composite_judge.py -v`  
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git commit -am "feat(benchmark): per-metric anchored judge via OpenAI"
```

---

### Task 5: Pairwise judge module + CLI

**Files:**
- Create: `benchmark/pairwise_judge.py`
- Modify: `benchmark/paths.py`
- Modify: `benchmark/cli.py`
- Create: `tests/benchmark/test_pairwise_judge.py`

- [ ] **Step 1: Add path constant**

```python
# benchmark/paths.py
PAIRWISE_JUDGE_SCORES_PATH = RESULTS_DIR / 'pairwise_judge_scores.csv'
```

- [ ] **Step 2: Write failing tests**

```python
def test_parse_pairwise_response():
    from benchmark.pairwise_judge import parse_pairwise_response
    raw = '{"winner":"KG_ON","rubric_alignment_winner":"KG_ON","confidence":"high","one_line_reason":"More rubric coverage"}'
    parsed = parse_pairwise_response(raw)
    assert parsed['winner'] == 'KG_ON'
    assert parsed['reason'] == 'More rubric coverage'


def test_pairwise_skips_incomplete_pair(monkeypatch, tmp_path):
    # mock runs with only KG_ON for paper X → pairwise_all skips
```

- [ ] **Step 3: Implement `benchmark/pairwise_judge.py`**

Core exports:
- `PAIRWISE_FIELDS`
- `parse_pairwise_response(raw: str) -> dict`
- `pairwise_compare_row(kg_off: dict, kg_on: dict) -> dict` — builds prompt, calls OpenAI via `build_judge_model().generate` or structured client
- `pairwise_all(*, paper_id: str | None = None) -> list[dict]`

Use `judge.load_judge_artifacts` for both jobs. Resumable skip on existing `paper_id` in CSV.

JSON schema keys: `winner`, `rubric_alignment_winner`, `confidence`, `one_line_reason` → map `one_line_reason` to CSV column `reason`.

- [ ] **Step 4: CLI**

```python
# benchmark/cli.py
PIPELINE_EVAL_STEPS = ('collect', 'check', 'judge', 'pairwise-judge', 'faithfulness', 'compare', 'report')

def _cmd_pairwise_judge(args):
    from benchmark.pairwise_judge import PAIRWISE_JUDGE_SCORES_PATH, pairwise_all
    rows = pairwise_all(paper_id=args.paper_id)
    print(f'Wrote {len(rows)} rows to {PAIRWISE_JUDGE_SCORES_PATH}')
    return 0
```

Add `--paper-id` to pairwise parser.

- [ ] **Step 5: Run tests**

Run: `pytest tests/benchmark/test_pairwise_judge.py -v`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add benchmark/pairwise_judge.py benchmark/paths.py benchmark/cli.py tests/benchmark/test_pairwise_judge.py
git commit -m "feat(benchmark): pairwise KG_ON vs KG_OFF judge"
```

---

### Task 6: Compare + report pairwise integration

**Files:**
- Modify: `benchmark/compare.py`
- Modify: `benchmark/report.py`
- Modify: `tests/benchmark/test_compare.py`

- [ ] **Step 1: Write failing test**

```python
def test_paired_comparison_includes_pairwise_columns(tmp_path):
    # write minimal paired_comparison inputs + pairwise_judge_scores.csv
    # assert output has pairwise_winner, pairwise_rubric_winner, pairwise_confidence
```

- [ ] **Step 2: Implement compare merge**

In `build_paired_comparison`, after building base paired rows, left-join `pairwise_judge_scores.csv` on `paper_id`.

Add columns: `pairwise_winner`, `pairwise_rubric_winner`, `pairwise_confidence`.

- [ ] **Step 3: Implement report headline**

In `build_benchmark_summary`, add section:

```markdown
## Pairwise judge (KG_ON vs KG_OFF)

- KG_ON wins: **{kg_on_wins}** / {n_pairs} ({pct}%)
- KG_OFF wins: **{kg_off_wins}**
- Ties: **{ties}**
```

Compute from `paired['pairwise_winner'].value_counts()`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/benchmark/test_compare.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(benchmark): pairwise win rate in compare and report"
```

---

### Task 7: Env, operator docs, archive helper

**Files:**
- Modify: `.env.example`
- Modify: `docs/operator-benchmark-rerun.md`
- Create: `benchmark/archive_results.py` (minimal CLI)

- [ ] **Step 1: Update `.env.example`**

```env
BENCHMARK_JUDGE_PROVIDER=openai
BENCHMARK_FAITHFULNESS_PROVIDER=google
BENCHMARK_JUDGE_MODEL=gpt-5-mini
BENCHMARK_FAITHFULNESS_MODEL=gemma-4-31b-it
BENCHMARK_JUDGE_MODE=per_metric
BENCHMARK_JUDGE_MAX_MANUSCRIPT_CHARS=25000
BENCHMARK_JUDGE_MAX_REPORT_CHARS=8000
BENCHMARK_JUDGE_MAX_CRITERIA_CHARS=4000
BENCHMARK_RAGAS_ANNOTATION_MATCH_MIN_JACCARD=0.15
```

- [ ] **Step 2: Archive helper**

```python
# benchmark/archive_results.py
def archive_eval_results(tag: str = 'pre-v2') -> Path:
    """Copy active result CSVs/JSONL to benchmark/results/archive/{tag}/"""
```

CLI: `python -m benchmark archive-results --tag pre-v2-2026-06-08`

- [ ] **Step 3: Update operator doc v2 section**

Document sanity gate commands from spec §8 and pass criteria.

- [ ] **Step 4: Commit**

```bash
git add .env.example docs/operator-benchmark-rerun.md benchmark/archive_results.py benchmark/cli.py
git commit -m "docs(benchmark): v2 eval env and operator archive protocol"
```

---

### Task 8: Full pytest sweep

- [ ] **Step 1: Run full benchmark tests**

Run: `pytest tests/benchmark/ -v`  
Expected: all PASS (no live API — mocks only)

- [ ] **Step 2: Fix any regressions**

- [ ] **Step 3: Commit if fixes needed**

```bash
git commit -am "test(benchmark): fix regressions for eval v2"
```

---

### Task 9: Operator validation — v2 re-run (manual, live API)

**Files:** none (operator task)

**Pre-flight:**

- [ ] Archive: `python -m benchmark archive-results --tag pre-v2-2026-06-08`
- [ ] Truncate/delete active `review_quality_scores.csv`, `faithfulness_run_scores.csv`, `claim_scores.jsonl`
- [ ] Set `.env` per spec §4

**Sanity gate (ACL short pair):**

```powershell
cd .worktrees/gemma-ragas-rerun
python -m benchmark judge --job-id d9f100a7-874d-47b6-9c9b-b80d3f177af5
python -m benchmark judge --job-id 330c5928-a8d1-4ec1-8efe-ed2081001477
python -m benchmark pairwise-judge --paper-id acl_2024.acl-short.8
python -m benchmark faithfulness --job-id d9f100a7-874d-47b6-9c9b-b80d3f177af5
```

**Pass criteria (spec §8):**

- [ ] Judge scores not all 5.0 on both ACL runs
- [ ] Faithfulness mean > 0.1; `context_source` ≠ citation label in `claim_scores.jsonl`
- [ ] Pairwise row has `winner` + `reason`

**Full v2 eval:**

```powershell
python -m benchmark pipeline --phase eval
```

Or step-by-step: `judge` → `pairwise-judge` → `faithfulness` → `compare` → `report`

**Success criteria (spec §13):**

- [ ] 18 judge rows
- [ ] ≥ 9 pairwise rows
- [ ] Corpus median `faithfulness_mean` > 0.15
- [ ] `benchmark_summary.md` shows pairwise KG win rate
- [ ] < 50% runs with all metrics ≥ 4.5

**Do NOT commit** live API result files unless operator explicitly requests.

---

## Spec coverage checklist

| Spec § | Task |
|--------|------|
| §4 Split providers | Task 1 |
| §5 Per-metric judge + anchors | Task 4 |
| §6 Pairwise judge | Task 5 |
| §7 Evidence resolution | Task 2, 3 |
| §8 Re-run protocol | Task 7, 9 |
| §9 Compare/report | Task 6 |
| §10 Error handling | Tasks 4, 5 (retry in judge_run / pairwise) |
| §11 Testing | All tasks |
| §13 Success criteria | Task 9 |

---

## Cost reminder (18-run v2 re-run)

| Step | Est. cost |
|------|-----------|
| Judge (4×18) | ~$0.18 |
| Pairwise (9 pairs) | ~$0.04 |
| Faithfulness (Gemma) | $0 free tier |
| **Total** | **~$0.22** |
