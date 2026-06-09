from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from benchmark.checks import validate_run_completion
from benchmark.collect import RUNS_JSONL_PATH, collect_run
from benchmark.eval_llm import build_judge_model, eval_model_name, pause_between_eval_calls
from benchmark.paths import DATA_JOBS_DIR, RESULTS_DIR
from benchmark.registry import list_runs

REVIEW_QUALITY_SCORES_PATH = RESULTS_DIR / 'review_quality_scores.csv'
DEFAULT_JUDGE_MODEL = 'gpt-5-mini'
DEFAULT_MANUSCRIPT_MAX_CHARS = 25_000

JUDGE_SCORE_ANCHORS = (
    'Anchors: 1=major failures; 3=adequate but generic; 5=excellent and evidence-grounded. '
    'Return only a single number 1-5.'
)

CORE_JUDGE_METRICS = (
    'factual_correctness',
    'evidence_support',
    'rubric_alignment',
    'criterion_grounded_valid_critique',
)

METRIC_NAMES = CORE_JUDGE_METRICS

COMPOSITE_JUDGE_CRITERIA = """Return ONLY valid JSON with numeric scores 1-5 for:
factual_correctness, evidence_support, rubric_alignment, criterion_grounded_valid_critique.
Score the peer review against the manuscript excerpt and venue criteria in context."""

_EVAL_PARAMS = [LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.CONTEXT]


def _judge_mode() -> str:
    return os.environ.get('BENCHMARK_JUDGE_MODE', 'per_metric').strip().lower()


def _anchored(criteria: str) -> str:
    return f'{criteria} {JUDGE_SCORE_ANCHORS}'


def _manuscript_max_chars() -> int:
    raw = os.environ.get('BENCHMARK_JUDGE_MAX_MANUSCRIPT_CHARS')
    if raw is None:
        return DEFAULT_MANUSCRIPT_MAX_CHARS
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MANUSCRIPT_MAX_CHARS


def _truncate_report(text: str) -> str:
    max_chars = int(os.environ.get('BENCHMARK_JUDGE_MAX_REPORT_CHARS', '8000'))
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + '\n\n...[truncated]...'


def _truncate_criteria(text: str) -> str:
    max_chars = int(os.environ.get('BENCHMARK_JUDGE_MAX_CRITERIA_CHARS', '4000'))
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + '\n\n...[truncated]...'


def parse_composite_judge_response(raw: str) -> dict[str, float]:
    text = raw.strip()
    if '```' in text:
        text = text.split('```', 1)[-1]
        if text.startswith('json'):
            text = text[4:]
        text = text.split('```', 1)[0]
    payload = json.loads(text.strip())
    return {name: float(payload[name]) for name in CORE_JUDGE_METRICS}


def build_composite_metric() -> GEval:
    return GEval(
        name='composite_review_quality',
        criteria=COMPOSITE_JUDGE_CRITERIA,
        evaluation_params=_EVAL_PARAMS,
        model=build_judge_model(),
    )


def composite_judge_scores(row: dict[str, Any]) -> dict[str, float]:
    """Single Gemma call returning JSON with 4 metric scores (1-5)."""
    import time

    from google import genai
    from google.genai import errors as genai_errors

    api_key = os.environ.get('GOOGLE_API_KEY', '').strip()
    if not api_key:
        raise RuntimeError('GOOGLE_API_KEY is required for composite judge')
    client = genai.Client(api_key=api_key)
    venue = str(row.get('venue') or 'unknown')
    manuscript = str(row.get('manuscript_excerpt') or '')
    report = str(row.get('final_markdown') or '')
    criteria = _truncate_criteria(str(row.get('criteria_json') or ''))
    prompt = (
        f'Venue: {venue}\n\n'
        f'Manuscript excerpt:\n{manuscript}\n\n'
        f'Venue criteria:\n{criteria}\n\n'
        f'Peer review:\n{report}\n\n'
        f'{COMPOSITE_JUDGE_CRITERIA}'
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=eval_model_name(),
                contents=prompt,
            )
            return parse_composite_judge_response(response.text or '')
        except genai_errors.ServerError as exc:
            last_error = exc
            time.sleep(5 * (attempt + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError('composite judge failed without response')


def _build_geval_metric(*, name: str, criteria: str) -> GEval:
    return GEval(
        name=name,
        criteria=criteria,
        evaluation_params=_EVAL_PARAMS,
        model=build_judge_model(),
    )


def build_factual_correctness_metric() -> GEval:
    return _build_geval_metric(
        name='factual_correctness',
        criteria=_anchored(
            'Score 1-5 how factually correct the peer review is relative to the manuscript. '
            '1 = major factual errors or misreadings; 5 = claims align with the paper.'
        ),
    )


def build_evidence_support_metric() -> GEval:
    return _build_geval_metric(
        name='evidence_support',
        criteria=_anchored(
            'Score 1-5 how well critiques and strengths cite or reflect manuscript evidence. '
            '1 = mostly unsupported assertions; 5 = critiques are well grounded in the text.'
        ),
    )


def build_rubric_alignment_metric() -> GEval:
    return _build_geval_metric(
        name='rubric_alignment',
        criteria=_anchored(
            'Score 1-5 how well the review addresses venue-specific review criteria when provided. '
            '1 = ignores criteria; 5 = systematically covers the rubric.'
        ),
    )


def build_specificity_metric() -> GEval:
    return _build_geval_metric(
        name='specificity',
        criteria=(
            'Score 1-5 how specific the review is about methods, experiments, and results. '
            '1 = vague boilerplate; 5 = concrete, paper-specific analysis.'
        ),
    )


def build_actionability_metric() -> GEval:
    return _build_geval_metric(
        name='actionability',
        criteria=(
            'Score 1-5 how actionable the revision suggestions are. '
            '1 = no useful guidance; 5 = clear, prioritized fixes the authors can implement.'
        ),
    )


def build_unsupported_critique_rate_metric() -> GEval:
    return _build_geval_metric(
        name='unsupported_critique_rate',
        criteria=(
            'Score 1-5 where higher means FEWER unsupported negative critiques '
            '(lower unsupported critique rate). '
            '1 = many critiques lack manuscript evidence; 5 = critiques are evidence-backed.'
        ),
    )


def build_criterion_grounded_valid_critique_metric() -> GEval:
    return _build_geval_metric(
        name='criterion_grounded_valid_critique',
        criteria=_anchored(
            'Score 1-5 for criterion-grounded, valid critiques when venue criteria are provided. '
            '1 = criteria ignored or invalid critiques; 5 = valid critiques tied to rubric items. '
            'If no criteria are provided, score based on whether critiques would be valid for a venue review.'
        ),
    )


def build_all_metrics() -> dict[str, GEval]:
    return {
        'factual_correctness': build_factual_correctness_metric(),
        'evidence_support': build_evidence_support_metric(),
        'rubric_alignment': build_rubric_alignment_metric(),
        'specificity': build_specificity_metric(),
        'actionability': build_actionability_metric(),
        'unsupported_critique_rate': build_unsupported_critique_rate_metric(),
        'criterion_grounded_valid_critique': build_criterion_grounded_valid_critique_metric(),
    }


def _truncate_manuscript(text: str) -> str:
    max_chars = _manuscript_max_chars()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + '\n\n...[truncated]...'


def _job_dir(job_id: str) -> Path:
    return DATA_JOBS_DIR / job_id


def load_judge_artifacts(job_id: str) -> dict[str, str]:
    from benchmark.review_sources import load_job_review_artifacts

    return load_job_review_artifacts(job_id)


def _registry_row_for_job(job_id: str) -> dict[str, Any]:
    for run in list_runs():
        if run.job_id == job_id:
            return {
                'paper_id': run.paper_id,
                'venue': run.venue,
                'condition': run.condition,
                'git_commit': run.git_commit,
            }
    return {}


def _resolve_collected_row(collected_row: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(collected_row, str):
        job_id = collected_row
        row = collect_run(job_id)
        row.update(_registry_row_for_job(job_id))
        return row
    return dict(collected_row)


def _enrich_row_for_judge(row: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(row)
    job_id = str(enriched.get('job_id') or '').strip()
    if not job_id:
        raise ValueError('collected_row must include job_id')

    from benchmark.review_sources import load_review_artifacts

    artifacts = load_review_artifacts(enriched)
    for key, value in artifacts.items():
        if not str(enriched.get(key) or '').strip():
            enriched[key] = value
    return enriched


def _build_test_case(row: dict[str, Any]) -> LLMTestCase:
    venue = str(row.get('venue') or 'unknown')
    criteria_json = str(row.get('criteria_json') or '')
    manuscript_excerpt = str(row.get('manuscript_excerpt') or '')
    final_markdown = str(row.get('final_markdown') or '')

    context = [manuscript_excerpt]
    if criteria_json.strip():
        context.append(criteria_json)

    return LLMTestCase(
        input=f'Venue: {venue}',
        actual_output=final_markdown,
        context=context,
    )


def judge_run(collected_row: dict[str, Any] | str) -> dict[str, Any]:
    row = _enrich_row_for_judge(_resolve_collected_row(collected_row))
    test_case = _build_test_case(row)
    scores: dict[str, Any] = {
        'paper_id': row.get('paper_id', ''),
        'job_id': row.get('job_id', ''),
        'venue': row.get('venue', ''),
        'condition': row.get('condition', ''),
    }
    if _judge_mode() == 'composite':
        scores.update(composite_judge_scores(row))
        return scores
    all_metrics = build_all_metrics()
    metrics = (
        {name: all_metrics[name] for name in CORE_JUDGE_METRICS}
        if _judge_mode() == 'per_metric'
        else all_metrics
    )
    for name, metric in metrics.items():
        scores[name] = metric.measure(test_case)
    return scores


def _read_runs_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _existing_judged_job_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        return {str(row.get('job_id', '')) for row in reader if row.get('job_id')}


def judge_all(
    *,
    job_id: str | None = None,
    runs_path: Path | None = None,
    output_path: Path | None = None,
) -> list[dict[str, Any]]:
    source = runs_path or RUNS_JSONL_PATH
    destination = output_path or REVIEW_QUALITY_SCORES_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)

    if job_id:
        candidate_rows = [_resolve_collected_row(job_id)]
        skip_ids: set[str] = set()
    else:
        candidate_rows = _read_runs_jsonl(source)
        skip_ids = _existing_judged_job_ids(destination)

    results: list[dict[str, Any]] = []
    if destination.exists() and destination.stat().st_size > 0:
        with destination.open(encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            results = list(reader)

    for row in candidate_rows:
        completion_errors = validate_run_completion(row)
        if completion_errors:
            continue
        jid = str(row.get('job_id', ''))
        if jid in skip_ids:
            continue
        results.append(judge_run(row))
        pause_between_eval_calls()

    if not results:
        destination.write_text('', encoding='utf-8')
        return results

    fieldnames = list(results[0].keys())
    with destination.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    return results


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
    destination.parent.mkdir(parents=True, exist_ok=True)

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
        destination.write_text('', encoding='utf-8')
        return trad_results

    fieldnames = list(combined[0].keys())
    with destination.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined)
    return trad_results
