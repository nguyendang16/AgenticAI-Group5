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
from benchmark.paths import DATA_JOBS_DIR, RESULTS_DIR
from benchmark.registry import list_runs

REVIEW_QUALITY_SCORES_PATH = RESULTS_DIR / 'review_quality_scores.csv'
DEFAULT_JUDGE_MODEL = 'gpt-5-mini'
DEFAULT_MANUSCRIPT_MAX_CHARS = 50_000

METRIC_NAMES = (
    'factual_correctness',
    'evidence_support',
    'rubric_alignment',
    'specificity',
    'actionability',
    'unsupported_critique_rate',
    'criterion_grounded_valid_critique',
)

_EVAL_PARAMS = [LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.CONTEXT]


def _judge_model() -> str:
    return os.environ.get('BENCHMARK_JUDGE_MODEL', DEFAULT_JUDGE_MODEL)


def _manuscript_max_chars() -> int:
    raw = os.environ.get('BENCHMARK_JUDGE_MAX_MANUSCRIPT_CHARS')
    if raw is None:
        return DEFAULT_MANUSCRIPT_MAX_CHARS
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MANUSCRIPT_MAX_CHARS


def _build_geval_metric(*, name: str, criteria: str) -> GEval:
    return GEval(
        name=name,
        criteria=criteria,
        evaluation_params=_EVAL_PARAMS,
        model=_judge_model(),
    )


def build_factual_correctness_metric() -> GEval:
    return _build_geval_metric(
        name='factual_correctness',
        criteria=(
            'Score 1-5 how factually correct the peer review is relative to the manuscript. '
            '1 = major factual errors or misreadings; 5 = claims align with the paper.'
        ),
    )


def build_evidence_support_metric() -> GEval:
    return _build_geval_metric(
        name='evidence_support',
        criteria=(
            'Score 1-5 how well critiques and strengths cite or reflect manuscript evidence. '
            '1 = mostly unsupported assertions; 5 = critiques are well grounded in the text.'
        ),
    )


def build_rubric_alignment_metric() -> GEval:
    return _build_geval_metric(
        name='rubric_alignment',
        criteria=(
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
        criteria=(
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
    job_dir = _job_dir(job_id)

    mineru_path = job_dir / 'mineru_full.md'
    manuscript_excerpt = ''
    if mineru_path.exists():
        manuscript_excerpt = _truncate_manuscript(mineru_path.read_text(encoding='utf-8'))

    final_path = job_dir / 'final_report.md'
    final_markdown = final_path.read_text(encoding='utf-8') if final_path.exists() else ''

    criteria_json = ''
    criteria_path = job_dir / 'review_criteria_bundle.json'
    if criteria_path.exists():
        criteria_json = criteria_path.read_text(encoding='utf-8')

    return {
        'manuscript_excerpt': manuscript_excerpt,
        'final_markdown': final_markdown,
        'criteria_json': criteria_json,
    }


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

    artifacts = load_judge_artifacts(job_id)
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
    metrics = build_all_metrics()

    scores: dict[str, Any] = {
        'paper_id': row.get('paper_id', ''),
        'job_id': row.get('job_id', ''),
        'venue': row.get('venue', ''),
        'condition': row.get('condition', ''),
    }

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
    else:
        candidate_rows = _read_runs_jsonl(source)

    results: list[dict[str, Any]] = []
    for row in candidate_rows:
        completion_errors = validate_run_completion(row)
        if completion_errors:
            continue
        results.append(judge_run(row))

    if not results:
        destination.write_text('', encoding='utf-8')
        return results

    fieldnames = list(results[0].keys())
    with destination.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    return results
