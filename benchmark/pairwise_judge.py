from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from benchmark.checks import validate_run_completion
from benchmark.collect import RUNS_JSONL_PATH
from benchmark.eval_llm import judge_model_name, judge_provider, pause_between_eval_calls
from benchmark.judge import _read_runs_jsonl, load_judge_artifacts
from benchmark.paths import PAIRWISE_JUDGE_SCORES_PATH

PAIRWISE_FIELDS = (
    'paper_id',
    'venue',
    'winner',
    'rubric_alignment_winner',
    'confidence',
    'reason',
    'job_id_kg_on',
    'job_id_kg_off',
)

_PAIRWISE_JSON_KEYS = ('winner', 'rubric_alignment_winner', 'confidence', 'one_line_reason')

_PAIRWISE_PROMPT = (
    'Compare Review A (KG_OFF) and Review B (KG_ON) for venue {venue}. '
    'Which better satisfies rubric_alignment, evidence_support, and actionable critique quality? '
    'Return JSON with keys: winner (KG_ON|KG_OFF|tie), rubric_alignment_winner (KG_ON|KG_OFF|tie), '
    'confidence (high|medium|low), one_line_reason.\n\n'
    'Venue criteria:\n{criteria}\n\n'
    'Manuscript excerpt:\n{manuscript}\n\n'
    'Review A (KG_OFF):\n{review_a}\n\n'
    'Review B (KG_ON):\n{review_b}'
)


def parse_pairwise_response(raw: str) -> dict[str, str]:
    text = raw.strip()
    if '```' in text:
        text = text.split('```', 1)[-1]
        if text.startswith('json'):
            text = text[4:]
        text = text.split('```', 1)[0]
    payload = json.loads(text.strip())
    return {
        'winner': str(payload.get('winner', '')),
        'rubric_alignment_winner': str(payload.get('rubric_alignment_winner', '')),
        'confidence': str(payload.get('confidence', '')),
        'reason': str(payload.get('one_line_reason') or payload.get('reason') or ''),
    }


def _call_judge_llm(prompt: str) -> str:
    provider = judge_provider()
    if provider == 'openai':
        from openai import OpenAI

        api_key = os.environ.get('OPENAI_API_KEY') or os.environ.get('API_KEY')
        base_url = (
            os.environ.get('OPENAI_BASE_URL')
            or os.environ.get('BASE_URL')
            or os.environ.get('OPENAI_API_BASE')
            or None
        )
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=judge_model_name(),
            temperature=0,
            response_format={'type': 'json_object'},
            messages=[
                {
                    'role': 'system',
                    'content': (
                        'You compare peer reviews for benchmark evaluation. '
                        f'Return JSON with keys: {", ".join(_PAIRWISE_JSON_KEYS)}.'
                    ),
                },
                {'role': 'user', 'content': prompt},
            ],
        )
        return response.choices[0].message.content or ''
    raise ValueError(f'Unsupported BENCHMARK_JUDGE_PROVIDER for pairwise judge: {provider}')


def pairwise_compare(kg_off_row: dict[str, Any], kg_on_row: dict[str, Any]) -> dict[str, str]:
    job_id_kg_off = str(kg_off_row.get('job_id') or '')
    job_id_kg_on = str(kg_on_row.get('job_id') or '')
    paper_id = str(kg_off_row.get('paper_id') or kg_on_row.get('paper_id') or '')
    venue = str(kg_off_row.get('venue') or kg_on_row.get('venue') or '')

    artifacts_off = load_judge_artifacts(job_id_kg_off)
    artifacts_on = load_judge_artifacts(job_id_kg_on)

    manuscript = artifacts_on.get('manuscript_excerpt') or artifacts_off.get('manuscript_excerpt') or ''
    criteria = artifacts_on.get('criteria_json') or artifacts_off.get('criteria_json') or ''
    review_a = artifacts_off.get('final_markdown') or ''
    review_b = artifacts_on.get('final_markdown') or ''

    prompt = _PAIRWISE_PROMPT.format(
        venue=venue,
        criteria=criteria,
        manuscript=manuscript,
        review_a=review_a,
        review_b=review_b,
    )
    parsed = parse_pairwise_response(_call_judge_llm(prompt))

    return {
        'paper_id': paper_id,
        'venue': venue,
        'winner': parsed['winner'],
        'rubric_alignment_winner': parsed['rubric_alignment_winner'],
        'confidence': parsed['confidence'],
        'reason': parsed['reason'],
        'job_id_kg_on': job_id_kg_on,
        'job_id_kg_off': job_id_kg_off,
    }


def _existing_paper_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        return {str(row.get('paper_id', '')) for row in reader if row.get('paper_id')}


def _group_valid_pairs(
    runs: list[dict[str, Any]],
    *,
    paper_id: str | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in runs:
        pid = str(row.get('paper_id') or '').strip()
        if not pid:
            continue
        if paper_id is not None and pid != paper_id:
            continue
        if validate_run_completion(row):
            continue
        condition = str(row.get('condition') or '').strip().upper()
        if condition not in {'KG_ON', 'KG_OFF'}:
            continue
        grouped.setdefault(pid, {})[condition] = row
    return grouped


def _write_results(path: Path, results: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PAIRWISE_FIELDS))
        writer.writeheader()
        writer.writerows(results)


def pairwise_all(
    *,
    paper_id: str | None = None,
    runs_path: Path | None = None,
    output_path: Path | None = None,
) -> list[dict[str, str]]:
    source = runs_path or RUNS_JSONL_PATH
    destination = output_path or PAIRWISE_JUDGE_SCORES_PATH

    grouped = _group_valid_pairs(_read_runs_jsonl(source), paper_id=paper_id)
    skip_paper_ids: set[str] = set() if paper_id else _existing_paper_ids(destination)

    results: list[dict[str, str]] = []
    if destination.exists() and destination.stat().st_size > 0:
        with destination.open(encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            results = [dict(row) for row in reader]

    for pid, conditions in grouped.items():
        if pid in skip_paper_ids:
            continue
        kg_off_row = conditions.get('KG_OFF')
        kg_on_row = conditions.get('KG_ON')
        if kg_off_row is None or kg_on_row is None:
            continue
        results.append(pairwise_compare(kg_off_row, kg_on_row))
        pause_between_eval_calls()

    _write_results(destination, results)
    return results
