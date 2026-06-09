from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from benchmark.analysis_subset import filter_paper_ids
from benchmark.checks import validate_run_completion
from benchmark.collect import RUNS_JSONL_PATH
from benchmark.eval_llm import judge_model_name, judge_provider, pause_between_eval_calls
from benchmark.judge import _read_runs_jsonl, _read_trad_runs_jsonl
from benchmark.paths import TRAD_PAIRWISE_JUDGE_SCORES_PATH, TRAD_RUNS_JSONL_PATH
from benchmark.review_sources import load_review_artifacts

TRAD_PAIRWISE_FIELDS = (
    'paper_id',
    'venue',
    'pair_type',
    'winner',
    'rubric_alignment_winner',
    'confidence',
    'reason',
    'job_id_trad',
    'job_id_agent',
)

TRAD_PAIRWISE_JSON_KEYS = (
    'winner',
    'rubric_alignment_winner',
    'confidence',
    'one_line_reason',
)

_PAIR_TYPES = {'TRAD_VS_KG_OFF', 'TRAD_VS_KG_ON'}


def parse_trad_pairwise_response(raw: str) -> dict[str, str]:
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
            response_format={'type': 'json_object'},
            messages=[
                {
                    'role': 'system',
                    'content': (
                        'You compare peer reviews for benchmark evaluation. '
                        f'Return JSON with keys: {", ".join(TRAD_PAIRWISE_JSON_KEYS)}.'
                    ),
                },
                {'role': 'user', 'content': prompt},
            ],
        )
        return response.choices[0].message.content or ''
    raise ValueError(f'Unsupported BENCHMARK_JUDGE_PROVIDER for trad pairwise judge: {provider}')


def trad_pairwise_compare(
    trad_row: dict[str, Any],
    agent_row: dict[str, Any],
    *,
    pair_type: str,
) -> dict[str, str]:
    if pair_type not in _PAIR_TYPES:
        raise ValueError(f'Unsupported pair_type: {pair_type}')

    agent_label = 'KG_OFF' if pair_type == 'TRAD_VS_KG_OFF' else 'KG_ON'

    job_id_trad = str(trad_row.get('job_id') or '')
    job_id_agent = str(agent_row.get('job_id') or '')
    paper_id = str(trad_row.get('paper_id') or agent_row.get('paper_id') or '')
    venue = str(trad_row.get('venue') or agent_row.get('venue') or '')

    trad_artifacts = load_review_artifacts(trad_row)
    agent_artifacts = load_review_artifacts(agent_row)

    manuscript = (
        trad_artifacts.get('manuscript_excerpt')
        or agent_artifacts.get('manuscript_excerpt')
        or ''
    )
    criteria = trad_artifacts.get('criteria_json') or agent_artifacts.get('criteria_json') or ''
    review_a = trad_artifacts.get('final_markdown') or ''
    review_b = agent_artifacts.get('final_markdown') or ''

    prompt = (
        f'Compare Review A (TRAD) and Review B ({agent_label}) for venue {venue}. '
        'Which better satisfies rubric_alignment, evidence_support, and actionable critique quality? '
        f'Return JSON with keys: winner (TRAD|{agent_label}|tie), '
        f'rubric_alignment_winner (TRAD|{agent_label}|tie), confidence (high|medium|low), '
        'one_line_reason.\n\n'
        f'Venue criteria:\n{criteria}\n\n'
        f'Manuscript excerpt:\n{manuscript}\n\n'
        f'Review A (TRAD):\n{review_a}\n\n'
        f'Review B ({agent_label}):\n{review_b}'
    )
    parsed = parse_trad_pairwise_response(_call_judge_llm(prompt))
    return {
        'paper_id': paper_id,
        'venue': venue,
        'pair_type': pair_type,
        'winner': parsed['winner'],
        'rubric_alignment_winner': parsed['rubric_alignment_winner'],
        'confidence': parsed['confidence'],
        'reason': parsed['reason'],
        'job_id_trad': job_id_trad,
        'job_id_agent': job_id_agent,
    }


def _write_results(path: Path, results: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TRAD_PAIRWISE_FIELDS))
        writer.writeheader()
        writer.writerows(results)


def trad_pairwise_all(
    *,
    paper_id: str | None = None,
    use_subset: bool = True,
) -> list[dict[str, str]]:
    trad_rows = _read_trad_runs_jsonl(TRAD_RUNS_JSONL_PATH)
    agent_rows = _read_runs_jsonl(RUNS_JSONL_PATH)

    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in trad_rows:
        pid = str(row.get('paper_id') or '').strip()
        if not pid:
            continue
        if paper_id is not None and pid != paper_id:
            continue
        condition = str(row.get('condition') or '').strip().upper()
        if condition in {'TRAD_LLM', 'TRAD'}:
            grouped.setdefault(pid, {})['TRAD'] = row

    for row in agent_rows:
        pid = str(row.get('paper_id') or '').strip()
        if not pid:
            continue
        if paper_id is not None and pid != paper_id:
            continue
        if validate_run_completion(row):
            continue
        condition = str(row.get('condition') or '').strip().upper()
        if condition in {'KG_OFF', 'KG_ON'}:
            grouped.setdefault(pid, {})[condition] = row

    paper_ids = list(grouped.keys())
    if use_subset and paper_id is None:
        paper_ids = filter_paper_ids(paper_ids, use_subset=True)

    results: list[dict[str, str]] = []
    for pid in paper_ids:
        rows = grouped.get(pid, {})
        trad_row = rows.get('TRAD')
        kg_off_row = rows.get('KG_OFF')
        kg_on_row = rows.get('KG_ON')
        if trad_row is None or kg_off_row is None or kg_on_row is None:
            continue
        results.append(
            trad_pairwise_compare(trad_row, kg_off_row, pair_type='TRAD_VS_KG_OFF')
        )
        pause_between_eval_calls()
        results.append(
            trad_pairwise_compare(trad_row, kg_on_row, pair_type='TRAD_VS_KG_ON')
        )
        pause_between_eval_calls()

    _write_results(TRAD_PAIRWISE_JUDGE_SCORES_PATH, results)
    return results
