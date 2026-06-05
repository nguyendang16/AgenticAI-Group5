from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from benchmark.collect import RUNS_JSONL_PATH
from benchmark.paths import RESULTS_DIR

DETERMINISTIC_SCORES_PATH = RESULTS_DIR / 'deterministic_scores.csv'
MIN_FINAL_REPORT_BYTES = 2048

_NUMERIC_FIELDS = (
    'runtime_seconds',
    'input_tokens',
    'output_tokens',
    'total_tokens',
    'tool_calls',
    'annotation_count',
    'criteria_count',
    'paper_search_calls',
    'final_md_bytes',
)

_BOOLEAN_FIELDS = (
    'has_legend',
    'has_claim_audit',
)


def validate_run_completion(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    status = str(row.get('status') or '').strip().lower()
    if status != 'completed':
        errors.append(f'status must be completed (got {row.get("status")!r})')

    final_md_bytes = row.get('final_md_bytes')
    if final_md_bytes is None:
        errors.append('final_report.md missing or unreadable')
    elif int(final_md_bytes) <= MIN_FINAL_REPORT_BYTES:
        errors.append(
            f'final_report.md too small ({final_md_bytes} bytes; need > {MIN_FINAL_REPORT_BYTES})'
        )

    runtime_seconds = row.get('runtime_seconds')
    if runtime_seconds is None:
        errors.append('events.jsonl missing or runtime not recorded')

    input_tokens = int(row.get('input_tokens') or 0)
    output_tokens = int(row.get('output_tokens') or 0)
    total_tokens = int(row.get('total_tokens') or 0)
    if total_tokens <= 0 and (input_tokens + output_tokens) <= 0:
        errors.append('token usage not recorded')

    paper_search_calls = row.get('paper_search_calls')
    if paper_search_calls is None:
        errors.append('paper_search call count not recorded')
    elif int(paper_search_calls) != 0:
        errors.append(f'paper_search calls must be 0 (got {paper_search_calls})')

    return errors


def validate_condition(row: dict[str, Any]) -> list[str]:
    condition = str(row.get('condition') or '').strip().upper()
    if condition not in {'KG_ON', 'KG_OFF'}:
        return [f'unknown condition {row.get("condition")!r}']

    criteria_count = int(row.get('criteria_count') or 0)
    has_legend = bool(row.get('has_legend'))
    has_claim_audit = bool(row.get('has_claim_audit'))
    venue_criterion_ids = row.get('venue_criterion_ids')
    has_criterion_id_in_annotations = row.get('has_criterion_id_in_annotations')
    has_criteria_bundle = row.get('has_criteria_bundle')

    errors: list[str] = []

    if condition == 'KG_ON':
        if criteria_count <= 0:
            errors.append('KG_ON requires criteria_count > 0')
        if has_criteria_bundle is False:
            errors.append('KG_ON requires review_criteria_bundle.json')
        if not has_legend:
            errors.append('KG_ON requires Criterion Legend section')
        if not has_claim_audit:
            errors.append('KG_ON requires Claim-Level Audit section')
        if venue_criterion_ids is not None and not venue_criterion_ids:
            errors.append('KG_ON requires venue-prefixed criterion IDs')
    else:
        if criteria_count != 0:
            errors.append(f'KG_OFF requires criteria_count == 0 (got {criteria_count})')
        if has_legend:
            errors.append('KG_OFF must not include Criterion Legend section')
        if has_claim_audit:
            errors.append('KG_OFF must not include Claim-Level Audit section')
        if venue_criterion_ids:
            errors.append('KG_OFF must not include venue-prefixed criterion IDs')
        if has_criterion_id_in_annotations:
            errors.append('KG_OFF must not include criterion_id in annotations.json')

    return errors


def score_deterministic(row: dict[str, Any]) -> dict[str, Any]:
    completion_errors = validate_run_completion(row)
    condition_errors = validate_condition(row)

    score: dict[str, Any] = {
        'paper_id': row.get('paper_id', ''),
        'job_id': row.get('job_id', ''),
        'venue': row.get('venue', ''),
        'condition': row.get('condition', ''),
        'completion_valid': not completion_errors,
        'condition_valid': not condition_errors,
        'completion_errors': '; '.join(completion_errors),
        'condition_errors': '; '.join(condition_errors),
    }

    for key in _NUMERIC_FIELDS:
        if key in row and row[key] is not None:
            score[key] = row[key]

    for key in _BOOLEAN_FIELDS:
        if key in row:
            score[key] = row[key]

    return score


def _read_runs_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def run_all_checks(
    *,
    runs_path: Path | None = None,
    output_path: Path | None = None,
) -> list[dict[str, Any]]:
    source = runs_path or RUNS_JSONL_PATH
    destination = output_path or DETERMINISTIC_SCORES_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)

    scores = [score_deterministic(row) for row in _read_runs_jsonl(source)]
    if not scores:
        destination.write_text('', encoding='utf-8')
        return scores

    fieldnames = list(scores[0].keys())
    with destination.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scores)

    return scores
