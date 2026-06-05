from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark.paths import DATA_JOBS_DIR, RESULTS_DIR
from benchmark.registry import list_runs

VENUE_CRITERION_ID_RE = re.compile(r'([A-Z]{2,}_C\d{2}_[A-Z0-9_]+)')
RUNS_JSONL_PATH = RESULTS_DIR / 'runs.jsonl'


def _job_dir(job_id: str) -> Path:
    return DATA_JOBS_DIR / job_id


def _read_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def _load_state(job_dir: Path) -> dict[str, Any]:
    for name in ('state.json', 'job.json'):
        path = job_dir / name
        if path.exists():
            payload = _read_json_file(path)
            return payload if isinstance(payload, dict) else {}
    return {}


def _load_events(job_dir: Path) -> list[dict[str, Any]]:
    path = job_dir / 'events.jsonl'
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))
    return events


def _parse_iso_ts(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith('Z'):
        text = text[:-1] + '+00:00'
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _runtime_seconds(events: list[dict[str, Any]], state: dict[str, Any]) -> float | None:
    timestamps = [_parse_iso_ts(event.get('ts')) for event in events]
    timestamps = [ts for ts in timestamps if ts is not None]
    if len(timestamps) >= 2:
        return max(0.0, (max(timestamps) - min(timestamps)).total_seconds())
    if len(timestamps) == 1:
        return 0.0

    created = _parse_iso_ts(state.get('created_at'))
    updated = _parse_iso_ts(state.get('updated_at'))
    if created is not None and updated is not None:
        return max(0.0, (updated - created).total_seconds())
    return None


def _last_criteria_resolution(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in reversed(events):
        if event.get('event') == 'review_criteria_resolved':
            return event
    return {}


def _nested_int(payload: dict[str, Any], *keys: str, default: int = 0) -> int:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    try:
        return int(current or 0)
    except (TypeError, ValueError):
        return default


def _load_annotations(job_dir: Path) -> list[Any]:
    path = job_dir / 'annotations.json'
    if not path.exists():
        return []
    payload = _read_json_file(path)
    if isinstance(payload, list):
        return payload
    return []


def _load_criteria_bundle(job_dir: Path) -> dict[str, Any]:
    path = job_dir / 'review_criteria_bundle.json'
    if not path.exists():
        return {}
    payload = _read_json_file(path)
    return payload if isinstance(payload, dict) else {}


def _extract_venue_criterion_ids(*texts: str) -> list[str]:
    found: set[str] = set()
    for text in texts:
        if text:
            found.update(VENUE_CRITERION_ID_RE.findall(text))
    return sorted(found)


def collect_run(job_id: str) -> dict[str, Any]:
    job_dir = _job_dir(job_id)
    state = _load_state(job_dir)
    events = _load_events(job_dir)
    criteria_resolution = _last_criteria_resolution(events)
    criteria_bundle = _load_criteria_bundle(job_dir)
    annotations = _load_annotations(job_dir)

    final_report_path = job_dir / 'final_report.md'
    final_report_text = ''
    final_md_bytes = 0
    if final_report_path.exists():
        final_report_text = final_report_path.read_text(encoding='utf-8')
        final_md_bytes = len(final_report_text.encode('utf-8'))

    criteria_count = _nested_int(criteria_resolution, 'criteria_count')
    if criteria_count == 0:
        criteria_count = _nested_int(state, 'metadata', 'review_criteria_count')
    if criteria_count == 0:
        criteria_count = _nested_int(criteria_bundle, 'criteria_count')

    annotation_count = _nested_int(state, 'annotation_count')
    if annotation_count == 0 and annotations:
        annotation_count = len(annotations)

    usage = state.get('usage') if isinstance(state.get('usage'), dict) else {}
    token_usage = usage.get('token') if isinstance(usage.get('token'), dict) else {}
    tool_usage = usage.get('tool') if isinstance(usage.get('tool'), dict) else {}
    paper_search_usage = usage.get('paper_search') if isinstance(usage.get('paper_search'), dict) else {}

    bundle_text = json.dumps(criteria_bundle, ensure_ascii=False) if criteria_bundle else ''
    venue_criterion_ids = _extract_venue_criterion_ids(final_report_text, bundle_text)

    status = state.get('status')
    if isinstance(status, dict):
        status = status.get('value') or status.get('name')
    status_text = str(status or '').strip()
    if status_text.startswith('JobStatus.'):
        status_text = status_text.split('.', 1)[1]

    runtime = _runtime_seconds(events, state)

    return {
        'job_id': job_id,
        'status': status_text,
        'runtime_seconds': runtime,
        'input_tokens': _nested_int(token_usage, 'input_tokens'),
        'output_tokens': _nested_int(token_usage, 'output_tokens'),
        'total_tokens': _nested_int(token_usage, 'total_tokens'),
        'tool_calls': _nested_int(tool_usage, 'total_calls'),
        'annotation_count': annotation_count,
        'criteria_count': criteria_count,
        'paper_search_calls': _nested_int(paper_search_usage, 'total_calls'),
        'final_md_bytes': final_md_bytes,
        'has_legend': '## Criterion Legend' in final_report_text,
        'has_claim_audit': '## Claim-Level Audit' in final_report_text,
        'venue_criterion_ids': venue_criterion_ids,
    }


def harvest_all(*, output_path: Path | None = None) -> list[dict[str, Any]]:
    path = output_path or RUNS_JSONL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for run in list_runs():
        row = collect_run(run.job_id)
        row['paper_id'] = run.paper_id
        row['venue'] = run.venue
        row['condition'] = run.condition
        row['git_commit'] = run.git_commit
        rows.append(row)

    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    return rows
