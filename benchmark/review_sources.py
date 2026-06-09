from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.judge import _truncate_criteria, _truncate_manuscript, _truncate_report
from benchmark.paths import DATA_JOBS_DIR, REPO_ROOT


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
