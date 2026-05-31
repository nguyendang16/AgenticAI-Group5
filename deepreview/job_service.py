from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from deepreview.config import get_settings
from deepreview.state import ensure_artifact_paths, load_job_state, save_job_state
from deepreview.storage import append_event, events_path, job_dir
from deepreview.types import JobState, JobStatus


def status_snapshot(job: JobState) -> dict[str, Any]:
    return {
        'job_id': str(job.id),
        'status': job.status.value,
        'message': job.message,
        'error': job.error,
        'annotation_count': job.annotation_count,
        'final_report_ready': job.final_report_ready,
        'pdf_ready': job.pdf_ready,
        'usage': job.usage.model_dump(mode='json'),
        'created_at': job.created_at.isoformat(),
        'updated_at': job.updated_at.isoformat(),
        'artifacts': job.artifacts.model_dump(mode='json'),
        'metadata': job.metadata,
    }


def create_job_from_pdf(pdf_path: Path, title: str | None = None) -> JobState:
    pdf_path = pdf_path.expanduser().resolve()
    if not pdf_path.exists() or not pdf_path.is_file():
        raise FileNotFoundError(f'PDF not found: {pdf_path}')

    settings = get_settings()
    file_size = int(pdf_path.stat().st_size)
    if file_size <= 0:
        raise ValueError(f'PDF is empty: {pdf_path}')
    if file_size > int(settings.max_pdf_bytes):
        raise ValueError(
            f'PDF too large: {file_size} bytes, max allowed {int(settings.max_pdf_bytes)} bytes'
        )

    job = JobState(
        title=(title or pdf_path.stem).strip() or pdf_path.stem,
        source_pdf_name=pdf_path.name,
    )
    save_job_state(job)

    artifacts = ensure_artifact_paths(job.id)
    source_pdf_path = Path(artifacts['source_pdf'])
    source_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(pdf_path), str(source_pdf_path))

    loaded = load_job_state(job.id)
    if loaded is None:
        raise RuntimeError(f'failed to reload job after create: {job.id}')

    loaded.artifacts.source_pdf_path = str(source_pdf_path)
    save_job_state(loaded)

    append_event(loaded.id, 'created', source_pdf=str(source_pdf_path), title=loaded.title)
    return loaded


def spawn_worker(job_id: str) -> int:
    root = Path(__file__).resolve().parent.parent
    main_py = root / 'main.py'
    logs_dir = job_dir(job_id)
    stdout_path = logs_dir / 'worker.stdout.log'
    stderr_path = logs_dir / 'worker.stderr.log'

    stdout_f = stdout_path.open('ab')
    stderr_f = stderr_path.open('ab')

    try:
        process = subprocess.Popen(
            [sys.executable, str(main_py), '_run-job', '--job-id', str(job_id)],
            cwd=str(root),
            start_new_session=True,
            stdout=stdout_f,
            stderr=stderr_f,
        )
    finally:
        stdout_f.close()
        stderr_f.close()

    append_event(job_id, 'worker_spawned', pid=process.pid)
    return process.pid


def read_events(job_id: str, *, after_line: int = 0) -> tuple[list[dict[str, Any]], int]:
    path = events_path(job_id)
    if not path.exists():
        return [], after_line

    rows: list[dict[str, Any]] = []
    line_no = 0
    with path.open(encoding='utf-8') as handle:
        for raw in handle:
            line_no += 1
            if line_no <= after_line:
                continue
            text = raw.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except Exception:
                continue
    return rows, line_no


def artifact_path(job_id: str, name: str) -> Path | None:
    job = load_job_state(job_id)
    if job is None:
        return None
    artifacts = ensure_artifact_paths(job_id)
    key_map = {
        'markdown': 'final_markdown',
        'md': 'final_markdown',
        'pdf': 'report_pdf',
        'source': 'source_pdf',
    }
    key = key_map.get(name, name)
    path = artifacts.get(key)
    if path is None or not path.exists():
        return None
    return path
