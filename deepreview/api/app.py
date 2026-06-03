from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from deepreview.config import get_settings
from deepreview.evaluation.tier1 import evaluate_job_dir, save_job_evaluation
from deepreview.job_service import (
    artifact_path,
    create_job_from_pdf,
    read_events,
    spawn_worker,
    status_snapshot,
)
from deepreview.state import ensure_artifact_paths, load_job_state
from deepreview.types import JobStatus

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_WEB_DIR = _REPO_ROOT / 'web'
_LOGO_CANDIDATES = (
    _REPO_ROOT / 'assets/branding/logo.png',
    _REPO_ROOT / 'assets/logo.png',
    _REPO_ROOT / 'logo.png',
)


def _resolve_logo_path() -> Path | None:
    for candidate in _LOGO_CANDIDATES:
        if candidate.exists():
            return candidate
    return None

app = FastAPI(title='Evidence-Verified Agentic Peer Review', version='0.1.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/api/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.get('/logo.png')
def logo() -> FileResponse:
    path = _resolve_logo_path()
    if path is None:
        raise HTTPException(status_code=404, detail='logo.png not found')
    return FileResponse(path, media_type='image/png')


@app.get('/favicon.ico')
def favicon() -> FileResponse:
    path = _resolve_logo_path()
    if path is None:
        raise HTTPException(status_code=404, detail='favicon not found')
    return FileResponse(path, media_type='image/png')


@app.post('/api/jobs')
async def submit_job(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    review_venue: str | None = Form(None),
    review_journal: str | None = Form(None),
    review_domain: str | None = Form(None),
    review_article_type: str | None = Form(None),
) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail='Only PDF files are supported')

    suffix = Path(file.filename).suffix or '.pdf'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail='Uploaded file is empty')
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        job = create_job_from_pdf(
            tmp_path,
            title=title or Path(file.filename).stem,
            review_venue=review_venue,
            review_journal=review_journal,
            review_domain=review_domain,
            review_article_type=review_article_type,
        )
        spawn_worker(str(job.id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    return {
        'job_id': str(job.id),
        'status': job.status.value,
        'message': job.message,
    }


@app.get('/api/jobs/{job_id}')
def get_job(job_id: str) -> dict[str, Any]:
    job = load_job_state(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f'Job not found: {job_id}')
    return status_snapshot(job)


@app.get('/api/jobs/{job_id}/events')
def get_job_events(job_id: str, after: int = 0) -> dict[str, Any]:
    job = load_job_state(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f'Job not found: {job_id}')

    events, next_line = read_events(job_id, after_line=max(0, after))
    return {
        'job_id': job_id,
        'events': events,
        'next_after': next_line,
        'status': job.status.value,
    }


@app.get('/api/jobs/{job_id}/report.md')
def get_report_markdown(job_id: str) -> PlainTextResponse:
    job = load_job_state(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f'Job not found: {job_id}')
    if not job.final_report_ready:
        raise HTTPException(status_code=409, detail='Final report not ready yet')

    path = artifact_path(job_id, 'markdown')
    if path is None:
        raise HTTPException(status_code=404, detail='Markdown report missing')
    return PlainTextResponse(path.read_text(encoding='utf-8'), media_type='text/markdown; charset=utf-8')


@app.get('/api/jobs/{job_id}/report.pdf')
def get_report_pdf(job_id: str) -> FileResponse:
    job = load_job_state(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f'Job not found: {job_id}')
    if not job.pdf_ready:
        raise HTTPException(status_code=409, detail='PDF report not ready yet')

    path = artifact_path(job_id, 'pdf')
    if path is None:
        raise HTTPException(status_code=404, detail='PDF report missing')
    return FileResponse(path, media_type='application/pdf', filename=f'review-{job_id}.pdf')


@app.post('/api/jobs/{job_id}/evaluate')
def evaluate_job(job_id: str) -> dict[str, Any]:
    job = load_job_state(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f'Job not found: {job_id}')

    job_dir = ensure_artifact_paths(job_id)['source_pdf'].parent
    try:
        result = evaluate_job_dir(job_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'{type(exc).__name__}: {exc}') from exc

    save_job_evaluation(job_dir, result)
    return result


@app.get('/api/jobs/{job_id}/evaluation')
def get_job_evaluation(job_id: str) -> dict[str, Any]:
    job = load_job_state(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f'Job not found: {job_id}')

    eval_path = ensure_artifact_paths(job_id)['source_pdf'].parent / 'evaluation.json'
    if not eval_path.exists():
        raise HTTPException(status_code=404, detail='Evaluation not found. POST /evaluate first.')
    return json.loads(eval_path.read_text(encoding='utf-8'))


@app.get('/api/evaluations/harvest')
def harvest_all_evaluations() -> dict[str, Any]:
    settings = get_settings()
    jobs_root = settings.data_dir / 'jobs'
    from deepreview.evaluation.tier1 import evaluate_jobs_root, write_evaluation_report

    rows = evaluate_jobs_root(jobs_root)
    if not rows:
        raise HTTPException(status_code=404, detail=f'No jobs under {jobs_root}')

    output_dir = settings.data_dir / 'evaluations' / 'latest'
    paths = write_evaluation_report(rows, output_dir=output_dir)
    for row in rows:
        job_id = str(row.get('job_id') or '').strip()
        if job_id:
            save_job_evaluation(jobs_root / job_id, row)

    payload = json.loads(paths['json'].read_text(encoding='utf-8'))
    payload['paths'] = {key: str(path) for key, path in paths.items()}
    return payload


if _WEB_DIR.is_dir():
    app.mount('/', StaticFiles(directory=str(_WEB_DIR), html=True), name='web')
