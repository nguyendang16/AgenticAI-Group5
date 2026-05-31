from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from deepreview.job_service import (
    artifact_path,
    create_job_from_pdf,
    read_events,
    spawn_worker,
    status_snapshot,
)
from deepreview.state import load_job_state
from deepreview.types import JobStatus

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_WEB_DIR = _REPO_ROOT / 'web'

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
    path = _REPO_ROOT / 'logo.png'
    if not path.exists():
        raise HTTPException(status_code=404, detail='logo.png not found')
    return FileResponse(path, media_type='image/png')


@app.get('/favicon.ico')
def favicon() -> FileResponse:
    path = _REPO_ROOT / 'logo.png'
    if not path.exists():
        raise HTTPException(status_code=404, detail='favicon not found')
    return FileResponse(path, media_type='image/png')


@app.post('/api/jobs')
async def submit_job(
    file: UploadFile = File(...),
    title: str | None = Form(None),
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
        job = create_job_from_pdf(tmp_path, title=title or Path(file.filename).stem)
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


if _WEB_DIR.is_dir():
    app.mount('/', StaticFiles(directory=str(_WEB_DIR), html=True), name='web')
