from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
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


@app.middleware('http')
async def add_no_cache_for_benchmark(request: Request, call_next):
    response = await call_next(request)
    if (
        request.url.path.startswith('/api/kg-evidence-tests')
        or request.url.path.endswith('kg-evidence-tests.html')
    ):
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response


@app.get('/api/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}


def _kg_evidence_tests_payload() -> dict[str, Any]:
    path = _REPO_ROOT / 'outputs' / 'kg_evidence_retrieval_test_results.json'
    if not path.exists():
        return {
            'ready': False,
            'message': 'Evidence retrieval test result not found. Run the test suite first.',
        }
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Failed to read evidence test result: {exc}') from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail='Evidence test result has invalid format')

    by_source: dict[str, dict[str, int]] = {}
    for case in payload.get('cases', []):
        if not isinstance(case, dict):
            continue
        source_ids = case.get('expected_source_ids') if isinstance(case.get('expected_source_ids'), list) else []
        for source_id in source_ids:
            key = str(source_id or '').strip()
            if not key:
                continue
            row = by_source.setdefault(key, {'cases': 0, 'passed': 0, 'failed': 0})
            row['cases'] += 1
            if case.get('passed'):
                row['passed'] += 1
            else:
                row['failed'] += 1

    payload['ready'] = True
    payload['by_source'] = by_source
    return payload


@app.get('/api/kg-evidence-tests')
def get_kg_evidence_tests() -> dict[str, Any]:
    return _kg_evidence_tests_payload()


@app.post('/api/kg-evidence-tests/run')
def run_kg_evidence_tests() -> dict[str, Any]:
    command = [
        sys.executable,
        '-m',
        'src.kg_evidence_retrieval_tests',
        '--cases',
        'data/kg_evidence_retrieval_test_cases.json',
        '--output-json',
        'outputs/kg_evidence_retrieval_test_results.json',
        '--output-md',
        'docs/kg_evidence_retrieval_test_results.md',
    ]
    completed = subprocess.run(
        command,
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        result_path = _REPO_ROOT / 'outputs' / 'kg_evidence_retrieval_test_results.json'
        if not result_path.exists():
            raise HTTPException(
                status_code=500,
                detail={
                    'message': 'Evidence retrieval tests failed before writing results',
                    'stdout': completed.stdout[-4000:],
                    'stderr': completed.stderr[-4000:],
                },
            )
    payload = _kg_evidence_tests_payload()
    payload['run_return_code'] = completed.returncode
    return payload

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


if _WEB_DIR.is_dir():
    app.mount('/', StaticFiles(directory=str(_WEB_DIR), html=True), name='web')
