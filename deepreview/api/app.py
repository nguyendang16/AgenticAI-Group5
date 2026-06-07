from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from html import escape
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
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
        request.url.path.startswith('/api/kg-benchmark')
        or request.url.path.startswith('/api/kg-evidence-tests')
        or request.url.path.endswith('benchmark-graph-evaluation.html')
        or request.url.path.endswith('kg-evidence-tests.html')
    ):
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response


def _parse_dt(value: Any) -> datetime:
    if not isinstance(value, str):
        return datetime.min
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00')).replace(tzinfo=None)
    except ValueError:
        return datetime.min


def _paper_input_from_job(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get('metadata') if isinstance(payload.get('metadata'), dict) else {}
    criteria_query = metadata.get('review_criteria_query')
    if not isinstance(criteria_query, dict):
        criteria_query = {}
    criteria_graph = metadata.get('review_criteria_graph_evaluation')
    if not isinstance(criteria_graph, dict):
        criteria_graph = {}
    usage = payload.get('usage') if isinstance(payload.get('usage'), dict) else {}
    tool_usage = usage.get('tool') if isinstance(usage.get('tool'), dict) else {}
    per_tool = tool_usage.get('per_tool') if isinstance(tool_usage.get('per_tool'), dict) else {}
    return {
        'job_id': payload.get('id'),
        'title': payload.get('title'),
        'source_pdf_name': payload.get('source_pdf_name'),
        'status': payload.get('status'),
        'created_at': payload.get('created_at'),
        'updated_at': payload.get('updated_at'),
        'venue': criteria_query.get('venue') or metadata.get('review_venue'),
        'journal': criteria_query.get('journal') or metadata.get('review_journal'),
        'domain': criteria_query.get('domain') or metadata.get('review_domain'),
        'article_type': criteria_query.get('article_type') or metadata.get('review_article_type'),
        'criteria_source': metadata.get('review_criteria_source'),
        'criteria_count': metadata.get('review_criteria_count') or criteria_graph.get('criteria_count'),
        'active_criteria_count': criteria_graph.get('active_criteria_count'),
        'evidence_requirements_count': criteria_graph.get('evidence_requirements_count'),
        'annotation_count': payload.get('annotation_count'),
        'pdf_search_calls': per_tool.get('pdf_search'),
        'pdf_annotate_calls': per_tool.get('pdf_annotate'),
        'final_report_ready': payload.get('final_report_ready'),
        'pdf_ready': payload.get('pdf_ready'),
    }


def _latest_paper_inputs() -> dict[str, Any]:
    jobs_root = _REPO_ROOT / 'data' / 'jobs'
    if not jobs_root.exists():
        return {}
    jobs: list[dict[str, Any]] = []
    for path in jobs_root.glob('*/job.json'):
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        if isinstance(payload, dict):
            jobs.append(payload)
    jobs.sort(key=lambda item: _parse_dt(item.get('updated_at')), reverse=True)
    latest = jobs[0] if jobs else None
    latest_completed = next(
        (
            job
            for job in jobs
            if job.get('status') == 'completed'
            and (job.get('final_report_ready') or job.get('pdf_ready'))
        ),
        None,
    )
    result: dict[str, Any] = {}
    if latest:
        paper_input = _paper_input_from_job(latest)
        paper_status = str(paper_input.get('status') or '')
        paper_input['is_running'] = paper_status in {
            'queued',
            'pdf_uploading_to_mineru',
            'pdf_parsing',
            'agent_running',
            'final_report_persisting',
            'pdf_exporting',
        }
        paper_input['is_finished'] = paper_status == 'completed'
        paper_input['is_failed'] = paper_status == 'failed'
        result['paper_input'] = paper_input
    if latest_completed:
        result['latest_completed_paper_input'] = _paper_input_from_job(latest_completed)
    return result


@app.get('/api/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.get('/api/kg-benchmark')
def get_kg_benchmark() -> dict[str, Any]:
    return _kg_benchmark_payload()


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


def _kg_benchmark_payload() -> dict[str, Any]:
    path = _REPO_ROOT / 'outputs' / 'kg_evaluation.json'
    if not path.exists():
        return {
            'ready': False,
            'message': 'Benchmark result not found. Rebuild the KG and run src.kg_evaluation first.',
        }
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Failed to read benchmark result: {exc}') from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail='Benchmark result has invalid format')
    payload['ready'] = True
    payload.update(_latest_paper_inputs())
    return payload


@app.get('/benchmark-graph-evaluation.html', response_class=HTMLResponse)
def benchmark_graph_evaluation_page() -> HTMLResponse:
    path = _WEB_DIR / 'benchmark-graph-evaluation.html'
    if not path.exists():
        raise HTTPException(status_code=404, detail='benchmark page not found')
    html = path.read_text(encoding='utf-8')
    try:
        payload = _kg_benchmark_payload()
    except Exception:
        payload = {}
    paper = payload.get('paper_input') if isinstance(payload.get('paper_input'), dict) else {}
    completed = (
        payload.get('latest_completed_paper_input')
        if isinstance(payload.get('latest_completed_paper_input'), dict)
        else {}
    )
    display = paper if paper.get('title') else completed
    title = escape(str(display.get('title') or display.get('source_pdf_name') or 'No evaluated paper yet'))
    html = html.replace(
        'Evaluated paper: Loading paper title...',
        f'Evaluated paper: {title}',
    )
    html = html.replace(
        '<strong id="benchmark-paper-title">Loading paper title...</strong>',
        f'<strong id="benchmark-paper-title">{title}</strong>',
    )
    html = html.replace(
        '<strong id="paper-title">Loading paper title...</strong>',
        f'<strong id="paper-title">{title}</strong>',
    )
    return HTMLResponse(
        html,
        headers={
            'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
            'Pragma': 'no-cache',
            'Expires': '0',
        },
    )


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
