from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.checks import MIN_FINAL_REPORT_BYTES
from benchmark.paths import REPO_ROOT, TRAD_REVIEWS_DIR, TRAD_RUNS_JSONL_PATH
from benchmark.trad_registry import load_trad_registry, resolve_source_pdf, trad_job_id


def extract_pdf_text(pdf_path: Path) -> str:
    import fitz

    doc = fitz.open(pdf_path)
    try:
        parts: list[str] = []
        for page in doc:
            parts.append(page.get_text())
        return '\n\n'.join(parts).strip()
    finally:
        doc.close()


def validate_trad_review_bytes(data: bytes) -> list[str]:
    if len(data) <= MIN_FINAL_REPORT_BYTES:
        return [f'review too small ({len(data)} bytes; need > {MIN_FINAL_REPORT_BYTES})']
    return []


def ingest_trad_reviews(*, paper_id: str | None = None) -> list[dict[str, Any]]:
    TRAD_REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    TRAD_RUNS_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict[str, Any]] = []
    for entry in load_trad_registry():
        pid = str(entry['paper_id'])
        if paper_id is not None and pid != paper_id:
            continue

        pdf_path = resolve_source_pdf(str(entry['source_pdf']))
        if not pdf_path.exists():
            raise FileNotFoundError(f'trad source PDF missing: {pdf_path}')

        text = extract_pdf_text(pdf_path)
        md_path = TRAD_REVIEWS_DIR / f'{pid}.md'
        md_path.write_text(text, encoding='utf-8')

        errors = validate_trad_review_bytes(text.encode('utf-8'))
        if errors:
            raise ValueError(f'{pid}: {"; ".join(errors)}')

        rel_review = md_path.relative_to(REPO_ROOT).as_posix()
        run_row = {
            'job_id': trad_job_id(pid),
            'paper_id': pid,
            'venue': entry['venue'],
            'condition': 'TRAD_LLM',
            'status': 'completed',
            'manuscript_job_id': entry['manuscript_job_id'],
            'criteria_job_id': entry['criteria_job_id'],
            'review_path': rel_review,
            'final_md_bytes': len(text.encode('utf-8')),
        }
        rows_out.append(run_row)

    with TRAD_RUNS_JSONL_PATH.open('w', encoding='utf-8') as handle:
        for row in rows_out:
            handle.write(json.dumps(row) + '\n')

    return rows_out
