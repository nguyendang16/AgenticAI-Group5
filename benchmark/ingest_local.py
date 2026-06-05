from __future__ import annotations

import re
import shutil
from pathlib import Path

import fitz

from benchmark.models import PaperRecord
from benchmark.paths import KG_TEST_PAPERS_DIR, PAPERS_DIR, REPO_ROOT

LOCAL_VENUE_OVERRIDES: dict[str, str] = {
    '1412.6980v9.pdf': 'ICLR',
    '9447_Tabular_Insights_Visual_I.pdf': 'ICLR',
    '1706.03762v7.pdf': 'NeurIPS',
    '2402.05602v2.pdf': 'NeurIPS',
    '2311.10263v2.pdf': 'ICML',
    '2402.01869v2.pdf': 'ICML',
    '2404.01847v3.pdf': 'AAAI',
    '2024.acl-short.8.pdf': 'ACL',
    '2024.findings-acl.438.pdf': 'ACL',
    '1-s2.0-S0360131524002380-main.pdf': 'Computers And Education',
    'Liu-UsingAIBasedObject-2023.pdf': 'ETS',
}

VENUE_PATTERNS: list[tuple[str, str]] = [
    ('ICLR', r'International Conference on Learning Representations|Published as a conference paper at ICLR'),
    ('NeurIPS', r'NeurIPS|Neural Information Processing Systems'),
    ('ACL', r'Association for Computational Linguistics|Findings of the Association'),
    ('Computers And Education', r'Computers\s*&\s*Education'),
    ('ETS', r'Educational Technology\s*&\s*Society'),
]


def _venue_slug(venue: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]+', '_', venue.lower()).strip('_')


def detect_venue_from_pdf(pdf_path: Path) -> str | None:
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return None
    try:
        text_parts: list[str] = []
        for page_idx in range(min(2, doc.page_count)):
            text_parts.append(doc[page_idx].get_text())
        text = '\n'.join(text_parts)
        for venue, pattern in VENUE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return venue
        return None
    finally:
        doc.close()


def extract_title_from_pdf(pdf_path: Path) -> str:
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return pdf_path.stem
    try:
        text = doc[0].get_text() if doc.page_count else ''
        snippet = text[:500]
        for line in snippet.splitlines():
            cleaned = line.strip()
            if len(cleaned) >= 10:
                return cleaned
        return pdf_path.stem
    finally:
        doc.close()


def ingest_kg_test_papers() -> list[PaperRecord]:
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    records: list[PaperRecord] = []
    for pdf_path in sorted(KG_TEST_PAPERS_DIR.glob('*.pdf')):
        filename = pdf_path.name
        venue = LOCAL_VENUE_OVERRIDES.get(filename)
        if venue is None:
            venue = detect_venue_from_pdf(pdf_path)
        if venue is None:
            venue = 'Unknown'

        title = extract_title_from_pdf(pdf_path)
        paper_id = f'{_venue_slug(venue)}_{pdf_path.stem}'.lower()

        dest = PAPERS_DIR / f'{paper_id}.pdf'
        if not dest.exists():
            shutil.copy2(pdf_path, dest)

        rel_pdf = dest.relative_to(REPO_ROOT).as_posix()
        records.append(
            PaperRecord(
                paper_id=paper_id,
                venue=venue,
                title=title,
                pdf_path=rel_pdf,
                source='local',
            )
        )
    return records
