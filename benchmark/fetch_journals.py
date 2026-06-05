from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

import httpx

from benchmark.ingest_local import extract_title_from_pdf
from benchmark.models import PaperRecord
from benchmark.paths import INCOMING_DIR, PAPERS_DIR, REPO_ROOT

logger = logging.getLogger(__name__)

JOURNAL_SEED_URLS: dict[str, list[dict[str, str]]] = {
    'TWELF': [],
    'ETRD': [],
    'Computers And Education': [],
    'ETS': [],
}

INCOMING_VENUE_PREFIXES: dict[str, str] = {
    'TWELF': 'TWELF',
    'ETRD': 'ETRD',
    'ETS': 'ETS',
    'COMPUTERS_AND_EDUCATION': 'Computers And Education',
}


def _venue_slug(venue: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]+', '_', venue.lower()).strip('_')


def _slugify(text: str) -> str:
    slug = re.sub(r'[^a-zA-Z0-9]+', '_', text.lower()).strip('_')
    return slug or 'paper'


def _venue_from_incoming_filename(filename: str) -> tuple[str, str] | None:
    stem = Path(filename).stem
    if '_' not in stem:
        return None
    prefix, slug = stem.split('_', 1)
    venue = INCOMING_VENUE_PREFIXES.get(prefix.upper())
    if venue is None or not slug:
        return None
    return venue, slug


def scan_incoming_papers(
    needed: dict[str, int],
    *,
    existing_paper_ids: set[str] | None = None,
) -> list[PaperRecord]:
    """Register manually dropped PDFs from benchmark/papers/incoming/."""
    if not needed or not any(count > 0 for count in needed.values()):
        return []

    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    seen_ids = set(existing_paper_ids or ())
    records: list[PaperRecord] = []
    filled: dict[str, int] = {venue: 0 for venue in needed}

    for pdf_path in sorted(INCOMING_DIR.glob('*.pdf')):
        parsed = _venue_from_incoming_filename(pdf_path.name)
        if parsed is None:
            continue
        venue, slug = parsed
        remaining = needed.get(venue, 0) - filled.get(venue, 0)
        if remaining <= 0:
            continue

        paper_id = f'{_venue_slug(venue)}_{slug}'.lower()
        if paper_id in seen_ids:
            continue

        dest = PAPERS_DIR / f'{paper_id}.pdf'
        if not dest.exists():
            shutil.copy2(pdf_path, dest)

        title = extract_title_from_pdf(dest)
        rel_pdf = dest.relative_to(REPO_ROOT).as_posix()
        records.append(
            PaperRecord(
                paper_id=paper_id,
                venue=venue,
                title=title,
                pdf_path=rel_pdf,
                source='incoming',
                metadata_source='incoming',
            )
        )
        seen_ids.add(paper_id)
        filled[venue] = filled.get(venue, 0) + 1

    return records


def fetch_journal_gaps(needed: dict[str, int]) -> list[PaperRecord]:
    if not needed:
        return []

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    records: list[PaperRecord] = []

    try:
        with httpx.Client(follow_redirects=True, timeout=60.0) as client:
            for venue, count in needed.items():
                if count <= 0 or venue not in JOURNAL_SEED_URLS:
                    continue

                fetched_for_venue = 0
                for entry in JOURNAL_SEED_URLS[venue]:
                    if fetched_for_venue >= count:
                        break

                    title = entry.get('title', '').strip()
                    url = entry.get('url', '').strip()
                    entry_venue = entry.get('venue', venue).strip() or venue
                    if not url:
                        continue

                    slug = _slugify(title or Path(url).stem)
                    paper_id = f'{_venue_slug(entry_venue)}_{slug}'.lower()
                    dest = PAPERS_DIR / f'{paper_id}.pdf'

                    if not dest.exists():
                        try:
                            response = client.get(url)
                        except httpx.HTTPError as exc:
                            logger.warning('venue_gap: %s request failed for %s: %s', venue, url, exc)
                            continue

                        if response.status_code in (403, 404):
                            logger.warning(
                                'venue_gap: %s HTTP %s for %s',
                                venue,
                                response.status_code,
                                url,
                            )
                            continue

                        try:
                            response.raise_for_status()
                        except httpx.HTTPStatusError as exc:
                            logger.warning('venue_gap: %s download failed for %s: %s', venue, url, exc)
                            continue

                        content_type = response.headers.get('content-type', '')
                        if 'pdf' not in content_type.lower() and not url.lower().endswith('.pdf'):
                            logger.warning('venue_gap: %s non-PDF response for %s', venue, url)
                            continue

                        if not response.content:
                            logger.warning('venue_gap: %s empty response for %s', venue, url)
                            continue

                        dest.write_bytes(response.content)

                    if not title:
                        title = extract_title_from_pdf(dest)

                    rel_pdf = dest.relative_to(REPO_ROOT).as_posix()
                    records.append(
                        PaperRecord(
                            paper_id=paper_id,
                            venue=entry_venue,
                            title=title,
                            pdf_path=rel_pdf,
                            source='journal',
                            metadata_source='journal',
                        )
                    )
                    fetched_for_venue += 1
    except Exception as exc:
        logger.warning('Journal fetch unavailable: %s', exc)

    return records
