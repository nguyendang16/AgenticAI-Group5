from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from benchmark.models import PaperRecord
from benchmark.paths import MANIFEST_PATH, PAPERS_DIR, REPO_ROOT

logger = logging.getLogger(__name__)

VENUE_QUERIES: dict[str, dict[str, str]] = {
    'CHI': {'term': 'venue:CHI', 'group': 'all', 'fallback_term': 'CHI accepted'},
    'AAAI': {'term': 'venue:AAAI', 'group': 'AAAI', 'fallback_term': 'AAAI accepted'},
}

SEARCH_BATCH_SIZE = 50
MAX_SEARCH_OFFSET = 500


def _venue_slug(venue: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]+', '_', venue.lower()).strip('_')


def _content_value(field: object) -> str | None:
    if field is None:
        return None
    if isinstance(field, dict):
        value = field.get('value')
        return str(value) if value is not None else None
    return str(field)


def _load_existing_titles() -> list[str]:
    titles: list[str] = []
    if MANIFEST_PATH.exists():
        for line in MANIFEST_PATH.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            title = row.get('title')
            if isinstance(title, str) and title.strip():
                titles.append(title.strip())

    try:
        from benchmark.ingest_local import ingest_kg_test_papers

        for record in ingest_kg_test_papers():
            if record.title.strip():
                titles.append(record.title.strip())
    except Exception:
        logger.debug('Could not load local paper titles for duplicate check', exc_info=True)

    return titles


def _is_duplicate_title(title: str, existing_titles: list[str]) -> bool:
    normalized = title.lower().strip()
    if not normalized:
        return False
    for existing in existing_titles:
        existing_norm = existing.lower().strip()
        if not existing_norm:
            continue
        if normalized in existing_norm or existing_norm in normalized:
            return True
    return False


def _venue_matches(venue_key: str, venue_label: str | None) -> bool:
    if not venue_label:
        return False
    return venue_key.upper() in venue_label.upper()


def _is_accepted_submission(venue_label: str | None, decision: str | None) -> bool:
    if decision:
        decision_lower = decision.lower()
        if 'accept' in decision_lower:
            return True
        if 'reject' in decision_lower:
            return False
    if not venue_label:
        return False
    venue_lower = venue_label.lower()
    if 'submission' in venue_lower and 'accepted' not in venue_lower:
        return False
    return True


def _extract_year(note: object, venue_label: str | None) -> int | None:
    for source in (venue_label, str(getattr(note, 'cdate', '')), str(getattr(note, 'tcdate', ''))):
        if not source:
            continue
        match = re.search(r'20\d{2}', source)
        if match:
            return int(match.group())
    return None


def _note_sort_key(note: object) -> int:
    for attr in ('tcdate', 'cdate', 'tmdate'):
        value = getattr(note, attr, None)
        if isinstance(value, int):
            return value
    return 0


def _search_venue_notes(client: object, venue_key: str, query: dict[str, str]) -> list[object]:
    notes: list[object] = []
    for term in (query['term'], query['fallback_term']):
        offset = 0
        while offset <= MAX_SEARCH_OFFSET:
            try:
                batch = client.search_notes(
                    term=term,
                    group=query['group'],
                    limit=SEARCH_BATCH_SIZE,
                    offset=offset,
                )
            except Exception as exc:
                logger.warning('OpenReview search failed for %s (%s): %s', venue_key, term, exc)
                break
            if not batch:
                break
            notes.extend(batch)
            if len(batch) < SEARCH_BATCH_SIZE:
                break
            offset += SEARCH_BATCH_SIZE
        if notes:
            break
    return notes


def _candidate_notes(client: object, venue_key: str, count: int, existing_titles: list[str]) -> list[object]:
    query = VENUE_QUERIES[venue_key]
    raw_notes = _search_venue_notes(client, venue_key, query)
    candidates: list[object] = []
    seen_ids: set[str] = set()

    for note in sorted(raw_notes, key=_note_sort_key, reverse=True):
        note_id = getattr(note, 'id', None)
        if not note_id or note_id in seen_ids:
            continue
        seen_ids.add(note_id)

        content = getattr(note, 'content', {}) or {}
        title = _content_value(content.get('title'))
        venue_label = _content_value(content.get('venue'))
        decision = _content_value(content.get('decision'))
        pdf_path = _content_value(content.get('pdf'))

        if not title or not pdf_path:
            continue
        if not _venue_matches(venue_key, venue_label):
            continue
        if not _is_accepted_submission(venue_label, decision):
            continue
        if _is_duplicate_title(title, existing_titles):
            continue

        candidates.append(note)
        if len(candidates) >= count:
            break

    return candidates


def fetch_openreview_gaps(needed: dict[str, int]) -> list[PaperRecord]:
    if not needed:
        return []

    try:
        from openreview.api import OpenReviewClient
    except ImportError:
        logger.warning('openreview-py is not installed; skipping OpenReview gap fetch')
        return []

    try:
        client = OpenReviewClient(baseurl='https://api2.openreview.net')
    except Exception as exc:
        logger.warning('OpenReview API unavailable: %s', exc)
        return []

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    existing_titles = _load_existing_titles()
    records: list[PaperRecord] = []

    for venue_key, count in needed.items():
        if count <= 0 or venue_key not in VENUE_QUERIES:
            continue

        try:
            candidates = _candidate_notes(client, venue_key, count, existing_titles)
        except Exception as exc:
            logger.warning('OpenReview fetch failed for %s: %s', venue_key, exc)
            continue

        for note in candidates:
            content = getattr(note, 'content', {}) or {}
            title = _content_value(content.get('title')) or ''
            venue_label = _content_value(content.get('venue')) or venue_key
            decision = _content_value(content.get('decision'))
            paper_id = f'{_venue_slug(venue_key)}_{note.id}'.lower()
            dest = PAPERS_DIR / f'{paper_id}.pdf'

            if not dest.exists():
                try:
                    pdf_bytes = client.get_pdf(note.id)
                except Exception as exc:
                    logger.warning('Failed to download PDF for %s: %s', paper_id, exc)
                    continue
                if not pdf_bytes:
                    logger.warning('Empty PDF for %s', paper_id)
                    continue
                dest.write_bytes(pdf_bytes)

            rel_pdf = dest.relative_to(REPO_ROOT).as_posix()
            record = PaperRecord(
                paper_id=paper_id,
                venue=venue_key,
                title=title,
                pdf_path=rel_pdf,
                source='openreview',
                year=_extract_year(note, venue_label),
                expected_decision=decision,
                metadata_source='openreview',
            )
            records.append(record)
            existing_titles.append(title)

            if sum(1 for r in records if r.venue == venue_key) >= count:
                break

            # Gentle pacing to reduce rate-limit errors during multi-paper fetch.
            time.sleep(0.5)

    return records
