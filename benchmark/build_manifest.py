from __future__ import annotations

import json
import logging
from collections import defaultdict

from benchmark.fetch_journals import fetch_journal_gaps, scan_incoming_papers
from benchmark.fetch_openreview import fetch_openreview_gaps
from benchmark.ingest_local import ingest_benchmark_papers_dir, ingest_kg_test_papers
from benchmark.models import PaperRecord
from benchmark.paths import MANIFEST_PATH

logger = logging.getLogger(__name__)

TARGET_VENUES = [
    'ICLR',
    'NeurIPS',
    'ICML',
    'ACL',
    'CHI',
    'AAAI',
    'TWELF',
    'Computers And Education',
    'ETRD',
    'ETS',
]
PAPERS_PER_VENUE = 2

OPENREVIEW_VENUES = frozenset({'CHI', 'AAAI'})
JOURNAL_VENUES = frozenset({'TWELF', 'ETRD', 'Computers And Education', 'ETS'})


def group_by_venue(rows: list[PaperRecord]) -> dict[str, list[PaperRecord]]:
    by_venue: dict[str, list[PaperRecord]] = defaultdict(list)
    for row in rows:
        by_venue[row.venue].append(row)
    return dict(by_venue)


def compute_gaps(rows: list[PaperRecord]) -> dict[str, int]:
    by_venue = group_by_venue(rows)
    return {venue: max(0, PAPERS_PER_VENUE - len(by_venue.get(venue, []))) for venue in TARGET_VENUES}


def _dedupe_rows(rows: list[PaperRecord]) -> list[PaperRecord]:
    seen: set[str] = set()
    deduped: list[PaperRecord] = []
    for row in rows:
        if row.paper_id in seen:
            continue
        seen.add(row.paper_id)
        deduped.append(row)
    return deduped


def _select_manifest_rows(rows: list[PaperRecord]) -> list[PaperRecord]:
    by_venue = group_by_venue(_dedupe_rows(rows))
    selected: list[PaperRecord] = []
    for venue in TARGET_VENUES:
        selected.extend(by_venue.get(venue, [])[:PAPERS_PER_VENUE])
    return selected


def build_manifest(*, local_only: bool = False) -> list[PaperRecord]:
    if local_only:
        rows = ingest_benchmark_papers_dir()
        manifest_rows = _select_manifest_rows(rows)
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with MANIFEST_PATH.open('w', encoding='utf-8') as handle:
            for row in manifest_rows:
                handle.write(json.dumps(row.to_dict()) + '\n')
        logger.info('Wrote %d papers to %s (local-only)', len(manifest_rows), MANIFEST_PATH)
        return manifest_rows

    rows = ingest_kg_test_papers()

    gaps = compute_gaps(rows)
    openreview_needed = {venue: count for venue, count in gaps.items() if venue in OPENREVIEW_VENUES and count > 0}
    if openreview_needed:
        rows.extend(fetch_openreview_gaps(openreview_needed))

    gaps = compute_gaps(rows)
    journal_needed = {venue: count for venue, count in gaps.items() if venue in JOURNAL_VENUES and count > 0}
    if journal_needed:
        rows.extend(fetch_journal_gaps(journal_needed))

    gaps = compute_gaps(rows)
    incoming_needed = {venue: count for venue, count in gaps.items() if venue in JOURNAL_VENUES and count > 0}
    if incoming_needed:
        rows.extend(
            scan_incoming_papers(
                incoming_needed,
                existing_paper_ids={row.paper_id for row in rows},
            )
        )

    manifest_rows = _select_manifest_rows(rows)

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open('w', encoding='utf-8') as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row.to_dict()) + '\n')

    final_gaps = compute_gaps(manifest_rows)
    for venue, gap in final_gaps.items():
        if gap > 0:
            logger.warning('Unfilled venue gap: %s needs %d more paper(s)', venue, gap)

    logger.info('Wrote %d papers to %s', len(manifest_rows), MANIFEST_PATH)
    return manifest_rows
