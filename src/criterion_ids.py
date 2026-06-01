from __future__ import annotations

import re

_GENERIC_CRITERION_ID = re.compile(r'^criterion_\d+$', re.I)


def venue_prefix_from_name(name: str) -> str:
    """Derive a short venue acronym (TWELF, CHI, ICLR) from a venue/journal label."""
    token = re.sub(r'[^a-zA-Z0-9]+', '', str(name or '').strip())
    if not token:
        return 'VENUE'
    upper = token.upper()
    m = re.match(r'^([A-Z]{2,12})', upper)
    if m:
        return m.group(1)
    return upper[:8] if len(upper) > 8 else upper


def venue_prefix_from_query(venue: str = '', journal: str = '') -> str:
    return venue_prefix_from_name(venue or journal or 'VENUE')


def is_generic_criterion_id(criterion_id: str) -> bool:
    cid = str(criterion_id or '').strip()
    if not cid:
        return True
    if _GENERIC_CRITERION_ID.match(cid):
        return True
    if re.search(r'(?:^|:)criterion_\d+$', cid, re.I):
        return True
    return False


def semantic_criterion_id(*, venue_prefix: str, index: int, criterion_group: str) -> str:
    group_token = re.sub(r'[^A-Z0-9]+', '_', str(criterion_group or 'OTHER').upper()).strip('_')
    if not group_token:
        group_token = 'OTHER'
    prefix = venue_prefix_from_name(venue_prefix)
    return f'{prefix}_C{int(index):02d}_{group_token}'


def assign_semantic_criterion_ids(
    criteria: list[dict],
    *,
    venue: str = '',
    journal: str = '',
    always_reassign: bool = False,
) -> list[dict]:
    """Return criteria copies with stable venue-prefixed IDs when IDs are generic."""
    prefix = venue_prefix_from_query(venue, journal)
    out: list[dict] = []
    for index, item in enumerate(criteria, start=1):
        if not isinstance(item, dict):
            continue
        row = dict(item)
        cid = str(row.get('criterion_id') or '').strip()
        group = str(row.get('criterion_group') or 'OTHER').strip() or 'OTHER'
        if always_reassign or not cid or is_generic_criterion_id(cid):
            row['criterion_id'] = semantic_criterion_id(
                venue_prefix=prefix,
                index=index,
                criterion_group=group,
            )
        out.append(row)
    return out
