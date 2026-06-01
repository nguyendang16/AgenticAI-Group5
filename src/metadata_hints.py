from __future__ import annotations

import re
from pathlib import Path


_DOMAIN_MAP = {
    'iclr': ['Machine Learning', 'Deep Learning'],
    'neurips': ['Machine Learning', 'AI'],
    'icml': ['Machine Learning'],
    'aaai': ['Artificial Intelligence'],
    'acl': ['Natural Language Processing', 'Computational Linguistics'],
    'chi': ['Human-Computer Interaction'],
    'twelf': ['Educational Technology', 'Learning Sciences'],
    'ets': ['Educational Technology'],
    'etrd': ['Educational Technology', 'Instructional Design'],
    'computers_and_education': ['Educational Technology', 'Learning Sciences'],
}


def _slug_token(text: str) -> str:
    token = re.sub(r'[^a-zA-Z0-9]+', '_', text.lower()).strip('_')
    return token or 'unknown'


def infer_metadata_from_path(path: Path) -> dict:
    name = path.stem.replace('_template', '')
    parts = name.split('_')
    kind = parts[0] if parts else 'unknown'
    file_name = path.name

    year_match = re.search(r'(20\d{2})', name)
    year = int(year_match.group(1)) if year_match else None

    venue_key = ''
    venue_display = ''
    source_type = 'guideline'
    article_types: list[str] = ['full_paper']

    if kind == 'conference':
        source_type = 'conference_guideline'
        if 'call_for_papers' in name or 'cfp' in name:
            source_type = 'call_for_papers'
            article_types = ['proposal', 'full_paper', 'short_paper']
        if 'review_form' in name:
            source_type = 'review_form'
        venue_parts = [p for p in parts[1:] if p not in {'review', 'reviewer', 'guidelines', 'guide', 'instructions', 'policies', 'arr', 'form', 'call', 'for', 'papers', 'cfp'} and not re.fullmatch(r'20\d{2}', p)]
        venue_key = venue_parts[0] if venue_parts else (parts[1] if len(parts) > 1 else 'conference')
        venue_display = venue_key.upper()
        if venue_key == 'acl':
            venue_display = 'ACL ARR'
    elif kind == 'journal':
        source_type = 'journal_guideline'
        journal_parts = [p for p in parts[1:] if p not in {'author', 'submission', 'guidelines', 'guide'}]
        venue_key = '_'.join(journal_parts) if journal_parts else 'journal'
        venue_display = venue_key.replace('_', ' ').title()
    else:
        venue_key = _slug_token(name)
        venue_display = venue_key.replace('_', ' ').title()

    domains: list[str] = []
    for key, domain_list in _DOMAIN_MAP.items():
        if key in name:
            domains.extend(domain_list)
    domains = list(dict.fromkeys(domains))

    source_id = _slug_token(name)
    title = venue_display
    if year:
        title = f'{title} {year}'
    if 'reviewer' in name:
        title += ' Reviewer Guideline'
    elif 'review_form' in name:
        title += ' Review Form'
    elif 'call_for_papers' in name:
        title += ' Call for Papers'
    elif kind == 'journal':
        title += ' Author Guidelines'

    return {
        'source_id': source_id,
        'source_title': title.strip(),
        'source_type': source_type,
        'venue_or_journal_name': venue_display,
        'publisher': venue_display if kind == 'conference' else '',
        'domain': domains,
        'year': year,
        'article_types': article_types,
        'file_name': file_name,
        'is_conference': kind == 'conference',
        'is_journal': kind == 'journal',
    }
