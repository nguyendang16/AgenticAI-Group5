from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.criterion_ids import assign_semantic_criterion_ids

_REPO_ROOT = Path(__file__).resolve().parent.parent

VERIFICATION_STATUSES: tuple[str, ...] = (
    'SUPPORTED_BY_TEXT',
    'LITERATURE_SUPPORTED_NEEDS_QUALIFICATION',
    'MISSING_REQUIRED_EVIDENCE',
    'PARTIALLY_SUPPORTED',
    'UNSUPPORTED',
    'NEEDS_HUMAN_CHECK',
)

CONFIDENCE_LEVELS: tuple[str, ...] = ('High', 'Medium', 'Low')

HUMAN_CHECK_VALUES: tuple[str, ...] = ('Yes', 'No')

SCORING_EXCLUDED_GROUPS: frozenset[str] = frozenset({'OTHER'})

_CRITERION_ID_TOKEN_RE = re.compile(r'\b([A-Z]{2,12}_C\d{2}_[A-Z0-9_]+)\b')

_FORMAT_COMPLIANCE_HINT = re.compile(
    r'\b(format|presentation|page\s*length|pdf\s+submission|registration|presentation\s+requirements)\b',
    re.I,
)

_STATUS_ALIASES: dict[str, str] = {
    'supported': 'SUPPORTED_BY_TEXT',
    'supported by text': 'SUPPORTED_BY_TEXT',
    'supported_by_text': 'SUPPORTED_BY_TEXT',
    'literature supported': 'LITERATURE_SUPPORTED_NEEDS_QUALIFICATION',
    'literature_supported_needs_qualification': 'LITERATURE_SUPPORTED_NEEDS_QUALIFICATION',
    'missing': 'MISSING_REQUIRED_EVIDENCE',
    'missing required evidence': 'MISSING_REQUIRED_EVIDENCE',
    'missing_required_evidence': 'MISSING_REQUIRED_EVIDENCE',
    'not verifiable': 'MISSING_REQUIRED_EVIDENCE',
    'not verifiable from manuscript': 'MISSING_REQUIRED_EVIDENCE',
    'partial': 'PARTIALLY_SUPPORTED',
    'partially supported': 'PARTIALLY_SUPPORTED',
    'partially_supported': 'PARTIALLY_SUPPORTED',
    'unsupported': 'UNSUPPORTED',
    'needs human check': 'NEEDS_HUMAN_CHECK',
    'needs_human_check': 'NEEDS_HUMAN_CHECK',
    'human check': 'NEEDS_HUMAN_CHECK',
}

# Loose patterns: acronym + optional year digits (TWELF2026, ICLR2026). No trailing \b after digits
# so names like TWELF2026_Proposal still match (_ is a word char in Python regex).
_VENUE_HINTS: list[tuple[re.Pattern[str], str, str | None]] = [
    (re.compile(r'\bICLR\d*', re.I), 'ICLR', 'Machine Learning'),
    (re.compile(r'\bNeurIPS\d*|\bNIPS\d*', re.I), 'NeurIPS', 'Machine Learning'),
    (re.compile(r'\bICML\d*', re.I), 'ICML', 'Machine Learning'),
    (re.compile(r'\bAAAI[\s-]*\d*', re.I), 'AAAI', 'Artificial Intelligence'),
    (re.compile(r'\bACL\s+ARR\b|\bACL\d*|\bARR\d*', re.I), 'ACL ARR', 'Natural Language Processing'),
    (re.compile(r'\bACM\s+CHI\b|\bCHI\d*', re.I), 'CHI', 'Human-Computer Interaction'),
    (re.compile(r'\bTWELF\d*', re.I), 'TWELF', 'Educational Technology'),
]

_JOURNAL_HINTS: list[tuple[re.Pattern[str], str, str | None]] = [
    (re.compile(r'Computers\s*&\s*Education', re.I), 'Computers And Education', 'Educational Technology'),
    (re.compile(r'\bETR&D\b|\bETRD\b', re.I), 'Etrd', 'Educational Technology'),
    (re.compile(r'\bETS\b', re.I), 'Ets', 'Educational Technology'),
]


def _clean(value: Any) -> str:
    return str(value or '').strip()


def _first_non_empty(*values: Any) -> str:
    for value in values:
        token = _clean(value)
        if token:
            return token
    return ''


def infer_venue_journal_from_text(text: str, *, max_chars: int = 24000) -> dict[str, str]:
    """Heuristic venue/journal detection from any text (markdown, job title, filename)."""
    sample = (text or '')[:max_chars]
    for pattern, venue, domain in _VENUE_HINTS:
        if pattern.search(sample):
            out: dict[str, str] = {'venue': venue}
            if domain:
                out['domain'] = domain
            return out
    for pattern, journal, domain in _JOURNAL_HINTS:
        if pattern.search(sample):
            out = {'journal': journal}
            if domain:
                out['domain'] = domain
            return out
    return {}


def infer_venue_journal_from_markdown(markdown: str, *, max_chars: int = 12000) -> dict[str, str]:
    return infer_venue_journal_from_text(markdown, max_chars=max_chars)


def _build_hint_corpus(
    *,
    job_metadata: dict[str, Any] | None,
    extra_inference_text: str = '',
) -> str:
    """Title, filename, and form hints — searched before full paper body (avoids bibliography false positives)."""
    metadata = job_metadata if isinstance(job_metadata, dict) else {}
    chunks = [
        extra_inference_text,
        _clean(metadata.get('title')),
        _clean(metadata.get('source_pdf_name')),
        _clean(metadata.get('job_title')),
    ]
    return '\n'.join(chunk for chunk in chunks if chunk)


def _infer_venue_with_priority(
    *,
    paper_markdown: str,
    job_metadata: dict[str, Any] | None,
    extra_inference_text: str = '',
) -> dict[str, str]:
    hints = _build_hint_corpus(job_metadata=job_metadata, extra_inference_text=extra_inference_text)
    if hints.strip():
        found = infer_venue_journal_from_text(hints, max_chars=8000)
        if found:
            return found
    if paper_markdown.strip():
        return infer_venue_journal_from_text(paper_markdown)
    return {}


def merge_criteria_query(
    *,
    settings: Any,
    job_metadata: dict[str, Any] | None,
    paper_markdown: str,
    extra_inference_text: str = '',
) -> dict[str, str]:
    metadata = job_metadata if isinstance(job_metadata, dict) else {}
    inferred = (
        _infer_venue_with_priority(
            paper_markdown=paper_markdown,
            job_metadata=metadata,
            extra_inference_text=extra_inference_text,
        )
        if bool(getattr(settings, 'review_infer_venue_from_paper', True))
        else {}
    )

    venue = _first_non_empty(
        metadata.get('review_venue'),
        getattr(settings, 'review_venue', None),
        inferred.get('venue'),
    )
    journal = _first_non_empty(
        metadata.get('review_journal'),
        getattr(settings, 'review_journal', None),
        inferred.get('journal'),
    )
    domain = _first_non_empty(
        metadata.get('review_domain'),
        getattr(settings, 'review_domain', None),
        inferred.get('domain'),
    )
    article_type = _first_non_empty(
        metadata.get('review_article_type'),
        getattr(settings, 'review_article_type', None),
    )
    return {
        'venue': venue,
        'journal': journal,
        'domain': domain,
        'article_type': article_type,
    }


def _criteria_json_dir(settings: Any) -> Path:
    raw = getattr(settings, 'review_criteria_json_dir', None)
    path = Path(raw) if raw else Path('outputs/extracted_json')
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path


def _load_bundle_from_json_dir(
    json_dir: Path,
    *,
    venue: str,
    journal: str,
    domain: str,
    article_type: str,
) -> dict[str, Any] | None:
    if not json_dir.is_dir():
        return None

    venue_lower = venue.lower()
    journal_lower = journal.lower()
    domain_lower = domain.lower()

    best: dict[str, Any] | None = None
    best_score = -1

    for path in sorted(json_dir.glob('*.json')):
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        sd = payload.get('source_document') if isinstance(payload, dict) else None
        if not isinstance(sd, dict):
            continue
        host = _clean(sd.get('venue_or_journal_name'))
        host_lower = host.lower()
        score = 0
        if venue and venue_lower in host_lower:
            score += 3
        if journal and journal_lower in host_lower:
            score += 3
        if domain:
            domains = sd.get('domain') or []
            if isinstance(domains, list) and any(domain_lower in str(d).lower() for d in domains):
                score += 1
        if article_type:
            types = sd.get('article_types') or []
            if isinstance(types, list) and article_type.lower() in [str(t).lower() for t in types]:
                score += 1
        if score <= 0:
            continue
        criteria = payload.get('review_criteria') if isinstance(payload.get('review_criteria'), list) else []
        if not criteria:
            continue
        if score > best_score:
            best_score = score
            best = payload

    if best is None:
        return None

    from collections import defaultdict

    criteria_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    provenance: list[dict[str, str]] = []
    sd = best['source_document']
    sid = _clean(sd.get('source_id'))
    if sid:
        provenance.append({'source_id': sid, 'file_name': _clean(sd.get('file_name'))})

    for item in best.get('review_criteria', []):
        if not isinstance(item, dict):
            continue
        group = _clean(item.get('criterion_group')) or 'OTHER'
        criteria_by_group[group].append(
            {
                'criterion_id': item.get('criterion_id'),
                'criterion_name': item.get('criterion_name'),
                'criterion_group': group,
                'description': item.get('description'),
                'severity_if_missing': item.get('severity_if_missing'),
                'source_quote': item.get('source_quote'),
                'confidence': item.get('confidence'),
                'applies_to_domain': item.get('applies_to_domain') or [],
                'applies_to_article_type': item.get('applies_to_article_type') or [],
                'evidence_required': item.get('evidence_required') or [],
                'provenance': {'source_id': sid, 'file_name': _clean(sd.get('file_name'))},
            }
        )

    return {
        'query': {
            'venue': venue or None,
            'journal': journal or None,
            'domain': domain or None,
            'article_type': article_type or None,
        },
        'criteria_by_group': dict(criteria_by_group),
        'criteria_count': sum(len(v) for v in criteria_by_group.values()),
        'provenance': provenance,
        'verification_ready': {
            'must_reference_criterion_ids': True,
            'groups_present': list(criteria_by_group.keys()),
        },
        'source': 'json_fallback',
    }


def _criteria_query_variants(query: dict[str, str]) -> list[dict[str, str]]:
    """Alternate venue/journal placements when UI or inference uses the wrong host label."""
    variants: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()

    def _add(venue: str, journal: str, domain: str, article_type: str) -> None:
        key = (venue, journal, domain, article_type)
        if key in seen:
            return
        seen.add(key)
        variants.append(
            {
                'venue': venue,
                'journal': journal,
                'domain': domain,
                'article_type': article_type,
            }
        )

    venue = _clean(query.get('venue'))
    journal = _clean(query.get('journal'))
    domain = _clean(query.get('domain'))
    article_type = _clean(query.get('article_type'))

    _add(venue, journal, domain, article_type)
    # e.g. TWELF2026 is stored as Journal in Neo4j but often entered in the venue field
    if venue and not journal:
        _add('', venue, domain, article_type)
    if journal and not venue:
        _add(journal, '', domain, article_type)
    if venue.upper().startswith('TWELF'):
        for jname in ('TWELF2026', 'TWELF'):
            if jname != journal:
                _add('', jname, domain, article_type)
    return variants


def _retrieve_from_neo4j(
    *,
    venue: str,
    journal: str,
    domain: str,
    article_type: str,
    settings: Any,
) -> dict[str, Any] | None:
    uri = _clean(getattr(settings, 'neo4j_uri', None))
    password = _clean(getattr(settings, 'neo4j_password', None))
    if not uri or not password:
        return None
    try:
        from src.retriever import retrieve_criteria_bundle

        bundle = retrieve_criteria_bundle(
            venue=venue or None,
            journal=journal or None,
            domain=domain or None,
            article_type=article_type or None,
        )
        if int(bundle.get('criteria_count') or 0) > 0:
            bundle['source'] = 'neo4j'
            return bundle
    except Exception:
        return None
    return None


def resolve_review_criteria_bundle(
    *,
    settings: Any,
    paper_markdown: str,
    job_metadata: dict[str, Any] | None = None,
    extra_inference_text: str = '',
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Resolve venue criteria for the review agent. Never raises."""
    resolution: dict[str, Any] = {
        'enabled': bool(getattr(settings, 'review_criteria_enabled', True)),
        'query': {},
        'source': None,
        'criteria_count': 0,
        'error': None,
    }
    if not resolution['enabled']:
        resolution['skipped'] = 'disabled'
        return None, resolution

    query = merge_criteria_query(
        settings=settings,
        job_metadata=job_metadata,
        paper_markdown=paper_markdown,
        extra_inference_text=extra_inference_text,
    )
    resolution['query'] = query
    if inferred_hint := _infer_venue_with_priority(
        paper_markdown=paper_markdown,
        job_metadata=job_metadata if isinstance(job_metadata, dict) else {},
        extra_inference_text=extra_inference_text,
    ):
        resolution['inferred'] = inferred_hint

    if not any(query.values()):
        resolution['skipped'] = 'no_venue_or_journal'
        return None, resolution

    bundle = None
    resolved_query = query
    for variant in _criteria_query_variants(query):
        bundle = _retrieve_from_neo4j(
            venue=variant['venue'],
            journal=variant['journal'],
            domain=variant['domain'],
            article_type=variant['article_type'],
            settings=settings,
        )
        if bundle is not None:
            resolved_query = variant
            break
    if bundle is None:
        for variant in _criteria_query_variants(query):
            bundle = _load_bundle_from_json_dir(
                _criteria_json_dir(settings),
                venue=variant['venue'],
                journal=variant['journal'],
                domain=variant['domain'],
                article_type=variant['article_type'],
            )
            if bundle is not None:
                resolved_query = variant
                break

    resolution['query'] = resolved_query
    if resolved_query != query:
        resolution['query_fallback'] = True

    if bundle is None:
        resolution['skipped'] = 'no_criteria_found'
        return None, resolution

    bundle = enrich_criteria_bundle(bundle)
    resolution['source'] = bundle.get('source')
    resolution['criteria_count'] = int(bundle.get('criteria_count') or 0)
    return bundle, resolution


def is_scoring_active_criterion(item: dict[str, Any]) -> bool:
    group = _clean(item.get('criterion_group')).upper()
    if group in SCORING_EXCLUDED_GROUPS:
        return False
    return not bool(item.get('exclude_from_scoring'))


def normalize_criterion_metadata(item: dict[str, Any]) -> dict[str, Any]:
    """Fix groups (format vs scope) and mark routing-only criteria as non-scoring."""
    row = dict(item)
    name = _clean(row.get('criterion_name'))
    desc = _clean(row.get('description'))
    group = _clean(row.get('criterion_group')) or 'OTHER'
    combined = f'{name} {desc}'
    if group == 'SCOPE_FIT' and _FORMAT_COMPLIANCE_HINT.search(combined):
        row['criterion_group'] = 'FORMAT_COMPLIANCE'
    elif group == 'OTHER':
        row['exclude_from_scoring'] = True
    return row


def enrich_criteria_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Normalize groups, assign semantic IDs, and rebuild criteria_by_group."""
    query = bundle.get('query') if isinstance(bundle.get('query'), dict) else {}
    venue = _clean(query.get('venue'))
    journal = _clean(query.get('journal'))
    groups = bundle.get('criteria_by_group')
    if not isinstance(groups, dict):
        return bundle

    flat: list[dict[str, Any]] = []
    seen_flat: set[str] = set()
    for group_name in sorted(groups.keys()):
        for item in groups[group_name]:
            if isinstance(item, dict):
                row = normalize_criterion_metadata(dict(item))
                row.setdefault('criterion_group', group_name)
                cid = _clean(row.get('criterion_id'))
                dedupe_key = cid or f"{row.get('criterion_name')}|{row.get('description')}"
                if dedupe_key in seen_flat:
                    continue
                seen_flat.add(dedupe_key)
                flat.append(row)

    remapped = assign_semantic_criterion_ids(
        flat,
        venue=venue,
        journal=journal,
        always_reassign=True,
    )
    id_map = {
        _clean(old.get('criterion_id')): _clean(new.get('criterion_id'))
        for old, new in zip(flat, remapped, strict=True)
        if _clean(old.get('criterion_id')) and _clean(new.get('criterion_id'))
    }

    new_groups: dict[str, list[dict[str, Any]]] = {}
    for item in remapped:
        group = _clean(item.get('criterion_group')) or 'OTHER'
        new_groups.setdefault(group, []).append(item)

    out = dict(bundle)
    out['criteria_by_group'] = new_groups
    out['criteria_count'] = sum(len(v) for v in new_groups.values())
    if id_map:
        out['criterion_id_map'] = id_map
    out['evaluation'] = build_graph_evaluation_summary(out)
    return out


def enrich_criteria_bundle_semantic_ids(bundle: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible alias for enrich_criteria_bundle."""
    return enrich_criteria_bundle(bundle)


def _source_template_label(item: dict[str, Any]) -> str:
    prov = item.get('provenance') if isinstance(item.get('provenance'), dict) else {}
    return _clean(prov.get('file_name')) or _clean(prov.get('source_id')) or '-'


def _short_criterion_id(cid: str) -> str:
    """Extract short ID like C01 from TWELF_C01_CLARITY_PRESENTATION, or keep original if short."""
    match = re.search(r'(C\d{2})', cid)
    if match:
        return match.group(1)
    return cid if len(cid) <= 12 else cid[:12]


def _short_group_label(group: str) -> str:
    """Shorten group name: CLARITY_PRESENTATION -> CLARITY."""
    parts = group.split('_')
    return parts[0] if parts else group


def build_graph_evaluation_summary(bundle: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize the KG retrieval result for reports and UI progress."""
    if not bundle:
        return {
            'status': 'not_available',
            'criteria_count': 0,
            'active_criteria_count': 0,
            'reference_only_count': 0,
            'source_document_count': 0,
            'source_documents': [],
            'groups_present': [],
            'evidence_requirements_count': 0,
        }

    groups = bundle.get('criteria_by_group') if isinstance(bundle.get('criteria_by_group'), dict) else {}
    criteria: list[dict[str, Any]] = []
    for group_items in groups.values():
        if isinstance(group_items, list):
            criteria.extend(item for item in group_items if isinstance(item, dict))

    active_count = sum(1 for item in criteria if is_scoring_active_criterion(item))
    reference_count = max(0, len(criteria) - active_count)
    evidence_count = 0
    for item in criteria:
        evidence = item.get('evidence_required')
        if isinstance(evidence, list):
            evidence_count += len(evidence)

    provenance = bundle.get('provenance') if isinstance(bundle.get('provenance'), list) else []
    seen_sources: set[tuple[str, str]] = set()
    source_documents: list[dict[str, str]] = []
    for row in provenance:
        if not isinstance(row, dict):
            continue
        source_id = _clean(row.get('source_id'))
        file_name = _clean(row.get('file_name'))
        key = (source_id, file_name)
        if key in seen_sources:
            continue
        seen_sources.add(key)
        source_documents.append({'source_id': source_id, 'file_name': file_name})

    return {
        'status': 'ready' if criteria else 'empty',
        'criteria_count': len(criteria),
        'active_criteria_count': active_count,
        'reference_only_count': reference_count,
        'source_document_count': len(source_documents),
        'source_documents': source_documents,
        'groups_present': sorted(groups.keys()),
        'evidence_requirements_count': evidence_count,
    }


def normalize_verification_status(raw: str) -> str:
    token = _clean(raw)
    if not token:
        return 'NEEDS_HUMAN_CHECK'
    upper = token.upper().replace(' ', '_').replace('-', '_')
    if upper in VERIFICATION_STATUSES:
        return upper
    lower = token.lower().rstrip('.')
    if lower in _STATUS_ALIASES:
        return _STATUS_ALIASES[lower]
    for key, canonical in _STATUS_ALIASES.items():
        if key in lower:
            return canonical
    return 'NEEDS_HUMAN_CHECK'


def normalize_confidence(raw: str) -> str:
    token = _clean(raw)
    if not token:
        return 'Medium'
    upper = token.upper().replace(' ', '_')
    if upper in VERIFICATION_STATUSES or upper == 'NEEDS_HUMAN_CHECK':
        return 'Medium'
    lower = token.lower()
    if lower in {'high', 'h'}:
        return 'High'
    if lower in {'medium', 'med', 'm'}:
        return 'Medium'
    if lower in {'low', 'l'}:
        return 'Low'
    if token in CONFIDENCE_LEVELS:
        return token
    return 'Medium'


def normalize_human_check(raw: str) -> str:
    token = _clean(raw).lower()
    if token in {'yes', 'y', 'true', 'required'}:
        return 'Yes'
    if token in {'no', 'n', 'false', 'not required'}:
        return 'No'
    return 'No'


def _extract_criterion_id_token(cell: str) -> str:
    match = _CRITERION_ID_TOKEN_RE.search(cell)
    return match.group(1) if match else _clean(cell.split('—')[0].split('-')[0])


def _find_column_index(headers: list[str], *needles: str) -> int | None:
    lowered = [h.lower() for h in headers]
    for idx, header in enumerate(lowered):
        if all(needle in header for needle in needles):
            return idx
    for idx, header in enumerate(lowered):
        if any(needle in header for needle in needles):
            return idx
    return None


def _repair_audit_pipe_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith('- '):
        stripped = stripped[2:].strip()
    if '|' not in stripped:
        return None
    if not stripped.startswith('|'):
        stripped = f'| {stripped}'
    if not stripped.endswith('|'):
        stripped = f'{stripped} |'
    cells = [cell.strip() for cell in stripped.strip('|').split('|')]
    cleaned_cells = []
    for cell in cells:
        if cell.startswith('- '):
            cell = cell[2:].strip()
        cleaned_cells.append(cell)
    return '| ' + ' | '.join(cleaned_cells) + ' |'


def _default_human_check_for_status(status: str) -> str:
    if status in {'NEEDS_HUMAN_CHECK', 'LITERATURE_SUPPORTED_NEEDS_QUALIFICATION', 'PARTIALLY_SUPPORTED'}:
        return 'Yes'
    if status == 'MISSING_REQUIRED_EVIDENCE':
        return 'No'
    return 'No'


def _default_confidence_for_status(status: str) -> str:
    if status == 'MISSING_REQUIRED_EVIDENCE':
        return 'High'
    if status in {'LITERATURE_SUPPORTED_NEEDS_QUALIFICATION', 'PARTIALLY_SUPPORTED'}:
        return 'Medium'
    if status == 'SUPPORTED_BY_TEXT':
        return 'High'
    return 'Medium'


def _column_index_map(headers: list[str]) -> dict[str, int | None]:
    return {
        'id': _find_column_index(headers, 'id') or _find_column_index(headers, 'critique'),
        'critique': _find_column_index(headers, 'critique'),
        'criterion': _find_column_index(headers, 'crit'),
        'evidence': _find_column_index(headers, 'evid'),
        'status': _find_column_index(headers, 'status') or _find_column_index(headers, 'verification'),
        'confidence': _find_column_index(headers, 'conf'),
        'human': _find_column_index(headers, 'human'),
        'fix': _find_column_index(headers, 'fix') or _find_column_index(headers, 'suggest'),
    }


def _cell_at(cells: list[str], index: int | None) -> str:
    if index is None or index < 0 or index >= len(cells):
        return ''
    return _clean(cells[index])


def normalize_claim_level_audit_markdown(
    audit_text: str,
    *,
    criteria_bundle: dict[str, Any] | None = None,
) -> str:
    """Repair and normalize Claim-Level Audit tables (status, confidence, human check)."""
    del criteria_bundle  # reserved for future criterion-id validation
    text = str(audit_text or '').strip()
    if not text:
        return text

    table_lines: list[str] = []
    preamble: list[str] = []
    for line in text.splitlines():
        repaired = _repair_audit_pipe_line(line)
        if repaired:
            table_lines.append(repaired)
        elif not table_lines:
            preamble.append(line)

    if not table_lines:
        return text

    header_cells = [c.strip() for c in table_lines[0].strip('|').split('|')]
    data_start = 1
    if len(table_lines) > 1:
        sep_cells = [c.strip() for c in table_lines[1].strip('|').split('|')]
        if sep_cells and all(set(c) <= {'-', ':'} for c in sep_cells):
            data_start = 2

    colmap = _column_index_map(header_cells)
    compact_header = '| ID | Evidence | Status | Conf | Fix |'
    compact_sep = '| --- | --- | --- | --- | --- |'
    out_rows = [compact_header, compact_sep]

    for line in table_lines[data_start:]:
        cells = [c.strip() for c in line.strip('|').split('|')]
        if not cells or all(not c for c in cells):
            continue

        row_id = _cell_at(cells, colmap['id']) or _cell_at(cells, 0)
        if row_id.lower().startswith('critique id'):
            continue

        evidence = _cell_at(cells, colmap['evidence'])
        if not evidence and len(cells) > 3:
            evidence = cells[3]

        status_raw = _cell_at(cells, colmap['status'])
        if not status_raw:
            for cell in cells:
                if normalize_verification_status(cell) in VERIFICATION_STATUSES and cell.upper() == normalize_verification_status(cell):
                    status_raw = cell
                    break
        status = normalize_verification_status(status_raw)

        conf_raw = _cell_at(cells, colmap['confidence'])
        if conf_raw and normalize_verification_status(conf_raw) in VERIFICATION_STATUSES:
            misplaced_status = normalize_verification_status(conf_raw)
            if not status_raw:
                status = misplaced_status
            confidence = _default_confidence_for_status(status)
        else:
            confidence = normalize_confidence(conf_raw) if conf_raw else _default_confidence_for_status(status)

        fix = _cell_at(cells, colmap['fix']) or (cells[-1] if cells else '')
        if fix.upper() in VERIFICATION_STATUSES or fix in CONFIDENCE_LEVELS:
            fix = _cell_at(cells, colmap['fix'])

        short_status = status.replace('_', ' ').replace('MISSING REQUIRED EVIDENCE', 'Missing').replace('LITERATURE SUPPORTED NEEDS QUALIFICATION', 'Lit. Support').replace('PARTIALLY SUPPORTED', 'Partial').replace('SUPPORTED BY TEXT', 'Supported').replace('NEEDS HUMAN CHECK', 'Check').replace('UNSUPPORTED', 'Unsupported')
        short_conf = confidence[0] if confidence else 'M'
        short_evidence = evidence[:100] + '...' if len(evidence) > 100 else evidence
        out_rows.append(
            f'| {row_id} | {short_evidence} | {short_status} | {short_conf} | {fix} |'
        )

    if len(out_rows) <= 2:
        return text

    body = '\n'.join(out_rows)
    footer = '\n\n_ID = Criterion from Legend (C01-C05). Conf = H (High), M (Medium), L (Low)._'
    if preamble:
        return '\n'.join(preamble).strip() + '\n' + body + footer
    return body + footer


def format_criteria_bundle_for_prompt(
    bundle: dict[str, Any] | None,
    *,
    max_criteria: int = 48,
    max_description_chars: int = 280,
) -> str:
    if not bundle or int(bundle.get('criteria_count') or 0) <= 0:
        return ''

    query = bundle.get('query') if isinstance(bundle.get('query'), dict) else {}
    lines = [
        '[VENUE REVIEW CRITERIA — KNOWLEDGE GRAPH]',
        'Use these venue/journal-specific criteria to structure your review.',
        'When a weakness clearly maps to a criterion, cite its criterion_id in the final report '
        '(e.g. under Key Issues or Actionable Suggestions).',
        'Use each criterion only for its stated meaning; do not repurpose reviewer-process criteria '
        'as author submission requirements.',
        'Do not invent criterion IDs; only use IDs listed below.',
        'Skip criteria that clearly do not apply to this paper.',
        'Do NOT map critiques, annotations, or scores to routing-only criteria (group OTHER / SIG session).',
        'A Criterion Legend (bullet list) is auto-prepended to the saved report.',
        '',
        f"Query: venue={query.get('venue') or '-'} journal={query.get('journal') or '-'} "
        f"domain={query.get('domain') or '-'} article_type={query.get('article_type') or '-'}",
        f"Total criteria: {bundle.get('criteria_count', 0)}",
        '',
        '### Active criteria (use for critiques, annotations, and scores)',
        '',
    ]

    emitted = 0
    groups = bundle.get('criteria_by_group') if isinstance(bundle.get('criteria_by_group'), dict) else {}
    reference_items: list[dict[str, Any]] = []
    for group in sorted(groups.keys()):
        for item in groups[group]:
            if isinstance(item, dict) and not is_scoring_active_criterion(item):
                reference_items.append(item)

    for group in sorted(groups.keys()):
        if emitted >= max_criteria:
            break
        for item in groups[group]:
            if emitted >= max_criteria:
                break
            if not isinstance(item, dict) or not is_scoring_active_criterion(item):
                continue
            cid = _clean(item.get('criterion_id'))
            short_id = _short_criterion_id(cid)
            name = _clean(item.get('criterion_name'))
            desc = _clean(item.get('description'))
            if len(desc) > max_description_chars:
                desc = f'{desc[:max_description_chars]}…'
            grp = _short_group_label(_clean(item.get('criterion_group')) or group)
            severity = _clean(item.get('severity_if_missing')) or 'major'
            lines.append(f'- **{cid}** = {short_id} ({grp}): {name} — {desc} [severity={severity}]')
            emitted += 1

    if reference_items:
        lines.extend(['', '### Reference only (do not map critiques or scores)', ''])
        for item in reference_items:
            cid = _clean(item.get('criterion_id'))
            short_id = _short_criterion_id(cid)
            name = _clean(item.get('criterion_name'))
            lines.append(f'- {short_id}: {name} [reference only]')
        lines.append('')

    if emitted >= max_criteria:
        lines.append(f'[... truncated to {max_criteria} criteria for context ...]')
        lines.append('')

    return '\n'.join(lines).strip() + '\n\n'


def format_criteria_legend_markdown(bundle: dict[str, Any] | None) -> str:
    """Criterion legend as compact bullet lists (PDF-friendly)."""
    if not bundle or int(bundle.get('criteria_count') or 0) <= 0:
        return ''

    query = bundle.get('query') if isinstance(bundle.get('query'), dict) else {}
    venue = query.get('venue') or 'VENUE'
    lines = [
        '## Criterion Legend',
        '',
        f"Venue: {venue} | Domain: {query.get('domain') or '-'}",
        '',
    ]
    groups = bundle.get('criteria_by_group') if isinstance(bundle.get('criteria_by_group'), dict) else {}
    reference: list[str] = []
    for group in sorted(groups.keys()):
        for item in groups[group]:
            if not isinstance(item, dict):
                continue
            cid = _clean(item.get('criterion_id'))
            short_id = _short_criterion_id(cid)
            name = _clean(item.get('criterion_name'))
            grp = _short_group_label(_clean(item.get('criterion_group')) or group)
            desc = _clean(item.get('description'))
            if len(desc) > 150:
                desc = f'{desc[:150]}…'
            entry = f'- **{short_id}** ({grp}): {name} — {desc}'
            if is_scoring_active_criterion(item):
                lines.append(entry)
            else:
                reference.append(f'- {short_id} ({grp}): {name} [reference only]')

    if reference:
        lines.extend(['', '**Reference only:**'])
        lines.extend(reference)
    lines.append('')
    return '\n'.join(lines)


def format_graph_evaluation_markdown(bundle: dict[str, Any] | None) -> str:
    """Compact report section explaining what the criteria graph returned."""
    if not bundle or int(bundle.get('criteria_count') or 0) <= 0:
        return ''

    evaluation = bundle.get('evaluation')
    if not isinstance(evaluation, dict):
        evaluation = build_graph_evaluation_summary(bundle)

    query = bundle.get('query') if isinstance(bundle.get('query'), dict) else {}
    venue = _clean(query.get('venue')) or '-'
    journal = _clean(query.get('journal')) or '-'
    domain = _clean(query.get('domain')) or '-'
    article_type = _clean(query.get('article_type')) or '-'
    groups = evaluation.get('groups_present') if isinstance(evaluation.get('groups_present'), list) else []
    source_documents = (
        evaluation.get('source_documents') if isinstance(evaluation.get('source_documents'), list) else []
    )

    lines = [
        '## KG Criteria Result',
        '',
        f'- Query: venue={venue}; journal={journal}; domain={domain}; article_type={article_type}',
        f"- Criteria returned: {int(evaluation.get('criteria_count') or 0)} "
        f"({int(evaluation.get('active_criteria_count') or 0)} active, "
        f"{int(evaluation.get('reference_only_count') or 0)} reference-only)",
        f"- Evidence requirements linked: {int(evaluation.get('evidence_requirements_count') or 0)}",
    ]
    if groups:
        lines.append(f"- Groups: {', '.join(str(group) for group in groups)}")
    if source_documents:
        lines.append('- Source documents / literature:')
        for row in source_documents[:8]:
            if not isinstance(row, dict):
                continue
            source_id = _clean(row.get('source_id')) or '-'
            file_name = _clean(row.get('file_name')) or '-'
            lines.append(f'  - {file_name} ({source_id})')
    else:
        lines.append('- Source documents / literature: none returned by graph')
    lines.append('')
    return '\n'.join(lines)


_INLINE_SECTION_LINE = re.compile(
    r'^\s*(?:[-*]\s*)?(?P<title>summary|strengths|weaknesses|key issues|issues|'
    r'actionable suggestions|suggestions|claim[- ]level audit|audit table|scores)\s*:?\s*$',
    re.I,
)

_NESTED_SECTION_MARKERS = (
    '- strengths:',
    '- weaknesses:',
    '- key issues:',
    '- actionable suggestions:',
    '- scores:',
    'strengths:',
    'weaknesses:',
    'key issues:',
    '## strengths',
    '## weaknesses',
    '## key issues',
    '## actionable suggestions',
    '## scores',
    '## claim-level audit',
)


def _inline_section_id(title: str, *, review_fast_mode: bool) -> str | None:
    token = _clean(title).lower()
    mapping = {
        'summary': 'summary',
        'strengths': 'strengths',
        'weaknesses': 'weaknesses',
        'key issues': 'key_issues',
        'issues': 'key_issues',
        'actionable suggestions': 'actionable_suggestions',
        'suggestions': 'actionable_suggestions',
        'claim-level audit': 'claim_level_audit',
        'claim level audit': 'claim_level_audit',
        'audit table': 'claim_level_audit',
        'scores': 'scores',
    }
    if not review_fast_mode:
        mapping.pop('claim-level audit', None)
        mapping.pop('claim level audit', None)
        mapping.pop('audit table', None)
    return mapping.get(token)


def split_inline_sections_from_summary(
    summary: str,
    *,
    review_fast_mode: bool = True,
) -> dict[str, str]:
    """Parse summary blobs that embed Strengths/Weaknesses/etc. as plain headings."""
    text = str(summary or '').strip()
    if not text:
        return {}

    lines = text.splitlines()
    sections: dict[str, list[str]] = {}
    current_id: str | None = None

    for line in lines:
        stripped = line.strip()
        match = _INLINE_SECTION_LINE.match(stripped)
        if match:
            section_id = _inline_section_id(match.group('title'), review_fast_mode=review_fast_mode)
            if section_id:
                current_id = section_id
                sections.setdefault(current_id, [])
                continue
        if current_id:
            sections.setdefault(current_id, []).append(line)

    if len(sections) < 2:
        return {}

    return {sid: '\n'.join(buf).strip() for sid, buf in sections.items() if '\n'.join(buf).strip()}


def redistribute_stuffed_summary_sections(
    sections: dict[str, str],
    *,
    review_fast_mode: bool = True,
) -> dict[str, str]:
    """Move inline subsections out of summary into dedicated section keys."""
    cleaned = {key: str(value or '').strip() for key, value in sections.items() if str(value or '').strip()}
    summary = cleaned.get('summary', '')
    if not summary:
        return cleaned

    parsed = split_inline_sections_from_summary(summary, review_fast_mode=review_fast_mode)
    if not parsed:
        return cleaned

    out = dict(cleaned)
    for section_id, content in parsed.items():
        if section_id == 'summary':
            out['summary'] = content
        elif section_id not in out or len(content) > len(out.get(section_id, '')):
            out[section_id] = content
    return out


def _summary_contains_nested_sections(summary: str) -> bool:
    lower = summary.lower()
    hits = sum(1 for marker in _NESTED_SECTION_MARKERS if marker in lower)
    if hits >= 2:
        return True
    return bool(split_inline_sections_from_summary(summary))


def sanitize_fast_report_sections(sections: dict[str, str]) -> dict[str, str]:
    """Drop duplicate full-report content stuffed into the summary section."""
    cleaned = redistribute_stuffed_summary_sections(sections, review_fast_mode=True)
    cleaned = {key: str(value or '').strip() for key, value in cleaned.items() if str(value or '').strip()}
    summary = cleaned.get('summary', '')
    if not summary:
        return cleaned

    if _summary_contains_nested_sections(summary):
        trimmed: list[str] = []
        for line in summary.splitlines():
            lower = line.strip().lower()
            if _INLINE_SECTION_LINE.match(line.strip()):
                break
            if any(lower.startswith(marker) for marker in _NESTED_SECTION_MARKERS):
                break
            trimmed.append(line)
        new_summary = '\n'.join(trimmed).strip()
        if new_summary:
            cleaned['summary'] = new_summary

    other_keys = [key for key in cleaned if key != 'summary']
    if len(other_keys) >= 2 and cleaned.get('summary'):
        nested = split_inline_sections_from_summary(cleaned['summary'])
        if nested and len(nested) >= 2:
            cleaned['summary'] = nested.get('summary', cleaned['summary'])
    return cleaned


def build_final_report_markdown(
    sections: dict[str, str],
    *,
    section_order: list[str],
    section_titles: dict[str, str],
    criteria_bundle: dict[str, Any] | None = None,
    review_fast_mode: bool = False,
) -> str:
    """Assemble report with optional criterion legend prepended."""
    normalized = sanitize_fast_report_sections(sections) if review_fast_mode else sections
    if review_fast_mode and 'claim_level_audit' in normalized:
        normalized = dict(normalized)
        normalized['claim_level_audit'] = normalize_claim_level_audit_markdown(
            normalized['claim_level_audit'],
            criteria_bundle=criteria_bundle,
        )
    blocks: list[str] = []
    legend = format_criteria_legend_markdown(criteria_bundle)
    if legend:
        blocks.append(legend.strip())
    graph_evaluation = format_graph_evaluation_markdown(criteria_bundle)
    if graph_evaluation:
        blocks.append(graph_evaluation.strip())
    for section_id in section_order:
        content = str(normalized.get(section_id) or '').strip()
        if not content:
            continue
        heading = section_titles.get(section_id, section_id)
        blocks.append(f'## {heading}\n{content}')
    return '\n\n'.join(blocks).strip()
