from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from src.criterion_ids import is_generic_criterion_id
from src.metadata_hints import infer_metadata_from_path
from src.normalizer import merge_duplicate_criteria, normalize_template
from src.schema import (
    EvidenceRequired,
    ExtractedTemplate,
    ReviewCriterion,
    ReviewFormField,
    SourceDocument,
)
from src.validator import validate_template


def _iter_block_items(document: Document):
    body = document.element.body
    for child in body.iterchildren():
        tag = child.tag.split('}')[-1]
        if tag == 'p':
            yield Paragraph(child, document)
        elif tag == 'tbl':
            yield Table(child, document)


def _clean(text: Any) -> str:
    return re.sub(r'\s+', ' ', str(text or '')).strip()


def _key(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


def _truthy(text: str) -> bool:
    return _clean(text).lower() in {'true', 'yes', 'y', 'required', '1'}


def _split_multi(text: str) -> list[str]:
    values = [_clean(part) for part in re.split(r'[;\n]+|\s+/\s+', text or '')]
    return [value for value in values if value]


def _rows_from_table(table: Table) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in table.rows:
        cells = [_clean(cell.text) for cell in row.cells]
        if any(cells):
            rows.append(cells)
    return rows


def _dict_rows(table: Table) -> tuple[list[str], list[dict[str, str]]]:
    rows = _rows_from_table(table)
    if not rows:
        return [], []

    headers = [_key(cell) for cell in rows[0]]
    mapped_rows: list[dict[str, str]] = []
    for row in rows[1:]:
        item: dict[str, str] = {}
        for index, header in enumerate(headers):
            item[header] = row[index] if index < len(row) else ''
        if any(item.values()):
            mapped_rows.append(item)
    return headers, mapped_rows


def _parse_metadata_from_paragraphs(paragraphs: list[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    in_metadata = False
    for paragraph in paragraphs:
        marker = paragraph.strip()
        if marker == 'SOURCE_METADATA':
            in_metadata = True
            continue
        if in_metadata and marker.isupper() and ':' not in marker:
            break
        if not in_metadata or ':' not in paragraph:
            continue

        raw_key, raw_value = paragraph.split(':', 1)
        metadata[_key(raw_key)] = _clean(raw_value)
    return metadata


def _parse_metadata_table(headers: list[str], rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {}
    if {'field', 'value'}.issubset(set(headers)):
        return {_key(row.get('field', '')): _clean(row.get('value', '')) for row in rows if row.get('field')}
    if {'key', 'value'}.issubset(set(headers)):
        return {_key(row.get('key', '')): _clean(row.get('value', '')) for row in rows if row.get('key')}
    return {}


def _metadata_value(metadata: dict[str, Any], *names: str) -> str:
    for name in names:
        value = _clean(metadata.get(_key(name), ''))
        if value:
            return value
    return ''


def _metadata_list(metadata: dict[str, Any], *names: str) -> list[str]:
    return _split_multi(_metadata_value(metadata, *names))


def _metadata_year(metadata: dict[str, Any], fallback: int | None) -> int | None:
    value = _metadata_value(metadata, 'year')
    match = re.search(r'20\d{2}', value)
    return int(match.group(0)) if match else fallback


def _build_source_document(path: Path, metadata: dict[str, Any]) -> SourceDocument:
    hints = infer_metadata_from_path(path)
    venue = _metadata_value(metadata, 'venue_or_journal', 'venue_or_journal_name')
    return SourceDocument(
        source_id=_metadata_value(metadata, 'source_id') or hints['source_id'],
        source_title=_metadata_value(metadata, 'source_title') or hints['source_title'],
        source_type=_metadata_value(metadata, 'source_type') or hints['source_type'],
        venue_or_journal_name=venue or hints['venue_or_journal_name'],
        publisher=_metadata_value(metadata, 'publisher') or hints.get('publisher', ''),
        domain=_metadata_list(metadata, 'domain') or hints.get('domain', []),
        year=_metadata_year(metadata, hints.get('year')),
        source_url=_metadata_value(metadata, 'source_url'),
        article_types=_metadata_list(metadata, 'article_type', 'article_types') or hints.get('article_types', []),
        file_name=path.name,
    )


def _criterion_evidence(criterion_id: str, evidence_text: str) -> list[EvidenceRequired]:
    evidence_items: list[EvidenceRequired] = []
    for evidence_name in _split_multi(evidence_text):
        evidence_items.append(
            EvidenceRequired(
                evidence_id='',
                name=evidence_name,
                description=f'Manuscript evidence for: {evidence_name}',
                evidence_type='manuscript_span',
                required=True,
            )
        )
    return evidence_items


def _parse_criteria(headers: list[str], rows: list[dict[str, str]]) -> list[ReviewCriterion]:
    if 'criterion_name' not in headers and 'name' not in headers:
        return []
    if 'criterion_group' not in headers and 'group' not in headers:
        return []

    criteria: list[ReviewCriterion] = []
    for index, row in enumerate(rows, start=1):
        criterion_name = row.get('criterion_name') or row.get('name') or ''
        if not criterion_name:
            continue
        criterion_id = row.get('criterion_id') or ''
        description = row.get('description') or row.get('criterion_description') or criterion_name
        evidence_text = row.get('evidence_required') or row.get('required_evidence') or ''
        criteria.append(
            ReviewCriterion(
                criterion_id=criterion_id,
                criterion_name=criterion_name,
                criterion_group=(row.get('criterion_group') or row.get('group') or 'OTHER').replace('/', ' '),
                description=description,
                evidence_required=_criterion_evidence(criterion_id or f'criterion_{index}', evidence_text),
                severity_if_missing=row.get('severity_if_missing') or 'major',
                source_quote=row.get('source_quote') or description,
                confidence='high',
            )
        )
    return criteria


def _parse_review_fields(headers: list[str], rows: list[dict[str, str]]) -> list[ReviewFormField]:
    if 'field_name' not in headers:
        return []

    fields: list[ReviewFormField] = []
    for row in rows:
        field_name = row.get('field_name') or ''
        if not field_name:
            continue
        maps_to = row.get('maps_to_criterion') or row.get('maps_to_criteria') or ''
        fields.append(
            ReviewFormField(
                field_id=row.get('field_id') or '',
                field_name=field_name,
                field_type=row.get('field_type') or 'text',
                description=row.get('description') or row.get('scale_or_values') or '',
                required=_truthy(row.get('required') or ''),
                maps_to_criteria=_split_multi(maps_to.replace(',', ';')),
            )
        )
    return fields


def _evidence_by_name(headers: list[str], rows: list[dict[str, str]]) -> dict[str, EvidenceRequired]:
    if 'evidence_name' not in headers:
        return {}
    evidence: dict[str, EvidenceRequired] = {}
    for index, row in enumerate(rows, start=1):
        name = row.get('evidence_name') or ''
        if not name:
            continue
        evidence[_key(name)] = EvidenceRequired(
            evidence_id=row.get('evidence_id') or f'evidence_{index}',
            name=name,
            description=row.get('description') or name,
            evidence_type=row.get('evidence_type') or 'manuscript_span',
            required=_truthy(row.get('required') or 'yes'),
        )
    return evidence


def extract_template_locally(path: Path) -> tuple[ExtractedTemplate, list[dict[str, Any]]]:
    document = Document(str(path))
    paragraphs: list[str] = []
    tables: list[tuple[list[str], list[dict[str, str]]]] = []

    for block in _iter_block_items(document):
        if isinstance(block, Paragraph):
            for line in block.text.splitlines():
                text = _clean(line)
                if text:
                    paragraphs.append(text)
        elif isinstance(block, Table):
            headers, rows = _dict_rows(block)
            if headers and rows:
                tables.append((headers, rows))

    metadata = _parse_metadata_from_paragraphs(paragraphs)
    criteria: list[ReviewCriterion] = []
    fields: list[ReviewFormField] = []
    named_evidence: dict[str, EvidenceRequired] = {}

    for headers, rows in tables:
        metadata.update({k: v for k, v in _parse_metadata_table(headers, rows).items() if v})
        criteria.extend(_parse_criteria(headers, rows))
        fields.extend(_parse_review_fields(headers, rows))
        named_evidence.update(_evidence_by_name(headers, rows))

    for criterion in criteria:
        enriched: list[EvidenceRequired] = []
        for ev in criterion.evidence_required:
            named = named_evidence.get(_key(ev.name))
            if named:
                enriched.append(
                    EvidenceRequired(
                        evidence_id=ev.evidence_id,
                        name=named.name,
                        description=named.description,
                        evidence_type=named.evidence_type,
                        required=named.required,
                    )
                )
            else:
                enriched.append(ev)
        criterion.evidence_required = enriched

    template = ExtractedTemplate(
        source_document=_build_source_document(path, metadata),
        review_criteria=criteria,
        review_form_fields=fields,
        checklist_items=[],
        graph_relations=[],
    )
    for index, criterion in enumerate(template.review_criteria, start=1):
        if is_generic_criterion_id(criterion.criterion_id):
            criterion.criterion_id = f'{template.source_document.source_id}:c{index:02d}'
    template = normalize_template(template)
    template, dup_warnings = merge_duplicate_criteria(template)
    warnings = dup_warnings + validate_template(template)
    return template, warnings
