from __future__ import annotations

import re
from typing import Any

from src.normalizer import _slug

_SOURCE_KEYS = (
    'source_id',
    'source_title',
    'source_type',
    'venue_or_journal_name',
    'publisher',
    'domain',
    'year',
    'source_url',
    'article_types',
    'file_name',
)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    token = str(value).strip().lower()
    if token in {'true', 'yes', '1', 'required', 'mandatory'}:
        return True
    if token in {'false', 'no', '0', 'optional', 'conditional', 'n/a', 'na', ''}:
        return False
    return bool(value)


def _coerce_year(value: Any) -> int | None:
    if value is None or value == '':
        return None
    if isinstance(value, int):
        return value
    match = re.search(r'(20\d{2})', str(value))
    return int(match.group(1)) if match else None


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _coerce_source_document(payload: dict[str, Any], hints: dict[str, Any]) -> dict[str, Any]:
    sd = payload.get('source_document')
    if not isinstance(sd, dict):
        sd = {}
    merged = dict(sd)
    for key in _SOURCE_KEYS:
        hint_val = hints.get(key)
        if hint_val in (None, '', []):
            continue
        if not merged.get(key):
            merged[key] = hint_val
    merged.setdefault('file_name', hints.get('file_name', ''))
    merged.setdefault('source_id', hints.get('source_id', _slug(merged.get('file_name', 'unknown'))))
    merged.setdefault('source_title', hints.get('source_title', merged['source_id']))
    merged.setdefault('source_type', hints.get('source_type', 'guideline'))
    merged['domain'] = _coerce_str_list(merged.get('domain') or hints.get('domain'))
    merged['article_types'] = _coerce_str_list(merged.get('article_types') or hints.get('article_types'))
    merged['year'] = _coerce_year(merged.get('year') if merged.get('year') is not None else hints.get('year'))
    payload['source_document'] = merged
    return payload


def _coerce_evidence_items(items: Any, *, criterion_id: str) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        eid = str(item.get('evidence_id') or item.get('id') or f'{criterion_id}:ev{index}').strip()
        name = str(item.get('name') or item.get('evidence_name') or eid).strip()
        out.append(
            {
                'evidence_id': eid,
                'name': name,
                'description': str(item.get('description') or ''),
                'evidence_type': str(item.get('evidence_type') or 'manuscript_span'),
                'required': _parse_bool(item.get('required', True)),
            }
        )
    return out


def _coerce_review_criteria(payload: dict[str, Any], *, prefix: str) -> None:
    raw = payload.get('review_criteria')
    if not isinstance(raw, list):
        payload['review_criteria'] = []
        return
    criteria: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        name = str(item.get('criterion_name') or item.get('name') or f'Criterion {index}').strip()
        cid = str(item.get('criterion_id') or '').strip() or f'{prefix}:c{index}'
        description = str(item.get('description') or name).strip()
        criteria.append(
            {
                'criterion_id': cid,
                'criterion_name': name,
                'criterion_group': item.get('criterion_group') or 'OTHER',
                'description': description,
                'applies_to_domain': _coerce_str_list(item.get('applies_to_domain')),
                'applies_to_article_type': _coerce_str_list(item.get('applies_to_article_type')),
                'evidence_required': _coerce_evidence_items(
                    item.get('evidence_required'), criterion_id=cid
                ),
                'severity_if_missing': item.get('severity_if_missing') or 'major',
                'source_quote': str(item.get('source_quote') or '')[:200],
                'confidence': item.get('confidence') or 'medium',
            }
        )
    payload['review_criteria'] = criteria


def _coerce_review_form_fields(payload: dict[str, Any], *, prefix: str) -> None:
    raw = payload.get('review_form_fields')
    if not isinstance(raw, list):
        payload['review_form_fields'] = []
        return
    fields: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        name = str(item.get('field_name') or item.get('name') or f'Field {index}').strip()
        fid = str(item.get('field_id') or '').strip() or f'{prefix}:field_{index}'
        maps = item.get('maps_to_criteria') or item.get('maps_to') or []
        fields.append(
            {
                'field_id': fid,
                'field_name': name,
                'field_type': str(item.get('field_type') or 'text'),
                'scale_min': item.get('scale_min'),
                'scale_max': item.get('scale_max'),
                'description': str(item.get('description') or ''),
                'required': _parse_bool(item.get('required')),
                'maps_to_criteria': _coerce_str_list(maps),
            }
        )
    payload['review_form_fields'] = fields


def _coerce_checklist_items(payload: dict[str, Any], *, prefix: str) -> None:
    raw = payload.get('checklist_items')
    if not isinstance(raw, list):
        payload['checklist_items'] = []
        return
    items: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        name = str(item.get('item_name') or item.get('name') or f'Item {index}').strip()
        iid = str(item.get('item_id') or '').strip() or f'{prefix}:chk_{index}'
        items.append(
            {
                'item_id': iid,
                'item_name': name,
                'checklist_group': str(item.get('checklist_group') or ''),
                'description': str(item.get('description') or ''),
                'required_evidence': _coerce_str_list(item.get('required_evidence')),
                'applies_to_method': _coerce_str_list(item.get('applies_to_method')),
                'severity_if_missing': item.get('severity_if_missing') or 'major',
                'source_quote': str(item.get('source_quote') or '')[:200],
            }
        )
    payload['checklist_items'] = items


def coerce_extraction_payload(payload: dict[str, Any], hints: dict[str, Any]) -> dict[str, Any]:
    """Normalize LLM JSON into shapes Pydantic accepts; drop unreliable graph_relations."""
    if not isinstance(payload, dict):
        payload = {}

    payload = _coerce_source_document(payload, hints)
    sd = payload['source_document']
    prefix = _slug(str(sd.get('source_id') or hints.get('source_id') or 'template'))

    _coerce_review_criteria(payload, prefix=prefix)
    _coerce_review_form_fields(payload, prefix=prefix)
    _coerce_checklist_items(payload, prefix=prefix)

    # LLM graph_relations use inconsistent keys; normalizer rebuilds when empty.
    payload['graph_relations'] = []

    return payload
