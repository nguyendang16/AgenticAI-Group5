from __future__ import annotations

import re
from typing import Any

from src.criterion_ids import (
    is_generic_criterion_id,
    semantic_criterion_id,
    venue_prefix_from_name,
)
from src.schema import ExtractedTemplate, ReviewCriterion, ReviewFormField, ChecklistItem, EvidenceRequired


def _slug(text: str, *, max_len: int = 64) -> str:
    token = re.sub(r'[^a-zA-Z0-9]+', '_', str(text or '').lower()).strip('_')
    return (token[:max_len] if token else 'item')


def normalize_template(template: ExtractedTemplate) -> ExtractedTemplate:
    sd = template.source_document
    prefix = _slug(sd.source_id or sd.file_name)

    sd.source_id = prefix
    if not sd.file_name:
        sd.file_name = sd.source_id

    venue_slug = _slug(sd.venue_or_journal_name or prefix)
    venue_acronym = venue_prefix_from_name(sd.venue_or_journal_name or venue_slug)

    seen_criterion: set[str] = set()
    normalized_criteria: list[ReviewCriterion] = []
    for index, criterion in enumerate(template.review_criteria, start=1):
        base = _slug(criterion.criterion_name or f'criterion_{index}')
        cid = criterion.criterion_id.strip() if criterion.criterion_id else ''
        group = str(criterion.criterion_group or 'OTHER').strip() or 'OTHER'
        if not cid or is_generic_criterion_id(cid):
            cid = semantic_criterion_id(
                venue_prefix=venue_acronym,
                index=index,
                criterion_group=group,
            )
        elif cid in seen_criterion:
            cid = f'{prefix}:{base}'
            suffix = 2
            while cid in seen_criterion:
                cid = f'{prefix}:{base}_{suffix}'
                suffix += 1
        seen_criterion.add(cid)
        criterion.criterion_id = cid

        evidence_items: list[EvidenceRequired] = []
        for ev_index, ev in enumerate(criterion.evidence_required, start=1):
            eid = ev.evidence_id.strip() if ev.evidence_id else f'{cid}:ev{ev_index}'
            evidence_items.append(
                ev.model_copy(update={'evidence_id': _slug(eid, max_len=80)})
            )
        if not evidence_items and criterion.description:
            evidence_items.append(
                EvidenceRequired(
                    evidence_id=f'{cid}:evidence_1',
                    name='Supporting evidence',
                    description='Evidence in the manuscript supporting or refuting this criterion.',
                    evidence_type='manuscript_span',
                    required=True,
                )
            )
        criterion.evidence_required = evidence_items
        if not criterion.applies_to_domain and sd.domain:
            criterion.applies_to_domain = list(sd.domain)
        if not criterion.applies_to_article_type and sd.article_types:
            criterion.applies_to_article_type = list(sd.article_types)
        normalized_criteria.append(criterion)

    seen_fields: set[str] = set()
    normalized_fields: list[ReviewFormField] = []
    for index, field in enumerate(template.review_form_fields, start=1):
        fid = field.field_id.strip() if field.field_id else f'{prefix}:field_{index}'
        if fid in seen_fields:
            fid = f'{prefix}:field_{index}'
        seen_fields.add(fid)
        normalized_fields.append(field.model_copy(update={'field_id': _slug(fid, max_len=80)}))

    seen_items: set[str] = set()
    normalized_checklist: list[ChecklistItem] = []
    for index, item in enumerate(template.checklist_items, start=1):
        iid = item.item_id.strip() if item.item_id else f'{prefix}:chk_{index}'
        if iid in seen_items:
            iid = f'{prefix}:chk_{index}'
        seen_items.add(iid)
        normalized_checklist.append(item.model_copy(update={'item_id': _slug(iid, max_len=80)}))

    node_ids = {sd.source_id}
    node_ids.add(venue_slug)
    for c in normalized_criteria:
        node_ids.add(c.criterion_id)
        for ev in c.evidence_required:
            node_ids.add(ev.evidence_id)
    for f in normalized_fields:
        node_ids.add(f.field_id)
    for item in normalized_checklist:
        node_ids.add(item.item_id)

    relations = list(template.graph_relations)
    if not relations:
        host_type = 'Venue' if 'conference' in sd.source_type or 'cfp' in sd.source_type or 'review_form' in sd.source_type else 'Journal'
        relations.append(
            {
                'source_node_type': 'SourceDocument',
                'source_node_id': sd.source_id,
                'relation': 'DESCRIBES',
                'target_node_type': host_type,
                'target_node_id': venue_slug,
            }
        )
        for c in normalized_criteria:
            relations.append(
                {
                    'source_node_type': host_type,
                    'source_node_id': venue_slug,
                    'relation': 'HAS_CRITERION',
                    'target_node_type': 'ReviewCriterion',
                    'target_node_id': c.criterion_id,
                }
            )
            relations.append(
                {
                    'source_node_type': 'ReviewCriterion',
                    'source_node_id': c.criterion_id,
                    'relation': 'SUPPORTED_BY_SOURCE',
                    'target_node_type': 'SourceDocument',
                    'target_node_id': sd.source_id,
                }
            )
            for ev in c.evidence_required:
                relations.append(
                    {
                        'source_node_type': 'ReviewCriterion',
                        'source_node_id': c.criterion_id,
                        'relation': 'REQUIRES_EVIDENCE',
                        'target_node_type': 'EvidenceRequirement',
                        'target_node_id': ev.evidence_id,
                    }
                )
        for f in normalized_fields:
            relations.append(
                {
                    'source_node_type': host_type,
                    'source_node_id': venue_slug,
                    'relation': 'HAS_REVIEW_FIELD',
                    'target_node_type': 'ReviewFormField',
                    'target_node_id': f.field_id,
                }
            )
        for item in normalized_checklist:
            relations.append(
                {
                    'source_node_type': 'ChecklistItem',
                    'source_node_id': item.item_id,
                    'relation': 'SUPPORTED_BY_SOURCE',
                    'target_node_type': 'SourceDocument',
                    'target_node_id': sd.source_id,
                }
            )

    from src.schema import GraphRelation

    validated_relations: list[GraphRelation] = []
    for rel in relations:
        if isinstance(rel, dict):
            gr = GraphRelation.model_validate(rel)
        else:
            gr = rel
        if gr.source_node_id in node_ids and gr.target_node_id in node_ids:
            validated_relations.append(gr)

    return ExtractedTemplate(
        source_document=sd,
        review_criteria=normalized_criteria,
        review_form_fields=normalized_fields,
        checklist_items=normalized_checklist,
        graph_relations=validated_relations,
    )


def merge_duplicate_criteria(
    template: ExtractedTemplate,
) -> tuple[ExtractedTemplate, list[dict[str, Any]]]:
    """Flag near-duplicate criterion names within the same template."""
    warnings: list[dict[str, Any]] = []
    seen_names: dict[str, str] = {}
    for c in template.review_criteria:
        key = re.sub(r'\s+', ' ', c.criterion_name.lower().strip())
        if key in seen_names:
            warnings.append(
                {
                    'type': 'duplicate_criterion_name',
                    'criterion_id': c.criterion_id,
                    'duplicate_of': seen_names[key],
                    'name': c.criterion_name,
                }
            )
        else:
            seen_names[key] = c.criterion_id
    return template, warnings
