from __future__ import annotations

from typing import Any

from src.schema import ExtractedTemplate


def validate_template(template: ExtractedTemplate) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    sd = template.source_document

    for field in ('source_id', 'source_title', 'source_type', 'file_name'):
        if not str(getattr(sd, field, '') or '').strip():
            warnings.append({'type': 'missing_source_field', 'field': field})

    node_ids: set[str] = {sd.source_id}
    venue_id = _venue_node_id(sd.venue_or_journal_name)
    node_ids.add(venue_id)

    for c in template.review_criteria:
        if not c.criterion_id:
            warnings.append({'type': 'missing_criterion_id', 'name': c.criterion_name})
        if not c.criterion_name:
            warnings.append({'type': 'missing_criterion_name', 'id': c.criterion_id})
        if not c.description.strip():
            warnings.append({'type': 'missing_criterion_description', 'id': c.criterion_id})
        if not c.source_quote.strip():
            warnings.append({'type': 'missing_source_quote', 'id': c.criterion_id})
        if not c.evidence_required:
            warnings.append({'type': 'missing_evidence_required', 'id': c.criterion_id})
        if c.confidence == 'low':
            warnings.append({'type': 'low_confidence', 'id': c.criterion_id})
        node_ids.add(c.criterion_id)
        for ev in c.evidence_required:
            node_ids.add(ev.evidence_id)

    for f in template.review_form_fields:
        node_ids.add(f.field_id)
    for item in template.checklist_items:
        node_ids.add(item.item_id)

    for rel in template.graph_relations:
        if rel.source_node_id not in node_ids:
            warnings.append(
                {
                    'type': 'invalid_graph_relation',
                    'reason': 'unknown_source_node_id',
                    'relation': rel.model_dump(),
                }
            )
        if rel.target_node_id not in node_ids:
            warnings.append(
                {
                    'type': 'invalid_graph_relation',
                    'reason': 'unknown_target_node_id',
                    'relation': rel.model_dump(),
                }
            )

    return warnings


def _venue_node_id(name: str) -> str:
    import re

    token = re.sub(r'[^a-zA-Z0-9]+', '_', str(name or '').lower()).strip('_')
    return token or 'venue'
