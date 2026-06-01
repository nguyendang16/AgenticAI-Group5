from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from src.config import EXTRACT_MODEL, get_pipeline_settings, require_openai_api_key
from src.payload_coercion import coerce_extraction_payload
from src.schema import ExtractedTemplate

EXTRACTION_SYSTEM_PROMPT = """You extract structured review knowledge from academic conference/journal documents.
Return ONLY valid JSON matching the user schema. No markdown fences.

Rules:
- Extract only information supported by the document. Do not invent criteria.
- Every review criterion needs criterion_id, criterion_name, criterion_group, description, confidence.
- Include a short source_quote (max 200 chars) per criterion when possible.
- If implied but not explicit, set confidence to medium or low.
- Call-for-papers: focus on SCOPE_FIT, article types, formatting, submission requirements.
- Reviewer forms: populate review_form_fields and map to criteria via maps_to_criteria.
- Journal guidelines: METHODOLOGICAL_RIGOR, ETHICS_PRIVACY, REPRODUCIBILITY_TRANSPARENCY, CLARITY_PRESENTATION.
- Avoid long passages in source_quote.

criterion_group must be one of:
NOVELTY_CONTRIBUTION, TECHNICAL_SOUNDNESS, METHODOLOGICAL_RIGOR, EVALUATION_VALIDITY,
THEORETICAL_GROUNDING, EDUCATIONAL_CONTRIBUTION, ETHICS_PRIVACY, REPRODUCIBILITY_TRANSPARENCY,
CLARITY_PRESENTATION, SCOPE_FIT, REVIEWER_CONFIDENCE, OTHER

severity_if_missing: minor | major | critical
confidence: low | medium | high

Populate graph_relations for:
- SourceDocument DESCRIBES Venue or Journal (use venue_or_journal_name as target_node_id slug)
- Venue/Journal HAS_CRITERION ReviewCriterion
- ReviewCriterion REQUIRES_EVIDENCE EvidenceRequirement
- ReviewCriterion SUPPORTED_BY_SOURCE SourceDocument
- Venue/Journal HAS_REVIEW_FIELD ReviewFormField
- ChecklistItem SUPPORTED_BY_SOURCE SourceDocument
Use consistent IDs across relations.
"""


def _build_user_prompt(*, raw_text: str, hints: dict[str, Any]) -> str:
    return (
        'Extract structured review knowledge from this document.\n\n'
        f'Filename hints (use but correct if document contradicts):\n{json.dumps(hints, ensure_ascii=False, indent=2)}\n\n'
        'Required top-level JSON keys: source_document, review_criteria, review_form_fields, '
        'checklist_items, graph_relations\n\n'
        'Document text:\n'
        f'{raw_text}'
    )


def extract_with_llm(*, raw_text: str, hints: dict[str, Any]) -> dict[str, Any]:
    settings = get_pipeline_settings()
    api_key = require_openai_api_key()
    client_kwargs: dict[str, Any] = {'api_key': api_key}
    if settings.openai_base_url:
        client_kwargs['base_url'] = settings.openai_base_url
    client = OpenAI(**client_kwargs)

    response = client.chat.completions.create(
        model=EXTRACT_MODEL,
        temperature=0.1,
        response_format={'type': 'json_object'},
        messages=[
            {'role': 'system', 'content': EXTRACTION_SYSTEM_PROMPT},
            {'role': 'user', 'content': _build_user_prompt(raw_text=raw_text, hints=hints)},
        ],
    )
    content = response.choices[0].message.content or '{}'
    return json.loads(content)


def llm_extract_template(*, raw_text: str, hints: dict[str, Any]) -> ExtractedTemplate:
    payload = extract_with_llm(raw_text=raw_text, hints=hints)
    payload = coerce_extraction_payload(payload, hints)
    return ExtractedTemplate.model_validate(payload)
