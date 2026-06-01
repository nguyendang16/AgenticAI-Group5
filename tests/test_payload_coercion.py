from src.payload_coercion import coerce_extraction_payload
from src.schema import ExtractedTemplate


def test_coerce_iclr_style_llm_payload():
    hints = {
        'source_id': 'conference_iclr_2026_reviewer_guide',
        'source_title': 'ICLR 2026 Reviewer Guideline',
        'source_type': 'conference_guideline',
        'venue_or_journal_name': 'ICLR',
        'file_name': 'conference_iclr_2026_reviewer_guide_template.docx',
        'domain': ['Machine Learning'],
        'year': 2026,
        'article_types': ['full_paper'],
    }
    payload = {
        'source_document': {
            'source_id': 'conference_iclr_2026_reviewer_guide',
            'source_url': 'https://example.com/ReviewerGuide',
        },
        'review_criteria': [
            {
                'criterion_name': 'Novelty',
                'criterion_group': 'NOVELTY_CONTRIBUTION',
                'description': 'Assess novelty.',
            }
        ],
        'review_form_fields': [],
        'checklist_items': [],
        'graph_relations': [
            {
                'start_node': 'ICLR',
                'relationship': 'HAS_CRITERION',
                'end_node': 'ICLR_C01',
            }
        ],
    }
    coerced = coerce_extraction_payload(payload, hints)
    template = ExtractedTemplate.model_validate(coerced)
    assert template.source_document.file_name == hints['file_name']
    assert template.review_criteria[0].criterion_id
    assert coerced['graph_relations'] == []


def test_coerce_form_field_required_strings():
    hints = {'source_id': 'x', 'file_name': 'a.docx', 'source_title': 'T', 'source_type': 'review_form'}
    payload = {
        'source_document': {'source_id': 'x', 'source_title': 'T', 'source_type': 'review_form', 'venue_or_journal_name': 'AAAI'},
        'review_form_fields': [
            {'field_name': 'Ethics', 'required': 'true'},
            {'field_name': 'Optional', 'required': 'conditional'},
        ],
        'review_criteria': [],
        'checklist_items': [],
        'graph_relations': [{'source_node_id': 'a', 'target_node_id': 'b'}],
    }
    coerced = coerce_extraction_payload(payload, hints)
    template = ExtractedTemplate.model_validate(coerced)
    assert template.review_form_fields[0].required is True
    assert template.review_form_fields[1].required is False
