from __future__ import annotations

from deepreview.criteria_kg import (
    _infer_venue_with_priority,
    build_graph_evaluation_summary,
    build_final_report_markdown,
    enrich_criteria_bundle,
    enrich_criteria_bundle_semantic_ids,
    format_criteria_bundle_for_prompt,
    format_graph_evaluation_markdown,
    format_criteria_legend_markdown,
    normalize_claim_level_audit_markdown,
    normalize_confidence,
    normalize_verification_status,
    redistribute_stuffed_summary_sections,
    sanitize_fast_report_sections,
)


def test_twelf_title_wins_over_chi_citation_in_body():
    body = 'See prior work (CHI \'08). https://doi.org/10.1145/example'
    found = _infer_venue_with_priority(
        paper_markdown=body,
        job_metadata={'title': 'TWELF2026_Proposal_Dang_Khoi_Nguyen'},
        extra_inference_text='TWELF2026_Proposal_Dang_Khoi_Nguyen',
    )
    assert found.get('venue') == 'TWELF'


def test_sanitize_summary_strips_nested_sections():
    sections = {
        'summary': (
            '- Summary: Short overview.\n'
            '- Strengths:\n- - item\n'
            '- Weaknesses:\n- - item'
        ),
        'strengths': '- Real strength',
        'weaknesses': '- Real weakness',
        'key_issues': '- Issue',
    }
    cleaned = sanitize_fast_report_sections(sections)
    assert '- Strengths:' not in cleaned['summary']
    assert cleaned['strengths'] == '- Real strength'


def test_enrich_bundle_semantic_ids():
    bundle = {
        'criteria_count': 1,
        'query': {'venue': 'TWELF'},
        'criteria_by_group': {
            'SCOPE_FIT': [
                {
                    'criterion_id': 'criterion_1',
                    'criterion_name': 'Fit',
                    'criterion_group': 'SCOPE_FIT',
                    'description': 'Scope.',
                    'provenance': {'file_name': 'twelf_template.docx'},
                }
            ]
        },
    }
    enriched = enrich_criteria_bundle_semantic_ids(bundle)
    cid = enriched['criteria_by_group']['SCOPE_FIT'][0]['criterion_id']
    assert cid == 'TWELF_C01_SCOPE_FIT'


def test_legend_uses_bullet_list():
    bundle = enrich_criteria_bundle(
        {
            'criteria_count': 1,
            'query': {'venue': 'TWELF'},
            'criteria_by_group': {
                'SCOPE_FIT': [
                    {
                        'criterion_id': 'criterion_1',
                        'criterion_name': 'Fit',
                        'criterion_group': 'SCOPE_FIT',
                        'description': 'Scope.',
                        'provenance': {'file_name': 'twelf.docx'},
                    }
                ]
            },
        }
    )
    legend = format_criteria_legend_markdown(bundle)
    assert '**C01**' in legend
    assert '(SCOPE)' in legend
    assert '| criterion_id |' not in legend


def test_format_compliance_group_remap():
    bundle = enrich_criteria_bundle(
        {
            'criteria_count': 1,
            'query': {'venue': 'TWELF'},
            'criteria_by_group': {
                'SCOPE_FIT': [
                    {
                        'criterion_id': 'criterion_5',
                        'criterion_name': 'Format and presentation compliance',
                        'criterion_group': 'SCOPE_FIT',
                        'description': 'Check page length and PDF submission.',
                    }
                ]
            },
        }
    )
    item = bundle['criteria_by_group']['FORMAT_COMPLIANCE'][0]
    assert 'FORMAT_COMPLIANCE' in item['criterion_id']


def test_prompt_excludes_other_from_active():
    bundle = enrich_criteria_bundle(
        {
            'criteria_count': 2,
            'query': {'venue': 'TWELF'},
            'criteria_by_group': {
                'OTHER': [
                    {
                        'criterion_id': 'criterion_4',
                        'criterion_name': 'SIG routing',
                        'criterion_group': 'OTHER',
                        'description': 'Session routing.',
                    }
                ],
                'SCOPE_FIT': [
                    {
                        'criterion_id': 'criterion_1',
                        'criterion_name': 'Fit',
                        'criterion_group': 'SCOPE_FIT',
                        'description': 'Scope.',
                    }
                ],
            },
        }
    )
    prompt = format_criteria_bundle_for_prompt(bundle)
    assert 'Active criteria' in prompt
    assert 'Reference only' in prompt
    assert 'SIG routing' in prompt
    assert 'reference only' in prompt


def test_redistribute_plain_strengths_from_summary():
    sections = {
        'summary': 'Overview.\nStrengths\n- Good topic\nWeaknesses\n- Gap',
        'weaknesses': '- Existing weakness',
    }
    out = redistribute_stuffed_summary_sections(sections, review_fast_mode=True)
    assert 'Good topic' in out.get('strengths', '')
    assert 'Overview' in out.get('summary', '')


def test_normalize_verification_status():
    assert normalize_verification_status('not verifiable from manuscript') == 'MISSING_REQUIRED_EVIDENCE'
    assert normalize_verification_status('SUPPORTED_BY_TEXT') == 'SUPPORTED_BY_TEXT'


def test_normalize_audit_table_status_column():
    audit = (
        '- Critique ID | Critique | Criterion ID | Verification status | Confidence | Fix\n'
        '| - C01 | Missing methods | TWELF_C01_CLARITY_PRESENTATION | Not verifiable from manuscript | NEEDS_HUMAN_CHECK | Add detail |\n'
    )
    out = normalize_claim_level_audit_markdown(audit)
    assert 'Missing' in out
    assert '| H |' in out or 'H' in out


def test_normalize_confidence_rejects_status_enum():
    assert normalize_confidence('NEEDS_HUMAN_CHECK') == 'Medium'


def test_build_report_includes_legend():
    bundle = enrich_criteria_bundle_semantic_ids(
        {
            'criteria_count': 1,
            'query': {'venue': 'TWELF', 'journal': None, 'domain': 'Educational Technology', 'article_type': None},
            'criteria_by_group': {
                'SCOPE_FIT': [
                    {
                        'criterion_id': 'criterion_1',
                        'criterion_name': 'Fit to digital learning',
                        'criterion_group': 'SCOPE_FIT',
                        'description': 'Scope fit check.',
                    }
                ]
            },
        }
    )
    md = build_final_report_markdown(
        {'summary': 'Brief summary only.'},
        section_order=['summary'],
        section_titles={'summary': 'Summary'},
        criteria_bundle=bundle,
        review_fast_mode=True,
    )
    assert '## Criterion Legend' in md
    assert '## Graph Evaluation' in md
    assert 'C01' in md
    assert '## Summary' in md


def test_graph_evaluation_summarizes_sources():
    bundle = enrich_criteria_bundle_semantic_ids(
        {
            'criteria_count': 1,
            'query': {'venue': 'TWELF', 'journal': None, 'domain': 'Educational Technology', 'article_type': None},
            'provenance': [{'source_id': 'twelf_guidelines', 'file_name': 'twelf.docx'}],
            'criteria_by_group': {
                'SCOPE_FIT': [
                    {
                        'criterion_id': 'criterion_1',
                        'criterion_name': 'Fit',
                        'criterion_group': 'SCOPE_FIT',
                        'description': 'Scope fit.',
                        'evidence_required': [{'name': 'Learning context'}],
                    }
                ]
            },
        }
    )
    evaluation = build_graph_evaluation_summary(bundle)
    assert evaluation['criteria_count'] == 1
    assert evaluation['active_criteria_count'] == 1
    assert evaluation['source_document_count'] == 1
    assert evaluation['source_documents'][0]['file_name'] == 'twelf.docx'

    md = format_graph_evaluation_markdown(bundle)
    assert '## Graph Evaluation' in md
    assert 'twelf.docx' in md
    assert 'Evidence requirements linked: 1' in md
