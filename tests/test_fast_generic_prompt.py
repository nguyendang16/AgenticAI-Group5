from __future__ import annotations

from deepreview.prompts.review_agent_prompt import build_review_agent_system_prompt
from deepreview.tools.review_tools import _required_final_report_section_order


def test_fast_kg_off_prompt_omits_claim_audit_and_criterion_id():
    prompt = build_review_agent_system_prompt(
        source_file_id='job-1',
        source_file_name='paper.pdf',
        paper_markdown='# Title\n\nBody.',
        review_fast_mode=True,
        criteria_bundle=None,
    )
    assert 'Claim-Level Audit' not in prompt
    assert 'criterion_id' not in prompt
    assert 'VENUE REVIEW CRITERIA' not in prompt
    assert 'Summary, Strengths, Weaknesses, Key Issues, Actionable Suggestions, Scores' in prompt


def test_fast_kg_on_prompt_keeps_claim_audit():
    bundle = {
        'criteria_count': 1,
        'query': {'venue': 'ICLR'},
        'criteria_by_group': {
            'SCOPE_FIT': [
                {
                    'criterion_id': 'ICLR_C01_SCOPE_FIT',
                    'criterion_name': 'Scope',
                    'criterion_group': 'SCOPE_FIT',
                    'description': 'Scope fit.',
                }
            ]
        },
    }
    prompt = build_review_agent_system_prompt(
        source_file_id='job-1',
        source_file_name='paper.pdf',
        paper_markdown='# Title\n\nBody.',
        review_fast_mode=True,
        criteria_bundle=bundle,
    )
    assert 'Claim-Level Audit' in prompt
    assert 'criterion_id' in prompt
    assert 'VENUE REVIEW CRITERIA' in prompt


def test_fast_generic_section_order_without_criteria():
    assert _required_final_report_section_order(
        review_fast_mode=True,
        kg_criteria_active=False,
    ) == [
        'summary',
        'strengths',
        'weaknesses',
        'key_issues',
        'actionable_suggestions',
        'scores',
    ]


def test_fast_kg_section_order_unchanged():
    assert _required_final_report_section_order(
        review_fast_mode=True,
        kg_criteria_active=True,
    ) == [
        'summary',
        'strengths',
        'weaknesses',
        'key_issues',
        'actionable_suggestions',
        'claim_level_audit',
        'scores',
    ]
