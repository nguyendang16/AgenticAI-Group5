from __future__ import annotations

from types import SimpleNamespace

from deepreview.criteria_kg import (
    format_criteria_bundle_for_prompt,
    infer_venue_journal_from_markdown,
    merge_criteria_query,
    resolve_review_criteria_bundle,
)
from deepreview.prompts.review_agent_prompt import build_review_agent_system_prompt


def test_infer_iclr_from_markdown():
    inferred = infer_venue_journal_from_markdown('Submitted to ICLR 2026 workshop track.')
    assert inferred.get('venue') == 'ICLR'
    assert inferred.get('domain') == 'Machine Learning'


def test_infer_twelf2026_from_job_title():
    inferred = infer_venue_journal_from_markdown('TWELF2026_Proposal_Dang_Khoi_Nguyen')
    assert inferred.get('venue') == 'TWELF'
    assert inferred.get('domain') == 'Educational Technology'


def test_merge_query_uses_job_title_before_body_chi_cite():
    settings = SimpleNamespace(
        review_venue=None,
        review_journal=None,
        review_domain=None,
        review_article_type=None,
        review_infer_venue_from_paper=True,
    )
    query = merge_criteria_query(
        settings=settings,
        job_metadata={'title': 'TWELF2026_Proposal'},
        paper_markdown='Prior work (CHI \'08) is cited here.',
        extra_inference_text='TWELF2026_Proposal',
    )
    assert query['venue'] == 'TWELF'


def test_merge_query_uses_job_title_when_markdown_silent():
    settings = SimpleNamespace(
        review_venue=None,
        review_journal=None,
        review_domain=None,
        review_article_type=None,
        review_infer_venue_from_paper=True,
    )
    query = merge_criteria_query(
        settings=settings,
        job_metadata={'title': 'TWELF2026_Proposal_Dang_Khoi_Nguyen'},
        paper_markdown='A proposal about data visualization education.',
        extra_inference_text='TWELF2026_Proposal_Dang_Khoi_Nguyen',
    )
    assert query['venue'] == 'TWELF'


def test_merge_query_prefers_job_metadata():
    settings = SimpleNamespace(
        review_venue='NeurIPS',
        review_journal=None,
        review_domain=None,
        review_article_type=None,
        review_infer_venue_from_paper=True,
    )
    query = merge_criteria_query(
        settings=settings,
        job_metadata={'review_venue': 'ICLR'},
        paper_markdown='NeurIPS 2025 paper',
    )
    assert query['venue'] == 'ICLR'


def test_json_fallback_bundle():
    settings = SimpleNamespace(
        review_criteria_enabled=True,
        review_venue='ICLR',
        review_journal='',
        review_domain='Machine Learning',
        review_article_type='',
        review_infer_venue_from_paper=False,
        neo4j_uri=None,
        neo4j_password=None,
        review_criteria_json_dir='outputs/extracted_json',
    )
    bundle, resolution = resolve_review_criteria_bundle(
        settings=settings,
        paper_markdown='',
        job_metadata={},
    )
    assert resolution.get('source') == 'json_fallback'
    assert bundle is not None
    assert int(bundle.get('criteria_count') or 0) > 0


def test_prompt_includes_criteria_section():
    bundle = {
        'criteria_count': 1,
        'query': {'venue': 'ICLR', 'journal': None, 'domain': 'ML', 'article_type': None},
        'criteria_by_group': {
            'NOVELTY_CONTRIBUTION': [
                {
                    'criterion_id': 'ICLR_C01',
                    'criterion_name': 'Novelty',
                    'description': 'Assess novelty.',
                    'severity_if_missing': 'major',
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
    assert 'VENUE REVIEW CRITERIA' in prompt
    assert 'ICLR_C01' in prompt
