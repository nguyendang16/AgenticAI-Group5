from __future__ import annotations

import pandas as pd
import pytest

from benchmark.compare import (
    bootstrap_median_ci,
    build_paired_comparison,
    paired_wilcoxon_pvalue,
)


def test_paired_wilcoxon_pvalue_requires_eight_pairs():
    assert paired_wilcoxon_pvalue(pd.Series([0.1, 0.2, 0.3])) is None


def test_paired_wilcoxon_pvalue_returns_float_for_large_sample():
    deltas = pd.Series([0.1, 0.2, 0.0, 0.3, 0.1, 0.4, 0.2, 0.5])
    pvalue = paired_wilcoxon_pvalue(deltas)
    assert pvalue is not None
    assert 0.0 <= pvalue <= 1.0


def test_bootstrap_median_ci_returns_bounds():
    deltas = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    low, high = bootstrap_median_ci(deltas, n_boot=200, random_state=0)
    assert low is not None
    assert high is not None
    assert low <= 3.0 <= high


def test_build_paired_comparison_writes_outputs(tmp_path):
    deterministic = tmp_path / 'deterministic_scores.csv'
    judge = tmp_path / 'review_quality_scores.csv'
    decision = tmp_path / 'decision_metrics.csv'

    det_rows = []
    judge_rows = []
    for paper_id, venue in (('p1', 'ICLR'), ('p2', 'NeurIPS')):
        for condition, criteria_count, rubric in (
            ('KG_ON', 5, 4.0),
            ('KG_OFF', 0, 3.0),
        ):
            det_rows.append(
                {
                    'paper_id': paper_id,
                    'job_id': f'{paper_id}-{condition}',
                    'venue': venue,
                    'condition': condition,
                    'completion_valid': True,
                    'condition_valid': True,
                    'completion_errors': '',
                    'condition_errors': '',
                    'runtime_seconds': 100 if condition == 'KG_ON' else 90,
                    'total_tokens': 1000 if condition == 'KG_ON' else 900,
                    'criteria_count': criteria_count,
                    'input_tokens': 700,
                    'output_tokens': 300,
                    'tool_calls': 1,
                    'annotation_count': 2,
                    'paper_search_calls': 0,
                    'final_md_bytes': 4096,
                }
            )
            judge_rows.append(
                {
                    'paper_id': paper_id,
                    'job_id': f'{paper_id}-{condition}',
                    'venue': venue,
                    'condition': condition,
                    'rubric_alignment': rubric,
                    'factual_correctness': rubric,
                    'evidence_support': rubric,
                    'specificity': rubric,
                    'actionability': rubric,
                    'unsupported_critique_rate': rubric,
                    'criterion_grounded_valid_critique': rubric,
                }
            )

    pd.DataFrame(det_rows).to_csv(deterministic, index=False)
    pd.DataFrame(judge_rows).to_csv(judge, index=False)
    pd.DataFrame(
        [
            {
                'paper_id': 'p1',
                'condition': 'KG_ON',
                'expected_decision': 'accept',
                'predicted_decision': 'accept',
                'decision_correct': 1,
            },
            {
                'paper_id': 'p1',
                'condition': 'KG_OFF',
                'expected_decision': 'accept',
                'predicted_decision': 'reject',
                'decision_correct': 0,
            },
        ]
    ).to_csv(decision, index=False)

    metadata = build_paired_comparison(
        deterministic_path=deterministic,
        judge_path=judge,
        decision_path=decision,
        paired_output_path=tmp_path / 'paired_comparison.csv',
        overall_output_path=tmp_path / 'overall_summary.csv',
        venue_output_path=tmp_path / 'venue_summary.csv',
    )

    paired = pd.read_csv(tmp_path / 'paired_comparison.csv')
    overall = pd.read_csv(tmp_path / 'overall_summary.csv')
    venue = pd.read_csv(tmp_path / 'venue_summary.csv')

    assert metadata['valid_pairs'] == 2
    assert len(paired) == 2
    assert paired['delta_rubric_alignment'].tolist() == pytest.approx([1.0, 1.0])
    assert 'rubric_alignment' in overall['metric'].values
    assert len(venue) == 2
