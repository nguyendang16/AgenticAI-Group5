import pandas as pd

from benchmark.analysis_subset import load_analysis_subset
from benchmark.report import build_benchmark_summary


def test_build_benchmark_summary_includes_subset_sections(tmp_path):
    subset = load_analysis_subset()
    winners_map = {
        paper_id: 'KG_ON'
        for paper_id in subset['included_paper_ids']
    }
    winners_map['ets_liu-usingaibasedobject-2023'] = 'KG_OFF'
    winners_map['iclr_1412.6980v9'] = 'KG_OFF'
    winners_map['icml_2402.01869v2'] = 'tie'
    for excluded in subset['excluded_paper_ids']:
        winners_map[excluded] = 'KG_OFF'

    paired = pd.DataFrame({
        'paper_id': list(winners_map.keys()),
        'venue': ['TEST'] * len(winners_map),
        'pairwise_winner': list(winners_map.values()),
    })
    overall = pd.DataFrame([
        {
            'metric': 'rubric_alignment',
            'n_pairs': 7,
            'median_delta': 0.05,
            'mean_delta': 0.05,
            'bootstrap_ci_low': -0.1,
            'bootstrap_ci_high': 0.2,
            'wilcoxon_pvalue': 0.5,
        }
    ])
    three_way = pd.DataFrame([
        {'paper_id': 'p1', 'median_rubric_alignment_TRAD_LLM': 0.3, 'median_rubric_alignment_KG_OFF': 0.5, 'median_rubric_alignment_KG_ON': 0.7},
        {'paper_id': '__overall__', 'median_rubric_alignment_TRAD_LLM': 0.35, 'median_rubric_alignment_KG_OFF': 0.45, 'median_rubric_alignment_KG_ON': 0.65},
    ])
    trad_pairwise = pd.DataFrame([
        {'paper_id': 'p1', 'pair_type': 'TRAD_VS_KG_OFF', 'winner': 'TRAD'},
        {'paper_id': 'p2', 'pair_type': 'TRAD_VS_KG_OFF', 'winner': 'KG_OFF'},
        {'paper_id': 'p1', 'pair_type': 'TRAD_VS_KG_ON', 'winner': 'KG_ON'},
        {'paper_id': 'p2', 'pair_type': 'TRAD_VS_KG_ON', 'winner': 'tie'},
    ])

    paired_path = tmp_path / 'paired_comparison.csv'
    overall_path = tmp_path / 'overall_summary.csv'
    venue_path = tmp_path / 'venue_summary.csv'
    three_way_path = tmp_path / 'three_way_summary.csv'
    trad_path = tmp_path / 'trad_pairwise_judge_scores.csv'
    output_path = tmp_path / 'benchmark_summary.md'

    paired.to_csv(paired_path, index=False)
    overall.to_csv(overall_path, index=False)
    venue_path.write_text('', encoding='utf-8')
    three_way.to_csv(three_way_path, index=False)
    trad_pairwise.to_csv(trad_path, index=False)

    markdown = build_benchmark_summary(
        paired_path=paired_path,
        overall_path=overall_path,
        venue_path=venue_path,
        three_way_path=three_way_path,
        trad_pairwise_path=trad_path,
        output_path=output_path,
        metadata={'valid_pairs': 7, 'excluded_pairs': 2, 'completion_rate': 0.778},
        use_subset=True,
    )

    assert '## Analysis subset' in markdown
    assert 'Included papers: **7**' in markdown
    for excluded in subset['excluded_paper_ids']:
        assert excluded in markdown
    assert subset['reason'] in markdown

    assert 'KG_ON wins: **4** / 7' in markdown
    assert 'KG_OFF wins: **2** / 7' in markdown
    assert 'Ties: **1** / 7' in markdown

    assert '## Three-way medians' in markdown
    assert 'TRAD_LLM: **0.350**' in markdown
    assert 'KG_OFF: **0.450**' in markdown
    assert 'KG_ON: **0.650**' in markdown

    assert '## Trad pairwise win rates' in markdown
    assert '**TRAD_VS_KG_OFF**' in markdown
    assert '**TRAD_VS_KG_ON**' in markdown
