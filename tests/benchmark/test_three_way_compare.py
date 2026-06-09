import pandas as pd

from benchmark.three_way_compare import build_three_way_summary


def test_build_three_way_summary_medians():
    judge = pd.DataFrame([
        {'paper_id': 'p1', 'condition': 'TRAD_LLM', 'rubric_alignment': 0.3, 'job_id': 't1'},
        {'paper_id': 'p1', 'condition': 'KG_OFF', 'rubric_alignment': 0.5, 'job_id': 'o1'},
        {'paper_id': 'p1', 'condition': 'KG_ON', 'rubric_alignment': 0.7, 'job_id': 'n1'},
        {'paper_id': 'p2', 'condition': 'TRAD_LLM', 'rubric_alignment': 0.4, 'job_id': 't2'},
        {'paper_id': 'p2', 'condition': 'KG_OFF', 'rubric_alignment': 0.4, 'job_id': 'o2'},
        {'paper_id': 'p2', 'condition': 'KG_ON', 'rubric_alignment': 0.6, 'job_id': 'n2'},
    ])
    faith = pd.DataFrame([
        {'job_id': 't1', 'faithfulness_mean': 0.1},
        {'job_id': 'o1', 'faithfulness_mean': 0.2},
        {'job_id': 'n1', 'faithfulness_mean': 0.3},
        {'job_id': 't2', 'faithfulness_mean': 0.15},
        {'job_id': 'o2', 'faithfulness_mean': 0.25},
        {'job_id': 'n2', 'faithfulness_mean': 0.35},
    ])
    summary = build_three_way_summary(
        judge_df=judge,
        faithfulness_df=faith,
        use_subset=False,
        output_path=None,
    )
    overall = summary[summary['paper_id'] == '__overall__'].iloc[0]
    assert overall['median_rubric_alignment_TRAD_LLM'] == 0.35
    assert overall['median_rubric_alignment_KG_ON'] == 0.65
