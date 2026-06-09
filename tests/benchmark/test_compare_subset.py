import pandas as pd

from benchmark.analysis_subset import load_analysis_subset
from benchmark.compare import apply_paper_id_filter


def test_apply_paper_id_filter_b2a():
    subset = load_analysis_subset()
    included = subset['included_paper_ids']
    # Build 7 rows for included with correct winners + 2 excluded
    winners_map = {
        'acl_2024.acl-short.8': 'tie',
        'ets_liu-usingaibasedobject-2023': 'KG_ON',
        'iclr_1412.6980v9': 'KG_OFF',
        'iclr_9447_tabular_insights_visual_i': 'KG_ON',
        'icml_2402.01869v2': 'KG_OFF',
        'neurips_1706.03762v7': 'KG_ON',
        'neurips_2402.05602v2': 'KG_ON',
        'acl_2024.findings-acl.438': 'KG_OFF',
        'icml_2311.10263v2': 'KG_OFF',
    }
    df = pd.DataFrame({
        'paper_id': list(winners_map.keys()),
        'pairwise_winner': list(winners_map.values()),
    })
    filtered = apply_paper_id_filter(df, use_subset=True)
    assert len(filtered) == 7
    winners = filtered['pairwise_winner'].value_counts()
    assert winners.get('KG_ON', 0) == 4
    assert winners.get('KG_OFF', 0) == 2
    assert winners.get('tie', 0) == 1
