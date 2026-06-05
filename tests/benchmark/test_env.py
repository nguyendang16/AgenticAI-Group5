from benchmark.env import build_benchmark_env


def test_kg_on_env():
    env = build_benchmark_env('KG_ON', venue='ICLR', base={'AGENT_MODEL': 'gpt-5-mini'})
    assert env['REVIEW_CRITERIA_ENABLED'] == 'true'
    assert env['REVIEW_VENUE'] == 'ICLR'
    assert env['REVIEW_FAST_MODE'] == 'true'
    assert env['PAPER_SEARCH_ENABLED'] == 'false'
    assert env['REVIEW_INFER_VENUE_FROM_PAPER'] == 'false'


def test_kg_off_env():
    env = build_benchmark_env('KG_OFF', venue='ICLR', base={})
    assert env['REVIEW_CRITERIA_ENABLED'] == 'false'
