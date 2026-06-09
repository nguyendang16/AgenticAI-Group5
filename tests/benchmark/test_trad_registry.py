from benchmark.trad_registry import load_trad_registry, trad_job_id


def test_trad_job_id_format():
    assert trad_job_id('acl_2024.acl-short.8') == 'trad:acl_2024.acl-short.8'


def test_load_trad_registry_nine_rows():
    rows = load_trad_registry()
    assert len(rows) == 9
    paper_ids = {row['paper_id'] for row in rows}
    assert 'neurips_1706.03762v7' in paper_ids
    assert all(row.get('manuscript_job_id') for row in rows)
    assert all(row.get('criteria_job_id') for row in rows)
