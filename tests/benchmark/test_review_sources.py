from pathlib import Path

from benchmark.review_sources import load_review_artifacts, load_trad_review_artifacts

FIXTURE_MD = Path('tests/fixtures/benchmark/trad_sample.md')


def test_load_trad_review_artifacts_from_paths(tmp_path, monkeypatch):
    review_path = tmp_path / 'review.md'
    review_path.write_text(FIXTURE_MD.read_text(encoding='utf-8'), encoding='utf-8')
    ms_path = tmp_path / 'mineru_full.md'
    ms_path.write_text('# Manuscript\n\nBody text.' * 200, encoding='utf-8')
    crit_path = tmp_path / 'review_criteria_bundle.json'
    crit_path.write_text('{"criteria": []}', encoding='utf-8')

    row = {
        'review_path': str(review_path),
        'manuscript_job_id': 'unused',
        'criteria_job_id': 'unused',
    }
    from benchmark import review_sources as rs

    def fake_job_dir(job_id: str) -> Path:
        return tmp_path

    monkeypatch.setattr(rs, '_job_dir', fake_job_dir)
    artifacts = load_trad_review_artifacts(row)
    assert 'Weaknesses' in artifacts['final_markdown']
    assert 'Manuscript' in artifacts['manuscript_excerpt']
    assert artifacts['criteria_json']


def test_load_review_artifacts_dispatches_trad_llm_row(tmp_path, monkeypatch):
    review_path = tmp_path / 'review.md'
    review_path.write_text(FIXTURE_MD.read_text(encoding='utf-8'), encoding='utf-8')
    (tmp_path / 'mineru_full.md').write_text('# Manuscript\n\nBody text.' * 200, encoding='utf-8')
    (tmp_path / 'review_criteria_bundle.json').write_text('{"criteria": []}', encoding='utf-8')

    from benchmark import review_sources as rs

    def fake_job_dir(job_id: str) -> Path:
        return tmp_path

    monkeypatch.setattr(rs, '_job_dir', fake_job_dir)

    row = {
        'job_id': 'trad:test-paper',
        'condition': 'TRAD_LLM',
        'review_path': str(review_path),
        'manuscript_job_id': 'unused',
        'criteria_job_id': 'unused',
    }
    artifacts = load_review_artifacts(row)
    assert 'Weaknesses' in artifacts['final_markdown']
    assert 'Manuscript' in artifacts['manuscript_excerpt']
    assert artifacts['criteria_json']
