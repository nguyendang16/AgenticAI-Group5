from __future__ import annotations

from unittest.mock import patch

from benchmark.judge import judge_trad_all


def test_judge_trad_all_skips_completion_validation(tmp_path, monkeypatch):
    trad_runs = tmp_path / 'trad_runs.jsonl'
    trad_runs.write_text(
        '{"job_id":"trad:test","paper_id":"test","venue":"ACL","condition":"TRAD_LLM",'
        '"review_path":"tests/fixtures/benchmark/trad_sample.md",'
        '"manuscript_job_id":"x","criteria_job_id":"y","final_md_bytes":3000}\n',
        encoding='utf-8',
    )
    monkeypatch.setenv('BENCHMARK_JUDGE_MODE', 'composite')

    with patch('benchmark.judge.judge_run') as mock_judge:
        mock_judge.return_value = {
            'paper_id': 'test',
            'job_id': 'trad:test',
            'venue': 'ACL',
            'condition': 'TRAD_LLM',
            'rubric_alignment': 0.5,
        }
        results = judge_trad_all(runs_path=trad_runs, output_path=tmp_path / 'scores.csv')
    assert len(results) == 1
    mock_judge.assert_called_once()
