from __future__ import annotations

from unittest.mock import patch

from benchmark.faithfulness import faithfulness_trad_all


def test_faithfulness_trad_all_no_annotations(tmp_path):
    trad_runs = tmp_path / 'trad_runs.jsonl'
    trad_runs.write_text(
        '{"job_id":"trad:test","paper_id":"test","venue":"ACL","condition":"TRAD_LLM",'
        '"review_path":"tests/fixtures/benchmark/trad_sample.md",'
        '"manuscript_job_id":"x","criteria_job_id":"y","final_md_bytes":3000}\n',
        encoding='utf-8',
    )
    with patch('benchmark.faithfulness.score_claims') as mock_score:
        mock_score.return_value = {
            'faithfulness_mean': 0.1,
            'faithfulness_n': 2,
            'claim_rows': [],
        }
        rows = faithfulness_trad_all(
            runs_path=trad_runs,
            run_scores_path=tmp_path / 'faith.csv',
        )
    assert len(rows) == 1
