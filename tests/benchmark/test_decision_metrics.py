from __future__ import annotations

import json

import pytest

sklearn = pytest.importorskip('sklearn')


def test_parse_accept_from_scores_section():
    from benchmark.decision_metrics import parse_predicted_decision

    md = '## Scores\n\n**Recommendation:** Accept\n'
    assert parse_predicted_decision(md) == 'accept'


def test_parse_reject_from_scores_section():
    from benchmark.decision_metrics import parse_predicted_decision

    md = '## Scores\n\n**Recommendation:** Reject\n'
    assert parse_predicted_decision(md) == 'reject'


def test_parse_ignores_text_outside_scores_section():
    from benchmark.decision_metrics import parse_predicted_decision

    md = '## Summary\n\nRecommendation: Accept\n\n## Scores\n\nFinal Score: 4/10\n'
    assert parse_predicted_decision(md) is None


def test_sklearn_metrics_on_labeled_rows():
    from benchmark.decision_metrics import compute_decision_metrics

    rows = [
        {'predicted_decision': 'accept', 'expected_decision': 'accept'},
        {'predicted_decision': 'reject', 'expected_decision': 'accept'},
    ]
    out = compute_decision_metrics(rows)
    assert 'balanced_accuracy' in out
    assert 0.0 <= out['balanced_accuracy'] <= 1.0
    assert out['accuracy'] == 0.5


def test_score_labeled_runs_skips_unlabeled_rows():
    from benchmark.decision_metrics import score_labeled_runs

    rows = [
        {
            'paper_id': 'p1',
            'job_id': 'j1',
            'venue': 'ICLR',
            'condition': 'KG_ON',
            'expected_decision': 'Accept (Poster)',
            'final_markdown': '## Scores\n\n**Recommendation:** Accept\n',
        },
        {
            'paper_id': 'p2',
            'job_id': 'j2',
            'venue': 'ICLR',
            'condition': 'KG_OFF',
            'expected_decision': None,
            'final_markdown': '## Scores\n\n**Recommendation:** Reject\n',
        },
    ]
    scored = score_labeled_runs(rows)
    assert len(scored) == 1
    assert scored[0]['paper_id'] == 'p1'
    assert scored[0]['predicted_decision'] == 'accept'
    assert scored[0]['decision_correct'] is True


def test_run_decision_metrics_skips_when_no_labels(tmp_path, monkeypatch):
    from benchmark import decision_metrics as module

    manifest_path = tmp_path / 'manifest.jsonl'
    runs_path = tmp_path / 'runs.jsonl'
    output_path = tmp_path / 'decision_metrics.csv'

    manifest_path.write_text(
        json.dumps({'paper_id': 'p1', 'expected_decision': None}) + '\n',
        encoding='utf-8',
    )
    runs_path.write_text(
        json.dumps(
            {
                'paper_id': 'p1',
                'job_id': 'j1',
                'venue': 'ICLR',
                'condition': 'KG_ON',
                'status': 'completed',
                'final_md_bytes': 3000,
                'runtime_seconds': 10.0,
                'input_tokens': 1,
                'output_tokens': 1,
                'total_tokens': 2,
                'paper_search_calls': 0,
            }
        )
        + '\n',
        encoding='utf-8',
    )

    monkeypatch.setattr(module, 'MANIFEST_PATH', manifest_path)
    monkeypatch.setattr(module, 'RUNS_JSONL_PATH', runs_path)

    rows = module.run_decision_metrics(
        runs_path=runs_path,
        manifest_path=manifest_path,
        output_path=output_path,
    )
    assert rows == []
    assert output_path.read_text(encoding='utf-8') == ''


def test_run_decision_metrics_writes_csv(tmp_path):
    from benchmark.decision_metrics import run_decision_metrics

    manifest_path = tmp_path / 'manifest.jsonl'
    runs_path = tmp_path / 'runs.jsonl'
    output_path = tmp_path / 'decision_metrics.csv'

    manifest_path.write_text(
        json.dumps({'paper_id': 'p1', 'expected_decision': 'Accept'}) + '\n',
        encoding='utf-8',
    )
    runs_path.write_text(
        json.dumps(
            {
                'paper_id': 'p1',
                'job_id': 'j1',
                'venue': 'ICLR',
                'condition': 'KG_ON',
                'status': 'completed',
                'final_md_bytes': 3000,
                'runtime_seconds': 10.0,
                'input_tokens': 1,
                'output_tokens': 1,
                'total_tokens': 2,
                'paper_search_calls': 0,
                'final_markdown': '## Scores\n\n**Recommendation:** Accept\n',
            }
        )
        + '\n',
        encoding='utf-8',
    )

    rows = run_decision_metrics(
        runs_path=runs_path,
        manifest_path=manifest_path,
        output_path=output_path,
    )
    assert len(rows) == 1
    assert rows[0]['decision_correct'] is True

    csv_text = output_path.read_text(encoding='utf-8')
    assert 'p1' in csv_text
    assert '_corpus_summary' in csv_text
    assert 'balanced_accuracy' in csv_text or '1.0' in csv_text
