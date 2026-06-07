from __future__ import annotations

import pytest


def test_parse_pairwise_response():
    from benchmark.pairwise_judge import parse_pairwise_response

    raw = (
        '{"winner":"KG_ON","rubric_alignment_winner":"KG_ON",'
        '"confidence":"high","one_line_reason":"More rubric coverage"}'
    )
    parsed = parse_pairwise_response(raw)
    assert parsed['winner'] == 'KG_ON'
    assert parsed['reason'] == 'More rubric coverage'


def test_pairwise_skips_incomplete_pair(monkeypatch: pytest.MonkeyPatch, tmp_path):
    from benchmark import pairwise_judge as pairwise_module

    runs_path = tmp_path / 'runs.jsonl'
    runs_path.write_text(
        '\n'.join(
            [
                (
                    '{"job_id":"j1","paper_id":"p1","venue":"ICLR","condition":"KG_ON",'
                    '"status":"completed","final_md_bytes":5000,"runtime_seconds":10,'
                    '"input_tokens":1,"output_tokens":1,"total_tokens":2,"paper_search_calls":0}'
                ),
            ]
        ),
        encoding='utf-8',
    )
    output_path = tmp_path / 'pairwise_judge_scores.csv'
    compared: list[tuple[dict, dict]] = []

    def fake_pairwise_compare(kg_off_row, kg_on_row):
        compared.append((kg_off_row, kg_on_row))
        return {
            'paper_id': kg_off_row['paper_id'],
            'venue': kg_off_row['venue'],
            'winner': 'KG_ON',
            'rubric_alignment_winner': 'KG_ON',
            'confidence': 'high',
            'reason': 'test',
            'job_id_kg_on': kg_on_row['job_id'],
            'job_id_kg_off': kg_off_row['job_id'],
        }

    monkeypatch.setattr(pairwise_module, 'pairwise_compare', fake_pairwise_compare)
    monkeypatch.setattr(pairwise_module, 'pause_between_eval_calls', lambda: None)

    results = pairwise_module.pairwise_all(runs_path=runs_path, output_path=output_path)
    assert compared == []
    assert results == []
    assert not output_path.exists() or output_path.read_text(encoding='utf-8') == ''


def test_pairwise_compare_builds_result(monkeypatch: pytest.MonkeyPatch):
    from benchmark import judge as judge_module
    from benchmark import pairwise_judge as pairwise_module

    monkeypatch.setattr(
        judge_module,
        'load_judge_artifacts',
        lambda _job_id: {
            'manuscript_excerpt': 'manuscript',
            'final_markdown': 'review text',
            'criteria_json': '{"criteria_count": 1}',
        },
    )
    monkeypatch.setattr(
        pairwise_module,
        '_call_judge_llm',
        lambda _prompt: (
            '{"winner":"KG_OFF","rubric_alignment_winner":"KG_OFF",'
            '"confidence":"medium","one_line_reason":"Clearer evidence"}'
        ),
    )

    kg_off = {'job_id': 'off-1', 'paper_id': 'p1', 'venue': 'ICLR', 'condition': 'KG_OFF'}
    kg_on = {'job_id': 'on-1', 'paper_id': 'p1', 'venue': 'ICLR', 'condition': 'KG_ON'}
    result = pairwise_module.pairwise_compare(kg_off, kg_on)

    assert result['winner'] == 'KG_OFF'
    assert result['reason'] == 'Clearer evidence'
    assert result['job_id_kg_off'] == 'off-1'
    assert result['job_id_kg_on'] == 'on-1'
