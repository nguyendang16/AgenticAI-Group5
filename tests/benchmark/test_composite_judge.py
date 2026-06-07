import json
import pytest

CORE_JUDGE_METRICS = (
    'factual_correctness',
    'evidence_support',
    'rubric_alignment',
    'criterion_grounded_valid_critique',
)


def test_parse_composite_judge_json():
    from benchmark.judge import parse_composite_judge_response

    raw = json.dumps({
        'factual_correctness': 4,
        'evidence_support': 3.5,
        'rubric_alignment': 5,
        'criterion_grounded_valid_critique': 4,
    })
    scores = parse_composite_judge_response(raw)
    assert scores['rubric_alignment'] == 5.0
    assert set(scores) == set(CORE_JUDGE_METRICS)


def test_composite_judge_run_mocked(monkeypatch):
    from benchmark import judge as judge_module

    class FakeCompositeMetric:
        def measure(self, test_case, **_kwargs):
            return (
                '{"factual_correctness":4,"evidence_support":4,'
                '"rubric_alignment":5,"criterion_grounded_valid_critique":4}'
            )

    monkeypatch.setenv('BENCHMARK_JUDGE_MODE', 'composite')
    monkeypatch.setattr(judge_module, 'build_composite_metric', lambda: FakeCompositeMetric())
    monkeypatch.setattr(judge_module, 'pause_between_eval_calls', lambda: None)
    monkeypatch.setattr(
        judge_module,
        'load_judge_artifacts',
        lambda _job_id: {
            'manuscript_excerpt': 'Paper body',
            'final_markdown': '## Summary\nReview text',
            'criteria_json': '{}',
        },
    )
    row = {
        'job_id': 'j1',
        'paper_id': 'p1',
        'venue': 'ACL',
        'condition': 'KG_ON',
        'manuscript_excerpt': 'Paper body',
        'final_markdown': '## Summary\nReview text',
        'criteria_json': '{}',
    }
    scores = judge_module.judge_run(row)
    assert scores['rubric_alignment'] == 5.0
    assert len([k for k in scores if k in CORE_JUDGE_METRICS]) == 4
