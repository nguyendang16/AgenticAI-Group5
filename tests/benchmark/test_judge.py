from __future__ import annotations

import pytest

deepeval = pytest.importorskip('deepeval')
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


@pytest.fixture(autouse=True)
def _deepeval_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')


def test_build_rubric_alignment_metric():
    from benchmark.judge import build_rubric_alignment_metric

    metric = build_rubric_alignment_metric()
    assert isinstance(metric, GEval)


@pytest.mark.parametrize(
    'builder_name',
    [
        'build_factual_correctness_metric',
        'build_evidence_support_metric',
        'build_rubric_alignment_metric',
        'build_specificity_metric',
        'build_actionability_metric',
        'build_unsupported_critique_rate_metric',
        'build_criterion_grounded_valid_critique_metric',
    ],
)
def test_build_geval_metrics(builder_name: str):
    import benchmark.judge as judge_module

    builder = getattr(judge_module, builder_name)
    metric = builder()
    assert isinstance(metric, GEval)


def test_load_judge_artifacts(tmp_path, monkeypatch: pytest.MonkeyPatch):
    from benchmark import judge as judge_module

    job_id = 'job-test-1'
    job_dir = tmp_path / job_id
    job_dir.mkdir()
    (job_dir / 'mineru_full.md').write_text('# Paper\n\nBody text.', encoding='utf-8')
    (job_dir / 'final_report.md').write_text('## Summary\n\nReview.', encoding='utf-8')
    (job_dir / 'review_criteria_bundle.json').write_text('{"criteria_count": 1}', encoding='utf-8')

    monkeypatch.setattr(judge_module, 'DATA_JOBS_DIR', tmp_path)

    artifacts = judge_module.load_judge_artifacts(job_id)
    assert artifacts['manuscript_excerpt'].startswith('# Paper')
    assert '## Summary' in artifacts['final_markdown']
    assert 'criteria_count' in artifacts['criteria_json']


def test_judge_run_uses_measure(monkeypatch: pytest.MonkeyPatch):
    from benchmark import judge as judge_module

    captured: list[LLMTestCase] = []

    class FakeMetric:
        def __init__(self, name: str):
            self.name = name

        def measure(self, test_case: LLMTestCase, **_kwargs) -> float:
            captured.append(test_case)
            return 4.0

    def fake_build_all_metrics():
        return {name: FakeMetric(name) for name in judge_module.METRIC_NAMES}

    monkeypatch.setattr(judge_module, 'build_all_metrics', fake_build_all_metrics)
    monkeypatch.setattr(
        judge_module,
        'load_judge_artifacts',
        lambda _job_id: {
            'manuscript_excerpt': 'manuscript body',
            'final_markdown': '## Summary\n\nGood paper.',
            'criteria_json': '{"criteria_count": 2}',
        },
    )

    row = {
        'job_id': 'job-abc',
        'paper_id': 'paper-1',
        'venue': 'ICLR',
        'condition': 'KG_ON',
    }
    scores = judge_module.judge_run(row)

    assert len(captured) == len(judge_module.METRIC_NAMES)
    assert captured[0].input == 'Venue: ICLR'
    assert captured[0].actual_output == '## Summary\n\nGood paper.'
    assert captured[0].context == ['manuscript body', '{"criteria_count": 2}']
    assert scores['paper_id'] == 'paper-1'
    assert scores['factual_correctness'] == 4.0


def test_judge_all_skips_incomplete_runs(tmp_path, monkeypatch: pytest.MonkeyPatch):
    from benchmark import judge as judge_module

    runs_path = tmp_path / 'runs.jsonl'
    runs_path.write_text(
        '\n'.join(
            [
                '{"job_id":"j1","paper_id":"p1","venue":"ICLR","condition":"KG_ON","status":"failed","final_md_bytes":100}',
                '{"job_id":"j2","paper_id":"p1","venue":"ICLR","condition":"KG_OFF","status":"completed","final_md_bytes":5000,"runtime_seconds":10,"input_tokens":1,"output_tokens":1,"total_tokens":2,"paper_search_calls":0}',
            ]
        ),
        encoding='utf-8',
    )
    output_path = tmp_path / 'review_quality_scores.csv'

    judged: list[str] = []

    def fake_judge_run(row):
        judged.append(row['job_id'])
        return {
            'paper_id': row.get('paper_id', ''),
            'job_id': row['job_id'],
            'venue': row.get('venue', ''),
            'condition': row.get('condition', ''),
            'factual_correctness': 3.0,
        }

    monkeypatch.setattr(judge_module, 'judge_run', fake_judge_run)

    results = judge_module.judge_all(runs_path=runs_path, output_path=output_path)
    assert judged == ['j2']
    assert len(results) == 1
    assert output_path.read_text(encoding='utf-8').startswith('paper_id,')
