from unittest.mock import MagicMock


def test_build_faithfulness_rows():
    from benchmark.faithfulness import build_ragas_dataset_rows
    from benchmark.claims import AtomicClaim

    claims = [AtomicClaim('id1', 'Weak stats', 'Weaknesses', 'Table 1')]
    rows = build_ragas_dataset_rows(claims)
    assert rows[0]['user_input'] == 'Weak stats'
    assert rows[0]['retrieved_contexts'] == ['Table 1']


def test_score_claims_mocked(monkeypatch):
    from benchmark import faithfulness as mod
    from benchmark import eval_llm

    monkeypatch.setattr(eval_llm, 'build_ragas_llm', lambda: MagicMock())
    monkeypatch.setattr(
        'ragas.evaluate',
        lambda ds, metrics, **_kwargs: {'faithfulness': [0.8]},
    )
    from benchmark.claims import AtomicClaim

    result = mod.score_claims([AtomicClaim('id1', 'text', 'Weaknesses', 'ctx')], job_meta={})
    assert result['faithfulness_mean'] == 0.8
