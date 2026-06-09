from unittest.mock import MagicMock


def test_build_faithfulness_rows():
    from benchmark.faithfulness import build_ragas_dataset_rows
    from benchmark.claims import AtomicClaim

    claims = [
        AtomicClaim(
            'id1',
            'Weak stats',
            'Weaknesses',
            'Table 1',
            context_source='section_slice',
            resolved_context_preview='Table 1 preview',
        )
    ]
    rows = build_ragas_dataset_rows(claims)
    assert rows[0]['user_input'] == 'Weak stats'
    assert rows[0]['retrieved_contexts'] == ['Table 1']


def test_score_claims_mocked(monkeypatch):
    from benchmark import faithfulness as mod
    from benchmark import eval_llm

    monkeypatch.setattr(eval_llm, 'build_faithfulness_llm', lambda: MagicMock())
    monkeypatch.setattr(
        'ragas.evaluate',
        lambda ds, metrics, **_kwargs: {'faithfulness': [0.8]},
    )
    from benchmark.claims import AtomicClaim

    result = mod.score_claims([AtomicClaim('id1', 'text', 'Weaknesses', 'ctx')], job_meta={})
    assert result['faithfulness_mean'] == 0.8


def test_faithfulness_emits_context_source(monkeypatch):
    from benchmark import faithfulness as mod
    from benchmark import eval_llm
    from benchmark.claims import AtomicClaim

    monkeypatch.setattr(eval_llm, 'build_faithfulness_llm', lambda: MagicMock())
    monkeypatch.setattr(
        'ragas.evaluate',
        lambda ds, metrics, **_kwargs: {'faithfulness': [0.5]},
    )
    claim = AtomicClaim(
        'id1',
        'Stats weak',
        'Weaknesses',
        'We report 230% increase',
        context_source='section_slice',
        resolved_context_preview='We report 230%',
    )
    result = mod.score_claims([claim], job_meta={'job_id': 'jid'})
    row = result['claim_rows'][0]
    assert row['context_source'] == 'section_slice'
    assert row['resolved_context_preview'] == 'We report 230%'
    assert row['eval_version'] == 'v2'
    assert result['faithfulness_mean'] == 0.5
