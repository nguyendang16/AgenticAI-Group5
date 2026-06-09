def test_submit_cap(tmp_path, monkeypatch):
    from benchmark import run_attempts as mod
    from benchmark.paths import RUN_ATTEMPTS_PATH

    attempts_path = tmp_path / 'run_attempts.json'
    monkeypatch.setattr(mod, 'RUN_ATTEMPTS_PATH', attempts_path)

    assert mod.can_submit('paper-a', 'KG_OFF')
    mod.record_submit('paper-a', 'KG_OFF')
    assert mod.can_submit('paper-a', 'KG_OFF')
    mod.record_submit('paper-a', 'KG_OFF')
    assert not mod.can_submit('paper-a', 'KG_OFF')
    assert mod.submit_count('paper-a', 'KG_OFF') == 2
