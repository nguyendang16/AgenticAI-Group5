import json
import os

import pytest


def test_acquire_and_release_lock(tmp_path, monkeypatch):
    from benchmark import run_lock as mod
    from benchmark.paths import RUN_LOCK_PATH

    lock_path = tmp_path / '.run.lock'
    monkeypatch.setattr(mod, 'RUN_LOCK_PATH', lock_path)

    mod.acquire_run_lock(paper_id='p1', condition='KG_OFF')
    payload = json.loads(lock_path.read_text(encoding='utf-8'))
    assert payload['pid'] == os.getpid()
    assert payload['paper_id'] == 'p1'

    mod.release_run_lock()
    assert not lock_path.exists()


def test_acquire_raises_when_lock_held(tmp_path, monkeypatch):
    from benchmark import run_lock as mod

    lock_path = tmp_path / '.run.lock'
    monkeypatch.setattr(mod, 'RUN_LOCK_PATH', lock_path)
    monkeypatch.setattr(mod, '_pid_alive', lambda _pid: True)

    lock_path.write_text(
        json.dumps({'pid': 99999, 'started_at': 't', 'paper_id': 'x', 'condition': 'KG_ON'}),
        encoding='utf-8',
    )

    with pytest.raises(RuntimeError, match='lock held'):
        mod.acquire_run_lock()
