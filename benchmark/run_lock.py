from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

from benchmark.paths import RUN_LOCK_PATH

_LOCK_FIELDS = ('pid', 'started_at', 'paper_id', 'condition')


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_lock() -> dict[str, Any] | None:
    if not RUN_LOCK_PATH.exists():
        return None
    try:
        payload = json.loads(RUN_LOCK_PATH.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def acquire_run_lock(*, paper_id: str = '', condition: str = '') -> None:
    existing = _read_lock()
    if existing is not None:
        pid = int(existing.get('pid') or 0)
        if _pid_alive(pid):
            raise RuntimeError(
                f'Benchmark run lock held by pid {pid} '
                f'({existing.get("paper_id")}/{existing.get("condition")} since {existing.get("started_at")})'
            )
    payload = {
        'pid': os.getpid(),
        'started_at': datetime.now(UTC).isoformat(),
        'paper_id': paper_id,
        'condition': condition,
    }
    RUN_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUN_LOCK_PATH.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def update_run_lock(*, paper_id: str, condition: str) -> None:
    payload = _read_lock() or {}
    payload.update({
        'pid': os.getpid(),
        'paper_id': paper_id,
        'condition': condition,
    })
    RUN_LOCK_PATH.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def release_run_lock() -> None:
    existing = _read_lock()
    if existing is None:
        return
    pid = int(existing.get('pid') or 0)
    if pid == os.getpid() and RUN_LOCK_PATH.exists():
        RUN_LOCK_PATH.unlink()
