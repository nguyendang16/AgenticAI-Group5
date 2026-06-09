from __future__ import annotations

import json
from typing import Any

from benchmark.paths import RUN_ATTEMPTS_PATH

MAX_SUBMITS_PER_CONDITION = 2


def _load_attempts() -> dict[str, Any]:
    if not RUN_ATTEMPTS_PATH.exists():
        return {}
    try:
        payload = json.loads(RUN_ATTEMPTS_PATH.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_attempts(data: dict[str, Any]) -> None:
    RUN_ATTEMPTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUN_ATTEMPTS_PATH.write_text(json.dumps(data, indent=2, sort_keys=True), encoding='utf-8')


def _attempt_key(paper_id: str, condition: str) -> str:
    return f'{paper_id}::{condition}'


def submit_count(paper_id: str, condition: str) -> int:
    data = _load_attempts()
    return int(data.get(_attempt_key(paper_id, condition), 0))


def can_submit(paper_id: str, condition: str, *, max_submits: int = MAX_SUBMITS_PER_CONDITION) -> bool:
    return submit_count(paper_id, condition) < max_submits


def record_submit(paper_id: str, condition: str) -> int:
    data = _load_attempts()
    key = _attempt_key(paper_id, condition)
    count = int(data.get(key, 0)) + 1
    data[key] = count
    _save_attempts(data)
    return count
