from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from benchmark.paths import REPO_ROOT, TRAD_REGISTRY_PATH


def trad_job_id(paper_id: str) -> str:
    return f'trad:{paper_id}'


@lru_cache(maxsize=1)
def load_trad_registry() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in TRAD_REGISTRY_PATH.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def registry_row_for_paper(paper_id: str) -> dict[str, Any] | None:
    for row in load_trad_registry():
        if row.get('paper_id') == paper_id:
            return row
    return None


def resolve_source_pdf(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path
