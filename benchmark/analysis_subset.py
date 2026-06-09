from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from benchmark.paths import BENCHMARK_DIR

ANALYSIS_SUBSET_PATH = BENCHMARK_DIR / 'analysis_subset.json'


@lru_cache(maxsize=1)
def load_analysis_subset() -> dict[str, Any]:
    payload = json.loads(ANALYSIS_SUBSET_PATH.read_text(encoding='utf-8'))
    included = list(payload.get('included_paper_ids') or [])
    excluded = list(payload.get('excluded_paper_ids') or [])
    return {
        'version': str(payload.get('version') or ''),
        'reason': str(payload.get('reason') or ''),
        'included_paper_ids': included,
        'excluded_paper_ids': excluded,
    }


def filter_paper_ids(paper_ids: list[str], *, use_subset: bool = True) -> list[str]:
    if not use_subset:
        return list(paper_ids)
    included = set(load_analysis_subset()['included_paper_ids'])
    return [pid for pid in paper_ids if pid in included]
