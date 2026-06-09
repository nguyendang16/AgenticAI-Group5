from __future__ import annotations

import shutil
from pathlib import Path

from benchmark.paths import PARSES_DIR

_CACHE_FILES = ('mineru_full.md', 'mineru_content_list.json')


def _cache_dir(paper_id: str) -> Path:
    return PARSES_DIR / paper_id


def save_parse_cache(paper_id: str, job_dir: Path) -> None:
    cache = _cache_dir(paper_id)
    cache.mkdir(parents=True, exist_ok=True)
    for name in _CACHE_FILES:
        src = job_dir / name
        if src.exists() and src.stat().st_size > 0:
            shutil.copy2(src, cache / name)


def load_parse_cache(paper_id: str, job_dir: Path) -> bool:
    cache = _cache_dir(paper_id)
    copied = False
    for name in _CACHE_FILES:
        src = cache / name
        if src.exists() and src.stat().st_size > 0:
            job_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, job_dir / name)
            copied = True
    return copied


def has_parse_cache(paper_id: str) -> bool:
    cache = _cache_dir(paper_id)
    md = cache / 'mineru_full.md'
    return md.exists() and md.stat().st_size > 0
