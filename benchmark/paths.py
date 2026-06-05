from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
KG_TEST_PAPERS_DIR = REPO_ROOT / 'KG_testPapers'
BENCHMARK_DIR = REPO_ROOT / 'benchmark'
PAPERS_DIR = BENCHMARK_DIR / 'papers'
RESULTS_DIR = BENCHMARK_DIR / 'results'
MANIFEST_PATH = BENCHMARK_DIR / 'manifest.jsonl'
REGISTRY_PATH = BENCHMARK_DIR / 'registry.sqlite'
DATA_JOBS_DIR = REPO_ROOT / 'data' / 'jobs'
