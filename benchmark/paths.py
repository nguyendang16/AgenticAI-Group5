from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
KG_TEST_PAPERS_DIR = REPO_ROOT / 'KG_testPapers'
BENCHMARK_DIR = REPO_ROOT / 'benchmark'
PAPERS_DIR = BENCHMARK_DIR / 'papers'
INCOMING_DIR = PAPERS_DIR / 'incoming'
RESULTS_DIR = BENCHMARK_DIR / 'results'
PAIRWISE_JUDGE_SCORES_PATH = RESULTS_DIR / 'pairwise_judge_scores.csv'
MANIFEST_PATH = BENCHMARK_DIR / 'manifest.jsonl'
REGISTRY_PATH = BENCHMARK_DIR / 'registry.sqlite'
DATA_JOBS_DIR = REPO_ROOT / 'data' / 'jobs'
PARSES_DIR = BENCHMARK_DIR / 'parses'
RUN_LOCK_PATH = BENCHMARK_DIR / '.run.lock'
RUN_ATTEMPTS_PATH = BENCHMARK_DIR / 'run_attempts.json'
TRAD_LLM_DIR = REPO_ROOT / 'trad_LLM'
TRAD_REVIEWS_DIR = BENCHMARK_DIR / 'trad_reviews'
TRAD_REGISTRY_PATH = BENCHMARK_DIR / 'trad_registry.jsonl'
TRAD_RUNS_JSONL_PATH = RESULTS_DIR / 'trad_runs.jsonl'
TRAD_PAIRWISE_JUDGE_SCORES_PATH = RESULTS_DIR / 'trad_pairwise_judge_scores.csv'
THREE_WAY_SUMMARY_PATH = RESULTS_DIR / 'three_way_summary.csv'
