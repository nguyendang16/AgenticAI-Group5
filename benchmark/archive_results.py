from __future__ import annotations

import shutil
from pathlib import Path

from benchmark.compare import PAIRED_COMPARISON_PATH
from benchmark.faithfulness import CLAIM_SCORES_PATH, FAITHFULNESS_RUN_SCORES_PATH
from benchmark.judge import REVIEW_QUALITY_SCORES_PATH
from benchmark.paths import PAIRWISE_JUDGE_SCORES_PATH, RESULTS_DIR
from benchmark.report import BENCHMARK_SUMMARY_PATH

ARCHIVE_RESULT_FILES: tuple[Path, ...] = (
    REVIEW_QUALITY_SCORES_PATH,
    FAITHFULNESS_RUN_SCORES_PATH,
    CLAIM_SCORES_PATH,
    PAIRED_COMPARISON_PATH,
    BENCHMARK_SUMMARY_PATH,
    PAIRWISE_JUDGE_SCORES_PATH,
)


def archive_eval_results(tag: str = 'pre-v2') -> Path:
    """Copy active result CSVs/JSONL to benchmark/results/archive/{tag}/."""
    archive_dir = RESULTS_DIR / 'archive' / tag
    archive_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    skipped: list[str] = []

    for source in ARCHIVE_RESULT_FILES:
        if not source.exists() or source.stat().st_size == 0:
            skipped.append(source.name)
            continue
        destination = archive_dir / source.name
        shutil.copy2(source, destination)
        copied.append(source.name)

    print(f'Archived {len(copied)} file(s) to {archive_dir}')
    if copied:
        print('  copied:', ', '.join(copied))
    if skipped:
        print('  skipped (missing or empty):', ', '.join(skipped))

    return archive_dir
