#!/usr/bin/env python3
"""Harvest Tier 1 evaluation metrics from data/jobs/*/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deepreview.config import get_settings
from deepreview.evaluation.tier1 import (
    evaluate_job_dir,
    evaluate_jobs_root,
    save_job_evaluation,
    write_evaluation_report,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Tier 1 evaluation harvest (see docs/2026-06-01-tiered-system-evaluation-design.md)',
    )
    parser.add_argument(
        '--jobs-root',
        type=Path,
        default=None,
        help='Directory containing job folders (default: <data_dir>/jobs)',
    )
    parser.add_argument(
        '--job-id',
        type=str,
        default=None,
        help='Evaluate a single job id instead of the whole corpus',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/evaluations/latest'),
        help='Where to write eval_results.json, eval_run_sheet.csv, eval_summary.md',
    )
    parser.add_argument(
        '--save-per-job',
        action='store_true',
        help='Also write evaluation.json into each job directory',
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Print aggregate JSON to stdout',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    settings = get_settings()
    jobs_root = args.jobs_root or (settings.data_dir / 'jobs')

    if args.job_id:
        job_dir = jobs_root / args.job_id
        rows = [evaluate_job_dir(job_dir)]
    else:
        rows = evaluate_jobs_root(jobs_root)

    if not rows:
        print(f'No jobs found under {jobs_root}', file=sys.stderr)
        return 1

    paths = write_evaluation_report(rows, output_dir=args.output_dir)
    print(f'Wrote {paths["json"]}')
    print(f'Wrote {paths["csv"]}')
    print(f'Wrote {paths["markdown"]}')

    if args.save_per_job:
        for row in rows:
            job_id = str(row.get('job_id') or '').strip()
            if not job_id:
                continue
            eval_path = save_job_evaluation(jobs_root / job_id, row)
            print(f'Wrote {eval_path}')

    aggregate = json.loads(paths['json'].read_text(encoding='utf-8')).get('aggregate', {})
    print(
        f"Verdict: {aggregate.get('verdict', 'n/a')} · "
        f"completion={aggregate.get('completed_count')}/{aggregate.get('job_count')} · "
        f"median_oqi={aggregate.get('median_oqi')}"
    )

    if args.json:
        print(json.dumps({'jobs': rows, 'aggregate': aggregate}, indent=2, ensure_ascii=False))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
