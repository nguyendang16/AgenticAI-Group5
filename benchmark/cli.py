from __future__ import annotations

import argparse
import logging
import sys
from typing import Callable

from benchmark.progress import ProgressReporter, phase_banner, pipeline_exit, warn_if_live

PIPELINE_REVIEWS_STEPS: tuple[str, ...] = ('build-manifest', 'run')
PIPELINE_EVAL_STEPS: tuple[str, ...] = (
    'collect',
    'check',
    'judge',
    'pairwise-judge',
    'faithfulness',
    'compare',
    'report',
)


def _cmd_build_manifest(args: argparse.Namespace) -> int:
    from benchmark.build_manifest import build_manifest

    local_only = getattr(args, 'local_only', False)
    rows = build_manifest(local_only=local_only)
    print(f'Wrote {len(rows)} papers to benchmark/manifest.jsonl')
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from benchmark.runner import run_paired_benchmark

    return run_paired_benchmark(
        timeout_seconds=args.timeout,
        dry_run=args.dry_run,
        paper_id_filter=args.paper_id,
    )


def _cmd_collect(_args: argparse.Namespace) -> int:
    from benchmark.collect import harvest_all

    rows = harvest_all()
    print(f'Wrote {len(rows)} runs to benchmark/results/runs.jsonl')
    return 0


def _cmd_check(_args: argparse.Namespace) -> int:
    from benchmark.checks import DETERMINISTIC_SCORES_PATH, run_all_checks

    scores = run_all_checks()
    print(f'Wrote {len(scores)} rows to {DETERMINISTIC_SCORES_PATH}')
    return 0


def _cmd_judge(args: argparse.Namespace) -> int:
    from benchmark.judge import REVIEW_QUALITY_SCORES_PATH, judge_all, judge_trad_all

    if getattr(args, 'source', None) == 'trad':
        scores = judge_trad_all(paper_id=args.paper_id)
    else:
        scores = judge_all(job_id=args.job_id)
    print(f'Wrote judge scores ({len(scores)} new rows) to {REVIEW_QUALITY_SCORES_PATH}')
    return 0


def _cmd_pairwise_judge(args: argparse.Namespace) -> int:
    from benchmark.pairwise_judge import PAIRWISE_JUDGE_SCORES_PATH, pairwise_all

    rows = pairwise_all(paper_id=args.paper_id)
    print(f'Wrote {len(rows)} rows to {PAIRWISE_JUDGE_SCORES_PATH}')
    return 0


def _cmd_trad_pairwise_judge(args: argparse.Namespace) -> int:
    from benchmark.paths import TRAD_PAIRWISE_JUDGE_SCORES_PATH
    from benchmark.trad_pairwise_judge import trad_pairwise_all

    rows = trad_pairwise_all(paper_id=args.paper_id, use_subset=not args.all_papers)
    print(f'Wrote {len(rows)} rows to {TRAD_PAIRWISE_JUDGE_SCORES_PATH}')
    return 0


def _cmd_faithfulness(args: argparse.Namespace) -> int:
    from benchmark.faithfulness import (
        FAITHFULNESS_RUN_SCORES_PATH,
        faithfulness_all,
        faithfulness_trad_all,
    )

    if getattr(args, 'source', None) == 'trad':
        rows = faithfulness_trad_all(paper_id=args.paper_id)
    else:
        rows = faithfulness_all(job_id=args.job_id)
    print(f'Wrote {len(rows)} rows to {FAITHFULNESS_RUN_SCORES_PATH}')
    return 0


def _cmd_decision_metrics(_args: argparse.Namespace) -> int:
    try:
        from benchmark.decision_metrics import DECISION_METRICS_PATH, run_decision_metrics
    except ImportError as exc:
        print(f'Skipping decision-metrics: {exc}')
        return 0

    rows = run_decision_metrics()
    print(f'Wrote {len(rows)} labeled runs to {DECISION_METRICS_PATH}')
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from benchmark.compare import (
        OVERALL_SUMMARY_PATH,
        PAIRED_COMPARISON_PATH,
        VENUE_SUMMARY_PATH,
        build_paired_comparison,
    )

    metadata = build_paired_comparison(use_subset=not args.all_papers)
    print(
        'Wrote paired comparison outputs: '
        f'{PAIRED_COMPARISON_PATH}, {OVERALL_SUMMARY_PATH}, {VENUE_SUMMARY_PATH} '
        f'({metadata["valid_pairs"]} valid pairs, {metadata["excluded_pairs"]} excluded)'
    )
    return 0


def _cmd_report(_args: argparse.Namespace) -> int:
    from benchmark.report import BENCHMARK_SUMMARY_PATH, generate_report

    path = generate_report(rebuild_comparison=True)
    print(f'Wrote {path}')
    return 0


def _cmd_archive_results(args: argparse.Namespace) -> int:
    from benchmark.archive_results import archive_eval_results

    archive_eval_results(tag=args.tag)
    return 0


def _dispatch_step(name: str, args: argparse.Namespace) -> int:
    handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        'build-manifest': _cmd_build_manifest,
        'run': _cmd_run,
        'collect': _cmd_collect,
        'check': _cmd_check,
        'judge': _cmd_judge,
        'pairwise-judge': _cmd_pairwise_judge,
        'faithfulness': _cmd_faithfulness,
        'decision-metrics': _cmd_decision_metrics,
        'compare': _cmd_compare,
        'report': _cmd_report,
    }
    handler = handlers.get(name)
    if handler is None:
        print(f'Unknown pipeline step: {name}', file=sys.stderr)
        return 1
    warn_if_live(name)
    return handler(args)


def _cmd_pipeline(args: argparse.Namespace) -> int:
    if args.phase == 'reviews':
        steps = PIPELINE_REVIEWS_STEPS
        phase_banner('reviews', detail='OpenAI reviews only — gap-fill missing papers')
    elif args.phase == 'eval':
        steps = PIPELINE_EVAL_STEPS
        phase_banner(
            'eval',
            detail='OpenAI judge + pairwise + Gemma RAGAS — run >=60 min after reviews',
        )
    else:
        print(f'Unknown phase: {args.phase}', file=sys.stderr)
        return 1

    if args.phase == 'reviews' and not args.dry_run:
        print('Reminder: kill orphan review workers before starting.', flush=True)

    progress = ProgressReporter(label=args.phase, total=len(steps))
    progress.start()

    for step in steps:
        progress.advance(step)
        if args.dry_run and step == 'build-manifest':
            print(
                'DRY-RUN: skip build-manifest (use existing benchmark/manifest.jsonl).',
                flush=True,
            )
            continue
        rc = _dispatch_step(step, args)
        if rc != 0:
            return pipeline_exit(rc, step)

    progress.finish('success')
    if args.phase == 'reviews' and not args.dry_run:
        print('Wait ≥60 minutes before: python -m benchmark pipeline --phase eval', flush=True)
    return 0


def _cmd_all(args: argparse.Namespace) -> int:
    steps: list[tuple[str, Callable[[argparse.Namespace], int]]] = [
        ('build-manifest', _cmd_build_manifest),
        ('run', _cmd_run),
        ('collect', _cmd_collect),
        ('check', _cmd_check),
        ('judge', _cmd_judge),
        ('decision-metrics', _cmd_decision_metrics),
        ('compare', _cmd_compare),
        ('report', _cmd_report),
    ]

    for name, handler in steps:
        print(f'==> benchmark {name}')
        rc = handler(args)
        if rc != 0:
            print(f'benchmark {name} failed with exit code {rc}')
            return rc
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

    epilog = (
        'Operator rerun (production): use explicit phases, not "all".\n'
        '  python -m benchmark pipeline --phase reviews\n'
        '  # wait ≥60 min\n'
        '  python -m benchmark pipeline --phase eval\n'
        'See docs/operator-benchmark-rerun.md'
    )
    parser = argparse.ArgumentParser(
        prog='benchmark',
        description='KG benchmark harness',
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command')

    build_parser = subparsers.add_parser('build-manifest', help='Build benchmark/manifest.jsonl')
    build_parser.add_argument(
        '--local-only',
        action='store_true',
        help='Use existing benchmark/papers only (no OpenReview/journal fetch)',
    )
    build_parser.set_defaults(func=_cmd_build_manifest, local_only=False)

    run_parser = subparsers.add_parser('run', help='Run paired KG_ON/KG_OFF benchmark')
    run_parser.add_argument('--dry-run', action='store_true', help='Print planned commands only')
    run_parser.add_argument('--paper-id', default=None, help='Run a single paper from the manifest')
    run_parser.add_argument('--timeout', type=int, default=3600, help='Watch timeout in seconds')
    run_parser.set_defaults(func=_cmd_run)

    collect_parser = subparsers.add_parser('collect', help='Harvest job artifacts to runs.jsonl')
    collect_parser.set_defaults(func=_cmd_collect)

    check_parser = subparsers.add_parser('check', help='Run deterministic validators on runs.jsonl')
    check_parser.set_defaults(func=_cmd_check)

    judge_parser = subparsers.add_parser('judge', help='DeepEval report-level quality scoring')
    judge_parser.add_argument('--job-id', default=None, help='Judge a single benchmark job')
    judge_parser.add_argument(
        '--source',
        choices=('agent', 'trad'),
        default='agent',
        help='Review source: agent benchmark jobs or TRAD_LLM PDF reviews',
    )
    judge_parser.add_argument('--paper-id', default=None, help='Judge a single paper (trad source)')
    judge_parser.set_defaults(func=_cmd_judge, local_only=False)

    pairwise_parser = subparsers.add_parser(
        'pairwise-judge',
        help='OpenAI pairwise comparison of KG_ON vs KG_OFF reviews per paper',
    )
    pairwise_parser.add_argument('--paper-id', default=None, help='Compare a single paper')
    pairwise_parser.set_defaults(func=_cmd_pairwise_judge)

    trad_pairwise_parser = subparsers.add_parser(
        'trad-pairwise-judge',
        help='OpenAI pairwise comparison of TRAD vs KG_OFF/KG_ON reviews per paper',
    )
    trad_pairwise_parser.add_argument('--paper-id', default=None, help='Compare a single paper')
    trad_pairwise_parser.add_argument(
        '--all-papers',
        action='store_true',
        help='Include all papers; default uses analysis subset',
    )
    trad_pairwise_parser.set_defaults(func=_cmd_trad_pairwise_judge)

    faithfulness_parser = subparsers.add_parser(
        'faithfulness',
        help='RAGAS claim-level faithfulness scoring',
    )
    faithfulness_parser.add_argument('--job-id', default=None, help='Score a single benchmark job')
    faithfulness_parser.add_argument(
        '--source',
        choices=('agent', 'trad'),
        default='agent',
        help='Review source: agent benchmark jobs or TRAD_LLM PDF reviews',
    )
    faithfulness_parser.add_argument(
        '--paper-id',
        default=None,
        help='Score a single paper (trad source)',
    )
    faithfulness_parser.set_defaults(func=_cmd_faithfulness, local_only=False)

    pipeline_parser = subparsers.add_parser(
        'pipeline',
        help='Run operator phases with terminal progress (see docs/operator-benchmark-rerun.md)',
    )
    pipeline_parser.add_argument(
        '--phase',
        required=True,
        choices=('reviews', 'eval'),
        help='reviews=gap-fill OpenAI; eval=OpenAI judge+pairwise+Gemma RAGAS+report',
    )
    pipeline_parser.add_argument('--dry-run', action='store_true', help='Pass --dry-run to review run step')
    pipeline_parser.add_argument('--paper-id', default=None, help='Run a single paper (reviews phase)')
    pipeline_parser.add_argument('--timeout', type=int, default=3600, help='Watch timeout for review run')
    pipeline_parser.add_argument('--job-id', default=None, help='Single job for judge/faithfulness in eval phase')
    pipeline_parser.set_defaults(func=_cmd_pipeline, local_only=True)

    decision_metrics_parser = subparsers.add_parser(
        'decision-metrics',
        help='sklearn accept/reject metrics for labeled manifest papers',
    )
    decision_metrics_parser.set_defaults(func=_cmd_decision_metrics)

    compare_parser = subparsers.add_parser('compare', help='Build paired KG_ON vs KG_OFF comparison CSVs')
    compare_parser.add_argument(
        '--all-papers',
        action='store_true',
        help='Include all papers; default uses B2a analysis subset',
    )
    compare_parser.set_defaults(func=_cmd_compare)

    report_parser = subparsers.add_parser('report', help='Generate benchmark_summary.md')
    report_parser.set_defaults(func=_cmd_report)

    archive_parser = subparsers.add_parser(
        'archive-results',
        help='Copy active eval result files to benchmark/results/archive/{tag}/',
    )
    archive_parser.add_argument(
        '--tag',
        default='pre-v2',
        help='Archive subdirectory name (default: pre-v2)',
    )
    archive_parser.set_defaults(func=_cmd_archive_results)

    all_parser = subparsers.add_parser(
        'all',
        help='Run build-manifest through report (full benchmark pipeline; not for production reruns)',
    )
    all_parser.add_argument('--dry-run', action='store_true', help='Print planned run commands only')
    all_parser.add_argument('--paper-id', default=None, help='Run a single paper from the manifest')
    all_parser.add_argument('--timeout', type=int, default=3600, help='Watch timeout in seconds')
    all_parser.set_defaults(func=_cmd_all)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1

    func: Callable[[argparse.Namespace], int] = args.func
    return func(args)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
