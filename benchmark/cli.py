from __future__ import annotations

import argparse
import logging
import sys
from typing import Callable


def _cmd_build_manifest(_args: argparse.Namespace) -> int:
    from benchmark.build_manifest import build_manifest

    rows = build_manifest()
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
    from benchmark.judge import REVIEW_QUALITY_SCORES_PATH, judge_all

    scores = judge_all(job_id=args.job_id)
    print(f'Wrote {len(scores)} rows to {REVIEW_QUALITY_SCORES_PATH}')
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


def _cmd_compare(_args: argparse.Namespace) -> int:
    from benchmark.compare import (
        OVERALL_SUMMARY_PATH,
        PAIRED_COMPARISON_PATH,
        VENUE_SUMMARY_PATH,
        build_paired_comparison,
    )

    metadata = build_paired_comparison()
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

    parser = argparse.ArgumentParser(prog='benchmark', description='KG benchmark harness')
    subparsers = parser.add_subparsers(dest='command')

    build_parser = subparsers.add_parser('build-manifest', help='Build benchmark/manifest.jsonl')
    build_parser.set_defaults(func=_cmd_build_manifest)

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
    judge_parser.set_defaults(func=_cmd_judge)

    decision_metrics_parser = subparsers.add_parser(
        'decision-metrics',
        help='sklearn accept/reject metrics for labeled manifest papers',
    )
    decision_metrics_parser.set_defaults(func=_cmd_decision_metrics)

    compare_parser = subparsers.add_parser('compare', help='Build paired KG_ON vs KG_OFF comparison CSVs')
    compare_parser.set_defaults(func=_cmd_compare)

    report_parser = subparsers.add_parser('report', help='Generate benchmark_summary.md')
    report_parser.set_defaults(func=_cmd_report)

    all_parser = subparsers.add_parser(
        'all',
        help='Run build-manifest through report (full benchmark pipeline)',
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
