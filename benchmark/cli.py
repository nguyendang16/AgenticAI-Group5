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

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1

    func: Callable[[argparse.Namespace], int] = args.func
    return func(args)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
