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


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

    parser = argparse.ArgumentParser(prog='benchmark', description='KG benchmark harness')
    subparsers = parser.add_subparsers(dest='command')

    build_parser = subparsers.add_parser('build-manifest', help='Build benchmark/manifest.jsonl')
    build_parser.set_defaults(func=_cmd_build_manifest)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1

    func: Callable[[argparse.Namespace], int] = args.func
    return func(args)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
