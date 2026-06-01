from __future__ import annotations

import argparse
from pathlib import Path

from src.neo4j_loader import load_all_templates


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Load extracted JSON into Neo4j')
    parser.add_argument('--json_dir', type=Path, default=Path('outputs/extracted_json'))
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Delete all nodes before load (destructive)',
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    count = load_all_templates(args.json_dir.resolve(), clear=args.clear)
    print(f'Loaded {count} templates into Neo4j')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
