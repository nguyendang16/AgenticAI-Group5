from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from benchmark.env import BENCHMARK_ENV_DEFAULTS, build_benchmark_env
from benchmark.models import PaperRecord, RunRecord
from benchmark.paths import MANIFEST_PATH, REPO_ROOT
from benchmark.registry import upsert_run

logger = logging.getLogger(__name__)

CONDITIONS = ('KG_ON', 'KG_OFF')
BENCHMARK_ENV_KEYS = set(BENCHMARK_ENV_DEFAULTS) | {'REVIEW_VENUE', 'REVIEW_CRITERIA_ENABLED'}


def load_manifest(manifest_path: Path | None = None) -> list[PaperRecord]:
    path = manifest_path or MANIFEST_PATH
    if not path.exists():
        raise FileNotFoundError(f'Manifest not found: {path}')
    records: list[PaperRecord] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        records.append(
            PaperRecord(
                paper_id=data['paper_id'],
                venue=data['venue'],
                title=data['title'],
                pdf_path=data['pdf_path'],
                source=data['source'],
                year=data.get('year'),
                expected_decision=data.get('expected_decision'),
                expected_score=data.get('expected_score'),
                metadata_source=data.get('metadata_source', 'local'),
            )
        )
    return records


def get_git_commit() -> str:
    result = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        logger.warning('git rev-parse failed: %s', result.stderr.strip())
        return ''
    return result.stdout.strip()


def _parse_json_objects(text: str) -> list[dict]:
    decoder = json.JSONDecoder()
    objects: list[dict] = []
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, end = decoder.raw_decode(text, idx)
        objects.append(obj)
        idx = end
    return objects


def _resolve_pdf(paper: PaperRecord) -> Path:
    pdf = Path(paper.pdf_path)
    if not pdf.is_absolute():
        pdf = REPO_ROOT / pdf
    return pdf.resolve()


def _benchmark_env_snapshot(bench_env: dict[str, str]) -> dict[str, str]:
    return {key: bench_env[key] for key in BENCHMARK_ENV_KEYS if key in bench_env}


def _extract_status(watch_stdout: str, *, fallback: str) -> str:
    objects = _parse_json_objects(watch_stdout)
    if not objects:
        return fallback
    last = objects[-1]
    if last.get('status') == 'timeout':
        return str(last.get('current_status') or 'timeout')
    return str(last.get('status') or fallback)


def run_paired_benchmark(
    manifest_path: Path | str | None = None,
    *,
    timeout_seconds: int = 3600,
    dry_run: bool = False,
    paper_id_filter: str | None = None,
) -> int:
    path = Path(manifest_path) if manifest_path is not None else MANIFEST_PATH
    papers = load_manifest(path)
    if paper_id_filter:
        papers = [paper for paper in papers if paper.paper_id == paper_id_filter]
        if not papers:
            logger.error('No paper found with paper_id=%s', paper_id_filter)
            return 1

    git_commit = get_git_commit()
    main_py = str(REPO_ROOT / 'main.py')
    pairs_run = 0

    for paper in papers:
        pdf_path = _resolve_pdf(paper)
        if not dry_run and not pdf_path.exists():
            logger.error('PDF missing for %s: %s', paper.paper_id, pdf_path)
            return 1

        for condition in CONDITIONS:
            bench_env = build_benchmark_env(
                condition,
                venue=paper.venue,
                base=dict(os.environ),
            )
            merged_env = {**os.environ, **bench_env}
            env_snapshot = _benchmark_env_snapshot(bench_env)

            submit_cmd = [
                sys.executable,
                main_py,
                'submit',
                '--pdf',
                str(pdf_path),
                '--wait-seconds',
                '0',
            ]
            watch_cmd = [
                sys.executable,
                main_py,
                'watch',
                '--job-id',
                '<job_id>',
                '--timeout',
                str(timeout_seconds),
            ]

            if dry_run:
                print(f'[{paper.paper_id}/{condition}] submit: {" ".join(submit_cmd)}')
                print(f'  cwd={REPO_ROOT}')
                print(f'  env={json.dumps(env_snapshot, sort_keys=True)}')
                print(f'[{paper.paper_id}/{condition}] watch: {" ".join(watch_cmd)}')
                print(f'  cwd={REPO_ROOT}')
                print(f'  upsert_run paper_id={paper.paper_id} condition={condition} git_commit={git_commit}')
                pairs_run += 1
                continue

            logger.info('Submitting %s / %s', paper.paper_id, condition)
            submit_result = subprocess.run(
                submit_cmd,
                env=merged_env,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            if submit_result.returncode != 0:
                logger.error(
                    'submit failed for %s / %s (exit %d): %s',
                    paper.paper_id,
                    condition,
                    submit_result.returncode,
                    submit_result.stderr.strip() or submit_result.stdout.strip(),
                )
                return 1

            try:
                submit_payload = json.loads(submit_result.stdout)
            except json.JSONDecodeError:
                logger.error(
                    'submit returned invalid JSON for %s / %s: %s',
                    paper.paper_id,
                    condition,
                    submit_result.stdout.strip(),
                )
                return 1

            if submit_payload.get('status') == 'error' or 'job_id' not in submit_payload:
                logger.error(
                    'submit error for %s / %s: %s',
                    paper.paper_id,
                    condition,
                    submit_payload.get('message') or submit_payload,
                )
                return 1

            job_id = str(submit_payload['job_id'])
            initial_status = str(submit_payload.get('status') or '')

            started_at = datetime.now(UTC).isoformat()
            watch_result = subprocess.run(
                [
                    sys.executable,
                    main_py,
                    'watch',
                    '--job-id',
                    job_id,
                    '--timeout',
                    str(timeout_seconds),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            finished_at = datetime.now(UTC).isoformat()

            if watch_result.returncode != 0:
                logger.error(
                    'watch failed for %s / %s job %s (exit %d): %s',
                    paper.paper_id,
                    condition,
                    job_id,
                    watch_result.returncode,
                    watch_result.stderr.strip() or watch_result.stdout.strip(),
                )
                return 1

            status = _extract_status(watch_result.stdout, fallback=initial_status)
            upsert_run(
                RunRecord(
                    paper_id=paper.paper_id,
                    venue=paper.venue,
                    condition=condition,
                    job_id=job_id,
                    status=status,
                    git_commit=git_commit,
                    env_snapshot=env_snapshot,
                ),
                started_at=started_at,
                finished_at=finished_at,
            )
            logger.info('Recorded %s / %s -> job %s status=%s', paper.paper_id, condition, job_id, status)
            pairs_run += 1

    logger.info('Completed %d paired condition(s)', pairs_run)
    return 0
