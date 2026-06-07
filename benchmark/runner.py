from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from benchmark.checks import validate_run_completion
from benchmark.collect import collect_run
from benchmark.env import BENCHMARK_ENV_DEFAULTS, build_benchmark_env
from benchmark.models import PaperRecord, RunRecord
from benchmark.parse_cache import has_parse_cache, save_parse_cache
from benchmark.paths import DATA_JOBS_DIR, MANIFEST_PATH, PARSES_DIR, REPO_ROOT
from benchmark.registry import list_runs, upsert_run
from benchmark.run_attempts import can_submit, record_submit
from benchmark.run_lock import acquire_run_lock, release_run_lock, update_run_lock

logger = logging.getLogger(__name__)

GAP_FILL_PAPER_ORDER = (
    'ets_liu-usingaibasedobject-2023',
    'computers_and_education_1-s2.0-s0360131524002380-main',
    'aaai_t1yqsvzeoo',
    'aaai_2404.01847v3',
)
CONDITIONS = ('KG_OFF', 'KG_ON')
BENCHMARK_ENV_KEYS = set(BENCHMARK_ENV_DEFAULTS) | {'REVIEW_VENUE', 'REVIEW_CRITERIA_ENABLED'}
BASE_PAUSE_SECONDS = 120
TOKENS_PER_PAUSE_STEP = 400_000
PAUSE_STEP_SECONDS = 60
MAX_RATE_LIMIT_RETRIES = 4
RATE_LIMIT_RETRY_SLEEP_SECONDS = 120


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


def _sort_papers(papers: list[PaperRecord]) -> list[PaperRecord]:
    order_index = {paper_id: index for index, paper_id in enumerate(GAP_FILL_PAPER_ORDER)}

    def sort_key(paper: PaperRecord) -> tuple[int, str]:
        return (order_index.get(paper.paper_id, len(GAP_FILL_PAPER_ORDER)), paper.paper_id)

    return sorted(papers, key=sort_key)


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


def _adaptive_pause_seconds(input_tokens: int) -> int:
    extra_steps = math.ceil(max(0, input_tokens) / TOKENS_PER_PAUSE_STEP)
    return BASE_PAUSE_SECONDS + extra_steps * PAUSE_STEP_SECONDS


def _job_has_valid_final_report(job_id: str) -> bool:
    row = collect_run(job_id)
    return not validate_run_completion(row)


def _job_has_parse_artifacts(job_id: str) -> bool:
    md_path = DATA_JOBS_DIR / job_id / 'mineru_full.md'
    return md_path.exists() and md_path.stat().st_size > 0


def _is_rate_limit_failure(job_id: str, status: str) -> bool:
    if status != 'failed':
        return False
    from deepreview.state import load_job_state

    job = load_job_state(job_id)
    if job is None:
        return False
    error_text = f'{job.error or ""} {job.message or ""}'.lower()
    return 'rate_limit' in error_text or 'ratelimiterror' in error_text


def _watch_job(
    *,
    job_id: str,
    main_py: str,
    merged_env: dict[str, str],
    timeout_seconds: int,
    initial_status: str = '',
) -> tuple[str, str, str]:
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
        env={**merged_env, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'},
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        check=False,
    )
    finished_at = datetime.now(UTC).isoformat()

    if watch_result.returncode != 0:
        from deepreview.state import load_job_state

        job = load_job_state(job_id)
        terminal = job is not None and job.status.value in {'completed', 'failed'}
        if not terminal:
            raise RuntimeError(
                f'watch failed for job {job_id} (exit {watch_result.returncode}): '
                f'{watch_result.stderr.strip() or watch_result.stdout.strip()}'
            )
        logger.warning(
            'watch exited %d but job %s is %s; continuing',
            watch_result.returncode,
            job_id,
            job.status.value,
        )

    status = _extract_status(watch_result.stdout, fallback=initial_status)
    return job_id, status, started_at, finished_at


def _resume_failed_job(
    *,
    job_id: str,
    main_py: str,
    merged_env: dict[str, str],
    timeout_seconds: int,
) -> tuple[str, str, str, str] | None:
    resume_result = subprocess.run(
        [sys.executable, main_py, '_run-job', '--job-id', job_id],
        env={**merged_env, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'},
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        check=False,
    )
    if resume_result.returncode != 0:
        logger.warning(
            'resume failed for job %s (exit %d): %s',
            job_id,
            resume_result.returncode,
            resume_result.stderr.strip() or resume_result.stdout.strip(),
        )
        return None
    jid, status, started_at, finished_at = _watch_job(
        job_id=job_id,
        main_py=main_py,
        merged_env=merged_env,
        timeout_seconds=timeout_seconds,
    )
    return jid, status, started_at, finished_at


def _run_one_condition(
    *,
    paper: PaperRecord,
    condition: str,
    pdf_path: Path,
    main_py: str,
    merged_env: dict[str, str],
    timeout_seconds: int,
    prior_job_id: str | None = None,
    prior_status: str | None = None,
    reuse_parse: bool = False,
) -> tuple[str, str, str, str] | None:
    if prior_job_id and prior_status == 'failed' and _job_has_parse_artifacts(prior_job_id):
        logger.info('Resuming failed job %s for %s / %s', prior_job_id, paper.paper_id, condition)
        resumed = _resume_failed_job(
            job_id=prior_job_id,
            main_py=main_py,
            merged_env=merged_env,
            timeout_seconds=timeout_seconds,
        )
        if resumed is not None:
            return resumed
        logger.warning('Resume failed for job %s; will submit new job if allowed', prior_job_id)

    if not can_submit(paper.paper_id, condition):
        logger.error('Submit cap reached for %s / %s', paper.paper_id, condition)
        return None

    submit_cmd = [
        sys.executable,
        main_py,
        'submit',
        '--pdf',
        str(pdf_path),
        '--wait-seconds',
        '0',
    ]
    if reuse_parse and has_parse_cache(paper.paper_id):
        submit_cmd.extend(['--reuse-parse-dir', str(PARSES_DIR / paper.paper_id)])

    for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 1):
        if attempt > 1:
            logger.info(
                'Retrying %s / %s (attempt %d/%d) after rate limit',
                paper.paper_id,
                condition,
                attempt,
                MAX_RATE_LIMIT_RETRIES,
            )
            time.sleep(RATE_LIMIT_RETRY_SLEEP_SECONDS)

        record_submit(paper.paper_id, condition)
        submit_result = subprocess.run(
            submit_cmd,
            env={**merged_env, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'},
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
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
            return None

        try:
            submit_payload = json.loads(submit_result.stdout)
        except json.JSONDecodeError:
            logger.error(
                'submit returned invalid JSON for %s / %s: %s',
                paper.paper_id,
                condition,
                submit_result.stdout.strip(),
            )
            return None

        if submit_payload.get('status') == 'error' or 'job_id' not in submit_payload:
            logger.error(
                'submit error for %s / %s: %s',
                paper.paper_id,
                condition,
                submit_payload.get('message') or submit_payload,
            )
            return None

        job_id = str(submit_payload['job_id'])
        initial_status = str(submit_payload.get('status') or '')
        jid, status, started_at, finished_at = _watch_job(
            job_id=job_id,
            main_py=main_py,
            merged_env=merged_env,
            timeout_seconds=timeout_seconds,
            initial_status=initial_status,
        )
        if status == 'completed':
            return jid, status, started_at, finished_at
        if _is_rate_limit_failure(job_id, status) and attempt < MAX_RATE_LIMIT_RETRIES:
            logger.warning(
                'Rate limit failure for %s / %s job %s; will retry',
                paper.paper_id,
                condition,
                job_id,
            )
            continue
        return jid, status, started_at, finished_at

    return None


def run_paired_benchmark(
    manifest_path: Path | str | None = None,
    *,
    timeout_seconds: int = 3600,
    dry_run: bool = False,
    paper_id_filter: str | None = None,
) -> int:
    path = Path(manifest_path) if manifest_path is not None else MANIFEST_PATH
    papers = _sort_papers(load_manifest(path))
    if paper_id_filter:
        papers = [paper for paper in papers if paper.paper_id == paper_id_filter]
        if not papers:
            logger.error('No paper found with paper_id=%s', paper_id_filter)
            return 1

    if not dry_run:
        try:
            acquire_run_lock()
        except RuntimeError as exc:
            logger.error('%s', exc)
            return 1

    try:
        git_commit = get_git_commit()
        main_py = str(REPO_ROOT / 'main.py')
        pairs_run = 0
        existing_runs = {(run.paper_id, run.condition): run for run in list_runs()}
        last_input_tokens = 0

        for paper in papers:
            pdf_path = _resolve_pdf(paper)
            if not dry_run and not pdf_path.exists():
                logger.error('PDF missing for %s: %s', paper.paper_id, pdf_path)
                return 1

            first_condition_done = False
            for condition in CONDITIONS:
                if not dry_run:
                    update_run_lock(paper_id=paper.paper_id, condition=condition)

                prior = existing_runs.get((paper.paper_id, condition))
                if prior is not None and prior.status == 'completed':
                    logger.info(
                        'Skipping %s / %s (already completed as job %s)',
                        paper.paper_id,
                        condition,
                        prior.job_id,
                    )
                    if condition == CONDITIONS[0]:
                        first_condition_done = _job_has_parse_artifacts(prior.job_id)
                    pairs_run += 1
                    continue

                if prior is not None and _job_has_valid_final_report(prior.job_id):
                    logger.info(
                        'Skipping %s / %s (valid final_report for job %s)',
                        paper.paper_id,
                        condition,
                        prior.job_id,
                    )
                    pairs_run += 1
                    continue

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
                if not first_condition_done and has_parse_cache(paper.paper_id):
                    submit_cmd.extend(['--reuse-parse-dir', str(PARSES_DIR / paper.paper_id)])
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
                    first_condition_done = True
                    continue

                logger.info('Submitting %s / %s', paper.paper_id, condition)
                run_result = _run_one_condition(
                    paper=paper,
                    condition=condition,
                    pdf_path=pdf_path,
                    main_py=main_py,
                    merged_env=merged_env,
                    timeout_seconds=timeout_seconds,
                    prior_job_id=prior.job_id if prior else None,
                    prior_status=prior.status if prior else None,
                    reuse_parse=not first_condition_done and has_parse_cache(paper.paper_id),
                )
                if run_result is None:
                    return 1
                job_id, status, started_at, finished_at = run_result
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
                existing_runs[(paper.paper_id, condition)] = RunRecord(
                    paper_id=paper.paper_id,
                    venue=paper.venue,
                    condition=condition,
                    job_id=job_id,
                    status=status,
                    git_commit=git_commit,
                    env_snapshot=env_snapshot,
                )
                logger.info('Recorded %s / %s -> job %s status=%s', paper.paper_id, condition, job_id, status)

                if status == 'completed' and not first_condition_done:
                    save_parse_cache(paper.paper_id, DATA_JOBS_DIR / job_id)
                    first_condition_done = True

                collected = collect_run(job_id)
                last_input_tokens = int(collected.get('input_tokens') or 0)
                pairs_run += 1

                pause_seconds = _adaptive_pause_seconds(last_input_tokens)
                if pause_seconds > 0:
                    logger.info(
                        'Pausing %ds before next run (TPM cooldown; last_input_tokens=%d)',
                        pause_seconds,
                        last_input_tokens,
                    )
                    time.sleep(pause_seconds)

        logger.info('Completed %d paired condition(s)', pairs_run)
        return 0
    finally:
        if not dry_run:
            release_run_lock()
