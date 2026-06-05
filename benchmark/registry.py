from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from benchmark.models import RunRecord
from benchmark.paths import REGISTRY_PATH

RUNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY,
  paper_id TEXT NOT NULL,
  venue TEXT NOT NULL,
  condition TEXT NOT NULL,
  job_id TEXT NOT NULL,
  status TEXT,
  git_commit TEXT,
  started_at TEXT,
  finished_at TEXT,
  env_json TEXT,
  UNIQUE(paper_id, condition)
);
"""


def init_registry(db_path: Path | None = None) -> Path:
    path = db_path or REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(RUNS_SCHEMA)
    return path


def upsert_run(
    record: RunRecord,
    *,
    started_at: str | None = None,
    finished_at: str | None = None,
    db_path: Path | None = None,
) -> None:
    path = db_path or REGISTRY_PATH
    init_registry(path)
    env_json = json.dumps(record.env_snapshot, sort_keys=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO runs (
              paper_id, venue, condition, job_id, status, git_commit,
              started_at, finished_at, env_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(paper_id, condition) DO UPDATE SET
              venue = excluded.venue,
              job_id = excluded.job_id,
              status = excluded.status,
              git_commit = excluded.git_commit,
              started_at = COALESCE(excluded.started_at, runs.started_at),
              finished_at = COALESCE(excluded.finished_at, runs.finished_at),
              env_json = excluded.env_json
            """,
            (
                record.paper_id,
                record.venue,
                record.condition,
                record.job_id,
                record.status,
                record.git_commit,
                started_at,
                finished_at,
                env_json,
            ),
        )


def _row_to_run_record(row: sqlite3.Row) -> RunRecord:
    env_snapshot: dict[str, str] = {}
    if row['env_json']:
        env_snapshot = json.loads(row['env_json'])
    return RunRecord(
        paper_id=row['paper_id'],
        venue=row['venue'],
        condition=row['condition'],
        job_id=row['job_id'],
        status=row['status'] or '',
        git_commit=row['git_commit'] or '',
        env_snapshot=env_snapshot,
    )


def list_runs(db_path: Path | None = None) -> list[RunRecord]:
    path = db_path or REGISTRY_PATH
    if not path.exists():
        return []
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            'SELECT * FROM runs ORDER BY paper_id, condition'
        ).fetchall()
    return [_row_to_run_record(row) for row in rows]
