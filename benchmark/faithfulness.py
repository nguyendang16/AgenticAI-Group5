from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from benchmark.claims import AtomicClaim, extract_critique_claims
from benchmark.collect import RUNS_JSONL_PATH, collect_run
from benchmark.eval_llm import pause_between_eval_calls
from benchmark.judge import load_judge_artifacts
from benchmark.paths import RESULTS_DIR
from benchmark.registry import list_runs

CLAIM_SCORES_PATH = RESULTS_DIR / 'claim_scores.jsonl'
FAITHFULNESS_RUN_SCORES_PATH = RESULTS_DIR / 'faithfulness_run_scores.csv'

_RUN_SCORE_FIELDS = (
    'job_id',
    'paper_id',
    'venue',
    'condition',
    'faithfulness_mean',
    'faithfulness_n',
)


def build_ragas_dataset_rows(claims: list[AtomicClaim]) -> list[dict[str, Any]]:
    return [
        {
            'user_input': claim.text,
            'response': claim.text,
            'retrieved_contexts': [claim.evidence_span or ''],
        }
        for claim in claims
    ]


def score_claims(claims: list[AtomicClaim], *, job_meta: dict[str, Any]) -> dict[str, Any]:
    if not claims:
        return {'faithfulness_mean': None, 'faithfulness_n': 0, 'claim_rows': []}
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import Faithfulness

    from benchmark.eval_llm import build_ragas_llm

    llm = build_ragas_llm()
    metric = Faithfulness(llm=llm)
    ds = Dataset.from_list(build_ragas_dataset_rows(claims))
    result = evaluate(ds, metrics=[metric])
    scores = list(result['faithfulness'])
    claim_rows = []
    for claim, score in zip(claims, scores):
        claim_rows.append({
            **job_meta,
            'claim_id': claim.claim_id,
            'section': claim.section,
            'claim_text': claim.text,
            'faithfulness': float(score),
        })
    valid = [float(s) for s in scores if s is not None]
    mean = sum(valid) / len(valid) if valid else None
    return {'faithfulness_mean': mean, 'faithfulness_n': len(valid), 'claim_rows': claim_rows}


def _existing_scored_job_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        return {str(row.get('job_id', '')) for row in reader if row.get('job_id')}


def _append_claim_scores(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')


def _append_run_scores(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open('a', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_RUN_SCORE_FIELDS))
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in _RUN_SCORE_FIELDS})


def faithfulness_all(*, job_id: str | None = None) -> list[dict[str, Any]]:
    from benchmark.checks import validate_run_completion
    from benchmark.judge import _read_runs_jsonl

    if job_id:
        rows = [collect_run(job_id)]
        for run in list_runs():
            if run.job_id == job_id:
                rows[0].setdefault('paper_id', run.paper_id)
                rows[0].setdefault('venue', run.venue)
                rows[0].setdefault('condition', run.condition)
                break
    else:
        rows = _read_runs_jsonl(RUNS_JSONL_PATH)

    skip_ids = _existing_scored_job_ids(FAITHFULNESS_RUN_SCORES_PATH)
    run_results: list[dict[str, Any]] = []

    for row in rows:
        if validate_run_completion(row):
            continue
        jid = str(row.get('job_id', ''))
        if jid in skip_ids:
            continue
        artifacts = load_judge_artifacts(jid)
        manuscript = artifacts.get('manuscript_excerpt') or ''
        claims = extract_critique_claims(artifacts.get('final_markdown') or '', manuscript=manuscript)
        meta = {
            'job_id': jid,
            'paper_id': row.get('paper_id', ''),
            'venue': row.get('venue', ''),
            'condition': row.get('condition', ''),
        }
        scored = score_claims(claims, job_meta=meta)
        run_row = {
            **meta,
            'faithfulness_mean': scored['faithfulness_mean'],
            'faithfulness_n': scored['faithfulness_n'],
        }
        run_results.append(run_row)
        _append_claim_scores(CLAIM_SCORES_PATH, scored['claim_rows'])
        _append_run_scores(FAITHFULNESS_RUN_SCORES_PATH, [run_row])
        pause_between_eval_calls()

    return run_results
