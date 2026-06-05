from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from benchmark.checks import validate_run_completion
from benchmark.collect import RUNS_JSONL_PATH
from benchmark.paths import DATA_JOBS_DIR, MANIFEST_PATH, RESULTS_DIR

logger = logging.getLogger(__name__)

DECISION_METRICS_PATH = RESULTS_DIR / 'decision_metrics.csv'
SUMMARY_PAPER_ID = '_corpus_summary'

ACCEPT_TOKENS = ('accept', 'poster', 'oral', 'spotlight')
REJECT_TOKENS = ('reject', 'withdraw')

_SCORES_SECTION_RE = re.compile(r'^##\s+Scores\b.*$', re.IGNORECASE | re.MULTILINE)
_NEXT_SECTION_RE = re.compile(r'^##\s+', re.MULTILINE)
_RECOMMENDATION_RE = re.compile(
    r'(?:\*\*)?(?:recommendation|decision|verdict)(?:\*\*)?\s*:?\s*(.+)',
    re.IGNORECASE,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _extract_scores_section(final_markdown: str) -> str:
    if not final_markdown:
        return ''
    match = _SCORES_SECTION_RE.search(final_markdown)
    if not match:
        return ''
    start = match.end()
    remainder = final_markdown[start:]
    next_heading = _NEXT_SECTION_RE.search(remainder)
    if next_heading:
        remainder = remainder[: next_heading.start()]
    return remainder


def normalize_decision(text: str | None) -> str | None:
    if text is None:
        return None
    normalized = str(text).strip().lower()
    if not normalized:
        return None
    if any(token in normalized for token in ACCEPT_TOKENS):
        return 'accept'
    if any(token in normalized for token in REJECT_TOKENS):
        return 'reject'
    if 'borderline' in normalized or 'weak accept' in normalized or 'weak reject' in normalized:
        return 'borderline'
    return None


def parse_predicted_decision(final_markdown: str) -> str | None:
    scores_section = _extract_scores_section(final_markdown)
    if not scores_section:
        return None

    for line in scores_section.splitlines():
        match = _RECOMMENDATION_RE.search(line)
        if match:
            return normalize_decision(match.group(1))

    return normalize_decision(scores_section)


def _load_final_markdown(row: dict[str, Any]) -> str:
    existing = str(row.get('final_markdown') or '').strip()
    if existing:
        return existing

    job_id = str(row.get('job_id') or '').strip()
    if not job_id:
        return ''

    final_path = DATA_JOBS_DIR / job_id / 'final_report.md'
    if final_path.exists():
        return final_path.read_text(encoding='utf-8')
    return ''


def _load_manifest_labels() -> dict[str, str]:
    labels: dict[str, str] = {}
    for row in _read_jsonl(MANIFEST_PATH):
        paper_id = str(row.get('paper_id') or '').strip()
        expected = row.get('expected_decision')
        if paper_id and expected is not None and str(expected).strip():
            labels[paper_id] = str(expected).strip()
    return labels


def score_labeled_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row in rows:
        expected_raw = row.get('expected_decision')
        if expected_raw is None or not str(expected_raw).strip():
            continue

        expected = normalize_decision(str(expected_raw))
        final_markdown = _load_final_markdown(row)
        predicted = parse_predicted_decision(final_markdown)

        scored.append(
            {
                'paper_id': row.get('paper_id', ''),
                'job_id': row.get('job_id', ''),
                'venue': row.get('venue', ''),
                'condition': row.get('condition', ''),
                'expected_decision': expected,
                'expected_decision_raw': str(expected_raw).strip(),
                'predicted_decision': predicted,
                'decision_correct': (
                    predicted is not None and expected is not None and predicted == expected
                ),
            }
        )
    return scored


def compute_decision_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    eval_rows = [
        row
        for row in rows
        if row.get('expected_decision') is not None and row.get('predicted_decision') is not None
    ]
    if not eval_rows:
        return {
            'labeled_run_count': len(rows),
            'evaluated_run_count': 0,
            'accuracy': 0.0,
            'balanced_accuracy': 0.0,
            'macro_f1': 0.0,
        }

    y_true = [str(row['expected_decision']) for row in eval_rows]
    y_pred = [str(row['predicted_decision']) for row in eval_rows]
    labels = sorted(set(y_true) | set(y_pred))

    return {
        'labeled_run_count': len(rows),
        'evaluated_run_count': len(eval_rows),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)),
    }


def _write_decision_metrics_csv(
    *,
    per_run_rows: list[dict[str, Any]],
    summary: dict[str, float | int],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not per_run_rows:
        output_path.write_text('', encoding='utf-8')
        return

    per_run_fieldnames = list(per_run_rows[0].keys())
    summary_row = {
        'paper_id': SUMMARY_PAPER_ID,
        'job_id': '',
        'venue': '',
        'condition': '',
        'expected_decision': '',
        'expected_decision_raw': '',
        'predicted_decision': '',
        'decision_correct': '',
        **{key: summary[key] for key in summary},
    }
    fieldnames = list(dict.fromkeys([*per_run_fieldnames, *summary.keys()]))

    with output_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(per_run_rows)
        writer.writerow(summary_row)


def run_decision_metrics(
    *,
    runs_path: Path | None = None,
    manifest_path: Path | None = None,
    output_path: Path | None = None,
) -> list[dict[str, Any]]:
    source_runs = runs_path or RUNS_JSONL_PATH
    source_manifest = manifest_path or MANIFEST_PATH
    destination = output_path or DECISION_METRICS_PATH

    manifest_rows = _read_jsonl(source_manifest)
    labeled_paper_ids = {
        str(row.get('paper_id') or '').strip()
        for row in manifest_rows
        if row.get('expected_decision') is not None and str(row.get('expected_decision')).strip()
    }
    if not labeled_paper_ids:
        logger.info('No manifest rows with expected_decision; skipping decision metrics')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text('', encoding='utf-8')
        return []

    labels_by_paper = {
        str(row.get('paper_id') or '').strip(): str(row.get('expected_decision')).strip()
        for row in manifest_rows
        if str(row.get('paper_id') or '').strip() in labeled_paper_ids
    }

    joined_rows: list[dict[str, Any]] = []
    for row in _read_jsonl(source_runs):
        paper_id = str(row.get('paper_id') or '').strip()
        if paper_id not in labels_by_paper:
            continue
        if validate_run_completion(row):
            continue
        enriched = dict(row)
        enriched['expected_decision'] = labels_by_paper[paper_id]
        joined_rows.append(enriched)

    scored_rows = score_labeled_runs(joined_rows)
    summary = compute_decision_metrics(scored_rows)
    _write_decision_metrics_csv(
        per_run_rows=scored_rows,
        summary=summary,
        output_path=destination,
    )
    return scored_rows
