from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark.analysis_subset import load_analysis_subset
from benchmark.compare import PAIRED_COMPARISON_PATH
from benchmark.faithfulness import FAITHFULNESS_RUN_SCORES_PATH
from benchmark.judge import REVIEW_QUALITY_SCORES_PATH
from benchmark.paths import RESULTS_DIR, TRAD_PAIRWISE_JUDGE_SCORES_PATH
from benchmark.checks import DETERMINISTIC_SCORES_PATH

COMBINED_RESULTS_PATH = RESULTS_DIR / 'benchmark_results_combined.csv'
PAIRWISE_KG_PATH = RESULTS_DIR / 'pairwise_judge_scores.csv'

_JUDGE_METRICS = (
    'factual_correctness',
    'evidence_support',
    'rubric_alignment',
    'criterion_grounded_valid_critique',
)

_ENGINEERING = (
    'runtime_seconds',
    'total_tokens',
    'annotation_count',
    'criteria_count',
)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _condition_metrics(
    judge: pd.DataFrame,
    faith: pd.DataFrame,
    det: pd.DataFrame,
    *,
    paper_id: str,
    condition: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    prefix = condition.lower()

    j = judge[(judge['paper_id'] == paper_id) & (judge['condition'] == condition)]
    if not j.empty:
        row = j.iloc[0]
        for metric in _JUDGE_METRICS:
            out[f'{prefix}_{metric}'] = row.get(metric)

    f = faith[(faith['paper_id'] == paper_id) & (faith['condition'] == condition)]
    if not f.empty:
        row = f.iloc[0]
        out[f'{prefix}_faithfulness_mean'] = row.get('faithfulness_mean')
        out[f'{prefix}_faithfulness_n'] = row.get('faithfulness_n')

    d = det[(det['paper_id'] == paper_id) & (det['condition'] == condition)]
    if not d.empty:
        row = d.iloc[0]
        for col in _ENGINEERING:
            out[f'{prefix}_{col}'] = row.get(col)
        out[f'{prefix}_job_id'] = row.get('job_id')

    return out


def build_combined_results_csv(*, output_path: Path | None = None) -> Path:
    subset = load_analysis_subset()
    included = set(subset['included_paper_ids'])

    judge = _read_csv(REVIEW_QUALITY_SCORES_PATH)
    faith = _read_csv(FAITHFULNESS_RUN_SCORES_PATH)
    det = _read_csv(DETERMINISTIC_SCORES_PATH)
    paired = _read_csv(PAIRED_COMPARISON_PATH)
    trad_pw = _read_csv(TRAD_PAIRWISE_JUDGE_SCORES_PATH)

    if judge.empty:
        raise FileNotFoundError(f'No judge scores at {REVIEW_QUALITY_SCORES_PATH}')

    paper_ids = sorted(judge['paper_id'].astype(str).unique())
    rows: list[dict[str, Any]] = []

    for paper_id in paper_ids:
        venue = str(judge[judge['paper_id'] == paper_id].iloc[0].get('venue', ''))
        row: dict[str, Any] = {
            'paper_id': paper_id,
            'venue': venue,
            'in_analysis_subset': paper_id in included,
        }

        for condition in ('KG_OFF', 'KG_ON', 'TRAD_LLM'):
            row.update(_condition_metrics(judge, faith, det, paper_id=paper_id, condition=condition))

        if not paired.empty and paper_id in paired['paper_id'].astype(str).values:
            p = paired[paired['paper_id'] == paper_id].iloc[0]
            row['pairwise_kg_winner'] = p.get('pairwise_winner')
            row['pairwise_kg_rubric_winner'] = p.get('pairwise_rubric_winner')
            row['pairwise_kg_confidence'] = p.get('pairwise_confidence')
            row['delta_rubric_alignment'] = p.get('delta_rubric_alignment')
            row['delta_faithfulness_mean'] = p.get('delta_faithfulness_mean')
            row['delta_total_tokens'] = p.get('delta_total_tokens')

        if not trad_pw.empty:
            off = trad_pw[
                (trad_pw['paper_id'] == paper_id) & (trad_pw['pair_type'] == 'TRAD_VS_KG_OFF')
            ]
            on = trad_pw[
                (trad_pw['paper_id'] == paper_id) & (trad_pw['pair_type'] == 'TRAD_VS_KG_ON')
            ]
            if not off.empty:
                row['trad_vs_kg_off_winner'] = off.iloc[0].get('winner')
            if not on.empty:
                row['trad_vs_kg_on_winner'] = on.iloc[0].get('winner')

        rows.append(row)

    destination = output_path or COMBINED_RESULTS_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(destination, index=False, quoting=csv.QUOTE_MINIMAL)
    return destination
