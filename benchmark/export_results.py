from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark.analysis_subset import load_analysis_subset
from benchmark.compare import PAIRED_COMPARISON_PATH
from benchmark.faithfulness import FAITHFULNESS_RUN_SCORES_PATH
from benchmark.judge import REVIEW_QUALITY_SCORES_PATH
from benchmark.paths import RESULTS_DIR, THREE_WAY_SUMMARY_PATH, TRAD_PAIRWISE_JUDGE_SCORES_PATH
from benchmark.checks import DETERMINISTIC_SCORES_PATH

COMBINED_RESULTS_PATH = RESULTS_DIR / 'benchmark_results_combined.csv'
WORKBOOK_RESULTS_PATH = RESULTS_DIR / 'benchmark_results.xlsx'
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

_CONDITIONS = ('TRAD_LLM', 'KG_OFF', 'KG_ON')


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

    d = pd.DataFrame()
    if not det.empty and 'paper_id' in det.columns and 'condition' in det.columns:
        d = det[(det['paper_id'] == paper_id) & (det['condition'] == condition)]
    if not d.empty:
        row = d.iloc[0]
        for col in _ENGINEERING:
            out[f'{prefix}_{col}'] = row.get(col)
        out[f'{prefix}_job_id'] = row.get('job_id')

    return out


def build_combined_results_frame() -> pd.DataFrame:
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

        for condition in _CONDITIONS:
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

    return pd.DataFrame(rows)


def build_combined_results_csv(*, output_path: Path | None = None) -> Path:
    destination = output_path or COMBINED_RESULTS_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = build_combined_results_frame()
    frame.to_csv(destination, index=False, quoting=csv.QUOTE_MINIMAL)
    return destination


def _overview_sheet(combined: pd.DataFrame, three_way: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    subset = load_analysis_subset()
    subset_papers = combined[combined['in_analysis_subset'] == True]  # noqa: E712

    rows: list[dict[str, Any]] = [
        {'section': 'Analysis subset', 'metric': 'included_papers', 'value': len(subset['included_paper_ids'])},
        {'section': 'Analysis subset', 'metric': 'rationale', 'value': subset['reason']},
    ]

    if not paired.empty and 'pairwise_winner' in paired.columns:
        winners = paired['pairwise_winner'].fillna('').astype(str).str.strip().value_counts()
        for winner in ('KG_ON', 'KG_OFF', 'tie'):
            rows.append({
                'section': 'KG_ON vs KG_OFF (pairwise)',
                'metric': f'{winner}_wins',
                'value': int(winners.get(winner, 0)),
            })

    overall = three_way[three_way['paper_id'].astype(str) == '__overall__'] if not three_way.empty else pd.DataFrame()
    if not overall.empty:
        row = overall.iloc[0]
        for metric in ('rubric_alignment', 'factual_correctness', 'evidence_support', 'faithfulness_mean'):
            for condition in _CONDITIONS:
                col = f'median_{metric}_{condition}'
                if col in row.index:
                    rows.append({
                        'section': 'Three-way medians (7-paper subset)',
                        'metric': f'{metric}_{condition}',
                        'value': row.get(col),
                    })

    trad_pw_wins = _read_csv(TRAD_PAIRWISE_JUDGE_SCORES_PATH)
    if not trad_pw_wins.empty:
        for pair_type, group in trad_pw_wins.groupby('pair_type', sort=True):
            for winner, count in group['winner'].fillna('').astype(str).str.strip().value_counts().items():
                if winner:
                    rows.append({
                        'section': str(pair_type),
                        'metric': f'{winner}_wins',
                        'value': int(count),
                    })

    if not subset_papers.empty and 'delta_total_tokens' in subset_papers.columns:
        tokens = pd.to_numeric(subset_papers['delta_total_tokens'], errors='coerce').dropna()
        if not tokens.empty:
            rows.append({
                'section': 'Engineering',
                'metric': 'median_delta_total_tokens',
                'value': float(tokens.median()),
            })

    return pd.DataFrame(rows)


def _judge_comparison_sheet(judge: pd.DataFrame, *, subset_only: bool = True) -> pd.DataFrame:
    if judge.empty:
        return pd.DataFrame()

    frame = judge.copy()
    frame['condition'] = frame['condition'].astype(str).str.strip().str.upper()
    frame = frame[frame['condition'].isin(_CONDITIONS)]

    if subset_only:
        included = set(load_analysis_subset()['included_paper_ids'])
        frame = frame[frame['paper_id'].astype(str).isin(included)]

    long_rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        for metric in _JUDGE_METRICS:
            if metric not in row.index:
                continue
            long_rows.append({
                'paper_id': row.get('paper_id'),
                'venue': row.get('venue'),
                'condition': row.get('condition'),
                'metric': metric,
                'score': row.get(metric),
            })
    return pd.DataFrame(long_rows)


def _faithfulness_sheet(faith: pd.DataFrame, *, subset_only: bool = True) -> pd.DataFrame:
    if faith.empty:
        return pd.DataFrame()

    frame = faith.copy()
    frame['condition'] = frame['condition'].astype(str).str.strip().str.upper()
    frame = frame[frame['condition'].isin(_CONDITIONS)]

    if subset_only:
        included = set(load_analysis_subset()['included_paper_ids'])
        frame = frame[frame['paper_id'].astype(str).isin(included)]

    cols = [c for c in ('paper_id', 'venue', 'condition', 'faithfulness_mean', 'faithfulness_n', 'job_id') if c in frame.columns]
    return frame[cols].sort_values(['paper_id', 'condition']).reset_index(drop=True)


def _metric_matrix_sheet(three_way: pd.DataFrame) -> pd.DataFrame:
    if three_way.empty:
        return pd.DataFrame()

    overall = three_way[three_way['paper_id'].astype(str) == '__overall__']
    if overall.empty:
        return pd.DataFrame()

    row = overall.iloc[0]
    metrics = ('rubric_alignment', 'factual_correctness', 'evidence_support', 'faithfulness_mean')
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        entry: dict[str, Any] = {'metric': metric}
        for condition in _CONDITIONS:
            col = f'median_{metric}_{condition}'
            entry[condition] = row.get(col) if col in row.index else None
        rows.append(entry)
    return pd.DataFrame(rows)


def build_results_workbook(*, output_path: Path | None = None) -> Path:
    destination = output_path or WORKBOOK_RESULTS_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)

    combined = build_combined_results_frame()
    judge = _read_csv(REVIEW_QUALITY_SCORES_PATH)
    faith = _read_csv(FAITHFULNESS_RUN_SCORES_PATH)
    paired = _read_csv(PAIRED_COMPARISON_PATH)
    three_way = _read_csv(THREE_WAY_SUMMARY_PATH)
    trad_pw = _read_csv(TRAD_PAIRWISE_JUDGE_SCORES_PATH)

    subset_combined = combined[combined['in_analysis_subset'] == True].copy()  # noqa: E712
    id_cols = ['paper_id', 'venue', 'in_analysis_subset']
    judge_cols = [f'{c.lower()}_{m}' for c in ('trad_llm', 'kg_off', 'kg_on') for m in (*_JUDGE_METRICS, 'faithfulness_mean', 'faithfulness_n')]
    pairwise_cols = [
        'pairwise_kg_winner',
        'pairwise_kg_rubric_winner',
        'trad_vs_kg_off_winner',
        'trad_vs_kg_on_winner',
        'delta_rubric_alignment',
        'delta_faithfulness_mean',
        'delta_total_tokens',
    ]
    ordered = id_cols + [c for c in judge_cols if c in subset_combined.columns] + [c for c in pairwise_cols if c in subset_combined.columns]
    subset_combined = subset_combined[ordered]

    with pd.ExcelWriter(destination, engine='openpyxl') as writer:
        _overview_sheet(combined, three_way, paired).to_excel(writer, sheet_name='Overview', index=False)
        _metric_matrix_sheet(three_way).to_excel(writer, sheet_name='Metric Matrix', index=False)
        subset_combined.to_excel(writer, sheet_name='By Paper (subset)', index=False)
        combined.to_excel(writer, sheet_name='By Paper (all)', index=False)
        _judge_comparison_sheet(judge).to_excel(writer, sheet_name='Judge (long)', index=False)
        _faithfulness_sheet(faith).to_excel(writer, sheet_name='Faithfulness', index=False)
        if not paired.empty:
            paired.to_excel(writer, sheet_name='KG Pairwise', index=False)
        if not trad_pw.empty:
            trad_pw.to_excel(writer, sheet_name='Trad Pairwise', index=False)
        if not three_way.empty:
            three_way.to_excel(writer, sheet_name='Three-Way Detail', index=False)

    return destination


def export_all_results() -> tuple[Path, Path]:
    csv_path = build_combined_results_csv()
    xlsx_path = build_results_workbook()
    return csv_path, xlsx_path
