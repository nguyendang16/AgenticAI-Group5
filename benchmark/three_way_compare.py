from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchmark.compare import apply_paper_id_filter
from benchmark.faithfulness import FAITHFULNESS_RUN_SCORES_PATH
from benchmark.judge import METRIC_NAMES, REVIEW_QUALITY_SCORES_PATH
from benchmark.paths import THREE_WAY_SUMMARY_PATH

CONDITIONS = ('TRAD_LLM', 'KG_ON', 'KG_OFF')
OVERALL_PAPER_ID = '__overall__'
THREE_WAY_METRICS = (
    'rubric_alignment',
    'factual_correctness',
    'evidence_support',
    'faithfulness_mean',
    *(
        metric
        for metric in METRIC_NAMES
        if metric not in {'rubric_alignment', 'factual_correctness', 'evidence_support'}
    ),
)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        path.write_text('', encoding='utf-8')
        return
    frame.to_csv(path, index=False)


def _median_for(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors='coerce').dropna()
    if numeric.empty:
        return None
    return round(float(numeric.median()), 10)


def build_three_way_summary(
    *,
    judge_df: pd.DataFrame | None = None,
    faithfulness_df: pd.DataFrame | None = None,
    use_subset: bool = True,
    output_path: Path | None = None,
) -> pd.DataFrame:
    judge = judge_df.copy() if judge_df is not None else _read_csv(REVIEW_QUALITY_SCORES_PATH)
    faithfulness = (
        faithfulness_df.copy()
        if faithfulness_df is not None
        else _read_csv(FAITHFULNESS_RUN_SCORES_PATH)
    )

    if judge.empty:
        summary = pd.DataFrame()
        _write_dataframe(output_path or THREE_WAY_SUMMARY_PATH, summary)
        return summary

    if not faithfulness.empty and 'job_id' in judge.columns and 'job_id' in faithfulness.columns:
        faith_cols = [
            col for col in ('job_id', 'faithfulness_mean') if col in faithfulness.columns
        ]
        judge = judge.merge(faithfulness[faith_cols], on='job_id', how='left')

    judge['condition'] = judge['condition'].astype(str).str.strip().str.upper()
    judge = judge[judge['condition'].isin(CONDITIONS)].copy()
    judge = apply_paper_id_filter(judge, use_subset=use_subset)

    if judge.empty:
        summary = pd.DataFrame()
        _write_dataframe(output_path or THREE_WAY_SUMMARY_PATH, summary)
        return summary

    paper_rows: list[dict[str, object]] = []
    for paper_id, group in judge.groupby('paper_id', sort=True):
        row: dict[str, object] = {'paper_id': paper_id}
        for metric in THREE_WAY_METRICS:
            if metric not in group.columns:
                continue
            for condition in CONDITIONS:
                subset = group[group['condition'] == condition]
                row[f'median_{metric}_{condition}'] = _median_for(subset[metric])
        paper_rows.append(row)

    summary = pd.DataFrame(paper_rows)

    overall: dict[str, object] = {'paper_id': OVERALL_PAPER_ID}
    metric_cols = [col for col in summary.columns if col != 'paper_id']
    for col in metric_cols:
        overall[col] = _median_for(summary[col])
    summary = pd.concat([summary, pd.DataFrame([overall])], ignore_index=True)

    _write_dataframe(output_path or THREE_WAY_SUMMARY_PATH, summary)
    return summary
