from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from benchmark.analysis_subset import filter_paper_ids
from benchmark.checks import DETERMINISTIC_SCORES_PATH
from benchmark.decision_metrics import SUMMARY_PAPER_ID
from benchmark.faithfulness import FAITHFULNESS_RUN_SCORES_PATH
from benchmark.judge import METRIC_NAMES, REVIEW_QUALITY_SCORES_PATH
from benchmark.paths import PAIRWISE_JUDGE_SCORES_PATH, RESULTS_DIR

PAIRED_COMPARISON_PATH = RESULTS_DIR / 'paired_comparison.csv'
OVERALL_SUMMARY_PATH = RESULTS_DIR / 'overall_summary.csv'
VENUE_SUMMARY_PATH = RESULTS_DIR / 'venue_summary.csv'
DECISION_METRICS_PATH = RESULTS_DIR / 'decision_metrics.csv'

_DETERMINISTIC_NUMERIC = (
    'runtime_seconds',
    'input_tokens',
    'output_tokens',
    'total_tokens',
    'tool_calls',
    'annotation_count',
    'criteria_count',
    'paper_search_calls',
    'final_md_bytes',
)

_MIN_WILCOXON_N = 8
_MIN_DECISION_VENUE_N = 5
_BOOTSTRAP_ITERATIONS = 2000
_BOOTSTRAP_SEED = 42


def paired_wilcoxon_pvalue(deltas: pd.Series) -> float | None:
    if len(deltas) < _MIN_WILCOXON_N:
        return None
    cleaned = deltas.dropna()
    if len(cleaned) < _MIN_WILCOXON_N:
        return None
    _stat, p = wilcoxon(cleaned)
    return float(p)


def bootstrap_median_ci(
    values: pd.Series,
    *,
    n_boot: int = _BOOTSTRAP_ITERATIONS,
    alpha: float = 0.05,
    random_state: int = _BOOTSTRAP_SEED,
) -> tuple[float | None, float | None]:
    cleaned = values.dropna().to_numpy(dtype=float)
    if cleaned.size == 0:
        return None, None

    rng = np.random.default_rng(random_state)
    boot_medians = np.empty(n_boot, dtype=float)
    sample_size = cleaned.size
    for index in range(n_boot):
        sample = rng.choice(cleaned, size=sample_size, replace=True)
        boot_medians[index] = np.median(sample)

    lower = float(np.percentile(boot_medians, 100 * alpha / 2))
    upper = float(np.percentile(boot_medians, 100 * (1 - alpha / 2)))
    return lower, upper


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _merge_run_tables(
    deterministic: pd.DataFrame,
    judge: pd.DataFrame,
    decision: pd.DataFrame,
    faithfulness: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if deterministic.empty:
        return pd.DataFrame()

    merged = deterministic.copy()
    if not judge.empty:
        judge_cols = ['paper_id', 'condition', *METRIC_NAMES]
        judge_subset = judge[[col for col in judge_cols if col in judge.columns]].copy()
        merged = merged.merge(judge_subset, on=['paper_id', 'condition'], how='left')

    if faithfulness is not None and not faithfulness.empty:
        faith_cols = [col for col in ('job_id', 'faithfulness_mean', 'faithfulness_n') if col in faithfulness.columns]
        if 'job_id' in faith_cols and 'job_id' in merged.columns:
            merged = merged.merge(
                faithfulness[faith_cols].copy(),
                on='job_id',
                how='left',
            )

    if not decision.empty:
        decision = decision[decision['paper_id'] != SUMMARY_PAPER_ID].copy()
        decision_cols = [
            col
            for col in (
                'paper_id',
                'condition',
                'predicted_decision',
                'expected_decision',
                'decision_correct',
            )
            if col in decision.columns
        ]
        decision_subset = decision[decision_cols].copy()
        merged = merged.merge(decision_subset, on=['paper_id', 'condition'], how='left')

    return merged


def _truthy(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, (int, float)) and value == 1:
        return True
    return str(value).strip().lower() in {'true', '1', 'yes'}


def _is_valid_pair_row(row: pd.Series, *, condition: str) -> bool:
    suffix = '_kg_on' if condition == 'KG_ON' else '_kg_off'

    if not _truthy(row.get(f'condition_valid{suffix}')):
        return False
    if not _truthy(row.get(f'completion_valid{suffix}')):
        return False

    condition_errors = row.get(f'condition_errors{suffix}')
    if isinstance(condition_errors, str) and condition_errors.strip():
        return False

    if condition == 'KG_ON':
        try:
            if float(row.get(f'criteria_count{suffix}') or 0) <= 0:
                return False
        except (TypeError, ValueError):
            return False

    return True


def _build_valid_pairs(merged: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if merged.empty:
        return pd.DataFrame(), 0

    required = {'paper_id', 'condition'}
    if not required.issubset(merged.columns):
        return pd.DataFrame(), 0

    valid = merged.copy()
    if 'condition_valid' in valid.columns:
        valid = valid[valid['condition_valid'].astype(str).str.lower().isin({'true', '1'})]
    if 'completion_valid' in valid.columns:
        valid = valid[valid['completion_valid'].astype(str).str.lower().isin({'true', '1'})]
    if 'condition_errors' in valid.columns:
        valid = valid[valid['condition_errors'].fillna('').astype(str).str.strip() == '']

    kg_on = valid[valid['condition'] == 'KG_ON'].copy()
    kg_off = valid[valid['condition'] == 'KG_OFF'].copy()

    candidate_paper_ids = set(kg_on['paper_id']).intersection(set(kg_off['paper_id']))
    excluded = len(set(merged['paper_id'].unique())) - len(candidate_paper_ids)

    if not candidate_paper_ids:
        return pd.DataFrame(), max(excluded, 0)

    kg_on = kg_on[kg_on['paper_id'].isin(candidate_paper_ids)]
    kg_off = kg_off[kg_off['paper_id'].isin(candidate_paper_ids)]

    pairs = kg_on.merge(kg_off, on='paper_id', suffixes=('_kg_on', '_kg_off'))

    venue_col = 'venue_kg_on' if 'venue_kg_on' in pairs.columns else 'venue'
    if venue_col in pairs.columns:
        pairs['venue'] = pairs[venue_col]
    elif 'venue' not in pairs.columns:
        pairs['venue'] = ''

    mask = pairs.apply(
        lambda row: _is_valid_pair_row(row, condition='KG_ON')
        and _is_valid_pair_row(row, condition='KG_OFF'),
        axis=1,
    )
    invalid_within_candidates = int((~mask).sum())
    pairs = pairs[mask].copy()
    excluded += invalid_within_candidates

    kg_on_criteria = pairs.get('criteria_count_kg_on')
    if kg_on_criteria is not None:
        pairs = pairs[pd.to_numeric(kg_on_criteria, errors='coerce').fillna(0) > 0]

    return pairs, excluded


def _numeric_delta_columns() -> tuple[str, ...]:
    return (*_DETERMINISTIC_NUMERIC, *METRIC_NAMES, 'faithfulness_mean', 'decision_correct')


def _merge_pairwise_judge(
    paired: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> pd.DataFrame:
    if paired.empty:
        return paired

    output = paired.copy()
    if pairwise.empty or 'paper_id' not in pairwise.columns:
        for col in ('pairwise_winner', 'pairwise_rubric_winner', 'pairwise_confidence'):
            output[col] = pd.NA
        return output

    pairwise_cols = [
        col
        for col in ('paper_id', 'winner', 'rubric_alignment_winner', 'confidence')
        if col in pairwise.columns
    ]
    subset = pairwise[pairwise_cols].copy()
    subset = subset.rename(
        columns={
            'winner': 'pairwise_winner',
            'rubric_alignment_winner': 'pairwise_rubric_winner',
            'confidence': 'pairwise_confidence',
        }
    )
    return output.merge(subset, on='paper_id', how='left')


def _compute_pair_deltas(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pairs

    output = pairs[['paper_id', 'venue']].copy()
    for metric in _numeric_delta_columns():
        on_col = f'{metric}_kg_on'
        off_col = f'{metric}_kg_off'
        if on_col in pairs.columns and off_col in pairs.columns:
            on_values = pd.to_numeric(pairs[on_col], errors='coerce')
            off_values = pd.to_numeric(pairs[off_col], errors='coerce')
            if metric == 'decision_correct':
                on_values = pairs[on_col].map(_truthy).astype('Int64')
                off_values = pairs[off_col].map(_truthy).astype('Int64')
            output[f'delta_{metric}'] = on_values - off_values

    if 'job_id_kg_on' in pairs.columns:
        output['job_id_kg_on'] = pairs['job_id_kg_on']
    if 'job_id_kg_off' in pairs.columns:
        output['job_id_kg_off'] = pairs['job_id_kg_off']

    return output


def _sklearn_decision_metrics(rows: pd.DataFrame) -> dict[str, float | None]:
    if rows.empty or 'expected_decision' not in rows.columns or 'predicted_decision' not in rows.columns:
        return {'balanced_accuracy': None, 'macro_f1': None}

    labeled = rows.dropna(subset=['expected_decision', 'predicted_decision']).copy()
    labeled = labeled[
        labeled['expected_decision'].astype(str).str.strip().ne('')
        & labeled['predicted_decision'].astype(str).str.strip().ne('')
    ]
    if labeled.empty:
        return {'balanced_accuracy': None, 'macro_f1': None}

    from sklearn.metrics import balanced_accuracy_score, f1_score

    y_true = labeled['expected_decision'].astype(str).tolist()
    y_pred = labeled['predicted_decision'].astype(str).tolist()
    labels = sorted(set(y_true) | set(y_pred))

    try:
        balanced = float(balanced_accuracy_score(y_true, y_pred))
    except ValueError:
        balanced = None

    try:
        macro_f1 = float(f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0))
    except ValueError:
        macro_f1 = None

    return {'balanced_accuracy': balanced, 'macro_f1': macro_f1}


def _append_decision_metric_deltas(
    paired: pd.DataFrame,
    merged: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | None], dict[str, dict[str, float | None]]]:
    if merged.empty or 'expected_decision' not in merged.columns:
        return paired, {}, {}

    overall_on = _sklearn_decision_metrics(merged[merged['condition'] == 'KG_ON'])
    overall_off = _sklearn_decision_metrics(merged[merged['condition'] == 'KG_OFF'])
    overall = {
        'delta_balanced_accuracy': _metric_delta(overall_on['balanced_accuracy'], overall_off['balanced_accuracy']),
        'delta_macro_f1': _metric_delta(overall_on['macro_f1'], overall_off['macro_f1']),
    }

    venue_deltas: dict[str, dict[str, float | None]] = {}
    if 'venue' in merged.columns:
        for venue, group in merged.groupby('venue'):
            on_rows = group[group['condition'] == 'KG_ON']
            off_rows = group[group['condition'] == 'KG_OFF']
            labeled_ids = set(on_rows['paper_id']).intersection(set(off_rows['paper_id']))
            if len(labeled_ids) < _MIN_DECISION_VENUE_N:
                continue
            on_metrics = _sklearn_decision_metrics(on_rows[on_rows['paper_id'].isin(labeled_ids)])
            off_metrics = _sklearn_decision_metrics(off_rows[off_rows['paper_id'].isin(labeled_ids)])
            venue_deltas[str(venue)] = {
                'delta_balanced_accuracy': _metric_delta(
                    on_metrics['balanced_accuracy'],
                    off_metrics['balanced_accuracy'],
                ),
                'delta_macro_f1': _metric_delta(on_metrics['macro_f1'], off_metrics['macro_f1']),
            }

    return paired, overall, venue_deltas


def _metric_delta(kg_on: float | None, kg_off: float | None) -> float | None:
    if kg_on is None or kg_off is None:
        return None
    return float(kg_on - kg_off)


def _summarize_deltas(paired: pd.DataFrame) -> pd.DataFrame:
    delta_cols = [col for col in paired.columns if col.startswith('delta_')]
    rows: list[dict[str, Any]] = []

    for metric_col in delta_cols:
        series = paired[metric_col]
        ci_low, ci_high = bootstrap_median_ci(series)
        rows.append(
            {
                'metric': metric_col.removeprefix('delta_'),
                'n_pairs': int(series.dropna().shape[0]),
                'median_delta': float(series.median()) if series.dropna().size else None,
                'mean_delta': float(series.mean()) if series.dropna().size else None,
                'bootstrap_ci_low': ci_low,
                'bootstrap_ci_high': ci_high,
                'wilcoxon_pvalue': paired_wilcoxon_pvalue(series),
            }
        )

    return pd.DataFrame(rows)


def _summarize_by_venue(
    paired: pd.DataFrame,
    venue_decision_deltas: dict[str, dict[str, float | None]],
) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()

    delta_cols = [col for col in paired.columns if col.startswith('delta_')]
    grouped = paired.groupby('venue', dropna=False)
    rows: list[dict[str, Any]] = []

    for venue, group in grouped:
        row: dict[str, Any] = {
            'venue': venue,
            'n_pairs': int(len(group)),
        }
        for metric_col in delta_cols:
            metric = metric_col.removeprefix('delta_')
            series = group[metric_col]
            row[f'median_delta_{metric}'] = float(series.median()) if series.dropna().size else None
            row[f'mean_delta_{metric}'] = float(series.mean()) if series.dropna().size else None
            ci_low, ci_high = bootstrap_median_ci(series)
            row[f'bootstrap_ci_low_{metric}'] = ci_low
            row[f'bootstrap_ci_high_{metric}'] = ci_high
            row[f'wilcoxon_pvalue_{metric}'] = paired_wilcoxon_pvalue(series)

        venue_key = str(venue)
        if venue_key in venue_decision_deltas:
            row.update(venue_decision_deltas[venue_key])

        rows.append(row)

    return pd.DataFrame(rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return

    fieldnames = list(rows[0].keys())
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def apply_paper_id_filter(frame: pd.DataFrame, *, use_subset: bool = True) -> pd.DataFrame:
    if frame.empty or 'paper_id' not in frame.columns:
        return frame
    allowed = filter_paper_ids(frame['paper_id'].astype(str).tolist(), use_subset=use_subset)
    allowed_set = set(allowed)
    return frame[frame['paper_id'].astype(str).isin(allowed_set)].copy()


def _write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        path.write_text('', encoding='utf-8')
        return
    frame.to_csv(path, index=False)


def build_paired_comparison(
    *,
    deterministic_path: Path | None = None,
    judge_path: Path | None = None,
    decision_path: Path | None = None,
    pairwise_path: Path | None = None,
    paired_output_path: Path | None = None,
    overall_output_path: Path | None = None,
    venue_output_path: Path | None = None,
    use_subset: bool = True,
) -> dict[str, Any]:
    deterministic = _read_csv(deterministic_path or DETERMINISTIC_SCORES_PATH)
    judge = _read_csv(judge_path or REVIEW_QUALITY_SCORES_PATH)
    decision = _read_csv(decision_path or DECISION_METRICS_PATH)
    faithfulness = _read_csv(FAITHFULNESS_RUN_SCORES_PATH)
    pairwise = _read_csv(pairwise_path or PAIRWISE_JUDGE_SCORES_PATH)

    merged = _merge_run_tables(deterministic, judge, decision, faithfulness)
    pairs_raw, excluded_pairs = _build_valid_pairs(merged)
    paired = _compute_pair_deltas(pairs_raw)
    paired = _merge_pairwise_judge(paired, pairwise)
    paired = apply_paper_id_filter(paired, use_subset=use_subset)
    paired, overall_decision_deltas, venue_decision_deltas = _append_decision_metric_deltas(paired, merged)

    overall = _summarize_deltas(paired)
    if overall_decision_deltas:
        for key, value in overall_decision_deltas.items():
            metric_name = key.removeprefix('delta_')
            overall = pd.concat(
                [
                    overall,
                    pd.DataFrame(
                        [
                            {
                                'metric': metric_name,
                                'n_pairs': int(paired['paper_id'].nunique()) if not paired.empty else 0,
                                'median_delta': value,
                                'mean_delta': value,
                                'bootstrap_ci_low': None,
                                'bootstrap_ci_high': None,
                                'wilcoxon_pvalue': None,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    venue_summary = _summarize_by_venue(paired, venue_decision_deltas)

    total_candidate_papers = int(merged['paper_id'].nunique()) if not merged.empty else 0
    valid_pairs = int(len(paired))
    completion_rate = (valid_pairs / total_candidate_papers) if total_candidate_papers else 0.0

    metadata = {
        'valid_pairs': valid_pairs,
        'excluded_pairs': excluded_pairs,
        'completion_rate': completion_rate,
        'overall_decision_deltas': overall_decision_deltas,
    }

    _write_dataframe(paired_output_path or PAIRED_COMPARISON_PATH, paired)
    _write_dataframe(overall_output_path or OVERALL_SUMMARY_PATH, overall)
    _write_dataframe(venue_output_path or VENUE_SUMMARY_PATH, venue_summary)

    return metadata
