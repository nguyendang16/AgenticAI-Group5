from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from benchmark.compare import (
    OVERALL_SUMMARY_PATH,
    PAIRED_COMPARISON_PATH,
    VENUE_SUMMARY_PATH,
    build_paired_comparison,
)
from benchmark.paths import RESULTS_DIR

BENCHMARK_SUMMARY_PATH = RESULTS_DIR / 'benchmark_summary.md'


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _fmt(value: Any, *, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 'n/a'
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return str(value)


def _metric_row(overall: pd.DataFrame, metric: str) -> pd.Series | None:
    if overall.empty or 'metric' not in overall.columns:
        return None
    matches = overall[overall['metric'] == metric]
    if matches.empty:
        return None
    return matches.iloc[0]


def _top_venues_by_metric(venue: pd.DataFrame, metric: str, *, ascending: bool, limit: int = 3) -> list[tuple[str, float]]:
    col = f'median_delta_{metric}'
    if venue.empty or col not in venue.columns:
        return []

    ranked = venue.dropna(subset=[col]).sort_values(col, ascending=ascending)
    output: list[tuple[str, float]] = []
    for _, row in ranked.head(limit).iterrows():
        output.append((str(row.get('venue', '')), float(row[col])))
    return output


def build_benchmark_summary(
    *,
    paired_path: Path | None = None,
    overall_path: Path | None = None,
    venue_path: Path | None = None,
    output_path: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    paired = _read_csv(paired_path or PAIRED_COMPARISON_PATH)
    overall = _read_csv(overall_path or OVERALL_SUMMARY_PATH)
    venue = _read_csv(venue_path or VENUE_SUMMARY_PATH)

    meta = metadata or {}
    valid_pairs = int(meta.get('valid_pairs', len(paired)))
    excluded_pairs = int(meta.get('excluded_pairs', 0))
    completion_rate = float(meta.get('completion_rate', 0.0))

    rubric = _metric_row(overall, 'rubric_alignment')
    factual = _metric_row(overall, 'factual_correctness')
    faithfulness = _metric_row(overall, 'faithfulness_mean')
    total_tokens = _metric_row(overall, 'total_tokens')
    runtime = _metric_row(overall, 'runtime_seconds')
    balanced_acc = _metric_row(overall, 'balanced_accuracy')
    macro_f1 = _metric_row(overall, 'macro_f1')

    helps = _top_venues_by_metric(venue, 'rubric_alignment', ascending=False)
    hurts = _top_venues_by_metric(venue, 'rubric_alignment', ascending=True)

    pairwise_n = len(paired)
    if pairwise_n and 'pairwise_winner' in paired.columns:
        winner_counts = paired['pairwise_winner'].fillna('').astype(str).str.strip().value_counts()
        kg_on_wins = int(winner_counts.get('KG_ON', 0))
        kg_off_wins = int(winner_counts.get('KG_OFF', 0))
        ties = int(winner_counts.get('tie', 0))
        kg_on_pct = (kg_on_wins / pairwise_n) * 100
        kg_off_pct = (kg_off_wins / pairwise_n) * 100
        ties_pct = (ties / pairwise_n) * 100
    else:
        kg_on_wins = kg_off_wins = ties = 0
        kg_on_pct = kg_off_pct = ties_pct = 0.0

    lines = [
        '# KG Benchmark Summary',
        '',
        '## Executive summary',
        '',
        f'- Valid paired papers: **{valid_pairs}**',
        f'- Excluded / invalid pairs: **{excluded_pairs}**',
        f'- Pair completion rate: **{_fmt(completion_rate * 100, digits=1)}%**',
        f'- Median Δ rubric_alignment (KG_ON − KG_OFF): **{_fmt(rubric.get("median_delta") if rubric is not None else None)}**',
        f'- Median Δ factual_correctness: **{_fmt(factual.get("median_delta") if factual is not None else None)}**',
        f'- Median Δ faithfulness_mean: **{_fmt(faithfulness.get("median_delta") if faithfulness is not None else None)}**',
        '',
        '## Pairwise judge (KG_ON vs KG_OFF)',
        '',
        f'- KG_ON wins: **{kg_on_wins}** / {pairwise_n} ({_fmt(kg_on_pct, digits=1)}%)',
        f'- KG_OFF wins: **{kg_off_wins}** / {pairwise_n} ({_fmt(kg_off_pct, digits=1)}%)',
        f'- Ties: **{ties}** / {pairwise_n} ({_fmt(ties_pct, digits=1)}%)',
        '',
        '## Per-venue rubric alignment',
        '',
        '| Venue | n_pairs | median Δ rubric_alignment | 95% CI | Wilcoxon p |',
        '| --- | ---: | ---: | --- | ---: |',
    ]

    rubric_col = 'median_delta_rubric_alignment'
    ci_low_col = 'bootstrap_ci_low_rubric_alignment'
    ci_high_col = 'bootstrap_ci_high_rubric_alignment'
    p_col = 'wilcoxon_pvalue_rubric_alignment'

    if venue.empty or rubric_col not in venue.columns:
        lines.append('| n/a | 0 | n/a | n/a | n/a |')
    else:
        for _, row in venue.sort_values('venue').iterrows():
            ci = f'[{_fmt(row.get(ci_low_col))}, {_fmt(row.get(ci_high_col))}]'
            lines.append(
                '| {venue} | {n_pairs} | {median} | {ci} | {p} |'.format(
                    venue=row.get('venue', ''),
                    n_pairs=int(row.get('n_pairs', 0)),
                    median=_fmt(row.get(rubric_col)),
                    ci=ci,
                    p=_fmt(row.get(p_col)),
                )
            )

    lines.extend(
        [
            '',
            '## Decision metrics (sklearn, when labels exist)',
            '',
        ]
    )
    if balanced_acc is None and macro_f1 is None:
        lines.append('No labeled decision metrics available.')
    else:
        lines.append(
            f'- Δ balanced_accuracy: **{_fmt(balanced_acc.get("median_delta") if balanced_acc is not None else None)}**'
        )
        lines.append(
            f'- Δ macro_f1: **{_fmt(macro_f1.get("median_delta") if macro_f1 is not None else None)}**'
        )

    lines.extend(['', '## Top venues where KG helps / hurts (rubric_alignment)', ''])
    if helps:
        lines.append('**KG helps most:**')
        for venue_name, delta in helps:
            lines.append(f'- {venue_name}: median Δ = {_fmt(delta)}')
    else:
        lines.append('- KG helps most: n/a')

    lines.append('')
    if hurts:
        lines.append('**KG hurts most:**')
        for venue_name, delta in hurts:
            lines.append(f'- {venue_name}: median Δ = {_fmt(delta)}')
    else:
        lines.append('- KG hurts most: n/a')

    lines.extend(
        [
            '',
            '## Statistical test',
            '',
        ]
    )

    if rubric is not None and rubric.get('wilcoxon_pvalue') is not None:
        lines.append(
            f'Wilcoxon signed-rank p-value on rubric_alignment deltas (n ≥ 8): **{_fmt(rubric.get("wilcoxon_pvalue"))}**'
        )
    else:
        lines.append('Wilcoxon signed-rank test not run (requires n ≥ 8 valid pairs).')

    lines.extend(
        [
            '',
            '## Data quality',
            '',
            f'- Invalid or incomplete pairs excluded from comparison: **{excluded_pairs}**',
            '',
            '## Appendix: Engineering metrics',
            '',
            f'- Median Δ total_tokens: **{_fmt(total_tokens.get("median_delta") if total_tokens is not None else None)}**',
            f'- Median Δ runtime_seconds: **{_fmt(runtime.get("median_delta") if runtime is not None else None)}**',
        ]
    )

    markdown = '\n'.join(lines) + '\n'
    destination = output_path or BENCHMARK_SUMMARY_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(markdown, encoding='utf-8')
    return markdown


def generate_report(*, rebuild_comparison: bool = True) -> Path:
    metadata: dict[str, Any] | None = None
    if rebuild_comparison:
        metadata = build_paired_comparison()
    build_benchmark_summary(metadata=metadata)
    return BENCHMARK_SUMMARY_PATH
