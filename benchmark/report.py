from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from benchmark.analysis_subset import load_analysis_subset
from benchmark.compare import (
    OVERALL_SUMMARY_PATH,
    PAIRED_COMPARISON_PATH,
    VENUE_SUMMARY_PATH,
    apply_paper_id_filter,
    build_paired_comparison,
)
from benchmark.paths import RESULTS_DIR, THREE_WAY_SUMMARY_PATH, TRAD_PAIRWISE_JUDGE_SCORES_PATH

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


def _analysis_subset_lines() -> list[str]:
    subset = load_analysis_subset()
    included = subset['included_paper_ids']
    excluded = subset['excluded_paper_ids']
    reason = subset['reason']
    lines = [
        '## Analysis subset',
        '',
        f'- Included papers: **{len(included)}**',
        f'- Subset rationale: {reason}',
        '',
        '**Excluded papers:**',
    ]
    for paper_id in excluded:
        lines.append(f'- {paper_id}: {reason}')
    if not excluded:
        lines.append('- n/a')
    return lines


def _three_way_median_lines(three_way: pd.DataFrame) -> list[str]:
    lines = ['## Three-way medians', '']
    if three_way.empty or 'paper_id' not in three_way.columns:
        lines.append('No three-way summary available.')
        return lines

    overall_rows = three_way[three_way['paper_id'].astype(str) == '__overall__']
    if overall_rows.empty:
        lines.append('No overall three-way row available.')
        return lines

    row = overall_rows.iloc[0]
    lines.extend(
        [
            'Median rubric_alignment by condition:',
            f'- TRAD_LLM: **{_fmt(row.get("median_rubric_alignment_TRAD_LLM"))}**',
            f'- KG_OFF: **{_fmt(row.get("median_rubric_alignment_KG_OFF"))}**',
            f'- KG_ON: **{_fmt(row.get("median_rubric_alignment_KG_ON"))}**',
        ]
    )
    return lines


def _trad_pairwise_win_lines(trad_pairwise: pd.DataFrame) -> list[str]:
    lines = ['## Trad pairwise win rates', '']
    if trad_pairwise.empty or 'pair_type' not in trad_pairwise.columns or 'winner' not in trad_pairwise.columns:
        lines.append('No trad pairwise scores available.')
        return lines

    for pair_type, group in trad_pairwise.groupby('pair_type', sort=True):
        winner_counts = group['winner'].fillna('').astype(str).str.strip().value_counts()
        parts = [f'{winner}: **{int(count)}**' for winner, count in winner_counts.items() if winner]
        lines.append(f'- **{pair_type}**: {", ".join(parts) if parts else "n/a"}')
    return lines


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
    three_way_path: Path | None = None,
    trad_pairwise_path: Path | None = None,
    output_path: Path | None = None,
    metadata: dict[str, Any] | None = None,
    use_subset: bool = True,
) -> str:
    paired = _read_csv(paired_path or PAIRED_COMPARISON_PATH)
    if use_subset and not paired.empty and 'paper_id' in paired.columns:
        paired = apply_paper_id_filter(paired, use_subset=True)
    overall = _read_csv(overall_path or OVERALL_SUMMARY_PATH)
    venue = _read_csv(venue_path or VENUE_SUMMARY_PATH)
    three_way = _read_csv(three_way_path or THREE_WAY_SUMMARY_PATH)
    trad_pairwise = _read_csv(trad_pairwise_path or TRAD_PAIRWISE_JUDGE_SCORES_PATH)

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
        *_analysis_subset_lines(),
        '',
        '## Pairwise judge (KG_ON vs KG_OFF)',
        '',
        f'- KG_ON wins: **{kg_on_wins}** / {pairwise_n} ({_fmt(kg_on_pct, digits=1)}%)',
        f'- KG_OFF wins: **{kg_off_wins}** / {pairwise_n} ({_fmt(kg_off_pct, digits=1)}%)',
        f'- Ties: **{ties}** / {pairwise_n} ({_fmt(ties_pct, digits=1)}%)',
        '',
        *_three_way_median_lines(three_way),
        '',
        *_trad_pairwise_win_lines(trad_pairwise),
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


def generate_report(*, rebuild_comparison: bool = True, use_subset: bool = True) -> Path:
    metadata: dict[str, Any] | None = None
    if rebuild_comparison:
        metadata = build_paired_comparison(use_subset=use_subset)
    build_benchmark_summary(metadata=metadata, use_subset=use_subset)
    return BENCHMARK_SUMMARY_PATH
