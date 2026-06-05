from __future__ import annotations

BENCHMARK_ENV_DEFAULTS = {
    'REVIEW_FAST_MODE': 'true',
    'PAPER_SEARCH_ENABLED': 'false',
    'REVIEW_INFER_VENUE_FROM_PAPER': 'false',
    'ENABLE_FINAL_GATES': 'false',
}


def build_benchmark_env(
    condition: str,
    *,
    venue: str,
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(base or {})
    env.update(BENCHMARK_ENV_DEFAULTS)
    env['REVIEW_VENUE'] = venue
    env['REVIEW_CRITERIA_ENABLED'] = 'true' if condition == 'KG_ON' else 'false'
    return env
