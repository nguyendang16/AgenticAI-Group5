from __future__ import annotations

import csv
import json
import random
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from deepreview.adapters.markdown_parser import build_page_index
from deepreview.state import ensure_artifact_paths
from deepreview.storage import read_json
from deepreview.types import JobState, JobStatus

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_REQUIRED_SECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ('summary', re.compile(r'(?im)^\s*#{1,3}\s*summary\b')),
    ('strengths', re.compile(r'(?im)^\s*#{1,3}\s*strengths\b')),
    ('weaknesses', re.compile(r'(?im)^\s*#{1,3}\s*weaknesses\b')),
    ('suggestions', re.compile(r'(?im)^\s*#{1,3}\s*(actionable\s+suggestions|suggestions)\b')),
)

_COVERAGE_KEYWORDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ('method', re.compile(r'\b(method|methodology|approach|design)\b', re.I)),
    ('experiments', re.compile(r'\b(experiment|evaluation|result|empirical|ablation)\b', re.I)),
    ('limitations', re.compile(r'\b(limitation|threat|validity|generaliz)\b', re.I)),
)

_PLACEHOLDER_RE = re.compile(r'\b(TBD|N/A|TODO|placeholder|to be determined)\b', re.I)

_WHITESPACE_RE = re.compile(r'\s+')


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    token = str(value).strip()
    if token.endswith('Z'):
        token = f'{token[:-1]}+00:00'
    try:
        return datetime.fromisoformat(token)
    except ValueError:
        return None


def _normalize_text(value: str) -> str:
    return _WHITESPACE_RE.sub(' ', (value or '').strip().lower())


def _token_overlap(left: str, right: str) -> float:
    left_norm = _normalize_text(left)
    right_norm = _normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm in right_norm or right_norm in left_norm:
        return 1.0
    left_tokens = set(left_norm.split())
    right_tokens = set(right_norm.split())
    if not left_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def _load_events(job_dir: Path) -> list[dict[str, Any]]:
    path = job_dir / 'events.jsonl'
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _load_annotations(job_dir: Path) -> list[dict[str, Any]]:
    path = job_dir / 'annotations.json'
    if not path.exists():
        return []
    payload = read_json(path)
    rows = payload.get('annotations') if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _load_page_index(job_dir: Path) -> dict[int, list[str]]:
    artifacts = ensure_artifact_paths(job_dir.name)
    markdown_path = job_dir / 'mineru_full.md'
    content_list_path = job_dir / 'mineru_content_list.json'
    markdown = markdown_path.read_text(encoding='utf-8') if markdown_path.exists() else ''
    content_list: list[dict[str, Any]] | None = None
    if content_list_path.exists():
        raw = read_json(content_list_path)
        if isinstance(raw, dict):
            rows = raw.get('content_list')
            content_list = rows if isinstance(rows, list) else None
        elif isinstance(raw, list):
            content_list = raw
    return build_page_index(markdown, content_list)


def _span_from_page_index(
    page_index: dict[int, list[str]],
    *,
    page: int,
    start_line: int,
    end_line: int,
) -> str:
    lines = page_index.get(int(page)) or []
    if not lines:
        return ''
    start = max(1, int(start_line))
    end = max(start, int(end_line))
    start_idx = start - 1
    end_idx = min(len(lines), end)
    return '\n'.join(lines[start_idx:end_idx]).strip()


def check_annotation_grounding(
    annotation: dict[str, Any],
    page_index: dict[int, list[str]],
    *,
    overlap_threshold: float = 0.8,
) -> dict[str, Any]:
    page = int(annotation.get('page') or 0)
    start_line = int(annotation.get('start_line') or 0)
    end_line = int(annotation.get('end_line') or start_line)
    quoted = str(annotation.get('text') or '').strip()
    page_span = _span_from_page_index(
        page_index,
        page=page,
        start_line=start_line,
        end_line=end_line,
    )
    overlap = _token_overlap(quoted, page_span)
    passed = overlap >= overlap_threshold
    return {
        'annotation_id': annotation.get('id'),
        'page': page,
        'overlap': round(overlap, 3),
        'passed': passed,
        'quoted_preview': quoted[:120],
    }


def sample_grounding_checks(
    annotations: list[dict[str, Any]],
    page_index: dict[int, list[str]],
    *,
    sample_size: int = 3,
    seed: str = '',
    overlap_threshold: float = 0.8,
) -> dict[str, Any]:
    if not annotations:
        return {
            'sample_size': 0,
            'passed': 0,
            'pass_rate': None,
            'checks': [],
            'score': 0,
        }

    pool = list(annotations)
    rng = random.Random(seed or 'grounding')
    if len(pool) > sample_size:
        pool = rng.sample(pool, sample_size)
    checks = [
        check_annotation_grounding(row, page_index, overlap_threshold=overlap_threshold)
        for row in pool
    ]
    passed = sum(1 for row in checks if row.get('passed'))
    sample_n = len(checks)
    pass_rate = passed / sample_n if sample_n else None
    score = 2 if sample_n and passed == sample_n else (1 if passed > 0 else 0)
    return {
        'sample_size': sample_n,
        'passed': passed,
        'pass_rate': round(pass_rate, 3) if pass_rate is not None else None,
        'checks': checks,
        'score': score,
    }


def _score_q1_structure(report_text: str) -> tuple[int, dict[str, Any]]:
    found = [name for name, pattern in _REQUIRED_SECTION_PATTERNS if pattern.search(report_text)]
    word_count = len(report_text.split())
    section_ok = len(found) >= 4
    length_ok = word_count >= 800
    if section_ok and length_ok:
        score = 2
    elif section_ok or length_ok:
        score = 1
    else:
        score = 0
    return score, {
        'sections_found': found,
        'word_count': word_count,
        'section_ok': section_ok,
        'length_ok': length_ok,
    }


def _score_q2_coverage(report_text: str) -> tuple[int, dict[str, Any]]:
    hits = {
        name: bool(pattern.search(report_text))
        for name, pattern in _COVERAGE_KEYWORDS
    }
    hit_count = sum(1 for ok in hits.values() if ok)
    if hit_count >= 3:
        score = 2
    elif hit_count >= 2:
        score = 1
    else:
        score = 0
    return score, {'keyword_hits': hits, 'hit_count': hit_count}


def _score_q3_specificity(annotations: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    count = len(annotations)
    pages = {int(row.get('page') or 0) for row in annotations if int(row.get('page') or 0) > 0}
    comments = [str(row.get('comment') or '').strip() for row in annotations if str(row.get('comment') or '').strip()]
    avg_comment_len = statistics.mean([len(c) for c in comments]) if comments else 0.0
    count_ok = count >= 8
    pages_ok = len(pages) >= 3
    comment_ok = avg_comment_len > 40
    passed = sum((count_ok, pages_ok, comment_ok))
    if passed == 3:
        score = 2
    elif passed >= 2:
        score = 1
    else:
        score = 0
    return score, {
        'annotation_count': count,
        'distinct_pages': len(pages),
        'avg_comment_length': round(avg_comment_len, 1),
        'count_ok': count_ok,
        'pages_ok': pages_ok,
        'comment_ok': comment_ok,
    }


def _score_q5_actionability(report_text: str) -> tuple[int, dict[str, Any]]:
    suggestions_match = re.search(
        r'(?is)(?:^|\n)\s*#{1,3}\s*(?:actionable\s+suggestions|suggestions)\s*\n(.*?)(?:\n\s*#{1,3}\s|\Z)',
        report_text,
    )
    section = suggestions_match.group(1) if suggestions_match else report_text
    bullets = [
        line.strip()
        for line in section.splitlines()
        if line.strip().startswith(('-', '*', '•')) or re.match(r'^\d+[\).\]]\s', line.strip())
    ]
    lines = [line.strip() for line in section.splitlines() if line.strip()]
    placeholder_lines = sum(1 for line in lines if _PLACEHOLDER_RE.search(line))
    placeholder_ratio = (placeholder_lines / len(lines)) if lines else 0.0
    bullets_ok = len(bullets) >= 3
    placeholder_ok = placeholder_ratio < 0.30
    if bullets_ok and placeholder_ok:
        score = 2
    elif bullets_ok or placeholder_ok:
        score = 1
    else:
        score = 0
    return score, {
        'suggestion_bullets': len(bullets),
        'placeholder_ratio': round(placeholder_ratio, 3),
        'bullets_ok': bullets_ok,
        'placeholder_ok': placeholder_ok,
    }


def _wall_clock_minutes(events: list[dict[str, Any]], job: JobState) -> float | None:
    created = _parse_ts(job.created_at.isoformat() if job.created_at else None)
    terminal_ts: datetime | None = None
    for row in reversed(events):
        event = str(row.get('event') or '')
        if event in {'completed', 'failed', 'completed_recovered', 'pipeline_exception'}:
            terminal_ts = _parse_ts(str(row.get('ts') or ''))
            break
    if terminal_ts is None and job.updated_at:
        terminal_ts = job.updated_at
    if created is None or terminal_ts is None:
        return None
    return round((terminal_ts - created).total_seconds() / 60.0, 2)


def _event_count(events: list[dict[str, Any]], event_name: str) -> int:
    return sum(1 for row in events if str(row.get('event') or '') == event_name)


def _classify_root_cause_bucket(
    *,
    job: JobState,
    events: list[dict[str, Any]],
    reliability: dict[str, Any],
    oqi: dict[str, Any] | None,
) -> str | None:
    if job.status == JobStatus.completed and reliability.get('completed'):
        if oqi and int(oqi.get('total') or 0) <= 5:
            return 'R5'
        grounding = (oqi or {}).get('q4_grounding') or {}
        if grounding.get('sample_size') and int(grounding.get('passed') or 0) <= 1:
            return 'R5'
        if any(str(row.get('event') or '') == 'completed_recovered' for row in events):
            return 'R6'
        return None

    metadata = job.metadata if isinstance(job.metadata, dict) else {}
    last_status = ''
    for row in reversed(events):
        if str(row.get('event') or '') == 'status':
            last_status = str(row.get('status') or '')
            break

    if last_status == 'pdf_parsing' or metadata.get('parse_warning'):
        return 'R1'
    if _event_count(events, 'agent_run_incomplete') >= 1:
        return 'R2'
    if job.annotation_count > 0 and not job.final_report_ready:
        return 'R3'
    paper_state = metadata.get('paper_search_runtime_state')
    if isinstance(paper_state, dict) and paper_state.get('error'):
        return 'R4'
    criteria = metadata.get('review_criteria_resolution')
    if isinstance(criteria, dict) and criteria.get('error'):
        return 'R7'
    usage = job.usage
    if int(usage.token.total_tokens or 0) > 0 and int(job.annotation_count or 0) < 3:
        return 'R8'
    if job.error and 'APIConnectionError' in str(job.error):
        return 'R2'
    if job.status == JobStatus.failed:
        return 'R2'
    return None


def evaluate_job_dir(job_dir: Path, *, repo_root: Path | None = None) -> dict[str, Any]:
    job_dir = job_dir.resolve()
    job_path = job_dir / 'job.json'
    if not job_path.exists():
        raise FileNotFoundError(f'job.json not found in {job_dir}')

    job = JobState.model_validate(read_json(job_path))
    events = _load_events(job_dir)
    artifacts = ensure_artifact_paths(job.id)

    final_md_path = job_dir / 'final_report.md'
    final_pdf_path = job_dir / 'final_report.pdf'
    report_text = final_md_path.read_text(encoding='utf-8') if final_md_path.exists() else ''
    annotations = _load_annotations(job_dir)

    has_md = final_md_path.exists() and final_md_path.stat().st_size > 2048
    has_pdf = final_pdf_path.exists()
    final_write_event = any(
        str(row.get('event') or '') in {'final_report_persisted', 'review_final_markdown_write'}
        for row in events
    )
    completed = job.status == JobStatus.completed

    reliability = {
        'completed': completed,
        'has_final_markdown': has_md,
        'has_final_pdf': has_pdf,
        'no_fatal_error': job.error is None or completed,
        'final_write_gate': bool(job.final_report_ready or final_write_event or has_md),
        'tier1_pass': completed and has_md and (job.error is None or has_pdf),
    }

    usage = job.usage
    wall_min = _wall_clock_minutes(events, job)
    efficiency = {
        'wall_clock_minutes': wall_min,
        'tokens_total': int(usage.token.total_tokens or 0),
        'tokens_input': int(usage.token.input_tokens or 0),
        'tokens_output': int(usage.token.output_tokens or 0),
        'tool_calls_total': int(usage.tool.total_calls or 0),
        'paper_search_calls': int(usage.paper_search.total_calls or 0),
        'agent_run_incomplete_count': _event_count(events, 'agent_run_incomplete'),
        'flags': [],
    }
    if wall_min is not None and wall_min > 45:
        efficiency['flags'].append('slow_fast_mode')
    if int(usage.tool.total_calls or 0) > 150:
        efficiency['flags'].append('high_tool_calls')
    if int(usage.tool.total_calls or 0) < 5 and completed:
        efficiency['flags'].append('low_tool_calls')

    oqi: dict[str, Any] | None = None
    if completed and report_text.strip():
        q1_score, q1_detail = _score_q1_structure(report_text)
        q2_score, q2_detail = _score_q2_coverage(report_text)
        q3_score, q3_detail = _score_q3_specificity(annotations)
        page_index = _load_page_index(job_dir) if annotations else {}
        q4 = sample_grounding_checks(
            annotations,
            page_index,
            sample_size=3,
            seed=str(job.id),
        )
        q5_score, q5_detail = _score_q5_actionability(report_text)
        total = q1_score + q2_score + q3_score + int(q4['score']) + q5_score
        oqi = {
            'q1_structure': {'score': q1_score, **q1_detail},
            'q2_coverage': {'score': q2_score, **q2_detail},
            'q3_specificity': {'score': q3_score, **q3_detail},
            'q4_grounding': q4,
            'q5_actionability': {'score': q5_score, **q5_detail},
            'total': total,
            'max': 10,
            'tier1_quality_pass': total >= 7,
        }

    tier2_needed = (
        not reliability['tier1_pass']
        or (oqi is not None and int(oqi['total']) <= 5)
        or (
            oqi is not None
            and int((oqi.get('q4_grounding') or {}).get('passed') or 0) <= 1
            and int((oqi.get('q4_grounding') or {}).get('sample_size') or 0) >= 2
        )
    )
    root_cause = _classify_root_cause_bucket(
        job=job,
        events=events,
        reliability=reliability,
        oqi=oqi,
    )

    return {
        'job_id': str(job.id),
        'title': job.title,
        'source_pdf_name': job.source_pdf_name,
        'status': job.status.value,
        'error': job.error,
        'annotation_count': int(job.annotation_count or len(annotations)),
        'reliability': reliability,
        'efficiency': efficiency,
        'oqi': oqi,
        'tier2_needed': tier2_needed,
        'root_cause_bucket': root_cause,
        'evaluated_at': datetime.now().astimezone().isoformat(),
        'spec': 'docs/2026-06-01-tiered-system-evaluation-design.md Tier 1',
        'artifacts': {
            'job_json': str(job_path),
            'final_report_md': str(final_md_path) if final_md_path.exists() else None,
            'annotations_json': str(job_dir / 'annotations.json')
            if (job_dir / 'annotations.json').exists()
            else None,
            'mineru_markdown': str(job_dir / 'mineru_full.md')
            if (job_dir / 'mineru_full.md').exists()
            else None,
        },
    }


def evaluate_jobs_root(jobs_root: Path) -> list[dict[str, Any]]:
    jobs_root = jobs_root.resolve()
    if not jobs_root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for child in sorted(jobs_root.iterdir()):
        if not child.is_dir() or not (child / 'job.json').exists():
            continue
        try:
            rows.append(evaluate_job_dir(child))
        except Exception as exc:
            rows.append(
                {
                    'job_id': child.name,
                    'status': 'error',
                    'error': f'{type(exc).__name__}: {exc}',
                    'tier2_needed': True,
                    'root_cause_bucket': None,
                }
            )
    return rows


def aggregate_verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    completed_rows = [row for row in rows if (row.get('reliability') or {}).get('completed')]
    completed = len(completed_rows)
    completion_rate = (completed / total) if total else 0.0

    oqi_values = [
        int((row.get('oqi') or {}).get('total'))
        for row in completed_rows
        if isinstance(row.get('oqi'), dict) and (row.get('oqi') or {}).get('total') is not None
    ]
    median_oqi = statistics.median(oqi_values) if oqi_values else None

    grounding_rates: list[float] = []
    for row in completed_rows:
        q4 = (row.get('oqi') or {}).get('q4_grounding') or {}
        rate = q4.get('pass_rate')
        if isinstance(rate, (int, float)):
            grounding_rates.append(float(rate))
    grounding_pass_rate = statistics.mean(grounding_rates) if grounding_rates else None

    bucket_counts: dict[str, int] = {}
    for row in rows:
        bucket = row.get('root_cause_bucket')
        if bucket:
            bucket_counts[str(bucket)] = bucket_counts.get(str(bucket), 0) + 1

    verdict = 'no-go'
    if (
        completion_rate >= 0.85
        and median_oqi is not None
        and median_oqi >= 7
        and grounding_pass_rate is not None
        and grounding_pass_rate >= 0.80
    ):
        verdict = 'go'
    elif completion_rate >= 0.70 and (median_oqi is None or median_oqi >= 6):
        verdict = 'go-with-fixes'

    top_buckets = sorted(bucket_counts.items(), key=lambda item: (-item[1], item[0]))[:3]
    return {
        'job_count': total,
        'completed_count': completed,
        'completion_rate': round(completion_rate, 3),
        'median_oqi': median_oqi,
        'grounding_pass_rate_mean': round(grounding_pass_rate, 3) if grounding_pass_rate is not None else None,
        'root_cause_counts': bucket_counts,
        'top_root_causes': [{'bucket': b, 'count': c} for b, c in top_buckets],
        'verdict': verdict,
    }


def write_evaluation_report(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate_verdict(rows)

    json_path = output_dir / 'eval_results.json'
    json_path.write_text(
        json.dumps({'jobs': rows, 'aggregate': summary}, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )

    csv_path = output_dir / 'eval_run_sheet.csv'
    fieldnames = [
        'job_id',
        'title',
        'status',
        'completed',
        'wall_clock_min',
        'tokens_total',
        'tool_calls_total',
        'annotation_count',
        'oqi',
        'q4_passed',
        'q4_sample_size',
        'tier2_needed',
        'root_cause_bucket',
        'error',
    ]
    with csv_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            rel = row.get('reliability') or {}
            eff = row.get('efficiency') or {}
            oqi = row.get('oqi') or {}
            q4 = oqi.get('q4_grounding') or {}
            writer.writerow(
                {
                    'job_id': row.get('job_id'),
                    'title': row.get('title'),
                    'status': row.get('status'),
                    'completed': rel.get('completed'),
                    'wall_clock_min': eff.get('wall_clock_minutes'),
                    'tokens_total': eff.get('tokens_total'),
                    'tool_calls_total': eff.get('tool_calls_total'),
                    'annotation_count': row.get('annotation_count'),
                    'oqi': oqi.get('total'),
                    'q4_passed': q4.get('passed'),
                    'q4_sample_size': q4.get('sample_size'),
                    'tier2_needed': row.get('tier2_needed'),
                    'root_cause_bucket': row.get('root_cause_bucket'),
                    'error': row.get('error'),
                }
            )

    md_lines = [
        '# Tier 1 Evaluation Summary',
        '',
        f'- Jobs evaluated: **{summary["job_count"]}**',
        f'- Completion rate: **{summary["completed_count"]}/{summary["job_count"]}** ({summary["completion_rate"]:.0%})',
        f'- Median OQI: **{summary["median_oqi"]}**/10' if summary['median_oqi'] is not None else '- Median OQI: **n/a**',
    ]
    if summary['grounding_pass_rate_mean'] is not None:
        md_lines.append(
            f'- Mean grounding pass rate (Q4 sample): **{summary["grounding_pass_rate_mean"]:.0%}**'
        )
    md_lines.extend(
        [
            f'- Verdict: **{summary["verdict"].upper().replace("-", " ")}**',
            '',
            '## Top root-cause buckets',
            '',
        ]
    )
    if summary['top_root_causes']:
        for item in summary['top_root_causes']:
            md_lines.append(f'- {item["bucket"]}: {item["count"]} job(s)')
    else:
        md_lines.append('- None flagged')

    md_lines.extend(['', '## Per-job results', ''])
    for row in rows:
        oqi_total = (row.get('oqi') or {}).get('total')
        md_lines.append(
            f'- `{row.get("job_id")}` · {row.get("status")} · OQI={oqi_total if oqi_total is not None else "n/a"} · tier2={row.get("tier2_needed")} · bucket={row.get("root_cause_bucket") or "-"}'
        )

    md_path = output_dir / 'eval_summary.md'
    md_path.write_text('\n'.join(md_lines) + '\n', encoding='utf-8')

    return {'json': json_path, 'csv': csv_path, 'markdown': md_path}


def save_job_evaluation(job_dir: Path, payload: dict[str, Any]) -> Path:
    path = job_dir / 'evaluation.json'
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return path
