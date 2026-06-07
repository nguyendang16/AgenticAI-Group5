from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.retriever import retrieve_criteria_bundle


def _flatten_criteria(bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    groups = bundle.get('criteria_by_group') if isinstance(bundle.get('criteria_by_group'), dict) else {}
    criteria: dict[str, dict[str, Any]] = {}
    for items in groups.values():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            cid = str(item.get('criterion_id') or '').strip()
            if cid:
                criteria[cid] = item
    return criteria


def _evidence_ids(item: dict[str, Any]) -> set[str]:
    evidence = item.get('evidence_required') if isinstance(item.get('evidence_required'), list) else []
    return {
        str(ev.get('evidence_id') or '').strip()
        for ev in evidence
        if isinstance(ev, dict) and str(ev.get('evidence_id') or '').strip()
    }


def _source_ids(item: dict[str, Any]) -> set[str]:
    sources: set[str] = set()
    provenance = item.get('provenance') if isinstance(item.get('provenance'), dict) else {}
    if provenance.get('source_id'):
        sources.add(str(provenance['source_id']).strip())
    provenance_sources = (
        item.get('provenance_sources') if isinstance(item.get('provenance_sources'), list) else []
    )
    for source in provenance_sources:
        if isinstance(source, dict) and source.get('source_id'):
            sources.add(str(source['source_id']).strip())
    return sources


def _evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    query = case.get('query') if isinstance(case.get('query'), dict) else {}
    bundle = retrieve_criteria_bundle(
        venue=query.get('venue'),
        journal=query.get('journal'),
        domain=query.get('domain'),
        article_type=query.get('article_type'),
    )
    retrieved = _flatten_criteria(bundle)
    failures: list[str] = []
    expected_criteria = (
        case.get('expected_criteria') if isinstance(case.get('expected_criteria'), list) else []
    )
    expected_source_ids = set(case.get('expected_source_ids') or [])
    retrieved_source_ids = {
        source_id
        for item in retrieved.values()
        for source_id in _source_ids(item)
    }
    if expected_source_ids and not expected_source_ids.issubset(retrieved_source_ids):
        failures.append(
            f'missing expected source ids: {sorted(expected_source_ids - retrieved_source_ids)}'
        )
    forbidden_source_ids = set(case.get('forbidden_source_ids') or [])
    forbidden_source_hits = forbidden_source_ids & retrieved_source_ids
    if forbidden_source_hits:
        failures.append(f'forbidden source ids returned: {sorted(forbidden_source_hits)}')

    expected_empty = bool(case.get('expected_empty'))
    if expected_empty and retrieved:
        failures.append(f'expected no criteria, retrieved {len(retrieved)}')

    min_criteria_count = case.get('min_criteria_count')
    if isinstance(min_criteria_count, int) and len(retrieved) < min_criteria_count:
        failures.append(f'expected at least {min_criteria_count} criteria, retrieved {len(retrieved)}')

    max_criteria_count = case.get('max_criteria_count')
    if isinstance(max_criteria_count, int) and len(retrieved) > max_criteria_count:
        failures.append(f'expected at most {max_criteria_count} criteria, retrieved {len(retrieved)}')

    expected_criterion_ids: list[str] = []
    expected_evidence_ids: list[str] = []
    matched_criterion_ids: list[str] = []
    matched_evidence_ids: list[str] = []
    for expected in expected_criteria:
        if not isinstance(expected, dict):
            continue
        cid = str(expected.get('criterion_id') or '').strip()
        if not cid:
            continue
        expected_criterion_ids.append(cid)
        if cid not in retrieved:
            failures.append(f'missing criterion: {cid}')
            continue
        matched_criterion_ids.append(cid)
        evidence_expected = {
            str(eid).strip()
            for eid in expected.get('evidence_ids', [])
            if str(eid).strip()
        }
        expected_evidence_ids.extend(sorted(evidence_expected))
        evidence_actual = _evidence_ids(retrieved[cid])
        missing_evidence = evidence_expected - evidence_actual
        if missing_evidence:
            failures.append(f'{cid} missing evidence ids: {sorted(missing_evidence)}')
        else:
            matched_evidence_ids.extend(sorted(evidence_expected))

    forbidden_criterion_ids = {
        str(cid).strip()
        for cid in case.get('forbidden_criterion_ids', [])
        if str(cid).strip()
    }
    forbidden_criterion_hits = forbidden_criterion_ids & set(retrieved)
    if forbidden_criterion_hits:
        failures.append(f'forbidden criteria returned: {sorted(forbidden_criterion_hits)}')

    return {
        'id': case.get('id'),
        'category': case.get('category', 'positive_retrieval'),
        'scenario': case.get('scenario'),
        'question': case.get('question'),
        'query': query,
        'passed': not failures,
        'failures': failures,
        'retrieved_criteria_count': len(retrieved),
        'retrieved_source_ids': sorted(retrieved_source_ids),
        'expected_source_ids': sorted(expected_source_ids),
        'forbidden_source_ids': sorted(forbidden_source_ids),
        'expected_empty': expected_empty,
        'min_criteria_count': min_criteria_count,
        'max_criteria_count': max_criteria_count,
        'expected_criterion_ids': expected_criterion_ids,
        'matched_criterion_ids': matched_criterion_ids,
        'expected_evidence_ids': expected_evidence_ids,
        'matched_evidence_ids': matched_evidence_ids,
        'forbidden_criterion_ids': sorted(forbidden_criterion_ids),
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    category_rows = [
        '| {category} | {cases} | {passed} | {failed} |'.format(
            category=str(category).replace('|', '\\|'),
            cases=summary['case_count'],
            passed=summary['passed'],
            failed=summary['failed'],
        )
        for category, summary in sorted(payload.get('by_category', {}).items())
    ]
    rows = []
    for case in payload['cases']:
        status = 'PASS' if case['passed'] else 'FAIL'
        rows.append(
            '| {id} | {category} | {status} | {retrieved} | {criteria} | {evidence} | {scenario} |'.format(
                id=case['id'],
                category=str(case.get('category', '')).replace('|', '\\|'),
                status=status,
                retrieved=case['retrieved_criteria_count'],
                criteria=f"{len(case['matched_criterion_ids'])}/{len(case['expected_criterion_ids'])}",
                evidence=f"{len(case['matched_evidence_ids'])}/{len(case['expected_evidence_ids'])}",
                scenario=str(case['scenario']).replace('|', '\\|'),
            )
        )
    return f"""# KG Evidence Retrieval Test Results

Generated: {payload['generated_at']}

| Metric | Value |
|---|---:|
| Test cases | {payload['summary']['case_count']} |
| Passed | {payload['summary']['passed']} |
| Failed | {payload['summary']['failed']} |
| Pass rate | {payload['summary']['pass_rate']} |

## Category Summary

| Category | Cases | Passed | Failed |
|---|---:|---:|---:|
{chr(10).join(category_rows)}

## Case Results

| Case | Category | Status | Retrieved Criteria | Criteria Matched | Evidence Matched | Scenario |
|---|---|---|---:|---:|---:|---|
{chr(10).join(rows)}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description='Run hard KG evidence retrieval test cases')
    parser.add_argument(
        '--cases',
        type=Path,
        default=Path('data/kg_evidence_retrieval_test_cases.json'),
    )
    parser.add_argument(
        '--output-json',
        type=Path,
        default=Path('outputs/kg_evidence_retrieval_test_results.json'),
    )
    parser.add_argument(
        '--output-md',
        type=Path,
        default=Path('docs/kg_evidence_retrieval_test_results.md'),
    )
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding='utf-8'))
    if not isinstance(cases, list):
        raise ValueError('test cases file must contain a JSON list')
    results = [_evaluate_case(case) for case in cases]
    passed = sum(1 for case in results if case['passed'])
    by_category: dict[str, dict[str, int]] = {}
    for result in results:
        category = str(result.get('category') or 'positive_retrieval')
        row = by_category.setdefault(category, {'case_count': 0, 'passed': 0, 'failed': 0})
        row['case_count'] += 1
        if result['passed']:
            row['passed'] += 1
        else:
            row['failed'] += 1
    payload = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'cases_file': str(args.cases),
        'summary': {
            'case_count': len(results),
            'passed': passed,
            'failed': len(results) - passed,
            'pass_rate': round(passed / len(results), 4) if results else 0.0,
        },
        'by_category': by_category,
        'cases': results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    args.output_md.write_text(_markdown_report(payload), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload['summary']['failed'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
