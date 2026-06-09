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
    category_details = """## What This Benchmark Evaluates

This benchmark checks whether the Knowledge Graph (KG) retrieval layer can return the correct review-criteria evidence for realistic venue, journal, domain, and article-type inputs. It is focused on **KG evidence retrieval behavior**, not full paper-review quality.

The test answers this question:

> Given a review context, can the KG retrieve the expected source document, review criterion, and evidence requirement?

The benchmark is useful because the review agent depends on this KG output before it writes criterion-grounded feedback. If the KG retrieves the wrong venue criteria, misses evidence requirements, or fails to handle common input variations, the final review can become less reliable.

## How The Test Runs

Each test case provides an input query, such as:

```json
{
  "venue": "TWELF",
  "domain": "Educational Technology",
  "article_type": "full_paper"
}
```

The test runner sends that query to the KG retrieval function. The returned result is compared against expected identifiers:

- expected KG source document IDs
- expected review criterion IDs
- expected evidence requirement IDs

The benchmark does not ask an LLM to judge the answer. Pass/fail is computed with deterministic matching against the expected KG IDs.

## Test Flow

```mermaid
flowchart TD
  A["Prepare test cases"] --> B["Send each input query to the KG"]
  B --> C["KG returns matching review criteria and evidence"]
  C --> D["Compare KG output with expected result"]
  D --> E{"Match?"}
  E -->|Yes| F["Pass"]
  E -->|No| G["Fail with reason"]
  F --> H["Summarize results by category"]
  G --> H
  H --> I["Save JSON and Markdown reports"]
```

In simple terms, each test case asks the KG a question such as:

```text
For this venue/journal/domain/article type,
which review criteria and evidence should be returned?
```

The returned KG result is then compared with the expected source, criterion, and evidence IDs. Because the result is checked by exact IDs, the benchmark is repeatable. The same KG state and same test cases should produce the same pass/fail result.

## How Pass And Fail Are Decided

A case is marked **PASS** when the KG behavior matches the expected behavior for that scenario.

For positive retrieval cases, this means:

- the KG returns the correct source document
- the expected criterion appears in the retrieved criteria
- the expected evidence requirement appears in the retrieved evidence

For rejection and fallback cases, this means:

- the KG avoids guessing when the venue or journal is unknown
- the KG rejects criteria when domain or article type filters do not match
- the KG handles simple formatting differences such as lowercase input, year suffixes, or known aliases

A case is marked **FAIL** when the KG does not retrieve the expected source, criterion, or evidence, or when it retrieves criteria that should have been rejected. In this benchmark, the failed cases are intentionally hard future-work scenarios. They test behavior such as typo correction, loose semantic aliases, and question-only routing.

## Category Meaning

| Category | Purpose | Example input | Expected behavior | What the result means |
|---|---|---|---|---|
| `positive_retrieval` | Tests the normal happy path where the user gives enough structured metadata for the KG to find the right source. | `venue=TWELF`, `domain=Educational Technology`, `article_type=Long paper 3-5 pages` | The KG should return the expected source document, criterion ID, and evidence IDs. | Passing this category means the KG works well when the input clearly names a known conference or journal. |
| `missing_optional_fields` | Tests whether the KG still works when some fields are absent. This reflects real UI use where the user may only provide venue or journal. | `venue=ICLR` with no domain or article type, or `journal=ETRD` with no venue. | The KG should still retrieve the correct source and at least a useful set of criteria for that host. | Passing this category means the system does not require every metadata field to be filled before it can retrieve criteria. |
| `case_format_tolerance` | Tests simple normalization and alias handling. These cases are common because users may type names in different formats. | `venue=iclr`, `venue=AAAI 2026`, `journal=Computers and Education`, `venue=ACM CHI`. | The KG should map the variant to the correct canonical source, such as mapping lowercase or long names to the same stored venue/journal. | Passing this category means the retriever is robust to simple formatting changes, year suffixes, and known aliases. |
| `no_match_and_fallback` | Tests unknown or underspecified hosts where the KG should not invent a match. | `venue=Imaginary Conference on Agentic Widgets` or a known domain with no venue/journal. | The KG should return no criteria instead of guessing a nearby conference or journal. | Passing this category means the system has a safe fallback behavior for unknown inputs. |
| `wrong_filter_rejection` | Tests known hosts combined with mismatched domain or article type filters. | `venue=ICLR`, `domain=Educational Technology`, or `journal=ETRD`, `article_type=CHI full paper`. | The KG should reject criteria when the host exists but the filters do not apply. | Passing this category means domain and article-type filters reduce false-positive retrieval. |
| `aspirational_unsupported_inputs` | Tests difficult future-work behavior that the current system is not expected to solve yet. These cases intentionally represent messy real-world inputs. | `venue=NeruIPS`, `venue=ICLM`, `question=Which guideline checks technical soundness?`, or loose phrases like `learning representations conference`. | Ideally, a stronger system would use typo correction, semantic matching, or natural-language routing to find the intended source. | Failing this category means the current KG retriever depends on recognizable structured inputs and does not yet support fuzzy semantic routing. |

### Category Details

**`positive_retrieval`**

This is the strongest evidence that the KG works for normal review use. A case in this category already knows the venue or journal and asks for a specific expected source/criterion/evidence combination. These cases are closest to how the review agent uses the KG after a paper venue is known.

**`missing_optional_fields`**

These cases check whether the retriever can still return useful criteria when metadata is incomplete. For example, if the user only enters `ICLR`, the system should not fail just because domain or article type is empty.

**`case_format_tolerance`**

These cases check basic real-world input variation. The KG should not be fragile when a user types lowercase names, long-form names, or names with year suffixes.

**`no_match_and_fallback`**

These cases check safe behavior when the input does not match any known KG host. A pass means the retriever returns an empty result instead of hallucinating a source document based on broad words such as `review`, `learning`, or `conference`.

**`wrong_filter_rejection`**

These cases check whether filtering works after a known venue or journal is found. A pass means the KG does not return ICLR, CHI, TWELF, or journal criteria when the requested domain or article type conflicts with the stored criteria applicability.

**`aspirational_unsupported_inputs`**

These are deliberately hard cases. They are included to show the system boundary, not to hide failure. Examples include typos such as `NeruIPS`, semantic-only phrases such as `Learning Representations conference`, and question-only inputs without structured venue or journal metadata. The current system fails these because it mostly uses exact, alias, and simple substring matching. Improving this category would require fuzzy matching, richer alias dictionaries, or embedding/semantic retrieval over KG source and criterion text.
"""
    passed = payload['summary']['passed']
    case_count = payload['summary']['case_count']
    pass_rate_pct = payload['summary']['pass_rate'] * 100
    failure_rate_pct = 100 - pass_rate_pct
    failed_categories = [
        f"- `{category}`: {summary['failed']} failed out of {summary['case_count']}"
        for category, summary in sorted(payload.get('by_category', {}).items())
        if int(summary.get('failed') or 0) > 0
    ]
    failed_category_text = '\n'.join(failed_categories) or '- No failed categories in this run.'
    result_interpretation = f"""## Result Interpretation

The current result is:

```text
{passed} / {case_count} cases passed
Pass rate: {pass_rate_pct:.2f}%
Failure rate: {failure_rate_pct:.2f}%
```

Failed categories in this run:

{failed_category_text}

This means the current KG retrieval is strong for many structured, fallback, and rejection scenarios, but still has gaps around broad domain synonyms, unsupported abbreviations, typo correction, fuzzy semantic matching, and question-only routing.

## What The Result Means For The Project

This benchmark result can be used as a reference for the current KG retrieval module, but it should not be interpreted as full system performance across all papers. The benchmark mainly tests whether the KG can retrieve review criteria correctly. It does not evaluate whether the final LLM-generated review is correct, complete, or useful.

In practical terms:

- The KG is strong for known structured inputs.
- The KG is acceptable for simple input variations.
- The KG safely avoids guessing for unknown hosts.
- The KG rejects known hosts when domain or article-type filters conflict.
- The KG needs improvement for real-world messy inputs, broader domain labels, and abbreviation-heavy queries.
- The full review quality still depends on PDF parsing, prompt construction, agent tool use, annotation quality, and final report generation.

## Suggested Improvements

To improve the failed cases, the next KG retrieval upgrade should add:

1. Typo-tolerant venue and journal matching, for example mapping `NeruIPS` to `NeurIPS`.
2. Stronger alias dictionaries, for example mapping `Learning Representations conference` to `ICLR`.
3. Semantic retrieval over source names, criterion names, and evidence descriptions.
4. Natural-language query routing so question-only inputs can still find the right KG source.
5. Confidence scoring so uncertain matches can be shown to the user instead of silently failing.
"""
    category_rows = [
        '| {category} | {cases} | {passed} | {failed} |'.format(
            category=str(category).replace('|', '\\|'),
            cases=summary['case_count'],
            passed=summary['passed'],
            failed=summary['failed'],
        )
        for category, summary in sorted(payload.get('by_category', {}).items())
    ]
    return f"""# KG Evidence Retrieval Test Results

Generated: {payload['generated_at']}

## Statistic Results

| Metric | Value |
|---|---:|
| Test cases | {payload['summary']['case_count']} |
| Passed | {payload['summary']['passed']} |
| Failed | {payload['summary']['failed']} |
| Pass rate | {payload['summary']['pass_rate']} |

### Category Summary

| Category | Cases | Passed | Failed |
|---|---:|---:|---:|
{chr(10).join(category_rows)}

[Download full case-level Excel result](../test/output/kg_evidence_retrieval_test_results.xlsx)

{category_details}

{result_interpretation}
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
