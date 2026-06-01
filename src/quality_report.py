from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.schema import ExtractedTemplate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Quality report for extracted template JSON')
    parser.add_argument('--json_dir', type=Path, default=Path('outputs/extracted_json'))
    parser.add_argument('--output', type=Path, default=Path('outputs/logs/quality_report.json'))
    return parser.parse_args()


def _load_templates(json_dir: Path) -> list[ExtractedTemplate]:
    templates: list[ExtractedTemplate] = []
    for path in sorted(json_dir.glob('*.json')):
        payload = json.loads(path.read_text(encoding='utf-8'))
        templates.append(ExtractedTemplate.model_validate(payload))
    return templates


def build_quality_report(templates: list[ExtractedTemplate]) -> dict:
    group_counts: Counter[str] = Counter()
    missing_evidence: list[str] = []
    missing_quote: list[str] = []
    low_confidence: list[str] = []
    duplicate_names: dict[str, list[str]] = defaultdict(list)

    for template in templates:
        sd = template.source_document
        name_index: dict[str, str] = {}
        for c in template.review_criteria:
            group_counts[c.criterion_group] += 1
            if not c.evidence_required:
                missing_evidence.append(f'{sd.source_id}:{c.criterion_id}')
            if not c.source_quote.strip():
                missing_quote.append(f'{sd.source_id}:{c.criterion_id}')
            if c.confidence == 'low':
                low_confidence.append(f'{sd.source_id}:{c.criterion_id}')
            key = c.criterion_name.strip().lower()
            if key in name_index:
                duplicate_names[key].append(c.criterion_id)
            else:
                name_index[key] = c.criterion_id

    return {
        'template_count': len(templates),
        'total_criteria': sum(len(t.review_criteria) for t in templates),
        'total_form_fields': sum(len(t.review_form_fields) for t in templates),
        'total_checklist_items': sum(len(t.checklist_items) for t in templates),
        'criteria_by_group': dict(group_counts),
        'criteria_missing_evidence': missing_evidence,
        'criteria_missing_source_quote': missing_quote,
        'criteria_low_confidence': low_confidence,
        'possible_duplicate_names': {k: v for k, v in duplicate_names.items() if v},
        'templates': [
            {
                'source_id': t.source_document.source_id,
                'file_name': t.source_document.file_name,
                'venue_or_journal': t.source_document.venue_or_journal_name,
                'criteria_count': len(t.review_criteria),
            }
            for t in templates
        ],
    }


def main() -> int:
    args = _parse_args()
    json_dir = args.json_dir.resolve()
    templates = _load_templates(json_dir)
    if not templates:
        print(f'No JSON files in {json_dir}')
        return 1

    report = build_quality_report(templates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'Templates: {report["template_count"]}')
    print(f'Total criteria: {report["total_criteria"]}')
    print('Criteria by group:')
    for group, count in sorted(report['criteria_by_group'].items()):
        print(f'  {group}: {count}')
    print(f'Missing evidence: {len(report["criteria_missing_evidence"])}')
    print(f'Missing source_quote: {len(report["criteria_missing_source_quote"])}')
    print(f'Low confidence: {len(report["criteria_low_confidence"])}')
    print(f'Possible duplicate name groups: {len(report["possible_duplicate_names"])}')
    print(f'Full report: {args.output.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
