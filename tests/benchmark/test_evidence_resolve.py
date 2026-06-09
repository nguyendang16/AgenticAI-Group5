import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures' / 'benchmark'


@pytest.fixture
def manuscript() -> str:
    return (FIXTURES / 'acl_short_manuscript.md').read_text(encoding='utf-8')


@pytest.fixture
def annotations() -> list[dict]:
    payload = json.loads((FIXTURES / 'acl_short_annotations.json').read_text(encoding='utf-8'))
    return payload['annotations']


def test_section_slice_abstract(manuscript):
    from benchmark.evidence_resolve import resolve_evidence_span

    bullet = (
        'Statistical reporting is weak (evidence: Abstract and Section 2.2/2.3; C06).'
    )
    resolved = resolve_evidence_span(bullet, manuscript=manuscript, annotations=[])
    assert resolved.source == 'section_slice'
    assert '230%' in resolved.text or 'Abstract' in resolved.text
    assert 'Abstract and Section 2.2' not in resolved.text


def test_annotation_match(manuscript, annotations):
    from benchmark.evidence_resolve import resolve_evidence_span

    bullet = (
        'Statistical reporting is weak: quantitative claims lack confidence intervals '
        '(evidence: Abstract; C06).'
    )
    resolved = resolve_evidence_span(
        bullet, manuscript=manuscript, annotations=annotations
    )
    assert resolved.source == 'annotation_match'
    assert '230%' in resolved.text


def test_never_returns_citation_label(manuscript):
    from benchmark.evidence_resolve import resolve_evidence_span

    bullet = 'Gap (evidence: Appendix C; C06).'
    resolved = resolve_evidence_span(bullet, manuscript=manuscript, annotations=[])
    assert resolved.source != 'citation_label'
    assert 'Appendix C; C06' not in resolved.text
