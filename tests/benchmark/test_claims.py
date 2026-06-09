from benchmark.claims import extract_critique_claims

SAMPLE = """## Weaknesses
- Major gap (evidence: "Table 1"; C06).
- Another issue without inline evidence.

## Key Issues
1. Statistical reporting weak (See Section 2.2).

## Strengths
- Should be ignored.

## Scores
- 6/10
"""


def test_extract_weaknesses_and_key_issues_only():
    claims = extract_critique_claims(SAMPLE, manuscript='A' * 5000, max_claims=10)
    assert len(claims) == 3
    assert 'Should be ignored' not in ' '.join(c.text for c in claims)


def test_extract_uses_manuscript_not_citation_label():
    ms = '## Abstract\n\nWe report 230% increase in confidence.\n\n## 1 Introduction\n'
    report = '## Weaknesses\n- Stats weak (evidence: Abstract; C06).\n'
    claims = extract_critique_claims(report, manuscript=ms, max_claims=5)
    assert len(claims) == 1
    assert '230%' in (claims[0].evidence_span or '')
    assert claims[0].context_source in ('section_slice', 'manuscript_prefix', 'annotation_match')


def test_cap_at_max_claims():
    text = '## Weaknesses\n' + '\n'.join(f'- item {i}' for i in range(20))
    claims = extract_critique_claims(text, manuscript='ms', max_claims=10)
    assert len(claims) == 10
