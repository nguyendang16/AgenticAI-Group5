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
    assert claims[0].evidence_span == 'Table 1'
    assert 'Should be ignored' not in ' '.join(c.text for c in claims)


def test_cap_at_max_claims():
    text = '## Weaknesses\n' + '\n'.join(f'- item {i}' for i in range(20))
    claims = extract_critique_claims(text, manuscript='ms', max_claims=10)
    assert len(claims) == 10
