from __future__ import annotations


def test_parse_trad_pairwise_response_basic_json():
    from benchmark.trad_pairwise_judge import parse_trad_pairwise_response

    raw = (
        '{"winner":"TRAD","rubric_alignment_winner":"KG_OFF",'
        '"confidence":"high","one_line_reason":"More venue-specific critique"}'
    )
    parsed = parse_trad_pairwise_response(raw)
    assert parsed['winner'] == 'TRAD'
    assert parsed['rubric_alignment_winner'] == 'KG_OFF'
    assert parsed['confidence'] == 'high'
    assert parsed['reason'] == 'More venue-specific critique'


def test_parse_trad_pairwise_response_json_fence():
    from benchmark.trad_pairwise_judge import parse_trad_pairwise_response

    raw = """```json
{"winner":"KG_ON","rubric_alignment_winner":"tie","confidence":"medium","one_line_reason":"Both are balanced"}
```"""
    parsed = parse_trad_pairwise_response(raw)
    assert parsed['winner'] == 'KG_ON'
    assert parsed['rubric_alignment_winner'] == 'tie'
    assert parsed['confidence'] == 'medium'
    assert parsed['reason'] == 'Both are balanced'


def test_parse_trad_pairwise_response_reason_fallback():
    from benchmark.trad_pairwise_judge import parse_trad_pairwise_response

    raw = (
        '{"winner":"tie","rubric_alignment_winner":"TRAD",'
        '"confidence":"low","reason":"Close call on evidence"}'
    )
    parsed = parse_trad_pairwise_response(raw)
    assert parsed['winner'] == 'tie'
    assert parsed['rubric_alignment_winner'] == 'TRAD'
    assert parsed['confidence'] == 'low'
    assert parsed['reason'] == 'Close call on evidence'
