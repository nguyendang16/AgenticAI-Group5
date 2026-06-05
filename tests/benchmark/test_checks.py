from benchmark.checks import validate_condition, validate_run_completion


def test_kg_on_requires_legend():
    row = {
        'condition': 'KG_ON',
        'criteria_count': 5,
        'has_legend': True,
        'has_claim_audit': True,
        'status': 'completed',
    }
    assert validate_condition(row) == []


def test_kg_off_rejects_legend():
    row = {
        'condition': 'KG_OFF',
        'criteria_count': 0,
        'has_legend': True,
        'status': 'completed',
    }
    errs = validate_condition(row)
    assert any('legend' in e.lower() for e in errs)
