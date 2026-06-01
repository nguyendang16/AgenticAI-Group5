from __future__ import annotations

from src.criterion_ids import (
    assign_semantic_criterion_ids,
    is_generic_criterion_id,
    semantic_criterion_id,
    venue_prefix_from_name,
)


def test_venue_prefix_twelf2026():
    assert venue_prefix_from_name('TWELF2026') == 'TWELF'


def test_semantic_id_format():
    cid = semantic_criterion_id(venue_prefix='TWELF', index=1, criterion_group='SCOPE_FIT')
    assert cid == 'TWELF_C01_SCOPE_FIT'


def test_assign_semantic_ids_replaces_generic():
    criteria = [
        {'criterion_id': 'criterion_1', 'criterion_name': 'A', 'criterion_group': 'SCOPE_FIT'},
        {'criterion_id': 'criterion_2', 'criterion_name': 'B', 'criterion_group': 'CLARITY'},
    ]
    out = assign_semantic_criterion_ids(criteria, venue='TWELF')
    assert out[0]['criterion_id'] == 'TWELF_C01_SCOPE_FIT'
    assert out[1]['criterion_id'] == 'TWELF_C02_CLARITY'
    assert not is_generic_criterion_id(out[0]['criterion_id'])
