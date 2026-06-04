from __future__ import annotations

from src.retriever import _dedupe_criteria_by_group, _normalize_source_docs


def test_dedupe_criteria_by_group_keeps_one_per_criterion_id():
    grouped = {
        'CLARITY': [
            {'criterion_id': 'criterion_1', 'criterion_name': 'A'},
            {'criterion_id': 'criterion_1', 'criterion_name': 'A'},
        ],
        'NOVELTY': [{'criterion_id': 'criterion_2', 'criterion_name': 'B'}],
    }
    out = _dedupe_criteria_by_group(grouped)
    assert len(out['CLARITY']) == 1
    assert len(out['NOVELTY']) == 1
    assert out['CLARITY'][0]['criterion_id'] == 'criterion_1'


def test_normalize_source_docs_dedupes_by_source_id():
    raw = [
        {'source_id': 's1', 'file_name': 'a.docx'},
        {'source_id': 's1', 'file_name': 'a.docx'},
        {'source_id': 's2', 'file_name': 'b.docx'},
    ]
    docs = _normalize_source_docs(raw)
    assert len(docs) == 2
    assert docs[0]['source_id'] == 's1'
    assert docs[1]['source_id'] == 's2'
