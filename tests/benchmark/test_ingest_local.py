from __future__ import annotations

from pathlib import Path

from benchmark.ingest_local import ingest_kg_test_papers, LOCAL_VENUE_OVERRIDES


def test_local_override_table_has_eleven_entries():
    assert len(LOCAL_VENUE_OVERRIDES) == 11


def test_ingest_finds_acl_paper(tmp_path, monkeypatch):
    # Use real KG_testPapers if present; skip otherwise
    from benchmark.paths import KG_TEST_PAPERS_DIR
    if not KG_TEST_PAPERS_DIR.exists():
        return
    rows = ingest_kg_test_papers()
    assert len(rows) >= 11
    venues = {r.venue for r in rows}
    assert 'ACL' in venues
    assert 'ICLR' in venues
