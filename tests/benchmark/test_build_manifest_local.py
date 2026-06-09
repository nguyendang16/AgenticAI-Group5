def test_build_manifest_local_only(tmp_path, monkeypatch):
    import benchmark.build_manifest as mod
    import benchmark.ingest_local as ingest_mod

    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    (papers_dir / 'acl_test-paper.pdf').write_bytes(b'%PDF-1.4 fake')

    manifest_path = tmp_path / 'manifest.jsonl'
    monkeypatch.setattr(ingest_mod, 'PAPERS_DIR', papers_dir)
    monkeypatch.setattr(ingest_mod, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(mod, 'MANIFEST_PATH', manifest_path)

    rows = mod.build_manifest(local_only=True)
    assert len(rows) >= 1
    assert manifest_path.exists()
    assert rows[0].paper_id == 'acl_test-paper'
