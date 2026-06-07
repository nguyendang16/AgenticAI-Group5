def test_save_and_load_parse_cache(tmp_path, monkeypatch):
    import benchmark.parse_cache as mod

    cache_root = tmp_path / 'parses'
    monkeypatch.setattr(mod, 'PARSES_DIR', cache_root)

    from benchmark.parse_cache import has_parse_cache, load_parse_cache, save_parse_cache

    paper_id = 'test-paper'
    job_dir = tmp_path / 'job-1'
    job_dir.mkdir()
    (job_dir / 'mineru_full.md').write_text('# Parsed paper', encoding='utf-8')
    (job_dir / 'mineru_content_list.json').write_text('[]', encoding='utf-8')

    save_parse_cache(paper_id, job_dir)
    assert has_parse_cache(paper_id)

    target = tmp_path / 'job-2'
    assert load_parse_cache(paper_id, target)
    assert (target / 'mineru_full.md').read_text(encoding='utf-8') == '# Parsed paper'
