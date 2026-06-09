from benchmark.trad_ingest import extract_pdf_text, validate_trad_review_bytes


def test_validate_trad_review_bytes_min_size():
    assert validate_trad_review_bytes(b'x' * 2049) == []
    assert validate_trad_review_bytes(b'short') != []


def test_extract_pdf_text_roundtrip(tmp_path):
    import fitz

    pdf_path = tmp_path / 'sample.pdf'
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), 'Peer review summary for benchmark test.')
    doc.save(pdf_path)
    doc.close()
    text = extract_pdf_text(pdf_path)
    assert 'Peer review summary' in text
