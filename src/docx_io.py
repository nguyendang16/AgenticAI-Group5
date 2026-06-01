from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph


def _iter_block_items(document: Document):
    body = document.element.body
    for child in body.iterchildren():
        tag = child.tag.split('}')[-1]
        if tag == 'p':
            yield Paragraph(child, document)
        elif tag == 'tbl':
            yield Table(child, document)


def _table_to_text(table: Table) -> str:
    rows: list[str] = []
    for row in table.rows:
        cells = [cell.text.strip().replace('\n', ' ') for cell in row.cells]
        if any(cells):
            rows.append(' | '.join(cells))
    return '\n'.join(rows)


def extract_docx_text(path: Path, *, max_chars: int = 120_000) -> str:
    document = Document(str(path))
    parts: list[str] = []
    for block in _iter_block_items(document):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if text:
                parts.append(text)
        elif isinstance(block, Table):
            table_text = _table_to_text(block)
            if table_text:
                parts.append('[TABLE]\n' + table_text)
    combined = '\n\n'.join(parts).strip()
    if len(combined) > max_chars:
        return combined[:max_chars] + '\n\n[...truncated for extraction...]'
    return combined
