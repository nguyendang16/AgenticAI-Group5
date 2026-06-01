from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.config import EXTRACT_MODEL
from src.docx_io import extract_docx_text
from src.llm_extractor import llm_extract_template
from src.metadata_hints import infer_metadata_from_path
from src.normalizer import merge_duplicate_criteria, normalize_template
from src.validator import validate_template


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Extract review templates from .docx via LLM')
    parser.add_argument('--template_dir', type=Path, default=Path('templates'))
    parser.add_argument('--output_dir', type=Path, default=Path('outputs/extracted_json'))
    parser.add_argument('--log_dir', type=Path, default=Path('outputs/logs'))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    template_dir = args.template_dir.resolve()
    output_dir = args.output_dir.resolve()
    log_dir = args.log_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    docx_files = sorted(template_dir.glob('*.docx'))
    if not docx_files:
        print(f'No .docx files found in {template_dir}')
        return 1

    print(f'Extraction model: {EXTRACT_MODEL}')
    print(f'Found {len(docx_files)} templates')

    all_warnings: list[dict] = []
    ok_count = 0
    fail_count = 0
    for path in tqdm(docx_files, desc='Extracting'):
        file_warnings: list[dict] = [{'type': 'file', 'file_name': path.name}]
        try:
            hints = infer_metadata_from_path(path)
            raw_text = extract_docx_text(path)
            if not raw_text.strip():
                file_warnings.append({'type': 'empty_document', 'file_name': path.name})
                all_warnings.append({'file': path.name, 'warnings': file_warnings})
                continue

            template = llm_extract_template(raw_text=raw_text, hints=hints)
            template = normalize_template(template)
            template, dup_warnings = merge_duplicate_criteria(template)
            file_warnings.extend(dup_warnings)
            file_warnings.extend(validate_template(template))

            out_path = output_dir / f'{template.source_document.source_id}.json'
            out_path.write_text(
                json.dumps(template.model_dump(mode='json'), ensure_ascii=False, indent=2),
                encoding='utf-8',
            )
            warn_path = log_dir / f'{template.source_document.source_id}.warnings.json'
            warn_path.write_text(
                json.dumps(file_warnings, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )
            all_warnings.append({'file': path.name, 'source_id': template.source_document.source_id, 'warnings': file_warnings})
            ok_count += 1
        except Exception as exc:
            fail_count += 1
            file_warnings.append({'type': 'extraction_failed', 'error': str(exc)})
            all_warnings.append({'file': path.name, 'warnings': file_warnings})
            (log_dir / f'{path.stem}.errors.json').write_text(
                json.dumps(file_warnings, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )

    summary_path = log_dir / 'extraction_summary.json'
    summary_path.write_text(json.dumps(all_warnings, ensure_ascii=False, indent=2), encoding='utf-8')
    json_written = len(list(output_dir.glob('*.json'))) - (1 if (output_dir / '.gitkeep').exists() else 0)
    print(f'Extracted OK: {ok_count}/{len(docx_files)}, failed: {fail_count}')
    print(f'JSON files in {output_dir}: {json_written}')
    print(f'Warnings/logs in {log_dir}')
    return 1 if fail_count else 0


if __name__ == '__main__':
    raise SystemExit(main())
