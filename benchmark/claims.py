from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass

from benchmark.evidence_resolve import ResolvedEvidence, resolve_evidence_span

_SECTION_NAMES = ('Weaknesses', 'Key Issues')
_BULLET_RE = re.compile(r'^(?:[-•]\s+|\d+\.\s+)(.+)$', re.MULTILINE)
_MD_STOP_HEADERS = (
    '## Weaknesses',
    '## Key Issues',
    '## Strengths',
    '## Actionable',
    '## Claim-Level',
    '## Scores',
    '## Criterion',
)


@dataclass
class AtomicClaim:
    claim_id: str
    text: str
    section: str
    evidence_span: str | None
    context_source: str = 'legacy'
    resolved_context_preview: str = ''


def _max_claims() -> int:
    return int(os.environ.get('BENCHMARK_RAGAS_MAX_CLAIMS', '10'))


def _section_slice(report_md: str, section_name: str) -> str:
    md_match = re.search(
        rf'^##\s+{re.escape(section_name)}\s*$',
        report_md,
        re.MULTILINE | re.IGNORECASE,
    )
    if md_match:
        part = report_md[md_match.end() :]
        for stop in _MD_STOP_HEADERS:
            if stop.lower() == f'## {section_name}'.lower():
                continue
            if stop in part:
                part = part.split(stop, 1)[0]
        return part

    num_match = re.search(
        rf'^\d+\.\s+{re.escape(section_name)}\s*$',
        report_md,
        re.MULTILINE | re.IGNORECASE,
    )
    if num_match:
        part = report_md[num_match.end() :]
        next_numbered = re.search(r'^\d+\.\s+', part, re.MULTILINE)
        if next_numbered:
            part = part[: next_numbered.start()]
        if 'PART B' in part:
            part = part.split('PART B', 1)[0]
        return part

    return ''


def _evidence_for_bullet(
    bullet: str,
    manuscript: str,
    annotations: list[dict] | None = None,
) -> ResolvedEvidence:
    return resolve_evidence_span(bullet, manuscript=manuscript, annotations=annotations or [])


def extract_critique_claims(
    report_md: str,
    *,
    manuscript: str,
    annotations: list[dict] | None = None,
    max_claims: int | None = None,
) -> list[AtomicClaim]:
    cap = max_claims if max_claims is not None else _max_claims()
    claims: list[AtomicClaim] = []
    for section_name in _SECTION_NAMES:
        for match in _BULLET_RE.finditer(_section_slice(report_md, section_name)):
            text = match.group(1).strip()
            if not text:
                continue
            resolved = _evidence_for_bullet(text, manuscript, annotations)
            claims.append(
                AtomicClaim(
                    claim_id=str(uuid.uuid4()),
                    text=text,
                    section=section_name,
                    evidence_span=resolved.text,
                    context_source=resolved.source,
                    resolved_context_preview=resolved.preview,
                )
            )
            if len(claims) >= cap:
                return claims
    return claims
