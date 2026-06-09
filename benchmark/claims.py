from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass

from benchmark.evidence_resolve import ResolvedEvidence, resolve_evidence_span

_SECTIONS = ('## Weaknesses', '## Key Issues')
_BULLET_RE = re.compile(r'^(?:-\s+|\d+\.\s+)(.+)$', re.MULTILINE)
_EVIDENCE_RE = re.compile(r'\(evidence:\s*["\']?([^"\')]+)', re.IGNORECASE)
_PAREN_REF_RE = re.compile(r'\((?:See|Appendix|Section|Figure|Table|Abstract)[^)]+\)', re.IGNORECASE)


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


def _max_context_chars() -> int:
    return int(os.environ.get('BENCHMARK_RAGAS_MAX_CONTEXT_CHARS', '2000'))


def _section_slice(report_md: str, header: str) -> str:
    if header not in report_md:
        return ''
    part = report_md.split(header, 1)[1]
    stop_headers = (
        *(h for h in _SECTIONS if h != header),
        '## Strengths',
        '## Actionable',
        '## Claim-Level',
        '## Scores',
        '## Criterion',
    )
    for stop in stop_headers:
        if stop in part:
            part = part.split(stop, 1)[0]
    return part


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
    for header in _SECTIONS:
        for match in _BULLET_RE.finditer(_section_slice(report_md, header)):
            text = match.group(1).strip()
            if not text:
                continue
            resolved = _evidence_for_bullet(text, manuscript, annotations)
            claims.append(
                AtomicClaim(
                    claim_id=str(uuid.uuid4()),
                    text=text,
                    section=header.removeprefix('## '),
                    evidence_span=resolved.text,
                    context_source=resolved.source,
                    resolved_context_preview=resolved.preview,
                )
            )
            if len(claims) >= cap:
                return claims
    return claims
