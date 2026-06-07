from __future__ import annotations

import os
import re
from dataclasses import dataclass

_STOPWORDS = frozenset({'the', 'a', 'an', 'and', 'or', 'to', 'of', 'in', 'on', 'for', 'is', 'are'})

_SECTION_REF_RE = re.compile(
    r'\b(abstract|section\s+[\d.]+|appendix\s+[a-z0-9]+|figure\s+\d+|table\s+\d+)\b',
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE)
_QUOTE_RE = re.compile(r'["\']([^"\']{20,})["\']')
_CRITERION_TAG_RE = re.compile(r'\bC0?\d\b', re.IGNORECASE)


@dataclass
class ResolvedEvidence:
    text: str
    source: str
    preview: str


def _content_tokens(text: str) -> set[str]:
    return {
        t.lower()
        for t in re.findall(r'[a-zA-Z]{3,}', text)
        if t.lower() not in _STOPWORDS
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _min_jaccard() -> float:
    return float(os.environ.get('BENCHMARK_RAGAS_ANNOTATION_MATCH_MIN_JACCARD', '0.15'))


def _match_annotation(bullet: str, annotations: list[dict]) -> ResolvedEvidence | None:
    bullet_tokens = _content_tokens(bullet)
    bullet_criteria = {m.group(0).upper() for m in _CRITERION_TAG_RE.finditer(bullet)}
    best_score = 0.0
    best_text = ''
    for item in annotations:
        comment = str(item.get('comment') or item.get('summary') or '')
        criterion = str(item.get('criterion_id') or '')
        ann_tokens = _content_tokens(comment)
        score = _jaccard(bullet_tokens, ann_tokens)
        if bullet_criteria and criterion:
            if any(tag in criterion.upper() for tag in bullet_criteria):
                score = max(score, 0.25)
        if score > best_score:
            best_score = score
            best_text = str(item.get('text') or '')
    if best_score >= _min_jaccard() and best_text.strip():
        text = best_text.strip()
        return ResolvedEvidence(text=text[: _max_chars()], source='annotation_match', preview=text[:120])
    return None


def _max_chars() -> int:
    return int(os.environ.get('BENCHMARK_RAGAS_MAX_CONTEXT_CHARS', '2000'))


def _slice_section(manuscript: str, ref: str) -> str:
    ref_lower = ref.lower().strip()
    headings = list(_HEADING_RE.finditer(manuscript))
    start_idx = None
    for match in headings:
        heading = match.group(1).lower()
        if ref_lower in heading or heading.startswith(ref_lower):
            start_idx = match.start()
            break
    if start_idx is None:
        if ref_lower == 'abstract':
            for match in headings:
                if 'abstract' in match.group(1).lower():
                    start_idx = match.start()
                    break
    if start_idx is None:
        return ''
    end_idx = len(manuscript)
    for match in headings:
        if match.start() > start_idx:
            end_idx = match.start()
            break
    return manuscript[start_idx:end_idx].strip()[: _max_chars()]


def _section_slice(bullet: str, manuscript: str) -> ResolvedEvidence | None:
    refs = _SECTION_REF_RE.findall(bullet)
    for ref in refs:
        chunk = _slice_section(manuscript, ref)
        if len(chunk) >= 80:
            return ResolvedEvidence(text=chunk, source='section_slice', preview=chunk[:120])
    return None


def _quoted_in_claim(bullet: str, manuscript: str) -> ResolvedEvidence | None:
    for match in _QUOTE_RE.finditer(bullet):
        quote = match.group(1).strip()
        if quote in manuscript:
            return ResolvedEvidence(
                text=quote[: _max_chars()],
                source='quoted_in_claim',
                preview=quote[:120],
            )
    return None


def resolve_evidence_span(
    bullet: str,
    *,
    manuscript: str,
    annotations: list[dict] | None = None,
    max_chars: int | None = None,
) -> ResolvedEvidence:
    cap = max_chars if max_chars is not None else _max_chars()
    ann = annotations or []

    matched = _match_annotation(bullet, ann)
    if matched:
        matched.text = matched.text[:cap]
        return matched

    sliced = _section_slice(bullet, manuscript)
    if sliced:
        sliced.text = sliced.text[:cap]
        return sliced

    quoted = _quoted_in_claim(bullet, manuscript)
    if quoted:
        quoted.text = quoted.text[:cap]
        return quoted

    prefix = manuscript[:cap]
    return ResolvedEvidence(text=prefix, source='manuscript_prefix', preview=prefix[:120])
