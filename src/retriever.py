from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

from src.config import get_pipeline_settings


def _slug(text: str) -> str:
    import re

    return re.sub(r'[^a-zA-Z0-9]+', '_', str(text or '').lower()).strip('_')


def _alias_slugs(text: str) -> list[str]:
    import re

    slug = _slug(text)
    aliases = [slug] if slug else []
    phrase_aliases = {
        'international conference on machine learning': 'icml',
        'international conference on learning representations': 'iclr',
        'conference on neural information processing systems': 'neurips',
        'neural information processing systems': 'neurips',
        'association for computational linguistics': 'acl',
        'acm conference on human factors in computing systems': 'chi',
    }
    lowered = str(text or '').lower()
    for phrase, alias in phrase_aliases.items():
        if phrase in lowered and alias not in aliases:
            aliases.append(alias)
    short_year = re.sub(r'(^|_)20(\d{2})(_|$)', r'\1\2\3', slug)
    long_year = re.sub(r'(^|_)(\d{2})(_|$)', r'\g<1>20\2\3', slug)
    for alias in (short_year, long_year):
        if alias and alias not in aliases:
            aliases.append(alias)
    return aliases


def _normalize_source_docs(raw: Any) -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    seen: set[str] = set()
    if not isinstance(raw, list):
        return docs
    for item in raw:
        if item is None:
            continue
        props = dict(item) if not isinstance(item, dict) else item
        sid = str(props.get('source_id') or '').strip()
        if not sid or sid in seen:
            continue
        seen.add(sid)
        docs.append({'source_id': sid, 'file_name': str(props.get('file_name') or '').strip()})
    return docs


def _dedupe_criteria_by_group(
    criteria_by_group: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """One entry per criterion_id (fallback: name+group) within each group."""
    deduped: dict[str, list[dict[str, Any]]] = {}
    for group, items in criteria_by_group.items():
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in items:
            cid = str(item.get('criterion_id') or '').strip()
            key = cid or f"{item.get('criterion_name')}|{item.get('description')}"
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        if unique:
            deduped[group] = unique
    return deduped


def retrieve_criteria_bundle(
    *,
    venue: str | None = None,
    journal: str | None = None,
    domain: str | None = None,
    article_type: str | None = None,
) -> dict[str, Any]:
    settings = get_pipeline_settings()
    if not settings.neo4j_uri or not settings.neo4j_password:
        raise RuntimeError('NEO4J_URI and NEO4J_PASSWORD are required')

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    database = settings.neo4j_database or 'neo4j'

    venue_slug = _slug(venue) if venue else ''
    journal_slug = _slug(journal) if journal else ''
    venue_slug_aliases = _alias_slugs(venue or '')
    journal_slug_aliases = _alias_slugs(journal or '')
    domain_name = (domain or '').strip()
    article_type_name = (article_type or '').strip()

    query = """
    OPTIONAL MATCH (v_direct:Venue)
    WHERE ($venue_slug <> '' AND v_direct.venue_id IN $venue_slug_aliases)
       OR ($venue <> '' AND toLower(v_direct.name) CONTAINS toLower($venue))
    OPTIONAL MATCH (venue_doc:SourceDocument)-[:DESCRIBES]->(v_from_source:Venue)
    WHERE $venue <> ''
      AND (
        ANY(alias IN $venue_slug_aliases WHERE toLower(venue_doc.source_id) CONTAINS toLower(alias))
        OR toLower(venue_doc.file_name) CONTAINS toLower($venue)
        OR toLower(venue_doc.source_title) CONTAINS toLower($venue)
      )
    OPTIONAL MATCH (j_direct:Journal)
    WHERE ($journal_slug <> '' AND j_direct.journal_id IN $journal_slug_aliases)
       OR ($journal <> '' AND toLower(j_direct.name) CONTAINS toLower($journal))
    OPTIONAL MATCH (journal_doc:SourceDocument)-[:DESCRIBES]->(j_from_source:Journal)
    WHERE $journal <> ''
      AND (
        ANY(alias IN $journal_slug_aliases WHERE toLower(journal_doc.source_id) CONTAINS toLower(alias))
        OR toLower(journal_doc.file_name) CONTAINS toLower($journal)
        OR toLower(journal_doc.source_title) CONTAINS toLower($journal)
      )
    WITH coalesce(v_direct, v_from_source, j_direct, j_from_source) AS host,
         labels(coalesce(v_direct, v_from_source, j_direct, j_from_source))[0] AS host_label
    WHERE host IS NOT NULL
    OPTIONAL MATCH (host)-[:HAS_CRITERION]->(c:ReviewCriterion)
    OPTIONAL MATCH (c)-[:REQUIRES_EVIDENCE]->(e:EvidenceRequirement)
    OPTIONAL MATCH (c)-[:SUPPORTED_BY_SOURCE]->(d:SourceDocument)
    WITH host, host_label, c,
         collect(DISTINCT e) AS evidence_list,
         [doc IN collect(DISTINCT d) WHERE doc IS NOT NULL |
           {source_id: doc.source_id, file_name: doc.file_name}] AS source_docs
    WHERE c IS NOT NULL
      AND (
        $domain_name = ''
        OR ANY(domain IN coalesce(c.applies_to_domain, []) WHERE toLower(domain) = toLower($domain_name))
        OR EXISTS { MATCH (host)-[:BELONGS_TO_DOMAIN]->(dom:Domain) WHERE toLower(dom.name) = toLower($domain_name) }
      )
      AND (
        $article_type_name = ''
        OR ANY(article_type IN coalesce(c.applies_to_article_type, []) WHERE toLower(article_type) = toLower($article_type_name))
      )
    RETURN host, host_label,
           c.criterion_id AS criterion_id,
           c.criterion_name AS criterion_name,
           c.criterion_group AS criterion_group,
           c.description AS description,
           c.severity_if_missing AS severity_if_missing,
           c.source_quote AS source_quote,
           c.confidence AS confidence,
           c.applies_to_domain AS applies_to_domain,
           c.applies_to_article_type AS applies_to_article_type,
           evidence_list,
           source_docs
    """

    criteria_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    provenance: list[dict[str, str]] = []
    seen_sources: set[str] = set()

    with driver.session(database=database) as session:
        rows = session.run(
            query,
            venue=venue or '',
            venue_slug=venue_slug,
            venue_slug_aliases=venue_slug_aliases,
            journal=journal or '',
            journal_slug=journal_slug,
            journal_slug_aliases=journal_slug_aliases,
            domain_name=domain_name,
            article_type_name=article_type_name,
        )
        for record in rows:
            group = record['criterion_group'] or 'OTHER'
            evidence_items = []
            for ev in record['evidence_list'] or []:
                if ev is None:
                    continue
                props = dict(ev) if not isinstance(ev, dict) else ev
                evidence_items.append(
                    {
                        'evidence_id': props.get('evidence_id'),
                        'name': props.get('name'),
                        'description': props.get('description'),
                        'evidence_type': props.get('evidence_type'),
                        'required': props.get('required'),
                    }
                )
            source_docs = _normalize_source_docs(record.get('source_docs'))
            primary = source_docs[0] if source_docs else {'source_id': '', 'file_name': ''}
            entry = {
                'criterion_id': record['criterion_id'],
                'criterion_name': record['criterion_name'],
                'criterion_group': group,
                'description': record['description'],
                'severity_if_missing': record['severity_if_missing'],
                'source_quote': record['source_quote'],
                'confidence': record['confidence'],
                'applies_to_domain': record['applies_to_domain'] or [],
                'applies_to_article_type': record['applies_to_article_type'] or [],
                'evidence_required': evidence_items,
                'provenance': {
                    'source_id': primary['source_id'],
                    'file_name': primary['file_name'],
                },
                'provenance_sources': source_docs,
            }
            criteria_by_group[group].append(entry)
            for doc in source_docs:
                sid = doc['source_id']
                if sid not in seen_sources:
                    seen_sources.add(sid)
                    provenance.append(doc)

    driver.close()

    criteria_by_group = _dedupe_criteria_by_group(dict(criteria_by_group))

    return {
        'query': {
            'venue': venue,
            'journal': journal,
            'domain': domain,
            'article_type': article_type,
        },
        'criteria_by_group': dict(criteria_by_group),
        'criteria_count': sum(len(v) for v in criteria_by_group.values()),
        'provenance': provenance,
        'verification_ready': {
            'must_reference_criterion_ids': True,
            'groups_present': list(criteria_by_group.keys()),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Retrieve review criteria from Neo4j')
    parser.add_argument('--venue', type=str, default=None)
    parser.add_argument('--journal', type=str, default=None)
    parser.add_argument('--domain', type=str, default=None)
    parser.add_argument('--article-type', type=str, default=None)
    parser.add_argument('--output', type=Path, default=None, help='Write JSON to file')
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    bundle = retrieve_criteria_bundle(
        venue=args.venue,
        journal=args.journal,
        domain=args.domain,
        article_type=args.article_type,
    )
    text = json.dumps(bundle, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
        print(f'Wrote {args.output.resolve()}')
    else:
        print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
