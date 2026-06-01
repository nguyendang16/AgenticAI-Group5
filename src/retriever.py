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
    domain_name = (domain or '').strip()
    article_type_name = (article_type or '').strip()

    query = """
    OPTIONAL MATCH (v:Venue)
    WHERE ($venue_slug <> '' AND v.venue_id = $venue_slug)
       OR ($venue <> '' AND toLower(v.name) CONTAINS toLower($venue))
    OPTIONAL MATCH (j:Journal)
    WHERE ($journal_slug <> '' AND j.journal_id = $journal_slug)
       OR ($journal <> '' AND toLower(j.name) CONTAINS toLower($journal))
    WITH coalesce(v, j) AS host, labels(coalesce(v, j))[0] AS host_label
    WHERE host IS NOT NULL
    OPTIONAL MATCH (host)-[:HAS_CRITERION]->(c:ReviewCriterion)
    OPTIONAL MATCH (c)-[:REQUIRES_EVIDENCE]->(e:EvidenceRequirement)
    OPTIONAL MATCH (c)-[:SUPPORTED_BY_SOURCE]->(d:SourceDocument)
    OPTIONAL MATCH (host)-[:HAS_REVIEW_FIELD]->(f:ReviewFormField)
    WITH host, host_label, c, collect(DISTINCT e) AS evidence_list, d, collect(DISTINCT f) AS fields
    WHERE c IS NOT NULL
      AND (
        $domain_name = '' OR $domain_name IN coalesce(c.applies_to_domain, [])
        OR EXISTS { MATCH (host)-[:BELONGS_TO_DOMAIN]->(dom:Domain) WHERE dom.name = $domain_name }
      )
      AND (
        $article_type_name = '' OR $article_type_name IN coalesce(c.applies_to_article_type, [])
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
           d.source_id AS source_id,
           d.file_name AS file_name
    """

    criteria_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    provenance: list[dict[str, str]] = []
    seen_sources: set[str] = set()

    with driver.session(database=database) as session:
        rows = session.run(
            query,
            venue=venue or '',
            venue_slug=venue_slug,
            journal=journal or '',
            journal_slug=journal_slug,
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
                    'source_id': record['source_id'],
                    'file_name': record['file_name'],
                },
            }
            criteria_by_group[group].append(entry)
            sid = record['source_id'] or ''
            if sid and sid not in seen_sources:
                seen_sources.add(sid)
                provenance.append({'source_id': sid, 'file_name': record['file_name'] or ''})

    driver.close()

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
