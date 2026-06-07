from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

from deepreview.config import get_settings


def _neo4j_session():
    settings = get_settings()
    if not settings.neo4j_uri or not settings.neo4j_password:
        raise RuntimeError('NEO4J_URI and NEO4J_PASSWORD are required')
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    return driver, settings.neo4j_database or 'neo4j'


def _query_value(session: Any, query: str, **params: Any) -> Any:
    record = session.run(query, **params).single()
    return None if record is None else record[0]


def _graph_snapshot(session: Any) -> dict[str, Any]:
    labels = [
        'SourceDocument',
        'Venue',
        'Journal',
        'ReviewCriterion',
        'EvidenceRequirement',
        'ReviewFormField',
        'Domain',
        'ArticleType',
    ]
    required_properties = {
        'SourceDocument': ['source_id', 'source_title', 'source_type', 'file_name'],
        'Venue': ['venue_id', 'name'],
        'Journal': ['journal_id', 'name'],
        'ReviewCriterion': ['criterion_id', 'criterion_name', 'criterion_group', 'description'],
        'EvidenceRequirement': ['evidence_id', 'name', 'description', 'evidence_type'],
        'ReviewFormField': ['field_id', 'field_name', 'field_type'],
        'Domain': ['name'],
        'ArticleType': ['name'],
    }

    node_counts: dict[str, int] = {}
    missing_by_label: dict[str, dict[str, int]] = {}
    total_required_slots = 0
    missing_required_slots = 0
    for label in labels:
        count = int(_query_value(session, f'MATCH (n:{label}) RETURN count(n)') or 0)
        node_counts[label] = count
        missing_by_label[label] = {}
        for prop in required_properties[label]:
            missing = int(
                _query_value(
                    session,
                    f"""
                    MATCH (n:{label})
                    WHERE n.{prop} IS NULL OR trim(toString(n.{prop})) = ''
                    RETURN count(n)
                    """,
                )
                or 0
            )
            missing_by_label[label][prop] = missing
            total_required_slots += count
            missing_required_slots += missing

    relation_counts = {
        row['rel']: int(row['count'])
        for row in session.run(
            'MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS count ORDER BY rel'
        )
    }
    total_relations = sum(relation_counts.values())

    criterion_count = node_counts['ReviewCriterion']
    source_count = node_counts['SourceDocument']
    criterion_with_source = int(
        _query_value(
            session,
            'MATCH (c:ReviewCriterion)-[:SUPPORTED_BY_SOURCE]->(:SourceDocument) RETURN count(DISTINCT c)',
        )
        or 0
    )
    criterion_with_evidence = int(
        _query_value(
            session,
            'MATCH (c:ReviewCriterion)-[:REQUIRES_EVIDENCE]->(:EvidenceRequirement) RETURN count(DISTINCT c)',
        )
        or 0
    )
    source_describes_count = int(
        _query_value(
            session,
            'MATCH (s:SourceDocument)-[:DESCRIBES]->() RETURN count(DISTINCT s)',
        )
        or 0
    )
    duplicate_criteria = [
        dict(row)
        for row in session.run(
            """
            MATCH (c:ReviewCriterion)
            WITH c.criterion_id AS id, count(c) AS count
            WHERE count > 1
            RETURN id, count
            ORDER BY id
            """
        )
    ]
    duplicate_sources = [
        dict(row)
        for row in session.run(
            """
            MATCH (s:SourceDocument)
            WITH s.source_id AS id, count(s) AS count
            WHERE count > 1
            RETURN id, count
            ORDER BY id
            """
        )
    ]

    property_completeness = (
        1.0
        if total_required_slots == 0
        else (total_required_slots - missing_required_slots) / total_required_slots
    )
    return {
        'node_counts': node_counts,
        'relation_counts': relation_counts,
        'total_nodes': sum(node_counts.values()),
        'total_relations': total_relations,
        'health': {
            'required_property_completeness': round(property_completeness, 4),
            'missing_required_properties': missing_by_label,
            'criteria_with_evidence_ratio': round(
                criterion_with_evidence / criterion_count if criterion_count else 0.0,
                4,
            ),
            'criteria_with_source_ratio': round(
                criterion_with_source / criterion_count if criterion_count else 0.0,
                4,
            ),
            'source_documents_describing_host_ratio': round(
                source_describes_count / source_count if source_count else 0.0,
                4,
            ),
            'duplicate_criterion_ids': duplicate_criteria,
            'duplicate_source_ids': duplicate_sources,
            'neo4j_reachable': True,
        },
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    snapshot = payload['graph_summary']
    counts = snapshot['node_counts']
    rels = snapshot['relation_counts']
    health = snapshot['health']
    return f"""# Knowledge Graph Snapshot

Generated: {payload['generated_at']}

This report intentionally contains no framework-based evaluation. It only shows the current local Neo4j knowledge graph snapshot and basic health checks.

## Graph Size

| Item | Count |
|---|---:|
| SourceDocument | {counts.get('SourceDocument', 0)} |
| Venue | {counts.get('Venue', 0)} |
| Journal | {counts.get('Journal', 0)} |
| ReviewCriterion | {counts.get('ReviewCriterion', 0)} |
| EvidenceRequirement | {counts.get('EvidenceRequirement', 0)} |
| ReviewFormField | {counts.get('ReviewFormField', 0)} |
| Domain | {counts.get('Domain', 0)} |
| ArticleType | {counts.get('ArticleType', 0)} |
| Total nodes | {snapshot.get('total_nodes', 0)} |
| Total relations | {snapshot.get('total_relations', 0)} |

## Graph Health

| Check | Result |
|---|---:|
| Required property completeness | {health['required_property_completeness']} |
| Criteria with evidence | {health['criteria_with_evidence_ratio']} |
| Criteria with source provenance | {health['criteria_with_source_ratio']} |
| Source documents describing host | {health['source_documents_describing_host_ratio']} |
| Duplicate criterion IDs | {len(health['duplicate_criterion_ids'])} |
| Duplicate source IDs | {len(health['duplicate_source_ids'])} |
| Neo4j reachable | {health['neo4j_reachable']} |

## Relation Coverage

| Relation | Count |
|---|---:|
{chr(10).join(f'| {rel} | {count} |' for rel, count in sorted(rels.items()))}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description='Summarize the local review-criteria KG')
    parser.add_argument('--output-json', type=Path, default=Path('outputs/kg_evaluation.json'))
    parser.add_argument('--output-md', type=Path, default=Path('docs/kg_evaluation_report.md'))
    args = parser.parse_args()

    driver, database = _neo4j_session()
    with driver.session(database=database) as session:
        summary = _graph_snapshot(session)
    driver.close()

    payload = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'graph_summary': summary,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    args.output_md.write_text(_markdown_report(payload), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
