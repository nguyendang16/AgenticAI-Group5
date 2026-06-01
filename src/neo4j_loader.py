from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

from src.config import get_pipeline_settings
from src.schema import ExtractedTemplate


CONSTRAINTS = [
    'CREATE CONSTRAINT source_document_id IF NOT EXISTS FOR (n:SourceDocument) REQUIRE n.source_id IS UNIQUE',
    'CREATE CONSTRAINT venue_id IF NOT EXISTS FOR (n:Venue) REQUIRE n.venue_id IS UNIQUE',
    'CREATE CONSTRAINT journal_id IF NOT EXISTS FOR (n:Journal) REQUIRE n.journal_id IS UNIQUE',
    'CREATE CONSTRAINT domain_name IF NOT EXISTS FOR (n:Domain) REQUIRE n.name IS UNIQUE',
    'CREATE CONSTRAINT article_type_name IF NOT EXISTS FOR (n:ArticleType) REQUIRE n.name IS UNIQUE',
    'CREATE CONSTRAINT criterion_id IF NOT EXISTS FOR (n:ReviewCriterion) REQUIRE n.criterion_id IS UNIQUE',
    'CREATE CONSTRAINT evidence_id IF NOT EXISTS FOR (n:EvidenceRequirement) REQUIRE n.evidence_id IS UNIQUE',
    'CREATE CONSTRAINT form_field_id IF NOT EXISTS FOR (n:ReviewFormField) REQUIRE n.field_id IS UNIQUE',
    'CREATE CONSTRAINT checklist_item_id IF NOT EXISTS FOR (n:ChecklistItem) REQUIRE n.item_id IS UNIQUE',
]


def _slug(text: str) -> str:
    import re

    return re.sub(r'[^a-zA-Z0-9]+', '_', str(text or '').lower()).strip('_') or 'unknown'


def load_json_templates(json_dir: Path) -> list[ExtractedTemplate]:
    templates: list[ExtractedTemplate] = []
    for path in sorted(json_dir.glob('*.json')):
        payload = json.loads(path.read_text(encoding='utf-8'))
        templates.append(ExtractedTemplate.model_validate(payload))
    return templates


def _is_conference(source_type: str) -> bool:
    token = source_type.lower()
    return any(x in token for x in ('conference', 'cfp', 'review_form', 'reviewer'))


def load_template_to_neo4j(session: Any, template: ExtractedTemplate) -> None:
    sd = template.source_document
    venue_slug = _slug(sd.venue_or_journal_name)
    host_label = 'Venue' if _is_conference(sd.source_type) else 'Journal'
    host_id_field = 'venue_id' if host_label == 'Venue' else 'journal_id'

    session.run(
        """
        MERGE (d:SourceDocument {source_id: $source_id})
        SET d.source_title = $source_title,
            d.source_type = $source_type,
            d.file_name = $file_name,
            d.year = $year,
            d.source_url = $source_url,
            d.publisher = $publisher
        """,
        source_id=sd.source_id,
        source_title=sd.source_title,
        source_type=sd.source_type,
        file_name=sd.file_name,
        year=sd.year,
        source_url=sd.source_url,
        publisher=sd.publisher,
    )

    if host_label == 'Venue':
        session.run(
            """
            MERGE (v:Venue {venue_id: $host_id})
            SET v.name = $name
            WITH v
            MATCH (d:SourceDocument {source_id: $source_id})
            MERGE (d)-[:DESCRIBES]->(v)
            """,
            host_id=venue_slug,
            name=sd.venue_or_journal_name,
            source_id=sd.source_id,
        )
    else:
        session.run(
            """
            MERGE (j:Journal {journal_id: $host_id})
            SET j.name = $name
            WITH j
            MATCH (d:SourceDocument {source_id: $source_id})
            MERGE (d)-[:DESCRIBES]->(j)
            """,
            host_id=venue_slug,
            name=sd.venue_or_journal_name,
            source_id=sd.source_id,
        )

    for domain_name in sd.domain:
        session.run(
            f"""
            MERGE (dom:Domain {{name: $name}})
            WITH dom
            MATCH (h:{host_label} {{{host_id_field}: $host_id}})
            MERGE (h)-[:BELONGS_TO_DOMAIN]->(dom)
            """,
            name=domain_name,
            host_id=venue_slug,
        )

    for article_type in sd.article_types:
        session.run(
            f"""
            MERGE (at:ArticleType {{name: $name}})
            WITH at
            MATCH (h:{host_label} {{{host_id_field}: $host_id}})
            MERGE (h)-[:ACCEPTS_ARTICLE_TYPE]->(at)
            """,
            name=article_type,
            host_id=venue_slug,
        )

    for criterion in template.review_criteria:
        session.run(
            f"""
            MERGE (c:ReviewCriterion {{criterion_id: $criterion_id}})
            SET c.criterion_name = $criterion_name,
                c.criterion_group = $criterion_group,
                c.description = $description,
                c.severity_if_missing = $severity_if_missing,
                c.source_quote = $source_quote,
                c.confidence = $confidence,
                c.applies_to_domain = $applies_to_domain,
                c.applies_to_article_type = $applies_to_article_type
            WITH c
            MATCH (h:{host_label} {{{host_id_field}: $host_id}})
            MERGE (h)-[:HAS_CRITERION]->(c)
            WITH c
            MATCH (d:SourceDocument {{source_id: $source_id}})
            MERGE (c)-[:SUPPORTED_BY_SOURCE]->(d)
            """,
            criterion_id=criterion.criterion_id,
            criterion_name=criterion.criterion_name,
            criterion_group=criterion.criterion_group,
            description=criterion.description,
            severity_if_missing=criterion.severity_if_missing,
            source_quote=criterion.source_quote,
            confidence=criterion.confidence,
            applies_to_domain=criterion.applies_to_domain,
            applies_to_article_type=criterion.applies_to_article_type,
            host_id=venue_slug,
            source_id=sd.source_id,
        )
        for ev in criterion.evidence_required:
            session.run(
                """
                MERGE (e:EvidenceRequirement {evidence_id: $evidence_id})
                SET e.name = $name,
                    e.description = $description,
                    e.evidence_type = $evidence_type,
                    e.required = $required
                WITH e
                MATCH (c:ReviewCriterion {criterion_id: $criterion_id})
                MERGE (c)-[:REQUIRES_EVIDENCE]->(e)
                """,
                evidence_id=ev.evidence_id,
                name=ev.name,
                description=ev.description,
                evidence_type=ev.evidence_type,
                required=ev.required,
                criterion_id=criterion.criterion_id,
            )

    for field in template.review_form_fields:
        session.run(
            f"""
            MERGE (f:ReviewFormField {{field_id: $field_id}})
            SET f.field_name = $field_name,
                f.field_type = $field_type,
                f.scale_min = $scale_min,
                f.scale_max = $scale_max,
                f.description = $description,
                f.required = $required,
                f.maps_to_criteria = $maps_to_criteria
            WITH f
            MATCH (h:{host_label} {{{host_id_field}: $host_id}})
            MERGE (h)-[:HAS_REVIEW_FIELD]->(f)
            """,
            field_id=field.field_id,
            field_name=field.field_name,
            field_type=field.field_type,
            scale_min=field.scale_min,
            scale_max=field.scale_max,
            description=field.description,
            required=field.required,
            maps_to_criteria=field.maps_to_criteria,
            host_id=venue_slug,
        )

    for item in template.checklist_items:
        session.run(
            """
            MERGE (ci:ChecklistItem {item_id: $item_id})
            SET ci.item_name = $item_name,
                ci.checklist_group = $checklist_group,
                ci.description = $description,
                ci.required_evidence = $required_evidence,
                ci.applies_to_method = $applies_to_method,
                ci.severity_if_missing = $severity_if_missing,
                ci.source_quote = $source_quote
            WITH ci
            MATCH (d:SourceDocument {source_id: $source_id})
            MERGE (ci)-[:SUPPORTED_BY_SOURCE]->(d)
            """,
            item_id=item.item_id,
            item_name=item.item_name,
            checklist_group=item.checklist_group,
            description=item.description,
            required_evidence=item.required_evidence,
            applies_to_method=item.applies_to_method,
            severity_if_missing=item.severity_if_missing,
            source_quote=item.source_quote,
            source_id=sd.source_id,
        )


def load_all_templates(json_dir: Path, *, clear: bool = False) -> int:
    settings = get_pipeline_settings()
    if not settings.neo4j_uri or not settings.neo4j_password:
        raise RuntimeError('NEO4J_URI and NEO4J_PASSWORD are required')

    templates = load_json_templates(json_dir)
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    database = settings.neo4j_database or 'neo4j'
    count = 0
    with driver.session(database=database) as session:
        if clear:
            session.run('MATCH (n) DETACH DELETE n')
        for statement in CONSTRAINTS:
            session.run(statement)
        for template in templates:
            load_template_to_neo4j(session, template)
            count += 1
    driver.close()
    return count
