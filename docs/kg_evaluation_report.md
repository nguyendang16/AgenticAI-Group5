# Knowledge Graph Snapshot

Generated: 2026-06-07T13:44:03.970207+00:00

This report intentionally contains no framework-based evaluation. It only shows the current local Neo4j knowledge graph snapshot and basic health checks.

## Graph Size

| Item | Count |
|---|---:|
| SourceDocument | 10 |
| Venue | 7 |
| Journal | 3 |
| ReviewCriterion | 55 |
| EvidenceRequirement | 115 |
| ReviewFormField | 51 |
| Domain | 15 |
| ArticleType | 24 |
| Total nodes | 280 |
| Total relations | 332 |

## Graph Health

| Check | Result |
|---|---:|
| Required property completeness | 1.0 |
| Criteria with evidence | 1.0 |
| Criteria with source provenance | 1.0 |
| Source documents describing host | 1.0 |
| Duplicate criterion IDs | 0 |
| Duplicate source IDs | 0 |
| Neo4j reachable | True |

## Relation Coverage

| Relation | Count |
|---|---:|
| ACCEPTS_ARTICLE_TYPE | 24 |
| BELONGS_TO_DOMAIN | 22 |
| DESCRIBES | 10 |
| HAS_CRITERION | 55 |
| HAS_REVIEW_FIELD | 51 |
| REQUIRES_EVIDENCE | 115 |
| SUPPORTED_BY_SOURCE | 55 |
