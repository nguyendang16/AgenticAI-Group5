# Knowledge Graph Evaluation Report

Generated: 2026-06-04T08:39:38.159890+00:00

## Framework 1: Zaveri et al. Linked Data Quality Dimensions

Applied dimensions: completeness, consistency, provenance, conciseness, and availability.

### Graph Size

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

### Quality Results

| Dimension | Result |
|---|---:|
| Required property completeness | 1.0 |
| Criteria with evidence | 1.0 |
| Criteria with domain | 1.0 |
| Criteria with article type | 1.0 |
| Source documents describing host | 1.0 |
| Criteria with source provenance | 1.0 |
| Duplicate criterion IDs | 0 |
| Duplicate source IDs | 0 |
| Neo4j reachable | True |

### Relation Coverage

| Relation | Count |
|---|---:|
| ACCEPTS_ARTICLE_TYPE | 24 |
| BELONGS_TO_DOMAIN | 22 |
| DESCRIBES | 10 |
| HAS_CRITERION | 55 |
| HAS_REVIEW_FIELD | 51 |
| REQUIRES_EVIDENCE | 115 |
| SUPPORTED_BY_SOURCE | 55 |

## Framework 2: PyKEEN-Style Link Prediction

PyKEEN is installed locally. The report includes a real PyKEEN TransE run and the previous transparent typed tail-popularity baseline for comparison.

### Real PyKEEN Run

| Metric | Value |
|---|---:|
| PyKEEN version | 1.11.1 |
| Model | TransE |
| Epochs | 30 |
| Triples | 332 |
| Train triples | 265 |
| Test triples | 67 |
| MRR | 0.0394 |
| MR | 86.3881 |
| Hits@1 | 0.0 |
| Hits@3 | 0.0299 |
| Hits@10 | 0.1045 |

### Typed Popularity Baseline

| Metric | Value |
|---|---:|
| Triples | 332 |
| Train triples | 266 |
| Test triples | 66 |
| MRR | 0.0641 |
| MR | 50.0 |
| Hits@1 | 0.0 |
| Hits@3 | 0.0455 |
| Hits@10 | 0.2273 |

### Link Prediction By Relation

| Relation | Test Count | MRR | Hits@10 |
|---|---:|---:|---:|
| ACCEPTS_ARTICLE_TYPE | 4 | 0.0482 | 0.0 |
| BELONGS_TO_DOMAIN | 5 | 0.1682 | 0.4 |
| HAS_CRITERION | 15 | 0.0226 | 0.0 |
| HAS_REVIEW_FIELD | 10 | 0.0235 | 0.0 |
| REQUIRES_EVIDENCE | 19 | 0.0096 | 0.0 |
| SUPPORTED_BY_SOURCE | 13 | 0.1877 | 1.0 |

## Interpretation

The graph is strong as a curated retrieval knowledge graph: every criterion has evidence and source provenance, source documents are connected to venues/journals, and no duplicate source or criterion IDs are present.

The PyKEEN scores should be interpreted carefully. The graph is small and highly schema-driven, so link prediction is less important than quality, provenance, and retrieval correctness. A stronger experiment should compare TransE/DistMult/ComplEx on a larger graph with more venues and paper-derived nodes.
