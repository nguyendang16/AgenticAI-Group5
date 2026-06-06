# Knowledge Graph Benchmark

This benchmark evaluates the local Neo4j knowledge graph for review criteria using two complementary frameworks:

1. Xu, Gao & Yu AI-based KG quality evaluation model from IEEE ICNLP 2021
2. PyKEEN link-prediction evaluation

## Graph Snapshot

| Node / Relation | Count |
|---|---:|
| SourceDocument | 10 |
| Venue | 7 |
| Journal | 3 |
| ReviewCriterion | 55 |
| EvidenceRequirement | 115 |
| ReviewFormField | 51 |
| Domain | 15 |
| ArticleType | 24 |
| HAS_CRITERION | 55 |
| REQUIRES_EVIDENCE | 115 |
| SUPPORTED_BY_SOURCE | 55 |

## Framework 1: AI-Based KG Quality Evaluation Model

Reference: Xu, Gao & Yu, *Quality Evaluation Model of AI-based Knowledge Graph System*, 2021 3rd International Conference on Natural Language Processing (ICNLP), IEEE, pp. 73-78.

This framework is suitable because it evaluates a knowledge graph as part of an AI-based system, not only as a collection of triples. In this project, the KG supports review-criteria retrieval and grounding for the review agent. Therefore, we evaluate whether the KG has complete criterion fields, evidence requirements, domain/article applicability, source provenance, duplicate consistency, and Neo4j availability.

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

Interpretation: the KG is strong as a curated retrieval and provenance graph. Every review criterion has required properties, evidence links, domain/article metadata, and source provenance.

## Framework 2: PyKEEN Link Prediction

Reference: Ali et al., *PyKEEN 1.0: A Python Library for Training and Evaluating Knowledge Graph Embeddings*.

### Real PyKEEN Model

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

### Baseline Comparison

Baseline: typed tail-popularity baseline.

The baseline does not learn embeddings. It ranks candidate tail nodes by how frequently they appeared with the same relation type in the training graph.

| Method | MRR | MR | Hits@1 | Hits@3 | Hits@10 |
|---|---:|---:|---:|---:|---:|
| TransE | 0.0394 | 86.3881 | 0.0 | 0.0299 | 0.1045 |
| Typed tail-popularity baseline | 0.0641 | 50.0 | 0.0 | 0.0455 | 0.2273 |

Interpretation: the simple baseline performs better than TransE. This means the current KG is too small and schema-driven for embedding-based graph completion to be very useful. The KG is currently better evaluated through quality, provenance, and retrieval correctness.

## Conclusion

The current KG is suitable for the review system because it is complete, consistent, provenance-backed, and queryable from Neo4j. However, it is not yet a strong graph-completion benchmark. To improve PyKEEN performance, the graph should include more source documents, more venues/journals, paper-derived nodes, claim/evidence nodes, and richer cross-document relations.

## References

- Xu, Z., Gao, Y., & Yu, F. (2021). *Quality Evaluation Model of AI-based Knowledge Graph System*. 2021 3rd International Conference on Natural Language Processing (ICNLP), IEEE, pp. 73-78. https://ieeexplore.ieee.org/document/9537861
- Ali, M., et al. (2021). *PyKEEN 1.0: A Python Library for Training and Evaluating Knowledge Graph Embeddings*. JMLR. https://www.jmlr.org/papers/v22/20-825.html
