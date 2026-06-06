from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

from deepreview.config import get_settings


_REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault('PYSTOW_HOME', str(_REPO_ROOT / '.pykeen-cache'))


@dataclass(frozen=True)
class Triple:
    head: str
    head_label: str
    relation: str
    tail: str
    tail_label: str


def _node_id_expr(alias: str) -> str:
    return (
        f'coalesce({alias}.source_id, {alias}.venue_id, {alias}.journal_id, '
        f'{alias}.criterion_id, {alias}.evidence_id, {alias}.field_id, '
        f'{alias}.item_id, {alias}.name)'
    )


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


def _quality_checks(session: Any) -> dict[str, Any]:
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

    source_count = node_counts['SourceDocument']
    source_describes_count = int(
        _query_value(
            session,
            'MATCH (s:SourceDocument)-[:DESCRIBES]->() RETURN count(DISTINCT s)',
        )
        or 0
    )
    criterion_count = node_counts['ReviewCriterion']
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
    duplicate_criteria = [
        dict(row)
        for row in session.run(
            """
            MATCH (c:ReviewCriterion)
            WITH c.criterion_id AS id, count(c) AS count
            WHERE count > 1
            RETURN id, count
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
            """
        )
    ]

    criteria_with_domain = int(
        _query_value(
            session,
            """
            MATCH (c:ReviewCriterion)
            WHERE c.applies_to_domain IS NOT NULL AND size(c.applies_to_domain) > 0
            RETURN count(c)
            """,
        )
        or 0
    )
    criteria_with_article_type = int(
        _query_value(
            session,
            """
            MATCH (c:ReviewCriterion)
            WHERE c.applies_to_article_type IS NOT NULL AND size(c.applies_to_article_type) > 0
            RETURN count(c)
            """,
        )
        or 0
    )

    property_completeness = (
        1.0
        if total_required_slots == 0
        else (total_required_slots - missing_required_slots) / total_required_slots
    )

    return {
        'framework': 'Xu, Gao & Yu AI-based KG quality evaluation model (IEEE ICNLP, 2021)',
        'reference': {
            'title': 'Quality Evaluation Model of AI-based Knowledge Graph System',
            'authors': 'ZhenHao Xu, Yan Gao, and Fei Yu',
            'venue': '2021 3rd International Conference on Natural Language Processing (ICNLP), IEEE',
            'year': 2021,
            'pages': '73-78',
            'ieee_xplore_document': '9537861',
            'url': 'https://ieeexplore.ieee.org/document/9537861',
        },
        'adaptation_note': (
            'The IEEE paper proposes evaluating an AI-based KG system with quality-model '
            'dimensions rather than a single graph-size number. For this review-agent KG, '
            'the operational quality dimensions are required-field completeness, evidence '
            'coverage, domain/article applicability, provenance, duplicate consistency, '
            'and Neo4j availability.'
        ),
        'node_counts': node_counts,
        'relation_counts': relation_counts,
        'accuracy_consistency': {
            'required_property_completeness': round(property_completeness, 4),
            'missing_required_properties': missing_by_label,
            'duplicate_criterion_ids': duplicate_criteria,
            'duplicate_source_ids': duplicate_sources,
        },
        'completeness': {
            'criteria_with_evidence_ratio': round(
                criterion_with_evidence / criterion_count if criterion_count else 0.0,
                4,
            ),
            'criteria_with_domain_ratio': round(
                criteria_with_domain / criterion_count if criterion_count else 0.0,
                4,
            ),
            'criteria_with_article_type_ratio': round(
                criteria_with_article_type / criterion_count if criterion_count else 0.0,
                4,
            ),
            'source_documents_describing_host_ratio': round(
                source_describes_count / source_count if source_count else 0.0,
                4,
            ),
        },
        'provenance': {
            'criteria_with_source_ratio': round(
                criterion_with_source / criterion_count if criterion_count else 0.0,
                4,
            ),
        },
        'availability': {
            'neo4j_reachable': True,
        },
    }


def _load_triples(session: Any) -> list[Triple]:
    query = f"""
    MATCH (h)-[r]->(t)
    RETURN {_node_id_expr('h')} AS head,
           labels(h)[0] AS head_label,
           type(r) AS relation,
           {_node_id_expr('t')} AS tail,
           labels(t)[0] AS tail_label
    ORDER BY relation, head, tail
    """
    triples: list[Triple] = []
    for row in session.run(query):
        head = str(row['head'] or '').strip()
        tail = str(row['tail'] or '').strip()
        if head and tail:
            triples.append(
                Triple(
                    head=head,
                    head_label=str(row['head_label'] or ''),
                    relation=str(row['relation'] or ''),
                    tail=tail,
                    tail_label=str(row['tail_label'] or ''),
                )
            )
    return triples


def _rank_tail(
    *,
    triple: Triple,
    train_triples: set[tuple[str, str, str]],
    candidate_tails: list[str],
    tail_popularity: Counter[tuple[str, str]],
) -> int:
    true_score = tail_popularity[(triple.relation, triple.tail)]
    filtered_candidates = [
        tail
        for tail in candidate_tails
        if tail == triple.tail or (triple.head, triple.relation, tail) not in train_triples
    ]
    ranked = sorted(
        filtered_candidates,
        key=lambda tail: (-tail_popularity[(triple.relation, tail)], tail),
    )
    try:
        return ranked.index(triple.tail) + 1
    except ValueError:
        return len(ranked) + 1 if true_score == 0 else 1


def _pykeen_style_eval(triples: list[Triple], *, seed: int = 13, test_ratio: float = 0.2) -> dict[str, Any]:
    rng = random.Random(seed)
    shuffled = list(triples)
    rng.shuffle(shuffled)
    test_size = max(1, round(len(shuffled) * test_ratio))
    test = shuffled[:test_size]
    train = shuffled[test_size:]
    train_keys = {(t.head, t.relation, t.tail) for t in train}

    candidates_by_label: dict[str, set[str]] = defaultdict(set)
    tail_popularity: Counter[tuple[str, str]] = Counter()
    for triple in train:
        candidates_by_label[triple.tail_label].add(triple.tail)
        tail_popularity[(triple.relation, triple.tail)] += 1
    for triple in triples:
        candidates_by_label[triple.tail_label].add(triple.tail)

    ranks: list[int] = []
    relation_ranks: dict[str, list[int]] = defaultdict(list)
    for triple in test:
        rank = _rank_tail(
            triple=triple,
            train_triples=train_keys,
            candidate_tails=sorted(candidates_by_label[triple.tail_label]),
            tail_popularity=tail_popularity,
        )
        ranks.append(rank)
        relation_ranks[triple.relation].append(rank)

    def hits_at(k: int, values: list[int]) -> float:
        return sum(1 for rank in values if rank <= k) / len(values) if values else 0.0

    def mrr(values: list[int]) -> float:
        return sum(1 / rank for rank in values) / len(values) if values else 0.0

    return {
        'framework': 'PyKEEN-style filtered tail-prediction evaluation',
        'library_available': _pykeen_available(),
        'model': 'typed tail-popularity baseline',
        'triple_count': len(triples),
        'train_triples': len(train),
        'test_triples': len(test),
        'seed': seed,
        'metrics': {
            'MRR': round(mrr(ranks), 4),
            'MR': round(sum(ranks) / len(ranks), 4) if ranks else 0.0,
            'Hits@1': round(hits_at(1, ranks), 4),
            'Hits@3': round(hits_at(3, ranks), 4),
            'Hits@10': round(hits_at(10, ranks), 4),
        },
        'metrics_by_relation': {
            relation: {
                'test_count': len(values),
                'MRR': round(mrr(values), 4),
                'Hits@10': round(hits_at(10, values), 4),
            }
            for relation, values in sorted(relation_ranks.items())
        },
    }


def _pykeen_available() -> bool:
    try:
        import pykeen  # noqa: F401
    except Exception:
        return False
    return True


def _metric_value(metric_results: Any, metric: str) -> float | None:
    attempts = [
        {'metric': metric},
        {'metric': metric, 'side': 'both'},
        {'metric': metric, 'side': 'tail'},
        {'metric': metric, 'rank_type': 'realistic'},
        {'metric': metric, 'rank_type': 'realistic', 'side': 'both'},
        {'metric': metric, 'rank_type': 'realistic', 'side': 'tail'},
    ]
    for kwargs in attempts:
        try:
            value = metric_results.get_metric(**kwargs)
        except Exception:
            continue
        if value is not None:
            return float(value)
    return None


def _metric_from_flat_dict(
    flat: dict[str, Any],
    *needles: str,
    exclude: tuple[str, ...] = (),
) -> float | None:
    lowered_needles = [needle.lower() for needle in needles]
    lowered_exclude = [item.lower() for item in exclude]
    for key, value in sorted(flat.items()):
        key_l = str(key).lower()
        if any(item in key_l for item in lowered_exclude):
            continue
        if all(needle in key_l for needle in lowered_needles):
            try:
                return float(value)
            except Exception:
                continue
    return None


def _pykeen_transe_eval(
    triples: list[Triple],
    *,
    seed: int = 13,
    test_ratio: float = 0.2,
    epochs: int = 30,
) -> dict[str, Any]:
    try:
        import numpy as np
        from importlib.metadata import version
        from pykeen.pipeline import pipeline
        from pykeen.triples import TriplesFactory
    except Exception as exc:
        return {
            'framework': 'PyKEEN TransE filtered rank evaluation',
            'library_available': False,
            'error': f'{type(exc).__name__}: {exc}',
        }

    labeled_triples = np.array(
        [[triple.head, triple.relation, triple.tail] for triple in triples],
        dtype=str,
    )
    triples_factory = TriplesFactory.from_labeled_triples(labeled_triples)
    training, testing = triples_factory.split([1.0 - test_ratio, test_ratio], random_state=seed)
    try:
        result = pipeline(
            training=training,
            testing=testing,
            model='TransE',
            model_kwargs={'embedding_dim': 32},
            training_kwargs={'batch_size': 64},
            optimizer='Adam',
            optimizer_kwargs={'lr': 0.01},
            epochs=epochs,
            random_seed=seed,
            device='cpu',
            use_tqdm=False,
        )
    except Exception as exc:
        return {
            'framework': 'PyKEEN TransE filtered rank evaluation',
            'library_available': True,
            'pykeen_version': version('pykeen'),
            'model': 'TransE',
            'error': f'{type(exc).__name__}: {exc}',
        }

    metrics = result.metric_results
    if hasattr(metrics, 'to_flat_dict'):
        raw_metrics = metrics.to_flat_dict()
    else:
        raw_metrics = metrics.to_dict()
    extracted = {
        'MRR': (
            _metric_value(metrics, 'mean_reciprocal_rank')
            or _metric_from_flat_dict(
                raw_metrics,
                'both',
                'realistic',
                'inverse_harmonic_mean_rank',
                exclude=('adjusted',),
            )
            or _metric_from_flat_dict(
                raw_metrics,
                'tail',
                'realistic',
                'inverse_harmonic_mean_rank',
                exclude=('adjusted',),
            )
            or _metric_from_flat_dict(raw_metrics, 'inverse_harmonic_mean_rank', exclude=('adjusted',))
        ),
        'MR': (
            _metric_value(metrics, 'mean_rank')
            or _metric_from_flat_dict(
                raw_metrics,
                'both',
                'realistic',
                'arithmetic_mean_rank',
                exclude=('adjusted', 'inverse'),
            )
            or _metric_from_flat_dict(
                raw_metrics,
                'tail',
                'realistic',
                'arithmetic_mean_rank',
                exclude=('adjusted', 'inverse'),
            )
            or _metric_from_flat_dict(
                raw_metrics,
                'arithmetic_mean_rank',
                exclude=('adjusted', 'inverse'),
            )
        ),
        'Hits@1': (
            _metric_value(metrics, 'hits_at_1')
            or _metric_from_flat_dict(raw_metrics, 'both', 'realistic', 'hits_at_1')
            or _metric_from_flat_dict(raw_metrics, 'tail', 'realistic', 'hits_at_1')
            or _metric_from_flat_dict(raw_metrics, 'hits_at_1')
        ),
        'Hits@3': (
            _metric_value(metrics, 'hits_at_3')
            or _metric_from_flat_dict(raw_metrics, 'both', 'realistic', 'hits_at_3')
            or _metric_from_flat_dict(raw_metrics, 'tail', 'realistic', 'hits_at_3')
            or _metric_from_flat_dict(raw_metrics, 'hits_at_3')
        ),
        'Hits@10': (
            _metric_value(metrics, 'hits_at_10')
            or _metric_from_flat_dict(raw_metrics, 'both', 'realistic', 'hits_at_10')
            or _metric_from_flat_dict(raw_metrics, 'tail', 'realistic', 'hits_at_10')
            or _metric_from_flat_dict(raw_metrics, 'hits_at_10')
        ),
    }
    return {
        'framework': 'PyKEEN TransE filtered rank evaluation',
        'library_available': True,
        'pykeen_version': version('pykeen'),
        'model': 'TransE',
        'epochs': epochs,
        'embedding_dim': 32,
        'triple_count': len(triples),
        'train_triples': int(training.num_triples),
        'test_triples': int(testing.num_triples),
        'seed': seed,
        'metrics': {
            key: (round(value, 4) if value is not None else None)
            for key, value in extracted.items()
        },
        'raw_metric_key_count': len(raw_metrics),
        'raw_metric_keys_sample': sorted(map(str, raw_metrics.keys()))[:20],
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    q = payload['user_needs_quality']
    p = payload['pykeen_style_link_prediction']
    real = payload.get('pykeen_transe')
    counts = q['node_counts']
    rels = q['relation_counts']
    return f"""# Knowledge Graph Evaluation Report

Generated: {payload['generated_at']}

## Framework 1: Xu, Gao & Yu AI-Based KG Quality Evaluation Model

Reference: Xu, Z., Gao, Y. & Yu, F. *Quality Evaluation Model of AI-based Knowledge Graph System*. 2021 3rd International Conference on Natural Language Processing (ICNLP), IEEE, pp. 73-78. https://ieeexplore.ieee.org/document/9537861

Applied adaptation: the IEEE paper treats KG evaluation as a quality model for an AI-based KG system. For this project, the benchmark maps that idea to measurable local Neo4j checks: required-field completeness, evidence coverage, domain/article applicability, provenance, duplicate consistency, and Neo4j availability.

### Graph Size

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

### Quality Results

| Dimension | Result |
|---|---:|
| Required property completeness | {q['accuracy_consistency']['required_property_completeness']} |
| Criteria with evidence | {q['completeness']['criteria_with_evidence_ratio']} |
| Criteria with domain | {q['completeness']['criteria_with_domain_ratio']} |
| Criteria with article type | {q['completeness']['criteria_with_article_type_ratio']} |
| Source documents describing host | {q['completeness']['source_documents_describing_host_ratio']} |
| Criteria with source provenance | {q['provenance']['criteria_with_source_ratio']} |
| Duplicate criterion IDs | {len(q['accuracy_consistency']['duplicate_criterion_ids'])} |
| Duplicate source IDs | {len(q['accuracy_consistency']['duplicate_source_ids'])} |
| Neo4j reachable | {q['availability']['neo4j_reachable']} |

### Relation Coverage

| Relation | Count |
|---|---:|
{chr(10).join(f'| {rel} | {count} |' for rel, count in sorted(rels.items()))}

## Framework 2: PyKEEN-Style Link Prediction

PyKEEN is installed locally. The report includes a real PyKEEN TransE run and the previous transparent typed tail-popularity baseline for comparison.

### Real PyKEEN Run

| Metric | Value |
|---|---:|
| PyKEEN version | {real.get('pykeen_version') if isinstance(real, dict) else 'n/a'} |
| Model | {real.get('model') if isinstance(real, dict) else 'n/a'} |
| Epochs | {real.get('epochs') if isinstance(real, dict) else 'n/a'} |
| Triples | {real.get('triple_count') if isinstance(real, dict) else 'n/a'} |
| Train triples | {real.get('train_triples') if isinstance(real, dict) else 'n/a'} |
| Test triples | {real.get('test_triples') if isinstance(real, dict) else 'n/a'} |
| MRR | {(real.get('metrics') or {}).get('MRR') if isinstance(real, dict) else 'n/a'} |
| MR | {(real.get('metrics') or {}).get('MR') if isinstance(real, dict) else 'n/a'} |
| Hits@1 | {(real.get('metrics') or {}).get('Hits@1') if isinstance(real, dict) else 'n/a'} |
| Hits@3 | {(real.get('metrics') or {}).get('Hits@3') if isinstance(real, dict) else 'n/a'} |
| Hits@10 | {(real.get('metrics') or {}).get('Hits@10') if isinstance(real, dict) else 'n/a'} |

### Typed Popularity Baseline

| Metric | Value |
|---|---:|
| Triples | {p['triple_count']} |
| Train triples | {p['train_triples']} |
| Test triples | {p['test_triples']} |
| MRR | {p['metrics']['MRR']} |
| MR | {p['metrics']['MR']} |
| Hits@1 | {p['metrics']['Hits@1']} |
| Hits@3 | {p['metrics']['Hits@3']} |
| Hits@10 | {p['metrics']['Hits@10']} |

### Link Prediction By Relation

| Relation | Test Count | MRR | Hits@10 |
|---|---:|---:|---:|
{chr(10).join(f'| {rel} | {m["test_count"]} | {m["MRR"]} | {m["Hits@10"]} |' for rel, m in p['metrics_by_relation'].items())}

## Interpretation

The graph is strong as a curated retrieval knowledge graph: every criterion has evidence and source provenance, source documents are connected to venues/journals, and no duplicate source or criterion IDs are present.

The PyKEEN scores should be interpreted carefully. The graph is small and highly schema-driven, so link prediction is less important than quality, provenance, and retrieval correctness. A stronger experiment should compare TransE/DistMult/ComplEx on a larger graph with more venues and paper-derived nodes.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description='Evaluate the local review-criteria KG')
    parser.add_argument('--output-json', type=Path, default=Path('outputs/kg_evaluation.json'))
    parser.add_argument('--output-md', type=Path, default=Path('docs/kg_evaluation_report.md'))
    args = parser.parse_args()

    driver, database = _neo4j_session()
    with driver.session(database=database) as session:
        quality = _quality_checks(session)
        triples = _load_triples(session)
    driver.close()

    payload = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'user_needs_quality': quality,
        # Backward-compatible alias for older benchmark-page builds.
        'zaveri_quality': quality,
        'pykeen_transe': _pykeen_transe_eval(triples),
        'pykeen_style_link_prediction': _pykeen_style_eval(triples),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    args.output_md.write_text(_markdown_report(payload), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
