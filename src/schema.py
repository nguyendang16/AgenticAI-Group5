from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class CriterionGroup(str, Enum):
    NOVELTY_CONTRIBUTION = 'NOVELTY_CONTRIBUTION'
    TECHNICAL_SOUNDNESS = 'TECHNICAL_SOUNDNESS'
    METHODOLOGICAL_RIGOR = 'METHODOLOGICAL_RIGOR'
    EVALUATION_VALIDITY = 'EVALUATION_VALIDITY'
    THEORETICAL_GROUNDING = 'THEORETICAL_GROUNDING'
    EDUCATIONAL_CONTRIBUTION = 'EDUCATIONAL_CONTRIBUTION'
    ETHICS_PRIVACY = 'ETHICS_PRIVACY'
    REPRODUCIBILITY_TRANSPARENCY = 'REPRODUCIBILITY_TRANSPARENCY'
    CLARITY_PRESENTATION = 'CLARITY_PRESENTATION'
    SCOPE_FIT = 'SCOPE_FIT'
    REVIEWER_CONFIDENCE = 'REVIEWER_CONFIDENCE'
    OTHER = 'OTHER'


class Severity(str, Enum):
    minor = 'minor'
    major = 'major'
    critical = 'critical'


class Confidence(str, Enum):
    low = 'low'
    medium = 'medium'
    high = 'high'


CRITERION_GROUPS = {item.value for item in CriterionGroup}
SEVERITIES = {item.value for item in Severity}
CONFIDENCES = {item.value for item in Confidence}

GROUP_ALIASES: dict[str, str] = {
    'NOVELTY': 'NOVELTY_CONTRIBUTION',
    'CONTRIBUTION': 'NOVELTY_CONTRIBUTION',
    'TECHNICAL': 'TECHNICAL_SOUNDNESS',
    'SOUNDNESS': 'TECHNICAL_SOUNDNESS',
    'METHODOLOGY': 'METHODOLOGICAL_RIGOR',
    'METHOD': 'METHODOLOGICAL_RIGOR',
    'EVALUATION': 'EVALUATION_VALIDITY',
    'EXPERIMENT': 'EVALUATION_VALIDITY',
    'THEORY': 'THEORETICAL_GROUNDING',
    'EDUCATION': 'EDUCATIONAL_CONTRIBUTION',
    'ETHICS': 'ETHICS_PRIVACY',
    'REPRODUCIBILITY': 'REPRODUCIBILITY_TRANSPARENCY',
    'CLARITY': 'CLARITY_PRESENTATION',
    'PRESENTATION': 'CLARITY_PRESENTATION',
    'WRITING': 'CLARITY_PRESENTATION',
    'SCOPE': 'SCOPE_FIT',
    'FIT': 'SCOPE_FIT',
    'CONFIDENCE': 'REVIEWER_CONFIDENCE',
}


class SourceDocument(BaseModel):
    source_id: str
    source_title: str
    source_type: str
    venue_or_journal_name: str = ''
    publisher: str = ''
    domain: list[str] = Field(default_factory=list)
    year: int | None = None
    source_url: str = ''
    article_types: list[str] = Field(default_factory=list)
    file_name: str


class EvidenceRequired(BaseModel):
    evidence_id: str
    name: str
    description: str = ''
    evidence_type: str = ''
    required: bool = True


class ReviewCriterion(BaseModel):
    criterion_id: str
    criterion_name: str
    criterion_group: str
    description: str
    applies_to_domain: list[str] = Field(default_factory=list)
    applies_to_article_type: list[str] = Field(default_factory=list)
    evidence_required: list[EvidenceRequired] = Field(default_factory=list)
    severity_if_missing: str = 'major'
    source_quote: str = ''
    confidence: str = 'medium'

    @field_validator('criterion_group', mode='before')
    @classmethod
    def normalize_group(cls, value: Any) -> str:
        token = str(value or '').strip().upper().replace(' ', '_').replace('-', '_')
        if token in CRITERION_GROUPS:
            return token
        if token in GROUP_ALIASES:
            return GROUP_ALIASES[token]
        return CriterionGroup.OTHER.value

    @field_validator('severity_if_missing', mode='before')
    @classmethod
    def normalize_severity(cls, value: Any) -> str:
        token = str(value or 'major').strip().lower()
        return token if token in SEVERITIES else Severity.major.value

    @field_validator('confidence', mode='before')
    @classmethod
    def normalize_confidence(cls, value: Any) -> str:
        token = str(value or 'medium').strip().lower()
        return token if token in CONFIDENCES else Confidence.medium.value


class ReviewFormField(BaseModel):
    field_id: str
    field_name: str
    field_type: str = 'text'
    scale_min: float | None = None
    scale_max: float | None = None
    description: str = ''
    required: bool = False
    maps_to_criteria: list[str] = Field(default_factory=list)


class ChecklistItem(BaseModel):
    item_id: str
    item_name: str
    checklist_group: str = ''
    description: str = ''
    required_evidence: list[str] = Field(default_factory=list)
    applies_to_method: list[str] = Field(default_factory=list)
    severity_if_missing: str = 'major'
    source_quote: str = ''

    @field_validator('severity_if_missing', mode='before')
    @classmethod
    def normalize_severity(cls, value: Any) -> str:
        token = str(value or 'major').strip().lower()
        return token if token in SEVERITIES else Severity.major.value


class GraphRelation(BaseModel):
    source_node_type: str
    source_node_id: str
    relation: str
    target_node_type: str
    target_node_id: str


class ExtractedTemplate(BaseModel):
    source_document: SourceDocument
    review_criteria: list[ReviewCriterion] = Field(default_factory=list)
    review_form_fields: list[ReviewFormField] = Field(default_factory=list)
    checklist_items: list[ChecklistItem] = Field(default_factory=list)
    graph_relations: list[GraphRelation] = Field(default_factory=list)
