from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PaperRecord:
    paper_id: str
    venue: str
    title: str
    pdf_path: str
    source: str
    year: int | None = None
    expected_decision: str | None = None
    expected_score: float | None = None
    metadata_source: str = 'local'

    def to_dict(self) -> dict[str, Any]:
        return {
            'paper_id': self.paper_id,
            'venue': self.venue,
            'title': self.title,
            'pdf_path': self.pdf_path,
            'source': self.source,
            'year': self.year,
            'expected_decision': self.expected_decision,
            'expected_score': self.expected_score,
            'metadata_source': self.metadata_source,
        }


@dataclass
class RunRecord:
    paper_id: str
    venue: str
    condition: str  # KG_ON | KG_OFF
    job_id: str
    status: str
    git_commit: str
    env_snapshot: dict[str, str] = field(default_factory=dict)
