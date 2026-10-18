"""Domain models and DTOs for Semantic Gap Discovery (Phase 4).

Defines gap status, candidate models, validation schemas, and persistence DTOs.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class GapStatus(str, enum.Enum):
    """Lifecycle status of a discovered research gap."""
    CANDIDATE = "CANDIDATE"
    VALIDATED = "VALIDATED"
    REJECTED = "REJECTED"


class ValidationVerdict(str, enum.Enum):
    """LLM validation decision for a candidate research gap."""
    VALID = "VALID"
    REJECT = "REJECT"


class EvidenceRole(str, enum.Enum):
    """Role of an evidence statement relative to a research gap."""
    SUPPORTING = "SUPPORTING"
    LIMITATION = "LIMITATION"
    FUTURE_WORK = "FUTURE_WORK"
    COUNTER = "COUNTER"


class StatementEvidenceDTO(BaseModel):
    """Statement metadata used as evidence for a research gap."""
    model_config = ConfigDict(frozen=True)

    statement_id: UUID
    paper_id: UUID
    statement_type: str
    statement_text: str
    section: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    evidence_role: EvidenceRole = EvidenceRole.SUPPORTING


class GapCandidate(BaseModel):
    """Unvalidated research gap candidate derived from pattern analysis."""
    model_config = ConfigDict(extra="ignore")

    gap_statement: str = Field(description="Clear statement of the discovered research gap.")
    research_problem: str = Field(description="The underlying research challenge or problem domain.")
    why_existing_work_insufficient: str = Field(description="Explanation of why current methodologies or findings fall short.")
    supporting_statement_ids: List[UUID] = Field(default_factory=list, description="IDs of statements providing evidence.")
    supporting_paper_ids: List[UUID] = Field(default_factory=list, description="IDs of papers contributing evidence.")
    relevant_limitations: List[str] = Field(default_factory=list, description="Explicit limitations supporting this gap.")
    relevant_future_work: List[str] = Field(default_factory=list, description="Future work directions pointing to this gap.")
    signals: Dict[str, Any] = Field(default_factory=dict, description="Pattern signals (e.g. recurrence_count, cross_paper_evidence).")


class GapValidationResult(BaseModel):
    """Structured output from LLM gap validation."""
    model_config = ConfigDict(extra="ignore")

    verdict: ValidationVerdict = Field(description="Whether the candidate is supported by evidence (VALID) or rejected (REJECT).")
    reasoning: str = Field(description="Detailed scientific rationale for the validation verdict.")
    supporting_evidence_statement_ids: List[str] = Field(default_factory=list, description="UUIDs of verified supporting statements.")
    counter_evidence_statement_ids: List[str] = Field(default_factory=list, description="UUIDs of counter-evidence statements if any.")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence in the validation verdict.")


class ResearchGapCreateDTO(BaseModel):
    """DTO for persisting a research gap with verified evidence."""
    model_config = ConfigDict(frozen=True)

    gap_statement: str
    research_problem: str
    status: GapStatus
    supporting_statement_ids: List[UUID]
    supporting_paper_ids: List[UUID]
    evidence_count: int
    limitations: List[str] = Field(default_factory=list)
    future_work: List[str] = Field(default_factory=list)
    validation_reasoning: Optional[str] = None
    validation_confidence: Optional[float] = None
    evidence_items: List[StatementEvidenceDTO] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ResearchGapResponseDTO(BaseModel):
    """DTO returned by APIs representing a discovered research gap."""
    id: UUID
    gap_statement: str
    research_problem: str
    status: GapStatus
    evidence_count: int
    supporting_statement_ids: List[UUID]
    supporting_paper_ids: List[UUID]
    limitations: List[str]
    future_work: List[str]
    validation_reasoning: Optional[str]
    validation_confidence: Optional[float]
    created_at: datetime
    metadata: Dict[str, Any]
