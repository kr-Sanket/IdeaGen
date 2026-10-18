"""Domain models and DTOs for Phase 5 Research Opportunity Generation & Prior-Art Validation."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import List, Optional
import uuid
from pydantic import BaseModel, Field


class OpportunityStatus(str, enum.Enum):
    DRAFT = "DRAFT"
    VALIDATED = "VALIDATED"
    REFINED = "REFINED"
    REJECTED_INSUFFICIENTLY_DIFFERENTIATED = "REJECTED_INSUFFICIENTLY_DIFFERENTIATED"
    INSUFFICIENTLY_DIFFERENTIATED = "INSUFFICIENTLY_DIFFERENTIATED"


class FeasibilityAssessment(BaseModel):
    required_data: str = Field(description="Datasets required for experimentation")
    required_compute: str = Field(description="Compute resources needed (GPUs, memory, execution time)")
    implementation_complexity: str = Field(description="Algorithmic and engineering complexity assessment")
    experimental_setup: str = Field(description="Proposed baseline comparisons and evaluation setup")
    feasibility_score: float = Field(default=0.7, ge=0.0, le=1.0)
    summary: str = Field(default="Feasible under standard academic resources.")


class NoveltyAssessment(BaseModel):
    assessment: str = Field(
        default="Potentially novel within the analyzed corpus",
        description="Strictly bounded novelty claim restricted to analyzed corpus"
    )
    reasoning: str
    differentiation: str
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    overlapping_statement_ids: List[uuid.UUID] = Field(default_factory=list)
    overlapping_paper_ids: List[uuid.UUID] = Field(default_factory=list)


class PriorArtAssessment(BaseModel):
    assessment: str = Field(description="HIGH_OVERLAP, PARTIAL_OVERLAP, LOW_OVERLAP, or DIFFERENTIATED")
    overlapping_statement_ids: List[uuid.UUID] = Field(default_factory=list)
    overlapping_paper_ids: List[uuid.UUID] = Field(default_factory=list)
    reasoning: str
    differentiation: str
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class ResearchOpportunityInputDTO(BaseModel):
    gap_id: uuid.UUID
    gap_statement: str
    research_problem: str
    supporting_statement_ids: List[uuid.UUID] = Field(default_factory=list)
    supporting_paper_ids: List[uuid.UUID] = Field(default_factory=list)
    supporting_statements_text: List[str] = Field(default_factory=list)
    recurring_limitations: List[str] = Field(default_factory=list)
    future_work_statements: List[str] = Field(default_factory=list)
    existing_methods: List[str] = Field(default_factory=list)
    relevant_findings: List[str] = Field(default_factory=list)
    contradictory_evidence: List[str] = Field(default_factory=list)


class CandidateOpportunity(BaseModel):
    research_problem: str
    identified_gap: str
    proposed_research_direction: str
    reasoning: str
    technical_approach: str
    expected_contribution: str
    feasibility: FeasibilityAssessment
    supporting_statement_ids: List[uuid.UUID] = Field(default_factory=list)
    supporting_paper_ids: List[uuid.UUID] = Field(default_factory=list)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class RefinementCritique(BaseModel):
    critique: str
    revised_direction: str
    should_reject: bool = False
    reason: Optional[str] = None


class ResearchOpportunityDTO(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    gap_id: uuid.UUID
    research_problem: str
    identified_gap: str
    proposed_research_direction: str
    reasoning: str
    technical_approach: str
    expected_contribution: str
    feasibility: FeasibilityAssessment
    novelty_assessment: NoveltyAssessment
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    supporting_statement_ids: List[uuid.UUID] = Field(default_factory=list)
    supporting_paper_ids: List[uuid.UUID] = Field(default_factory=list)
    prior_art_statement_ids: List[uuid.UUID] = Field(default_factory=list)
    status: OpportunityStatus
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Reproducibility metadata
    generation_model: str = "gemini-2.5-flash"
    validation_model: str = "gemini-2.5-flash"
    embedding_model: str = "gemini-embedding-001"
    refinement_iteration_count: int = 0
    pipeline_version: str = "v1.0"
