import enum
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field

class OpportunityStatus(str, enum.Enum):
    OPEN = "OPEN"
    STRENGTHENED = "STRENGTHENED"
    PARTIALLY_ADDRESSED = "PARTIALLY_ADDRESSED"
    RESOLVED = "RESOLVED"

class OpportunityRelationship(str, enum.Enum):
    SUPPORTS = "SUPPORTS"
    ADDRESSES = "ADDRESSES"
    CONTRADICTS = "CONTRADICTS"
    EXTENDS = "EXTENDS"
    UNRELATED = "UNRELATED"

class OpportunityCreateDTO(BaseModel):
    title: str
    problem_statement: str
    description: str
    status: OpportunityStatus = OpportunityStatus.OPEN
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    research_opportunity_id: Optional[UUID] = None
    gap_id: Optional[UUID] = None

class OpportunityEvidenceCreateDTO(BaseModel):
    opportunity_id: UUID
    paper_id: UUID
    evidence_type: OpportunityRelationship
    supporting_text: Optional[str] = None

class OpportunityHistoryCreateDTO(BaseModel):
    opportunity_id: UUID
    previous_status: OpportunityStatus
    new_status: OpportunityStatus
    trigger_type: OpportunityRelationship
    evidence_id: Optional[UUID] = None
