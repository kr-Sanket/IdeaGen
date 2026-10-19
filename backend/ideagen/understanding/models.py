"""Domain models and DTOs for AI Paper Understanding (Phase 3).

Defines structured ResearchKnowledge, ResearchStatements, statement types,
and model extraction schemas with full provenance tracking.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class StatementType(str, enum.Enum):
    """Canonical classification for extracted scientific statements."""
    PROBLEM = "PROBLEM"
    QUESTION = "QUESTION"
    METHOD = "METHOD"
    DATASET = "DATASET"
    FINDING = "FINDING"
    LIMITATION = "LIMITATION"
    FUTURE_WORK = "FUTURE_WORK"
    ASSUMPTION = "ASSUMPTION"
    CONTRIBUTION = "CONTRIBUTION"


class ExtractionStatus(str, enum.Enum):
    """Execution status for the AI extraction process."""
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class RawStatementExtraction(BaseModel):
    """Model-generated research statement prior to provenance resolution."""
    model_config = ConfigDict(extra="ignore")

    statement_text: str = Field(description="The scientific claim, statement, or finding extracted from the text.")
    statement_type: StatementType = Field(description="The category of the scientific statement.")
    passage_index: Optional[int] = Field(
        default=None,
        description="0-based integer index of the input passage (matching the [PASSAGE X] headers) providing supporting evidence. Required for all grounded statements (use null if ungrounded).",
    )
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence score if provided by the model.")


class StructuredPaperExtractionOutput(BaseModel):
    """Pydantic schema enforcing structured model output for scientific paper understanding."""
    model_config = ConfigDict(extra="ignore")

    research_problem: Optional[str] = Field(default=None, description="Primary research problem or gap addressed by the paper.")
    research_questions: List[str] = Field(default_factory=list, description="Specific research questions formulated.")
    methodology: Optional[str] = Field(default=None, description="Core methodology, architecture, or algorithmic technique proposed.")
    datasets: List[str] = Field(default_factory=list, description="Datasets, benchmarks, or corpora utilized or introduced.")
    findings: List[str] = Field(default_factory=list, description="Key experimental results or core findings.")
    limitations: List[str] = Field(default_factory=list, description="Explicit limitations or threats to validity identified.")
    future_work: List[str] = Field(default_factory=list, description="Directions or open questions suggested for future research.")
    statements: List[RawStatementExtraction] = Field(default_factory=list, description="Granular scientific statements with passage references.")


class ResearchStatementCreateDTO(BaseModel):
    """DTO for persisting a validated research statement with source provenance."""
    model_config = ConfigDict(frozen=True)

    statement_text: str
    statement_type: StatementType
    source_passage_id: Optional[UUID] = None
    source_page_start: Optional[int] = None
    source_page_end: Optional[int] = None
    section: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ResearchKnowledgeCreateDTO(BaseModel):
    """DTO for persisting complete paper research knowledge and provenance."""
    model_config = ConfigDict(frozen=True)

    paper_id: UUID
    research_problem: Optional[str] = None
    research_questions: List[str] = Field(default_factory=list)
    methodology: Optional[str] = None
    datasets: List[str] = Field(default_factory=list)
    findings: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    future_work: List[str] = Field(default_factory=list)
    research_statements: List[ResearchStatementCreateDTO] = Field(default_factory=list)

    # Model versioning & provenance
    model_name: str
    model_version: Optional[str] = None
    extraction_schema_version: str = "v1.0"
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    extraction_status: ExtractionStatus = ExtractionStatus.SUCCESS
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ResearchKnowledgeResponseDTO(BaseModel):
    """DTO returned by APIs to represent extracted knowledge."""
    id: UUID
    paper_id: UUID
    research_problem: Optional[str]
    research_questions: List[str]
    methodology: Optional[str]
    datasets: List[str]
    findings: List[str]
    limitations: List[str]
    future_work: List[str]
    statement_count: int
    model_name: str
    extraction_schema_version: str
    extracted_at: datetime
    extraction_status: str
    error_message: Optional[str]
