"""Pydantic schemas and DTOs for IdeaGen API endpoints (Phase 6)."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import datetime
import uuid


# ── Overview / Workspace Stats ──────────────────────────────────────────

class WorkspaceStatsDTO(BaseModel):
    total_papers: int
    analyzed_papers: int
    total_passages: int
    total_statements: int
    total_gaps: int
    candidate_gaps: int
    validated_gaps: int
    rejected_gaps: int
    total_opportunities: int
    validated_opportunities: int
    analysis_status: str = "IDLE"


# ── Papers & Evidence ───────────────────────────────────────────────────

class PaperResponseDTO(BaseModel):
    id: uuid.UUID
    source_type: str
    source_id: str
    title: Optional[str] = None
    publication_date: Optional[str] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None
    created_at: Optional[str] = None
    document_count: int = 0
    passage_count: int = 0
    knowledge_status: str = "UNANALYZED"
    content_hash: Optional[str] = None


class EvidenceDetailDTO(BaseModel):
    statement_id: uuid.UUID
    statement_text: str
    statement_type: str
    paper_id: uuid.UUID
    paper_title: str
    passage_id: Optional[uuid.UUID] = None
    section: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    passage_text: Optional[str] = None


class StatementResponseDTO(BaseModel):
    id: uuid.UUID
    paper_id: uuid.UUID
    paper_title: str
    passage_id: Optional[uuid.UUID] = None
    statement_text: str
    statement_type: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section: Optional[str] = None
    confidence: Optional[float] = None
    created_at: Optional[str] = None


# ── Final Opportunity Report Schema ──────────────────────────────────────

class FeasibilityDTO(BaseModel):
    required_data: str
    required_compute: str
    implementation_complexity: str
    experimental_setup: str
    feasibility_score: float = 0.8
    summary: str = ""


class NoveltyDTO(BaseModel):
    assessment: str = "Potentially novel within the analyzed corpus"
    reasoning: str = ""
    differentiation: str = ""
    confidence: float = 0.8
    overlapping_statement_ids: List[uuid.UUID] = Field(default_factory=list)
    overlapping_paper_ids: List[uuid.UUID] = Field(default_factory=list)


class OpportunityReportDTO(BaseModel):
    opportunity_id: uuid.UUID
    gap_id: uuid.UUID
    
    # 16 Required Report Sections
    research_problem: str
    evidence_of_problem: List[EvidenceDetailDTO] = Field(default_factory=list)
    existing_approaches: List[str] = Field(default_factory=list)
    identified_gap: str
    why_existing_work_is_insufficient: str
    proposed_research_direction: str
    reasoning: str
    technical_approach: str
    expected_contribution: str
    feasibility: FeasibilityDTO
    prior_art_assessment: Dict[str, Any] = Field(default_factory=dict)
    differentiation: str
    supporting_papers: List[PaperResponseDTO] = Field(default_factory=list)
    supporting_evidence: List[EvidenceDetailDTO] = Field(default_factory=list)
    confidence: float = 0.8
    novelty_assessment: NoveltyDTO
    
    # Metadata
    status: str
    created_at: str
    generation_model: str
    validation_model: str
    embedding_model: str
    refinement_iteration_count: int
    pipeline_version: str = "v1.0"


# ── Legacy Demo Schemas (Preserved for compatibility) ─────────────────────

class DemoPaper(BaseModel):
    id: str
    title: str
    date: Optional[datetime.date]

class DemoDiscovery(BaseModel):
    cluster_size: int
    papers: List[DemoPaper]

class DemoGroundedClaim(BaseModel):
    text: str
    refs: List[str]

class DemoEvidence(BaseModel):
    source_id: str
    title: str
    text: str

class DemoGrounding(BaseModel):
    claims: List[DemoGroundedClaim]
    evidence: List[DemoEvidence]

class DemoTimelineEvent(BaseModel):
    relationship: str
    paper_title: str
    paper_date: Optional[datetime.date]
    status_change: str
    excerpt: str

class DemoTimelineBatch(BaseModel):
    batch_id: str
    name: str
    events: List[DemoTimelineEvent]

class DemoLeakageProtection(BaseModel):
    future_paper: str
    future_date: Optional[datetime.date]
    result: str

class DemoScenarioResponse(BaseModel):
    opportunity_id: str
    title: str
    problem_statement: str
    initial_status: str
    final_status: str
    discovery: DemoDiscovery
    grounding: DemoGrounding
    timeline: List[DemoTimelineBatch]
    leakage_protection: DemoLeakageProtection

class DemoOpportunityOverview(BaseModel):
    opportunity_id: str
    title: str
    problem_statement: str
    status: str
    confidence: float
    supporting_papers_count: int
    evidence_count: int

class SearchResultPaper(BaseModel):
    id: str
    source: str
    title: str
    authors: List[str]
    publication_year: Optional[int] = None
    publication_date: Optional[str] = None
    abstract: str
    url: Optional[str] = None
    doi: Optional[str] = None
    raw_metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    query: str
    source: str
    count: int
    papers: List[SearchResultPaper]

class AnalyzeRequest(BaseModel):
    papers: List[SearchResultPaper]
