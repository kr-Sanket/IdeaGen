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


class RunMetadataCountsDTO(BaseModel):
    papers: int = 0
    baseline_papers: int = 0
    later_evaluation_papers: int = 0
    statements: int = 0
    research_gaps: int = 0
    opportunities: int = 0
    lifecycle_opportunities: int = 0
    evidence_records: int = 0
    history_records: int = 0


class RunMetadataDTO(BaseModel):
    run_fingerprint: Optional[Dict[str, Any]] = None
    baseline_cutoff: Optional[str] = None
    evaluation_cutoffs: Optional[List[str]] = None
    stage_5_enabled: bool = False
    stage_5_batch_window_days: Optional[int] = None
    relevance_screening: Optional[Dict[str, Any]] = None
    discovery_completion: Optional[Dict[str, Any]] = None
    total_papers: int = 0
    total_statements: int = 0
    total_gaps: int = 0
    total_opportunities: int = 0
    counts: Optional[RunMetadataCountsDTO] = None
    database_status: str = Field(
        "CONNECTED",
        description="Connection observed while serving this request, with the SQL dialect, e.g. 'CONNECTED (sqlite)'.",
    )
    metric_interpretation_note: Optional[str] = None


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
    temporal_role: Optional[str] = None  # "BASELINE", "LATER_EVALUATION", "UNDATED"
    is_baseline: Optional[bool] = None


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
    passage_text: Optional[str] = None



# ── Final Opportunity Report Schema ──────────────────────────────────────

class FeasibilityDTO(BaseModel):
    # Model-generated values; None means the stored opportunity has no value (not measured).
    required_data: Optional[str] = None
    required_compute: Optional[str] = None
    implementation_complexity: Optional[str] = None
    experimental_setup: Optional[str] = None
    feasibility_score: Optional[float] = None
    summary: Optional[str] = None


class NoveltyDTO(BaseModel):
    # Model-generated prior-art assessment; None means not recorded for this opportunity.
    assessment: Optional[str] = None
    reasoning: Optional[str] = None
    differentiation: Optional[str] = None
    confidence: Optional[float] = None
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
    why_existing_work_is_insufficient: Optional[str] = None
    proposed_research_direction: str
    reasoning: str
    technical_approach: str
    expected_contribution: str
    feasibility: FeasibilityDTO
    prior_art_assessment: Dict[str, Any] = Field(default_factory=dict)
    differentiation: Optional[str] = None
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
    lifecycle_status: Optional[str] = None
    lifecycle_opportunity_id: Optional[uuid.UUID] = None



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

DEMO_DATA_NOTICE = (
    "Demonstration scenario: paper titles, dates and abstracts come from the bundled prototype "
    "corpus, but embeddings are synthetic and every relationship classification and confidence "
    "is scripted. These results are not model output and are not evidence of system accuracy."
)


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
    data_notice: str = DEMO_DATA_NOTICE

class DemoOpportunityOverview(BaseModel):
    opportunity_id: str
    title: str
    problem_statement: str
    status: str
    confidence: float
    supporting_papers_count: int
    evidence_count: int
    data_notice: str = DEMO_DATA_NOTICE

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


# ── Opportunity Evolution Subsystem DTOs (Checkpoint 10) ───────────────────

class OpportunityEvolveRequest(BaseModel):
    paper_ids: List[uuid.UUID] = Field(..., min_length=1, max_length=100, description="Bounded set of eligible paper IDs")
    cutoff_date: Optional[str] = Field(None, description="Optional ISO cutoff date YYYY-MM-DD")


class OpportunityEvolutionEventDTO(BaseModel):
    paper_id: uuid.UUID
    relationship: str
    previous_status: str
    new_status: str
    state_changed: bool
    evidence_id: Optional[uuid.UUID] = None
    history_id: Optional[uuid.UUID] = None


class PaperEvolutionErrorDTO(BaseModel):
    paper_id: uuid.UUID
    reason: str


class OpportunityEvolveResponse(BaseModel):
    opportunity_id: uuid.UUID
    initial_status: str
    final_status: str
    processed_count: int
    skipped_count: int
    failed_count: int
    events: List[OpportunityEvolutionEventDTO] = Field(default_factory=list)
    errors: List[PaperEvolutionErrorDTO] = Field(default_factory=list)


class OpportunityHistoryDTO(BaseModel):
    id: uuid.UUID
    opportunity_id: uuid.UUID
    previous_status: str
    new_status: str
    trigger_type: str
    evidence_id: Optional[uuid.UUID] = None
    created_at: Optional[str] = None


class OpportunityEvidenceDTO(BaseModel):
    id: uuid.UUID
    opportunity_id: uuid.UUID
    paper_id: uuid.UUID
    paper_title: Optional[str] = None
    paper_publication_date: Optional[str] = None
    batch_id: Optional[str] = None
    evaluation_cutoff: Optional[str] = None
    relationship: str
    confidence: Optional[float] = None
    supporting_text: Optional[str] = None
    verified: bool
    created_at: Optional[str] = None


class OpportunityListItemDTO(BaseModel):
    id: uuid.UUID = Field(..., description="Unique opportunity record identifier")
    gap_id: uuid.UUID = Field(..., description="Originating research gap UUID")
    research_problem: str = Field(..., description="High-level research problem statement")
    identified_gap: str = Field(..., description="Specific identified research gap")
    proposed_research_direction: str = Field(..., description="Proposed research direction")
    status: str = Field(..., description="Opportunity validation status (e.g. VALIDATED)")
    confidence: float = Field(0.8, description="Confidence score")
    created_at: Optional[str] = Field(None, description="ISO timestamp of record creation")
    supporting_statement_count: int = Field(0, description="Count of supporting research statements")
    supporting_paper_count: int = Field(0, description="Count of supporting papers")
    lifecycle_opportunity_id: Optional[uuid.UUID] = Field(None, description="Bridged native lifecycle opportunity UUID if linked")
    lifecycle_status: Optional[str] = Field(None, description="Lifecycle state (e.g. OPEN, VALIDATED, EXTENDED)")
    lifecycle_confidence: Optional[float] = Field(None, description="Native lifecycle confidence score")
    evidence_count: int = Field(0, description="Count of temporal evidence records")


# ── OpenAlex Ingestion DTOs (Stage 6A) ───────────────────────────────────

class OpenAlexIngestRequest(BaseModel):
    openalex_id: Optional[str] = Field(
        None,
        description="OpenAlex Work ID (e.g. 'W2741809807' or 'https://openalex.org/W2741809807')",
    )
    doi: Optional[str] = Field(
        None,
        description="Digital Object Identifier (e.g. '10.48550/arxiv.2005.11401' or 'https://doi.org/...')",
    )
    enable_evolution: Optional[bool] = Field(
        None,
        description=(
            "Opt out of post-ingestion evolution with false. true only takes effect when the server "
            "enables it (IDEAGEN_ENABLE_POST_INGESTION_EVOLUTION); otherwise it is ignored."
        ),
    )


class OpenAlexIngestResponse(BaseModel):
    paper_id: Optional[uuid.UUID] = Field(
        None, description="Persisted Paper UUID if successfully ingested or persisted as NO_PDF; existing Paper UUID if SKIPPED_DUPLICATE; None if error or ambiguous"
    )
    source_id: str = Field(..., description="Canonical source ID (e.g. 'W2741809807')")
    status: str = Field(..., description="Ingestion status: 'INGESTED', 'NO_PDF', 'SKIPPED_DUPLICATE', or 'ERROR'")
    has_pdf: bool = Field(..., description="True if full-text PDF was downloaded and parsed; False if NO_PDF metadata-only")
    document_id: Optional[uuid.UUID] = Field(None, description="Persisted PaperDocument UUID if full text is available")
    passage_count: int = Field(0, description="Number of text passages chunked and persisted")
    error_message: Optional[str] = Field(None, description="Error reason or fallback explanation if status is NO_PDF/ERROR/SKIPPED_DUPLICATE")
    batch_id: str = Field(..., description="Assigned temporal batch ID")
    doi: Optional[str] = Field(None, description="Normalized DOI if present")
    evolution_status: Optional[str] = Field(
        None, description="Post-ingestion evolution hook status: 'DISABLED', 'SUCCESS', 'PARTIAL_SUCCESS', 'FAILED', or 'SKIPPED'"
    )
    evolution_result: Optional[Dict[str, Any]] = Field(
        None, description="Structured outcome from IncrementalOpportunityEvolutionService if evolution was triggered"
    )


# ── OpenAlex Search DTOs (Stage 6B) ──────────────────────────────────────

class OpenAlexPaperResultDTO(BaseModel):
    id: str = Field(..., description="Canonical OpenAlex work identifier (e.g. 'W2741809807')")
    doi: Optional[str] = Field(None, description="Normalized DOI if available")
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(default_factory=list, description="List of author display names")
    publication_year: Optional[int] = Field(None, description="Publication year")
    publication_date: Optional[str] = Field(None, description="ISO publication date YYYY-MM-DD")
    abstract: str = Field("", description="Reconstructed abstract text")
    is_oa: Optional[bool] = Field(None, description="Open-access status")
    pdf_url: Optional[str] = Field(None, description="Direct URL to open-access PDF if available")
    landing_page_url: Optional[str] = Field(None, description="Primary landing page or DOI URL")
    citation_count: Optional[int] = Field(None, description="Cited by count from OpenAlex")


class OpenAlexSearchResponse(BaseModel):
    query: str = Field(..., description="Search query string")
    source: str = Field("OPENALEX", description="Search provider source identifier")
    count: int = Field(..., description="Number of paper results returned in this batch")
    total_count: Optional[int] = Field(None, description="Upstream OpenAlex total matching hit count if reported by provider")
    papers: List[OpenAlexPaperResultDTO] = Field(default_factory=list, description="List of matching paper results")
