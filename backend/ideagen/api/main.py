"""IdeaGen REST API Main Application (Phase 6).

FastAPI backend application exposing literature, research landscape, gaps, evidence tracing,
opportunity synthesis, workspace metrics, and 16-section final opportunity reports.
"""

from __future__ import annotations

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import text, func, select
import json
import logging
import uuid
import datetime
import tempfile
import os
from pathlib import Path

logger = logging.getLogger(__name__)

from ideagen.db import (
    get_db,
    Paper,
    PaperDocument,
    PaperPassage,
    ResearchKnowledge,
    ResearchStatement,
    ResearchGap,
    GapEvidence,
    Opportunity,
    OpportunityEvidence,
    OpportunityHistory,
    ResearchOpportunityRecord,
    ExperimentMetadataRecord,
    SQLAlchemyPersistenceLayer,
)
from ideagen.ingestion.models import PaperCreateDTO, PaperDocumentCreateDTO, SourceType
from ideagen.api.schema import (
    WorkspaceStatsDTO,
    RunMetadataDTO,
    RunMetadataCountsDTO,
    PaperResponseDTO,
    EvidenceDetailDTO,
    StatementResponseDTO,
    OpportunityReportDTO,
    OpportunityEvolveRequest,
    OpportunityEvolveResponse,
    OpportunityHistoryDTO,
    OpportunityEvidenceDTO,
    OpportunityListItemDTO,

    OpportunityEvolutionEventDTO,
    PaperEvolutionErrorDTO,
    DemoScenarioResponse,
    DemoOpportunityOverview,
    SearchResponse,
    AnalyzeRequest,
    SearchResultPaper,
    OpenAlexIngestRequest,
    OpenAlexIngestResponse,
    OpenAlexPaperResultDTO,
    OpenAlexSearchResponse,
)
from ideagen.api.demo_service import build_demo_scenario
from ideagen.ingestion.search import ArxivSearchProvider, OpenAlexSearchProvider, OpenAlexSearchError
from ideagen.ingestion.pdf import ManualPDFIngestionService, PDFExtractionError, DuplicateDocumentError
from ideagen.ingestion.openalex_ingestion import (
    OpenAlexPaperIngestionService,
    IngestionStatus,
    EvolutionHookStatus,
)
from ideagen.ingestion.parser import PaperNormalizer
from ideagen.core.config import get_settings
from ideagen.understanding import PaperUnderstandingPipeline, ResearchKnowledgeResponseDTO
from ideagen.discovery import SemanticGapDiscoveryPipeline, ResearchGapResponseDTO, GapStatus
from ideagen.discovery.vector_index import StatementVectorIndex
from ideagen.discovery.embeddings import get_default_embedding_provider
from ideagen.llm import get_default_llm_provider
from ideagen.opportunity.classifier import RelationshipClassifier
from ideagen.generation.pipeline import ResearchOpportunityPipeline
from ideagen.generation.models import ResearchOpportunityInputDTO, OpportunityStatus
from ideagen.reporting.report_service import OpportunityReportService

app = FastAPI(title="IdeaGen API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health Check ─────────────────────────────────────────────────────────

@app.get("/api/health")
@app.get("/api/v1/health")
@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Verifies application health and database connectivity."""
    try:
        db.execute(text("SELECT 1"))
        dialect = db.bind.dialect.name if db.bind else "unknown"
        return {"status": "healthy", "database": f"{dialect}_connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database connection error: {e}")


# ── Workspace Overview & Stats ───────────────────────────────────────────

@app.get("/api/v1/stats", response_model=WorkspaceStatsDTO)
@app.get("/api/stats", response_model=WorkspaceStatsDTO)
def get_workspace_stats(db: Session = Depends(get_db)):
    """Calculates real workspace metrics directly from the PostgreSQL database."""
    total_papers = db.query(func.count(Paper.id)).scalar() or 0
    analyzed_papers = db.query(func.count(ResearchKnowledge.id)).scalar() or 0
    total_passages = db.query(func.count(PaperPassage.id)).scalar() or 0
    total_statements = db.query(func.count(ResearchStatement.id)).scalar() or 0

    total_gaps = db.query(func.count(ResearchGap.id)).scalar() or 0
    candidate_gaps = db.query(func.count(ResearchGap.id)).filter(ResearchGap.status == "CANDIDATE").scalar() or 0
    validated_gaps = db.query(func.count(ResearchGap.id)).filter(ResearchGap.status == "VALIDATED").scalar() or 0
    rejected_gaps = db.query(func.count(ResearchGap.id)).filter(ResearchGap.status == "REJECTED").scalar() or 0

    total_opps = db.query(func.count(ResearchOpportunityRecord.id)).scalar() or 0
    val_opps = db.query(func.count(ResearchOpportunityRecord.id)).filter(
        ResearchOpportunityRecord.status.in_(["VALIDATED", "REFINED"])
    ).scalar() or 0

    status = "COMPLETE" if total_opps > 0 else ("ANALYZING" if total_papers > 0 else "IDLE")

    return WorkspaceStatsDTO(
        total_papers=total_papers,
        analyzed_papers=analyzed_papers,
        total_passages=total_passages,
        total_statements=total_statements,
        total_gaps=total_gaps,
        candidate_gaps=candidate_gaps,
        validated_gaps=validated_gaps,
        rejected_gaps=rejected_gaps,
        total_opportunities=total_opps,
        validated_opportunities=val_opps,
        analysis_status=status,
    )


# ── Run Overview & Metadata Endpoints ────────────────────────────────────

def get_configured_temporal_cutoff(db: Session) -> Optional[datetime.date]:
    """Helper to load configured baseline temporal cutoff date from experiment metadata."""
    meta = db.query(ExperimentMetadataRecord).filter(ExperimentMetadataRecord.key == "configuration_fingerprint").first()
    if meta and meta.value:
        try:
            fp = json.loads(meta.value)
            tc = fp.get("temporal_cutoff")
            if tc:
                return datetime.date.fromisoformat(str(tc))
        except Exception:
            pass
    return None


def resolve_paper_temporal_role(
    paper_date: Optional[datetime.date],
    temporal_cutoff_date: Optional[datetime.date],
) -> tuple[Optional[str], Optional[bool]]:
    """Derives (temporal_role, is_baseline) for a paper against the configured baseline cutoff."""
    if paper_date is None:
        return ("UNDATED", None)
    if temporal_cutoff_date is None:
        return ("DATED", None)
    if paper_date <= temporal_cutoff_date:
        return ("BASELINE", True)
    return ("LATER_EVALUATION", False)


@app.get("/api/v1/run-metadata", response_model=RunMetadataDTO)
@app.get("/api/v1/run_metadata", response_model=RunMetadataDTO)
@app.get("/api/run-metadata", response_model=RunMetadataDTO)
def get_run_metadata(db: Session = Depends(get_db)):
    """Retrieves experiment configuration fingerprint, cutoff rules, and pipeline metrics for the completed run."""
    meta_rows = db.query(ExperimentMetadataRecord).all()
    meta_map = {row.key: row.value for row in meta_rows}

    fingerprint = None
    if "configuration_fingerprint" in meta_map:
        try:
            fingerprint = json.loads(meta_map["configuration_fingerprint"])
        except Exception as e:
            logger.warning(f"Failed to parse configuration_fingerprint: {e}")

    screening_report = None
    if "relevance_screening_report" in meta_map:
        try:
            screening_report = json.loads(meta_map["relevance_screening_report"])
        except Exception as e:
            logger.warning(f"Failed to parse relevance_screening_report: {e}")

    discovery_comp = None
    if "stage_3_discovery_completion" in meta_map:
        try:
            discovery_comp = json.loads(meta_map["stage_3_discovery_completion"])
        except Exception as e:
            logger.warning(f"Failed to parse stage_3_discovery_completion: {e}")

    baseline_cutoff = fingerprint.get("temporal_cutoff") if fingerprint else None
    eval_cutoffs = fingerprint.get("stage_5_cutoffs") if fingerprint else None
    stage_5_enabled = bool(fingerprint.get("enable_stage_5", False)) if fingerprint else False
    stage_5_batch_window_days = fingerprint.get("stage_5_batch_window_days") if fingerprint else None

    # Pipeline limitations note
    metric_interpretation_note = (
        "Stage 5 measures descriptive temporal evolution counts (relationships, transitions, status distribution). "
        "These values represent event volume and trajectory progression, not empirical accuracy, precision, or research "
        "validity without human-annotated gold standard data."
    )

    total_papers = db.query(func.count(Paper.id)).scalar() or 0
    total_statements = db.query(func.count(ResearchStatement.id)).scalar() or 0
    total_gaps = db.query(func.count(ResearchGap.id)).scalar() or 0
    total_opps = db.query(func.count(ResearchOpportunityRecord.id)).scalar() or 0

    cutoff_dt = get_configured_temporal_cutoff(db)
    if cutoff_dt:
        baseline_papers = db.query(func.count(Paper.id)).filter(Paper.publication_date <= cutoff_dt).scalar() or 0
        later_papers = db.query(func.count(Paper.id)).filter(Paper.publication_date > cutoff_dt).scalar() or 0
    else:
        baseline_papers = total_papers
        later_papers = 0

    lifecycle_opps = db.query(func.count(Opportunity.id)).scalar() or 0
    evidence_records = db.query(func.count(OpportunityEvidence.id)).scalar() or 0
    history_records = db.query(func.count(OpportunityHistory.id)).scalar() or 0

    counts = RunMetadataCountsDTO(
        papers=total_papers,
        baseline_papers=baseline_papers,
        later_evaluation_papers=later_papers,
        statements=total_statements,
        research_gaps=total_gaps,
        opportunities=total_opps,
        lifecycle_opportunities=lifecycle_opps,
        evidence_records=evidence_records,
        history_records=history_records,
    )

    return RunMetadataDTO(
        run_fingerprint=fingerprint,
        baseline_cutoff=baseline_cutoff,
        evaluation_cutoffs=eval_cutoffs,
        stage_5_enabled=stage_5_enabled,
        stage_5_batch_window_days=stage_5_batch_window_days,
        relevance_screening=screening_report,
        discovery_completion=discovery_comp,
        total_papers=total_papers,
        total_statements=total_statements,
        total_gaps=total_gaps,
        total_opportunities=total_opps,
        counts=counts,
        # Observed, not assumed: this request has just read the counts above from the database.
        database_status=f"CONNECTED ({db.get_bind().dialect.name})",
        metric_interpretation_note=metric_interpretation_note,
    )


# ── Literature & Papers Endpoints ────────────────────────────────────────

@app.get("/api/v1/papers", response_model=List[PaperResponseDTO])
@app.get("/api/papers", response_model=List[PaperResponseDTO])
def list_papers(limit: int = 50, offset: int = 0, db: Session = Depends(get_db)):
    """Lists real persisted papers from PostgreSQL with pagination and metadata."""
    cutoff_dt = get_configured_temporal_cutoff(db)
    papers = db.query(Paper).order_by(Paper.created_at.desc()).offset(offset).limit(limit).all()

    results: List[PaperResponseDTO] = []
    for p in papers:
        doc_count = len(p.documents) if p.documents else 0
        passage_count = 0
        content_hash = None
        if p.documents:
            for d in p.documents:
                passage_count += len(d.passages) if d.passages else 0
            if p.documents[0].content_hash:
                content_hash = p.documents[0].content_hash

        knowledge_status = "ANALYZED" if p.research_knowledge else "UNANALYZED"
        pub_date_str = str(p.publication_date) if p.publication_date else None
        created_str = p.created_at.isoformat() if p.created_at else None
        temporal_role, is_baseline = resolve_paper_temporal_role(p.publication_date, cutoff_dt)

        results.append(
            PaperResponseDTO(
                id=p.id,
                source_type=p.source_type,
                source_id=p.source_id,
                title=p.title or f"Paper {p.source_id}",
                publication_date=pub_date_str,
                doi=p.doi,
                abstract=p.abstract,
                created_at=created_str,
                document_count=doc_count,
                passage_count=passage_count,
                knowledge_status=knowledge_status,
                content_hash=content_hash,
                temporal_role=temporal_role,
                is_baseline=is_baseline,
            )
        )
    return results


@app.get("/api/v1/papers/{paper_id}", response_model=PaperResponseDTO)
def get_paper_details(paper_id: uuid.UUID, db: Session = Depends(get_db)):
    """Retrieves single paper details with passage and knowledge status."""
    p = db.query(Paper).filter(Paper.id == paper_id).first()
    if not p:
        raise HTTPException(status_code=404, detail=f"Paper {paper_id} not found")

    cutoff_dt = get_configured_temporal_cutoff(db)
    doc_count = len(p.documents) if p.documents else 0
    passage_count = 0
    content_hash = None
    if p.documents:
        for d in p.documents:
            passage_count += len(d.passages) if d.passages else 0
        if p.documents[0].content_hash:
            content_hash = p.documents[0].content_hash

    knowledge_status = "ANALYZED" if p.research_knowledge else "UNANALYZED"
    pub_date_str = str(p.publication_date) if p.publication_date else None
    created_str = p.created_at.isoformat() if p.created_at else None
    temporal_role, is_baseline = resolve_paper_temporal_role(p.publication_date, cutoff_dt)

    return PaperResponseDTO(
        id=p.id,
        source_type=p.source_type,
        source_id=p.source_id,
        title=p.title or f"Paper {p.source_id}",
        publication_date=pub_date_str,
        doi=p.doi,
        abstract=p.abstract,
        created_at=created_str,
        document_count=doc_count,
        passage_count=passage_count,
        knowledge_status=knowledge_status,
        content_hash=content_hash,
        temporal_role=temporal_role,
        is_baseline=is_baseline,
    )



@app.get("/api/v1/documents/{document_id}/passages")
@app.get("/api/documents/{document_id}/passages")
def get_document_passages(document_id: uuid.UUID, db: Session = Depends(get_db)):
    """Retrieves all deterministic passages for an ingested document."""
    persistence = SQLAlchemyPersistenceLayer(db)
    passages = persistence.get_passages_by_document(document_id)
    return [
        {
            "id": str(p.id),
            "paper_id": str(p.paper_id),
            "document_id": str(p.document_id),
            "passage_index": p.passage_index,
            "page_start": p.page_start,
            "page_end": p.page_end,
            "section": p.section,
            "text": p.text,
        }
        for p in passages
    ]


@app.post("/api/v1/papers/{paper_id}/understand", response_model=ResearchKnowledgeResponseDTO)
@app.post("/api/papers/{paper_id}/understand", response_model=ResearchKnowledgeResponseDTO)
def understand_paper(paper_id: uuid.UUID, db: Session = Depends(get_db)):
    """Triggers AI paper understanding pipeline on an ingested paper."""
    persistence = SQLAlchemyPersistenceLayer(db)
    pipeline = PaperUnderstandingPipeline(persistence=persistence)
    try:
        return pipeline.understand_paper(paper_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Understanding pipeline failed: {str(e)}")


@app.get("/api/v1/papers/{paper_id}/knowledge", response_model=ResearchKnowledgeResponseDTO)
@app.get("/api/papers/{paper_id}/knowledge", response_model=ResearchKnowledgeResponseDTO)
def get_paper_knowledge(paper_id: uuid.UUID, db: Session = Depends(get_db)):
    """Retrieves extracted research knowledge for a paper."""
    persistence = SQLAlchemyPersistenceLayer(db)
    pipeline = PaperUnderstandingPipeline(persistence=persistence)
    knowledge = pipeline.get_paper_knowledge(paper_id)
    if not knowledge:
        raise HTTPException(status_code=404, detail=f"No research knowledge found for paper {paper_id}")
    return knowledge


@app.get("/api/v1/papers/{paper_id}/statements", response_model=List[StatementResponseDTO])
@app.get("/api/papers/{paper_id}/statements", response_model=List[StatementResponseDTO])
def get_paper_statements(paper_id: uuid.UUID, db: Session = Depends(get_db)):
    """Retrieves all grounded research statements with provenance for a paper."""
    persistence = SQLAlchemyPersistenceLayer(db)
    statements = persistence.get_statements_by_paper(paper_id)
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    paper_title = paper.title if paper and paper.title else "Unknown Paper"

    return [
        StatementResponseDTO(
            id=s.id,
            paper_id=s.paper_id,
            paper_title=paper_title,
            passage_id=s.passage_id,
            statement_text=s.statement_text,
            statement_type=s.statement_type,
            page_start=s.page_start,
            page_end=s.page_end,
            section=s.section,
            confidence=s.confidence,
            created_at=s.created_at.isoformat() if s.created_at else None,
            passage_text=s.passage.text if s.passage else None,
        )
        for s in statements
    ]


@app.get("/api/v1/statements", response_model=List[StatementResponseDTO])
def search_statements(
    type: Optional[str] = Query(None, description="LIMITATION, FUTURE_WORK, METHODOLOGY, FINDING"),
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Retrieves extracted research statements across the corpus."""
    query = db.query(ResearchStatement)
    if type:
        query = query.filter(ResearchStatement.statement_type == type.upper())

    statements = query.order_by(ResearchStatement.created_at.desc()).limit(limit).all()

    results: List[StatementResponseDTO] = []
    for s in statements:
        paper = db.query(Paper).filter(Paper.id == s.paper_id).first()
        paper_title = paper.title if paper and paper.title else f"Paper {s.paper_id.hex[:8]}"
        created_str = s.created_at.isoformat() if s.created_at else None

        results.append(
            StatementResponseDTO(
                id=s.id,
                paper_id=s.paper_id,
                paper_title=paper_title,
                passage_id=s.passage_id,
                statement_text=s.statement_text,
                statement_type=s.statement_type,
                page_start=s.page_start,
                page_end=s.page_end,
                section=s.section,
                confidence=s.confidence,
                created_at=created_str,
                passage_text=s.passage.text if s.passage else None,
            )
        )
    return results



# ── Research Gaps Endpoints ──────────────────────────────────────────────

@app.post("/api/v1/gaps/discover", response_model=List[ResearchGapResponseDTO])
@app.post("/api/gaps/discover", response_model=List[ResearchGapResponseDTO])
def discover_research_gaps(
    cutoff_date: Optional[str] = Query(None, description="Optional ISO publication cutoff date YYYY-MM-DD"),
    db: Session = Depends(get_db)
):
    """Triggers end-to-end semantic gap discovery across all research statements."""
    persistence = SQLAlchemyPersistenceLayer(db)
    pipeline = SemanticGapDiscoveryPipeline(persistence=persistence)
    try:
        return pipeline.discover_gaps(cutoff_date=cutoff_date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Discovery pipeline failed: {str(e)}")


@app.get("/api/v1/gaps", response_model=List[ResearchGapResponseDTO])
@app.get("/api/gaps", response_model=List[ResearchGapResponseDTO])
def list_research_gaps(status: Optional[GapStatus] = None, db: Session = Depends(get_db)):
    """Retrieves all discovered research gaps, optionally filtered by status."""
    persistence = SQLAlchemyPersistenceLayer(db)
    pipeline = SemanticGapDiscoveryPipeline(persistence=persistence)
    return pipeline.get_gaps(status=status)


@app.get("/api/v1/gaps/{gap_id}", response_model=ResearchGapResponseDTO)
@app.get("/api/gaps/{gap_id}", response_model=ResearchGapResponseDTO)
def get_research_gap(gap_id: uuid.UUID, db: Session = Depends(get_db)):
    """Retrieves a specific research gap by ID with evidence count and details."""
    persistence = SQLAlchemyPersistenceLayer(db)
    gap = persistence.get_research_gap(gap_id)
    if not gap:
        raise HTTPException(status_code=404, detail=f"Research gap {gap_id} not found")

    evidence_records = persistence.get_gap_evidence(gap_id)
    stmt_ids = [e.statement_id for e in evidence_records]
    paper_ids = list({e.paper_id for e in evidence_records})

    return ResearchGapResponseDTO(
        id=gap.id,
        gap_statement=gap.gap_statement,
        research_problem=gap.research_problem,
        status=GapStatus(gap.status),
        evidence_count=gap.evidence_count,
        supporting_statement_ids=stmt_ids,
        supporting_paper_ids=paper_ids,
        limitations=gap.limitations or [],
        future_work=gap.future_work or [],
        validation_reasoning=gap.validation_reasoning,
        validation_confidence=gap.validation_confidence,
        created_at=gap.created_at,
        metadata=gap.metadata_json or {},
    )


# ── Opportunities & Final Report Endpoints ──────────────────────────────

@app.post("/api/v1/gaps/{gap_id}/opportunity", response_model=OpportunityReportDTO)
@app.post("/api/v1/gaps/{gap_id}/generate_opportunity", response_model=OpportunityReportDTO)
def generate_opportunity_from_gap(
    gap_id: uuid.UUID,
    cutoff_date: Optional[str] = Query(None, description="Optional ISO cutoff date YYYY-MM-DD to restrict prior-art retrieval"),
    db: Session = Depends(get_db)
):
    """Synthesizes and persists a research opportunity from a validated research gap."""
    persistence = SQLAlchemyPersistenceLayer(db)
    gap = persistence.get_research_gap(gap_id)
    if not gap:
        raise HTTPException(status_code=404, detail=f"Research gap {gap_id} not found.")

    if gap.status != "VALIDATED":
        raise HTTPException(status_code=400, detail=f"Gap {gap_id} must be VALIDATED before opportunity generation (current status: {gap.status}).")

    evidence_records = persistence.get_gap_evidence(gap_id)
    stmt_ids = [e.statement_id for e in evidence_records]
    paper_ids = list({e.paper_id for e in evidence_records})

    # Fetch statement texts
    supporting_stmts = db.query(ResearchStatement).filter(ResearchStatement.id.in_(stmt_ids)).all() if stmt_ids else []
    stmt_texts = [s.statement_text for s in supporting_stmts]
    methods = [s.statement_text for s in supporting_stmts if s.statement_type in {"METHOD", "METHODOLOGY", "BASELINE"}]

    input_dto = ResearchOpportunityInputDTO(
        gap_id=gap.id,
        gap_statement=gap.gap_statement,
        research_problem=gap.research_problem,
        supporting_statement_ids=stmt_ids,
        supporting_paper_ids=paper_ids,
        supporting_statements_text=stmt_texts,
        recurring_limitations=gap.limitations or [],
        future_work_statements=gap.future_work or [],
        existing_methods=methods,
    )

    # Initialize Phase 4 vector index & embedder using configured provider
    embedder = get_default_embedding_provider()
    dimension = embedder.dimension
    vector_index = StatementVectorIndex(dimension=dimension)

    # Resolve effective cutoff (from query param or gap metadata)
    effective_cutoff = cutoff_date
    if not effective_cutoff and gap.metadata_json and isinstance(gap.metadata_json, dict):
        effective_cutoff = gap.metadata_json.get("temporal_cutoff")

    # Populate FAISS index from db statements using genuine embeddings
    if effective_cutoff:
        from ideagen.discovery.temporal import filter_statements_by_cutoff
        all_stmts = persistence.get_all_research_statements(cutoff_date=effective_cutoff)
        all_stmts = filter_statements_by_cutoff(all_stmts, effective_cutoff)
    else:
        all_stmts = db.query(ResearchStatement).all()

    if all_stmts:
        stmt_texts_to_embed = [s.statement_text for s in all_stmts]
        stmt_vectors = embedder.embed_texts(stmt_texts_to_embed)
        vector_index.add_statements(all_stmts, stmt_vectors)

    llm = get_default_llm_provider()
    pipeline = ResearchOpportunityPipeline(
        vector_index=vector_index,
        embedder=embedder,
        llm_provider=llm,
        max_refinement_iterations=2,
    )

    try:
        opp_dto = pipeline.process_validated_gap(input_dto=input_dto, db_session=db)
    except Exception as e:
        logger.error(f"Opportunity generation failed for gap {gap_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Opportunity generation failed: {e}")

    report_service = OpportunityReportService(db)
    report = report_service.generate_opportunity_report(opp_dto.id)
    if not report:
        raise HTTPException(status_code=500, detail="Failed to compile opportunity report.")
    return report


@app.get("/api/v1/opportunities", response_model=List[OpportunityListItemDTO])
@app.get("/api/opportunities", response_model=List[OpportunityListItemDTO])
def list_opportunities(db: Session = Depends(get_db)):
    """Lists all validated research opportunities persisted in PostgreSQL."""
    opps = db.query(ResearchOpportunityRecord).order_by(ResearchOpportunityRecord.created_at.desc()).all()
    results = []
    for o in opps:
        lifecycle_id = str(o.lifecycle_opportunity.id) if o.lifecycle_opportunity else None
        lifecycle_status = o.lifecycle_opportunity.status if o.lifecycle_opportunity else None
        lifecycle_conf = o.lifecycle_opportunity.confidence if o.lifecycle_opportunity else None
        ev_count = 0
        if o.lifecycle_opportunity:
            ev_count = db.query(func.count(OpportunityEvidence.id)).filter(
                OpportunityEvidence.opportunity_id == o.lifecycle_opportunity.id
            ).scalar() or 0

        results.append({
            "id": str(o.id),
            "gap_id": str(o.gap_id),
            "research_problem": o.research_problem,
            "identified_gap": o.identified_gap,
            "proposed_research_direction": o.proposed_research_direction,
            "status": o.status,
            "confidence": o.confidence,
            "created_at": o.created_at.isoformat() if o.created_at else None,
            "supporting_statement_count": len(o.supporting_statement_ids or []),
            "supporting_paper_count": len(o.supporting_paper_ids or []),
            "lifecycle_opportunity_id": lifecycle_id,
            "lifecycle_status": lifecycle_status,
            "lifecycle_confidence": lifecycle_conf,
            "evidence_count": ev_count,
        })
    return results



@app.get("/api/v1/opportunities/{opportunity_id}/report", response_model=OpportunityReportDTO)
@app.get("/api/opportunities/{opportunity_id}/report", response_model=OpportunityReportDTO)
def get_opportunity_report(opportunity_id: uuid.UUID, db: Session = Depends(get_db)):
    """Retrieves the complete 16-section Final Opportunity Report for an opportunity."""
    report_service = OpportunityReportService(db)
    report = report_service.generate_opportunity_report(opportunity_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"Opportunity report for ID {opportunity_id} not found.")
    return report


# ── Opportunity Lifecycle Evolution Endpoints (Checkpoint 10) ──────────────

def get_relationship_classifier() -> RelationshipClassifier:
    """Dependency provider for RelationshipClassifier using configured LLM provider."""
    provider = get_default_llm_provider()
    return RelationshipClassifier(provider=provider)


def resolve_lifecycle_opportunity(
    opportunity_id: uuid.UUID,
    db: Session,
    create_if_missing: bool = False,
) -> Optional[Opportunity]:
    """Resolves an Opportunity lifecycle entity from an Opportunity ID or bridged ResearchOpportunityRecord ID.

    By default (create_if_missing=False), strictly read-only: does not create or flush lifecycle entities.
    """
    # 1. Match directly on Opportunity.id
    opp = db.query(Opportunity).filter(Opportunity.id == opportunity_id).first()
    if opp is not None:
        return opp

    # 2. Match on Opportunity.research_opportunity_id
    opp = db.query(Opportunity).filter(Opportunity.research_opportunity_id == opportunity_id).first()
    if opp is not None:
        return opp

    # 3. Check if opportunity_id belongs to a ResearchOpportunityRecord
    rec = db.query(ResearchOpportunityRecord).filter(ResearchOpportunityRecord.id == opportunity_id).first()
    if rec is not None:
        if create_if_missing:
            from ideagen.opportunity.bridge import OpportunityBridgeService
            bridge = OpportunityBridgeService(db)
            return bridge.bridge_opportunity(rec)
        return None

    return None


@app.post("/api/v1/opportunities/{opportunity_id}/evolve", response_model=OpportunityEvolveResponse)
@app.post("/api/opportunities/{opportunity_id}/evolve", response_model=OpportunityEvolveResponse)
def evolve_opportunity(
    opportunity_id: uuid.UUID,
    req: OpportunityEvolveRequest,
    db: Session = Depends(get_db),
    classifier: RelationshipClassifier = Depends(get_relationship_classifier),
):
    """Evolves an opportunity's lifecycle status using independently verified paper evidence."""
    opp = resolve_lifecycle_opportunity(opportunity_id, db, create_if_missing=True)
    if opp is None:
        raise HTTPException(status_code=404, detail=f"Opportunity {opportunity_id} not found.")
    initial_status = opp.status

    if not req.paper_ids:
        raise HTTPException(status_code=400, detail="paper_ids list cannot be empty.")
    if len(req.paper_ids) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 paper IDs allowed per request.")

    # Validate cutoff date if provided
    norm_cutoff: Optional[datetime.date] = None
    if req.cutoff_date is not None:
        from ideagen.discovery.temporal import normalize_cutoff_date
        try:
            norm_cutoff = normalize_cutoff_date(req.cutoff_date)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid cutoff date: {e}")

    # Lower temporal bound: only papers published on/after the opportunity existed can be
    # evolution evidence (same rule as incremental post-ingestion evolution).
    from ideagen.opportunity.incremental import get_opportunity_effective_creation_date
    opp_date = get_opportunity_effective_creation_date(opp)

    if norm_cutoff is not None:
        # Ensure the opportunity itself existed at or before the requested cutoff
        if opp_date is not None:
            if opp_date > norm_cutoff:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Opportunity {opportunity_id} was created on {opp_date.isoformat()}, "
                        f"which is after the requested cutoff date {norm_cutoff.isoformat()}."
                    ),
                )

    # Validate paper existence
    papers = db.query(Paper).filter(Paper.id.in_(req.paper_ids)).all()
    found_map = {p.id: p for p in papers}
    missing_ids = [pid for pid in req.paper_ids if pid not in found_map]
    if missing_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Papers not found: {', '.join(str(pid) for pid in missing_ids)}",
        )

    # Exclude papers already recorded for this opportunity
    existing_evidences = db.query(OpportunityEvidence).filter(
        OpportunityEvidence.opportunity_id == opp.id
    ).all()
    already_recorded_ids = {ev.paper_id for ev in existing_evidences}

    # Partition into eligible vs skipped
    eligible_papers: List[Paper] = []
    skipped_count = 0
    errors: List[PaperEvolutionErrorDTO] = []
    events: List[OpportunityEvolutionEventDTO] = []

    for pid in req.paper_ids:
        p = found_map[pid]
        if p.id in already_recorded_ids:
            skipped_count += 1
            continue

        if norm_cutoff is not None or opp_date is not None:
            if p.publication_date is None:
                skipped_count += 1
                continue
        if norm_cutoff is not None and p.publication_date > norm_cutoff:
            skipped_count += 1
            continue
        if opp_date is not None and p.publication_date < opp_date:
            skipped_count += 1
            continue

        eligible_papers.append(p)

    processed_count = 0
    failed_count = 0

    if eligible_papers:
        from ideagen.opportunity.service import OpportunityTransitionService
        from ideagen.opportunity.integration import OpportunityClassificationIntegration
        from ideagen.opportunity.temporal import OpportunityEvolutionService, TemporalBatch

        trans_service = OpportunityTransitionService(db)
        integration = OpportunityClassificationIntegration(classifier, trans_service, db)
        evolution_service = OpportunityEvolutionService(integration)

        start_date = opp_date if opp_date is not None else datetime.date(1900, 1, 1)
        end_date = norm_cutoff if norm_cutoff is not None else datetime.date(2100, 1, 1)
        batch = TemporalBatch(
            batch_id=f"evolve_{uuid.uuid4().hex[:8]}",
            start_date=start_date,
            end_date=end_date,
            sequence_number=1,
        )

        for p in eligible_papers:
            try:
                batch_res = evolution_service.process_batch(
                    db=db,
                    opportunity_id=opp.id,
                    batch=batch,
                    papers=[p],
                )
                processed_count += batch_res.processed_paper_count
                skipped_count += batch_res.skipped_paper_count
                for ev in batch_res.events:
                    events.append(
                        OpportunityEvolutionEventDTO(
                            paper_id=ev.paper_id,
                            relationship=ev.relationship.value,
                            previous_status=ev.previous_status.value,
                            new_status=ev.new_status.value,
                            state_changed=ev.state_changed,
                            evidence_id=ev.evidence_id,
                            history_id=ev.history_id,
                        )
                    )
            except Exception as e:
                failed_count += 1
                errors.append(
                    PaperEvolutionErrorDTO(
                        paper_id=p.id,
                        reason="Classification or transition failed for paper.",
                    )
                )

    db.commit()
    db.refresh(opp)

    return OpportunityEvolveResponse(
        opportunity_id=opp.id,
        initial_status=initial_status,
        final_status=opp.status,
        processed_count=processed_count,
        skipped_count=skipped_count,
        failed_count=failed_count,
        events=events,
        errors=errors,
    )


@app.get("/api/v1/opportunities/{opportunity_id}/history", response_model=List[OpportunityHistoryDTO])
@app.get("/api/opportunities/{opportunity_id}/history", response_model=List[OpportunityHistoryDTO])
def get_opportunity_history(opportunity_id: uuid.UUID, db: Session = Depends(get_db)):
    """Retrieves persisted lifecycle history in chronological order."""
    opp = resolve_lifecycle_opportunity(opportunity_id, db, create_if_missing=False)
    if opp is None:
        rec = db.query(ResearchOpportunityRecord).filter(ResearchOpportunityRecord.id == opportunity_id).first()
        if rec is not None:
            return []
        raise HTTPException(status_code=404, detail=f"Opportunity {opportunity_id} not found.")

    histories = (
        db.query(OpportunityHistory)
        .filter(OpportunityHistory.opportunity_id == opp.id)
        .order_by(OpportunityHistory.created_at.asc(), OpportunityHistory.id.asc())
        .all()
    )
    return [
        OpportunityHistoryDTO(
            id=h.id,
            opportunity_id=h.opportunity_id,
            previous_status=h.previous_status,
            new_status=h.new_status,
            trigger_type=h.trigger_type,
            evidence_id=h.evidence_id,
            created_at=h.created_at.isoformat() if h.created_at else None,
        )
        for h in histories
    ]


@app.get("/api/v1/opportunities/{opportunity_id}/evidence", response_model=List[OpportunityEvidenceDTO])
@app.get("/api/opportunities/{opportunity_id}/evidence", response_model=List[OpportunityEvidenceDTO])
def get_opportunity_evidence(opportunity_id: uuid.UUID, db: Session = Depends(get_db)):
    """Retrieves persisted evidence records linked to the opportunity."""
    opp = resolve_lifecycle_opportunity(opportunity_id, db, create_if_missing=False)
    if opp is None:
        rec = db.query(ResearchOpportunityRecord).filter(ResearchOpportunityRecord.id == opportunity_id).first()
        if rec is not None:
            return []
        raise HTTPException(status_code=404, detail=f"Opportunity {opportunity_id} not found.")

    evidences = (
        db.query(OpportunityEvidence)
        .filter(OpportunityEvidence.opportunity_id == opp.id)
        .order_by(OpportunityEvidence.created_at.asc(), OpportunityEvidence.id.asc())
        .all()
    )

    ev_paper_ids = list({ev.paper_id for ev in evidences})
    papers = db.query(Paper).filter(Paper.id.in_(ev_paper_ids)).all() if ev_paper_ids else []
    paper_map = {p.id: p for p in papers}

    passages = db.query(PaperPassage).filter(PaperPassage.paper_id.in_(ev_paper_ids)).all() if ev_paper_ids else []
    passage_map: Dict[uuid.UUID, List[PaperPassage]] = {}
    for psg in passages:
        passage_map.setdefault(psg.paper_id, []).append(psg)

    # Load Stage 5 cutoffs from experiment_metadata if present to derive batch/cutoff windows
    meta_fp = db.query(ExperimentMetadataRecord).filter(ExperimentMetadataRecord.key == "configuration_fingerprint").first()
    stage_5_cutoffs: List[datetime.date] = []
    if meta_fp and meta_fp.value:
        try:
            fp_data = json.loads(meta_fp.value)
            for c in fp_data.get("stage_5_cutoffs") or []:
                try:
                    stage_5_cutoffs.append(datetime.date.fromisoformat(str(c)))
                except Exception:
                    pass
            stage_5_cutoffs.sort()
        except Exception:
            pass

    from ideagen.opportunity.service import is_evidence_grounded_for_paper

    results = []
    for ev in evidences:
        p = paper_map.get(ev.paper_id)
        p_passages = passage_map.get(ev.paper_id, [])
        is_verified = is_evidence_grounded_for_paper(ev, p, p_passages) if p else False

        pub_date = p.publication_date if p else None
        pub_date_str = str(pub_date) if pub_date else None
        batch_id = p.batch_id if p and p.batch_id else None
        eval_cutoff = None
        if pub_date and stage_5_cutoffs:
            for idx, c in enumerate(stage_5_cutoffs):
                if pub_date <= c:
                    eval_cutoff = str(c)
                    if not batch_id:
                        batch_id = f"batch_{idx}_{c}"
                    break
            if not eval_cutoff:
                eval_cutoff = str(stage_5_cutoffs[-1])
                if not batch_id:
                    batch_id = f"batch_{len(stage_5_cutoffs)-1}_{stage_5_cutoffs[-1]}"

        results.append(
            OpportunityEvidenceDTO(
                id=ev.id,
                opportunity_id=ev.opportunity_id,
                paper_id=ev.paper_id,
                paper_title=p.title if p and p.title else None,
                paper_publication_date=pub_date_str,
                batch_id=batch_id,
                evaluation_cutoff=eval_cutoff,
                relationship=ev.evidence_type,
                confidence=None,
                supporting_text=ev.supporting_text,
                verified=is_verified,
                created_at=ev.created_at.isoformat() if ev.created_at else None,
            )
        )
    return results



@app.get("/api/v1/evidence/{statement_id}", response_model=EvidenceDetailDTO)
def get_evidence_detail(statement_id: uuid.UUID, db: Session = Depends(get_db)):
    """Retrieves full evidence trace for a statement (paper, section, page, passage snippet)."""
    stmt = db.query(ResearchStatement).filter(ResearchStatement.id == statement_id).first()
    if not stmt:
        raise HTTPException(status_code=404, detail=f"Research statement {statement_id} not found.")

    paper = db.query(Paper).filter(Paper.id == stmt.paper_id).first()
    paper_title = paper.title if paper and paper.title else f"Paper {stmt.paper_id.hex[:8]}"

    passage: Optional[PaperPassage] = None
    if stmt.passage_id:
        passage = db.query(PaperPassage).filter(PaperPassage.id == stmt.passage_id).first()

    return EvidenceDetailDTO(
        statement_id=stmt.id,
        statement_text=stmt.statement_text,
        statement_type=stmt.statement_type,
        paper_id=stmt.paper_id,
        paper_title=paper_title,
        passage_id=stmt.passage_id,
        section=stmt.section or (passage.section if passage else None),
        page_start=stmt.page_start or (passage.page_start if passage else None),
        page_end=stmt.page_end or (passage.page_end if passage else None),
        passage_text=passage.text if passage else None,
    )


# ── OpenAlex Paper Ingestion Endpoint (Stage 6A) ──────────────────────────

@app.post("/api/v1/papers/ingest/openalex", response_model=OpenAlexIngestResponse)
@app.post("/api/papers/ingest/openalex", response_model=OpenAlexIngestResponse)
def ingest_openalex_paper(
    req: OpenAlexIngestRequest,
    db: Session = Depends(get_db),
):
    """Ingests a research paper from OpenAlex by Work ID or DOI.

    Deterministic ID resolution:
    - If both 'openalex_id' and 'doi' are provided, 'openalex_id' takes precedence as the
      primary canonical OpenAlex identifier. If lookup by 'openalex_id' returns no match,
      the endpoint falls back to 'doi'.
    - If neither identifier is provided, raises HTTP 400.
    - If provided identifiers cannot be normalized, raises HTTP 400.

    Persistence and evolution:
    - Constructs OpenAlexPaperIngestionService with request-scoped SQLAlchemy session and persistence layer.
    - Post-ingestion evolution runs only when enabled in system configuration
      (IDEAGEN_ENABLE_POST_INGESTION_EVOLUTION, default off). A request can opt out with
      'enable_evolution=false' but cannot enable it when the server disables it. Each run classifies
      at most IDEAGEN_POST_INGESTION_EVOLUTION_MAX_CANDIDATES eligible opportunities.
    - Preserves fail-closed NO_PDF metadata-only fallback if open-access PDF is absent, inaccessible, or invalid.
    - Rejects duplicate papers gracefully without creating orphaned or partial records.
    """
    raw_oa_id = req.openalex_id.strip() if req.openalex_id else None
    raw_doi = req.doi.strip() if req.doi else None

    if not raw_oa_id and not raw_doi:
        raise HTTPException(
            status_code=400,
            detail="Either 'openalex_id' or 'doi' must be provided.",
        )

    def _is_valid_openalex_id(raw: Optional[str]) -> bool:
        if not raw:
            return False
        norm = PaperNormalizer.normalize_openalex_id(raw)
        return norm is not None and norm.startswith("W") and norm[1:].isdigit()

    def _is_valid_doi(raw: Optional[str]) -> bool:
        if not raw:
            return False
        norm = PaperNormalizer.normalize_doi(raw)
        return norm is not None and norm.startswith("10.") and "/" in norm

    valid_oa = _is_valid_openalex_id(raw_oa_id)
    valid_doi = _is_valid_doi(raw_doi)

    if raw_oa_id and not valid_oa and not valid_doi:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid OpenAlex ID format: '{raw_oa_id}'. Must be a valid OpenAlex ID (e.g. 'W123456789') or URL.",
        )
    if raw_doi and not valid_doi and not valid_oa:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid DOI format: '{raw_doi}'. Must be a valid DOI (e.g. '10.1234/...').",
        )
    if not valid_oa and not valid_doi:
        raise HTTPException(
            status_code=400,
            detail="No valid identifier could be extracted from the request.",
        )

    norm_oa_id = PaperNormalizer.normalize_openalex_id(raw_oa_id) if valid_oa else None
    norm_doi = PaperNormalizer.normalize_doi(raw_doi) if valid_doi else None

    search_provider = OpenAlexSearchProvider(raise_on_error=True)
    results = []
    lookup_term = None

    try:
        # Deterministic precedence: openalex_id first, fallback to doi
        if norm_oa_id:
            lookup_term = norm_oa_id
            results = search_provider.search(norm_oa_id, limit=1)

        if not results and norm_doi:
            lookup_term = f"https://doi.org/{norm_doi}"
            results = search_provider.search(lookup_term, limit=1)
    except Exception as search_err:
        logger.error(f"OpenAlex metadata lookup error for {lookup_term}: {search_err}", exc_info=True)
        raise HTTPException(
            status_code=502,
            detail=f"Upstream OpenAlex service error during metadata lookup: {search_err}",
        )

    if not results or not results[0].raw_metadata:
        raise HTTPException(
            status_code=404,
            detail=f"Paper not found in OpenAlex for identifier: {lookup_term or norm_oa_id or norm_doi}",
        )

    work = results[0].raw_metadata
    persistence = SQLAlchemyPersistenceLayer(db)

    # A request may opt out of evolution, but cannot enable it when the server disables it:
    # evolution makes one synchronous LLM call per candidate opportunity.
    evolution_allowed = bool(get_settings().enable_post_ingestion_evolution)
    if req.enable_evolution and not evolution_allowed:
        logger.info("Ignoring enable_evolution=true: post-ingestion evolution is disabled by server configuration.")
    run_evolution = evolution_allowed and (req.enable_evolution is not False)

    service = OpenAlexPaperIngestionService(
        persistence=persistence,
        auto_evolve=run_evolution,
    )

    try:
        res = service.ingest_work(work)
    except DuplicateDocumentError as dup_err:
        existing_paper_id = None
        if norm_oa_id and hasattr(persistence, "get_paper_id_by_source_id"):
            existing_paper_id = persistence.get_paper_id_by_source_id(SourceType.OPENALEX_API.value, norm_oa_id)
        if not existing_paper_id and norm_doi and hasattr(persistence, "get_paper_id_by_doi"):
            existing_paper_id = persistence.get_paper_id_by_doi(norm_doi)
        if not existing_paper_id and hasattr(persistence, "get_document_by_hash"):
            import re
            m = re.search(r"content hash ([a-f0-9]+)", str(dup_err))
            if m:
                doc = persistence.get_document_by_hash(m.group(1))
                if doc:
                    existing_paper_id = doc.paper_id
        return OpenAlexIngestResponse(
            paper_id=existing_paper_id,
            source_id=work.get("id") or norm_oa_id or "UNKNOWN",
            status=IngestionStatus.SKIPPED_DUPLICATE,
            has_pdf=False,
            document_id=None,
            passage_count=0,
            error_message=str(dup_err),
            batch_id="UNKNOWN",
            doi=norm_doi or PaperNormalizer.normalize_doi(work.get("doi")),
            evolution_status=EvolutionHookStatus.SKIPPED if service.auto_evolve else EvolutionHookStatus.DISABLED,
        )
    except Exception as e:
        logger.error(f"OpenAlex ingestion failed for work {work.get('id')}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"OpenAlex ingestion pipeline error: {str(e)}",
        )

    return OpenAlexIngestResponse(
        paper_id=res.paper_id,
        source_id=res.source_id,
        status=res.status,
        has_pdf=res.has_pdf,
        document_id=res.document_id,
        passage_count=res.passage_count,
        error_message=res.error_message,
        batch_id=res.batch_id,
        doi=res.doi,
        evolution_status=res.evolution_status,
        evolution_result=_format_evolution_result(res.evolution_result),
    )


def _format_evolution_result(evo_res: Any) -> Optional[Dict[str, Any]]:
    """Safely formats IncrementalEvolutionResult or error dict into json-serializable dict."""
    if evo_res is None:
        return None
    if isinstance(evo_res, dict):
        return evo_res
    if hasattr(evo_res, "paper_id") and hasattr(evo_res, "run_status"):
        return {
            "paper_id": str(evo_res.paper_id),
            "run_status": evo_res.run_status.value if hasattr(evo_res.run_status, "value") else str(evo_res.run_status),
            "paper_title": getattr(evo_res, "paper_title", None),
            "publication_date": evo_res.publication_date.isoformat() if getattr(evo_res, "publication_date", None) else None,
            "candidate_count": getattr(evo_res, "candidate_count", 0),
            "evaluated_count": getattr(evo_res, "evaluated_count", 0),
            "skipped_count": getattr(evo_res, "skipped_count", 0),
            "grounding_rejected_count": getattr(evo_res, "grounding_rejected_count", 0),
            "failed_count": getattr(evo_res, "failed_count", 0),
            "outcomes": [
                {
                    "opportunity_id": str(o.opportunity_id),
                    "status": o.status.value if hasattr(o.status, "value") else str(o.status),
                    "initial_status": o.initial_status.value if hasattr(o.initial_status, "value") and o.initial_status else (str(o.initial_status) if o.initial_status else None),
                    "final_status": o.final_status.value if hasattr(o.final_status, "value") and o.final_status else (str(o.final_status) if o.final_status else None),
                    "relationship": o.relationship.value if hasattr(o.relationship, "value") and o.relationship else (str(o.relationship) if o.relationship else None),
                    "state_changed": o.state_changed,
                    "evidence_id": str(o.evidence_id) if o.evidence_id else None,
                    "history_id": str(o.history_id) if o.history_id else None,
                    "skip_reason": o.skip_reason,
                    "error_message": o.error_message,
                }
                for o in getattr(evo_res, "outcomes", [])
            ],
            "error": getattr(evo_res, "error", None),
        }
    return None


# ── PDF Upload & Demo Endpoints (Preserved for compatibility) ─────────────

@app.post("/api/papers/upload", response_model=SearchResponse)
async def upload_papers(files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 files allowed.")

    persistence = SQLAlchemyPersistenceLayer(db)
    service = ManualPDFIngestionService(persistence=persistence)
    results = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            extracted_text = service.extract_text(tmp_path)
            content_hash = service.compute_sha256(tmp_path)

            if persistence.document_exists_by_hash(content_hash):
                existing_doc = persistence.get_document_by_hash(content_hash)
                if existing_doc and existing_doc.paper:
                    p = existing_doc.paper
                    raw_meta = dict(p.raw_metadata or {})
                    raw_meta["content_hash"] = content_hash
                    results.append(SearchResultPaper(
                        id=f"W{p.id.hex[:10]}",
                        source="UPLOAD",
                        title=p.title or file.filename,
                        authors=p.authors or [],
                        publication_year=p.publication_date.year if p.publication_date else None,
                        publication_date=p.publication_date.isoformat() if p.publication_date else None,
                        abstract=p.abstract,
                        url=None,
                        doi=p.doi,
                        raw_metadata={"content_hash": content_hash} if not raw_meta else raw_meta
                    ))
                    continue

            abstract = extracted_text[:1000]
            unique_source_id = f"upload_{uuid.uuid4().hex}"
            pub_date = None
            now_ts = datetime.datetime.now(datetime.timezone.utc)

            paper_dto = PaperCreateDTO(
                source_type=SourceType.MANUAL_UPLOAD,
                source_id=unique_source_id,
                title=file.filename,
                publication_date=pub_date,
                batch_id="manual",
                abstract=abstract,
                authors=[],
                referenced_works=[],
                raw_metadata={"original_filename": file.filename, "content_hash": content_hash}
            )
            saved_id = persistence.save_paper(paper_dto)

            doc_dto = PaperDocumentCreateDTO(
                paper_id=saved_id,
                full_text=extracted_text,
                extraction_method=service.extraction_method,
                extraction_version=service.extraction_version,
                extracted_at=now_ts,
                content_hash=content_hash,
                original_filename=file.filename,
                acquisition_timestamp=now_ts,
                provenance_metadata={"source_type": SourceType.MANUAL_UPLOAD.value, "source_id": unique_source_id}
            )
            saved_doc_id = persistence.save_paper_document(doc_dto)

            try:
                _, block_units = service.extract_structured(tmp_path)
            except Exception:
                from ideagen.ingestion.passage_chunker import BlockUnit
                block_units = [BlockUnit(text=p.strip(), page_number=1, section=None) for p in extracted_text.split("\n\n") if p.strip()]

            passages = service.chunker.create_passages(blocks=block_units, paper_id=saved_id, document_id=saved_doc_id)
            persistence.save_passages(passages)

            paper_id_str = f"W{saved_id.hex[:10]}"
            results.append(SearchResultPaper(
                id=paper_id_str,
                source="UPLOAD",
                title=file.filename,
                authors=[],
                publication_year=pub_date.year if pub_date else None,
                publication_date=pub_date.isoformat() if pub_date else None,
                abstract=abstract,
                url=None,
                doi=None,
                raw_metadata={"id": paper_id_str, "title": file.filename, "abstract": abstract, "content_hash": content_hash}
            ))
        except PDFExtractionError:
            pass
        finally:
            if tmp_path.exists():
                os.unlink(tmp_path)

    if not results:
        raise HTTPException(status_code=400, detail="No readable PDFs found.")

    return SearchResponse(query="Upload", source="UPLOAD", count=len(results), papers=[r.model_dump() for r in results])


@app.get("/api/demo/scenario", response_model=DemoScenarioResponse)
def run_demo_scenario():
    return build_demo_scenario()


@app.get("/api/demo/opportunities", response_model=List[DemoOpportunityOverview])
def get_demo_opportunities():
    res = build_demo_scenario()
    return [DemoOpportunityOverview(
        opportunity_id=res.opportunity_id,
        title=res.title,
        problem_statement=res.problem_statement,
        status=res.final_status,
        confidence=0.9,
        supporting_papers_count=res.discovery.cluster_size,
        evidence_count=len(res.grounding.evidence)
    )]


@app.get("/api/search/papers", response_model=SearchResponse)
def search_papers(q: str, limit: int = 10):
    provider = ArxivSearchProvider()
    results = provider.search(q, limit)
    return SearchResponse(query=q, source="ARXIV", count=len(results), papers=[r.model_dump() for r in results])


@app.get("/api/v1/search/openalex", response_model=OpenAlexSearchResponse)
@app.get("/api/search/openalex", response_model=OpenAlexSearchResponse)
def search_openalex_papers(
    query: Optional[str] = Query(None, description="Topic, title, keywords, OpenAlex ID, or DOI"),
    q: Optional[str] = Query(None, description="Alternative alias for query"),
    limit: int = Query(10, description="Maximum number of results to return (1-100)"),
    start_date: Optional[str] = Query(None, description="Filter papers published on or after ISO date YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="Filter papers published on or before ISO date YYYY-MM-DD"),
):
    """Searches scientific literature using the OpenAlex Works API.

    Returns structured metadata including canonical work ID, DOI, title,
    authors, publication date, abstract, open-access status, PDF URL, and citation count.
    Does not persist results or trigger paper ingestion.
    """
    search_term = (query if query is not None else (q if q is not None else "")).strip()
    if not search_term:
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty. Please provide 'query' or 'q'.",
        )

    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid limit: {limit}. Must be between 1 and 100.",
        )

    parsed_start = None
    parsed_end = None
    if start_date is not None and start_date.strip():
        try:
            parsed_start = datetime.date.fromisoformat(start_date.strip())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid start_date format: '{start_date}'. Expected ISO format YYYY-MM-DD.",
            )

    if end_date is not None and end_date.strip():
        try:
            parsed_end = datetime.date.fromisoformat(end_date.strip())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid end_date format: '{end_date}'. Expected ISO format YYYY-MM-DD.",
            )

    if parsed_start and parsed_end and parsed_start > parsed_end:
        raise HTTPException(
            status_code=400,
            detail=f"start_date ({start_date}) cannot be later than end_date ({end_date}).",
        )

    try:
        provider = OpenAlexSearchProvider(raise_on_error=True)
        results = provider.search(
            query=search_term,
            limit=limit,
            start_date=start_date.strip() if start_date else None,
            end_date=end_date.strip() if end_date else None,
        )
    except OpenAlexSearchError as e:
        logger.error(f"Upstream OpenAlex search error for '{search_term}': {e}", exc_info=True)
        raise HTTPException(
            status_code=502,
            detail=f"Upstream OpenAlex service error: {e}",
        )
    except Exception as e:
        logger.error(f"Unexpected error during OpenAlex search for '{search_term}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal search error: {e}",
        )

    paper_dtos = [
        OpenAlexPaperResultDTO(
            id=r.id,
            doi=r.doi,
            title=r.title,
            authors=r.authors or [],
            publication_year=r.publication_year,
            publication_date=r.publication_date,
            abstract=r.abstract or "",
            is_oa=r.is_oa,
            pdf_url=r.pdf_url,
            landing_page_url=r.url,
            citation_count=r.citation_count,
        )
        for r in results
    ]

    return OpenAlexSearchResponse(
        query=search_term,
        source="OPENALEX",
        count=len(paper_dtos),
        total_count=getattr(provider, "last_total_count", None),
        papers=paper_dtos,
    )


@app.post("/api/search/analyze", response_model=DemoScenarioResponse)
def analyze_search_papers(req: AnalyzeRequest):
    raw_papers = [p.raw_metadata for p in req.papers]
    for p, raw in zip(req.papers, raw_papers):
        raw["abstract"] = p.abstract
    return build_demo_scenario(custom_raw_papers=raw_papers)
