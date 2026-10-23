import uuid
from datetime import date, datetime, timezone
from typing import Optional
import sqlalchemy
from sqlalchemy import String, Date, Text, DateTime, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB, ENUM
from sqlalchemy.sql import func, text
from ideagen.ingestion.models import SourceType

class Base(DeclarativeBase):
    pass

class Paper(Base):
    __tablename__ = "papers"
    
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    source_type: Mapped[str] = mapped_column(
        String(64).with_variant(ENUM(SourceType, name="source_type_enum", create_type=False), "postgresql")
    )
    source_id: Mapped[str] = mapped_column(String, index=True)
    doi: Mapped[str] = mapped_column(String, unique=True, nullable=True)
    title: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    publication_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    batch_id: Mapped[str] = mapped_column(String, index=True)
    abstract: Mapped[str] = mapped_column(Text)
    authors: Mapped[list] = mapped_column(JSONB().with_variant(sqlalchemy.JSON, "sqlite"))
    referenced_works: Mapped[list] = mapped_column(JSONB().with_variant(sqlalchemy.JSON, "sqlite"))
    raw_metadata: Mapped[dict] = mapped_column(JSONB().with_variant(sqlalchemy.JSON, "sqlite"))
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index(
            'idx_paper_openalex_identity', 
            'source_id', 
            unique=True, 
            postgresql_where=text("source_type IN ('OPENALEX_API', 'OPENALEX_SNAPSHOT')")
        ),
        Index(
            'idx_paper_manual_identity', 
            'source_id', 
            unique=True, 
            postgresql_where=text("source_type = 'MANUAL_UPLOAD'")
        ),
    )
    
    documents: Mapped[list["PaperDocument"]] = sqlalchemy.orm.relationship(back_populates="paper", cascade="all, delete-orphan")
    research_knowledge: Mapped[Optional["ResearchKnowledge"]] = sqlalchemy.orm.relationship(back_populates="paper", cascade="all, delete-orphan", uselist=False)

class PaperDocument(Base):
    __tablename__ = "paper_documents"
    
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    paper_id: Mapped[uuid.UUID] = mapped_column(sqlalchemy.ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    full_text: Mapped[str] = mapped_column(Text)
    extraction_method: Mapped[str] = mapped_column(String)
    extraction_version: Mapped[str] = mapped_column(String)
    extracted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    content_hash: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)
    original_filename: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    acquisition_timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    provenance_metadata: Mapped[Optional[dict]] = mapped_column(JSONB().with_variant(sqlalchemy.JSON, "sqlite"), nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    paper: Mapped["Paper"] = sqlalchemy.orm.relationship(back_populates="documents")
    passages: Mapped[list["PaperPassage"]] = sqlalchemy.orm.relationship(back_populates="document", cascade="all, delete-orphan", order_by="PaperPassage.passage_index")

class PaperPassage(Base):
    __tablename__ = "paper_passages"
    
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    paper_id: Mapped[uuid.UUID] = mapped_column(sqlalchemy.ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    document_id: Mapped[uuid.UUID] = mapped_column(sqlalchemy.ForeignKey("paper_documents.id", ondelete="CASCADE"), index=True)
    passage_index: Mapped[int] = mapped_column(sqlalchemy.Integer)
    page_start: Mapped[int] = mapped_column(sqlalchemy.Integer)
    page_end: Mapped[int] = mapped_column(sqlalchemy.Integer)
    section: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    text: Mapped[str] = mapped_column(Text)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_passage_doc_order', 'document_id', 'passage_index'),
    )
    
    document: Mapped["PaperDocument"] = sqlalchemy.orm.relationship(back_populates="passages")
    paper: Mapped["Paper"] = sqlalchemy.orm.relationship()

from ideagen.opportunity.models import OpportunityStatus, OpportunityRelationship
from sqlalchemy import Float, CheckConstraint, UniqueConstraint

class Opportunity(Base):
    __tablename__ = "opportunities"
    
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    title: Mapped[str] = mapped_column(String)
    problem_statement: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        String(64).with_variant(ENUM(OpportunityStatus, name="opportunity_status_enum", create_type=False), "postgresql")
    )
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        CheckConstraint('confidence >= 0.0 AND confidence <= 1.0', name='check_confidence_bounds'),
    )
    
    research_opportunity_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        sqlalchemy.ForeignKey("research_opportunities.id", ondelete="SET NULL"),
        nullable=True,
        unique=True,
        index=True,
    )
    gap_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        sqlalchemy.ForeignKey("research_gaps.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    evidence: Mapped[list["OpportunityEvidence"]] = relationship(back_populates="opportunity", cascade="all, delete-orphan")
    history: Mapped[list["OpportunityHistory"]] = relationship(back_populates="opportunity", cascade="all, delete-orphan")
    research_opportunity: Mapped[Optional["ResearchOpportunityRecord"]] = relationship(back_populates="lifecycle_opportunity")
    gap: Mapped[Optional["ResearchGap"]] = relationship()

class OpportunityEvidence(Base):
    __tablename__ = "opportunity_evidence"
    
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    opportunity_id: Mapped[uuid.UUID] = mapped_column(sqlalchemy.ForeignKey("opportunities.id", ondelete="CASCADE"), index=True)
    paper_id: Mapped[uuid.UUID] = mapped_column(sqlalchemy.ForeignKey("papers.id", ondelete="RESTRICT"), index=True)
    evidence_type: Mapped[str] = mapped_column(
        String(64).with_variant(ENUM(OpportunityRelationship, name="opportunity_relationship_enum", create_type=False), "postgresql")
    )
    supporting_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    __table_args__ = (
        UniqueConstraint("opportunity_id", "paper_id", name="uq_opportunity_evidence_opp_paper"),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )
    
    opportunity: Mapped["Opportunity"] = relationship(back_populates="evidence")

class OpportunityHistory(Base):
    __tablename__ = "opportunity_history"
    
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    opportunity_id: Mapped[uuid.UUID] = mapped_column(sqlalchemy.ForeignKey("opportunities.id", ondelete="CASCADE"), index=True)
    previous_status: Mapped[str] = mapped_column(
        String(64).with_variant(ENUM(OpportunityStatus, name="opportunity_status_enum", create_type=False), "postgresql")
    )
    new_status: Mapped[str] = mapped_column(
        String(64).with_variant(ENUM(OpportunityStatus, name="opportunity_status_enum", create_type=False), "postgresql")
    )
    trigger_type: Mapped[str] = mapped_column(
        String(64).with_variant(ENUM(OpportunityRelationship, name="opportunity_relationship_enum", create_type=False), "postgresql")
    )
    evidence_id: Mapped[Optional[uuid.UUID]] = mapped_column(sqlalchemy.ForeignKey("opportunity_evidence.id", ondelete="SET NULL"), nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )
    
    opportunity: Mapped["Opportunity"] = relationship(back_populates="history")


class ResearchKnowledge(Base):
    __tablename__ = "research_knowledge"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    paper_id: Mapped[uuid.UUID] = mapped_column(sqlalchemy.ForeignKey("papers.id", ondelete="CASCADE"), unique=True, index=True)
    research_problem: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    research_questions: Mapped[list] = mapped_column(JSONB().with_variant(sqlalchemy.JSON, "sqlite"), default=list)
    methodology: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    datasets: Mapped[list] = mapped_column(JSONB().with_variant(sqlalchemy.JSON, "sqlite"), default=list)
    findings: Mapped[list] = mapped_column(JSONB().with_variant(sqlalchemy.JSON, "sqlite"), default=list)
    limitations: Mapped[list] = mapped_column(JSONB().with_variant(sqlalchemy.JSON, "sqlite"), default=list)
    future_work: Mapped[list] = mapped_column(JSONB().with_variant(sqlalchemy.JSON, "sqlite"), default=list)

    model_name: Mapped[str] = mapped_column(String(255))
    model_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    extraction_schema_version: Mapped[str] = mapped_column(String(32), default="v1.0")
    extracted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    extraction_status: Mapped[str] = mapped_column(String(32), default="SUCCESS")
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[Optional[dict]] = mapped_column("metadata", JSONB().with_variant(sqlalchemy.JSON, "sqlite"), default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    paper: Mapped["Paper"] = sqlalchemy.orm.relationship(back_populates="research_knowledge")
    statements: Mapped[list["ResearchStatement"]] = sqlalchemy.orm.relationship(back_populates="knowledge", cascade="all, delete-orphan", order_by="ResearchStatement.created_at")


class ResearchStatement(Base):
    __tablename__ = "research_statements"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    knowledge_id: Mapped[uuid.UUID] = mapped_column(sqlalchemy.ForeignKey("research_knowledge.id", ondelete="CASCADE"), index=True)
    paper_id: Mapped[uuid.UUID] = mapped_column(sqlalchemy.ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    passage_id: Mapped[Optional[uuid.UUID]] = mapped_column(sqlalchemy.ForeignKey("paper_passages.id", ondelete="SET NULL"), nullable=True, index=True)
    statement_text: Mapped[str] = mapped_column(Text)
    statement_type: Mapped[str] = mapped_column(String(64), index=True)
    page_start: Mapped[Optional[int]] = mapped_column(sqlalchemy.Integer, nullable=True)
    page_end: Mapped[Optional[int]] = mapped_column(sqlalchemy.Integer, nullable=True)
    section: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(sqlalchemy.Float, nullable=True)
    metadata_json: Mapped[Optional[dict]] = mapped_column("metadata", JSONB().with_variant(sqlalchemy.JSON, "sqlite"), default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    knowledge: Mapped["ResearchKnowledge"] = sqlalchemy.orm.relationship(back_populates="statements")
    paper: Mapped["Paper"] = sqlalchemy.orm.relationship()
    passage: Mapped[Optional["PaperPassage"]] = sqlalchemy.orm.relationship()


class ResearchGap(Base):
    __tablename__ = "research_gaps"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    gap_statement: Mapped[str] = mapped_column(Text)
    research_problem: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), index=True)
    evidence_count: Mapped[int] = mapped_column(sqlalchemy.Integer, default=0)
    limitations: Mapped[list] = mapped_column(JSONB().with_variant(sqlalchemy.JSON, "sqlite"), default=list)
    future_work: Mapped[list] = mapped_column(JSONB().with_variant(sqlalchemy.JSON, "sqlite"), default=list)
    validation_reasoning: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    validation_confidence: Mapped[Optional[float]] = mapped_column(sqlalchemy.Float, nullable=True)
    metadata_json: Mapped[Optional[dict]] = mapped_column("metadata", JSONB().with_variant(sqlalchemy.JSON, "sqlite"), default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    evidence: Mapped[list["GapEvidence"]] = sqlalchemy.orm.relationship(back_populates="gap", cascade="all, delete-orphan")


class GapEvidence(Base):
    __tablename__ = "gap_evidence"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    gap_id: Mapped[uuid.UUID] = mapped_column(sqlalchemy.ForeignKey("research_gaps.id", ondelete="CASCADE"), index=True)
    statement_id: Mapped[uuid.UUID] = mapped_column(sqlalchemy.ForeignKey("research_statements.id", ondelete="CASCADE"), index=True)
    paper_id: Mapped[uuid.UUID] = mapped_column(sqlalchemy.ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    evidence_role: Mapped[str] = mapped_column(String(32), default="SUPPORTING")
    metadata_json: Mapped[Optional[dict]] = mapped_column("metadata", JSONB().with_variant(sqlalchemy.JSON, "sqlite"), default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    gap: Mapped["ResearchGap"] = sqlalchemy.orm.relationship(back_populates="evidence")
    statement: Mapped["ResearchStatement"] = sqlalchemy.orm.relationship()
    paper: Mapped["Paper"] = sqlalchemy.orm.relationship()


class ResearchOpportunityRecord(Base):
    __tablename__ = "research_opportunities"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    gap_id: Mapped[uuid.UUID] = mapped_column(sqlalchemy.ForeignKey("research_gaps.id", ondelete="CASCADE"), index=True)
    research_problem: Mapped[str] = mapped_column(Text)
    identified_gap: Mapped[str] = mapped_column(Text)
    proposed_research_direction: Mapped[str] = mapped_column(Text)
    reasoning: Mapped[str] = mapped_column(Text)
    technical_approach: Mapped[str] = mapped_column(Text)
    expected_contribution: Mapped[str] = mapped_column(Text)
    feasibility: Mapped[dict] = mapped_column(JSONB().with_variant(sqlalchemy.JSON, "sqlite"))
    novelty_assessment: Mapped[dict] = mapped_column(JSONB().with_variant(sqlalchemy.JSON, "sqlite"))
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    supporting_statement_ids: Mapped[list] = mapped_column(JSONB().with_variant(sqlalchemy.JSON, "sqlite"), default=list)
    supporting_paper_ids: Mapped[list] = mapped_column(JSONB().with_variant(sqlalchemy.JSON, "sqlite"), default=list)
    prior_art_statement_ids: Mapped[list] = mapped_column(JSONB().with_variant(sqlalchemy.JSON, "sqlite"), default=list)
    status: Mapped[str] = mapped_column(String(64), index=True)

    generation_model: Mapped[str] = mapped_column(String(255))
    validation_model: Mapped[str] = mapped_column(String(255))
    embedding_model: Mapped[str] = mapped_column(String(255))
    refinement_iteration_count: Mapped[int] = mapped_column(sqlalchemy.Integer, default=0)
    pipeline_version: Mapped[str] = mapped_column(String(32), default="v1.0")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    gap: Mapped["ResearchGap"] = sqlalchemy.orm.relationship()
    assessments: Mapped[list["PriorArtAssessmentRecord"]] = sqlalchemy.orm.relationship(back_populates="opportunity", cascade="all, delete-orphan")
    iterations: Mapped[list["OpportunityRefinementRecord"]] = sqlalchemy.orm.relationship(back_populates="opportunity", cascade="all, delete-orphan")
    lifecycle_opportunity: Mapped[Optional["Opportunity"]] = sqlalchemy.orm.relationship(back_populates="research_opportunity", uselist=False)


class PriorArtAssessmentRecord(Base):
    __tablename__ = "prior_art_assessments"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    opportunity_id: Mapped[uuid.UUID] = mapped_column(sqlalchemy.ForeignKey("research_opportunities.id", ondelete="CASCADE"), index=True)
    iteration: Mapped[int] = mapped_column(sqlalchemy.Integer)
    assessment: Mapped[str] = mapped_column(String(64))
    overlapping_statement_ids: Mapped[list] = mapped_column(JSONB().with_variant(sqlalchemy.JSON, "sqlite"), default=list)
    overlapping_paper_ids: Mapped[list] = mapped_column(JSONB().with_variant(sqlalchemy.JSON, "sqlite"), default=list)
    reasoning: Mapped[str] = mapped_column(Text)
    differentiation: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    opportunity: Mapped["ResearchOpportunityRecord"] = sqlalchemy.orm.relationship(back_populates="assessments")


class OpportunityRefinementRecord(Base):
    __tablename__ = "opportunity_refinement_iterations"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    opportunity_id: Mapped[uuid.UUID] = mapped_column(sqlalchemy.ForeignKey("research_opportunities.id", ondelete="CASCADE"), index=True)
    iteration: Mapped[int] = mapped_column(sqlalchemy.Integer)
    original_direction: Mapped[str] = mapped_column(Text)
    critique: Mapped[str] = mapped_column(Text)
    revised_direction: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(64))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    opportunity: Mapped["ResearchOpportunityRecord"] = sqlalchemy.orm.relationship(back_populates="iterations")


class ExperimentMetadataRecord(Base):
    """Stores key-value experiment metadata such as configuration fingerprints."""
    __tablename__ = "experiment_metadata"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[str] = mapped_column(Text)


class ResearchProjectRecord(Base):
    """Stores personal workspace research projects."""
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    topic: Mapped[str] = mapped_column(Text)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    temporal_cutoff: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="ACTIVE")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    selections: Mapped[list["ProjectPaperSelectionRecord"]] = sqlalchemy.orm.relationship(back_populates="project", cascade="all, delete-orphan")
    runs: Mapped[list["ResearchRunRecord"]] = sqlalchemy.orm.relationship(back_populates="project", cascade="all, delete-orphan")
    papers: Mapped[list["ProjectPaperRecord"]] = sqlalchemy.orm.relationship(back_populates="project", cascade="all, delete-orphan")

    def __init__(self, **kwargs):
        if "title" in kwargs and "name" not in kwargs:
            kwargs["name"] = kwargs.pop("title")
        if "topic" not in kwargs:
            kwargs["topic"] = kwargs.get("name", "")
        super().__init__(**kwargs)

    @property
    def title(self) -> str:
        return self.name

    @title.setter
    def title(self, value: str):
        self.name = value



class ProjectPaperRecord(Base):
    """Associates a corpus Paper with a project."""
    __tablename__ = "project_papers"

    project_id: Mapped[str] = mapped_column(String(64), sqlalchemy.ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True, index=True)
    paper_id: Mapped[uuid.UUID] = mapped_column(sqlalchemy.ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True, index=True)
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    paper: Mapped["Paper"] = sqlalchemy.orm.relationship()
    project: Mapped["ResearchProjectRecord"] = sqlalchemy.orm.relationship(back_populates="papers")



class ProjectPaperSelectionRecord(Base):
    """Stores persisted user paper selections for a research project."""
    __tablename__ = "project_paper_selections"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(String(64), sqlalchemy.ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    openalex_id: Mapped[str] = mapped_column(String(64), index=True)
    doi: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    title: Mapped[str] = mapped_column(Text)
    authors: Mapped[list] = mapped_column(JSONB().with_variant(sqlalchemy.JSON, "sqlite"), default=list)
    publication_date: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    pdf_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    landing_page_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    citation_count: Mapped[Optional[int]] = mapped_column(sqlalchemy.Integer, nullable=True)
    abstract: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    selected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    project: Mapped["ResearchProjectRecord"] = sqlalchemy.orm.relationship(back_populates="selections")

    def __init__(self, **kwargs):
        if "id" not in kwargs or kwargs["id"] is None:
            kwargs["id"] = f"psel_{uuid.uuid4().hex[:12]}"
        elif not isinstance(kwargs["id"], str):
            kwargs["id"] = str(kwargs["id"])
        pub_yr = kwargs.pop("publication_year", None)
        if pub_yr and "publication_date" not in kwargs:
            kwargs["publication_date"] = str(pub_yr)
        super().__init__(**kwargs)

    @property
    def publication_year(self) -> Optional[int]:
        if self.publication_date:
            try:
                return int(self.publication_date[:4])
            except Exception:
                return None
        return None

    @publication_year.setter
    def publication_year(self, val: Optional[int]):
        if val is not None and not self.publication_date:
            self.publication_date = str(val)



class ResearchRunRecord(Base):
    """Stores persistent research run executions, stage states, and summaries."""
    __tablename__ = "research_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(String(64), sqlalchemy.ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True, default="PENDING")
    current_stage: Mapped[str] = mapped_column(String(64), default="INGESTION")
    planned_stages: Mapped[list] = mapped_column(JSONB().with_variant(sqlalchemy.JSON, "sqlite"), default=list)
    stage_states: Mapped[dict] = mapped_column(JSONB().with_variant(sqlalchemy.JSON, "sqlite"), default=dict)
    summary: Mapped[Optional[dict]] = mapped_column(JSONB().with_variant(sqlalchemy.JSON, "sqlite"), nullable=True)
    logs: Mapped[list] = mapped_column(JSONB().with_variant(sqlalchemy.JSON, "sqlite"), default=list)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    project: Mapped["ResearchProjectRecord"] = sqlalchemy.orm.relationship(back_populates="runs")
    papers: Mapped[list["RunPaperRecord"]] = sqlalchemy.orm.relationship(back_populates="run", cascade="all, delete-orphan")

    def __init__(self, **kwargs):
        if "id" not in kwargs or kwargs["id"] is None:
            kwargs["id"] = f"run_{uuid.uuid4().hex[:12]}"
        elif not isinstance(kwargs["id"], str):
            kwargs["id"] = str(kwargs["id"])
        super().__init__(**kwargs)

    @property
    def stage_progress(self) -> dict:
        if self.stage_states and "stages" in self.stage_states:
            return self.stage_states
        # Transform stage_states dict to standard stage list if needed
        stages = []
        for s in (self.planned_stages or []):
            st_info = (self.stage_states or {}).get(s, {})
            stages.append({
                "stage_id": s,
                "name": s.replace("_", " ").title(),
                "status": st_info.get("status", "PENDING"),
                "message": st_info.get("message") or (st_info.get("details") or {}).get("error"),
            })
        return {"stages": stages, "logs": self.logs or []}

    @stage_progress.setter
    def stage_progress(self, val: dict):
        self.stage_states = val


class RunPaperRecord(Base):
    """Stores per-paper processing state within a research run."""
    __tablename__ = "research_run_papers"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), sqlalchemy.ForeignKey("research_runs.id", ondelete="CASCADE"), index=True)
    openalex_id: Mapped[str] = mapped_column(String(64), index=True)
    doi: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    title: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="PENDING")
    paper_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    passage_count: Mapped[int] = mapped_column(sqlalchemy.Integer, default=0)
    statement_count: Mapped[int] = mapped_column(sqlalchemy.Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    run: Mapped["ResearchRunRecord"] = sqlalchemy.orm.relationship(back_populates="papers")

    def __init__(self, **kwargs):
        if "id" not in kwargs or kwargs["id"] is None:
            kwargs["id"] = f"rp_{uuid.uuid4().hex[:12]}"
        elif not isinstance(kwargs["id"], str):
            kwargs["id"] = str(kwargs["id"])
        super().__init__(**kwargs)


    @property
    def passages_count(self) -> int:
        return self.passage_count or 0

    @passages_count.setter
    def passages_count(self, v: int):
        self.passage_count = v

    @property
    def statements_count(self) -> int:
        return self.statement_count or 0

    @statements_count.setter
    def statements_count(self, v: int):
        self.statement_count = v

    @property
    def corpus_paper_id(self) -> Optional[str]:
        return self.paper_id

    @corpus_paper_id.setter
    def corpus_paper_id(self, v: Optional[str]):
        self.paper_id = v

    @property
    def has_pdf(self) -> bool:
        return (self.passage_count or 0) > 0
