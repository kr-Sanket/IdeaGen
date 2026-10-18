import uuid
from datetime import date, datetime
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
from sqlalchemy import Float, CheckConstraint

class Opportunity(Base):
    __tablename__ = "opportunities"
    
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
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
    
    evidence: Mapped[list["OpportunityEvidence"]] = relationship(back_populates="opportunity", cascade="all, delete-orphan")
    history: Mapped[list["OpportunityHistory"]] = relationship(back_populates="opportunity", cascade="all, delete-orphan")

class OpportunityEvidence(Base):
    __tablename__ = "opportunity_evidence"
    
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    opportunity_id: Mapped[uuid.UUID] = mapped_column(sqlalchemy.ForeignKey("opportunities.id", ondelete="CASCADE"), index=True)
    paper_id: Mapped[uuid.UUID] = mapped_column(sqlalchemy.ForeignKey("papers.id", ondelete="RESTRICT"), index=True)
    evidence_type: Mapped[str] = mapped_column(
        String(64).with_variant(ENUM(OpportunityRelationship, name="opportunity_relationship_enum", create_type=False), "postgresql")
    )
    supporting_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
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
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
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


