"""Tests for Phase 5 SQLAlchemy Persistence (SQLite / Postgres)."""

import pytest
import uuid
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ideagen.db.models import Base, ResearchGap, ResearchOpportunityRecord, PriorArtAssessmentRecord
from ideagen.generation.models import (
    ResearchOpportunityDTO,
    OpportunityStatus,
    FeasibilityAssessment,
    NoveltyAssessment,
    PriorArtAssessment,
)
from ideagen.generation.persistence import OpportunityPersistenceLayer


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionMaker = sessionmaker(bind=engine)
    session = SessionMaker()
    yield session
    session.close()


def test_persistence_save_and_retrieve_opportunity(db_session):
    # Create parent ResearchGap record
    gap_id = uuid.uuid4()
    gap_record = ResearchGap(
        id=gap_id,
        gap_statement="Unquantified hallucination rates in RAG",
        research_problem="Multi-hop error accumulation",
        status="VALIDATED",
        evidence_count=1,
    )
    db_session.add(gap_record)
    db_session.commit()

    s_id = uuid.uuid4()
    p_id = uuid.uuid4()

    feasibility = FeasibilityAssessment(
        required_data="Data",
        required_compute="Compute",
        implementation_complexity="Med",
        experimental_setup="Setup",
    )
    novelty = NoveltyAssessment(
        assessment="Potentially novel within the analyzed corpus",
        reasoning="Reasoning",
        differentiation="Differentiation",
    )

    dto = ResearchOpportunityDTO(
        id=uuid.uuid4(),
        gap_id=gap_id,
        research_problem="Problem",
        identified_gap="Gap",
        proposed_research_direction="Direction",
        reasoning="Reasoning",
        technical_approach="Approach",
        expected_contribution="Contribution",
        feasibility=feasibility,
        novelty_assessment=novelty,
        confidence=0.85,
        supporting_statement_ids=[s_id],
        supporting_paper_ids=[p_id],
        prior_art_statement_ids=[],
        status=OpportunityStatus.VALIDATED,
        created_at=datetime.utcnow(),
        generation_model="gemini-2.5-flash",
        validation_model="gemini-2.5-flash",
        embedding_model="gemini-embedding-001",
        refinement_iteration_count=0,
        pipeline_version="v1.0",
    )

    assessment = PriorArtAssessment(
        assessment="LOW_OVERLAP",
        overlapping_statement_ids=[],
        overlapping_paper_ids=[],
        reasoning="No overlap found",
        differentiation="Clear separation",
        confidence=0.90,
    )

    persistence = OpportunityPersistenceLayer(db_session)
    record = persistence.save_opportunity(
        dto=dto,
        assessments=[(0, assessment)],
        iterations=[],
    )

    assert record.id == dto.id
    assert record.gap_id == gap_id
    assert record.status == OpportunityStatus.VALIDATED.value

    retrieved = persistence.get_opportunity_by_id(dto.id)
    assert retrieved is not None
    assert retrieved.research_problem == "Problem"
    assert len(retrieved.assessments) == 1
    assert retrieved.assessments[0].assessment == "LOW_OVERLAP"
