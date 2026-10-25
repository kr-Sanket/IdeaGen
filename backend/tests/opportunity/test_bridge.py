"""Tests for Opportunity Bridge Service (Checkpoint 7).

Verifies:
- A validated record creates an OPEN lifecycle entity.
- Repeating the bridge creates no duplicate (idempotent).
- Repeating it after a lifecycle status change does not reset that status.
- Ineligible generation statuses are rejected or safely ignored.
- Missing optional source fields are handled safely.
- Confidence bounds are clamped to [0.0, 1.0].
- DTO and SQLAlchemy record inputs are both supported.
- OpportunityPersistenceLayer integration bridges validated opportunities atomically.
- Schema, foreign key, uniqueness, and SET NULL cascade behavior.
"""

import pytest
import uuid
from datetime import datetime
from sqlalchemy import create_engine, select, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

from ideagen.db.models import (
    Base,
    Opportunity,
    OpportunityHistory,
    ResearchGap,
    ResearchOpportunityRecord,
)
from ideagen.opportunity.models import OpportunityStatus as LifecycleStatus
from ideagen.generation.models import (
    ResearchOpportunityDTO,
    OpportunityStatus as GenerationStatus,
    FeasibilityAssessment,
    NoveltyAssessment,
)
from ideagen.opportunity.bridge import OpportunityBridgeService
from ideagen.generation.persistence import OpportunityPersistenceLayer


@pytest.fixture
def db_session():
    """Isolated SQLite in-memory test database with foreign keys enabled."""
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    SessionMaker = sessionmaker(bind=engine)
    session = SessionMaker()
    yield session
    session.close()


def _create_test_gap(session, gap_id=None):
    gap = ResearchGap(
        id=gap_id or uuid.uuid4(),
        gap_statement="Lack of temporal consistency checks in streaming agents",
        research_problem="Temporal incoherence across multi-step execution",
        status="VALIDATED",
        evidence_count=2,
    )
    session.add(gap)
    session.commit()
    return gap


def _create_test_opportunity_record(
    session,
    gap_id,
    record_id=None,
    status="VALIDATED",
    confidence=0.88,
    proposed_direction="Develop temporal anchoring graphs for streaming LLM agents",
):
    rec = ResearchOpportunityRecord(
        id=record_id or uuid.uuid4(),
        gap_id=gap_id,
        research_problem="Temporal incoherence across multi-step execution",
        identified_gap="Lack of temporal consistency checks in streaming agents",
        proposed_research_direction=proposed_direction,
        reasoning="Multi-step tool executions drift temporally without anchored constraints.",
        technical_approach="Construct a dynamic DAG anchoring temporal assertions.",
        expected_contribution="Provable reduction in temporal hallucination rates.",
        feasibility={"required_compute": "1x GPU", "implementation_complexity": "Moderate"},
        novelty_assessment={"assessment": "Potentially novel within the analyzed corpus"},
        confidence=confidence,
        supporting_statement_ids=[str(uuid.uuid4())],
        supporting_paper_ids=[str(uuid.uuid4())],
        prior_art_statement_ids=[],
        status=status,
        generation_model="qwen3:8b",
        validation_model="qwen3:8b",
        embedding_model="nomic-embed-text",
        refinement_iteration_count=0,
        pipeline_version="v1.0",
    )
    session.add(rec)
    session.commit()
    return rec


def test_bridge_validated_record_creates_open_entity(db_session):
    gap = _create_test_gap(db_session)
    rec = _create_test_opportunity_record(db_session, gap.id, status="VALIDATED", confidence=0.88)

    bridge_svc = OpportunityBridgeService(db_session)
    lifecycle_opp = bridge_svc.bridge_opportunity(rec)

    assert lifecycle_opp is not None
    assert lifecycle_opp.research_opportunity_id == rec.id
    assert lifecycle_opp.gap_id == gap.id
    assert lifecycle_opp.status == LifecycleStatus.OPEN.value
    assert lifecycle_opp.confidence == pytest.approx(0.88)
    assert lifecycle_opp.title == rec.proposed_research_direction
    assert lifecycle_opp.problem_statement == rec.research_problem
    assert "Identified Gap:" in lifecycle_opp.description
    assert "Technical Approach:" in lifecycle_opp.description
    assert "Expected Contribution:" in lifecycle_opp.description

    # Verify bidirectional SQLAlchemy relationship navigation
    db_session.commit()
    retrieved_opp = db_session.get(Opportunity, lifecycle_opp.id)
    assert retrieved_opp is not None
    assert retrieved_opp.research_opportunity.id == rec.id
    assert retrieved_opp.gap.id == gap.id
    assert rec.lifecycle_opportunity.id == retrieved_opp.id


def test_bridge_refined_record_creates_open_entity(db_session):
    gap = _create_test_gap(db_session)
    rec = _create_test_opportunity_record(db_session, gap.id, status="REFINED", confidence=0.75)

    bridge_svc = OpportunityBridgeService(db_session)
    lifecycle_opp = bridge_svc.bridge_opportunity(rec)

    assert lifecycle_opp is not None
    assert lifecycle_opp.status == LifecycleStatus.OPEN.value
    assert lifecycle_opp.research_opportunity_id == rec.id


def test_bridge_idempotency_no_duplicates(db_session):
    gap = _create_test_gap(db_session)
    rec = _create_test_opportunity_record(db_session, gap.id)

    bridge_svc = OpportunityBridgeService(db_session)
    opp1 = bridge_svc.bridge_opportunity(rec)
    db_session.commit()

    opp2 = bridge_svc.bridge_opportunity(rec)
    db_session.commit()

    assert opp1.id == opp2.id

    # Verify exactly one row in the database
    rows = db_session.execute(
        select(Opportunity).where(Opportunity.research_opportunity_id == rec.id)
    ).scalars().all()
    assert len(rows) == 1


def test_bridge_preserves_lifecycle_status_and_history(db_session):
    gap = _create_test_gap(db_session)
    rec = _create_test_opportunity_record(db_session, gap.id)

    bridge_svc = OpportunityBridgeService(db_session)
    lifecycle_opp = bridge_svc.bridge_opportunity(rec)
    db_session.commit()

    # Simulate downstream lifecycle evolution transition to STRENGTHENED
    lifecycle_opp.status = LifecycleStatus.STRENGTHENED.value
    history_entry = OpportunityHistory(
        opportunity_id=lifecycle_opp.id,
        previous_status=LifecycleStatus.OPEN.value,
        new_status=LifecycleStatus.STRENGTHENED.value,
        trigger_type="SUPPORTS",
    )
    db_session.add(history_entry)
    db_session.commit()

    # Re-invoking the bridge for the same research opportunity record MUST NOT reset status
    rebridged = bridge_svc.bridge_opportunity(rec)
    db_session.commit()

    assert rebridged.id == lifecycle_opp.id
    assert rebridged.status == LifecycleStatus.STRENGTHENED.value
    assert len(rebridged.history) == 1
    assert rebridged.history[0].new_status == LifecycleStatus.STRENGTHENED.value


def test_bridge_ineligible_statuses_safely_ignored_or_rejected(db_session):
    gap = _create_test_gap(db_session)
    bridge_svc = OpportunityBridgeService(db_session)

    # 1. DRAFT status -> ignored by default (returns None)
    draft_rec = _create_test_opportunity_record(db_session, gap.id, status="DRAFT")
    result_draft = bridge_svc.bridge_opportunity(draft_rec)
    assert result_draft is None

    # 2. REJECTED_INSUFFICIENTLY_DIFFERENTIATED status -> ignored by default
    rejected_rec = _create_test_opportunity_record(
        db_session,
        gap.id,
        status="REJECTED_INSUFFICIENTLY_DIFFERENTIATED",
    )
    result_rejected = bridge_svc.bridge_opportunity(rejected_rec)
    assert result_rejected is None

    # 3. With raise_on_ineligible=True -> raises ValueError
    with pytest.raises(ValueError, match="ineligible status"):
        bridge_svc.bridge_opportunity(draft_rec, raise_on_ineligible=True)

    with pytest.raises(ValueError, match="ineligible status"):
        bridge_svc.bridge_opportunity(rejected_rec, raise_on_ineligible=True)

    # Ensure no rows were added to opportunities table
    total_opps = db_session.execute(select(Opportunity)).scalars().all()
    assert len(total_opps) == 0


def test_bridge_missing_optional_fields_handled_safely(db_session):
    gap = _create_test_gap(db_session)
    # Create record with minimal / empty optional fields
    rec = ResearchOpportunityRecord(
        id=uuid.uuid4(),
        gap_id=gap.id,
        research_problem="Bare problem",
        identified_gap="",
        proposed_research_direction="",
        reasoning="",
        technical_approach="",
        expected_contribution="",
        feasibility={},
        novelty_assessment={},
        confidence=None,
        supporting_statement_ids=[],
        supporting_paper_ids=[],
        prior_art_statement_ids=[],
        status="VALIDATED",
        generation_model="test",
        validation_model="test",
        embedding_model="test",
        refinement_iteration_count=0,
        pipeline_version="v1.0",
    )
    db_session.add(rec)
    db_session.commit()

    bridge_svc = OpportunityBridgeService(db_session)
    lifecycle_opp = bridge_svc.bridge_opportunity(rec)

    assert lifecycle_opp is not None
    assert lifecycle_opp.title == "Research Opportunity"  # Clean fallback
    assert lifecycle_opp.problem_statement == "Bare problem"
    assert lifecycle_opp.description == "Bare problem"
    assert lifecycle_opp.confidence == 0.0
    assert lifecycle_opp.status == LifecycleStatus.OPEN.value


def test_bridge_confidence_bounds_clamping(db_session):
    gap = _create_test_gap(db_session)
    bridge_svc = OpportunityBridgeService(db_session)

    # Upper bound clamping (> 1.0)
    rec_high = _create_test_opportunity_record(db_session, gap.id, confidence=1.8)
    opp_high = bridge_svc.bridge_opportunity(rec_high)
    assert opp_high.confidence == 1.0

    # Lower bound clamping (< 0.0)
    rec_low = _create_test_opportunity_record(db_session, gap.id, confidence=-0.5)
    opp_low = bridge_svc.bridge_opportunity(rec_low)
    assert opp_low.confidence == 0.0


def test_bridge_dto_support(db_session):
    gap = _create_test_gap(db_session)
    dto = ResearchOpportunityDTO(
        id=uuid.uuid4(),
        gap_id=gap.id,
        research_problem="Problem from DTO",
        identified_gap="Gap from DTO",
        proposed_research_direction="Direction from DTO",
        reasoning="Reasoning from DTO",
        technical_approach="Approach from DTO",
        expected_contribution="Contribution from DTO",
        feasibility=FeasibilityAssessment(
            required_data="Data",
            required_compute="Compute",
            implementation_complexity="Low",
            experimental_setup="Setup",
        ),
        novelty_assessment=NoveltyAssessment(
            assessment="Potentially novel within the analyzed corpus",
            reasoning="Reasoning",
            differentiation="Differentiation",
        ),
        confidence=0.92,
        supporting_statement_ids=[uuid.uuid4()],
        supporting_paper_ids=[uuid.uuid4()],
        prior_art_statement_ids=[],
        status=GenerationStatus.VALIDATED,
        created_at=datetime.utcnow(),
        generation_model="test",
        validation_model="test",
        embedding_model="test",
        refinement_iteration_count=0,
        pipeline_version="v1.0",
    )

    p_layer = OpportunityPersistenceLayer(db_session)
    p_layer.save_opportunity(dto, bridge=False)

    bridge_svc = OpportunityBridgeService(db_session)
    lifecycle_opp = bridge_svc.bridge_opportunity(dto)

    assert lifecycle_opp is not None
    assert lifecycle_opp.research_opportunity_id == dto.id
    assert lifecycle_opp.title == "Direction from DTO"
    assert lifecycle_opp.confidence == pytest.approx(0.92)
    assert lifecycle_opp.status == LifecycleStatus.OPEN.value


def test_opportunity_persistence_layer_automatic_bridge(db_session):
    gap = _create_test_gap(db_session)
    dto = ResearchOpportunityDTO(
        id=uuid.uuid4(),
        gap_id=gap.id,
        research_problem="Integrated problem",
        identified_gap="Integrated gap",
        proposed_research_direction="Integrated direction",
        reasoning="Integrated reasoning",
        technical_approach="Integrated approach",
        expected_contribution="Integrated contribution",
        feasibility=FeasibilityAssessment(
            required_data="Data",
            required_compute="Compute",
            implementation_complexity="Low",
            experimental_setup="Setup",
        ),
        novelty_assessment=NoveltyAssessment(
            assessment="Potentially novel within the analyzed corpus",
            reasoning="Reasoning",
            differentiation="Differentiation",
        ),
        confidence=0.84,
        supporting_statement_ids=[uuid.uuid4()],
        supporting_paper_ids=[uuid.uuid4()],
        prior_art_statement_ids=[],
        status=GenerationStatus.VALIDATED,
        created_at=datetime.utcnow(),
        generation_model="test",
        validation_model="test",
        embedding_model="test",
        refinement_iteration_count=0,
        pipeline_version="v1.0",
    )

    p_layer = OpportunityPersistenceLayer(db_session)
    saved_record = p_layer.save_opportunity(dto, bridge=True)

    # Opportunity lifecycle entity should have been created automatically
    lifecycle_opp = db_session.execute(
        select(Opportunity).where(Opportunity.research_opportunity_id == saved_record.id)
    ).scalar_one_or_none()

    assert lifecycle_opp is not None
    assert lifecycle_opp.status == LifecycleStatus.OPEN.value
    assert lifecycle_opp.research_opportunity_id == saved_record.id


def test_opportunity_persistence_layer_bridge_disabled(db_session):
    gap = _create_test_gap(db_session)
    dto = ResearchOpportunityDTO(
        id=uuid.uuid4(),
        gap_id=gap.id,
        research_problem="No bridge problem",
        identified_gap="No bridge gap",
        proposed_research_direction="No bridge direction",
        reasoning="No bridge reasoning",
        technical_approach="No bridge approach",
        expected_contribution="No bridge contribution",
        feasibility=FeasibilityAssessment(
            required_data="Data",
            required_compute="Compute",
            implementation_complexity="Low",
            experimental_setup="Setup",
        ),
        novelty_assessment=NoveltyAssessment(
            assessment="Potentially novel within the analyzed corpus",
            reasoning="Reasoning",
            differentiation="Differentiation",
        ),
        confidence=0.80,
        supporting_statement_ids=[uuid.uuid4()],
        supporting_paper_ids=[uuid.uuid4()],
        prior_art_statement_ids=[],
        status=GenerationStatus.VALIDATED,
        created_at=datetime.utcnow(),
        generation_model="test",
        validation_model="test",
        embedding_model="test",
        refinement_iteration_count=0,
        pipeline_version="v1.0",
    )

    p_layer = OpportunityPersistenceLayer(db_session)
    saved_record = p_layer.save_opportunity(dto, bridge=False)

    lifecycle_opp = db_session.execute(
        select(Opportunity).where(Opportunity.research_opportunity_id == saved_record.id)
    ).scalar_one_or_none()

    assert lifecycle_opp is None


def test_uniqueness_constraint_enforced_for_source_record(db_session):
    gap = _create_test_gap(db_session)
    rec = _create_test_opportunity_record(db_session, gap.id)

    # First lifecycle opportunity
    opp1 = Opportunity(
        id=uuid.uuid4(),
        research_opportunity_id=rec.id,
        gap_id=gap.id,
        title="Title 1",
        problem_statement="Problem 1",
        description="Desc 1",
        status=LifecycleStatus.OPEN.value,
        confidence=0.5,
    )
    db_session.add(opp1)
    db_session.commit()

    # Second lifecycle opportunity with identical research_opportunity_id must violate unique constraint
    opp2 = Opportunity(
        id=uuid.uuid4(),
        research_opportunity_id=rec.id,
        gap_id=gap.id,
        title="Title 2",
        problem_statement="Problem 2",
        description="Desc 2",
        status=LifecycleStatus.OPEN.value,
        confidence=0.6,
    )
    db_session.add(opp2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_foreign_key_set_null_on_source_deletion(db_session):
    gap = _create_test_gap(db_session)
    rec = _create_test_opportunity_record(db_session, gap.id)

    bridge_svc = OpportunityBridgeService(db_session)
    lifecycle_opp = bridge_svc.bridge_opportunity(rec)
    db_session.commit()

    opp_id = lifecycle_opp.id

    # Delete source ResearchOpportunityRecord
    db_session.delete(rec)
    db_session.commit()

    # Opportunity must still exist, with research_opportunity_id set to NULL
    surviving_opp = db_session.get(Opportunity, opp_id)
    assert surviving_opp is not None
    assert surviving_opp.research_opportunity_id is None


def test_schema_model_metadata():
    """Verify table schema metadata for opportunities contains expected columns and constraints."""
    table = Base.metadata.tables["opportunities"]
    assert "research_opportunity_id" in table.columns
    assert "gap_id" in table.columns

    col_ro = table.columns["research_opportunity_id"]
    assert col_ro.nullable is True
    assert col_ro.unique is True

    col_gap = table.columns["gap_id"]
    assert col_gap.nullable is True

    # Check foreign keys
    fk_targets = {fk.target_fullname for fk in table.foreign_keys}
    assert "research_opportunities.id" in fk_targets
    assert "research_gaps.id" in fk_targets


def test_alembic_migration_graph():
    """Verify that Alembic head is the new bridge revision and migration chain is unbroken."""
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    from pathlib import Path

    config = Config(str(Path(__file__).resolve().parents[2] / "alembic.ini"))
    script = ScriptDirectory.from_config(config)

    heads = script.get_heads()
    assert heads == ["8b2c3d4e5f6a"], f"Expected single head, got {heads}"
    assert script.get_revision("8b2c3d4e5f6a").down_revision == "7a1b2c3d4e5f"
    assert script.get_revision("7a1b2c3d4e5f").down_revision == "e7b3c9a14f52"
    assert script.get_revision("e7b3c9a14f52").down_revision == "d5e8a71b2c40"
    assert script.get_revision("d5e8a71b2c40").down_revision == "b4e67f1a9c23"

    bridge_rev = script.get_revision("b4e67f1a9c23")
    assert bridge_rev is not None
    assert bridge_rev.down_revision == "a95f51a7e212"


# ── Atomicity of generation persistence + lifecycle bridging ────────────────


@pytest.fixture(params=[True, False], ids=["autoflush", "no_autoflush_like_production"])
def atomic_session(request):
    """Isolated SQLite session; also run with autoflush=False, matching ideagen.db.session."""
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=request.param)()
    yield session
    session.close()


def _make_generation_dto(gap_id, direction="Atomic direction"):
    return ResearchOpportunityDTO(
        id=uuid.uuid4(),
        gap_id=gap_id,
        research_problem="Atomic problem",
        identified_gap="Atomic gap",
        proposed_research_direction=direction,
        reasoning="Atomic reasoning",
        technical_approach="Atomic approach",
        expected_contribution="Atomic contribution",
        feasibility=FeasibilityAssessment(
            required_data="Data",
            required_compute="Compute",
            implementation_complexity="Low",
            experimental_setup="Setup",
        ),
        novelty_assessment=NoveltyAssessment(
            assessment="Potentially novel within the analyzed corpus",
            reasoning="Reasoning",
            differentiation="Differentiation",
        ),
        confidence=0.81,
        supporting_statement_ids=[uuid.uuid4()],
        supporting_paper_ids=[uuid.uuid4()],
        prior_art_statement_ids=[],
        status=GenerationStatus.VALIDATED,
        created_at=datetime.utcnow(),
        generation_model="test",
        validation_model="test",
        embedding_model="test",
        refinement_iteration_count=1,
        pipeline_version="v1.0",
    )


def _row_counts(session):
    from ideagen.db.models import PriorArtAssessmentRecord, OpportunityRefinementRecord

    return {
        "research_opportunities": session.query(ResearchOpportunityRecord).count(),
        "lifecycle_opportunities": session.query(Opportunity).count(),
        "prior_art": session.query(PriorArtAssessmentRecord).count(),
        "refinements": session.query(OpportunityRefinementRecord).count(),
    }


def test_bridge_failure_before_insert_rolls_back_generation_record(atomic_session, monkeypatch):
    """A bridge error must not leave an unbridged generation record (or its children) committed."""
    from ideagen.generation.models import PriorArtAssessment

    session = atomic_session
    gap = _create_test_gap(session)
    p_layer = OpportunityPersistenceLayer(session)
    earlier = p_layer.save_opportunity(_make_generation_dto(gap.id, "Earlier direction"), bridge=True)
    before = _row_counts(session)

    def failing_bridge(self, record, session=None, raise_on_ineligible=False):
        raise RuntimeError("bridge unavailable")

    monkeypatch.setattr(OpportunityBridgeService, "bridge_opportunity", failing_bridge)
    dto = _make_generation_dto(gap.id)
    with pytest.raises(RuntimeError, match="bridge unavailable"):
        p_layer.save_opportunity(
            dto,
            assessments=[(0, PriorArtAssessment(assessment="LOW_OVERLAP", reasoning="r", differentiation="d"))],
            iterations=[(1, "orig", "critique", "revised", "REFINED")],
            bridge=True,
        )

    session.expire_all()
    assert _row_counts(session) == before
    assert session.get(ResearchOpportunityRecord, dto.id) is None
    assert session.get(ResearchOpportunityRecord, earlier.id) is not None

    # Session remains usable: a later save with a working bridge succeeds and is bridged.
    monkeypatch.undo()
    later = p_layer.save_opportunity(_make_generation_dto(gap.id, "Later direction"), bridge=True)
    assert session.execute(
        select(Opportunity).where(Opportunity.research_opportunity_id == later.id)
    ).scalar_one().status == LifecycleStatus.OPEN.value


def test_database_failure_during_bridge_flush_rolls_back_and_session_recovers(atomic_session, monkeypatch):
    """A real IntegrityError while flushing the bridge must propagate, persist nothing new,
    and leave the session usable (no PendingRollbackError on the next save)."""
    from types import SimpleNamespace
    import ideagen.opportunity.bridge as bridge_module

    session = atomic_session
    gap = _create_test_gap(session)
    p_layer = OpportunityPersistenceLayer(session)
    existing = p_layer.save_opportunity(_make_generation_dto(gap.id, "Existing direction"), bridge=True)
    existing_lifecycle = session.execute(
        select(Opportunity).where(Opportunity.research_opportunity_id == existing.id)
    ).scalar_one()
    before = _row_counts(session)

    # Force the bridge to reuse an existing Opportunity primary key -> genuine IntegrityError on flush.
    monkeypatch.setattr(bridge_module, "uuid", SimpleNamespace(uuid4=lambda: existing_lifecycle.id))
    dto = _make_generation_dto(gap.id, "Colliding direction")
    with pytest.raises(IntegrityError):
        p_layer.save_opportunity(dto, bridge=True)
    monkeypatch.undo()

    session.expire_all()
    assert _row_counts(session) == before
    assert session.get(ResearchOpportunityRecord, dto.id) is None
    unchanged = session.get(Opportunity, existing_lifecycle.id)
    assert unchanged.research_opportunity_id == existing.id
    assert unchanged.status == LifecycleStatus.OPEN.value

    recovered = p_layer.save_opportunity(_make_generation_dto(gap.id, "Recovered direction"), bridge=True)
    assert session.get(ResearchOpportunityRecord, recovered.id) is not None
    assert _row_counts(session)["lifecycle_opportunities"] == before["lifecycle_opportunities"] + 1


def test_successful_bridge_and_bridge_disabled_persist_expected_rows(atomic_session):
    session = atomic_session
    gap = _create_test_gap(session)
    p_layer = OpportunityPersistenceLayer(session)

    bridged = p_layer.save_opportunity(_make_generation_dto(gap.id, "Bridged direction"), bridge=True)
    lifecycle = session.execute(
        select(Opportunity).where(Opportunity.research_opportunity_id == bridged.id)
    ).scalar_one()
    assert lifecycle.status == LifecycleStatus.OPEN.value
    assert lifecycle.gap_id == gap.id
    assert lifecycle.title == "Bridged direction"
    assert lifecycle.confidence == pytest.approx(0.81)

    unbridged = p_layer.save_opportunity(_make_generation_dto(gap.id, "Unbridged direction"), bridge=False)
    assert session.get(ResearchOpportunityRecord, unbridged.id) is not None
    assert session.execute(
        select(Opportunity).where(Opportunity.research_opportunity_id == unbridged.id)
    ).scalar_one_or_none() is None

    # Saving an ineligible (rejected) record with bridge=True commits it without a lifecycle entity.
    rejected_dto = _make_generation_dto(gap.id, "Rejected direction")
    rejected_dto.status = GenerationStatus.REJECTED_INSUFFICIENTLY_DIFFERENTIATED
    rejected = p_layer.save_opportunity(rejected_dto, bridge=True)
    assert session.get(ResearchOpportunityRecord, rejected.id) is not None
    assert session.execute(
        select(Opportunity).where(Opportunity.research_opportunity_id == rejected.id)
    ).scalar_one_or_none() is None
