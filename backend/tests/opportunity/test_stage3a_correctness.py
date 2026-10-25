"""Focused regression tests for Stage 3A Correctness Hardening.

Verifies:
1. Database-level unique constraint on OpportunityEvidence(opportunity_id, paper_id).
2. Uniqueness conflicts handled idempotently without session corruption.
3. Unrelated integrity errors propagate properly without being hidden.
4. Old addressing evidence not counting toward resolution after contradiction reopening.
5. Two new distinct addressing papers required for re-resolution.
6. Replay determinism when run more than once with isolated=True.
"""

from __future__ import annotations

import datetime
import uuid
import pytest
from sqlalchemy import create_engine, select, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from ideagen.db.models import (
    Base,
    Opportunity,
    Paper,
    PaperPassage,
    OpportunityEvidence,
    OpportunityHistory,
)
from ideagen.opportunity.models import OpportunityRelationship, OpportunityStatus
from ideagen.opportunity.service import OpportunityTransitionService
from ideagen.opportunity.classifier import (
    RelationshipClassifier,
    RelationshipClassification,
)
from ideagen.opportunity.integration import OpportunityClassificationIntegration
from ideagen.opportunity.temporal import (
    OpportunityEvolutionService,
    TemporalBatch,
)


@pytest.fixture
def isolated_db():
    """In-memory SQLite database with foreign keys and unique constraints enforced."""
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = session_factory()
    yield session
    session.close()


def _create_test_paper(
    session: Session,
    title: str,
    doi: str,
    text: str = "Evidence excerpt solving the target problem.",
    pub_date: datetime.date = datetime.date(2023, 1, 1),
) -> Paper:
    paper_id = uuid.uuid4()
    p = Paper(
        id=paper_id,
        source_type="TEST",
        source_id=f"src_{paper_id.hex[:8]}",
        doi=doi,
        title=title,
        publication_date=pub_date,
        batch_id="batch-1",
        abstract="Abstract of paper.",
        authors=["Author A"],
        referenced_works=[],
        raw_metadata={},
    )
    session.add(p)
    session.flush()

    from ideagen.db.models import PaperDocument
    doc_id = uuid.uuid4()
    doc = PaperDocument(
        id=doc_id,
        paper_id=paper_id,
        full_text="Document full text",
        extraction_method="pdf_test",
        extraction_version="v1.0",
        extracted_at=datetime.datetime.now(datetime.timezone.utc),
    )
    session.add(doc)
    session.flush()

    passage = PaperPassage(
        id=uuid.uuid4(),
        paper_id=paper_id,
        document_id=doc_id,
        passage_index=0,
        page_start=1,
        page_end=1,
        section="METHODS",
        text=text,
    )
    session.add(passage)
    session.flush()
    return p


def _create_test_opportunity(
    session: Session,
    status: OpportunityStatus = OpportunityStatus.OPEN,
) -> Opportunity:
    opp = Opportunity(
        id=uuid.uuid4(),
        title="Scalable Oversight Opportunity",
        problem_statement="How to oversee superhuman models?",
        description="Scalable oversight remains an open scientific challenge.",
        status=status.value,
        confidence=0.85,
    )
    session.add(opp)
    session.flush()
    return opp


# ── 1. Database-Level Unique Constraint ───────────────────────────────────────

def test_duplicate_evidence_rejected_at_db_level(isolated_db):
    """Direct database insertion of duplicate (opportunity_id, paper_id) raises IntegrityError."""
    opp = _create_test_opportunity(isolated_db)
    paper = _create_test_paper(isolated_db, "Paper 1", "10.1000/1")

    ev1 = OpportunityEvidence(
        id=uuid.uuid4(),
        opportunity_id=opp.id,
        paper_id=paper.id,
        evidence_type=OpportunityRelationship.SUPPORTS.value,
        supporting_text="Evidence 1",
    )
    isolated_db.add(ev1)
    isolated_db.flush()

    # Attempt second insertion with same opportunity_id and paper_id
    ev2 = OpportunityEvidence(
        id=uuid.uuid4(),
        opportunity_id=opp.id,
        paper_id=paper.id,
        evidence_type=OpportunityRelationship.ADDRESSES.value,
        supporting_text="Evidence 2",
    )
    isolated_db.add(ev2)

    with pytest.raises(IntegrityError):
        isolated_db.flush()

    isolated_db.rollback()


# ── 2. Idempotent Conflict Handling & Session Usability ──────────────────────

def test_duplicate_conflicts_handled_idempotently_without_session_corruption(isolated_db):
    """When a uniqueness conflict occurs, apply_relationship handles it cleanly."""
    opp = _create_test_opportunity(isolated_db)
    paper = _create_test_paper(isolated_db, "Paper 1", "10.1000/1", "addresses problem directly")
    service = OpportunityTransitionService(isolated_db)

    # First application succeeds
    service.apply_relationship(
        opp.id, paper.id, OpportunityRelationship.ADDRESSES, "addresses problem directly"
    )
    isolated_db.flush()

    # Second application raises ValueError as an idempotent duplicate conflict
    with pytest.raises(ValueError, match="already exists"):
        service.apply_relationship(
            opp.id, paper.id, OpportunityRelationship.ADDRESSES, "addresses problem directly"
        )

    # Session must remain clean and usable for subsequent queries and operations
    paper2 = _create_test_paper(isolated_db, "Paper 2", "10.1000/2", "addresses problem independently")
    service.apply_relationship(
        opp.id, paper2.id, OpportunityRelationship.ADDRESSES, "addresses problem independently"
    )
    isolated_db.flush()

    assert opp.status == OpportunityStatus.RESOLVED.value


def test_concurrent_duplicate_after_precheck_is_reported_as_duplicate(isolated_db):
    """A row inserted by a concurrent writer after the pre-check must surface as a duplicate,
    not as UnboundLocalError, and leave the session usable."""
    opp = _create_test_opportunity(isolated_db)
    paper = _create_test_paper(isolated_db, "Paper 1", "10.1000/1", "addresses problem directly")
    service = OpportunityTransitionService(isolated_db)
    inserted = {"done": False}

    # The passage lookup runs after the duplicate pre-check and before the savepoint,
    # which is exactly the window a concurrent writer would race into.
    @event.listens_for(isolated_db, "do_orm_execute")
    def _race_in_concurrent_evidence(orm_execute_state):
        if inserted["done"] or not orm_execute_state.is_select:
            return
        if PaperPassage not in [d["entity"] for d in orm_execute_state.statement.column_descriptions]:
            return
        inserted["done"] = True
        orm_execute_state.session.connection().execute(
            OpportunityEvidence.__table__.insert().values(
                id=uuid.uuid4(),
                opportunity_id=opp.id,
                paper_id=paper.id,
                evidence_type=OpportunityRelationship.SUPPORTS.value,
                supporting_text="concurrent writer",
                created_at=datetime.datetime.now(datetime.timezone.utc),
            )
        )

    with pytest.raises(ValueError, match="already exists"):
        service.apply_relationship(
            opp.id, paper.id, OpportunityRelationship.ADDRESSES, "addresses problem directly"
        )
    event.remove(isolated_db, "do_orm_execute", _race_in_concurrent_evidence)

    assert inserted["done"]
    rows = isolated_db.execute(
        select(OpportunityEvidence).where(OpportunityEvidence.opportunity_id == opp.id)
    ).scalars().all()
    assert [r.supporting_text for r in rows] == ["concurrent writer"]
    assert opp.status == OpportunityStatus.OPEN.value

    # Session remains usable after the savepoint rollback.
    paper2 = _create_test_paper(isolated_db, "Paper 2", "10.1000/2", "supports problem framing")
    service.apply_relationship(
        opp.id, paper2.id, OpportunityRelationship.SUPPORTS, "supports problem framing"
    )
    isolated_db.flush()


def test_integrity_error_from_unrelated_pending_state_propagates(isolated_db):
    """A conflict in state pending before the call is flushed by begin_nested() and must
    propagate as IntegrityError (previously this path raised UnboundLocalError)."""
    opp = _create_test_opportunity(isolated_db)
    paper_a = _create_test_paper(isolated_db, "Paper A", "10.1000/a", "supports problem framing")
    paper_b = _create_test_paper(isolated_db, "Paper B", "10.1000/b", "supports problem framing")
    isolated_db.add(OpportunityEvidence(
        opportunity_id=opp.id, paper_id=paper_a.id,
        evidence_type=OpportunityRelationship.SUPPORTS.value, supporting_text="first",
    ))
    isolated_db.flush()
    # Pending duplicate for paper A, not yet flushed.
    isolated_db.add(OpportunityEvidence(
        opportunity_id=opp.id, paper_id=paper_a.id,
        evidence_type=OpportunityRelationship.SUPPORTS.value, supporting_text="pending duplicate",
    ))

    service = OpportunityTransitionService(isolated_db)
    with pytest.raises(IntegrityError):
        service.apply_relationship(
            opp.id, paper_b.id, OpportunityRelationship.SUPPORTS, "supports problem framing"
        )
    isolated_db.rollback()


def test_unrelated_integrity_error_propagates_without_being_hidden(isolated_db):
    """An IntegrityError from an invalid foreign key or other failure is not hidden."""
    opp = _create_test_opportunity(isolated_db)
    service = OpportunityTransitionService(isolated_db)
    non_existent_paper_id = uuid.uuid4()

    with pytest.raises(ValueError, match="Paper .* not found"):
        service.apply_relationship(
            opp.id, non_existent_paper_id, OpportunityRelationship.SUPPORTS
        )


# ── 3. Post-Contradiction Resolution Rule ────────────────────────────────────

def test_old_addressing_evidence_does_not_count_after_contradiction_reopening(isolated_db):
    """Old addressing papers prior to RESOLVED -> OPEN contradiction must not count toward re-resolution."""
    opp = _create_test_opportunity(isolated_db, status=OpportunityStatus.OPEN)
    p1 = _create_test_paper(isolated_db, "Paper A", "10.1001/a", "evidence solving oversight")
    p2 = _create_test_paper(isolated_db, "Paper B", "10.1001/b", "evidence solving oversight")
    p_critique = _create_test_paper(isolated_db, "Critique", "10.1001/critique", "claims of oversight solving are flawed")
    p3 = _create_test_paper(isolated_db, "Paper C", "10.1001/c", "new certified oversight method")

    service = OpportunityTransitionService(isolated_db)

    # Step 1: Resolve opportunity with P1 and P2
    service.apply_relationship(opp.id, p1.id, OpportunityRelationship.ADDRESSES, "evidence solving oversight")
    service.apply_relationship(opp.id, p2.id, OpportunityRelationship.ADDRESSES, "evidence solving oversight")
    isolated_db.flush()
    assert opp.status == OpportunityStatus.RESOLVED.value

    # Step 2: Contradiction re-opens RESOLVED to OPEN
    service.apply_relationship(opp.id, p_critique.id, OpportunityRelationship.CONTRADICTS, "claims of oversight solving are flawed")
    isolated_db.flush()
    assert opp.status == OpportunityStatus.OPEN.value

    # Verify history recorded the reopening
    histories = isolated_db.execute(
        select(OpportunityHistory)
        .where(OpportunityHistory.opportunity_id == opp.id)
        .order_by(OpportunityHistory.created_at.desc())
    ).scalars().all()
    assert len(histories) >= 2
    reopen_hist = histories[0]
    assert reopen_hist.previous_status == OpportunityStatus.RESOLVED.value
    assert reopen_hist.new_status == OpportunityStatus.OPEN.value
    assert reopen_hist.trigger_type == OpportunityRelationship.CONTRADICTS.value

    # Step 3: First new addressing paper after reopening
    # Under the corrected rule, P1 and P2 MUST NOT count toward this post-reopening evolution.
    # Therefore, addressing_paper_count is 1, and the opportunity must transition to PARTIALLY_ADDRESSED,
    # NOT jump to RESOLVED.
    service.apply_relationship(opp.id, p3.id, OpportunityRelationship.ADDRESSES, "new certified oversight method")
    isolated_db.flush()
    assert opp.status == OpportunityStatus.PARTIALLY_ADDRESSED.value

    # Verify all 3 addressing evidence records (P1, P2, P3) remain in history/evidence table
    all_evidence = isolated_db.execute(
        select(OpportunityEvidence)
        .where(
            OpportunityEvidence.opportunity_id == opp.id,
            OpportunityEvidence.evidence_type == OpportunityRelationship.ADDRESSES.value,
        )
    ).scalars().all()
    assert len(all_evidence) == 3


def test_two_new_distinct_addressing_papers_required_for_reresolution(isolated_db):
    """Exactly two distinct new addressing papers post-reopening are required to re-resolve."""
    opp = _create_test_opportunity(isolated_db, status=OpportunityStatus.OPEN)
    p1 = _create_test_paper(isolated_db, "Paper A", "10.1001/a", "evidence solving oversight")
    p2 = _create_test_paper(isolated_db, "Paper B", "10.1001/b", "evidence solving oversight")
    p_critique = _create_test_paper(isolated_db, "Critique", "10.1001/critique", "claims of oversight solving are flawed")
    p3 = _create_test_paper(isolated_db, "Paper C", "10.1001/c", "new certified oversight method C")
    p4 = _create_test_paper(isolated_db, "Paper D", "10.1001/d", "new certified oversight method D")

    service = OpportunityTransitionService(isolated_db)

    # Initial resolution: P1 + P2 -> RESOLVED
    service.apply_relationship(opp.id, p1.id, OpportunityRelationship.ADDRESSES, "evidence solving oversight")
    service.apply_relationship(opp.id, p2.id, OpportunityRelationship.ADDRESSES, "evidence solving oversight")
    isolated_db.flush()
    assert opp.status == OpportunityStatus.RESOLVED.value

    # Contradiction: RESOLVED -> OPEN
    service.apply_relationship(opp.id, p_critique.id, OpportunityRelationship.CONTRADICTS, "claims of oversight solving are flawed")
    isolated_db.flush()
    assert opp.status == OpportunityStatus.OPEN.value

    # 1st new paper post-reopening: OPEN -> PARTIALLY_ADDRESSED
    service.apply_relationship(opp.id, p3.id, OpportunityRelationship.ADDRESSES, "new certified oversight method C")
    isolated_db.flush()
    assert opp.status == OpportunityStatus.PARTIALLY_ADDRESSED.value

    # 2nd new paper post-reopening: PARTIALLY_ADDRESSED -> RESOLVED
    service.apply_relationship(opp.id, p4.id, OpportunityRelationship.ADDRESSES, "new certified oversight method D")
    isolated_db.flush()
    assert opp.status == OpportunityStatus.RESOLVED.value


# ── 4. Replay Determinism & Isolation ────────────────────────────────────────

def test_isolated_replay_determinism_multiple_runs(isolated_db):
    """Calling replay(..., isolated=True) multiple times yields identical results without DB mutation."""
    opp = _create_test_opportunity(isolated_db, status=OpportunityStatus.OPEN)
    p1 = _create_test_paper(isolated_db, "Paper 1", "10.2001/1", "solving problem in 2024", pub_date=datetime.date(2024, 2, 1))
    p2 = _create_test_paper(isolated_db, "Paper 2", "10.2001/2", "solving problem in 2024", pub_date=datetime.date(2024, 8, 1))

    class MockProvider:
        def generate_structured(self, prompt, schema):
            return RelationshipClassification(
                relationship=OpportunityRelationship.ADDRESSES,
                confidence=0.9,
                rationale="Addresses problem.",
                evidence_excerpt="solving problem in 2024",
                prompt_version="1.0.0",
            )

    classifier = RelationshipClassifier(provider=MockProvider())
    transition_svc = OpportunityTransitionService(isolated_db)
    integration = OpportunityClassificationIntegration(classifier, transition_svc, isolated_db)
    evolution_svc = OpportunityEvolutionService(integration)

    b1 = TemporalBatch("batch-1", datetime.date(2024, 1, 1), datetime.date(2024, 6, 30), 1)
    b2 = TemporalBatch("batch-2", datetime.date(2024, 7, 1), datetime.date(2024, 12, 31), 2)
    batches = [b1, b2]
    papers = [p1, p2]

    # Run 1: isolated replay
    results1 = evolution_svc.replay(isolated_db, opp.id, batches, papers, isolated=True)
    assert len(results1) == 2
    assert results1[0].final_status == OpportunityStatus.PARTIALLY_ADDRESSED
    assert results1[1].final_status == OpportunityStatus.RESOLVED

    # Ensure DB was NOT mutated
    isolated_db.refresh(opp)
    assert opp.status == OpportunityStatus.OPEN.value
    ev_count = isolated_db.execute(
        select(OpportunityEvidence).where(OpportunityEvidence.opportunity_id == opp.id)
    ).scalars().all()
    assert len(ev_count) == 0

    # Run 2: isolated replay produces identical results
    results2 = evolution_svc.replay(isolated_db, opp.id, batches, papers, isolated=True)
    assert len(results2) == 2
    assert results2[0].final_status == OpportunityStatus.PARTIALLY_ADDRESSED
    assert results2[1].final_status == OpportunityStatus.RESOLVED

    # Run 3: out-of-order batches are still deterministically processed chronologically
    results3 = evolution_svc.replay(isolated_db, opp.id, [b2, b1], papers, isolated=True)
    assert results3[0].events[0].paper_id == p1.id
    assert results3[1].events[0].paper_id == p2.id
    assert results3[0].final_status == OpportunityStatus.PARTIALLY_ADDRESSED
    assert results3[1].final_status == OpportunityStatus.RESOLVED
