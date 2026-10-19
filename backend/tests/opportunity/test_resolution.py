"""Tests for Deterministic Multi-Paper Lifecycle Resolution (Checkpoint 9).

Verifies:
- One ADDRESSES paper results in PARTIALLY_ADDRESSED, never RESOLVED.
- Two distinct, grounded ADDRESSES papers transition the opportunity to RESOLVED.
- Duplicate paper replay does not increase the distinct-paper count or inflate history.
- Ungrounded evidence cannot contribute to resolution.
- Canonical paper identity deduplication (by DOI, normalized title, or source) prevents
  pseudo-multi-paper resolution from preprint/published duplicates.
- Additional ADDRESSES on an already RESOLVED opportunity leaves it RESOLVED without duplicate history.
- CONTRADICTS re-opens RESOLVED to OPEN with history recorded.
- Re-opened opportunities require two new grounded papers to re-resolve.
- SUPPORTS, CONTRADICTS, EXTENDS, and UNRELATED preserve their intended semantics.
- Transactional consistency between evidence creation and status transitions.
"""

import datetime
import uuid
import pytest
from sqlalchemy import create_engine, select, event
from sqlalchemy.orm import sessionmaker

from ideagen.db.models import (
    Base,
    Opportunity,
    Paper,
    PaperPassage,
    OpportunityEvidence,
    OpportunityHistory,
)
from ideagen.opportunity.models import OpportunityRelationship, OpportunityStatus
from ideagen.opportunity.state_machine import transition
from ideagen.opportunity.service import (
    OpportunityTransitionService,
    get_paper_canonical_identity,
)
from ideagen.opportunity.grounding import UngroundedEvidenceError
from ideagen.opportunity.classifier import (
    RelationshipClassifier,
    RelationshipClassification,
    RelationshipClassificationInput,
)
from ideagen.opportunity.integration import OpportunityClassificationIntegration
from ideagen.opportunity.temporal import (
    OpportunityEvolutionService,
    TemporalBatch,
)
from tests.opportunity.fake_provider import FakeLLMProvider


@pytest.fixture
def db_session():
    """Isolated in-memory SQLite database with foreign keys enabled."""
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


def _create_paper(
    session,
    title: str,
    abstract: str = "",
    doi: str = None,
    source_type: str = "MANUAL_UPLOAD",
    source_id: str = None,
    publication_date: datetime.date = None,
    passages: list[str] = None,
) -> Paper:
    paper_id = uuid.uuid4()
    p = Paper(
        id=paper_id,
        source_type=source_type,
        source_id=source_id or str(paper_id),
        doi=doi,
        title=title,
        publication_date=publication_date or datetime.date(2023, 1, 1),
        batch_id="batch-1",
        abstract=abstract,
        authors=["Author A", "Author B"],
        referenced_works=[],
        raw_metadata={},
    )
    session.add(p)
    session.flush()

    if passages:
        from ideagen.db.models import PaperDocument
        doc = PaperDocument(
            id=uuid.uuid4(),
            paper_id=paper_id,
            full_text="Document full text",
            extraction_method="pdf_test",
            extraction_version="v1.0",
            extracted_at=datetime.datetime.now(datetime.timezone.utc),
        )
        session.add(doc)
        session.flush()

        for idx, text in enumerate(passages):
            psg = PaperPassage(
                id=uuid.uuid4(),
                paper_id=paper_id,
                document_id=doc.id,
                passage_index=idx,
                page_start=idx + 1,
                page_end=idx + 1,
                section="RESULTS",
                text=text,
            )
            session.add(psg)
        session.flush()

    return p


def _create_opportunity(
    session,
    title: str = "Test Research Opportunity",
    status: OpportunityStatus = OpportunityStatus.OPEN,
) -> Opportunity:
    opp = Opportunity(
        id=uuid.uuid4(),
        title=title,
        problem_statement="Investigating scalable alignment in large neural models.",
        description="Detailed opportunity description.",
        status=status.value,
        confidence=0.8,
    )
    session.add(opp)
    session.flush()
    return opp


# ── State Machine Pure Unit Tests ──────────────────────────────────────────


def test_state_machine_addresses_single_paper_never_resolves():
    """Single ADDRESSES paper must never resolve an opportunity regardless of starting state."""
    assert transition(OpportunityStatus.OPEN, OpportunityRelationship.ADDRESSES, addressing_paper_count=1) == OpportunityStatus.PARTIALLY_ADDRESSED
    assert transition(OpportunityStatus.STRENGTHENED, OpportunityRelationship.ADDRESSES, addressing_paper_count=1) == OpportunityStatus.PARTIALLY_ADDRESSED
    assert transition(OpportunityStatus.PARTIALLY_ADDRESSED, OpportunityRelationship.ADDRESSES, addressing_paper_count=1) == OpportunityStatus.PARTIALLY_ADDRESSED


def test_state_machine_addresses_multi_paper_resolves():
    """Two or more addressing papers transition to RESOLVED from PARTIALLY_ADDRESSED."""
    assert transition(OpportunityStatus.PARTIALLY_ADDRESSED, OpportunityRelationship.ADDRESSES, addressing_paper_count=2) == OpportunityStatus.RESOLVED
    assert transition(OpportunityStatus.PARTIALLY_ADDRESSED, OpportunityRelationship.ADDRESSES, addressing_paper_count=3) == OpportunityStatus.RESOLVED
    # Single step from OPEN always yields PARTIALLY_ADDRESSED first
    assert transition(OpportunityStatus.OPEN, OpportunityRelationship.ADDRESSES, addressing_paper_count=2) == OpportunityStatus.PARTIALLY_ADDRESSED


def test_state_machine_backwards_compatibility():
    """Default addressing_paper_count=1 preserves all existing state machine behavior."""
    assert transition(OpportunityStatus.OPEN, OpportunityRelationship.ADDRESSES) == OpportunityStatus.PARTIALLY_ADDRESSED
    assert transition(OpportunityStatus.PARTIALLY_ADDRESSED, OpportunityRelationship.ADDRESSES) == OpportunityStatus.PARTIALLY_ADDRESSED
    assert transition(OpportunityStatus.RESOLVED, OpportunityRelationship.ADDRESSES) == OpportunityStatus.RESOLVED


# ── Canonical Paper Identity Deduplication ──────────────────────────────────


def test_canonical_identity_doi_normalized(db_session):
    p1 = _create_paper(db_session, title="Paper Alpha", doi="10.1000/182")
    p2 = _create_paper(db_session, title="Paper Alpha (v2)", doi=" 10.1000/182 ")
    assert get_paper_canonical_identity(p1) == get_paper_canonical_identity(p2) == "doi:10.1000/182"


def test_canonical_identity_title_normalized(db_session):
    p1 = _create_paper(db_session, title="Attention Is All You Need", doi=None)
    p2 = _create_paper(db_session, title="attention is all you need!", doi=None)
    assert get_paper_canonical_identity(p1) == get_paper_canonical_identity(p2) == "title:attention is all you need"


def test_canonical_identity_distinct_papers(db_session):
    p1 = _create_paper(db_session, title="Deep Residual Learning", doi="10.1000/res")
    p2 = _create_paper(db_session, title="Dense Convolutional Networks", doi="10.1000/dense")
    assert get_paper_canonical_identity(p1) != get_paper_canonical_identity(p2)


# ── Lifecycle Transitions with Service ─────────────────────────────────────


def test_one_addresses_paper_results_in_partially_addressed_never_resolved(db_session):
    opp = _create_opportunity(db_session, status=OpportunityStatus.OPEN)
    p1 = _create_paper(
        db_session,
        title="Paper Alpha",
        passages=["We successfully present an architecture that addresses neural model alignment."],
    )

    service = OpportunityTransitionService(db_session)
    service.apply_relationship(
        opportunity_id=opp.id,
        paper_id=p1.id,
        relationship=OpportunityRelationship.ADDRESSES,
        evidence_excerpt="architecture that addresses neural model alignment",
    )
    db_session.flush()

    assert opp.status == OpportunityStatus.PARTIALLY_ADDRESSED.value

    # Verify history
    history = db_session.execute(
        select(OpportunityHistory).where(OpportunityHistory.opportunity_id == opp.id)
    ).scalars().all()
    assert len(history) == 1
    assert history[0].previous_status == OpportunityStatus.OPEN.value
    assert history[0].new_status == OpportunityStatus.PARTIALLY_ADDRESSED.value
    assert history[0].trigger_type == OpportunityRelationship.ADDRESSES.value


def test_two_distinct_grounded_addresses_papers_transition_to_resolved(db_session):
    opp = _create_opportunity(db_session, status=OpportunityStatus.OPEN)
    p1 = _create_paper(
        db_session,
        title="First Study on Alignment",
        doi="10.1001/first",
        passages=["We solve the neural alignment issue through constitutional training."],
    )
    p2 = _create_paper(
        db_session,
        title="Second Study on Alignment",
        doi="10.1002/second",
        passages=["In an independent evaluation, we confirm solving neural alignment with preference ranking."],
    )

    service = OpportunityTransitionService(db_session)

    # 1. First paper transitions OPEN -> PARTIALLY_ADDRESSED
    service.apply_relationship(
        opportunity_id=opp.id,
        paper_id=p1.id,
        relationship=OpportunityRelationship.ADDRESSES,
        evidence_excerpt="solve the neural alignment issue through constitutional training",
    )
    db_session.flush()
    assert opp.status == OpportunityStatus.PARTIALLY_ADDRESSED.value

    # 2. Second distinct paper transitions PARTIALLY_ADDRESSED -> RESOLVED
    service.apply_relationship(
        opportunity_id=opp.id,
        paper_id=p2.id,
        relationship=OpportunityRelationship.ADDRESSES,
        evidence_excerpt="solving neural alignment with preference ranking",
    )
    db_session.flush()
    assert opp.status == OpportunityStatus.RESOLVED.value

    # Verify complete history progression
    history = db_session.execute(
        select(OpportunityHistory)
        .where(OpportunityHistory.opportunity_id == opp.id)
        .order_by(OpportunityHistory.created_at.asc())
    ).scalars().all()
    assert len(history) == 2
    assert history[0].previous_status == OpportunityStatus.OPEN.value
    assert history[0].new_status == OpportunityStatus.PARTIALLY_ADDRESSED.value
    assert history[1].previous_status == OpportunityStatus.PARTIALLY_ADDRESSED.value
    assert history[1].new_status == OpportunityStatus.RESOLVED.value


def test_duplicate_paper_does_not_inflate_count_or_resolve(db_session):
    opp = _create_opportunity(db_session, status=OpportunityStatus.OPEN)

    # Paper 1: ArXiv preprint (without DOI)
    p1 = _create_paper(
        db_session,
        title="Scalable Alignment of Neural Networks",
        doi=None,
        passages=["We address the alignment challenge using recursive critique."],
    )
    # Paper 2: Journal publication with same normalized title (different UUID and source_id)
    p2 = _create_paper(
        db_session,
        title="scalable alignment of neural networks!",
        doi=None,
        passages=["We address the alignment challenge using recursive critique."],
    )

    service = OpportunityTransitionService(db_session)
    service.apply_relationship(
        opportunity_id=opp.id,
        paper_id=p1.id,
        relationship=OpportunityRelationship.ADDRESSES,
        evidence_excerpt="address the alignment challenge using recursive critique",
    )
    db_session.flush()
    assert opp.status == OpportunityStatus.PARTIALLY_ADDRESSED.value

    # Applying duplicate paper representation
    service.apply_relationship(
        opportunity_id=opp.id,
        paper_id=p2.id,
        relationship=OpportunityRelationship.ADDRESSES,
        evidence_excerpt="address the alignment challenge using recursive critique",
    )
    db_session.flush()

    # Must NOT transition to RESOLVED because distinct count is still 1
    assert opp.status == OpportunityStatus.PARTIALLY_ADDRESSED.value

    # History should have exactly 1 record (no state change on duplicate)
    history = db_session.execute(
        select(OpportunityHistory).where(OpportunityHistory.opportunity_id == opp.id)
    ).scalars().all()
    assert len(history) == 1


def test_ungrounded_evidence_cannot_contribute_to_resolution(db_session):
    opp = _create_opportunity(db_session, status=OpportunityStatus.OPEN)
    p1 = _create_paper(
        db_session,
        title="Grounded Paper",
        passages=["We address the alignment problem through supervised fine-tuning."],
    )
    p2 = _create_paper(
        db_session,
        title="Paper With Unrelated Passages",
        passages=["We investigate protein folding and thermodynamic stability."],
    )

    service = OpportunityTransitionService(db_session)
    service.apply_relationship(
        opportunity_id=opp.id,
        paper_id=p1.id,
        relationship=OpportunityRelationship.ADDRESSES,
        evidence_excerpt="address the alignment problem through supervised fine-tuning",
    )
    db_session.flush()
    assert opp.status == OpportunityStatus.PARTIALLY_ADDRESSED.value

    # Attempting to apply ungrounded excerpt on paper 2 fails closed
    with pytest.raises(UngroundedEvidenceError):
        service.apply_relationship(
            opportunity_id=opp.id,
            paper_id=p2.id,
            relationship=OpportunityRelationship.ADDRESSES,
            evidence_excerpt="completely solve scalable alignment in deep networks",
        )

    # Opportunity remains PARTIALLY_ADDRESSED
    assert opp.status == OpportunityStatus.PARTIALLY_ADDRESSED.value


def test_already_resolved_stays_resolved_without_duplicate_history(db_session):
    opp = _create_opportunity(db_session, status=OpportunityStatus.RESOLVED)
    p3 = _create_paper(
        db_session,
        title="Third Valid Paper",
        passages=["We also address and confirm the alignment solution."],
    )

    service = OpportunityTransitionService(db_session)
    service.apply_relationship(
        opportunity_id=opp.id,
        paper_id=p3.id,
        relationship=OpportunityRelationship.ADDRESSES,
        evidence_excerpt="address and confirm the alignment solution",
    )
    db_session.flush()

    assert opp.status == OpportunityStatus.RESOLVED.value

    # No history added because status did not change
    history = db_session.execute(
        select(OpportunityHistory).where(OpportunityHistory.opportunity_id == opp.id)
    ).scalars().all()
    assert len(history) == 0

    # Evidence row was successfully persisted
    evidence = db_session.execute(
        select(OpportunityEvidence).where(OpportunityEvidence.opportunity_id == opp.id)
    ).scalars().all()
    assert len(evidence) == 1


def test_contradicts_reopens_resolved_opportunity_to_open_with_history(db_session):
    opp = _create_opportunity(db_session, status=OpportunityStatus.RESOLVED)
    p_refute = _create_paper(
        db_session,
        title="Refutation Study",
        passages=["We show that prior alignment solutions are flawed and do not hold under adversarial attacks."],
    )

    service = OpportunityTransitionService(db_session)
    service.apply_relationship(
        opportunity_id=opp.id,
        paper_id=p_refute.id,
        relationship=OpportunityRelationship.CONTRADICTS,
        evidence_excerpt="prior alignment solutions are flawed and do not hold",
    )
    db_session.flush()

    assert opp.status == OpportunityStatus.OPEN.value

    history = db_session.execute(
        select(OpportunityHistory).where(OpportunityHistory.opportunity_id == opp.id)
    ).scalars().all()
    assert len(history) == 1
    assert history[0].previous_status == OpportunityStatus.RESOLVED.value
    assert history[0].new_status == OpportunityStatus.OPEN.value
    assert history[0].trigger_type == OpportunityRelationship.CONTRADICTS.value


def test_reopened_opportunity_requires_two_new_distinct_papers_to_resolve(db_session):
    opp = _create_opportunity(db_session, status=OpportunityStatus.OPEN)
    p1 = _create_paper(
        db_session,
        title="Old Paper 1",
        passages=["We address alignment with method 1."],
    )
    p2 = _create_paper(
        db_session,
        title="Old Paper 2",
        passages=["We address alignment with method 2."],
    )
    p_critique = _create_paper(
        db_session,
        title="Critique Paper",
        passages=["We refute all prior claims: alignment remains unsolved."],
    )
    p3 = _create_paper(
        db_session,
        title="New Paper 3",
        passages=["We address the critique and solve alignment with certified defenses."],
    )
    p4 = _create_paper(
        db_session,
        title="New Paper 4",
        passages=["We independently address and verify certified defenses."],
    )

    service = OpportunityTransitionService(db_session)

    # 1. Resolve opportunity with P1 + P2
    service.apply_relationship(opp.id, p1.id, OpportunityRelationship.ADDRESSES, "address alignment with method 1")
    service.apply_relationship(opp.id, p2.id, OpportunityRelationship.ADDRESSES, "address alignment with method 2")
    db_session.flush()
    assert opp.status == OpportunityStatus.RESOLVED.value

    # 2. Contradiction re-opens to OPEN
    service.apply_relationship(opp.id, p_critique.id, OpportunityRelationship.CONTRADICTS, "refute all prior claims: alignment remains unsolved")
    db_session.flush()
    assert opp.status == OpportunityStatus.OPEN.value

    # 3. First new paper transitions OPEN -> PARTIALLY_ADDRESSED (must NOT immediately jump to RESOLVED)
    service.apply_relationship(opp.id, p3.id, OpportunityRelationship.ADDRESSES, "address the critique and solve alignment with certified defenses")
    db_session.flush()
    assert opp.status == OpportunityStatus.PARTIALLY_ADDRESSED.value

    # 4. Second new distinct paper transitions PARTIALLY_ADDRESSED -> RESOLVED
    service.apply_relationship(opp.id, p4.id, OpportunityRelationship.ADDRESSES, "independently address and verify certified defenses")
    db_session.flush()
    assert opp.status == OpportunityStatus.RESOLVED.value


def test_evolution_service_batch_resolution_and_replay(db_session):
    opp = _create_opportunity(db_session, status=OpportunityStatus.OPEN)
    p1 = _create_paper(
        db_session,
        title="Paper Alpha",
        doi="10.1001/a",
        publication_date=datetime.date(2023, 1, 10),
        passages=["We address scalable oversight for complex models."],
    )
    p2 = _create_paper(
        db_session,
        title="Paper Beta",
        doi="10.1002/b",
        publication_date=datetime.date(2023, 1, 20),
        passages=["We address scalable oversight with cross-model debate."],
    )

    class DynamicProvider:
        def generate_structured(self, prompt, schema):
            if "Paper Alpha" in prompt:
                return RelationshipClassification(
                    relationship=OpportunityRelationship.ADDRESSES,
                    confidence=0.92,
                    rationale="Addresses scalable oversight",
                    evidence_excerpt="address scalable oversight for complex models",
                    prompt_version="1.0.0",
                )
            elif "Paper Beta" in prompt:
                return RelationshipClassification(
                    relationship=OpportunityRelationship.ADDRESSES,
                    confidence=0.95,
                    rationale="Addresses scalable oversight with debate",
                    evidence_excerpt="address scalable oversight with cross-model debate",
                    prompt_version="1.0.0",
                )
            raise ValueError(f"Unknown prompt: {prompt}")

    classifier = RelationshipClassifier(provider=DynamicProvider())
    trans_service = OpportunityTransitionService(db_session)
    integration = OpportunityClassificationIntegration(classifier, trans_service, db_session)
    evolution_service = OpportunityEvolutionService(integration)

    batch = TemporalBatch(
        batch_id="batch-1",
        start_date=datetime.date(2023, 1, 1),
        end_date=datetime.date(2023, 1, 31),
        sequence_number=1,
    )

    result = evolution_service.process_batch(
        db=db_session,
        opportunity_id=opp.id,
        batch=batch,
        papers=[p1, p2],
    )

    assert result.initial_status == OpportunityStatus.OPEN
    assert result.final_status == OpportunityStatus.RESOLVED
    assert result.processed_paper_count == 2
    assert result.skipped_paper_count == 0

    # Replay same batch: papers must be safely skipped without inflating count or duplicating history
    replay_result = evolution_service.process_batch(
        db=db_session,
        opportunity_id=opp.id,
        batch=batch,
        papers=[p1, p2],
    )

    assert replay_result.initial_status == OpportunityStatus.RESOLVED
    assert replay_result.final_status == OpportunityStatus.RESOLVED
    assert replay_result.processed_paper_count == 0
    assert replay_result.skipped_paper_count == 2

    history = db_session.execute(
        select(OpportunityHistory).where(OpportunityHistory.opportunity_id == opp.id)
    ).scalars().all()
    # 2 transitions: OPEN -> PARTIALLY_ADDRESSED, PARTIALLY_ADDRESSED -> RESOLVED
    assert len(history) == 2


def test_supports_contradicts_extends_unrelated_preserve_semantics(db_session):
    """Verify that SUPPORTS, CONTRADICTS, EXTENDS, and UNRELATED preserve their defined semantics and history logging."""
    opp = _create_opportunity(db_session, status=OpportunityStatus.OPEN)

    p_supp = _create_paper(db_session, title="Supports Paper", passages=["Supports finding X."])
    p_ext = _create_paper(db_session, title="Extends Paper", passages=["Extends finding X in direction Y."])
    p_unrel = _create_paper(db_session, title="Unrelated Paper", passages=["Unrelated topic."])
    p_contra = _create_paper(db_session, title="Contradicts Paper", passages=["Contradicts finding X entirely."])

    service = OpportunityTransitionService(db_session)

    # 1. OPEN + SUPPORTS -> STRENGTHENED (state changed -> history created)
    service.apply_relationship(opp.id, p_supp.id, OpportunityRelationship.SUPPORTS, "Supports finding X")
    db_session.flush()
    assert opp.status == OpportunityStatus.STRENGTHENED.value

    # 2. STRENGTHENED + EXTENDS -> STRENGTHENED (no state change -> no history)
    service.apply_relationship(opp.id, p_ext.id, OpportunityRelationship.EXTENDS, "Extends finding X in direction Y")
    db_session.flush()
    assert opp.status == OpportunityStatus.STRENGTHENED.value

    # 3. STRENGTHENED + UNRELATED -> STRENGTHENED (no state change -> no history)
    service.apply_relationship(opp.id, p_unrel.id, OpportunityRelationship.UNRELATED, None)
    db_session.flush()
    assert opp.status == OpportunityStatus.STRENGTHENED.value

    # 4. STRENGTHENED + CONTRADICTS -> OPEN (state changed -> history created)
    service.apply_relationship(opp.id, p_contra.id, OpportunityRelationship.CONTRADICTS, "Contradicts finding X entirely")
    db_session.flush()
    assert opp.status == OpportunityStatus.OPEN.value

    # Verify history has exactly 2 records
    history = db_session.execute(
        select(OpportunityHistory)
        .where(OpportunityHistory.opportunity_id == opp.id)
        .order_by(OpportunityHistory.created_at.asc())
    ).scalars().all()
    assert len(history) == 2
    assert history[0].previous_status == OpportunityStatus.OPEN.value
    assert history[0].new_status == OpportunityStatus.STRENGTHENED.value
    assert history[0].trigger_type == OpportunityRelationship.SUPPORTS.value
    assert history[1].previous_status == OpportunityStatus.STRENGTHENED.value
    assert history[1].new_status == OpportunityStatus.OPEN.value
    assert history[1].trigger_type == OpportunityRelationship.CONTRADICTS.value
