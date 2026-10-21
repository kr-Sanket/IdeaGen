"""Tests for Evidence-Grounded Opportunity Evolution (Checkpoint 8).

Verifies:
- Classifier receives bounded actual stored paper passages.
- Valid evidence excerpts grounded in passages are accepted and applied.
- Fabricated/hallucinated excerpts are rejected fail-closed and not persisted.
- Invalid evidence cannot cause a lifecycle status change.
- UNRELATED relationship does not cause a status transition.
- Approximate matching threshold (0.85 default) behaves predictably and explicitly.
- Duplicate paper replay skips deterministically without duplicating evidence/history or crashing.
- In multi-paper batches, different papers remain independently processable even if one has invalid evidence.
- Unrelated database and classifier errors are not swallowed.
"""

import pytest
import uuid
import datetime
from sqlalchemy import create_engine, select, event
from sqlalchemy.orm import sessionmaker

from ideagen.db.models import (
    Base,
    Opportunity,
    Paper,
    PaperDocument,
    PaperPassage,
    OpportunityEvidence,
    OpportunityHistory,
)
from ideagen.opportunity.models import OpportunityRelationship, OpportunityStatus
from ideagen.opportunity.classifier import (
    RelationshipClassifier,
    RelationshipClassification,
    RelationshipClassificationInput,
)
from ideagen.opportunity.service import OpportunityTransitionService
from ideagen.opportunity.integration import (
    OpportunityClassificationIntegration,
    ClassificationTransitionResult,
)
from ideagen.opportunity.temporal import (
    OpportunityEvolutionService,
    TemporalBatch,
)
from ideagen.opportunity.grounding import (
    UngroundedEvidenceError,
    is_excerpt_grounded,
    normalize_text,
    DEFAULT_GROUNDING_THRESHOLD,
)
from ideagen.llm.exceptions import ProviderFailureError
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


def _make_paper_with_passages(
    session,
    title="Longitudinal Biomarkers in Caloric Restriction",
    abstract="Study of 10-year fasting cohorts.",
    pub_date=datetime.date(2024, 3, 1),
    passages=None,
):
    paper = Paper(
        id=uuid.uuid4(),
        source_type="MANUAL_UPLOAD",
        source_id=f"test-source-{uuid.uuid4()}",
        title=title,
        publication_date=pub_date,
        batch_id="test-batch",
        abstract=abstract,
        authors=["Alice Smith", "Bob Jones"],
        referenced_works=[],
        raw_metadata={},
    )
    session.add(paper)
    session.flush()

    doc = PaperDocument(
        id=uuid.uuid4(),
        paper_id=paper.id,
        full_text="Document full text placeholder",
        extraction_method="pdf_test",
        extraction_version="v1.0",
        extracted_at=datetime.datetime.now(datetime.timezone.utc),
    )
    session.add(doc)
    session.flush()

    default_passages = [
        (0, "Introduction", "Caloric restriction and intermittent fasting have shown longevity benefits in animal models."),
        (1, "Results", "Longitudinal fasting over 5 years demonstrates preserved bone mineral density and reduced inflammatory markers in elderly cohorts."),
        (2, "Discussion", "However, compliance challenges remain significant across diverse socio-economic groups."),
    ]
    passage_defs = passages if passages is not None else default_passages

    for idx, sec, txt in passage_defs:
        p_row = PaperPassage(
            id=uuid.uuid4(),
            paper_id=paper.id,
            document_id=doc.id,
            passage_index=idx,
            page_start=idx + 1,
            page_end=idx + 1,
            section=sec,
            text=txt,
        )
        session.add(p_row)

    session.commit()
    return paper


def _make_opportunity(session, status=OpportunityStatus.OPEN):
    opp = Opportunity(
        id=uuid.uuid4(),
        title="Long-term bone density in senior fasting regimens",
        problem_statement="Uncertainty regarding long-term musculoskeletal impact of fasting in seniors.",
        description="Need longitudinal RCT evidence on bone density and fracture risk.",
        status=status.value,
        confidence=0.5,
    )
    session.add(opp)
    session.commit()
    return opp


# ═══════════════════════════════════════════════════════════════════════════
# Grounding Unit Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_grounding_normalization_and_exact_substring():
    passage = '“Longitudinal fasting over 5 years—demonstrates preserved bone mineral density.”'
    # Test normalization
    norm = normalize_text(passage)
    assert '"' in norm
    assert "-" in norm
    assert "longitudinal fasting" in norm

    # Exact substring with differing quotes and spaces
    excerpt = "preserved bone mineral density"
    assert is_excerpt_grounded(excerpt, [passage]) is True

    # Substring with different casing and extra whitespace
    excerpt_whitespace = "  Preserved   Bone  Mineral   Density  "
    assert is_excerpt_grounded(excerpt_whitespace, [passage]) is True


def test_grounding_approximate_matching_threshold():
    passage = "Longitudinal fasting over 5 years demonstrates preserved bone mineral density in elderly cohorts."

    # Near match: 1 minor word deviation ("shows" vs "demonstrates")
    excerpt_near = "fasting over 5 years shows preserved bone mineral density"
    # Should match with default 0.85 threshold
    assert is_excerpt_grounded(excerpt_near, [passage], threshold=0.85) is True

    # Far match: heavily rewritten / fabricated claims
    excerpt_far = "fasting completely cures arthritis and reverses all biological aging"
    assert is_excerpt_grounded(excerpt_far, [passage], threshold=0.85) is False

    # Explicit threshold testing: strict threshold 0.98 rejects slight deviation
    assert is_excerpt_grounded(excerpt_near, [passage], threshold=0.98) is False
    # Generous threshold 0.50 accepts partial match
    assert is_excerpt_grounded(excerpt_near, [passage], threshold=0.50) is True


def test_grounding_rejects_empty_and_whitespace():
    passage = "Some real scientific text."
    assert is_excerpt_grounded("", [passage]) is False
    assert is_excerpt_grounded("   ", [passage]) is False
    assert is_excerpt_grounded("real text", []) is False
    assert is_excerpt_grounded(None, [passage]) is False


# ═══════════════════════════════════════════════════════════════════════════
# Passage Context & Retrieval Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_classifier_receives_real_stored_passages(db_session):
    paper = _make_paper_with_passages(db_session)
    opp = _make_opportunity(db_session)

    captured_inputs = []

    class CapturingClassifier:
        def classify(self, inp: RelationshipClassificationInput):
            captured_inputs.append(inp)
            return RelationshipClassification(
                relationship=OpportunityRelationship.SUPPORTS,
                confidence=0.9,
                rationale="Passage confirms bone density preservation.",
                evidence_excerpt="preserved bone mineral density and reduced inflammatory markers",
            )

    transition_svc = OpportunityTransitionService(db_session)
    integration = OpportunityClassificationIntegration(
        classifier=CapturingClassifier(),
        transition_service=transition_svc,
        session=db_session,
    )

    integration.classify_and_apply(opp.id, paper.id)

    assert len(captured_inputs) == 1
    inp = captured_inputs[0]
    # Verify bounded stored passages were delivered into the classifier context
    assert inp.paper_excerpt is not None
    assert "Results" in inp.paper_excerpt
    assert "preserved bone mineral density" in inp.paper_excerpt
    assert "Discussion" in inp.paper_excerpt


# ═══════════════════════════════════════════════════════════════════════════
# Fail-Closed Evidence Validation Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_valid_excerpt_accepted_and_transition_applied(db_session):
    paper = _make_paper_with_passages(db_session)
    opp = _make_opportunity(db_session, status=OpportunityStatus.OPEN)

    provider = FakeLLMProvider(response=RelationshipClassification(
        relationship=OpportunityRelationship.SUPPORTS,
        confidence=0.95,
        rationale="Evidence directly supports senior bone density hypothesis.",
        evidence_excerpt="preserved bone mineral density and reduced inflammatory markers in elderly cohorts",
    ))
    classifier = RelationshipClassifier(provider)
    transition_svc = OpportunityTransitionService(db_session)
    integration = OpportunityClassificationIntegration(classifier, transition_svc, db_session)

    result = integration.classify_and_apply(opp.id, paper.id)
    db_session.commit()

    assert result.state_changed is True
    assert result.previous_status == OpportunityStatus.OPEN
    assert result.new_status == OpportunityStatus.STRENGTHENED

    # Verify evidence row was saved with exact excerpt
    evidence = db_session.get(OpportunityEvidence, result.evidence_id)
    assert evidence is not None
    assert "preserved bone mineral density" in evidence.supporting_text

    # Verify history was recorded
    history = db_session.get(OpportunityHistory, result.history_id)
    assert history is not None
    assert history.new_status == OpportunityStatus.STRENGTHENED.value


def test_fabricated_excerpt_rejected_and_not_persisted(db_session):
    paper = _make_paper_with_passages(db_session)
    opp = _make_opportunity(db_session, status=OpportunityStatus.OPEN)

    # LLM hallucinates an excerpt not present anywhere in the paper
    provider = FakeLLMProvider(response=RelationshipClassification(
        relationship=OpportunityRelationship.SUPPORTS,
        confidence=0.99,
        rationale="Hallucinated claims.",
        evidence_excerpt="Intermittent fasting completely eliminates osteoporosis and doubles bone mineral density.",
    ))
    classifier = RelationshipClassifier(provider)
    transition_svc = OpportunityTransitionService(db_session)
    integration = OpportunityClassificationIntegration(classifier, transition_svc, db_session)

    with pytest.raises(UngroundedEvidenceError, match="not grounded in stored passages"):
        integration.classify_and_apply(opp.id, paper.id)

    db_session.rollback()

    # Verify Opportunity status was NOT modified
    opp_refreshed = db_session.get(Opportunity, opp.id)
    assert opp_refreshed.status == OpportunityStatus.OPEN.value

    # Verify NO evidence or history was created
    ev_count = db_session.execute(select(OpportunityEvidence)).scalars().all()
    assert len(ev_count) == 0

    hist_count = db_session.execute(select(OpportunityHistory)).scalars().all()
    assert len(hist_count) == 0


def test_unrelated_relationship_does_not_create_status_transition(db_session):
    paper = _make_paper_with_passages(db_session)
    opp = _make_opportunity(db_session, status=OpportunityStatus.OPEN)

    provider = FakeLLMProvider(response=RelationshipClassification(
        relationship=OpportunityRelationship.UNRELATED,
        confidence=0.90,
        rationale="Paper is about unrelated topics.",
        evidence_excerpt=None,
    ))
    classifier = RelationshipClassifier(provider)
    transition_svc = OpportunityTransitionService(db_session)
    integration = OpportunityClassificationIntegration(classifier, transition_svc, db_session)

    result = integration.classify_and_apply(opp.id, paper.id)
    db_session.commit()

    assert result.state_changed is False
    assert result.previous_status == OpportunityStatus.OPEN
    assert result.new_status == OpportunityStatus.OPEN
    assert result.history_id is None

    opp_refreshed = db_session.get(Opportunity, opp.id)
    assert opp_refreshed.status == OpportunityStatus.OPEN.value


# ═══════════════════════════════════════════════════════════════════════════
# Duplicate-Safe Replay & Batch Processing Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_duplicate_paper_replay_does_not_crash_or_duplicate(db_session):
    paper = _make_paper_with_passages(db_session)
    opp = _make_opportunity(db_session, status=OpportunityStatus.OPEN)

    provider = FakeLLMProvider(response=RelationshipClassification(
        relationship=OpportunityRelationship.SUPPORTS,
        confidence=0.95,
        rationale="Valid support.",
        evidence_excerpt="preserved bone mineral density",
    ))
    classifier = RelationshipClassifier(provider)
    transition_svc = OpportunityTransitionService(db_session)
    integration = OpportunityClassificationIntegration(classifier, transition_svc, db_session)
    service = OpportunityEvolutionService(integration)

    batch1 = TemporalBatch("b1", datetime.date(2024, 1, 1), datetime.date(2024, 6, 30), 1)

    # First run
    res1 = service.process_batch(db_session, opp.id, batch1, [paper])
    db_session.commit()

    assert res1.final_status == OpportunityStatus.STRENGTHENED
    assert res1.processed_paper_count == 1
    assert res1.skipped_paper_count == 0
    assert len(res1.events) == 1

    # Second run (replaying same batch and paper)
    res2 = service.process_batch(db_session, opp.id, batch1, [paper])
    db_session.commit()

    # Replay must NOT crash, must NOT duplicate evidence or history, and must preserve status
    assert res2.final_status == OpportunityStatus.STRENGTHENED
    assert res2.processed_paper_count == 0
    assert res2.skipped_paper_count == 1
    assert len(res2.events) == 0

    # Verify exactly 1 evidence row and 1 history row in database
    ev_rows = db_session.execute(select(OpportunityEvidence)).scalars().all()
    assert len(ev_rows) == 1

    hist_rows = db_session.execute(select(OpportunityHistory)).scalars().all()
    assert len(hist_rows) == 1


def test_different_papers_remain_independently_processable(db_session):
    """In a batch with multiple papers, if one fails evidence validation,
    the others still process cleanly without partial corruption."""
    p1 = _make_paper_with_passages(
        db_session,
        title="Paper 1 Valid",
        pub_date=datetime.date(2024, 2, 1),
        passages=[(0, "Results", "Intermittent fasting reduces systemic inflammation.")],
    )
    p2 = _make_paper_with_passages(
        db_session,
        title="Paper 2 Fabricated",
        pub_date=datetime.date(2024, 3, 1),
        passages=[(0, "Results", "Caloric restriction shows modest metabolic improvements.")],
    )
    p3 = _make_paper_with_passages(
        db_session,
        title="Paper 3 Valid Addresses",
        pub_date=datetime.date(2024, 4, 1),
        passages=[(0, "Results", "RCT resolves bone density uncertainty with DEXA scan verification.")],
    )
    opp = _make_opportunity(db_session, status=OpportunityStatus.OPEN)

    class DynamicProvider:
        def generate_structured(self, prompt, schema):
            if "Paper 1 Valid" in prompt:
                return RelationshipClassification(
                    relationship=OpportunityRelationship.SUPPORTS,
                    confidence=0.9,
                    rationale="Supports gap.",
                    evidence_excerpt="reduces systemic inflammation",
                )
            elif "Paper 2 Fabricated" in prompt:
                # Returns fabricated excerpt
                return RelationshipClassification(
                    relationship=OpportunityRelationship.SUPPORTS,
                    confidence=0.9,
                    rationale="Fabricated.",
                    evidence_excerpt="Completely solves all human aging and eliminates mortality.",
                )
            elif "Paper 3 Valid Addresses" in prompt:
                return RelationshipClassification(
                    relationship=OpportunityRelationship.ADDRESSES,
                    confidence=0.85,
                    rationale="Directly addresses uncertainty.",
                    evidence_excerpt="resolves bone density uncertainty with DEXA scan verification",
                )
            raise ValueError("Unknown paper prompt")

    classifier = RelationshipClassifier(DynamicProvider())
    transition_svc = OpportunityTransitionService(db_session)
    integration = OpportunityClassificationIntegration(classifier, transition_svc, db_session)
    service = OpportunityEvolutionService(integration)

    batch = TemporalBatch("batch-mixed", datetime.date(2024, 1, 1), datetime.date(2024, 6, 30), 1)
    res = service.process_batch(db_session, opp.id, batch, [p1, p2, p3])
    db_session.commit()

    # p1 (SUPPORTS) applied -> STRENGTHENED
    # p2 (Fabricated) rejected -> skipped!
    # p3 (ADDRESSES) applied -> PARTIALLY_ADDRESSED
    assert res.processed_paper_count == 2
    assert res.skipped_paper_count == 1
    assert res.final_status == OpportunityStatus.PARTIALLY_ADDRESSED

    # Evidence stored for p1 and p3 only
    ev_rows = db_session.execute(select(OpportunityEvidence)).scalars().all()
    assert len(ev_rows) == 2
    saved_paper_ids = {e.paper_id for e in ev_rows}
    assert p1.id in saved_paper_ids
    assert p3.id in saved_paper_ids
    assert p2.id not in saved_paper_ids


def test_unrelated_database_or_classifier_error_not_swallowed(db_session):
    paper = _make_paper_with_passages(db_session)
    opp = _make_opportunity(db_session)

    provider = FakeLLMProvider(should_fail=True)  # Raises ProviderFailureError
    classifier = RelationshipClassifier(provider)
    transition_svc = OpportunityTransitionService(db_session)
    integration = OpportunityClassificationIntegration(classifier, transition_svc, db_session)
    service = OpportunityEvolutionService(integration)

    batch = TemporalBatch("b1", datetime.date(2024, 1, 1), datetime.date(2024, 6, 30), 1)

    with pytest.raises(ProviderFailureError):
        service.process_batch(db_session, opp.id, batch, [paper])


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint 4: Fail-Closed Evidence Grounding for Papers Without Passages
# ═══════════════════════════════════════════════════════════════════════════

def _make_paper_without_passages(
    session,
    title="Paper Without Passages",
    abstract="Caloric restriction significantly improves metabolic health in human trials.",
    pub_date=datetime.date(2024, 3, 1),
):
    paper = Paper(
        id=uuid.uuid4(),
        source_type="MANUAL_UPLOAD",
        source_id=f"test-source-{uuid.uuid4()}",
        title=title,
        publication_date=pub_date,
        batch_id="test-batch",
        abstract=abstract if abstract is not None else "",
        authors=["Alice Smith"],
        referenced_works=[],
        raw_metadata={},
    )
    session.add(paper)
    session.commit()
    return paper


def test_no_passages_substantive_relationship_null_quote_rejected(db_session):
    paper = _make_paper_without_passages(db_session)
    opp = _make_opportunity(db_session, status=OpportunityStatus.OPEN)

    provider = FakeLLMProvider(response=RelationshipClassification(
        relationship=OpportunityRelationship.SUPPORTS,
        confidence=0.9,
        rationale="Support claim without quote.",
        evidence_excerpt=None,
    ))
    classifier = RelationshipClassifier(provider)
    transition_svc = OpportunityTransitionService(db_session)
    integration = OpportunityClassificationIntegration(classifier, transition_svc, db_session)

    with pytest.raises(UngroundedEvidenceError):
        integration.classify_and_apply(opp.id, paper.id)

    db_session.rollback()
    assert db_session.get(Opportunity, opp.id).status == OpportunityStatus.OPEN.value
    assert len(db_session.execute(select(OpportunityEvidence)).scalars().all()) == 0
    assert len(db_session.execute(select(OpportunityHistory)).scalars().all()) == 0


def test_no_passages_substantive_relationship_quote_verified_against_abstract(db_session):
    paper = _make_paper_without_passages(
        db_session,
        abstract="Caloric restriction significantly improves metabolic health in human trials."
    )
    opp = _make_opportunity(db_session, status=OpportunityStatus.OPEN)

    provider = FakeLLMProvider(response=RelationshipClassification(
        relationship=OpportunityRelationship.SUPPORTS,
        confidence=0.95,
        rationale="Abstract shows metabolic improvement.",
        evidence_excerpt="improves metabolic health in human trials",
    ))
    classifier = RelationshipClassifier(provider)
    transition_svc = OpportunityTransitionService(db_session)
    integration = OpportunityClassificationIntegration(classifier, transition_svc, db_session)

    result = integration.classify_and_apply(opp.id, paper.id)
    db_session.commit()

    assert result.state_changed is True
    assert result.previous_status == OpportunityStatus.OPEN
    assert result.new_status == OpportunityStatus.STRENGTHENED

    opp_refreshed = db_session.get(Opportunity, opp.id)
    assert opp_refreshed.status == OpportunityStatus.STRENGTHENED.value

    ev_rows = db_session.execute(select(OpportunityEvidence)).scalars().all()
    assert len(ev_rows) == 1
    assert "improves metabolic health" in ev_rows[0].supporting_text


def test_no_passages_substantive_relationship_quote_absent_from_abstract_rejected(db_session):
    paper = _make_paper_without_passages(
        db_session,
        abstract="Caloric restriction significantly improves metabolic health in human trials."
    )
    opp = _make_opportunity(db_session, status=OpportunityStatus.OPEN)

    provider = FakeLLMProvider(response=RelationshipClassification(
        relationship=OpportunityRelationship.SUPPORTS,
        confidence=0.95,
        rationale="Completely hallucinated claim.",
        evidence_excerpt="completely eliminates Alzheimer's disease and reverses biological clock",
    ))
    classifier = RelationshipClassifier(provider)
    transition_svc = OpportunityTransitionService(db_session)
    integration = OpportunityClassificationIntegration(classifier, transition_svc, db_session)

    with pytest.raises(UngroundedEvidenceError, match="is not grounded in stored passages"):
        integration.classify_and_apply(opp.id, paper.id)

    db_session.rollback()
    assert db_session.get(Opportunity, opp.id).status == OpportunityStatus.OPEN.value
    assert len(db_session.execute(select(OpportunityEvidence)).scalars().all()) == 0
    assert len(db_session.execute(select(OpportunityHistory)).scalars().all()) == 0


def test_no_passages_no_abstract_substantive_relationship_rejected(db_session):
    paper = _make_paper_without_passages(
        db_session,
        abstract=""
    )
    opp = _make_opportunity(db_session, status=OpportunityStatus.OPEN)

    provider = FakeLLMProvider(response=RelationshipClassification(
        relationship=OpportunityRelationship.ADDRESSES,
        confidence=0.85,
        rationale="Paper claims to address problem.",
        evidence_excerpt="addresses problem statement",
    ))
    classifier = RelationshipClassifier(provider)
    transition_svc = OpportunityTransitionService(db_session)
    integration = OpportunityClassificationIntegration(classifier, transition_svc, db_session)

    with pytest.raises(UngroundedEvidenceError, match="neither stored passages nor abstract"):
        integration.classify_and_apply(opp.id, paper.id)

    db_session.rollback()
    assert db_session.get(Opportunity, opp.id).status == OpportunityStatus.OPEN.value
    assert len(db_session.execute(select(OpportunityEvidence)).scalars().all()) == 0
    assert len(db_session.execute(select(OpportunityHistory)).scalars().all()) == 0


def test_no_passages_unrelated_relationship_accepted_without_quote(db_session):
    paper = _make_paper_without_passages(
        db_session,
        abstract=""
    )
    opp = _make_opportunity(db_session, status=OpportunityStatus.OPEN)

    provider = FakeLLMProvider(response=RelationshipClassification(
        relationship=OpportunityRelationship.UNRELATED,
        confidence=0.99,
        rationale="Completely different topic.",
        evidence_excerpt=None,
    ))
    classifier = RelationshipClassifier(provider)
    transition_svc = OpportunityTransitionService(db_session)
    integration = OpportunityClassificationIntegration(classifier, transition_svc, db_session)

    result = integration.classify_and_apply(opp.id, paper.id)
    db_session.commit()

    assert result.state_changed is False
    assert result.previous_status == OpportunityStatus.OPEN
    assert result.new_status == OpportunityStatus.OPEN

    opp_refreshed = db_session.get(Opportunity, opp.id)
    assert opp_refreshed.status == OpportunityStatus.OPEN.value

    # UNRELATED creates an evidence record to mark examination, but no history transition
    ev_rows = db_session.execute(select(OpportunityEvidence)).scalars().all()
    assert len(ev_rows) == 1
    assert ev_rows[0].evidence_type == OpportunityRelationship.UNRELATED.value

    hist_rows = db_session.execute(select(OpportunityHistory)).scalars().all()
    assert len(hist_rows) == 0


def test_rejected_evidence_creates_no_record_or_transition(db_session):
    # Test service level rejection directly
    paper = _make_paper_without_passages(db_session, abstract="Real abstract text.")
    opp = _make_opportunity(db_session, status=OpportunityStatus.OPEN)

    svc = OpportunityTransitionService(db_session)

    # 1. Null quote rejection
    with pytest.raises(UngroundedEvidenceError):
        svc.apply_relationship(opp.id, paper.id, OpportunityRelationship.SUPPORTS, evidence_excerpt=None)
    db_session.rollback()

    # 2. Unverifiable quote rejection
    with pytest.raises(UngroundedEvidenceError):
        svc.apply_relationship(opp.id, paper.id, OpportunityRelationship.CONTRADICTS, evidence_excerpt="Nonexistent text")
    db_session.rollback()

    # Verify database remains completely untouched
    opp_refreshed = db_session.get(Opportunity, opp.id)
    assert opp_refreshed.status == OpportunityStatus.OPEN.value

    assert len(db_session.execute(select(OpportunityEvidence)).scalars().all()) == 0
    assert len(db_session.execute(select(OpportunityHistory)).scalars().all()) == 0
