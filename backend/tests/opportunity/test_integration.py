"""Integration tests for OpportunityClassificationIntegration.

Uses FakeLLMProvider to test the full classify-and-apply pipeline
without network calls.  All transitions are verified against the
existing deterministic state machine.
"""

import pytest
import uuid
from datetime import date
from sqlalchemy import select

from ideagen.db.models import (
    Opportunity, Paper, OpportunityEvidence, OpportunityHistory,
)
from ideagen.opportunity.models import OpportunityRelationship, OpportunityStatus
from ideagen.opportunity.classifier import (
    RelationshipClassifier,
    RelationshipClassification,
)
from ideagen.opportunity.service import OpportunityTransitionService
from ideagen.opportunity.integration import (
    OpportunityClassificationIntegration,
    ClassificationTransitionResult,
)
from ideagen.llm.exceptions import ProviderFailureError, MalformedResponseError
from ideagen.ingestion.models import SourceType
from tests.opportunity.fake_provider import FakeLLMProvider


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_paper(session, title="Test Paper", abstract="Test abstract."):
    """Create and persist a minimal Paper."""
    paper = Paper(
        source_type=SourceType.OPENALEX_API.value,
        source_id=f"W{uuid.uuid4()}",
        title=title,
        publication_date=date(2024, 1, 1),
        batch_id="integration-batch",
        abstract=abstract,
        authors=[],
        referenced_works=[],
        raw_metadata={},
    )
    session.add(paper)
    session.commit()
    return paper


def _make_opportunity(session, status=OpportunityStatus.OPEN):
    """Create and persist a minimal Opportunity in a known state."""
    opp = Opportunity(
        title="Gap in longitudinal fasting studies",
        problem_statement="No 5+ year studies on older adults.",
        description="Need RCTs assessing bone density in seniors.",
        status=status.value,
        confidence=0.5,
    )
    session.add(opp)
    session.commit()
    return opp


def _build_integration(session, fake_provider):
    """Wire up the integration from a FakeLLMProvider."""
    classifier = RelationshipClassifier(provider=fake_provider)
    transition_service = OpportunityTransitionService(session)
    return OpportunityClassificationIntegration(
        classifier=classifier,
        transition_service=transition_service,
        session=session,
    )


def _fake_provider_for(relationship, confidence=0.9):
    """Return a FakeLLMProvider configured to emit *relationship*."""
    return FakeLLMProvider(response=RelationshipClassification(
        relationship=relationship,
        confidence=confidence,
        rationale=f"Mock rationale for {relationship.value}",
        evidence_excerpt="mock excerpt",
        prompt_version="v1",
    ))


# ═══════════════════════════════════════════════════════════════════════
# §7  Five-relationship coverage from OPEN
# ═══════════════════════════════════════════════════════════════════════

class TestAllRelationshipsFromOpen:
    """OPEN + each relationship → expected status per state machine."""

    def test_open_supports(self, session):
        opp = _make_opportunity(session)
        paper = _make_paper(session)
        integration = _build_integration(
            session, _fake_provider_for(OpportunityRelationship.SUPPORTS)
        )

        result = integration.classify_and_apply(opp.id, paper.id)
        session.commit()

        assert isinstance(result, ClassificationTransitionResult)
        assert result.relationship == OpportunityRelationship.SUPPORTS
        assert result.previous_status == OpportunityStatus.OPEN
        assert result.new_status == OpportunityStatus.STRENGTHENED
        assert result.state_changed is True
        assert result.evidence_id is not None
        assert result.history_id is not None
        assert result.classifier_confidence == 0.9
        assert result.prompt_version == "v1"

        # Verify DB
        session.refresh(opp)
        assert opp.status == OpportunityStatus.STRENGTHENED.value

    def test_open_addresses(self, session):
        opp = _make_opportunity(session)
        paper = _make_paper(session)
        integration = _build_integration(
            session, _fake_provider_for(OpportunityRelationship.ADDRESSES)
        )

        result = integration.classify_and_apply(opp.id, paper.id)
        session.commit()

        assert result.previous_status == OpportunityStatus.OPEN
        assert result.new_status == OpportunityStatus.PARTIALLY_ADDRESSED
        assert result.state_changed is True
        # OPEN + ADDRESSES must NOT become RESOLVED
        assert result.new_status != OpportunityStatus.RESOLVED

    def test_open_contradicts(self, session):
        opp = _make_opportunity(session)
        paper = _make_paper(session)
        integration = _build_integration(
            session, _fake_provider_for(OpportunityRelationship.CONTRADICTS)
        )

        result = integration.classify_and_apply(opp.id, paper.id)
        session.commit()

        assert result.previous_status == OpportunityStatus.OPEN
        assert result.new_status == OpportunityStatus.OPEN
        assert result.state_changed is False
        assert result.history_id is None

    def test_open_extends(self, session):
        opp = _make_opportunity(session)
        paper = _make_paper(session)
        integration = _build_integration(
            session, _fake_provider_for(OpportunityRelationship.EXTENDS)
        )

        result = integration.classify_and_apply(opp.id, paper.id)
        session.commit()

        assert result.previous_status == OpportunityStatus.OPEN
        assert result.new_status == OpportunityStatus.OPEN
        assert result.state_changed is False
        assert result.history_id is None

    def test_open_unrelated(self, session):
        opp = _make_opportunity(session)
        paper = _make_paper(session)
        integration = _build_integration(
            session, _fake_provider_for(OpportunityRelationship.UNRELATED)
        )

        result = integration.classify_and_apply(opp.id, paper.id)
        session.commit()

        assert result.previous_status == OpportunityStatus.OPEN
        assert result.new_status == OpportunityStatus.OPEN
        assert result.state_changed is False
        assert result.evidence_id is not None  # evidence is always created
        assert result.history_id is None


# ═══════════════════════════════════════════════════════════════════════
# §8  State-machine ownership — non-OPEN transitions
# ═══════════════════════════════════════════════════════════════════════

class TestStateMachineOwnership:
    """Proves that the integration layer delegates to the state machine
    rather than hard-coding outcomes."""

    def test_strengthened_contradicts_returns_open(self, session):
        opp = _make_opportunity(session, status=OpportunityStatus.STRENGTHENED)
        paper = _make_paper(session)
        integration = _build_integration(
            session, _fake_provider_for(OpportunityRelationship.CONTRADICTS)
        )

        result = integration.classify_and_apply(opp.id, paper.id)
        session.commit()

        assert result.previous_status == OpportunityStatus.STRENGTHENED
        assert result.new_status == OpportunityStatus.OPEN
        assert result.state_changed is True

    def test_partially_addressed_contradicts_stays(self, session):
        opp = _make_opportunity(session, status=OpportunityStatus.PARTIALLY_ADDRESSED)
        paper = _make_paper(session)
        integration = _build_integration(
            session, _fake_provider_for(OpportunityRelationship.CONTRADICTS)
        )

        result = integration.classify_and_apply(opp.id, paper.id)
        session.commit()

        assert result.previous_status == OpportunityStatus.PARTIALLY_ADDRESSED
        assert result.new_status == OpportunityStatus.PARTIALLY_ADDRESSED
        assert result.state_changed is False

    def test_resolved_contradicts_returns_open(self, session):
        opp = _make_opportunity(session, status=OpportunityStatus.RESOLVED)
        paper = _make_paper(session)
        integration = _build_integration(
            session, _fake_provider_for(OpportunityRelationship.CONTRADICTS)
        )

        result = integration.classify_and_apply(opp.id, paper.id)
        session.commit()

        assert result.previous_status == OpportunityStatus.RESOLVED
        assert result.new_status == OpportunityStatus.OPEN
        assert result.state_changed is True

    def test_strengthened_addresses_partially_addressed(self, session):
        opp = _make_opportunity(session, status=OpportunityStatus.STRENGTHENED)
        paper = _make_paper(session)
        integration = _build_integration(
            session, _fake_provider_for(OpportunityRelationship.ADDRESSES)
        )

        result = integration.classify_and_apply(opp.id, paper.id)
        session.commit()

        assert result.previous_status == OpportunityStatus.STRENGTHENED
        assert result.new_status == OpportunityStatus.PARTIALLY_ADDRESSED
        assert result.state_changed is True

    def test_resolved_addresses_stays_resolved(self, session):
        opp = _make_opportunity(session, status=OpportunityStatus.RESOLVED)
        paper = _make_paper(session)
        integration = _build_integration(
            session, _fake_provider_for(OpportunityRelationship.ADDRESSES)
        )

        result = integration.classify_and_apply(opp.id, paper.id)
        session.commit()

        assert result.previous_status == OpportunityStatus.RESOLVED
        assert result.new_status == OpportunityStatus.RESOLVED
        assert result.state_changed is False


# ═══════════════════════════════════════════════════════════════════════
# §9  ADDRESSES / Resolution safety
# ═══════════════════════════════════════════════════════════════════════

class TestAddressesResolutionSafety:
    """Single ADDRESSES never resolves an opportunity."""

    def test_open_addresses_not_resolved(self, session):
        opp = _make_opportunity(session)
        paper = _make_paper(session)
        integration = _build_integration(
            session, _fake_provider_for(OpportunityRelationship.ADDRESSES)
        )
        result = integration.classify_and_apply(opp.id, paper.id)
        session.commit()
        assert result.new_status == OpportunityStatus.PARTIALLY_ADDRESSED
        assert result.new_status != OpportunityStatus.RESOLVED

    def test_strengthened_addresses_not_resolved(self, session):
        opp = _make_opportunity(session, status=OpportunityStatus.STRENGTHENED)
        paper = _make_paper(session)
        integration = _build_integration(
            session, _fake_provider_for(OpportunityRelationship.ADDRESSES)
        )
        result = integration.classify_and_apply(opp.id, paper.id)
        session.commit()
        assert result.new_status == OpportunityStatus.PARTIALLY_ADDRESSED

    def test_resolved_stays_resolved(self, session):
        opp = _make_opportunity(session, status=OpportunityStatus.RESOLVED)
        paper = _make_paper(session)
        integration = _build_integration(
            session, _fake_provider_for(OpportunityRelationship.ADDRESSES)
        )
        result = integration.classify_and_apply(opp.id, paper.id)
        session.commit()
        assert result.new_status == OpportunityStatus.RESOLVED


# ═══════════════════════════════════════════════════════════════════════
# §10  Failure injection
# ═══════════════════════════════════════════════════════════════════════

class TestFailureInjection:
    """Classifier failures must not produce any DB side effects."""

    def test_provider_failure_no_transition(self, session):
        opp = _make_opportunity(session)
        paper = _make_paper(session)
        opp_id = opp.id
        provider = FakeLLMProvider(should_fail=True)
        integration = _build_integration(session, provider)

        with pytest.raises(ProviderFailureError):
            integration.classify_and_apply(opp_id, paper.id)

        # The classification error occurs before any DB mutation, so
        # no rollback is needed — the session is still clean.
        opp_reloaded = session.get(Opportunity, opp_id)
        assert opp_reloaded.status == OpportunityStatus.OPEN.value

        evidence = session.execute(
            select(OpportunityEvidence).where(
                OpportunityEvidence.opportunity_id == opp_id
            )
        ).scalars().all()
        assert len(evidence) == 0

        history = session.execute(
            select(OpportunityHistory).where(
                OpportunityHistory.opportunity_id == opp_id
            )
        ).scalars().all()
        assert len(history) == 0

    def test_malformed_response_no_transition(self, session):
        opp = _make_opportunity(session)
        paper = _make_paper(session)
        opp_id = opp.id
        provider = FakeLLMProvider(should_malform=True)
        integration = _build_integration(session, provider)

        with pytest.raises(MalformedResponseError):
            integration.classify_and_apply(opp_id, paper.id)

        opp_reloaded = session.get(Opportunity, opp_id)
        assert opp_reloaded.status == OpportunityStatus.OPEN.value

        evidence = session.execute(
            select(OpportunityEvidence).where(
                OpportunityEvidence.opportunity_id == opp_id
            )
        ).scalars().all()
        assert len(evidence) == 0

    def test_transition_failure_atomicity(self, session, monkeypatch):
        """If the transition service fails after classification, no
        partial state persists."""
        opp = _make_opportunity(session)
        paper = _make_paper(session)
        provider = _fake_provider_for(OpportunityRelationship.SUPPORTS)
        integration = _build_integration(session, provider)

        # Sabotage the flush to simulate a DB failure during transition
        original_flush = session.flush

        def exploding_flush(*args, **kwargs):
            raise RuntimeError("Simulated DB failure")

        monkeypatch.setattr(session, "flush", exploding_flush)

        with pytest.raises(RuntimeError, match="Simulated DB failure"):
            integration.classify_and_apply(opp.id, paper.id)

        monkeypatch.undo()
        session.rollback()
        session.refresh(opp)
        assert opp.status == OpportunityStatus.OPEN.value

        evidence = session.execute(
            select(OpportunityEvidence).where(
                OpportunityEvidence.opportunity_id == opp.id
            )
        ).scalars().all()
        assert len(evidence) == 0

        history = session.execute(
            select(OpportunityHistory).where(
                OpportunityHistory.opportunity_id == opp.id
            )
        ).scalars().all()
        assert len(history) == 0


# ═══════════════════════════════════════════════════════════════════════
# §6  Idempotency (AD-22)
# ═══════════════════════════════════════════════════════════════════════

class TestIdempotency:
    """Duplicate (opportunity_id, paper_id) must be rejected."""

    def test_duplicate_paper_rejected(self, session):
        opp = _make_opportunity(session)
        paper = _make_paper(session)
        integration = _build_integration(
            session, _fake_provider_for(OpportunityRelationship.SUPPORTS)
        )

        result = integration.classify_and_apply(opp.id, paper.id)
        session.commit()
        assert result.state_changed is True

        # Second application of same paper must be rejected
        integration2 = _build_integration(
            session, _fake_provider_for(OpportunityRelationship.ADDRESSES)
        )
        with pytest.raises(ValueError, match="Evidence from Paper .* already exists"):
            integration2.classify_and_apply(opp.id, paper.id)


# ═══════════════════════════════════════════════════════════════════════
# Evidence excerpt passthrough
# ═══════════════════════════════════════════════════════════════════════

class TestEvidenceExcerpt:
    """The classifier's evidence excerpt is passed through to evidence."""

    def test_excerpt_persisted(self, session):
        opp = _make_opportunity(session)
        paper = _make_paper(session)
        integration = _build_integration(
            session, _fake_provider_for(OpportunityRelationship.SUPPORTS)
        )

        result = integration.classify_and_apply(
            opp.id, paper.id, evidence_excerpt="a bounded excerpt"
        )
        session.commit()

        evidence = session.execute(
            select(OpportunityEvidence).where(
                OpportunityEvidence.id == result.evidence_id
            )
        ).scalar_one()
        # The classifier's own evidence_excerpt is what gets stored
        assert evidence.supporting_text == "mock excerpt"
