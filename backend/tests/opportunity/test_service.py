import pytest
import uuid
from datetime import date
from sqlalchemy.exc import IntegrityError
from sqlalchemy import select

from ideagen.db.models import Opportunity, Paper, OpportunityEvidence, OpportunityHistory
from ideagen.opportunity.models import OpportunityRelationship, OpportunityStatus
from ideagen.opportunity.service import OpportunityTransitionService
from ideagen.ingestion.models import SourceType

@pytest.fixture
def test_paper(session):
    paper = Paper(
        source_type=SourceType.OPENALEX_API.value,
        source_id=f"W{uuid.uuid4()}",
        title="Test Paper",
        publication_date=date(2023, 1, 1),
        batch_id="batch1",
        abstract="",
        authors=[],
        referenced_works=[],
        raw_metadata={}
    )
    session.add(paper)
    session.commit()
    return paper

@pytest.fixture
def test_opportunity(session):
    opp = Opportunity(
        title="Test Opp",
        problem_statement="Problem",
        description="Desc",
        status=OpportunityStatus.OPEN.value,
        confidence=0.5
    )
    session.add(opp)
    session.commit()
    return opp

def test_transition_open_to_strengthened(session, test_opportunity, test_paper):
    service = OpportunityTransitionService(session)
    
    # 1. OPEN + SUPPORTS -> STRENGTHENED
    service.apply_relationship(test_opportunity.id, test_paper.id, OpportunityRelationship.SUPPORTS)
    session.commit()
    
    session.refresh(test_opportunity)
    assert test_opportunity.status == OpportunityStatus.STRENGTHENED.value
    
    evidence = session.execute(select(OpportunityEvidence).where(OpportunityEvidence.opportunity_id == test_opportunity.id)).scalars().all()
    assert len(evidence) == 1
    assert evidence[0].paper_id == test_paper.id
    assert evidence[0].evidence_type == OpportunityRelationship.SUPPORTS.value
    
    history = session.execute(select(OpportunityHistory).where(OpportunityHistory.opportunity_id == test_opportunity.id)).scalars().all()
    assert len(history) == 1
    assert history[0].previous_status == OpportunityStatus.OPEN.value
    assert history[0].new_status == OpportunityStatus.STRENGTHENED.value
    assert history[0].trigger_type == OpportunityRelationship.SUPPORTS.value
    assert history[0].evidence_id == evidence[0].id

def test_transition_open_to_partially_addressed(session, test_opportunity, test_paper):
    service = OpportunityTransitionService(session)
    service.apply_relationship(test_opportunity.id, test_paper.id, OpportunityRelationship.ADDRESSES)
    session.commit()
    session.refresh(test_opportunity)
    assert test_opportunity.status == OpportunityStatus.PARTIALLY_ADDRESSED.value

def test_transition_strengthened_to_partially_addressed(session, test_opportunity, test_paper):
    test_opportunity.status = OpportunityStatus.STRENGTHENED.value
    session.commit()
    service = OpportunityTransitionService(session)
    service.apply_relationship(test_opportunity.id, test_paper.id, OpportunityRelationship.ADDRESSES)
    session.commit()
    session.refresh(test_opportunity)
    assert test_opportunity.status == OpportunityStatus.PARTIALLY_ADDRESSED.value

def test_transition_resolved_to_resolved(session, test_opportunity, test_paper):
    test_opportunity.status = OpportunityStatus.RESOLVED.value
    session.commit()
    service = OpportunityTransitionService(session)
    service.apply_relationship(test_opportunity.id, test_paper.id, OpportunityRelationship.ADDRESSES)
    session.commit()
    session.refresh(test_opportunity)
    assert test_opportunity.status == OpportunityStatus.RESOLVED.value
    
    # Should not create history since status didn't change
    history = session.execute(select(OpportunityHistory).where(OpportunityHistory.opportunity_id == test_opportunity.id)).scalars().all()
    assert len(history) == 0

def test_noop_strengthened_supports(session, test_opportunity, test_paper):
    test_opportunity.status = OpportunityStatus.STRENGTHENED.value
    session.commit()
    service = OpportunityTransitionService(session)
    service.apply_relationship(test_opportunity.id, test_paper.id, OpportunityRelationship.SUPPORTS)
    session.commit()
    session.refresh(test_opportunity)
    assert test_opportunity.status == OpportunityStatus.STRENGTHENED.value
    
    history = session.execute(select(OpportunityHistory).where(OpportunityHistory.opportunity_id == test_opportunity.id)).scalars().all()
    assert len(history) == 0

def test_noop_relationship(session, test_opportunity, test_paper):
    service = OpportunityTransitionService(session)
    
    # 5. UNRELATED leaves status unchanged
    service.apply_relationship(test_opportunity.id, test_paper.id, OpportunityRelationship.UNRELATED)
    session.commit()
    
    session.refresh(test_opportunity)
    assert test_opportunity.status == OpportunityStatus.OPEN.value
    
    evidence = session.execute(select(OpportunityEvidence).where(OpportunityEvidence.opportunity_id == test_opportunity.id)).scalars().all()
    assert len(evidence) == 1
    
    # No history for no-op
    history = session.execute(select(OpportunityHistory).where(OpportunityHistory.opportunity_id == test_opportunity.id)).scalars().all()
    assert len(history) == 0

def test_nonexistent_opportunity(session, test_paper):
    service = OpportunityTransitionService(session)
    with pytest.raises(ValueError, match="Opportunity .* not found."):
        service.apply_relationship(uuid.uuid4(), test_paper.id, OpportunityRelationship.SUPPORTS)

def test_nonexistent_paper(session, test_opportunity):
    service = OpportunityTransitionService(session)
    with pytest.raises(ValueError, match="Paper .* not found."):
        service.apply_relationship(test_opportunity.id, uuid.uuid4(), OpportunityRelationship.SUPPORTS)

def test_duplicate_paper_evidence(session, test_opportunity, test_paper):
    service = OpportunityTransitionService(session)
    service.apply_relationship(test_opportunity.id, test_paper.id, OpportunityRelationship.SUPPORTS)
    session.commit()
    
    with pytest.raises(ValueError, match="Evidence from Paper .* already exists"):
        service.apply_relationship(test_opportunity.id, test_paper.id, OpportunityRelationship.ADDRESSES)
        
def test_atomicity(session, test_opportunity, test_paper, monkeypatch):
    service = OpportunityTransitionService(session)
    
    # Mock transition to raise an exception AFTER evidence is validated,
    # or just force an error during flush.
    
    def fake_flush():
        raise Exception("Database failure")
        
    monkeypatch.setattr(session, "flush", fake_flush)
    
    try:
        service.apply_relationship(test_opportunity.id, test_paper.id, OpportunityRelationship.SUPPORTS)
        session.commit()
    except Exception:
        session.rollback()
    
    monkeypatch.undo()
    session.refresh(test_opportunity)
    assert test_opportunity.status == OpportunityStatus.OPEN.value
    
    evidence = session.execute(select(OpportunityEvidence).where(OpportunityEvidence.opportunity_id == test_opportunity.id)).scalars().all()
    assert len(evidence) == 0
    
    history = session.execute(select(OpportunityHistory).where(OpportunityHistory.opportunity_id == test_opportunity.id)).scalars().all()
    assert len(history) == 0
