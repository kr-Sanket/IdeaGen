import pytest
from datetime import date
import uuid
from sqlalchemy.exc import IntegrityError
from ideagen.db.models import Paper, Opportunity, OpportunityEvidence, OpportunityHistory
from ideagen.opportunity.models import OpportunityStatus, OpportunityRelationship
from ideagen.ingestion.models import SourceType

def test_opportunity_creation_and_bounds(session):
    opp = Opportunity(
        title="Test Opportunity",
        problem_statement="Problem statement",
        description="Description",
        status=OpportunityStatus.OPEN,
        confidence=0.5
    )
    session.add(opp)
    session.commit()
    
    assert opp.id is not None
    assert opp.created_at is not None
    assert opp.updated_at is not None

def test_confidence_bounds_enforced(session):
    opp = Opportunity(
        title="Test Opportunity",
        problem_statement="Problem statement",
        description="Description",
        status=OpportunityStatus.OPEN,
        confidence=1.5
    )
    session.add(opp)
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    opp2 = Opportunity(
        title="Test Opportunity 2",
        problem_statement="Problem statement",
        description="Description",
        status=OpportunityStatus.OPEN,
        confidence=-0.1
    )
    session.add(opp2)
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

def test_opportunity_relations(session):
    # 1. Create Paper
    paper = Paper(
        source_type=SourceType.MANUAL_UPLOAD.value,
        source_id="test-source-id",
        title="Test Paper",
        publication_date=date(2023, 1, 1),
        batch_id="test-batch",
        abstract="",
        authors=[],
        referenced_works=[],
        raw_metadata={}
    )
    session.add(paper)
    session.commit()
    
    # 2. Create Opportunity
    opp = Opportunity(
        title="Test Opportunity",
        problem_statement="Problem statement",
        description="Description",
        status=OpportunityStatus.OPEN,
        confidence=0.5
    )
    session.add(opp)
    session.commit()
    
    # 3. Create OpportunityEvidence linking Paper and Opportunity
    evidence = OpportunityEvidence(
        opportunity_id=opp.id,
        paper_id=paper.id,
        evidence_type=OpportunityRelationship.SUPPORTS.value,
        supporting_text="This paper supports the opportunity."
    )
    session.add(evidence)
    session.commit()
    
    # 4. Create OpportunityHistory
    history = OpportunityHistory(
        opportunity_id=opp.id,
        previous_status=OpportunityStatus.OPEN.value,
        new_status=OpportunityStatus.STRENGTHENED.value,
        trigger_type=OpportunityRelationship.SUPPORTS.value,
        evidence_id=evidence.id
    )
    session.add(history)
    session.commit()
    
    # Retrieve and verify all three
    retrieved_opp = session.query(Opportunity).filter_by(id=opp.id).first()
    assert len(retrieved_opp.evidence) == 1
    assert len(retrieved_opp.history) == 1
    assert retrieved_opp.evidence[0].paper_id == paper.id
    assert retrieved_opp.history[0].new_status == OpportunityStatus.STRENGTHENED.value
    
    # 6. Deleting Opportunity deletes Evidence and History but not Paper
    session.delete(retrieved_opp)
    session.commit()
    
    assert session.query(OpportunityEvidence).filter_by(id=evidence.id).first() is None
    assert session.query(OpportunityHistory).filter_by(id=history.id).first() is None
    
    # Paper must still exist
    retrieved_paper = session.query(Paper).filter_by(id=paper.id).first()
    assert retrieved_paper is not None
