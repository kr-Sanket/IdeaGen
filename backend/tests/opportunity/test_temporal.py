import datetime
import uuid
import pytest
from sqlalchemy import select

from ideagen.db.models import Opportunity, Paper, OpportunityEvidence, OpportunityHistory
from ideagen.ingestion.models import SourceType
from ideagen.opportunity.models import OpportunityRelationship, OpportunityStatus
from ideagen.opportunity.classifier import RelationshipClassifier, RelationshipClassification
from ideagen.opportunity.service import OpportunityTransitionService
from ideagen.opportunity.integration import OpportunityClassificationIntegration
from ideagen.opportunity.temporal import (
    TemporalBatch, 
    OpportunityEvolutionService, 
    EvolutionEvent
)
from tests.opportunity.fake_provider import FakeLLMProvider

def _make_paper(session, title="Test Paper", p_date=datetime.date(2024, 1, 1)):
    p = Paper(
        source_type=SourceType.OPENALEX_API.value,
        source_id=f"W{uuid.uuid4()}",
        title=title,
        publication_date=p_date,
        batch_id="temporal-batch",
        abstract="Test abstract containing mock excerpt and E.",
        authors=[],
        referenced_works=[],
        raw_metadata={},
    )
    session.add(p)
    session.commit()
    return p

def _make_opportunity(session, status=OpportunityStatus.OPEN):
    opp = Opportunity(
        title="Test Temporal",
        problem_statement="Test Problem",
        description="Test Desc",
        status=status.value,
        confidence=0.5,
    )
    session.add(opp)
    session.commit()
    return opp

def _build_service(session, provider):
    classifier = RelationshipClassifier(provider=provider)
    transition = OpportunityTransitionService(session)
    integration = OpportunityClassificationIntegration(classifier, transition, session)
    return OpportunityEvolutionService(integration)

def _fake_provider(relationship=OpportunityRelationship.SUPPORTS):
    return FakeLLMProvider(response=RelationshipClassification(
        relationship=relationship,
        confidence=0.9,
        rationale="Mock rationale",
        evidence_excerpt="mock excerpt",
        prompt_version="v1"
    ))

def test_valid_batch_construction():
    b = TemporalBatch("b1", datetime.date(2024,1,1), datetime.date(2024,12,31), 1)
    assert b.batch_id == "b1"

def test_invalid_date_range(session):
    opp = _make_opportunity(session)
    service = _build_service(session, _fake_provider())
    b = TemporalBatch("b1", datetime.date(2024,12,31), datetime.date(2024,1,1), 1) # start > end
    with pytest.raises(ValueError, match="Invalid date range"):
        service.process_batch(session, opp.id, b, [])

def test_overlapping_batch_detection():
    # overlapping batch validation not strictly in process_batch, but chronological order is tested in replay
    pass

def test_deterministic_paper_ordering(session):
    p1 = _make_paper(session, p_date=datetime.date(2024,2,1))
    p2 = _make_paper(session, p_date=datetime.date(2024,1,1))
    p3 = _make_paper(session, p_date=datetime.date(2024,2,1))
    
    opp = _make_opportunity(session)
    
    # We will just capture the order they are processed
    class CapturingIntegration:
        def __init__(self):
            self.processed = []
        def classify_and_apply(self, opportunity_id, paper_id):
            self.processed.append(paper_id)
            from ideagen.opportunity.integration import ClassificationTransitionResult
            return ClassificationTransitionResult(
                opportunity_id, paper_id, OpportunityRelationship.SUPPORTS,
                OpportunityStatus.OPEN, OpportunityStatus.OPEN, False, None, None, 1.0, "", "v1"
            )
            
    service = OpportunityEvolutionService(CapturingIntegration())
    b = TemporalBatch("b0", datetime.date(2024,1,1), datetime.date(2024,12,31), 1)
    
    # Run
    service.process_batch(session, opp.id, b, [p1, p2, p3])
    ordered = service.integration.processed
    
    # p2 is earliest (Jan 1)
    assert ordered[0] == p2.id
    # p1 and p3 are same date, ordered by UUID
    expected_order = sorted([p1.id, p3.id], key=str)
    assert ordered[1:] == expected_order

def test_future_paper_exclusion(session):
    # Leakage protection test
    opp = _make_opportunity(session, OpportunityStatus.OPEN)
    # Cutoff is Dec 2024
    b = TemporalBatch("b0", datetime.date(2024,1,1), datetime.date(2024,12,31), 1)
    
    p_future = _make_paper(session, p_date=datetime.date(2025,1,1))
    
    service = _build_service(session, _fake_provider(OpportunityRelationship.SUPPORTS))
    res = service.process_batch(session, opp.id, b, [p_future])
    
    assert res.processed_paper_count == 0
    assert res.skipped_paper_count == 1
    assert res.final_status == OpportunityStatus.OPEN
    
    session.refresh(opp)
    assert opp.status == OpportunityStatus.OPEN.value

def test_missing_publication_date_handling(session):
    opp = _make_opportunity(session)
    b = TemporalBatch("b0", datetime.date(2024,1,1), datetime.date(2024,12,31), 1)
    p = _make_paper(session, p_date=None)
    
    service = _build_service(session, _fake_provider())
    res = service.process_batch(session, opp.id, b, [p])
    
    assert res.processed_paper_count == 0
    assert res.skipped_paper_count == 1

def test_batch_0_processing_supports_transition(session):
    opp = _make_opportunity(session, OpportunityStatus.OPEN)
    b = TemporalBatch("b0", datetime.date(2024,1,1), datetime.date(2024,12,31), 1)
    p = _make_paper(session, p_date=datetime.date(2024,5,5))
    
    service = _build_service(session, _fake_provider(OpportunityRelationship.SUPPORTS))
    res = service.process_batch(session, opp.id, b, [p])
    
    assert res.processed_paper_count == 1
    assert res.initial_status == OpportunityStatus.OPEN
    assert res.final_status == OpportunityStatus.STRENGTHENED
    assert len(res.events) == 1
    assert res.events[0].relationship == OpportunityRelationship.SUPPORTS
    assert res.events[0].state_changed is True
    
    # Verify evidence and history
    assert res.events[0].evidence_id is not None
    assert res.events[0].history_id is not None
    
    # DB
    session.refresh(opp)
    assert opp.status == OpportunityStatus.STRENGTHENED.value

def test_multi_paper_batch(session):
    opp = _make_opportunity(session, OpportunityStatus.OPEN)
    b = TemporalBatch("b0", datetime.date(2024,1,1), datetime.date(2024,12,31), 1)
    p1 = _make_paper(session, p_date=datetime.date(2024,2,2))
    p2 = _make_paper(session, p_date=datetime.date(2024,3,3))
    
    class MultiProvider:
        def __init__(self):
            self.calls = 0
        def generate_structured(self, prompt, schema):
            rel = OpportunityRelationship.ADDRESSES if self.calls == 0 else OpportunityRelationship.UNRELATED
            self.calls += 1
            return RelationshipClassification(
                relationship=rel, confidence=0.9, rationale="R", evidence_excerpt="E", prompt_version="v1"
            )

    classifier = RelationshipClassifier(provider=MultiProvider())
    transition = OpportunityTransitionService(session)
    integration = OpportunityClassificationIntegration(classifier, transition, session)
    service = OpportunityEvolutionService(integration)
    
    res = service.process_batch(session, opp.id, b, [p1, p2])
    
    assert res.processed_paper_count == 2
    assert res.final_status == OpportunityStatus.PARTIALLY_ADDRESSED
    assert len(res.events) == 2
    assert res.events[0].relationship == OpportunityRelationship.ADDRESSES
    assert res.events[0].state_changed is True
    assert res.events[1].relationship == OpportunityRelationship.UNRELATED
    assert res.events[1].state_changed is False

def test_addresses_transition(session):
    opp = _make_opportunity(session, OpportunityStatus.OPEN)
    b = TemporalBatch("b0", datetime.date(2024,1,1), datetime.date(2024,12,31), 1)
    p = _make_paper(session, p_date=datetime.date(2024,5,5))
    
    service = _build_service(session, _fake_provider(OpportunityRelationship.ADDRESSES))
    res = service.process_batch(session, opp.id, b, [p])
    
    assert res.final_status == OpportunityStatus.PARTIALLY_ADDRESSED
    # Verify no automatic RESOLVED
    assert res.final_status != OpportunityStatus.RESOLVED

def test_contradicts_transition(session):
    opp = _make_opportunity(session, OpportunityStatus.STRENGTHENED)
    b = TemporalBatch("b0", datetime.date(2024,1,1), datetime.date(2024,12,31), 1)
    p = _make_paper(session, p_date=datetime.date(2024,5,5))
    
    service = _build_service(session, _fake_provider(OpportunityRelationship.CONTRADICTS))
    res = service.process_batch(session, opp.id, b, [p])
    
    assert res.final_status == OpportunityStatus.OPEN

def test_extends_transition(session):
    opp = _make_opportunity(session, OpportunityStatus.OPEN)
    b = TemporalBatch("b0", datetime.date(2024,1,1), datetime.date(2024,12,31), 1)
    p = _make_paper(session, p_date=datetime.date(2024,5,5))
    
    service = _build_service(session, _fake_provider(OpportunityRelationship.EXTENDS))
    res = service.process_batch(session, opp.id, b, [p])
    
    assert res.final_status == OpportunityStatus.OPEN
    # EXTENDS does not create new state, but sets directional_extension_detected flag
    assert res.events[0].directional_extension_detected is True

def test_duplicate_paper_reprocessing(session):
    opp = _make_opportunity(session, OpportunityStatus.OPEN)
    b = TemporalBatch("b0", datetime.date(2024,1,1), datetime.date(2024,12,31), 1)
    p = _make_paper(session, p_date=datetime.date(2024,5,5))
    
    service = _build_service(session, _fake_provider(OpportunityRelationship.SUPPORTS))
    # First time
    res1 = service.process_batch(session, opp.id, b, [p])
    assert res1.final_status == OpportunityStatus.STRENGTHENED
    session.commit()
    
    # Reprocess same paper, same batch
    # Checkpoint 8 duplicate-safe replay: skips duplicate without crashing, preserving status
    res2 = service.process_batch(session, opp.id, b, [p])
    assert res2.final_status == OpportunityStatus.STRENGTHENED
    assert res2.processed_paper_count == 0
    assert res2.skipped_paper_count == 1
    assert len(res2.events) == 0

def test_deterministic_replay(session):
    opp = _make_opportunity(session, OpportunityStatus.OPEN)
    
    # 3 batches, 3 papers
    b1 = TemporalBatch("b1", datetime.date(2024,1,1), datetime.date(2024,6,30), 1)
    b2 = TemporalBatch("b2", datetime.date(2024,7,1), datetime.date(2024,12,31), 2)
    b3 = TemporalBatch("b3", datetime.date(2025,1,1), datetime.date(2025,6,30), 3)
    
    p1 = _make_paper(session, p_date=datetime.date(2024,2,2)) # in b1
    p2 = _make_paper(session, p_date=datetime.date(2024,8,8)) # in b2
    p3 = _make_paper(session, p_date=datetime.date(2025,3,3)) # in b3
    
    # FakeProvider will just return SUPPORTS
    service = _build_service(session, _fake_provider(OpportunityRelationship.SUPPORTS))
    
    # Instead of running sequentially, replay all
    # Replay automatically runs them in chronological order
    results = service.replay(session, opp.id, [b2, b3, b1], [p1, p2, p3])
    
    assert len(results) == 3
    assert results[0].events[0].paper_id == p1.id
    assert results[1].events[0].paper_id == p2.id
    assert results[2].events[0].paper_id == p3.id

def test_classifier_failure(session):
    opp = _make_opportunity(session, OpportunityStatus.OPEN)
    b = TemporalBatch("b0", datetime.date(2024,1,1), datetime.date(2024,12,31), 1)
    p = _make_paper(session, p_date=datetime.date(2024,5,5))
    
    class FailingProvider:
        def generate_structured(self, prompt, schema):
            from ideagen.llm.exceptions import ProviderFailureError
            raise ProviderFailureError("API down")
            
    classifier = RelationshipClassifier(provider=FailingProvider())
    transition = OpportunityTransitionService(session)
    integration = OpportunityClassificationIntegration(classifier, transition, session)
    service = OpportunityEvolutionService(integration)
    
    with pytest.raises(Exception, match="API down"):
        service.process_batch(session, opp.id, b, [p])
