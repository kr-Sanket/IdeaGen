import uuid
import datetime
import pytest
from typing import List
from sqlalchemy import select

from ideagen.db.models import Opportunity, Paper, OpportunityEvidence, OpportunityHistory
from ideagen.ingestion.models import SourceType
from ideagen.opportunity.discovery import (
    ResearchGapSignal, 
    OpportunityDiscoveryService,
    OpportunityCandidate
)
from tests.opportunity.test_discovery import FakeGapEmbedder

from ideagen.retrieval.evidence import EvidenceResult
from ideagen.rag.explanation import GroundedOpportunityService, GroundedExplanation, GroundedClaim
from tests.rag.test_grounding import FakeRetriever, FakeExplanationGenerator

from ideagen.opportunity.temporal import TemporalBatch, OpportunityEvolutionService
from ideagen.opportunity.classifier import RelationshipClassifier, RelationshipClassification
from ideagen.opportunity.models import OpportunityRelationship, OpportunityStatus
from ideagen.opportunity.service import OpportunityTransitionService
from ideagen.opportunity.integration import OpportunityClassificationIntegration

from ideagen.pipeline.end_to_end import IdeaGenOrchestrator

# ── Fixtures ──

def make_paper(session, title, p_date, batch_id):
    p = Paper(
        source_type=SourceType.OPENALEX_API.value,
        source_id=f"W{uuid.uuid4()}",
        title=title,
        publication_date=p_date,
        batch_id=batch_id,
        abstract="Test abstract",
        authors=[],
        referenced_works=[],
        raw_metadata={},
    )
    session.add(p)
    return p

def make_signal(paper_id, statement, text):
    return ResearchGapSignal(
        id=uuid.uuid4(),
        paper_id=paper_id,
        statement=statement,
        normalized_statement=statement,
        confidence=0.9
    )

class ControlledProvider:
    def __init__(self, mapping):
        self.mapping = mapping
    def generate_structured(self, prompt, schema):
        rel = OpportunityRelationship.UNRELATED
        for k, v in self.mapping.items():
            if k in prompt:
                rel = v
                break
        return RelationshipClassification(
            relationship=rel,
            confidence=0.9,
            rationale="Controlled rationale",
            evidence_excerpt="Excerpt",
            prompt_version="v1"
        )


def test_end_to_end_pipeline(session):
    # ── 1. Ingestion / Persistence (Fixture Setup) ──
    # Batch 0 (Baseline Discovery Batch)
    b0 = TemporalBatch("b0", datetime.date(2024, 1, 1), datetime.date(2024, 12, 31), 0)
    p_gap1 = make_paper(session, "Gap Paper 1", datetime.date(2024, 2, 1), "b0")
    p_gap2 = make_paper(session, "Gap Paper 2", datetime.date(2024, 5, 1), "b0")
    p_gap3 = make_paper(session, "Gap Paper 3", datetime.date(2024, 8, 1), "b0")
    p_distract = make_paper(session, "Unrelated A", datetime.date(2024, 9, 1), "b0")
    
    # Batch 1 (Evolution 1)
    b1 = TemporalBatch("b1", datetime.date(2025, 1, 1), datetime.date(2025, 6, 30), 1)
    p_supports = make_paper(session, "Supports Paper", datetime.date(2025, 3, 1), "b1")
    p_unrelated1 = make_paper(session, "Unrelated B", datetime.date(2025, 4, 1), "b1")
    
    # Batch 2 (Evolution 2)
    b2 = TemporalBatch("b2", datetime.date(2025, 7, 1), datetime.date(2025, 12, 31), 2)
    p_addresses = make_paper(session, "Addresses Paper", datetime.date(2025, 9, 1), "b2")
    p_unrelated2 = make_paper(session, "Unrelated C", datetime.date(2025, 11, 1), "b2")
    
    # Future Leakage Paper
    b3 = TemporalBatch("b3", datetime.date(2026, 1, 1), datetime.date(2026, 12, 31), 3)
    p_future = make_paper(session, "Future Addresses", datetime.date(2026, 5, 1), "b3")

    session.commit()
    papers = [p_gap1, p_gap2, p_gap3, p_distract, p_supports, p_unrelated1, p_addresses, p_unrelated2, p_future]

    # ── 2. Signal Creation ──
    signals = [
        make_signal(p_gap1.id, "recurring gap", "text1"),
        make_signal(p_gap2.id, "recurring gap", "text2"),
        make_signal(p_gap3.id, "recurring gap", "text3"),
        make_signal(p_distract.id, "distractor noise", "text4"),
    ]

    # ── 3. Wire Up Services ──
    # Discovery
    embedder_map = {
        "recurring gap": [1.0, 0.0, 0.0],
        "distractor noise": [0.0, 1.0, 0.0],
    }
    discovery = OpportunityDiscoveryService(
        embedder=FakeGapEmbedder(embedder_map),
        similarity_threshold=0.8,
        min_cluster_size=3
    )

    # RAG
    evidence = [EvidenceResult(
        paper_id=p_gap1.id, title="Gap Paper 1", publication_date=datetime.date(2024,2,1),
        source_type="openalex", source_id=p_gap1.source_id, evidence_text="text1",
        evidence_type="ABSTRACT", score=1.0, rank=1, provenance="DB"
    )]
    grounding = GroundedOpportunityService(
        retriever=FakeRetriever(evidence),
        generator=FakeExplanationGenerator(
            claims=[GroundedClaim("Grounded claim", [p_gap1.source_id])],
            ev_refs=[p_gap1.source_id]
        )
    )

    # Evolution
    provider_map = {
        "Supports Paper": OpportunityRelationship.SUPPORTS,
        "Addresses Paper": OpportunityRelationship.ADDRESSES,
        "Future Addresses": OpportunityRelationship.ADDRESSES,
    }
    classifier = RelationshipClassifier(ControlledProvider(provider_map))
    transition = OpportunityTransitionService(session)
    integration = OpportunityClassificationIntegration(classifier, transition, session)
    evolution = OpportunityEvolutionService(integration)

    # Orchestrator
    orchestrator = IdeaGenOrchestrator(discovery, grounding, evolution)

    # ── 4. Execution ──
    def select_first(candidates):
        return candidates[0] if candidates else None

    # Step 4a: Discover, Validate, Ground
    opp_id, candidate = orchestrator.discover_and_validate(session, signals, select_first)
    
    assert opp_id is not None
    assert candidate.cluster_size == 3
    
    opp = session.query(Opportunity).filter(Opportunity.id == opp_id).first()
    assert opp.status == OpportunityStatus.OPEN.value

    exp = orchestrator.ground_evidence(candidate)
    assert len(exp.claims) == 1
    assert exp.claims[0].evidence_refs == [p_gap1.source_id]
    
    # Verify temporal isolation (Leakage Test)
    # Processing Batch 0 with a future paper should NOT affect the opportunity.
    res_b0 = orchestrator.evolve_temporally(session, opp_id, [b0], papers)
    assert len(res_b0) == 1
    # None of the papers in b0 have mapping except they are unrelated, so they do nothing.
    assert res_b0[0].final_status == OpportunityStatus.OPEN
    
    session.refresh(opp)
    assert opp.status == OpportunityStatus.OPEN.value

    # Step 4b: Evolve B1 and B2
    results = orchestrator.evolve_temporally(session, opp_id, [b1, b2], papers)
    
    # B1 results
    assert results[0].events[0].batch_id == "b1"
    assert results[0].final_status == OpportunityStatus.STRENGTHENED
    # Events in B1: supports and unrelated
    rel_events_b1 = [e.relationship for e in results[0].events]
    assert OpportunityRelationship.SUPPORTS in rel_events_b1
    assert OpportunityRelationship.UNRELATED in rel_events_b1
    
    # B2 results
    assert results[1].events[0].batch_id == "b2"
    assert results[1].final_status == OpportunityStatus.PARTIALLY_ADDRESSED
    
    session.refresh(opp)
    assert opp.status == OpportunityStatus.PARTIALLY_ADDRESSED.value

    # Step 4c: AD-22 Duplicate Handling (Idempotency)
    # Re-running the identical batches should fail due to duplicates in DB
    with pytest.raises(ValueError, match="already exists for Opportunity"):
        orchestrator.evolve_temporally(session, opp_id, [b1], papers)


def test_failure_safety_ingestion_discovery(session):
    orchestrator = IdeaGenOrchestrator(
        OpportunityDiscoveryService(FakeGapEmbedder({}), 0.8, 3),
        None, None
    )
    # If discovery returns nothing
    with pytest.raises(ValueError, match="No candidates"):
        orchestrator.discover_and_validate(session, [], lambda x: None)
        
    # If human rejects all candidates
    signals = [
        make_signal(uuid.uuid4(), "gap", "t"),
        make_signal(uuid.uuid4(), "gap", "t"),
        make_signal(uuid.uuid4(), "gap", "t")
    ]
    orchestrator.discovery = OpportunityDiscoveryService(FakeGapEmbedder({"gap": [1.0]}), 0.8, 3)
    
    with pytest.raises(ValueError, match="No candidate selected"):
        orchestrator.discover_and_validate(session, signals, lambda x: None)
