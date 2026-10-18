import uuid
import pytest
from typing import List

from ideagen.opportunity.discovery import OpportunityCandidate
from ideagen.retrieval.evidence import EvidenceResult
from ideagen.rag.explanation import (
    GroundedOpportunityService, 
    ExplanationGenerator,
    GroundedExplanation,
    GroundedClaim,
    UnsupportedClaim
)


class FakeRetriever:
    def __init__(self, evidence: List[EvidenceResult]):
        self.evidence = evidence
        
    def retrieve(self, query: str, top_k: int = 5) -> List[EvidenceResult]:
        return self.evidence[:top_k]


class FakeExplanationGenerator:
    def __init__(self, claims=None, unsupported=None, ev_refs=None, fail=False):
        self.claims = claims or []
        self.unsupported = unsupported or []
        self.ev_refs = ev_refs or []
        self.fail = fail

    def generate(self, candidate: OpportunityCandidate, evidence: List[EvidenceResult]) -> GroundedExplanation:
        if self.fail:
            raise RuntimeError("Generator failure")
            
        # Default behavior if not explicitly configured
        if not self.claims and not self.fail and evidence:
            self.claims = [GroundedClaim("default claim", [str(evidence[0].source_id)])]
            self.ev_refs = [str(evidence[0].source_id)]
            
        return GroundedExplanation(
            summary="Test Summary",
            claims=self.claims,
            unsupported_claims=self.unsupported,
            evidence_refs=self.ev_refs,
            generation_metadata={"fake": True}
        )

def make_candidate() -> OpportunityCandidate:
    return OpportunityCandidate(
        cluster_id="cluster-1",
        title="Test Opportunity",
        problem_statement="Test Problem",
        description="Test Desc",
        supporting_paper_ids=[uuid.uuid4()],
        supporting_signal_ids=[uuid.uuid4()],
        cluster_size=1,
        confidence=1.0,
        similarity_summary={}
    )


def make_evidence(s_id: str) -> EvidenceResult:
    from ideagen.retrieval.evidence import EvidenceResult
    import datetime
    return EvidenceResult(
        paper_id=uuid.uuid4(),
        title="Title",
        publication_date=datetime.date(2023, 1, 1),
        source_type="test",
        source_id=s_id,
        evidence_text="test text",
        evidence_type="ABSTRACT",
        score=1.0,
        rank=1,
        provenance="Test Corpus"
    )

@pytest.fixture
def evidence():
    return [make_evidence("source-1"), make_evidence("source-2")]

@pytest.fixture
def candidate():
    return make_candidate()


def test_explanation_generated_from_evidence(candidate, evidence):
    retriever = FakeRetriever(evidence)
    generator = FakeExplanationGenerator()
    service = GroundedOpportunityService(retriever, generator)
    
    explanation = service.ground(candidate)
    assert explanation.summary == "Test Summary"
    assert len(explanation.claims) == 1


def test_evidence_references_preserved(candidate, evidence):
    retriever = FakeRetriever(evidence)
    generator = FakeExplanationGenerator(
        claims=[GroundedClaim("c1", ["source-1"])],
        ev_refs=["source-1"]
    )
    service = GroundedOpportunityService(retriever, generator)
    
    explanation = service.ground(candidate)
    assert "source-1" in explanation.claims[0].evidence_refs


def test_unknown_evidence_reference_rejected(candidate, evidence):
    retriever = FakeRetriever(evidence)
    generator = FakeExplanationGenerator(
        claims=[GroundedClaim("c1", ["fake-source"])],
        ev_refs=["fake-source"]
    )
    service = GroundedOpportunityService(retriever, generator)
    
    with pytest.raises(ValueError, match="Fabricated citation detected: fake-source"):
        service.ground(candidate)


def test_fabricated_paper_id_rejected(candidate, evidence):
    retriever = FakeRetriever(evidence)
    generator = FakeExplanationGenerator(
        claims=[GroundedClaim("c1", ["source-1"])],
        ev_refs=["source-1", "fabricated-paper-id"]
    )
    service = GroundedOpportunityService(retriever, generator)
    
    with pytest.raises(ValueError, match="Fabricated citation detected: fabricated-paper-id"):
        service.ground(candidate)


def test_fabricated_doi_rejected(candidate, evidence):
    retriever = FakeRetriever(evidence)
    generator = FakeExplanationGenerator(
        claims=[GroundedClaim("c1", ["10.1234/fake.doi"])],
        ev_refs=["10.1234/fake.doi"]
    )
    service = GroundedOpportunityService(retriever, generator)
    
    with pytest.raises(ValueError, match="Fabricated citation detected: 10.1234/fake.doi"):
        service.ground(candidate)


def test_unsupported_factual_claim_rejected_flagged(candidate, evidence):
    retriever = FakeRetriever(evidence)
    generator = FakeExplanationGenerator(
        claims=[GroundedClaim("c1", ["source-1"])],
        unsupported=[UnsupportedClaim("c2", "No evidence found")],
        ev_refs=["source-1"]
    )
    service = GroundedOpportunityService(retriever, generator)
    
    exp = service.ground(candidate)
    assert len(exp.unsupported_claims) == 1
    assert exp.unsupported_claims[0].text == "c2"


def test_multiple_evidence_references_supported(candidate, evidence):
    retriever = FakeRetriever(evidence)
    generator = FakeExplanationGenerator(
        claims=[GroundedClaim("c1", ["source-1", "source-2"])],
        ev_refs=["source-1", "source-2"]
    )
    service = GroundedOpportunityService(retriever, generator)
    
    exp = service.ground(candidate)
    assert len(exp.claims[0].evidence_refs) == 2


def test_empty_evidence_produces_no_grounded_factual_claim(candidate):
    retriever = FakeRetriever([])
    generator = FakeExplanationGenerator(claims=[], ev_refs=[])
    service = GroundedOpportunityService(retriever, generator)
    
    exp = service.ground(candidate)
    assert len(exp.claims) == 0


def test_provider_failure_propagates(candidate, evidence):
    retriever = FakeRetriever(evidence)
    generator = FakeExplanationGenerator(fail=True)
    service = GroundedOpportunityService(retriever, generator)
    
    with pytest.raises(RuntimeError, match="Generator failure"):
        service.ground(candidate)


def test_deterministic_fake_provider_behavior(candidate, evidence):
    retriever = FakeRetriever(evidence)
    generator = FakeExplanationGenerator()
    service = GroundedOpportunityService(retriever, generator)
    
    exp1 = service.ground(candidate)
    exp2 = service.ground(candidate)
    
    assert exp1.claims[0].text == exp2.claims[0].text

def test_malformed_llm_output_rejected(candidate, evidence):
    retriever = FakeRetriever(evidence)
    # Simulate malformed output by putting non-list in evidence_refs
    generator = FakeExplanationGenerator(
        claims=[GroundedClaim("c1", "source-1")], # should be list
        ev_refs=["source-1"]
    )
    service = GroundedOpportunityService(retriever, generator)
    
    with pytest.raises(ValueError):
        # We expect a value error because a string will be iterated over character by character
        service.ground(candidate)
