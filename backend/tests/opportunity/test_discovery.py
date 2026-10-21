import uuid
import pytest
from typing import List

from ideagen.opportunity.discovery import (
    ResearchGapSignal,
    OpportunityCandidate,
    OpportunityDiscoveryService,
    cosine_similarity,
    GapEmbedder
)

class FakeGapEmbedder:
    def __init__(self, mapping=None, fail=False):
        self.mapping = mapping or {}
        self.fail = fail
        
    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.fail:
            raise RuntimeError("Embedder failure")
        results = []
        for text in texts:
            vec = [0.0, 0.0, 1.0] # Default fallback
            # Simple keyword matching to assign deterministic vectors
            for key, mapped_vec in self.mapping.items():
                if key in text:
                    vec = mapped_vec
                    break
            results.append(vec)
        return results

@pytest.fixture
def fake_embedder():
    mapping = {
        "memory overhead": [1.0, 0.0, 0.0],
        "latency issue": [0.0, 1.0, 0.0],
        "unrelated gap": [0.0, 0.0, 1.0],
        "borderline": [0.707, 0.707, 0.0],
        "unique": [0.5, 0.5, 0.5]
    }
    return FakeGapEmbedder(mapping)

@pytest.fixture
def discovery_service(fake_embedder):
    return OpportunityDiscoveryService(fake_embedder, similarity_threshold=0.8, min_cluster_size=2)

def make_signal(statement: str, p_id: uuid.UUID = None) -> ResearchGapSignal:
    return ResearchGapSignal(
        id=uuid.uuid4(),
        paper_id=p_id or uuid.uuid4(),
        statement=statement,
        normalized_statement=""
    )

def test_normalization_whitespace(discovery_service):
    raw = "   This   is a    gap.  "
    assert discovery_service.normalize(raw) == "this is a gap"

def test_normalization_preserves_domain_terms(discovery_service):
    raw = "The P-NP problem remains unresolved."
    assert discovery_service.normalize(raw) == "the p-np problem remains unresolved"

def test_cosine_similarity():
    v1 = [1.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0]
    assert cosine_similarity(v1, v2) == 0.0
    assert abs(cosine_similarity(v1, [1.0, 1.0, 0.0]) - 0.707) < 0.01

def test_empty_input_behavior(discovery_service):
    assert discovery_service.discover_opportunities([]) == []

def test_one_signal_input_behavior(discovery_service):
    sig = make_signal("memory overhead is bad")
    assert discovery_service.discover_opportunities([sig]) == []

def test_configurable_threshold(fake_embedder):
    sig1 = make_signal("memory overhead")
    sig2 = make_signal("borderline memory") # gets [0.707, 0.707, 0]
    
    # Cosine sim is 0.707
    strict_service = OpportunityDiscoveryService(fake_embedder, similarity_threshold=0.8, min_cluster_size=2)
    assert len(strict_service.discover_opportunities([sig1, sig2])) == 0
    
    loose_service = OpportunityDiscoveryService(fake_embedder, similarity_threshold=0.7, min_cluster_size=2)
    assert len(loose_service.discover_opportunities([sig1, sig2])) == 1

def test_recurring_cluster_formation(discovery_service):
    p1, p2, p3 = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    signals = [
        make_signal("high memory overhead is a problem", p1),
        make_signal("suffers from memory overhead", p2),
        make_signal("the memory overhead restricts scaling", p3),
        make_signal("latency issue found", uuid.uuid4())
    ]
    
    candidates = discovery_service.discover_opportunities(signals)
    assert len(candidates) == 1
    
    candidate = candidates[0]
    assert candidate.cluster_size == 3
    assert len(candidate.supporting_paper_ids) == 3
    assert set(candidate.supporting_paper_ids) == {p1, p2, p3}
    assert "memory overhead" in candidate.title
    assert "latency" not in candidate.problem_statement

def test_isolated_and_unrelated_statements(discovery_service):
    signals = [
        make_signal("memory overhead 1", uuid.uuid4()),
        make_signal("memory overhead 2", uuid.uuid4()),
        make_signal("completely unrelated gap", uuid.uuid4()),
        make_signal("unique problem", uuid.uuid4())
    ]
    
    candidates = discovery_service.discover_opportunities(signals)
    assert len(candidates) == 1
    # Only memory overhead clustered
    assert candidates[0].cluster_size == 2

def test_duplicate_handling(discovery_service):
    # Two identical statements from different papers
    p1, p2 = uuid.uuid4(), uuid.uuid4()
    signals = [
        make_signal("memory overhead", p1),
        make_signal("memory overhead", p2)
    ]
    
    candidates = discovery_service.discover_opportunities(signals)
    assert len(candidates) == 1
    assert candidates[0].cluster_size == 2

def test_minimum_cluster_size(fake_embedder):
    service = OpportunityDiscoveryService(fake_embedder, similarity_threshold=0.8, min_cluster_size=3)
    signals = [
        make_signal("memory overhead 1", uuid.uuid4()),
        make_signal("memory overhead 2", uuid.uuid4())
    ]
    # Cluster size is 2, min is 3
    assert len(service.discover_opportunities(signals)) == 0

def test_cluster_evidence_traceability(discovery_service):
    s1 = make_signal("latency issue 1")
    s2 = make_signal("latency issue 2")
    
    candidates = discovery_service.discover_opportunities([s1, s2])
    assert len(candidates) == 1
    
    candidate = candidates[0]
    assert s1.id in candidate.supporting_signal_ids
    assert s2.id in candidate.supporting_signal_ids
    assert candidate.similarity_summary is not None
    # Key for the pair
    assert any(str(s1.id) in k and str(s2.id) in k for k in candidate.similarity_summary.keys())

def test_embedder_failure_propagation():
    embedder = FakeGapEmbedder(fail=True)
    service = OpportunityDiscoveryService(embedder, similarity_threshold=0.8, min_cluster_size=2)
    sig = make_signal("test")
    
    with pytest.raises(RuntimeError, match="Embedder failure"):
        service.discover_opportunities([sig, sig])

def test_deterministic_output(discovery_service):
    s1 = make_signal("memory overhead 1")
    s2 = make_signal("memory overhead 2")
    
    res1 = discovery_service.discover_opportunities([s1, s2])
    res2 = discovery_service.discover_opportunities([s1, s2])
    
    assert res1[0].cluster_id == res2[0].cluster_id
    assert res1[0].title == res2[0].title

def test_no_unsupported_claims_introduced(discovery_service):
    s1 = make_signal("memory overhead A")
    s2 = make_signal("memory overhead B")
    
    candidate = discovery_service.discover_opportunities([s1, s2])[0]
    # The problem statement should just be a combination of existing statements
    assert "memory overhead A" in candidate.problem_statement
    assert "memory overhead B" in candidate.problem_statement
    assert "novel architecture" not in candidate.problem_statement # No hallucination

def test_malformed_missing_signal_handling(discovery_service):
    # Empty statement but normalizer handles it
    s1 = make_signal("")
    s2 = make_signal("")
    candidates = discovery_service.discover_opportunities([s1, s2])
    # they will fallback to [0,0,1], cluster together
    assert len(candidates) == 1


def test_single_paper_signals_produce_documented_gap(discovery_service):
    p1 = uuid.uuid4()
    s1 = make_signal("high memory overhead is an issue", p_id=p1)
    s2 = make_signal("the memory overhead restricts scaling", p_id=p1)

    candidates = discovery_service.discover_opportunities([s1, s2])
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.cluster_size == 2
    assert len(candidate.supporting_paper_ids) == 1
    assert candidate.supporting_paper_ids == [p1]
    assert candidate.title.startswith("Documented Gap: ")
    assert not candidate.title.startswith("Recurring Gap: ")
    assert "within 1 paper" in candidate.description
    assert "2 similar statements within 1 paper" in candidate.description


def test_multi_paper_signals_produce_recurring_gap(discovery_service):
    p1, p2 = uuid.uuid4(), uuid.uuid4()
    s1 = make_signal("high memory overhead is an issue", p_id=p1)
    s2 = make_signal("the memory overhead restricts scaling", p_id=p2)

    candidates = discovery_service.discover_opportunities([s1, s2])
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.cluster_size == 2
    assert len(candidate.supporting_paper_ids) == 2
    assert set(candidate.supporting_paper_ids) == {p1, p2}
    assert candidate.title.startswith("Recurring Gap: ")
    assert not candidate.title.startswith("Documented Gap: ")
    assert "across 2 papers" in candidate.description
    assert "2 similar statements across 2 papers" in candidate.description
