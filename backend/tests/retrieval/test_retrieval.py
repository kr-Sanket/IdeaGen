import uuid
import datetime
import pytest
from typing import List

from ideagen.retrieval.evidence import EvidenceResult
from ideagen.retrieval.local import LocalEvidenceRetriever, PaperEvidenceEmbedder


class FakePaperEvidenceEmbedder:
    def __init__(self, mapping=None, fail=False):
        self.mapping = mapping or {}
        self.fail = fail

    def embed_query(self, text: str) -> List[float]:
        if self.fail:
            raise RuntimeError("Embedder failure")
        return self._embed(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.fail:
            raise RuntimeError("Embedder failure")
        return [self._embed(t) for t in texts]

    def _embed(self, text: str) -> List[float]:
        vec = [0.0, 0.0, 1.0] # Default vector
        for key, mapped_vec in self.mapping.items():
            if key.lower() in text.lower():
                vec = mapped_vec
                break
        return vec


def make_evidence(text: str, p_id: uuid.UUID = None, s_id: str = None) -> EvidenceResult:
    return EvidenceResult(
        paper_id=p_id or uuid.uuid4(),
        title=f"Title for {text[:10]}",
        publication_date=datetime.date(2023, 1, 1),
        source_type="openalex",
        source_id=s_id or str(uuid.uuid4()),
        evidence_text=text,
        evidence_type="ABSTRACT",
        score=0.0,
        rank=0,
        provenance="Test Corpus"
    )

@pytest.fixture
def fake_embedder():
    mapping = {
        "RAG systems": [1.0, 0.0, 0.0],
        "distractor": [0.707, 0.707, 0.0],
        "unrelated": [0.0, 1.0, 0.0]
    }
    return FakePaperEvidenceEmbedder(mapping)


def test_empty_corpus(fake_embedder):
    retriever = LocalEvidenceRetriever(fake_embedder, [])
    assert retriever.retrieve("query") == []


def test_empty_query(fake_embedder):
    e1 = make_evidence("RAG systems are great")
    retriever = LocalEvidenceRetriever(fake_embedder, [e1])
    # Empty query should still return results, just scoring against the default fallback vector
    # because FakeEmbedder maps "" to [0,0,1].
    res = retriever.retrieve("")
    assert len(res) == 1
    assert res[0].score == 0.0 # [0,0,1] vs [1,0,0] is 0


def test_top_k_behavior(fake_embedder):
    corpus = [make_evidence(f"unrelated {i}") for i in range(10)]
    retriever = LocalEvidenceRetriever(fake_embedder, corpus)
    assert len(retriever.retrieve("query", top_k=3)) == 3
    assert len(retriever.retrieve("query", top_k=15)) == 10


def test_deterministic_ranking(fake_embedder):
    e1 = make_evidence("RAG systems A")
    e2 = make_evidence("RAG systems B")
    retriever = LocalEvidenceRetriever(fake_embedder, [e1, e2])
    
    res1 = retriever.retrieve("RAG systems")
    res2 = retriever.retrieve("RAG systems")
    
    assert res1[0].paper_id == res2[0].paper_id
    assert res1[1].paper_id == res2[1].paper_id


def test_relevant_and_unrelated_ranking(fake_embedder):
    e_unrelated = make_evidence("unrelated stuff")
    e_relevant = make_evidence("RAG systems research")
    e_distractor = make_evidence("distractor text")
    
    retriever = LocalEvidenceRetriever(fake_embedder, [e_unrelated, e_relevant, e_distractor])
    
    res = retriever.retrieve("RAG systems")
    
    assert res[0].paper_id == e_relevant.paper_id
    assert res[0].score == 1.0
    assert res[1].paper_id == e_distractor.paper_id
    assert abs(res[1].score - 0.707) < 0.01
    assert res[2].paper_id == e_unrelated.paper_id
    assert res[2].score == 0.0


def test_tie_breaking(fake_embedder):
    p1 = uuid.UUID(int=1)
    p2 = uuid.UUID(int=2)
    # both match "RAG systems", so score is 1.0
    e1 = make_evidence("RAG systems A", p_id=p1)
    e2 = make_evidence("RAG systems B", p_id=p2)
    
    # Even if they are added in different orders, tie breaker by UUID should make order stable.
    # UUID 2 comes after UUID 1 as string? UUID 1 is '00...001', UUID 2 is '00...002'. 
    # '00...02' > '00...01', reverse=True means p2 comes first.
    retriever = LocalEvidenceRetriever(fake_embedder, [e1, e2])
    res = retriever.retrieve("RAG systems")
    
    assert res[0].paper_id == p2
    assert res[1].paper_id == p1

def test_provenance_preservation(fake_embedder):
    e = make_evidence("RAG systems", s_id="source-123")
    retriever = LocalEvidenceRetriever(fake_embedder, [e])
    res = retriever.retrieve("RAG systems")
    assert res[0].source_id == "source-123"
    assert res[0].provenance == "Test Corpus"


def test_missing_paperdocument_handling(fake_embedder):
    # This represents abstract only, where FULL_TEXT missing is normal.
    e = make_evidence("RAG systems")
    retriever = LocalEvidenceRetriever(fake_embedder, [e])
    assert retriever.retrieve("RAG systems")[0].evidence_type == "ABSTRACT"


def test_embedder_failure_propagation():
    embedder = FakePaperEvidenceEmbedder(fail=True)
    e = make_evidence("RAG systems")
    with pytest.raises(RuntimeError, match="Embedder failure"):
        LocalEvidenceRetriever(embedder, [e])
