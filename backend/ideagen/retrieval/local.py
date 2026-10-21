from typing import List, Protocol

from ideagen.opportunity.discovery import cosine_similarity
from ideagen.retrieval.evidence import EvidenceResult

class PaperEvidenceEmbedder(Protocol):
    """Protocol for embedding paper text for evidence retrieval."""
    def embed_query(self, text: str) -> List[float]:
        ...
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...

class LocalEvidenceRetriever:
    """A deterministic, local in-memory retriever for controlled tests."""
    
    def __init__(self, embedder: PaperEvidenceEmbedder, corpus: List[EvidenceResult]):
        self.embedder = embedder
        self.corpus = corpus
        self.corpus_embeddings = self.embedder.embed_documents([doc.evidence_text for doc in corpus]) if corpus else []

    def retrieve(self, query: str, top_k: int = 5) -> List[EvidenceResult]:
        if not self.corpus:
            return []
            
        q_emb = self.embedder.embed_query(query)
        scored = []
        for i, (doc, doc_emb) in enumerate(zip(self.corpus, self.corpus_embeddings)):
            sim = cosine_similarity(q_emb, doc_emb)
            scored.append((sim, str(doc.paper_id), i, doc))
            
        # Sort by descending score, then deterministically by paper_id, then original index
        scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        
        results = []
        for rank, (score, _, _, doc) in enumerate(scored[:top_k], start=1):
            results.append(EvidenceResult(
                paper_id=doc.paper_id,
                title=doc.title,
                publication_date=doc.publication_date,
                source_type=doc.source_type,
                source_id=doc.source_id,
                evidence_text=doc.evidence_text,
                evidence_type=doc.evidence_type,
                score=score,
                rank=rank,
                provenance=doc.provenance
            ))
        return results
