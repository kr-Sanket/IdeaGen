import datetime
import math
import re
import unicodedata
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, Set, Tuple

@dataclass(frozen=True)
class ResearchGapSignal:
    """A research gap or limitation extracted from a paper."""
    id: uuid.UUID
    paper_id: uuid.UUID
    statement: str
    normalized_statement: str
    source_type: str = "manual"
    confidence: float = 1.0
    created_at: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))

@dataclass(frozen=True)
class OpportunityCandidate:
    """A candidate opportunity formed from recurring research gaps."""
    cluster_id: str
    title: str
    problem_statement: str
    description: str
    supporting_paper_ids: List[uuid.UUID]
    supporting_signal_ids: List[uuid.UUID]
    cluster_size: int
    confidence: float
    similarity_summary: Dict[str, Any]

class GapEmbedder(Protocol):
    """Protocol for semantic embedding of gap statements."""
    def embed(self, texts: List[str]) -> List[List[float]]:
        ...

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes cosine similarity between two vectors."""
    if not v1 or not v2:
        return 0.0
    dot_prod = sum(x * y for x, y in zip(v1, v2))
    mag1 = math.sqrt(sum(x * x for x in v1))
    mag2 = math.sqrt(sum(x * x for x in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_prod / (mag1 * mag2)

class OpportunityDiscoveryService:
    """Discovers recurring research gaps from signals to form candidate opportunities."""
    
    def __init__(self, embedder: GapEmbedder, similarity_threshold: float = 0.8, min_cluster_size: int = 2):
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size

    def normalize(self, statement: str) -> str:
        """Normalizes a statement while preserving domain terms."""
        # Unicode normalize
        text = unicodedata.normalize("NFKC", statement)
        # Lowercase for semantic uniformity
        text = text.lower()
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text)
        # Strip trailing punctuation/whitespace
        text = text.strip(" .;,")
        return text
        
    def _find_connected_components(self, num_nodes: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
        """Finds connected components in a graph using DFS."""
        adj_list = {i: [] for i in range(num_nodes)}
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
            
        visited = set()
        components = []
        
        for i in range(num_nodes):
            if i not in visited:
                comp = []
                stack = [i]
                while stack:
                    node = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        comp.append(node)
                        stack.extend(adj_list[node])
                components.append(comp)
        return components

    def discover_opportunities(self, signals: List[ResearchGapSignal]) -> List[OpportunityCandidate]:
        """Runs the discovery pipeline over a set of signals."""
        if not signals:
            return []
            
        # 1. Normalize
        normalized_texts = [sig.normalized_statement or self.normalize(sig.statement) for sig in signals]
        
        # 2. Embed
        embeddings = self.embedder.embed(normalized_texts)
        
        # 3. Calculate similarities & build edges
        n = len(signals)
        edges = []
        similarity_cache = {}
        for i in range(n):
            for j in range(i + 1, n):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                similarity_cache[(i, j)] = sim
                if sim >= self.similarity_threshold:
                    edges.append((i, j))
                    
        # 4. Cluster using connected components
        components = self._find_connected_components(n, edges)
        
        # 5. Filter and form candidates
        candidates = []
        for idx, comp in enumerate(components):
            if len(comp) < self.min_cluster_size:
                continue
                
            cluster_signals = [signals[i] for i in comp]
            supporting_paper_ids = list({sig.paper_id for sig in cluster_signals})
            
            # Form candidate fields
            shortest_sig = min(cluster_signals, key=lambda s: len(s.normalized_statement or s.statement))
            
            title = f"Recurring Gap: {shortest_sig.normalized_statement or shortest_sig.statement}"
            problem_statement = " | ".join([s.statement for s in cluster_signals])
            description = f"Discovered from {len(cluster_signals)} similar statements across {len(supporting_paper_ids)} papers."
            
            # Summarize similarities within cluster
            sim_summary = {}
            for i in range(len(comp)):
                for j in range(i + 1, len(comp)):
                    orig_i, orig_j = comp[i], comp[j]
                    pair_key = (min(orig_i, orig_j), max(orig_i, orig_j))
                    sim_summary[f"{signals[orig_i].id}_{signals[orig_j].id}"] = similarity_cache.get(pair_key, 0.0)
            
            candidate = OpportunityCandidate(
                cluster_id=f"cluster-{idx}",
                title=title[:200],  # truncate if too long
                problem_statement=problem_statement,
                description=description,
                supporting_paper_ids=supporting_paper_ids,
                supporting_signal_ids=[s.id for s in cluster_signals],
                cluster_size=len(cluster_signals),
                confidence=min([s.confidence for s in cluster_signals]), # conservative confidence
                similarity_summary=sim_summary
            )
            candidates.append(candidate)
            
        return candidates
