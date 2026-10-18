import datetime
import uuid
from dataclasses import dataclass
from typing import List, Optional, Protocol

@dataclass(frozen=True)
class EvidenceResult:
    """A single retrieved evidence item from the corpus."""
    paper_id: uuid.UUID
    title: str
    publication_date: Optional[datetime.date]
    source_type: str
    source_id: str
    evidence_text: str
    evidence_type: str  # e.g., "ABSTRACT", "FULL_TEXT"
    score: float
    rank: int
    provenance: str

class EvidenceRetriever(Protocol):
    """Protocol for retrieving source evidence to support an opportunity."""
    def retrieve(self, query: str, top_k: int = 5) -> List[EvidenceResult]:
        ...
