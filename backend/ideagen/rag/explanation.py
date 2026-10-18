from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

from ideagen.opportunity.discovery import OpportunityCandidate
from ideagen.retrieval.evidence import EvidenceResult, EvidenceRetriever


@dataclass(frozen=True)
class GroundedClaim:
    """A factual claim supported by specific evidence."""
    text: str
    evidence_refs: List[str]  # Must map to source_ids of retrieved evidence


@dataclass(frozen=True)
class UnsupportedClaim:
    """A claim generated without adequate evidence support."""
    text: str
    reason: str


@dataclass(frozen=True)
class GroundedExplanation:
    """A verified, evidence-grounded explanation for a research opportunity."""
    summary: str
    claims: List[GroundedClaim]
    unsupported_claims: List[UnsupportedClaim]
    evidence_refs: List[str]  # Distinct list of valid references used
    generation_metadata: Dict[str, Any]


class ExplanationGenerator(Protocol):
    """Protocol for generating grounded explanations from candidates and evidence."""
    def generate(
        self, candidate: OpportunityCandidate, evidence: List[EvidenceResult]
    ) -> GroundedExplanation:
        ...


class GroundedOpportunityService:
    """Orchestrates evidence retrieval and explanation generation."""

    def __init__(self, retriever: EvidenceRetriever, generator: ExplanationGenerator):
        self.retriever = retriever
        self.generator = generator

    def ground(self, candidate: OpportunityCandidate, top_k: int = 5) -> GroundedExplanation:
        """Retrieves evidence and generates a validated explanation."""
        
        # 1. Construct query safely
        query = f"{candidate.title}. {candidate.problem_statement}"
        
        # 2. Retrieve evidence
        retrieved_evidence = self.retriever.retrieve(query, top_k=top_k)
        
        # 3. Generate explanation
        explanation = self.generator.generate(candidate, retrieved_evidence)
        
        # 4. Validate evidence references against retrieved items to prevent hallucination
        valid_source_ids = {str(e.source_id) for e in retrieved_evidence}
        
        for claim in explanation.claims:
            for ref in claim.evidence_refs:
                if str(ref) not in valid_source_ids:
                    raise ValueError(f"Fabricated citation detected: {ref}")
                    
        # Ensure top-level evidence_refs are also strictly valid
        validated_refs = []
        for ref in explanation.evidence_refs:
            if str(ref) not in valid_source_ids:
                raise ValueError(f"Fabricated citation detected: {ref}")
            validated_refs.append(str(ref))
            
        return GroundedExplanation(
            summary=explanation.summary,
            claims=explanation.claims,
            unsupported_claims=explanation.unsupported_claims,
            evidence_refs=list(set(validated_refs)),
            generation_metadata=explanation.generation_metadata
        )
