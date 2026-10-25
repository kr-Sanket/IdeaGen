"""Prior-Art Retrieval and Validation Engine (Phase 5).

Reuses the Phase 4 FAISS vector index (StatementVectorIndex) and EmbeddingProvider to retrieve
semantically related prior-art research statements, and performs LLM-based overlap critique.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Set
import uuid
import numpy as np
from pydantic import BaseModel, Field

from ideagen.discovery.vector_index import StatementVectorIndex, StatementSearchResult
from ideagen.discovery.embeddings import EmbeddingProvider
from ideagen.llm.provider import LLMProvider
from ideagen.llm.exceptions import LLMError, ProviderFailureError
from ideagen.generation.models import CandidateOpportunity, PriorArtAssessment
from ideagen.generation.validator import sanitize_novelty_language

logger = logging.getLogger(__name__)


class LLMPriorArtValidationSchema(BaseModel):
    """Internal Pydantic schema for structured LLM prior-art critique."""
    assessment: str = Field(description="HIGH_OVERLAP, PARTIAL_OVERLAP, LOW_OVERLAP, or DIFFERENTIATED")
    reasoning: str = Field(description="Detailed critique comparing proposed opportunity against retrieved prior work")
    differentiation: str = Field(description="Specific points where the proposed direction differs from prior art")
    overlapping_statement_indices: List[int] = Field(default_factory=list, description="0-based indices of retrieved statements that overlap")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class PriorArtRetriever:
    """Retrieves semantically related statements from the Phase 4 FAISS StatementVectorIndex."""

    def __init__(self, vector_index: StatementVectorIndex, embedder: EmbeddingProvider):
        self.vector_index = vector_index
        self.embedder = embedder

    def retrieve_prior_art(
        self,
        query_text: str,
        top_k: int = 10,
        min_similarity: float = 0.2
    ) -> List[StatementSearchResult]:
        """Embeds query_text and retrieves top_k nearest prior-art statements from FAISS."""
        if not query_text or self.vector_index.total_vectors == 0:
            return []

        embeddings = self.embedder.embed_texts([query_text])
        if embeddings.shape[0] == 0:
            return []

        query_vector = embeddings[0]
        results = self.vector_index.search(query_vector=query_vector, top_k=top_k, min_similarity=min_similarity)
        return results


class PriorArtValidator:
    """Evaluates candidate opportunities against retrieved prior art for overlap and differentiation using LLM."""

    def __init__(self, llm_provider: Optional[LLMProvider] = None, model_name: Optional[str] = None):
        self.llm_provider = llm_provider
        self.model_name = model_name or getattr(llm_provider, "model_name", "gemini-2.5-flash")

    def validate_prior_art(
        self,
        candidate: CandidateOpportunity,
        retrieved_art: List[StatementSearchResult]
    ) -> PriorArtAssessment:
        """Compares candidate opportunity against retrieved prior art using LLM.

        Raises:
            ProviderFailureError: If LLM provider is unavailable or validation fails.
        """
        if not retrieved_art:
            return PriorArtAssessment(
                assessment="LOW_OVERLAP",
                overlapping_statement_ids=[],
                overlapping_paper_ids=[],
                reasoning="No closely matching prior-art statements were retrieved from the analyzed corpus.",
                differentiation="The proposed direction exhibits no direct textual overlap with indexed statements in the corpus.",
                confidence=0.85,
            )

        if self.llm_provider is None:
            raise ProviderFailureError("LLM provider is not configured for prior-art validation.")

        try:
            prompt = self._build_prior_art_prompt(candidate, retrieved_art)
            llm_response = self.llm_provider.generate_structured(
                prompt=prompt,
                response_schema=LLMPriorArtValidationSchema
            )
            return self._map_llm_response(retrieved_art, llm_response)
        except Exception as e:
            logger.error(f"LLM prior-art validation failed: {e}")
            if isinstance(e, LLMError):
                raise
            raise ProviderFailureError(f"LLM prior-art validation failed: {e}") from e

    def _build_prior_art_prompt(
        self,
        candidate: CandidateOpportunity,
        retrieved_art: List[StatementSearchResult]
    ) -> str:
        art_lines = []
        for idx, art in enumerate(retrieved_art):
            art_lines.append(
                f"[{idx}] Paper ID: {art.paper_id} | Type: {art.statement_type} | Similarity: {art.similarity:.3f}\n"
                f"    Text: \"{art.statement_text}\""
            )
        art_formatted = "\n".join(art_lines)

        return f"""
Compare the PROPOSED RESEARCH OPPORTUNITY against the RETRIEVED PRIOR-ART STATEMENTS from existing literature.

PROPOSED RESEARCH OPPORTUNITY:
- Problem: {candidate.research_problem}
- Identified Gap: {candidate.identified_gap}
- Proposed Direction: {candidate.proposed_research_direction}
- Technical Approach: {candidate.technical_approach}

RETRIEVED PRIOR-ART STATEMENTS:
{art_formatted}

INSTRUCTIONS:
1. Determine if the proposed direction is already substantially explored in the prior art.
2. Classify assessment as one of: 'HIGH_OVERLAP' (substantially explored), 'PARTIAL_OVERLAP' (partially explored), 'LOW_OVERLAP' (minor overlap), or 'DIFFERENTIATED' (distinct approach).
3. Identify 0-based indices of retrieved statements that strongly overlap.
4. Explain specific differentiation points.
5. NEVER claim 'this is definitely novel'. Use cautious language ('Potentially novel within the analyzed corpus').
"""

    def _map_llm_response(
        self,
        retrieved_art: List[StatementSearchResult],
        response: LLMPriorArtValidationSchema
    ) -> PriorArtAssessment:
        overlapping_stmt_ids: List[uuid.UUID] = []
        overlapping_paper_ids_set: Set[uuid.UUID] = set()

        for idx in response.overlapping_statement_indices:
            if 0 <= idx < len(retrieved_art):
                art = retrieved_art[idx]
                overlapping_stmt_ids.append(art.statement_id)
                overlapping_paper_ids_set.add(art.paper_id)

        assessment_str = response.assessment.upper()
        if assessment_str not in {"HIGH_OVERLAP", "PARTIAL_OVERLAP", "LOW_OVERLAP", "DIFFERENTIATED"}:
            assessment_str = "PARTIAL_OVERLAP"

        reasoning = sanitize_novelty_language(response.reasoning)
        differentiation = sanitize_novelty_language(response.differentiation)

        return PriorArtAssessment(
            assessment=assessment_str,
            overlapping_statement_ids=overlapping_stmt_ids,
            overlapping_paper_ids=list(overlapping_paper_ids_set),
            reasoning=reasoning,
            differentiation=differentiation,
            confidence=response.confidence,
        )
