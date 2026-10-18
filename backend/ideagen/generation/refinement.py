"""Iterative Refinement Engine (Phase 5).

Manages the critique-driven direction revision loop when prior-art overlap is detected using LLM.
Limits iterations via configurable bounds and rejects opportunities that cannot be differentiated.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple
import uuid
from pydantic import BaseModel, Field

from ideagen.llm.provider import LLMProvider
from ideagen.llm.exceptions import LLMError, ProviderFailureError
from ideagen.generation.models import (
    CandidateOpportunity,
    PriorArtAssessment,
    RefinementCritique,
    OpportunityStatus,
)
from ideagen.generation.validator import sanitize_novelty_language
from ideagen.generation.prior_art import PriorArtRetriever, PriorArtValidator

logger = logging.getLogger(__name__)


class LLMRefinementSchema(BaseModel):
    """Internal Pydantic schema for structured LLM refinement critique and revision."""
    critique: str = Field(description="Critique explaining how prior art overlaps with current proposed direction")
    revised_direction: str = Field(description="Differentiated proposed research direction avoiding prior art overlap")
    revised_technical_approach: str = Field(description="Refined technical approach incorporating new constraints")
    should_reject: bool = Field(default=False, description="True if direction cannot be meaningfully differentiated")
    reason: Optional[str] = Field(default=None, description="Reason for rejection if applicable")


class IterativeRefinementEngine:
    """Orchestrates iterative refinement and prior-art re-evaluation loops."""

    def __init__(
        self,
        retriever: PriorArtRetriever,
        validator: PriorArtValidator,
        llm_provider: Optional[LLMProvider] = None,
        max_refinement_iterations: int = 2,
    ):
        self.retriever = retriever
        self.validator = validator
        self.llm_provider = llm_provider
        self.max_refinement_iterations = max_refinement_iterations

    def refine_until_differentiated(
        self,
        initial_candidate: CandidateOpportunity,
        initial_assessment: PriorArtAssessment,
    ) -> Tuple[CandidateOpportunity, PriorArtAssessment, List[Tuple[int, str, str, str, str]], OpportunityStatus]:
        """Runs the refinement loop: candidate -> critique -> revise -> retrieve -> re-evaluate.

        Returns:
            Tuple of (final_candidate, final_assessment, refinement_history, final_status)

        Raises:
            ProviderFailureError: If LLM refinement critique fails.
        """
        current_candidate = initial_candidate
        current_assessment = initial_assessment
        refinement_history: List[Tuple[int, str, str, str, str]] = []  # (iteration, original_dir, critique, revised_dir, status)

        iteration = 0
        while iteration < self.max_refinement_iterations:
            if current_assessment.assessment not in {"HIGH_OVERLAP"}:
                # Prior art is acceptable (LOW_OVERLAP, PARTIAL_OVERLAP, or DIFFERENTIATED)
                break

            iteration += 1
            logger.info(f"Refinement iteration {iteration}/{self.max_refinement_iterations} triggered for candidate.")

            critique = self._generate_refinement_critique(current_candidate, current_assessment)
            refinement_history.append((
                iteration,
                current_candidate.proposed_research_direction,
                critique.critique,
                critique.revised_direction,
                "REFINED" if not critique.should_reject else "REJECTED_INSUFFICIENTLY_DIFFERENTIATED",
            ))

            if critique.should_reject:
                logger.info("Refinement engine requested early rejection due to unresolvable overlap.")
                return (
                    current_candidate,
                    current_assessment,
                    refinement_history,
                    OpportunityStatus.REJECTED_INSUFFICIENTLY_DIFFERENTIATED,
                )

            # Update candidate direction & technical approach
            current_candidate = current_candidate.model_copy(deep=True)
            current_candidate.proposed_research_direction = sanitize_novelty_language(critique.revised_direction)
            current_candidate.reasoning += f" [Refinement Iteration {iteration}: {sanitize_novelty_language(critique.critique)}]"

            # Retrieve prior art for updated direction
            retrieved_art = self.retriever.retrieve_prior_art(
                query_text=current_candidate.proposed_research_direction,
                top_k=10
            )

            # Re-evaluate prior art overlap
            current_assessment = self.validator.validate_prior_art(current_candidate, retrieved_art)

        # Final status check after exiting refinement loop
        if current_assessment.assessment == "HIGH_OVERLAP":
            final_status = OpportunityStatus.REJECTED_INSUFFICIENTLY_DIFFERENTIATED
        elif iteration > 0:
            final_status = OpportunityStatus.REFINED
        else:
            final_status = OpportunityStatus.VALIDATED

        return current_candidate, current_assessment, refinement_history, final_status

    def _generate_refinement_critique(
        self,
        candidate: CandidateOpportunity,
        assessment: PriorArtAssessment
    ) -> RefinementCritique:
        if self.llm_provider is None:
            raise ProviderFailureError("LLM provider is not configured for refinement critique.")

        try:
            prompt = f"""
The proposed research direction has HIGH OVERLAP with prior art in the literature corpus.

PROPOSED DIRECTION:
{candidate.proposed_research_direction}

PRIOR ART OVERLAP CRITIQUE:
{assessment.reasoning}

DIFFERENTIATION ATTEMPT:
{assessment.differentiation}

INSTRUCTIONS:
1. Provide a sharp critique explaining why the proposed direction overlaps with existing work.
2. Propose a revised research direction that pivots away from the overlap while maintaining focus on resolving the gap.
3. If the direction cannot be differentiated from prior work, set should_reject=True.
"""
            response = self.llm_provider.generate_structured(
                prompt=prompt,
                response_schema=LLMRefinementSchema
            )
            return RefinementCritique(
                critique=sanitize_novelty_language(response.critique),
                revised_direction=sanitize_novelty_language(response.revised_direction),
                should_reject=response.should_reject,
                reason=sanitize_novelty_language(response.reason or ""),
            )
        except Exception as e:
            logger.error(f"LLM refinement critique failed: {e}")
            if isinstance(e, LLMError):
                raise
            raise ProviderFailureError(f"LLM refinement critique failed: {e}") from e
