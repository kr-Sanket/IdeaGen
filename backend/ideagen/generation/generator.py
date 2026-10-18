"""Opportunity Generator Engine (Phase 5).

Generates evidence-grounded candidate research opportunities from validated research gaps
using structured LLM prompts backed by the LLMProvider interface.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Dict, Any
import uuid
from pydantic import BaseModel, Field

from ideagen.llm.provider import LLMProvider
from ideagen.llm.exceptions import LLMError, ProviderFailureError
from ideagen.generation.models import (
    ResearchOpportunityInputDTO,
    CandidateOpportunity,
    FeasibilityAssessment,
)
from ideagen.generation.validator import sanitize_novelty_language

logger = logging.getLogger(__name__)


class LLMCandidateOpportunitySchema(BaseModel):
    """Internal Pydantic schema for structured LLM generation response."""
    proposed_research_direction: str = Field(description="Clear research hypothesis and target direction")
    reasoning: str = Field(description="Detailed justification linking proposed direction to validated gap evidence")
    technical_approach: str = Field(description="Concrete methodology, architecture, or algorithmic approach")
    expected_contribution: str = Field(description="Scientific or practical impact expected")
    required_data: str = Field(description="Datasets needed for evaluation")
    required_compute: str = Field(description="Compute requirements (GPUs, memory)")
    implementation_complexity: str = Field(description="Complexity evaluation")
    experimental_setup: str = Field(description="Experimental baselines and evaluation protocol")
    feasibility_score: float = Field(default=0.8, ge=0.0, le=1.0)
    confidence: float = Field(default=0.75, ge=0.0, le=1.0)


class OpportunityGenerator:
    """Generates grounded candidate research opportunities from validated research gaps."""

    def __init__(self, llm_provider: Optional[LLMProvider] = None, model_name: Optional[str] = None):
        self.llm_provider = llm_provider
        self.model_name = model_name or getattr(llm_provider, "model_name", "gemini-2.5-flash")

    def generate_candidate(self, input_dto: ResearchOpportunityInputDTO) -> CandidateOpportunity:
        """Generates a candidate research opportunity grounded in the input evidence using LLM."""
        if self.llm_provider is None:
            raise ProviderFailureError("LLM provider is not configured for research opportunity generation.")

        prompt = self._build_generation_prompt(input_dto)
        try:
            llm_response = self.llm_provider.generate_structured(
                prompt=prompt,
                response_schema=LLMCandidateOpportunitySchema
            )
            return self._map_llm_response_to_candidate(input_dto, llm_response)
        except Exception as e:
            logger.error(f"LLM candidate opportunity generation failed: {e}")
            if isinstance(e, LLMError):
                raise
            raise ProviderFailureError(f"LLM candidate opportunity generation failed: {e}") from e

    def _build_generation_prompt(self, input_dto: ResearchOpportunityInputDTO) -> str:
        limitations_str = "\n- ".join(input_dto.recurring_limitations) if input_dto.recurring_limitations else "None specified"
        future_work_str = "\n- ".join(input_dto.future_work_statements) if input_dto.future_work_statements else "None specified"
        methods_str = "\n- ".join(input_dto.existing_methods) if input_dto.existing_methods else "None specified"
        statements_str = "\n- ".join(input_dto.supporting_statements_text) if input_dto.supporting_statements_text else "None specified"

        return f"""
You are an expert scientific researcher. Analyze the following VALIDATED RESEARCH GAP and grounded literature evidence to propose a concrete, evidence-backed research opportunity.

VALIDATED RESEARCH GAP:
- Gap Statement: {input_dto.gap_statement}
- Research Problem: {input_dto.research_problem}

GROUNDED EVIDENCE FROM CORPUS:
- Supporting Research Statements:
- {statements_str}

- Recurring Limitations in Existing Work:
- {limitations_str}

- Future Work Directions Mentioned in Literature:
- {future_work_str}

- Existing Baseline Methods:
- {methods_str}

INSTRUCTIONS:
1. Propose a concrete research direction addressing the gap above.
2. Ensure the proposal directly references and builds upon the provided limitations and methods.
3. Clearly explain what should be investigated, why it addresses the gap, and what technical approach to take.
4. Assess feasibility carefully considering data, compute, complexity, and experimental setup.
5. NEVER claim absolute novelty (do NOT use terms like 'novel research idea', 'guaranteed novel', or 'first ever').
"""

    def _map_llm_response_to_candidate(
        self,
        input_dto: ResearchOpportunityInputDTO,
        response: LLMCandidateOpportunitySchema
    ) -> CandidateOpportunity:
        feasibility = FeasibilityAssessment(
            required_data=sanitize_novelty_language(response.required_data),
            required_compute=sanitize_novelty_language(response.required_compute),
            implementation_complexity=sanitize_novelty_language(response.implementation_complexity),
            experimental_setup=sanitize_novelty_language(response.experimental_setup),
            feasibility_score=response.feasibility_score,
            summary=f"Data: {response.required_data}; Compute: {response.required_compute}",
        )

        return CandidateOpportunity(
            research_problem=sanitize_novelty_language(input_dto.research_problem),
            identified_gap=sanitize_novelty_language(input_dto.gap_statement),
            proposed_research_direction=sanitize_novelty_language(response.proposed_research_direction),
            reasoning=sanitize_novelty_language(response.reasoning),
            technical_approach=sanitize_novelty_language(response.technical_approach),
            expected_contribution=sanitize_novelty_language(response.expected_contribution),
            feasibility=feasibility,
            supporting_statement_ids=list(input_dto.supporting_statement_ids),
            supporting_paper_ids=list(input_dto.supporting_paper_ids),
            confidence=response.confidence,
        )
