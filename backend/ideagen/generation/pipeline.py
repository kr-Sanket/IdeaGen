"""Research Opportunity Generation & Prior-Art Validation Pipeline (Phase 5).

Orchestrates: VALIDATED GAP -> opportunity generation -> prior-art retrieval -> critique -> refinement -> persisted opportunity.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Set
import uuid
from datetime import datetime
from sqlalchemy.orm import Session

from ideagen.discovery.vector_index import StatementVectorIndex
from ideagen.discovery.embeddings import EmbeddingProvider
from ideagen.llm.provider import LLMProvider

from ideagen.generation.models import (
    ResearchOpportunityInputDTO,
    ResearchOpportunityDTO,
    CandidateOpportunity,
    PriorArtAssessment,
    NoveltyAssessment,
    OpportunityStatus,
)
from ideagen.generation.validator import EvidenceIntegrityValidator, sanitize_novelty_language
from ideagen.generation.generator import OpportunityGenerator
from ideagen.generation.prior_art import PriorArtRetriever, PriorArtValidator
from ideagen.generation.refinement import IterativeRefinementEngine
from ideagen.generation.persistence import OpportunityPersistenceLayer

logger = logging.getLogger(__name__)


class ResearchOpportunityPipeline:
    """Phase 5 End-to-End Pipeline for Research Opportunity Generation & Validation."""

    def __init__(
        self,
        vector_index: StatementVectorIndex,
        embedder: EmbeddingProvider,
        llm_provider: Optional[LLMProvider] = None,
        max_refinement_iterations: int = 2,
        generation_model: str = "gemini-2.5-flash",
        validation_model: str = "gemini-2.5-flash",
        pipeline_version: str = "v1.0",
    ):
        self.vector_index = vector_index
        self.embedder = embedder
        self.llm_provider = llm_provider
        self.max_refinement_iterations = max_refinement_iterations

        resolved_model = getattr(llm_provider, "model_name", "gemini-2.5-flash") if llm_provider else "gemini-2.5-flash"
        self.generation_model = generation_model or resolved_model
        self.validation_model = validation_model or resolved_model
        self.embedding_model = getattr(embedder, "model_name", "nomic-embed-text")
        self.pipeline_version = pipeline_version

        # Initialize sub-components
        self.generator = OpportunityGenerator(llm_provider=llm_provider, model_name=self.generation_model)
        self.retriever = PriorArtRetriever(vector_index=vector_index, embedder=embedder)
        self.validator = PriorArtValidator(llm_provider=llm_provider, model_name=self.validation_model)
        self.refinement_engine = IterativeRefinementEngine(
            retriever=self.retriever,
            validator=self.validator,
            llm_provider=llm_provider,
            max_refinement_iterations=max_refinement_iterations,
        )

    def process_validated_gap(
        self,
        input_dto: ResearchOpportunityInputDTO,
        db_session: Optional[Session] = None,
        valid_statement_ids: Optional[Set[uuid.UUID]] = None,
        valid_paper_ids: Optional[Set[uuid.UUID]] = None,
    ) -> ResearchOpportunityDTO:
        """Executes full Phase 5 pipeline on a validated research gap input."""

        # 1. Evidence Integrity & Input Pre-Validation
        integrity_validator = EvidenceIntegrityValidator(db_session=db_session)

        # 2. Opportunity Generation
        candidate = self.generator.generate_candidate(input_dto)
        candidate = integrity_validator.validate_and_sanitize_candidate(
            candidate,
            valid_statement_ids=valid_statement_ids,
            valid_paper_ids=valid_paper_ids,
        )

        # 3. Prior-Art Retrieval
        retrieved_art = self.retriever.retrieve_prior_art(
            query_text=f"{candidate.proposed_research_direction} {candidate.technical_approach}",
            top_k=10,
        )

        # 4. Initial Prior-Art Validation / Critique
        initial_assessment = self.validator.validate_prior_art(candidate, retrieved_art)
        assessments_log: List[Tuple[int, PriorArtAssessment]] = [(0, initial_assessment)]

        # 5. Iterative Refinement
        (
            final_candidate,
            final_assessment,
            refinement_history,
            final_status,
        ) = self.refinement_engine.refine_until_differentiated(candidate, initial_assessment)

        if len(refinement_history) > 0:
            assessments_log.append((len(refinement_history), final_assessment))

        # 6. Novelty Assessment Construction (strictly bounded novelty language)
        novelty = NoveltyAssessment(
            assessment="Potentially novel within the analyzed corpus",
            reasoning=final_assessment.reasoning,
            differentiation=final_assessment.differentiation,
            confidence=final_assessment.confidence,
            overlapping_statement_ids=final_assessment.overlapping_statement_ids,
            overlapping_paper_ids=final_assessment.overlapping_paper_ids,
        )
        novelty = integrity_validator.sanitize_novelty_assessment(novelty)

        # Collect prior art statement IDs
        prior_art_stmt_ids = list(final_assessment.overlapping_statement_ids)

        # 7. Final DTO Assembly
        opportunity_dto = ResearchOpportunityDTO(
            id=uuid.uuid4(),
            gap_id=input_dto.gap_id,
            research_problem=final_candidate.research_problem,
            identified_gap=final_candidate.identified_gap,
            proposed_research_direction=final_candidate.proposed_research_direction,
            reasoning=final_candidate.reasoning,
            technical_approach=final_candidate.technical_approach,
            expected_contribution=final_candidate.expected_contribution,
            feasibility=final_candidate.feasibility,
            novelty_assessment=novelty,
            confidence=final_candidate.confidence,
            supporting_statement_ids=final_candidate.supporting_statement_ids,
            supporting_paper_ids=final_candidate.supporting_paper_ids,
            prior_art_statement_ids=prior_art_stmt_ids,
            status=final_status,
            created_at=datetime.utcnow(),
            generation_model=self.generation_model,
            validation_model=self.validation_model,
            embedding_model=self.embedding_model,
            refinement_iteration_count=len(refinement_history),
            pipeline_version=self.pipeline_version,
        )

        # 8. Database Persistence
        if db_session is not None:
            persistence = OpportunityPersistenceLayer(db_session)
            persistence.save_opportunity(
                dto=opportunity_dto,
                assessments=assessments_log,
                iterations=refinement_history,
            )

        return opportunity_dto
