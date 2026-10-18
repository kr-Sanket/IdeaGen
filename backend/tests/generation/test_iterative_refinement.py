"""Tests for Iterative Refinement Engine and Rejection Logic."""

import pytest
import uuid
import numpy as np

from ideagen.discovery.vector_index import StatementVectorIndex, StatementSearchResult
from ideagen.generation.models import CandidateOpportunity, PriorArtAssessment, FeasibilityAssessment, OpportunityStatus
from ideagen.generation.prior_art import PriorArtRetriever, PriorArtValidator
from ideagen.generation.refinement import IterativeRefinementEngine, LLMRefinementSchema
from ideagen.llm.exceptions import ProviderFailureError
from tests.generation.test_opportunity_generator import MockLLMProvider
from tests.generation.test_prior_art import FakeEmbeddingProvider


def test_refinement_raises_when_llm_provider_is_none():
    embedder = FakeEmbeddingProvider(dimension=128)
    index = StatementVectorIndex(dimension=128)
    retriever = PriorArtRetriever(vector_index=index, embedder=embedder)
    validator = PriorArtValidator(llm_provider=None)

    refinement_engine = IterativeRefinementEngine(
        retriever=retriever,
        validator=validator,
        llm_provider=None,
        max_refinement_iterations=2,
    )

    candidate = CandidateOpportunity(
        research_problem="Problem",
        identified_gap="Gap",
        proposed_research_direction="Flat index approach",
        reasoning="Reasoning",
        technical_approach="Approach",
        expected_contribution="Contribution",
        feasibility=FeasibilityAssessment(
            required_data="Data",
            required_compute="Compute",
            implementation_complexity="Med",
            experimental_setup="Setup",
        ),
    )

    initial_assessment = PriorArtAssessment(
        assessment="HIGH_OVERLAP",
        overlapping_statement_ids=[uuid.uuid4()],
        overlapping_paper_ids=[uuid.uuid4()],
        reasoning="Direct duplicate of prior work",
        differentiation="None",
        confidence=0.9,
    )

    with pytest.raises(ProviderFailureError) as exc_info:
        refinement_engine.refine_until_differentiated(candidate, initial_assessment)
    assert "LLM provider is not configured" in str(exc_info.value)


def test_refinement_with_llm_pivots_direction():
    embedder = FakeEmbeddingProvider(dimension=128)
    index = StatementVectorIndex(dimension=128)
    retriever = PriorArtRetriever(vector_index=index, embedder=embedder)

    mock_llm_refinement = LLMRefinementSchema(
        critique="The flat index direction directly duplicates prior work.",
        revised_direction="Pivoted direction: hierarchical graph pruning for sub-linear vector retrieval",
        revised_technical_approach="HNSW tree pruning algorithm",
        should_reject=False,
    )
    mock_provider = MockLLMProvider(response_obj=mock_llm_refinement)

    validator = PriorArtValidator(llm_provider=mock_provider)
    refinement_engine = IterativeRefinementEngine(
        retriever=retriever,
        validator=validator,
        llm_provider=mock_provider,
        max_refinement_iterations=2,
    )

    candidate = CandidateOpportunity(
        research_problem="Problem",
        identified_gap="Gap",
        proposed_research_direction="Flat index approach",
        reasoning="Reasoning",
        technical_approach="Approach",
        expected_contribution="Contribution",
        feasibility=FeasibilityAssessment(
            required_data="Data",
            required_compute="Compute",
            implementation_complexity="Med",
            experimental_setup="Setup",
        ),
    )

    initial_assessment = PriorArtAssessment(
        assessment="HIGH_OVERLAP",
        overlapping_statement_ids=[uuid.uuid4()],
        overlapping_paper_ids=[uuid.uuid4()],
        reasoning="Direct duplicate of prior work",
        differentiation="None",
        confidence=0.9,
    )

    final_candidate, final_assessment, history, status = refinement_engine.refine_until_differentiated(
        candidate, initial_assessment
    )

    assert status == OpportunityStatus.REFINED
    assert len(history) == 1
    assert "Pivoted direction" in final_candidate.proposed_research_direction
