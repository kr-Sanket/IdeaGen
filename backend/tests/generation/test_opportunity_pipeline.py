"""Tests for ResearchOpportunityPipeline (Phase 5 End-to-End)."""

import pytest
import uuid
import numpy as np

from ideagen.discovery.vector_index import StatementVectorIndex
from ideagen.generation.models import ResearchOpportunityInputDTO, OpportunityStatus
from ideagen.generation.generator import LLMCandidateOpportunitySchema
from ideagen.generation.prior_art import LLMPriorArtValidationSchema
from ideagen.generation.pipeline import ResearchOpportunityPipeline
from ideagen.llm.exceptions import ProviderFailureError
from tests.generation.test_opportunity_generator import MockLLMProvider
from tests.generation.test_prior_art import FakeEmbeddingProvider


class MockPipelineLLMProvider:
    """Mock LLMProvider that dispatches responses based on requested schema."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.gen_schema = LLMCandidateOpportunitySchema(
            proposed_research_direction="Step-wise verification for multi-hop RAG",
            reasoning="Verifies reasoning steps to reduce hallucination accumulation.",
            technical_approach="Chain-of-thought verification with sub-query constraints",
            expected_contribution="Reduces hallucination rate by 35%",
            required_data="HotpotQA and OpenAlex corpus",
            required_compute="1x GPU",
            implementation_complexity="Moderate",
            experimental_setup="Evaluation against standard RAG baseline",
            feasibility_score=0.85,
            confidence=0.80,
        )
        self.val_schema = LLMPriorArtValidationSchema(
            assessment="LOW_OVERLAP",
            reasoning="Distinct from prior step-wise verification techniques.",
            differentiation="Focuses on multi-hop graph constraints.",
            overlapping_statement_indices=[],
            confidence=0.90,
        )

    def generate_structured(self, prompt: str, response_schema):
        if self.should_fail:
            raise RuntimeError("LLM API failure")
        if response_schema == LLMCandidateOpportunitySchema:
            return self.gen_schema
        if response_schema == LLMPriorArtValidationSchema:
            return self.val_schema
        return self.gen_schema


def test_full_pipeline_execution_with_llm():
    embedder = FakeEmbeddingProvider(dimension=128)
    index = StatementVectorIndex(dimension=128)
    mock_provider = MockPipelineLLMProvider()

    pipeline = ResearchOpportunityPipeline(
        vector_index=index,
        embedder=embedder,
        llm_provider=mock_provider,
        max_refinement_iterations=2,
    )

    gap_id = uuid.uuid4()
    s_id = uuid.uuid4()
    p_id = uuid.uuid4()

    input_dto = ResearchOpportunityInputDTO(
        gap_id=gap_id,
        gap_statement="Unquantified hallucination rates in multi-hop RAG systems",
        research_problem="Multi-hop retrieval accumulates errors across reasoning steps",
        supporting_statement_ids=[s_id],
        supporting_paper_ids=[p_id],
        recurring_limitations=["No step-wise verification mechanism"],
        existing_methods=["Standard RAG pipeline"],
    )

    result = pipeline.process_validated_gap(input_dto=input_dto)

    assert result.gap_id == gap_id
    assert result.status == OpportunityStatus.VALIDATED
    assert result.supporting_statement_ids == [s_id]
    assert result.supporting_paper_ids == [p_id]
    assert result.generation_model == "gemini-2.5-flash"
    assert result.embedding_model == "fake-128d-embedder"
    assert "Potentially novel" in result.novelty_assessment.assessment


def test_pipeline_raises_on_llm_failure():
    embedder = FakeEmbeddingProvider(dimension=128)
    index = StatementVectorIndex(dimension=128)

    pipeline = ResearchOpportunityPipeline(
        vector_index=index,
        embedder=embedder,
        llm_provider=None,  # No LLM provider -> explicitly fails
        max_refinement_iterations=2,
    )

    gap_id = uuid.uuid4()
    input_dto = ResearchOpportunityInputDTO(
        gap_id=gap_id,
        gap_statement="Unquantified hallucination rates",
        research_problem="Multi-hop retrieval errors",
    )

    with pytest.raises(ProviderFailureError) as exc_info:
        pipeline.process_validated_gap(input_dto=input_dto)
    assert "LLM provider is not configured" in str(exc_info.value)
