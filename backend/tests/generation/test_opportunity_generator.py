"""Tests for Candidate Opportunity Generation Engine (LLM-only, no fallbacks)."""

import pytest
import uuid
from typing import Type, TypeVar
from pydantic import BaseModel

from ideagen.generation.models import ResearchOpportunityInputDTO
from ideagen.generation.generator import OpportunityGenerator, LLMCandidateOpportunitySchema
from ideagen.llm.provider import LLMProvider
from ideagen.llm.exceptions import ProviderFailureError

T = TypeVar("T")


class MockLLMProvider:
    """Mock LLMProvider implementing generate_structured for deterministic tests."""

    def __init__(self, response_obj: BaseModel = None, should_fail: bool = False):
        self.response_obj = response_obj
        self.should_fail = should_fail

    def generate_structured(self, prompt: str, response_schema: Type[T]) -> T:
        if self.should_fail:
            raise RuntimeError("API key invalid or rate limit exceeded")
        return self.response_obj


def test_generator_raises_when_llm_provider_is_none():
    generator = OpportunityGenerator(llm_provider=None)
    gap_id = uuid.uuid4()
    input_dto = ResearchOpportunityInputDTO(
        gap_id=gap_id,
        gap_statement="High latency in RAG retrieval",
        research_problem="Retrieval latency limits search",
    )

    with pytest.raises(ProviderFailureError) as exc_info:
        generator.generate_candidate(input_dto)
    assert "LLM provider is not configured" in str(exc_info.value)


def test_generator_raises_when_llm_provider_fails():
    mock_provider = MockLLMProvider(should_fail=True)
    generator = OpportunityGenerator(llm_provider=mock_provider)

    gap_id = uuid.uuid4()
    input_dto = ResearchOpportunityInputDTO(
        gap_id=gap_id,
        gap_statement="High latency in RAG retrieval",
        research_problem="Retrieval latency limits search",
    )

    with pytest.raises(ProviderFailureError) as exc_info:
        generator.generate_candidate(input_dto)
    assert "candidate opportunity generation failed" in str(exc_info.value)


def test_llm_backed_generator_success():
    mock_llm_response = LLMCandidateOpportunitySchema(
        proposed_research_direction="Hierarchical tree-structured indexing for sub-linear RAG retrieval",
        reasoning="Tree indices prune search space to reduce vector comparison latency.",
        technical_approach="Build an HNSW index with dynamic leaf clustering.",
        expected_contribution="Reduces average retrieval latency by 40% on 10M document corpora.",
        required_data="MS MARCO and OpenAlex benchmark text collections",
        required_compute="1x NVIDIA RTX 4090 GPU",
        implementation_complexity="Moderate algorithm complexity",
        experimental_setup="Compare retrieval time and recall@10 against flat FAISS baseline",
        feasibility_score=0.9,
        confidence=0.85,
    )

    mock_provider = MockLLMProvider(response_obj=mock_llm_response)
    generator = OpportunityGenerator(llm_provider=mock_provider)

    gap_id = uuid.uuid4()
    input_dto = ResearchOpportunityInputDTO(
        gap_id=gap_id,
        gap_statement="High retrieval latency",
        research_problem="Slow search response",
    )

    candidate = generator.generate_candidate(input_dto)
    assert candidate.proposed_research_direction == mock_llm_response.proposed_research_direction
    assert candidate.feasibility.feasibility_score == 0.9
    assert candidate.confidence == 0.85
