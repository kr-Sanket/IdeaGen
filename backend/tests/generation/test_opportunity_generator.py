"""Tests for Candidate Opportunity Generation Engine (LLM-only, no fallbacks)."""

import pytest
import uuid
from typing import Type, TypeVar
from pydantic import BaseModel, ValidationError

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


def test_generation_prompt_specifies_normalized_feasibility_scale():
    generator = OpportunityGenerator(llm_provider=MockLLMProvider())
    input_dto = ResearchOpportunityInputDTO(
        gap_id=uuid.uuid4(),
        gap_statement="Test gap",
        research_problem="Test problem",
    )
    prompt = generator._build_generation_prompt(input_dto)
    assert "feasibility_score as a normalized float from 0.0 to 1.0" in prompt
    assert "Do NOT use a 1-10 scale or a percentage" in prompt


def test_candidate_opportunity_schema_accepts_valid_feasibility_score():
    candidate = LLMCandidateOpportunitySchema(
        proposed_research_direction="Direction",
        reasoning="Reasoning",
        technical_approach="Approach",
        expected_contribution="Contribution",
        required_data="Data",
        required_compute="Compute",
        implementation_complexity="Complexity",
        experimental_setup="Setup",
        feasibility_score=0.85,
        confidence=0.75,
    )
    assert candidate.feasibility_score == 0.85


def test_candidate_opportunity_schema_rejects_out_of_range_feasibility_score():
    with pytest.raises(ValidationError) as exc_info:
        LLMCandidateOpportunitySchema(
            proposed_research_direction="Direction",
            reasoning="Reasoning",
            technical_approach="Approach",
            expected_contribution="Contribution",
            required_data="Data",
            required_compute="Compute",
            implementation_complexity="Complexity",
            experimental_setup="Setup",
            feasibility_score=8,
            confidence=0.75,
        )
    assert "Input should be less than or equal to 1" in str(exc_info.value)
