"""Tests for Prior-Art Retrieval and LLM-based Prior-Art Validation."""

import pytest
import uuid
import numpy as np

from ideagen.discovery.vector_index import StatementVectorIndex, StatementSearchResult
from ideagen.generation.models import CandidateOpportunity, FeasibilityAssessment
from ideagen.generation.prior_art import PriorArtRetriever, PriorArtValidator, LLMPriorArtValidationSchema
from ideagen.llm.exceptions import ProviderFailureError
from tests.generation.test_opportunity_generator import MockLLMProvider


class FakeEmbeddingProvider:
    """Fake embedding provider producing reproducible 128-d float32 vectors."""

    def __init__(self, dimension: int = 128):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return "fake-128d-embedder"

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)
        vecs = []
        for t in texts:
            h = hash(t) % 1000
            v = np.full((self._dimension,), fill_value=float(h) / 1000.0, dtype=np.float32)
            vecs.append(v)
        return np.vstack(vecs)


def test_prior_art_retrieval():
    embedder = FakeEmbeddingProvider(dimension=128)
    index = StatementVectorIndex(dimension=128)

    stmt1_id = uuid.uuid4()
    paper1_id = uuid.uuid4()
    statements = [
        {
            "id": stmt1_id,
            "paper_id": paper1_id,
            "statement_type": "LIMITATION",
            "statement_text": "FAISS flat index suffers from high memory footprint at scale.",
            "section": "Limitations",
            "page_start": 4,
            "page_end": 5,
        }
    ]

    vecs = embedder.embed_texts(["FAISS flat index suffers from high memory footprint at scale."])
    index.add_statements(statements, vecs)

    retriever = PriorArtRetriever(vector_index=index, embedder=embedder)
    results = retriever.retrieve_prior_art("FAISS flat index memory bottleneck", top_k=5)

    assert len(results) == 1
    assert results[0].statement_id == stmt1_id
    assert results[0].paper_id == paper1_id


def test_prior_art_validator_raises_when_llm_provider_is_none():
    validator = PriorArtValidator(llm_provider=None)

    candidate = CandidateOpportunity(
        research_problem="Search latency",
        identified_gap="High vector retrieval time",
        proposed_research_direction="Tree pruning for vector indices",
        reasoning="Reduces distance computations",
        technical_approach="HNSW tree",
        expected_contribution="Sub-linear search",
        feasibility=FeasibilityAssessment(
            required_data="Data",
            required_compute="Compute",
            implementation_complexity="Med",
            experimental_setup="Setup",
        ),
    )

    retrieved_art = [
        StatementSearchResult(
            statement_id=uuid.uuid4(),
            paper_id=uuid.uuid4(),
            statement_type="LIMITATION",
            statement_text="Existing vector indices have high query latency.",
            section="Intro",
            page_start=1,
            page_end=2,
            similarity=0.82,
        )
    ]

    with pytest.raises(ProviderFailureError) as exc_info:
        validator.validate_prior_art(candidate, retrieved_art)
    assert "LLM provider is not configured" in str(exc_info.value)


def test_prior_art_validator_raises_when_llm_fails():
    mock_provider = MockLLMProvider(should_fail=True)
    validator = PriorArtValidator(llm_provider=mock_provider)

    candidate = CandidateOpportunity(
        research_problem="Search latency",
        identified_gap="High vector retrieval time",
        proposed_research_direction="Tree pruning for vector indices",
        reasoning="Reduces distance computations",
        technical_approach="HNSW tree",
        expected_contribution="Sub-linear search",
        feasibility=FeasibilityAssessment(
            required_data="Data",
            required_compute="Compute",
            implementation_complexity="Med",
            experimental_setup="Setup",
        ),
    )

    retrieved_art = [
        StatementSearchResult(
            statement_id=uuid.uuid4(),
            paper_id=uuid.uuid4(),
            statement_type="LIMITATION",
            statement_text="Existing vector indices have high query latency.",
            section="Intro",
            page_start=1,
            page_end=2,
            similarity=0.82,
        )
    ]

    with pytest.raises(ProviderFailureError) as exc_info:
        validator.validate_prior_art(candidate, retrieved_art)
    assert "prior-art validation failed" in str(exc_info.value)


def test_prior_art_validation_with_llm_provider():
    stmt_id = uuid.uuid4()
    paper_id = uuid.uuid4()

    mock_llm_response = LLMPriorArtValidationSchema(
        assessment="HIGH_OVERLAP",
        reasoning="The proposed direction directly matches existing index techniques in retrieved literature.",
        differentiation="Minimal differentiation observed.",
        overlapping_statement_indices=[0],
        confidence=0.90,
    )

    mock_provider = MockLLMProvider(response_obj=mock_llm_response)
    validator = PriorArtValidator(llm_provider=mock_provider)

    candidate = CandidateOpportunity(
        research_problem="Search latency",
        identified_gap="High vector retrieval time",
        proposed_research_direction="Tree pruning for vector indices",
        reasoning="Reduces distance computations",
        technical_approach="HNSW tree",
        expected_contribution="Sub-linear search",
        feasibility=FeasibilityAssessment(
            required_data="Data",
            required_compute="Compute",
            implementation_complexity="Med",
            experimental_setup="Setup",
        ),
    )

    retrieved_art = [
        StatementSearchResult(
            statement_id=stmt_id,
            paper_id=paper_id,
            statement_type="LIMITATION",
            statement_text="Existing vector indices have high query latency.",
            section="Intro",
            page_start=1,
            page_end=2,
            similarity=0.82,
        )
    ]

    assessment = validator.validate_prior_art(candidate, retrieved_art)
    assert assessment.assessment == "HIGH_OVERLAP"
    assert assessment.overlapping_statement_ids == [stmt_id]
    assert assessment.overlapping_paper_ids == [paper_id]
