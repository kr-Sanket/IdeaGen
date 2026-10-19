"""Unit and Integration Tests for Ollama LLM and Embedding Providers."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import uuid
import numpy as np
import pytest
import requests
from pydantic import BaseModel, Field

from ideagen.core.config import Settings, get_settings
from ideagen.llm.ollama_provider import OllamaLLMProvider
from ideagen.llm.gemini_provider import GeminiLLMProvider
from ideagen.llm import get_default_llm_provider
from ideagen.llm.exceptions import ProviderFailureError, MalformedResponseError
from ideagen.discovery.embeddings import (
    OllamaEmbeddingProvider,
    GeminiEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    get_default_embedding_provider,
)
from ideagen.discovery.vector_index import StatementVectorIndex


class SampleResearchHypothesis(BaseModel):
    hypothesis: str = Field(description="Scientific hypothesis statement")
    confidence: float = Field(ge=0.0, le=1.0)
    is_falsifiable: bool


# ── OllamaLLMProvider Unit Tests ─────────────────────────────────────────────

def test_ollama_provider_defaults():
    provider = OllamaLLMProvider()
    assert provider.base_url == "http://localhost:11434"
    assert provider.model_name == "qwen3:8b"
    assert provider.timeout == 300.0
    assert provider.num_ctx == 8192


def test_ollama_provider_custom_config():
    provider = OllamaLLMProvider(
        base_url="http://custom-host:11434/",
        model_name="custom-model",
        timeout=45.0,
        num_ctx=4096,
    )
    assert provider.base_url == "http://custom-host:11434"
    assert provider.model_name == "custom-model"
    assert provider.timeout == 45.0
    assert provider.num_ctx == 4096


@patch("requests.post")
def test_ollama_generate_structured_success(mock_post):
    payload = {
        "message": {
            "content": json.dumps({
                "hypothesis": "Retrieval augmentation reduces hallucination in domain-specific tasks.",
                "confidence": 0.88,
                "is_falsifiable": True,
            })
        }
    }
    mock_post.return_value = MagicMock(
        status_code=200,
        json=MagicMock(return_value=payload),
    )

    provider = OllamaLLMProvider()
    res = provider.generate_structured("Generate a hypothesis", SampleResearchHypothesis)

    assert isinstance(res, SampleResearchHypothesis)
    assert "Retrieval augmentation" in res.hypothesis
    assert res.confidence == 0.88
    assert res.is_falsifiable is True

    # Verify request payload
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args[1]
    assert call_kwargs["json"]["model"] == "qwen3:8b"
    assert call_kwargs["json"]["stream"] is False
    assert call_kwargs["json"]["options"]["temperature"] == 0.0
    assert call_kwargs["json"]["options"]["num_ctx"] == 8192


@patch("requests.post")
def test_ollama_generate_structured_cleans_markdown_fences(mock_post):
    raw_json = json.dumps({
        "hypothesis": "Dense retrieval improves evidence extraction recall.",
        "confidence": 0.95,
        "is_falsifiable": True,
    })
    wrapped_content = f"```json\n{raw_json}\n```"

    mock_post.return_value = MagicMock(
        status_code=200,
        json=MagicMock(return_value={"message": {"content": wrapped_content}}),
    )

    provider = OllamaLLMProvider()
    res = provider.generate_structured("Generate a hypothesis", SampleResearchHypothesis)

    assert isinstance(res, SampleResearchHypothesis)
    assert res.hypothesis == "Dense retrieval improves evidence extraction recall."
    assert res.confidence == 0.95


@patch("requests.post")
def test_ollama_http_error_raises_provider_failure(mock_post):
    mock_post.return_value = MagicMock(
        status_code=500,
        text="Internal Server Error: Model failed to load",
    )

    provider = OllamaLLMProvider()
    with pytest.raises(ProviderFailureError, match="HTTP 500"):
        provider.generate_structured("Generate a hypothesis", SampleResearchHypothesis)


@patch("requests.post")
def test_ollama_timeout_raises_provider_failure(mock_post):
    mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

    provider = OllamaLLMProvider(timeout=5.0)
    with pytest.raises(ProviderFailureError, match="timed out after 5.0s"):
        provider.generate_structured("Generate a hypothesis", SampleResearchHypothesis)


@patch("requests.post")
def test_ollama_connection_error_raises_provider_failure(mock_post):
    mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

    provider = OllamaLLMProvider()
    with pytest.raises(ProviderFailureError, match="Ollama connection error"):
        provider.generate_structured("Generate a hypothesis", SampleResearchHypothesis)


@patch("requests.post")
def test_ollama_empty_response_raises_provider_failure(mock_post):
    mock_post.return_value = MagicMock(
        status_code=200,
        json=MagicMock(return_value={"message": {"content": "   "}}),
    )

    provider = OllamaLLMProvider()
    with pytest.raises(ProviderFailureError, match="empty response"):
        provider.generate_structured("Generate a hypothesis", SampleResearchHypothesis)


@patch("requests.post")
def test_ollama_malformed_json_raises_malformed_response(mock_post):
    mock_post.return_value = MagicMock(
        status_code=200,
        json=MagicMock(return_value={"message": {"content": "not-valid-json { hypothesis:"}}),
    )

    provider = OllamaLLMProvider()
    with pytest.raises(MalformedResponseError):
        provider.generate_structured("Generate a hypothesis", SampleResearchHypothesis)


@patch("requests.post")
def test_ollama_schema_violation_raises_malformed_response(mock_post):
    mock_post.return_value = MagicMock(
        status_code=200,
        json=MagicMock(return_value={"message": {"content": json.dumps({"confidence": 1.5})}}),
    )

    provider = OllamaLLMProvider()
    with pytest.raises(MalformedResponseError, match="failed schema validation"):
        provider.generate_structured("Generate a hypothesis", SampleResearchHypothesis)


# ── OllamaEmbeddingProvider Unit Tests ───────────────────────────────────────

def test_ollama_embedding_defaults():
    provider = OllamaEmbeddingProvider()
    assert provider.dimension == 768
    assert provider.model_name == "nomic-embed-text"


def test_ollama_embedding_empty_texts():
    provider = OllamaEmbeddingProvider()
    res = provider.embed_texts([])
    assert isinstance(res, np.ndarray)
    assert res.shape == (0, 768)


@patch("requests.post")
def test_ollama_embedding_success_via_api_embed(mock_post):
    raw_vec1 = [0.1] * 768
    raw_vec2 = [0.2] * 768
    mock_post.return_value = MagicMock(
        status_code=200,
        json=MagicMock(return_value={"embeddings": [raw_vec1, raw_vec2]}),
    )

    provider = OllamaEmbeddingProvider()
    embeddings = provider.embed_texts(["Statement 1", "Statement 2"])

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 768)
    assert embeddings.dtype == np.float32

    # Verify unit-length normalization
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], rtol=1e-5)


@patch("requests.post")
def test_ollama_embedding_fallback_to_embeddings_endpoint(mock_post):
    # First call to /api/embed fails with 404, fallback calls to /api/embeddings succeed
    def mock_post_side_effect(url, **kwargs):
        if url.endswith("/api/embed"):
            return MagicMock(status_code=404, text="Not Found")
        return MagicMock(
            status_code=200,
            json=MagicMock(return_value={"embedding": [0.5] * 768}),
        )

    mock_post.side_effect = mock_post_side_effect

    provider = OllamaEmbeddingProvider()
    embeddings = provider.embed_texts(["Fallback statement"])

    assert embeddings.shape == (1, 768)
    norm = np.linalg.norm(embeddings[0])
    np.testing.assert_allclose(norm, 1.0, rtol=1e-5)


@patch("requests.post")
def test_ollama_embedding_failure_raises_runtime_error(mock_post):
    mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

    provider = OllamaEmbeddingProvider()
    with pytest.raises(RuntimeError, match="Failed to generate Ollama embeddings"):
        provider.embed_texts(["Failing text"])


@patch("requests.post")
def test_ollama_embedding_dimension_mismatch_raises_value_error(mock_post):
    # Ollama returns 1024-dimension vectors when provider expects 768
    wrong_dim_vec = [0.1] * 1024
    mock_post.return_value = MagicMock(
        status_code=200,
        json=MagicMock(return_value={"embeddings": [wrong_dim_vec]}),
    )

    provider = OllamaEmbeddingProvider(dimension=768)
    with pytest.raises(ValueError, match="Embedding dimension mismatch: expected 768"):
        provider.embed_texts(["Text with wrong returned dim"])


@patch("requests.post")
def test_ollama_embedding_empty_embeddings_raises_runtime_error(mock_post):
    mock_post.return_value = MagicMock(
        status_code=200,
        json=MagicMock(return_value={"embeddings": []}),
    )

    provider = OllamaEmbeddingProvider()
    with pytest.raises(RuntimeError, match="returned no embeddings"):
        provider.embed_texts(["Some text"])


@patch("requests.post")
def test_ollama_embedding_empty_vector_in_list_raises_runtime_error(mock_post):
    mock_post.return_value = MagicMock(
        status_code=200,
        json=MagicMock(return_value={"embeddings": [[]]}),
    )

    provider = OllamaEmbeddingProvider()
    with pytest.raises(RuntimeError, match="empty embedding vector"):
        provider.embed_texts(["Some text"])


def test_statement_vector_index_dimension_safety():
    index = StatementVectorIndex(dimension=768)

    # 1. Dimension mismatch in add_statements
    wrong_dim_vecs = np.zeros((1, 1024), dtype=np.float32)
    fake_statement = MagicMock(id=uuid.uuid4(), paper_id=uuid.uuid4(), statement_type="FINDING", statement_text="test", section=None, page_start=None, page_end=None)
    with pytest.raises(ValueError, match="Vector dimension .* does not match index dimension"):
        index.add_statements([fake_statement], wrong_dim_vecs)

    # 2. Dimension mismatch in rebuild
    with pytest.raises(ValueError, match="Vector dimension .* does not match index dimension"):
        index.rebuild([fake_statement], wrong_dim_vecs)

    # 3. Dimension mismatch in search
    wrong_query_vec = np.zeros((1024,), dtype=np.float32)
    with pytest.raises(ValueError, match="Query vector dimension .* does not match index dimension"):
        index.search(wrong_query_vec)


# ── Centralized Provider Factory Tests ───────────────────────────────────────

def test_default_llm_provider_selection():
    provider = get_default_llm_provider()
    assert isinstance(provider, OllamaLLMProvider)
    assert provider.model_name == "qwen3:8b"


def test_custom_llm_provider_selection_gemini(monkeypatch):
    monkeypatch.setenv("IDEAGEN_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "mock-key")
    provider = get_default_llm_provider()
    assert isinstance(provider, GeminiLLMProvider)


def test_invalid_llm_provider_selection(monkeypatch):
    monkeypatch.setenv("IDEAGEN_LLM_PROVIDER", "unsupported-provider")
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        get_default_llm_provider()


def test_default_embedding_provider_selection():
    provider = get_default_embedding_provider()
    assert isinstance(provider, OllamaEmbeddingProvider)
    assert provider.model_name == "nomic-embed-text"
    assert provider.dimension == 768


def test_custom_embedding_provider_selection_gemini(monkeypatch):
    monkeypatch.setenv("IDEAGEN_EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "mock-key")
    provider = get_default_embedding_provider()
    assert isinstance(provider, GeminiEmbeddingProvider)


def test_custom_embedding_provider_selection_sentence_transformers(monkeypatch):
    monkeypatch.setenv("IDEAGEN_EMBEDDING_PROVIDER", "sentence_transformers")
    # SentenceTransformerEmbeddingProvider lazy-loads model, instantiate directly
    with patch("sentence_transformers.SentenceTransformer") as mock_st:
        mock_st.return_value.get_sentence_embedding_dimension.return_value = 384
        provider = get_default_embedding_provider()
        assert isinstance(provider, SentenceTransformerEmbeddingProvider)


def test_invalid_embedding_provider_selection(monkeypatch):
    monkeypatch.setenv("IDEAGEN_EMBEDDING_PROVIDER", "unsupported-provider")
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        get_default_embedding_provider()


# ── Optional Live Smoke Test (Skipped Offline) ───────────────────────────────

def test_live_ollama_smoke():
    """Optional smoke test against a live local Ollama server if available."""
    settings = get_settings()
    try:
        r = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=1.0)
        if r.status_code != 200:
            pytest.skip("Ollama server not responding with HTTP 200")
    except Exception:
        pytest.skip("Ollama server not reachable locally; skipping live smoke test.")

    provider = OllamaLLMProvider(base_url=settings.ollama_base_url, model_name=settings.ollama_model)
    res = provider.generate_structured("Say hello and return true", SampleResearchHypothesis)
    assert isinstance(res, SampleResearchHypothesis)
