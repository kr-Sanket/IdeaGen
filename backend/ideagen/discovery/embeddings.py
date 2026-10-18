"""Pretrained scientific and text embedding providers (Phase 4).

Generates real, reproducible dense vector representations of research statements
supporting Gemini GenAI embeddings and SentenceTransformers.
"""

from __future__ import annotations

import logging
from typing import List, Protocol, runtime_checkable
import numpy as np

from ideagen.core.config import get_settings

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for dense vector embedding generators."""

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generates L2-normalized float32 embeddings for a batch of strings."""
        ...

    @property
    def dimension(self) -> int:
        """Returns the dimensionality of the generated vectors."""
        ...

    @property
    def model_name(self) -> str:
        """Returns the model identifier for provenance and reproducibility."""
        ...


class GeminiEmbeddingProvider:
    """Generates embeddings using Google GenAI pretrained embedding models."""

    def __init__(self, api_key: str, model_name: str = "models/gemini-embedding-001"):
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name
        self._dimension = 3072

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embeds a list of texts and returns an (N, D) float32 numpy array."""
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        vectors: List[List[float]] = []
        # Batch requests to respect API thresholds
        batch_size = 20
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            for text in chunk:
                # Handle empty strings safely
                safe_text = text.strip() if text and text.strip() else "scientific research statement"
                resp = self._client.models.embed_content(
                    model=self._model_name,
                    contents=safe_text,
                )
                if not resp.embeddings or not resp.embeddings[0].values:
                    raise RuntimeError(f"Embedding API returned empty vector for text: {safe_text[:30]}")
                vectors.append(resp.embeddings[0].values)

        arr = np.array(vectors, dtype=np.float32)
        # Normalize to unit length for cosine similarity via inner product
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


class SentenceTransformerEmbeddingProvider:
    """Generates embeddings locally using SentenceTransformers pretrained models."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embeds texts using SentenceTransformer with L2 normalization."""
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        clean_texts = [t.strip() if t and t.strip() else "statement" for t in texts]
        embeddings = self._model.encode(
            clean_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)


class OllamaEmbeddingProvider:
    """Generates embeddings locally using Ollama embedding models."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "nomic-embed-text",
        dimension: int = 768,
        timeout: float = 60.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._dimension = dimension
        self._timeout = timeout

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embeds a list of texts and returns an (N, D) float32 numpy array with L2 normalization."""
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        import requests

        clean_texts = [t.strip() if t and t.strip() else "statement" for t in texts]

        embed_url = f"{self._base_url}/api/embed"
        payload = {
            "model": self._model_name,
            "input": clean_texts,
        }

        try:
            resp = requests.post(embed_url, json=payload, timeout=self._timeout)
            if resp.status_code == 200:
                data = resp.json()
                vectors = data.get("embeddings", [])
            else:
                # Fallback to per-item /api/embeddings endpoint if /api/embed is unavailable
                vectors = []
                for text in clean_texts:
                    r = requests.post(
                        f"{self._base_url}/api/embeddings",
                        json={"model": self._model_name, "prompt": text},
                        timeout=self._timeout,
                    )
                    if r.status_code != 200:
                        raise RuntimeError(f"Ollama embedding API error ({r.status_code}): {r.text}")
                    vectors.append(r.json()["embedding"])
        except Exception as e:
            raise RuntimeError(f"Failed to generate Ollama embeddings: {e}") from e

        if not vectors or len(vectors) == 0:
            raise RuntimeError("Ollama embedding API returned no embeddings.")

        if any(len(v) == 0 for v in vectors):
            raise RuntimeError("Ollama embedding API returned an empty embedding vector.")

        arr = np.array(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            raise RuntimeError("Ollama embedding API returned an invalid embedding array.")

        if arr.shape[1] != self._dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._dimension} (configured for {self._model_name}), "
                f"but Ollama returned vectors of dimension {arr.shape[1]}."
            )

        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


def get_default_embedding_provider() -> EmbeddingProvider:
    """Returns the configured default embedding provider."""
    settings = get_settings()
    provider_name = (settings.embedding_provider or "ollama").lower()
    if provider_name == "ollama":
        return OllamaEmbeddingProvider(
            base_url=settings.ollama_base_url,
            model_name=settings.ollama_embedding_model,
            dimension=settings.ollama_embedding_dimension,
            timeout=settings.ollama_timeout,
        )
    elif provider_name == "gemini":
        return GeminiEmbeddingProvider(api_key=settings.gemini_api_key or "")
    elif provider_name in ("sentence_transformers", "sentencetransformer", "local"):
        return SentenceTransformerEmbeddingProvider()
    raise ValueError(f"Unsupported embedding provider: {provider_name}")
