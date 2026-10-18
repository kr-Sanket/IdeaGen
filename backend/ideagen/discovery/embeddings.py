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


def get_default_embedding_provider() -> EmbeddingProvider:
    """Returns the configured default embedding provider."""
    settings = get_settings()
    if settings.gemini_api_key:
        return GeminiEmbeddingProvider(api_key=settings.gemini_api_key)
    return SentenceTransformerEmbeddingProvider()
