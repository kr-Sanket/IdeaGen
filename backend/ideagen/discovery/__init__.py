"""Semantic Gap Discovery package (Phase 4)."""

from .models import (
    GapStatus,
    ValidationVerdict,
    EvidenceRole,
    StatementEvidenceDTO,
    GapCandidate,
    GapValidationResult,
    ResearchGapCreateDTO,
    ResearchGapResponseDTO,
)
from .embeddings import (
    EmbeddingProvider,
    GeminiEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    get_default_embedding_provider,
)
from .vector_index import StatementVectorIndex, StatementSearchResult
from .grouper import SemanticGrouper, SemanticCluster
from .pattern_analyzer import ResearchPatternAnalyzer
from .validator import GapValidator
from .pipeline import SemanticGapDiscoveryPipeline

__all__ = [
    "GapStatus",
    "ValidationVerdict",
    "EvidenceRole",
    "StatementEvidenceDTO",
    "GapCandidate",
    "GapValidationResult",
    "ResearchGapCreateDTO",
    "ResearchGapResponseDTO",
    "EmbeddingProvider",
    "GeminiEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "get_default_embedding_provider",
    "StatementVectorIndex",
    "StatementSearchResult",
    "SemanticGrouper",
    "SemanticCluster",
    "ResearchPatternAnalyzer",
    "GapValidator",
    "SemanticGapDiscoveryPipeline",
]
