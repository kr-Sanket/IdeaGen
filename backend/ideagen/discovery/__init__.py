"""Semantic Gap Discovery package (Phase 4)."""

from .models import (
    GapStatus,
    GapCategory,
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
    OllamaEmbeddingProvider,
    GeminiEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    get_default_embedding_provider,
)
from .vector_index import StatementVectorIndex, StatementSearchResult
from .grouper import SemanticGrouper, SemanticCluster
from .pattern_analyzer import ResearchPatternAnalyzer
from .validator import GapValidator
from .pipeline import SemanticGapDiscoveryPipeline
from .temporal import normalize_cutoff_date, is_paper_eligible, filter_statements_by_cutoff

__all__ = [
    "GapStatus",
    "GapCategory",
    "ValidationVerdict",
    "EvidenceRole",
    "StatementEvidenceDTO",
    "GapCandidate",
    "GapValidationResult",
    "ResearchGapCreateDTO",
    "ResearchGapResponseDTO",
    "EmbeddingProvider",
    "OllamaEmbeddingProvider",
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
    "normalize_cutoff_date",
    "is_paper_eligible",
    "filter_statements_by_cutoff",
]
