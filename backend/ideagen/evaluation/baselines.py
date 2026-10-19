from dataclasses import dataclass
import logging
import re
from typing import Any, Collection, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from pydantic import BaseModel

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_distances
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

# Default regex patterns for identifying candidate research limitations
DEFAULT_LIMITATION_PATTERNS: Tuple[str, ...] = (
    r"\blimitations?\b",
    r"\bfuture\s+work\b",
    r"\bremains?\s+(challenging|unresolved|difficult|open|problematic)\b",
    r"\bfails?\s+to\b",
    r"\bunable\s+to\b",
    r"\bsuffers?\s+from\b",
    r"\bdoes\s+not\s+(address|handle|resolve|support|scale)\b",
    r"\black\s+of\b",
    r"\bcannot\s+(handle|scale|generalize|resolve|mitigate)\b",
    r"\bunresolved\b",
    r"\bbottlenecks?\b",
    r"\bopen\s+(problem|question|challenge)s?\b",
    r"\bdrawbacks?\b",
    r"\bshortcomings?\b",
    r"\bvulnerabilit(y|ies)\b",
)

# Negation and qualification patterns that exclude false-positive limitation claims
DEFAULT_EXCLUSION_PATTERNS: Tuple[str, ...] = (
    r"\bno\s+limitations?\b",
    r"\bwithout\s+(any\s+)?limitations?\b",
    r"\bnot\s+a\s+limitation\b",
    r"\bno\s+significant\s+limitation\b",
    r"\bnever\s+fails?\s+to\b",
    r"\blacks?\s+no\b",
)


@dataclass(frozen=True)
class TFIDFBaselineResult:
    """Documented output of TFIDFDiscoveryBaseline.fit_predict().

    Attributes:
        clusters: List of sets of statement IDs for each predicted cluster.
        cluster_labels: Mapping from statement ID to assigned integer cluster index.
        candidate_statements: List of candidate statements that passed lexical filtering.
        filtered_out_statements: List of statements excluded during lexical filtering.
        parameters: Configuration dictionary used during vectorization and clustering.
    """
    clusters: List[Set[str]]
    cluster_labels: Dict[str, int]
    candidate_statements: List[Dict[str, Any]]
    filtered_out_statements: List[Dict[str, Any]]
    parameters: Dict[str, Any]

    def __iter__(self):
        """Allows direct iteration over cluster sets for pairwise metric compatibility."""
        return iter(self.clusters)

    def __len__(self) -> int:
        """Returns the number of predicted clusters."""
        return len(self.clusters)

    def __getitem__(self, index: int) -> Set[str]:
        """Returns the cluster at the specified index."""
        return self.clusters[index]


class TFIDFDiscoveryBaseline:
    """Lexical TF-IDF baseline for research-gap discovery.

    Filters input statements using configurable lexical limitation patterns,
    vectorizes remaining candidates with TF-IDF, and clusters them using
    agglomerative hierarchical clustering with cosine distance.

    This baseline provides a reproducible, non-neural comparative reference
    for evaluating deep-learning and embedding-based research-gap discovery.
    """

    def __init__(
        self,
        similarity_threshold: Optional[float] = None,
        distance_threshold: float = 0.85,
        ngram_range: Tuple[int, int] = (1, 1),
        stop_words: Optional[str] = "english",
        min_df: int = 1,
        limitation_patterns: Optional[Sequence[str]] = None,
        exclusion_patterns: Optional[Sequence[str]] = None,
    ):
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required to use TFIDFDiscoveryBaseline. "
                "Install with optional dependencies: pip install scikit-learn"
            )

        if similarity_threshold is not None:
            # Consistent with IdeaGen SemanticGrouper: distance = 1 - similarity
            self.distance_threshold = max(0.01, 1.0 - float(similarity_threshold))
        else:
            self.distance_threshold = float(distance_threshold)

        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.min_df = min_df

        self.limitation_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in (limitation_patterns if limitation_patterns is not None else DEFAULT_LIMITATION_PATTERNS)
        ]
        self.exclusion_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in (exclusion_patterns if exclusion_patterns is not None else DEFAULT_EXCLUSION_PATTERNS)
        ]

    def is_limitation_candidate(self, text: str) -> bool:
        """Determines whether a statement is a candidate limitation using lexical rules."""
        if not text or not isinstance(text, str):
            return False

        # Check exclusion patterns first (e.g. 'no limitations observed')
        for excl in self.exclusion_patterns:
            if excl.search(text):
                return False

        # Check positive limitation patterns
        for pattern in self.limitation_patterns:
            if pattern.search(text):
                return True

        return False

    def _extract_id_and_text(self, stmt: Any, default_idx: int) -> Tuple[str, str]:
        """Extracts stable statement ID and text from diverse statement representations."""
        default_id = f"stmt_{default_idx}"

        if stmt is None:
            raise ValueError(f"Encountered None statement at index {default_idx}")

        if isinstance(stmt, dict):
            s_id = stmt.get("id") or stmt.get("statement_id") or default_id
            s_text = stmt.get("statement_text") or stmt.get("text") or ""
        elif isinstance(stmt, tuple) and len(stmt) == 2:
            s_id = str(stmt[0])
            s_text = str(stmt[1])
        elif isinstance(stmt, str):
            s_id = default_id
            s_text = stmt
        elif hasattr(stmt, "statement_text"):
            s_id = str(stmt.id) if hasattr(stmt, "id") and stmt.id is not None else default_id
            s_text = str(stmt.statement_text or "")
        elif hasattr(stmt, "text"):
            s_id = str(stmt.id) if hasattr(stmt, "id") and stmt.id is not None else default_id
            s_text = str(stmt.text or "")
        else:
            raise ValueError(
                f"Unsupported statement type at index {default_idx}: {type(stmt).__name__}. "
                f"Expected dict, ResearchStatement, tuple, or string."
            )

        if not isinstance(s_text, str):
            s_text = str(s_text)

        return str(s_id), s_text

    def fit_predict(self, statements: Sequence[Any]) -> TFIDFBaselineResult:
        """Filters limitation candidates and groups them into lexical clusters.

        Args:
            statements: Sequence of statements (ResearchStatement instances,
                dictionaries with 'id' and 'statement_text', tuples, or strings).

        Returns:
            TFIDFBaselineResult containing predicted cluster sets, cluster labels mapping,
            and candidate/filtered breakdown.
        """
        if statements is None or not isinstance(statements, (list, tuple, Sequence)):
            raise ValueError("Input 'statements' must be a non-None sequence.")

        candidate_records: List[Dict[str, Any]] = []
        filtered_out_records: List[Dict[str, Any]] = []

        # 1. Deterministic extraction and lexical filtering
        for idx, s in enumerate(statements):
            s_id, s_text = self._extract_id_and_text(s, idx)
            record = {"id": s_id, "text": s_text, "source_index": idx}

            if self.is_limitation_candidate(s_text):
                candidate_records.append(record)
            else:
                filtered_out_records.append(record)

        params = {
            "distance_threshold": self.distance_threshold,
            "ngram_range": self.ngram_range,
            "stop_words": self.stop_words,
            "total_input_statements": len(statements),
            "candidate_count": len(candidate_records),
            "filtered_out_count": len(filtered_out_records),
        }

        # 2. Handle zero candidates
        if len(candidate_records) == 0:
            return TFIDFBaselineResult(
                clusters=[],
                cluster_labels={},
                candidate_statements=candidate_records,
                filtered_out_statements=filtered_out_records,
                parameters=params,
            )

        # 3. Handle single candidate (deterministic singleton)
        if len(candidate_records) == 1:
            single_id = candidate_records[0]["id"]
            return TFIDFBaselineResult(
                clusters=[{single_id}],
                cluster_labels={single_id: 0},
                candidate_statements=candidate_records,
                filtered_out_statements=filtered_out_records,
                parameters=params,
            )

        # 4. TF-IDF Vectorization for >= 2 candidates
        candidate_texts = [c["text"] for c in candidate_records]

        try:
            vectorizer = TfidfVectorizer(
                ngram_range=self.ngram_range,
                stop_words=self.stop_words,
                min_df=self.min_df,
                norm="l2",
            )
            X = vectorizer.fit_transform(candidate_texts).toarray()
        except ValueError as e:
            # Handle empty vocabulary (e.g. all words are English stop words or pure punctuation)
            logger.warning(f"TfidfVectorizer failed with empty vocabulary ({e}); falling back without stop words.")
            try:
                fallback_vec = TfidfVectorizer(
                    ngram_range=self.ngram_range,
                    stop_words=None,
                    token_pattern=r"(?u)\S+",
                    norm="l2",
                )
                X = fallback_vec.fit_transform(candidate_texts).toarray()
            except Exception:
                # If even fallback tokenization fails, treat each candidate as an individual singleton cluster
                clusters = [{c["id"]} for c in candidate_records]
                labels = {c["id"]: idx for idx, c in enumerate(candidate_records)}
                return TFIDFBaselineResult(
                    clusters=clusters,
                    cluster_labels=labels,
                    candidate_statements=candidate_records,
                    filtered_out_statements=filtered_out_records,
                    parameters=params,
                )

        # Check for zero-norm rows (e.g. empty candidate texts)
        row_norms = np.linalg.norm(X, axis=1, keepdims=True)
        zero_rows = (row_norms == 0).flatten()
        if zero_rows.all():
            # All texts are completely zero-norm
            clusters = [{c["id"]} for c in candidate_records]
            labels = {c["id"]: idx for idx, c in enumerate(candidate_records)}
            return TFIDFBaselineResult(
                clusters=clusters,
                cluster_labels=labels,
                candidate_statements=candidate_records,
                filtered_out_statements=filtered_out_records,
                parameters=params,
            )

        # 5. Agglomerative Clustering using precomputed cosine distances
        dist_matrix = cosine_distances(X)
        clustering = AgglomerativeClustering(
            metric="precomputed",
            linkage="average",
            distance_threshold=self.distance_threshold,
            n_clusters=None,
        )
        raw_labels = clustering.fit_predict(dist_matrix)

        # 6. Deterministic Cluster Mapping & Re-indexing
        # Re-index clusters by first appearance in candidate_records
        clusters_by_label: Dict[int, Set[str]] = {}
        for c_idx, raw_lbl in enumerate(raw_labels):
            stmt_id = candidate_records[c_idx]["id"]
            clusters_by_label.setdefault(int(raw_lbl), set()).add(stmt_id)

        # Sort clusters deterministically by the earliest source index
        id_to_src_idx = {c["id"]: c["source_index"] for c in candidate_records}
        sorted_cluster_groups = sorted(
            clusters_by_label.values(),
            key=lambda c_set: min(id_to_src_idx[sid] for sid in c_set),
        )

        final_clusters: List[Set[str]] = []
        final_labels: Dict[str, int] = {}

        for new_idx, cluster_set in enumerate(sorted_cluster_groups):
            final_clusters.append(cluster_set)
            for sid in cluster_set:
                final_labels[sid] = new_idx

        return TFIDFBaselineResult(
            clusters=final_clusters,
            cluster_labels=final_labels,
            candidate_statements=candidate_records,
            filtered_out_statements=filtered_out_records,
            parameters=params,
        )


class BaselineDefinition(BaseModel):
    name: str
    target_component: str
    description: str
    metrics: List[str]
    is_implemented: bool


def get_baselines() -> List[BaselineDefinition]:
    return [
        BaselineDefinition(
            name="TF-IDF Discovery",
            target_component="OpportunityDiscoveryService",
            description="Lexical similarity clustering without semantic research-gap normalization.",
            metrics=["pairwise_precision", "pairwise_recall"],
            is_implemented=True,
        ),
        BaselineDefinition(
            name="BM25 Retrieval",
            target_component="GroundedOpportunityService",
            description="Standard BM25 lexical retrieval against entire document text.",
            metrics=["top_k_recall"],
            is_implemented=False,
        ),
        BaselineDefinition(
            name="Static Opportunity",
            target_component="OpportunityEvolutionService",
            description="Assessing the gap once at the end of the timeline without tracking intermediate states.",
            metrics=["temporal_state_accuracy"],
            is_implemented=False,
        ),
        BaselineDefinition(
            name="Unconstrained Generation",
            target_component="ExplanationGenerator",
            description="Explanation generation without explicit evidence-reference validation.",
            metrics=["citation_validity_rate"],
            is_implemented=False,
        ),
    ]

