"""Semantic grouping and landscape clustering for research statements (Phase 4).

Constructs semantic clusters of related research statements using hierarchical
cosine distance clustering to form the research landscape.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, NamedTuple, Set
import uuid

import numpy as np
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger(__name__)


class SemanticCluster(NamedTuple):
    """A semantic cluster representing a coherent research topic or problem space."""
    cluster_id: int
    statements: List[Any]
    paper_ids: Set[uuid.UUID]
    centroid: np.ndarray
    statement_type_counts: Dict[str, int]


class SemanticGrouper:
    """Groups dense statement embeddings into semantic research topics."""

    def __init__(self, similarity_threshold: float = 0.65, min_cluster_size: int = 2):
        self.similarity_threshold = similarity_threshold
        # Distance threshold for cosine metric (distance = 1 - similarity)
        self.distance_threshold = max(0.01, 1.0 - similarity_threshold)
        self.min_cluster_size = min_cluster_size

    def group_statements(
        self,
        statements: List[Any],
        vectors: np.ndarray,
    ) -> List[SemanticCluster]:
        """Clusters research statements based on semantic similarity."""
        if len(statements) == 0:
            return []

        if len(statements) == 1:
            # Single statement form its own cluster if allowed, or return 1 cluster
            p_id = statements[0].paper_id if hasattr(statements[0], "paper_id") else statements[0]["paper_id"]
            if isinstance(p_id, str):
                p_id = uuid.UUID(p_id)
            st_type = statements[0].statement_type if hasattr(statements[0], "statement_type") else statements[0]["statement_type"]
            return [
                SemanticCluster(
                    cluster_id=0,
                    statements=statements,
                    paper_ids={p_id},
                    centroid=vectors[0],
                    statement_type_counts={str(st_type): 1},
                )
            ]

        # Ensure vectors are float64 for scikit-learn precision
        v_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        clustering = AgglomerativeClustering(
            metric="cosine",
            linkage="average",
            distance_threshold=self.distance_threshold,
            n_clusters=None,
        )

        labels = clustering.fit_predict(v_norm)

        # Group statements by label
        clusters_by_label: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            clusters_by_label.setdefault(int(label), []).append(idx)

        semantic_clusters: List[SemanticCluster] = []
        cluster_counter = 0

        for label, indices in clusters_by_label.items():
            if len(indices) < self.min_cluster_size and len(clusters_by_label) > 1:
                # Discard singleton noise if we have multiple clusters
                continue

            cluster_stmts = [statements[i] for i in indices]
            cluster_vecs = v_norm[indices]
            centroid = np.mean(cluster_vecs, axis=0)
            centroid = centroid / np.linalg.norm(centroid)

            paper_ids: Set[uuid.UUID] = set()
            type_counts: Dict[str, int] = {}

            for s in cluster_stmts:
                p_id = s.paper_id if hasattr(s, "paper_id") else s["paper_id"]
                if isinstance(p_id, str):
                    p_id = uuid.UUID(p_id)
                paper_ids.add(p_id)

                st_type = str(s.statement_type if hasattr(s, "statement_type") else s["statement_type"])
                type_counts[st_type] = type_counts.get(st_type, 0) + 1

            semantic_clusters.append(
                SemanticCluster(
                    cluster_id=cluster_counter,
                    statements=cluster_stmts,
                    paper_ids=paper_ids,
                    centroid=centroid.astype(np.float32),
                    statement_type_counts=type_counts,
                )
            )
            cluster_counter += 1

        return semantic_clusters
