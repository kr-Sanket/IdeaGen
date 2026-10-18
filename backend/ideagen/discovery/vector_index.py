"""FAISS local vector index for research statements (Phase 4).

Provides deterministic vector storage, metadata mapping, persistence to disk,
rebuilding capabilities, and nearest-neighbor semantic search.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional
import uuid

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class StatementSearchResult(NamedTuple):
    statement_id: uuid.UUID
    paper_id: uuid.UUID
    statement_type: str
    statement_text: str
    section: Optional[str]
    page_start: Optional[int]
    page_end: Optional[int]
    similarity: float


class StatementVectorIndex:
    """Local FAISS vector index maintaining 1-to-1 mappings with research statements."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.id_list: List[uuid.UUID] = []
        self.metadata_store: Dict[str, Dict[str, Any]] = {}

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal

    def add_statements(
        self,
        statements: List[Any],
        vectors: np.ndarray,
    ) -> None:
        """Adds statements and their corresponding dense vectors to the FAISS index."""
        if len(statements) == 0:
            return

        if vectors.shape[0] != len(statements):
            raise ValueError(f"Vectors count ({vectors.shape[0]}) does not match statements ({len(statements)})")

        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension ({vectors.shape[1]}) does not match index dimension ({self.dimension})")

        # Ensure float32 and normalized
        v_float32 = vectors.astype(np.float32)
        faiss.normalize_L2(v_float32)

        self.index.add(v_float32)

        for stmt in statements:
            s_id = stmt.id if hasattr(stmt, "id") else stmt["id"]
            if isinstance(s_id, str):
                s_id = uuid.UUID(s_id)
            self.id_list.append(s_id)

            paper_id = stmt.paper_id if hasattr(stmt, "paper_id") else stmt["paper_id"]
            stmt_type = stmt.statement_type if hasattr(stmt, "statement_type") else stmt["statement_type"]
            stmt_text = stmt.statement_text if hasattr(stmt, "statement_text") else stmt["statement_text"]
            section = stmt.section if hasattr(stmt, "section") else stmt.get("section")
            page_start = stmt.page_start if hasattr(stmt, "page_start") else stmt.get("page_start")
            page_end = stmt.page_end if hasattr(stmt, "page_end") else stmt.get("page_end")

            self.metadata_store[str(s_id)] = {
                "paper_id": str(paper_id),
                "statement_type": str(stmt_type),
                "statement_text": str(stmt_text),
                "section": section,
                "page_start": page_start,
                "page_end": page_end,
            }

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> List[StatementSearchResult]:
        # Ensure vector shape (1, D)
        q = query_vector.astype(np.float32)
        if len(q.shape) == 1:
            q = np.expand_dims(q, axis=0)

        if q.shape[1] != self.dimension:
            raise ValueError(f"Query vector dimension ({q.shape[1]}) does not match index dimension ({self.dimension})")

        if self.total_vectors == 0:
            return []

        faiss.normalize_L2(q)
        k = min(top_k, self.total_vectors)

        scores, indices = self.index.search(q, k)

        results: List[StatementSearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < min_similarity:
                continue

            s_id = self.id_list[idx]
            meta = self.metadata_store.get(str(s_id), {})

            results.append(
                StatementSearchResult(
                    statement_id=s_id,
                    paper_id=uuid.UUID(meta["paper_id"]),
                    statement_type=meta.get("statement_type", "UNKNOWN"),
                    statement_text=meta.get("statement_text", ""),
                    section=meta.get("section"),
                    page_start=meta.get("page_start"),
                    page_end=meta.get("page_end"),
                    similarity=float(score),
                )
            )

        return results

    def save(self, index_file: Path, meta_file: Path) -> None:
        """Persists the FAISS index and statement metadata mapping to disk."""
        index_file.parent.mkdir(parents=True, exist_ok=True)
        meta_file.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_file))

        data = {
            "dimension": self.dimension,
            "id_list": [str(i) for i in self.id_list],
            "metadata_store": self.metadata_store,
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, index_file: Path, meta_file: Path) -> StatementVectorIndex:
        """Loads a persistent FAISS index and metadata mapping from disk."""
        if not index_file.exists() or not meta_file.exists():
            raise FileNotFoundError(f"Index or metadata file missing: {index_file}, {meta_file}")

        with open(meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        dimension = data["dimension"]
        instance = cls(dimension=dimension)
        instance.index = faiss.read_index(str(index_file))
        instance.id_list = [uuid.UUID(i) for i in data["id_list"]]
        instance.metadata_store = data["metadata_store"]
        return instance

    def rebuild(self, statements: List[Any], vectors: np.ndarray) -> None:
        """Rebuilds the entire index from scratch with the provided data."""
        if len(statements) > 0 and vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension ({vectors.shape[1]}) does not match index dimension ({self.dimension})")
        self.index = faiss.IndexFlatIP(self.dimension)
        self.id_list = []
        self.metadata_store = {}
        self.add_statements(statements, vectors)
