"""Semantic Gap Discovery Pipeline (Phase 4).

Coordinates the complete discovery pipeline:
Research statements → Scientific embeddings → FAISS index → Semantic grouping →
Research-pattern analysis → Candidate gap detection → AI gap validation →
Evidence integrity verification → PostgreSQL persistence.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Collection, Dict, List, Optional
import uuid

from ideagen.core.config import get_settings
from ideagen.llm import LLMProvider, get_default_llm_provider
from ideagen.llm.exceptions import ProviderFailureError
from .models import (
    GapStatus,
    ResearchGapResponseDTO,
)
from .embeddings import EmbeddingProvider, get_default_embedding_provider
from .vector_index import StatementVectorIndex
from .grouper import SemanticGrouper
from .pattern_analyzer import ResearchPatternAnalyzer
from .validator import GapValidator

logger = logging.getLogger(__name__)


class SemanticGapDiscoveryPipeline:
    """Orchestrates end-to-end semantic research gap discovery and validation."""

    def __init__(
        self,
        persistence: SQLAlchemyPersistenceLayer,
        embedding_provider: Optional[EmbeddingProvider] = None,
        llm_provider: Optional[LLMProvider] = None,
        index_dir: Optional[Path] = None,
        similarity_threshold: float = 0.65,
        min_supporting_evidence: int = 1,
        max_evidence_per_candidate: int = 3,
    ):
        self.persistence = persistence
        settings = get_settings()

        # 1. Embedding Provider
        self.embedding_provider = embedding_provider or get_default_embedding_provider()

        # 2. Vector Index Setup
        self.index_dir = index_dir or (settings.index_dir / "statements")
        self.index_file = self.index_dir / "index.faiss"
        self.meta_file = self.index_dir / "meta.json"
        self.vector_index = StatementVectorIndex(dimension=self.embedding_provider.dimension)

        # 3. Grouper & Pattern Analyzer
        self.grouper = SemanticGrouper(similarity_threshold=similarity_threshold)
        self.pattern_analyzer = ResearchPatternAnalyzer(min_supporting_evidence=min_supporting_evidence)

        # 4. LLM Validator
        val_provider = llm_provider if llm_provider is not None else get_default_llm_provider()
        self.validator = GapValidator(
            provider=val_provider,
            min_valid_evidence=min_supporting_evidence,
            max_evidence_per_candidate=max_evidence_per_candidate,
        )

    def discover_gaps(
        self,
        persist_index: bool = True,
        cutoff_date: Optional[Union[datetime.date, datetime.datetime, str]] = None,
        paper_ids: Optional[Collection[uuid.UUID]] = None,
        project_id: Optional[str] = None,
        gap_metadata: Optional[Dict[str, object]] = None,
    ) -> List[ResearchGapResponseDTO]:
        """Executes the full semantic gap discovery pipeline across eligible research statements.

        ``paper_ids`` restricts discovery to statements and passages from those papers (a
        project's corpus). ``None`` means the whole database; an empty collection discovers nothing.
        ``project_id`` and ``gap_metadata`` are stored with each gap as it is saved, so gaps persisted
        before a later failure still belong to their project and run.
        """
        from .temporal import normalize_cutoff_date, filter_statements_by_cutoff

        norm_cutoff = normalize_cutoff_date(cutoff_date)
        if norm_cutoff is not None:
            logger.info(f"Executing semantic gap discovery with temporal cutoff: {norm_cutoff.isoformat()}")

        # 1. Load statements from database (enforcing cutoff if provided)
        try:
            statements = self.persistence.get_all_research_statements(cutoff_date=norm_cutoff)
        except TypeError:
            statements = self.persistence.get_all_research_statements()

        # In-memory defense-in-depth filtering
        if norm_cutoff is not None:
            statements = filter_statements_by_cutoff(statements, norm_cutoff)

        scoped = paper_ids is not None
        if scoped:
            allowed_paper_ids = set(paper_ids)
            statements = [s for s in statements if s.paper_id in allowed_paper_ids]
            # A scoped discovery must not persist over the shared, whole-corpus FAISS index.
            persist_index = False

        if not statements:
            logger.info("No eligible research statements found in database for discovery. Skipped.")
            return []

        statements_by_id = {s.id: s for s in statements}
        texts = [s.statement_text for s in statements]
        eligible_paper_ids = {s.paper_id for s in statements}

        # Load supporting passages for evidence grounding (strictly bounded to eligible papers)
        passages_by_id = {}
        if hasattr(self.persistence, "get_all_passages"):
            try:
                passages = self.persistence.get_all_passages(cutoff_date=norm_cutoff)
            except TypeError:
                passages = self.persistence.get_all_passages()
            except Exception as e:
                logger.warning(f"Could not load passages from persistence: {e}")
                passages = []

            for p in passages:
                p_paper_id = getattr(p, "paper_id", None) if hasattr(p, "paper_id") else (
                    p.get("paper_id") if isinstance(p, dict) else None
                )
                if (norm_cutoff is None and not scoped) or p_paper_id in eligible_paper_ids:
                    p_id = getattr(p, "id", None) if hasattr(p, "id") else (
                        p.get("id") if isinstance(p, dict) else None
                    )
                    if p_id:
                        passages_by_id[p_id] = p

        # 2. Generate dense scientific embeddings
        logger.info(f"Generating embeddings for {len(statements)} statements with {self.embedding_provider.model_name}")
        vectors = self.embedding_provider.embed_texts(texts)

        # 3. Update / Rebuild FAISS local index
        self.vector_index.rebuild(statements=statements, vectors=vectors)
        if persist_index and norm_cutoff is None:
            self.vector_index.save(index_file=self.index_file, meta_file=self.meta_file)
        elif persist_index and norm_cutoff is not None:
            logger.info("Skipping FAISS index persistence to shared directory for temporal cutoff run.")

        # 4. Construct semantic landscape via grouping
        clusters = self.grouper.group_statements(statements=statements, vectors=vectors)
        logger.info(f"Formed {len(clusters)} semantic clusters across statements")

        # 5. Analyze research patterns and extract candidate gaps
        candidates = self.pattern_analyzer.analyze_clusters(clusters)
        logger.info(f"Detected {len(candidates)} candidate gaps across clusters")
        if not candidates:
            return []

        # 6. Validate candidates and enforce evidence integrity
        persisted_dtos: List[ResearchGapResponseDTO] = []
        for candidate in candidates:
            try:
                gap_dto = self.validator.validate_candidate(
                    candidate=candidate,
                    statements_by_id=statements_by_id,
                    passages_by_id=passages_by_id,
                )
            except ProviderFailureError as e:
                logger.error(
                    f"Gap discovery validation aborted due to provider failure: {e}. "
                    f"Stopping candidate processing."
                )
                raise

            # Record temporal cutoff in gap metadata if present
            if norm_cutoff is not None:
                gap_dto.metadata["temporal_cutoff"] = norm_cutoff.isoformat()
            if gap_metadata:
                gap_dto.metadata.update(gap_metadata)

            # 7. Persist to PostgreSQL
            if project_id is not None:
                gap_id = self.persistence.save_research_gap(gap_dto, project_id=project_id)
            else:
                gap_id = self.persistence.save_research_gap(gap_dto)

            # 8. Add to output
            persisted_dtos.append(
                ResearchGapResponseDTO(
                    id=gap_id,
                    gap_statement=gap_dto.gap_statement,
                    research_problem=gap_dto.research_problem,
                    status=gap_dto.status,
                    evidence_count=gap_dto.evidence_count,
                    supporting_statement_ids=gap_dto.supporting_statement_ids,
                    supporting_paper_ids=gap_dto.supporting_paper_ids,
                    limitations=gap_dto.limitations,
                    future_work=gap_dto.future_work,
                    validation_reasoning=gap_dto.validation_reasoning,
                    validation_confidence=gap_dto.validation_confidence,
                    created_at=gap_dto.metadata.get("created_at") or self.persistence.get_research_gap(gap_id).created_at,
                    metadata=gap_dto.metadata,
                )
            )

        return persisted_dtos

    def get_gaps(self, status: Optional[GapStatus] = None) -> List[ResearchGapResponseDTO]:
        """Retrieves stored research gaps from PostgreSQL."""
        status_filter = status.value if status else None
        db_gaps = self.persistence.get_all_research_gaps(status=status_filter)
        results: List[ResearchGapResponseDTO] = []

        for g in db_gaps:
            evidence_records = self.persistence.get_gap_evidence(g.id)
            stmt_ids = [e.statement_id for e in evidence_records]
            paper_ids = list({e.paper_id for e in evidence_records})

            results.append(
                ResearchGapResponseDTO(
                    id=g.id,
                    gap_statement=g.gap_statement,
                    research_problem=g.research_problem,
                    status=GapStatus(g.status),
                    evidence_count=g.evidence_count,
                    supporting_statement_ids=stmt_ids,
                    supporting_paper_ids=paper_ids,
                    limitations=g.limitations or [],
                    future_work=g.future_work or [],
                    validation_reasoning=g.validation_reasoning,
                    validation_confidence=g.validation_confidence,
                    created_at=g.created_at,
                    metadata=g.metadata_json or {},
                )
            )

        return results
