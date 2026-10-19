"""Controlled Multi-Paper Experiment Runner for IdeaGen F.

Executes a reproducible, isolated end-to-end experiment across a corpus of scientific PDFs:
1. Ingestion & passage chunking with PyMuPDF and SHA-256 provenance.
2. Structured scientific understanding & statement extraction via Ollama (qwen3:8b).
3. Semantic statement clustering & gap validation via Ollama & nomic-embed-text (768-d).
4. Research opportunity synthesis with genuine prior-art FAISS retrieval.
5. Strict database isolation via persistent/in-memory SQLite.
6. Comprehensive metrics collection, configuration fingerprinting, checkpointing, and artifact export.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Set
import uuid

# Ensure backend/ is in pythonpath
_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_DIR = _REPO_ROOT / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from sqlalchemy import create_engine, select, String, Text
from sqlalchemy.orm import sessionmaker, Session, Mapped, mapped_column
from sqlalchemy.pool import StaticPool

from ideagen.core.config import get_settings
from ideagen.db.models import (
    Base,
    Paper,
    PaperDocument,
    PaperPassage,
    ResearchKnowledge,
    ResearchStatement,
    ResearchGap,
    GapEvidence,
    ResearchOpportunityRecord,
    PriorArtAssessmentRecord,
    OpportunityRefinementRecord,
    Opportunity,
    OpportunityEvidence,
    OpportunityHistory,
)
from ideagen.db.persistence import SQLAlchemyPersistenceLayer
from ideagen.ingestion.pdf import ManualPDFIngestionService, PDFExtractionError
from ideagen.understanding.pipeline import PaperUnderstandingPipeline
from ideagen.understanding.passage_selector import PassageSelector
from ideagen.discovery.pipeline import SemanticGapDiscoveryPipeline
from ideagen.discovery.vector_index import StatementVectorIndex
from ideagen.discovery.embeddings import EmbeddingProvider, get_default_embedding_provider
from ideagen.discovery.models import GapStatus
from ideagen.discovery.temporal import normalize_cutoff_date
from ideagen.llm.provider import LLMProvider
from ideagen.llm import get_default_llm_provider
from ideagen.generation.pipeline import ResearchOpportunityPipeline
from ideagen.generation.models import ResearchOpportunityInputDTO
from ideagen.reporting.report_service import OpportunityReportService
from ideagen.opportunity.models import OpportunityRelationship, OpportunityStatus
from ideagen.opportunity.classifier import RelationshipClassifier
from ideagen.opportunity.service import OpportunityTransitionService
from ideagen.opportunity.integration import OpportunityClassificationIntegration
from ideagen.opportunity.temporal import (
    TemporalBatch,
    OpportunityEvolutionService,
    OpportunityEvolutionResult,
)
from ideagen.opportunity.bridge import OpportunityBridgeService
from ideagen.db.session import configure_sqlite_foreign_keys
from ideagen.ingestion.preflight import run_corpus_preflight, CorpusPreflightReport

logger = logging.getLogger(__name__)


class ExperimentMetadataRecord(Base):
    """Stores key-value experiment metadata such as configuration fingerprints."""
    __tablename__ = "experiment_metadata"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[str] = mapped_column(Text)


def compute_configuration_fingerprint(
    llm_provider: Any,
    llm_model: str,
    embedding_provider: Any,
    embedding_model: str,
    embedding_dimension: int,
    max_passages: int,
    max_total_words: int,
    extraction_schema_version: str,
    similarity_threshold: float,
    min_supporting_evidence: int,
    ollama_timeout: float = 300.0,
    max_refinement_iterations: int = 2,
    temporal_cutoff: Optional[str] = None,
    enable_stage_5: bool = False,
    stage_5_batch_window_days: Optional[int] = None,
    stage_5_cutoffs: Optional[List[str]] = None,
    stage_5_chunk_size: Optional[int] = None,
    stage_5_max_retries: Optional[int] = None,
) -> Dict[str, Any]:
    """Generates a canonical configuration dictionary of all material run settings."""
    return {
        "llm_provider": llm_provider.__class__.__name__ if llm_provider else "None",
        "llm_model": str(llm_model),
        "embedding_provider": embedding_provider.__class__.__name__ if embedding_provider else "None",
        "embedding_model": str(embedding_model),
        "embedding_dimension": int(embedding_dimension),
        "ollama_timeout": float(ollama_timeout),
        "passage_selection": {
            "max_passages": int(max_passages),
            "max_total_words": int(max_total_words),
        },
        "extraction_schema_version": str(extraction_schema_version),
        "gap_discovery": {
            "similarity_threshold": float(similarity_threshold),
            "min_supporting_evidence": int(min_supporting_evidence),
        },
        "max_refinement_iterations": int(max_refinement_iterations),
        "temporal_cutoff": str(temporal_cutoff) if temporal_cutoff else None,
        "enable_stage_5": bool(enable_stage_5),
        "stage_5_batch_window_days": stage_5_batch_window_days,
        "stage_5_cutoffs": [str(c) for c in stage_5_cutoffs] if stage_5_cutoffs else None,
        "stage_5_chunk_size": stage_5_chunk_size,
        "stage_5_max_retries": stage_5_max_retries,
    }


def compare_fingerprints(stored: Dict[str, Any], current: Dict[str, Any]) -> List[str]:
    """Returns a list of human-readable mismatch descriptions if material settings differ."""
    mismatches: List[str] = []

    # 1. LLM
    if stored.get("llm_model") != current.get("llm_model"):
        mismatches.append(
            f"LLM model mismatch: stored='{stored.get('llm_model')}', current='{current.get('llm_model')}'"
        )
    if stored.get("llm_provider") != current.get("llm_provider"):
        mismatches.append(
            f"LLM provider mismatch: stored='{stored.get('llm_provider')}', current='{current.get('llm_provider')}'"
        )

    # 2. Embedding
    if stored.get("embedding_model") != current.get("embedding_model"):
        mismatches.append(
            f"Embedding model mismatch: stored='{stored.get('embedding_model')}', current='{current.get('embedding_model')}'"
        )
    if stored.get("embedding_dimension") != current.get("embedding_dimension"):
        mismatches.append(
            f"Embedding dimension mismatch: stored={stored.get('embedding_dimension')}, current={current.get('embedding_dimension')}"
        )

    # 3. Request timeout
    if stored.get("ollama_timeout") != current.get("ollama_timeout"):
        mismatches.append(
            f"Ollama timeout mismatch: stored={stored.get('ollama_timeout')}, current={current.get('ollama_timeout')}"
        )

    # 4. Passage selection limits
    stored_ps = stored.get("passage_selection", {})
    current_ps = current.get("passage_selection", {})
    if stored_ps.get("max_passages") != current_ps.get("max_passages"):
        mismatches.append(
            f"Passage limit max_passages mismatch: stored={stored_ps.get('max_passages')}, current={current_ps.get('max_passages')}"
        )
    if stored_ps.get("max_total_words") != current_ps.get("max_total_words"):
        mismatches.append(
            f"Passage budget max_total_words mismatch: stored={stored_ps.get('max_total_words')}, current={current_ps.get('max_total_words')}"
        )

    # 5. Schema version
    if stored.get("extraction_schema_version") != current.get("extraction_schema_version"):
        mismatches.append(
            f"Extraction schema version mismatch: stored='{stored.get('extraction_schema_version')}', current='{current.get('extraction_schema_version')}'"
        )

    # 6. Gap discovery settings
    stored_gd = stored.get("gap_discovery", {})
    current_gd = current.get("gap_discovery", {})
    if stored_gd.get("similarity_threshold") != current_gd.get("similarity_threshold"):
        mismatches.append(
            f"Gap discovery similarity_threshold mismatch: stored={stored_gd.get('similarity_threshold')}, current={current_gd.get('similarity_threshold')}"
        )
    if stored_gd.get("min_supporting_evidence") != current_gd.get("min_supporting_evidence"):
        mismatches.append(
            f"Evidence threshold min_supporting_evidence mismatch: stored={stored_gd.get('min_supporting_evidence')}, current={current_gd.get('min_supporting_evidence')}"
        )

    # 7. Opportunity generation settings
    if stored.get("max_refinement_iterations") != current.get("max_refinement_iterations"):
        mismatches.append(
            f"Opportunity max refinement iterations mismatch: stored={stored.get('max_refinement_iterations')}, current={current.get('max_refinement_iterations')}"
        )

    # 8. Temporal cutoff
    if stored.get("temporal_cutoff") != current.get("temporal_cutoff"):
        mismatches.append(
            f"Temporal cutoff mismatch: stored={stored.get('temporal_cutoff')!r}, current={current.get('temporal_cutoff')!r}"
        )

    # 9. Stage 5 settings
    stored_s5 = stored.get("enable_stage_5", False)
    current_s5 = current.get("enable_stage_5", False)
    if stored_s5 != current_s5:
        mismatches.append(
            f"Stage 5 enabled mismatch: stored={stored_s5}, current={current_s5}"
        )
    if current_s5 and stored_s5:
        if stored.get("stage_5_batch_window_days") != current.get("stage_5_batch_window_days"):
            mismatches.append(
                f"Stage 5 batch window mismatch: stored={stored.get('stage_5_batch_window_days')}, current={current.get('stage_5_batch_window_days')}"
            )
        if stored.get("stage_5_cutoffs") != current.get("stage_5_cutoffs"):
            mismatches.append(
                f"Stage 5 cutoffs mismatch: stored={stored.get('stage_5_cutoffs')}, current={current.get('stage_5_cutoffs')}"
            )
        if stored.get("stage_5_chunk_size") != current.get("stage_5_chunk_size"):
            mismatches.append(
                f"Stage 5 chunk size mismatch: stored={stored.get('stage_5_chunk_size')}, current={current.get('stage_5_chunk_size')}"
            )

    return mismatches


def is_opportunity_available_at_cutoff(
    opp: Opportunity,
    cutoff_date: datetime.date,
) -> bool:
    """Returns True if the opportunity was created/available on or before the cutoff date."""
    opp_dt = opp.created_at
    if opp_dt is None and getattr(opp, "research_opportunity", None):
        opp_dt = opp.research_opportunity.created_at

    if opp_dt is not None:
        opp_date = opp_dt.date() if isinstance(opp_dt, datetime.datetime) else opp_dt
        if opp_date > cutoff_date:
            return False
    return True


def build_temporal_batches(
    papers: List[Paper],
    baseline_cutoff: Optional[datetime.date] = None,
    explicit_cutoffs: Optional[List[datetime.date | str]] = None,
    batch_window_days: Optional[int] = None,
) -> List[TemporalBatch]:
    """Constructs deterministic chronological batches for temporal evaluation."""
    dated_papers = [p for p in papers if p.publication_date is not None]
    if baseline_cutoff:
        eval_papers = [p for p in dated_papers if p.publication_date > baseline_cutoff]
    else:
        eval_papers = dated_papers

    if not eval_papers:
        return []

    eval_papers.sort(key=lambda p: (p.publication_date, str(p.id)))
    min_date = eval_papers[0].publication_date
    max_date = eval_papers[-1].publication_date

    batches: List[TemporalBatch] = []

    if explicit_cutoffs:
        normalized_cutoffs: List[datetime.date] = []
        for c in explicit_cutoffs:
            if isinstance(c, str):
                normalized_cutoffs.append(datetime.date.fromisoformat(c.split("T")[0]))
            elif isinstance(c, datetime.datetime):
                normalized_cutoffs.append(c.date())
            elif isinstance(c, datetime.date):
                normalized_cutoffs.append(c)
        normalized_cutoffs.sort()

        current_start = (
            (baseline_cutoff + datetime.timedelta(days=1))
            if baseline_cutoff
            else min_date
        )

        for seq, c_date in enumerate(normalized_cutoffs):
            if c_date < current_start:
                continue
            batch_id = f"batch_{seq}_{c_date.isoformat()}"
            batches.append(
                TemporalBatch(
                    batch_id=batch_id,
                    start_date=current_start,
                    end_date=c_date,
                    sequence_number=seq,
                )
            )
            current_start = c_date + datetime.timedelta(days=1)

    elif batch_window_days and batch_window_days > 0:
        current_start = (
            (baseline_cutoff + datetime.timedelta(days=1))
            if baseline_cutoff
            else min_date
        )
        seq = 0
        while current_start <= max_date:
            current_end = current_start + datetime.timedelta(days=batch_window_days - 1)
            batch_id = f"batch_{seq}_{current_start.isoformat()}_{current_end.isoformat()}"
            batches.append(
                TemporalBatch(
                    batch_id=batch_id,
                    start_date=current_start,
                    end_date=current_end,
                    sequence_number=seq,
                )
            )
            current_start = current_end + datetime.timedelta(days=1)
            seq += 1

    else:
        distinct_dates = sorted(list({p.publication_date for p in eval_papers}))
        current_start = (
            (baseline_cutoff + datetime.timedelta(days=1))
            if baseline_cutoff
            else min_date
        )
        for seq, d in enumerate(distinct_dates):
            batch_id = f"batch_{seq}_{d.isoformat()}"
            batches.append(
                TemporalBatch(
                    batch_id=batch_id,
                    start_date=current_start,
                    end_date=d,
                    sequence_number=seq,
                )
            )
            current_start = d + datetime.timedelta(days=1)

    return batches


def _atomic_write_json(filepath: Path, data: Dict[str, Any]) -> None:
    """Writes a dictionary as formatted JSON atomically using a temporary file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = filepath.with_suffix(f"{filepath.suffix}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(filepath)


def _atomic_write_text(filepath: Path, content: str) -> None:
    """Writes a string to a file atomically using a temporary file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = filepath.with_suffix(f"{filepath.suffix}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
    tmp_path.replace(filepath)


class PaperInputMetadata:
    """Metadata for an input PDF file."""

    def __init__(
        self,
        filepath: Path,
        title: Optional[str] = None,
        publication_date: Optional[datetime.date] = None,
        doi: Optional[str] = None,
        authors: Optional[List[Dict[str, Any]]] = None,
        is_explicit: bool = False,
        content_hash: Optional[str] = None,
    ):
        self.filepath = filepath
        self.title = title
        self.publication_date = publication_date
        self.doi = doi
        self.authors = authors or []
        self.is_explicit = is_explicit
        self.content_hash = content_hash or self.compute_sha256()

    def compute_sha256(self) -> str:
        """Computes deterministic SHA-256 hash of PDF bytes."""
        hasher = hashlib.sha256()
        with open(self.filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filepath.name,
            "filepath": str(self.filepath),
            "content_hash": self.content_hash,
            "title": self.title,
            "publication_date": str(self.publication_date) if self.publication_date else None,
            "doi": self.doi,
            "authors_count": len(self.authors),
            "is_explicit_metadata": self.is_explicit,
        }


class ExperimentResult:
    """Container for experiment execution outcome and metrics."""

    def __init__(
        self,
        success: bool,
        metrics: Dict[str, Any],
        metrics_file: Optional[Path] = None,
        summary_file: Optional[Path] = None,
        error: Optional[str] = None,
    ):
        self.success = success
        self.metrics = metrics
        self.metrics_file = metrics_file
        self.summary_file = summary_file
        self.error = error


class ControlledExperimentRunner:
    """Executes an interruption-safe, persistent, resumable multi-paper experiment."""

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        db_path: Optional[str | Path] = None,
        artifacts_dir: Optional[str | Path] = None,
        min_supporting_evidence: int = 1,
        retry_failed: bool = True,
        max_passages: int = 12,
        max_total_words: int = 2500,
        similarity_threshold: float = 0.65,
        extraction_schema_version: str = "v1.0",
        ollama_timeout: Optional[float] = None,
        max_refinement_iterations: int = 2,
        temporal_cutoff: Optional[datetime.date | str | datetime.datetime] = None,
        enable_stage_5: bool = False,
        stage_5_batches: Optional[List[TemporalBatch]] = None,
        stage_5_cutoffs: Optional[List[datetime.date | str]] = None,
        stage_5_batch_window_days: Optional[int] = None,
        stage_5_max_retries: int = 2,
        stage_5_backoff_factor: float = 0.5,
        stage_5_timeout: Optional[float] = None,
        stage_5_chunk_size: Optional[int] = None,
    ):
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        self.db_path = db_path
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
        self.min_supporting_evidence = min_supporting_evidence
        self.retry_failed = retry_failed
        self.max_passages = max_passages
        self.max_total_words = max_total_words
        self.similarity_threshold = similarity_threshold
        self.extraction_schema_version = extraction_schema_version
        self.ollama_timeout = ollama_timeout
        self.max_refinement_iterations = max_refinement_iterations
        self.temporal_cutoff = normalize_cutoff_date(temporal_cutoff)
        self.enable_stage_5 = enable_stage_5
        self.stage_5_batches = stage_5_batches
        self.stage_5_cutoffs = stage_5_cutoffs
        self.stage_5_batch_window_days = stage_5_batch_window_days
        self.stage_5_max_retries = stage_5_max_retries
        self.stage_5_backoff_factor = stage_5_backoff_factor
        self.stage_5_timeout = stage_5_timeout
        self.stage_5_chunk_size = stage_5_chunk_size

    def _resolve_manifest(self, input_dir: Path) -> List[PaperInputMetadata]:
        """Discovers PDF files in input_dir and matches against optional manifest."""
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

        pdf_files = sorted(list(input_dir.glob("*.pdf")))
        if not pdf_files:
            raise ValueError(f"No PDF files found in directory: {input_dir}")

        # Check for manifest.json or metadata.json
        manifest_data: Dict[str, Dict[str, Any]] = {}
        for m_name in ["manifest.json", "metadata.json"]:
            m_path = input_dir / m_name
            if m_path.exists():
                try:
                    with open(m_path, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                    if isinstance(raw, list):
                        for item in raw:
                            if isinstance(item, dict) and "filename" in item:
                                manifest_data[item["filename"]] = item
                    elif isinstance(raw, dict):
                        if "papers" in raw and isinstance(raw["papers"], list):
                            for item in raw["papers"]:
                                if isinstance(item, dict) and "filename" in item:
                                    manifest_data[item["filename"]] = item
                        else:
                            for k, v in raw.items():
                                if isinstance(v, dict):
                                    manifest_data[k] = v
                except Exception as e:
                    logger.warning(f"Could not parse manifest {m_path}: {e}")
                break

        paper_inputs: List[PaperInputMetadata] = []
        for pdf_path in pdf_files:
            meta = manifest_data.get(pdf_path.name, {})
            pub_date = None
            if "publication_date" in meta and meta["publication_date"]:
                try:
                    pub_date = datetime.date.fromisoformat(str(meta["publication_date"]).split("T")[0])
                except (ValueError, TypeError):
                    pub_date = None

            is_explicit = bool(meta)
            paper_inputs.append(
                PaperInputMetadata(
                    filepath=pdf_path,
                    title=meta.get("title"),
                    publication_date=pub_date,
                    doi=meta.get("doi"),
                    authors=meta.get("authors") or [],
                    is_explicit=is_explicit,
                )
            )

        return paper_inputs

    def _checkpoint_metrics(self, metrics: Dict[str, Any]) -> None:
        """Atomically checkpoints current metrics and summary to artifacts directory."""
        if not self.artifacts_dir:
            return
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = self.artifacts_dir / "experiment_metrics.json"
        summary_file = self.artifacts_dir / "experiment_summary.md"

        _atomic_write_json(metrics_file, metrics)
        summary_md = self._render_summary_markdown(metrics)
        _atomic_write_text(summary_file, summary_md)

    def _update_understanding_metrics(
        self,
        metrics: Dict[str, Any],
        understood_papers: List[Dict[str, Any]],
        understanding_failures: List[Dict[str, Any]],
        under_start: float,
        session: Session,
    ) -> None:
        """Helper to recompute statement and provenance counts for metrics."""
        all_stmts = session.query(ResearchStatement).all()
        corpus_stmts_by_type: Dict[str, int] = {}
        total_prov_valid = 0
        for s in all_stmts:
            corpus_stmts_by_type[s.statement_type] = corpus_stmts_by_type.get(s.statement_type, 0) + 1
            if s.passage_id:
                pass_exists = (
                    session.query(PaperPassage)
                    .filter(PaperPassage.id == s.passage_id, PaperPassage.paper_id == s.paper_id)
                    .first()
                )
                if pass_exists is not None:
                    total_prov_valid += 1

        under_dict: Dict[str, Any] = {
            "understood_count": len(understood_papers),
            "understood_papers_count": len(understood_papers),
            "resumed_count": sum(1 for p in understood_papers if p.get("resumed_from_cache")),
            "failed_count": len(understanding_failures),
            "total_statements_extracted": len(all_stmts),
            "corpus_statements_by_type": corpus_stmts_by_type,
            "overall_provenance_valid_count": total_prov_valid,
            "overall_provenance_validity_rate": (
                round(total_prov_valid / len(all_stmts), 3) if all_stmts else 0.0
            ),
            "understood_papers": understood_papers,
            "failures": understanding_failures,
        }
        if self.temporal_cutoff:
            eligible_stmts = (
                session.query(ResearchStatement)
                .join(ResearchStatement.paper)
                .filter(
                    Paper.publication_date.is_not(None),
                    Paper.publication_date <= self.temporal_cutoff,
                )
                .all()
            )
            under_dict["temporal_cutoff"] = self.temporal_cutoff.isoformat()
            under_dict["eligible_statements_extracted"] = len(eligible_stmts)

        metrics["understanding"] = under_dict
        metrics["timings_seconds"]["understanding_total"] = round(time.time() - under_start, 2)

    def run(self, input_dir: Path | str) -> ExperimentResult:
        """Executes the controlled experiment with interruption safety and resumption."""
        start_time = time.time()
        input_dir_path = Path(input_dir)
        temp_db_file: Optional[Path] = None
        is_persistent_db = False
        interrupted = False

        # 0. Run offline corpus metadata and publication-date preflight
        try:
            preflight_report = run_corpus_preflight(input_dir_path)
            logger.info(
                f"Corpus Preflight: {preflight_report.total_papers} papers "
                f"({preflight_report.dated_papers_count} dated, "
                f"{preflight_report.undated_papers_count} undated, "
                f"{preflight_report.malformed_dates_count} malformed)"
            )
        except Exception as pf_err:
            preflight_report = None
            logger.warning(f"Corpus preflight check could not be completed: {pf_err}")

        # 1. Resolve and validate input papers
        try:
            paper_inputs = self._resolve_manifest(input_dir_path)
        except Exception as e:
            return ExperimentResult(
                success=False,
                metrics={"error": f"Input validation failed: {str(e)}"},
                error=str(e),
            )

        # Build corpus hash mapping
        corpus_hashes = {p.filepath.name: p.content_hash for p in paper_inputs}

        # 2. Setup isolated SQLite database
        if self.db_path == ":memory:":
            db_uri = "sqlite:///:memory:"
            database_mode = "in-memory"
            is_persistent_db = False
        elif self.db_path:
            db_file = Path(self.db_path).resolve()
            db_file.parent.mkdir(parents=True, exist_ok=True)
            db_uri = f"sqlite:///{db_file}"
            database_mode = "persistent-sqlite"
            is_persistent_db = True
        elif self.artifacts_dir:
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            db_file = (self.artifacts_dir / "experiment_state.db").resolve()
            db_uri = f"sqlite:///{db_file}"
            database_mode = "persistent-sqlite"
            is_persistent_db = True
        else:
            fd, tmp_name = tempfile.mkstemp(prefix="ideagen_experiment_", suffix=".db")
            os.close(fd)
            temp_db_file = Path(tmp_name)
            db_uri = f"sqlite:///{temp_db_file}"
            database_mode = "temporary-sqlite"
            is_persistent_db = False

        engine = create_engine(
            db_uri,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        configure_sqlite_foreign_keys(engine)
        Base.metadata.create_all(engine)
        SessionMaker = sessionmaker(bind=engine)
        session: Session = SessionMaker()

        # 3. Resolve providers and compute configuration fingerprint
        llm = self.llm_provider or get_default_llm_provider()
        embedder = self.embedding_provider or get_default_embedding_provider()
        resolved_llm_name = getattr(llm, "model_name", "qwen3:8b")
        resolved_embedder_name = getattr(embedder, "model_name", "nomic-embed-text")
        resolved_dim = embedder.dimension

        app_settings = get_settings()
        if self.ollama_timeout is not None:
            resolved_timeout = float(self.ollama_timeout)
        elif hasattr(llm, "timeout") and getattr(llm, "timeout") is not None:
            resolved_timeout = float(getattr(llm, "timeout"))
        else:
            resolved_timeout = float(app_settings.ollama_timeout)

        # Synchronize resolved timeout to active provider if writable
        if hasattr(llm, "timeout"):
            try:
                setattr(llm, "timeout", resolved_timeout)
            except (AttributeError, TypeError):
                pass

        current_fingerprint = compute_configuration_fingerprint(
            llm_provider=llm,
            llm_model=resolved_llm_name,
            embedding_provider=embedder,
            embedding_model=resolved_embedder_name,
            embedding_dimension=resolved_dim,
            max_passages=self.max_passages,
            max_total_words=self.max_total_words,
            extraction_schema_version=self.extraction_schema_version,
            similarity_threshold=self.similarity_threshold,
            min_supporting_evidence=self.min_supporting_evidence,
            ollama_timeout=resolved_timeout,
            max_refinement_iterations=self.max_refinement_iterations,
            temporal_cutoff=str(self.temporal_cutoff) if self.temporal_cutoff else None,
            enable_stage_5=self.enable_stage_5,
            stage_5_batch_window_days=self.stage_5_batch_window_days,
            stage_5_cutoffs=[str(c) for c in self.stage_5_cutoffs] if self.stage_5_cutoffs else None,
            stage_5_chunk_size=self.stage_5_chunk_size,
            stage_5_max_retries=self.stage_5_max_retries,
        )

        # 4. Check configuration consistency against existing database
        existing_docs = session.query(PaperDocument).all()
        if existing_docs:
            meta_row = (
                session.query(ExperimentMetadataRecord)
                .filter(ExperimentMetadataRecord.key == "configuration_fingerprint")
                .first()
            )
            if not meta_row or not meta_row.value:
                session.close()
                engine.dispose()
                err_msg = (
                    "Configuration mismatch detected: Existing database contains populated experiment data "
                    "but stored configuration fingerprint is missing. Refusing unsafe resume operation."
                )
                return ExperimentResult(
                    success=False,
                    metrics={"error": err_msg},
                    error=err_msg,
                )

            try:
                stored_fingerprint = json.loads(meta_row.value)
                if not isinstance(stored_fingerprint, dict):
                    raise ValueError("Stored configuration fingerprint is not a valid JSON dictionary")
            except Exception as e:
                session.close()
                engine.dispose()
                err_msg = (
                    f"Configuration mismatch detected: Stored configuration fingerprint is malformed: {e}. "
                    "Refusing unsafe resume operation."
                )
                return ExperimentResult(
                    success=False,
                    metrics={"error": err_msg},
                    error=err_msg,
                )

            mismatches = compare_fingerprints(stored_fingerprint, current_fingerprint)
            if mismatches:
                session.close()
                engine.dispose()
                err_msg = (
                    "Configuration mismatch detected against existing database:\n- "
                    + "\n- ".join(mismatches)
                    + "\nRefusing to resume with incompatible settings."
                )
                return ExperimentResult(
                    success=False,
                    metrics={"error": err_msg, "configuration_mismatches": mismatches},
                    error=err_msg,
                )

            # Check corpus consistency against existing database
            existing_hashes = {d.content_hash for d in existing_docs if d.content_hash}
            current_hashes = set(corpus_hashes.values())

            # Detect if a file with the same name was changed
            for d in existing_docs:
                if d.original_filename and d.original_filename in corpus_hashes:
                    if d.content_hash != corpus_hashes[d.original_filename]:
                        session.close()
                        engine.dispose()
                        err_msg = (
                            f"Corpus mismatch detected: File '{d.original_filename}' has content hash "
                            f"'{corpus_hashes[d.original_filename]}' in input corpus, but existing database "
                            f"recorded content hash '{d.content_hash}'. Refusing unsafe resume operation."
                        )
                        return ExperimentResult(
                            success=False,
                            metrics={"error": err_msg},
                            error=err_msg,
                        )

            # Detect if database contains papers absent from input corpus
            orphan_hashes = existing_hashes - current_hashes
            if orphan_hashes:
                session.close()
                engine.dispose()
                err_msg = (
                    f"Corpus mismatch detected: Existing database contains {len(orphan_hashes)} paper(s) "
                    f"not present in the current input directory. Refusing unsafe resume operation."
                )
                return ExperimentResult(
                    success=False,
                    metrics={"error": err_msg},
                    error=err_msg,
                )
        else:
            # Fresh database: persist initial configuration fingerprint
            meta_row = (
                session.query(ExperimentMetadataRecord)
                .filter(ExperimentMetadataRecord.key == "configuration_fingerprint")
                .first()
            )
            if not meta_row:
                session.add(
                    ExperimentMetadataRecord(
                        key="configuration_fingerprint",
                        value=json.dumps(current_fingerprint),
                    )
                )
                session.commit()

        # Track paper statuses
        paper_statuses: Dict[str, str] = {p.filepath.name: "PENDING" for p in paper_inputs}

        # Metrics structure
        metrics: Dict[str, Any] = {
            "metadata": {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "input_directory": str(input_dir_path),
                "database_mode": database_mode,
                "database_isolated": True,
                "production_postgresql_connected": False,
                "corpus_hashes": corpus_hashes,
                "configuration_fingerprint": current_fingerprint,
                "corpus_preflight": preflight_report.to_dict() if preflight_report else None,
                "interrupted": False,
            },
            "papers_input": [p.to_dict() for p in paper_inputs],
            "paper_statuses": paper_statuses,
            "ingestion": {},
            "understanding": {},
            "discovery": {},
            "opportunities": {},
            "stage_5_evolution": {},
            "timings_seconds": {},
        }

        current_in_flight_paper: Optional[str] = None
        current_in_flight_paper_id: Optional[uuid.UUID] = None

        try:
            persistence = SQLAlchemyPersistenceLayer(session)

            metrics["metadata"]["llm_model"] = resolved_llm_name
            metrics["metadata"]["embedding_model"] = resolved_embedder_name
            metrics["metadata"]["embedding_dimension"] = resolved_dim
            metrics["metadata"]["ollama_timeout"] = resolved_timeout
            metrics["metadata"]["max_refinement_iterations"] = self.max_refinement_iterations

            # ── STAGE 1: INGESTION (RESUMABLE) ─────────────────────────────
            ingest_start = time.time()
            ingestion_service = ManualPDFIngestionService(persistence=persistence)
            ingested_papers: List[Dict[str, Any]] = []
            ingestion_failures: List[Dict[str, Any]] = []

            for p_in in paper_inputs:
                p_start = time.time()
                try:
                    # Check if already ingested in database
                    existing_doc = (
                        session.query(PaperDocument)
                        .filter(PaperDocument.content_hash == p_in.content_hash)
                        .first()
                    )
                    if existing_doc and existing_doc.paper:
                        db_p = existing_doc.paper
                        p_id = db_p.id
                        passages_count = (
                            session.query(PaperPassage)
                            .filter(PaperPassage.paper_id == p_id)
                            .count()
                        )
                        ingested_papers.append({
                            "filename": p_in.filepath.name,
                            "paper_id": str(p_id),
                            "title": db_p.title or p_in.title,
                            "publication_date": str(db_p.publication_date) if db_p and db_p.publication_date else None,
                            "content_hash": p_in.content_hash,
                            "passages_count": passages_count,
                            "wall_time_s": 0.0,
                            "reused_from_db": True,
                        })
                        paper_statuses[p_in.filepath.name] = "INGESTED"
                        logger.info(f"Reusing existing ingested paper '{p_in.filepath.name}' (id: {p_id}, passages: {passages_count})")
                    else:
                        p_dto, doc_dto = ingestion_service.ingest_pdf(
                            pdf_path=p_in.filepath,
                            title=p_in.title,
                            publication_date=p_in.publication_date,
                            doi=p_in.doi,
                            authors=p_in.authors,
                            check_duplicates=True,
                        )
                        db_p = session.query(Paper).filter(Paper.source_id == p_dto.source_id).first()
                        p_id = db_p.id if db_p else None
                        passages_count = (
                            session.query(PaperPassage).filter(PaperPassage.paper_id == p_id).count() if p_id else 0
                        )
                        ingested_papers.append({
                            "filename": p_in.filepath.name,
                            "paper_id": str(p_id) if p_id else None,
                            "title": db_p.title if db_p else p_in.title,
                            "publication_date": str(db_p.publication_date) if db_p and db_p.publication_date else None,
                            "content_hash": doc_dto.content_hash,
                            "passages_count": passages_count,
                            "wall_time_s": round(time.time() - p_start, 2),
                            "reused_from_db": False,
                        })
                        paper_statuses[p_in.filepath.name] = "INGESTED"
                except Exception as e:
                    logger.error(f"Ingestion failed for {p_in.filepath.name}: {e}")
                    paper_statuses[p_in.filepath.name] = "FAILED"
                    ingestion_failures.append({
                        "filename": p_in.filepath.name,
                        "error": str(e),
                    })

            metrics["ingestion"] = {
                "total_papers": len(paper_inputs),
                "ingested_count": len(ingested_papers),
                "success_count": len(ingested_papers),
                "reused_count": sum(1 for p in ingested_papers if p.get("reused_from_db")),
                "failed_count": len(ingestion_failures),
                "total_passages_created": session.query(PaperPassage).count(),
                "total_passages_extracted": session.query(PaperPassage).count(),
                "ingested_papers": ingested_papers,
                "failures": ingestion_failures,
            }
            metrics["timings_seconds"]["ingestion_total"] = round(time.time() - ingest_start, 2)
            self._checkpoint_metrics(metrics)

            if not ingested_papers:
                metrics["discovery"]["status"] = "SKIPPED_NO_PAPERS"
                return self._finalize_result(True, metrics, session)

            # ── STAGE 2: PAPER UNDERSTANDING (RESUMABLE & CHECKPOINTED) ──
            under_start = time.time()
            understanding_pipeline = PaperUnderstandingPipeline(
                persistence=persistence,
                provider=llm,
                model_name=resolved_llm_name,
            )
            understanding_pipeline.selector = PassageSelector(
                max_passages=self.max_passages,
                max_total_words=self.max_total_words,
            )

            understood_papers: List[Dict[str, Any]] = []
            understanding_failures: List[Dict[str, Any]] = []

            for p_info in ingested_papers:
                p_id_str = p_info["paper_id"]
                if not p_id_str:
                    continue
                p_id = uuid.UUID(p_id_str)
                fname = p_info["filename"]
                pu_start = time.time()

                # Check existing understanding in DB
                existing_k = session.query(ResearchKnowledge).filter(ResearchKnowledge.paper_id == p_id).first()
                if existing_k is not None:
                    if existing_k.extraction_status == "SUCCESS":
                        stmts = session.query(ResearchStatement).filter(ResearchStatement.paper_id == p_id).all()
                        stmts_by_type: Dict[str, int] = {}
                        provenance_valid_count = 0
                        for s in stmts:
                            stmts_by_type[s.statement_type] = stmts_by_type.get(s.statement_type, 0) + 1
                            if s.passage_id:
                                pass_exists = (
                                    session.query(PaperPassage)
                                    .filter(PaperPassage.id == s.passage_id, PaperPassage.paper_id == p_id)
                                    .first()
                                )
                                if pass_exists is not None:
                                    provenance_valid_count += 1

                        understood_papers.append({
                            "paper_id": str(p_id),
                            "filename": fname,
                            "status": "COMPLETED",
                            "statements_count": len(stmts),
                            "statements_by_type": stmts_by_type,
                            "provenance_valid_statements": provenance_valid_count,
                            "provenance_validity_rate": round(provenance_valid_count / len(stmts), 3) if stmts else 1.0,
                            "wall_time_s": 0.0,
                            "resumed_from_cache": True,
                        })
                        paper_statuses[fname] = "COMPLETED"
                        logger.info(f"Skipping LLM understanding for paper '{fname}' (reused {len(stmts)} statements from DB)")
                        self._update_understanding_metrics(metrics, understood_papers, understanding_failures, under_start, session)
                        self._checkpoint_metrics(metrics)
                        continue
                    else:
                        # Previously failed
                        if self.retry_failed:
                            logger.info(f"Retrying understanding for previously failed paper '{fname}'...")
                            session.delete(existing_k)
                            session.commit()
                        else:
                            logger.info(f"Skipping retry for previously failed paper '{fname}' (retry_failed=False)")
                            paper_statuses[fname] = "FAILED"
                            understanding_failures.append({
                                "paper_id": str(p_id),
                                "filename": fname,
                                "status": "FAILED",
                                "error": existing_k.error_message or "Previous extraction failed",
                            })
                            self._update_understanding_metrics(metrics, understood_papers, understanding_failures, under_start, session)
                            self._checkpoint_metrics(metrics)
                            continue

                current_in_flight_paper = fname
                current_in_flight_paper_id = p_id

                try:
                    k_dto = understanding_pipeline.understand_paper(paper_id=p_id)
                    if k_dto.extraction_status == "SUCCESS":
                        stmts = session.query(ResearchStatement).filter(ResearchStatement.paper_id == p_id).all()
                        stmts_by_type = {}
                        provenance_valid_count = 0
                        for s in stmts:
                            stmts_by_type[s.statement_type] = stmts_by_type.get(s.statement_type, 0) + 1
                            if s.passage_id:
                                pass_exists = (
                                    session.query(PaperPassage)
                                    .filter(PaperPassage.id == s.passage_id, PaperPassage.paper_id == p_id)
                                    .first()
                                )
                                if pass_exists is not None:
                                    provenance_valid_count += 1

                        understood_papers.append({
                            "paper_id": str(p_id),
                            "filename": fname,
                            "status": "COMPLETED",
                            "statements_count": len(stmts),
                            "statements_by_type": stmts_by_type,
                            "provenance_valid_statements": provenance_valid_count,
                            "provenance_validity_rate": round(provenance_valid_count / len(stmts), 3) if stmts else 1.0,
                            "wall_time_s": round(time.time() - pu_start, 2),
                            "resumed_from_cache": False,
                        })
                        paper_statuses[fname] = "COMPLETED"
                    else:
                        paper_statuses[fname] = "FAILED"
                        understanding_failures.append({
                            "paper_id": str(p_id),
                            "filename": fname,
                            "status": "FAILED",
                            "error": k_dto.error_message or "Model extraction failed",
                        })
                    current_in_flight_paper = None
                    current_in_flight_paper_id = None
                except Exception as e:
                    current_in_flight_paper = None
                    current_in_flight_paper_id = None
                    logger.error(f"Understanding failed for paper {p_id}: {e}")
                    paper_statuses[fname] = "FAILED"
                    understanding_failures.append({
                        "paper_id": str(p_id),
                        "filename": fname,
                        "status": "FAILED",
                        "error": str(e),
                    })

                self._update_understanding_metrics(metrics, understood_papers, understanding_failures, under_start, session)
                self._checkpoint_metrics(metrics)

            self._update_understanding_metrics(metrics, understood_papers, understanding_failures, under_start, session)
            self._checkpoint_metrics(metrics)

            all_stmts = session.query(ResearchStatement).all()
            if not all_stmts:
                failure_reason = "Zero statements extracted across all papers."
                metrics["discovery"]["status"] = "SKIPPED_NO_STATEMENTS"
                metrics["fatal_error"] = failure_reason
                return self._finalize_result(False, metrics, session, error=failure_reason)

            if self.temporal_cutoff:
                from ideagen.discovery.temporal import filter_statements_by_cutoff
                eligible_stmts = persistence.get_all_research_statements(cutoff_date=self.temporal_cutoff)
                eligible_stmts = filter_statements_by_cutoff(eligible_stmts, self.temporal_cutoff)
                if not eligible_stmts:
                    failure_reason = f"Zero eligible statements found for temporal cutoff {self.temporal_cutoff.isoformat()}."
                    metrics["discovery"]["status"] = "SKIPPED_NO_STATEMENTS"
                    metrics["fatal_error"] = failure_reason
                    return self._finalize_result(False, metrics, session, error=failure_reason)

            # ── STAGE 3: SEMANTIC GAP DISCOVERY (DEDUPLICATED) ─────────────
            disc_start = time.time()
            all_db_gaps = session.query(ResearchGap).all()
            validated_gaps: List[Dict[str, Any]] = []
            rejected_gaps: List[Dict[str, Any]] = []

            if all_db_gaps:
                stmt_ids_in_db = {s.id for s in all_stmts}
                gap_evidence_valid = True
                for g in all_db_gaps:
                    ev_records = persistence.get_gap_evidence(g.id)
                    if not ev_records or any(e.statement_id not in stmt_ids_in_db for e in ev_records):
                        gap_evidence_valid = False
                        break

                newly_understood_in_this_run = any(not p.get("resumed_from_cache") for p in understood_papers)

                if gap_evidence_valid and not newly_understood_in_this_run:
                    logger.info(f"Reusing {len(all_db_gaps)} verified research gaps from existing database.")
                    for g in all_db_gaps:
                        ev_records = persistence.get_gap_evidence(g.id)
                        p_ids = list({e.paper_id for e in ev_records})
                        g_dict = {
                            "gap_id": str(g.id),
                            "gap_statement": g.gap_statement,
                            "research_problem": g.research_problem,
                            "status": g.status,
                            "evidence_count": g.evidence_count,
                            "distinct_supporting_papers_count": len(p_ids),
                            "supporting_paper_ids": [str(pid) for pid in p_ids],
                            "validation_confidence": g.validation_confidence,
                            "validation_reasoning": g.validation_reasoning,
                            "resumed_from_cache": True,
                        }
                        if g.status == GapStatus.VALIDATED.value or g.status == "VALIDATED":
                            validated_gaps.append(g_dict)
                        else:
                            rejected_gaps.append(g_dict)
                elif newly_understood_in_this_run:
                    raise RuntimeError(
                        "Cannot safely reuse existing research gaps: new statements were extracted in this run, "
                        "so existing gaps do not reflect the complete statement corpus. "
                        "Please run with a clean database or specify a new --db-path."
                    )
                else:
                    raise RuntimeError(
                        "Cannot safely reuse existing research gaps: gap evidence records reference statements "
                        "missing from current database. Refusing to mix inconsistent results."
                    )
            else:
                discovery_pipeline = SemanticGapDiscoveryPipeline(
                    persistence=persistence,
                    embedding_provider=embedder,
                    llm_provider=llm,
                    similarity_threshold=self.similarity_threshold,
                    min_supporting_evidence=self.min_supporting_evidence,
                )
                discovered_gaps = discovery_pipeline.discover_gaps(
                    persist_index=False,
                    cutoff_date=self.temporal_cutoff,
                )
                all_db_gaps = session.query(ResearchGap).all()

                for g in all_db_gaps:
                    ev_records = persistence.get_gap_evidence(g.id)
                    p_ids = list({e.paper_id for e in ev_records})
                    g_dict = {
                        "gap_id": str(g.id),
                        "gap_statement": g.gap_statement,
                        "research_problem": g.research_problem,
                        "status": g.status,
                        "evidence_count": g.evidence_count,
                        "distinct_supporting_papers_count": len(p_ids),
                        "supporting_paper_ids": [str(pid) for pid in p_ids],
                        "validation_confidence": g.validation_confidence,
                        "validation_reasoning": g.validation_reasoning,
                        "resumed_from_cache": False,
                    }
                    if g.status == GapStatus.VALIDATED.value or g.status == "VALIDATED":
                        validated_gaps.append(g_dict)
                    else:
                        rejected_gaps.append(g_dict)

            metrics["discovery"] = {
                "candidate_gaps_total": len(all_db_gaps),
                "validated_gaps_count": len(validated_gaps),
                "rejected_gaps_count": len(rejected_gaps),
                "validated_gaps": validated_gaps,
                "rejected_gaps": rejected_gaps,
            }
            metrics["timings_seconds"]["discovery_total"] = round(time.time() - disc_start, 2)
            self._checkpoint_metrics(metrics)

            # ── STAGE 4: OPPORTUNITY GENERATION (RESUMABLE) ────────────────
            opp_start = time.time()
            opportunities_generated: List[Dict[str, Any]] = []
            opportunity_failures: List[Dict[str, Any]] = []

            if not validated_gaps:
                metrics["opportunities"] = {
                    "status": "NO_VALIDATED_GAPS",
                    "opportunities_count": 0,
                    "note": "Opportunity generation skipped because no candidate gaps satisfied validation criteria.",
                }
            else:
                vector_index = None
                opp_pipeline = None

                for vg in validated_gaps:
                    gap_id = uuid.UUID(vg["gap_id"])
                    gap = persistence.get_research_gap(gap_id)
                    if not gap:
                        continue

                    # Check if opportunity already exists
                    existing_opp = (
                        session.query(ResearchOpportunityRecord)
                        .filter(ResearchOpportunityRecord.gap_id == gap_id)
                        .first()
                    )
                    if existing_opp:
                        logger.info(f"Reusing existing research opportunity for gap {gap_id}")
                        try:
                            bridge_svc = OpportunityBridgeService(session)
                            bridge_svc.bridge_opportunity(existing_opp)
                            session.commit()
                        except Exception as bridge_err:
                            logger.warning(f"Failed to bridge cached opportunity {existing_opp.id}: {bridge_err}")
                        report_service = OpportunityReportService(session)
                        report = report_service.generate_opportunity_report(existing_opp.id)
                        opportunities_generated.append({
                            "opportunity_id": str(existing_opp.id),
                            "gap_id": str(gap_id),
                            "research_problem": existing_opp.research_problem,
                            "proposed_research_direction": existing_opp.proposed_research_direction,
                            "technical_approach": existing_opp.technical_approach,
                            "status": existing_opp.status,
                            "confidence": existing_opp.confidence,
                            "refinement_iteration_count": existing_opp.refinement_iteration_count,
                            "prior_art_statements_count": len(existing_opp.prior_art_statement_ids or []),
                            "novelty_assessment": (
                                existing_opp.novelty_assessment.get("assessment")
                                if isinstance(existing_opp.novelty_assessment, dict)
                                else str(existing_opp.novelty_assessment)
                            ),
                            "report_sections_count": 16,
                            "supporting_papers_count": len(report.supporting_papers),
                            "existing_approaches_count": len(report.existing_approaches),
                            "resumed_from_cache": True,
                        })
                        continue

                    if opp_pipeline is None:
                        vector_index = StatementVectorIndex(dimension=resolved_dim)
                        if self.temporal_cutoff:
                            from ideagen.discovery.temporal import filter_statements_by_cutoff
                            prior_art_stmts = persistence.get_all_research_statements(cutoff_date=self.temporal_cutoff)
                            prior_art_stmts = filter_statements_by_cutoff(prior_art_stmts, self.temporal_cutoff)
                        else:
                            prior_art_stmts = all_stmts

                        if prior_art_stmts:
                            stmt_texts_to_embed = [s.statement_text for s in prior_art_stmts]
                            stmt_vectors = embedder.embed_texts(stmt_texts_to_embed)
                            vector_index.add_statements(prior_art_stmts, stmt_vectors)

                        opp_pipeline = ResearchOpportunityPipeline(
                            vector_index=vector_index,
                            embedder=embedder,
                            llm_provider=llm,
                            max_refinement_iterations=self.max_refinement_iterations,
                            generation_model=resolved_llm_name,
                            validation_model=resolved_llm_name,
                        )

                    try:
                        ev_records = persistence.get_gap_evidence(gap_id)
                        stmt_ids = [e.statement_id for e in ev_records]
                        paper_ids = list({e.paper_id for e in ev_records})

                        supporting_stmts = (
                            session.query(ResearchStatement).filter(ResearchStatement.id.in_(stmt_ids)).all()
                            if stmt_ids
                            else []
                        )
                        stmt_texts = [s.statement_text for s in supporting_stmts]
                        methods = [
                            s.statement_text
                            for s in supporting_stmts
                            if s.statement_type in {"METHOD", "METHODOLOGY", "BASELINE"}
                        ]

                        input_dto = ResearchOpportunityInputDTO(
                            gap_id=gap.id,
                            gap_statement=gap.gap_statement,
                            research_problem=gap.research_problem,
                            supporting_statement_ids=stmt_ids,
                            supporting_paper_ids=paper_ids,
                            supporting_statements_text=stmt_texts,
                            recurring_limitations=gap.limitations or [],
                            future_work_statements=gap.future_work or [],
                            existing_methods=methods,
                        )

                        opp_dto = opp_pipeline.process_validated_gap(input_dto=input_dto, db_session=session)
                        report_service = OpportunityReportService(session)
                        report = report_service.generate_opportunity_report(opp_dto.id)

                        opportunities_generated.append({
                            "opportunity_id": str(opp_dto.id),
                            "gap_id": str(gap_id),
                            "research_problem": opp_dto.research_problem,
                            "proposed_research_direction": opp_dto.proposed_research_direction,
                            "technical_approach": opp_dto.technical_approach,
                            "status": opp_dto.status.value if hasattr(opp_dto.status, "value") else opp_dto.status,
                            "confidence": opp_dto.confidence,
                            "refinement_iteration_count": opp_dto.refinement_iteration_count,
                            "prior_art_statements_count": len(opp_dto.prior_art_statement_ids),
                            "novelty_assessment": (
                                opp_dto.novelty_assessment.assessment
                                if hasattr(opp_dto.novelty_assessment, "assessment")
                                else opp_dto.novelty_assessment.get("assessment")
                            ),
                            "report_sections_count": 16,
                            "supporting_papers_count": len(report.supporting_papers),
                            "existing_approaches_count": len(report.existing_approaches),
                            "resumed_from_cache": False,
                        })
                    except Exception as e:
                        logger.error(f"Opportunity generation failed for gap {gap_id}: {e}")
                        opportunity_failures.append({
                            "gap_id": str(gap_id),
                            "error": str(e),
                        })

                metrics["opportunities"] = {
                    "opportunities_count": len(opportunities_generated),
                    "failed_count": len(opportunity_failures),
                    "opportunities": opportunities_generated,
                    "failures": opportunity_failures,
                }

            metrics["timings_seconds"]["opportunity_total"] = round(time.time() - opp_start, 2)
            self._checkpoint_metrics(metrics)

            # ── STAGE 5: TEMPORAL OPPORTUNITY EVOLUTION (OPTIONAL) ─────────
            if self.enable_stage_5:
                s5_start = time.time()
                logger.info("Executing Stage 5: Temporal Opportunity Evolution...")
                try:
                    # 1. Ensure all validated opportunities are bridged to lifecycle entities
                    bridge_svc = OpportunityBridgeService(session)
                    for rec in session.query(ResearchOpportunityRecord).all():
                        try:
                            opp_entity = bridge_svc.bridge_opportunity(rec)
                            if opp_entity and self.temporal_cutoff:
                                cutoff_dt = datetime.datetime.combine(self.temporal_cutoff, datetime.time.min)
                                if opp_entity.created_at and opp_entity.created_at > cutoff_dt:
                                    opp_entity.created_at = cutoff_dt
                        except Exception as b_err:
                            logger.warning(f"Failed to bridge opportunity {rec.id} for Stage 5: {b_err}")
                    session.commit()

                    all_opportunities = session.query(Opportunity).all()
                    if not all_opportunities:
                        logger.info("Stage 5: No opportunities available to evolve.")
                        metrics["stage_5_evolution"] = {
                            "status": "NO_OPPORTUNITIES",
                            "enabled": True,
                            "batches_evaluated_count": 0,
                            "total_opportunities_evaluated": 0,
                            "total_events_count": 0,
                            "total_status_transitions_count": 0,
                            "per_batch_metrics": [],
                            "metric_interpretation_note": (
                                "Stage 5 measures descriptive temporal evolution counts (relationships, transitions, status distribution). "
                                "These values represent event volume and trajectory progression, not empirical accuracy, precision, or research "
                                "validity without human-annotated gold standard data."
                            ),
                        }
                        metrics["timings_seconds"]["stage_5_evolution_total"] = round(time.time() - s5_start, 2)
                        self._checkpoint_metrics(metrics)
                    else:
                        all_db_papers = session.query(Paper).all()
                        dated_papers = [p for p in all_db_papers if p.publication_date is not None]
                        undated_papers = [p for p in all_db_papers if p.publication_date is None]

                        # 2. Determine temporal batches
                        if self.stage_5_batches:
                            batches = sorted(self.stage_5_batches, key=lambda b: (b.sequence_number, b.start_date))
                        else:
                            batches = build_temporal_batches(
                                papers=all_db_papers,
                                baseline_cutoff=self.temporal_cutoff,
                                explicit_cutoffs=self.stage_5_cutoffs,
                                batch_window_days=self.stage_5_batch_window_days,
                            )

                        if not batches:
                            logger.info("Stage 5: No temporal batches could be formed from available papers.")
                            metrics["stage_5_evolution"] = {
                                "status": "NO_EVALUATION_BATCHES",
                                "enabled": True,
                                "batches_evaluated_count": 0,
                                "total_opportunities_evaluated": len(all_opportunities),
                                "total_events_count": 0,
                                "total_status_transitions_count": 0,
                                "per_batch_metrics": [],
                                "metric_interpretation_note": (
                                    "Stage 5 measures descriptive temporal evolution counts (relationships, transitions, status distribution). "
                                    "These values represent event volume and trajectory progression, not empirical accuracy, precision, or research "
                                    "validity without human-annotated gold standard data."
                                ),
                            }
                            metrics["timings_seconds"]["stage_5_evolution_total"] = round(time.time() - s5_start, 2)
                            self._checkpoint_metrics(metrics)
                        else:
                            # 3. Instantiate evolution services
                            classifier = RelationshipClassifier(
                                provider=llm,
                                timeout=self.stage_5_timeout or self.ollama_timeout,
                                max_retries=self.stage_5_max_retries,
                                backoff_factor=self.stage_5_backoff_factor,
                            )
                            transition_svc = OpportunityTransitionService(session)
                            integration_svc = OpportunityClassificationIntegration(classifier, transition_svc, session)
                            evolution_svc = OpportunityEvolutionService(integration_svc)

                            per_batch_metrics: List[Dict[str, Any]] = []
                            total_events = 0
                            total_transitions = 0

                            for batch in batches:
                                # Papers strictly within batch date range (future-data isolated)
                                batch_eligible = [
                                    p for p in dated_papers
                                    if batch.start_date <= p.publication_date <= batch.end_date
                                ]
                                batch_eligible.sort(key=lambda p: (p.publication_date, str(p.id)))

                                # Precise exclusions
                                undated_excluded = len(undated_papers)
                                future_excluded = sum(1 for p in dated_papers if p.publication_date > batch.end_date)
                                past_excluded = sum(1 for p in dated_papers if p.publication_date < batch.start_date)
                                total_excluded = undated_excluded + future_excluded + past_excluded

                                # Opportunities created/available at or before batch cutoff
                                eligible_opps = [
                                    opp for opp in all_opportunities
                                    if is_opportunity_available_at_cutoff(opp, batch.end_date)
                                ]
                                future_opps_skipped = len(all_opportunities) - len(eligible_opps)

                                batch_processed = 0
                                batch_skipped = 0
                                batch_rejections = 0
                                batch_transitions: List[Dict[str, Any]] = []
                                batch_rel_counts = {r.value: 0 for r in OpportunityRelationship}

                                for opp in eligible_opps:
                                    res = evolution_svc.process_batch(
                                        db=session,
                                        opportunity_id=opp.id,
                                        batch=batch,
                                        papers=batch_eligible,
                                        chunk_size=self.stage_5_chunk_size,
                                    )
                                    for ev in res.events:
                                        batch_rel_counts[ev.relationship.value] = batch_rel_counts.get(ev.relationship.value, 0) + 1
                                        total_events += 1
                                        if ev.state_changed:
                                            batch_transitions.append({
                                                "opportunity_id": str(opp.id),
                                                "paper_id": str(ev.paper_id),
                                                "publication_date": str(ev.publication_date),
                                                "previous_status": ev.previous_status.value,
                                                "new_status": ev.new_status.value,
                                                "relationship": ev.relationship.value,
                                            })
                                            total_transitions += 1
                                    batch_processed += res.processed_paper_count
                                    batch_skipped += res.skipped_paper_count
                                    batch_rejections += getattr(res, "grounding_rejections_count", 0)

                                # Lifecycle status distribution of eligible opportunities at end of batch
                                status_dist: Dict[str, int] = {}
                                for opp in eligible_opps:
                                    session.refresh(opp)
                                    status_dist[opp.status] = status_dist.get(opp.status, 0) + 1

                                batch_m = {
                                    "batch_id": batch.batch_id,
                                    "sequence_number": batch.sequence_number,
                                    "start_date": batch.start_date.isoformat(),
                                    "cutoff_date": batch.end_date.isoformat(),
                                    "eligible_paper_count": len(batch_eligible),
                                    "excluded_paper_count": total_excluded,
                                    "undated_papers_excluded_count": undated_excluded,
                                    "future_papers_excluded_count": future_excluded,
                                    "opportunities_evaluated": len(eligible_opps),
                                    "opportunities_skipped_future_count": future_opps_skipped,
                                    "processed_papers": batch_processed,
                                    "skipped_papers": batch_skipped,
                                    "relationship_counts": batch_rel_counts,
                                    "lifecycle_status_counts": status_dist,
                                    "status_transitions_count": len(batch_transitions),
                                    "status_transitions": batch_transitions,
                                    "grounding_rejections_count": batch_rejections,
                                    "failures_count": 0,
                                    "failures": [],
                                }
                                per_batch_metrics.append(batch_m)
                                self._checkpoint_metrics(metrics)

                            metrics["stage_5_evolution"] = {
                                "status": "COMPLETED",
                                "enabled": True,
                                "batches_evaluated_count": len(batches),
                                "total_opportunities_evaluated": len(all_opportunities),
                                "total_events_count": total_events,
                                "total_status_transitions_count": total_transitions,
                                "per_batch_metrics": per_batch_metrics,
                                "metric_interpretation_note": (
                                    "Stage 5 measures descriptive temporal evolution counts (relationships, transitions, status distribution). "
                                    "These values represent event volume and trajectory progression, not empirical accuracy, precision, or research "
                                    "validity without human-annotated gold standard data."
                                ),
                            }
                            metrics["timings_seconds"]["stage_5_evolution_total"] = round(time.time() - s5_start, 2)
                            self._checkpoint_metrics(metrics)

                except Exception as s5_err:
                    logger.error(f"Stage 5 temporal evolution failed: {s5_err}")
                    metrics["stage_5_evolution"] = {
                        "status": "FAILED",
                        "enabled": True,
                        "error": str(s5_err),
                        "failures": [str(s5_err)],
                        "per_batch_metrics": per_batch_metrics if "per_batch_metrics" in locals() else [],
                    }
                    metrics["timings_seconds"]["stage_5_evolution_total"] = round(time.time() - s5_start, 2)
                    failure_reason = f"Stage 5 evolution failed: {s5_err}"
                    metrics["fatal_error"] = failure_reason
                    self._checkpoint_metrics(metrics)
                    return self._finalize_result(False, metrics, session, error=failure_reason)
            else:
                metrics["stage_5_evolution"] = {
                    "status": "DISABLED",
                    "enabled": False,
                }

            metrics["timings_seconds"]["pipeline_wall_time_total"] = round(time.time() - start_time, 2)

            return self._finalize_result(True, metrics, session)

        except KeyboardInterrupt:
            interrupted = True
            metrics["metadata"]["interrupted"] = True
            metrics["fatal_error"] = "Interrupted by user (SIGINT / KeyboardInterrupt)"
            metrics["timings_seconds"]["pipeline_wall_time_total"] = round(time.time() - start_time, 2)
            if current_in_flight_paper:
                paper_statuses[current_in_flight_paper] = "INTERRUPTED"
                if "understanding" not in metrics:
                    metrics["understanding"] = {}
                failures = metrics["understanding"].get("failures", [])
                failures.append({
                    "filename": current_in_flight_paper,
                    "paper_id": str(current_in_flight_paper_id) if current_in_flight_paper_id else None,
                    "status": "INTERRUPTED",
                    "error": "Process interrupted during active LLM inference",
                })
                metrics["understanding"]["failures"] = failures
            metrics["paper_statuses"] = paper_statuses
            logger.warning("Controlled experiment interrupted by SIGINT. Preserving state and exporting partial metrics...")
            return self._finalize_result(False, metrics, session, error="Interrupted by user (SIGINT)")

        except Exception as e:
            metrics["timings_seconds"]["pipeline_wall_time_total"] = round(time.time() - start_time, 2)
            metrics["fatal_error"] = str(e)
            metrics["paper_statuses"] = paper_statuses
            return self._finalize_result(False, metrics, session, error=str(e))

        finally:
            session.close()
            engine.dispose()
            if temp_db_file and temp_db_file.exists():
                if interrupted:
                    logger.info(f"Run interrupted. Preserving temporary SQLite database at {temp_db_file}")
                else:
                    try:
                        temp_db_file.unlink(missing_ok=True)
                    except Exception as e:
                        logger.warning(f"Could not remove temporary DB {temp_db_file}: {e}")

    def _finalize_result(
        self,
        success: bool,
        metrics: Dict[str, Any],
        session: Session,
        error: Optional[str] = None,
    ) -> ExperimentResult:
        """Exports metrics and markdown summary to artifacts directory."""
        metrics_file = None
        summary_file = None

        if error and not metrics.get("fatal_error"):
            metrics["fatal_error"] = error

        if not success and not metrics.get("fatal_error"):
            metrics["fatal_error"] = error or "Experiment failed."

        if self.artifacts_dir:
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            metrics_file = self.artifacts_dir / "experiment_metrics.json"
            summary_file = self.artifacts_dir / "experiment_summary.md"

            _atomic_write_json(metrics_file, metrics)
            summary_md = self._render_summary_markdown(metrics)
            _atomic_write_text(summary_file, summary_md)

        return ExperimentResult(
            success=success,
            metrics=metrics,
            metrics_file=metrics_file,
            summary_file=summary_file,
            error=error,
        )

    def _render_summary_markdown(self, metrics: Dict[str, Any]) -> str:
        """Renders human-readable markdown experiment summary."""
        meta = metrics.get("metadata", {})
        ingest = metrics.get("ingestion", {})
        under = metrics.get("understanding", {})
        disc = metrics.get("discovery", {})
        opp = metrics.get("opportunities", {})
        timings = metrics.get("timings_seconds", {})

        status_str = "SUCCESS"
        if meta.get("interrupted"):
            status_str = "INTERRUPTED (Interrupted by user / SIGINT)"
        elif metrics.get("fatal_error"):
            status_str = f"FAILED ({metrics.get('fatal_error')})"

        passage_limits = meta.get("configuration_fingerprint", {}).get("passage_selection", {})
        budget_str = f"{passage_limits.get('max_total_words', 2500)} words (Max: {passage_limits.get('max_passages', 12)} passages)"
        timeout_val = meta.get("configuration_fingerprint", {}).get("ollama_timeout", "N/A")
        refine_val = meta.get("configuration_fingerprint", {}).get("max_refinement_iterations", "N/A")

        md = [
            "# IdeaGen Controlled Multi-Paper Experiment Summary",
            "",
            f"- **Status**: {status_str}",
            f"- **Timestamp**: {meta.get('timestamp', 'N/A')}",
            f"- **LLM Model**: `{meta.get('llm_model', 'N/A')}`",
            f"- **Embedding Model**: `{meta.get('embedding_model', 'N/A')}` ({meta.get('embedding_dimension', 'N/A')}-d)",
            f"- **Passage Budget**: `{budget_str}`",
            f"- **Ollama Timeout**: `{timeout_val}s`",
            f"- **Max Refinement Iterations**: `{refine_val}`",
            f"- **Total Wall Time**: `{timings.get('pipeline_wall_time_total', 'N/A')}s`",
            f"- **Database Isolation**: `{meta.get('database_mode', 'N/A')}` (PostgreSQL disconnected: `True`)",
        ]

        if meta.get("corpus_preflight"):
            pf = meta["corpus_preflight"]
            dr = pf.get("date_range", {})
            md.extend([
                "",
                "## 0. Corpus Preflight Summary",
                f"- Total Papers: **{pf.get('total_papers', 0)}** ({pf.get('dated_papers_count', 0)} dated, {pf.get('undated_papers_count', 0)} undated, {pf.get('malformed_dates_count', 0)} malformed)",
                f"- Date Range: `{dr.get('earliest', 'N/A')}` to `{dr.get('latest', 'N/A')}`",
                f"- Temporal Impact: *{pf.get('temporal_evaluation_impact', '')}*",
            ])

        md.extend([
            "",
            "## 1. Ingestion Summary",
            f"- Papers Ingested: **{ingest.get('ingested_count', 0)}** (Reused: {ingest.get('reused_count', 0)}, Failed: {ingest.get('failed_count', 0)})",
            f"- Total Passages Created: **{ingest.get('total_passages_created', 0)}**",
            f"- Ingestion Wall Time: **{timings.get('ingestion_total', 'N/A')}s**",
            "",
            "## 2. Paper Understanding",
            f"- Papers Processed: **{under.get('understood_count', 0)}** (Resumed: {under.get('resumed_count', 0)})",
            f"- Statements Extracted: **{under.get('total_statements_extracted', 0)}**",
        ])
        if "eligible_statements_extracted" in under:
            md.append(f"- Statements Eligible for Cutoff ({under.get('temporal_cutoff')}): **{under.get('eligible_statements_extracted')}**")
        md.extend([
            f"- Provenance Validity Rate: **{round(under.get('overall_provenance_validity_rate', 0.0) * 100, 1)}%**",
            "- Statements by Type:",
        ])

        for st_type, count in under.get("corpus_statements_by_type", {}).items():
            md.append(f"  - `{st_type}`: {count}")

        md.extend([
            f"- Understanding Wall Time: **{timings.get('understanding_total', 'N/A')}s**",
            "",
            "## 3. Semantic Gap Discovery",
            f"- Candidate Gaps Analyzed: **{disc.get('candidate_gaps_total', 0)}**",
            f"- Validated Gaps: **{disc.get('validated_gaps_count', 0)}**",
            f"- Rejected Gaps: **{disc.get('rejected_gaps_count', 0)}**",
            f"- Discovery Wall Time: **{timings.get('discovery_total', 'N/A')}s**",
            "",
            "## 4. Research Opportunity Generation",
            f"- Opportunities Generated: **{opp.get('opportunities_count', 0)}**",
            f"- Opportunity Wall Time: **{timings.get('opportunity_total', 'N/A')}s**",
        ])

        if opp.get("opportunities"):
            md.append("")
            md.append("### Synthesized Opportunities:")
            for o in opp["opportunities"]:
                md.append(f"- **ID**: `{o.get('opportunity_id')}`")
                md.append(f"  - Direction: *{o.get('proposed_research_direction')}*")
                md.append(f"  - Status: `{o.get('status')}` (Confidence: {o.get('confidence')})")
                md.append(f"  - Prior Art Statements: {o.get('prior_art_statements_count')}")
                md.append(f"  - Novelty: `{o.get('novelty_assessment')}`")

        s5 = metrics.get("stage_5_evolution", {})
        if s5.get("enabled"):
            md.extend([
                "",
                "## 5. Temporal Opportunity Evolution (Stage 5)",
                f"- Status: **{s5.get('status', 'N/A')}**",
                f"- Batches Evaluated: **{s5.get('batches_evaluated_count', 0)}**",
                f"- Opportunities Evaluated: **{s5.get('total_opportunities_evaluated', 0)}**",
                f"- Evolution Events: **{s5.get('total_events_count', 0)}**",
                f"- Status Transitions: **{s5.get('total_status_transitions_count', 0)}**",
                f"- Evolution Wall Time: **{timings.get('stage_5_evolution_total', 'N/A')}s**",
            ])
            if s5.get("per_batch_metrics"):
                md.append("")
                md.append("### Batch Breakdown:")
                for b in s5["per_batch_metrics"]:
                    md.append(
                        f"- Batch `{b.get('batch_id')}` (Cutoff: {b.get('cutoff_date')}): "
                        f"{b.get('eligible_paper_count')} eligible papers, "
                        f"{b.get('opportunities_evaluated')} opps evaluated, "
                        f"{b.get('status_transitions_count')} transitions, "
                        f"{b.get('grounding_rejections_count', 0)} grounding rejections"
                    )

        md.append("")
        return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="IdeaGen Controlled Multi-Paper Experiment Runner")
    parser.add_argument("--input-dir", required=True, type=str, help="Directory containing scientific PDF files")
    parser.add_argument("--artifacts-dir", default=None, type=str, help="Directory to export metrics JSON and summary MD")
    parser.add_argument("--db-path", default=None, type=str, help="Path for SQLite database (or ':memory:')")
    parser.add_argument("--min-evidence", default=1, type=int, help="Minimum supporting evidence count for gap validation")
    parser.add_argument("--retry-failed", action="store_true", default=True, help="Retry previously failed paper understanding on resumption")
    parser.add_argument("--no-retry-failed", dest="retry_failed", action="store_false", help="Do not retry previously failed paper understanding")
    parser.add_argument("--max-passages", default=12, type=int, help="Maximum passages for paper understanding")
    parser.add_argument("--max-total-words", default=2500, type=int, help="Word budget for passage selection")
    parser.add_argument("--similarity-threshold", default=0.65, type=float, help="Similarity threshold for gap clustering")
    parser.add_argument("--ollama-timeout", default=None, type=float, help="Ollama request timeout in seconds")
    parser.add_argument("--max-refinement-iterations", default=2, type=int, help="Maximum refinement iterations for opportunity generation")
    parser.add_argument("--cutoff-date", default=None, type=str, help="Historical evaluation cutoff date (YYYY-MM-DD)")
    parser.add_argument("--enable-stage-5", action="store_true", default=False, help="Enable Stage 5 temporal opportunity evolution evaluation")
    parser.add_argument("--stage-5-batch-window-days", default=None, type=int, help="Window size in days for Stage 5 temporal batches")
    parser.add_argument("--stage-5-cutoffs", default=None, type=str, nargs="+", help="Explicit chronological cutoff dates for Stage 5 (YYYY-MM-DD ...)")
    parser.add_argument("--stage-5-chunk-size", default=None, type=int, help="Chunk size for Stage 5 batch processing")
    parser.add_argument("--stage-5-max-retries", default=2, type=int, help="Maximum retries for Stage 5 LLM relationship classification")
    parser.add_argument("--stage-5-backoff-factor", default=0.5, type=float, help="Exponential backoff factor for Stage 5 retries")
    parser.add_argument("--stage-5-timeout", default=None, type=float, help="Per-call timeout for Stage 5 relationship classification")
    parser.add_argument("--preflight-only", action="store_true", default=False, help="Run offline corpus preflight report and exit")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if args.preflight_only:
        report = run_corpus_preflight(args.input_dir)
        print(report.summary())
        sys.exit(0)

    runner = ControlledExperimentRunner(
        db_path=args.db_path,
        artifacts_dir=args.artifacts_dir,
        min_supporting_evidence=args.min_evidence,
        retry_failed=args.retry_failed,
        max_passages=args.max_passages,
        max_total_words=args.max_total_words,
        similarity_threshold=args.similarity_threshold,
        ollama_timeout=args.ollama_timeout,
        max_refinement_iterations=args.max_refinement_iterations,
        temporal_cutoff=args.cutoff_date,
        enable_stage_5=args.enable_stage_5,
        stage_5_batch_window_days=args.stage_5_batch_window_days,
        stage_5_cutoffs=args.stage_5_cutoffs,
        stage_5_max_retries=args.stage_5_max_retries,
        stage_5_backoff_factor=args.stage_5_backoff_factor,
        stage_5_timeout=args.stage_5_timeout,
        stage_5_chunk_size=args.stage_5_chunk_size,
    )

    result = runner.run(input_dir=args.input_dir)
    if result.success:
        print("\n=== EXPERIMENT COMPLETED SUCCESSFULLY ===")
        if result.summary_file:
            print(f"Summary written to: {result.summary_file}")
        if result.metrics_file:
            print(f"Metrics written to: {result.metrics_file}")
    else:
        print(f"\n=== EXPERIMENT FAILED: {result.error} ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
