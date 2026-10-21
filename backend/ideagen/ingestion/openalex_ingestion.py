"""OpenAlex Paper Ingestion Service with Open-Access PDF support.

Coordinates:
- OpenAlex work metadata parsing and canonical ID normalization.
- Duplicate detection by source ID, DOI, and document content hash.
- Open-access PDF download via OpenAccessPDFDownloader.
- Text extraction, section detection, and passage generation via ManualPDFIngestionService.
- Metadata-only 'NO_PDF' persistence fallback when PDF is missing, inaccessible, or invalid.
- Prevention of partial database records upon failure.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Union
from uuid import UUID
from pydantic import BaseModel

from ideagen.ingestion.models import PaperCreateDTO, PaperDocumentCreateDTO, SourceType
from ideagen.ingestion.parser import OpenAlexResponseParser
from ideagen.ingestion.pipeline import (
    PersistenceLayer,
    DefaultDuplicateDetector,
    DuplicateDetectionInterface,
)
from ideagen.ingestion.pdf import (
    ManualPDFIngestionService,
    DuplicateDocumentError,
    PDFExtractionError,
)
from ideagen.ingestion.downloader import (
    OpenAccessPDFDownloader,
    PDFDownloadError,
    PDFUnavailableError,
    OversizedPDFError,
    InvalidPDFSignatureError,
)

logger = logging.getLogger(__name__)


class IngestionStatus:
    INGESTED = "INGESTED"
    NO_PDF = "NO_PDF"
    SKIPPED_DUPLICATE = "SKIPPED_DUPLICATE"
    ERROR = "ERROR"


class EvolutionHookStatus:
    DISABLED = "DISABLED"
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class PaperIngestionResult(BaseModel):
    paper_id: Optional[UUID] = None
    source_id: str
    status: str  # "INGESTED", "NO_PDF", "SKIPPED_DUPLICATE", "ERROR"
    has_pdf: bool
    document_id: Optional[UUID] = None
    passage_count: int = 0
    error_message: Optional[str] = None
    batch_id: str
    doi: Optional[str] = None
    evolution_status: Optional[str] = None  # "DISABLED", "SUCCESS", "PARTIAL_SUCCESS", "FAILED", "SKIPPED"
    evolution_result: Optional[Any] = None


class OpenAlexPaperIngestionService:
    """Production ingestion service for OpenAlex research papers."""

    def __init__(
        self,
        persistence: PersistenceLayer,
        downloader: Optional[OpenAccessPDFDownloader] = None,
        pdf_service: Optional[ManualPDFIngestionService] = None,
        duplicate_detector: Optional[DuplicateDetectionInterface] = None,
        auto_evolve: Optional[bool] = None,
        evolution_service: Optional[Any] = None,
    ):
        self.persistence = persistence
        self.downloader = downloader or OpenAccessPDFDownloader()
        self.pdf_service = pdf_service or ManualPDFIngestionService(persistence=persistence)
        self.duplicate_detector = duplicate_detector or DefaultDuplicateDetector()

        if auto_evolve is not None:
            self.auto_evolve = bool(auto_evolve)
        else:
            from ideagen.core.config import get_settings
            self.auto_evolve = bool(get_settings().enable_post_ingestion_evolution)

        self.evolution_service = evolution_service

    def ingest_work(
        self,
        work: Dict[str, Any],
        check_duplicates: bool = True,
        allow_no_pdf: bool = True,
    ) -> PaperIngestionResult:
        """Ingests an OpenAlex work record.

        Downloads and chunks open-access PDF if accessible; otherwise saves paper
        as a metadata-only NO_PDF record.
        """
        # 1. Parse metadata & canonicalize IDs
        paper_dto = OpenAlexResponseParser.parse_work(work, source_type=SourceType.OPENALEX_API)

        # 2. Check for duplicate paper
        if check_duplicates and self.duplicate_detector.is_duplicate(paper_dto, self.persistence):
            logger.info(f"Skipping duplicate paper {paper_dto.source_id}")
            existing_id = None
            if hasattr(self.duplicate_detector, "resolve_duplicate_paper_id"):
                existing_id = self.duplicate_detector.resolve_duplicate_paper_id(paper_dto, self.persistence)
            return PaperIngestionResult(
                paper_id=existing_id,
                source_id=paper_dto.source_id,
                status=IngestionStatus.SKIPPED_DUPLICATE,
                has_pdf=False,
                batch_id=paper_dto.batch_id,
                doi=paper_dto.doi,
                evolution_status=EvolutionHookStatus.SKIPPED if self.auto_evolve else EvolutionHookStatus.DISABLED,
            )

        # 3. Resolve PDF URL
        pdf_url = self.downloader.extract_pdf_url(work)

        # If no PDF URL available, save as NO_PDF metadata-only
        if not pdf_url:
            return self._save_no_pdf(paper_dto, reason="NO_PDF_URL")

        # 4. Attempt download and structured text processing
        try:
            with self.downloader.download_pdf(pdf_url) as temp_pdf_path:
                # Content-hash duplicate check
                if check_duplicates and self.pdf_service.is_content_duplicate(temp_pdf_path):
                    content_hash = self.pdf_service.compute_sha256(temp_pdf_path)
                    raise DuplicateDocumentError(
                        f"Duplicate document detected (content hash {content_hash} already exists)"
                    )

                # Extract structured representation BEFORE persisting paper
                # Guarantees no partial paper is saved if extraction fails
                extraction_result, block_units = self.pdf_service.extract_structured(temp_pdf_path)

                # Update paper metadata
                paper_dto.raw_metadata["pdf_status"] = "ACCESSIBLE"
                paper_dto.raw_metadata["has_pdf"] = True
                paper_dto.raw_metadata["pdf_url"] = pdf_url
                paper_dto.raw_metadata["content_hash"] = extraction_result.content_hash

                if not paper_dto.abstract:
                    for sec in extraction_result.sections:
                        if sec.name == "Abstract":
                            paper_dto.abstract = sec.text
                            break
                    if not paper_dto.abstract:
                        paper_dto.abstract = extraction_result.full_text[:500]

                # Persist Paper, PaperDocument & Passages atomically within a scoped transaction
                paper_id = None
                try:
                    tx_ctx = self.persistence.transaction() if hasattr(self.persistence, "transaction") else nullcontext()
                    with tx_ctx:
                        paper_id = self.persistence.save_paper(paper_dto)

                        now_ts = datetime.now(timezone.utc)
                        doc_dto = PaperDocumentCreateDTO(
                            paper_id=paper_id,
                            full_text=extraction_result.full_text,
                            extraction_method=self.pdf_service.extraction_method,
                            extraction_version=self.pdf_service.extraction_version,
                            extracted_at=extraction_result.extracted_at,
                            content_hash=extraction_result.content_hash,
                            original_filename=temp_pdf_path.name,
                            acquisition_timestamp=now_ts,
                            provenance_metadata={
                                "source_type": SourceType.OPENALEX_API.value,
                                "source_id": paper_dto.source_id,
                                "content_hash": extraction_result.content_hash,
                                "pdf_url": pdf_url,
                                "num_pages": len(extraction_result.pages),
                                "num_sections": len(extraction_result.sections),
                            },
                        )
                        saved_doc_id = self.persistence.save_paper_document(doc_dto)
                        doc_id = saved_doc_id or getattr(doc_dto, "id", None)

                        # Create and persist deterministic passages
                        passages = self.pdf_service.chunker.create_passages(
                            blocks=block_units,
                            paper_id=paper_id,
                            document_id=doc_id,
                        )
                        if hasattr(self.persistence, "save_passages"):
                            self.persistence.save_passages(passages)

                except Exception as persistence_err:
                    logger.error(
                        f"Failed persisting paper/document/passages for {paper_dto.source_id}: {persistence_err}."
                    )
                    # If persistence layer lacked transaction support, attempt fallback compensating deletion
                    rolled_back = hasattr(self.persistence, "transaction")
                    if not rolled_back and paper_id and hasattr(self.persistence, "delete_paper"):
                        try:
                            rolled_back = self.persistence.delete_paper(paper_id)
                        except Exception as rb_err:
                            logger.error(f"Failed fallback rollback of paper {paper_id}: {rb_err}")

                    # Do NOT call save_paper again; report persistence failure distinctly as ERROR
                    return PaperIngestionResult(
                        paper_id=None if rolled_back else paper_id,
                        source_id=paper_dto.source_id,
                        status=IngestionStatus.ERROR,
                        has_pdf=True,
                        error_message=f"Persistence error: {persistence_err}",
                        batch_id=paper_dto.batch_id,
                        doi=paper_dto.doi,
                    )

                ingestion_res = PaperIngestionResult(
                    paper_id=paper_id,
                    source_id=paper_dto.source_id,
                    status=IngestionStatus.INGESTED,
                    has_pdf=True,
                    document_id=doc_id,
                    passage_count=len(passages),
                    batch_id=paper_dto.batch_id,
                    doi=paper_dto.doi,
                )

                # Post-ingestion evolution hook runs after persistence transaction is committed
                if self.auto_evolve:
                    ingestion_res = self._run_evolution_hook(ingestion_res, paper_id)
                else:
                    ingestion_res.evolution_status = EvolutionHookStatus.DISABLED

                return ingestion_res

        except DuplicateDocumentError:
            raise

        except (PDFDownloadError, PDFExtractionError, Exception) as e:
            if not allow_no_pdf:
                raise
            logger.warning(
                f"Failed to acquire/extract PDF for {paper_dto.source_id}: {e}. Falling back to NO_PDF."
            )
            return self._save_no_pdf(paper_dto, reason=str(e))

    def _save_no_pdf(self, paper_dto: PaperCreateDTO, reason: str) -> PaperIngestionResult:
        """Persists paper metadata-only as NO_PDF record without document or passages."""
        paper_dto.raw_metadata["pdf_status"] = "NO_PDF"
        paper_dto.raw_metadata["pdf_error"] = reason
        paper_dto.raw_metadata["has_pdf"] = False

        paper_id = self.persistence.save_paper(paper_dto)

        res = PaperIngestionResult(
            paper_id=paper_id,
            source_id=paper_dto.source_id,
            status=IngestionStatus.NO_PDF,
            has_pdf=False,
            error_message=reason,
            batch_id=paper_dto.batch_id,
            doi=paper_dto.doi,
            evolution_status=EvolutionHookStatus.SKIPPED if self.auto_evolve else EvolutionHookStatus.DISABLED,
        )
        if self.auto_evolve:
            logger.info(
                f"Post-ingestion evolution skipped for paper {paper_id}: "
                f"NO_PDF metadata-only record has no full-text document or passages."
            )
        return res

    def _get_or_create_evolution_service(self) -> Optional[Any]:
        """Resolves or lazily constructs the IncrementalOpportunityEvolutionService.

        Uses injected self.evolution_service if provided.
        Otherwise, if self.persistence provides an active SQLAlchemy Session,
        constructs the integration, transition service, and evolution service explicitly.
        """
        if self.evolution_service is not None:
            return self.evolution_service

        session = getattr(self.persistence, "session", None)
        if session is None:
            logger.warning(
                "Cannot construct IncrementalOpportunityEvolutionService: "
                "persistence layer does not expose a SQLAlchemy session and no evolution_service was injected."
            )
            return None

        from ideagen.llm import get_default_llm_provider
        from ideagen.opportunity.classifier import RelationshipClassifier
        from ideagen.opportunity.service import OpportunityTransitionService
        from ideagen.opportunity.integration import OpportunityClassificationIntegration
        from ideagen.opportunity.incremental import IncrementalOpportunityEvolutionService

        provider = get_default_llm_provider()
        classifier = RelationshipClassifier(provider=provider)
        trans_svc = OpportunityTransitionService(session=session)
        integration = OpportunityClassificationIntegration(
            classifier=classifier,
            transition_service=trans_svc,
            session=session,
        )
        from ideagen.core.config import get_settings
        self.evolution_service = IncrementalOpportunityEvolutionService(
            integration=integration,
            session=session,
            max_candidates=get_settings().post_ingestion_evolution_max_candidates,
        )
        return self.evolution_service

    def _is_caller_owned_transaction(self, session: Optional[Any] = None) -> bool:
        """Determines if the persistence layer or session is currently participating in an outer caller-owned transaction."""
        if getattr(self.persistence, "_in_transaction", False):
            return True
        if session is not None and getattr(session, "in_nested_transaction", lambda: False)():
            return True
        return False

    def _run_evolution_hook(
        self,
        ingestion_result: PaperIngestionResult,
        paper_id: UUID,
    ) -> PaperIngestionResult:
        """Executes the post-ingestion evolution hook in an isolated error boundary.

        Guarantees:
        - Ingested paper, document, and passages are already committed before evolution begins.
        - In standalone mode (no caller-owned outer transaction), successful evolution is explicitly
          committed to the database, ensuring durability across fresh sessions and connection pools.
        - In standalone mode, evolution failure or exception rolls back only uncommitted evolution state,
          leaving the successfully committed paper, document, and passages intact.
        - When a caller owns the outer transaction, evolution does not issue standalone commits or rollbacks,
          allowing the caller's transaction context to decide atomicity.
        - Any error in evolution (LLM outage, schema error, commit failure) is caught and reported as FAILED
          without rolling back or deleting the committed paper records.
        """
        from ideagen.opportunity.incremental import IncrementalEvolutionRunStatus

        session = getattr(self.persistence, "session", None)
        is_caller_owned = self._is_caller_owned_transaction(session)

        try:
            evolution_svc = self._get_or_create_evolution_service()
            if evolution_svc is None:
                logger.warning(
                    f"Post-ingestion evolution skipped for paper {paper_id}: "
                    f"evolution service could not be initialized."
                )
                ingestion_result.evolution_status = EvolutionHookStatus.SKIPPED
                return ingestion_result

            # Run incremental opportunity evolution
            evo_res = evolution_svc.evaluate_paper(paper_id=paper_id)

            # Map IncrementalEvolutionRunStatus to EvolutionHookStatus
            if evo_res.run_status == IncrementalEvolutionRunStatus.PAPER_NOT_FOUND:
                ingestion_result.evolution_status = EvolutionHookStatus.FAILED
                logger.error(
                    f"Post-ingestion evolution failed for paper {paper_id}: "
                    f"paper record not found in database by evolution service."
                )
            elif evo_res.run_status == IncrementalEvolutionRunStatus.SUCCESS:
                if getattr(evo_res, "evaluated_count", 0) > 0:
                    ingestion_result.evolution_status = EvolutionHookStatus.SUCCESS
                else:
                    ingestion_result.evolution_status = EvolutionHookStatus.SKIPPED
            elif evo_res.run_status == IncrementalEvolutionRunStatus.PARTIAL_SUCCESS:
                ingestion_result.evolution_status = EvolutionHookStatus.PARTIAL_SUCCESS
            elif evo_res.run_status in (
                IncrementalEvolutionRunStatus.SKIPPED_UNDATED,
                IncrementalEvolutionRunStatus.SKIPPED_INSUFFICIENT_CONTENT,
                IncrementalEvolutionRunStatus.NO_CANDIDATES,
            ):
                ingestion_result.evolution_status = EvolutionHookStatus.SKIPPED
            elif evo_res.run_status == IncrementalEvolutionRunStatus.FAILED:
                ingestion_result.evolution_status = EvolutionHookStatus.FAILED
            else:
                ingestion_result.evolution_status = EvolutionHookStatus.SKIPPED

            ingestion_result.evolution_result = evo_res

            # In standalone mode, commit or roll back the evolution transaction explicitly
            if not is_caller_owned and session is not None:
                if ingestion_result.evolution_status in (
                    EvolutionHookStatus.SUCCESS,
                    EvolutionHookStatus.PARTIAL_SUCCESS,
                ):
                    try:
                        session.commit()
                    except Exception as commit_err:
                        logger.error(
                            f"Failed to commit post-ingestion evolution for paper {paper_id}: {commit_err}",
                            exc_info=True,
                        )
                        try:
                            session.rollback()
                        except Exception as rb_err:
                            logger.error(f"Rollback failed following commit error for paper {paper_id}: {rb_err}")
                        ingestion_result.evolution_status = EvolutionHookStatus.FAILED
                        ingestion_result.evolution_result = {
                            "error": f"Evolution commit failed: {type(commit_err).__name__}: {commit_err}",
                            "partial_result": evo_res,
                        }
                elif ingestion_result.evolution_status == EvolutionHookStatus.FAILED:
                    try:
                        session.rollback()
                    except Exception as rb_err:
                        logger.error(f"Rollback failed following evolution failure for paper {paper_id}: {rb_err}")

            logger.info(
                f"Post-ingestion evolution completed for paper {paper_id}: "
                f"status={ingestion_result.evolution_status}, "
                f"evaluated={getattr(evo_res, 'evaluated_count', 0)}, "
                f"skipped={getattr(evo_res, 'skipped_count', 0)}, "
                f"grounding_rejected={getattr(evo_res, 'grounding_rejected_count', 0)}, "
                f"failed={getattr(evo_res, 'failed_count', 0)}."
            )

        except Exception as evo_err:
            logger.error(
                f"Post-ingestion evolution failed for paper {paper_id}: {type(evo_err).__name__}: {evo_err}",
                exc_info=True,
            )
            # In standalone mode, roll back only the uncommitted evolution session state
            if not is_caller_owned and session is not None:
                try:
                    session.rollback()
                except Exception as rb_err:
                    logger.error(f"Rollback failed following unhandled evolution exception for paper {paper_id}: {rb_err}")

            ingestion_result.evolution_status = EvolutionHookStatus.FAILED
            ingestion_result.evolution_result = {
                "error": f"{type(evo_err).__name__}: {evo_err}",
            }

        return ingestion_result
