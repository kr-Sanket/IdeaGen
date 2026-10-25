import json
import logging
import threading
import uuid
import copy
from datetime import datetime, timezone, date
from typing import List, Dict, Any, Optional

from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified
import sqlalchemy

from ideagen.db.session import get_session_factory
from ideagen.db.models import (
    Paper,
    PaperPassage,
    ResearchKnowledge,
    ResearchStatement,
    ResearchProjectRecord,
    ProjectPaperSelectionRecord,
    ResearchRunRecord,
    RunPaperRecord,
    ProjectPaperRecord,
)
from ideagen.db.persistence import SQLAlchemyPersistenceLayer
from ideagen.ingestion.openalex_ingestion import OpenAlexPaperIngestionService
from ideagen.ingestion.search import OpenAlexSearchProvider, OpenAlexSearchError
from ideagen.ingestion.parser import PaperNormalizer
from ideagen.ingestion.pdf import DuplicateDocumentError
from ideagen.ingestion.models import SourceType
from ideagen.understanding.pipeline import PaperUnderstandingPipeline
from ideagen.discovery.pipeline import SemanticGapDiscoveryPipeline
from ideagen.core.config import get_settings

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Stages a research run may execute, in execution order. Opportunity generation is deliberately
# absent: it only happens when a researcher explicitly triggers it for a VALIDATED gap
# (POST /api/v1/gaps/{gap_id}/opportunity) after reviewing the run's discovered gaps.
RUN_STAGES = ("INGESTION", "UNDERSTANDING", "GAP_DISCOVERY")
RETRYABLE_RUN_STATUSES = {"FAILED", "PARTIAL_SUCCESS", "INTERRUPTED", "PENDING"}
# A PENDING run not executing in this process for longer than this was orphaned (e.g. a crash
# between creating the run and its worker thread starting).
PENDING_RECOVERY_GRACE_SECONDS = 30
_FINISHED_STAGE_STATUSES = {"COMPLETED", "SKIPPED"}


def normalize_planned_stages(planned_stages: Optional[List[str]]) -> List[str]:
    """Validates requested stages and returns them in canonical execution order."""
    if not planned_stages:
        return list(RUN_STAGES)
    requested = [str(s).strip().upper() for s in planned_stages]
    if "OPPORTUNITY_GENERATION" in requested:
        raise ValueError(
            "OPPORTUNITY_GENERATION cannot be part of a research run. Review the run's gaps and "
            "generate an opportunity explicitly from a VALIDATED gap."
        )
    unknown = sorted(set(requested) - set(RUN_STAGES))
    if unknown:
        raise ValueError(f"Unknown run stage(s): {', '.join(unknown)}. Supported stages: {', '.join(RUN_STAGES)}.")
    return [s for s in RUN_STAGES if s in requested]


class ResearchWorkflowEngine:
    """Manages the complete paper-selection-to-research-results workflow.
    
    Provides:
    - Preflight evaluation (duplicate detection, PDF availability, planned stages, warnings)
    - Persistent run records and per-paper execution state
    - Sequential stage execution with iterative per-paper commits
    - Paper-level fault tolerance (failures do not halt unrelated papers)
    - Idempotent retries and recovery from interruption
    - Auditable technical execution logs
    """

    _active_runs: set = set()
    _runs_lock: threading.Lock = threading.Lock()

    @classmethod
    def is_run_active(cls, run_id: str) -> bool:
        """Checks if a run is currently being actively executed in-process."""
        with cls._runs_lock:
            return run_id in cls._active_runs

    @classmethod
    def mark_run_active(cls, run_id: str) -> bool:
        """Attempts to register run_id as active. Returns False if already running."""
        with cls._runs_lock:
            if run_id in cls._active_runs:
                return False
            cls._active_runs.add(run_id)
            return True

    @classmethod
    def mark_run_inactive(cls, run_id: str) -> None:
        """Unregisters run_id from active execution."""
        with cls._runs_lock:
            cls._active_runs.discard(run_id)

    @classmethod
    def recover_interrupted_runs(cls, db: Session) -> int:
        """Finds runs marked RUNNING (or PENDING beyond a short grace period) that are not active in
        memory (e.g. after a process crash or restart) and transitions them to INTERRUPTED so they can
        be safely retried.
        """
        with cls._runs_lock:
            active_ids = set(cls._active_runs)

        candidates = db.query(ResearchRunRecord).filter(ResearchRunRecord.status.in_(["RUNNING", "PENDING"])).all()
        now = datetime.now(timezone.utc)
        recovered = 0
        for r in candidates:
            if r.id in active_ids:
                continue
            if r.status == "PENDING":
                updated = r.updated_at or r.created_at
                if updated is not None and updated.tzinfo is None:
                    updated = updated.replace(tzinfo=timezone.utc)
                if updated is not None and (now - updated).total_seconds() < PENDING_RECOVERY_GRACE_SECONDS:
                    continue
            r.status = "INTERRUPTED"
            r.updated_at = datetime.now(timezone.utc)
            stage_states = copy.deepcopy(r.stage_states) if r.stage_states else {}
            if r.current_stage and r.current_stage in stage_states:
                if stage_states[r.current_stage].get("status") in {"RUNNING", "PENDING"}:
                    stage_states[r.current_stage]["status"] = "INTERRUPTED"
                    stage_states[r.current_stage]["completed_at"] = _utc_now_iso()
            r.stage_states = stage_states
            flag_modified(r, "stage_states")
            for p in r.papers:
                if p.status in {"INGESTING", "EXTRACTING_STATEMENTS"}:
                    p.status = "INTERRUPTED"
                    p.updated_at = datetime.now(timezone.utc)
            cls._append_log(
                r,
                "WARNING",
                "RECOVERY",
                f"Run {r.id} was not executing (system restart or crash) during stage {r.current_stage}. Marked INTERRUPTED and ready for retry."
            )
            recovered += 1

        if recovered > 0:
            db.commit()
            logger.info(f"Recovered {recovered} interrupted research run(s).")
        return recovered

    def __init__(self, db: Optional[Session] = None):
        self.db = db

    @staticmethod
    def _llm_label() -> str:
        settings = get_settings()
        provider = (settings.llm_provider or "ollama").lower()
        if provider == "ollama":
            return f"Ollama {settings.ollama_model}"
        if provider == "gemini":
            return f"Gemini {settings.gemini_model_name}"
        return provider

    @staticmethod
    def _corpus_summary(db: Session, project_id: str) -> Dict[str, int]:
        """Counts the project's corpus papers by what a run can do with them."""
        passage_counts = dict(
            db.query(PaperPassage.paper_id, sqlalchemy.func.count(PaperPassage.id))
            .join(ProjectPaperRecord, ProjectPaperRecord.paper_id == PaperPassage.paper_id)
            .filter(ProjectPaperRecord.project_id == project_id)
            .group_by(PaperPassage.paper_id)
            .all()
        )
        extracted = {
            row[0]
            for row in db.query(ResearchKnowledge.paper_id)
            .join(ProjectPaperRecord, ProjectPaperRecord.paper_id == ResearchKnowledge.paper_id)
            .filter(ProjectPaperRecord.project_id == project_id, ResearchKnowledge.extraction_status == "SUCCESS")
            .all()
        }
        paper_ids = ResearchWorkflowEngine._project_paper_ids(db, project_id)
        with_passages = {pid for pid in paper_ids if passage_counts.get(pid, 0) > 0}
        return {
            "total": len(paper_ids),
            "with_passages": len(with_passages),
            "statements_extracted": len(with_passages & extracted),
            "pending_extraction": len(with_passages - extracted),
            "metadata_only": len(paper_ids) - len(with_passages),
        }

    @staticmethod
    def _project_paper_ids(db: Session, project_id: str) -> set:
        return {
            row[0]
            for row in db.query(ProjectPaperRecord.paper_id).filter(ProjectPaperRecord.project_id == project_id).all()
        }

    @staticmethod
    def _normalize_paper_ids(raw_id: Any) -> tuple[Optional[str], Optional[uuid.UUID]]:
        """Normalizes an ingested paper ID into canonical (str_id, uuid_obj).

        Returns:
            (str_id, uuid_obj):
                - str_id: String representation suitable for RunPaperRecord.paper_id.
                - uuid_obj: uuid.UUID representation suitable for ProjectPaperRecord.paper_id,
                  or None if raw_id is None or cannot be parsed as a valid UUID.
        """
        if raw_id is None:
            return None, None
        if isinstance(raw_id, uuid.UUID):
            return str(raw_id), raw_id
        if isinstance(raw_id, str):
            cleaned = raw_id.strip()
            if not cleaned:
                return None, None
            try:
                parsed = uuid.UUID(cleaned)
                return str(parsed), parsed
            except (ValueError, AttributeError, TypeError):
                return cleaned, None
        try:
            parsed = uuid.UUID(str(raw_id))
            return str(parsed), parsed
        except (ValueError, AttributeError, TypeError):
            return str(raw_id), None

    def preflight_check(
        self,
        project_id: str,
        selections: Optional[List[Dict[str, Any]]] = None,
        db: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """Performs deterministic preflight analysis on a list of candidate papers."""
        session = db or getattr(self, "db", None)
        if session is None:
            raise ValueError("Database session must be provided")

        if selections is None:
            sel_records = session.query(ProjectPaperSelectionRecord).filter(
                ProjectPaperSelectionRecord.project_id == project_id
            ).all()
            selections = [
                {
                    "openalex_id": s.openalex_id,
                    "doi": s.doi,
                    "title": s.title,
                    "authors": s.authors or [],
                    "publication_date": s.publication_date,
                    "pdf_url": s.pdf_url,
                    "abstract": s.abstract,
                }
                for s in sel_records
            ]

        project = session.query(ResearchProjectRecord).filter(ResearchProjectRecord.id == project_id).first()
        cutoff = project.temporal_cutoff if project else None

        items = []
        new_count = 0
        corpus_count = 0
        pdf_count = 0
        meta_only_count = 0
        warnings = []

        for p in selections:
            raw_oa_id = (p.get("openalex_id") or "").strip()
            raw_doi = (p.get("doi") or "").strip()
            title = p.get("title") or "Untitled Paper"
            pdf_url = p.get("pdf_url")
            pub_date = p.get("publication_date")
            authors = p.get("authors") or []
            pub_year = p.get("publication_year")

            norm_oa_id = PaperNormalizer.normalize_openalex_id(raw_oa_id) if raw_oa_id else None
            norm_doi = PaperNormalizer.normalize_doi(raw_doi) if raw_doi else None

            # Check if paper already exists in corpus by OpenAlex ID or DOI
            existing_paper = None
            if norm_oa_id:
                existing_paper = session.query(Paper).filter(Paper.source_id == norm_oa_id).first()
            if not existing_paper and norm_doi:
                existing_paper = session.query(Paper).filter(Paper.doi == norm_doi).first()

            in_corpus = existing_paper is not None
            has_pdf = bool(pdf_url)

            if in_corpus:
                corpus_count += 1
                paper_status = "CORPUS_MATCH"
                passage_count = session.query(sqlalchemy.func.count(PaperPassage.id)).filter(
                    PaperPassage.paper_id == existing_paper.id
                ).scalar() or 0
            else:
                new_count += 1
                paper_status = "NEW" if has_pdf else "METADATA_ONLY"
                passage_count = 0

            if has_pdf:
                pdf_count += 1
                pdf_status = "OPEN_ACCESS_AVAILABLE"
            else:
                meta_only_count += 1
                pdf_status = "METADATA_ONLY"

            is_pre_cutoff = True
            if pub_date and cutoff:
                try:
                    p_date = date.fromisoformat(pub_date[:10])
                    is_pre_cutoff = p_date <= cutoff
                except Exception:
                    pass

            warning = None
            if not has_pdf and not in_corpus:
                warning = "No open-access PDF available; stored as metadata only."

            items.append({
                "openalex_id": raw_oa_id or (norm_oa_id or ""),
                "doi": norm_doi,
                "title": title,
                "authors": authors,
                "publication_year": pub_year,
                "publication_date": pub_date,
                "pdf_url": pdf_url,
                "has_pdf": has_pdf,
                "status": paper_status,
                "pdf_status": pdf_status,
                "is_in_corpus": in_corpus,
                "corpus_paper_id": str(existing_paper.id) if existing_paper else None,
                "existing_paper_id": str(existing_paper.id) if existing_paper else None,
                "existing_passages": passage_count,
                "is_pre_cutoff": is_pre_cutoff,
                "warning": warning,
            })

        if meta_only_count > 0:
            warnings.append(
                f"{meta_only_count} selected paper(s) have no open-access PDF listed. They will be stored as metadata only without passage text; research statements cannot be extracted from them."
            )

        if corpus_count > 0:
            warnings.append(
                f"{corpus_count} paper(s) already exist in the corpus and will be reused without re-downloading."
            )

        planned_stages = [
            {
                "stage_id": "INGESTION",
                "stage": "INGESTION",
                "name": "Literature Ingestion",
                "description": f"Harvest full-text open-access PDFs and extract passages ({new_count} new papers, {corpus_count} existing).",
                "estimated_action": f"Download & parse {new_count} papers",
            },
            {
                "stage_id": "UNDERSTANDING",
                "stage": "UNDERSTANDING",
                "name": "Statement Extraction",
                "description": (
                    "Extract verifiable findings, limitations, and methodology claims from stored passages "
                    f"of the project's papers using the configured LLM ({self._llm_label()})."
                ),
                "estimated_action": "Extract statements from new passages",
            },
            {
                "stage_id": "GAP_DISCOVERY",
                "stage": "GAP_DISCOVERY",
                "name": "Research Gap Discovery",
                "description": (
                    "Cluster this project's limitation and future-work statements into evidence-backed candidate gaps. "
                    "The run then pauses for researcher review; opportunities are generated only on explicit request."
                ),
                "estimated_action": "Discover semantic gaps",
            },
        ]

        # Papers already in the project corpus (Paper Search "Add") are not queued selections, but a
        # run processes them: UNDERSTANDING extracts their statements, GAP_DISCOVERY reads them.
        corpus = self._corpus_summary(session, project_id)
        if not selections and corpus["total"] == 0:
            warnings.append(
                "No papers are selected and this project's corpus is empty. Add papers before starting a run."
            )
        elif corpus["total"] > 0 and corpus["with_passages"] == 0:
            warnings.append(
                f"None of the {corpus['total']} corpus paper(s) has full text, so no research statements can be "
                "extracted from them and gap discovery will find nothing in the corpus."
            )
        elif corpus["metadata_only"] > 0:
            warnings.append(
                f"{corpus['metadata_only']} of {corpus['total']} corpus paper(s) are metadata only (no full text); "
                "statements cannot be extracted from them."
            )

        return {
            "project_id": project_id,
            "project_name": project.name if project else "Active Project",
            "temporal_cutoff": str(cutoff) if cutoff else None,
            "total_selected": len(selections),
            "new_papers": new_count,
            "new_papers_count": new_count,
            "corpus_matches": corpus_count,
            "already_in_corpus_count": corpus_count,
            "pdf_available": pdf_count,
            "pdf_available_count": pdf_count,
            "metadata_only": meta_only_count,
            "metadata_only_count": meta_only_count,
            "papers": items,
            "items": items,
            "planned_stages": planned_stages,
            "warnings": warnings,
            "corpus": corpus,
            "runnable": bool(selections) or corpus["total"] > 0,
        }

    def create_run(
        self,
        project_id: str,
        selections: Optional[List[Dict[str, Any]]] = None,
        planned_stages: Optional[List[str]] = None,
        db: Optional[Session] = None,
    ) -> ResearchRunRecord:
        """Persists a new research run and associated paper tracking records."""
        session = db or getattr(self, "db", None)
        if session is None:
            raise ValueError("Database session must be provided")

        if selections is None:
            sel_records = session.query(ProjectPaperSelectionRecord).filter(
                ProjectPaperSelectionRecord.project_id == project_id
            ).all()
            selections = [
                {
                    "openalex_id": s.openalex_id,
                    "doi": s.doi,
                    "title": s.title,
                    "authors": s.authors or [],
                    "publication_date": s.publication_date,
                    "pdf_url": s.pdf_url,
                    "abstract": s.abstract,
                }
                for s in sel_records
            ]

        stages = normalize_planned_stages(planned_stages)
        if not selections and not self._project_paper_ids(session, project_id):
            raise ValueError(
                "Nothing to process: no papers are selected and this project's corpus is empty. "
                "Add papers to the project before starting a run."
            )
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        stage_states = {
            s: {
                "status": "PENDING",
                "started_at": None,
                "completed_at": None,
                "details": {},
            }
            for s in stages
        }

        stage_progress = {
            "stages": [
                {
                    "stage_id": s,
                    "name": s.replace("_", " ").title(),
                    "status": "PENDING",
                    "message": None,
                }
                for s in stages
            ],
            "logs": [
                {
                    "timestamp": _utc_now_iso(),
                    "level": "INFO",
                    "stage": "INITIALIZATION",
                    "message": f"Initialized research run {run_id} for project {project_id} with {len(selections)} papers across stages: {', '.join(stages)}.",
                }
            ]
        }

        run = ResearchRunRecord(
            id=run_id,
            project_id=project_id,
            status="PENDING",
            current_stage=stages[0] if stages else "COMPLETED",
            planned_stages=stages,
            stage_states=stage_states,
            summary=None,
            logs=stage_progress["logs"],
            created_at=now,
            updated_at=now,
        )
        session.add(run)

        # Create paper records
        for p in selections:
            paper_rec_id = f"rp_{uuid.uuid4().hex[:12]}"
            doi = p.get("doi")
            norm_doi = PaperNormalizer.normalize_doi(doi) if doi else None
            run_paper = RunPaperRecord(
                id=paper_rec_id,
                run_id=run_id,
                openalex_id=p.get("openalex_id") or "",
                doi=norm_doi,
                title=p.get("title") or "Untitled Paper",
                status="PENDING",
                created_at=now,
                updated_at=now,
            )
            session.add(run_paper)

        session.commit()
        session.refresh(run)
        return run

    @classmethod
    def execute_run(cls, run_id: str, db_override: Optional[Session] = None) -> None:
        """Executes planned workflow stages with safe per-paper persistence and idempotent updates."""
        if not cls.mark_run_active(run_id):
            logger.warning(f"Run {run_id} is already actively running. Rejecting duplicate invocation.")
            raise RuntimeError(f"Run {run_id} is already actively running.")
        cls._execute_marked_run(run_id, db_override)

    @classmethod
    def _execute_marked_run(cls, run_id: str, db_override: Optional[Session] = None) -> None:
        """Runs an execution whose run_id the caller has already registered as active."""
        close_session = False
        if db_override is not None:
            db = db_override
        else:
            session_factory = get_session_factory()
            db = session_factory()
            close_session = True

        try:
            run = db.query(ResearchRunRecord).filter(ResearchRunRecord.id == run_id).first()
            if not run:
                logger.error(f"Cannot execute run {run_id}: not found.")
                return

            run.status = "RUNNING"
            run.updated_at = datetime.now(timezone.utc)
            cls._append_log(run, "INFO", "RUNNER", f"Started execution of run {run_id}.")
            db.commit()

            stages = [s for s in RUN_STAGES if s in (run.planned_stages or [])]
            stage_states = dict(run.stage_states or {})

            def _finished(stage: str) -> bool:
                return (stage_states.get(stage) or {}).get("status") in _FINISHED_STAGE_STATUSES

            # On retry, finished stages keep their results. UNDERSTANDING is idempotent per paper, so
            # it re-runs whenever ingestion re-ran. A finished GAP_DISCOVERY is never repeated within
            # the same run, which would persist a second, duplicate set of gaps.
            ingestion_ran = False
            if "INGESTION" in stages and not _finished("INGESTION"):
                cls._execute_stage_ingestion(db, run, stage_states)
                ingestion_ran = True

            if "UNDERSTANDING" in stages and (ingestion_ran or not _finished("UNDERSTANDING")):
                cls._execute_stage_understanding(db, run, stage_states)

            if "GAP_DISCOVERY" in stages:
                if not _finished("GAP_DISCOVERY"):
                    cls._execute_stage_gap_discovery(db, run, stage_states)
                else:
                    if ((stage_states.get("GAP_DISCOVERY") or {}).get("details") or {}).get("gaps_discovered", 0) > 0:
                        # The preserved discovery's gaps still await researcher review.
                        run.status = "AWAITING_REVIEW"
                    if ingestion_ran:
                        cls._append_log(
                            run,
                            "INFO",
                            "GAP_DISCOVERY",
                            "Gap discovery already completed in this run and was not repeated, to avoid duplicate gaps. "
                            "Start a new run to include newly ingested papers in gap discovery.",
                        )
                        db.commit()

            # Finalize run status
            cls._finalize_run_summary(db, run, stage_states)

        except Exception as e:
            logger.error(f"Fatal error in run {run_id}: {e}", exc_info=True)
            try:
                run = db.query(ResearchRunRecord).filter(ResearchRunRecord.id == run_id).first()
                if run:
                    run.status = "FAILED"
                    run.error_message = str(e)
                    run.updated_at = datetime.now(timezone.utc)
                    cls._append_log(run, "ERROR", "FATAL", f"Run failed due to unhandled error: {e}")
                    db.commit()
            except Exception:
                db.rollback()
        finally:
            cls.mark_run_inactive(run_id)
            if close_session:
                db.close()

    @classmethod
    def start_run_async(cls, run_id: str) -> None:
        """Spawns an asynchronous background thread for non-blocking execution.

        The run is registered as active before the thread starts, so a concurrent start or a
        recovery sweep cannot observe it as idle in between.
        """
        if not cls.mark_run_active(run_id):
            raise RuntimeError(f"Run {run_id} is already actively running.")
        worker = threading.Thread(
            target=cls._execute_marked_run,
            args=(run_id,),
            daemon=True,
            name=f"IdeaGen-Run-{run_id}",
        )
        try:
            worker.start()
        except Exception:
            cls.mark_run_inactive(run_id)
            raise

    def retry_run(self, run_id: str, db_override: Optional[Session] = None) -> ResearchRunRecord:
        """Resets failed or interrupted papers and stages for a safe idempotent retry."""
        if self.is_run_active(run_id):
            raise ValueError(f"Run {run_id} is currently running.")
        session = db_override or getattr(self, "db", None)
        close_session = False
        if session is None:
            session_factory = get_session_factory()
            session = session_factory()
            close_session = True
        try:
            run = session.query(ResearchRunRecord).filter(ResearchRunRecord.id == run_id).first()
            if not run:
                raise ValueError(f"Run {run_id} not found.")

            if run.status == "RUNNING" and self.is_run_active(run_id):
                raise ValueError(f"Run {run_id} is currently running.")

            def _is_stage_retryable(stage_name: str, stage_dict: Any) -> bool:
                if not isinstance(stage_dict, dict):
                    return False
                if stage_dict.get("status") in {"FAILED", "ERROR", "RUNNING", "PARTIAL_SUCCESS", "INTERRUPTED"}:
                    return True
                # If GAP_DISCOVERY was marked completed but discovered zero gaps, it is eligible for retry
                if stage_name == "GAP_DISCOVERY":
                    details = stage_dict.get("details")
                    if isinstance(details, dict):
                        gaps_count = details.get("gaps_discovered")
                        if gaps_count == 0 and not isinstance(gaps_count, bool):
                            return True
                return False

            has_failed_papers = any(p.status in {"FAILED", "ERROR", "INTERRUPTED"} for p in run.papers)
            has_unfinished_stages = any(
                (run.stage_states or {}).get(s, {}).get("status") not in _FINISHED_STAGE_STATUSES
                or _is_stage_retryable(s, (run.stage_states or {}).get(s, {}))
                for s in (run.planned_stages or [])
            )
            if run.status not in RETRYABLE_RUN_STATUSES and not has_failed_papers:
                raise ValueError(f"Run {run_id} has status {run.status} and has nothing to retry.")
            if not has_failed_papers and not has_unfinished_stages:
                raise ValueError(f"Run {run_id} has no failed papers or unfinished stages to retry.")

            # Reset failed or interrupted papers
            for p in run.papers:
                if p.status in {"FAILED", "ERROR", "PENDING", "INTERRUPTED"}:
                    p.status = "PENDING"
                    p.error_message = None
                    p.updated_at = datetime.now(timezone.utc)

            # Reset failed or running or interrupted stages, plus GAP_DISCOVERY if zero gaps were found
            stage_states = copy.deepcopy(run.stage_states) if run.stage_states else {}
            for s, st in stage_states.items():
                if _is_stage_retryable(s, st):
                    st["status"] = "PENDING"
                    st["started_at"] = None
                    st["completed_at"] = None

            run.status = "PENDING"
            run.stage_states = stage_states
            flag_modified(run, "stage_states")
            run.error_message = None
            run.updated_at = datetime.now(timezone.utc)
            self._append_log(run, "INFO", "RETRY", f"Run {run_id} reset for retry.")
            session.commit()
            session.refresh(run)

            if db_override is not None:
                self.execute_run(run_id, db_override=session)
                session.refresh(run)
            else:
                self.start_run_async(run_id)
            return run
        finally:
            if close_session:
                session.close()

    # ── Internal Stage Handlers ──────────────────────────────────────────────

    @classmethod
    def _execute_stage_ingestion(cls, db: Session, run: ResearchRunRecord, stage_states: dict) -> None:
        run.current_stage = "INGESTION"
        stage_states["INGESTION"] = {
            "status": "RUNNING",
            "started_at": _utc_now_iso(),
            "completed_at": None,
            "details": {},
        }
        run.stage_states = stage_states
        flag_modified(run, "stage_states")
        cls._append_log(run, "INFO", "INGESTION", "Stage 1: Beginning literature ingestion and passage extraction.")
        db.commit()

        persistence = SQLAlchemyPersistenceLayer(db)
        search_provider = OpenAlexSearchProvider(raise_on_error=True)
        ingestion_service = OpenAlexPaperIngestionService(persistence=persistence, auto_evolve=False)

        ingested_count = 0
        skipped_count = 0
        failed_count = 0

        for p in run.papers:
            # Check if paper is already processed in this run
            if p.status in {"INGESTED", "SKIPPED_DUPLICATE", "COMPLETED"}:
                skipped_count += 1
                continue

            p.status = "INGESTING"
            p.updated_at = datetime.now(timezone.utc)
            db.commit()

            norm_oa_id = PaperNormalizer.normalize_openalex_id(p.openalex_id) if p.openalex_id else None
            norm_doi = PaperNormalizer.normalize_doi(p.doi) if p.doi else None

            # 1. Check existing corpus first
            existing = None
            if norm_oa_id:
                existing = db.query(Paper).filter(Paper.source_id == norm_oa_id).first()
            if not existing and norm_doi:
                existing = db.query(Paper).filter(Paper.doi == norm_doi).first()

            if existing:
                p.paper_id = str(existing.id)
                # Ensure association with project in project_papers
                existing_assoc = db.query(ProjectPaperRecord).filter(
                    (ProjectPaperRecord.project_id == run.project_id) & (ProjectPaperRecord.paper_id == existing.id)
                ).first()
                if not existing_assoc:
                    db.add(ProjectPaperRecord(project_id=run.project_id, paper_id=existing.id))
                    db.flush()

                passages = db.query(sqlalchemy.func.count(PaperPassage.id)).filter(
                    PaperPassage.paper_id == existing.id
                ).scalar() or 0
                p.passage_count = passages
                p.status = "SKIPPED_DUPLICATE"
                p.updated_at = datetime.now(timezone.utc)
                skipped_count += 1
                cls._append_log(run, "INFO", "INGESTION", f"Paper '{p.title}' found in existing corpus ({passages} passages). Skipping re-download.")
                db.commit()
                continue

            # 2. Ingest via OpenAlex
            try:
                lookup_id = norm_oa_id or (f"https://doi.org/{norm_doi}" if norm_doi else None)
                if not lookup_id:
                    p.status = "FAILED"
                    p.error_message = "No valid OpenAlex ID or DOI available."
                    failed_count += 1
                    cls._append_log(run, "WARNING", "INGESTION", f"Skipping '{p.title}': No valid identifier.")
                    db.commit()
                    continue

                try:
                    search_res = search_provider.search(lookup_id, limit=1)
                except OpenAlexSearchError as search_err:
                    # Network, proxy, rate-limit and upstream failures are not "paper not found".
                    p.status = "FAILED"
                    p.error_message = f"OpenAlex lookup failed: {search_err.message}"
                    p.updated_at = datetime.now(timezone.utc)
                    failed_count += 1
                    cls._append_log(run, "WARNING", "INGESTION", f"OpenAlex lookup failed for '{p.title}': {search_err.message}")
                    db.commit()
                    continue
                if not search_res or not search_res[0].raw_metadata:
                    p.status = "FAILED"
                    p.error_message = "Paper metadata not found in OpenAlex."
                    failed_count += 1
                    cls._append_log(run, "WARNING", "INGESTION", f"Could not find OpenAlex metadata for '{p.title}'.")
                    db.commit()
                    continue

                work = search_res[0].raw_metadata
                res = ingestion_service.ingest_work(work)

                paper_id_str, paper_uuid = cls._normalize_paper_ids(res.paper_id)

                if res.status == "INGESTED":
                    p.paper_id = paper_id_str
                    p.passage_count = res.passage_count
                    p.status = "INGESTED"
                    if paper_uuid:
                        existing_assoc = db.query(ProjectPaperRecord).filter(
                            (ProjectPaperRecord.project_id == run.project_id) & (ProjectPaperRecord.paper_id == paper_uuid)
                        ).first()
                        if not existing_assoc:
                            db.add(ProjectPaperRecord(project_id=run.project_id, paper_id=paper_uuid))
                            db.flush()
                    ingested_count += 1
                    cls._append_log(run, "INFO", "INGESTION", f"Successfully ingested '{p.title}' with {res.passage_count} passages stored.")
                elif res.status == "NO_PDF":
                    p.paper_id = paper_id_str
                    p.passage_count = 0
                    p.status = "NO_PDF"
                    if paper_uuid:
                        existing_assoc = db.query(ProjectPaperRecord).filter(
                            (ProjectPaperRecord.project_id == run.project_id) & (ProjectPaperRecord.paper_id == paper_uuid)
                        ).first()
                        if not existing_assoc:
                            db.add(ProjectPaperRecord(project_id=run.project_id, paper_id=paper_uuid))
                            db.flush()
                    ingested_count += 1
                    cls._append_log(run, "INFO", "INGESTION", f"Stored metadata for '{p.title}' (no open-access PDF available).")
                elif res.status == "SKIPPED_DUPLICATE":
                    p.paper_id = paper_id_str
                    p.status = "SKIPPED_DUPLICATE"
                    if paper_uuid:
                        existing_assoc = db.query(ProjectPaperRecord).filter(
                            (ProjectPaperRecord.project_id == run.project_id) & (ProjectPaperRecord.paper_id == paper_uuid)
                        ).first()
                        if not existing_assoc:
                            db.add(ProjectPaperRecord(project_id=run.project_id, paper_id=paper_uuid))
                            db.flush()
                    skipped_count += 1
                    cls._append_log(run, "INFO", "INGESTION", f"Duplicate paper '{p.title}' skipped.")
                else:
                    p.status = "FAILED"
                    p.error_message = res.error_message or "Ingestion failed."
                    failed_count += 1
                    cls._append_log(run, "WARNING", "INGESTION", f"Ingestion error for '{p.title}': {p.error_message}")

                p.updated_at = datetime.now(timezone.utc)
                db.commit()

            except Exception as e:
                db.rollback()
                p.status = "FAILED"
                p.error_message = str(e)
                if p.paper_id is not None:
                    p.paper_id = str(p.paper_id)
                p.updated_at = datetime.now(timezone.utc)
                failed_count += 1
                cls._append_log(run, "ERROR", "INGESTION", f"Unexpected failure ingesting '{p.title}': {e}")
                try:
                    db.commit()
                except Exception as commit_err:
                    logger.error(f"Failed committing failure status for paper '{p.title}': {commit_err}")
                    db.rollback()

        stage_states["INGESTION"] = {
            "status": "COMPLETED" if failed_count == 0 else ("PARTIAL_SUCCESS" if ingested_count > 0 or skipped_count > 0 else "FAILED"),
            "started_at": stage_states["INGESTION"]["started_at"],
            "completed_at": _utc_now_iso(),
            "details": {
                "ingested": ingested_count,
                "skipped_duplicate": skipped_count,
                "failed": failed_count,
            },
        }
        run.stage_states = stage_states
        flag_modified(run, "stage_states")
        db.commit()

    @classmethod
    def _execute_stage_understanding(cls, db: Session, run: ResearchRunRecord, stage_states: dict) -> None:
        run.current_stage = "UNDERSTANDING"
        stage_states["UNDERSTANDING"] = {
            "status": "RUNNING",
            "started_at": _utc_now_iso(),
            "completed_at": None,
            "details": {},
        }
        run.stage_states = stage_states
        flag_modified(run, "stage_states")
        cls._append_log(run, "INFO", "UNDERSTANDING", "Stage 2: Extracting evidenced research statements using local LLM.")
        db.commit()

        persistence = SQLAlchemyPersistenceLayer(db)
        understanding_pipeline = PaperUnderstandingPipeline(persistence=persistence)

        extracted_count = 0
        already_extracted_count = 0
        succeeded_count = 0
        failed_count = 0
        processed_paper_ids: set = set()

        for p in run.papers:
            if not p.paper_id:
                continue

            _, p_uuid = cls._normalize_paper_ids(p.paper_id)
            if not p_uuid:
                continue
            processed_paper_ids.add(p_uuid)

            # Reuse only a successful extraction with statements; a recorded failure or empty extraction on full-text is retried
            existing_knowledge = db.query(ResearchKnowledge).filter(
                ResearchKnowledge.paper_id == p_uuid, ResearchKnowledge.extraction_status == "SUCCESS"
            ).first()
            if existing_knowledge:
                stmts = db.query(sqlalchemy.func.count(ResearchStatement.id)).filter(
                    ResearchStatement.knowledge_id == existing_knowledge.id
                ).scalar() or 0
                if stmts > 0:
                    p.statement_count = stmts
                    p.status = "COMPLETED"
                    p.updated_at = datetime.now(timezone.utc)
                    already_extracted_count += 1
                    cls._append_log(run, "INFO", "UNDERSTANDING", f"Reusing {stmts} existing statements for '{p.title}'.")
                    db.commit()
                    continue

            # Skip metadata-only papers with 0 passages
            if p.passage_count == 0:
                p.status = "COMPLETED"
                p.statement_count = 0
                p.updated_at = datetime.now(timezone.utc)
                cls._append_log(run, "INFO", "UNDERSTANDING", f"Skipping statement extraction for '{p.title}': No full-text passages stored.")
                db.commit()
                continue

            p.status = "EXTRACTING_STATEMENTS"
            p.updated_at = datetime.now(timezone.utc)
            db.commit()

            try:
                knowledge_res = understanding_pipeline.understand_paper(p_uuid)
                stmts_found = cls._extracted_statement_count(knowledge_res)
                p.statement_count = stmts_found
                p.status = "COMPLETED"
                p.updated_at = datetime.now(timezone.utc)
                extracted_count += stmts_found
                succeeded_count += 1
                cls._append_log(run, "INFO", "UNDERSTANDING", f"Extracted {stmts_found} research statements from '{p.title}'.")
                db.commit()
            except Exception as e:
                db.rollback()
                p.status = "FAILED"
                p.error_message = f"Understanding error: {str(e)}"
                p.updated_at = datetime.now(timezone.utc)
                failed_count += 1
                cls._append_log(run, "WARNING", "UNDERSTANDING", f"Statement extraction failed for '{p.title}': {e}")
                db.commit()

        # Papers added to the project corpus directly (Paper Search "Add") are not run papers, but
        # they are part of the corpus gap discovery reads, so extract their statements as well.
        # Reuse only papers with a successful extraction that actually contains statements.
        corpus_pending = (
            db.query(Paper)
            .join(ProjectPaperRecord, ProjectPaperRecord.paper_id == Paper.id)
            .filter(ProjectPaperRecord.project_id == run.project_id)
            .filter(~Paper.id.in_(
                db.query(ResearchKnowledge.paper_id)
                .join(ResearchStatement, ResearchStatement.knowledge_id == ResearchKnowledge.id)
                .filter(ResearchKnowledge.extraction_status == "SUCCESS")
            ))
            .all()
        )
        for paper in corpus_pending:
            if paper.id in processed_paper_ids:
                continue
            passages = db.query(sqlalchemy.func.count(PaperPassage.id)).filter(PaperPassage.paper_id == paper.id).scalar() or 0
            if passages == 0:
                continue
            title = paper.title or str(paper.id)
            try:
                knowledge_res = understanding_pipeline.understand_paper(paper.id)
                stmts_found = cls._extracted_statement_count(knowledge_res)
                extracted_count += stmts_found
                succeeded_count += 1
                cls._append_log(run, "INFO", "UNDERSTANDING", f"Extracted {stmts_found} research statements from corpus paper '{title}'.")
                db.commit()
            except Exception as e:
                db.rollback()
                failed_count += 1
                cls._append_log(run, "WARNING", "UNDERSTANDING", f"Statement extraction failed for corpus paper '{title}': {e}")
                db.commit()

        if failed_count == 0:
            status = "COMPLETED"
        elif succeeded_count > 0 or already_extracted_count > 0:
            status = "PARTIAL_SUCCESS"
        else:
            status = "FAILED"
        stage_states["UNDERSTANDING"] = {
            "status": status,
            "started_at": stage_states["UNDERSTANDING"]["started_at"],
            "completed_at": _utc_now_iso(),
            "details": {
                "statements_extracted": extracted_count,
                "reused_existing": already_extracted_count,
                "failed": failed_count,
            },
        }
        run.stage_states = stage_states
        flag_modified(run, "stage_states")
        db.commit()

    @staticmethod
    def _extracted_statement_count(knowledge_res) -> int:
        """Statement count of a ResearchKnowledgeResponseDTO; raises if extraction did not succeed."""
        status = getattr(knowledge_res, "extraction_status", "SUCCESS")
        if status != "SUCCESS":
            raise RuntimeError(getattr(knowledge_res, "error_message", None) or f"extraction status {status}")
        return int(getattr(knowledge_res, "statement_count", 0) or 0)

    @classmethod
    def _execute_stage_gap_discovery(cls, db: Session, run: ResearchRunRecord, stage_states: dict) -> None:
        run.current_stage = "GAP_DISCOVERY"
        stage_states["GAP_DISCOVERY"] = {
            "status": "RUNNING",
            "started_at": _utc_now_iso(),
            "completed_at": None,
            "details": {},
        }
        run.stage_states = stage_states
        flag_modified(run, "stage_states")
        cls._append_log(run, "INFO", "GAP_DISCOVERY", "Stage 3: Running semantic gap discovery across this project's statements.")
        db.commit()

        project = db.query(ResearchProjectRecord).filter(ResearchProjectRecord.id == run.project_id).first()
        cutoff_str = str(project.temporal_cutoff) if (project and project.temporal_cutoff) else None
        project_paper_ids = cls._project_paper_ids(db, run.project_id)

        try:
            persistence = SQLAlchemyPersistenceLayer(db)
            discovery_pipeline = SemanticGapDiscoveryPipeline(persistence=persistence)
            # Only this project's papers: other projects' statements must never shape its gaps.
            # Gaps are tagged with the project and run as each one is saved.
            discovered_gaps = discovery_pipeline.discover_gaps(
                cutoff_date=cutoff_str,
                paper_ids=project_paper_ids,
                project_id=run.project_id,
                gap_metadata={"run_id": run.id},
            )
            gap_count = len(discovered_gaps)

            stage_states["GAP_DISCOVERY"] = {
                "status": "COMPLETED",
                "started_at": stage_states["GAP_DISCOVERY"]["started_at"],
                "completed_at": _utc_now_iso(),
                "details": {
                    "gaps_discovered": gap_count,
                    "project_papers_considered": len(project_paper_ids),
                },
            }
            cls._append_log(
                run, "INFO", "GAP_DISCOVERY",
                f"Discovered {gap_count} research gap(s) from {len(project_paper_ids)} project paper(s).",
            )

            if gap_count > 0:
                # Researcher approval requirement: the run pauses for gap review; opportunity
                # generation only happens on explicit request for a VALIDATED gap.
                run.status = "AWAITING_REVIEW"
                cls._append_log(
                    run,
                    "INFO",
                    "AWAITING_REVIEW",
                    f"Pipeline paused at Awaiting Review: inspect the {gap_count} discovered gap(s) and their evidence "
                    "before explicitly generating any opportunity."
                )

        except Exception as e:
            db.rollback()
            stage_states["GAP_DISCOVERY"] = {
                "status": "FAILED",
                "started_at": stage_states["GAP_DISCOVERY"]["started_at"],
                "completed_at": _utc_now_iso(),
                "details": {"error": str(e)},
            }
            run.error_message = f"Gap discovery failed: {e}"
            cls._append_log(run, "ERROR", "GAP_DISCOVERY", f"Gap discovery failed: {e}")

        run.stage_states = stage_states
        flag_modified(run, "stage_states")
        db.commit()

    @classmethod
    def _finalize_run_summary(cls, db: Session, run: ResearchRunRecord, stage_states: dict) -> None:
        failed_papers = sum(1 for p in run.papers if p.status in {"FAILED", "ERROR"})
        completed_papers = sum(1 for p in run.papers if p.status in {"COMPLETED", "INGESTED", "SKIPPED_DUPLICATE", "ANALYZED", "NO_PDF"})
        total_stmts = sum(p.statement_count for p in run.papers)
        planned = run.planned_stages or []
        stage_status = {s: (stage_states.get(s) or {}).get("status") for s in planned}
        failed_stages = [s for s, st in stage_status.items() if st == "FAILED"]
        partial_stages = [s for s, st in stage_status.items() if st == "PARTIAL_SUCCESS"]

        gaps_awaiting_review = run.status == "AWAITING_REVIEW"
        if failed_stages:
            # A failed stage is never reported as success. If gap discovery still produced gaps
            # (from the evidence that was available), they remain reviewable: PARTIAL_SUCCESS.
            run.status = "PARTIAL_SUCCESS" if gaps_awaiting_review and "GAP_DISCOVERY" not in failed_stages else "FAILED"
            if not run.error_message:
                run.error_message = f"Stage(s) failed: {', '.join(failed_stages)}."
        elif not gaps_awaiting_review:
            if failed_papers == 0 and not partial_stages:
                run.status = "COMPLETED"
            elif completed_papers > 0 or partial_stages:
                run.status = "PARTIAL_SUCCESS"
            else:
                run.status = "FAILED"

        run.current_stage = "COMPLETED"
        next_actions = []
        if gaps_awaiting_review:
            next_actions.append("Review discovered research gaps and their evidence; generate an opportunity explicitly from a validated gap.")
        if not gaps_awaiting_review and run.status in {"COMPLETED", "PARTIAL_SUCCESS"}:
            next_actions.append("Inspect extracted statements in Corpus Explorer.")
            gap_details = (stage_states.get("GAP_DISCOVERY") or {}).get("details") or {}
            if stage_status.get("GAP_DISCOVERY") == "COMPLETED" and gap_details.get("gaps_discovered", 0) == 0:
                next_actions.append("No research gaps were discovered; add more papers to the project corpus and start a new run.")
        if failed_papers > 0 or failed_stages or partial_stages:
            next_actions.append("Use the Retry Action to reprocess failed papers or stages after fixing the reported cause.")

        run.summary = {
            "total_papers": len(run.papers),
            "completed_papers": completed_papers,
            "successful_papers": completed_papers,
            "failed_papers": failed_papers,
            "failed_stages": failed_stages,
            "total_statements": total_stmts,
            "next_actions": next_actions,
            "completed_at": _utc_now_iso(),
        }
        run.updated_at = datetime.now(timezone.utc)
        cls._append_log(run, "INFO", "COMPLETED", f"Run finished with status {run.status}. {completed_papers}/{len(run.papers)} papers processed successfully.")
        db.commit()

    @staticmethod
    def _append_log(run: ResearchRunRecord, level: str, stage: str, message: str) -> None:
        log_entry = {
            "timestamp": _utc_now_iso(),
            "level": level,
            "stage": stage,
            "message": message,
        }
        current_logs = list(run.logs or [])
        current_logs.append(log_entry)
        run.logs = current_logs
        flag_modified(run, "logs")


WorkflowEngine = ResearchWorkflowEngine

