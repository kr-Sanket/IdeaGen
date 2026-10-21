"""Incremental Opportunity Evolution Service (Stage 3B, Part 1).

Connects newly ingested and persisted papers to eligible research opportunities
for incremental lifecycle evolution, using the existing classification integration,
deterministic transition service, fail-closed evidence grounding, and lifecycle state machine.

Architecture & Design Decisions:
1. Pure Orchestration: Reuses ``OpportunityClassificationIntegration``, ``OpportunityTransitionService``,
   and ``state_machine.transition()``. Does NOT introduce a new lifecycle engine or transition rules.
2. Separation of Concerns: LLM classifies relationship semantics; deterministic application code applies transitions.
3. Candidate Bounds & Relevance Trade-off:
   - In the current IdeaGen architecture, research opportunities are generated from gaps and lack an
     inverted semantic index or vector catalog for incoming papers.
   - In the absence of an empirical, validated paper-to-opportunity relevance filter, applying an
     ad-hoc heuristic threshold would risk silent false negatives (ignoring opportunities that a paper
     addresses or contradicts) and violate the repository's fail-closed correctness guarantees.
   - Therefore, candidate selection bounds queries strictly by:
     (a) Valid lifecycle statuses (``OpportunityStatus``),
     (b) Temporal consistency (``effective_creation_date <= paper.publication_date``),
     (c) Idempotency exclusion (skipping already-evaluated opportunity/paper pairs),
     (d) Deterministic chronological ordering (``created_at.asc(), id.asc()``),
     (e) Configurable candidate limit (``max_candidates``).
4. Atomicity & Transaction Safety:
   - Evaluates each candidate opportunity inside an isolated nested savepoint (``session.begin_nested()``).
   - If classification, evidence grounding, or validation fails on one opportunity, only that savepoint
     is rolled back; neither the outer session, the persisted paper, nor other candidate evaluations are corrupted.
   - Preserves caller-owned transaction boundaries: never calls ``session.commit()``.
5. Idempotent Deduplication:
   - Pre-checks for existing ``OpportunityEvidence(opportunity_id, paper_id)``.
   - Handles concurrent database uniqueness conflicts gracefully via savepoint rollback and skip reporting.
"""

from __future__ import annotations

import datetime
from enum import Enum
import logging
from typing import Any, List, Optional, Sequence, Union
import uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from ideagen.db.models import Opportunity, Paper, PaperPassage, OpportunityEvidence
from ideagen.opportunity.models import OpportunityStatus, OpportunityRelationship
from ideagen.opportunity.integration import (
    OpportunityClassificationIntegration,
    ClassificationTransitionResult,
)
from ideagen.opportunity.grounding import UngroundedEvidenceError
from ideagen.opportunity.bridge import ELIGIBLE_GENERATION_STATUSES

logger = logging.getLogger(__name__)


class OpportunityEvaluationStatus(str, Enum):
    """Status of an individual opportunity evaluation against a target paper."""
    EVALUATED = "EVALUATED"
    SKIPPED = "SKIPPED"
    GROUNDING_REJECTED = "GROUNDING_REJECTED"
    FAILED = "FAILED"


class IncrementalEvolutionRunStatus(str, Enum):
    """Overall status of an incremental evolution evaluation run for a paper."""
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    NO_CANDIDATES = "NO_CANDIDATES"
    SKIPPED_UNDATED = "SKIPPED_UNDATED"
    SKIPPED_INSUFFICIENT_CONTENT = "SKIPPED_INSUFFICIENT_CONTENT"
    PAPER_NOT_FOUND = "PAPER_NOT_FOUND"
    FAILED = "FAILED"


@dataclass(frozen=True)
class OpportunityEvaluationOutcome:
    """Outcome of evaluating a single opportunity against a paper."""
    opportunity_id: uuid.UUID
    status: OpportunityEvaluationStatus
    initial_status: Optional[OpportunityStatus] = None
    final_status: Optional[OpportunityStatus] = None
    relationship: Optional[OpportunityRelationship] = None
    state_changed: bool = False
    evidence_id: Optional[uuid.UUID] = None
    history_id: Optional[uuid.UUID] = None
    skip_reason: Optional[str] = None
    error_message: Optional[str] = None


@dataclass(frozen=True)
class IncrementalEvolutionResult:
    """Structured result returned by IncrementalOpportunityEvolutionService."""
    paper_id: uuid.UUID
    run_status: IncrementalEvolutionRunStatus
    paper_title: Optional[str] = None
    publication_date: Optional[datetime.date] = None
    candidate_count: int = 0
    evaluated_count: int = 0
    skipped_count: int = 0
    grounding_rejected_count: int = 0
    failed_count: int = 0
    outcomes: List[OpportunityEvaluationOutcome] = field(default_factory=list)
    error: Optional[str] = None


def get_opportunity_effective_creation_date(opp: Opportunity) -> Optional[datetime.date]:
    """Resolves the effective creation date of an opportunity.

    Prefers native opportunity ``created_at``. If missing, checks the bridged
    ``research_opportunity.created_at``. Returns ``datetime.date`` or None if undetermined.
    """
    opp_dt = opp.created_at
    if opp_dt is None and getattr(opp, "research_opportunity", None):
        opp_dt = opp.research_opportunity.created_at

    if opp_dt is not None:
        return opp_dt.date() if isinstance(opp_dt, datetime.datetime) else opp_dt
    return None


class IncrementalOpportunityEvolutionService:
    """Orchestrates incremental evaluation of a persisted paper against candidate opportunities."""

    def __init__(
        self,
        integration: OpportunityClassificationIntegration,
        session: Optional[Session] = None,
        max_candidates: Optional[int] = None,
    ) -> None:
        """Initializes the incremental opportunity evolution service.

        Args:
            integration: Existing OpportunityClassificationIntegration instance.
            session: Optional SQLAlchemy session. Can also be supplied per invocation.
            max_candidates: Optional default cap on candidate opportunities evaluated per paper.
        """
        self.integration = integration
        self.session = session
        self.max_candidates = max_candidates

    def evaluate_paper(
        self,
        paper_id: uuid.UUID,
        session: Optional[Session] = None,
        cutoff_date: Optional[Union[datetime.date, datetime.datetime, str]] = None,
        max_candidates: Optional[int] = None,
        candidate_opportunity_ids: Optional[Sequence[uuid.UUID]] = None,
    ) -> IncrementalEvolutionResult:
        """Evaluates a persisted paper against eligible candidate opportunities incrementally.

        Args:
            paper_id: UUID of the target persisted paper.
            session: Optional SQLAlchemy session (defaults to self.session).
            cutoff_date: Optional temporal evaluation cutoff. Papers published after or
                opportunities created after this date will be excluded.
            max_candidates: Optional limit on the number of candidate opportunities to evaluate.
                Overrides self.max_candidates if provided.
            candidate_opportunity_ids: Optional explicit list of opportunity UUIDs to evaluate against.
                If provided, filters candidate pool to these opportunities.

        Returns:
            An ``IncrementalEvolutionResult`` describing the complete evaluation outcome.
        """
        db = session or self.session
        if db is None:
            raise ValueError("A database session is required for incremental opportunity evolution.")

        # Ensure integration and transition service operate on the active session
        if getattr(self.integration, "_session", None) is not db:
            self.integration._session = db
        if getattr(self.integration, "_transition_service", None) is not None:
            if getattr(self.integration._transition_service, "session", None) is not db:
                self.integration._transition_service.session = db

        # ── 1. Resolve persisted paper and stored passages ─────────────────────────
        paper = db.execute(
            select(Paper).where(Paper.id == paper_id)
        ).scalar_one_or_none()

        if paper is None:
            logger.warning(f"Incremental evolution aborted: Paper {paper_id} not found in database.")
            return IncrementalEvolutionResult(
                paper_id=paper_id,
                run_status=IncrementalEvolutionRunStatus.PAPER_NOT_FOUND,
                error=f"Paper {paper_id} not found in database.",
            )

        stored_passages = db.execute(
            select(PaperPassage)
            .where(PaperPassage.paper_id == paper_id)
            .order_by(PaperPassage.passage_index)
        ).scalars().all()

        # ── 2. Validate paper publication date (temporal integrity) ───────────────
        if paper.publication_date is None:
            logger.info(f"Skipping incremental evolution: Paper {paper_id} is undated (publication_date is None).")
            return IncrementalEvolutionResult(
                paper_id=paper_id,
                run_status=IncrementalEvolutionRunStatus.SKIPPED_UNDATED,
                paper_title=paper.title,
                publication_date=None,
                error=f"Paper {paper_id} is undated (publication_date is None). Undated papers cannot be evaluated temporally.",
            )

        # Optional external cutoff date enforcement
        norm_cutoff: Optional[datetime.date] = None
        if cutoff_date is not None:
            from ideagen.discovery.temporal import normalize_cutoff_date
            try:
                norm_cutoff = normalize_cutoff_date(cutoff_date)
            except Exception as e:
                logger.error(f"Invalid cutoff date provided to evaluate_paper: {e}")
                return IncrementalEvolutionResult(
                    paper_id=paper_id,
                    run_status=IncrementalEvolutionRunStatus.FAILED,
                    paper_title=paper.title,
                    publication_date=paper.publication_date,
                    error=f"Invalid cutoff date: {e}",
                )

            if norm_cutoff is not None and paper.publication_date > norm_cutoff:
                logger.info(
                    f"Skipping paper {paper_id}: publication date {paper.publication_date.isoformat()} "
                    f"is after requested cutoff date {norm_cutoff.isoformat()}."
                )
                return IncrementalEvolutionResult(
                    paper_id=paper_id,
                    run_status=IncrementalEvolutionRunStatus.SKIPPED_UNDATED,
                    paper_title=paper.title,
                    publication_date=paper.publication_date,
                    error=(
                        f"Paper publication date {paper.publication_date.isoformat()} "
                        f"is after requested cutoff date {norm_cutoff.isoformat()}."
                    ),
                )

        # ── 3. Validate usable paper content ─────────────────────────────────────
        has_title = bool(paper.title and paper.title.strip())
        has_abstract = bool(paper.abstract and paper.abstract.strip())
        has_passage_text = any(bool(p.text and p.text.strip()) for p in stored_passages)

        if not has_title or (not has_abstract and not has_passage_text):
            logger.info(
                f"Skipping paper {paper_id}: lacks sufficient usable textual content "
                f"(has_title={has_title}, has_abstract={has_abstract}, has_passages={has_passage_text})."
            )
            return IncrementalEvolutionResult(
                paper_id=paper_id,
                run_status=IncrementalEvolutionRunStatus.SKIPPED_INSUFFICIENT_CONTENT,
                paper_title=paper.title,
                publication_date=paper.publication_date,
                error=(
                    f"Paper {paper_id} lacks sufficient usable textual content for relationship classification "
                    f"(must have title and either abstract or passage text)."
                ),
            )

        # ── 4. Query candidate opportunities deterministically ───────────────────
        query = select(Opportunity).where(
            Opportunity.status.in_([s.value for s in OpportunityStatus])
        )
        if candidate_opportunity_ids is not None:
            query = query.where(Opportunity.id.in_(candidate_opportunity_ids))

        # Initial deterministic ordering at query level
        query = query.order_by(Opportunity.created_at.asc(), Opportunity.id.asc())
        all_candidates = db.execute(query).scalars().all()

        # Deterministic chronological sort resolving effective creation dates
        sorted_candidates = sorted(
            all_candidates,
            key=lambda o: (
                get_opportunity_effective_creation_date(o) or datetime.date.min,
                str(o.id),
            ),
        )

        # The cap bounds classifier calls, so it is applied to candidates that pass every
        # eligibility check (step 5), not to the raw query result.
        limit_to_apply = max_candidates if max_candidates is not None else self.max_candidates
        if limit_to_apply is not None and limit_to_apply <= 0:
            limit_to_apply = None
        classified_slots = 0

        outcomes: List[OpportunityEvaluationOutcome] = []

        if not sorted_candidates:
            return IncrementalEvolutionResult(
                paper_id=paper_id,
                run_status=IncrementalEvolutionRunStatus.NO_CANDIDATES,
                paper_title=paper.title,
                publication_date=paper.publication_date,
                candidate_count=0,
                evaluated_count=0,
                skipped_count=0,
                grounding_rejected_count=0,
                failed_count=0,
                outcomes=[],
            )

        # ── 5. Evaluate each candidate opportunity atomically ────────────────────
        for opp in sorted_candidates:
            opp_initial_status: Optional[OpportunityStatus] = None
            try:
                opp_initial_status = OpportunityStatus(opp.status)
            except (ValueError, TypeError):
                outcomes.append(
                    OpportunityEvaluationOutcome(
                        opportunity_id=opp.id,
                        status=OpportunityEvaluationStatus.SKIPPED,
                        skip_reason=f"Opportunity status '{opp.status}' is not a valid lifecycle status.",
                    )
                )
                continue

            # 5a. Check effective creation date against paper publication date
            opp_date = get_opportunity_effective_creation_date(opp)
            if opp_date is None:
                outcomes.append(
                    OpportunityEvaluationOutcome(
                        opportunity_id=opp.id,
                        status=OpportunityEvaluationStatus.SKIPPED,
                        initial_status=opp_initial_status,
                        final_status=opp_initial_status,
                        skip_reason="Opportunity effective creation date is missing or undetermined.",
                    )
                )
                continue

            if opp_date > paper.publication_date:
                outcomes.append(
                    OpportunityEvaluationOutcome(
                        opportunity_id=opp.id,
                        status=OpportunityEvaluationStatus.SKIPPED,
                        initial_status=opp_initial_status,
                        final_status=opp_initial_status,
                        skip_reason=(
                            f"Opportunity effective creation date ({opp_date.isoformat()}) "
                            f"is after paper publication date ({paper.publication_date.isoformat()})."
                        ),
                    )
                )
                continue

            if norm_cutoff is not None and opp_date > norm_cutoff:
                outcomes.append(
                    OpportunityEvaluationOutcome(
                        opportunity_id=opp.id,
                        status=OpportunityEvaluationStatus.SKIPPED,
                        initial_status=opp_initial_status,
                        final_status=opp_initial_status,
                        skip_reason=(
                            f"Opportunity effective creation date ({opp_date.isoformat()}) "
                            f"is after requested cutoff date ({norm_cutoff.isoformat()})."
                        ),
                    )
                )
                continue

            # 5b. Check generation bridge eligibility if linked
            if opp.research_opportunity_id is not None and getattr(opp, "research_opportunity", None):
                gen_status = getattr(opp.research_opportunity, "status", None)
                if gen_status and str(gen_status).upper() not in ELIGIBLE_GENERATION_STATUSES:
                    outcomes.append(
                        OpportunityEvaluationOutcome(
                            opportunity_id=opp.id,
                            status=OpportunityEvaluationStatus.SKIPPED,
                            initial_status=opp_initial_status,
                            final_status=opp_initial_status,
                            skip_reason=(
                                f"Bridged research opportunity status '{gen_status}' is ineligible "
                                f"(must be in {ELIGIBLE_GENERATION_STATUSES})."
                            ),
                        )
                    )
                    continue

            # 5c. Idempotency: skip if already recorded in OpportunityEvidence
            existing_evidence = db.execute(
                select(OpportunityEvidence).where(
                    OpportunityEvidence.opportunity_id == opp.id,
                    OpportunityEvidence.paper_id == paper.id,
                )
            ).scalar_one_or_none()

            if existing_evidence is not None:
                outcomes.append(
                    OpportunityEvaluationOutcome(
                        opportunity_id=opp.id,
                        status=OpportunityEvaluationStatus.SKIPPED,
                        initial_status=opp_initial_status,
                        final_status=opp_initial_status,
                        skip_reason=f"Evidence already recorded for paper {paper.id} on opportunity {opp.id}.",
                    )
                )
                continue

            # 5d. Candidate limit, counted over eligible candidates only
            if limit_to_apply is not None and classified_slots >= limit_to_apply:
                outcomes.append(
                    OpportunityEvaluationOutcome(
                        opportunity_id=opp.id,
                        status=OpportunityEvaluationStatus.SKIPPED,
                        initial_status=opp_initial_status,
                        final_status=opp_initial_status,
                        skip_reason=f"Candidate limit ({limit_to_apply}) reached; not classified in this run.",
                    )
                )
                continue
            classified_slots += 1

            # 5e. Atomic evaluation inside nested savepoint
            savepoint = db.begin_nested()
            try:
                transition_res: ClassificationTransitionResult = self.integration.classify_and_apply(
                    opportunity_id=opp.id,
                    paper_id=paper.id,
                )
                savepoint.commit()

                outcomes.append(
                    OpportunityEvaluationOutcome(
                        opportunity_id=opp.id,
                        status=OpportunityEvaluationStatus.EVALUATED,
                        initial_status=transition_res.previous_status,
                        final_status=transition_res.new_status,
                        relationship=transition_res.relationship,
                        state_changed=transition_res.state_changed,
                        evidence_id=transition_res.evidence_id,
                        history_id=transition_res.history_id,
                    )
                )
            except UngroundedEvidenceError as uee:
                savepoint.rollback()
                logger.warning(
                    f"Rejecting ungrounded evidence for paper {paper.id} against opportunity {opp.id}: {uee}. "
                    f"Status transition discarded."
                )
                outcomes.append(
                    OpportunityEvaluationOutcome(
                        opportunity_id=opp.id,
                        status=OpportunityEvaluationStatus.GROUNDING_REJECTED,
                        initial_status=opp_initial_status,
                        final_status=opp_initial_status,
                        skip_reason=f"Ungrounded evidence rejected: {uee}",
                        error_message=str(uee),
                    )
                )
            except ValueError as ve:
                savepoint.rollback()
                if "already exists" in str(ve):
                    logger.info(
                        f"Skipping duplicate evidence for paper {paper.id} on opportunity {opp.id}: {ve}"
                    )
                    outcomes.append(
                        OpportunityEvaluationOutcome(
                            opportunity_id=opp.id,
                            status=OpportunityEvaluationStatus.SKIPPED,
                            initial_status=opp_initial_status,
                            final_status=opp_initial_status,
                            skip_reason=f"Duplicate evidence conflict: {ve}",
                        )
                    )
                else:
                    logger.error(
                        f"Validation error evaluating paper {paper.id} on opportunity {opp.id}: {ve}"
                    )
                    outcomes.append(
                        OpportunityEvaluationOutcome(
                            opportunity_id=opp.id,
                            status=OpportunityEvaluationStatus.FAILED,
                            initial_status=opp_initial_status,
                            final_status=opp_initial_status,
                            error_message=str(ve),
                        )
                    )
            except Exception as e:
                savepoint.rollback()
                logger.error(
                    f"Unexpected error evaluating paper {paper.id} on opportunity {opp.id}: {e}",
                    exc_info=True,
                )
                outcomes.append(
                    OpportunityEvaluationOutcome(
                        opportunity_id=opp.id,
                        status=OpportunityEvaluationStatus.FAILED,
                        initial_status=opp_initial_status,
                        final_status=opp_initial_status,
                        error_message=str(e),
                    )
                )

        # ── 6. Aggregate run summary and return ──────────────────────────────────
        candidate_count = len(sorted_candidates)
        evaluated_count = sum(1 for o in outcomes if o.status == OpportunityEvaluationStatus.EVALUATED)
        skipped_count = sum(1 for o in outcomes if o.status == OpportunityEvaluationStatus.SKIPPED)
        grounding_rejected_count = sum(1 for o in outcomes if o.status == OpportunityEvaluationStatus.GROUNDING_REJECTED)
        failed_count = sum(1 for o in outcomes if o.status == OpportunityEvaluationStatus.FAILED)

        if candidate_count == 0:
            run_status = IncrementalEvolutionRunStatus.NO_CANDIDATES
        elif failed_count == 0:
            run_status = IncrementalEvolutionRunStatus.SUCCESS
        elif evaluated_count > 0:
            run_status = IncrementalEvolutionRunStatus.PARTIAL_SUCCESS
        else:
            run_status = IncrementalEvolutionRunStatus.FAILED

        return IncrementalEvolutionResult(
            paper_id=paper.id,
            run_status=run_status,
            paper_title=paper.title,
            publication_date=paper.publication_date,
            candidate_count=candidate_count,
            evaluated_count=evaluated_count,
            skipped_count=skipped_count,
            grounding_rejected_count=grounding_rejected_count,
            failed_count=failed_count,
            outcomes=outcomes,
        )
