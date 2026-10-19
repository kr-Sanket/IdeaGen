import datetime
import uuid
from dataclasses import dataclass
from typing import List, Optional

from sqlalchemy.orm import Session

from ideagen.db.models import Opportunity, Paper
from ideagen.opportunity.models import OpportunityRelationship, OpportunityStatus
from ideagen.opportunity.integration import OpportunityClassificationIntegration

@dataclass(frozen=True)
class TemporalBatch:
    batch_id: str
    start_date: datetime.date
    end_date: datetime.date
    sequence_number: int

@dataclass(frozen=True)
class EvolutionEvent:
    batch_id: str
    paper_id: uuid.UUID
    publication_date: datetime.date
    relationship: OpportunityRelationship
    previous_status: OpportunityStatus
    new_status: OpportunityStatus
    state_changed: bool
    evidence_id: Optional[uuid.UUID]
    history_id: Optional[uuid.UUID]
    directional_extension_detected: bool = False

@dataclass(frozen=True)
class OpportunityEvolutionResult:
    opportunity_id: uuid.UUID
    initial_status: OpportunityStatus
    final_status: OpportunityStatus
    events: List[EvolutionEvent]
    processed_paper_count: int
    skipped_paper_count: int
    grounding_rejections_count: int = 0

class OpportunityEvolutionService:
    def __init__(self, integration_service: OpportunityClassificationIntegration):
        self.integration = integration_service

    def process_batch(
        self,
        db: Session,
        opportunity_id: uuid.UUID,
        batch: TemporalBatch,
        papers: List[Paper],
        chunk_size: Optional[int] = None,
    ) -> OpportunityEvolutionResult:
        if batch.start_date > batch.end_date:
            raise ValueError(f"Invalid date range for batch {batch.batch_id}")

        opp = db.query(Opportunity).filter(Opportunity.id == opportunity_id).first()
        if not opp:
            raise ValueError("Opportunity not found")

        initial_status = OpportunityStatus(opp.status)
        events = []
        skipped = 0
        
        eligible_papers = []
        for p in papers:
            if p.publication_date is None:
                skipped += 1
                continue
            if not (batch.start_date <= p.publication_date <= batch.end_date):
                skipped += 1
                continue
            eligible_papers.append(p)
            
        # Deterministic chronological order
        eligible_papers.sort(key=lambda p: (p.publication_date, str(p.id)))
        
        from sqlalchemy import select
        from ideagen.db.models import OpportunityEvidence
        from ideagen.opportunity.grounding import UngroundedEvidenceError
        import logging
        logger = logging.getLogger(__name__)

        processed_count = 0
        grounding_rejections = 0

        chunks: List[List[Paper]] = (
            [eligible_papers[i:i + chunk_size] for i in range(0, len(eligible_papers), chunk_size)]
            if chunk_size and chunk_size > 0
            else [eligible_papers]
        )

        for chunk_idx, chunk in enumerate(chunks):
            if len(chunks) > 1:
                logger.info(
                    f"Processing batch {batch.batch_id} chunk {chunk_idx + 1}/{len(chunks)} "
                    f"({len(chunk)} papers) for opportunity {opportunity_id}"
                )
            for p in chunk:
                # Duplicate-safe replay: skip already recorded (opportunity, paper) pairs deterministically
                existing_evidence = db.execute(
                    select(OpportunityEvidence).where(
                        OpportunityEvidence.opportunity_id == opportunity_id,
                        OpportunityEvidence.paper_id == p.id
                    )
                ).scalar_one_or_none()

                if existing_evidence is not None:
                    skipped += 1
                    logger.info(
                        f"Skipping paper {p.id}: evidence already recorded for opportunity {opportunity_id}."
                    )
                    continue

                savepoint = db.begin_nested()
                try:
                    res = self.integration.classify_and_apply(
                        opportunity_id=opportunity_id,
                        paper_id=p.id
                    )
                    savepoint.commit()
                    processed_count += 1
                except UngroundedEvidenceError as ev_err:
                    savepoint.rollback()
                    skipped += 1
                    grounding_rejections += 1
                    logger.warning(
                        f"Rejecting ungrounded evidence for paper {p.id} against opportunity {opportunity_id}: {ev_err}. "
                        f"Status transition discarded."
                    )
                    continue
                except Exception as e:
                    savepoint.rollback()
                    raise e

                directional_ext = (res.relationship == OpportunityRelationship.EXTENDS)

                events.append(EvolutionEvent(
                    batch_id=batch.batch_id,
                    paper_id=res.paper_id,
                    publication_date=p.publication_date,
                    relationship=res.relationship,
                    previous_status=res.previous_status,
                    new_status=res.new_status,
                    state_changed=res.state_changed,
                    evidence_id=res.evidence_id,
                    history_id=res.history_id,
                    directional_extension_detected=directional_ext
                ))

        # Refetch opp to get final status
        opp_final = db.query(Opportunity).filter(Opportunity.id == opportunity_id).first()
        final_status = OpportunityStatus(opp_final.status)

        return OpportunityEvolutionResult(
            opportunity_id=opportunity_id,
            initial_status=initial_status,
            final_status=final_status,
            events=events,
            processed_paper_count=processed_count,
            skipped_paper_count=skipped,
            grounding_rejections_count=grounding_rejections,
        )

    def replay(
        self,
        db: Session,
        opportunity_id: uuid.UUID,
        batches: List[TemporalBatch],
        papers: List[Paper]
    ) -> List[OpportunityEvolutionResult]:
        """Replay processing over multiple sequential batches."""
        # Ensure chronological batch order
        sorted_batches = sorted(batches, key=lambda b: b.sequence_number)
        results = []
        for b in sorted_batches:
            res = self.process_batch(db, opportunity_id, b, papers)
            results.append(res)
        return results
