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

class OpportunityEvolutionService:
    def __init__(self, integration_service: OpportunityClassificationIntegration):
        self.integration = integration_service

    def process_batch(
        self,
        db: Session,
        opportunity_id: uuid.UUID,
        batch: TemporalBatch,
        papers: List[Paper]
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
        
        for p in eligible_papers:
            try:
                res = self.integration.classify_and_apply(
                    opportunity_id=opportunity_id,
                    paper_id=p.id
                )
            except Exception as e:
                # Existing AD-22 deduplication might raise ValueError depending on how transition service handles it,
                # but typically if transition fails, we either skip or propagate.
                # Let's propagate it unless it's a known skip. The requirements say respect idempotency conflict.
                # If OpportunityTransitionService raises ValueError on duplicate evidence, we skip or bubble up.
                # We will just let it bubble up, tests will verify atomicity/failure behavior.
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
            processed_paper_count=len(eligible_papers),
            skipped_paper_count=skipped
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
