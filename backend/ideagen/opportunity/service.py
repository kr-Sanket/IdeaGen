from typing import Optional
import uuid
from sqlalchemy.orm import Session
from sqlalchemy import select

from ideagen.db.models import Opportunity, Paper, OpportunityEvidence, OpportunityHistory
from ideagen.opportunity.models import OpportunityRelationship, OpportunityStatus
from ideagen.opportunity.state_machine import transition

class OpportunityTransitionService:
    def __init__(self, session: Session):
        self.session = session

    def apply_relationship(
        self,
        opportunity_id: uuid.UUID,
        paper_id: uuid.UUID,
        relationship: OpportunityRelationship,
        evidence_excerpt: Optional[str] = None
    ) -> Opportunity:
        """
        Applies a relationship classification to an Opportunity deterministically.
        Uses SELECT ... FOR UPDATE to prevent concurrent modifications.
        Idempotency: Rejects duplicate applications of the same paper-opportunity pair.
        """
        # Validate existence and lock the opportunity row
        opportunity = self.session.execute(
            select(Opportunity)
            .where(Opportunity.id == opportunity_id)
            .with_for_update()
        ).scalar_one_or_none()

        if opportunity is None:
            raise ValueError(f"Opportunity {opportunity_id} not found.")

        # Validate paper exists
        paper = self.session.execute(
            select(Paper).where(Paper.id == paper_id)
        ).scalar_one_or_none()
        
        if paper is None:
            raise ValueError(f"Paper {paper_id} not found.")
            
        # Check for idempotency: if this exact relationship from this paper is already applied, we reject to avoid duplicates.
        # Actually, let's just reject if ANY evidence exists from this paper for this opportunity. 
        # A paper shouldn't provide multiple conflicting evidences.
        existing_evidence = self.session.execute(
            select(OpportunityEvidence)
            .where(
                OpportunityEvidence.opportunity_id == opportunity_id,
                OpportunityEvidence.paper_id == paper_id
            )
        ).scalar_one_or_none()
        
        if existing_evidence is not None:
            raise ValueError(f"Evidence from Paper {paper_id} already exists for Opportunity {opportunity_id}.")

        previous_status = OpportunityStatus(opportunity.status)
        new_status = transition(previous_status, relationship)

        # 1. Create Evidence
        evidence = OpportunityEvidence(
            opportunity_id=opportunity_id,
            paper_id=paper_id,
            evidence_type=relationship.value,
            supporting_text=evidence_excerpt
        )
        self.session.add(evidence)
        self.session.flush()

        # 2. Update Status if changed
        if new_status != previous_status:
            opportunity.status = new_status.value
            
            # 3. Create History
            history = OpportunityHistory(
                opportunity_id=opportunity_id,
                previous_status=previous_status.value,
                new_status=new_status.value,
                trigger_type=relationship.value,
                evidence_id=evidence.id
            )
            self.session.add(history)
            
        # The caller is responsible for committing the session (this ensures it is part of a larger unit of work if needed).
        # We flush to ensure constraints are checked immediately
        self.session.flush()

        return opportunity
