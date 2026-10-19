import re
from typing import Optional, Sequence, Union
import uuid
from sqlalchemy.orm import Session
from sqlalchemy import select

from ideagen.db.models import Opportunity, Paper, PaperPassage, OpportunityEvidence, OpportunityHistory
from ideagen.opportunity.models import OpportunityRelationship, OpportunityStatus
from ideagen.opportunity.state_machine import transition
from ideagen.opportunity.grounding import (
    is_excerpt_grounded,
    validate_evidence_grounding,
    SUBSTANTIVE_RELATIONSHIPS,
    UngroundedEvidenceError,
)


def get_paper_canonical_identity(paper: Paper) -> str:
    """Extracts a canonical identity string for safe paper deduplication.

    Uses the strongest safe identity fields available in the schema:
    1. Canonical DOI (normalized: lowercase, stripped)
    2. Normalized Title (lowercase, stripped of punctuation and extra whitespace)
    3. Source identity (source_type:source_id)
    4. Fallback to paper UUID

    Limitation: True scientific independence (different lab, no co-authors,
    or preprint-to-journal publication lineage) cannot be fully established
    without author entity disambiguation and complete citation graph tracking.
    We implement the strongest safe deduplication possible using the existing schema.
    """
    if paper.doi and paper.doi.strip():
        return f"doi:{paper.doi.strip().lower()}"

    if paper.title and paper.title.strip():
        norm_title = re.sub(r"[^\w\s]", "", paper.title.lower())
        norm_title = re.sub(r"\s+", " ", norm_title).strip()
        if len(norm_title) >= 5:
            return f"title:{norm_title}"

    if paper.source_type and paper.source_id:
        return f"source:{paper.source_type}:{paper.source_id.strip()}"

    return f"id:{str(paper.id)}"


def is_evidence_grounded_for_paper(
    evidence: OpportunityEvidence,
    paper: Paper,
    passages: Sequence[Union[str, any]],
) -> bool:
    """Verifies that an evidence row represents verified grounded literature evidence.

    Ungrounded excerpts or evidences without supporting text cannot contribute
    towards multi-paper resolution.
    """
    if not evidence.supporting_text or not evidence.supporting_text.strip():
        return False
    if passages:
        return is_excerpt_grounded(evidence.supporting_text, passages)
    if paper.abstract:
        return is_excerpt_grounded(evidence.supporting_text, [paper.abstract])
    return False


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
        Deterministic Resolution: Requires >= 2 distinct papers with independently
        grounded ADDRESSES evidence to reach the RESOLVED state.
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
        existing_evidence = self.session.execute(
            select(OpportunityEvidence)
            .where(
                OpportunityEvidence.opportunity_id == opportunity_id,
                OpportunityEvidence.paper_id == paper_id
            )
        ).scalar_one_or_none()
        
        if existing_evidence is not None:
            raise ValueError(f"Evidence from Paper {paper_id} already exists for Opportunity {opportunity_id}.")

        # Fail-closed evidence grounding validation
        stored_passages = self.session.execute(
            select(PaperPassage)
            .where(PaperPassage.paper_id == paper_id)
            .order_by(PaperPassage.passage_index)
        ).scalars().all()

        if stored_passages and relationship in SUBSTANTIVE_RELATIONSHIPS:
            validate_evidence_grounding(
                excerpt=evidence_excerpt,
                passages=stored_passages,
                relationship=relationship,
                paper_id=paper_id,
            )
        elif evidence_excerpt is not None and relationship in SUBSTANTIVE_RELATIONSHIPS:
            sources = [paper.abstract] if paper.abstract else []
            if sources:
                validate_evidence_grounding(
                    excerpt=evidence_excerpt,
                    passages=sources,
                    relationship=relationship,
                    paper_id=paper_id,
                )
            else:
                raise UngroundedEvidenceError(
                    f"Evidence excerpt cannot be grounded: paper {paper_id} has neither stored passages nor abstract."
                )

        # 1. Create Evidence and flush to persist in current transaction
        evidence = OpportunityEvidence(
            opportunity_id=opportunity_id,
            paper_id=paper_id,
            evidence_type=relationship.value,
            supporting_text=evidence_excerpt
        )
        self.session.add(evidence)
        self.session.flush()

        # 2. Determine addressing paper count for multi-paper resolution rule
        addressing_paper_count = 1
        if relationship == OpportunityRelationship.ADDRESSES:
            addressing_evidences = self.session.execute(
                select(OpportunityEvidence)
                .where(
                    OpportunityEvidence.opportunity_id == opportunity_id,
                    OpportunityEvidence.evidence_type == OpportunityRelationship.ADDRESSES.value,
                )
            ).scalars().all()
            ev_paper_ids = list({ev.paper_id for ev in addressing_evidences})
            if ev_paper_ids:
                papers = self.session.execute(
                    select(Paper).where(Paper.id.in_(ev_paper_ids))
                ).scalars().all()
                paper_map = {p.id: p for p in papers}

                passages = self.session.execute(
                    select(PaperPassage).where(PaperPassage.paper_id.in_(ev_paper_ids))
                ).scalars().all()
                passage_map: dict[uuid.UUID, list[PaperPassage]] = {}
                for psg in passages:
                    passage_map.setdefault(psg.paper_id, []).append(psg)

                distinct_identities = set()
                for ev in addressing_evidences:
                    p = paper_map.get(ev.paper_id)
                    if not p:
                        continue
                    p_passages = passage_map.get(ev.paper_id, [])
                    # Count only verified grounded evidence
                    if is_evidence_grounded_for_paper(ev, p, p_passages):
                        distinct_identities.add(get_paper_canonical_identity(p))

                addressing_paper_count = len(distinct_identities)

        previous_status = OpportunityStatus(opportunity.status)
        new_status = transition(
            previous_status,
            relationship,
            addressing_paper_count=addressing_paper_count,
        )

        # 3. Update Status if changed
        if new_status != previous_status:
            opportunity.status = new_status.value
            
            # 4. Create History
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
