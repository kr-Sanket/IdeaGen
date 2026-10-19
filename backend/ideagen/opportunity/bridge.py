"""Opportunity Bridge Service.

Connects validated production ResearchOpportunityRecord objects (generation pipeline)
to the Opportunity lifecycle entity (evolution subsystem) using an explicit, idempotent bridge.
"""

from __future__ import annotations

import logging
from typing import Optional, Set, Union
import uuid

from sqlalchemy.orm import Session
from sqlalchemy import select

from ideagen.db.models import Opportunity, ResearchOpportunityRecord
from ideagen.opportunity.models import OpportunityStatus as LifecycleStatus
from ideagen.generation.models import (
    ResearchOpportunityDTO,
    OpportunityStatus as GenerationStatus,
)

logger = logging.getLogger(__name__)

# Generation statuses explicitly approved and eligible for lifecycle entity creation
ELIGIBLE_GENERATION_STATUSES: Set[str] = {
    GenerationStatus.VALIDATED.value,
    GenerationStatus.REFINED.value,
}


class OpportunityBridgeService:
    """Explicit, idempotent bridge connecting production generation artifacts to lifecycle entities."""

    def __init__(self, session: Optional[Session] = None):
        self.session = session

    def is_eligible(self, record_or_dto: Union[ResearchOpportunityRecord, ResearchOpportunityDTO]) -> bool:
        """Determines if a generation record is eligible for bridging into the lifecycle subsystem."""
        raw_status = getattr(record_or_dto, "status", None)
        if raw_status is None:
            return False
        status_str = raw_status.value if hasattr(raw_status, "value") else str(raw_status)
        return status_str.upper() in ELIGIBLE_GENERATION_STATUSES

    def bridge_opportunity(
        self,
        record: Union[ResearchOpportunityRecord, ResearchOpportunityDTO],
        session: Optional[Session] = None,
        raise_on_ineligible: bool = False,
    ) -> Optional[Opportunity]:
        """Bridges a validated generation record to the lifecycle Opportunity entity.

        Args:
            record: ResearchOpportunityRecord or ResearchOpportunityDTO from generation.
            session: Optional SQLAlchemy session (defaults to self.session).
            raise_on_ineligible: If True, raises ValueError for ineligible generation status.
                                 If False (default), safely ignores and returns None.

        Returns:
            Opportunity lifecycle entity if bridged or existing, or None if ineligible.
        """
        db = session or self.session
        if db is None:
            raise ValueError("A database session is required to bridge opportunities.")

        source_id = getattr(record, "id", None)
        if source_id is None:
            raise ValueError("Source record must have a valid id.")

        # 1. Idempotency check: if lifecycle entity already exists for this source record, return it untouched
        existing = db.execute(
            select(Opportunity).where(Opportunity.research_opportunity_id == source_id)
        ).scalar_one_or_none()

        if existing is not None:
            logger.debug(
                f"Opportunity for research_opportunity_id={source_id} already exists "
                f"(id={existing.id}, status={existing.status}). Returning existing entity."
            )
            return existing

        # 2. Check eligibility: only VALIDATED or REFINED generation statuses are eligible
        if not self.is_eligible(record):
            status_val = getattr(record, "status", None)
            if raise_on_ineligible:
                raise ValueError(
                    f"Research opportunity {source_id} has ineligible status '{status_val}'. "
                    f"Eligible statuses are {ELIGIBLE_GENERATION_STATUSES}."
                )
            logger.info(
                f"Skipping bridge for research_opportunity {source_id}: "
                f"status '{status_val}' is ineligible (must be in {ELIGIBLE_GENERATION_STATUSES})."
            )
            return None

        # 3. Safe field mapping without inventing facts
        direction = getattr(record, "proposed_research_direction", None)
        gap_stmt = getattr(record, "identified_gap", None)
        title = (direction or gap_stmt or "Research Opportunity")[:255].strip()

        problem_stmt = getattr(record, "research_problem", "") or ""

        # Construct factual description from available source fields
        desc_parts = []
        if gap_stmt and str(gap_stmt).strip():
            desc_parts.append(f"Identified Gap: {str(gap_stmt).strip()}")
        reasoning = getattr(record, "reasoning", None)
        if reasoning and str(reasoning).strip():
            desc_parts.append(f"Reasoning: {str(reasoning).strip()}")
        tech_approach = getattr(record, "technical_approach", None)
        if tech_approach and str(tech_approach).strip():
            desc_parts.append(f"Technical Approach: {str(tech_approach).strip()}")
        expected_contribution = getattr(record, "expected_contribution", None)
        if expected_contribution and str(expected_contribution).strip():
            desc_parts.append(f"Expected Contribution: {str(expected_contribution).strip()}")

        description = "\n\n".join(desc_parts) if desc_parts else problem_stmt

        # Bounded confidence in [0.0, 1.0]
        raw_conf = getattr(record, "confidence", 0.0)
        try:
            conf_val = float(raw_conf) if raw_conf is not None else 0.0
        except (ValueError, TypeError):
            conf_val = 0.0
        confidence = max(0.0, min(1.0, conf_val))

        gap_id = getattr(record, "gap_id", None)
        rec_created_at = getattr(record, "created_at", None)

        opp_kwargs = {
            "id": uuid.uuid4(),
            "research_opportunity_id": source_id,
            "gap_id": gap_id,
            "title": title,
            "problem_statement": problem_stmt,
            "description": description,
            "status": LifecycleStatus.OPEN.value,
            "confidence": confidence,
        }
        if rec_created_at is not None:
            opp_kwargs["created_at"] = rec_created_at

        lifecycle_opp = Opportunity(**opp_kwargs)

        db.add(lifecycle_opp)
        db.flush()

        logger.info(
            f"Successfully bridged ResearchOpportunityRecord {source_id} "
            f"to Opportunity {lifecycle_opp.id} (status={lifecycle_opp.status})."
        )
        return lifecycle_opp
