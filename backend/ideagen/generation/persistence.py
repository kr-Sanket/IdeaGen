"""Database Persistence Layer for Phase 5 Research Opportunities (SQLAlchemy).

Persists ResearchOpportunityRecord, PriorArtAssessmentRecord, and OpportunityRefinementRecord
records to PostgreSQL / SQLite database with full evidence relationship integrity.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Dict, Any
import uuid
from datetime import datetime
from sqlalchemy.orm import Session

from ideagen.db.models import (
    ResearchGap,
    ResearchOpportunityRecord,
    PriorArtAssessmentRecord,
    OpportunityRefinementRecord,
)
from ideagen.generation.models import (
    ResearchOpportunityDTO,
    PriorArtAssessment,
    OpportunityStatus,
    FeasibilityAssessment,
    NoveltyAssessment,
)

logger = logging.getLogger(__name__)


class OpportunityPersistenceLayer:
    """SQLAlchemy persistence manager for Phase 5 Research Opportunities."""

    def __init__(self, session: Session):
        self.session = session

    def save_opportunity(
        self,
        dto: ResearchOpportunityDTO,
        assessments: Optional[List[Tuple[int, PriorArtAssessment]]] = None,
        iterations: Optional[List[Tuple[int, str, str, str, str]]] = None,
        bridge: bool = True,
    ) -> ResearchOpportunityRecord:
        """Persists a ResearchOpportunityDTO and associated prior-art assessments & refinement records.

        With ``bridge=True`` the generation record and its lifecycle bridge are one atomic unit:
        if bridging fails, the unit is rolled back to a savepoint (leaving any earlier session
        state untouched and the session usable) and the original exception is re-raised, so a
        generation record is never committed without its lifecycle Opportunity.
        """
        if not bridge:
            record = self._add_records(dto, assessments, iterations)
            self.session.commit()
            self.session.refresh(record)
            return record

        from ideagen.opportunity.bridge import OpportunityBridgeService

        savepoint = self.session.begin_nested()
        try:
            record = self._add_records(dto, assessments, iterations)
            OpportunityBridgeService(self.session).bridge_opportunity(record)
            savepoint.commit()
        except Exception:
            savepoint.rollback()
            logger.error(
                f"Failed to persist and bridge ResearchOpportunityRecord {dto.id}; "
                "the generation record was not saved."
            )
            raise

        self.session.commit()
        self.session.refresh(record)
        return record

    def _add_records(
        self,
        dto: ResearchOpportunityDTO,
        assessments: Optional[List[Tuple[int, PriorArtAssessment]]],
        iterations: Optional[List[Tuple[int, str, str, str, str]]],
    ) -> ResearchOpportunityRecord:
        """Adds the generation record, prior-art assessments and refinement records to the session."""
        # Convert Pydantic feasibility & novelty dicts
        feasibility_dict = dto.feasibility.model_dump()
        novelty_dict = dto.novelty_assessment.model_dump(mode="json")

        # The opportunity belongs to its source gap's project, set atomically with the record.
        source_gap = self.session.get(ResearchGap, dto.gap_id) if dto.gap_id else None
        record = ResearchOpportunityRecord(
            id=dto.id,
            project_id=source_gap.project_id if source_gap is not None else None,
            gap_id=dto.gap_id,
            research_problem=dto.research_problem,
            identified_gap=dto.identified_gap,
            proposed_research_direction=dto.proposed_research_direction,
            reasoning=dto.reasoning,
            technical_approach=dto.technical_approach,
            expected_contribution=dto.expected_contribution,
            feasibility=feasibility_dict,
            novelty_assessment=novelty_dict,
            confidence=dto.confidence,
            supporting_statement_ids=[str(i) for i in dto.supporting_statement_ids],
            supporting_paper_ids=[str(i) for i in dto.supporting_paper_ids],
            prior_art_statement_ids=[str(i) for i in dto.prior_art_statement_ids],
            status=dto.status.value if isinstance(dto.status, OpportunityStatus) else str(dto.status),
            generation_model=dto.generation_model,
            validation_model=dto.validation_model,
            embedding_model=dto.embedding_model,
            refinement_iteration_count=dto.refinement_iteration_count,
            pipeline_version=dto.pipeline_version,
            created_at=dto.created_at,
        )

        self.session.add(record)

        # Save prior-art assessments
        if assessments:
            for iter_num, assessment in assessments:
                assessment_rec = PriorArtAssessmentRecord(
                    id=uuid.uuid4(),
                    opportunity_id=record.id,
                    iteration=iter_num,
                    assessment=assessment.assessment,
                    overlapping_statement_ids=[str(i) for i in assessment.overlapping_statement_ids],
                    overlapping_paper_ids=[str(i) for i in assessment.overlapping_paper_ids],
                    reasoning=assessment.reasoning,
                    differentiation=assessment.differentiation,
                    confidence=assessment.confidence,
                )
                self.session.add(assessment_rec)

        # Save refinement iterations
        if iterations:
            for iter_num, orig_dir, critique_text, rev_dir, iter_status in iterations:
                iter_rec = OpportunityRefinementRecord(
                    id=uuid.uuid4(),
                    opportunity_id=record.id,
                    iteration=iter_num,
                    original_direction=orig_dir,
                    critique=critique_text,
                    revised_direction=rev_dir,
                    status=iter_status,
                )
                self.session.add(iter_rec)

        return record

    def get_opportunity_by_id(self, opportunity_id: uuid.UUID) -> Optional[ResearchOpportunityRecord]:
        """Retrieves a ResearchOpportunityRecord by UUID."""
        return self.session.query(ResearchOpportunityRecord).filter(
            ResearchOpportunityRecord.id == opportunity_id
        ).first()

    def list_opportunities_for_gap(self, gap_id: uuid.UUID) -> List[ResearchOpportunityRecord]:
        """Lists all persisted ResearchOpportunityRecords for a given gap_id."""
        return self.session.query(ResearchOpportunityRecord).filter(
            ResearchOpportunityRecord.gap_id == gap_id
        ).all()
