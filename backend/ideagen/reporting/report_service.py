"""Opportunity Report Generator Service (Phase 6).

Compiles the 16-section Final Opportunity Report from database models (ResearchOpportunityRecord,
ResearchGap, ResearchStatement, PaperPassage, and Paper).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Dict, Any
import uuid
import datetime
from sqlalchemy.orm import Session
from sqlalchemy import select

from ideagen.db.models import (
    ResearchOpportunityRecord,
    ResearchGap,
    ResearchStatement,
    PaperPassage,
    Paper,
    ResearchKnowledge,
    GapEvidence,
)
from ideagen.api.schema import (
    OpportunityReportDTO,
    EvidenceDetailDTO,
    PaperResponseDTO,
    FeasibilityDTO,
    NoveltyDTO,
)
from ideagen.generation.validator import sanitize_novelty_language

logger = logging.getLogger(__name__)


class OpportunityReportService:
    """Service for generating comprehensive, evidence-grounded research opportunity reports."""

    def __init__(self, session: Session):
        self.session = session

    def generate_opportunity_report(self, opportunity_id: uuid.UUID) -> Optional[OpportunityReportDTO]:
        """Compiles the complete 16-section opportunity report for a given opportunity_id."""
        opportunity: Optional[ResearchOpportunityRecord] = self.session.query(ResearchOpportunityRecord).filter(
            ResearchOpportunityRecord.id == opportunity_id
        ).first()

        if not opportunity:
            return None

        # 1. Fetch parent ResearchGap
        gap: Optional[ResearchGap] = self.session.query(ResearchGap).filter(
            ResearchGap.id == opportunity.gap_id
        ).first()

        gap_statement = gap.gap_statement if gap else opportunity.identified_gap
        gap_limitations = gap.limitations if gap else []

        # 2. Gather supporting statement IDs
        statement_ids_set = set()
        if opportunity.supporting_statement_ids:
            for s_id in opportunity.supporting_statement_ids:
                try:
                    statement_ids_set.add(uuid.UUID(str(s_id)))
                except (ValueError, TypeError):
                    pass

        if gap:
            gap_evidence = self.session.query(GapEvidence).filter(GapEvidence.gap_id == gap.id).all()
            for ge in gap_evidence:
                statement_ids_set.add(ge.statement_id)

        # 3. Query supporting ResearchStatement records
        supporting_statements: List[ResearchStatement] = []
        if statement_ids_set:
            supporting_statements = self.session.query(ResearchStatement).filter(
                ResearchStatement.id.in_(list(statement_ids_set))
            ).all()

        # 4. Map evidence details
        evidence_of_problem: List[EvidenceDetailDTO] = []
        supporting_evidence: List[EvidenceDetailDTO] = []
        supporting_paper_ids_set = set()

        for stmt in supporting_statements:
            paper = self.session.query(Paper).filter(Paper.id == stmt.paper_id).first()
            paper_title = paper.title if paper and paper.title else f"Paper {stmt.paper_id.hex[:8]}"
            if paper:
                supporting_paper_ids_set.add(paper.id)

            passage: Optional[PaperPassage] = None
            if stmt.passage_id:
                passage = self.session.query(PaperPassage).filter(PaperPassage.id == stmt.passage_id).first()

            detail = EvidenceDetailDTO(
                statement_id=stmt.id,
                statement_text=stmt.statement_text,
                statement_type=stmt.statement_type,
                paper_id=stmt.paper_id,
                paper_title=paper_title,
                passage_id=stmt.passage_id,
                section=stmt.section or (passage.section if passage else None),
                page_start=stmt.page_start or (passage.page_start if passage else None),
                page_end=stmt.page_end or (passage.page_end if passage else None),
                passage_text=passage.text if passage else None,
            )
            supporting_evidence.append(detail)
            if stmt.statement_type in {"LIMITATION", "FAILURE_MODE", "PROBLEM"}:
                evidence_of_problem.append(detail)

        if not evidence_of_problem and supporting_evidence:
            evidence_of_problem = supporting_evidence[:3]

        # 5. Gather paper DTOs
        if opportunity.supporting_paper_ids:
            for p_id in opportunity.supporting_paper_ids:
                try:
                    supporting_paper_ids_set.add(uuid.UUID(str(p_id)))
                except (ValueError, TypeError):
                    pass

        supporting_papers_list: List[PaperResponseDTO] = []
        if supporting_paper_ids_set:
            papers = self.session.query(Paper).filter(Paper.id.in_(list(supporting_paper_ids_set))).all()
            for p in papers:
                doc_count = len(p.documents) if p.documents else 0
                passages_count = 0
                if p.documents:
                    for d in p.documents:
                        passages_count += len(d.passages) if d.passages else 0

                knowledge_status = "ANALYZED" if p.research_knowledge else "UNANALYZED"
                pub_date_str = str(p.publication_date) if p.publication_date else None
                created_str = p.created_at.isoformat() if p.created_at else None

                supporting_papers_list.append(
                    PaperResponseDTO(
                        id=p.id,
                        source_type=p.source_type,
                        source_id=p.source_id,
                        title=p.title or f"Paper {p.source_id}",
                        publication_date=pub_date_str,
                        doi=p.doi,
                        abstract=p.abstract,
                        created_at=created_str,
                        document_count=doc_count,
                        passage_count=passages_count,
                        knowledge_status=knowledge_status,
                    )
                )

        # 6. Existing Approaches
        existing_approaches: List[str] = []
        for stmt in supporting_statements:
            if stmt.statement_type in {"METHOD", "METHODOLOGY", "BASELINE"}:
                existing_approaches.append(stmt.statement_text)

        # No template fallback: an empty list means no method statements support this opportunity.

        # 7. Why Existing Work Is Insufficient
        why_insufficient_parts = []
        if gap_limitations:
            why_insufficient_parts.append(f"Recurring limitations identified across papers: {'; '.join(gap_limitations)}.")
        for stmt in supporting_statements:
            if stmt.statement_type == "LIMITATION":
                why_insufficient_parts.append(stmt.statement_text)

        why_insufficient_text = " ".join(why_insufficient_parts) if why_insufficient_parts else None

        # 8. Feasibility DTO
        feasibility_dict = opportunity.feasibility or {}
        # Values come from the stored (model-generated) assessment only; absent keys stay None.
        feasibility_dto = FeasibilityDTO(
            required_data=feasibility_dict.get("required_data"),
            required_compute=feasibility_dict.get("required_compute"),
            implementation_complexity=feasibility_dict.get("implementation_complexity"),
            experimental_setup=feasibility_dict.get("experimental_setup"),
            feasibility_score=feasibility_dict.get("feasibility_score"),
            summary=feasibility_dict.get("summary"),
        )

        # 9. Novelty Assessment DTO
        novelty_dict = opportunity.novelty_assessment or {}
        novelty_dto = NoveltyDTO(
            assessment=sanitize_novelty_language(novelty_dict.get("assessment")),
            reasoning=sanitize_novelty_language(novelty_dict.get("reasoning")),
            differentiation=sanitize_novelty_language(novelty_dict.get("differentiation")),
            confidence=novelty_dict.get("confidence"),
        )

        # 10. Prior Art Assessment DTO
        prior_art_dict = {
            "assessment": novelty_dict.get("assessment"),
            "reasoning": novelty_dict.get("reasoning"),
            "differentiation": novelty_dict.get("differentiation"),
            "confidence": novelty_dict.get("confidence"),
        }

        differentiation_text = novelty_dict.get("differentiation") or None

        created_at_str = opportunity.created_at.isoformat() if opportunity.created_at else datetime.datetime.now(datetime.timezone.utc).isoformat()
        lifecycle_status = opportunity.lifecycle_opportunity.status if opportunity.lifecycle_opportunity else None
        lifecycle_opp_id = opportunity.lifecycle_opportunity.id if opportunity.lifecycle_opportunity else None

        return OpportunityReportDTO(
            opportunity_id=opportunity.id,
            gap_id=opportunity.gap_id,
            research_problem=opportunity.research_problem,
            evidence_of_problem=evidence_of_problem,
            existing_approaches=existing_approaches,
            identified_gap=gap_statement,
            why_existing_work_is_insufficient=why_insufficient_text,
            proposed_research_direction=opportunity.proposed_research_direction,
            reasoning=opportunity.reasoning,
            technical_approach=opportunity.technical_approach,
            expected_contribution=opportunity.expected_contribution,
            feasibility=feasibility_dto,
            prior_art_assessment=prior_art_dict,
            differentiation=differentiation_text,
            supporting_papers=supporting_papers_list,
            supporting_evidence=supporting_evidence,
            confidence=opportunity.confidence,
            novelty_assessment=novelty_dto,
            status=opportunity.status,
            created_at=created_at_str,
            generation_model=opportunity.generation_model,
            validation_model=opportunity.validation_model,
            embedding_model=opportunity.embedding_model,
            refinement_iteration_count=opportunity.refinement_iteration_count,
            pipeline_version=opportunity.pipeline_version,
            lifecycle_status=lifecycle_status,
            lifecycle_opportunity_id=lifecycle_opp_id,
        )
