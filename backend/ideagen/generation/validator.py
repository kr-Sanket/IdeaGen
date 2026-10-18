"""Evidence Integrity and Language Safety Validator (Phase 5).

Enforces strict citation integrity, verifies evidence references against stored corpus records,
and replaces exaggerated novelty claims with cautious, corpus-bounded phrasing.
"""

from __future__ import annotations

import logging
import re
from typing import List, Set, Tuple, Optional, Dict, Any
import uuid
from sqlalchemy.orm import Session

from ideagen.db.models import Paper, ResearchStatement, ResearchGap, GapEvidence
from ideagen.generation.models import CandidateOpportunity, NoveltyAssessment

logger = logging.getLogger(__name__)

FORBIDDEN_PATTERNS = [
    (re.compile(r"\bnovel research idea\b", re.IGNORECASE), "potentially novel research direction within the analyzed corpus"),
    (re.compile(r"\bguaranteed novel\b", re.IGNORECASE), "potentially novel within the analyzed corpus"),
    (re.compile(r"\bfirst ever\b", re.IGNORECASE), "previously unobserved within the analyzed corpus"),
    (re.compile(r"\bdefinitely novel\b", re.IGNORECASE), "potentially novel within the analyzed corpus"),
    (re.compile(r"\bguaranteed novelty\b", re.IGNORECASE), "potential differentiation within the analyzed corpus"),
]


def sanitize_novelty_language(text: str) -> str:
    """Sanitizes text by replacing prohibited absolute novelty assertions with cautious phrasing."""
    if not text:
        return text
    sanitized = text
    for pattern, replacement in FORBIDDEN_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


class EvidenceIntegrityValidator:
    """Validates that candidate research opportunities strictly reference existing corpus evidence."""

    def __init__(self, db_session: Optional[Session] = None):
        self.db_session = db_session

    def validate_and_sanitize_candidate(
        self,
        candidate: CandidateOpportunity,
        valid_statement_ids: Optional[Set[uuid.UUID]] = None,
        valid_paper_ids: Optional[Set[uuid.UUID]] = None,
    ) -> CandidateOpportunity:
        """Validates statement & paper IDs, purges fabricated citations, and sanitizes novelty language."""
        
        # 1. Gather valid IDs from DB session if provided
        db_stmt_ids: Set[uuid.UUID] = set(valid_statement_ids) if valid_statement_ids is not None else set()
        db_paper_ids: Set[uuid.UUID] = set(valid_paper_ids) if valid_paper_ids is not None else set()

        if self.db_session is not None:
            if not db_stmt_ids:
                stmt_rows = self.db_session.query(ResearchStatement.id).all()
                db_stmt_ids = {row[0] for row in stmt_rows}
            if not db_paper_ids:
                paper_rows = self.db_session.query(Paper.id).all()
                db_paper_ids = {row[0] for row in paper_rows}

        # 2. Filter supporting statement and paper IDs against known corpus if set
        filtered_stmt_ids: List[uuid.UUID] = []
        if db_stmt_ids:
            for s_id in candidate.supporting_statement_ids:
                if s_id in db_stmt_ids:
                    filtered_stmt_ids.append(s_id)
                else:
                    logger.warning(f"Purged fabricated or unknown statement ID: {s_id}")
        else:
            filtered_stmt_ids = list(candidate.supporting_statement_ids)

        filtered_paper_ids: List[uuid.UUID] = []
        if db_paper_ids:
            for p_id in candidate.supporting_paper_ids:
                if p_id in db_paper_ids:
                    filtered_paper_ids.append(p_id)
                else:
                    logger.warning(f"Purged fabricated or unknown paper ID: {p_id}")
        else:
            filtered_paper_ids = list(candidate.supporting_paper_ids)

        # 3. Sanitize fields against prohibited novelty language
        sanitized_problem = sanitize_novelty_language(candidate.research_problem)
        sanitized_gap = sanitize_novelty_language(candidate.identified_gap)
        sanitized_direction = sanitize_novelty_language(candidate.proposed_research_direction)
        sanitized_reasoning = sanitize_novelty_language(candidate.reasoning)
        sanitized_approach = sanitize_novelty_language(candidate.technical_approach)
        sanitized_contribution = sanitize_novelty_language(candidate.expected_contribution)

        # 4. Construct sanitized candidate copy
        feasibility_copy = candidate.feasibility.model_copy()
        feasibility_copy.summary = sanitize_novelty_language(feasibility_copy.summary)

        return CandidateOpportunity(
            research_problem=sanitized_problem,
            identified_gap=sanitized_gap,
            proposed_research_direction=sanitized_direction,
            reasoning=sanitized_reasoning,
            technical_approach=sanitized_approach,
            expected_contribution=sanitized_contribution,
            feasibility=feasibility_copy,
            supporting_statement_ids=filtered_stmt_ids,
            supporting_paper_ids=filtered_paper_ids,
            confidence=candidate.confidence,
        )

    def sanitize_novelty_assessment(self, assessment: NoveltyAssessment) -> NoveltyAssessment:
        """Sanitizes NoveltyAssessment reasoning and differentiation fields."""
        cleaned_assessment = sanitize_novelty_language(assessment.assessment)
        if "definitely novel" in cleaned_assessment.lower() or "guaranteed novel" in cleaned_assessment.lower():
            cleaned_assessment = "Potentially novel within the analyzed corpus"

        return NoveltyAssessment(
            assessment=cleaned_assessment,
            reasoning=sanitize_novelty_language(assessment.reasoning),
            differentiation=sanitize_novelty_language(assessment.differentiation),
            confidence=assessment.confidence,
            overlapping_statement_ids=list(assessment.overlapping_statement_ids),
            overlapping_paper_ids=list(assessment.overlapping_paper_ids),
        )
