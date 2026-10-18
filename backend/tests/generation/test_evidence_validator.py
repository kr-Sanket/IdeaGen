"""Tests for Evidence Integrity and Cautious Language Sanitization."""

import pytest
import uuid

from ideagen.generation.models import CandidateOpportunity, FeasibilityAssessment, NoveltyAssessment
from ideagen.generation.validator import sanitize_novelty_language, EvidenceIntegrityValidator


def test_sanitize_novelty_language():
    text_with_forbidden = "This is a novel research idea with guaranteed novel results and first ever implementation."
    sanitized = sanitize_novelty_language(text_with_forbidden)

    assert "novel research idea" not in sanitized.lower()
    assert "guaranteed novel" not in sanitized.lower()
    assert "first ever" not in sanitized.lower()
    assert "potentially novel" in sanitized.lower()


def test_evidence_integrity_validator_purges_unknown_ids():
    valid_stmt_id = uuid.uuid4()
    unknown_stmt_id = uuid.uuid4()
    valid_paper_id = uuid.uuid4()
    unknown_paper_id = uuid.uuid4()

    candidate = CandidateOpportunity(
        research_problem="Test problem",
        identified_gap="Test gap",
        proposed_research_direction="Test direction guaranteed novel",
        reasoning="Test reasoning",
        technical_approach="Test approach",
        expected_contribution="First ever approach",
        feasibility=FeasibilityAssessment(
            required_data="Data",
            required_compute="GPU",
            implementation_complexity="Med",
            experimental_setup="Setup",
        ),
        supporting_statement_ids=[valid_stmt_id, unknown_stmt_id],
        supporting_paper_ids=[valid_paper_id, unknown_paper_id],
    )

    validator = EvidenceIntegrityValidator()
    sanitized = validator.validate_and_sanitize_candidate(
        candidate,
        valid_statement_ids={valid_stmt_id},
        valid_paper_ids={valid_paper_id},
    )

    assert sanitized.supporting_statement_ids == [valid_stmt_id]
    assert sanitized.supporting_paper_ids == [valid_paper_id]
    assert "guaranteed novel" not in sanitized.proposed_research_direction.lower()
    assert "first ever" not in sanitized.expected_contribution.lower()
