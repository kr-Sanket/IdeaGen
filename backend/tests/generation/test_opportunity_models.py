"""Tests for Phase 5 Opportunity Data Models and Schemas."""

import pytest
import uuid
from datetime import datetime

from ideagen.generation.models import (
    OpportunityStatus,
    FeasibilityAssessment,
    NoveltyAssessment,
    PriorArtAssessment,
    ResearchOpportunityInputDTO,
    CandidateOpportunity,
    ResearchOpportunityDTO,
)


def test_feasibility_assessment_schema():
    feasibility = FeasibilityAssessment(
        required_data="OpenAlex abstracts corpus",
        required_compute="1x NVIDIA RTX 4090 GPU",
        implementation_complexity="Moderate algorithm development",
        experimental_setup="Comparative evaluation on 100 benchmark gaps",
        feasibility_score=0.85,
        summary="Feasible under standard academic setup.",
    )
    assert feasibility.feasibility_score == 0.85
    assert feasibility.required_compute == "1x NVIDIA RTX 4090 GPU"


def test_novelty_assessment_schema():
    novelty = NoveltyAssessment(
        assessment="Potentially novel within the analyzed corpus",
        reasoning="Distinct from prior indexing focus",
        differentiation="Focuses on dynamic bounds",
        confidence=0.80,
    )
    assert novelty.assessment == "Potentially novel within the analyzed corpus"


def test_research_opportunity_dto_schema():
    gap_id = uuid.uuid4()
    s_id = uuid.uuid4()
    p_id = uuid.uuid4()

    feasibility = FeasibilityAssessment(
        required_data="Test dataset",
        required_compute="CPU",
        implementation_complexity="Low",
        experimental_setup="Unit test setup",
    )
    novelty = NoveltyAssessment(
        assessment="Potentially novel within the analyzed corpus",
        reasoning="Test reasoning",
        differentiation="Test differentiation",
    )

    opp = ResearchOpportunityDTO(
        id=uuid.uuid4(),
        gap_id=gap_id,
        research_problem="Problem statement",
        identified_gap="Gap statement",
        proposed_research_direction="Proposed direction",
        reasoning="Reasoning statement",
        technical_approach="Technical approach",
        expected_contribution="Expected contribution",
        feasibility=feasibility,
        novelty_assessment=novelty,
        confidence=0.90,
        supporting_statement_ids=[s_id],
        supporting_paper_ids=[p_id],
        prior_art_statement_ids=[],
        status=OpportunityStatus.VALIDATED,
        created_at=datetime.utcnow(),
        generation_model="gemini-2.5-flash",
        validation_model="gemini-2.5-flash",
        embedding_model="gemini-embedding-001",
        refinement_iteration_count=0,
        pipeline_version="v1.0",
    )

    assert opp.gap_id == gap_id
    assert opp.status == OpportunityStatus.VALIDATED
    assert opp.supporting_statement_ids == [s_id]
    assert opp.generation_model == "gemini-2.5-flash"
