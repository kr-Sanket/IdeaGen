"""Deterministic unit and regression tests for GapValidator citation resolution and schema integrity.

Verifies:
1. A cited E1 alias resolves to the exact corresponding statement UUID.
2. Multiple aliases (e.g. E1, E2, [E1], case-insensitive) resolve correctly.
3. Exact UUID citations remain supported.
4. Unknown aliases (e.g. E99) and foreign statement IDs are rejected.
5. Empty citations with a VALID verdict are still rejected with EMPTY_EVIDENCE_CITATIONS.
6. Missing citation fields are handled safely according to the schema contract.
7. A VALID result with valid citations passes through all existing taxonomy and precision gates.
8. Background motivation, broad challenges, and single-study limitations retain their existing safeguards.
9. Missing confidence remains None.
10. Evidence provenance and passage metadata remain unchanged.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
import pytest

from ideagen.llm.provider import LLMProvider
from ideagen.discovery.models import (
    GapCandidate,
    GapValidationResult,
    GapCategory,
    ValidationVerdict,
    GapStatus,
)
from ideagen.discovery.validator import GapValidator, resolve_evidence_citation


class MockValidatorProvider(LLMProvider):
    """Deterministic LLM provider returning pre-configured GapValidationResult instances."""

    def __init__(
        self,
        verdict: ValidationVerdict = ValidationVerdict.VALID,
        reasoning: str = "Candidate gap is verified by provided empirical evidence.",
        supporting_citations: Optional[List[str]] = None,
        counter_citations: Optional[List[str]] = None,
        confidence: Optional[float] = None,
        gap_category: Optional[GapCategory] = None,
        needs_refinement: bool = False,
    ):
        self.verdict = verdict
        self.reasoning = reasoning
        self.supporting_citations = supporting_citations if supporting_citations is not None else []
        self.counter_citations = counter_citations if counter_citations is not None else []
        self.confidence = confidence
        self.gap_category = gap_category
        self.needs_refinement = needs_refinement
        self.last_prompt: Optional[str] = None

    def generate_structured(self, prompt: str, response_schema: Any) -> Any:
        self.last_prompt = prompt
        return response_schema(
            verdict=self.verdict,
            reasoning=self.reasoning,
            supporting_evidence_statement_ids=self.supporting_citations,
            counter_evidence_statement_ids=self.counter_citations,
            confidence=self.confidence,
            gap_category=self.gap_category,
            needs_refinement=self.needs_refinement,
        )


def _create_mock_statement(
    statement_id: uuid.UUID,
    paper_id: uuid.UUID,
    statement_text: str,
    statement_type: str = "LIMITATION",
    section: str = "Discussion",
    passage_text: str = "Source passage verifying the limitation.",
) -> Dict[str, Any]:
    passage_id = uuid.uuid4()
    return {
        "id": statement_id,
        "statement_id": statement_id,
        "paper_id": paper_id,
        "statement_type": statement_type,
        "statement_text": statement_text,
        "section": section,
        "passage_id": passage_id,
        "passage_text": passage_text,
    }


# Test 1: A cited E1 alias resolves to the exact corresponding UUID
def test_cited_e1_alias_resolves_to_exact_uuid():
    s1_id = uuid.uuid4()
    p1_id = uuid.uuid4()
    stmt = _create_mock_statement(s1_id, p1_id, "Direct STNLF methods cannot be applied to microgrids.")
    statements_by_id = {s1_id: stmt}

    candidate = GapCandidate(
        gap_statement="Lack of direct STNLF methods for utility-scale solar microgrids.",
        research_problem="Direct STNLF methods cannot be applied to microgrids.",
        why_existing_work_insufficient="Current methods require centralized telemetry.",
        supporting_statement_ids=[s1_id],
        supporting_paper_ids=[p1_id],
        relevant_limitations=[],
        relevant_future_work=[],
        signals={"pattern_type": "LIMITATION_AND_FUTURE_WORK"},
    )

    provider = MockValidatorProvider(
        verdict=ValidationVerdict.VALID,
        supporting_citations=["E1"],
        gap_category=GapCategory.EVIDENCE_GROUNDED_GAP,
    )
    validator = GapValidator(provider=provider, min_valid_evidence=1)
    dto = validator.validate_candidate(candidate, statements_by_id)

    assert dto.status == GapStatus.VALIDATED
    assert dto.supporting_statement_ids == [s1_id]
    assert dto.evidence_count == 1
    assert len(dto.evidence_items) == 1
    assert dto.evidence_items[0].statement_id == s1_id
    assert "[E1]" in provider.last_prompt


# Test 2: Multiple aliases resolve correctly
def test_multiple_aliases_resolve_correctly():
    s1_id = uuid.uuid4()
    s2_id = uuid.uuid4()
    p1_id = uuid.uuid4()
    p2_id = uuid.uuid4()

    statements_by_id = {
        s1_id: _create_mock_statement(s1_id, p1_id, "Inverter telemetry is noisy."),
        s2_id: _create_mock_statement(s2_id, p2_id, "Future work must handle inverter noise robustly."),
    }

    candidate = GapCandidate(
        gap_statement="Inverter telemetry noise impedes solar forecasting.",
        research_problem="Noisy sensor data degrades STNLF accuracy.",
        why_existing_work_insufficient="Prior algorithms assume clean irradiance sensors.",
        supporting_statement_ids=[s1_id, s2_id],
        supporting_paper_ids=[p1_id, p2_id],
        relevant_limitations=[],
        relevant_future_work=[],
        signals={"pattern_type": "LIMITATION_AND_FUTURE_WORK"},
    )

    # Test lowercase "e1" and bracketed "[E2]"
    provider = MockValidatorProvider(
        verdict=ValidationVerdict.VALID,
        supporting_citations=["e1", "[E2]"],
        gap_category=GapCategory.EVIDENCE_GROUNDED_GAP,
    )
    validator = GapValidator(provider=provider, min_valid_evidence=2)
    dto = validator.validate_candidate(candidate, statements_by_id)

    assert dto.status == GapStatus.VALIDATED
    assert set(dto.supporting_statement_ids) == {s1_id, s2_id}
    assert dto.evidence_count == 2
    assert set(dto.supporting_paper_ids) == {p1_id, p2_id}


# Test 3: Exact UUID citations remain supported
def test_exact_uuid_citations_remain_supported():
    s1_id = uuid.uuid4()
    p1_id = uuid.uuid4()
    statements_by_id = {s1_id: _create_mock_statement(s1_id, p1_id, "Battery wear models are inaccurate.")}

    candidate = GapCandidate(
        gap_statement="Inaccurate battery wear modeling under solar cycling.",
        research_problem="Battery degradation models fail under erratic microgrid load.",
        why_existing_work_insufficient="Existing models assume constant charge/discharge cycles.",
        supporting_statement_ids=[s1_id],
        supporting_paper_ids=[p1_id],
        relevant_limitations=[],
        relevant_future_work=[],
        signals={},
    )

    provider = MockValidatorProvider(
        verdict=ValidationVerdict.VALID,
        supporting_citations=[str(s1_id)],
        gap_category=GapCategory.EVIDENCE_GROUNDED_GAP,
    )
    validator = GapValidator(provider=provider, min_valid_evidence=1)
    dto = validator.validate_candidate(candidate, statements_by_id)

    assert dto.status == GapStatus.VALIDATED
    assert dto.supporting_statement_ids == [s1_id]


# Test 4: Unknown aliases and foreign statement IDs are rejected
def test_unknown_aliases_and_foreign_statement_ids_rejected():
    s1_id = uuid.uuid4()
    p1_id = uuid.uuid4()
    foreign_id = uuid.uuid4()
    statements_by_id = {s1_id: _create_mock_statement(s1_id, p1_id, "Cloud cover transient forecasting limitation.")}

    candidate = GapCandidate(
        gap_statement="Unpredictable cloud transients in microgrid forecasting.",
        research_problem="Ramp-rate prediction lacks sub-minute fidelity.",
        why_existing_work_insufficient="Satellite imagery resolution is 15-minute coarse.",
        supporting_statement_ids=[s1_id],
        supporting_paper_ids=[p1_id],
        relevant_limitations=[],
        relevant_future_work=[],
        signals={},
    )

    # Case A: Unknown alias E99
    provider_unknown = MockValidatorProvider(
        verdict=ValidationVerdict.VALID,
        supporting_citations=["E99"],
        gap_category=GapCategory.EVIDENCE_GROUNDED_GAP,
    )
    validator = GapValidator(provider=provider_unknown, min_valid_evidence=1)
    dto_unknown = validator.validate_candidate(candidate, statements_by_id)
    assert dto_unknown.status == GapStatus.REJECTED
    assert dto_unknown.metadata.get("rejection_reason") == "INVALID_OR_FABRICATED_CITATIONS"

    # Case B: Foreign UUID not in candidate's bounded statements
    provider_foreign = MockValidatorProvider(
        verdict=ValidationVerdict.VALID,
        supporting_citations=[str(foreign_id)],
        gap_category=GapCategory.EVIDENCE_GROUNDED_GAP,
    )
    dto_foreign = validator.validate_candidate(candidate, statements_by_id)
    assert dto_foreign.status == GapStatus.REJECTED
    assert dto_foreign.metadata.get("rejection_reason") == "INVALID_OR_FABRICATED_CITATIONS"


# Test 5: Empty citations with a VALID verdict are still rejected
def test_empty_citations_with_valid_verdict_rejected():
    s1_id = uuid.uuid4()
    p1_id = uuid.uuid4()
    statements_by_id = {s1_id: _create_mock_statement(s1_id, p1_id, "Solar inverter degradation statement.")}

    candidate = GapCandidate(
        gap_statement="Solar inverter degradation estimation.",
        research_problem="No lifetime degradation model.",
        why_existing_work_insufficient="Only manufacturer lab specs available.",
        supporting_statement_ids=[s1_id],
        supporting_paper_ids=[p1_id],
        relevant_limitations=[],
        relevant_future_work=[],
        signals={},
    )

    provider = MockValidatorProvider(
        verdict=ValidationVerdict.VALID,
        supporting_citations=[],  # Empty citations
        gap_category=GapCategory.EVIDENCE_GROUNDED_GAP,
    )
    validator = GapValidator(provider=provider, min_valid_evidence=1)
    dto = validator.validate_candidate(candidate, statements_by_id)

    assert dto.status == GapStatus.REJECTED
    assert dto.metadata.get("rejection_reason") == "EMPTY_EVIDENCE_CITATIONS"
    assert "provided no evidence statement citations" in dto.validation_reasoning


# Test 6: Missing citation fields are handled safely according to the schema contract
def test_missing_citation_fields_handled_safely_by_schema():
    # 1. Schema contract verifies citation fields are in 'required'
    schema = GapValidationResult.model_json_schema()
    assert "supporting_evidence_statement_ids" in schema["required"]
    assert "counter_evidence_statement_ids" in schema["required"]

    # 2. Deserializing a payload where citation fields are omitted defaults safely to []
    raw_payload = {
        "verdict": "VALID",
        "reasoning": "Valid but omitted citation lists in JSON.",
    }
    parsed = GapValidationResult.model_validate(raw_payload)
    assert parsed.supporting_evidence_statement_ids == []
    assert parsed.counter_evidence_statement_ids == []

    # 3. GapValidator handles this safely by rejecting due to empty citations
    s1_id = uuid.uuid4()
    p1_id = uuid.uuid4()
    statements_by_id = {s1_id: _create_mock_statement(s1_id, p1_id, "Microgrid power flow limitation.")}

    candidate = GapCandidate(
        gap_statement="Microgrid power flow limitation.",
        research_problem="Unbalanced three-phase load.",
        why_existing_work_insufficient="Assumes balanced load.",
        supporting_statement_ids=[s1_id],
        supporting_paper_ids=[p1_id],
        relevant_limitations=[],
        relevant_future_work=[],
        signals={},
    )

    class CustomOmissionProvider(LLMProvider):
        def generate_structured(self, prompt: str, response_schema: Any) -> Any:
            return response_schema.model_validate(raw_payload)

    validator = GapValidator(provider=CustomOmissionProvider(), min_valid_evidence=1)
    dto = validator.validate_candidate(candidate, statements_by_id)
    assert dto.status == GapStatus.REJECTED
    assert dto.metadata.get("rejection_reason") == "EMPTY_EVIDENCE_CITATIONS"


# Test 7: A VALID result with valid citations passes through all existing taxonomy and precision gates
def test_valid_result_with_valid_citations_passes_taxonomy_and_precision_gates():
    s1_id = uuid.uuid4()
    s2_id = uuid.uuid4()
    p1_id = uuid.uuid4()
    p2_id = uuid.uuid4()

    statements_by_id = {
        s1_id: _create_mock_statement(s1_id, p1_id, "Utility-scale microgrids lack STNLF benchmarks."),
        s2_id: _create_mock_statement(s2_id, p2_id, "Independent validation in real-world microgrids is missing."),
    }

    candidate = GapCandidate(
        gap_statement="Lack of demonstrated STNLF methods in utility-scale solar microgrids.",
        research_problem="Existing direct STNLF methods have only been tested in residential simulations.",
        why_existing_work_insufficient="Utility-scale microgrids exhibit non-linear phase imbalances.",
        supporting_statement_ids=[s1_id, s2_id],
        supporting_paper_ids=[p1_id, p2_id],
        relevant_limitations=[],
        relevant_future_work=[],
        signals={"pattern_type": "LIMITATION_AND_FUTURE_WORK"},
    )

    provider = MockValidatorProvider(
        verdict=ValidationVerdict.VALID,
        supporting_citations=["E1", "E2"],
        confidence=0.89,
        gap_category=GapCategory.EVIDENCE_GROUNDED_GAP,
    )
    validator = GapValidator(provider=provider, min_valid_evidence=1)
    dto = validator.validate_candidate(candidate, statements_by_id)

    assert dto.status == GapStatus.VALIDATED
    assert dto.evidence_count == 2
    assert dto.metadata["gap_category"] == "EVIDENCE_GROUNDED_GAP"
    assert dto.metadata["needs_refinement"] is False
    assert "single_study_scope" not in dto.metadata
    assert dto.validation_confidence == 0.89


# Test 8: Background motivation, broad challenges, and single-study limitations retain their existing safeguards
def test_taxonomy_safeguards_retained():
    s1_id = uuid.uuid4()
    p1_id = uuid.uuid4()
    statements_by_id = {s1_id: _create_mock_statement(s1_id, p1_id, "Introductory background motivation.")}

    # Case A: Background motivation is REJECTED even when model votes VALID and cites E1
    cand_bg = GapCandidate(
        gap_statement="Centralized solar forecasting creates single-point-of-failure vulnerabilities.",
        research_problem="Centralized data storage creates privacy risks and storage demands.",
        why_existing_work_insufficient="Centralized architecture vulnerability.",
        supporting_statement_ids=[s1_id],
        supporting_paper_ids=[p1_id],
        relevant_limitations=[],
        relevant_future_work=[],
        signals={"potential_background_motivation": True},
    )
    provider_bg = MockValidatorProvider(
        verdict=ValidationVerdict.VALID,
        supporting_citations=["E1"],
        gap_category=GapCategory.BACKGROUND_MOTIVATION,
    )
    validator = GapValidator(provider=provider_bg, min_valid_evidence=1)
    dto_bg = validator.validate_candidate(cand_bg, statements_by_id)
    assert dto_bg.status == GapStatus.REJECTED
    assert dto_bg.metadata.get("rejection_reason") == "BACKGROUND_MOTIVATION"

    # Case B: Broad challenge is retained as CANDIDATE with needs_refinement=True
    cand_broad = GapCandidate(
        gap_statement="Weather uncertainty in solar microgrids.",
        research_problem="Non-IID weather variation across microgrid locations.",
        why_existing_work_insufficient="General weather variation challenge.",
        supporting_statement_ids=[s1_id],
        supporting_paper_ids=[p1_id],
        relevant_limitations=[],
        relevant_future_work=[],
        signals={},
    )
    provider_broad = MockValidatorProvider(
        verdict=ValidationVerdict.VALID,
        supporting_citations=["E1"],
        gap_category=GapCategory.BROAD_CHALLENGE,
        needs_refinement=True,
    )
    validator_broad = GapValidator(provider=provider_broad, min_valid_evidence=1)
    dto_broad = validator_broad.validate_candidate(cand_broad, statements_by_id)
    assert dto_broad.status == GapStatus.CANDIDATE
    assert dto_broad.metadata.get("needs_refinement") is True

    # Case C: Single-study limitation is retained as CANDIDATE with single_study_scope=True
    cand_study = GapCandidate(
        gap_statement="Evaluation limited to a single 50kW array in Arizona.",
        research_problem="Testing only performed on one specific PV array.",
        why_existing_work_insufficient="Single site evaluation.",
        supporting_statement_ids=[s1_id],
        supporting_paper_ids=[p1_id],
        relevant_limitations=[],
        relevant_future_work=[],
        signals={"potential_study_specific_limitation": True},
    )
    provider_study = MockValidatorProvider(
        verdict=ValidationVerdict.VALID,
        supporting_citations=["E1"],
        gap_category=GapCategory.STUDY_LIMITATION,
    )
    validator_study = GapValidator(provider=provider_study, min_valid_evidence=1)
    dto_study = validator_study.validate_candidate(cand_study, statements_by_id)
    assert dto_study.status == GapStatus.CANDIDATE
    assert dto_study.metadata.get("single_study_scope") is True


# Test 9: Missing confidence remains None
def test_missing_confidence_remains_none():
    s1_id = uuid.uuid4()
    p1_id = uuid.uuid4()
    statements_by_id = {s1_id: _create_mock_statement(s1_id, p1_id, "Model confidence omitted test.")}

    candidate = GapCandidate(
        gap_statement="Confidence test gap.",
        research_problem="Testing nullable confidence column.",
        why_existing_work_insufficient="Confidence test rationale.",
        supporting_statement_ids=[s1_id],
        supporting_paper_ids=[p1_id],
        relevant_limitations=[],
        relevant_future_work=[],
        signals={},
    )

    provider = MockValidatorProvider(
        verdict=ValidationVerdict.VALID,
        supporting_citations=["E1"],
        confidence=None,  # Explicitly omitted confidence
        gap_category=GapCategory.EVIDENCE_GROUNDED_GAP,
    )
    validator = GapValidator(provider=provider, min_valid_evidence=1)
    dto = validator.validate_candidate(candidate, statements_by_id)

    assert dto.status == GapStatus.VALIDATED
    assert dto.validation_confidence is None


# Test 10: Evidence provenance remains unchanged
def test_evidence_provenance_remains_unchanged():
    s1_id = uuid.uuid4()
    p1_id = uuid.uuid4()
    stmt = _create_mock_statement(
        s1_id,
        p1_id,
        "Original statement text with exact provenance.",
        statement_type="LIMITATION",
        section="Results Section 4.2",
        passage_text="Detailed source excerpt from section 4.2.",
    )
    statements_by_id = {s1_id: stmt}

    candidate = GapCandidate(
        gap_statement="Provenance check gap.",
        research_problem="Testing evidence item attributes.",
        why_existing_work_insufficient="Provenance test rationale.",
        supporting_statement_ids=[s1_id],
        supporting_paper_ids=[p1_id],
        relevant_limitations=[],
        relevant_future_work=[],
        signals={},
    )

    provider = MockValidatorProvider(
        verdict=ValidationVerdict.VALID,
        supporting_citations=["E1"],
        gap_category=GapCategory.EVIDENCE_GROUNDED_GAP,
    )
    validator = GapValidator(provider=provider, min_valid_evidence=1)
    dto = validator.validate_candidate(candidate, statements_by_id)

    assert len(dto.evidence_items) == 1
    item = dto.evidence_items[0]
    assert item.statement_id == s1_id
    assert item.paper_id == p1_id
    assert item.statement_text == "Original statement text with exact provenance."
    assert item.section == "Results Section 4.2"


# Test 11: Background motivation statement rejected despite model VALID verdict
def test_background_motivation_net_load_statement_rejected():
    s1_id = uuid.uuid4()
    p1_id = uuid.uuid4()
    stmt = _create_mock_statement(
        s1_id,
        p1_id,
        "Forecasting net load is required to safeguard power system operation under variable renewable generation.",
        statement_type="PROBLEM",
        section="Introduction",
    )
    statements_by_id = {s1_id: stmt}

    candidate = GapCandidate(
        gap_statement="Forecasting net load is required to safeguard power system operation.",
        research_problem="Forecasting net load is required to safeguard power system operation under variable renewable generation.",
        why_existing_work_insufficient="Growing renewable integration poses new challenges to power system stability.",
        supporting_statement_ids=[s1_id],
        supporting_paper_ids=[p1_id],
        relevant_limitations=[],
        relevant_future_work=[],
        signals={},
    )

    # Model attempts to validate background motivation as EVIDENCE_GROUNDED_GAP
    provider = MockValidatorProvider(
        verdict=ValidationVerdict.VALID,
        supporting_citations=["E1"],
        gap_category=GapCategory.EVIDENCE_GROUNDED_GAP,
    )
    validator = GapValidator(provider=provider, min_valid_evidence=1)
    dto = validator.validate_candidate(candidate, statements_by_id)

    assert dto.status == GapStatus.REJECTED
    assert dto.metadata.get("rejection_reason") == "BACKGROUND_MOTIVATION"
    assert dto.metadata.get("gap_category") == "BACKGROUND_MOTIVATION"
    assert "background motivation" in dto.validation_reasoning.lower()


# Test 12: Background rejection protects against false positives when genuine deficits are present
def test_background_rejection_protects_against_false_positives():
    s1_id = uuid.uuid4()
    p1_id = uuid.uuid4()
    # Statement contains purpose keywords ('required to safeguard') AND an explicit technical deficit ('suffer from severe phase imbalance and cannot converge')
    stmt = _create_mock_statement(
        s1_id,
        p1_id,
        "Accurate forecasting is required to safeguard microgrid stability, but current models suffer from severe phase imbalance and cannot converge in real-time.",
        statement_type="LIMITATION",
        section="Discussion",
    )
    statements_by_id = {s1_id: stmt}

    candidate = GapCandidate(
        gap_statement="Failure of real-time forecasting convergence under severe phase imbalance.",
        research_problem="Accurate forecasting is required to safeguard microgrid stability, but current models suffer from severe phase imbalance and cannot converge in real-time.",
        why_existing_work_insufficient="Existing matrix factorization algorithms incur intractably high latency under three-phase unbalance.",
        supporting_statement_ids=[s1_id],
        supporting_paper_ids=[p1_id],
        relevant_limitations=["Current models suffer from severe phase imbalance and cannot converge in real-time."],
        relevant_future_work=[],
        signals={},
    )

    provider = MockValidatorProvider(
        verdict=ValidationVerdict.VALID,
        supporting_citations=["E1"],
        gap_category=GapCategory.EVIDENCE_GROUNDED_GAP,
    )
    validator = GapValidator(provider=provider, min_valid_evidence=1)
    dto = validator.validate_candidate(candidate, statements_by_id)

    # Must NOT be falsely rejected as background motivation
    assert dto.status == GapStatus.VALIDATED
    assert dto.metadata.get("gap_category") == "EVIDENCE_GROUNDED_GAP"
    assert dto.metadata.get("rejection_reason") is None


# Test 13: Duplicate passage evidence is not counted as independent corroboration
def test_duplicate_passage_evidence_not_counted_as_independent_corroboration():
    s1_id = uuid.uuid4()
    s2_id = uuid.uuid4()
    p1_id = uuid.uuid4()
    shared_passage_id = uuid.uuid4()

    s1 = _create_mock_statement(s1_id, p1_id, "Model fails to generalize under partial shading.", section="Results")
    s1["passage_id"] = shared_passage_id

    s2 = _create_mock_statement(s2_id, p1_id, "Partial shading degrades forecasting accuracy by 34%.", section="Results")
    s2["passage_id"] = shared_passage_id

    statements_by_id = {s1_id: s1, s2_id: s2}

    candidate = GapCandidate(
        gap_statement="Forecasting degradation under partial shading conditions.",
        research_problem="Model fails to generalize under partial shading.",
        why_existing_work_insufficient="Uniform irradiance assumption fails during intermittent cloud occlusion.",
        supporting_statement_ids=[s1_id, s2_id],
        supporting_paper_ids=[p1_id],
        supporting_passage_ids=[shared_passage_id],
        relevant_limitations=[],
        relevant_future_work=[],
        signals={"distinct_passage_count": 1, "same_passage_evidence": True},
    )

    provider = MockValidatorProvider(
        verdict=ValidationVerdict.VALID,
        supporting_citations=["E1", "E2"],
        gap_category=GapCategory.EVIDENCE_GROUNDED_GAP,
    )
    validator = GapValidator(provider=provider, min_valid_evidence=1)
    dto = validator.validate_candidate(candidate, statements_by_id)

    # Legitimate single-paper limitation remains eligible
    assert dto.status == GapStatus.VALIDATED
    # Evidence scope accurately recorded
    assert dto.metadata["distinct_passage_count"] == 1
    assert dto.metadata["single_passage_evidence"] is True
    assert len(dto.evidence_items) == 2
    assert dto.evidence_items[0].passage_id == shared_passage_id
    assert dto.evidence_items[1].passage_id == shared_passage_id


# Test 14: Introductory limitation from a single paper flags uncertainty and requires refinement
def test_introductory_limitation_from_single_paper_requires_refinement():
    s1_id = uuid.uuid4()
    p1_id = uuid.uuid4()
    # Statement sourced from Introduction describing prior work
    stmt = _create_mock_statement(
        s1_id,
        p1_id,
        "Prior deep neural networks suffer from high computational overhead during streaming inference.",
        statement_type="LIMITATION",
        section="Introduction",
    )
    statements_by_id = {s1_id: stmt}

    candidate = GapCandidate(
        gap_statement="High computational overhead during streaming solar inference.",
        research_problem="Prior deep neural networks suffer from high computational overhead during streaming inference.",
        why_existing_work_insufficient="Convolutional layers require recurrent retraining.",
        supporting_statement_ids=[s1_id],
        supporting_paper_ids=[p1_id],
        relevant_limitations=[],
        relevant_future_work=[],
        signals={},
    )

    # Model claims VALID without refinement
    provider = MockValidatorProvider(
        verdict=ValidationVerdict.VALID,
        supporting_citations=["E1"],
        gap_category=GapCategory.EVIDENCE_GROUNDED_GAP,
        needs_refinement=False,
    )
    validator = GapValidator(provider=provider, min_valid_evidence=1)
    dto = validator.validate_candidate(candidate, statements_by_id)

    # Preserved as CANDIDATE flagged for researcher review rather than falsely rejected or blindly validated
    assert dto.status == GapStatus.CANDIDATE
    assert dto.metadata["needs_refinement"] is True
    assert dto.metadata["introductory_limitation_single_paper"] is True
    assert "uncertainty flagged for researcher review" in dto.validation_reasoning.lower()


# Test 15: Scale/scope divergence (utility vs residential) requires refinement & model VALID cannot override
def test_scale_scope_divergence_utility_vs_residential_requires_refinement():
    s1_id = uuid.uuid4()
    p1_id = uuid.uuid4()
    # Evidence specifically addresses utility-scale solar farms
    stmt = _create_mock_statement(
        s1_id,
        p1_id,
        "Direct STNLF algorithms cannot be deployed on utility-scale solar farms due to telemetry communication limits.",
        statement_type="LIMITATION",
        section="Discussion",
    )
    statements_by_id = {s1_id: stmt}

    # Candidate claims residential microgrid scope
    candidate = GapCandidate(
        gap_statement="Direct STNLF methods cannot scale to residential microgrids under variable weather.",
        research_problem="Centralized telemetry communication bottlenecks impede forecasting.",
        why_existing_work_insufficient="Communication bandwidth constraints.",
        supporting_statement_ids=[s1_id],
        supporting_paper_ids=[p1_id],
        relevant_limitations=[],
        relevant_future_work=[],
        signals={},
    )

    # Model votes VALID without refinement
    provider = MockValidatorProvider(
        verdict=ValidationVerdict.VALID,
        supporting_citations=["E1"],
        gap_category=GapCategory.EVIDENCE_GROUNDED_GAP,
        needs_refinement=False,
    )
    validator = GapValidator(provider=provider, min_valid_evidence=1)
    dto = validator.validate_candidate(candidate, statements_by_id)

    # Model cannot override deterministic safeguard
    assert dto.status == GapStatus.CANDIDATE
    assert dto.metadata["needs_refinement"] is True
    assert dto.metadata["scale_scope_divergence"] == "utility_scale_evidence_generalized_to_residential_scope"
    assert "retained as candidate requiring researcher refinement" in dto.validation_reasoning.lower()

