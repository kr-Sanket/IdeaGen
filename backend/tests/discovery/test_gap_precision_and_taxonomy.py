"""Deterministic unit and regression tests for research-gap extraction precision and taxonomy.

Verifies:
1. Introductory background motivation is not automatically treated as a validated open research gap.
2. A broad but legitimate challenge is retained as a candidate requiring refinement.
3. A single-study limitation is not presented as a field-wide conclusion without supporting evidence.
4. A genuinely unresolved, evidence-grounded problem remains discoverable and validated.
5. Evidence IDs and source provenance remain intact.
6. Malformed model output and provider failures continue to follow existing recovery behavior.
"""

from __future__ import annotations

import uuid
import pytest

from ideagen.llm.provider import LLMProvider
from ideagen.llm.exceptions import ProviderFailureError, MalformedResponseError
from ideagen.discovery.models import (
    GapCandidate,
    GapValidationResult,
    GapCategory,
    ValidationVerdict,
    GapStatus,
)
from ideagen.discovery.validator import GapValidator


class MockTaxonomyProvider(LLMProvider):
    """Deterministic LLM provider returning controlled validation results and taxonomy."""

    def __init__(
        self,
        verdict: ValidationVerdict = ValidationVerdict.VALID,
        reasoning: str = "Verified by empirical evidence.",
        cited_ids: list[str] = None,
        confidence: float | None = 0.90,
        gap_category: GapCategory = None,
        needs_refinement: bool = False,
        raise_provider_failure: bool = False,
        raise_malformed: bool = False,
    ):
        self.verdict = verdict
        self.reasoning = reasoning
        self.cited_ids = cited_ids or []
        self.confidence = confidence
        self.gap_category = gap_category
        self.needs_refinement = needs_refinement
        self.raise_provider_failure = raise_provider_failure
        self.raise_malformed = raise_malformed
        self.last_prompt = None

    def generate_structured(self, prompt: str, response_schema):
        self.last_prompt = prompt
        if self.raise_provider_failure:
            raise ProviderFailureError("Mock provider connection failed")
        if self.raise_malformed:
            raise MalformedResponseError("Mock unparseable JSON output")

        return response_schema(
            verdict=self.verdict,
            reasoning=self.reasoning,
            supporting_evidence_statement_ids=self.cited_ids,
            counter_evidence_statement_ids=[],
            confidence=self.confidence,
            gap_category=self.gap_category,
            needs_refinement=self.needs_refinement,
        )


def test_introductory_motivation_rejected_not_validated_gap():
    """1. Introductory motivation is not automatically treated as a validated open research gap (Audited Gap 1)."""
    s1_id = uuid.uuid4()
    p1_id = uuid.uuid4()

    statements_by_id = {
        s1_id: {
            "id": s1_id,
            "paper_id": p1_id,
            "statement_type": "PROBLEM",
            "statement_text": "Traditional centralized learning requires all data to be stored centrally, raising privacy risks and storage demands.",
            "section": "Introduction",
        }
    }

    candidate = GapCandidate(
        gap_statement="Identified research problem lacking comprehensive empirical resolution: Traditional centralized learning requires all data to be stored centrally, raising privacy risks and storage demands.",
        research_problem="Traditional centralized learning requires all data to be stored centrally, raising privacy risks and storage demands.",
        why_existing_work_insufficient="Centralized storage creates data leakage risks.",
        supporting_statement_ids=[s1_id],
        supporting_paper_ids=[p1_id],
        relevant_limitations=[],
        relevant_future_work=[],
        signals={"pattern_type": "UNRESOLVED_PROBLEM", "potential_background_motivation": True},
    )

    # Even if model erroneously marked VALID or classified as BACKGROUND_MOTIVATION:
    provider = MockTaxonomyProvider(
        verdict=ValidationVerdict.VALID,
        reasoning="Describes standard motivation for distributed training.",
        cited_ids=[str(s1_id)],
        gap_category=GapCategory.BACKGROUND_MOTIVATION,
    )

    validator = GapValidator(provider=provider, min_valid_evidence=1)
    res = validator.validate_candidate(candidate, statements_by_id)

    # Must be REJECTED, not VALIDATED
    assert res.status == GapStatus.REJECTED
    assert res.metadata.get("gap_category") == GapCategory.BACKGROUND_MOTIVATION.value
    assert res.metadata.get("rejection_reason") == "BACKGROUND_MOTIVATION"
    assert "background motivation" in res.validation_reasoning.lower()


def test_broad_challenge_retained_as_candidate_requiring_refinement():
    """2. A broad but legitimate challenge can be retained as a candidate requiring refinement (Audited Gap 2)."""
    s1_id = uuid.uuid4()
    s2_id = uuid.uuid4()
    p1_id = uuid.uuid4()
    p2_id = uuid.uuid4()

    statements_by_id = {
        s1_id: {
            "id": s1_id,
            "paper_id": p1_id,
            "statement_type": "LIMITATION",
            "statement_text": "FL is known to have limitations due to its assumption of Independent and Identically Distributed (IID) data.",
            "section": "Introduction",
        },
        s2_id: {
            "id": s2_id,
            "paper_id": p2_id,
            "statement_type": "PROBLEM",
            "statement_text": "FL faces challenges such as communication cost, energy consumption, and handling heterogeneous data distributions.",
            "section": "Challenges",
        },
    }

    candidate = GapCandidate(
        gap_statement="Persistent scientific limitation across current methodologies: FL is known to have limitations due to its assumption of Independent and Identically Distributed (IID) data.",
        research_problem="FL faces challenges such as communication cost, energy consumption, and handling heterogeneous data distributions.",
        why_existing_work_insufficient="Non-IID distributions degrade convergence across institutions.",
        supporting_statement_ids=[s1_id, s2_id],
        supporting_paper_ids=[p1_id, p2_id],
        relevant_limitations=["FL has limitations under non-IID data."],
        relevant_future_work=[],
        signals={"pattern_type": "RECURRING_LIMITATIONS", "contributing_paper_count": 2},
    )

    provider = MockTaxonomyProvider(
        verdict=ValidationVerdict.VALID,
        reasoning="Legitimate high-level challenge supported by multi-paper evidence, but broad.",
        cited_ids=[str(s1_id), str(s2_id)],
        gap_category=GapCategory.BROAD_CHALLENGE,
        needs_refinement=True,
    )

    validator = GapValidator(provider=provider, min_valid_evidence=1)
    res = validator.validate_candidate(candidate, statements_by_id)

    # Must be retained as CANDIDATE requiring refinement, not blindly promoted to VALIDATED
    assert res.status == GapStatus.CANDIDATE
    assert res.metadata.get("gap_category") == GapCategory.BROAD_CHALLENGE.value
    assert res.metadata.get("needs_refinement") is True
    # Provenance preserved
    assert set(res.supporting_statement_ids) == {s1_id, s2_id}
    assert set(res.supporting_paper_ids) == {p1_id, p2_id}


def test_single_study_limitation_not_presented_as_field_wide_conclusion():
    """3. A single-study limitation is not presented as a field-wide conclusion without supporting evidence (Audited Gap 3)."""
    s1_id = uuid.uuid4()
    p1_id = uuid.uuid4()

    statements_by_id = {
        s1_id: {
            "id": s1_id,
            "paper_id": p1_id,
            "statement_type": "LIMITATION",
            "statement_text": "The study is limited to a simulated environment and real-world hospital data, which may not fully represent all real-world scenarios.",
            "section": "Discussion",
        }
    }

    candidate = GapCandidate(
        gap_statement="Documented scientific limitation in methodology: The study is limited to a simulated environment and real-world hospital data, which may not fully represent all real-world scenarios.",
        research_problem="The paper evaluates the performance of FL for histopathology image classification using simulated and real-world data.",
        why_existing_work_insufficient="Single-study evaluation was conducted in simulated environment.",
        supporting_statement_ids=[s1_id],
        supporting_paper_ids=[p1_id],
        relevant_limitations=["Limited to simulated environment and real-world hospital data."],
        relevant_future_work=[],
        signals={"pattern_type": "DOCUMENTED_LIMITATION", "contributing_paper_count": 1, "potential_study_specific_limitation": True},
    )

    provider = MockTaxonomyProvider(
        verdict=ValidationVerdict.VALID,
        reasoning="Procedural author disclaimer describing experimental scope of a single paper.",
        cited_ids=[str(s1_id)],
        gap_category=GapCategory.STUDY_LIMITATION,
    )

    validator = GapValidator(provider=provider, min_valid_evidence=1)
    res = validator.validate_candidate(candidate, statements_by_id)

    # Retained as CANDIDATE with single-study scope, not promoted to field-wide conclusion
    assert res.status == GapStatus.CANDIDATE
    assert res.metadata.get("gap_category") == GapCategory.STUDY_LIMITATION.value
    assert res.metadata.get("single_study_scope") is True
    assert res.supporting_paper_ids == [p1_id]


def test_genuinely_unresolved_problem_remains_discoverable():
    """4. A genuinely unresolved, evidence-grounded problem remains discoverable and validated (Category D)."""
    s1_id = uuid.uuid4()
    s2_id = uuid.uuid4()
    p1_id = uuid.uuid4()
    p2_id = uuid.uuid4()

    statements_by_id = {
        s1_id: {
            "id": s1_id,
            "paper_id": p1_id,
            "statement_type": "LIMITATION",
            "statement_text": "Client drift under extreme Dirichlet label skew causes FedAvg to diverge after 50 rounds.",
            "section": "Experiments",
        },
        s2_id: {
            "id": s2_id,
            "paper_id": p2_id,
            "statement_type": "LIMITATION",
            "statement_text": "Normalized gradient clipping fails to prevent weight divergence when label distribution variance exceeds threshold.",
            "section": "Results",
        },
    }

    candidate = GapCandidate(
        gap_statement="Persistent scientific limitation across current methodologies: Client drift under extreme Dirichlet label skew causes FedAvg divergence.",
        research_problem="Weight divergence in non-IID federated optimization.",
        why_existing_work_insufficient="Existing clipping methods fail under extreme skew.",
        supporting_statement_ids=[s1_id, s2_id],
        supporting_paper_ids=[p1_id, p2_id],
        relevant_limitations=["FedAvg diverges after 50 rounds.", "Gradient clipping fails under skew."],
        relevant_future_work=[],
        signals={"pattern_type": "RECURRING_LIMITATIONS", "contributing_paper_count": 2},
    )

    provider = MockTaxonomyProvider(
        verdict=ValidationVerdict.VALID,
        reasoning="Concrete empirical bottleneck replicated across 2 independent studies.",
        cited_ids=[str(s1_id), str(s2_id)],
        gap_category=GapCategory.EVIDENCE_GROUNDED_GAP,
    )

    validator = GapValidator(provider=provider, min_valid_evidence=1)
    res = validator.validate_candidate(candidate, statements_by_id)

    # Validated and eligible for opportunity generation
    assert res.status == GapStatus.VALIDATED
    assert res.metadata.get("gap_category") == GapCategory.EVIDENCE_GROUNDED_GAP.value
    assert set(res.supporting_statement_ids) == {s1_id, s2_id}
    assert res.evidence_count == 2
    assert res.validation_confidence == 0.90


def test_evidence_provenance_and_ids_strictly_preserved():
    """5. Evidence IDs and source provenance remain intact."""
    s1_id = uuid.uuid4()
    p1_id = uuid.uuid4()

    statements_by_id = {
        s1_id: {
            "id": s1_id,
            "paper_id": p1_id,
            "statement_type": "LIMITATION",
            "statement_text": "Differential privacy noise reduces classification AUROC by 14% on rare tumor classes.",
            "section": "Privacy-Utility Tradeoff",
            "page_start": 7,
            "page_end": 8,
        }
    }

    candidate = GapCandidate(
        gap_statement="Empirical limitation: DP noise degrades rare-class performance.",
        research_problem="Privacy-utility tradeoff for rare clinical phenotypes.",
        why_existing_work_insufficient="Noise disproportionately harms low-prevalence classes.",
        supporting_statement_ids=[s1_id],
        supporting_paper_ids=[p1_id],
        relevant_limitations=["AUROC drops by 14% on rare classes."],
        relevant_future_work=[],
    )

    provider = MockTaxonomyProvider(
        verdict=ValidationVerdict.VALID,
        reasoning="Empirically substantiated in rare tumor class experiments.",
        cited_ids=[str(s1_id)],
    )

    validator = GapValidator(provider=provider, min_valid_evidence=1)
    res = validator.validate_candidate(candidate, statements_by_id)

    assert len(res.evidence_items) == 1
    ev = res.evidence_items[0]
    assert ev.statement_id == s1_id
    assert ev.paper_id == p1_id
    assert ev.section == "Privacy-Utility Tradeoff"
    assert ev.page_start == 7
    assert ev.page_end == 8
    assert "AUROC by 14%" in ev.statement_text


def test_malformed_model_output_and_provider_failures_follow_recovery():
    """6. Malformed model output and provider failures continue to follow existing recovery behavior."""
    s1_id = uuid.uuid4()
    p1_id = uuid.uuid4()

    statements_by_id = {
        s1_id: {
            "id": s1_id,
            "paper_id": p1_id,
            "statement_type": "LIMITATION",
            "statement_text": "Communication overhead is severe.",
        }
    }

    candidate = GapCandidate(
        gap_statement="Communication overhead limitation.",
        research_problem="Bandwidth constraints.",
        why_existing_work_insufficient="Network bottlenecks.",
        supporting_statement_ids=[s1_id],
        supporting_paper_ids=[p1_id],
        relevant_limitations=["Overhead is severe."],
        relevant_future_work=[],
    )

    # 1. Provider failure must propagate (not masked as scientific domain rejection)
    provider_fail = MockTaxonomyProvider(raise_provider_failure=True)
    validator_fail = GapValidator(provider=provider_fail, min_valid_evidence=1)
    with pytest.raises(ProviderFailureError):
        validator_fail.validate_candidate(candidate, statements_by_id)

    # 2. Malformed output falls back to safe REJECTED DTO with MODEL_INFERENCE_ERROR metadata
    provider_malformed = MockTaxonomyProvider(raise_malformed=True)
    validator_malformed = GapValidator(provider=provider_malformed, min_valid_evidence=1)
    res_malformed = validator_malformed.validate_candidate(candidate, statements_by_id)

    assert res_malformed.status == GapStatus.REJECTED
    assert res_malformed.metadata.get("rejection_reason") == "MODEL_INFERENCE_ERROR"
    assert "MalformedResponseError" in res_malformed.validation_reasoning or "provider error" in res_malformed.validation_reasoning.lower()


def test_omitted_confidence_preserves_none_across_all_statuses():
    """7. Missing confidence remains None (not fabricated 0.85, 0.75, or 0.0) across all gap statuses."""
    s1_id = uuid.uuid4()
    p1_id = uuid.uuid4()
    s2_id = uuid.uuid4()
    p2_id = uuid.uuid4()

    statements_by_id = {
        s1_id: {
            "id": s1_id,
            "paper_id": p1_id,
            "statement_type": "LIMITATION",
            "statement_text": "Gradient noise destabilizes local models.",
            "section": "Experiments",
        },
        s2_id: {
            "id": s2_id,
            "paper_id": p2_id,
            "statement_type": "PROBLEM",
            "statement_text": "Data heterogeneity slows convergence across clients.",
            "section": "Introduction",
        },
    }

    # Case A: VALIDATED status with confidence=None
    cand_validated = GapCandidate(
        gap_statement="Empirical limitation: Gradient noise destabilizes local models.",
        research_problem="Local model instability.",
        why_existing_work_insufficient="Existing optimizers diverge.",
        supporting_statement_ids=[s1_id],
        supporting_paper_ids=[p1_id],
        relevant_limitations=["Gradient noise destabilizes local models."],
        relevant_future_work=[],
    )
    provider_val = MockTaxonomyProvider(
        verdict=ValidationVerdict.VALID,
        reasoning="Empirically verified without model-reported confidence.",
        cited_ids=[str(s1_id)],
        confidence=None,
        gap_category=GapCategory.EVIDENCE_GROUNDED_GAP,
    )
    res_val = GapValidator(provider=provider_val, min_valid_evidence=1).validate_candidate(cand_validated, statements_by_id)
    assert res_val.status == GapStatus.VALIDATED
    assert res_val.validation_confidence is None

    # Case B: CANDIDATE status (BROAD_CHALLENGE) with confidence=None
    cand_broad = GapCandidate(
        gap_statement="High-level challenge: Data heterogeneity slows convergence.",
        research_problem="Convergence delays.",
        why_existing_work_insufficient="Heterogeneity remains open.",
        supporting_statement_ids=[s2_id],
        supporting_paper_ids=[p2_id],
        relevant_limitations=[],
        relevant_future_work=[],
    )
    provider_broad = MockTaxonomyProvider(
        verdict=ValidationVerdict.VALID,
        reasoning="Genuine but broad problem.",
        cited_ids=[str(s2_id)],
        confidence=None,
        gap_category=GapCategory.BROAD_CHALLENGE,
        needs_refinement=True,
    )
    res_broad = GapValidator(provider=provider_broad, min_valid_evidence=1).validate_candidate(cand_broad, statements_by_id)
    assert res_broad.status == GapStatus.CANDIDATE
    assert res_broad.validation_confidence is None
    assert res_broad.metadata.get("needs_refinement") is True

    # Case C: CANDIDATE status (STUDY_LIMITATION) with confidence=None
    provider_single = MockTaxonomyProvider(
        verdict=ValidationVerdict.VALID,
        reasoning="Single-study procedural disclaimer.",
        cited_ids=[str(s1_id)],
        confidence=None,
        gap_category=GapCategory.STUDY_LIMITATION,
    )
    res_single = GapValidator(provider=provider_single, min_valid_evidence=1).validate_candidate(cand_validated, statements_by_id)
    assert res_single.status == GapStatus.CANDIDATE
    assert res_single.validation_confidence is None
    assert res_single.metadata.get("single_study_scope") is True

    # Case D: REJECTED status (BACKGROUND_MOTIVATION) with confidence=None
    provider_bg = MockTaxonomyProvider(
        verdict=ValidationVerdict.VALID,
        reasoning="Standard paradigm motivation.",
        cited_ids=[str(s1_id)],
        confidence=None,
        gap_category=GapCategory.BACKGROUND_MOTIVATION,
    )
    res_bg = GapValidator(provider=provider_bg, min_valid_evidence=1).validate_candidate(cand_validated, statements_by_id)
    assert res_bg.status == GapStatus.REJECTED
    assert res_bg.validation_confidence is None
    assert res_bg.metadata.get("rejection_reason") == "BACKGROUND_MOTIVATION"

    # Case E: REJECTED status (explicit REJECT verdict) with confidence=None
    provider_reject = MockTaxonomyProvider(
        verdict=ValidationVerdict.REJECT,
        reasoning="Candidate is unsupported.",
        cited_ids=[str(s1_id)],
        confidence=None,
    )
    res_reject = GapValidator(provider=provider_reject, min_valid_evidence=1).validate_candidate(cand_validated, statements_by_id)
    assert res_reject.status == GapStatus.REJECTED
    assert res_reject.validation_confidence is None


def test_explicit_confidence_preserved_and_bounded():
    """8. Explicit model-produced confidence values are preserved and bounded to [0.0, 1.0]."""
    s1_id = uuid.uuid4()
    p1_id = uuid.uuid4()
    statements_by_id = {
        s1_id: {
            "id": s1_id,
            "paper_id": p1_id,
            "statement_type": "LIMITATION",
            "statement_text": "Sample efficiency drops under partial client participation.",
        }
    }
    candidate = GapCandidate(
        gap_statement="Sample efficiency drops under partial client participation.",
        research_problem="Partial client participation.",
        why_existing_work_insufficient="Unaddressed drop in efficiency.",
        supporting_statement_ids=[s1_id],
        supporting_paper_ids=[p1_id],
        relevant_limitations=[],
        relevant_future_work=[],
    )

    # Standard float
    prov1 = MockTaxonomyProvider(cited_ids=[str(s1_id)], confidence=0.88)
    res1 = GapValidator(provider=prov1, min_valid_evidence=1).validate_candidate(candidate, statements_by_id)
    assert res1.validation_confidence == 0.88

    # Edge cases 0.0 and 1.0
    prov2 = MockTaxonomyProvider(cited_ids=[str(s1_id)], confidence=0.0)
    res2 = GapValidator(provider=prov2, min_valid_evidence=1).validate_candidate(candidate, statements_by_id)
    assert res2.validation_confidence == 0.0

    prov3 = MockTaxonomyProvider(cited_ids=[str(s1_id)], confidence=1.0)
    res3 = GapValidator(provider=prov3, min_valid_evidence=1).validate_candidate(candidate, statements_by_id)
    assert res3.validation_confidence == 1.0


def test_classification_and_provenance_unchanged_when_confidence_missing():
    """9. Missing confidence does not affect taxonomy classification, evidence provenance, or metadata."""
    s1_id = uuid.uuid4()
    p1_id = uuid.uuid4()
    statements_by_id = {
        s1_id: {
            "id": s1_id,
            "paper_id": p1_id,
            "statement_type": "LIMITATION",
            "statement_text": "Straggler mitigation increases total communication rounds.",
            "section": "Systems Evaluation",
            "page_start": 12,
            "page_end": 14,
        }
    }
    candidate = GapCandidate(
        gap_statement="Straggler mitigation increases total communication rounds.",
        research_problem="Straggler effects in synchronous aggregation.",
        why_existing_work_insufficient="Asynchronous protocols trade accuracy for latency.",
        supporting_statement_ids=[s1_id],
        supporting_paper_ids=[p1_id],
        relevant_limitations=["Straggler mitigation increases rounds."],
        relevant_future_work=["Explore hybrid synchronization."],
        signals={"pattern_type": "UNRESOLVED_PROBLEM"},
    )

    prov_with_conf = MockTaxonomyProvider(
        verdict=ValidationVerdict.VALID,
        reasoning="Empirically documented straggler overhead.",
        cited_ids=[str(s1_id)],
        confidence=0.91,
        gap_category=GapCategory.EVIDENCE_GROUNDED_GAP,
    )
    prov_no_conf = MockTaxonomyProvider(
        verdict=ValidationVerdict.VALID,
        reasoning="Empirically documented straggler overhead.",
        cited_ids=[str(s1_id)],
        confidence=None,
        gap_category=GapCategory.EVIDENCE_GROUNDED_GAP,
    )

    validator1 = GapValidator(provider=prov_with_conf, min_valid_evidence=1)
    validator2 = GapValidator(provider=prov_no_conf, min_valid_evidence=1)

    res_with = validator1.validate_candidate(candidate, statements_by_id)
    res_without = validator2.validate_candidate(candidate, statements_by_id)

    # Identical status, category, citations, provenance
    assert res_with.status == res_without.status == GapStatus.VALIDATED
    assert res_with.supporting_statement_ids == res_without.supporting_statement_ids == [s1_id]
    assert res_with.supporting_paper_ids == res_without.supporting_paper_ids == [p1_id]
    assert res_with.evidence_count == res_without.evidence_count == 1
    assert len(res_with.evidence_items) == len(res_without.evidence_items) == 1
    assert res_with.evidence_items[0].page_start == res_without.evidence_items[0].page_start == 12
    assert res_with.metadata.get("gap_category") == res_without.metadata.get("gap_category") == GapCategory.EVIDENCE_GROUNDED_GAP.value

    # Only confidence differs
    assert res_with.validation_confidence == 0.91
    assert res_without.validation_confidence is None
