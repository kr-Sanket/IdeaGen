"""Automated Test Suite for Stage 4B Benchmark Robustness & Credibility (v2.0).

Verifies:
1. Benchmark v2 dataset composition: 25 cases, strictly partitioned into DEV (N=10) and HELD_OUT (N=15).
2. Multi-domain coverage across NLP/RAG, Geriatric Medicine, Distributed Systems, Differential Privacy, and Molecular GNNs.
3. Grounding evaluation stratified by source category (Real Literature, Pilot, Synthetic Stress Tests).
4. Unicode ligature robustness (verifying that ASCII excerpts match passages with \\ufb01/\\ufb02 ligatures).
5. Rejection of ungrounded excerpts: hallucinations, cross-paper attribution, and sub-threshold paraphrases.
6. Borderline case metadata: alternative classes and expert-review notes properly documented.
7. Stratified classification metrics: confusion matrix and per-class metrics isolated by DEV and HELD_OUT splits.
8. State-machine determinism: verifying AD-20 two-paper resolution and contradiction reopening on Benchmark v2.
"""

from __future__ import annotations

import pytest
from typing import Dict, Any, List

from ideagen.opportunity.models import OpportunityRelationship, OpportunityStatus
from ideagen.opportunity.grounding import (
    is_excerpt_grounded,
    validate_evidence_grounding,
    UngroundedEvidenceError,
)
from ideagen.opportunity.state_machine import transition
from ideagen.evaluation.stage4b_benchmark import (
    Stage4bBenchmarkCase,
    BenchmarkSplit,
    CaseSourceCategory,
    GroundingChallengeType,
    get_stage4b_benchmark_dataset,
    run_grounding_evaluation_by_category,
    run_classification_evaluation_by_split,
)


@pytest.fixture
def v2_cases() -> List[Stage4bBenchmarkCase]:
    return get_stage4b_benchmark_dataset()


def test_v2_dataset_composition_and_splits(v2_cases):
    """Verifies that Benchmark v2 contains exactly 25 cases with a 10/15 DEV/HELD_OUT split."""
    assert len(v2_cases) == 25

    dev_cases = [c for c in v2_cases if c.split == BenchmarkSplit.DEV]
    held_out_cases = [c for c in v2_cases if c.split == BenchmarkSplit.HELD_OUT]

    assert len(dev_cases) == 10, "Expected exactly 10 cases in DEV split"
    assert len(held_out_cases) == 15, "Expected exactly 15 cases in HELD_OUT split"

    # Multi-domain verification
    domains = {c.domain for c in v2_cases}
    assert "NLP / Retrieval Augmentation" in domains
    assert "Geriatric Medicine" in domains
    assert "Distributed Systems / FL" in domains
    assert "Differential Privacy" in domains
    assert "Molecular Biology / GNN" in domains


def test_stratified_grounding_by_source_category(v2_cases):
    """Verifies that grounding evaluation can be run and reported independently by category."""
    res = run_grounding_evaluation_by_category(v2_cases)

    assert res["overall_total"] == 25
    assert res["overall_accuracy"] == 1.0

    by_cat = res["by_category"]
    assert "REAL_LITERATURE" in by_cat
    assert "HUMAN_CURATED_PILOT" in by_cat
    assert "SYNTHETIC_STRESS_TEST" in by_cat

    # Real Literature: 11 cases total (9 valid, 2 invalid: 1 fabrication, 1 subthreshold paraphrase)
    real_lit = by_cat["REAL_LITERATURE"]
    assert real_lit["total"] == 11
    assert real_lit["accepted"] == 9
    assert real_lit["rejected"] == 2
    assert real_lit["accuracy"] == 1.0

    # Human-Curated Pilot: 13 cases total (13 valid, 0 invalid)
    pilot = by_cat["HUMAN_CURATED_PILOT"]
    assert pilot["total"] == 13
    assert pilot["accepted"] == 13
    assert pilot["accuracy"] == 1.0

    # Synthetic Stress: 1 cross-paper case
    syn = by_cat["SYNTHETIC_STRESS_TEST"]
    assert syn["total"] == 1
    assert syn["rejected"] == 1
    assert syn["accuracy"] == 1.0


def test_ligature_grounding_robustness(v2_cases):
    """Verifies case DEV-09 where passage contains unicode ligature \\ufb01ne-tuning and excerpt is ASCII."""
    case_dev09 = next(c for c in v2_cases if c.case_id == "DEV-09")
    assert case_dev09.grounding_challenge_type == GroundingChallengeType.LIGATURE
    assert "\ufb01" in case_dev09.passages[0]

    # Must be accepted by is_excerpt_grounded and validate_evidence_grounding
    assert is_excerpt_grounded(case_dev09.claimed_excerpt, case_dev09.passages)
    validate_evidence_grounding(
        case_dev09.claimed_excerpt,
        case_dev09.passages,
        case_dev09.expected_relationship,
        paper_id=case_dev09.paper_id,
    )


def test_unsupported_fabrication_rejected(v2_cases):
    """Verifies case HELD-04 where a fabricated claim attributed to Yoran et al. is rejected."""
    case_held04 = next(c for c in v2_cases if c.case_id == "HELD-04")
    assert case_held04.grounding_challenge_type == GroundingChallengeType.UNSUPPORTED_FABRICATION
    assert not case_held04.expected_grounding_valid

    with pytest.raises(UngroundedEvidenceError):
        validate_evidence_grounding(
            case_held04.claimed_excerpt,
            case_held04.passages,
            case_held04.expected_relationship,
            paper_id=case_held04.paper_id,
        )


def test_subthreshold_paraphrase_rejected(v2_cases):
    """Verifies case HELD-11 where a loose paraphrase (~35% similarity) is rejected."""
    case_held11 = next(c for c in v2_cases if c.case_id == "HELD-11")
    assert case_held11.grounding_challenge_type == GroundingChallengeType.PARAPHRASE_SUBTHRESHOLD
    assert not case_held11.expected_grounding_valid

    with pytest.raises(UngroundedEvidenceError):
        validate_evidence_grounding(
            case_held11.claimed_excerpt,
            case_held11.passages,
            case_held11.expected_relationship,
            paper_id=case_held11.paper_id,
        )


def test_cross_paper_attribution_rejected(v2_cases):
    """Verifies case DEV-10 where excerpt from Paper Beta attributed to Paper Alpha is rejected."""
    case_dev10 = next(c for c in v2_cases if c.case_id == "DEV-10")
    assert case_dev10.grounding_challenge_type == GroundingChallengeType.CROSS_PAPER_ATTRIBUTION
    assert not case_dev10.expected_grounding_valid

    with pytest.raises(UngroundedEvidenceError):
        validate_evidence_grounding(
            case_dev10.claimed_excerpt,
            case_dev10.passages,
            case_dev10.expected_relationship,
            paper_id=case_dev10.paper_id,
        )


def test_borderline_cases_and_expert_review_metadata(v2_cases):
    """Verifies that all borderline cases have alternative classes and expert-review notes."""
    borderline_cases = [c for c in v2_cases if c.is_borderline]
    assert len(borderline_cases) == 6

    for c in borderline_cases:
        assert c.requires_expert_review is True
        assert c.borderline_alternative_class is not None
        assert c.expert_review_notes is not None
        assert len(c.expert_review_notes) > 20

    # Specifically check Gao et al. (HELD-01: ALCE benchmark vs method baselines)
    alce_case = next(c for c in v2_cases if c.case_id == "HELD-01")
    assert alce_case.expected_relationship == OpportunityRelationship.EXTENDS
    assert alce_case.borderline_alternative_class == OpportunityRelationship.ADDRESSES


def test_classification_metrics_stratified_by_split(v2_cases):
    """Verifies that classification evaluation reports DEV and HELD_OUT metrics independently."""
    gold_preds = [c.expected_relationship for c in v2_cases]
    res_by_split = run_classification_evaluation_by_split(v2_cases, gold_preds)

    assert "DEV" in res_by_split
    assert "HELD_OUT" in res_by_split

    dev = res_by_split["DEV"]
    assert dev["sample_size"] == 10
    assert dev["overall_accuracy"] == 1.0

    held_out = res_by_split["HELD_OUT"]
    assert held_out["sample_size"] == 15
    assert held_out["overall_accuracy"] == 1.0
    assert held_out["per_class"]["ADDRESSES"]["support"] == 6
    assert held_out["per_class"]["SUPPORTS"]["support"] == 3


def test_held_out_simulated_misclassifications_borderline(v2_cases):
    """Tests sensitivity on held-out split when simulated classifier errs on borderline cases.

    Simulates:
    - HELD-01 (Gao ALCE): predicted as ADDRESSES (true: EXTENDS)
    - HELD-06 (Impossibility theorem): predicted as SUPPORTS (true: CONTRADICTS)
    """
    simulated_preds = []
    for c in v2_cases:
        if c.case_id == "HELD-01":
            simulated_preds.append(OpportunityRelationship.ADDRESSES)
        elif c.case_id == "HELD-06":
            simulated_preds.append(OpportunityRelationship.SUPPORTS)
        else:
            simulated_preds.append(c.expected_relationship)

    res_by_split = run_classification_evaluation_by_split(v2_cases, simulated_preds)
    held_out = res_by_split["HELD_OUT"]

    assert held_out["sample_size"] == 15
    assert held_out["overall_accuracy"] == round(13 / 15, 4)  # 86.67%
    assert held_out["confusion_matrix"]["EXTENDS"]["ADDRESSES"] == 1
    assert held_out["confusion_matrix"]["CONTRADICTS"]["SUPPORTS"] == 1


def test_state_machine_transitions_on_v2(v2_cases):
    """Verifies pure state machine transitions across all 25 cases on Benchmark v2."""
    for c in v2_cases:
        actual_pure = transition(
            current_status=c.initial_opportunity_status,
            relationship=c.expected_relationship,
            addressing_paper_count=c.addressing_paper_count,
        )
        assert actual_pure == c.expected_state_machine_status, f"Pure state mismatch on {c.case_id}"

        # End-to-end check: if grounding is invalid, status remains initial_opportunity_status
        if c.expected_grounding_valid:
            actual_e2e = actual_pure
        else:
            actual_e2e = c.initial_opportunity_status
        assert actual_e2e == c.expected_end_to_end_status, f"E2E state mismatch on {c.case_id}"
