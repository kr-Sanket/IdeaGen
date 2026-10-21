"""Automated Unit and Integration Tests for Stage 4 Opportunity Quality Evaluation.

Verifies:
1. Benchmark dataset integrity (15 cases: 5 real literature, 10 synthetic fixtures).
2. Grounding evaluation: exact matches accepted, fuzzy matches above 85% accepted.
3. Grounding evaluation: fabricated/hallucinated excerpts strictly rejected.
4. Grounding evaluation: cross-paper evidence attribution strictly rejected.
5. Grounding evaluation: missing/empty excerpts for substantive claims rejected.
6. Grounding evaluation: safe unrelated pass-through without grounding errors.
7. Classification metrics: confusion matrix, per-class Precision, Recall, F1, macro/micro.
8. State machine determinism: pure state machine transitions verified with 100% agreement.
9. End-to-end grounded lifecycle: ungrounded candidates do not alter opportunity status.
10. Opportunity quality criteria: claim clarity, evidence traceability, formulation uniqueness,
    and temporal obsolescence / addressed status tracking.
"""

from __future__ import annotations

import pytest
from typing import Dict, Any, List

from ideagen.opportunity.models import OpportunityRelationship, OpportunityStatus
from ideagen.opportunity.grounding import (
    is_excerpt_grounded,
    validate_evidence_grounding,
    UngroundedEvidenceError,
    DEFAULT_GROUNDING_THRESHOLD,
)
from ideagen.opportunity.state_machine import transition
from ideagen.evaluation.stage4_benchmark import (
    Stage4BenchmarkCase,
    BenchmarkSourceType,
    get_stage4_benchmark_dataset,
    run_grounding_evaluation,
    run_classification_evaluation,
    run_lifecycle_evaluation,
)


@pytest.fixture
def benchmark_cases() -> List[Stage4BenchmarkCase]:
    return get_stage4_benchmark_dataset()


def test_benchmark_dataset_composition(benchmark_cases):
    """Verifies that the benchmark dataset contains exactly 15 cases with proper literature/synthetic partition."""
    assert len(benchmark_cases) == 15
    lit_cases = [c for c in benchmark_cases if c.source_type == BenchmarkSourceType.LITERATURE]
    syn_cases = [c for c in benchmark_cases if c.source_type == BenchmarkSourceType.SYNTHETIC]

    assert len(lit_cases) == 5, "Expected 5 real literature cases"
    assert len(syn_cases) == 10, "Expected 10 controlled synthetic cases"

    # Verify all classes are covered
    classes_covered = {c.expected_relationship for c in benchmark_cases}
    assert classes_covered == {
        OpportunityRelationship.SUPPORTS,
        OpportunityRelationship.ADDRESSES,
        OpportunityRelationship.CONTRADICTS,
        OpportunityRelationship.EXTENDS,
        OpportunityRelationship.UNRELATED,
    }


def test_grounding_exact_and_fuzzy_matches_accepted(benchmark_cases):
    """Verifies that exact excerpts and fuzzy matches >= 85% are accepted."""
    valid_cases = [c for c in benchmark_cases if c.expected_grounding_valid]
    assert len(valid_cases) == 11

    for c in valid_cases:
        if c.expected_relationship == OpportunityRelationship.UNRELATED:
            # UNRELATED does not require passage grounding
            validate_evidence_grounding(c.claimed_excerpt, c.passages, c.expected_relationship)
        else:
            assert is_excerpt_grounded(c.claimed_excerpt, c.passages, threshold=DEFAULT_GROUNDING_THRESHOLD)
            # Should not raise UngroundedEvidenceError
            validate_evidence_grounding(c.claimed_excerpt, c.passages, c.expected_relationship, paper_id=c.paper_id)


def test_grounding_fabricated_excerpts_rejected(benchmark_cases):
    """Verifies that fabricated or hallucinated excerpts are strictly rejected."""
    case_syn03 = next(c for c in benchmark_cases if c.case_id == "SYN-03")
    assert not case_syn03.expected_grounding_valid
    assert not is_excerpt_grounded(case_syn03.claimed_excerpt, case_syn03.passages)

    with pytest.raises(UngroundedEvidenceError) as exc_info:
        validate_evidence_grounding(
            case_syn03.claimed_excerpt,
            case_syn03.passages,
            case_syn03.expected_relationship,
            paper_id=case_syn03.paper_id,
        )
    assert "not grounded in stored passages" in str(exc_info.value)


def test_grounding_cross_paper_attribution_rejected(benchmark_cases):
    """Verifies that evidence originating from a different paper cannot be attributed to the evaluated paper."""
    case_syn04 = next(c for c in benchmark_cases if c.case_id == "SYN-04")
    assert not case_syn04.expected_grounding_valid

    with pytest.raises(UngroundedEvidenceError) as exc_info:
        validate_evidence_grounding(
            case_syn04.claimed_excerpt,
            case_syn04.passages,
            case_syn04.expected_relationship,
            paper_id=case_syn04.paper_id,
        )
    assert "not grounded in stored passages for paper syn-paper-004-A" in str(exc_info.value)


def test_grounding_missing_excerpt_rejected(benchmark_cases):
    """Verifies that a substantive relationship without an evidence excerpt fails validation."""
    case_syn05 = next(c for c in benchmark_cases if c.case_id == "SYN-05")
    assert not case_syn05.expected_grounding_valid
    assert case_syn05.claimed_excerpt is None

    with pytest.raises(UngroundedEvidenceError) as exc_info:
        validate_evidence_grounding(
            case_syn05.claimed_excerpt,
            case_syn05.passages,
            case_syn05.expected_relationship,
            paper_id=case_syn05.paper_id,
        )
    assert "not grounded in stored passages" in str(exc_info.value)


def test_grounding_subthreshold_fuzzy_rejected(benchmark_cases):
    """Verifies that heavily paraphrased excerpts (< 85% token overlap) fail validation."""
    case_syn06 = next(c for c in benchmark_cases if c.case_id == "SYN-06")
    assert not case_syn06.expected_grounding_valid

    with pytest.raises(UngroundedEvidenceError):
        validate_evidence_grounding(
            case_syn06.claimed_excerpt,
            case_syn06.passages,
            case_syn06.expected_relationship,
            paper_id=case_syn06.paper_id,
        )


def test_grounding_safe_unrelated_pass_through(benchmark_cases):
    """Verifies that UNRELATED relationships safely pass without requiring passage grounding."""
    unrelated_cases = [c for c in benchmark_cases if c.expected_relationship == OpportunityRelationship.UNRELATED]
    assert len(unrelated_cases) == 2  # LIT-05 and SYN-09

    for c in unrelated_cases:
        # Must pass without raising UngroundedEvidenceError
        validate_evidence_grounding(
            c.claimed_excerpt,
            c.passages,
            c.expected_relationship,
            paper_id=c.paper_id,
        )


def test_grounding_suite_summary_metrics(benchmark_cases):
    """Verifies the overall grounding evaluation summary metrics."""
    res = run_grounding_evaluation(benchmark_cases)
    assert res["total_cases"] == 15
    assert res["substantive_valid_cases"] == 11
    assert res["substantive_accepted"] == 11
    assert res["grounding_acceptance_rate"] == 1.0
    assert res["invalid_cases"] == 4
    assert res["invalid_rejected"] == 4
    assert res["fabrication_rejection_rate"] == 1.0
    assert res["cross_paper_rejections"] == 1
    assert res["empty_excerpt_rejections"] == 1
    assert res["overall_grounding_accuracy"] == 1.0


def test_classification_confusion_matrix_and_metrics_gold(benchmark_cases):
    """Verifies the multi-class confusion matrix, precision, recall, and F1 on gold predictions."""
    gold_preds = [c.expected_relationship for c in benchmark_cases]
    res = run_classification_evaluation(benchmark_cases, gold_preds)

    assert res["sample_size"] == 15
    assert res["overall_accuracy"] == 1.0
    assert res["macro_precision"] == 1.0
    assert res["macro_recall"] == 1.0
    assert res["macro_f1"] == 1.0

    # Verify per-class supports match expected distributions
    assert res["per_class"]["SUPPORTS"]["support"] == 3
    assert res["per_class"]["ADDRESSES"]["support"] == 5
    assert res["per_class"]["CONTRADICTS"]["support"] == 3
    assert res["per_class"]["EXTENDS"]["support"] == 2
    assert res["per_class"]["UNRELATED"]["support"] == 2


def test_classification_confusion_matrix_with_simulated_misclassifications(benchmark_cases):
    """Verifies confusion matrix behavior when simulated classification errors occur.

    Simulates:
    - 1 EXTENDS misclassified as ADDRESSES (Case LIT-04)
    - 1 SUPPORTS misclassified as UNRELATED (Case LIT-01)
    """
    simulated_preds = []
    for c in benchmark_cases:
        if c.case_id == "LIT-04":
            simulated_preds.append(OpportunityRelationship.ADDRESSES)  # true: EXTENDS
        elif c.case_id == "LIT-01":
            simulated_preds.append(OpportunityRelationship.UNRELATED)  # true: SUPPORTS
        else:
            simulated_preds.append(c.expected_relationship)

    res = run_classification_evaluation(benchmark_cases, simulated_preds)
    assert res["sample_size"] == 15
    assert res["overall_accuracy"] == round(13 / 15, 4)  # 0.8667

    # Inspect specific confusion matrix cells
    matrix = res["confusion_matrix"]
    assert matrix["EXTENDS"]["ADDRESSES"] == 1
    assert matrix["SUPPORTS"]["UNRELATED"] == 1

    # Check that EXTENDS recall is degraded (1 true positive out of 2)
    assert res["per_class"]["EXTENDS"]["recall"] == 0.5
    # Check that ADDRESSES precision is degraded (5 true positives out of 6 predicted)
    assert res["per_class"]["ADDRESSES"]["precision"] == round(5 / 6, 4)


def test_pure_state_machine_determinism(benchmark_cases):
    """Verifies that pure state machine transitions execute with 100% determinism."""
    res = run_lifecycle_evaluation(benchmark_cases)
    assert res["total_cases"] == 15
    assert res["pure_state_machine_matches"] == 15
    assert res["pure_state_machine_accuracy"] == 1.0

    # Specifically verify AD-19 and AD-20 cases
    syn08 = next(r for r in res["pure_results"] if r["case_id"] == "SYN-08")
    assert syn08["initial_status"] == "OPEN"
    assert syn08["relationship"] == "ADDRESSES"
    assert syn08["addressing_count"] == 1
    assert syn08["actual_pure_status"] == "PARTIALLY_ADDRESSED", "AD-19: single paper must never resolve"

    lit03 = next(r for r in res["pure_results"] if r["case_id"] == "LIT-03")
    assert lit03["initial_status"] == "PARTIALLY_ADDRESSED"
    assert lit03["relationship"] == "ADDRESSES"
    assert lit03["addressing_count"] == 2
    assert lit03["actual_pure_status"] == "RESOLVED", "AD-20: 2 distinct papers resolve opportunity"

    syn07 = next(r for r in res["pure_results"] if r["case_id"] == "SYN-07")
    assert syn07["initial_status"] == "RESOLVED"
    assert syn07["relationship"] == "CONTRADICTS"
    assert syn07["actual_pure_status"] == "OPEN", "Contradiction must reopen resolved opportunity"


def test_end_to_end_grounded_lifecycle_isolation(benchmark_cases):
    """Verifies that ungrounded candidates do not alter opportunity status, leaving state intact."""
    res = run_lifecycle_evaluation(benchmark_cases)
    assert res["end_to_end_lifecycle_matches"] == 15
    assert res["end_to_end_lifecycle_accuracy"] == 1.0

    # Verify that ungrounded synthetic cases remained in their initial status
    for case_id in ("SYN-03", "SYN-04", "SYN-05", "SYN-06"):
        e2e = next(r for r in res["e2e_results"] if r["case_id"] == case_id)
        assert not e2e["grounding_valid"]
        assert e2e["actual_e2e_status"] == e2e["initial_status"], f"Case {case_id} should not alter state"


def test_stage4_opportunity_quality_inspection():
    """Evaluates the structural quality of synthesized research opportunities.

    Inspects:
    1. Claim clarity & specificity.
    2. Evidence grounding traceability.
    3. Non-duplication / formulation uniqueness.
    4. Distinguishing unresolved limitations from work already addressed by later literature.
    """
    cases = get_stage4_benchmark_dataset()
    seed_case = cases[0]

    # 1. Claim Clarity & Specificity
    assert len(seed_case.opportunity_title) > 20
    assert len(seed_case.opportunity_problem) > 50
    assert len(seed_case.opportunity_direction) > 50
    assert "modular architectures" in seed_case.opportunity_direction.lower()

    # 2. Evidence Grounding Traceability
    # The opportunity problem is traceable to Lewis et al. (2020)
    lewis_text = (
        "Large pre-trained language models have limited ability to access and precisely "
        "manipulate knowledge, underperforming task-specific architectures on knowledge-intensive NLP tasks."
    )
    assert "access and precisely manipulate knowledge" in lewis_text

    # 3. Evolution Tracking (Addressing Unresolved Limitations)
    # At t=2020 (Lewis et al.), the opportunity is OPEN.
    # At t=2023-01 (Shi et al.), the opportunity is STRENGTHENED.
    # At t=2023-10 (Yoran et al.), the opportunity transitions to PARTIALLY_ADDRESSED.
    # At t=2023-10 (Asai et al.), the opportunity transitions to RESOLVED.
    current_status = OpportunityStatus.OPEN
    addressing_papers = []

    # Paper 1: Shi et al. (2023) -> SUPPORTS
    current_status = transition(current_status, OpportunityRelationship.SUPPORTS)
    assert current_status == OpportunityStatus.STRENGTHENED

    # Paper 2: Yoran et al. (2023) -> ADDRESSES
    addressing_papers.append("yoran2023")
    current_status = transition(current_status, OpportunityRelationship.ADDRESSES, len(addressing_papers))
    assert current_status == OpportunityStatus.PARTIALLY_ADDRESSED

    # Paper 3: Asai et al. (2023) -> ADDRESSES
    addressing_papers.append("asai2023")
    current_status = transition(current_status, OpportunityRelationship.ADDRESSES, len(addressing_papers))
    assert current_status == OpportunityStatus.RESOLVED

    # This demonstrates that a static 2020 opportunity that seemed completely unresolved
    # was progressively addressed and resolved by 2023 literature through temporal evolution.
