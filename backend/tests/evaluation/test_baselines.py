"""Unit tests for TFIDFDiscoveryBaseline (Phase 6).

Verifies:
1. Baseline construction and configuration.
2. Identification of obvious limitation statements.
3. Exclusion of ordinary positive findings.
4. Handling of negative phrases such as "no limitations were observed".
5. Empty input handling.
6. One candidate statement handling.
7. No candidate statements remaining after filtering.
8. Multiple statements that are lexically similar group together.
9. Statements about unrelated limitations separate into distinct clusters.
10. Stable results across repeated calls with identical configuration.
11. Exactly one cluster assignment per candidate.
12. Non-candidate statements do not appear in predicted clusters.
13. Correct mapping from predicted assignments to original statement IDs.
14. Invalid input handling.
15. Empty vocabulary and tokenization edge cases.
16. Correct registration of the TF-IDF baseline in get_baselines().
17. Compatibility of baseline output with pairwise_cluster_metrics.
"""

import pytest

from ideagen.evaluation.baselines import (
    TFIDFDiscoveryBaseline,
    TFIDFBaselineResult,
    get_baselines,
)
from ideagen.evaluation.metrics import pairwise_cluster_metrics
from ideagen.understanding.models import ResearchStatementCreateDTO, StatementType


def test_baseline_construction_and_configuration():
    """Requirement 1: Baseline construction and parameter configuration."""
    baseline = TFIDFDiscoveryBaseline(
        similarity_threshold=0.4,
        ngram_range=(1, 3),
        stop_words="english",
    )
    # distance = max(0.01, 1 - similarity_threshold) = 0.6
    assert pytest.approx(baseline.distance_threshold, abs=1e-5) == 0.6
    assert baseline.ngram_range == (1, 3)
    assert baseline.stop_words == "english"

    # Direct distance_threshold constructor
    baseline_direct = TFIDFDiscoveryBaseline(distance_threshold=0.55)
    assert pytest.approx(baseline_direct.distance_threshold, abs=1e-5) == 0.55


def test_identification_of_obvious_limitations():
    """Requirement 2: Identification of obvious limitation statements."""
    baseline = TFIDFDiscoveryBaseline()
    examples = [
        "A critical limitation of current RAG models is factual hallucination.",
        "Future work should address latency bottlenecks in autoregressive decoding.",
        "The proposed approach fails to generalize to low-resource languages.",
        "Model calibration remains challenging under heavy domain shift.",
        "The dense vector index suffers from substantial GPU memory overhead.",
        "The retrieval pipeline cannot handle multi-step reasoning queries.",
        "Lack of labeled counterfactual training data restricts performance.",
    ]
    for text in examples:
        assert baseline.is_limitation_candidate(text) is True, f"Failed to identify: {text}"


def test_exclusion_of_ordinary_positive_findings():
    """Requirement 3: Exclusion of ordinary positive findings and contribution claims."""
    baseline = TFIDFDiscoveryBaseline()
    examples = [
        "Our method achieves state-of-the-art accuracy of 92.4% on GSM8K.",
        "We introduce a novel self-attention mechanism that speeds up training by 2x.",
        "The experimental results demonstrate strong empirical convergence across all seeds.",
        "Fine-tuning BERT on domain data is sufficient for strong classification accuracy.",
    ]
    for text in examples:
        assert baseline.is_limitation_candidate(text) is False, f"Incorrectly flagged: {text}"


def test_handling_of_negative_phrases():
    """Requirement 4: Exclusion of qualified phrases like 'no limitations were observed'."""
    baseline = TFIDFDiscoveryBaseline()
    examples = [
        "In our extensive evaluation, no limitations were observed in output fidelity.",
        "The model scales efficiently without limitations on context window length.",
        "This architectural choice is not a limitation in practical deployments.",
        "We observed no significant limitation during real-time benchmark streaming.",
    ]
    for text in examples:
        assert baseline.is_limitation_candidate(text) is False, f"False positive on negated phrase: {text}"


def test_empty_input():
    """Requirement 5: Empty input returns documented empty result without error."""
    baseline = TFIDFDiscoveryBaseline()
    res = baseline.fit_predict([])

    assert isinstance(res, TFIDFBaselineResult)
    assert len(res) == 0
    assert res.clusters == []
    assert res.cluster_labels == {}
    assert res.candidate_statements == []
    assert res.filtered_out_statements == []
    assert res.parameters["total_input_statements"] == 0


def test_one_candidate_statement():
    """Requirement 6: Single candidate statement forms exactly one singleton cluster."""
    baseline = TFIDFDiscoveryBaseline()
    statements = [
        {"id": "stmt_pos", "statement_text": "We achieve strong empirical performance."},
        {"id": "stmt_lim", "statement_text": "A key limitation is vulnerability to prompt injection."},
    ]
    res = baseline.fit_predict(statements)

    assert len(res) == 1
    assert res.clusters == [{"stmt_lim"}]
    assert res.cluster_labels == {"stmt_lim": 0}
    assert len(res.candidate_statements) == 1
    assert len(res.filtered_out_statements) == 1
    assert res.filtered_out_statements[0]["id"] == "stmt_pos"


def test_no_candidates_after_filtering():
    """Requirement 7: Input with only non-limitation statements returns empty clusters."""
    baseline = TFIDFDiscoveryBaseline()
    statements = [
        {"id": "s1", "statement_text": "We introduce a fast matrix multiplication kernel."},
        {"id": "s2", "statement_text": "The benchmark achieves 99% precision."},
    ]
    res = baseline.fit_predict(statements)

    assert len(res) == 0
    assert res.clusters == []
    assert res.cluster_labels == {}
    assert len(res.candidate_statements) == 0
    assert len(res.filtered_out_statements) == 2


def test_multiple_lexically_similar_statements():
    """Requirement 8: Multiple lexically similar limitation statements cluster together."""
    baseline = TFIDFDiscoveryBaseline(distance_threshold=0.85)
    statements = [
        {"id": "rag_1", "statement_text": "Dense retrieval suffers from severe hallucination in RAG pipelines."},
        {"id": "rag_2", "statement_text": "RAG models fail to prevent retrieval hallucination on long document queries."},
        {"id": "rag_3", "statement_text": "Retrieval hallucination remains an open challenge for RAG systems."},
    ]
    res = baseline.fit_predict(statements)

    # All three statements discuss retrieval hallucination in RAG and should be in 1 cluster
    assert len(res) == 1
    assert res.clusters[0] == {"rag_1", "rag_2", "rag_3"}
    assert res.cluster_labels["rag_1"] == res.cluster_labels["rag_2"] == res.cluster_labels["rag_3"]


def test_statements_about_unrelated_limitations():
    """Requirement 9: Statements about distinct, unrelated limitations form separate clusters."""
    baseline = TFIDFDiscoveryBaseline(distance_threshold=0.85)
    statements = [
        {"id": "rag_1", "statement_text": "Dense retrieval suffers from severe hallucination in RAG pipelines."},
        {"id": "rag_2", "statement_text": "RAG models fail to prevent retrieval hallucination on long document queries."},
        {"id": "quant_1", "statement_text": "Quantization suffers from catastrophic activation outlier clipping."},
        {"id": "quant_2", "statement_text": "Activation outliers remain an unresolved bottleneck in 4-bit quantization."},
    ]
    res = baseline.fit_predict(statements)

    # Should separate into 2 clusters: RAG hallucination vs Quantization outliers
    assert len(res) == 2
    # Verify RAG statements are clustered together
    assert res.cluster_labels["rag_1"] == res.cluster_labels["rag_2"]
    # Verify Quantization statements are clustered together
    assert res.cluster_labels["quant_1"] == res.cluster_labels["quant_2"]
    # Verify RAG and Quantization are in different clusters
    assert res.cluster_labels["rag_1"] != res.cluster_labels["quant_1"]


def test_stable_results_across_repeated_calls():
    """Requirement 10: Deterministic results across multiple calls on the same data."""
    baseline = TFIDFDiscoveryBaseline(distance_threshold=0.7)
    statements = [
        {"id": "s1", "statement_text": "Dense retrieval suffers from hallucination in RAG."},
        {"id": "s2", "statement_text": "Activation outliers degrade 4-bit quantization."},
        {"id": "s3", "statement_text": "RAG models exhibit retrieval hallucination on multi-hop questions."},
        {"id": "s4", "statement_text": "Positive contribution without any negative claims."},
    ]
    res1 = baseline.fit_predict(statements)
    res2 = baseline.fit_predict(statements)

    assert res1.clusters == res2.clusters
    assert res1.cluster_labels == res2.cluster_labels
    assert [c["id"] for c in res1.candidate_statements] == [c["id"] for c in res2.candidate_statements]


def test_exactly_one_cluster_assignment_per_candidate():
    """Requirement 11: Every candidate receives exactly one cluster assignment."""
    baseline = TFIDFDiscoveryBaseline()
    statements = [
        {"id": "c1", "statement_text": "A major limitation is memory consumption."},
        {"id": "c2", "statement_text": "Another bottleneck is inference latency."},
        {"id": "c3", "statement_text": "The model fails to process audio inputs."},
    ]
    res = baseline.fit_predict(statements)

    candidate_ids = {c["id"] for c in res.candidate_statements}
    assert set(res.cluster_labels.keys()) == candidate_ids

    # Check each candidate appears in exactly one cluster set
    for cid in candidate_ids:
        in_clusters = [cluster for cluster in res.clusters if cid in cluster]
        assert len(in_clusters) == 1, f"Candidate {cid} appeared in {len(in_clusters)} clusters."


def test_non_candidate_statements_excluded_from_clusters():
    """Requirement 12: Non-candidate statements never appear in predicted clusters."""
    baseline = TFIDFDiscoveryBaseline()
    statements = [
        {"id": "cand_1", "statement_text": "The system suffers from catastrophic forgetting."},
        {"id": "non_cand_1", "statement_text": "We achieve state-of-the-art results on ImageNet."},
        {"id": "non_cand_2", "statement_text": "The code is implemented in PyTorch."},
    ]
    res = baseline.fit_predict(statements)

    all_clustered_ids = set().union(*res.clusters)
    assert "cand_1" in all_clustered_ids
    assert "non_cand_1" not in all_clustered_ids
    assert "non_cand_2" not in all_clustered_ids


def test_mapping_predicted_assignments_to_original_statement_ids():
    """Requirement 13: Accurate mapping from predicted assignments back to statement IDs."""
    baseline = TFIDFDiscoveryBaseline()

    # Test with Pydantic DTO instances
    dto_stmt = ResearchStatementCreateDTO(
        statement_text="Our model fails to generalize under severe out-of-distribution shift.",
        statement_type=StatementType.LIMITATION,
    )
    # Test with tuple representation
    tuple_stmt = ("custom_tuple_id", "Memory footprint remains challenging during decoding.")
    # Test with dictionary representation
    dict_stmt = {"id": "custom_dict_id", "statement_text": "Bottlenecks exist in context caching."}

    statements = [dto_stmt, tuple_stmt, dict_stmt]
    res = baseline.fit_predict(statements)

    assert len(res.candidate_statements) == 3
    assert "custom_tuple_id" in res.cluster_labels
    assert "custom_dict_id" in res.cluster_labels
    # DTO without explicit id defaults to stmt_0
    assert "stmt_0" in res.cluster_labels


def test_invalid_input_handling():
    """Requirement 14: Clear error validation on invalid input structures."""
    baseline = TFIDFDiscoveryBaseline()

    with pytest.raises(ValueError, match="must be a non-None sequence"):
        baseline.fit_predict(None)

    with pytest.raises(ValueError, match="Encountered None statement"):
        baseline.fit_predict([{"id": "s1", "statement_text": "limitation"}, None])

    with pytest.raises(ValueError, match="Unsupported statement type"):
        baseline.fit_predict([12345])


def test_empty_vocabulary_or_tokenization_edge_cases():
    """Requirement 15: Safe handling of candidate statements with stop words or symbols."""
    baseline = TFIDFDiscoveryBaseline(stop_words="english")
    # Both candidates contain limitation keywords but only english stop words otherwise
    statements = [
        {"id": "sw_1", "statement_text": "limitation of the and or in of"},
        {"id": "sw_2", "statement_text": "bottleneck at the with of"},
    ]
    # Should not crash with ValueError('empty vocabulary')
    res = baseline.fit_predict(statements)
    assert len(res.candidate_statements) == 2
    assert len(res.clusters) >= 1


def test_correct_registration_of_tfidf_baseline():
    """Requirement 16: Verification of baseline registry in get_baselines()."""
    baselines = get_baselines()
    assert len(baselines) == 4

    tfidf_baseline = next((b for b in baselines if b.name == "TF-IDF Discovery"), None)
    assert tfidf_baseline is not None
    assert tfidf_baseline.is_implemented is True
    assert tfidf_baseline.target_component == "OpportunityDiscoveryService"
    assert "pairwise_precision" in tfidf_baseline.metrics
    assert "pairwise_recall" in tfidf_baseline.metrics

    # Other baselines remain unimplemented
    for b in baselines:
        if b.name != "TF-IDF Discovery":
            assert b.is_implemented is False, f"Baseline {b.name} should still be unimplemented."


def test_compatibility_with_pairwise_cluster_metrics():
    """Requirement 17: Baseline output is directly compatible with pairwise_cluster_metrics."""
    baseline = TFIDFDiscoveryBaseline(distance_threshold=0.85)
    statements = [
        {"id": "rag_1", "statement_text": "Dense retrieval suffers from hallucination in RAG pipelines."},
        {"id": "rag_2", "statement_text": "RAG models fail to prevent retrieval hallucination on long queries."},
        {"id": "quant_1", "statement_text": "Quantization suffers from catastrophic activation outlier clipping."},
        {"id": "quant_2", "statement_text": "Activation outliers remain an unresolved bottleneck in 4-bit quantization."},
    ]
    res = baseline.fit_predict(statements)

    # Define reference ground truth clusters for the candidate statements
    reference_clusters = [
        {"rag_1", "rag_2"},
        {"quant_1", "quant_2"},
    ]

    # Evaluate directly with the TFIDFBaselineResult instance
    metrics = pairwise_cluster_metrics(res, reference_clusters)
    assert metrics["pairwise_precision"] == 1.0
    assert metrics["pairwise_recall"] == 1.0
    assert metrics["pairwise_f1"] == 1.0
    assert metrics["rand_index"] == 1.0

    # Also evaluate using res.clusters explicitly
    metrics_explicit = pairwise_cluster_metrics(res.clusters, reference_clusters)
    assert metrics_explicit["pairwise_precision"] == 1.0
    assert metrics_explicit["rand_index"] == 1.0
