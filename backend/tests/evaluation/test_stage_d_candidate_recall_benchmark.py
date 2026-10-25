"""Tests for Stage D independent DEV / HELD_OUT candidate recall evaluation.

Verifies:
1. Strict disjointness between DEV and HELD_OUT splits (by query_id and paper_id).
2. Traceable label provenance across authentic, pilot, and synthetic partitions.
3. Simulated-date metadata enforcement (pilot and synthetic dates explicitly flagged).
4. Partition isolation, metric denominators, and zero-relevance control exclusion.
5. Deterministic offline cache behavior and clear failure without mock vectors.
6. EvaluationResult compliance, four-way ranking comparison, and held-out metrics.
"""

import datetime
from pathlib import Path
import pytest

from ideagen.evaluation.candidate_recall_benchmark import (
    BenchmarkCaseType,
    BenchmarkSplit,
    CandidateOpportunity,
    ChronologicalFIFORanker,
    PaperQuery,
    TFIDFLexicalRanker,
    filter_temporally_eligible_candidates,
    get_stage_d_candidate_recall_benchmark_dataset,
)
from ideagen.evaluation.dense_candidate_recall_benchmark import (
    DenseCandidateRecallBenchmarkRunner,
    DenseCosineRanker,
    ReciprocalRankFusionRanker,
    StageDCandidateRecallBenchmarkRunner,
)
from ideagen.evaluation.schemas import EvaluationResult, EvaluationType


def test_stage_d_split_disjointness():
    """DEV and HELD_OUT splits must have strictly disjoint query IDs and paper IDs."""
    cands_dev, queries_dev = get_stage_d_candidate_recall_benchmark_dataset(BenchmarkSplit.DEV)
    cands_held, queries_held = get_stage_d_candidate_recall_benchmark_dataset(BenchmarkSplit.HELD_OUT)

    dev_q_ids = {q.query_id for q in queries_dev}
    held_q_ids = {q.query_id for q in queries_held}
    assert dev_q_ids.isdisjoint(held_q_ids), f"Overlapping query IDs: {dev_q_ids & held_q_ids}"

    dev_paper_ids = {q.paper_id for q in queries_dev}
    held_paper_ids = {q.paper_id for q in queries_held}
    assert dev_paper_ids.isdisjoint(held_paper_ids), f"Overlapping paper IDs: {dev_paper_ids & held_paper_ids}"

    # Verify candidate pool is identical across both splits
    dev_cand_ids = [c.opportunity_id for c in cands_dev]
    held_cand_ids = [c.opportunity_id for c in cands_held]
    assert dev_cand_ids == held_cand_ids
    assert len(cands_dev) == 31


def test_stage_d_label_provenance():
    """Every scored relationship must trace to a verified source fixture or published paper."""
    for split in [BenchmarkSplit.DEV, BenchmarkSplit.HELD_OUT]:
        _, queries = get_stage_d_candidate_recall_benchmark_dataset(split)
        valid_relationships = {
            "ADDRESSES", "SUPPORTS", "CONTRADICTS", "EXTENDS", "HARD_NEGATIVE", "UNRELATED", "UNRELATED_CONTROL"
        }

        for q in queries:
            assert len(q.provenance_records) > 0, f"Query {q.query_id} has no provenance records."
            for prov in q.provenance_records:
                assert prov.expected_relationship in valid_relationships
                assert len(prov.rationale.strip()) > 10, f"Rationale too short for {q.query_id}"
                assert (
                    "experiments/data/" in prov.source_reference
                    or "classification_fixtures.json" in prov.source_reference
                    or "stage4b_benchmark.py" in prov.source_reference
                    or "stage4_benchmark.py" in prov.source_reference
                    or "candidate_recall_benchmark.py" in prov.source_reference
                    or "NeurIPS" in prov.source_reference
                    or "CVPR" in prov.source_reference
                ), f"Invalid source reference for {q.query_id}: {prov.source_reference}"


def test_stage_d_simulated_date_metadata():
    """Pilot and synthetic timestamps must be explicitly flagged as simulated benchmark dates."""
    for split in [BenchmarkSplit.DEV, BenchmarkSplit.HELD_OUT]:
        _, queries = get_stage_d_candidate_recall_benchmark_dataset(split)
        for q in queries:
            if q.case_type == BenchmarkCaseType.AUTHENTIC_LITERATURE:
                assert not q.is_simulated_date, f"Authentic query {q.query_id} should not be simulated."
            elif q.case_type == BenchmarkCaseType.HUMAN_CURATED_PILOT:
                assert q.is_simulated_date, f"Pilot query {q.query_id} must be flagged as simulated date."
                assert "Simulated benchmark date" in (q.notes or "")
            elif q.case_type == BenchmarkCaseType.SYNTHETIC_STRESS_TEST:
                assert q.is_simulated_date, f"Synthetic query {q.query_id} must be flagged as simulated date."


def test_stage_d_partition_isolation_and_denominators():
    """Denominators must accurately count evaluable queries vs. zero-relevance controls."""
    runner = StageDCandidateRecallBenchmarkRunner()
    results = runner.run()
    denoms = results["denominators"]

    # DEV: 13 queries = 11 evaluable (4 authentic, 5 pilot, 2 synthetic) + 2 controls (1 auth, 1 pilot)
    assert denoms["total_queries_dev"] == 13
    assert denoms["evaluable_queries_dev_overall"] == 11
    assert denoms["queries_without_relevant_dev_overall"] == 2
    assert denoms["evaluable_queries_dev_authentic"] == 4
    assert denoms["queries_without_relevant_dev_authentic"] == 1
    assert denoms["evaluable_queries_dev_pilot"] == 5
    assert denoms["queries_without_relevant_dev_pilot"] == 1
    assert denoms["evaluable_queries_dev_synthetic"] == 2
    assert denoms["queries_without_relevant_dev_synthetic"] == 0

    # HELD_OUT: 14 queries = 12 evaluable (1 authentic, 11 pilot, 0 synthetic) + 2 controls (1 auth, 1 synthetic)
    assert denoms["total_queries_held_out"] == 14
    assert denoms["evaluable_queries_held_out_overall"] == 12
    assert denoms["queries_without_relevant_held_out_overall"] == 2
    assert denoms["evaluable_queries_held_out_authentic"] == 1
    assert denoms["queries_without_relevant_held_out_authentic"] == 1
    assert denoms["evaluable_queries_held_out_pilot"] == 11
    assert denoms["queries_without_relevant_held_out_pilot"] == 0
    assert denoms["evaluable_queries_held_out_synthetic"] == 0
    assert denoms["queries_without_relevant_held_out_synthetic"] == 1

    assert denoms["candidate_pool_size"] == 31


def test_stage_d_offline_cache_contract():
    """All candidate texts and query texts for both splits must resolve from offline cache."""
    ranker = DenseCosineRanker()
    # Offline cache must have all 62 keys (31 candidates + 13 DEV queries + 14 HELD_OUT queries - shared controls)
    assert len(ranker._cache) >= 58

    for split in [BenchmarkSplit.DEV, BenchmarkSplit.HELD_OUT]:
        cands, queries = get_stage_d_candidate_recall_benchmark_dataset(split)
        for c in cands:
            t = f"{c.title} {c.problem_statement} {c.description}".strip()
            assert t in ranker._cache, f"Candidate {c.opportunity_id} missing from offline cache."
        for q in queries:
            t = f"{q.title} {q.abstract}".strip()
            assert t in ranker._cache, f"Query {q.query_id} missing from offline cache."


def test_stage_d_held_out_ranking_metrics_and_evaluation_result():
    """Evaluates the Stage D runner and checks comparative ranker behavior on HELD_OUT."""
    runner = StageDCandidateRecallBenchmarkRunner()
    results = runner.run()

    eval_res: EvaluationResult = results["evaluation_result"]
    assert eval_res.evaluation_name == "candidate_recall_stage_d"
    assert eval_res.evaluation_type == EvaluationType.EXECUTED
    assert not eval_res.is_engineering_validation

    metrics = eval_res.metrics

    # Check FIFO fails on held-out queries due to chronology
    assert metrics["fifo_held_out_overall_recall_at_1"] == 0.0

    # Check TF-IDF suffers on pilot terminology shifts (HELD-PILOT-06, HELD-PILOT-08)
    assert metrics["tfidf_held_out_pilot_recall_at_1"] < 1.0
    assert pytest.approx(metrics["tfidf_held_out_pilot_recall_at_1"], abs=1e-4) == 0.8182

    # Check Dense semantic retrieval achieves 1.0 Recall@1 across all pilot held-out cases
    assert metrics["dense_held_out_pilot_recall_at_1"] == 1.0

    # Check Hybrid RRF achieves 1.0 Recall@1 across overall and pilot held-out cases
    assert metrics["rrf_held_out_overall_recall_at_1"] == 1.0
    assert metrics["rrf_held_out_pilot_recall_at_1"] == 1.0
    assert metrics["rrf_held_out_authentic_recall_at_1"] == 1.0
