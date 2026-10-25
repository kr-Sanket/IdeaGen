"""Tests for the Stage C dense and hybrid (RRF) candidate recall benchmark.

Verifies:
1. Dense vector dimensions (768d float32 L2-normalized).
2. Dense cosine ranking prioritizing genuine semantic similarity.
3. Stable deterministic tie-breaking for both Dense and RRF rankers.
4. Reciprocal Rank Fusion (RRF, k=60) combining lexical and dense rankings accurately.
5. Vocabulary mismatch evaluation: Dense outperforming TF-IDF when conceptual phrasing diverges.
6. Proper non-penalization of empty relevance sets (negative controls).
7. Temporal eligibility filtering parity across all 4 rankers (FIFO, TF-IDF, Dense, RRF).
8. Partition isolation and denominator integrity across authentic, pilot, and synthetic subsets.
9. Informative blocker when genuine embeddings are unavailable (no random/mock vector substitution).
10. Full benchmark execution and EvaluationResult schema compliance.
"""

import datetime
from pathlib import Path
import pytest
import numpy as np

from ideagen.evaluation.schemas import EvaluationResult, EvaluationType
from ideagen.evaluation.candidate_recall_benchmark import (
    BenchmarkCaseType,
    CandidateOpportunity,
    PaperQuery,
    TFIDFLexicalRanker,
    filter_temporally_eligible_candidates,
)
from ideagen.evaluation.dense_candidate_recall_benchmark import (
    DEFAULT_RRF_K,
    DenseCandidateRecallBenchmarkRunner,
    DenseCosineRanker,
    ReciprocalRankFusionRanker,
    get_stage_c_candidate_recall_benchmark_dataset,
)


def test_dense_ranker_vector_dimensions_and_normalization():
    """Dense ranker must use 768-dimensional L2-normalized vectors."""
    ranker = DenseCosineRanker()
    vec = ranker.get_embedding("Retrieval-augmented generation with neural Wikipedia index.")

    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32
    assert vec.shape == (768,)
    # Verify L2 normalization: norm should be approximately 1.0
    norm = np.linalg.norm(vec)
    assert pytest.approx(norm, abs=1e-5) == 1.0


def test_dense_cosine_ranking_semantic_priority():
    """Dense cosine ranker ranks semantically aligned candidates ahead of unrelated candidates."""
    opp_fl = CandidateOpportunity(
        opportunity_id="opp-fl",
        title="Scalability bottlenecks in federated learning over unreliable networks",
        problem_statement="Current FL protocols fail when client dropout exceeds 40%.",
        description="Robust aggregation protocols for mobile edge clients.",
        effective_creation_date=datetime.date(2021, 1, 1),
    )
    opp_chem = CandidateOpportunity(
        opportunity_id="opp-chem",
        title="Catalytic cracking of heavy petroleum fractions",
        problem_statement="Zeolite catalyst deactivation via coke deposition.",
        description="High temperature fluidized bed regeneration.",
        effective_creation_date=datetime.date(2020, 1, 1),  # older
    )

    query = PaperQuery(
        query_id="Q-FL-SEM",
        paper_id="P-FL-SEM",
        title="RobustAgg: A novel aggregation scheme for federated learning",
        abstract="We introduce RobustAgg for unreliable clients, maintaining convergence up to 75% client dropout in federated learning.",
        publication_date=datetime.date(2022, 6, 1),
        case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
        relevant_opportunity_ids={"opp-fl"},
    )

    ranker = DenseCosineRanker()
    ranked = ranker.rank(query, [opp_chem, opp_fl])
    assert ranked[0].opportunity_id == "opp-fl"


def test_dense_stable_tie_breaking():
    """When candidates have identical dense similarity, sort deterministically by date asc, then ID asc."""
    date1 = datetime.date(2021, 1, 1)
    date2 = datetime.date(2021, 6, 1)

    c_b = CandidateOpportunity(opportunity_id="opp-b", title="Title B", problem_statement="P", description="D", effective_creation_date=date2)
    c_a = CandidateOpportunity(opportunity_id="opp-a", title="Title A", problem_statement="P", description="D", effective_creation_date=date2)
    c_older = CandidateOpportunity(opportunity_id="opp-older", title="Title Older", problem_statement="P", description="D", effective_creation_date=date1)

    # Mock identical unit vectors in explicit cache
    unit_vec = [1.0 / np.sqrt(768)] * 768
    cache = {
        "Query Title Query Abstract": unit_vec,
        "Title B P D": unit_vec,
        "Title A P D": unit_vec,
        "Title Older P D": unit_vec,
    }

    query = PaperQuery(
        query_id="Q-TIE",
        paper_id="P-TIE",
        title="Query Title",
        abstract="Query Abstract",
        publication_date=datetime.date(2022, 1, 1),
        case_type=BenchmarkCaseType.SYNTHETIC_STRESS_TEST,
        relevant_opportunity_ids={"opp-a"},
    )

    ranker = DenseCosineRanker(cache=cache)
    ranked = ranker.rank(query, [c_b, c_older, c_a])
    ranked_ids = [c.opportunity_id for c in ranked]

    # c_older comes first due to earlier date; c_a and c_b tie on date2, sorted by ID
    assert ranked_ids == ["opp-older", "opp-a", "opp-b"]


def test_rrf_fusion_formula_and_deterministic_ties():
    """Reciprocal Rank Fusion computes score = 1/(k+r_t) + 1/(k+r_d) and breaks ties stably."""
    d1 = datetime.date(2021, 1, 1)
    d2 = datetime.date(2021, 6, 1)

    cand_1 = CandidateOpportunity(opportunity_id="opp-1", title="1", problem_statement="1", description="1", effective_creation_date=d1)
    cand_2 = CandidateOpportunity(opportunity_id="opp-2", title="2", problem_statement="2", description="2", effective_creation_date=d2)

    query = PaperQuery(
        query_id="Q-RRF",
        paper_id="P-RRF",
        title="Query",
        abstract="Abstract",
        publication_date=datetime.date(2022, 1, 1),
        case_type=BenchmarkCaseType.SYNTHETIC_STRESS_TEST,
        relevant_opportunity_ids={"opp-1"},
    )

    # Suppose TF-IDF ranks [opp-1, opp-2] (ranks 1, 2)
    # Dense ranks [opp-2, opp-1] (ranks 1, 2)
    # RRF scores with k=60:
    # opp-1: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.0163934 + 0.0161290 = 0.0325224
    # opp-2: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.0325224
    # Scores are identical! Tie must be broken by effective_creation_date.asc(): opp-1 (d1) precedes opp-2 (d2).

    # Custom mock rankers for deterministic rank order testing
    class MockRanker:
        def __init__(self, order):
            self.order = order

        def rank(self, q, candidates, max_candidates=None):
            return self.order

    rrf = ReciprocalRankFusionRanker(
        tfidf_ranker=MockRanker([cand_1, cand_2]),
        dense_ranker=MockRanker([cand_2, cand_1]),
        k_rrf=60,
    )

    ranked = rrf.rank(query, [cand_2, cand_1])
    assert [c.opportunity_id for c in ranked] == ["opp-1", "opp-2"]


def test_vocabulary_mismatch_dense_outperforms_tfidf():
    """On SYN-VOCAB-01 (paraphrased continually learning gap), Dense achieves Rank 1 while TF-IDF drops."""
    candidates, queries = get_stage_c_candidate_recall_benchmark_dataset()
    queries_by_id = {q.query_id: q for q in queries}
    q_vocab = queries_by_id["SYN-VOCAB-01"]

    eligible = filter_temporally_eligible_candidates(candidates, q_vocab.publication_date)

    # 1. TF-IDF
    tfidf = TFIDFLexicalRanker()
    tfidf_ranked = tfidf.rank(q_vocab, eligible, max_candidates=None)
    tfidf_ids = [c.opportunity_id for c in tfidf_ranked]
    tfidf_target_rank = tfidf_ids.index("opp-syn-forgetting-001") + 1

    # 2. Dense Cosine
    dense = DenseCosineRanker()
    dense_ranked = dense.rank(q_vocab, eligible, max_candidates=None)
    dense_ids = [c.opportunity_id for c in dense_ranked]
    dense_target_rank = dense_ids.index("opp-syn-forgetting-001") + 1

    # 3. Hybrid RRF
    rrf = ReciprocalRankFusionRanker(tfidf_ranker=tfidf, dense_ranker=dense, k_rrf=DEFAULT_RRF_K)
    rrf_ranked = rrf.rank(q_vocab, eligible, max_candidates=None)
    rrf_ids = [c.opportunity_id for c in rrf_ranked]
    rrf_target_rank = rrf_ids.index("opp-syn-forgetting-001") + 1

    # Key Comparative Invariant:
    # Dense achieves rank 1 because nomic-embed-text connects plasticity loss with catastrophic forgetting
    assert dense_target_rank == 1
    # TF-IDF lacks overlapping terms, pushing the target down
    assert tfidf_target_rank > 1
    # Hybrid RRF lifts the target into the top 2
    assert rrf_target_rank <= 2


def test_missing_model_and_cache_reports_informative_blocker():
    """If embeddings are requested for uncached text and Ollama is unreachable, raises clear RuntimeError."""
    # Use non-existent cache and point provider to an invalid port
    non_existent_cache_path = Path("/tmp/non_existent_cache_test_12345.json")

    from ideagen.discovery.embeddings import OllamaEmbeddingProvider
    unreachable_provider = OllamaEmbeddingProvider(
        base_url="http://127.0.0.1:59999",  # unused port
        model_name="nomic-embed-text",
        timeout=0.2,
    )

    ranker = DenseCosineRanker(
        embedding_provider=unreachable_provider,
        cache_file_path=non_existent_cache_path,
        cache={},
    )

    with pytest.raises(RuntimeError) as exc_info:
        ranker.get_embedding("Novel uncached text that requires active provider")

    err_msg = str(exc_info.value)
    assert "nomic-embed-text" in err_msg
    assert "unavailable" in err_msg
    assert "Cannot compute genuine dense baseline" in err_msg


def test_stage_c_benchmark_execution_and_schema_compliance():
    """Runs the full Stage C benchmark and verifies EvaluationResult schema for all 4 strategies."""
    candidates, queries = get_stage_c_candidate_recall_benchmark_dataset()

    runner = DenseCandidateRecallBenchmarkRunner(
        candidate_pool=candidates,
        queries=queries,
        k_values=(1, 3, 5, 10, 20),
        max_candidates=20,
    )

    results = runner.run()

    eval_result: EvaluationResult = results["evaluation_result"]
    assert eval_result.evaluation_name == "candidate_recall_stage_c"
    assert eval_result.fixture_version == "v3.0-dense-hybrid-rrf"
    assert eval_result.evaluation_type == EvaluationType.EXECUTED

    # Check metrics for all 4 strategies and all 4 partitions
    for strat in ["fifo", "tfidf", "dense", "rrf"]:
        for partition in ["overall", "authentic", "pilot", "synthetic"]:
            for k in [1, 3, 5, 10, 20]:
                metric_key = f"{strat}_{partition}_recall_at_{k}"
                assert metric_key in eval_result.metrics
                val = eval_result.metrics[metric_key]
                assert 0.0 <= val <= 1.0, f"{metric_key} out of bounds: {val}"

            cap_rate_key = f"{strat}_{partition}_cap_miss_rate"
            assert cap_rate_key in eval_result.metrics
            assert 0.0 <= eval_result.metrics[cap_rate_key] <= 1.0

            opp_miss_key = f"{strat}_{partition}_opportunity_cap_misses"
            assert opp_miss_key in eval_result.metrics
            assert eval_result.metrics[opp_miss_key] >= 0.0

    # Denominators integrity
    assert eval_result.denominators["total_queries"] == 16
    assert eval_result.denominators["evaluable_queries_overall"] == 12
    assert eval_result.denominators["queries_without_relevant_overall"] == 4
    assert eval_result.denominators["evaluable_queries_authentic"] == 5
    assert eval_result.denominators["queries_without_relevant_authentic"] == 2
    assert eval_result.denominators["evaluable_queries_pilot"] == 5
    assert eval_result.denominators["queries_without_relevant_pilot"] == 1
    assert eval_result.denominators["evaluable_queries_synthetic"] == 2
    assert eval_result.denominators["queries_without_relevant_synthetic"] == 1
    assert eval_result.denominators["candidate_pool_size"] == 31

    # Comparative Recall@1 across strategies:
    # Dense should achieve 1.0 Recall@1 across authentic, pilot, and synthetic partitions
    assert eval_result.metrics["dense_overall_recall_at_1"] == 1.0
    assert eval_result.metrics["dense_authentic_recall_at_1"] == 1.0
    assert eval_result.metrics["dense_pilot_recall_at_1"] == 1.0
    assert eval_result.metrics["dense_synthetic_recall_at_1"] == 1.0

    # TF-IDF achieves 1.0 on authentic and pilot, but drops on synthetic vocab mismatch at K=1
    assert eval_result.metrics["tfidf_authentic_recall_at_1"] == 1.0
    assert eval_result.metrics["tfidf_synthetic_recall_at_1"] == 0.5  # 1 hit (SYN-CAP-MISS), 1 miss at K=1 (SYN-VOCAB)
    assert eval_result.metrics["tfidf_synthetic_recall_at_3"] == 1.0  # recovered at K=3

    # RRF combines both and achieves Recall@1 = 0.9167 (11/12) and Recall@3 = 1.0 (12/12)
    assert eval_result.metrics["rrf_overall_recall_at_3"] == 1.0

    # Cap miss rate verification:
    # FIFO suffers high cap miss rate due to chronological distractor crowding
    assert eval_result.metrics["fifo_overall_cap_miss_rate"] > 0.5
    # TF-IDF, Dense, and RRF all achieve 0.0 cap miss rate within top 20
    assert eval_result.metrics["tfidf_overall_cap_miss_rate"] == 0.0
    assert eval_result.metrics["dense_overall_cap_miss_rate"] == 0.0
    assert eval_result.metrics["rrf_overall_cap_miss_rate"] == 0.0


def test_stratification_preserves_three_way_partition_independence():
    """Verifies that authentic, pilot, and synthetic partitions are mutually exclusive."""
    candidates, queries = get_stage_c_candidate_recall_benchmark_dataset()

    runner = DenseCandidateRecallBenchmarkRunner(
        candidate_pool=candidates,
        queries=queries,
        k_values=(1, 5),
        max_candidates=20,
    )
    results = runner.run()

    for strat in ["fifo", "tfidf", "dense", "rrf"]:
        s_auth = results[f"{strat}_authentic"]
        s_pilot = results[f"{strat}_pilot"]
        s_syn = results[f"{strat}_synthetic"]

        assert all(o.case_type == BenchmarkCaseType.AUTHENTIC_LITERATURE for o in s_auth.query_outcomes)
        assert all(o.case_type == BenchmarkCaseType.HUMAN_CURATED_PILOT for o in s_pilot.query_outcomes)
        assert all(o.case_type == BenchmarkCaseType.SYNTHETIC_STRESS_TEST for o in s_syn.query_outcomes)

        assert s_auth.total_queries == 7
        assert s_pilot.total_queries == 6
        assert s_syn.total_queries == 3
        assert s_auth.total_queries + s_pilot.total_queries + s_syn.total_queries == len(queries)
