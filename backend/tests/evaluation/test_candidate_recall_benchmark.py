"""Tests for the Stage B in-memory candidate recall benchmark.

Verifies:
1. Temporal eligibility filtering strictly enforces temporal causality and cutoff constraints.
2. Stable tie-breaking for both FIFO and TF-IDF rankers.
3. Accurate detection and quantification of cap-related FIFO misses when distractors precede relevant candidates.
4. TF-IDF lexical ranking prioritizing keyword and terminology matches over distractor candidates.
5. Correct handling of empty relevance sets (not penalized as zero-recall queries).
6. Schema compliance and metric aggregation for EvaluationResult in Stage B.
7. Strict stratification preserving independence between authentic literature, human-curated pilot, and synthetic stress tests.
8. Provenance tracking for all scored and hard-negative relationships.
9. Defensible discrimination of same-subfield competing opportunities and hard negatives.
"""

import datetime
import pytest

from ideagen.evaluation.schemas import EvaluationResult, EvaluationType
from ideagen.evaluation.candidate_recall_benchmark import (
    BenchmarkCaseType,
    CandidateOpportunity,
    CandidateRecallBenchmarkRunner,
    ChronologicalFIFORanker,
    OpportunityRelationshipProvenance,
    PaperQuery,
    TFIDFLexicalRanker,
    filter_temporally_eligible_candidates,
    get_candidate_recall_benchmark_dataset,
    get_stage_b_candidate_recall_benchmark_dataset,
    summarize_strategy_results,
    evaluate_query_ranking,
)


def test_temporal_filtering_enforces_causality():
    """Opportunities created after the paper publication date must be excluded."""
    pub_date = datetime.date(2022, 6, 1)

    opp_past = CandidateOpportunity(
        opportunity_id="opp-past",
        title="Prior Gap",
        problem_statement="Problem past",
        description="Desc past",
        effective_creation_date=datetime.date(2021, 1, 1),
    )
    opp_same_day = CandidateOpportunity(
        opportunity_id="opp-same-day",
        title="Same Day Gap",
        problem_statement="Problem same",
        description="Desc same",
        effective_creation_date=pub_date,
    )
    opp_future = CandidateOpportunity(
        opportunity_id="opp-future",
        title="Future Gap",
        problem_statement="Problem future",
        description="Desc future",
        effective_creation_date=datetime.date(2022, 6, 2),
    )

    candidates = [opp_past, opp_same_day, opp_future]

    # Without external cutoff
    eligible = filter_temporally_eligible_candidates(candidates, pub_date)
    eligible_ids = {c.opportunity_id for c in eligible}
    assert eligible_ids == {"opp-past", "opp-same-day"}
    assert "opp-future" not in eligible_ids

    # With external cutoff date earlier than publication date
    cutoff = datetime.date(2021, 6, 1)
    eligible_with_cutoff = filter_temporally_eligible_candidates(candidates, pub_date, cutoff_date=cutoff)
    assert {c.opportunity_id for c in eligible_with_cutoff} == {"opp-past"}

    # Undated paper yields empty list
    assert filter_temporally_eligible_candidates(candidates, None) == []


def test_fifo_stable_tie_breaking():
    """When opportunities share the exact same creation date, FIFO must sort deterministically by string ID."""
    same_date = datetime.date(2021, 5, 1)
    candidates = [
        CandidateOpportunity(opportunity_id="opp-z", title="Z", problem_statement="Z", description="Z", effective_creation_date=same_date),
        CandidateOpportunity(opportunity_id="opp-a", title="A", problem_statement="A", description="A", effective_creation_date=same_date),
        CandidateOpportunity(opportunity_id="opp-m", title="M", problem_statement="M", description="M", effective_creation_date=same_date),
    ]

    query = PaperQuery(
        query_id="Q-TIE",
        paper_id="P-TIE",
        title="Query Paper",
        abstract="Query Abstract",
        publication_date=datetime.date(2022, 1, 1),
        case_type=BenchmarkCaseType.SYNTHETIC_STRESS_TEST,
        relevant_opportunity_ids={"opp-a"},
    )

    ranker = ChronologicalFIFORanker()
    ranked = ranker.rank(query, candidates)
    ranked_ids = [c.opportunity_id for c in ranked]
    assert ranked_ids == ["opp-a", "opp-m", "opp-z"]


def test_tfidf_stable_tie_breaking():
    """When candidates have zero lexical match, TF-IDF ranker breaks ties by date asc, then ID asc."""
    query = PaperQuery(
        query_id="Q-TFIDF-TIE",
        paper_id="P-TFIDF-TIE",
        title="Quantum chromodynamics lattice gauge simulations",
        abstract="SU(3) gauge theory gluon field calculations on high performance clusters.",
        publication_date=datetime.date(2022, 1, 1),
        case_type=BenchmarkCaseType.SYNTHETIC_STRESS_TEST,
        relevant_opportunity_ids=set(),
    )

    # 3 completely off-topic candidates with no vocabulary overlap
    d1 = datetime.date(2020, 1, 1)
    d2 = datetime.date(2020, 6, 1)
    candidates = [
        CandidateOpportunity(opportunity_id="opp-b", title="Ancient history Rome", problem_statement="Aqueduct engineering", description="Stone arches", effective_creation_date=d2),
        CandidateOpportunity(opportunity_id="opp-a", title="Medieval pottery glazes", problem_statement="Kiln temperatures", description="Clay mineralogy", effective_creation_date=d2),
        CandidateOpportunity(opportunity_id="opp-prior", title="Bronze age metallurgy", problem_statement="Tin bronze smelting", description="Slag analysis", effective_creation_date=d1),
    ]

    ranker = TFIDFLexicalRanker()
    ranked = ranker.rank(query, candidates)
    ranked_ids = [c.opportunity_id for c in ranked]

    # opp-prior (d1) comes first due to earlier date; opp-a and opp-b tie at d2, sorted by ID
    assert ranked_ids == ["opp-prior", "opp-a", "opp-b"]


def test_cap_related_fifo_miss_detected():
    """When relevant candidates are created after max_candidates distractors, FIFO misses them while TF-IDF captures them."""
    # 25 older distractors created in 2021
    distractors = [
        CandidateOpportunity(
            opportunity_id=f"distractor-{i:02d}",
            title=f"Distractor topic {i}",
            problem_statement=f"Unrelated problem in domain {i}",
            description=f"Description {i}",
            effective_creation_date=datetime.date(2021, 1, 1) + datetime.timedelta(days=i),
        )
        for i in range(25)
    ]

    # Target relevant opportunity created later in 2022
    target_opp = CandidateOpportunity(
        opportunity_id="opp-target-rag",
        title="Mitigating parametric knowledge limitations in retrieval-augmented models",
        problem_statement="Pre-trained language models suffer from limited parametric memory for knowledge-intensive QA.",
        description="Investigate hybrid non-parametric vector retrieval modules.",
        effective_creation_date=datetime.date(2022, 1, 1),
    )

    all_candidates = distractors + [target_opp]

    query = PaperQuery(
        query_id="Q-CAP-TEST",
        paper_id="P-CAP-TEST",
        title="Retrieval-Augmented Generation with Dense Wikipedia Indexes",
        abstract="We introduce RAG models combining parametric seq2seq models with non-parametric neural retrieval.",
        publication_date=datetime.date(2023, 1, 1),
        case_type=BenchmarkCaseType.SYNTHETIC_STRESS_TEST,
        relevant_opportunity_ids={"opp-target-rag"},
    )

    eligible = filter_temporally_eligible_candidates(all_candidates, query.publication_date)
    assert len(eligible) == 26

    # 1. FIFO with cap = 20
    fifo_ranker = ChronologicalFIFORanker()
    fifo_ranked = fifo_ranker.rank(query, eligible, max_candidates=20)
    assert len(fifo_ranked) == 20
    assert "opp-target-rag" not in [c.opportunity_id for c in fifo_ranked]

    fifo_outcome = evaluate_query_ranking(query, fifo_ranked, eligible, k_values=(1, 5, 20), max_candidates=20)
    assert fifo_outcome.recall_at_k[20] == 0.0
    assert fifo_outcome.has_cap_miss is True
    assert fifo_outcome.cap_misses_count == 1

    # 2. TF-IDF with cap = 20
    tfidf_ranker = TFIDFLexicalRanker()
    tfidf_ranked = tfidf_ranker.rank(query, eligible, max_candidates=20)
    assert len(tfidf_ranked) == 20
    assert tfidf_ranked[0].opportunity_id == "opp-target-rag"

    tfidf_outcome = evaluate_query_ranking(query, tfidf_ranked, eligible, k_values=(1, 5, 20), max_candidates=20)
    assert tfidf_outcome.recall_at_k[1] == 1.0
    assert tfidf_outcome.recall_at_k[20] == 1.0
    assert tfidf_outcome.has_cap_miss is False
    assert tfidf_outcome.cap_misses_count == 0


def test_tfidf_ranking_prioritizes_lexical_matches():
    """TF-IDF lexical ranker prioritizes candidate matching domain terms over older non-matching candidates."""
    opp_fl = CandidateOpportunity(
        opportunity_id="opp-fl",
        title="Scalability bottlenecks in federated learning over unreliable networks",
        problem_statement="Current FL protocols fail when client dropout exceeds 40%.",
        description="Robust aggregation protocols for mobile edge clients.",
        effective_creation_date=datetime.date(2022, 1, 1),
    )
    opp_chem = CandidateOpportunity(
        opportunity_id="opp-chem",
        title="Catalytic cracking of heavy petroleum fractions",
        problem_statement="Zeolite catalyst deactivation via coke deposition.",
        description="High temperature fluidized bed regeneration.",
        effective_creation_date=datetime.date(2020, 1, 1),  # older, would win in FIFO
    )

    query = PaperQuery(
        query_id="Q-FL",
        paper_id="P-FL",
        title="RobustAgg: A novel aggregation scheme for federated learning",
        abstract="We introduce RobustAgg for unreliable clients, maintaining convergence up to 75% client dropout in federated learning.",
        publication_date=datetime.date(2022, 6, 1),
        case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
        relevant_opportunity_ids={"opp-fl"},
    )

    ranker = TFIDFLexicalRanker()
    ranked = ranker.rank(query, [opp_chem, opp_fl])
    assert ranked[0].opportunity_id == "opp-fl"


def test_empty_relevance_set_handling():
    """Queries without verified relevant opportunities must not be counted as zero-recall failures."""
    query_control = PaperQuery(
        query_id="Q-CTRL",
        paper_id="P-CTRL",
        title="Control Paper on Ancient Irrigation Systems",
        abstract="Archaeological survey of subterranean canal networks in arid plateaus.",
        publication_date=datetime.date(2022, 1, 1),
        case_type=BenchmarkCaseType.AUTHENTIC_LITERATURE,
        relevant_opportunity_ids=set(),  # No relevant opportunities
    )

    candidate = CandidateOpportunity(
        opportunity_id="opp-01",
        title="Unrelated Gap",
        problem_statement="Unrelated Problem",
        description="Unrelated Desc",
        effective_creation_date=datetime.date(2021, 1, 1),
    )

    outcome = evaluate_query_ranking(query_control, [candidate], [candidate], k_values=(1, 5))
    assert outcome.relevant_count == 0
    assert outcome.has_cap_miss is False

    summary = summarize_strategy_results("FIFO", [outcome], k_values=(1, 5))
    assert summary.total_queries == 1
    assert summary.evaluable_queries == 0
    assert summary.queries_without_relevant == 1
    # Mean recall should be 0.0 with 0 evaluable queries, avoiding division by zero
    assert summary.mean_recall_at_k[1] == 0.0
    assert summary.mean_recall_at_k[5] == 0.0


def test_standard_benchmark_execution_and_schema_compliance():
    """Runs the Stage B candidate recall benchmark on the standard dataset and verifies EvaluationResult schema."""
    candidate_pool, queries = get_candidate_recall_benchmark_dataset()

    runner = CandidateRecallBenchmarkRunner(
        candidate_pool=candidate_pool,
        queries=queries,
        k_values=(1, 3, 5, 10, 20),
        max_candidates=20,
    )

    results = runner.run()

    eval_result: EvaluationResult = results["evaluation_result"]
    assert eval_result.evaluation_name == "candidate_recall_stage_b"
    assert eval_result.evaluation_type == EvaluationType.EXECUTED
    assert eval_result.fixture_version == "v2.0-source-grounded-hard-negatives"

    # Check that all Recall@K metrics are present and bounded [0.0, 1.0] for all partitions
    for strat in ["fifo", "tfidf"]:
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
    assert eval_result.denominators["total_queries"] == len(queries)
    assert (
        eval_result.denominators["evaluable_queries_overall"]
        + eval_result.denominators["queries_without_relevant_overall"]
        == len(queries)
    )
    assert eval_result.denominators["total_queries"] == 15
    assert eval_result.denominators["evaluable_queries_overall"] == 11
    assert eval_result.denominators["queries_without_relevant_overall"] == 4  # 2 authentic controls, 1 pilot neg control, 1 syn control
    assert eval_result.denominators["evaluable_queries_authentic"] == 5
    assert eval_result.denominators["queries_without_relevant_authentic"] == 2
    assert eval_result.denominators["evaluable_queries_pilot"] == 5
    assert eval_result.denominators["queries_without_relevant_pilot"] == 1
    assert eval_result.denominators["evaluable_queries_synthetic"] == 1
    assert eval_result.denominators["queries_without_relevant_synthetic"] == 1
    assert eval_result.denominators["candidate_pool_size"] == 30

    # Comparative hypothesis check:
    # TF-IDF should achieve higher or equal Recall@5 than FIFO because FIFO suffers from distractor crowding
    assert eval_result.metrics["tfidf_overall_recall_at_5"] >= eval_result.metrics["fifo_overall_recall_at_5"]
    assert eval_result.metrics["tfidf_overall_recall_at_1"] > eval_result.metrics["fifo_overall_recall_at_1"]

    # Cap miss rate verification
    assert eval_result.metrics["fifo_overall_cap_miss_rate"] == 0.6364
    assert eval_result.metrics["tfidf_overall_cap_miss_rate"] == 0.0

    # Synthetic cap miss verification: SYN-CAP-MISS-01 must register as a cap miss under FIFO and recovered under TF-IDF
    assert eval_result.metrics["fifo_synthetic_cap_miss_rate"] == 1.0
    assert eval_result.metrics["fifo_synthetic_recall_at_20"] == 0.0
    assert eval_result.metrics["tfidf_synthetic_recall_at_1"] == 1.0
    assert eval_result.metrics["tfidf_synthetic_cap_miss_rate"] == 0.0

    # Authentic literature metrics verification
    assert eval_result.metrics["fifo_authentic_cap_miss_rate"] == 0.8
    assert eval_result.metrics["fifo_authentic_recall_at_20"] == 0.2
    assert eval_result.metrics["tfidf_authentic_recall_at_1"] == 1.0

    # Human-curated pilot metrics verification
    assert eval_result.metrics["fifo_pilot_cap_miss_rate"] == 0.4
    assert eval_result.metrics["fifo_pilot_recall_at_20"] == 0.6
    assert eval_result.metrics["tfidf_pilot_recall_at_1"] == 1.0

    # Verify SYN-CAP-MISS-01 query outcome specifically
    syn_fifo_outcomes = {o.query_id: o for o in results["fifo_synthetic"].query_outcomes}
    syn_tfidf_outcomes = {o.query_id: o for o in results["tfidf_synthetic"].query_outcomes}
    assert syn_fifo_outcomes["SYN-CAP-MISS-01"].has_cap_miss is True
    assert syn_fifo_outcomes["SYN-CAP-MISS-01"].recall_at_k[20] == 0.0
    assert syn_tfidf_outcomes["SYN-CAP-MISS-01"].has_cap_miss is False
    assert syn_tfidf_outcomes["SYN-CAP-MISS-01"].recall_at_k[1] == 1.0


def test_stratification_preserves_partition_independence():
    """Stratified summaries must partition queries cleanly into authentic, pilot, and synthetic subsets."""
    candidate_pool, queries = get_stage_b_candidate_recall_benchmark_dataset()

    runner = CandidateRecallBenchmarkRunner(
        candidate_pool=candidate_pool,
        queries=queries,
        k_values=(1, 5),
        max_candidates=20,
    )
    results = runner.run()

    fifo_auth = results["fifo_authentic"]
    fifo_pilot = results["fifo_pilot"]
    fifo_syn = results["fifo_synthetic"]

    assert all(o.case_type == BenchmarkCaseType.AUTHENTIC_LITERATURE for o in fifo_auth.query_outcomes)
    assert all(o.case_type == BenchmarkCaseType.HUMAN_CURATED_PILOT for o in fifo_pilot.query_outcomes)
    assert all(o.case_type == BenchmarkCaseType.SYNTHETIC_STRESS_TEST for o in fifo_syn.query_outcomes)

    assert fifo_auth.total_queries == 7
    assert fifo_pilot.total_queries == 6
    assert fifo_syn.total_queries == 2
    assert fifo_auth.total_queries + fifo_pilot.total_queries + fifo_syn.total_queries == len(queries)


def test_label_provenance_records_integrity():
    """All queries must carry explicit, verifiable label provenance records with sources and rationales."""
    _, queries = get_stage_b_candidate_recall_benchmark_dataset()

    for q in queries:
        assert len(q.provenance_records) > 0, f"Query {q.query_id} is missing provenance records"
        for rec in q.provenance_records:
            assert isinstance(rec, OpportunityRelationshipProvenance)
            assert rec.opportunity_id, f"Empty opportunity_id in query {q.query_id}"
            assert rec.source_reference, f"Empty source_reference in query {q.query_id}"
            assert rec.rationale, f"Empty rationale in query {q.query_id}"
            assert rec.expected_relationship in {
                "ADDRESSES", "SUPPORTS", "CONTRADICTS", "EXTENDS", "HARD_NEGATIVE", "UNRELATED_CONTROL"
            }

            if rec.is_relevant:
                assert rec.opportunity_id in q.relevant_opportunity_ids
            else:
                assert rec.opportunity_id not in q.relevant_opportunity_ids


def test_same_subfield_hard_negative_discrimination():
    """Verifies that same-subfield hard negatives are present, verified, and correctly discriminated."""
    candidate_pool, queries = get_stage_b_candidate_recall_benchmark_dataset()
    cand_by_id = {c.opportunity_id: c for c in candidate_pool}
    queries_by_id = {q.query_id: q for q in queries}

    # 1. RAG Literature same-subfield hard negatives:
    # AUTH-RAG-02 (Shi 2023) and AUTH-RAG-03 (Yoran 2023) target distraction robustness,
    # with citation attribution as a same-subfield hard negative.
    q_yoran = queries_by_id["AUTH-RAG-03"]
    assert "opp-rag-distraction-robustness" in q_yoran.relevant_opportunity_ids
    assert "opp-rag-citations-attribution" in q_yoran.hard_negative_opportunity_ids
    assert cand_by_id["opp-rag-distraction-robustness"].domain == cand_by_id["opp-rag-citations-attribution"].domain

    # AUTH-RAG-05 (Gao 2023 ALCE) targets citation attribution, with distraction robustness as hard negative
    q_gao = queries_by_id["AUTH-RAG-05"]
    assert "opp-rag-citations-attribution" in q_gao.relevant_opportunity_ids
    assert "opp-rag-distraction-robustness" in q_gao.hard_negative_opportunity_ids

    # 2. Federated Learning same-subfield hard negatives:
    # PILOT-FL-01 (RobustAgg) targets dropout, with DP as hard negative
    q_robustagg = queries_by_id["PILOT-FL-01"]
    assert "opp-fl-dropout-bottlenecks" in q_robustagg.relevant_opportunity_ids
    assert "opp-fl-differential-privacy" in q_robustagg.hard_negative_opportunity_ids
    assert cand_by_id["opp-fl-dropout-bottlenecks"].domain == cand_by_id["opp-fl-differential-privacy"].domain

    # PILOT-FL-02 (DP in FL) targets DP, with dropout as hard negative
    q_dp_fl = queries_by_id["PILOT-FL-02"]
    assert "opp-fl-differential-privacy" in q_dp_fl.relevant_opportunity_ids
    assert "opp-fl-dropout-bottlenecks" in q_dp_fl.hard_negative_opportunity_ids

    # 3. Verify TF-IDF ranks the target ahead of the hard negative for these cases
    ranker = TFIDFLexicalRanker()
    for q in [q_yoran, q_gao, q_robustagg, q_dp_fl]:
        eligible = filter_temporally_eligible_candidates(candidate_pool, q.publication_date)
        ranked = ranker.rank(q, eligible)
        ranked_ids = [c.opportunity_id for c in ranked]
        target_id = next(iter(q.relevant_opportunity_ids))
        hard_neg_id = next(iter(q.hard_negative_opportunity_ids))
        assert target_id in ranked_ids
        assert hard_neg_id in ranked_ids
        assert ranked_ids.index(target_id) < ranked_ids.index(hard_neg_id), (
            f"Query {q.query_id}: Expected target {target_id} to outrank hard negative {hard_neg_id}"
        )


def test_temporal_filtering_on_authentic_literature_queries():
    """Authentic literature queries must strictly exclude future opportunities according to publication dates."""
    candidate_pool, queries = get_stage_b_candidate_recall_benchmark_dataset()
    queries_by_id = {q.query_id: q for q in queries}

    # Lewis et al. published 2020-05-22
    q_lewis = queries_by_id["AUTH-RAG-01"]
    eligible_lewis = filter_temporally_eligible_candidates(candidate_pool, q_lewis.publication_date)
    eligible_ids_lewis = {c.opportunity_id for c in eligible_lewis}

    # Must include opp-rag-knowledge-limits (created 2020-05-01)
    assert "opp-rag-knowledge-limits" in eligible_ids_lewis
    # Must exclude RAG opportunities created in 2023
    assert "opp-rag-distraction-robustness" not in eligible_ids_lewis
    assert "opp-rag-citations-attribution" not in eligible_ids_lewis
    assert "opp-rag-adaptive-retrieval" not in eligible_ids_lewis

    # ResNet control published 2016-06-01 predates entire pool
    q_resnet = queries_by_id["AUTH-CTRL-01"]
    eligible_resnet = filter_temporally_eligible_candidates(candidate_pool, q_resnet.publication_date)
    assert len(eligible_resnet) == 0


def test_unverified_relationships_excluded_from_scored_metrics():
    """Any unverified relationship must not be included in relevant_opportunity_ids or scored metrics."""
    query = PaperQuery(
        query_id="Q-UNVERIFIED",
        paper_id="P-UNVERIFIED",
        title="Unverified study on retrieval robustness",
        abstract="Exploratory discussion of potential knowledge boundaries.",
        publication_date=datetime.date(2023, 1, 1),
        case_type=BenchmarkCaseType.SYNTHETIC_STRESS_TEST,
        relevant_opportunity_ids=set(),  # Excluded from scored relevance
        unverified_opportunity_ids={"opp-unverified-001"},
        provenance_records=[
            OpportunityRelationshipProvenance(
                opportunity_id="opp-unverified-001",
                expected_relationship="BORDERLINE_UNVERIFIED",
                is_relevant=False,
                source_reference="exploratory_notes.md",
                rationale="Unverified hypothesis without sufficient experimental evidence.",
                is_verified=False,
            )
        ],
    )

    opp = CandidateOpportunity(
        opportunity_id="opp-unverified-001",
        title="Unverified gap",
        problem_statement="Problem",
        description="Desc",
        effective_creation_date=datetime.date(2022, 1, 1),
    )

    outcome = evaluate_query_ranking(query, [opp], [opp], k_values=(1, 5))
    assert outcome.relevant_count == 0
    assert outcome.has_cap_miss is False
    assert outcome.recall_at_k[1] == 0.0
