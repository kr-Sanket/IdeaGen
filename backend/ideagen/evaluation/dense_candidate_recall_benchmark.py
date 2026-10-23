"""Stage C Offline Dense and Hybrid Candidate-Retrieval Benchmark.

Evaluates dense semantic and hybrid (lexical + dense) candidate ranking for incremental
opportunity evolution without external cloud APIs, using local nomic-embed-text (768-dim)
and Reciprocal Rank Fusion (RRF).

Strategies compared:
1. Chronological FIFO baseline
2. TF-IDF Lexical Ranker
3. Dense Cosine Ranker (nomic-embed-text 768-dim)
4. Reciprocal Rank Fusion (RRF, k=60) combining TF-IDF and Dense rankings

Features:
- Reuses Stage B candidate pool, temporal causality filtering, and three-way partitioning
  (AUTHENTIC_LITERATURE, HUMAN_CURATED_PILOT, SYNTHETIC_STRESS_TEST).
- Adds defensible vocabulary-mismatch evaluation (e.g. catastrophic forgetting / EWC vs.
  plasticity preservation / Fisher information regularizers from Stage 4 SYN-01).
- Uses genuine local OllamaEmbeddingProvider (nomic-embed-text 768d) with an offline
  embedding cache to support hermetic, deterministic evaluation inside sandbox and CI.
- Halts with an informative RuntimeError blocker if embeddings are requested but the local
  model is unavailable and no genuine cached vectors exist (no random or mock vector substitution).
- Implements deterministic tie-breaking for both cosine ranking and RRF.
- Evaluates Recall@1, 3, 5, 10, 20 and query-level & opportunity-level cap misses.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ideagen.discovery.embeddings import EmbeddingProvider, OllamaEmbeddingProvider
from ideagen.evaluation.candidate_recall_benchmark import (
    DEFAULT_K_VALUES,
    BenchmarkCaseType,
    BenchmarkSplit,
    CandidateOpportunity,
    ChronologicalFIFORanker,
    OpportunityRelationshipProvenance,
    PaperQuery,
    QueryRankingOutcome,
    StrategyBenchmarkSummary,
    TFIDFLexicalRanker,
    evaluate_query_ranking,
    filter_temporally_eligible_candidates,
    get_stage_b_candidate_recall_benchmark_dataset,
    get_stage_d_candidate_recall_benchmark_dataset,
    summarize_strategy_results,
)
from ideagen.evaluation.schemas import EvaluationResult, EvaluationType

logger = logging.getLogger(__name__)

# Standard RRF smoothing constant (Cormack, Clarke, and Büttcher, SIGIR 2009)
DEFAULT_RRF_K: int = 60

# Default cache file path relative to this module
_DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "stage_c_embedding_cache.json"


class DenseCosineRanker:
    """Ranks candidate opportunities using dense vector cosine similarity.

    Uses an EmbeddingProvider (defaults to local Ollama nomic-embed-text, 768-dim)
    to generate L2-normalized float32 vectors over candidate opportunity text
    (title + problem_statement + description) and query paper text (title + abstract).

    Supports an offline embedding cache to enable fast, hermetic evaluation without
    repeated network calls or when external daemons are offline.

    Deterministic tie-breaking: (-similarity, effective_creation_date.asc(), opportunity_id.asc()).
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        cache: Optional[Dict[str, List[float]]] = None,
        cache_file_path: Optional[Path] = None,
        model_name: str = "nomic-embed-text",
        dimension: int = 768,
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self._provider = embedding_provider
        self._cache: Dict[str, np.ndarray] = {}

        # 1. Load from explicit memory cache if supplied
        if cache:
            for k, v in cache.items():
                self._cache[k] = np.array(v, dtype=np.float32)

        # 2. Load from disk cache file if available
        cache_path = cache_file_path or _DEFAULT_CACHE_PATH
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    disk_data = json.load(f)
                    for k, v in disk_data.items():
                        if k not in self._cache:
                            self._cache[k] = np.array(v, dtype=np.float32)
            except Exception as e:
                logger.warning(f"Could not load disk embedding cache from {cache_path}: {e}")

    def _get_provider(self) -> EmbeddingProvider:
        if self._provider is None:
            self._provider = OllamaEmbeddingProvider(
                model_name=self.model_name,
                dimension=self.dimension,
            )
        return self._provider

    def get_embedding(self, text: str) -> np.ndarray:
        """Retrieves or computes an L2-normalized vector for the given text."""
        clean_text = text.strip()
        if clean_text in self._cache:
            return self._cache[clean_text]

        # Attempt to compute using live provider
        try:
            provider = self._get_provider()
            vectors = provider.embed_texts([clean_text])
            if vectors.shape[0] == 0:
                raise RuntimeError("Embedding provider returned empty array.")
            vec = vectors[0].astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self._cache[clean_text] = vec
            return vec
        except Exception as e:
            raise RuntimeError(
                f"Embedding model '{self.model_name}' is unavailable and text is not in offline cache: {e}. "
                "Cannot compute genuine dense baseline without model weights or cached embeddings."
            ) from e

    def rank(
        self,
        query: PaperQuery,
        eligible_candidates: Sequence[CandidateOpportunity],
        max_candidates: Optional[int] = None,
    ) -> List[CandidateOpportunity]:
        if not eligible_candidates:
            return []

        query_text = f"{query.title} {query.abstract}"
        query_vec = self.get_embedding(query_text)

        cand_texts = [
            f"{c.title} {c.problem_statement} {c.description}"
            for c in eligible_candidates
        ]
        cand_vecs = np.array([self.get_embedding(t) for t in cand_texts], dtype=np.float32)

        # Cosine similarity via dot product of unit-normalized vectors
        similarities = (cand_vecs @ query_vec).flatten()

        scored = list(zip(similarities, eligible_candidates))
        # Deterministic sorting: -similarity, effective_creation_date, opportunity_id
        scored.sort(
            key=lambda item: (-item[0], item[1].effective_creation_date, str(item[1].opportunity_id))
        )

        ranked = [c for _, c in scored]
        if max_candidates is not None and max_candidates > 0:
            return ranked[:max_candidates]
        return ranked


class ReciprocalRankFusionRanker:
    """Ranks candidates by fusing TF-IDF lexical and Dense semantic rankings using RRF.

    Ranking formula (Cormack et al., 2009):
        RRF_score(c) = 1.0 / (k_rrf + r_tfidf(c)) + 1.0 / (k_rrf + r_dense(c))
    where:
        r_tfidf(c) is the 1-based rank position of candidate c in TF-IDF ranking.
        r_dense(c) is the 1-based rank position of candidate c in Dense ranking.
        k_rrf is the rank smoothing constant (default: 60).

    Deterministic tie-breaking:
        (-rrf_score, effective_creation_date.asc(), opportunity_id.asc()).
    """

    def __init__(
        self,
        tfidf_ranker: Optional[TFIDFLexicalRanker] = None,
        dense_ranker: Optional[DenseCosineRanker] = None,
        k_rrf: int = DEFAULT_RRF_K,
    ) -> None:
        self.tfidf_ranker = tfidf_ranker or TFIDFLexicalRanker()
        self.dense_ranker = dense_ranker or DenseCosineRanker()
        self.k_rrf = k_rrf

    def rank(
        self,
        query: PaperQuery,
        eligible_candidates: Sequence[CandidateOpportunity],
        max_candidates: Optional[int] = None,
    ) -> List[CandidateOpportunity]:
        if not eligible_candidates:
            return []

        # Obtain full un-truncated rankings across all eligible candidates
        tfidf_ranked = self.tfidf_ranker.rank(query, eligible_candidates, max_candidates=None)
        dense_ranked = self.dense_ranker.rank(query, eligible_candidates, max_candidates=None)

        tfidf_rank_map = {c.opportunity_id: i + 1 for i, c in enumerate(tfidf_ranked)}
        dense_rank_map = {c.opportunity_id: i + 1 for i, c in enumerate(dense_ranked)}

        scored: List[Tuple[float, CandidateOpportunity]] = []
        for c in eligible_candidates:
            r_t = tfidf_rank_map[c.opportunity_id]
            r_d = dense_rank_map[c.opportunity_id]
            rrf_score = (1.0 / (self.k_rrf + r_t)) + (1.0 / (self.k_rrf + r_d))
            scored.append((rrf_score, c))

        # Deterministic tie-breaking: -rrf_score, effective_creation_date, opportunity_id
        scored.sort(
            key=lambda item: (-item[0], item[1].effective_creation_date, str(item[1].opportunity_id))
        )

        ranked = [c for _, c in scored]
        if max_candidates is not None and max_candidates > 0:
            return ranked[:max_candidates]
        return ranked


# ══════════════════════════════════════════════════════════════════════════════
#  Stage C Dataset Construction (Includes Vocabulary Mismatch)
# ══════════════════════════════════════════════════════════════════════════════

def get_stage_c_candidate_recall_benchmark_dataset() -> Tuple[List[CandidateOpportunity], List[PaperQuery]]:
    """Constructs the Stage C dataset extending Stage B with vocabulary-mismatch evaluation."""
    candidates, queries = get_stage_b_candidate_recall_benchmark_dataset()

    # Dedicated vocabulary-mismatch opportunity (derived from Stage 4 SYN-01 continual learning gap)
    opp_forgetting = CandidateOpportunity(
        opportunity_id="opp-syn-forgetting-001",
        title="Catastrophic forgetting in continual domain adaptation",
        problem_statement=(
            "Pre-trained parametric weights suffer from catastrophic forgetting during sequential fine-tuning across disparate domains."
        ),
        description="Investigate continual representation learning via elastic weight consolidation.",
        effective_creation_date=datetime.date(2021, 1, 1),
        domain="Synthetic Continual Learning",
        case_type=BenchmarkCaseType.SYNTHETIC_STRESS_TEST,
    )
    candidates.append(opp_forgetting)

    # Vocabulary-mismatch query: describes catastrophic forgetting and EWC without using the exact phrasing
    q_vocab_mismatch = PaperQuery(
        query_id="SYN-VOCAB-01",
        paper_id="paper-syn-vocab-mismatch",
        title="Plasticity Preservation Under Sequential Distribution Shift",
        abstract=(
            "Deep neural networks exhibit severe representational interference and skill degradation when incrementally "
            "trained across distinct tasks. We protect previously consolidated capabilities using Fisher information matrix regularizers."
        ),
        publication_date=datetime.date(2022, 1, 1),
        case_type=BenchmarkCaseType.SYNTHETIC_STRESS_TEST,
        relevant_opportunity_ids={"opp-syn-forgetting-001"},
        provenance_records=[
            OpportunityRelationshipProvenance(
                opportunity_id="opp-syn-forgetting-001",
                expected_relationship="ADDRESSES",
                is_relevant=True,
                source_reference="stage4_benchmark.py:SYN-01",
                rationale=(
                    "Synthetic stress case testing vocabulary mismatch. Concepts of catastrophic forgetting and elastic weight "
                    "consolidation are expressed without shared keywords (plasticity preservation, representational interference, "
                    "Fisher information matrix regularizers)."
                ),
            )
        ],
        notes="Synthetic vocabulary mismatch query testing dense semantic vs. lexical retrieval.",
    )
    queries.append(q_vocab_mismatch)

    return candidates, queries


# ══════════════════════════════════════════════════════════════════════════════
#  Stage C Benchmark Runner
# ══════════════════════════════════════════════════════════════════════════════

class DenseCandidateRecallBenchmarkRunner:
    """Executes the offline benchmark comparing FIFO, TF-IDF, Dense Cosine, and Hybrid RRF."""

    def __init__(
        self,
        candidate_pool: Sequence[CandidateOpportunity],
        queries: Sequence[PaperQuery],
        k_values: Sequence[int] = DEFAULT_K_VALUES,
        max_candidates: Optional[int] = 20,
        cutoff_date: Optional[datetime.date] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        embedding_cache: Optional[Dict[str, List[float]]] = None,
        k_rrf: int = DEFAULT_RRF_K,
    ) -> None:
        self.candidate_pool = list(candidate_pool)
        self.queries = list(queries)
        self.k_values = tuple(k_values)
        self.max_candidates = max_candidates
        self.cutoff_date = cutoff_date
        self.k_rrf = k_rrf

        self.fifo_ranker = ChronologicalFIFORanker()
        self.tfidf_ranker = TFIDFLexicalRanker()
        self.dense_ranker = DenseCosineRanker(
            embedding_provider=embedding_provider,
            cache=embedding_cache,
        )
        self.rrf_ranker = ReciprocalRankFusionRanker(
            tfidf_ranker=self.tfidf_ranker,
            dense_ranker=self.dense_ranker,
            k_rrf=self.k_rrf,
        )

    def run(self) -> Dict[str, Any]:
        """Runs the four-way benchmark across all queries and returns comprehensive structured reports."""
        fifo_outcomes: List[QueryRankingOutcome] = []
        tfidf_outcomes: List[QueryRankingOutcome] = []
        dense_outcomes: List[QueryRankingOutcome] = []
        rrf_outcomes: List[QueryRankingOutcome] = []

        for q in self.queries:
            # 1. Temporal filtering (identical across all rankers)
            eligible = filter_temporally_eligible_candidates(
                self.candidate_pool,
                paper_publication_date=q.publication_date,
                cutoff_date=self.cutoff_date,
            )

            # 2. FIFO ranking
            fifo_ranked = self.fifo_ranker.rank(
                query=q,
                eligible_candidates=eligible,
                max_candidates=self.max_candidates,
            )
            fifo_outcomes.append(
                evaluate_query_ranking(
                    query=q,
                    ranked_candidates=fifo_ranked,
                    eligible_candidates=eligible,
                    k_values=self.k_values,
                    max_candidates=self.max_candidates,
                )
            )

            # 3. TF-IDF ranking
            tfidf_ranked = self.tfidf_ranker.rank(
                query=q,
                eligible_candidates=eligible,
                max_candidates=self.max_candidates,
            )
            tfidf_outcomes.append(
                evaluate_query_ranking(
                    query=q,
                    ranked_candidates=tfidf_ranked,
                    eligible_candidates=eligible,
                    k_values=self.k_values,
                    max_candidates=self.max_candidates,
                )
            )

            # 4. Dense Cosine ranking
            dense_ranked = self.dense_ranker.rank(
                query=q,
                eligible_candidates=eligible,
                max_candidates=self.max_candidates,
            )
            dense_outcomes.append(
                evaluate_query_ranking(
                    query=q,
                    ranked_candidates=dense_ranked,
                    eligible_candidates=eligible,
                    k_values=self.k_values,
                    max_candidates=self.max_candidates,
                )
            )

            # 5. Hybrid RRF ranking
            rrf_ranked = self.rrf_ranker.rank(
                query=q,
                eligible_candidates=eligible,
                max_candidates=self.max_candidates,
            )
            rrf_outcomes.append(
                evaluate_query_ranking(
                    query=q,
                    ranked_candidates=rrf_ranked,
                    eligible_candidates=eligible,
                    k_values=self.k_values,
                    max_candidates=self.max_candidates,
                )
            )

        # Summarize across all 4 strategies and partitions
        strategies = {
            "fifo": ("FIFO", fifo_outcomes),
            "tfidf": ("TF-IDF", tfidf_outcomes),
            "dense": ("Dense", dense_outcomes),
            "rrf": ("RRF", rrf_outcomes),
        }

        summaries: Dict[str, Dict[str, StrategyBenchmarkSummary]] = {}
        for strat_key, (strat_name, outcomes) in strategies.items():
            summaries[strat_key] = {
                "overall": summarize_strategy_results(strat_name, outcomes, None, self.k_values),
                "authentic": summarize_strategy_results(strat_name, outcomes, BenchmarkCaseType.AUTHENTIC_LITERATURE, self.k_values),
                "pilot": summarize_strategy_results(strat_name, outcomes, BenchmarkCaseType.HUMAN_CURATED_PILOT, self.k_values),
                "synthetic": summarize_strategy_results(strat_name, outcomes, BenchmarkCaseType.SYNTHETIC_STRESS_TEST, self.k_values),
            }

        # Assemble EvaluationResult metrics
        metrics: Dict[str, float] = {}
        for strat_key in ["fifo", "tfidf", "dense", "rrf"]:
            for part in ["overall", "authentic", "pilot", "synthetic"]:
                summary = summaries[strat_key][part]
                for k in self.k_values:
                    metrics[f"{strat_key}_{part}_recall_at_{k}"] = round(summary.mean_recall_at_k.get(k, 0.0), 4)
                metrics[f"{strat_key}_{part}_cap_miss_rate"] = round(summary.cap_miss_rate, 4)
                metrics[f"{strat_key}_{part}_opportunity_cap_misses"] = float(summary.total_cap_misses)

        ref_summary = summaries["fifo"]["overall"]
        auth_summary = summaries["fifo"]["authentic"]
        pilot_summary = summaries["fifo"]["pilot"]
        syn_summary = summaries["fifo"]["synthetic"]

        denominators = {
            "total_queries": len(self.queries),
            "evaluable_queries_overall": ref_summary.evaluable_queries,
            "queries_without_relevant_overall": ref_summary.queries_without_relevant,
            "evaluable_queries_authentic": auth_summary.evaluable_queries,
            "queries_without_relevant_authentic": auth_summary.queries_without_relevant,
            "evaluable_queries_pilot": pilot_summary.evaluable_queries,
            "queries_without_relevant_pilot": pilot_summary.queries_without_relevant,
            "evaluable_queries_synthetic": syn_summary.evaluable_queries,
            "queries_without_relevant_synthetic": syn_summary.queries_without_relevant,
            "candidate_pool_size": len(self.candidate_pool),
        }

        eval_result = EvaluationResult(
            evaluation_name="candidate_recall_stage_c",
            fixture_version="v3.0-dense-hybrid-rrf",
            evaluation_type=EvaluationType.EXECUTED,
            model_provider=f"Ollama ({self.dense_ranker.model_name} {self.dense_ranker.dimension}d) / TfidfVectorizer / RRF (k={self.k_rrf}) / FIFO",
            parameters={
                "k_values": list(self.k_values),
                "max_candidates": self.max_candidates,
                "cutoff_date": self.cutoff_date.isoformat() if self.cutoff_date else None,
                "embedding_model": self.dense_ranker.model_name,
                "embedding_dimension": self.dense_ranker.dimension,
                "rrf_k": self.k_rrf,
                "caching_behavior": "deterministic in-memory and local disk persistence",
            },
            metrics=metrics,
            denominators=denominators,
            errors=[],
            limitations=(
                "Stage C offline benchmark evaluates chronological FIFO vs. TF-IDF lexical vs. Dense nomic-embed-text (768d) "
                "vs. Hybrid Reciprocal Rank Fusion (k=60). The corpus uses authentic 5-paper RAG literature with source PDFs, "
                "human-curated pilot cases with same-subfield hard negatives, and synthetic vocabulary-mismatch and cap-truncation stress tests. "
                "Dense rankings use genuine local embeddings. Results characterize candidate-filtering recall under controlled candidate pools, "
                "not open-web generalization across arbitrary literature."
            ),
            is_engineering_validation=False,
        )

        return {
            "evaluation_result": eval_result,
            "summaries": summaries,
            "fifo_overall": summaries["fifo"]["overall"],
            "fifo_authentic": summaries["fifo"]["authentic"],
            "fifo_pilot": summaries["fifo"]["pilot"],
            "fifo_synthetic": summaries["fifo"]["synthetic"],
            "tfidf_overall": summaries["tfidf"]["overall"],
            "tfidf_authentic": summaries["tfidf"]["authentic"],
            "tfidf_pilot": summaries["tfidf"]["pilot"],
            "tfidf_synthetic": summaries["tfidf"]["synthetic"],
            "dense_overall": summaries["dense"]["overall"],
            "dense_authentic": summaries["dense"]["authentic"],
            "dense_pilot": summaries["dense"]["pilot"],
            "dense_synthetic": summaries["dense"]["synthetic"],
            "rrf_overall": summaries["rrf"]["overall"],
            "rrf_authentic": summaries["rrf"]["authentic"],
            "rrf_pilot": summaries["rrf"]["pilot"],
            "rrf_synthetic": summaries["rrf"]["synthetic"],
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Stage D Independent DEV / HELD_OUT Benchmark Runner
# ══════════════════════════════════════════════════════════════════════════════

class StageDCandidateRecallBenchmarkRunner:
    """Executes the Stage D independent candidate recall benchmark strictly separating DEV and HELD_OUT.

    Evaluates Chronological FIFO, TF-IDF Lexical, Dense Cosine (nomic-embed-text 768d),
    and Reciprocal Rank Fusion (RRF, k=60).

    Stratifies evaluation across:
    - Splits: DEV (construction/tuning, N=13) vs. HELD_OUT (evaluation, N=14)
    - Partitions: AUTHENTIC_LITERATURE, HUMAN_CURATED_PILOT, SYNTHETIC_STRESS_TEST
    - Rankers: FIFO, TF-IDF, Dense Cosine, Hybrid RRF
    """

    def __init__(
        self,
        split: Optional[BenchmarkSplit] = None,  # If None, evaluates both DEV and HELD_OUT
        k_values: Sequence[int] = DEFAULT_K_VALUES,
        max_candidates: Optional[int] = 20,
        cutoff_date: Optional[datetime.date] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        embedding_cache: Optional[Dict[str, List[float]]] = None,
        k_rrf: int = DEFAULT_RRF_K,
    ) -> None:
        self.target_split = split
        self.k_values = tuple(k_values)
        self.max_candidates = max_candidates
        self.cutoff_date = cutoff_date
        self.k_rrf = k_rrf

        self.fifo_ranker = ChronologicalFIFORanker()
        self.tfidf_ranker = TFIDFLexicalRanker()
        self.dense_ranker = DenseCosineRanker(
            embedding_provider=embedding_provider,
            cache=embedding_cache,
        )
        self.rrf_ranker = ReciprocalRankFusionRanker(
            tfidf_ranker=self.tfidf_ranker,
            dense_ranker=self.dense_ranker,
            k_rrf=self.k_rrf,
        )

    def run(self) -> Dict[str, Any]:
        """Runs the Stage D evaluation across the configured split(s) and partitions."""
        splits_to_run = (
            [self.target_split] if self.target_split is not None
            else [BenchmarkSplit.DEV, BenchmarkSplit.HELD_OUT]
        )

        all_summaries: Dict[str, Dict[str, Dict[str, StrategyBenchmarkSummary]]] = {}
        all_metrics: Dict[str, float] = {}
        all_denominators: Dict[str, Any] = {}
        split_results: Dict[str, Any] = {}

        candidate_pool_size = 0

        for s in splits_to_run:
            split_key = s.value.lower()
            candidate_pool, queries = get_stage_d_candidate_recall_benchmark_dataset(s)
            candidate_pool_size = len(candidate_pool)

            fifo_outcomes: List[QueryRankingOutcome] = []
            tfidf_outcomes: List[QueryRankingOutcome] = []
            dense_outcomes: List[QueryRankingOutcome] = []
            rrf_outcomes: List[QueryRankingOutcome] = []

            for q in queries:
                eligible = filter_temporally_eligible_candidates(
                    candidate_pool,
                    q.publication_date,
                    cutoff_date=self.cutoff_date,
                )

                # 1. FIFO
                r_fifo = self.fifo_ranker.rank(q, eligible, max_candidates=self.max_candidates)
                fifo_outcomes.append(
                    evaluate_query_ranking(q, r_fifo, eligible, self.k_values, self.max_candidates)
                )

                # 2. TF-IDF
                r_tfidf = self.tfidf_ranker.rank(q, eligible, max_candidates=self.max_candidates)
                tfidf_outcomes.append(
                    evaluate_query_ranking(q, r_tfidf, eligible, self.k_values, self.max_candidates)
                )

                # 3. Dense Cosine
                r_dense = self.dense_ranker.rank(q, eligible, max_candidates=self.max_candidates)
                dense_outcomes.append(
                    evaluate_query_ranking(q, r_dense, eligible, self.k_values, self.max_candidates)
                )

                # 4. Hybrid RRF
                r_rrf = self.rrf_ranker.rank(q, eligible, max_candidates=self.max_candidates)
                rrf_outcomes.append(
                    evaluate_query_ranking(q, r_rrf, eligible, self.k_values, self.max_candidates)
                )

            strategies = {
                "fifo": ("FIFO", fifo_outcomes),
                "tfidf": ("TF-IDF", tfidf_outcomes),
                "dense": ("Dense", dense_outcomes),
                "rrf": ("RRF", rrf_outcomes),
            }

            split_summaries: Dict[str, Dict[str, StrategyBenchmarkSummary]] = {}
            for strat_key, (strat_name, outcomes) in strategies.items():
                split_summaries[strat_key] = {
                    "overall": summarize_strategy_results(strat_name, outcomes, None, self.k_values),
                    "authentic": summarize_strategy_results(strat_name, outcomes, BenchmarkCaseType.AUTHENTIC_LITERATURE, self.k_values),
                    "pilot": summarize_strategy_results(strat_name, outcomes, BenchmarkCaseType.HUMAN_CURATED_PILOT, self.k_values),
                    "synthetic": summarize_strategy_results(strat_name, outcomes, BenchmarkCaseType.SYNTHETIC_STRESS_TEST, self.k_values),
                }

            all_summaries[split_key] = split_summaries

            # Assemble metrics for this split
            for strat_key in ["fifo", "tfidf", "dense", "rrf"]:
                for part in ["overall", "authentic", "pilot", "synthetic"]:
                    summary = split_summaries[strat_key][part]
                    for k in self.k_values:
                        all_metrics[f"{strat_key}_{split_key}_{part}_recall_at_{k}"] = round(summary.mean_recall_at_k.get(k, 0.0), 4)
                    all_metrics[f"{strat_key}_{split_key}_{part}_cap_miss_rate"] = round(summary.cap_miss_rate, 4)
                    all_metrics[f"{strat_key}_{split_key}_{part}_opportunity_cap_misses"] = float(summary.total_cap_misses)

            ref_summary = split_summaries["fifo"]["overall"]
            auth_summary = split_summaries["fifo"]["authentic"]
            pilot_summary = split_summaries["fifo"]["pilot"]
            syn_summary = split_summaries["fifo"]["synthetic"]

            all_denominators[f"total_queries_{split_key}"] = len(queries)
            all_denominators[f"evaluable_queries_{split_key}_overall"] = ref_summary.evaluable_queries
            all_denominators[f"queries_without_relevant_{split_key}_overall"] = ref_summary.queries_without_relevant
            all_denominators[f"evaluable_queries_{split_key}_authentic"] = auth_summary.evaluable_queries
            all_denominators[f"queries_without_relevant_{split_key}_authentic"] = auth_summary.queries_without_relevant
            all_denominators[f"evaluable_queries_{split_key}_pilot"] = pilot_summary.evaluable_queries
            all_denominators[f"queries_without_relevant_{split_key}_pilot"] = pilot_summary.queries_without_relevant
            all_denominators[f"evaluable_queries_{split_key}_synthetic"] = syn_summary.evaluable_queries
            all_denominators[f"queries_without_relevant_{split_key}_synthetic"] = syn_summary.queries_without_relevant

            split_results[split_key] = {
                "queries": queries,
                "summaries": split_summaries,
            }

        all_denominators["candidate_pool_size"] = candidate_pool_size

        eval_result = EvaluationResult(
            evaluation_name="candidate_recall_stage_d",
            fixture_version="v4.0-independent-dev-held-out",
            evaluation_type=EvaluationType.EXECUTED,
            model_provider=f"Ollama ({self.dense_ranker.model_name} {self.dense_ranker.dimension}d) / TfidfVectorizer / RRF (k={self.k_rrf}) / FIFO",
            parameters={
                "k_values": list(self.k_values),
                "max_candidates": self.max_candidates,
                "cutoff_date": self.cutoff_date.isoformat() if self.cutoff_date else None,
                "embedding_model": self.dense_ranker.model_name,
                "embedding_dimension": self.dense_ranker.dimension,
                "rrf_k": self.k_rrf,
                "target_split": self.target_split.value if self.target_split else "ALL",
                "caching_behavior": "deterministic in-memory and local disk persistence",
            },
            metrics=all_metrics,
            denominators=all_denominators,
            errors=[],
            limitations=(
                "Stage D offline benchmark strictly isolates DEV (construction/tuning, N=13) from HELD_OUT (evaluation, N=14). "
                "The authentic held-out subset contains a tiny sample (N=1 evaluable query: Yoran et al. 2023, plus 1 control) "
                "governed by the five-paper corpus boundary. The pilot held-out subset contains 11 unutilized human-curated vignettes "
                "with simulated benchmark dates. All rankers are evaluated frozen without hyperparameter tuning on HELD_OUT."
            ),
            is_engineering_validation=False,
        )

        return {
            "evaluation_result": eval_result,
            "all_summaries": all_summaries,
            "split_results": split_results,
            "metrics": all_metrics,
            "denominators": all_denominators,
        }

