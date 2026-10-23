"""Stage B In-Memory Candidate-Recall Benchmark.

Evaluates candidate opportunity selection and ranking for incremental lifecycle evolution
without calling LLMs or external embedding APIs, and without mutating database state.

Stage B Enhancements (Source-Grounded Hard-Negative Benchmark):
1. Candidate opportunities are built directly from source-backed literature (5-paper RAG corpus
   with committed PDFs) and human-curated opportunity classification fixtures.
2. Incorporates same-subfield competing opportunities and defensible hard negatives where
   relevance or irrelevance is supported by source evidence (e.g. differential privacy in federated
   learning as a hard negative against network client dropout; citation attribution vs. irrelevant
   context robustness in RAG).
3. Records explicit label provenance for every scored paper–opportunity pair: source fixture or PDF,
   expected relationship, and evidence rationale.
4. Partitions results into AUTHENTIC_LITERATURE, HUMAN_CURATED_PILOT, and SYNTHETIC_STRESS_TEST,
   preventing conflation of published evidence with pilot vignettes or synthetic edge cases.
5. Keeps uncertain or unverified relationships explicitly segregated from scored metrics.
6. Evaluates Chronological FIFO and TF-IDF lexical ranking under identical temporal eligibility
   and candidate cap constraints (max_candidates=20).
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ideagen.evaluation.schemas import EvaluationResult, EvaluationType


class BenchmarkSplit(str, Enum):
    """Evaluation split separating construction/tuning from held-out evaluation."""
    DEV = "DEV"
    HELD_OUT = "HELD_OUT"


class BenchmarkCaseType(str, Enum):
    """Categorization of benchmark cases to prevent conflating authentic evidence with pilot or stress data."""
    AUTHENTIC_LITERATURE = "AUTHENTIC_LITERATURE"
    HUMAN_CURATED_PILOT = "HUMAN_CURATED_PILOT"
    SYNTHETIC_STRESS_TEST = "SYNTHETIC_STRESS_TEST"


@dataclass(frozen=True)
class OpportunityRelationshipProvenance:
    """Provenance record for a paper-opportunity relationship annotation."""
    opportunity_id: str
    expected_relationship: str  # e.g. "ADDRESSES", "SUPPORTS", "CONTRADICTS", "EXTENDS", "HARD_NEGATIVE", "UNRELATED_CONTROL"
    is_relevant: bool           # True if positive target; False if hard negative or control
    source_reference: str       # e.g. "experiments/data/five_paper_rag_corpus/yoran2023_robust_ralm.pdf", "classification_fixtures.json:fix_10"
    rationale: str              # Concise evidence rationale
    is_verified: bool = True    # If False, uncertain and excluded from scored metrics


@dataclass(frozen=True)
class CandidateOpportunity:
    """An opportunity candidate available in the in-memory pool."""
    opportunity_id: str
    title: str
    problem_statement: str
    description: str
    effective_creation_date: datetime.date
    status: str = "OPEN"
    domain: Optional[str] = None
    case_type: BenchmarkCaseType = BenchmarkCaseType.AUTHENTIC_LITERATURE


@dataclass(frozen=True)
class PaperQuery:
    """A paper query evaluated against candidate opportunities."""
    query_id: str
    paper_id: str
    title: str
    abstract: str
    publication_date: datetime.date
    case_type: BenchmarkCaseType
    # Ground-truth set of opportunity IDs that this paper genuinely addresses, supports,
    # contradicts, or extends. Must be explicitly verified; never inferred from domain match.
    relevant_opportunity_ids: Set[str]
    hard_negative_opportunity_ids: Set[str] = field(default_factory=set)
    unverified_opportunity_ids: Set[str] = field(default_factory=set)
    provenance_records: List[OpportunityRelationshipProvenance] = field(default_factory=list)
    doi: Optional[str] = None
    notes: Optional[str] = None
    is_simulated_date: bool = False
    split: Optional[BenchmarkSplit] = None



@dataclass(frozen=True)
class QueryRankingOutcome:
    """Outcome of ranking candidate opportunities for a single query."""
    query_id: str
    paper_id: str
    case_type: BenchmarkCaseType
    eligible_candidate_count: int
    relevant_count: int
    ranked_opportunity_ids: List[str]
    recall_at_k: Dict[int, float]
    cap_misses_count: int
    has_cap_miss: bool


@dataclass(frozen=True)
class StrategyBenchmarkSummary:
    """Aggregated benchmark metrics for a single ranking strategy across a query subset."""
    strategy_name: str
    case_type_filter: Optional[BenchmarkCaseType]
    total_queries: int
    evaluable_queries: int
    queries_without_relevant: int
    mean_recall_at_k: Dict[int, float]
    total_cap_misses: int
    queries_with_cap_miss: int
    cap_miss_rate: float
    query_outcomes: List[QueryRankingOutcome]


# Default evaluation cutoffs evaluated
DEFAULT_K_VALUES: Tuple[int, ...] = (1, 3, 5, 10, 20)


def filter_temporally_eligible_candidates(
    candidates: Sequence[CandidateOpportunity],
    paper_publication_date: Optional[datetime.date],
    cutoff_date: Optional[datetime.date] = None,
) -> List[CandidateOpportunity]:
    """Filters candidate opportunities by temporal causality matching production incremental evolution.

    Production parity notes:
    - In ``ideagen.opportunity.incremental.IncrementalOpportunityEvolutionService``:
      An opportunity is eligible if ``effective_creation_date <= paper.publication_date``
      and ``effective_creation_date <= cutoff_date`` (if cutoff provided).
    - If ``paper.publication_date`` is missing or None, 0 candidates are eligible.
    """
    if paper_publication_date is None:
        return []

    eligible: List[CandidateOpportunity] = []
    for opp in candidates:
        if opp.effective_creation_date > paper_publication_date:
            continue
        if cutoff_date is not None and opp.effective_creation_date > cutoff_date:
            continue
        eligible.append(opp)
    return eligible


class ChronologicalFIFORanker:
    """Ranks candidate opportunities strictly in chronological order matching production.

    Sort key: (effective_creation_date.asc(), opportunity_id.asc()).
    Corresponds to ``query.order_by(Opportunity.created_at.asc(), Opportunity.id.asc())``
    in ``ideagen.opportunity.incremental``.
    """

    def rank(
        self,
        query: PaperQuery,
        eligible_candidates: Sequence[CandidateOpportunity],
        max_candidates: Optional[int] = None,
    ) -> List[CandidateOpportunity]:
        if not eligible_candidates:
            return []

        sorted_candidates = sorted(
            eligible_candidates,
            key=lambda c: (c.effective_creation_date, str(c.opportunity_id)),
        )

        if max_candidates is not None and max_candidates > 0:
            return sorted_candidates[:max_candidates]
        return sorted_candidates


class TFIDFLexicalRanker:
    """Ranks candidate opportunities using TF-IDF cosine similarity.

    Constructs document vectors over candidate opportunity text (title + problem_statement + description)
    and query paper text (title + abstract). Breaks ties deterministically by
    (effective_creation_date.asc(), opportunity_id.asc()).
    """

    def __init__(
        self,
        stop_words: str = "english",
        ngram_range: Tuple[int, int] = (1, 2),
    ) -> None:
        self.stop_words = stop_words
        self.ngram_range = ngram_range

    def rank(
        self,
        query: PaperQuery,
        eligible_candidates: Sequence[CandidateOpportunity],
        max_candidates: Optional[int] = None,
    ) -> List[CandidateOpportunity]:
        if not eligible_candidates:
            return []

        candidate_texts = [
            f"{c.title} {c.problem_statement} {c.description}"
            for c in eligible_candidates
        ]
        query_text = f"{query.title} {query.abstract}"

        vectorizer = TfidfVectorizer(
            stop_words=self.stop_words,
            ngram_range=self.ngram_range,
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(candidate_texts + [query_text])
        except ValueError:
            # Fallback if empty vocabulary
            sorted_candidates = sorted(
                eligible_candidates,
                key=lambda c: (c.effective_creation_date, str(c.opportunity_id)),
            )
            if max_candidates is not None and max_candidates > 0:
                return sorted_candidates[:max_candidates]
            return sorted_candidates

        cand_vectors = tfidf_matrix[:-1]
        query_vector = tfidf_matrix[-1]
        similarities = cosine_similarity(cand_vectors, query_vector).flatten()

        # Deterministic sorting: -similarity, effective_creation_date, opportunity_id
        scored = list(zip(similarities, eligible_candidates))
        scored.sort(
            key=lambda item: (-item[0], item[1].effective_creation_date, str(item[1].opportunity_id))
        )

        ranked = [c for _, c in scored]
        if max_candidates is not None and max_candidates > 0:
            return ranked[:max_candidates]
        return ranked


def evaluate_query_ranking(
    query: PaperQuery,
    ranked_candidates: Sequence[CandidateOpportunity],
    eligible_candidates: Sequence[CandidateOpportunity],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    max_candidates: Optional[int] = None,
) -> QueryRankingOutcome:
    """Evaluates ranking recall and cap-omission metrics for a single query."""
    relevant_ids = set(query.relevant_opportunity_ids)
    eligible_ids = {c.opportunity_id for c in eligible_candidates}
    ranked_ids = [c.opportunity_id for c in ranked_candidates]

    recall_at_k: Dict[int, float] = {}
    if len(relevant_ids) == 0:
        # Query has no verified relevant opportunities (e.g. control paper or unverified query).
        # We assign 0.0 placeholders in the outcome, but the aggregator will exclude
        # this query from evaluable recall denominators to avoid distorting metrics.
        for k in k_values:
            recall_at_k[k] = 0.0
        cap_misses = 0
    else:
        for k in k_values:
            top_k_ids = set(ranked_ids[:k])
            hits = len(top_k_ids.intersection(relevant_ids))
            recall_at_k[k] = hits / len(relevant_ids)

        # Detect candidate-cap misses:
        # Relevant opportunities that passed temporal eligibility but were pushed
        # beyond max_candidates by the ranking strategy.
        if max_candidates is not None and max_candidates > 0:
            capped_ranked_ids = set(ranked_ids[:max_candidates])
            temporally_eligible_relevant = relevant_ids.intersection(eligible_ids)
            missed_due_to_cap = temporally_eligible_relevant - capped_ranked_ids
            cap_misses = len(missed_due_to_cap)
        else:
            cap_misses = 0

    return QueryRankingOutcome(
        query_id=query.query_id,
        paper_id=query.paper_id,
        case_type=query.case_type,
        eligible_candidate_count=len(eligible_candidates),
        relevant_count=len(relevant_ids),
        ranked_opportunity_ids=ranked_ids,
        recall_at_k=recall_at_k,
        cap_misses_count=cap_misses,
        has_cap_miss=(cap_misses > 0),
    )


def summarize_strategy_results(
    strategy_name: str,
    outcomes: Sequence[QueryRankingOutcome],
    case_type_filter: Optional[BenchmarkCaseType] = None,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> StrategyBenchmarkSummary:
    """Aggregates per-query ranking outcomes into a stratified summary."""
    filtered_outcomes = [
        o for o in outcomes
        if case_type_filter is None or o.case_type == case_type_filter
    ]

    total_queries = len(filtered_outcomes)
    evaluable = [o for o in filtered_outcomes if o.relevant_count > 0]
    queries_without_relevant = total_queries - len(evaluable)

    mean_recall: Dict[int, float] = {}
    for k in k_values:
        if evaluable:
            mean_recall[k] = sum(o.recall_at_k.get(k, 0.0) for o in evaluable) / len(evaluable)
        else:
            mean_recall[k] = 0.0

    total_cap_misses = sum(o.cap_misses_count for o in evaluable)
    queries_with_cap_miss = sum(1 for o in evaluable if o.has_cap_miss)
    cap_miss_rate = (queries_with_cap_miss / len(evaluable)) if evaluable else 0.0

    return StrategyBenchmarkSummary(
        strategy_name=strategy_name,
        case_type_filter=case_type_filter,
        total_queries=total_queries,
        evaluable_queries=len(evaluable),
        queries_without_relevant=queries_without_relevant,
        mean_recall_at_k=mean_recall,
        total_cap_misses=total_cap_misses,
        queries_with_cap_miss=queries_with_cap_miss,
        cap_miss_rate=cap_miss_rate,
        query_outcomes=filtered_outcomes,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Stage B Source-Grounded Dataset Construction
# ══════════════════════════════════════════════════════════════════════════════

def get_stage_b_candidate_recall_benchmark_dataset() -> Tuple[List[CandidateOpportunity], List[PaperQuery]]:
    """Constructs the Stage B source-grounded candidate recall benchmark dataset.

    Features:
    1. Authentic Literature Opportunities from the committed 5-paper RAG corpus
       (Lewis et al., Shi et al., Yoran et al., Asai et al., Gao et al.).
    2. Same-subfield competing opportunities and hard negatives (e.g. DP in Federated Learning
       as a hard negative against network client dropout; citation attribution vs. irrelevant context robustness).
    3. Human-curated pilot opportunities from vetted fixtures (geriatric fasting, FL dropout, DP, GNNs).
    4. Explicit label provenance records for every scored relationship.
    5. Preserved synthetic cap-miss stress test and control queries.
    """
    # ── 1. Authentic Literature Opportunities (5-Paper RAG Corpus) ────────────
    opp_rag_knowledge = CandidateOpportunity(
        opportunity_id="opp-rag-knowledge-limits",
        title="Mitigating Knowledge Access & Manipulation Limits in RAG Models",
        problem_statement=(
            "Large pre-trained language models have limited ability to access and precisely manipulate knowledge, "
            "underperforming task-specific architectures on knowledge-intensive NLP tasks."
        ),
        description="Investigate modular architectures integrating task-specific knowledge retrieval with language models.",
        effective_creation_date=datetime.date(2020, 5, 1),
        domain="NLP / Retrieval-Augmented Generation",
        case_type=BenchmarkCaseType.AUTHENTIC_LITERATURE,
    )

    opp_rag_distraction = CandidateOpportunity(
        opportunity_id="opp-rag-distraction-robustness",
        title="Vulnerability of Retrieval-Augmented Language Models to Irrelevant Context Distraction",
        problem_statement=(
            "Retrieved documents frequently contain irrelevant context or adversarial noise that dramatically degrades "
            "the reasoning and accuracy of large language models."
        ),
        description=(
            "Develop robust retrieval filters and specialized decoding methods that train models to ignore irrelevant "
            "retrieved context and maintain reasoning fidelity."
        ),
        effective_creation_date=datetime.date(2023, 1, 15),
        domain="NLP / Retrieval-Augmented Generation",
        case_type=BenchmarkCaseType.AUTHENTIC_LITERATURE,
    )

    opp_rag_citations = CandidateOpportunity(
        opportunity_id="opp-rag-citations-attribution",
        title="Verifiable Citation and Statement-Level Attribution Generation in Language Models",
        problem_statement=(
            "Existing retrieval-augmented language models generate unsupported claims and lack verifiable fine-grained "
            "citation attributions linked to specific retrieved source spans."
        ),
        description=(
            "Design benchmark tasks, citation evaluation metrics, and end-to-end models capable of producing explicit "
            "sentence-level citations with high citation precision and recall."
        ),
        effective_creation_date=datetime.date(2023, 5, 1),
        domain="NLP / Retrieval-Augmented Generation",
        case_type=BenchmarkCaseType.AUTHENTIC_LITERATURE,
    )

    opp_rag_adaptive = CandidateOpportunity(
        opportunity_id="opp-rag-adaptive-retrieval",
        title="Selective and Adaptive Self-Reflective Retrieval for Knowledge Generation",
        problem_statement=(
            "Standard RAG models retrieve fixed numbers of passages regardless of whether retrieval is necessary, "
            "introducing unnecessary latency and context distortion on straightforward queries."
        ),
        description=(
            "Train language models with reflection tokens to dynamically critique and decide when to retrieve, "
            "when to skip retrieval, and when to generate critique evaluations."
        ),
        effective_creation_date=datetime.date(2023, 10, 1),
        domain="NLP / Retrieval-Augmented Generation",
        case_type=BenchmarkCaseType.AUTHENTIC_LITERATURE,
    )

    # ── 2. Human-Curated Pilot Opportunities (From Fixtures & Stage 4B) ────────
    opp_fl_dropout = CandidateOpportunity(
        opportunity_id="opp-fl-dropout-bottlenecks",
        title="Scalability bottlenecks in federated learning over unreliable networks",
        problem_statement="Current FL protocols fail when client dropout exceeds 40% due to loss of edge connectivity.",
        description="Develop robust aggregation protocols and asynchronous client reconciliation for high-dropout mobile environments.",
        effective_creation_date=datetime.date(2021, 3, 1),
        domain="Distributed Systems / Federated Learning",
        case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
    )

    opp_fl_dp = CandidateOpportunity(
        opportunity_id="opp-fl-differential-privacy",
        title="Differential privacy client budget depletion in cross-device federated learning",
        problem_statement=(
            "Repeated local model updates rapidly exhaust client privacy budgets epsilon under standard Gaussian mechanisms, "
            "forcing early participant dropout."
        ),
        description="Develop adaptive client privacy amplification via subsampling and secure aggregation.",
        effective_creation_date=datetime.date(2021, 4, 1),
        domain="Distributed Systems / Federated Learning",
        case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
    )

    opp_fasting = CandidateOpportunity(
        opportunity_id="opp-fasting-geriatrics",
        title="Lack of longitudinal studies on intermittent fasting in older adults",
        problem_statement="Most studies on intermittent fasting focus on short-term outcomes in young populations, leaving geriatric risks unknown.",
        description="Conduct multi-year trials assessing bone mineral density, sarcopenia, and fracture incidence in patients over 65.",
        effective_creation_date=datetime.date(2021, 1, 15),
        domain="Geriatric Medicine",
        case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
    )

    opp_dp_clinical = CandidateOpportunity(
        opportunity_id="opp-dp-clinical-trials",
        title="High privacy budget epsilon needed for high utility in clinical trial synthetic data generation",
        problem_statement="Existing differential privacy mechanisms require epsilon > 5.0 to maintain clinical trial statistical validity, compromising patient confidentiality.",
        description="Develop privacy amplification mechanisms to achieve sub-unitary epsilon for synthetic patient records.",
        effective_creation_date=datetime.date(2021, 5, 1),
        domain="Differential Privacy",
        case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
    )

    opp_gnn_squash = CandidateOpportunity(
        opportunity_id="opp-gnn-over-squashing",
        title="Over-squashing in message passing neural networks limits molecular property prediction for large polymers",
        problem_statement="Standard GNN message passing compresses exponential neighborhood information into fixed-size node vectors, losing long-range atomic interactions.",
        description="Design graph rewiring techniques based on Ricci curvature to eliminate bottleneck edges in molecular graphs.",
        effective_creation_date=datetime.date(2021, 8, 1),
        domain="Molecular Biology / GNN",
        case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
    )

    # ── 3. Distractor Opportunities (Created 2019-2022) ────────────────────────
    distractors: List[CandidateOpportunity] = [
        CandidateOpportunity(
            opportunity_id=f"distractor-{i:03d}",
            title=f"Open problem in domain {i}: {topic}",
            problem_statement=f"Documented limitation regarding {topic.lower()} in baseline architectures.",
            description=f"Detailed investigation needed for {topic.lower()}.",
            effective_creation_date=datetime.date(2019, 1, 1) + datetime.timedelta(days=i * 50),
            domain="Physical Sciences & Engineering",
            case_type=BenchmarkCaseType.SYNTHETIC_STRESS_TEST,
        )
        for i, topic in enumerate([
            "Turbulent fluid dynamics Navier-Stokes approximation",
            "Lithium-ion solid-state electrolyte dendrite formation",
            "Perovskite solar cell moisture degradation pathways",
            "Cryogenic CMOS quantum readout circuit thermal dissipation",
            "CRISPR-Cas9 off-target cleavage in polyploid crop genomes",
            "Superconducting qubit flux noise mitigation at millikelvin temperatures",
            "High-entropy alloy fatigue failure under cyclic cryogenic stress",
            "Direct air capture metal-organic framework regeneration enthalpy",
            "Neuromorphic memristor crossbar parasitic wire resistance",
            "Photonic crystal waveguide bending loss at telecom wavelengths",
            "Marine microplastic biodegradation via engineered bacterial consortia",
            "Gallium nitride power semiconductor avalanche breakdown reliability",
            "Radio interferometric calibration ionospheric phase disruption",
            "Anaerobic membrane bioreactor fouling under psychrophilic conditions",
            "Supercritical carbon dioxide turbine aerodynamic loss reduction",
            "Thermonuclear plasma divertor heat flux dispersion in stellarators",
            "Cellular cryo-electron tomography fiducial marker alignment accuracy",
            "Solid oxide fuel cell cathode chromium poisoning kinetics",
            "Deep-sea sediment geotechnical stability under methane hydrate dissociation",
            "Thermophotovoltaic emitter thermal emission spectral shaping",
        ], start=1)
    ]

    # ── 4. Dedicated Synthetic Stress-Test Opportunity (Created Late 2022) ────
    opp_syn_adapters = CandidateOpportunity(
        opportunity_id="opp-syn-adapters-001",
        title="Modular memory adapter interference in parameter-efficient model editing",
        problem_statement=(
            "Parameter-efficient adapters experience catastrophic interference and knowledge degradation "
            "when sequentially integrated into large pre-trained language models for specialized tasks."
        ),
        description=(
            "Design orthogonal modular adapter projection layers that prevent cross-task interference "
            "during sequential fine-tuning."
        ),
        effective_creation_date=datetime.date(2022, 12, 1),
        domain="Synthetic Stress Testing",
        case_type=BenchmarkCaseType.SYNTHETIC_STRESS_TEST,
    )

    candidate_pool = (
        [opp_rag_knowledge, opp_rag_distraction, opp_rag_citations, opp_rag_adaptive]
        + [opp_fl_dropout, opp_fl_dp, opp_fasting, opp_dp_clinical, opp_gnn_squash]
        + distractors
        + [opp_syn_adapters]
    )

    # ── 5. Query Papers (Authentic Literature, Human-Curated Pilot, Synthetic) ─
    queries: List[PaperQuery] = [
        # ══════════════════════════════════════════════════════════════════════
        #  AUTHENTIC LITERATURE PARTITION (5-Paper Corpus + Controls)
        # ══════════════════════════════════════════════════════════════════════
        PaperQuery(
            query_id="AUTH-RAG-01",
            paper_id="paper-lewis-2020",
            title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            abstract=(
                "We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG) — "
                "models which combine pre-trained parametric and non-parametric memory for language generation. "
                "We introduce RAG models where the parametric memory is a pre-trained seq2seq model and the "
                "non-parametric memory is a dense vector index of Wikipedia, accessed with a neural retriever."
            ),
            publication_date=datetime.date(2020, 5, 22),
            case_type=BenchmarkCaseType.AUTHENTIC_LITERATURE,
            relevant_opportunity_ids={"opp-rag-knowledge-limits"},
            hard_negative_opportunity_ids={"opp-rag-citations-attribution"},
            provenance_records=[
                OpportunityRelationshipProvenance(
                    opportunity_id="opp-rag-knowledge-limits",
                    expected_relationship="ADDRESSES",
                    is_relevant=True,
                    source_reference="experiments/data/five_paper_rag_corpus/lewis2020_rag.pdf",
                    rationale="Introduces original RAG architecture directly mitigating parametric knowledge manipulation limits.",
                ),
                OpportunityRelationshipProvenance(
                    opportunity_id="opp-rag-citations-attribution",
                    expected_relationship="HARD_NEGATIVE",
                    is_relevant=False,
                    source_reference="experiments/data/five_paper_rag_corpus/lewis2020_rag.pdf",
                    rationale="Evaluates open-domain QA generation, but does not address or evaluate sentence-level citation attribution.",
                ),
            ],
            doi="10.48550/arXiv.2005.11401",
            notes="Lewis et al. (2020) foundational RAG literature case.",
        ),
        PaperQuery(
            query_id="AUTH-RAG-02",
            paper_id="paper-shi-2023",
            title="Large Language Models Can Be Easily Distracted by Irrelevant Context",
            abstract=(
                "In this work, we investigate the distractibility of large language models, "
                "i.e., how the model problem-solving accuracy can be influenced by irrelevant context. "
                "We introduce Grade-School Math with Irrelevant Context (GSM-IC) and find that model "
                "performance is dramatically decreased when irrelevant information is included."
            ),
            publication_date=datetime.date(2023, 1, 31),
            case_type=BenchmarkCaseType.AUTHENTIC_LITERATURE,
            relevant_opportunity_ids={"opp-rag-distraction-robustness"},
            hard_negative_opportunity_ids={"opp-rag-citations-attribution"},
            provenance_records=[
                OpportunityRelationshipProvenance(
                    opportunity_id="opp-rag-distraction-robustness",
                    expected_relationship="SUPPORTS",
                    is_relevant=True,
                    source_reference="experiments/data/five_paper_rag_corpus/shi2023_distracted_context.pdf",
                    rationale="Proves empirical reasoning degradation caused by irrelevant context, establishing the distraction gap.",
                ),
                OpportunityRelationshipProvenance(
                    opportunity_id="opp-rag-citations-attribution",
                    expected_relationship="HARD_NEGATIVE",
                    is_relevant=False,
                    source_reference="experiments/data/five_paper_rag_corpus/shi2023_distracted_context.pdf",
                    rationale="Investigates mathematical reasoning under irrelevant context, entirely separate from citation verification.",
                ),
            ],
            doi="10.48550/arXiv.2302.00093",
            notes="Shi et al. (2023) supports context distraction gap.",
        ),
        PaperQuery(
            query_id="AUTH-RAG-03",
            paper_id="paper-yoran-2023",
            title="Making Retrieval-Augmented Language Models Robust to Irrelevant Context",
            abstract=(
                "Retrieval-augmented language models incorporate retrieved documents into context to enhance task performance. "
                "However, retrieved context often contains irrelevant passages that cause models to generate erroneous predictions. "
                "We propose robust training methods that explicitly teach language models to disregard irrelevant retrieved text."
            ),
            publication_date=datetime.date(2023, 10, 2),
            case_type=BenchmarkCaseType.AUTHENTIC_LITERATURE,
            relevant_opportunity_ids={"opp-rag-distraction-robustness"},
            hard_negative_opportunity_ids={"opp-rag-citations-attribution"},
            provenance_records=[
                OpportunityRelationshipProvenance(
                    opportunity_id="opp-rag-distraction-robustness",
                    expected_relationship="ADDRESSES",
                    is_relevant=True,
                    source_reference="experiments/data/five_paper_rag_corpus/yoran2023_robust_ralm.pdf",
                    rationale="Directly proposes robust retrieval-augmented language models resilient to distractor documents.",
                ),
                OpportunityRelationshipProvenance(
                    opportunity_id="opp-rag-citations-attribution",
                    expected_relationship="HARD_NEGATIVE",
                    is_relevant=False,
                    source_reference="experiments/data/five_paper_rag_corpus/yoran2023_robust_ralm.pdf",
                    rationale="Focuses on robust training against distractor passages, not on citation generation or sentence attribution.",
                ),
            ],
            doi="10.48550/arXiv.2310.01558",
            notes="Yoran et al. (2023) addresses distractor vulnerability in RALMs.",
        ),
        PaperQuery(
            query_id="AUTH-RAG-04",
            paper_id="paper-asai-2023",
            title="Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
            abstract=(
                "Despite the promise of retrieval-augmented generation, standard models retrieve unconditionally, often "
                "retrieving irrelevant or unnecessary text that impairs generation. We introduce Self-RAG, a framework that "
                "trains a single language model to dynamically retrieve on-demand and critique its own output using special reflection tokens."
            ),
            publication_date=datetime.date(2023, 10, 17),
            case_type=BenchmarkCaseType.AUTHENTIC_LITERATURE,
            relevant_opportunity_ids={"opp-rag-adaptive-retrieval"},
            hard_negative_opportunity_ids={"opp-rag-citations-attribution"},
            provenance_records=[
                OpportunityRelationshipProvenance(
                    opportunity_id="opp-rag-adaptive-retrieval",
                    expected_relationship="ADDRESSES",
                    is_relevant=True,
                    source_reference="experiments/data/five_paper_rag_corpus/asai2023_self_rag.pdf",
                    rationale="Trains arbitrary LM with reflection tokens to adaptively retrieve on-demand and evaluate relevance.",
                ),
                OpportunityRelationshipProvenance(
                    opportunity_id="opp-rag-citations-attribution",
                    expected_relationship="HARD_NEGATIVE",
                    is_relevant=False,
                    source_reference="experiments/data/five_paper_rag_corpus/asai2023_self_rag.pdf",
                    rationale="Targets adaptive retrieval and self-critique, distinct from fine-grained span citation evaluation benchmarks.",
                ),
            ],
            doi="10.48550/arXiv.2310.11511",
            notes="Asai et al. (2023) addresses unconditional retrieval via self-reflection.",
        ),
        PaperQuery(
            query_id="AUTH-RAG-05",
            paper_id="paper-gao-2023",
            title="Enabling Large Language Models to Generate Text with Citations",
            abstract=(
                "We introduce ALCE, the first benchmark for Automatic Language model Citation Evaluation. "
                "ALCE evaluates language models on their ability to generate answers grounded in retrieved documents "
                "with fine-grained sentence-level citations."
            ),
            publication_date=datetime.date(2023, 5, 24),
            case_type=BenchmarkCaseType.AUTHENTIC_LITERATURE,
            relevant_opportunity_ids={"opp-rag-citations-attribution"},
            hard_negative_opportunity_ids={"opp-rag-distraction-robustness"},
            provenance_records=[
                OpportunityRelationshipProvenance(
                    opportunity_id="opp-rag-citations-attribution",
                    expected_relationship="ADDRESSES",
                    is_relevant=True,
                    source_reference="experiments/data/five_paper_rag_corpus/gao2023_citations_alce.pdf",
                    rationale="Establishes citation attribution evaluation benchmark and citation generation models for RAG.",
                ),
                OpportunityRelationshipProvenance(
                    opportunity_id="opp-rag-distraction-robustness",
                    expected_relationship="HARD_NEGATIVE",
                    is_relevant=False,
                    source_reference="experiments/data/five_paper_rag_corpus/gao2023_citations_alce.pdf",
                    rationale="Evaluates citation correctness and attribution precision, not distractor resistance in mathematical reasoning.",
                ),
            ],
            doi="10.18653/v1/2023.emnlp-main.398",
            notes="Gao et al. (2023) addresses sentence-level citation attribution.",
        ),
        PaperQuery(
            query_id="AUTH-CTRL-01",
            paper_id="paper-resnet-control",
            title="Deep Residual Learning for Image Recognition",
            abstract=(
                "Deeper neural networks are more difficult to train. We present a residual learning framework "
                "to ease the training of networks that are substantially deeper than those used previously."
            ),
            publication_date=datetime.date(2016, 6, 1),
            case_type=BenchmarkCaseType.AUTHENTIC_LITERATURE,
            relevant_opportunity_ids=set(),
            provenance_records=[
                OpportunityRelationshipProvenance(
                    opportunity_id="none",
                    expected_relationship="UNRELATED_CONTROL",
                    is_relevant=False,
                    source_reference="CVPR 2016 / Stage 4B HELD-15",
                    rationale="Authentic computer vision literature control with no verified topical link to candidate pool.",
                )
            ],
            notes="Authentic literature negative control.",
        ),
        PaperQuery(
            query_id="AUTH-CTRL-02",
            paper_id="paper-transformer-control",
            title="Attention Is All You Need",
            abstract=(
                "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. "
                "We propose the Transformer, a model architecture eschewing recurrence and relying entirely on an attention mechanism."
            ),
            publication_date=datetime.date(2017, 6, 12),
            case_type=BenchmarkCaseType.AUTHENTIC_LITERATURE,
            relevant_opportunity_ids=set(),
            provenance_records=[
                OpportunityRelationshipProvenance(
                    opportunity_id="none",
                    expected_relationship="UNRELATED_CONTROL",
                    is_relevant=False,
                    source_reference="NeurIPS 2017",
                    rationale="Foundational architecture paper predating RAG/FL application gaps; valid zero-relevance control.",
                )
            ],
            notes="Authentic literature negative control predating opportunities.",
        ),

        # ══════════════════════════════════════════════════════════════════════
        #  HUMAN-CURATED PILOT PARTITION (Vetted Fixtures with Hard Negatives)
        # ══════════════════════════════════════════════════════════════════════
        PaperQuery(
            query_id="PILOT-FAST-01",
            paper_id="paper-fasting-5yr",
            title="A 5-year randomized controlled trial of intermittent fasting in adults over 65",
            abstract=(
                "This study presents the results of a 5-year trial assessing bone mineral density and sarcopenia "
                "in older adults undergoing 5:2 fasting. Bone density decreased slightly but muscle mass was preserved."
            ),
            publication_date=datetime.date(2022, 3, 15),
            case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
            relevant_opportunity_ids={"opp-fasting-geriatrics"},
            provenance_records=[
                OpportunityRelationshipProvenance(
                    opportunity_id="opp-fasting-geriatrics",
                    expected_relationship="ADDRESSES",
                    is_relevant=True,
                    source_reference="classification_fixtures.json:fix_2 / Stage 4B DEV-04",
                    rationale="5-year RCT directly resolves the documented absence of multi-year geriatric fasting outcomes.",
                )
            ],
            notes="Human-curated pilot case directly addressing geriatric fasting gap.",
        ),
        PaperQuery(
            query_id="PILOT-FAST-02",
            paper_id="paper-keto-athletes",
            title="Short-term metabolic impacts of the ketogenic diet in athletes",
            abstract=(
                "We evaluate short-term ketosis in 30 elite college sprinters over a 4-week high-fat regimen, "
                "measuring peak power output and lactate clearance."
            ),
            publication_date=datetime.date(2022, 5, 10),
            case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
            relevant_opportunity_ids=set(),
            hard_negative_opportunity_ids={"opp-fasting-geriatrics"},
            provenance_records=[
                OpportunityRelationshipProvenance(
                    opportunity_id="opp-fasting-geriatrics",
                    expected_relationship="HARD_NEGATIVE",
                    is_relevant=False,
                    source_reference="classification_fixtures.json:fix_15",
                    rationale="Dietary intervention study in athletic youth; hard negative against geriatric longitudinal fasting.",
                )
            ],
            notes="Human-curated pilot dietary hard negative with zero relevant opportunities.",
        ),
        PaperQuery(
            query_id="PILOT-FL-01",
            paper_id="paper-fl-robustagg",
            title="RobustAgg: A novel aggregation scheme for federated learning",
            abstract=(
                "We introduce RobustAgg, an algorithm specifically designed for environments with unreliable clients. "
                "Our protocol maintains convergence even when client dropout reaches up to 75%."
            ),
            publication_date=datetime.date(2022, 4, 12),
            case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
            relevant_opportunity_ids={"opp-fl-dropout-bottlenecks"},
            hard_negative_opportunity_ids={"opp-fl-differential-privacy"},
            provenance_records=[
                OpportunityRelationshipProvenance(
                    opportunity_id="opp-fl-dropout-bottlenecks",
                    expected_relationship="ADDRESSES",
                    is_relevant=True,
                    source_reference="classification_fixtures.json:fix_7 / Stage 4B DEV-07",
                    rationale="Proposes robust aggregation algorithm directly mitigating client dropout bottlenecks up to 75%.",
                ),
                OpportunityRelationshipProvenance(
                    opportunity_id="opp-fl-differential-privacy",
                    expected_relationship="HARD_NEGATIVE",
                    is_relevant=False,
                    source_reference="classification_fixtures.json:fix_7",
                    rationale="Solves network packet dropout and client loss, entirely separate from differential privacy noise mechanics.",
                ),
            ],
            notes="Human-curated pilot FL dropout solution.",
        ),
        PaperQuery(
            query_id="PILOT-FL-02",
            paper_id="paper-fl-dp-guarantees",
            title="Differential privacy guarantees in federated learning",
            abstract=(
                "We evaluate Gaussian noise perturbation under the moments accountant to guarantee strict "
                "participant differential privacy in distributed training across edge devices."
            ),
            publication_date=datetime.date(2022, 6, 1),
            case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
            relevant_opportunity_ids={"opp-fl-differential-privacy"},
            hard_negative_opportunity_ids={"opp-fl-dropout-bottlenecks"},
            provenance_records=[
                OpportunityRelationshipProvenance(
                    opportunity_id="opp-fl-differential-privacy",
                    expected_relationship="ADDRESSES",
                    is_relevant=True,
                    source_reference="classification_fixtures.json:fix_10",
                    rationale="Directly addresses client privacy budget consumption in federated learning architectures.",
                ),
                OpportunityRelationshipProvenance(
                    opportunity_id="opp-fl-dropout-bottlenecks",
                    expected_relationship="HARD_NEGATIVE",
                    is_relevant=False,
                    source_reference="classification_fixtures.json:fix_10",
                    rationale="Explicitly unrelated to client network dropout or connection instability, despite sharing federated learning domain terms.",
                ),
            ],
            notes="Human-curated pilot same-subfield FL hard negative.",
        ),
        PaperQuery(
            query_id="PILOT-DP-01",
            paper_id="paper-dp-clingan",
            title="A novel differentially private GAN achieving epsilon=0.5 with 92% clinical fidelity",
            abstract=(
                "We present DP-ClinGAN, a generative adversarial network that guarantees epsilon=0.5 differential privacy "
                "while achieving 92% utility on cardiovascular clinical trial benchmarks."
            ),
            publication_date=datetime.date(2022, 1, 10),
            case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
            relevant_opportunity_ids={"opp-dp-clinical-trials"},
            provenance_records=[
                OpportunityRelationshipProvenance(
                    opportunity_id="opp-dp-clinical-trials",
                    expected_relationship="ADDRESSES",
                    is_relevant=True,
                    source_reference="stage4b_benchmark.py:HELD-05",
                    rationale="Directly solves high epsilon dilemma by achieving epsilon=0.5 while retaining clinical trial data utility.",
                )
            ],
            notes="Human-curated pilot clinical differential privacy case.",
        ),
        PaperQuery(
            query_id="PILOT-GNN-01",
            paper_id="paper-gnn-curvature",
            title="Curvature-based rewiring eliminates over-squashing in polymer graphs",
            abstract=(
                "We propose Ollivier-Ricci curvature edge rewiring, which directly eliminates graph bottlenecks "
                "and improves polymer property prediction by 14.2%."
            ),
            publication_date=datetime.date(2022, 9, 20),
            case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
            relevant_opportunity_ids={"opp-gnn-over-squashing"},
            provenance_records=[
                OpportunityRelationshipProvenance(
                    opportunity_id="opp-gnn-over-squashing",
                    expected_relationship="ADDRESSES",
                    is_relevant=True,
                    source_reference="stage4b_benchmark.py:HELD-09",
                    rationale="Directly eliminates over-squashing bottleneck in polymer graphs via Ricci curvature rewiring.",
                )
            ],
            notes="Human-curated pilot GNN curvature rewiring case.",
        ),

        # ══════════════════════════════════════════════════════════════════════
        #  SYNTHETIC STRESS TEST PARTITION (Cap-Miss & Zero-Relevance Controls)
        # ══════════════════════════════════════════════════════════════════════
        PaperQuery(
            query_id="SYN-CAP-MISS-01",
            paper_id="paper-syn-cap-miss",
            title="Orthogonal Parameter-Efficient Memory Adapters for Sequential Model Editing",
            abstract=(
                "We mitigate parameter-efficient adapter interference and catastrophic knowledge degradation "
                "through specialized orthogonal modular memory adapters, eliminating cross-task interference in pre-trained models."
            ),
            publication_date=datetime.date(2023, 6, 1),
            case_type=BenchmarkCaseType.SYNTHETIC_STRESS_TEST,
            relevant_opportunity_ids={"opp-syn-adapters-001"},
            provenance_records=[
                OpportunityRelationshipProvenance(
                    opportunity_id="opp-syn-adapters-001",
                    expected_relationship="ADDRESSES",
                    is_relevant=True,
                    source_reference="candidate_recall_benchmark.py:opp_syn_adapters",
                    rationale="Constructed late-created opportunity (pos 31) to strictly test FIFO max_candidates=20 cap truncation.",
                )
            ],
            notes=(
                "Synthetic stress case testing candidate-cap truncation. Target opportunity (opp-syn-adapters-001) "
                "is created late (2022-12-01). Under FIFO with max_candidates=20, older distractors consume the budget, "
                "causing a cap miss. Under TF-IDF, lexical ranking retrieves it at rank 1."
            ),
        ),
        PaperQuery(
            query_id="SYN-EMPTY-REL-01",
            paper_id="paper-syn-empty-relevance",
            title="Synthetic Control Study on Archaeoastronomy Inscriptions",
            abstract="Comparative epigraphic translation of pre-Columbian celestial alignments in stone carvings.",
            publication_date=datetime.date(2023, 1, 1),
            case_type=BenchmarkCaseType.SYNTHETIC_STRESS_TEST,
            relevant_opportunity_ids=set(),
            provenance_records=[
                OpportunityRelationshipProvenance(
                    opportunity_id="none",
                    expected_relationship="UNRELATED_CONTROL",
                    is_relevant=False,
                    source_reference="candidate_recall_benchmark.py",
                    rationale="Synthetic empty-relevance query testing zero-recall non-penalization.",
                )
            ],
            notes="Synthetic empty-relevance query testing non-penalization of zero-relevance papers.",
        ),
    ]

    return candidate_pool, queries


def get_candidate_recall_benchmark_dataset() -> Tuple[List[CandidateOpportunity], List[PaperQuery]]:
    """Default candidate recall dataset (defaults to Stage B dataset)."""
    return get_stage_b_candidate_recall_benchmark_dataset()


# ══════════════════════════════════════════════════════════════════════════════
#  Stage D Independent DEV / HELD_OUT Dataset Construction
# ══════════════════════════════════════════════════════════════════════════════

def get_stage_d_candidate_recall_benchmark_dataset(
    split: BenchmarkSplit,
) -> Tuple[List[CandidateOpportunity], List[PaperQuery]]:
    """Constructs the Stage D dataset strictly partitioned into DEV or HELD_OUT.

    Properties:
    1. Candidate pool (N=31) contains all opportunities against which queries are retrieved:
       4 authentic RAG opportunities, 5 human-curated pilot opportunities, 20 distractors,
       1 adapter cap-miss opportunity, and 1 continual forgetting opportunity.
    2. DEV and HELD_OUT query sets are completely disjoint by query_id and paper_id.
    3. HELD_OUT authentic query (Yoran et al. 2023) evaluates retrieval without having been
       used to author or tune the target opportunity statement (Shi et al. 2023).
    4. HELD_OUT pilot queries (N=11) derive directly from pre-existing unutilized fixtures in
       classification_fixtures.json and stage4b_benchmark.py with genuine ground truth across
       ADDRESSES, SUPPORTS, CONTRADICTS, and EXTENDS.
    5. Pilot queries are explicitly marked with is_simulated_date=True to ensure transparent
       chronological semantics without misrepresenting vignettes as published literature.
    """
    # ── 1. Candidate Opportunity Pool (Shared across both splits) ─────────────
    opp_rag_knowledge = CandidateOpportunity(
        opportunity_id="opp-rag-knowledge-limits",
        title="Mitigating Knowledge Access & Manipulation Limits in RAG Models",
        problem_statement=(
            "Large pre-trained language models have limited ability to access and precisely manipulate knowledge, "
            "underperforming task-specific architectures on knowledge-intensive NLP tasks."
        ),
        description="Investigate modular architectures integrating task-specific knowledge retrieval with language models.",
        effective_creation_date=datetime.date(2020, 5, 1),
        domain="NLP / Retrieval-Augmented Generation",
        case_type=BenchmarkCaseType.AUTHENTIC_LITERATURE,
    )

    opp_rag_distraction = CandidateOpportunity(
        opportunity_id="opp-rag-distraction-robustness",
        title="Vulnerability of Retrieval-Augmented Language Models to Irrelevant Context Distraction",
        problem_statement=(
            "Retrieved documents frequently contain irrelevant context or adversarial noise that dramatically degrades "
            "the reasoning and accuracy of large language models."
        ),
        description=(
            "Develop robust retrieval filters and specialized decoding methods that train models to ignore irrelevant "
            "retrieved context and maintain reasoning fidelity."
        ),
        effective_creation_date=datetime.date(2023, 1, 15),
        domain="NLP / Retrieval-Augmented Generation",
        case_type=BenchmarkCaseType.AUTHENTIC_LITERATURE,
    )

    opp_rag_citations = CandidateOpportunity(
        opportunity_id="opp-rag-citations-attribution",
        title="Verifiable Citation and Statement-Level Attribution Generation in Language Models",
        problem_statement=(
            "Existing retrieval-augmented language models generate unsupported claims and lack verifiable fine-grained "
            "citation attributions linked to specific retrieved source spans."
        ),
        description=(
            "Design benchmark tasks, citation evaluation metrics, and end-to-end models capable of producing explicit "
            "sentence-level citations with high citation precision and recall."
        ),
        effective_creation_date=datetime.date(2023, 5, 1),
        domain="NLP / Retrieval-Augmented Generation",
        case_type=BenchmarkCaseType.AUTHENTIC_LITERATURE,
    )

    opp_rag_adaptive = CandidateOpportunity(
        opportunity_id="opp-rag-adaptive-retrieval",
        title="Selective and Adaptive Self-Reflective Retrieval for Knowledge Generation",
        problem_statement=(
            "Standard RAG models retrieve fixed numbers of passages regardless of whether retrieval is necessary, "
            "introducing unnecessary latency and context distortion on straightforward queries."
        ),
        description=(
            "Train language models with reflection tokens to dynamically critique and decide when to retrieve, "
            "when to skip retrieval, and when to generate critique evaluations."
        ),
        effective_creation_date=datetime.date(2023, 10, 1),
        domain="NLP / Retrieval-Augmented Generation",
        case_type=BenchmarkCaseType.AUTHENTIC_LITERATURE,
    )

    opp_fl_dropout = CandidateOpportunity(
        opportunity_id="opp-fl-dropout-bottlenecks",
        title="Scalability bottlenecks in federated learning over unreliable networks",
        problem_statement="Current FL protocols fail when client dropout exceeds 40% due to loss of edge connectivity.",
        description="Develop robust aggregation protocols and asynchronous client reconciliation for high-dropout mobile environments.",
        effective_creation_date=datetime.date(2021, 3, 1),
        domain="Distributed Systems / Federated Learning",
        case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
    )

    opp_fl_dp = CandidateOpportunity(
        opportunity_id="opp-fl-differential-privacy",
        title="Differential privacy client budget depletion in cross-device federated learning",
        problem_statement=(
            "Repeated local model updates rapidly exhaust client privacy budgets epsilon under standard Gaussian mechanisms, "
            "forcing early participant dropout."
        ),
        description="Develop adaptive client privacy amplification via subsampling and secure aggregation.",
        effective_creation_date=datetime.date(2021, 4, 1),
        domain="Distributed Systems / Federated Learning",
        case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
    )

    opp_fasting = CandidateOpportunity(
        opportunity_id="opp-fasting-geriatrics",
        title="Lack of longitudinal studies on intermittent fasting in older adults",
        problem_statement="Most studies on intermittent fasting focus on short-term outcomes in young populations, leaving geriatric risks unknown.",
        description="Conduct multi-year trials assessing bone mineral density, sarcopenia, and fracture incidence in patients over 65.",
        effective_creation_date=datetime.date(2021, 1, 15),
        domain="Geriatric Medicine",
        case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
    )

    opp_dp_clinical = CandidateOpportunity(
        opportunity_id="opp-dp-clinical-trials",
        title="High privacy budget epsilon needed for high utility in clinical trial synthetic data generation",
        problem_statement="Existing differential privacy mechanisms require epsilon > 5.0 to maintain clinical trial statistical validity, compromising patient confidentiality.",
        description="Develop privacy amplification mechanisms to achieve sub-unitary epsilon for synthetic patient records.",
        effective_creation_date=datetime.date(2021, 5, 1),
        domain="Differential Privacy",
        case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
    )

    opp_gnn_squash = CandidateOpportunity(
        opportunity_id="opp-gnn-over-squashing",
        title="Over-squashing in message passing neural networks limits molecular property prediction for large polymers",
        problem_statement="Standard GNN message passing compresses exponential neighborhood information into fixed-size node vectors, losing long-range atomic interactions.",
        description="Design graph rewiring techniques based on Ricci curvature to eliminate bottleneck edges in molecular graphs.",
        effective_creation_date=datetime.date(2021, 8, 1),
        domain="Molecular Biology / GNN",
        case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
    )

    distractors: List[CandidateOpportunity] = [
        CandidateOpportunity(
            opportunity_id=f"distractor-{i:03d}",
            title=f"Open problem in domain {i}: {topic}",
            problem_statement=f"Documented limitation regarding {topic.lower()} in baseline architectures.",
            description=f"Detailed investigation needed for {topic.lower()}.",
            effective_creation_date=datetime.date(2019, 1, 1) + datetime.timedelta(days=i * 50),
            domain="Physical Sciences & Engineering",
            case_type=BenchmarkCaseType.SYNTHETIC_STRESS_TEST,
        )
        for i, topic in enumerate([
            "Turbulent fluid dynamics Navier-Stokes approximation",
            "Lithium-ion solid-state electrolyte dendrite formation",
            "Perovskite solar cell moisture degradation pathways",
            "Cryogenic CMOS quantum readout circuit thermal dissipation",
            "CRISPR-Cas9 off-target cleavage in polyploid crop genomes",
            "Superconducting qubit flux noise mitigation at millikelvin temperatures",
            "High-entropy alloy fatigue failure under cyclic cryogenic stress",
            "Direct air capture metal-organic framework regeneration enthalpy",
            "Neuromorphic memristor crossbar parasitic wire resistance",
            "Photonic crystal waveguide bending loss at telecom wavelengths",
            "Marine microplastic biodegradation via engineered bacterial consortia",
            "Gallium nitride power semiconductor avalanche breakdown reliability",
            "Radio interferometric calibration ionospheric phase disruption",
            "Anaerobic membrane bioreactor fouling under psychrophilic conditions",
            "Supercritical carbon dioxide turbine aerodynamic loss reduction",
            "Thermonuclear plasma divertor heat flux dispersion in stellarators",
            "Cellular cryo-electron tomography fiducial marker alignment accuracy",
            "Solid oxide fuel cell cathode chromium poisoning kinetics",
            "Deep-sea sediment geotechnical stability under methane hydrate dissociation",
            "Thermophotovoltaic emitter thermal emission spectral shaping",
        ], start=1)
    ]

    opp_syn_adapters = CandidateOpportunity(
        opportunity_id="opp-syn-adapters-001",
        title="Modular memory adapter interference in parameter-efficient model editing",
        problem_statement=(
            "Parameter-efficient adapters experience catastrophic interference and knowledge degradation "
            "when sequentially integrated into large pre-trained language models for specialized tasks."
        ),
        description=(
            "Design orthogonal modular adapter projection layers that prevent cross-task interference "
            "during sequential fine-tuning."
        ),
        effective_creation_date=datetime.date(2022, 12, 1),
        domain="Synthetic Stress Testing",
        case_type=BenchmarkCaseType.SYNTHETIC_STRESS_TEST,
    )

    opp_syn_forgetting = CandidateOpportunity(
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

    candidate_pool = (
        [opp_rag_knowledge, opp_rag_distraction, opp_rag_citations, opp_rag_adaptive]
        + [opp_fl_dropout, opp_fl_dp, opp_fasting, opp_dp_clinical, opp_gnn_squash]
        + distractors
        + [opp_syn_adapters, opp_syn_forgetting]
    )

    # ── 2. Queries partitioned strictly by BenchmarkSplit ─────────────────────
    if split == BenchmarkSplit.DEV:
        queries = [
            PaperQuery(
                query_id="AUTH-RAG-01",
                paper_id="paper-lewis-2020",
                title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                abstract=(
                    "We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG) — "
                    "models which combine pre-trained parametric and non-parametric memory for language generation. "
                    "We introduce RAG models where the parametric memory is a pre-trained seq2seq model and the "
                    "non-parametric memory is a dense vector index of Wikipedia, accessed with a neural retriever."
                ),
                publication_date=datetime.date(2020, 5, 22),
                case_type=BenchmarkCaseType.AUTHENTIC_LITERATURE,
                relevant_opportunity_ids={"opp-rag-knowledge-limits"},
                hard_negative_opportunity_ids={"opp-rag-citations-attribution"},
                provenance_records=[
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-rag-knowledge-limits",
                        expected_relationship="ADDRESSES",
                        is_relevant=True,
                        source_reference="experiments/data/five_paper_rag_corpus/lewis2020_rag.pdf",
                        rationale="Introduces original RAG architecture directly mitigating parametric knowledge manipulation limits.",
                    ),
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-rag-citations-attribution",
                        expected_relationship="HARD_NEGATIVE",
                        is_relevant=False,
                        source_reference="experiments/data/five_paper_rag_corpus/lewis2020_rag.pdf",
                        rationale="Evaluates open-domain QA generation, but does not address or evaluate sentence-level citation attribution.",
                    ),
                ],
                doi="10.48550/arXiv.2005.11401",
                notes="Lewis et al. (2020) foundational RAG literature case in DEV split.",
                is_simulated_date=False,
                split=BenchmarkSplit.DEV,
            ),
            PaperQuery(
                query_id="AUTH-RAG-02",
                paper_id="paper-shi-2023",
                title="Large Language Models Can Be Easily Distracted by Irrelevant Context",
                abstract=(
                    "In this work, we investigate the distractibility of large language models, "
                    "i.e., how the model problem-solving accuracy can be influenced by irrelevant context. "
                    "We introduce Grade-School Math with Irrelevant Context (GSM-IC) and find that model "
                    "performance is dramatically decreased when irrelevant information is included."
                ),
                publication_date=datetime.date(2023, 1, 31),
                case_type=BenchmarkCaseType.AUTHENTIC_LITERATURE,
                relevant_opportunity_ids={"opp-rag-distraction-robustness"},
                hard_negative_opportunity_ids={"opp-rag-citations-attribution"},
                provenance_records=[
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-rag-distraction-robustness",
                        expected_relationship="SUPPORTS",
                        is_relevant=True,
                        source_reference="experiments/data/five_paper_rag_corpus/shi2023_distracted_context.pdf",
                        rationale="Proves empirical reasoning degradation caused by irrelevant context, establishing the distraction gap.",
                    ),
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-rag-citations-attribution",
                        expected_relationship="HARD_NEGATIVE",
                        is_relevant=False,
                        source_reference="experiments/data/five_paper_rag_corpus/shi2023_distracted_context.pdf",
                        rationale="Investigates mathematical reasoning under irrelevant context, entirely separate from citation verification.",
                    ),
                ],
                doi="10.48550/arXiv.2302.00093",
                notes="Shi et al. (2023) supports context distraction gap in DEV split.",
                is_simulated_date=False,
                split=BenchmarkSplit.DEV,
            ),
            PaperQuery(
                query_id="AUTH-RAG-04",
                paper_id="paper-asai-2023",
                title="Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
                abstract=(
                    "Despite the promise of retrieval-augmented generation, standard models retrieve unconditionally, often "
                    "retrieving irrelevant or unnecessary text that impairs generation. We introduce Self-RAG, a framework that "
                    "trains a single language model to dynamically retrieve on-demand and critique its own output using special reflection tokens."
                ),
                publication_date=datetime.date(2023, 10, 17),
                case_type=BenchmarkCaseType.AUTHENTIC_LITERATURE,
                relevant_opportunity_ids={"opp-rag-adaptive-retrieval"},
                hard_negative_opportunity_ids={"opp-rag-citations-attribution"},
                provenance_records=[
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-rag-adaptive-retrieval",
                        expected_relationship="ADDRESSES",
                        is_relevant=True,
                        source_reference="experiments/data/five_paper_rag_corpus/asai2023_self_rag.pdf",
                        rationale="Trains arbitrary LM with reflection tokens to adaptively retrieve on-demand and evaluate relevance.",
                    ),
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-rag-citations-attribution",
                        expected_relationship="HARD_NEGATIVE",
                        is_relevant=False,
                        source_reference="experiments/data/five_paper_rag_corpus/asai2023_self_rag.pdf",
                        rationale="Targets adaptive retrieval and self-critique, distinct from fine-grained span citation evaluation benchmarks.",
                    ),
                ],
                doi="10.48550/arXiv.2310.11511",
                notes="Asai et al. (2023) addresses unconditional retrieval via self-reflection in DEV split.",
                is_simulated_date=False,
                split=BenchmarkSplit.DEV,
            ),
            PaperQuery(
                query_id="AUTH-RAG-05",
                paper_id="paper-gao-2023",
                title="Enabling Large Language Models to Generate Text with Citations",
                abstract=(
                    "We introduce ALCE, the first benchmark for Automatic Language model Citation Evaluation. "
                    "ALCE evaluates language models on their ability to generate answers grounded in retrieved documents "
                    "with fine-grained sentence-level citations."
                ),
                publication_date=datetime.date(2023, 5, 24),
                case_type=BenchmarkCaseType.AUTHENTIC_LITERATURE,
                relevant_opportunity_ids={"opp-rag-citations-attribution"},
                hard_negative_opportunity_ids={"opp-rag-distraction-robustness"},
                provenance_records=[
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-rag-citations-attribution",
                        expected_relationship="ADDRESSES",
                        is_relevant=True,
                        source_reference="experiments/data/five_paper_rag_corpus/gao2023_citations_alce.pdf",
                        rationale="Establishes citation attribution evaluation benchmark and citation generation models for RAG.",
                    ),
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-rag-distraction-robustness",
                        expected_relationship="HARD_NEGATIVE",
                        is_relevant=False,
                        source_reference="experiments/data/five_paper_rag_corpus/gao2023_citations_alce.pdf",
                        rationale="Evaluates citation correctness and attribution precision, not distractor resistance in mathematical reasoning.",
                    ),
                ],
                doi="10.18653/v1/2023.emnlp-main.398",
                notes="Gao et al. (2023) addresses sentence-level citation attribution in DEV split.",
                is_simulated_date=False,
                split=BenchmarkSplit.DEV,
            ),
            PaperQuery(
                query_id="AUTH-CTRL-01",
                paper_id="paper-resnet-control",
                title="Deep Residual Learning for Image Recognition",
                abstract=(
                    "Deeper neural networks are more difficult to train. We present a residual learning framework "
                    "to ease the training of networks that are substantially deeper than those used previously."
                ),
                publication_date=datetime.date(2016, 6, 1),
                case_type=BenchmarkCaseType.AUTHENTIC_LITERATURE,
                relevant_opportunity_ids=set(),
                provenance_records=[
                    OpportunityRelationshipProvenance(
                        opportunity_id="none",
                        expected_relationship="UNRELATED_CONTROL",
                        is_relevant=False,
                        source_reference="CVPR 2016 / Stage 4B HELD-15",
                        rationale="Authentic computer vision literature control with no verified topical link to candidate pool.",
                    )
                ],
                notes="Authentic literature negative control in DEV split.",
                is_simulated_date=False,
                split=BenchmarkSplit.DEV,
            ),
            PaperQuery(
                query_id="PILOT-FAST-01",
                paper_id="paper-fasting-5yr",
                title="A 5-year randomized controlled trial of intermittent fasting in adults over 65",
                abstract=(
                    "This study presents the results of a 5-year trial assessing bone mineral density and sarcopenia "
                    "in older adults undergoing 5:2 fasting. Bone density decreased slightly but muscle mass was preserved."
                ),
                publication_date=datetime.date(2022, 3, 15),
                case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
                relevant_opportunity_ids={"opp-fasting-geriatrics"},
                provenance_records=[
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-fasting-geriatrics",
                        expected_relationship="ADDRESSES",
                        is_relevant=True,
                        source_reference="classification_fixtures.json:fix_2 / Stage 4B DEV-04",
                        rationale="5-year RCT directly resolves the documented absence of multi-year geriatric fasting outcomes.",
                    )
                ],
                notes="Human-curated pilot case directly addressing geriatric fasting gap in DEV split. Simulated benchmark date.",
                is_simulated_date=True,
                split=BenchmarkSplit.DEV,
            ),
            PaperQuery(
                query_id="PILOT-FAST-02",
                paper_id="paper-keto-athletes",
                title="Short-term metabolic impacts of the ketogenic diet in athletes",
                abstract=(
                    "We evaluate short-term ketosis in 30 elite college sprinters over a 4-week high-fat regimen, "
                    "measuring peak power output and lactate clearance."
                ),
                publication_date=datetime.date(2022, 5, 10),
                case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
                relevant_opportunity_ids=set(),
                hard_negative_opportunity_ids={"opp-fasting-geriatrics"},
                provenance_records=[
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-fasting-geriatrics",
                        expected_relationship="HARD_NEGATIVE",
                        is_relevant=False,
                        source_reference="classification_fixtures.json:fix_15",
                        rationale="Dietary intervention study in athletic youth; hard negative against geriatric longitudinal fasting.",
                    )
                ],
                notes="Human-curated pilot dietary hard negative with zero relevant opportunities in DEV split. Simulated benchmark date.",
                is_simulated_date=True,
                split=BenchmarkSplit.DEV,
            ),
            PaperQuery(
                query_id="PILOT-FL-01",
                paper_id="paper-fl-robustagg",
                title="RobustAgg: A novel aggregation scheme for federated learning",
                abstract=(
                    "We introduce RobustAgg, an algorithm specifically designed for environments with unreliable clients. "
                    "Our protocol maintains convergence even when client dropout reaches up to 75%."
                ),
                publication_date=datetime.date(2022, 4, 12),
                case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
                relevant_opportunity_ids={"opp-fl-dropout-bottlenecks"},
                hard_negative_opportunity_ids={"opp-fl-differential-privacy"},
                provenance_records=[
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-fl-dropout-bottlenecks",
                        expected_relationship="ADDRESSES",
                        is_relevant=True,
                        source_reference="classification_fixtures.json:fix_7 / Stage 4B DEV-07",
                        rationale="Proposes robust aggregation algorithm directly mitigating client dropout bottlenecks up to 75%.",
                    ),
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-fl-differential-privacy",
                        expected_relationship="HARD_NEGATIVE",
                        is_relevant=False,
                        source_reference="classification_fixtures.json:fix_7",
                        rationale="Solves network packet dropout and client loss, entirely separate from differential privacy noise mechanics.",
                    ),
                ],
                notes="Human-curated pilot FL dropout solution in DEV split. Simulated benchmark date.",
                is_simulated_date=True,
                split=BenchmarkSplit.DEV,
            ),
            PaperQuery(
                query_id="PILOT-FL-02",
                paper_id="paper-fl-dp-guarantees",
                title="Differential privacy guarantees in federated learning",
                abstract=(
                    "We evaluate Gaussian noise perturbation under the moments accountant to guarantee strict "
                    "participant differential privacy in distributed training across edge devices."
                ),
                publication_date=datetime.date(2022, 6, 1),
                case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
                relevant_opportunity_ids={"opp-fl-differential-privacy"},
                hard_negative_opportunity_ids={"opp-fl-dropout-bottlenecks"},
                provenance_records=[
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-fl-differential-privacy",
                        expected_relationship="ADDRESSES",
                        is_relevant=True,
                        source_reference="classification_fixtures.json:fix_10",
                        rationale="Directly addresses client privacy budget consumption in federated learning architectures.",
                    ),
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-fl-dropout-bottlenecks",
                        expected_relationship="HARD_NEGATIVE",
                        is_relevant=False,
                        source_reference="classification_fixtures.json:fix_10",
                        rationale="Explicitly unrelated to client network dropout or connection instability, despite sharing federated learning domain terms.",
                    ),
                ],
                notes="Human-curated pilot same-subfield FL hard negative in DEV split. Simulated benchmark date.",
                is_simulated_date=True,
                split=BenchmarkSplit.DEV,
            ),
            PaperQuery(
                query_id="PILOT-DP-01",
                paper_id="paper-dp-clingan",
                title="A novel differentially private GAN achieving epsilon=0.5 with 92% clinical fidelity",
                abstract=(
                    "We present DP-ClinGAN, a generative adversarial network that guarantees epsilon=0.5 differential privacy "
                    "while achieving 92% utility on cardiovascular clinical trial benchmarks."
                ),
                publication_date=datetime.date(2022, 1, 10),
                case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
                relevant_opportunity_ids={"opp-dp-clinical-trials"},
                provenance_records=[
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-dp-clinical-trials",
                        expected_relationship="ADDRESSES",
                        is_relevant=True,
                        source_reference="stage4b_benchmark.py:HELD-05",
                        rationale="Directly solves high epsilon dilemma by achieving epsilon=0.5 while retaining clinical trial data utility.",
                    )
                ],
                notes="Human-curated pilot clinical differential privacy case in DEV split. Simulated benchmark date.",
                is_simulated_date=True,
                split=BenchmarkSplit.DEV,
            ),
            PaperQuery(
                query_id="PILOT-GNN-01",
                paper_id="paper-gnn-curvature",
                title="Curvature-based rewiring eliminates over-squashing in polymer graphs",
                abstract=(
                    "We propose Ollivier-Ricci curvature edge rewiring, which directly eliminates graph bottlenecks "
                    "and improves polymer property prediction by 14.2%."
                ),
                publication_date=datetime.date(2022, 9, 20),
                case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
                relevant_opportunity_ids={"opp-gnn-over-squashing"},
                provenance_records=[
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-gnn-over-squashing",
                        expected_relationship="ADDRESSES",
                        is_relevant=True,
                        source_reference="stage4b_benchmark.py:HELD-09",
                        rationale="Directly eliminates over-squashing bottleneck in polymer graphs via Ricci curvature rewiring.",
                    )
                ],
                notes="Human-curated pilot GNN curvature rewiring case in DEV split. Simulated benchmark date.",
                is_simulated_date=True,
                split=BenchmarkSplit.DEV,
            ),
            PaperQuery(
                query_id="SYN-CAP-MISS-01",
                paper_id="paper-syn-cap-miss",
                title="Orthogonal Parameter-Efficient Memory Adapters for Sequential Model Editing",
                abstract=(
                    "We mitigate parameter-efficient adapter interference and catastrophic knowledge degradation "
                    "through specialized orthogonal modular memory adapters, eliminating cross-task interference in pre-trained models."
                ),
                publication_date=datetime.date(2023, 6, 1),
                case_type=BenchmarkCaseType.SYNTHETIC_STRESS_TEST,
                relevant_opportunity_ids={"opp-syn-adapters-001"},
                provenance_records=[
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-syn-adapters-001",
                        expected_relationship="ADDRESSES",
                        is_relevant=True,
                        source_reference="candidate_recall_benchmark.py:opp_syn_adapters",
                        rationale="Constructed late-created opportunity (pos 31) to strictly test FIFO max_candidates=20 cap truncation.",
                    )
                ],
                notes="Synthetic stress case testing candidate-cap truncation in DEV split. Simulated benchmark date.",
                is_simulated_date=True,
                split=BenchmarkSplit.DEV,
            ),
            PaperQuery(
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
                        rationale="Synthetic stress case testing vocabulary mismatch. Expresses catastrophic forgetting without exact keywords.",
                    )
                ],
                notes="Synthetic vocabulary mismatch query in DEV split. Simulated benchmark date.",
                is_simulated_date=True,
                split=BenchmarkSplit.DEV,
            ),
        ]
    elif split == BenchmarkSplit.HELD_OUT:
        queries = [
            # ── 1. Authentic Literature (Downstream & Independent) ─────────────
            PaperQuery(
                query_id="HELD-AUTH-01",
                paper_id="paper-yoran-2023",
                title="Making Retrieval-Augmented Language Models Robust to Irrelevant Context",
                abstract=(
                    "Retrieval-augmented language models incorporate retrieved documents into context to enhance task performance. "
                    "However, retrieved context often contains irrelevant passages that cause models to generate erroneous predictions. "
                    "We propose robust training methods that explicitly teach language models to disregard irrelevant retrieved text."
                ),
                publication_date=datetime.date(2023, 10, 2),
                case_type=BenchmarkCaseType.AUTHENTIC_LITERATURE,
                relevant_opportunity_ids={"opp-rag-distraction-robustness"},
                hard_negative_opportunity_ids={"opp-rag-citations-attribution"},
                provenance_records=[
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-rag-distraction-robustness",
                        expected_relationship="ADDRESSES",
                        is_relevant=True,
                        source_reference="experiments/data/five_paper_rag_corpus/yoran2023_robust_ralm.pdf",
                        rationale="Directly proposes robust retrieval-augmented language models resilient to distractor documents.",
                    ),
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-rag-citations-attribution",
                        expected_relationship="HARD_NEGATIVE",
                        is_relevant=False,
                        source_reference="experiments/data/five_paper_rag_corpus/yoran2023_robust_ralm.pdf",
                        rationale="Focuses on robust training against distractor passages, not on citation generation or sentence attribution.",
                    ),
                ],
                doi="10.48550/arXiv.2310.01558",
                notes="Yoran et al. (2023) held-out authentic query addressing distractor vulnerability in RALMs.",
                is_simulated_date=False,
                split=BenchmarkSplit.HELD_OUT,
            ),
            PaperQuery(
                query_id="HELD-AUTH-CTRL-02",
                paper_id="paper-transformer-control",
                title="Attention Is All You Need",
                abstract=(
                    "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. "
                    "We propose the Transformer, a model architecture eschewing recurrence and relying entirely on an attention mechanism."
                ),
                publication_date=datetime.date(2017, 6, 12),
                case_type=BenchmarkCaseType.AUTHENTIC_LITERATURE,
                relevant_opportunity_ids=set(),
                provenance_records=[
                    OpportunityRelationshipProvenance(
                        opportunity_id="none",
                        expected_relationship="UNRELATED_CONTROL",
                        is_relevant=False,
                        source_reference="NeurIPS 2017",
                        rationale="Foundational architecture paper predating RAG/FL application gaps; valid zero-relevance control.",
                    )
                ],
                doi="10.48550/arXiv.1706.03762",
                notes="Authentic literature negative control predating opportunities in HELD_OUT split.",
                is_simulated_date=False,
                split=BenchmarkSplit.HELD_OUT,
            ),

            # ── 2. Human-Curated Pilot Cases (Vetted Unused Fixtures) ───────────
            PaperQuery(
                query_id="HELD-PILOT-01",
                paper_id="paper-fasting-10yr",
                title="A decade of intermittent fasting: effects on skeletal health in postmenopausal women",
                abstract=(
                    "We followed 200 postmenopausal women (ages 60-75) practicing 16:8 fasting for 10 years. "
                    "DXA scans revealed accelerated bone mineral density loss compared to age-matched controls."
                ),
                publication_date=datetime.date(2022, 8, 15),
                case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
                relevant_opportunity_ids={"opp-fasting-geriatrics"},
                hard_negative_opportunity_ids={"opp-fl-dropout-bottlenecks"},
                provenance_records=[
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-fasting-geriatrics",
                        expected_relationship="ADDRESSES",
                        is_relevant=True,
                        source_reference="classification_fixtures.json:fix_17",
                        rationale="10-year observational study tracking bone mineral density in women over 60, directly addressing geriatric fasting gap.",
                    ),
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-fl-dropout-bottlenecks",
                        expected_relationship="UNRELATED",
                        is_relevant=False,
                        source_reference="classification_fixtures.json:fix_17",
                        rationale="Clinical geriatric nutrition study entirely unrelated to federated edge network dropouts.",
                    ),
                ],
                notes="Human-curated pilot held-out query (fix_17) directly addressing geriatric fasting gap. Simulated benchmark date.",
                is_simulated_date=True,
                split=BenchmarkSplit.HELD_OUT,
            ),
            PaperQuery(
                query_id="HELD-PILOT-02",
                paper_id="paper-fasting-meta",
                title="A meta-analysis of decades of intermittent fasting research in geriatric populations",
                abstract=(
                    "Contrary to popular belief, our comprehensive review found 42 distinct longitudinal trials "
                    "lasting 5-10 years focusing entirely on bone and muscle health in seniors undergoing intermittent fasting."
                ),
                publication_date=datetime.date(2022, 9, 1),
                case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
                relevant_opportunity_ids={"opp-fasting-geriatrics"},
                hard_negative_opportunity_ids={"opp-fl-dropout-bottlenecks"},
                provenance_records=[
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-fasting-geriatrics",
                        expected_relationship="CONTRADICTS",
                        is_relevant=True,
                        source_reference="classification_fixtures.json:fix_3",
                        rationale="Finds 42 distinct longitudinal trials in seniors, directly contradicting the premise that such studies are lacking.",
                    ),
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-fl-dropout-bottlenecks",
                        expected_relationship="UNRELATED",
                        is_relevant=False,
                        source_reference="classification_fixtures.json:fix_3",
                        rationale="Geriatric medical review unrelated to distributed systems.",
                    ),
                ],
                notes="Human-curated pilot held-out query (fix_3) contradicting geriatric fasting premise. Simulated benchmark date.",
                is_simulated_date=True,
                split=BenchmarkSplit.HELD_OUT,
            ),
            PaperQuery(
                query_id="HELD-PILOT-03",
                paper_id="paper-fasting-alzheimer",
                title="Long-term effects of intermittent fasting on cognitive decline in Alzheimer's models",
                abstract=(
                    "We explore a new dimension of long-term fasting by focusing entirely on cognitive health, "
                    "conducting a 4-year study on intermittent fasting's effect on dementia progression in older adults."
                ),
                publication_date=datetime.date(2022, 10, 10),
                case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
                relevant_opportunity_ids={"opp-fasting-geriatrics"},
                hard_negative_opportunity_ids={"opp-fl-dropout-bottlenecks"},
                provenance_records=[
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-fasting-geriatrics",
                        expected_relationship="EXTENDS",
                        is_relevant=True,
                        source_reference="classification_fixtures.json:fix_4",
                        rationale="Extends geriatric fasting investigation from bone/muscle mass to cognitive decline and dementia progression.",
                    ),
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-fl-dropout-bottlenecks",
                        expected_relationship="UNRELATED",
                        is_relevant=False,
                        source_reference="classification_fixtures.json:fix_4",
                        rationale="Medical Alzheimer study unrelated to distributed systems.",
                    ),
                ],
                notes="Human-curated pilot held-out query (fix_4) extending geriatric fasting domain. Simulated benchmark date.",
                is_simulated_date=True,
                split=BenchmarkSplit.HELD_OUT,
            ),
            PaperQuery(
                query_id="HELD-PILOT-04",
                paper_id="paper-fasting-mice",
                title="Lifespan fasting in mice exacerbates age-related bone loss",
                abstract=(
                    "Mice subjected to lifelong intermittent fasting showed severe osteoporosis in old age. "
                    "These alarming preclinical findings underscore the critical missing data regarding long-term fasting safety in human geriatric populations."
                ),
                publication_date=datetime.date(2022, 11, 5),
                case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
                relevant_opportunity_ids={"opp-fasting-geriatrics"},
                hard_negative_opportunity_ids={"opp-fl-dropout-bottlenecks"},
                provenance_records=[
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-fasting-geriatrics",
                        expected_relationship="SUPPORTS",
                        is_relevant=True,
                        source_reference="classification_fixtures.json:fix_16",
                        rationale="Preclinical animal model confirms alarming bone loss under lifelong fasting, supporting the urgent need for human geriatric data.",
                    ),
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-fl-dropout-bottlenecks",
                        expected_relationship="UNRELATED",
                        is_relevant=False,
                        source_reference="classification_fixtures.json:fix_16",
                        rationale="Preclinical bone density study unrelated to distributed systems.",
                    ),
                ],
                notes="Human-curated pilot held-out query (fix_16) supporting geriatric fasting gap. Simulated benchmark date.",
                is_simulated_date=True,
                split=BenchmarkSplit.HELD_OUT,
            ),
            PaperQuery(
                query_id="HELD-PILOT-05",
                paper_id="paper-fl-droptolerant",
                title="DropTolerant FL: Surviving extreme network partitions",
                abstract=(
                    "We present DropTolerant FL, an aggregation algorithm designed specifically for scenarios with "
                    "up to 80% client dropout in unreliable networks. It ensures consistent model updates despite extreme packet loss."
                ),
                publication_date=datetime.date(2022, 7, 20),
                case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
                relevant_opportunity_ids={"opp-fl-dropout-bottlenecks"},
                hard_negative_opportunity_ids={"opp-fl-differential-privacy"},
                provenance_records=[
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-fl-dropout-bottlenecks",
                        expected_relationship="ADDRESSES",
                        is_relevant=True,
                        source_reference="classification_fixtures.json:fix_21",
                        rationale="Introduces DropTolerant FL maintaining updates under 80% client dropout, directly addressing the bottleneck.",
                    ),
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-fl-differential-privacy",
                        expected_relationship="HARD_NEGATIVE",
                        is_relevant=False,
                        source_reference="classification_fixtures.json:fix_21",
                        rationale="Targets network packet loss and client dropout, entirely distinct from client differential privacy budget depletion.",
                    ),
                ],
                notes="Human-curated pilot held-out query (fix_21) addressing FL dropout bottlenecks. Simulated benchmark date.",
                is_simulated_date=True,
                split=BenchmarkSplit.HELD_OUT,
            ),
            PaperQuery(
                query_id="HELD-PILOT-06",
                paper_id="paper-fl-intermittent-avg",
                title="Empirical analysis of FedAvg on intermittent networks",
                abstract=(
                    "In our tests, standard FedAvg stopped converging entirely once network instability caused 45% of nodes "
                    "to drop out mid-round. This empirically validates the urgent need for new fault-tolerant approaches."
                ),
                publication_date=datetime.date(2022, 8, 1),
                case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
                relevant_opportunity_ids={"opp-fl-dropout-bottlenecks"},
                hard_negative_opportunity_ids={"opp-fl-differential-privacy"},
                provenance_records=[
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-fl-dropout-bottlenecks",
                        expected_relationship="SUPPORTS",
                        is_relevant=True,
                        source_reference="classification_fixtures.json:fix_11",
                        rationale="Empirically verifies FedAvg convergence failure at 45% node dropout, validating the existence of the bottleneck.",
                    ),
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-fl-differential-privacy",
                        expected_relationship="HARD_NEGATIVE",
                        is_relevant=False,
                        source_reference="classification_fixtures.json:fix_11",
                        rationale="Investigates network dropouts in FedAvg, not differential privacy mechanisms.",
                    ),
                ],
                notes="Human-curated pilot held-out query (fix_11) supporting FL dropout bottleneck. Simulated benchmark date.",
                is_simulated_date=True,
                split=BenchmarkSplit.HELD_OUT,
            ),
            PaperQuery(
                query_id="HELD-PILOT-07",
                paper_id="paper-fl-fedavg-robust",
                title="FedAvg is inherently robust to high dropout",
                abstract=(
                    "Contrary to recent claims, our mathematical proof demonstrates that vanilla FedAvg maintains identical "
                    "convergence bounds even with 99% client dropout, provided the learning rate is scaled correctly."
                ),
                publication_date=datetime.date(2022, 9, 15),
                case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
                relevant_opportunity_ids={"opp-fl-dropout-bottlenecks"},
                hard_negative_opportunity_ids={"opp-fl-differential-privacy"},
                provenance_records=[
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-fl-dropout-bottlenecks",
                        expected_relationship="CONTRADICTS",
                        is_relevant=True,
                        source_reference="classification_fixtures.json:fix_13",
                        rationale="Mathematical proof asserting vanilla FedAvg converges under 99% dropout, contradicting the premise that new protocols are needed.",
                    ),
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-fl-differential-privacy",
                        expected_relationship="HARD_NEGATIVE",
                        is_relevant=False,
                        source_reference="classification_fixtures.json:fix_13",
                        rationale="Theoretical convergence analysis of dropout in FedAvg, unrelated to privacy budget noise.",
                    ),
                ],
                notes="Human-curated pilot held-out query (fix_13) contradicting FL dropout premise. Simulated benchmark date.",
                is_simulated_date=True,
                split=BenchmarkSplit.HELD_OUT,
            ),
            PaperQuery(
                query_id="HELD-PILOT-08",
                paper_id="paper-fl-client-select",
                title="Predictive client selection for volatile federated networks",
                abstract=(
                    "Instead of modifying the aggregation protocol to handle dropped clients, we propose a machine learning model "
                    "that predicts client reliability beforehand. By only selecting stable clients, we circumvent the aggregation bottleneck entirely."
                ),
                publication_date=datetime.date(2022, 10, 25),
                case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
                relevant_opportunity_ids={"opp-fl-dropout-bottlenecks"},
                hard_negative_opportunity_ids={"opp-fl-differential-privacy"},
                provenance_records=[
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-fl-dropout-bottlenecks",
                        expected_relationship="EXTENDS",
                        is_relevant=True,
                        source_reference="classification_fixtures.json:fix_14",
                        rationale="Reframes client unreliability from an aggregation problem to a client selection and reliability prediction problem.",
                    ),
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-fl-differential-privacy",
                        expected_relationship="HARD_NEGATIVE",
                        is_relevant=False,
                        source_reference="classification_fixtures.json:fix_14",
                        rationale="Focuses on predictive node selection, not client privacy mechanisms.",
                    ),
                ],
                notes="Human-curated pilot held-out query (fix_14) extending FL problem focus. Simulated benchmark date.",
                is_simulated_date=True,
                split=BenchmarkSplit.HELD_OUT,
            ),
            PaperQuery(
                query_id="HELD-PILOT-09",
                paper_id="paper-dp-impossibility",
                title="Impossibility results for low-epsilon synthetic tabular data under realistic sample sizes",
                abstract=(
                    "We prove information-theoretically that no mechanism can achieve epsilon < 1.0 with greater than 80% utility "
                    "when sample size N < 50,000, contradicting recent empirical claims."
                ),
                publication_date=datetime.date(2022, 6, 15),
                case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
                relevant_opportunity_ids={"opp-dp-clinical-trials"},
                hard_negative_opportunity_ids={"opp-fl-differential-privacy"},
                provenance_records=[
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-dp-clinical-trials",
                        expected_relationship="CONTRADICTS",
                        is_relevant=True,
                        source_reference="stage4b_benchmark.py:HELD-06",
                        rationale="Information-theoretic proof contradicting the feasibility of high utility sub-unitary epsilon synthetic data generation.",
                    ),
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-fl-differential-privacy",
                        expected_relationship="HARD_NEGATIVE",
                        is_relevant=False,
                        source_reference="stage4b_benchmark.py:HELD-06",
                        rationale="Centralized tabular DP utility bounds, distinct from distributed edge federated learning budget amplification.",
                    ),
                ],
                notes="Human-curated pilot held-out query (HELD-06) contradicting DP synthetic data feasibility. Simulated benchmark date.",
                is_simulated_date=True,
                split=BenchmarkSplit.HELD_OUT,
            ),
            PaperQuery(
                query_id="HELD-PILOT-10",
                paper_id="paper-dp-icu-ecg",
                title="Extending differential privacy to continuous-time vital sign telemetry in ICU monitoring",
                abstract=(
                    "While existing DP tabular methods focus on static trial records, we extend formal differential privacy "
                    "guarantees to continuous streaming ECG waveforms in intensive care telemetry."
                ),
                publication_date=datetime.date(2022, 11, 20),
                case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
                relevant_opportunity_ids={"opp-dp-clinical-trials"},
                hard_negative_opportunity_ids={"opp-fl-differential-privacy"},
                provenance_records=[
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-dp-clinical-trials",
                        expected_relationship="EXTENDS",
                        is_relevant=True,
                        source_reference="stage4b_benchmark.py:HELD-07",
                        rationale="Extends differential privacy from static trial tabular databases to continuous real-time ICU physiological waveforms.",
                    ),
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-fl-differential-privacy",
                        expected_relationship="HARD_NEGATIVE",
                        is_relevant=False,
                        source_reference="stage4b_benchmark.py:HELD-07",
                        rationale="Continuous waveform DP in clinical care, distinct from distributed FL client parameter protection.",
                    ),
                ],
                notes="Human-curated pilot held-out query (HELD-07) extending DP to time series. Simulated benchmark date.",
                is_simulated_date=True,
                split=BenchmarkSplit.HELD_OUT,
            ),
            PaperQuery(
                query_id="HELD-PILOT-11",
                paper_id="paper-gnn-survey",
                title="Survey on GNN architectures in computational chemistry",
                abstract=(
                    "As highlighted in recent literature, information over-squashing at bottleneck nodes prevents standard message passing "
                    "from propagating long-range atomic interactions in macromolecules."
                ),
                publication_date=datetime.date(2022, 8, 10),
                case_type=BenchmarkCaseType.HUMAN_CURATED_PILOT,
                relevant_opportunity_ids={"opp-gnn-over-squashing"},
                hard_negative_opportunity_ids={"opp-fasting-geriatrics"},
                provenance_records=[
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-gnn-over-squashing",
                        expected_relationship="SUPPORTS",
                        is_relevant=True,
                        source_reference="stage4b_benchmark.py:HELD-08",
                        rationale="Survey confirms that over-squashing at bottleneck nodes prevents long-range message passing in macromolecules.",
                    ),
                    OpportunityRelationshipProvenance(
                        opportunity_id="opp-fasting-geriatrics",
                        expected_relationship="UNRELATED",
                        is_relevant=False,
                        source_reference="stage4b_benchmark.py:HELD-08",
                        rationale="GNN architecture survey completely unrelated to geriatric clinical studies.",
                    ),
                ],
                notes="Human-curated pilot held-out query (HELD-08) supporting GNN over-squashing bottleneck. Simulated benchmark date.",
                is_simulated_date=True,
                split=BenchmarkSplit.HELD_OUT,
            ),

            # ── 3. Synthetic Stress-Test Control ──────────────────────────────
            PaperQuery(
                query_id="HELD-SYN-01",
                paper_id="paper-syn-empty-relevance",
                title="Synthetic Control Study on Archaeoastronomy Inscriptions",
                abstract="Comparative epigraphic translation of pre-Columbian celestial alignments in stone carvings.",
                publication_date=datetime.date(2023, 1, 1),
                case_type=BenchmarkCaseType.SYNTHETIC_STRESS_TEST,
                relevant_opportunity_ids=set(),
                provenance_records=[
                    OpportunityRelationshipProvenance(
                        opportunity_id="none",
                        expected_relationship="UNRELATED_CONTROL",
                        is_relevant=False,
                        source_reference="candidate_recall_benchmark.py",
                        rationale="Synthetic empty-relevance query testing zero-recall non-penalization in held-out split.",
                    )
                ],
                notes="Synthetic empty-relevance query testing non-penalization of zero-relevance papers in held-out evaluation. Simulated benchmark date.",
                is_simulated_date=True,
                split=BenchmarkSplit.HELD_OUT,
            ),
        ]
    else:
        raise ValueError(f"Unknown BenchmarkSplit: {split}")

    return candidate_pool, queries



# ══════════════════════════════════════════════════════════════════════════════
#  Benchmark Runner
# ══════════════════════════════════════════════════════════════════════════════

class CandidateRecallBenchmarkRunner:
    """Executes the offline candidate recall benchmark comparing FIFO and TF-IDF."""

    def __init__(
        self,
        candidate_pool: Sequence[CandidateOpportunity],
        queries: Sequence[PaperQuery],
        k_values: Sequence[int] = DEFAULT_K_VALUES,
        max_candidates: Optional[int] = 20,
        cutoff_date: Optional[datetime.date] = None,
    ) -> None:
        self.candidate_pool = list(candidate_pool)
        self.queries = list(queries)
        self.k_values = tuple(k_values)
        self.max_candidates = max_candidates
        self.cutoff_date = cutoff_date

        self.fifo_ranker = ChronologicalFIFORanker()
        self.tfidf_ranker = TFIDFLexicalRanker()

    def run(self) -> Dict[str, Any]:
        """Runs the benchmark across all queries for both strategies and returns structured reports."""
        fifo_outcomes: List[QueryRankingOutcome] = []
        tfidf_outcomes: List[QueryRankingOutcome] = []

        for q in self.queries:
            # 1. Temporal filtering (matching production causality)
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

        # Summarize across partitions
        fifo_overall = summarize_strategy_results("FIFO", fifo_outcomes, None, self.k_values)
        fifo_auth = summarize_strategy_results("FIFO", fifo_outcomes, BenchmarkCaseType.AUTHENTIC_LITERATURE, self.k_values)
        fifo_pilot = summarize_strategy_results("FIFO", fifo_outcomes, BenchmarkCaseType.HUMAN_CURATED_PILOT, self.k_values)
        fifo_syn = summarize_strategy_results("FIFO", fifo_outcomes, BenchmarkCaseType.SYNTHETIC_STRESS_TEST, self.k_values)

        tfidf_overall = summarize_strategy_results("TF-IDF", tfidf_outcomes, None, self.k_values)
        tfidf_auth = summarize_strategy_results("TF-IDF", tfidf_outcomes, BenchmarkCaseType.AUTHENTIC_LITERATURE, self.k_values)
        tfidf_pilot = summarize_strategy_results("TF-IDF", tfidf_outcomes, BenchmarkCaseType.HUMAN_CURATED_PILOT, self.k_values)
        tfidf_syn = summarize_strategy_results("TF-IDF", tfidf_outcomes, BenchmarkCaseType.SYNTHETIC_STRESS_TEST, self.k_values)

        # Assemble EvaluationResult metrics
        metrics: Dict[str, float] = {}
        for k in self.k_values:
            metrics[f"fifo_overall_recall_at_{k}"] = round(fifo_overall.mean_recall_at_k.get(k, 0.0), 4)
            metrics[f"tfidf_overall_recall_at_{k}"] = round(tfidf_overall.mean_recall_at_k.get(k, 0.0), 4)
            metrics[f"fifo_authentic_recall_at_{k}"] = round(fifo_auth.mean_recall_at_k.get(k, 0.0), 4)
            metrics[f"tfidf_authentic_recall_at_{k}"] = round(tfidf_auth.mean_recall_at_k.get(k, 0.0), 4)
            metrics[f"fifo_pilot_recall_at_{k}"] = round(fifo_pilot.mean_recall_at_k.get(k, 0.0), 4)
            metrics[f"tfidf_pilot_recall_at_{k}"] = round(tfidf_pilot.mean_recall_at_k.get(k, 0.0), 4)
            metrics[f"fifo_synthetic_recall_at_{k}"] = round(fifo_syn.mean_recall_at_k.get(k, 0.0), 4)
            metrics[f"tfidf_synthetic_recall_at_{k}"] = round(tfidf_syn.mean_recall_at_k.get(k, 0.0), 4)

        metrics["fifo_overall_cap_miss_rate"] = round(fifo_overall.cap_miss_rate, 4)
        metrics["tfidf_overall_cap_miss_rate"] = round(tfidf_overall.cap_miss_rate, 4)
        metrics["fifo_authentic_cap_miss_rate"] = round(fifo_auth.cap_miss_rate, 4)
        metrics["tfidf_authentic_cap_miss_rate"] = round(tfidf_auth.cap_miss_rate, 4)
        metrics["fifo_pilot_cap_miss_rate"] = round(fifo_pilot.cap_miss_rate, 4)
        metrics["tfidf_pilot_cap_miss_rate"] = round(tfidf_pilot.cap_miss_rate, 4)
        metrics["fifo_synthetic_cap_miss_rate"] = round(fifo_syn.cap_miss_rate, 4)
        metrics["tfidf_synthetic_cap_miss_rate"] = round(tfidf_syn.cap_miss_rate, 4)

        metrics["fifo_overall_opportunity_cap_misses"] = float(fifo_overall.total_cap_misses)
        metrics["tfidf_overall_opportunity_cap_misses"] = float(tfidf_overall.total_cap_misses)
        metrics["fifo_authentic_opportunity_cap_misses"] = float(fifo_auth.total_cap_misses)
        metrics["tfidf_authentic_opportunity_cap_misses"] = float(tfidf_auth.total_cap_misses)
        metrics["fifo_pilot_opportunity_cap_misses"] = float(fifo_pilot.total_cap_misses)
        metrics["tfidf_pilot_opportunity_cap_misses"] = float(tfidf_pilot.total_cap_misses)
        metrics["fifo_synthetic_opportunity_cap_misses"] = float(fifo_syn.total_cap_misses)
        metrics["tfidf_synthetic_opportunity_cap_misses"] = float(tfidf_syn.total_cap_misses)

        denominators = {
            "total_queries": len(self.queries),
            "evaluable_queries_overall": fifo_overall.evaluable_queries,
            "queries_without_relevant_overall": fifo_overall.queries_without_relevant,
            "evaluable_queries_authentic": fifo_auth.evaluable_queries,
            "queries_without_relevant_authentic": fifo_auth.queries_without_relevant,
            "evaluable_queries_pilot": fifo_pilot.evaluable_queries,
            "queries_without_relevant_pilot": fifo_pilot.queries_without_relevant,
            "evaluable_queries_synthetic": fifo_syn.evaluable_queries,
            "queries_without_relevant_synthetic": fifo_syn.queries_without_relevant,
            "candidate_pool_size": len(self.candidate_pool),
        }

        eval_result = EvaluationResult(
            evaluation_name="candidate_recall_stage_b",
            fixture_version="v2.0-source-grounded-hard-negatives",
            evaluation_type=EvaluationType.EXECUTED,
            model_provider="scikit-learn (TfidfVectorizer) / in-memory FIFO",
            parameters={
                "k_values": list(self.k_values),
                "max_candidates": self.max_candidates,
                "cutoff_date": self.cutoff_date.isoformat() if self.cutoff_date else None,
            },
            metrics=metrics,
            denominators=denominators,
            errors=[],
            limitations=(
                "Stage B offline benchmark evaluates chronological FIFO vs. TF-IDF lexical retrieval with source-grounded "
                "same-subfield hard negatives. Dense vector retrieval is deferred to Stage C. The corpus uses authentic "
                "5-paper RAG literature with source PDFs and human-curated pilot cases with same-subfield hard negatives "
                "(e.g. differential privacy in federated learning as a hard negative against network client dropout). "
                "Results characterize candidate-filtering recall under controlled candidate pools, not open-web generalization."
            ),
            is_engineering_validation=False,
        )

        return {
            "evaluation_result": eval_result,
            "fifo_overall": fifo_overall,
            "fifo_authentic": fifo_auth,
            "fifo_pilot": fifo_pilot,
            "fifo_synthetic": fifo_syn,
            "tfidf_overall": tfidf_overall,
            "tfidf_authentic": tfidf_auth,
            "tfidf_pilot": tfidf_pilot,
            "tfidf_synthetic": tfidf_syn,
        }
