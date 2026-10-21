"""Stage 4B Robustness & Credibility Benchmark (v2.0).

Expanded 25-case evaluation benchmark addressing the Stage 4 credibility audit:
1. Explicit Development (DEV, N=10) vs Held-Out (HELD_OUT, N=15) partition.
2. Real scientific literature passages with full provenance hashes and authentic ligatures.
3. Multi-domain coverage (NLP/RAG, Geriatrics, Federated Learning, Differential Privacy, Molecular GNNs).
4. Explicitly flagged borderline cases with alternative classifications and expert-review notes.
5. Stress-testing grounding edge cases: ligatures, paraphrases, cross-paper attributions, hallucinations.
6. Stratified reporting: isolating Real Literature, Human-Curated Pilot, and Synthetic Stress Tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence
import uuid

from ideagen.opportunity.models import OpportunityRelationship, OpportunityStatus
from ideagen.opportunity.grounding import (
    is_excerpt_grounded,
    validate_evidence_grounding,
    UngroundedEvidenceError,
    DEFAULT_GROUNDING_THRESHOLD,
)
from ideagen.opportunity.state_machine import transition
from ideagen.evaluation.metrics import precision_score, recall_score, f1_score


class BenchmarkSplit(str, Enum):
    DEV = "DEV"
    HELD_OUT = "HELD_OUT"


class CaseSourceCategory(str, Enum):
    REAL_LITERATURE = "REAL_LITERATURE"
    HUMAN_CURATED_PILOT = "HUMAN_CURATED_PILOT"
    SYNTHETIC_STRESS_TEST = "SYNTHETIC_STRESS_TEST"


class GroundingChallengeType(str, Enum):
    EXACT_MATCH = "EXACT_MATCH"
    FUZZY_MATCH = "FUZZY_MATCH"
    LIGATURE = "LIGATURE"
    PARAPHRASE_SUBTHRESHOLD = "PARAPHRASE_SUBTHRESHOLD"
    UNSUPPORTED_FABRICATION = "UNSUPPORTED_FABRICATION"
    CROSS_PAPER_ATTRIBUTION = "CROSS_PAPER_ATTRIBUTION"
    AMBIGUOUS_EVIDENCE = "AMBIGUOUS_EVIDENCE"
    METHOD_AND_BENCHMARK = "METHOD_AND_BENCHMARK"
    NONE_REQUIRED = "NONE_REQUIRED"


@dataclass(frozen=True)
class Stage4bBenchmarkCase:
    case_id: str
    benchmark_version: str
    split: BenchmarkSplit
    source_category: CaseSourceCategory
    domain: str
    paper_id: str
    paper_title: str
    opportunity_id: str
    opportunity_title: str
    opportunity_problem: str
    opportunity_direction: str
    passages: List[str]
    claimed_excerpt: Optional[str]
    expected_relationship: OpportunityRelationship
    is_borderline: bool
    borderline_alternative_class: Optional[OpportunityRelationship]
    requires_expert_review: bool
    expert_review_notes: Optional[str]
    grounding_challenge_type: GroundingChallengeType
    expected_grounding_valid: bool
    initial_opportunity_status: OpportunityStatus
    addressing_paper_count: int
    expected_state_machine_status: OpportunityStatus
    expected_end_to_end_status: OpportunityStatus
    rationale: str


def get_stage4b_benchmark_dataset() -> List[Stage4bBenchmarkCase]:
    """Returns the 25-case Stage 4B Benchmark v2.0 dataset."""
    rag_opp_id = "ef562805-e2fc-4eb3-bf0c-c263c80f77b6"
    rag_opp_title = "Mitigating Knowledge Access & Manipulation Limits in RAG Models"
    rag_opp_prob = (
        "Large pre-trained language models have limited ability to access and precisely "
        "manipulate knowledge, underperforming task-specific architectures on knowledge-intensive NLP tasks."
    )
    rag_opp_dir = (
        "Investigate modular architectures integrating task-specific knowledge modules with "
        "large pre-trained language models while preserving generalization capabilities."
    )

    cases: List[Stage4bBenchmarkCase] = [
        # ══════════════════════════════════════════════════════════════════════
        #  DEVELOPMENT SPLIT (DEV, N=10)
        # ══════════════════════════════════════════════════════════════════════
        Stage4bBenchmarkCase(
            case_id="DEV-01",
            benchmark_version="v2.0-robustness",
            split=BenchmarkSplit.DEV,
            source_category=CaseSourceCategory.REAL_LITERATURE,
            domain="NLP / Retrieval Augmentation",
            paper_id="2376a0bd-733d-4307-9173-539313dbfec7",
            paper_title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)",
            opportunity_id=rag_opp_id,
            opportunity_title=rag_opp_title,
            opportunity_problem=rag_opp_prob,
            opportunity_direction=rag_opp_dir,
            passages=[
                "We explore a general-purpose \ufb01ne-tuning recipe for retrieval-augmented generation (RAG) — "
                "models which combine pre-trained parametric and non-parametric memory for language generation. "
                "We introduce RAG models where the parametric memory is a pre-trained seq2seq model and the "
                "non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever."
            ],
            claimed_excerpt="combine pre-trained parametric and non-parametric memory for language generation",
            expected_relationship=OpportunityRelationship.ADDRESSES,
            is_borderline=False,
            borderline_alternative_class=None,
            requires_expert_review=False,
            expert_review_notes=None,
            grounding_challenge_type=GroundingChallengeType.EXACT_MATCH,
            expected_grounding_valid=True,
            initial_opportunity_status=OpportunityStatus.OPEN,
            addressing_paper_count=1,
            expected_state_machine_status=OpportunityStatus.PARTIALLY_ADDRESSED,
            expected_end_to_end_status=OpportunityStatus.PARTIALLY_ADDRESSED,
            rationale="Foundational RAG paper directly addresses the parametric limitation by introducing dense Wikipedia vector retrieval.",
        ),
        Stage4bBenchmarkCase(
            case_id="DEV-02",
            benchmark_version="v2.0-robustness",
            split=BenchmarkSplit.DEV,
            source_category=CaseSourceCategory.REAL_LITERATURE,
            domain="NLP / Retrieval Augmentation",
            paper_id="32e1578f-6749-496d-a391-e33f002335a6",
            paper_title="Large Language Models Can Be Easily Distracted by Irrelevant Context (Shi et al., 2023)",
            opportunity_id=rag_opp_id,
            opportunity_title=rag_opp_title,
            opportunity_problem=rag_opp_prob,
            opportunity_direction=rag_opp_dir,
            passages=[
                "In this work, we investigate the distractibility of large language models, "
                "i.e., how the model problem-solving accuracy can be influenced by irrelevant context. "
                "We introduce Grade-School Math with Irrelevant Context (GSM-IC) and find that the model "
                "performance is dramatically decreased when irrelevant information is included."
            ],
            claimed_excerpt="find that the model performance is dramatically decreased when irrelevant information is included",
            expected_relationship=OpportunityRelationship.SUPPORTS,
            is_borderline=False,
            borderline_alternative_class=None,
            requires_expert_review=False,
            expert_review_notes=None,
            grounding_challenge_type=GroundingChallengeType.EXACT_MATCH,
            expected_grounding_valid=True,
            initial_opportunity_status=OpportunityStatus.OPEN,
            addressing_paper_count=0,
            expected_state_machine_status=OpportunityStatus.STRENGTHENED,
            expected_end_to_end_status=OpportunityStatus.STRENGTHENED,
            rationale="Empirically confirms that irrelevant retrieved passages degrade reasoning, strengthening the importance of the gap without resolving it.",
        ),
        Stage4bBenchmarkCase(
            case_id="DEV-03",
            benchmark_version="v2.0-robustness",
            split=BenchmarkSplit.DEV,
            source_category=CaseSourceCategory.HUMAN_CURATED_PILOT,
            domain="Geriatric Medicine",
            paper_id="pilot-fasting-001",
            paper_title="A 12-month observational study of time-restricted eating in seniors",
            opportunity_id="pilot-opp-fasting",
            opportunity_title="Lack of longitudinal studies on intermittent fasting in older adults",
            opportunity_problem="Most studies on intermittent fasting focus on short-term outcomes in young populations.",
            opportunity_direction="Conduct multi-year trials assessing bone mineral density and sarcopenia in patients over 65.",
            passages=[
                "We observed 50 individuals over age 65 adhering to a 16:8 fasting schedule. "
                "Results showed minor bone density loss over 12 months, highlighting the need for longer-term multi-year follow-ups."
            ],
            claimed_excerpt="highlighting the need for longer-term multi-year follow-ups.",
            expected_relationship=OpportunityRelationship.SUPPORTS,
            is_borderline=False,
            borderline_alternative_class=None,
            requires_expert_review=False,
            expert_review_notes=None,
            grounding_challenge_type=GroundingChallengeType.EXACT_MATCH,
            expected_grounding_valid=True,
            initial_opportunity_status=OpportunityStatus.OPEN,
            addressing_paper_count=0,
            expected_state_machine_status=OpportunityStatus.STRENGTHENED,
            expected_end_to_end_status=OpportunityStatus.STRENGTHENED,
            rationale="Observational 12-month cohort confirms the ongoing necessity for multi-year follow-up trials.",
        ),
        Stage4bBenchmarkCase(
            case_id="DEV-04",
            benchmark_version="v2.0-robustness",
            split=BenchmarkSplit.DEV,
            source_category=CaseSourceCategory.HUMAN_CURATED_PILOT,
            domain="Geriatric Medicine",
            paper_id="pilot-fasting-002",
            paper_title="A 5-year randomized controlled trial of intermittent fasting in adults over 65",
            opportunity_id="pilot-opp-fasting",
            opportunity_title="Lack of longitudinal studies on intermittent fasting in older adults",
            opportunity_problem="Most studies on intermittent fasting focus on short-term outcomes in young populations.",
            opportunity_direction="Conduct multi-year trials assessing bone mineral density and sarcopenia in patients over 65.",
            passages=[
                "This study presents the results of a 5-year trial assessing bone mineral density and sarcopenia "
                "in older adults undergoing 5:2 fasting. Bone density decreased slightly but muscle mass was preserved."
            ],
            claimed_excerpt="results of a 5-year trial assessing bone mineral density and sarcopenia in older adults",
            expected_relationship=OpportunityRelationship.ADDRESSES,
            is_borderline=False,
            borderline_alternative_class=None,
            requires_expert_review=False,
            expert_review_notes=None,
            grounding_challenge_type=GroundingChallengeType.EXACT_MATCH,
            expected_grounding_valid=True,
            initial_opportunity_status=OpportunityStatus.OPEN,
            addressing_paper_count=1,
            expected_state_machine_status=OpportunityStatus.PARTIALLY_ADDRESSED,
            expected_end_to_end_status=OpportunityStatus.PARTIALLY_ADDRESSED,
            rationale="A 5-year RCT directly provides the requested multi-year longitudinal evidence.",
        ),
        Stage4bBenchmarkCase(
            case_id="DEV-05",
            benchmark_version="v2.0-robustness",
            split=BenchmarkSplit.DEV,
            source_category=CaseSourceCategory.HUMAN_CURATED_PILOT,
            domain="Geriatric Medicine",
            paper_id="pilot-fasting-003",
            paper_title="A meta-analysis of decades of intermittent fasting research in geriatric populations",
            opportunity_id="pilot-opp-fasting",
            opportunity_title="Lack of longitudinal studies on intermittent fasting in older adults",
            opportunity_problem="Most studies on intermittent fasting focus on short-term outcomes in young populations.",
            opportunity_direction="Conduct multi-year trials assessing bone mineral density and sarcopenia in patients over 65.",
            passages=[
                "Contrary to popular belief, our comprehensive review found 42 distinct longitudinal trials "
                "lasting 5-10 years focusing entirely on bone and muscle health in seniors undergoing intermittent fasting."
            ],
            claimed_excerpt="found 42 distinct longitudinal trials lasting 5-10 years focusing entirely on bone and muscle health in seniors",
            expected_relationship=OpportunityRelationship.CONTRADICTS,
            is_borderline=False,
            borderline_alternative_class=None,
            requires_expert_review=False,
            expert_review_notes=None,
            grounding_challenge_type=GroundingChallengeType.EXACT_MATCH,
            expected_grounding_valid=True,
            initial_opportunity_status=OpportunityStatus.STRENGTHENED,
            addressing_paper_count=0,
            expected_state_machine_status=OpportunityStatus.OPEN,
            expected_end_to_end_status=OpportunityStatus.OPEN,
            rationale="Demonstrating 42 existing longitudinal trials directly contradicts the premise that such studies are lacking.",
        ),
        Stage4bBenchmarkCase(
            case_id="DEV-06",
            benchmark_version="v2.0-robustness",
            split=BenchmarkSplit.DEV,
            source_category=CaseSourceCategory.HUMAN_CURATED_PILOT,
            domain="Distributed Systems / FL",
            paper_id="pilot-fl-001",
            paper_title="A survey of federated learning applications in healthcare",
            opportunity_id="pilot-opp-fl",
            opportunity_title="Scalability bottlenecks in federated learning over unreliable networks",
            opportunity_problem="Current FL protocols fail when client dropout exceeds 40%.",
            opportunity_direction="Develop robust aggregation protocols for high-dropout mobile environments.",
            passages=[
                "FL is highly promising for hospitals. However, as noted in recent literature, high dropout rates "
                "in mobile health applications remain a severe limitation that future work must address before widespread adoption."
            ],
            claimed_excerpt="high dropout rates in mobile health applications remain a severe limitation that future work must address",
            expected_relationship=OpportunityRelationship.SUPPORTS,
            is_borderline=True,
            borderline_alternative_class=OpportunityRelationship.UNRELATED,
            requires_expert_review=True,
            expert_review_notes="Survey paper merely restates the limitation. Annotators may debate whether a secondary survey citation constitutes substantive SUPPORTS or UNRELATED background discussion.",
            grounding_challenge_type=GroundingChallengeType.AMBIGUOUS_EVIDENCE,
            expected_grounding_valid=True,
            initial_opportunity_status=OpportunityStatus.OPEN,
            addressing_paper_count=0,
            expected_state_machine_status=OpportunityStatus.STRENGTHENED,
            expected_end_to_end_status=OpportunityStatus.STRENGTHENED,
            rationale="Mentions the dropout bottleneck as a primary open challenge, providing secondary supporting evidence.",
        ),
        Stage4bBenchmarkCase(
            case_id="DEV-07",
            benchmark_version="v2.0-robustness",
            split=BenchmarkSplit.DEV,
            source_category=CaseSourceCategory.HUMAN_CURATED_PILOT,
            domain="Distributed Systems / FL",
            paper_id="pilot-fl-002",
            paper_title="RobustAgg: A novel aggregation scheme for federated learning",
            opportunity_id="pilot-opp-fl",
            opportunity_title="Scalability bottlenecks in federated learning over unreliable networks",
            opportunity_problem="Current FL protocols fail when client dropout exceeds 40%.",
            opportunity_direction="Develop robust aggregation protocols for high-dropout mobile environments.",
            passages=[
                "We introduce RobustAgg, an algorithm specifically designed for environments with unreliable clients. "
                "Our protocol maintains convergence even when client dropout reaches up to 75%."
            ],
            claimed_excerpt="maintains convergence even when client dropout reaches up to 75%.",
            expected_relationship=OpportunityRelationship.ADDRESSES,
            is_borderline=False,
            borderline_alternative_class=None,
            requires_expert_review=False,
            expert_review_notes=None,
            grounding_challenge_type=GroundingChallengeType.EXACT_MATCH,
            expected_grounding_valid=True,
            initial_opportunity_status=OpportunityStatus.OPEN,
            addressing_paper_count=1,
            expected_state_machine_status=OpportunityStatus.PARTIALLY_ADDRESSED,
            expected_end_to_end_status=OpportunityStatus.PARTIALLY_ADDRESSED,
            rationale="RobustAgg directly overcomes the dropout bottleneck by maintaining convergence up to 75% dropout.",
        ),
        Stage4bBenchmarkCase(
            case_id="DEV-08",
            benchmark_version="v2.0-robustness",
            split=BenchmarkSplit.DEV,
            source_category=CaseSourceCategory.HUMAN_CURATED_PILOT,
            domain="Distributed Systems / FL",
            paper_id="pilot-fl-003",
            paper_title="Empirical reassessment of client dropout in production edge networks",
            opportunity_id="pilot-opp-fl",
            opportunity_title="Scalability bottlenecks in federated learning over unreliable networks",
            opportunity_problem="Current FL protocols fail when client dropout exceeds 40%.",
            opportunity_direction="Develop robust aggregation protocols for high-dropout mobile environments.",
            passages=[
                "Across 10 million consumer mobile devices, client dropout was consistently under 1.2% per round, "
                "proving the 40% dropout bottleneck assumption is ungrounded in real-world deployments."
            ],
            claimed_excerpt="dropout was consistently under 1.2% per round, proving the 40% dropout bottleneck assumption is ungrounded",
            expected_relationship=OpportunityRelationship.CONTRADICTS,
            is_borderline=False,
            borderline_alternative_class=None,
            requires_expert_review=False,
            expert_review_notes=None,
            grounding_challenge_type=GroundingChallengeType.EXACT_MATCH,
            expected_grounding_valid=True,
            initial_opportunity_status=OpportunityStatus.PARTIALLY_ADDRESSED,
            addressing_paper_count=0,
            expected_state_machine_status=OpportunityStatus.PARTIALLY_ADDRESSED,
            expected_end_to_end_status=OpportunityStatus.PARTIALLY_ADDRESSED,
            rationale="Production telemetry contradicts the core premise that 40% client dropout occurs in mobile networks.",
        ),
        Stage4bBenchmarkCase(
            case_id="DEV-09",
            benchmark_version="v2.0-robustness",
            split=BenchmarkSplit.DEV,
            source_category=CaseSourceCategory.REAL_LITERATURE,
            domain="NLP / Retrieval Augmentation",
            paper_id="2376a0bd-733d-4307-9173-539313dbfec7",
            paper_title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)",
            opportunity_id=rag_opp_id,
            opportunity_title=rag_opp_title,
            opportunity_problem=rag_opp_prob,
            opportunity_direction=rag_opp_dir,
            passages=[
                "We explore a general-purpose \ufb01ne-tuning recipe for retrieval-augmented generation (RAG) — "
                "models which combine pre-trained parametric and non-parametric memory for language generation. "
                "We com- pare two RAG formulations, one which conditions on the same retrieved passages across the whole sequence."
            ],
            claimed_excerpt="recipe for retrieval-augmented generation (RAG)",
            expected_relationship=OpportunityRelationship.ADDRESSES,
            is_borderline=False,
            borderline_alternative_class=None,
            requires_expert_review=False,
            expert_review_notes=None,
            grounding_challenge_type=GroundingChallengeType.LIGATURE,
            expected_grounding_valid=True,
            initial_opportunity_status=OpportunityStatus.OPEN,
            addressing_paper_count=1,
            expected_state_machine_status=OpportunityStatus.PARTIALLY_ADDRESSED,
            expected_end_to_end_status=OpportunityStatus.PARTIALLY_ADDRESSED,
            rationale="Passage contains unicode ligature fi ('\\ufb01ne-tuning'); excerpt submitted in clean ASCII. Grounding must match robustly.",
        ),
        Stage4bBenchmarkCase(
            case_id="DEV-10",
            benchmark_version="v2.0-robustness",
            split=BenchmarkSplit.DEV,
            source_category=CaseSourceCategory.SYNTHETIC_STRESS_TEST,
            domain="Synthetic Adversarial",
            paper_id="syn-adv-001",
            paper_title="Synthetic Paper Alpha: Low-Resource Morphological Analysis",
            opportunity_id=rag_opp_id,
            opportunity_title=rag_opp_title,
            opportunity_problem=rag_opp_prob,
            opportunity_direction=rag_opp_dir,
            passages=[
                "Paper Alpha strictly investigates rule-based morphological inflection in Uralic language families."
            ],
            claimed_excerpt="Paper Beta demonstrates that dense vector retrieval resolves parametric hallucination",
            expected_relationship=OpportunityRelationship.SUPPORTS,
            is_borderline=False,
            borderline_alternative_class=None,
            requires_expert_review=False,
            expert_review_notes=None,
            grounding_challenge_type=GroundingChallengeType.CROSS_PAPER_ATTRIBUTION,
            expected_grounding_valid=False,
            initial_opportunity_status=OpportunityStatus.OPEN,
            addressing_paper_count=0,
            expected_state_machine_status=OpportunityStatus.STRENGTHENED,
            expected_end_to_end_status=OpportunityStatus.OPEN,
            rationale="Excerpt belongs to external Paper Beta. Attributing it to Paper Alpha must be strictly blocked by grounding.",
        ),

        # ══════════════════════════════════════════════════════════════════════
        #  HELD-OUT EVALUATION SPLIT (HELD_OUT, N=15)
        # ══════════════════════════════════════════════════════════════════════
        Stage4bBenchmarkCase(
            case_id="HELD-01",
            benchmark_version="v2.0-robustness",
            split=BenchmarkSplit.HELD_OUT,
            source_category=CaseSourceCategory.REAL_LITERATURE,
            domain="NLP / Retrieval Augmentation",
            paper_id="19654fd8-f3be-4b1f-92d0-e42d68729d31",
            paper_title="Enabling Large Language Models to Generate Text with Citations (Gao et al., 2023)",
            opportunity_id=rag_opp_id,
            opportunity_title=rag_opp_title,
            opportunity_problem=rag_opp_prob,
            opportunity_direction=rag_opp_dir,
            passages=[
                "In this work, our aim is to allow LLMs to generate text with citations, improving their factual "
                "correctness and verifiability. We propose ALCE, the first benchmark for Automatic LLMs' Citation Evaluation. "
                "ALCE collects a diverse set of questions and retrieval corpora and requires building end-to-end systems to "
                "retrieve supporting evidence and generate answers with citations."
            ],
            claimed_excerpt="We propose ALCE, the first benchmark for Automatic LLMs' Citation Evaluation",
            expected_relationship=OpportunityRelationship.EXTENDS,
            is_borderline=True,
            borderline_alternative_class=OpportunityRelationship.ADDRESSES,
            requires_expert_review=True,
            expert_review_notes="Gao et al. both propose an evaluation benchmark (ALCE) and test baseline generation models. Classified as EXTENDS because its primary contribution is defining citation metrics/benchmarks, but annotators could view baseline prompting as partially addressing hallucination.",
            grounding_challenge_type=GroundingChallengeType.METHOD_AND_BENCHMARK,
            expected_grounding_valid=True,
            initial_opportunity_status=OpportunityStatus.OPEN,
            addressing_paper_count=0,
            expected_state_machine_status=OpportunityStatus.OPEN,
            expected_end_to_end_status=OpportunityStatus.OPEN,
            rationale="Extends the retrieval-generation scope into verifiable citation benchmarks and attribution evaluation.",
        ),
        Stage4bBenchmarkCase(
            case_id="HELD-02",
            benchmark_version="v2.0-robustness",
            split=BenchmarkSplit.HELD_OUT,
            source_category=CaseSourceCategory.REAL_LITERATURE,
            domain="NLP / Retrieval Augmentation",
            paper_id="8e5f434b-e803-459b-a1d1-0675f03f1e84",
            paper_title="Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection (Asai et al., 2023)",
            opportunity_id=rag_opp_id,
            opportunity_title=rag_opp_title,
            opportunity_problem=rag_opp_prob,
            opportunity_direction=rag_opp_dir,
            passages=[
                "We introduce a new framework called Self-Reflective Retrieval-Augmented Generation (SELF-RAG) "
                "that enhances an LM's quality and factuality through retrieval and self-reflection. "
                "Our framework trains a single arbitrary LM that adaptively retrieves passages on-demand, "
                "and generates and reflects on retrieved passages and its own generations using reflection tokens."
            ],
            claimed_excerpt="single arbitrary LM that adaptively retrieves passages on-demand, and generates and reflects on retrieved passages",
            expected_relationship=OpportunityRelationship.ADDRESSES,
            is_borderline=True,
            borderline_alternative_class=OpportunityRelationship.EXTENDS,
            requires_expert_review=True,
            expert_review_notes="Self-RAG introduces a novel architectural concept (reflection tokens), which can be categorized as EXTENDS. However, because it explicitly optimizes for and resolves indiscriminate retrieval overhead, ADDRESSES is the primary functional classification.",
            grounding_challenge_type=GroundingChallengeType.METHOD_AND_BENCHMARK,
            expected_grounding_valid=True,
            initial_opportunity_status=OpportunityStatus.OPEN,
            addressing_paper_count=1,
            expected_state_machine_status=OpportunityStatus.PARTIALLY_ADDRESSED,
            expected_end_to_end_status=OpportunityStatus.PARTIALLY_ADDRESSED,
            rationale="Proposes adaptive on-demand retrieval with critique tokens, directly resolving indiscriminate retrieval overhead.",
        ),
        Stage4bBenchmarkCase(
            case_id="HELD-03",
            benchmark_version="v2.0-robustness",
            split=BenchmarkSplit.HELD_OUT,
            source_category=CaseSourceCategory.REAL_LITERATURE,
            domain="NLP / Retrieval Augmentation",
            paper_id="e5b7e483-16f6-4ea5-8426-b04e1a4f3d46",
            paper_title="Making Retrieval-Augmented Language Models Robust to Irrelevant Context (Yoran et al., 2023)",
            opportunity_id=rag_opp_id,
            opportunity_title=rag_opp_title,
            opportunity_problem=rag_opp_prob,
            opportunity_direction=rag_opp_dir,
            passages=[
                "Thus, we propose a method for automatically generating data to fine-tune the language model to "
                "properly leverage retrieved passages, including for challenging multi-hop tasks. "
                "We empirically show that even 1,000 examples suffice to train the model to be robust to irrelevant contexts while maintaining high performance on examples with relevant ones."
            ],
            claimed_excerpt="train the model to be robust to irrelevant contexts while maintaining high performance on examples with relevant ones.",
            expected_relationship=OpportunityRelationship.ADDRESSES,
            is_borderline=False,
            borderline_alternative_class=None,
            requires_expert_review=False,
            expert_review_notes=None,
            grounding_challenge_type=GroundingChallengeType.EXACT_MATCH,
            expected_grounding_valid=True,
            initial_opportunity_status=OpportunityStatus.STRENGTHENED,
            addressing_paper_count=1,
            expected_state_machine_status=OpportunityStatus.PARTIALLY_ADDRESSED,
            expected_end_to_end_status=OpportunityStatus.PARTIALLY_ADDRESSED,
            rationale="Empirically validates robust fine-tuning to ignore irrelevant context, directly addressing the distraction limitation.",
        ),
        Stage4bBenchmarkCase(
            case_id="HELD-04",
            benchmark_version="v2.0-robustness",
            split=BenchmarkSplit.HELD_OUT,
            source_category=CaseSourceCategory.REAL_LITERATURE,
            domain="NLP / Retrieval Augmentation",
            paper_id="e5b7e483-16f6-4ea5-8426-b04e1a4f3d46",
            paper_title="Making Retrieval-Augmented Language Models Robust to Irrelevant Context (Yoran et al., 2023)",
            opportunity_id=rag_opp_id,
            opportunity_title=rag_opp_title,
            opportunity_problem=rag_opp_prob,
            opportunity_direction=rag_opp_dir,
            passages=[
                "We present a thorough analysis on five open-domain question answering benchmarks, characterizing cases when retrieval reduces accuracy."
            ],
            claimed_excerpt="We completely eliminated all cascading multi-hop errors in RALMs with zero training cost by switching to quantum embeddings.",
            expected_relationship=OpportunityRelationship.ADDRESSES,
            is_borderline=False,
            borderline_alternative_class=None,
            requires_expert_review=False,
            expert_review_notes=None,
            grounding_challenge_type=GroundingChallengeType.UNSUPPORTED_FABRICATION,
            expected_grounding_valid=False,
            initial_opportunity_status=OpportunityStatus.OPEN,
            addressing_paper_count=0,
            expected_state_machine_status=OpportunityStatus.PARTIALLY_ADDRESSED,
            expected_end_to_end_status=OpportunityStatus.OPEN,
            rationale="Fabricated claim not present in Yoran et al. Grounding gatekeeper must reject it with UngroundedEvidenceError.",
        ),
        Stage4bBenchmarkCase(
            case_id="HELD-05",
            benchmark_version="v2.0-robustness",
            split=BenchmarkSplit.HELD_OUT,
            source_category=CaseSourceCategory.HUMAN_CURATED_PILOT,
            domain="Differential Privacy",
            paper_id="pilot-dp-001",
            paper_title="A novel differentially private GAN achieving epsilon=0.5 with 92% clinical fidelity",
            opportunity_id="pilot-opp-dp",
            opportunity_title="High privacy budget epsilon needed for high utility in clinical trial synthetic data generation",
            opportunity_problem="Existing DP methods require epsilon > 5.0 to maintain clinical trial statistical validity.",
            opportunity_direction="Develop privacy amplification mechanisms to achieve sub-unitary epsilon for synthetic patient records.",
            passages=[
                "We present DP-ClinGAN, a generative adversarial network that guarantees epsilon=0.5 differential privacy "
                "while achieving 92% utility on cardiovascular clinical trial benchmarks."
            ],
            claimed_excerpt="guarantees epsilon=0.5 differential privacy while achieving 92% utility on cardiovascular clinical trial benchmarks.",
            expected_relationship=OpportunityRelationship.ADDRESSES,
            is_borderline=False,
            borderline_alternative_class=None,
            requires_expert_review=False,
            expert_review_notes=None,
            grounding_challenge_type=GroundingChallengeType.EXACT_MATCH,
            expected_grounding_valid=True,
            initial_opportunity_status=OpportunityStatus.OPEN,
            addressing_paper_count=1,
            expected_state_machine_status=OpportunityStatus.PARTIALLY_ADDRESSED,
            expected_end_to_end_status=OpportunityStatus.PARTIALLY_ADDRESSED,
            rationale="DP-ClinGAN achieves sub-unitary epsilon=0.5 while maintaining utility, directly addressing the gap.",
        ),
        Stage4bBenchmarkCase(
            case_id="HELD-06",
            benchmark_version="v2.0-robustness",
            split=BenchmarkSplit.HELD_OUT,
            source_category=CaseSourceCategory.HUMAN_CURATED_PILOT,
            domain="Differential Privacy",
            paper_id="pilot-dp-002",
            paper_title="Impossibility results for low-epsilon synthetic tabular data under realistic sample sizes",
            opportunity_id="pilot-opp-dp",
            opportunity_title="High privacy budget epsilon needed for high utility in clinical trial synthetic data generation",
            opportunity_problem="Existing DP methods require epsilon > 5.0 to maintain clinical trial statistical validity.",
            opportunity_direction="Develop privacy amplification mechanisms to achieve sub-unitary epsilon for synthetic patient records.",
            passages=[
                "We prove information-theoretically that no mechanism can achieve epsilon < 1.0 with greater than 80% utility "
                "when sample size N < 50,000, contradicting recent empirical claims."
            ],
            claimed_excerpt="no mechanism can achieve epsilon < 1.0 with greater than 80% utility when sample size N < 50,000, contradicting recent empirical claims.",
            expected_relationship=OpportunityRelationship.CONTRADICTS,
            is_borderline=True,
            borderline_alternative_class=OpportunityRelationship.SUPPORTS,
            requires_expert_review=True,
            expert_review_notes="Theoretical impossibility paper. Could be annotated as SUPPORTS (reinforcing that low epsilon is difficult) or CONTRADICTS (refuting the feasibility premise of the proposed research direction). Gold label is CONTRADICTS because it proves the proposed direction is mathematically unachievable.",
            grounding_challenge_type=GroundingChallengeType.EXACT_MATCH,
            expected_grounding_valid=True,
            initial_opportunity_status=OpportunityStatus.STRENGTHENED,
            addressing_paper_count=0,
            expected_state_machine_status=OpportunityStatus.OPEN,
            expected_end_to_end_status=OpportunityStatus.OPEN,
            rationale="Information-theoretic lower bound contradicts the premise that sub-unitary epsilon with high utility is achievable at cohort scale.",
        ),
        Stage4bBenchmarkCase(
            case_id="HELD-07",
            benchmark_version="v2.0-robustness",
            split=BenchmarkSplit.HELD_OUT,
            source_category=CaseSourceCategory.HUMAN_CURATED_PILOT,
            domain="Differential Privacy",
            paper_id="pilot-dp-003",
            paper_title="Extending differential privacy to continuous-time vital sign telemetry in ICU monitoring",
            opportunity_id="pilot-opp-dp",
            opportunity_title="High privacy budget epsilon needed for high utility in clinical trial synthetic data generation",
            opportunity_problem="Existing DP methods require epsilon > 5.0 to maintain clinical trial statistical validity.",
            opportunity_direction="Develop privacy amplification mechanisms to achieve sub-unitary epsilon for synthetic patient records.",
            passages=[
                "While existing DP tabular methods focus on static trial records, we extend formal differential privacy "
                "guarantees to continuous streaming ECG waveforms in intensive care telemetry."
            ],
            claimed_excerpt="extend formal differential privacy guarantees to continuous streaming ECG waveforms",
            expected_relationship=OpportunityRelationship.EXTENDS,
            is_borderline=False,
            borderline_alternative_class=None,
            requires_expert_review=False,
            expert_review_notes=None,
            grounding_challenge_type=GroundingChallengeType.EXACT_MATCH,
            expected_grounding_valid=True,
            initial_opportunity_status=OpportunityStatus.OPEN,
            addressing_paper_count=0,
            expected_state_machine_status=OpportunityStatus.OPEN,
            expected_end_to_end_status=OpportunityStatus.OPEN,
            rationale="Applies differential privacy to streaming continuous waveforms rather than tabular clinical records, extending the problem domain.",
        ),
        Stage4bBenchmarkCase(
            case_id="HELD-08",
            benchmark_version="v2.0-robustness",
            split=BenchmarkSplit.HELD_OUT,
            source_category=CaseSourceCategory.HUMAN_CURATED_PILOT,
            domain="Molecular Biology / GNN",
            paper_id="pilot-gnn-001",
            paper_title="Survey on GNN architectures in computational chemistry",
            opportunity_id="pilot-opp-gnn",
            opportunity_title="Over-squashing in message passing neural networks limits molecular property prediction for large polymers",
            opportunity_problem="Standard GNN message passing compresses exponential neighborhood information into fixed-size node vectors.",
            opportunity_direction="Design graph rewiring techniques based on Ricci curvature to eliminate bottleneck edges.",
            passages=[
                "As highlighted in recent literature, information over-squashing at bottleneck nodes prevents standard message passing "
                "from propagating long-range atomic interactions in macromolecules."
            ],
            claimed_excerpt="information over-squashing at bottleneck nodes prevents standard message passing from propagating long-range atomic interactions",
            expected_relationship=OpportunityRelationship.SUPPORTS,
            is_borderline=True,
            borderline_alternative_class=OpportunityRelationship.UNRELATED,
            requires_expert_review=True,
            expert_review_notes="Survey paper discussing known GNN limitations. Categorized as SUPPORTS because it confirms the bottleneck in macromolecules, but reviewers could argue surveys without new experiments are UNRELATED.",
            grounding_challenge_type=GroundingChallengeType.AMBIGUOUS_EVIDENCE,
            expected_grounding_valid=True,
            initial_opportunity_status=OpportunityStatus.OPEN,
            addressing_paper_count=0,
            expected_state_machine_status=OpportunityStatus.STRENGTHENED,
            expected_end_to_end_status=OpportunityStatus.STRENGTHENED,
            rationale="Provides evidence supporting the presence and severity of information over-squashing in macromolecules.",
        ),
        Stage4bBenchmarkCase(
            case_id="HELD-09",
            benchmark_version="v2.0-robustness",
            split=BenchmarkSplit.HELD_OUT,
            source_category=CaseSourceCategory.HUMAN_CURATED_PILOT,
            domain="Molecular Biology / GNN",
            paper_id="pilot-gnn-002",
            paper_title="Curvature-based rewiring eliminates over-squashing in polymer graphs",
            opportunity_id="pilot-opp-gnn",
            opportunity_title="Over-squashing in message passing neural networks limits molecular property prediction for large polymers",
            opportunity_problem="Standard GNN message passing compresses exponential neighborhood information into fixed-size node vectors.",
            opportunity_direction="Design graph rewiring techniques based on Ricci curvature to eliminate bottleneck edges.",
            passages=[
                "We propose Ollivier-Ricci curvature edge rewiring, which directly eliminates graph bottlenecks "
                "and improves polymer property prediction by 14.2%."
            ],
            claimed_excerpt="Ollivier-Ricci curvature edge rewiring, which directly eliminates graph bottlenecks",
            expected_relationship=OpportunityRelationship.ADDRESSES,
            is_borderline=False,
            borderline_alternative_class=None,
            requires_expert_review=False,
            expert_review_notes=None,
            grounding_challenge_type=GroundingChallengeType.EXACT_MATCH,
            expected_grounding_valid=True,
            initial_opportunity_status=OpportunityStatus.OPEN,
            addressing_paper_count=1,
            expected_state_machine_status=OpportunityStatus.PARTIALLY_ADDRESSED,
            expected_end_to_end_status=OpportunityStatus.PARTIALLY_ADDRESSED,
            rationale="Implements exactly the proposed Ricci curvature graph rewiring, directly addressing the over-squashing bottleneck.",
        ),
        Stage4bBenchmarkCase(
            case_id="HELD-10",
            benchmark_version="v2.0-robustness",
            split=BenchmarkSplit.HELD_OUT,
            source_category=CaseSourceCategory.HUMAN_CURATED_PILOT,
            domain="Molecular Biology / GNN",
            paper_id="pilot-gnn-003",
            paper_title="Synthesizing biodegradable plastics from agricultural cellulose waste",
            opportunity_id="pilot-opp-gnn",
            opportunity_title="Over-squashing in message passing neural networks limits molecular property prediction for large polymers",
            opportunity_problem="Standard GNN message passing compresses exponential neighborhood information into fixed-size node vectors.",
            opportunity_direction="Design graph rewiring techniques based on Ricci curvature to eliminate bottleneck edges.",
            passages=[
                "We demonstrate an enzymatic protocol to convert corn husk cellulose into biodegradable thermoplastic polymers."
            ],
            claimed_excerpt=None,
            expected_relationship=OpportunityRelationship.UNRELATED,
            is_borderline=False,
            borderline_alternative_class=None,
            requires_expert_review=False,
            expert_review_notes=None,
            grounding_challenge_type=GroundingChallengeType.NONE_REQUIRED,
            expected_grounding_valid=True,
            initial_opportunity_status=OpportunityStatus.OPEN,
            addressing_paper_count=0,
            expected_state_machine_status=OpportunityStatus.OPEN,
            expected_end_to_end_status=OpportunityStatus.OPEN,
            rationale="Wet-lab chemical synthesis of bioplastics shares the keyword 'polymer' but is completely unrelated to graph neural network message-passing architectures.",
        ),
        Stage4bBenchmarkCase(
            case_id="HELD-11",
            benchmark_version="v2.0-robustness",
            split=BenchmarkSplit.HELD_OUT,
            source_category=CaseSourceCategory.REAL_LITERATURE,
            domain="NLP / Retrieval Augmentation",
            paper_id="32e1578f-6749-496d-a391-e33f002335a6",
            paper_title="Large Language Models Can Be Easily Distracted by Irrelevant Context (Shi et al., 2023)",
            opportunity_id=rag_opp_id,
            opportunity_title=rag_opp_title,
            opportunity_problem=rag_opp_prob,
            opportunity_direction=rag_opp_dir,
            passages=[
                "In this work, we investigate the distractibility of large language models, "
                "i.e., how the model problem-solving accuracy can be influenced by irrelevant context. "
                "We introduce Grade-School Math with Irrelevant Context (GSM-IC) and find that the model "
                "performance is dramatically decreased when irrelevant information is included."
            ],
            claimed_excerpt="Our testing reveals that supplementary text causes major drops in problem solving skills.",
            expected_relationship=OpportunityRelationship.SUPPORTS,
            is_borderline=False,
            borderline_alternative_class=None,
            requires_expert_review=False,
            expert_review_notes=None,
            grounding_challenge_type=GroundingChallengeType.PARAPHRASE_SUBTHRESHOLD,
            expected_grounding_valid=False,
            initial_opportunity_status=OpportunityStatus.OPEN,
            addressing_paper_count=0,
            expected_state_machine_status=OpportunityStatus.STRENGTHENED,
            expected_end_to_end_status=OpportunityStatus.OPEN,
            rationale="Sub-threshold paraphrase (~35% token similarity). Must be rejected by grounding to enforce strict evidence fidelity.",
        ),
        Stage4bBenchmarkCase(
            case_id="HELD-12",
            benchmark_version="v2.0-robustness",
            split=BenchmarkSplit.HELD_OUT,
            source_category=CaseSourceCategory.REAL_LITERATURE,
            domain="NLP / Retrieval Augmentation",
            paper_id="8e5f434b-e803-459b-a1d1-0675f03f1e84",
            paper_title="Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection (Asai et al., 2023)",
            opportunity_id=rag_opp_id,
            opportunity_title=rag_opp_title,
            opportunity_problem=rag_opp_prob,
            opportunity_direction=rag_opp_dir,
            passages=[
                "indiscriminately retrieving and incorporating a fixed number of retrieved passages, "
                "regardless of whether retrieval is necessary, or passages are relevant, diminishes LM versatility or can lead to unhelpful response generation."
            ],
            claimed_excerpt="indiscriminately retrieving and incorporating a fixed number of retrieved passages",
            expected_relationship=OpportunityRelationship.SUPPORTS,
            is_borderline=True,
            borderline_alternative_class=OpportunityRelationship.ADDRESSES,
            requires_expert_review=True,
            expert_review_notes="The paper as a whole ADDRESSES the problem, but this specific excerpt quotes the paper's description of prior art's limitation. Evaluated as SUPPORTS because the excerpt itself provides evidence of the limitation's existence.",
            grounding_challenge_type=GroundingChallengeType.AMBIGUOUS_EVIDENCE,
            expected_grounding_valid=True,
            initial_opportunity_status=OpportunityStatus.OPEN,
            addressing_paper_count=0,
            expected_state_machine_status=OpportunityStatus.STRENGTHENED,
            expected_end_to_end_status=OpportunityStatus.STRENGTHENED,
            rationale="Verbatim quote from Self-RAG's abstract describing the defect of indiscriminate retrieval, supporting the research opportunity.",
        ),
        Stage4bBenchmarkCase(
            case_id="HELD-13",
            benchmark_version="v2.0-robustness",
            split=BenchmarkSplit.HELD_OUT,
            source_category=CaseSourceCategory.REAL_LITERATURE,
            domain="NLP / Retrieval Augmentation",
            paper_id="8e5f434b-e803-459b-a1d1-0675f03f1e84",
            paper_title="Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection (Asai et al., 2023)",
            opportunity_id=rag_opp_id,
            opportunity_title=rag_opp_title,
            opportunity_problem=rag_opp_prob,
            opportunity_direction=rag_opp_dir,
            passages=[
                "SELF-RAG (7B and 13B parameters) significantly outperforms state-of-the-art LLMs and retrieval-augmented models on open-domain QA, reasoning and fact verification tasks."
            ],
            claimed_excerpt="SELF-RAG (7B and 13B parameters) significantly outperforms state-of-the-art LLMs and retrieval-augmented models on open-domain QA",
            expected_relationship=OpportunityRelationship.ADDRESSES,
            is_borderline=False,
            borderline_alternative_class=None,
            requires_expert_review=False,
            expert_review_notes=None,
            grounding_challenge_type=GroundingChallengeType.EXACT_MATCH,
            expected_grounding_valid=True,
            initial_opportunity_status=OpportunityStatus.PARTIALLY_ADDRESSED,
            addressing_paper_count=2,
            expected_state_machine_status=OpportunityStatus.RESOLVED,
            expected_end_to_end_status=OpportunityStatus.RESOLVED,
            rationale="AD-20 state-machine test: 2nd distinct addressing paper transitions PARTIALLY_ADDRESSED -> RESOLVED.",
        ),
        Stage4bBenchmarkCase(
            case_id="HELD-14",
            benchmark_version="v2.0-robustness",
            split=BenchmarkSplit.HELD_OUT,
            source_category=CaseSourceCategory.HUMAN_CURATED_PILOT,
            domain="Distributed Systems / FL",
            paper_id="pilot-fl-003",
            paper_title="Empirical reassessment of client dropout in production edge networks",
            opportunity_id="pilot-opp-fl",
            opportunity_title="Scalability bottlenecks in federated learning over unreliable networks",
            opportunity_problem="Current FL protocols fail when client dropout exceeds 40%.",
            opportunity_direction="Develop robust aggregation protocols for high-dropout mobile environments.",
            passages=[
                "Across 10 million consumer mobile devices, client dropout was consistently under 1.2% per round, "
                "proving the 40% dropout bottleneck assumption is ungrounded in real-world deployments."
            ],
            claimed_excerpt="dropout was consistently under 1.2% per round, proving the 40% dropout bottleneck assumption is ungrounded",
            expected_relationship=OpportunityRelationship.CONTRADICTS,
            is_borderline=False,
            borderline_alternative_class=None,
            requires_expert_review=False,
            expert_review_notes=None,
            grounding_challenge_type=GroundingChallengeType.EXACT_MATCH,
            expected_grounding_valid=True,
            initial_opportunity_status=OpportunityStatus.RESOLVED,
            addressing_paper_count=0,
            expected_state_machine_status=OpportunityStatus.OPEN,
            expected_end_to_end_status=OpportunityStatus.OPEN,
            rationale="State-machine reopening test: grounded CONTRADICTS against RESOLVED opportunity transitions back to OPEN.",
        ),
        Stage4bBenchmarkCase(
            case_id="HELD-15",
            benchmark_version="v2.0-robustness",
            split=BenchmarkSplit.HELD_OUT,
            source_category=CaseSourceCategory.REAL_LITERATURE,
            domain="Computer Vision Control",
            paper_id="he2016-resnet-control",
            paper_title="Deep Residual Learning for Image Recognition (He et al., 2016)",
            opportunity_id="pilot-opp-gnn",
            opportunity_title="Over-squashing in message passing neural networks limits molecular property prediction for large polymers",
            opportunity_problem="Standard GNN message passing compresses exponential neighborhood information into fixed-size node vectors.",
            opportunity_direction="Design graph rewiring techniques based on Ricci curvature to eliminate bottleneck edges.",
            passages=[
                "Deeper neural networks are more difficult to train. We present a residual learning framework "
                "to ease the training of networks that are substantially deeper than those used previously."
            ],
            claimed_excerpt=None,
            expected_relationship=OpportunityRelationship.UNRELATED,
            is_borderline=False,
            borderline_alternative_class=None,
            requires_expert_review=False,
            expert_review_notes=None,
            grounding_challenge_type=GroundingChallengeType.NONE_REQUIRED,
            expected_grounding_valid=True,
            initial_opportunity_status=OpportunityStatus.OPEN,
            addressing_paper_count=0,
            expected_state_machine_status=OpportunityStatus.OPEN,
            expected_end_to_end_status=OpportunityStatus.OPEN,
            rationale="Computer vision residual convolution learning is completely unrelated to graph over-squashing in molecular chemistry.",
        ),
    ]

    return cases


# ── EVALUATION RUNNERS ─────────────────────────────────────────────────────────

def run_grounding_evaluation_by_category(cases: Sequence[Stage4bBenchmarkCase]) -> Dict[str, Any]:
    """Evaluates evidence grounding validation stratified by source category."""
    categories = [
        CaseSourceCategory.REAL_LITERATURE,
        CaseSourceCategory.HUMAN_CURATED_PILOT,
        CaseSourceCategory.SYNTHETIC_STRESS_TEST,
    ]
    stratified: Dict[str, Dict[str, Any]] = {}
    overall_correct = 0

    for cat in categories:
        cat_cases = [c for c in cases if c.source_category == cat]
        if not cat_cases:
            continue

        cat_valid_count = sum(1 for c in cat_cases if c.expected_grounding_valid)
        cat_invalid_count = sum(1 for c in cat_cases if not c.expected_grounding_valid)
        cat_accepted = 0
        cat_rejected = 0

        for c in cat_cases:
            passed = False
            try:
                validate_evidence_grounding(
                    c.claimed_excerpt,
                    c.passages,
                    c.expected_relationship,
                    threshold=DEFAULT_GROUNDING_THRESHOLD,
                    paper_id=c.paper_id,
                )
                passed = True
            except UngroundedEvidenceError:
                passed = False

            if c.expected_grounding_valid and passed:
                cat_accepted += 1
                overall_correct += 1
            elif not c.expected_grounding_valid and not passed:
                cat_rejected += 1
                overall_correct += 1

        acc_rate = cat_accepted / cat_valid_count if cat_valid_count > 0 else 1.0
        rej_rate = cat_rejected / cat_invalid_count if cat_invalid_count > 0 else 1.0

        stratified[cat.value] = {
            "total": len(cat_cases),
            "valid_cases": cat_valid_count,
            "accepted": cat_accepted,
            "acceptance_rate": round(acc_rate, 4),
            "invalid_cases": cat_invalid_count,
            "rejected": cat_rejected,
            "rejection_rate": round(rej_rate, 4),
            "accuracy": round((cat_accepted + cat_rejected) / len(cat_cases), 4),
        }

    return {
        "overall_total": len(cases),
        "overall_accuracy": round(overall_correct / len(cases), 4) if cases else 0.0,
        "by_category": stratified,
    }


def run_classification_evaluation_by_split(
    cases: Sequence[Stage4bBenchmarkCase],
    predictions: Sequence[OpportunityRelationship],
) -> Dict[str, Any]:
    """Computes confusion matrix and per-class metrics separately for DEV and HELD_OUT splits."""
    splits = [BenchmarkSplit.DEV, BenchmarkSplit.HELD_OUT]
    results_by_split: Dict[str, Dict[str, Any]] = {}

    classes = [
        OpportunityRelationship.SUPPORTS,
        OpportunityRelationship.ADDRESSES,
        OpportunityRelationship.CONTRADICTS,
        OpportunityRelationship.EXTENDS,
        OpportunityRelationship.UNRELATED,
    ]
    class_names = [c.value for c in classes]

    for split in splits:
        split_pairs = [
            (c, p) for c, p in zip(cases, predictions) if c.split == split
        ]
        if not split_pairs:
            continue

        matrix: Dict[str, Dict[str, int]] = {
            true_cls: {pred_cls: 0 for pred_cls in class_names}
            for true_cls in class_names
        }

        for c, p in split_pairs:
            matrix[c.expected_relationship.value][p.value] += 1

        per_class: Dict[str, Dict[str, Any]] = {}
        total_tp = 0
        total_samples = len(split_pairs)

        for c_name in class_names:
            tp = matrix[c_name][c_name]
            fp = sum(matrix[other][c_name] for other in class_names if other != c_name)
            fn = sum(matrix[c_name][other] for other in class_names if other != c_name)
            tn = total_samples - (tp + fp + fn)

            p_score = precision_score(tp, fp)
            r_score = recall_score(tp, fn)
            f_score = f1_score(p_score, r_score)

            per_class[c_name] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "support": sum(matrix[c_name].values()),
                "precision": round(p_score, 4),
                "recall": round(r_score, 4),
                "f1": round(f_score, 4),
            }
            total_tp += tp

        macro_p = sum(m["precision"] for m in per_class.values()) / len(classes)
        macro_r = sum(m["recall"] for m in per_class.values()) / len(classes)
        macro_f1 = sum(m["f1"] for m in per_class.values()) / len(classes)

        results_by_split[split.value] = {
            "sample_size": total_samples,
            "overall_accuracy": round(total_tp / total_samples, 4) if total_samples > 0 else 0.0,
            "macro_precision": round(macro_p, 4),
            "macro_recall": round(macro_r, 4),
            "macro_f1": round(macro_f1, 4),
            "confusion_matrix": matrix,
            "per_class": per_class,
        }

    return results_by_split
