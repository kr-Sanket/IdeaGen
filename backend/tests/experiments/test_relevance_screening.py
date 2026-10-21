"""Deterministic offline tests for Checkpoint 7: Topic-Based Relevance Screening.

Tests verify:
1. Scoring formula and deterministic ranking.
2. Title/abstract and body passage contributions.
3. Excluding references from scoring.
4. Threshold equality behavior (score >= threshold).
5. Capacity-based exclusions (EXCLUDED_CAPACITY_LIMIT).
6. Minimum-selection floor and warnings (SELECTED_BELOW_THRESHOLD_FLOOR).
7. Zero eligible papers with minimum zero fails closed before Stage 2 LLM extraction.
8. Stable tie-breaking under shuffled input (by filename ascending).
9. Excluded papers remain in storage (documents, passages, and papers not pruned).
10. Post-cutoff papers remain available to Stage 5 without screening suppression.
11. Fingerprint mismatch when topic or settings change (refuses unsafe resume).
12. Reuse of screening results when inputs and configurations are unchanged.
13. Embedding failure stops processing before LLM extraction (fail-closed).
14. Screening-only mode produces an auditable report and makes zero LLM calls.
15. No topic preserves previous pipeline behavior.
16. Stage 5 contradictory and extension evidence is evaluated without similarity filtering.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import uuid

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_BACKEND_DIR = _REPO_ROOT / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from ideagen.db.models import (
    Base,
    Paper,
    PaperDocument,
    PaperPassage,
    ResearchKnowledge,
    ResearchStatement,
    ResearchOpportunityRecord,
    Opportunity,
    OpportunityEvidence,
)
from ideagen.discovery.embeddings import EmbeddingProvider
from ideagen.discovery.screening import (
    RelevanceScreeningService,
    ScreeningReport,
    compute_relevance_score,
    is_reference_passage,
    is_abstract_passage,
    DEFAULT_SCREENING_THRESHOLD,
    DEFAULT_SCREENING_MIN_PAPERS,
)
from experiments.runner import (
    ControlledExperimentRunner,
    ExperimentMetadataRecord,
    compute_configuration_fingerprint,
    compare_fingerprints,
)
from ideagen.opportunity.models import OpportunityRelationship
from ideagen.understanding.models import StructuredPaperExtractionOutput
from tests.experiments.test_controlled_experiment_runner import _create_synthetic_pdf
from tests.experiments.test_stage5_evolution_runner import Stage5MockLLMProvider


class MockTopicEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider mapping known test strings to explicit unit vectors."""

    def __init__(self, dimension: int = 16, model_name: str = "mock-topic-embedder-16d"):
        self._dim = dimension
        self._model_name = model_name
        self.call_count = 0
        self.should_raise = False

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        self.call_count += len(texts)
        if self.should_raise:
            raise RuntimeError("Simulated embedding provider failure")

        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        vectors = []
        for text in texts:
            t = text.lower()
            # Topic vector: aligned with axis 0
            if "quantum error correction" in t or "qec topic" in t:
                sim = 1.0
            elif "high_sim" in t or "syndrome extraction" in t or "fault-tolerant" in t or "teleportation" in t:
                sim = 0.85
            elif "med_sim" in t or "surface code" in t:
                sim = 0.60
            elif "threshold_exact" in t or "exact_threshold" in t:
                sim = 0.50
            elif "low_sim" in t or "classical routing" in t or "unrelated compiler" in t:
                sim = 0.20
            elif "ref_distractor" in t or "distractor reference" in t:
                # Very high similarity, but placed in reference section!
                sim = 0.99
            else:
                sim = 0.40

            vec = np.zeros(self._dim, dtype=np.float32)
            vec[0] = sim
            vec[1] = np.sqrt(max(0.0, 1.0 - sim * sim))
            norm = np.linalg.norm(vec)
            vectors.append(vec / (norm if norm > 0 else 1.0))

        return np.array(vectors, dtype=np.float32)


class TrackingMockLLM(Stage5MockLLMProvider):
    """LLM mock tracking total generation and understanding calls across Stages 1-5."""

    def __init__(
        self,
        relationship: OpportunityRelationship = OpportunityRelationship.SUPPORTS,
        evidence_excerpt: str = "high_sim teleportation protocol published after cutoff",
    ):
        super().__init__(relationship=relationship, evidence_excerpt=evidence_excerpt)
        self.understanding_calls = 0

    def generate_structured(self, prompt: str, response_schema):
        if response_schema == StructuredPaperExtractionOutput:
            self.understanding_calls += 1
        return super().generate_structured(prompt, response_schema)


@pytest.fixture
def multi_paper_corpus(tmp_path: Path) -> Path:
    """Creates a 4-paper corpus with synthetic PDFs and metadata."""
    corpus_dir = tmp_path / "screening_corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # Paper A: Highly relevant QEC paper
    p_a = corpus_dir / "paper_a.pdf"
    _create_synthetic_pdf(
        p_a,
        "Paper A: Fault-Tolerant Quantum Error Correction with Syndrome Extraction. "
        "Abstract: We introduce high_sim syndrome extraction for topological codes. "
        "Section 1 Introduction: high_sim fault-tolerant decoding on superconducting qubits. "
        "References: [1] Distractor reference on irrelevant topic."
    )

    # Paper B: Moderately relevant paper
    p_b = corpus_dir / "paper_b.pdf"
    _create_synthetic_pdf(
        p_b,
        "Paper B: Surface Code Architectures for Scalable Processing. "
        "Abstract: We analyze med_sim surface code lattices. "
        "Section 1 Body: med_sim logical qubit overhead in noisy environments."
    )

    # Paper C: Low-relevance paper
    p_c = corpus_dir / "paper_c.pdf"
    _create_synthetic_pdf(
        p_c,
        "Paper C: Classical Routing in Multi-Core Network Processors. "
        "Abstract: We study low_sim packet scheduling in classical switches. "
        "Section 1 Methods: low_sim packet queueing delays."
    )

    # Paper D: Post-cutoff future paper
    p_d = corpus_dir / "paper_d.pdf"
    _create_synthetic_pdf(
        p_d,
        "Paper D: Future Discoveries in Transmon Teleportation. "
        "Abstract: high_sim teleportation protocol published after cutoff. "
        "Section 1: high_sim experimental confirmation."
    )

    manifest = [
        {
            "filename": "paper_a.pdf",
            "title": "Fault-Tolerant Quantum Error Correction with Syndrome Extraction",
            "publication_date": "2021-03-01",
        },
        {
            "filename": "paper_b.pdf",
            "title": "Surface Code Architectures for Scalable Processing",
            "publication_date": "2021-06-01",
        },
        {
            "filename": "paper_c.pdf",
            "title": "Classical Routing in Multi-Core Network Processors",
            "publication_date": "2021-09-01",
        },
        {
            "filename": "paper_d.pdf",
            "title": "Future Discoveries in Transmon Teleportation",
            "publication_date": "2023-01-15",
        },
    ]
    with open(corpus_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    return corpus_dir


# ─────────────────────────────────────────────────────────────────────────────
# 1. Scoring formula and deterministic ranking
# ─────────────────────────────────────────────────────────────────────────────

def test_scoring_formula_and_deterministic_ranking():
    """Verifies alpha-weighted formula: Score = alpha * paper_sim + (1 - alpha) * avg(top_3_body_sims)."""
    # Test case 1: exactly 3 body passages
    score = compute_relevance_score(paper_sim=0.80, body_sims=[0.90, 0.70, 0.50], alpha=0.50)
    # top_3 avg = (0.90 + 0.70 + 0.50) / 3 = 0.70
    # score = 0.5 * 0.80 + 0.5 * 0.70 = 0.750000
    assert score == 0.750000

    # Test case 2: more than 3 body passages (top 3 selected)
    score_4 = compute_relevance_score(paper_sim=0.80, body_sims=[0.90, 0.70, 0.50, 0.10], alpha=0.50)
    assert score_4 == 0.750000

    # Test case 3: fewer than 3 body passages
    score_1 = compute_relevance_score(paper_sim=0.60, body_sims=[0.80], alpha=0.50)
    # top_1 avg = 0.80 -> score = 0.5*0.6 + 0.5*0.8 = 0.700000
    assert score_1 == 0.700000

    # Test case 4: empty body passages falls back to paper_sim
    score_empty = compute_relevance_score(paper_sim=0.623456, body_sims=[], alpha=0.50)
    assert score_empty == 0.623456


# ─────────────────────────────────────────────────────────────────────────────
# 2. Title/abstract and body passage contributions
# ─────────────────────────────────────────────────────────────────────────────

def test_title_abstract_and_body_contributions():
    """Verifies that both paper-level header (title/abstract) and body passages contribute."""
    # Paper X: high paper-level (0.9), low body (0.3)
    score_x = compute_relevance_score(paper_sim=0.90, body_sims=[0.30, 0.30, 0.30], alpha=0.50)
    # 0.5 * 0.90 + 0.5 * 0.30 = 0.60
    assert score_x == 0.600000

    # Paper Y: low paper-level (0.3), high body (0.9)
    score_y = compute_relevance_score(paper_sim=0.30, body_sims=[0.90, 0.90, 0.90], alpha=0.50)
    # 0.5 * 0.30 + 0.5 * 0.90 = 0.60
    assert score_y == 0.600000

    # Changing alpha weights one over the other
    score_x_paper_heavy = compute_relevance_score(paper_sim=0.90, body_sims=[0.30, 0.30, 0.30], alpha=0.80)
    # 0.8 * 0.90 + 0.2 * 0.30 = 0.72 + 0.06 = 0.78
    assert score_x_paper_heavy == 0.780000


# ─────────────────────────────────────────────────────────────────────────────
# 3. Excluding references from scoring
# ─────────────────────────────────────────────────────────────────────────────

def test_excluding_references_from_scoring():
    """Passages with reference/bibliography sections must not be included in body passage scoring."""
    class MockPassage:
        def __init__(self, section: str, text: str):
            self.id = uuid.uuid4()
            self.section = section
            self.text = text
            self.passage_index = 0
            self.page_start = 1
            self.page_end = 1

    ref_pass = MockPassage("References", "ref_distractor 1. High scoring citation")
    bib_pass = MockPassage("Bibliography", "ref_distractor 2. Second citation")
    body_pass = MockPassage("Section 2 Methods", "low_sim classical methods")

    assert is_reference_passage(ref_pass) is True
    assert is_reference_passage(bib_pass) is True
    assert is_reference_passage(body_pass) is False

    embedder = MockTopicEmbeddingProvider()
    service = RelevanceScreeningService(embedding_provider=embedder, threshold=0.50)

    class MockPaper:
        id = uuid.uuid4()
        title = "low_sim Classical Network Routing"
        abstract = "low_sim Abstract about packet queues"
        publication_date = datetime.date(2021, 1, 1)

    # Paper has low_sim body, but distractor reference with high similarity
    candidates = [(MockPaper(), [ref_pass, bib_pass, body_pass], "test_paper.pdf")]
    report = service.screen_papers(topic="quantum error correction", candidates=candidates)

    res = report.results[0]
    # Distractor reference (0.99) should NOT have inflated the score above 0.50
    assert res.score < 0.50
    assert res.decision == "EXCLUDED_BELOW_THRESHOLD"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Threshold equality behavior
# ─────────────────────────────────────────────────────────────────────────────

def test_threshold_equality_behavior():
    """Score exactly equal to threshold must be selected (score >= threshold)."""
    embedder = MockTopicEmbeddingProvider()
    service = RelevanceScreeningService(embedding_provider=embedder, threshold=0.50, min_papers=0)

    class MockPaperExact:
        id = uuid.uuid4()
        title = "threshold_exact title"
        abstract = "threshold_exact abstract"
        publication_date = datetime.date(2021, 1, 1)

    class MockPassageExact:
        id = uuid.uuid4()
        section = "Section 1"
        text = "threshold_exact body"
        passage_index = 0
        page_start = 1
        page_end = 1

    candidates = [(MockPaperExact(), [MockPassageExact()], "exact.pdf")]
    report = service.screen_papers(topic="quantum error correction", candidates=candidates)

    res = report.results[0]
    assert res.score == 0.500000
    assert res.decision == "SELECTED_ABOVE_THRESHOLD"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Capacity-based exclusions
# ─────────────────────────────────────────────────────────────────────────────

def test_capacity_based_exclusions():
    """When qualifying papers exceed max_papers, lower-ranked papers are EXCLUDED_CAPACITY_LIMIT."""
    embedder = MockTopicEmbeddingProvider()
    service = RelevanceScreeningService(
        embedding_provider=embedder,
        threshold=0.50,
        min_papers=1,
        max_papers=1,  # Capacity of 1
    )

    class MockPaper1:
        id = uuid.uuid4()
        title = "high_sim Paper 1"
        abstract = "high_sim abstract"
        publication_date = datetime.date(2021, 1, 1)

    class MockPaper2:
        id = uuid.uuid4()
        title = "med_sim Paper 2"
        abstract = "med_sim abstract"
        publication_date = datetime.date(2021, 2, 1)

    class MockPassage:
        id = uuid.uuid4()
        section = "Section 1"
        text = "body text"
        passage_index = 0
        page_start = 1
        page_end = 1

    candidates = [
        (MockPaper1(), [MockPassage()], "paper_1.pdf"),
        (MockPaper2(), [MockPassage()], "paper_2.pdf"),
    ]
    report = service.screen_papers(topic="quantum error correction", candidates=candidates)

    assert report.selected_count == 1
    assert report.excluded_count == 1
    assert report.results[0].decision == "SELECTED_ABOVE_THRESHOLD"
    assert report.results[1].decision == "EXCLUDED_CAPACITY_LIMIT"
    assert "max_papers" in report.results[1].reason
    assert report.results[1].temporal_workflow_note is not None


# ─────────────────────────────────────────────────────────────────────────────
# 6. Minimum-selection floor and warnings
# ─────────────────────────────────────────────────────────────────────────────

def test_minimum_selection_floor_and_warnings():
    """When fewer papers qualify than min_papers, floor pulls lower-scoring papers and warns."""
    embedder = MockTopicEmbeddingProvider()
    service = RelevanceScreeningService(
        embedding_provider=embedder,
        threshold=0.70,  # High threshold
        min_papers=2,    # Floor requires 2
    )

    class MockPaperHigh:
        id = uuid.uuid4()
        title = "high_sim Paper"
        abstract = "high_sim abstract"
        publication_date = datetime.date(2021, 1, 1)

    class MockPaperLow:
        id = uuid.uuid4()
        title = "low_sim Paper"
        abstract = "low_sim abstract"
        publication_date = datetime.date(2021, 2, 1)

    class MockPassage:
        def __init__(self, text: str = "high_sim body"):
            self.id = uuid.uuid4()
            self.section = "Section 1"
            self.text = text
            self.passage_index = 0
            self.page_start = 1
            self.page_end = 1

    candidates = [
        (MockPaperHigh(), [MockPassage("high_sim body")], "high.pdf"),
        (MockPaperLow(), [MockPassage("low_sim body")], "low.pdf"),
    ]
    report = service.screen_papers(topic="quantum error correction", candidates=candidates)

    assert report.selected_count == 2
    assert report.floor_warning is not None
    assert "Minimum selection floor" in report.floor_warning

    assert report.results[0].decision == "SELECTED_ABOVE_THRESHOLD"
    assert report.results[0].floor_included is False

    assert report.results[1].decision == "SELECTED_BELOW_THRESHOLD_FLOOR"
    assert report.results[1].floor_included is True


# ─────────────────────────────────────────────────────────────────────────────
# 7. Zero eligible papers with minimum zero fails closed
# ─────────────────────────────────────────────────────────────────────────────

def test_zero_eligible_papers_with_minimum_zero_fails_closed(tmp_path: Path):
    """When 0 papers qualify and min_papers == 0, runner halts before Stage 2 LLM extraction."""
    corpus_dir = tmp_path / "corpus_zero"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    p = corpus_dir / "classical.pdf"
    _create_synthetic_pdf(p, "Classical routing in network switches. Packet buffer management.")
    manifest = [{"filename": "classical.pdf", "title": "Classical Routing", "publication_date": "2021-01-01"}]
    with open(corpus_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    embedder = MockTopicEmbeddingProvider()
    llm = TrackingMockLLM()

    runner = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=tmp_path / "artifacts",
        db_path=":memory:",
        screening_topic="quantum error correction",
        screening_threshold=0.90,  # Very high, classical.pdf will get 0.20
        screening_min_papers=0,    # min_papers 0 allows 0 qualifying
    )

    result = runner.run(corpus_dir)
    assert result.success is False
    assert "Relevance screening qualification failure: 0 papers qualified" in result.error
    assert llm.understanding_calls == 0


# ─────────────────────────────────────────────────────────────────────────────
# 8. Stable tie-breaking under shuffled input
# ─────────────────────────────────────────────────────────────────────────────

def test_stable_tie_breaking_under_shuffled_input():
    """Papers with identical scores must break ties deterministically by filename ascending."""
    embedder = MockTopicEmbeddingProvider()
    service = RelevanceScreeningService(embedding_provider=embedder, threshold=0.10)

    class MockPaper:
        def __init__(self, pid: uuid.UUID):
            self.id = pid
            self.title = "low_sim Identical Title"
            self.abstract = "low_sim Identical Abstract"
            self.publication_date = datetime.date(2021, 1, 1)

    class MockPassage:
        id = uuid.uuid4()
        section = "Section"
        text = "low_sim Identical Body"
        passage_index = 0
        page_start = 1
        page_end = 1

    p1 = (MockPaper(uuid.UUID("00000000-0000-0000-0000-000000000001")), [MockPassage()], "zeta.pdf")
    p2 = (MockPaper(uuid.UUID("00000000-0000-0000-0000-000000000002")), [MockPassage()], "alpha.pdf")
    p3 = (MockPaper(uuid.UUID("00000000-0000-0000-0000-000000000003")), [MockPassage()], "beta.pdf")

    # Order 1: zeta, alpha, beta
    rep1 = service.screen_papers("quantum error correction", [p1, p2, p3])
    filenames_1 = [r.filename for r in rep1.results]

    # Order 2: beta, zeta, alpha
    rep2 = service.screen_papers("quantum error correction", [p3, p1, p2])
    filenames_2 = [r.filename for r in rep2.results]

    # Both must rank identically: alpha.pdf, beta.pdf, zeta.pdf
    assert filenames_1 == ["alpha.pdf", "beta.pdf", "zeta.pdf"]
    assert filenames_2 == ["alpha.pdf", "beta.pdf", "zeta.pdf"]


# ─────────────────────────────────────────────────────────────────────────────
# 9. Excluded papers remain in storage
# ─────────────────────────────────────────────────────────────────────────────

def test_excluded_papers_remaining_in_storage(multi_paper_corpus: Path, tmp_path: Path):
    """Screening excludes papers from Stage 2 extraction without deleting them from database."""
    db_file = tmp_path / "state.db"
    embedder = MockTopicEmbeddingProvider()
    llm = TrackingMockLLM()

    runner = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=tmp_path / "artifacts",
        db_path=db_file,
        screening_topic="quantum error correction",
        screening_threshold=0.50,
        screening_min_papers=1,
    )

    result = runner.run(multi_paper_corpus)
    assert result.success is True

    # Check database storage
    engine = create_engine(f"sqlite:///{db_file}")
    with Session(engine) as session:
        papers = session.query(Paper).all()
        # All 4 papers from corpus must exist in DB!
        assert len(papers) == 4
        # Passages for all 4 papers exist in DB
        passages = session.query(PaperPassage).all()
        assert len(passages) >= 4

        # Paper C (classical routing) was excluded, so no statements were extracted for it
        p_c = session.query(Paper).filter(Paper.title.contains("Classical Routing")).first()
        stmts_c = session.query(ResearchStatement).filter(ResearchStatement.paper_id == p_c.id).all()
        assert len(stmts_c) == 0

        # Paper A (QEC) was selected, so statements exist
        p_a = session.query(Paper).filter(Paper.title.contains("Fault-Tolerant Quantum")).first()
        stmts_a = session.query(ResearchStatement).filter(ResearchStatement.paper_id == p_a.id).all()
        assert len(stmts_a) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 10. Post-cutoff papers remain available to Stage 5
# ─────────────────────────────────────────────────────────────────────────────

def test_post_cutoff_papers_remaining_available_to_stage_5(multi_paper_corpus: Path, tmp_path: Path):
    """With temporal_cutoff, post-cutoff papers are not screened for discovery but remain for Stage 5."""
    db_file = tmp_path / "state.db"
    embedder = MockTopicEmbeddingProvider()
    llm = TrackingMockLLM()

    runner = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=tmp_path / "artifacts",
        db_path=db_file,
        temporal_cutoff="2022-01-01",  # Papers A, B, C are pre-cutoff; Paper D is 2023-01-15 (post-cutoff)
        screening_topic="quantum error correction",
        screening_threshold=0.50,
        screening_min_papers=1,
        enable_stage_5=True,
    )

    result = runner.run(multi_paper_corpus)
    assert result.success is True

    # Paper D was post-cutoff -> not screened for discovery
    rs_metrics = result.metrics["relevance_screening"]
    ranked_filenames = [r["filename"] for r in rs_metrics["ranked_papers"]]
    assert "paper_d.pdf" not in ranked_filenames
    assert result.metrics["paper_statuses"]["paper_d.pdf"] == "POST_CUTOFF_RETAINED_FOR_STAGE_5"

    # Stage 5 evaluated paper D
    s5_metrics = result.metrics.get("stage_5_evolution", {})
    assert s5_metrics.get("enabled") is True
    assert s5_metrics.get("batches_evaluated_count", 0) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 11. Fingerprint mismatch when topic or settings change
# ─────────────────────────────────────────────────────────────────────────────

def test_fingerprint_mismatch_when_topic_or_settings_change(multi_paper_corpus: Path, tmp_path: Path):
    """Resuming with a changed screening topic or threshold must refuse unsafe resume."""
    db_file = tmp_path / "state.db"
    embedder = MockTopicEmbeddingProvider()
    llm = TrackingMockLLM()

    # Run 1: initial settings
    runner1 = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=tmp_path / "artifacts",
        db_path=db_file,
        screening_topic="quantum error correction",
        screening_threshold=0.50,
        screening_min_papers=1,
    )
    res1 = runner1.run(multi_paper_corpus)
    assert res1.success is True

    # Run 2: changed topic
    runner2 = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=tmp_path / "artifacts",
        db_path=db_file,
        screening_topic="condensed matter superconductors",  # Different topic!
        screening_threshold=0.50,
        screening_min_papers=1,
    )
    res2 = runner2.run(multi_paper_corpus)
    assert res2.success is False
    assert "Configuration mismatch detected against existing database" in res2.error
    assert any("Relevance screening topic mismatch" in m for m in res2.metrics.get("configuration_mismatches", []))

    # Run 3: changed threshold
    runner3 = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=tmp_path / "artifacts",
        db_path=db_file,
        screening_topic="quantum error correction",
        screening_threshold=0.75,  # Different threshold!
        screening_min_papers=1,
    )
    res3 = runner3.run(multi_paper_corpus)
    assert res3.success is False
    assert any("Relevance screening threshold mismatch" in m for m in res3.metrics.get("configuration_mismatches", []))


# ─────────────────────────────────────────────────────────────────────────────
# 12. Reuse of screening results when inputs are unchanged
# ─────────────────────────────────────────────────────────────────────────────

def test_reuse_of_screening_results_when_inputs_unchanged(multi_paper_corpus: Path, tmp_path: Path):
    """When resuming with identical corpus and screening settings, prior decisions are reused."""
    db_file = tmp_path / "state.db"
    embedder = MockTopicEmbeddingProvider()
    llm = TrackingMockLLM()

    runner1 = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=tmp_path / "artifacts",
        db_path=db_file,
        screening_topic="quantum error correction",
        screening_threshold=0.50,
        screening_min_papers=1,
    )
    res1 = runner1.run(multi_paper_corpus)
    assert res1.success is True

    embed_calls_run1 = embedder.call_count

    # Run 2: exact same config and corpus
    runner2 = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=tmp_path / "artifacts",
        db_path=db_file,
        screening_topic="quantum error correction",
        screening_threshold=0.50,
        screening_min_papers=1,
    )
    res2 = runner2.run(multi_paper_corpus)
    assert res2.success is True
    # Screening report reused from database metadata, no extra embedding calls for screening
    assert embedder.call_count == embed_calls_run1


# ─────────────────────────────────────────────────────────────────────────────
# 13. Embedding failure stops processing before LLM extraction
# ─────────────────────────────────────────────────────────────────────────────

def test_embedding_failure_stops_processing_before_llm_extraction(multi_paper_corpus: Path, tmp_path: Path):
    """Fail-closed: if embedding provider fails, runner terminates before Stage 2 LLM calls."""
    embedder = MockTopicEmbeddingProvider()
    embedder.should_raise = True  # Simulate failure
    llm = TrackingMockLLM()

    runner = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=tmp_path / "artifacts",
        db_path=":memory:",
        screening_topic="quantum error correction",
        screening_threshold=0.50,
    )

    result = runner.run(multi_paper_corpus)
    assert result.success is False
    assert "Relevance screening embedding failure" in result.error
    assert llm.understanding_calls == 0


# ─────────────────────────────────────────────────────────────────────────────
# 14. Screening-only mode makes no LLM calls
# ─────────────────────────────────────────────────────────────────────────────

def test_screening_only_mode_makes_no_llm_calls(multi_paper_corpus: Path, tmp_path: Path):
    """Screening-only mode ranks papers, generates report, and exits before Stage 2 (0 LLM calls)."""
    embedder = MockTopicEmbeddingProvider()
    llm = TrackingMockLLM()

    runner = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=tmp_path / "artifacts",
        db_path=":memory:",
        screening_topic="quantum error correction",
        screening_threshold=0.50,
        screening_only=True,
    )

    result = runner.run(multi_paper_corpus)
    assert result.success is True
    assert llm.understanding_calls == 0
    assert result.metrics["understanding"]["status"] == "SKIPPED_SCREENING_ONLY"
    assert result.metrics["discovery"]["status"] == "SKIPPED_SCREENING_ONLY"
    assert result.metrics["relevance_screening"]["enabled"] is True
    assert result.summary_file.exists()


# ─────────────────────────────────────────────────────────────────────────────
# 15. No topic preserves previous pipeline behavior
# ─────────────────────────────────────────────────────────────────────────────

def test_no_topic_preserves_previous_pipeline_behavior(multi_paper_corpus: Path, tmp_path: Path):
    """When no screening topic is provided, screening is disabled and all ingested papers are extracted."""
    embedder = MockTopicEmbeddingProvider()
    llm = TrackingMockLLM()

    runner = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=tmp_path / "artifacts",
        db_path=":memory:",
        screening_topic=None,  # Disabled
    )

    result = runner.run(multi_paper_corpus)
    assert result.success is True
    assert result.metrics["relevance_screening"]["enabled"] is False
    # All 4 papers processed by understanding
    assert result.metrics["understanding"]["understood_count"] == 4


# ─────────────────────────────────────────────────────────────────────────────
# 16. Stage 5 contradictory and extension evidence not filtered
# ─────────────────────────────────────────────────────────────────────────────

def test_stage5_contradictory_and_extension_evidence_not_filtered(multi_paper_corpus: Path, tmp_path: Path):
    """Stage 5 evaluates contradictory and extension relationships without similarity filtering."""
    db_file = tmp_path / "state.db"
    embedder = MockTopicEmbeddingProvider()
    llm = TrackingMockLLM(
        relationship=OpportunityRelationship.CONTRADICTS,
        evidence_excerpt="high_sim teleportation protocol published after cutoff",
    )

    runner = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=tmp_path / "artifacts",
        db_path=db_file,
        temporal_cutoff="2022-01-01",
        screening_topic="quantum error correction",
        screening_threshold=0.50,
        screening_min_papers=1,
        enable_stage_5=True,
    )

    result = runner.run(multi_paper_corpus)
    assert result.success is True

    # Check Stage 5 metrics
    s5_metrics = result.metrics.get("stage_5_evolution", {})
    assert s5_metrics.get("status") in ("COMPLETED", "SUCCESS")
    assert s5_metrics.get("classifier_input_limits") is not None
    assert s5_metrics.get("total_events_count", 0) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 17. Checkpoint 8: Invalidation when paper membership changes
# ─────────────────────────────────────────────────────────────────────────────

def test_screening_invalidation_when_paper_added(tmp_path: Path):
    """Adding a new paper to the corpus must invalidate prior screening decisions and recompute."""
    corpus_dir = tmp_path / "corpus_dynamic"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    db_file = tmp_path / "state.db"

    # Initial corpus: 2 papers
    p_a = corpus_dir / "paper_a.pdf"
    p_b = corpus_dir / "paper_b.pdf"
    _create_synthetic_pdf(p_a, "Paper A: high_sim quantum syndrome decoding.")
    _create_synthetic_pdf(p_b, "Paper B: low_sim classical routing protocols.")
    manifest_initial = [
        {"filename": "paper_a.pdf", "title": "Paper A", "publication_date": "2021-01-01"},
        {"filename": "paper_b.pdf", "title": "Paper B", "publication_date": "2021-02-01"},
    ]
    with open(corpus_dir / "manifest.json", "w") as f:
        json.dump(manifest_initial, f)

    embedder = MockTopicEmbeddingProvider()
    llm = TrackingMockLLM()

    # Run 1
    runner1 = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=tmp_path / "artifacts1",
        db_path=db_file,
        screening_topic="quantum error correction",
        screening_threshold=0.50,
        screening_min_papers=1,
        screening_only=True,
    )
    res1 = runner1.run(corpus_dir)
    assert res1.success is True
    assert res1.metrics["relevance_screening"]["total_eligible_papers"] == 2

    # Verify input_signature is present in stored report
    engine = create_engine(f"sqlite:///{db_file}")
    with Session(engine) as session:
        meta = session.query(ExperimentMetadataRecord).filter(
            ExperimentMetadataRecord.key == "relevance_screening_report"
        ).first()
        stored_dict = json.loads(meta.value)
        assert "input_signature" in stored_dict
        assert stored_dict["input_signature"]["candidate_count"] == 2

    call_count_after_run1 = embedder.call_count

    # Add Paper C (moderately relevant QEC paper) to corpus
    p_c = corpus_dir / "paper_c.pdf"
    _create_synthetic_pdf(p_c, "Paper C: med_sim surface code lattices.")
    manifest_updated = manifest_initial + [
        {"filename": "paper_c.pdf", "title": "Paper C", "publication_date": "2021-03-01"}
    ]
    with open(corpus_dir / "manifest.json", "w") as f:
        json.dump(manifest_updated, f)

    # Run 2: Resume with added paper
    runner2 = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=tmp_path / "artifacts2",
        db_path=db_file,
        screening_topic="quantum error correction",
        screening_threshold=0.50,
        screening_min_papers=1,
        screening_only=True,
    )
    res2 = runner2.run(corpus_dir)
    assert res2.success is True

    # Screening must have recomputed and evaluated all 3 papers!
    rs_metrics2 = res2.metrics["relevance_screening"]
    assert rs_metrics2["total_eligible_papers"] == 3
    ranked_files = [r["filename"] for r in rs_metrics2["ranked_papers"]]
    assert "paper_c.pdf" in ranked_files
    # Embedder was called again to compute fresh rankings
    assert embedder.call_count > call_count_after_run1


# ─────────────────────────────────────────────────────────────────────────────
# 18. Checkpoint 8: Invalidation when passage content changes
# ─────────────────────────────────────────────────────────────────────────────

def test_screening_invalidation_when_passage_content_changes(tmp_path: Path):
    """Modifying a candidate paper's passage text must invalidate the screening signature."""
    corpus_dir = tmp_path / "corpus_passage"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    db_file = tmp_path / "state.db"

    p_a = corpus_dir / "paper_a.pdf"
    _create_synthetic_pdf(p_a, "Paper A: low_sim classical network switches.")
    manifest = [{"filename": "paper_a.pdf", "title": "Paper A", "publication_date": "2021-01-01"}]
    with open(corpus_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    embedder = MockTopicEmbeddingProvider()
    llm = TrackingMockLLM()

    runner1 = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=tmp_path / "artifacts1",
        db_path=db_file,
        screening_topic="quantum error correction",
        screening_threshold=0.50,
        screening_min_papers=1,
    )
    res1 = runner1.run(corpus_dir)
    assert res1.success is True
    initial_score = res1.metrics["relevance_screening"]["ranked_papers"][0]["score"]

    # Directly mutate the passage in DB to include high_sim content
    engine = create_engine(f"sqlite:///{db_file}")
    with Session(engine) as session:
        passage = session.query(PaperPassage).first()
        passage.text = "Paper A modified: high_sim quantum syndrome extraction and error correction."
        session.commit()

    call_count_after_mutation = embedder.call_count

    # Resume: passage text hash changes -> signature mismatch -> recomputed
    runner2 = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=tmp_path / "artifacts2",
        db_path=db_file,
        screening_topic="quantum error correction",
        screening_threshold=0.50,
        screening_min_papers=1,
    )
    res2 = runner2.run(corpus_dir)
    assert res2.success is True
    new_score = res2.metrics["relevance_screening"]["ranked_papers"][0]["score"]

    # Must have recomputed with higher score
    assert embedder.call_count > call_count_after_mutation
    assert new_score > initial_score


# ─────────────────────────────────────────────────────────────────────────────
# 19. Checkpoint 8: Malformed metadata handled safely without trusting old data
# ─────────────────────────────────────────────────────────────────────────────

def test_malformed_screening_metadata_recomputed_safely(tmp_path: Path):
    """Corrupted or malformed screening metadata must be handled safely by recomputing."""
    corpus_dir = tmp_path / "corpus_corrupt"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    db_file = tmp_path / "state.db"

    p_a = corpus_dir / "paper_a.pdf"
    _create_synthetic_pdf(p_a, "Paper A: high_sim quantum decoding.")
    manifest = [{"filename": "paper_a.pdf", "title": "Paper A", "publication_date": "2021-01-01"}]
    with open(corpus_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    embedder = MockTopicEmbeddingProvider()
    llm = TrackingMockLLM()

    # Pre-populate DB with malformed screening metadata
    engine = create_engine(f"sqlite:///{db_file}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(
            ExperimentMetadataRecord(
                key="relevance_screening_report",
                value='{"corrupted_json": [missing_brackets',
            )
        )
        session.commit()

    runner = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=tmp_path / "artifacts",
        db_path=db_file,
        screening_topic="quantum error correction",
        screening_threshold=0.50,
        screening_min_papers=1,
    )
    result = runner.run(corpus_dir)
    assert result.success is True
    assert result.metrics["relevance_screening"]["total_eligible_papers"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 20. Checkpoint 8: Inconsistent candidate IDs in metadata recomputed safely
# ─────────────────────────────────────────────────────────────────────────────

def test_inconsistent_screening_metadata_paper_id_mismatch_recomputed(tmp_path: Path):
    """Stored screening report referencing nonexistent paper IDs must not be reused."""
    corpus_dir = tmp_path / "corpus_inconsistent"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    db_file = tmp_path / "state.db"

    p_a = corpus_dir / "paper_a.pdf"
    _create_synthetic_pdf(p_a, "Paper A: high_sim quantum decoding.")
    manifest = [{"filename": "paper_a.pdf", "title": "Paper A", "publication_date": "2021-01-01"}]
    with open(corpus_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    embedder = MockTopicEmbeddingProvider()
    llm = TrackingMockLLM()

    # Pre-populate DB with screening report referencing a random foreign paper_id
    foreign_id = str(uuid.uuid4())
    bogus_report = {
        "enabled": True,
        "topic": "quantum error correction",
        "threshold": 0.50,
        "min_papers": 1,
        "max_papers": None,
        "total_eligible_papers": 1,
        "selected_count": 1,
        "excluded_count": 0,
        "ranked_papers": [
            {
                "paper_id": foreign_id,
                "filename": "foreign.pdf",
                "score": 0.99,
                "decision": "SELECTED_ABOVE_THRESHOLD",
                "reason": "Test",
            }
        ],
    }

    engine = create_engine(f"sqlite:///{db_file}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(
            ExperimentMetadataRecord(
                key="relevance_screening_report",
                value=json.dumps(bogus_report),
            )
        )
        session.commit()

    runner = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=tmp_path / "artifacts",
        db_path=db_file,
        screening_topic="quantum error correction",
        screening_threshold=0.50,
        screening_min_papers=1,
    )
    result = runner.run(corpus_dir)
    assert result.success is True

    # Bogus foreign paper must NOT be trusted; paper_a must be ranked
    rs_metrics = result.metrics["relevance_screening"]
    ranked_ids = [r["paper_id"] for r in rs_metrics["ranked_papers"]]
    assert foreign_id not in ranked_ids
    assert rs_metrics["ranked_papers"][0]["filename"] == "paper_a.pdf"


# ─────────────────────────────────────────────────────────────────────────────
# 21. Checkpoint 8: Fingerprint detects changed embedding provider
# ─────────────────────────────────────────────────────────────────────────────

def test_fingerprint_detects_changed_embedding_provider():
    """Fingerprint comparison must catch changes in embedding_provider."""
    fp1 = {
        "llm_provider": "MockLLM",
        "llm_model": "qwen3:8b",
        "embedding_provider": "MockTopicEmbeddingProvider",
        "embedding_model": "nomic-embed-text",
        "embedding_dimension": 768,
        "relevance_screening": {"enabled": True, "topic": "QEC", "threshold": 0.5, "min_papers": 0, "max_papers": None},
    }
    fp2 = dict(fp1)
    fp2["embedding_provider"] = "OllamaEmbeddingProvider"

    mismatches = compare_fingerprints(fp1, fp2)
    assert any("Embedding provider mismatch" in m for m in mismatches)


# ─────────────────────────────────────────────────────────────────────────────
# 22. Checkpoint 8: Legacy screening report without signature accepted if matching
# ─────────────────────────────────────────────────────────────────────────────

def test_legacy_screening_report_without_signature_accepted_when_matching(tmp_path: Path):
    """Legacy screening report without input_signature is accepted if candidate membership matches."""
    corpus_dir = tmp_path / "corpus_legacy"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    db_file = tmp_path / "state.db"

    p_a = corpus_dir / "paper_a.pdf"
    _create_synthetic_pdf(p_a, "Paper A: high_sim quantum decoding.")
    manifest = [{"filename": "paper_a.pdf", "title": "Paper A", "publication_date": "2021-01-01"}]
    with open(corpus_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    embedder = MockTopicEmbeddingProvider()
    llm = TrackingMockLLM()

    # Ingest paper A first
    runner1 = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=tmp_path / "artifacts1",
        db_path=db_file,
        screening_topic="quantum error correction",
        screening_threshold=0.50,
        screening_min_papers=1,
    )
    res1 = runner1.run(corpus_dir)
    assert res1.success is True

    # Strip the input_signature to simulate legacy report
    engine = create_engine(f"sqlite:///{db_file}")
    with Session(engine) as session:
        meta = session.query(ExperimentMetadataRecord).filter(
            ExperimentMetadataRecord.key == "relevance_screening_report"
        ).first()
        data = json.loads(meta.value)
        data.pop("input_signature", None)
        meta.value = json.dumps(data)
        session.commit()

    calls_before = embedder.call_count

    # Resume with matching candidate membership
    runner2 = ControlledExperimentRunner(
        llm_provider=llm,
        embedding_provider=embedder,
        artifacts_dir=tmp_path / "artifacts2",
        db_path=db_file,
        screening_topic="quantum error correction",
        screening_threshold=0.50,
        screening_min_papers=1,
    )
    res2 = runner2.run(corpus_dir)
    assert res2.success is True
    # Reused without recomputing
    assert embedder.call_count == calls_before



def test_screening_signature_changes_when_scored_abstract_changes():
    """The paper header embedded during scoring includes the bounded abstract, so the reuse
    signature must change when that abstract changes (from either source)."""
    import datetime as _dt
    from types import SimpleNamespace
    from ideagen.discovery.screening import compute_screening_input_signature

    def passage(idx, section, text):
        return SimpleNamespace(passage_index=idx, section=section, text=text)

    def signature(paper, passages):
        return compute_screening_input_signature(
            candidates=[(paper, passages, "paper.pdf")],
            topic="retrieval grounding", threshold=0.5, min_papers=1, max_papers=None,
            embedding_provider_name="Mock", embedding_model="mock", embedding_dimension=16,
        )["signature_hash"]

    pid = uuid.uuid4()
    body = [passage(2, "Methods", "Body text that does not change.")]

    def paper(abstract):
        return SimpleNamespace(id=pid, title="Same title", publication_date=_dt.date(2020, 1, 1), abstract=abstract)

    # 1. Abstract taken from an abstract-section passage.
    sig_a = signature(paper("unused"), [passage(1, "Abstract", "We study grounding in RAG.")] + body)
    sig_b = signature(paper("unused"), [passage(1, "Abstract", "We study citation recall in RAG.")] + body)
    assert sig_a != sig_b

    # 2. No abstract passage: the stored paper abstract is the header source.
    sig_c = signature(paper("Original abstract text."), body)
    sig_d = signature(paper("Revised abstract text."), body)
    assert sig_c != sig_d

    # Identical inputs remain stable.
    assert sig_c == signature(paper("Original abstract text."), body)
