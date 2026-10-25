"""Deterministic offline tests for Stage 5 Classifier Input Bounds and Passage Budgeting.

Verifies:
1. Extremely long abstracts are bounded deterministically.
2. A real abstract-section passage is preferred over Paper.abstract when reliably available.
3. Configurable passage and word budgets are enforced on PassageSelector.
4. Resolved classifier input settings appear in the fingerprint and experiment metrics.
5. Changing a setting prevents incompatible resume (fingerprint mismatch) and legacy fingerprints default safely.
6. Evidence grounding and provenance remain intact under bounded classifier input.
7. CONTRADICTS and EXTENDS pairs are evaluated without pre-filtering.
8. Chronological cutoff and duplicate pair skipping remain intact.
"""

from __future__ import annotations

import datetime
from pathlib import Path
import sys
import uuid
from typing import Any, Dict, List, Optional

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.runner import (
    ControlledExperimentRunner,
    compute_configuration_fingerprint,
    compare_fingerprints,
)
from ideagen.db.models import (
    Base,
    Paper,
    PaperDocument,
    PaperPassage,
    Opportunity,
    OpportunityEvidence,
    OpportunityHistory,
    ResearchOpportunityRecord,
)
from ideagen.opportunity.classifier import (
    RelationshipClassifier,
    RelationshipClassification,
    RelationshipClassificationInput,
)
from ideagen.opportunity.grounding import UngroundedEvidenceError
from ideagen.opportunity.integration import (
    OpportunityClassificationIntegration,
    resolve_paper_abstract,
    DEFAULT_STAGE_5_MAX_PASSAGES,
    DEFAULT_STAGE_5_MAX_TOTAL_WORDS,
    DEFAULT_STAGE_5_MAX_ABSTRACT_CHARS,
)
from ideagen.opportunity.models import OpportunityRelationship, OpportunityStatus
from ideagen.opportunity.service import OpportunityTransitionService
from ideagen.opportunity.temporal import TemporalBatch, OpportunityEvolutionService
from tests.experiments.test_controlled_experiment_runner import (
    DeterministicExperimentEmbeddingProvider,
    _create_synthetic_pdf,
)
from tests.experiments.test_stage5_evolution_runner import Stage5MockLLMProvider


@pytest.fixture
def sqlite_session():
    """Isolated in-memory SQLite database session."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


class RecordingMockClassifier(RelationshipClassifier):
    """Mock classifier that records received inputs for prompt bounding verification."""

    def __init__(
        self,
        relationship: OpportunityRelationship = OpportunityRelationship.SUPPORTS,
        evidence_excerpt: Optional[str] = None,
        confidence: float = 0.9,
    ):
        self.recorded_inputs: List[RelationshipClassificationInput] = []
        self.return_relationship = relationship
        self.return_excerpt = evidence_excerpt
        self.return_confidence = confidence

    def classify(self, input_data: RelationshipClassificationInput) -> RelationshipClassification:
        self.recorded_inputs.append(input_data)
        return RelationshipClassification(
            relationship=self.return_relationship,
            confidence=self.return_confidence,
            rationale="Deterministic test classification rationale.",
            evidence_excerpt=self.return_excerpt,
            prompt_version="v1",
        )


def _make_paper(
    session: Session,
    title: str = "Test Paper",
    abstract: str = "Test abstract",
    publication_date: Optional[datetime.date] = None,
) -> Paper:
    paper = Paper(
        id=uuid.uuid4(),
        source_type="pdf",
        source_id=str(uuid.uuid4()),
        batch_id="test_batch",
        title=title,
        abstract=abstract,
        publication_date=publication_date or datetime.date(2023, 1, 1),
        authors=[],
        referenced_works=[],
        raw_metadata={},
    )
    session.add(paper)
    session.flush()
    return paper


def test_resolve_paper_abstract_bounds_extremely_long_abstract():
    """Verify Paper.abstract is deterministically truncated when it exceeds max_abstract_chars."""
    paper = Paper(
        id=uuid.uuid4(),
        source_type="pdf",
        source_id="s1",
        title="Long Abstract Paper",
        abstract="A" * 10000,
        publication_date=datetime.date(2023, 1, 1),
    )

    bounded = resolve_paper_abstract(paper, stored_passages=[], max_abstract_chars=500)
    assert len(bounded) == 500
    assert bounded == "A" * 500

    # Default character budget (2000)
    default_bounded = resolve_paper_abstract(paper, stored_passages=[])
    assert len(default_bounded) == 2000
    assert default_bounded == "A" * 2000


def test_resolve_paper_abstract_prefers_real_abstract_passage():
    """Verify real abstract section passages from stored passages take precedence over paper.abstract."""
    paper = Paper(
        id=uuid.uuid4(),
        source_type="pdf",
        source_id="s2",
        title="Real Abstract Test Paper",
        abstract="This is the unverified or raw paper metadata abstract.",
        publication_date=datetime.date(2023, 2, 1),
    )

    doc_id = uuid.uuid4()
    p1 = PaperPassage(
        id=uuid.uuid4(),
        paper_id=paper.id,
        document_id=doc_id,
        passage_index=0,
        page_start=1,
        page_end=1,
        section="Abstract",
        text="Real authoritative abstract passage chunk 1.",
    )
    p2 = PaperPassage(
        id=uuid.uuid4(),
        paper_id=paper.id,
        document_id=doc_id,
        passage_index=1,
        page_start=1,
        page_end=1,
        section="1. Abstract",
        text="Real authoritative abstract passage chunk 2.",
    )
    other = PaperPassage(
        id=uuid.uuid4(),
        paper_id=paper.id,
        document_id=doc_id,
        passage_index=2,
        page_start=2,
        page_end=2,
        section="Introduction",
        text="Introduction text.",
    )

    resolved = resolve_paper_abstract(
        paper=paper,
        stored_passages=[p2, other, p1],  # unordered input to verify deterministic sorting
        max_abstract_chars=2000,
    )

    expected = "Real authoritative abstract passage chunk 1.\n\nReal authoritative abstract passage chunk 2."
    assert resolved == expected


def test_stage5_passage_and_word_budgets_enforced(sqlite_session: Session):
    """Verify configurable max_passages, max_total_words, and max_abstract_chars are enforced."""
    paper = _make_paper(
        session=sqlite_session,
        title="Passage Budget Test Paper",
        abstract="A" * 5000,
        publication_date=datetime.date(2023, 3, 1),
    )

    doc = PaperDocument(
        id=uuid.uuid4(),
        paper_id=paper.id,
        full_text="Sample full text for passage budget test.",
        extraction_method="pymupdf",
        extraction_version="1.0",
        extracted_at=datetime.datetime.now(datetime.timezone.utc),
        content_hash="dummy_hash_1",
        original_filename="test.pdf",
    )
    sqlite_session.add(doc)

    # Create 15 passages of 100 words each
    passages = []
    for i in range(15):
        p = PaperPassage(
            id=uuid.uuid4(),
            paper_id=paper.id,
            document_id=doc.id,
            passage_index=i,
            page_start=i + 1,
            page_end=i + 1,
            section=f"Section {i}",
            text=f"Passage content word {i} " * 20,  # ~80 words
        )
        sqlite_session.add(p)
        passages.append(p)

    opp = Opportunity(
        id=uuid.uuid4(),
        title="Test Opportunity",
        problem_statement="Test Problem Statement",
        description="Test Opportunity Description",
        status=OpportunityStatus.OPEN.value,
    )
    sqlite_session.add(opp)
    sqlite_session.commit()

    mock_classifier = RecordingMockClassifier(
        relationship=OpportunityRelationship.SUPPORTS,
        evidence_excerpt=passages[0].text.strip()[:60],
    )
    transition_svc = OpportunityTransitionService(sqlite_session)

    # Configure strictly: max 3 passages, max 150 words, max 300 abstract chars
    integration = OpportunityClassificationIntegration(
        classifier=mock_classifier,
        transition_service=transition_svc,
        session=sqlite_session,
        max_passages=3,
        max_total_words=150,
        max_abstract_chars=300,
    )

    result = integration.classify_and_apply(
        opportunity_id=opp.id,
        paper_id=paper.id,
    )

    assert result.relationship == OpportunityRelationship.SUPPORTS
    assert len(mock_classifier.recorded_inputs) == 1
    input_data = mock_classifier.recorded_inputs[0]

    # Abstract bound enforced
    assert len(input_data.paper_abstract) <= 300
    assert input_data.paper_abstract == "A" * 300

    # Passage excerpt budget enforced
    assert input_data.paper_excerpt is not None
    excerpt_passages = input_data.paper_excerpt.split("\n\n")
    assert len(excerpt_passages) <= 3
    total_words = sum(len(p.split()) for p in excerpt_passages)
    assert total_words <= 150


def test_fingerprint_includes_stage_5_passage_limits():
    """Verify stage_5_passage_selection appears in fingerprint when Stage 5 is enabled."""
    fp_enabled = compute_configuration_fingerprint(
        llm_provider=None,
        llm_model="test-llm",
        embedding_provider=None,
        embedding_model="test-embed",
        embedding_dimension=64,
        max_passages=12,
        max_total_words=2500,
        extraction_schema_version="v1.0",
        similarity_threshold=0.65,
        min_supporting_evidence=1,
        enable_stage_5=True,
        stage_5_max_passages=7,
        stage_5_max_total_words=1400,
        stage_5_max_abstract_chars=1800,
    )

    assert fp_enabled["enable_stage_5"] is True
    assert fp_enabled["stage_5_passage_selection"] == {
        "max_passages": 7,
        "max_total_words": 1400,
        "max_abstract_chars": 1800,
    }

    fp_disabled = compute_configuration_fingerprint(
        llm_provider=None,
        llm_model="test-llm",
        embedding_provider=None,
        embedding_model="test-embed",
        embedding_dimension=64,
        max_passages=12,
        max_total_words=2500,
        extraction_schema_version="v1.0",
        similarity_threshold=0.65,
        min_supporting_evidence=1,
        enable_stage_5=False,
    )
    assert fp_disabled["stage_5_passage_selection"] is None


def test_changing_stage_5_settings_prevents_incompatible_resume():
    """Verify fingerprint comparison detects mismatches in Stage 5 passage and abstract limits."""
    base_fp = compute_configuration_fingerprint(
        llm_provider=None,
        llm_model="test-llm",
        embedding_provider=None,
        embedding_model="test-embed",
        embedding_dimension=64,
        max_passages=12,
        max_total_words=2500,
        extraction_schema_version="v1.0",
        similarity_threshold=0.65,
        min_supporting_evidence=1,
        enable_stage_5=True,
        stage_5_max_passages=10,
        stage_5_max_total_words=2000,
        stage_5_max_abstract_chars=2000,
    )

    # 1. Matching fingerprint
    matching_fp = dict(base_fp)
    assert compare_fingerprints(base_fp, matching_fp) == []

    # 2. Changed max_passages
    changed_passages = compute_configuration_fingerprint(
        llm_provider=None,
        llm_model="test-llm",
        embedding_provider=None,
        embedding_model="test-embed",
        embedding_dimension=64,
        max_passages=12,
        max_total_words=2500,
        extraction_schema_version="v1.0",
        similarity_threshold=0.65,
        min_supporting_evidence=1,
        enable_stage_5=True,
        stage_5_max_passages=5,
        stage_5_max_total_words=2000,
        stage_5_max_abstract_chars=2000,
    )
    mismatches = compare_fingerprints(base_fp, changed_passages)
    assert any("Stage 5 max_passages mismatch" in m for m in mismatches)

    # 3. Changed max_total_words
    changed_words = compute_configuration_fingerprint(
        llm_provider=None,
        llm_model="test-llm",
        embedding_provider=None,
        embedding_model="test-embed",
        embedding_dimension=64,
        max_passages=12,
        max_total_words=2500,
        extraction_schema_version="v1.0",
        similarity_threshold=0.65,
        min_supporting_evidence=1,
        enable_stage_5=True,
        stage_5_max_passages=10,
        stage_5_max_total_words=1000,
        stage_5_max_abstract_chars=2000,
    )
    mismatches = compare_fingerprints(base_fp, changed_words)
    assert any("Stage 5 max_total_words mismatch" in m for m in mismatches)

    # 4. Changed max_abstract_chars
    changed_abstract = compute_configuration_fingerprint(
        llm_provider=None,
        llm_model="test-llm",
        embedding_provider=None,
        embedding_model="test-embed",
        embedding_dimension=64,
        max_passages=12,
        max_total_words=2500,
        extraction_schema_version="v1.0",
        similarity_threshold=0.65,
        min_supporting_evidence=1,
        enable_stage_5=True,
        stage_5_max_passages=10,
        stage_5_max_total_words=2000,
        stage_5_max_abstract_chars=500,
    )
    mismatches = compare_fingerprints(base_fp, changed_abstract)
    assert any("Stage 5 max_abstract_chars mismatch" in m for m in mismatches)

    # 5. Legacy fingerprint without stage_5_passage_selection: those runs used 10 passages /
    # 2000 words but passed the full, unbounded abstract, so they must not resume under a bound.
    legacy_fp = dict(base_fp)
    del legacy_fp["stage_5_passage_selection"]

    legacy_mismatches = compare_fingerprints(legacy_fp, base_fp)
    assert any("Stage 5 max_abstract_chars mismatch: stored=None, current=2000" in m for m in legacy_mismatches)
    assert not any("max_passages mismatch" in m or "max_total_words mismatch" in m for m in legacy_mismatches)

    # Resuming legacy run with custom non-default limits also reports the passage mismatch
    assert any("Stage 5 max_passages mismatch" in m for m in compare_fingerprints(legacy_fp, changed_passages))


def test_evidence_grounding_and_provenance_intact(sqlite_session: Session):
    """Verify evidence grounding and provenance remain fail-closed under bounded inputs."""
    paper = _make_paper(
        session=sqlite_session,
        title="Grounding Test Paper",
        abstract="Authoritative abstract text about retrieval grounding.",
        publication_date=datetime.date(2023, 4, 1),
    )

    doc = PaperDocument(
        id=uuid.uuid4(),
        paper_id=paper.id,
        full_text="Dense passage retrieval outperforms sparse BM25 across scientific QA tasks.",
        extraction_method="pymupdf",
        extraction_version="1.0",
        extracted_at=datetime.datetime.now(datetime.timezone.utc),
        content_hash="grounding_hash_1",
        original_filename="grounding.pdf",
    )
    sqlite_session.add(doc)

    p1 = PaperPassage(
        id=uuid.uuid4(),
        paper_id=paper.id,
        document_id=doc.id,
        passage_index=0,
        page_start=1,
        page_end=1,
        section="Methods",
        text="Dense passage retrieval outperforms sparse BM25 across scientific QA tasks.",
    )
    sqlite_session.add(p1)

    opp = Opportunity(
        id=uuid.uuid4(),
        title="Grounding Opportunity",
        problem_statement="Grounding problem statement",
        description="Grounding opportunity description",
        status=OpportunityStatus.OPEN.value,
    )
    sqlite_session.add(opp)
    sqlite_session.commit()

    transition_svc = OpportunityTransitionService(sqlite_session)

    # 1. Valid grounded substantive relationship succeeds and records provenance
    valid_classifier = RecordingMockClassifier(
        relationship=OpportunityRelationship.ADDRESSES,
        evidence_excerpt="Dense passage retrieval outperforms sparse BM25 across scientific QA tasks.",
    )
    integration = OpportunityClassificationIntegration(
        classifier=valid_classifier,
        transition_service=transition_svc,
        session=sqlite_session,
    )
    res = integration.classify_and_apply(opportunity_id=opp.id, paper_id=paper.id)
    assert res.relationship == OpportunityRelationship.ADDRESSES
    assert res.new_status == OpportunityStatus.PARTIALLY_ADDRESSED
    assert res.evidence_id is not None

    evidence = sqlite_session.query(OpportunityEvidence).filter(OpportunityEvidence.id == res.evidence_id).one()
    assert evidence.paper_id == paper.id
    assert evidence.opportunity_id == opp.id
    assert "Dense passage retrieval" in evidence.supporting_text

    # 2. Fabricated / ungrounded excerpt fails closed
    ungrounded_classifier = RecordingMockClassifier(
        relationship=OpportunityRelationship.SUPPORTS,
        evidence_excerpt="Hallucinated claim that does not exist anywhere in the paper.",
    )
    integration_ungrounded = OpportunityClassificationIntegration(
        classifier=ungrounded_classifier,
        transition_service=transition_svc,
        session=sqlite_session,
    )
    paper2 = _make_paper(
        session=sqlite_session,
        title="Second Paper",
        abstract="Another abstract.",
        publication_date=datetime.date(2023, 4, 2),
    )

    with pytest.raises(UngroundedEvidenceError):
        integration_ungrounded.classify_and_apply(opportunity_id=opp.id, paper_id=paper2.id)


def test_contradicts_and_extends_pairs_evaluated_without_filtering(sqlite_session: Session):
    """Verify CONTRADICTS and EXTENDS pairs transition opportunity states without being pre-filtered."""
    paper = _make_paper(
        session=sqlite_session,
        title="Evaluation Paper",
        abstract="Detailed abstract text about contradictory findings in RAG models.",
        publication_date=datetime.date(2023, 5, 1),
    )

    doc = PaperDocument(
        id=uuid.uuid4(),
        paper_id=paper.id,
        full_text="Empirical results directly contradict previous assumptions regarding retriever scaling.",
        extraction_method="pymupdf",
        extraction_version="1.0",
        extracted_at=datetime.datetime.now(datetime.timezone.utc),
        content_hash="eval_hash_1",
        original_filename="eval.pdf",
    )
    sqlite_session.add(doc)

    p1 = PaperPassage(
        id=uuid.uuid4(),
        paper_id=paper.id,
        document_id=doc.id,
        passage_index=0,
        page_start=1,
        page_end=1,
        section="Results",
        text="Empirical results directly contradict previous assumptions regarding retriever scaling.",
    )
    sqlite_session.add(p1)

    opp = Opportunity(
        id=uuid.uuid4(),
        title="Scaling Opportunity",
        problem_statement="Retriever scaling assumptions",
        description="Retriever scaling hypothesis",
        status=OpportunityStatus.RESOLVED.value,
    )
    sqlite_session.add(opp)
    sqlite_session.commit()

    transition_svc = OpportunityTransitionService(sqlite_session)

    # CONTRADICTS against RESOLVED must transition back to OPEN
    contradict_classifier = RecordingMockClassifier(
        relationship=OpportunityRelationship.CONTRADICTS,
        evidence_excerpt="Empirical results directly contradict previous assumptions regarding retriever scaling.",
    )
    integration = OpportunityClassificationIntegration(
        classifier=contradict_classifier,
        transition_service=transition_svc,
        session=sqlite_session,
    )
    result = integration.classify_and_apply(opportunity_id=opp.id, paper_id=paper.id)
    assert result.relationship == OpportunityRelationship.CONTRADICTS
    assert result.previous_status == OpportunityStatus.RESOLVED
    assert result.new_status == OpportunityStatus.OPEN
    assert result.state_changed is True

    # EXTENDS against OPEN preserves OPEN status while persisting directional extension evidence
    paper_ext = _make_paper(
        session=sqlite_session,
        title="Extension Paper",
        abstract="Extension abstract.",
        publication_date=datetime.date(2023, 5, 2),
    )
    doc_ext = PaperDocument(
        id=uuid.uuid4(),
        paper_id=paper_ext.id,
        full_text="We extend the formulation to multi-agent iterative retrieval benchmarks.",
        extraction_method="pymupdf",
        extraction_version="1.0",
        extracted_at=datetime.datetime.now(datetime.timezone.utc),
        content_hash="ext_hash",
        original_filename="ext.pdf",
    )
    sqlite_session.add(doc_ext)
    p_ext = PaperPassage(
        id=uuid.uuid4(),
        paper_id=paper_ext.id,
        document_id=doc_ext.id,
        passage_index=0,
        page_start=1,
        page_end=1,
        section="Discussion",
        text="We extend the formulation to multi-agent iterative retrieval benchmarks.",
    )
    sqlite_session.add(p_ext)
    sqlite_session.commit()

    extends_classifier = RecordingMockClassifier(
        relationship=OpportunityRelationship.EXTENDS,
        evidence_excerpt="We extend the formulation to multi-agent iterative retrieval benchmarks.",
    )
    integration_ext = OpportunityClassificationIntegration(
        classifier=extends_classifier,
        transition_service=transition_svc,
        session=sqlite_session,
    )
    result_ext = integration_ext.classify_and_apply(opportunity_id=opp.id, paper_id=paper_ext.id)
    assert result_ext.relationship == OpportunityRelationship.EXTENDS
    assert result_ext.previous_status == OpportunityStatus.OPEN
    assert result_ext.new_status == OpportunityStatus.OPEN
    assert result_ext.evidence_id is not None
    evidence_ext = sqlite_session.query(OpportunityEvidence).filter(OpportunityEvidence.id == result_ext.evidence_id).one()
    assert evidence_ext.evidence_type == OpportunityRelationship.EXTENDS.value
    assert "multi-agent iterative retrieval" in evidence_ext.supporting_text


def test_runner_records_classifier_input_limits_in_metrics(tmp_path: Path):
    """Verify ControlledExperimentRunner records resolved classifier input limits in metrics."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    db_file = tmp_path / "test.db"

    text_content = (
        "Retrieval-augmented generation enables large language models to ground their reasoning in external knowledge. "
        "However, distractors present in multi-document contexts cause severe factual degradation and reasoning drift. "
        "RAG hallucination rates degrade sharply when distractors exceed 3 documents. "
        "Context pruning fails to filter distractor passages in complex scientific QA. "
        "Further research is required to guarantee reliable factual synthesis under adversarial distraction."
    )
    _create_synthetic_pdf(corpus_dir / "paper1.pdf", text_content)

    import json
    manifest = [
        {"filename": "paper1.pdf", "title": "2020 Baseline Paper", "publication_date": "2020-06-01"},
    ]
    with open(corpus_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    runner = ControlledExperimentRunner(
        llm_provider=Stage5MockLLMProvider(),
        embedding_provider=DeterministicExperimentEmbeddingProvider(dimension=64),
        db_path=db_file,
        artifacts_dir=artifacts_dir,
        temporal_cutoff="2020-12-31",
        enable_stage_5=True,
        stage_5_max_passages=6,
        stage_5_max_total_words=1200,
        stage_5_max_abstract_chars=1500,
    )

    result = runner.run(input_dir=corpus_dir)
    assert result.success is True
    assert "stage_5_evolution" in result.metrics
    s5_metrics = result.metrics["stage_5_evolution"]

    assert s5_metrics["classifier_input_limits"] == {
        "max_passages": 6,
        "max_total_words": 1200,
        "max_abstract_chars": 1500,
    }
