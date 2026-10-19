"""Unit and integration tests for Semantic Gap Discovery (Phase 4).

Verifies embedding dimensions, FAISS local vector index, statement mapping,
semantic grouping, research-pattern analysis, evidence integrity checks,
and PostgreSQL persistence.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import uuid
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ideagen.core.config import get_settings
from ideagen.db.models import Base, Paper, PaperDocument, PaperPassage, ResearchKnowledge, ResearchStatement, ResearchGap, GapEvidence
from ideagen.db.persistence import SQLAlchemyPersistenceLayer
from ideagen.ingestion.models import PaperCreateDTO, SourceType
from ideagen.llm.provider import LLMProvider
from ideagen.discovery.models import (
    GapStatus,
    ValidationVerdict,
    EvidenceRole,
    GapCandidate,
    GapValidationResult,
    ResearchGapCreateDTO,
    StatementEvidenceDTO,
)
from ideagen.discovery.embeddings import EmbeddingProvider, GeminiEmbeddingProvider
from ideagen.discovery.vector_index import StatementVectorIndex, StatementSearchResult
from ideagen.discovery.grouper import SemanticGrouper, SemanticCluster
from ideagen.discovery.pattern_analyzer import ResearchPatternAnalyzer
from ideagen.discovery.validator import GapValidator


settings = get_settings()


@pytest.fixture(scope="session")
def engine():
    db_url = settings.database_url
    if not db_url:
        pytest.skip("IDEAGEN_DATABASE_URL is not configured; skipping gap discovery persistence tests.")
    eng = create_engine(db_url)
    Base.metadata.create_all(eng)
    yield eng


@pytest.fixture
def session(engine):
    connection = engine.connect()
    transaction = connection.begin()
    session_maker = sessionmaker(bind=connection, join_transaction_mode="create_savepoint")
    session = session_maker()
    yield session
    session.close()
    transaction.rollback()
    connection.close()


class DeterministicTestEmbeddingProvider(EmbeddingProvider):
    """Generates reproducible unit-length dense vectors for testing vector mathematics and indexing."""
    def __init__(self, dimension: int = 64):
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return "deterministic-test-embeddings-v1"

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        # Generate vectors with semantic clustering properties based on vocabulary
        vecs = []
        for text in texts:
            # Deterministic pseudo-random seed from string hash
            seed = sum(ord(c) for c in text) % 10000
            rng = np.random.RandomState(seed)
            # Add semantic directional bias for topics
            base = rng.randn(self._dim).astype(np.float32)
            if "hallucination" in text.lower() or "rag" in text.lower():
                base[:10] += 5.0
            elif "latency" in text.lower() or "speed" in text.lower():
                base[10:20] += 5.0
            elif "vision" in text.lower() or "multimodal" in text.lower():
                base[20:30] += 5.0

            # Normalize to unit length
            norm = np.linalg.norm(base)
            vecs.append(base / (norm if norm > 0 else 1.0))

        return np.array(vecs, dtype=np.float32)


class StubValidationProvider(LLMProvider):
    """Deterministic LLM provider for validating candidate gaps without external API calls."""
    def __init__(self, verdict: ValidationVerdict = ValidationVerdict.VALID, reasoning: str = "Evidence verified.", cited_ids: list[str] = None):
        self.verdict = verdict
        self.reasoning = reasoning
        self.cited_ids = cited_ids or []
        self.last_prompt = None

    def generate_structured(self, prompt: str, response_schema):
        self.last_prompt = prompt
        return response_schema(
            verdict=self.verdict,
            reasoning=self.reasoning,
            supporting_evidence_statement_ids=self.cited_ids,
            confidence=0.92,
        )


def test_embedding_provider_protocol_and_dimensions():
    """Verifies that embedding providers output correct dimensions and normalized vectors."""
    provider = DeterministicTestEmbeddingProvider(dimension=128)
    assert provider.dimension == 128
    assert "deterministic" in provider.model_name

    texts = [
        "Hallucination mitigation in RAG systems.",
        "Latency constraints during graph traversal.",
    ]
    vectors = provider.embed_texts(texts)
    assert vectors.shape == (2, 128)
    # Check L2 unit norm
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-5)


def test_faiss_vector_index_deterministic_search(tmp_path):
    """Verifies FAISS IndexFlatIP cosine similarity, ordering, and thresholding."""
    dim = 32
    index = StatementVectorIndex(dimension=dim)

    s1_id = uuid.uuid4()
    s2_id = uuid.uuid4()
    p_id = uuid.uuid4()

    statements = [
        {"id": s1_id, "paper_id": p_id, "statement_type": "LIMITATION", "statement_text": "High memory consumption during inference.", "section": "Limitations", "page_start": 3, "page_end": 3},
        {"id": s2_id, "paper_id": p_id, "statement_type": "METHOD", "statement_text": "Quantization of attention weights.", "section": "Methods", "page_start": 2, "page_end": 2},
    ]

    # Create distinct directional vectors
    v1 = np.zeros(dim, dtype=np.float32)
    v1[0] = 1.0
    v2 = np.zeros(dim, dtype=np.float32)
    v2[1] = 1.0
    vectors = np.stack([v1, v2])

    index.add_statements(statements, vectors)
    assert index.total_vectors == 2

    # Query with vector parallel to v1
    query = np.zeros(dim, dtype=np.float32)
    query[0] = 1.0
    results = index.search(query, top_k=2)

    assert len(results) == 2
    assert results[0].statement_id == s1_id
    assert pytest.approx(results[0].similarity, 1e-4) == 1.0
    assert pytest.approx(results[1].similarity, 1e-4) == 0.0

    # Query with min_similarity threshold
    filtered_results = index.search(query, top_k=2, min_similarity=0.5)
    assert len(filtered_results) == 1
    assert filtered_results[0].statement_id == s1_id


def test_faiss_vector_to_statement_mapping(tmp_path):
    """Verifies vector-to-statement metadata mapping and attribute preservation."""
    dim = 16
    index = StatementVectorIndex(dimension=dim)

    s_id = uuid.uuid4()
    p_id = uuid.uuid4()
    stmt = {
        "id": s_id,
        "paper_id": p_id,
        "statement_type": "FINDING",
        "statement_text": "Graph retrieval improves factual precision by 24%.",
        "section": "Results",
        "page_start": 4,
        "page_end": 5,
    }
    vec = np.random.randn(1, dim).astype(np.float32)

    index.add_statements([stmt], vec)
    res = index.search(vec[0], top_k=1)[0]

    assert res.statement_id == s_id
    assert res.paper_id == p_id
    assert res.statement_type == "FINDING"
    assert res.statement_text == "Graph retrieval improves factual precision by 24%."
    assert res.section == "Results"
    assert res.page_start == 4
    assert res.page_end == 5


def test_faiss_index_save_load_and_rebuild(tmp_path):
    """Verifies persistence to disk, loading, and deterministic index rebuild."""
    dim = 16
    index_file = tmp_path / "index.faiss"
    meta_file = tmp_path / "meta.json"

    idx1 = StatementVectorIndex(dimension=dim)
    s_id = uuid.uuid4()
    p_id = uuid.uuid4()
    stmt = {"id": s_id, "paper_id": p_id, "statement_type": "PROBLEM", "statement_text": "Hallucination.", "section": "Intro", "page_start": 1, "page_end": 1}
    vec = np.ones((1, dim), dtype=np.float32)

    idx1.add_statements([stmt], vec)
    idx1.save(index_file, meta_file)

    assert index_file.exists()
    assert meta_file.exists()

    # Load back
    loaded_idx = StatementVectorIndex.load(index_file, meta_file)
    assert loaded_idx.dimension == dim
    assert loaded_idx.total_vectors == 1
    assert loaded_idx.id_list[0] == s_id

    # Rebuild
    s2_id = uuid.uuid4()
    stmt2 = {"id": s2_id, "paper_id": p_id, "statement_type": "METHOD", "statement_text": "Reranking.", "section": "Method", "page_start": 2, "page_end": 2}
    vec2 = np.zeros((1, dim), dtype=np.float32)
    vec2[0, 0] = 1.0

    loaded_idx.rebuild([stmt2], vec2)
    assert loaded_idx.total_vectors == 1
    assert loaded_idx.id_list[0] == s2_id


def test_semantic_grouper_clustering():
    """Verifies cosine distance grouping into coherent semantic research clusters."""
    provider = DeterministicTestEmbeddingProvider(dimension=64)
    grouper = SemanticGrouper(similarity_threshold=0.60, min_cluster_size=2)

    p1 = uuid.uuid4()
    p2 = uuid.uuid4()

    # Group A: RAG & hallucination
    # Group B: Latency & speed
    statements = [
        {"id": uuid.uuid4(), "paper_id": p1, "statement_type": "LIMITATION", "statement_text": "RAG hallucination remains high under distractor documents."},
        {"id": uuid.uuid4(), "paper_id": p2, "statement_type": "FUTURE_WORK", "statement_text": "Mitigating hallucination in RAG benchmarks."},
        {"id": uuid.uuid4(), "paper_id": p1, "statement_type": "LIMITATION", "statement_text": "Latency constraints impede real-time speed in deployment."},
        {"id": uuid.uuid4(), "paper_id": p2, "statement_type": "PROBLEM", "statement_text": "Inference latency and execution speed bottlenecks."},
    ]

    texts = [s["statement_text"] for s in statements]
    vectors = provider.embed_texts(texts)

    clusters = grouper.group_statements(statements, vectors)
    assert len(clusters) >= 2

    # Verify cluster contents are separated by topic
    cluster_texts = [" ".join(s["statement_text"] for s in c.statements) for c in clusters]
    # One cluster should contain hallucination/RAG and another latency/speed
    assert any("hallucination" in t.lower() for t in cluster_texts)
    assert any("latency" in t.lower() for t in cluster_texts)


def test_research_pattern_analyzer_detects_candidate_gaps():
    """Verifies candidate gap detection from recurring limitations and future work."""
    analyzer = ResearchPatternAnalyzer(min_supporting_evidence=1)

    s1_id = uuid.uuid4()
    s2_id = uuid.uuid4()
    p1 = uuid.uuid4()
    p2 = uuid.uuid4()

    statements = [
        {"id": s1_id, "paper_id": p1, "statement_type": "LIMITATION", "statement_text": "Current graph traversal introduces significant latency."},
        {"id": s2_id, "paper_id": p2, "statement_type": "FUTURE_WORK", "statement_text": "Optimizing graph verification latency for streaming responses."},
    ]

    cluster = SemanticCluster(
        cluster_id=0,
        statements=statements,
        paper_ids={p1, p2},
        centroid=np.ones(10, dtype=np.float32),
        statement_type_counts={"LIMITATION": 1, "FUTURE_WORK": 1},
    )

    candidates = analyzer.analyze_clusters([cluster])
    assert len(candidates) == 1

    gap = candidates[0]
    assert "limitation" in gap.gap_statement.lower() or "latency" in gap.gap_statement.lower()
    assert s1_id in gap.supporting_statement_ids
    assert s2_id in gap.supporting_statement_ids
    assert p1 in gap.supporting_paper_ids
    assert p2 in gap.supporting_paper_ids
    assert len(gap.relevant_limitations) == 1
    assert len(gap.relevant_future_work) == 1


def test_pattern_analyzer_empty_when_no_gap_evidence():
    """Verifies that clusters with only positive findings without limitations or problems yield no gaps."""
    analyzer = ResearchPatternAnalyzer(min_supporting_evidence=1)

    statements = [
        {"id": uuid.uuid4(), "paper_id": uuid.uuid4(), "statement_type": "FINDING", "statement_text": "Accuracy reached 95%."},
        {"id": uuid.uuid4(), "paper_id": uuid.uuid4(), "statement_type": "CONTRIBUTION", "statement_text": "New dataset introduced."},
    ]

    cluster = SemanticCluster(
        cluster_id=0,
        statements=statements,
        paper_ids={uuid.uuid4()},
        centroid=np.ones(10, dtype=np.float32),
        statement_type_counts={"FINDING": 1, "CONTRIBUTION": 1},
    )

    candidates = analyzer.analyze_clusters([cluster])
    assert len(candidates) == 0


def test_evidence_integrity_rejects_hallucinated_statement_ids():
    """Verifies that validator strictly rejects hallucinated statement IDs not present in candidate pool."""
    s1_id = uuid.uuid4()
    fake_id = uuid.uuid4()
    p1 = uuid.uuid4()

    statements_by_id = {
        s1_id: {"id": s1_id, "paper_id": p1, "statement_type": "LIMITATION", "statement_text": "Real limitation statement."}
    }

    candidate = GapCandidate(
        gap_statement="Unaddressed problem.",
        research_problem="Problem.",
        why_existing_work_insufficient="Insufficient.",
        supporting_statement_ids=[s1_id],
        supporting_paper_ids=[p1],
        relevant_limitations=["Real limitation statement."],
        relevant_future_work=[],
    )

    # Provider cites ONLY the fake ID
    fake_provider = StubValidationProvider(
        verdict=ValidationVerdict.VALID,
        reasoning="Looks good based on evidence.",
        cited_ids=[str(fake_id)],
    )

    validator = GapValidator(provider=fake_provider, min_valid_evidence=1)
    res = validator.validate_candidate(candidate, statements_by_id)

    # Must fail closed: rejected, empty supporting statements, fake ID rejected
    assert res.status == GapStatus.REJECTED
    assert fake_id not in res.supporting_statement_ids
    assert res.supporting_statement_ids == []
    assert res.evidence_count == 0
    assert res.metadata.get("rejection_reason") == "INVALID_OR_FABRICATED_CITATIONS"


def test_validator_includes_original_passage_when_available():
    """Requirement 1: Original passage text is included in validator evidence block when available."""
    s1_id = uuid.uuid4()
    pass_id = uuid.uuid4()
    p1 = uuid.uuid4()

    passage_text = "Detailed empirical findings show distractor documents degrade precision from 85% to 22%."
    statements_by_id = {
        s1_id: {
            "id": s1_id,
            "paper_id": p1,
            "passage_id": pass_id,
            "statement_type": "LIMITATION",
            "statement_text": "Distractors cause severe degradation.",
            "section": "Experiments",
        }
    }
    passages_by_id = {
        pass_id: {
            "id": pass_id,
            "paper_id": p1,
            "text": passage_text,
        }
    }

    candidate = GapCandidate(
        gap_statement="Distractor sensitivity in RAG.",
        research_problem="Robustness under distraction.",
        why_existing_work_insufficient="Prior methods fail on noisy inputs.",
        supporting_statement_ids=[s1_id],
        supporting_paper_ids=[p1],
        relevant_limitations=["Distractors cause severe degradation."],
        relevant_future_work=[],
    )

    provider = StubValidationProvider(
        verdict=ValidationVerdict.VALID,
        reasoning="Passage confirms empirical degradation.",
        cited_ids=[str(s1_id)],
    )

    validator = GapValidator(provider=provider, min_valid_evidence=1)
    res = validator.validate_candidate(candidate, statements_by_id, passages_by_id=passages_by_id)

    assert res.status == GapStatus.VALIDATED
    assert provider.last_prompt is not None
    # Verify original passage text is present in the prompt evidence block
    assert passage_text in provider.last_prompt
    assert str(s1_id) in provider.last_prompt
    assert str(p1) in provider.last_prompt
    assert "Distractors cause severe degradation." in provider.last_prompt
    assert "Section: Experiments" in provider.last_prompt


def test_validator_handles_missing_or_unlinked_passage_explicitly():
    """Requirement 2: Missing or unlinked passage references are marked unavailable explicitly."""
    s1_id = uuid.uuid4()
    s2_id = uuid.uuid4()
    missing_pass_id = uuid.uuid4()
    p1 = uuid.uuid4()

    statements_by_id = {
        s1_id: {
            "id": s1_id,
            "paper_id": p1,
            "passage_id": None,  # Explicitly unlinked
            "statement_type": "LIMITATION",
            "statement_text": "Unlinked limitation.",
        },
        s2_id: {
            "id": s2_id,
            "paper_id": p1,
            "passage_id": missing_pass_id,  # Linked to ID not in passage map
            "statement_type": "LIMITATION",
            "statement_text": "Missing passage reference statement.",
        },
    }

    candidate = GapCandidate(
        gap_statement="Missing passage test gap.",
        research_problem="Robustness evaluation.",
        why_existing_work_insufficient="Insufficient.",
        supporting_statement_ids=[s1_id, s2_id],
        supporting_paper_ids=[p1],
        relevant_limitations=["Unlinked limitation."],
        relevant_future_work=[],
    )

    provider = StubValidationProvider(
        verdict=ValidationVerdict.VALID,
        reasoning="Valid verdict test.",
        cited_ids=[str(s1_id), str(s2_id)],
    )

    validator = GapValidator(provider=provider, min_valid_evidence=1)
    # passages_by_id is empty, so s1 has no passage_id and s2 has missing passage
    res = validator.validate_candidate(candidate, statements_by_id, passages_by_id={})

    assert provider.last_prompt is not None
    assert "[Source passage unavailable - no linked passage]" in provider.last_prompt
    assert "[Source passage unavailable - passage not found in database]" in provider.last_prompt


def test_validator_rejects_valid_verdict_with_empty_citations():
    """Requirement 3: A VALID verdict with an empty evidence-ID list is not accepted (fails closed)."""
    s1_id = uuid.uuid4()
    p1 = uuid.uuid4()

    statements_by_id = {
        s1_id: {"id": s1_id, "paper_id": p1, "statement_type": "LIMITATION", "statement_text": "Limitation."}
    }

    candidate = GapCandidate(
        gap_statement="Unaddressed gap.",
        research_problem="Problem.",
        why_existing_work_insufficient="Insufficient.",
        supporting_statement_ids=[s1_id],
        supporting_paper_ids=[p1],
        relevant_limitations=["Limitation."],
        relevant_future_work=[],
    )

    provider = StubValidationProvider(
        verdict=ValidationVerdict.VALID,
        reasoning="Model asserts valid without citing any evidence ID.",
        cited_ids=[],  # Empty citations
    )

    validator = GapValidator(provider=provider, min_valid_evidence=1)
    res = validator.validate_candidate(candidate, statements_by_id)

    # Must NOT adopt candidate pool; must fail closed and reject
    assert res.status == GapStatus.REJECTED
    assert res.supporting_statement_ids == []
    assert res.evidence_count == 0
    assert res.metadata.get("rejection_reason") == "EMPTY_EVIDENCE_CITATIONS"
    assert "no evidence statement citations" in res.validation_reasoning


def test_validator_rejects_valid_verdict_with_fabricated_ids():
    """Requirement 4: A VALID verdict with fabricated IDs is not accepted."""
    s1_id = uuid.uuid4()
    p1 = uuid.uuid4()

    statements_by_id = {
        s1_id: {"id": s1_id, "paper_id": p1, "statement_type": "LIMITATION", "statement_text": "Real statement."}
    }

    candidate = GapCandidate(
        gap_statement="Unaddressed gap.",
        research_problem="Problem.",
        why_existing_work_insufficient="Insufficient.",
        supporting_statement_ids=[s1_id],
        supporting_paper_ids=[p1],
        relevant_limitations=["Real statement."],
        relevant_future_work=[],
    )

    provider = StubValidationProvider(
        verdict=ValidationVerdict.VALID,
        reasoning="Fabricated IDs cited.",
        cited_ids=[str(uuid.uuid4()), "not-a-valid-uuid", str(uuid.uuid4())],
    )

    validator = GapValidator(provider=provider, min_valid_evidence=1)
    res = validator.validate_candidate(candidate, statements_by_id)

    assert res.status == GapStatus.REJECTED
    assert res.supporting_statement_ids == []
    assert res.evidence_count == 0
    assert res.metadata.get("rejection_reason") == "INVALID_OR_FABRICATED_CITATIONS"


def test_validator_rejects_ids_outside_candidate_evidence_pool():
    """Requirement 5: IDs belonging to statements outside candidate evidence pool are rejected."""
    s1_id = uuid.uuid4()
    s2_id = uuid.uuid4()  # In DB, but NOT in this candidate's pool
    p1 = uuid.uuid4()

    statements_by_id = {
        s1_id: {"id": s1_id, "paper_id": p1, "statement_type": "LIMITATION", "statement_text": "Candidate limitation."},
        s2_id: {"id": s2_id, "paper_id": p1, "statement_type": "LIMITATION", "statement_text": "Unrelated limitation."},
    }

    candidate = GapCandidate(
        gap_statement="Candidate gap statement.",
        research_problem="Problem.",
        why_existing_work_insufficient="Insufficient.",
        supporting_statement_ids=[s1_id],  # ONLY s1 is in candidate pool
        supporting_paper_ids=[p1],
        relevant_limitations=["Candidate limitation."],
        relevant_future_work=[],
    )

    # Case A: Model cites ONLY the external statement ID s2
    provider_external_only = StubValidationProvider(
        verdict=ValidationVerdict.VALID,
        reasoning="Cites statement outside candidate pool.",
        cited_ids=[str(s2_id)],
    )

    validator = GapValidator(provider=provider_external_only, min_valid_evidence=1)
    res_external = validator.validate_candidate(candidate, statements_by_id)

    assert res_external.status == GapStatus.REJECTED
    assert res_external.supporting_statement_ids == []
    assert s2_id not in res_external.supporting_statement_ids
    assert res_external.metadata.get("rejection_reason") == "INVALID_OR_FABRICATED_CITATIONS"

    # Case B: Model cites BOTH valid candidate ID (s1) and external ID (s2)
    provider_mixed = StubValidationProvider(
        verdict=ValidationVerdict.VALID,
        reasoning="Cites mixed IDs.",
        cited_ids=[str(s1_id), str(s2_id)],
    )

    validator_mixed = GapValidator(provider=provider_mixed, min_valid_evidence=1)
    res_mixed = validator_mixed.validate_candidate(candidate, statements_by_id)

    assert res_mixed.status == GapStatus.VALIDATED
    assert res_mixed.supporting_statement_ids == [s1_id]
    assert s2_id not in res_mixed.supporting_statement_ids
    assert res_mixed.evidence_count == 1


def test_validator_accepts_valid_verdict_with_valid_candidate_ids():
    """Requirement 6: A VALID verdict with valid candidate-associated IDs can still pass."""
    s1_id = uuid.uuid4()
    s2_id = uuid.uuid4()
    p1 = uuid.uuid4()
    p2 = uuid.uuid4()

    statements_by_id = {
        s1_id: {"id": s1_id, "paper_id": p1, "statement_type": "LIMITATION", "statement_text": "Limitation 1.", "section": "Limitations"},
        s2_id: {"id": s2_id, "paper_id": p2, "statement_type": "LIMITATION", "statement_text": "Limitation 2.", "section": "Discussion"},
    }

    candidate = GapCandidate(
        gap_statement="Valid recurring limitation.",
        research_problem="Problem.",
        why_existing_work_insufficient="Insufficient.",
        supporting_statement_ids=[s1_id, s2_id],
        supporting_paper_ids=[p1, p2],
        relevant_limitations=["Limitation 1.", "Limitation 2."],
        relevant_future_work=[],
    )

    provider = StubValidationProvider(
        verdict=ValidationVerdict.VALID,
        reasoning="Both empirical limitations substantiate the candidate gap.",
        cited_ids=[str(s1_id), str(s2_id)],
    )

    validator = GapValidator(provider=provider, min_valid_evidence=1)
    res = validator.validate_candidate(candidate, statements_by_id)

    assert res.status == GapStatus.VALIDATED
    assert set(res.supporting_statement_ids) == {s1_id, s2_id}
    assert res.evidence_count == 2
    assert len(res.evidence_items) == 2
    assert res.validation_reasoning == "Both empirical limitations substantiate the candidate gap."


def test_validator_preserves_explicit_rejection_behavior():
    """Requirement 7: Explicit model rejection behavior remains intact."""
    s1_id = uuid.uuid4()
    p1 = uuid.uuid4()

    statements_by_id = {
        s1_id: {"id": s1_id, "paper_id": p1, "statement_type": "LIMITATION", "statement_text": "Solvable constraint."}
    }

    candidate = GapCandidate(
        gap_statement="Purported gap.",
        research_problem="Problem.",
        why_existing_work_insufficient="Insufficient.",
        supporting_statement_ids=[s1_id],
        supporting_paper_ids=[p1],
        relevant_limitations=["Solvable constraint."],
        relevant_future_work=[],
    )

    provider = StubValidationProvider(
        verdict=ValidationVerdict.REJECT,
        reasoning="Candidate is trivial and already addressed by existing literature.",
        cited_ids=[str(s1_id)],
    )

    validator = GapValidator(provider=provider, min_valid_evidence=1)
    res = validator.validate_candidate(candidate, statements_by_id)

    assert res.status == GapStatus.REJECTED
    assert res.validation_reasoning == "Candidate is trivial and already addressed by existing literature."
    assert res.metadata.get("model_verdict") == "REJECT"


def test_gap_persistence_and_evidence_relationships(session):
    """Verifies PostgreSQL persistence of ResearchGap and GapEvidence records with cascade and relationships."""
    layer = SQLAlchemyPersistenceLayer(session)

    # 1. Create Paper & Statement
    paper_dto = PaperCreateDTO(
        source_type=SourceType.MANUAL_UPLOAD,
        source_id="GAP_TEST_PAPER",
        title="Paper for Gap Testing",
        batch_id="Batch 4",
        abstract="Abstract for gap persistence.",
        authors=[{"name": "Researcher"}],
        referenced_works=[],
        raw_metadata={},
    )
    paper_id = layer.save_paper(paper_dto)

    doc = PaperDocument(
        paper_id=paper_id,
        full_text="Sample text",
        extraction_method="pymupdf",
        extraction_version="v2.0",
        extracted_at=datetime.now(timezone.utc),
    )
    session.add(doc)
    session.commit()

    knowledge = ResearchKnowledge(
        paper_id=paper_id,
        model_name="test-model",
        extracted_at=datetime.now(timezone.utc),
        extraction_status="SUCCESS",
    )
    session.add(knowledge)
    session.commit()

    statement = ResearchStatement(
        knowledge_id=knowledge.id,
        paper_id=paper_id,
        statement_text="Multimodal distractors cause catastrophic retrieval failure.",
        statement_type="LIMITATION",
        page_start=4,
        page_end=4,
        section="Limitations",
        confidence=0.95,
    )
    session.add(statement)
    session.commit()

    # 2. Persist ResearchGap with GapEvidence
    evidence_item = StatementEvidenceDTO(
        statement_id=statement.id,
        paper_id=paper_id,
        statement_type="LIMITATION",
        statement_text=statement.statement_text,
        section="Limitations",
        page_start=4,
        page_end=4,
        evidence_role=EvidenceRole.LIMITATION,
    )

    gap_dto = ResearchGapCreateDTO(
        gap_statement="Lack of robust multimodal distractor filtering in RAG.",
        research_problem="Multimodal hallucination.",
        status=GapStatus.VALIDATED,
        supporting_statement_ids=[statement.id],
        supporting_paper_ids=[paper_id],
        evidence_count=1,
        limitations=[statement.statement_text],
        future_work=["Multimodal filtering benchmark."],
        validation_reasoning="Substantiated by explicit limitation in paper.",
        validation_confidence=0.95,
        evidence_items=[evidence_item],
        metadata={"domain": "RAG Reliability"},
    )

    gap_id = layer.save_research_gap(gap_dto)
    assert gap_id is not None

    # 3. Retrieve and assert
    gap = layer.get_research_gap(gap_id)
    assert gap is not None
    assert gap.gap_statement == "Lack of robust multimodal distractor filtering in RAG."
    assert gap.status == "VALIDATED"
    assert gap.evidence_count == 1

    evidence_records = layer.get_gap_evidence(gap_id)
    assert len(evidence_records) == 1
    assert evidence_records[0].statement_id == statement.id
    assert evidence_records[0].paper_id == paper_id
    assert evidence_records[0].evidence_role == "LIMITATION"


# ============================================================================
# Phase 1 Candidate Generation Quality Regression Tests
# ============================================================================


def test_pattern_analyzer_assumption_alone_produces_no_limitation_candidate():
    """Requirement 1: An ASSUMPTION statement alone produces no limitation candidate."""
    analyzer = ResearchPatternAnalyzer(min_supporting_evidence=1)
    s_id = uuid.uuid4()
    p_id = uuid.uuid4()

    statements = [
        {
            "id": s_id,
            "paper_id": p_id,
            "statement_type": "ASSUMPTION",
            "statement_text": "We assume that Wikipedia passages contain all necessary factual knowledge.",
        }
    ]

    cluster = SemanticCluster(
        cluster_id=0,
        statements=statements,
        paper_ids={p_id},
        centroid=np.ones(10, dtype=np.float32),
        statement_type_counts={"ASSUMPTION": 1},
    )

    candidates = analyzer.analyze_clusters([cluster])
    assert len(candidates) == 0


def test_pattern_analyzer_positive_sufficiency_statement_not_a_limitation():
    """Requirement 2: A positive sufficiency statement does not become a limitation candidate."""
    analyzer = ResearchPatternAnalyzer(min_supporting_evidence=1)
    s_id = uuid.uuid4()
    p_id = uuid.uuid4()

    statements = [
        {
            "id": s_id,
            "paper_id": p_id,
            "statement_type": "LIMITATION",  # Even if misclassified by upstream extractor
            "statement_text": "Fine-tuning the query encoder BERTq and BART generator is sufficient for strong performance.",
        }
    ]

    cluster = SemanticCluster(
        cluster_id=0,
        statements=statements,
        paper_ids={p_id},
        centroid=np.ones(10, dtype=np.float32),
        statement_type_counts={"LIMITATION": 1},
    )

    candidates = analyzer.analyze_clusters([cluster])
    assert len(candidates) == 0


def test_pattern_analyzer_feasibility_statement_not_a_limitation():
    """Requirement 3: A feasibility statement about adapting a model does not become a limitation."""
    analyzer = ResearchPatternAnalyzer(min_supporting_evidence=1)
    s_id = uuid.uuid4()
    p_id = uuid.uuid4()

    statements = [
        {
            "id": s_id,
            "paper_id": p_id,
            "statement_type": "LIMITATION",  # Even if misclassified as limitation
            "statement_text": "RAG models can be adapted for sequence classification by treating the target class as a single-token sequence.",
        }
    ]

    cluster = SemanticCluster(
        cluster_id=0,
        statements=statements,
        paper_ids={p_id},
        centroid=np.ones(10, dtype=np.float32),
        statement_type_counts={"LIMITATION": 1},
    )

    candidates = analyzer.analyze_clusters([cluster])
    assert len(candidates) == 0


def test_pattern_analyzer_explicit_limitation_produces_eligible_candidate():
    """Requirement 4: A genuine explicit limitation still produces an eligible candidate."""
    analyzer = ResearchPatternAnalyzer(min_supporting_evidence=1)
    s_id = uuid.uuid4()
    p_id = uuid.uuid4()

    statements = [
        {
            "id": s_id,
            "paper_id": p_id,
            "statement_type": "LIMITATION",
            "statement_text": "Current graph traversal introduces significant latency and inference overhead.",
        }
    ]

    cluster = SemanticCluster(
        cluster_id=0,
        statements=statements,
        paper_ids={p_id},
        centroid=np.ones(10, dtype=np.float32),
        statement_type_counts={"LIMITATION": 1},
    )

    candidates = analyzer.analyze_clusters([cluster])
    assert len(candidates) == 1

    gap = candidates[0]
    # Single-paper limitation must not claim to be "recurring across methodologies"
    assert gap.signals["pattern_type"] == "DOCUMENTED_LIMITATION"
    assert "Documented scientific limitation" in gap.gap_statement
    assert "Significant latency and inference overhead" in gap.gap_statement or "Current graph traversal" in gap.gap_statement


def test_pattern_analyzer_single_vs_multi_paper_corroboration():
    """Requirement 5: Multi-paper corroboration yields RECURRING_LIMITATIONS while single-paper yields DOCUMENTED_LIMITATION."""
    analyzer = ResearchPatternAnalyzer(min_supporting_evidence=1)
    s1_id = uuid.uuid4()
    s2_id = uuid.uuid4()
    p1 = uuid.uuid4()
    p2 = uuid.uuid4()

    statements = [
        {
            "id": s1_id,
            "paper_id": p1,
            "statement_type": "LIMITATION",
            "statement_text": "Context pruning fails to filter distractor passages in complex QA.",
        },
        {
            "id": s2_id,
            "paper_id": p2,
            "statement_type": "LIMITATION",
            "statement_text": "Irrelevant documents cause severe hallucination degradation in generation.",
        },
    ]

    cluster = SemanticCluster(
        cluster_id=0,
        statements=statements,
        paper_ids={p1, p2},
        centroid=np.ones(10, dtype=np.float32),
        statement_type_counts={"LIMITATION": 2},
    )

    candidates = analyzer.analyze_clusters([cluster])
    assert len(candidates) == 1

    gap = candidates[0]
    # With distinct papers, recurring pattern is valid
    assert gap.signals["pattern_type"] == "RECURRING_LIMITATIONS"
    assert "Persistent scientific limitation across current methodologies" in gap.gap_statement
    assert s1_id in gap.supporting_statement_ids
    assert s2_id in gap.supporting_statement_ids
    assert p1 in gap.supporting_paper_ids
    assert p2 in gap.supporting_paper_ids

