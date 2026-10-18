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
DB_URL = settings.database_url
if not DB_URL:
    raise ValueError("IDEAGEN_DATABASE_URL environment variable is required to run database integration tests.")


@pytest.fixture(scope="session")
def engine():
    engine = create_engine(DB_URL)
    Base.metadata.create_all(engine)
    yield engine


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

    def generate_structured(self, prompt: str, response_schema):
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

    # Must retain verified statements or reject if fake ID was provided
    # Verified statements from candidate pool should not contain the fake_id
    assert fake_id not in res.supporting_statement_ids


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
