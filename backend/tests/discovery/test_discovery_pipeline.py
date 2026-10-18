"""Pipeline and API integration tests for Semantic Gap Discovery (Phase 4)."""

from datetime import datetime, timezone
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ideagen.api.main import app
from ideagen.core.config import get_settings
from ideagen.db.models import Base, Paper, PaperDocument, ResearchKnowledge, ResearchStatement, ResearchGap, GapEvidence
from ideagen.db.persistence import SQLAlchemyPersistenceLayer
from ideagen.db.session import get_db
from ideagen.ingestion.models import PaperCreateDTO, SourceType
from ideagen.discovery.models import (
    GapStatus,
    ValidationVerdict,
    ResearchGapCreateDTO,
    StatementEvidenceDTO,
)
from ideagen.discovery.pipeline import SemanticGapDiscoveryPipeline
from tests.discovery.test_gap_discovery import DeterministicTestEmbeddingProvider, StubValidationProvider

settings = get_settings()
DB_URL = settings.database_url


@pytest.fixture(scope="session")
def engine():
    if not DB_URL:
        pytest.skip("IDEAGEN_DATABASE_URL is not configured; skipping discovery pipeline tests.")
    eng = create_engine(DB_URL)
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


@pytest.fixture
def client(session):
    def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_full_discovery_pipeline_flow(session, tmp_path):
    """Verifies end-to-end execution of SemanticGapDiscoveryPipeline from DB statements to validated gaps."""
    layer = SQLAlchemyPersistenceLayer(session)

    # 1. Insert 2 Papers with research statements exhibiting a clear gap pattern
    p1_dto = PaperCreateDTO(
        source_type=SourceType.MANUAL_UPLOAD,
        source_id="DISCOVERY_PAPER_1",
        title="Analysis of Distractor Injection in Scientific RAG",
        batch_id="Batch 4",
        abstract="Investigating distractor documents in RAG systems.",
        authors=[{"name": "Alice"}],
        referenced_works=[],
        raw_metadata={},
    )
    p1_id = layer.save_paper(p1_dto)

    p2_dto = PaperCreateDTO(
        source_type=SourceType.MANUAL_UPLOAD,
        source_id="DISCOVERY_PAPER_2",
        title="Faithfulness Evaluation Under Context Noise",
        batch_id="Batch 4",
        abstract="Evaluating noisy contexts in generative QA.",
        authors=[{"name": "Bob"}],
        referenced_works=[],
        raw_metadata={},
    )
    p2_id = layer.save_paper(p2_dto)

    k1 = ResearchKnowledge(paper_id=p1_id, model_name="test", extracted_at=datetime.now(timezone.utc), extraction_status="SUCCESS")
    k2 = ResearchKnowledge(paper_id=p2_id, model_name="test", extracted_at=datetime.now(timezone.utc), extraction_status="SUCCESS")
    session.add_all([k1, k2])
    session.commit()

    # Create semantic statements: both papers point to limitations in RAG hallucination
    s1 = ResearchStatement(
        knowledge_id=k1.id,
        paper_id=p1_id,
        statement_text="RAG hallucination rates degrade sharply when distractors exceed 3 documents.",
        statement_type="LIMITATION",
        section="Limitations",
        page_start=4,
        page_end=4,
    )
    s2 = ResearchStatement(
        knowledge_id=k2.id,
        paper_id=p2_id,
        statement_text="Developing robust distractor filtering for RAG remains an open future challenge.",
        statement_type="FUTURE_WORK",
        section="Future Work",
        page_start=6,
        page_end=6,
    )
    session.add_all([s1, s2])
    session.commit()

    # 2. Configure pipeline with deterministic providers
    embed_provider = DeterministicTestEmbeddingProvider(dimension=64)
    val_provider = StubValidationProvider(
        verdict=ValidationVerdict.VALID,
        reasoning="Validated based on mutual limitations across papers.",
        cited_ids=[str(s1.id), str(s2.id)],
    )

    pipeline = SemanticGapDiscoveryPipeline(
        persistence=layer,
        embedding_provider=embed_provider,
        llm_provider=val_provider,
        index_dir=tmp_path / "indexes",
        similarity_threshold=0.50,
        min_supporting_evidence=1,
    )

    # 3. Execute discovery
    gaps = pipeline.discover_gaps(persist_index=True)
    assert len(gaps) >= 1

    gap = gaps[0]
    assert gap.status == GapStatus.VALIDATED
    assert s1.id in gap.supporting_statement_ids
    assert s2.id in gap.supporting_statement_ids
    assert gap.evidence_count >= 2

    # Check FAISS index files on disk
    assert (tmp_path / "indexes" / "index.faiss").exists()
    assert (tmp_path / "indexes" / "meta.json").exists()


def test_discovery_api_endpoints(client, session):
    """Verifies GET /api/gaps and GET /api/gaps/{id} API endpoints."""
    layer = SQLAlchemyPersistenceLayer(session)

    # Insert a validated gap directly
    gap_dto = ResearchGapCreateDTO(
        gap_statement="API discovery verification gap.",
        research_problem="Problem statement.",
        status=GapStatus.VALIDATED,
        supporting_statement_ids=[],
        supporting_paper_ids=[],
        evidence_count=0,
        limitations=["Limitation."],
        future_work=["Future."],
        validation_reasoning="Verified.",
        validation_confidence=0.98,
        evidence_items=[],
        metadata={"test": True},
    )
    gap_id = layer.save_research_gap(gap_dto)

    # 1. GET /api/gaps
    resp = client.get("/api/gaps")
    assert resp.status_code == 200
    gaps_list = resp.json()
    assert len(gaps_list) >= 1
    found = [g for g in gaps_list if g["id"] == str(gap_id)]
    assert len(found) == 1
    assert found[0]["gap_statement"] == "API discovery verification gap."
    assert found[0]["status"] == "VALIDATED"

    # 2. GET /api/gaps?status=VALIDATED
    resp_val = client.get("/api/gaps?status=VALIDATED")
    assert resp_val.status_code == 200
    assert any(g["id"] == str(gap_id) for g in resp_val.json())

    # 3. GET /api/gaps/{gap_id}
    resp_single = client.get(f"/api/gaps/{gap_id}")
    assert resp_single.status_code == 200
    single_data = resp_single.json()
    assert single_data["id"] == str(gap_id)
    assert single_data["validation_reasoning"] == "Verified."

    # 4. GET /api/gaps/{non_existent} -> 404
    resp_404 = client.get(f"/api/gaps/{uuid.uuid4()}")
    assert resp_404.status_code == 404
