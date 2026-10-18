"""API integration tests for AI Paper Understanding endpoints (Phase 3)."""

from datetime import datetime, timezone
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ideagen.api.main import app
from ideagen.core.config import get_settings
from ideagen.db.models import Base, Paper, PaperDocument, PaperPassage
from ideagen.db.persistence import SQLAlchemyPersistenceLayer
from ideagen.db.session import get_db
from ideagen.ingestion.models import PaperCreateDTO, SourceType
from ideagen.understanding.models import (
    ResearchKnowledgeCreateDTO,
    ResearchStatementCreateDTO,
    StatementType,
)

settings = get_settings()
DB_URL = settings.database_url


@pytest.fixture(scope="session")
def engine():
    if not DB_URL:
        pytest.skip("IDEAGEN_DATABASE_URL is not configured; skipping understanding API tests.")
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


def test_api_understand_paper_not_found(client):
    """Verifies 404 returned when understanding non-existent paper."""
    random_id = uuid.uuid4()
    resp = client.post(f"/api/papers/{random_id}/understand")
    assert resp.status_code == 404


def test_api_get_knowledge_not_found(client):
    """Verifies 404 returned when retrieving non-existent knowledge."""
    random_id = uuid.uuid4()
    resp = client.get(f"/api/papers/{random_id}/knowledge")
    assert resp.status_code == 404


def test_api_knowledge_and_statements_retrieval(client, session):
    """Verifies GET /api/papers/{id}/knowledge and /statements return valid structured results."""
    layer = SQLAlchemyPersistenceLayer(session)

    # 1. Insert Paper & Passages
    paper_dto = PaperCreateDTO(
        source_type=SourceType.MANUAL_UPLOAD,
        source_id="API_UNDERSTAND_TEST_PAPER",
        title="API Understanding Test",
        batch_id="Batch 3",
        abstract="Test abstract for API verification.",
        authors=[{"name": "Author"}],
        referenced_works=[],
        raw_metadata={},
    )
    paper_id = layer.save_paper(paper_dto)

    doc = PaperDocument(
        paper_id=paper_id,
        full_text="Test full text.",
        extraction_method="pymupdf",
        extraction_version="v2.0",
        extracted_at=datetime.now(timezone.utc),
    )
    session.add(doc)
    session.commit()

    passage = PaperPassage(
        paper_id=paper_id,
        document_id=doc.id,
        passage_index=0,
        page_start=1,
        page_end=1,
        section="Abstract",
        text="A test passage text.",
    )
    session.add(passage)
    session.commit()

    # 2. Persist Knowledge & Statements
    stmt_dto = ResearchStatementCreateDTO(
        statement_text="A grounded API claim.",
        statement_type=StatementType.CONTRIBUTION,
        source_passage_id=passage.id,
        source_page_start=1,
        source_page_end=1,
        section="Abstract",
        confidence=0.94,
    )
    knowledge_dto = ResearchKnowledgeCreateDTO(
        paper_id=paper_id,
        research_problem="API verification problem.",
        research_questions=["Does the endpoint return expected DTO?"],
        methodology="TestClient HTTP request.",
        datasets=["Synthetic"],
        findings=["All assertions succeed."],
        limitations=["Unit test scope."],
        future_work=["End-to-end."],
        research_statements=[stmt_dto],
        model_name="api-test-model",
        model_version="1.0",
        extraction_schema_version="v1.0",
    )
    layer.save_research_knowledge(knowledge_dto)

    # 3. Test GET /api/papers/{paper_id}/knowledge
    resp_k = client.get(f"/api/papers/{paper_id}/knowledge")
    assert resp_k.status_code == 200
    data_k = resp_k.json()
    assert data_k["paper_id"] == str(paper_id)
    assert data_k["research_problem"] == "API verification problem."
    assert data_k["statement_count"] == 1
    assert data_k["model_name"] == "api-test-model"

    # 4. Test GET /api/papers/{paper_id}/statements
    resp_s = client.get(f"/api/papers/{paper_id}/statements")
    assert resp_s.status_code == 200
    data_s = resp_s.json()
    assert len(data_s) == 1
    assert data_s[0]["statement_text"] == "A grounded API claim."
    assert data_s[0]["statement_type"] == "CONTRIBUTION"
    assert data_s[0]["passage_id"] == str(passage.id)
    assert data_s[0]["page_start"] == 1
    assert data_s[0]["section"] == "Abstract"
    assert data_s[0]["confidence"] == 0.94
