"""Stage 6F — Targeted API Contract Fixes Regression & Contract Verification Suite.

Verifies:
1. Both old and new route paths work for every new alias:
   - POST /api/v1/papers/{id}/understand & POST /api/papers/{id}/understand
   - GET /api/v1/papers/{id}/knowledge & GET /api/papers/{id}/knowledge
   - GET /api/v1/documents/{id}/passages & GET /api/documents/{id}/passages
   - POST /api/v1/gaps/discover & POST /api/gaps/discover
2. Equivalent response schemas and status codes for all aliases.
3. Opportunity catalog conforms to OpportunityListItemDTO (including proposed_research_direction,
   gap_id, lifecycle_opportunity_id, lifecycle_status, evidence_count).
4. Duplicate OpenAlex ingestion returns the existing paper ID without creating duplicate papers,
   documents, passages, evidence, or history.
5. Ambiguous duplicate resolution fails safely with null ID.
6. Existing ArXiv search, manual upload, and OpenAlex search/ingestion contracts remain compatible
   (OpenAlexSearchResponse.count preserves batch count, total_count exposes upstream hits).
7. Existing error and transaction behavior remains unchanged.
"""

import io
import json
import uuid
import datetime
from typing import Dict, Any, List, Optional
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient
import numpy as np
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import pymupdf

from ideagen.api.main import app
from ideagen.api.schema import (
    OpportunityListItemDTO,
    OpenAlexSearchResponse,
    OpenAlexIngestResponse,
)
from ideagen.understanding import ResearchKnowledgeResponseDTO
from ideagen.discovery import ResearchGapResponseDTO
from ideagen.db import (
    Base,
    get_db,
    Paper,
    PaperDocument,
    PaperPassage,
    ResearchKnowledge,
    ResearchStatement,
    ResearchGap,
    GapEvidence,
    Opportunity,
    OpportunityEvidence,
    OpportunityHistory,
    ResearchOpportunityRecord,
    SQLAlchemyPersistenceLayer,
)
from ideagen.ingestion.models import (
    PaperCreateDTO,
    PaperDocumentCreateDTO,
    PassageCreateDTO,
    SourceType,
)
from ideagen.ingestion.openalex_ingestion import (
    OpenAlexPaperIngestionService,
    IngestionStatus,
    EvolutionHookStatus,
)
from ideagen.ingestion.pipeline import DefaultDuplicateDetector
from ideagen.ingestion.search import OpenAlexSearchProvider, PaperSearchResult
from ideagen.understanding.models import (
    RawStatementExtraction,
    StatementType,
    StructuredPaperExtractionOutput,
)
from ideagen.discovery.models import (
    GapValidationResult,
    ValidationVerdict,
)


class DeterministicMockLLMProvider:
    def __init__(self):
        self.model_name = "mock-stage6f-llm"

    def complete_structured(self, prompt: str, response_schema: Any, temperature: float = 0.0) -> Any:
        schema_name = getattr(response_schema, "__name__", "")
        if schema_name == "StructuredPaperExtractionOutput":
            return StructuredPaperExtractionOutput(
                research_problem="Consensus latency increases quadratically under network partitions.",
                research_questions=["How to adapt heartbeat under partition?"],
                methodology="Empirical validation",
                datasets=["Simulated WAN benchmark"],
                findings=["Adaptive heartbeat frequency reduces partition recovery delay."],
                limitations=["Scalability beyond 1000 nodes"],
                future_work=["Decentralized consensus"],
                statements=[
                    RawStatementExtraction(
                        statement_text="Consensus latency increases quadratically under network partitions.",
                        statement_type=StatementType.PROBLEM,
                        passage_index=0,
                        confidence=0.95,
                    ),
                    RawStatementExtraction(
                        statement_text="Adaptive heartbeat frequency reduces partition recovery delay.",
                        statement_type=StatementType.FINDING,
                        passage_index=0,
                        confidence=0.91,
                    ),
                ],
            )

        if schema_name == "GapValidationResult":
            return GapValidationResult(
                verdict=ValidationVerdict.VALID,
                reasoning="Gap is grounded in empirical consensus latency statements.",
                supporting_evidence_statement_ids=[],
                confidence=0.92,
            )

        data = {
            "proposed_research_direction": "Decentralized partition-tolerant adaptive consensus protocol.",
            "reasoning": "Static heartbeat schedules fail under bursty packet loss.",
            "technical_approach": "Reinforcement learning adaptive consensus timing.",
            "expected_contribution": "Sub-millisecond partition recovery.",
            "required_data": "Simulated network traces",
            "required_compute": "8x GPU cluster",
            "implementation_complexity": "Moderate",
            "experimental_setup": "WAN testbed",
            "feasibility_score": 0.88,
            "confidence": 0.90,
            "assessment": "High novelty in adaptive systems",
            "differentiation": "Prior protocols rely on static timeouts.",
            "overlapping_statement_indices": [],
        }
        if isinstance(response_schema, type) and issubclass(response_schema, Base):
            valid_fields = {k: v for k, v in data.items() if k in response_schema.model_fields}
            return response_schema(**valid_fields)
        return data

    def generate_structured(self, prompt: str, response_schema: Any, **kwargs) -> Any:
        return self.complete_structured(prompt, response_schema)


def create_sample_pdf_bytes(title: str = "Test Research", body: str = "Empirical distributed systems content. ") -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    lines = [title, "", "Abstract"]
    for i in range(20):
        lines.append(f"Abstract sentence {i}: {body}")
    lines.extend(["", "1. Introduction"])
    for i in range(20):
        lines.append(f"Introduction sentence {i}: {body}")
    page.insert_text((50, 50), "\n".join(lines))
    data = doc.tobytes()
    doc.close()
    return data


class DummyStreamResponse:
    def __init__(self, data: bytes, headers: Optional[Dict[str, str]] = None, status: int = 200):
        self.bio = io.BytesIO(data)
        self.headers = headers if headers is not None else {"Content-Type": "application/pdf"}
        self.status = status

    def read(self, amt: int = -1) -> bytes:
        return self.bio.read(amt)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class DeterministicEmbeddingProvider:
    def __init__(self, dimension: int = 64):
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return "mock-deterministic-embedder"

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        vecs = []
        for text in texts:
            seed = sum(ord(c) for c in text) % 10000
            rng = np.random.RandomState(seed)
            base = rng.randn(self._dim).astype(np.float32)
            norm = np.linalg.norm(base)
            vecs.append(base / (norm if norm > 0 else 1.0))
        return np.array(vecs, dtype=np.float32)


@pytest.fixture(name="stage6f_env")
def stage6f_env_fixture():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db

    mock_llm = DeterministicMockLLMProvider()
    mock_embedder = DeterministicEmbeddingProvider(dimension=64)
    with patch("ideagen.llm.get_default_llm_provider", return_value=mock_llm), \
         patch("ideagen.understanding.pipeline.get_default_llm_provider", return_value=mock_llm), \
         patch("ideagen.api.main.get_default_llm_provider", return_value=mock_llm), \
         patch("ideagen.discovery.pipeline.get_default_llm_provider", return_value=mock_llm), \
         patch("ideagen.discovery.embeddings.get_default_embedding_provider", return_value=mock_embedder), \
         patch("ideagen.discovery.pipeline.get_default_embedding_provider", return_value=mock_embedder):
        with TestClient(app) as client:
            yield client, TestingSessionLocal

    app.dependency_overrides.clear()


# ==============================================================================
# Task 1 & 2 Tests: Route Aliases & Opportunity Catalog DTO
# ==============================================================================

def test_1_route_aliases_exist_and_succeed(stage6f_env):
    """Verifies both old and new paths work for all 4 aliased endpoints."""
    client, SessionMaker = stage6f_env
    db = SessionMaker()
    persistence = SQLAlchemyPersistenceLayer(db)

    # Seed paper, document, passages
    p_id = uuid.uuid4()
    paper = Paper(
        id=p_id,
        source_type="OPENALEX_API",
        source_id="W9990001",
        doi="10.1000/alias1",
        title="Alias Test Paper",
        publication_date=datetime.date(2023, 1, 1),
        batch_id="Batch 0",
        abstract="Abstract for alias test.",
        authors=["Author One"],
        referenced_works=[],
        raw_metadata={"pdf_status": "ACCESSIBLE", "has_pdf": True},
    )
    db.add(paper)

    doc_id = uuid.uuid4()
    doc = PaperDocument(
        id=doc_id,
        paper_id=p_id,
        full_text="Sample text for passages testing.",
        content_hash="hash9990001",
        extraction_method="pymupdf",
        extraction_version="1.0",
        original_filename="alias.pdf",
        extracted_at=datetime.datetime.now(datetime.timezone.utc),
        acquisition_timestamp=datetime.datetime.now(datetime.timezone.utc),
    )
    db.add(doc)

    passage = PaperPassage(
        id=uuid.uuid4(),
        paper_id=p_id,
        document_id=doc_id,
        passage_index=0,
        page_start=1,
        page_end=1,
        section="Abstract",
        text="Sample text for passages testing.",
    )
    db.add(passage)
    db.commit()

    # 1. GET /api/v1/documents/{id}/passages vs GET /api/documents/{id}/passages
    res_v1_passages = client.get(f"/api/v1/documents/{doc_id}/passages")
    res_v0_passages = client.get(f"/api/documents/{doc_id}/passages")
    assert res_v1_passages.status_code == 200
    assert res_v0_passages.status_code == 200
    assert res_v1_passages.json() == res_v0_passages.json()
    assert len(res_v1_passages.json()) == 1

    # 2. POST /api/v1/papers/{id}/understand vs POST /api/papers/{id}/understand
    res_v1_understand = client.post(f"/api/v1/papers/{p_id}/understand")
    assert res_v1_understand.status_code == 200
    data_v1_und = res_v1_understand.json()
    assert "id" in data_v1_und
    assert data_v1_und["statement_count"] >= 1

    # Call legacy understand path on the same paper
    res_v0_understand = client.post(f"/api/papers/{p_id}/understand")
    assert res_v0_understand.status_code == 200
    assert res_v0_understand.json()["statement_count"] >= 1

    # 3. GET /api/v1/papers/{id}/knowledge vs GET /api/papers/{id}/knowledge
    res_v1_knowledge = client.get(f"/api/v1/papers/{p_id}/knowledge")
    res_v0_knowledge = client.get(f"/api/papers/{p_id}/knowledge")
    assert res_v1_knowledge.status_code == 200
    assert res_v0_knowledge.status_code == 200
    assert res_v1_knowledge.json() == res_v0_knowledge.json()

    # 4. POST /api/v1/gaps/discover vs POST /api/gaps/discover
    res_v1_disc = client.post("/api/v1/gaps/discover")
    res_v0_disc = client.post("/api/gaps/discover")
    assert res_v1_disc.status_code == 200
    assert res_v0_disc.status_code == 200
    assert isinstance(res_v1_disc.json(), list)
    assert isinstance(res_v0_disc.json(), list)


def test_2_route_aliases_equivalent_schemas_and_status_codes(stage6f_env):
    """Verifies identical error status codes and schemas between old and new aliases."""
    client, _ = stage6f_env
    non_existent_id = uuid.uuid4()

    # 404 on nonexistent paper understanding
    r_v1 = client.post(f"/api/v1/papers/{non_existent_id}/understand")
    r_v0 = client.post(f"/api/papers/{non_existent_id}/understand")
    assert r_v1.status_code == 404
    assert r_v0.status_code == 404
    assert r_v1.json() == r_v0.json()

    # 404 on nonexistent paper knowledge
    r_v1_k = client.get(f"/api/v1/papers/{non_existent_id}/knowledge")
    r_v0_k = client.get(f"/api/papers/{non_existent_id}/knowledge")
    assert r_v1_k.status_code == 404
    assert r_v0_k.status_code == 404
    assert r_v1_k.json() == r_v0_k.json()

    # Document passages for unknown document returns empty list on both
    r_v1_p = client.get(f"/api/v1/documents/{non_existent_id}/passages")
    r_v0_p = client.get(f"/api/documents/{non_existent_id}/passages")
    assert r_v1_p.status_code == 200
    assert r_v0_p.status_code == 200
    assert r_v1_p.json() == []
    assert r_v0_p.json() == []


def test_3_opportunity_catalog_conforms_to_opportunity_list_item_dto(stage6f_env):
    """Verifies GET /api/v1/opportunities and /api/opportunities conform to OpportunityListItemDTO."""
    client, SessionMaker = stage6f_env
    db = SessionMaker()

    # Seed native opportunity + bridged record
    opp_id = uuid.uuid4()
    gap_id = uuid.uuid4()

    rec = ResearchOpportunityRecord(
        id=opp_id,
        gap_id=gap_id,
        research_problem="Partition consensus failure",
        identified_gap="Static timeout in WAN consensus",
        proposed_research_direction="Adaptive reinforcement consensus timeouts",
        reasoning="Recurring partition delays in WAN deployments.",
        technical_approach="Reinforcement learning adaptive timeout tuning.",
        expected_contribution="Sub-second failover recovery.",
        feasibility={"feasibility_score": 0.85},
        novelty_assessment={"assessment": "Potentially novel within the analyzed corpus"},
        generation_model="qwen3:8b",
        validation_model="qwen3:8b",
        embedding_model="nomic-embed-text",
        status="VALIDATED",
        confidence=0.89,
        created_at=datetime.datetime(2023, 1, 2, tzinfo=datetime.timezone.utc),
        supporting_statement_ids=[str(uuid.uuid4()), str(uuid.uuid4())],
        supporting_paper_ids=[str(uuid.uuid4())],
    )
    db.add(rec)

    native_opp = Opportunity(
        id=opp_id,
        research_opportunity_id=rec.id,
        gap_id=gap_id,
        title="Partition-Tolerant Consensus",
        problem_statement="Network partition consensus delay.",
        description="Dynamic timing consensus.",
        status="OPEN",
        confidence=0.87,
        created_at=datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc),
    )
    db.add(native_opp)

    # Seed unbridged opportunity
    unbridged_id = uuid.uuid4()
    rec_unbridged = ResearchOpportunityRecord(
        id=unbridged_id,
        gap_id=gap_id,
        research_problem="Unbridged problem",
        identified_gap="Unbridged gap",
        proposed_research_direction="Unbridged direction",
        reasoning="Unbridged reasoning",
        technical_approach="Unbridged approach",
        expected_contribution="Unbridged contribution",
        feasibility={"feasibility_score": 0.8},
        novelty_assessment={"assessment": "Potentially novel"},
        generation_model="qwen3:8b",
        validation_model="qwen3:8b",
        embedding_model="nomic-embed-text",
        status="PROPOSED",
        confidence=0.75,
        created_at=datetime.datetime(2023, 1, 3, tzinfo=datetime.timezone.utc),
        supporting_statement_ids=[],
        supporting_paper_ids=[],
    )
    db.add(rec_unbridged)
    db.commit()

    # Test both /api/v1/opportunities and /api/opportunities
    for path in ("/api/v1/opportunities", "/api/opportunities"):
        res = client.get(path)
        assert res.status_code == 200
        items = res.json()
        assert len(items) == 2

        # Validate each item through Pydantic OpportunityListItemDTO
        for item in items:
            dto = OpportunityListItemDTO(**item)
            assert isinstance(dto.id, uuid.UUID)
            assert isinstance(dto.gap_id, uuid.UUID)
            assert dto.research_problem != ""
            assert dto.identified_gap != ""
            assert dto.proposed_research_direction != ""
            assert dto.status in ("VALIDATED", "PROPOSED")

        # Specific item checks
        item_map = {str(i["id"]): i for i in items}
        bridged = item_map[str(opp_id)]
        assert bridged["lifecycle_opportunity_id"] == str(opp_id)
        assert bridged["lifecycle_status"] == "OPEN"
        assert bridged["lifecycle_confidence"] == 0.87
        assert bridged["supporting_statement_count"] == 2
        assert bridged["supporting_paper_count"] == 1
        assert bridged["proposed_research_direction"] == "Adaptive reinforcement consensus timeouts"

        unbridged = item_map[str(unbridged_id)]
        assert unbridged["lifecycle_opportunity_id"] is None
        assert unbridged["lifecycle_status"] is None
        assert unbridged["evidence_count"] == 0
        assert unbridged["proposed_research_direction"] == "Unbridged direction"


# ==============================================================================
# Task 3 Tests: Duplicate-Ingestion ID Resolution & Safety
# ==============================================================================

def test_4_duplicate_ingestion_returns_existing_paper_id_without_leakage(stage6f_env):
    """Verifies duplicate ingestion returns existing paper UUID without creating duplicate DB rows."""
    client, SessionMaker = stage6f_env
    pdf_bytes = create_sample_pdf_bytes("Duplicate Idempotency")

    mock_work = {
        "id": "https://openalex.org/W8880001",
        "doi": "https://doi.org/10.1000/dedup1",
        "title": "Idempotent Ingestion Verification",
        "publication_year": 2023,
        "publication_date": "2023-05-01",
        "abstract_inverted_index": {"Idempotent": [0], "deduplication": [1]},
        "authorships": [{"author": {"display_name": "Author Dedup"}}],
        "open_access": {"is_oa": True, "oa_url": "https://example.org/dedup.pdf"},
        "best_oa_location": {"is_oa": True, "pdf_url": "https://example.org/dedup.pdf"},
        "cited_by_count": 5,
    }

    def mock_urlopen(req, timeout=30.0):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "api.openalex.org" in url:
            payload = json.dumps({"results": [mock_work]}).encode("utf-8")
            return DummyStreamResponse(payload, {"Content-Type": "application/json"})
        if "example.org/dedup.pdf" in url:
            return DummyStreamResponse(pdf_bytes, {"Content-Type": "application/pdf"})
        raise ValueError(f"Unexpected url: {url}")

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        # First ingestion
        res1 = client.post("/api/v1/papers/ingest/openalex", json={"openalex_id": "W8880001"})
        assert res1.status_code == 200
        data1 = res1.json()
        assert data1["status"] == IngestionStatus.INGESTED
        orig_paper_id = data1["paper_id"]
        assert orig_paper_id is not None

        db = SessionMaker()
        papers_count_1 = db.query(func.count(Paper.id)).scalar()
        docs_count_1 = db.query(func.count(PaperDocument.id)).scalar()
        passages_count_1 = db.query(func.count(PaperPassage.id)).scalar()
        assert papers_count_1 == 1
        assert docs_count_1 == 1

        # Second ingestion: duplicate request
        res2 = client.post("/api/v1/papers/ingest/openalex", json={"openalex_id": "W8880001"})
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["status"] == IngestionStatus.SKIPPED_DUPLICATE
        # Crucial Stage 6F requirement: returns existing paper UUID
        assert data2["paper_id"] == orig_paper_id

        # Verification: Row counts must remain strictly identical
        papers_count_2 = db.query(func.count(Paper.id)).scalar()
        docs_count_2 = db.query(func.count(PaperDocument.id)).scalar()
        passages_count_2 = db.query(func.count(PaperPassage.id)).scalar()
        assert papers_count_2 == papers_count_1
        assert docs_count_2 == docs_count_1
        assert passages_count_2 == passages_count_1


def test_5_ambiguous_duplicate_resolution_fails_safely(stage6f_env):
    """Verifies that conflicting source ID vs DOI matches or multiple matches return None rather than guessing."""
    _, SessionMaker = stage6f_env
    db = SessionMaker()
    persistence = SQLAlchemyPersistenceLayer(db)
    detector = DefaultDuplicateDetector()

    id_a = uuid.uuid4()
    paper_a = Paper(
        id=id_a,
        source_type="OPENALEX_API",
        source_id="W111111",
        doi="10.1000/doi_a",
        title="Paper A",
        publication_date=datetime.date(2023, 1, 1),
        batch_id="Batch 0",
        abstract="Abstract A",
        authors=["Author A"],
        referenced_works=[],
        raw_metadata={"has_pdf": False},
    )
    db.add(paper_a)

    id_b = uuid.uuid4()
    paper_b = Paper(
        id=id_b,
        source_type="OPENALEX_API",
        source_id="W222222",
        doi="10.1000/doi_b",
        title="Paper B",
        publication_date=datetime.date(2023, 2, 1),
        batch_id="Batch 0",
        abstract="Abstract B",
        authors=["Author B"],
        referenced_works=[],
        raw_metadata={"has_pdf": False},
    )
    db.add(paper_b)
    db.commit()

    # Case 1: Unambiguous match by source ID
    req_a = PaperCreateDTO(
        source_type=SourceType.OPENALEX_API,
        source_id="W111111",
        doi="10.1000/doi_a",
        title="Paper A duplicate",
        publication_date=datetime.date(2023, 1, 1),
        batch_id="Batch 0",
    )
    resolved_a = detector.resolve_duplicate_paper_id(req_a, persistence)
    assert resolved_a == id_a

    # Case 2: Ambiguous conflict — source ID points to Paper A, but DOI points to Paper B!
    req_conflicting = PaperCreateDTO(
        source_type=SourceType.OPENALEX_API,
        source_id="W111111",  # matches Paper A
        doi="10.1000/doi_b",  # matches Paper B
        title="Conflicting Paper",
        publication_date=datetime.date(2023, 1, 1),
        batch_id="Batch 0",
    )
    resolved_conflict = detector.resolve_duplicate_paper_id(req_conflicting, persistence)
    assert resolved_conflict is None, "Ambiguous conflict between source_id and DOI must return None!"

    # Case 3: Unknown paper identifiers return None
    req_unknown = PaperCreateDTO(
        source_type=SourceType.OPENALEX_API,
        source_id="W999999",
        doi="10.1000/doi_unknown",
        title="Unknown Paper",
        publication_date=datetime.date(2023, 1, 1),
        batch_id="Batch 0",
    )
    resolved_unknown = detector.resolve_duplicate_paper_id(req_unknown, persistence)
    assert resolved_unknown is None


# ==============================================================================
# Task 4 Tests: OpenAlex Search count & total_count Semantics
# ==============================================================================

def test_6_openalex_search_count_and_total_count_semantics(stage6f_env):
    """Verifies count represents batch size while total_count represents upstream hits."""
    client, _ = stage6f_env

    # 1. OpenAlex search response containing 2 results in batch out of 1542 total
    mock_payload = {
        "meta": {"count": 1542, "page": 1, "per_page": 2},
        "results": [
            {
                "id": "https://openalex.org/W101",
                "doi": "https://doi.org/10.1000/101",
                "title": "Quantum Error Correction I",
                "publication_year": 2023,
                "publication_date": "2023-01-01",
                "abstract_inverted_index": {"Quantum": [0]},
                "authorships": [{"author": {"display_name": "Author One"}}],
                "open_access": {"is_oa": True, "oa_url": "https://example.org/101.pdf"},
                "best_oa_location": {"is_oa": True, "pdf_url": "https://example.org/101.pdf"},
                "cited_by_count": 50,
            },
            {
                "id": "https://openalex.org/W102",
                "doi": "https://doi.org/10.1000/102",
                "title": "Quantum Error Correction II",
                "publication_year": 2023,
                "publication_date": "2023-02-01",
                "abstract_inverted_index": {"Quantum": [0]},
                "authorships": [{"author": {"display_name": "Author Two"}}],
                "open_access": {"is_oa": False},
                "best_oa_location": None,
                "cited_by_count": 12,
            },
        ],
    }

    with patch.object(OpenAlexSearchProvider, "_execute_request", return_value=mock_payload):
        # Both /api/v1/search/openalex and /api/search/openalex
        for search_url in ("/api/v1/search/openalex?query=quantum", "/api/search/openalex?query=quantum"):
            res = client.get(search_url)
            assert res.status_code == 200
            data = res.json()
            # Contract verification:
            assert data["count"] == 2, "count must preserve the number of returned results in this batch"
            assert data["total_count"] == 1542, "total_count must expose upstream total matching hit count"
            assert len(data["papers"]) == 2
            # Conforms to Pydantic schema
            parsed = OpenAlexSearchResponse(**data)
            assert parsed.count == 2
            assert parsed.total_count == 1542


def test_7_manual_upload_and_error_transaction_behavior(stage6f_env):
    """Verifies manual PDF upload compatibility and transactional rollback on ingestion failure."""
    client, SessionMaker = stage6f_env
    pdf_bytes = create_sample_pdf_bytes("Manual Upload Test")

    # 1. Manual PDF upload endpoint
    upload_res = client.post(
        "/api/papers/upload",
        files=[("files", ("manual_test.pdf", pdf_bytes, "application/pdf"))],
    )
    assert upload_res.status_code == 200
    upload_data = upload_res.json()
    assert upload_data["source"] == "UPLOAD"
    assert upload_data["count"] == 1
    assert len(upload_data["papers"]) == 1

    # 2. Transaction rollback on persistence failure in OpenAlex ingestion
    pdf_bytes_tx = create_sample_pdf_bytes("Tx Rollback Distinct Title", "Distinct body for rollback verification.")
    mock_work = {
        "id": "https://openalex.org/W999901",
        "doi": "https://doi.org/10.1000/txrollback",
        "title": "Tx Rollback Test",
        "publication_year": 2023,
        "publication_date": "2023-01-01",
        "abstract_inverted_index": {"Test": [0]},
        "authorships": [{"author": {"display_name": "Author Tx"}}],
        "open_access": {"is_oa": True, "oa_url": "https://example.org/tx.pdf"},
        "best_oa_location": {"is_oa": True, "pdf_url": "https://example.org/tx.pdf"},
    }

    def mock_urlopen(req, timeout=30.0):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "api.openalex.org" in url:
            payload = json.dumps({"results": [mock_work]}).encode("utf-8")
            return DummyStreamResponse(payload, {"Content-Type": "application/json"})
        if "example.org/tx.pdf" in url:
            return DummyStreamResponse(pdf_bytes_tx, {"Content-Type": "application/pdf"})
        raise ValueError(f"Unexpected url: {url}")

    db = SessionMaker()
    papers_before = db.query(func.count(Paper.id)).scalar()

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        # Force a failure in passage saving to trigger transaction rollback
        with patch.object(SQLAlchemyPersistenceLayer, "save_passages", side_effect=RuntimeError("Disk failure")):
            res = client.post("/api/v1/papers/ingest/openalex", json={"openalex_id": "W999901"})
            # Ingestion service catches persistence errors and returns status ERROR (fail-closed)
            assert res.status_code == 200
            assert res.json()["status"] == IngestionStatus.ERROR
            assert "Disk failure" in res.json()["error_message"]

    # Verify no partial paper was leaked into DB
    papers_after = db.query(func.count(Paper.id)).scalar()
    assert papers_after == papers_before
