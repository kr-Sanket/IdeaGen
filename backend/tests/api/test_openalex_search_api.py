"""Tests for OpenAlex Search API Integration (Stage 6B).

Verifies:
1. Successful topic search (/api/v1/search/openalex and /api/search/openalex).
2. Query and limit validation (empty query, limit < 1, limit > 100, query alias 'q').
3. Date-range validation and filtering (invalid format, start_date > end_date, valid dates).
4. Empty search results (200 OK with count=0, papers=[]).
5. Upstream timeout or provider error (502 Bad Gateway).
6. Missing optional metadata (graceful fallback for missing abstract, authors, doi, oa info).
7. Response schema consistency (strictly validates against OpenAlexSearchResponse).
8. Backward compatibility of existing ArXiv search endpoint (/api/search/papers).
9. Compatibility with Stage 6A ingestion request (passing search result identifiers).
10. No database persistence or ingestion side effects during search.
"""

from __future__ import annotations

import io
import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ideagen.db.models import Base, Paper, PaperDocument, PaperPassage, Opportunity
from ideagen.db import get_db
from ideagen.api.main import app
from ideagen.api.schema import OpenAlexSearchResponse, OpenAlexPaperResultDTO


# In-memory SQLite fixture for database isolation
@pytest.fixture(name="db_session")
def db_session_fixture():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(name="client")
def client_fixture(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    test_client = TestClient(app)
    try:
        yield test_client
    finally:
        app.dependency_overrides.clear()


class DummyStreamResponse:
    def __init__(self, data: bytes, headers: Optional[Dict[str, str]] = None, status: int = 200):
        self.bio = io.BytesIO(data)
        self.headers = headers if headers is not None else {"Content-Type": "application/json"}
        self.status = status

    def read(self, amt: int = -1) -> bytes:
        return self.bio.read(amt)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


SAMPLE_OPENALEX_WORK: Dict[str, Any] = {
    "id": "https://openalex.org/W2741809807",
    "doi": "https://doi.org/10.1145/3173574.3173772",
    "title": "Quantum Error Correction and Fault Tolerance in Superconducting Qubits",
    "display_name": "Quantum Error Correction and Fault Tolerance in Superconducting Qubits",
    "publication_year": 2024,
    "publication_date": "2024-03-15",
    "abstract_inverted_index": {
        "Quantum": [0],
        "error": [1],
        "correction": [2],
        "is": [3],
        "vital": [4],
        "for": [5],
        "scalable": [6],
        "quantum": [7],
        "computing.": [8],
    },
    "authorships": [
        {
            "author": {
                "id": "https://openalex.org/A5001234567",
                "display_name": "Alice Smith",
            }
        },
        {
            "author": {
                "id": "https://openalex.org/A5007654321",
                "display_name": "Bob Jones",
            }
        },
    ],
    "open_access": {
        "is_oa": True,
        "oa_status": "gold",
        "oa_url": "https://example.org/sample_paper.pdf",
    },
    "best_oa_location": {
        "is_oa": True,
        "pdf_url": "https://example.org/sample_paper.pdf",
        "landing_page_url": "https://example.org/sample_landing",
    },
    "primary_location": {
        "is_oa": True,
        "pdf_url": "https://example.org/sample_paper.pdf",
        "landing_page_url": "https://example.org/sample_landing",
    },
    "cited_by_count": 42,
}


def test_1_successful_topic_search(client, monkeypatch):
    """Test 1: Verify successful topic search across both /api/v1 and /api endpoints."""
    mock_payload = {
        "meta": {"count": 1, "db_response_time_ms": 12},
        "results": [SAMPLE_OPENALEX_WORK],
    }
    encoded = json.dumps(mock_payload).encode("utf-8")

    captured_urls = []

    def mock_urlopen(req, timeout=10):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        captured_urls.append(url)
        return DummyStreamResponse(encoded)

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    # 1. Primary endpoint: /api/v1/search/openalex
    res1 = client.get("/api/v1/search/openalex?query=quantum+error+correction&limit=5")
    assert res1.status_code == 200
    data1 = res1.json()

    assert data1["source"] == "OPENALEX"
    assert data1["query"] == "quantum error correction"
    assert data1["count"] == 1
    assert len(data1["papers"]) == 1

    paper = data1["papers"][0]
    assert paper["id"] == "W2741809807"
    assert paper["doi"] == "10.1145/3173574.3173772"
    assert paper["title"] == "Quantum Error Correction and Fault Tolerance in Superconducting Qubits"
    assert paper["authors"] == ["Alice Smith", "Bob Jones"]
    assert paper["publication_year"] == 2024
    assert paper["publication_date"] == "2024-03-15"
    assert paper["abstract"] == "Quantum error correction is vital for scalable quantum computing."
    assert paper["is_oa"] is True
    assert paper["pdf_url"] == "https://example.org/sample_paper.pdf"
    assert paper["landing_page_url"] == "https://example.org/sample_landing"
    assert paper["citation_count"] == 42

    # 2. Alias endpoint: /api/search/openalex
    res2 = client.get("/api/search/openalex?query=quantum+error+correction&limit=5")
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["count"] == 1
    assert data2["papers"][0]["id"] == "W2741809807"


def test_2_query_and_limit_validation(client):
    """Test 2: Validate empty query, alias 'q', and limit boundaries."""
    # Empty query or missing query
    res = client.get("/api/v1/search/openalex")
    assert res.status_code == 400
    assert "cannot be empty" in res.json()["detail"]

    res_empty = client.get("/api/v1/search/openalex?query=   ")
    assert res_empty.status_code == 400
    assert "cannot be empty" in res_empty.json()["detail"]

    # Limit below bounds (< 1)
    res_low = client.get("/api/v1/search/openalex?query=test&limit=0")
    assert res_low.status_code == 400
    assert "Invalid limit" in res_low.json()["detail"]

    res_neg = client.get("/api/v1/search/openalex?query=test&limit=-5")
    assert res_neg.status_code == 400
    assert "Invalid limit" in res_neg.json()["detail"]

    # Limit above bounds (> 100)
    res_high = client.get("/api/v1/search/openalex?query=test&limit=101")
    assert res_high.status_code == 400
    assert "Invalid limit" in res_high.json()["detail"]


def test_2b_query_alias_q_supported(client, monkeypatch):
    """Test 2b: Verify alias parameter 'q' functions identically to 'query'."""
    mock_payload = {"meta": {"count": 0}, "results": []}
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=10: DummyStreamResponse(json.dumps(mock_payload).encode("utf-8")),
    )

    res = client.get("/api/v1/search/openalex?q=neural+networks&limit=10")
    assert res.status_code == 200
    data = res.json()
    assert data["query"] == "neural networks"
    assert data["count"] == 0


def test_3_date_range_validation_and_filtering(client, monkeypatch):
    """Test 3: Validate ISO date parsing, start > end error, and valid filter passing."""
    # Invalid start_date format
    res = client.get("/api/v1/search/openalex?query=test&start_date=2024/01/01")
    assert res.status_code == 400
    assert "Invalid start_date format" in res.json()["detail"]

    # Invalid end_date format
    res = client.get("/api/v1/search/openalex?query=test&end_date=invalid-date")
    assert res.status_code == 400
    assert "Invalid end_date format" in res.json()["detail"]

    # start_date > end_date
    res = client.get("/api/v1/search/openalex?query=test&start_date=2024-05-01&end_date=2024-01-01")
    assert res.status_code == 400
    assert "cannot be later than end_date" in res.json()["detail"]

    # Valid date range passed to provider
    captured_urls = []

    def mock_urlopen(req, timeout=10):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        captured_urls.append(url)
        return DummyStreamResponse(json.dumps({"results": []}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    res = client.get(
        "/api/v1/search/openalex?query=transformers&start_date=2023-01-01&end_date=2023-12-31&limit=25"
    )
    assert res.status_code == 200
    assert len(captured_urls) == 1
    assert "from_publication_date%3A2023-01-01" in captured_urls[0] or "from_publication_date:2023-01-01" in captured_urls[0]
    assert "to_publication_date%3A2023-12-31" in captured_urls[0] or "to_publication_date:2023-12-31" in captured_urls[0]


def test_4_empty_search_results(client, monkeypatch):
    """Test 4: Upstream returns empty results list -> returns HTTP 200 with count=0, papers=[]."""
    mock_payload = {"meta": {"count": 0}, "results": []}
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=10: DummyStreamResponse(json.dumps(mock_payload).encode("utf-8")),
    )

    res = client.get("/api/v1/search/openalex?query=totallyunknownconceptdoesnotexist")
    assert res.status_code == 200
    data = res.json()
    assert data["query"] == "totallyunknownconceptdoesnotexist"
    assert data["source"] == "OPENALEX"
    assert data["count"] == 0
    assert data["papers"] == []


def test_5_upstream_timeout_and_provider_error(client, monkeypatch):
    """Test 5: Upstream OpenAlex failures or timeouts map cleanly to HTTP 502."""
    def mock_urlopen_fail(req, timeout=10):
        raise urllib.error.URLError("Connection timed out or DNS resolution failed")

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_fail)

    res = client.get("/api/v1/search/openalex?query=quantum")
    assert res.status_code == 502
    assert "Upstream OpenAlex service error" in res.json()["detail"]


def test_6_missing_optional_metadata(client, monkeypatch):
    """Test 6: Gracefully handle works with missing optional fields (None abstract, authors, doi)."""
    sparse_work = {
        "id": "W9999999999",
        "title": "Minimal Sparse Paper",
        # missing abstract_inverted_index
        # missing authorships
        # missing doi
        # missing open_access
        # missing best_oa_location
        # missing publication_date
        # missing cited_by_count
    }
    mock_payload = {"results": [sparse_work]}
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=10: DummyStreamResponse(json.dumps(mock_payload).encode("utf-8")),
    )

    res = client.get("/api/v1/search/openalex?query=sparse")
    assert res.status_code == 200
    data = res.json()
    assert data["count"] == 1
    p = data["papers"][0]
    assert p["id"] == "W9999999999"
    assert p["title"] == "Minimal Sparse Paper"
    assert p["abstract"] == ""
    assert p["authors"] == []
    assert p["doi"] is None
    assert p["is_oa"] is False
    assert p["pdf_url"] is None
    assert p["citation_count"] == 0


def test_7_response_schema_consistency(client, monkeypatch):
    """Test 7: Confirm strict schema validation using Pydantic OpenAlexSearchResponse."""
    mock_payload = {"results": [SAMPLE_OPENALEX_WORK]}
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=10: DummyStreamResponse(json.dumps(mock_payload).encode("utf-8")),
    )

    res = client.get("/api/v1/search/openalex?query=test")
    assert res.status_code == 200
    validated = OpenAlexSearchResponse.model_validate(res.json())
    assert validated.count == 1
    assert isinstance(validated.papers[0], OpenAlexPaperResultDTO)
    assert validated.papers[0].id == "W2741809807"


def test_8_arxiv_search_endpoint_compatibility(client, monkeypatch):
    """Test 8: Ensure existing GET /api/search/papers ArXiv search remains completely functional."""
    mock_arxiv_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2104.12345v1</id>
        <published>2021-04-20T00:00:00Z</published>
        <updated>2021-04-20T00:00:00Z</updated>
        <title>ArXiv Compatibility Paper</title>
        <summary>Summary of arxiv paper</summary>
        <author><name>ArXiv Author</name></author>
        <link title="doi" href="https://doi.org/10.48550/arXiv.2104.12345"/>
      </entry>
    </feed>
    """
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=10: DummyStreamResponse(mock_arxiv_xml, {"Content-Type": "application/xml"}),
    )

    res = client.get("/api/search/papers?q=deep+learning&limit=5")
    assert res.status_code == 200
    data = res.json()
    assert data["source"] == "ARXIV"
    assert data["query"] == "deep learning"
    assert data["count"] == 1
    assert data["papers"][0]["source"] == "ARXIV"
    assert data["papers"][0]["title"] == "ArXiv Compatibility Paper"
    assert "raw_metadata" in data["papers"][0]


def test_9_compatibility_with_stage6a_ingestion(client, monkeypatch):
    """Test 9: Verify returned search result identifiers can be passed directly to Stage 6A ingestion."""
    # 1. Search OpenAlex
    mock_search_payload = {"results": [SAMPLE_OPENALEX_WORK]}
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=10: DummyStreamResponse(json.dumps(mock_search_payload).encode("utf-8")),
    )

    search_res = client.get("/api/v1/search/openalex?query=quantum+error+correction")
    assert search_res.status_code == 200
    papers = search_res.json()["papers"]
    assert len(papers) > 0

    first_paper = papers[0]
    extracted_id = first_paper["id"]  # 'W2741809807'
    extracted_doi = first_paper["doi"]  # '10.1145/3173574.3173772'

    # 2. Feed extracted identifiers directly into Stage 6A endpoint
    # Mock OpenAlex work lookup & PDF download for ingestion
    work_for_ingest = {
        **SAMPLE_OPENALEX_WORK,
        "best_oa_location": {"is_oa": False, "pdf_url": None},  # NO_PDF path to keep test lightweight
    }
    mock_ingest_lookup = {"results": [work_for_ingest]}

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=10: DummyStreamResponse(json.dumps(mock_ingest_lookup).encode("utf-8")),
    )

    ingest_payload = {
        "openalex_id": extracted_id,
        "doi": extracted_doi,
        "enable_evolution": False,
    }
    ingest_res = client.post("/api/v1/papers/ingest/openalex", json=ingest_payload)
    assert ingest_res.status_code == 200
    ingest_data = ingest_res.json()
    assert ingest_data["source_id"] == extracted_id
    assert ingest_data["doi"] == extracted_doi
    assert ingest_data["status"] == "NO_PDF"


def test_10_no_database_persistence_side_effects(client, db_session, monkeypatch):
    """Test 10: Ensure search endpoint has zero side-effects on database state."""
    mock_payload = {"results": [SAMPLE_OPENALEX_WORK]}
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=10: DummyStreamResponse(json.dumps(mock_payload).encode("utf-8")),
    )

    # Initial counts
    papers_before = db_session.query(func.count(Paper.id)).scalar()
    docs_before = db_session.query(func.count(PaperDocument.id)).scalar()
    passages_before = db_session.query(func.count(PaperPassage.id)).scalar()
    opps_before = db_session.query(func.count(Opportunity.id)).scalar()

    # Perform multiple searches
    for _ in range(5):
        res = client.get("/api/v1/search/openalex?query=quantum+error+correction&limit=10")
        assert res.status_code == 200

    # Verification of zero mutations
    papers_after = db_session.query(func.count(Paper.id)).scalar()
    docs_after = db_session.query(func.count(PaperDocument.id)).scalar()
    passages_after = db_session.query(func.count(PaperPassage.id)).scalar()
    opps_after = db_session.query(func.count(Opportunity.id)).scalar()

    assert papers_before == papers_after == 0
    assert docs_before == docs_after == 0
    assert passages_before == passages_after == 0
    assert opps_before == opps_after == 0
