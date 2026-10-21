"""Tests for OpenAlex Paper Ingestion API Endpoint (Stage 6A).

Verifies:
1. Valid OpenAlex ID ingestion (/api/v1/papers/ingest/openalex).
2. DOI-based ingestion.
3. Missing identifier and invalid input validation (HTTP 400).
4. Deterministic precedence when both OpenAlex ID and DOI are supplied.
5. Fallback from OpenAlex ID to DOI if OpenAlex ID returns no work.
6. Metadata-only NO_PDF result when PDF is absent or extraction fails.
7. Duplicate paper detection (SKIPPED_DUPLICATE).
8. Upstream OpenAlex 404 (paper not found) and 502 (upstream error).
9. Request-scoped session durability across fresh sessions.
10. Default-disabled post-ingestion evolution hook and explicit override behavior.
11. Backward compatibility: existing /api/papers/upload route remains intact.
"""

from __future__ import annotations

import io
import json
import urllib.request
import urllib.error
import uuid
import datetime
from typing import Dict, Any, Optional
from unittest.mock import patch, MagicMock
import pymupdf
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ideagen.db.models import Base, Paper, PaperDocument, PaperPassage, Opportunity
from ideagen.db import get_db
from ideagen.api.main import app
from ideagen.ingestion.openalex_ingestion import IngestionStatus, EvolutionHookStatus
from ideagen.opportunity.models import OpportunityStatus


def create_sample_pdf_bytes(title: str = "Test Title", body: str = "Detailed scientific research text. ") -> bytes:
    """Generates valid, parseable in-memory PDF bytes."""
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
    """Simulates an urllib response stream for PDF downloading and search requests."""
    def __init__(self, data: bytes, headers: Optional[Dict[str, str]] = None, status: int = 200):
        self.bio = io.BytesIO(data)
        self.headers = headers if headers is not None else {"Content-Length": str(len(data))}
        self.status = status

    def read(self, amt: int = -1) -> bytes:
        return self.bio.read(amt)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


MOCK_WORK_DATA = {
    "id": "https://openalex.org/W2741809807",
    "doi": "https://doi.org/10.48550/arxiv.2005.11401",
    "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "publication_year": 2020,
    "publication_date": "2020-05-22",
    "cited_by_count": 1540,
    "abstract_inverted_index": {
        "We": [0],
        "explore": [1],
        "RAG": [2],
        "models": [3],
        "combining": [4],
        "parametric": [5],
        "and": [6],
        "non-parametric": [7],
        "memory.": [8],
    },
    "authorships": [
        {"author": {"display_name": "Patrick Lewis"}},
        {"author": {"display_name": "Ethan Perez"}},
    ],
    "open_access": {
        "is_oa": True,
        "oa_status": "green",
        "oa_url": "https://arxiv.org/abs/2005.11401",
    },
    "best_oa_location": {
        "pdf_url": "https://arxiv.org/pdf/2005.11401.pdf",
        "landing_page_url": "https://arxiv.org/abs/2005.11401",
        "is_oa": True,
    },
    "primary_location": {
        "pdf_url": "https://arxiv.org/pdf/2005.11401.pdf",
        "landing_page_url": "https://arxiv.org/abs/2005.11401",
        "is_oa": True,
    },
}


@pytest.fixture
def test_client_and_session():
    """Provides an isolated database and TestClient with clean dependency overrides."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionMaker = sessionmaker(bind=engine)

    def override_get_db():
        db = SessionMaker()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    yield client, SessionMaker

    app.dependency_overrides.clear()


# ── Scenario 1: Valid OpenAlex ID Ingestion ──────────────────────────────

def test_ingest_by_openalex_id_success(test_client_and_session):
    client, SessionMaker = test_client_and_session
    pdf_bytes = create_sample_pdf_bytes("RAG for Knowledge Tasks")

    def mock_urlopen(req, timeout=30.0):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "api.openalex.org" in url:
            payload = json.dumps({"results": [MOCK_WORK_DATA]}).encode("utf-8")
            return DummyStreamResponse(payload, {"Content-Type": "application/json"})
        if "arxiv.org/pdf" in url:
            return DummyStreamResponse(pdf_bytes, {"Content-Type": "application/pdf"})
        raise ValueError(f"Unexpected URL: {url}")

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        resp = client.post(
            "/api/v1/papers/ingest/openalex",
            json={"openalex_id": "W2741809807"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == IngestionStatus.INGESTED
    assert data["has_pdf"] is True
    assert data["source_id"] == "W2741809807"
    assert data["doi"] == "10.48550/arxiv.2005.11401"
    assert data["batch_id"] == "Batch 0"
    assert data["passage_count"] > 0
    assert data["paper_id"] is not None
    assert data["document_id"] is not None
    assert data["evolution_status"] == EvolutionHookStatus.DISABLED

    # Verify durable persistence in a fresh session
    fresh_session = SessionMaker()
    paper = fresh_session.execute(
        select(Paper).where(Paper.source_id == "W2741809807")
    ).scalar_one()
    assert paper.title == "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
    assert paper.abstract.startswith("We explore RAG models")

    doc = fresh_session.execute(
        select(PaperDocument).where(PaperDocument.paper_id == paper.id)
    ).scalar_one()
    assert doc is not None

    passages = fresh_session.execute(
        select(PaperPassage).where(PaperPassage.paper_id == paper.id)
    ).scalars().all()
    assert len(passages) == data["passage_count"]
    fresh_session.close()


# ── Scenario 2: DOI-based Ingestion ──────────────────────────────────────

def test_ingest_by_doi_success(test_client_and_session):
    client, SessionMaker = test_client_and_session
    pdf_bytes = create_sample_pdf_bytes("RAG for Knowledge Tasks")

    def mock_urlopen(req, timeout=30.0):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "api.openalex.org" in url:
            payload = json.dumps({"results": [MOCK_WORK_DATA]}).encode("utf-8")
            return DummyStreamResponse(payload, {"Content-Type": "application/json"})
        if "arxiv.org/pdf" in url:
            return DummyStreamResponse(pdf_bytes, {"Content-Type": "application/pdf"})
        raise ValueError(f"Unexpected URL: {url}")

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        resp = client.post(
            "/api/v1/papers/ingest/openalex",
            json={"doi": "10.48550/arxiv.2005.11401"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == IngestionStatus.INGESTED
    assert data["doi"] == "10.48550/arxiv.2005.11401"
    assert data["source_id"] == "W2741809807"
    assert data["paper_id"] is not None


# ── Scenario 3: Missing Identifier and Invalid Input (HTTP 400) ──────────

def test_missing_identifier_returns_400(test_client_and_session):
    client, _ = test_client_and_session
    resp = client.post("/api/v1/papers/ingest/openalex", json={})
    assert resp.status_code == 400
    assert "Either 'openalex_id' or 'doi' must be provided" in resp.json()["detail"]


def test_empty_string_identifier_returns_400(test_client_and_session):
    client, _ = test_client_and_session
    resp = client.post(
        "/api/v1/papers/ingest/openalex",
        json={"openalex_id": "   ", "doi": ""},
    )
    assert resp.status_code == 400
    assert "Either 'openalex_id' or 'doi' must be provided" in resp.json()["detail"]


def test_invalid_openalex_id_format_returns_400(test_client_and_session):
    client, _ = test_client_and_session
    resp = client.post(
        "/api/v1/papers/ingest/openalex",
        json={"openalex_id": "not_an_openalex_id"},
    )
    assert resp.status_code == 400
    assert "Invalid OpenAlex ID format" in resp.json()["detail"]


def test_invalid_doi_format_returns_400(test_client_and_session):
    client, _ = test_client_and_session
    resp = client.post(
        "/api/v1/papers/ingest/openalex",
        json={"doi": "not_a_valid_doi"},
    )
    assert resp.status_code == 400
    assert "Invalid DOI format" in resp.json()["detail"]


# ── Scenario 4: Deterministic Precedence When Both Supplied ──────────────

def test_deterministic_precedence_prioritizes_openalex_id(test_client_and_session):
    client, _ = test_client_and_session
    pdf_bytes = create_sample_pdf_bytes("RAG for Knowledge Tasks")

    requested_urls = []

    def mock_urlopen(req, timeout=30.0):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        requested_urls.append(url)
        if "api.openalex.org" in url:
            payload = json.dumps({"results": [MOCK_WORK_DATA]}).encode("utf-8")
            return DummyStreamResponse(payload, {"Content-Type": "application/json"})
        if "arxiv.org/pdf" in url:
            return DummyStreamResponse(pdf_bytes, {"Content-Type": "application/pdf"})
        raise ValueError(f"Unexpected URL: {url}")

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        resp = client.post(
            "/api/v1/papers/ingest/openalex",
            json={
                "openalex_id": "W2741809807",
                "doi": "10.1000/secondary.doi",
            },
        )

    assert resp.status_code == 200
    # First search request must filter by openalex:W2741809807
    assert any("filter=openalex%3AW2741809807" in u for u in requested_urls)


def test_fallback_to_doi_if_openalex_id_yields_no_work(test_client_and_session):
    client, _ = test_client_and_session
    pdf_bytes = create_sample_pdf_bytes("RAG for Knowledge Tasks")

    def mock_urlopen(req, timeout=30.0):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "openalex%3AW9999999999" in url:
            # First lookup by OpenAlex ID returns empty results
            payload = json.dumps({"results": []}).encode("utf-8")
            return DummyStreamResponse(payload, {"Content-Type": "application/json"})
        if "doi%3A10.48550%2Farxiv.2005.11401" in url:
            # Fallback lookup by DOI succeeds
            payload = json.dumps({"results": [MOCK_WORK_DATA]}).encode("utf-8")
            return DummyStreamResponse(payload, {"Content-Type": "application/json"})
        if "arxiv.org/pdf" in url:
            return DummyStreamResponse(pdf_bytes, {"Content-Type": "application/pdf"})
        raise ValueError(f"Unexpected URL: {url}")

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        resp = client.post(
            "/api/v1/papers/ingest/openalex",
            json={
                "openalex_id": "W9999999999",
                "doi": "10.48550/arxiv.2005.11401",
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == IngestionStatus.INGESTED
    assert data["source_id"] == "W2741809807"


# ── Scenario 5: Metadata-only NO_PDF Result ──────────────────────────────

def test_no_pdf_metadata_only_fallback(test_client_and_session):
    client, SessionMaker = test_client_and_session

    work_no_pdf = dict(MOCK_WORK_DATA)
    work_no_pdf["id"] = "https://openalex.org/W5555555555"
    work_no_pdf["best_oa_location"] = None
    work_no_pdf["primary_location"] = None
    work_no_pdf["open_access"] = {"is_oa": False}

    def mock_urlopen(req, timeout=30.0):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "api.openalex.org" in url:
            payload = json.dumps({"results": [work_no_pdf]}).encode("utf-8")
            return DummyStreamResponse(payload, {"Content-Type": "application/json"})
        raise ValueError(f"Unexpected URL: {url}")

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        resp = client.post(
            "/api/v1/papers/ingest/openalex",
            json={"openalex_id": "W5555555555"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == IngestionStatus.NO_PDF
    assert data["has_pdf"] is False
    assert data["paper_id"] is not None
    assert data["document_id"] is None
    assert data["passage_count"] == 0
    assert data["error_message"] == "NO_PDF_URL"

    # Verify in DB: paper exists, document and passages do NOT exist
    fresh_session = SessionMaker()
    paper = fresh_session.execute(
        select(Paper).where(Paper.source_id == "W5555555555")
    ).scalar_one()
    assert paper is not None

    docs = fresh_session.execute(
        select(PaperDocument).where(PaperDocument.paper_id == paper.id)
    ).scalars().all()
    assert len(docs) == 0

    passages = fresh_session.execute(
        select(PaperPassage).where(PaperPassage.paper_id == paper.id)
    ).scalars().all()
    assert len(passages) == 0
    fresh_session.close()


# ── Scenario 6: Duplicate Paper Detection ────────────────────────────────

def test_duplicate_paper_skips_without_error(test_client_and_session):
    client, _ = test_client_and_session
    pdf_bytes = create_sample_pdf_bytes("RAG for Knowledge Tasks")

    def mock_urlopen(req, timeout=30.0):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "api.openalex.org" in url:
            payload = json.dumps({"results": [MOCK_WORK_DATA]}).encode("utf-8")
            return DummyStreamResponse(payload, {"Content-Type": "application/json"})
        if "arxiv.org/pdf" in url:
            return DummyStreamResponse(pdf_bytes, {"Content-Type": "application/pdf"})
        raise ValueError(f"Unexpected URL: {url}")

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        # 1. First ingestion succeeds
        resp1 = client.post(
            "/api/v1/papers/ingest/openalex",
            json={"openalex_id": "W2741809807"},
        )
        assert resp1.status_code == 200
        assert resp1.json()["status"] == IngestionStatus.INGESTED

        # 2. Duplicate ingestion request
        resp2 = client.post(
            "/api/v1/papers/ingest/openalex",
            json={"openalex_id": "W2741809807"},
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["status"] == IngestionStatus.SKIPPED_DUPLICATE
        assert data2["paper_id"] == resp1.json()["paper_id"]


# ── Scenario 7: Upstream OpenAlex 404 and 502 Failures ───────────────────

def test_paper_not_found_in_openalex_returns_404(test_client_and_session):
    client, _ = test_client_and_session

    def mock_urlopen(req, timeout=30.0):
        payload = json.dumps({"results": []}).encode("utf-8")
        return DummyStreamResponse(payload, {"Content-Type": "application/json"})

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        resp = client.post(
            "/api/v1/papers/ingest/openalex",
            json={"openalex_id": "W1234567890"},
        )

    assert resp.status_code == 404
    assert "Paper not found in OpenAlex" in resp.json()["detail"]


def test_upstream_openalex_network_error_returns_502(test_client_and_session):
    client, _ = test_client_and_session

    def mock_urlopen(req, timeout=30.0):
        raise urllib.error.URLError("Connection refused by upstream host")

    with patch("ideagen.ingestion.search.OpenAlexSearchProvider.search", side_effect=Exception("Connection refused")):
        resp = client.post(
            "/api/v1/papers/ingest/openalex",
            json={"openalex_id": "W1234567890"},
        )

    assert resp.status_code == 502
    assert "Upstream OpenAlex service error" in resp.json()["detail"]


# ── Scenario 8: Evolution Hook Configuration Behavior ────────────────────

def test_evolution_hook_disabled_by_default(test_client_and_session):
    client, _ = test_client_and_session
    pdf_bytes = create_sample_pdf_bytes("RAG for Knowledge Tasks")

    def mock_urlopen(req, timeout=30.0):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "api.openalex.org" in url:
            payload = json.dumps({"results": [MOCK_WORK_DATA]}).encode("utf-8")
            return DummyStreamResponse(payload, {"Content-Type": "application/json"})
        if "arxiv.org/pdf" in url:
            return DummyStreamResponse(pdf_bytes, {"Content-Type": "application/pdf"})
        raise ValueError(f"Unexpected URL: {url}")

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        resp = client.post(
            "/api/v1/papers/ingest/openalex",
            json={"openalex_id": "W2741809807"},
        )

    assert resp.status_code == 200
    assert resp.json()["evolution_status"] == EvolutionHookStatus.DISABLED


def test_explicit_evolution_hook_enabled(test_client_and_session, monkeypatch):
    client, SessionMaker = test_client_and_session
    monkeypatch.setenv("IDEAGEN_ENABLE_POST_INGESTION_EVOLUTION", "true")
    pdf_bytes = create_sample_pdf_bytes("RAG for Knowledge Tasks")

    def mock_urlopen(req, timeout=30.0):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "api.openalex.org" in url:
            payload = json.dumps({"results": [MOCK_WORK_DATA]}).encode("utf-8")
            return DummyStreamResponse(payload, {"Content-Type": "application/json"})
        if "arxiv.org/pdf" in url:
            return DummyStreamResponse(pdf_bytes, {"Content-Type": "application/pdf"})
        raise ValueError(f"Unexpected URL: {url}")

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        resp = client.post(
            "/api/v1/papers/ingest/openalex",
            json={
                "openalex_id": "W2741809807",
                "enable_evolution": True,
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    # Since no opportunities exist in DB, evolution runs and returns SKIPPED (no candidates)
    assert data["evolution_status"] == EvolutionHookStatus.SKIPPED


# ── Scenario 9: Existing Manual PDF Upload Remains Unchanged ─────────────

def test_manual_pdf_upload_backward_compatibility(test_client_and_session):
    client, SessionMaker = test_client_and_session
    pdf_bytes = create_sample_pdf_bytes("Manual Upload Test Paper")

    files = [("files", ("manual_paper.pdf", pdf_bytes, "application/pdf"))]
    resp = client.post("/api/papers/upload", files=files)

    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "UPLOAD"
    assert data["count"] == 1
    assert data["papers"][0]["title"] == "manual_paper.pdf"

    fresh_session = SessionMaker()
    uploaded_paper = fresh_session.execute(
        select(Paper).where(Paper.title == "manual_paper.pdf")
    ).scalar_one()
    assert uploaded_paper.source_type == "MANUAL_UPLOAD"
    fresh_session.close()
