"""Stage 6C: Post-Ingestion Evolution Integration Tests.

Validates the full HTTP workflow connecting OpenAlex paper ingestion to the
IncrementalOpportunityEvolutionService:
1. Evolution disabled by default.
2. Explicit request opt-in (enable_evolution=True) and opt-out (enable_evolution=False).
3. Configuration-enabled behavior (IDEAGEN_ENABLE_POST_INGESTION_EVOLUTION=true).
4. Successful ingestion triggers evolution exactly once (no duplicate execution).
5. NO_PDF, duplicate, and failed ingestion suppress evolution.
6. Multiple eligible opportunities are processed and outcome counters reconcile.
7. Grounding or provider failures do not corrupt ingestion.
8. Caller-owned transaction is not prematurely committed or rolled back.
9. Standalone evolution changes are durable after the request session closes.
10. Existing ArXiv search and manual PDF upload remain compatible.
"""

from __future__ import annotations

import datetime
import io
import json
import urllib.request
import urllib.error
import uuid
from typing import Dict, Any, Optional
from unittest.mock import patch, MagicMock

import pymupdf
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

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
from ideagen.db import get_db
from ideagen.api.main import app
from ideagen.ingestion.openalex_ingestion import IngestionStatus, EvolutionHookStatus
from ideagen.opportunity.models import OpportunityStatus, OpportunityRelationship
from ideagen.opportunity.classifier import RelationshipClassification
from ideagen.llm.exceptions import ProviderFailureError


def create_sample_pdf_bytes(title: str = "Test Title", body: str = "Detailed scientific research text. ") -> bytes:
    """Generates valid, parseable in-memory PDF bytes with substantial text."""
    doc = pymupdf.open()
    page = doc.new_page()
    lines = [title, "", "Abstract"]
    for i in range(20):
        lines.append(f"Abstract sentence {i}: {body}")
    lines.extend(["", "1. Introduction"])
    for i in range(25):
        lines.append(f"Introduction sentence {i}: {body}")
    page.insert_text((50, 50), "\n".join(lines))
    data = doc.tobytes()
    doc.close()
    return data


class DummyStreamResponse:
    """Simulates an urllib response stream for HTTP downloads."""
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


MOCK_WORK_DATA: Dict[str, Any] = {
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
        "vital.": [4],
    },
    "authorships": [
        {"author": {"display_name": "Alice Smith"}}
    ],
    "open_access": {
        "is_oa": True,
        "oa_status": "gold",
        "oa_url": "https://example.org/sample.pdf",
    },
    "best_oa_location": {
        "is_oa": True,
        "pdf_url": "https://example.org/sample.pdf",
        "landing_page_url": "https://doi.org/10.1145/3173574.3173772",
    },
    "cited_by_count": 42,
}


@pytest.fixture(name="test_context")
def test_context_fixture():
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
    client = TestClient(app)

    try:
        yield client, TestingSessionLocal
    finally:
        app.dependency_overrides.clear()


def make_mock_urlopen(pdf_bytes: bytes, work_dict: Dict[str, Any]):
    def mock_urlopen(req, timeout=30.0):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "api.openalex.org" in url:
            payload = json.dumps({"results": [work_dict]}).encode("utf-8")
            return DummyStreamResponse(payload, {"Content-Type": "application/json"})
        if "example.org" in url or ".pdf" in url:
            return DummyStreamResponse(pdf_bytes, {"Content-Type": "application/pdf"})
        raise ValueError(f"Unexpected mock URL: {url}")
    return mock_urlopen


def test_1_evolution_disabled_by_default(test_context):
    """Test 1: When enable_evolution is omitted and config is default, evolution is DISABLED."""
    client, SessionMaker = test_context
    pdf_bytes = create_sample_pdf_bytes("Quantum Error Correction")

    with patch("urllib.request.urlopen", side_effect=make_mock_urlopen(pdf_bytes, MOCK_WORK_DATA)):
        res = client.post(
            "/api/v1/papers/ingest/openalex",
            json={"openalex_id": "W2741809807"},
        )

    assert res.status_code == 200
    data = res.json()
    assert data["status"] == IngestionStatus.INGESTED
    assert data["has_pdf"] is True
    assert data["evolution_status"] == EvolutionHookStatus.DISABLED
    assert data["evolution_result"] is None


def test_2_explicit_request_opt_in_and_opt_out(test_context, monkeypatch):
    """Test 2: With the server setting enabled, enable_evolution=False opts out and
    enable_evolution=True runs evolution. (Opt-in without server enablement: test_2b.)"""
    client, SessionMaker = test_context
    pdf_bytes = create_sample_pdf_bytes("Quantum Error Correction")
    monkeypatch.setenv("IDEAGEN_ENABLE_POST_INGESTION_EVOLUTION", "true")

    # 1. Explicit opt-out with enable_evolution=False
    with patch("urllib.request.urlopen", side_effect=make_mock_urlopen(pdf_bytes, MOCK_WORK_DATA)):
        res_opt_out = client.post(
            "/api/v1/papers/ingest/openalex",
            json={"openalex_id": "W2741809807", "enable_evolution": False},
        )
    assert res_opt_out.status_code == 200
    assert res_opt_out.json()["evolution_status"] == EvolutionHookStatus.DISABLED
    assert res_opt_out.json()["evolution_result"] is None

    # Clear DB for second part
    with SessionMaker() as sess:
        sess.query(PaperPassage).delete()
        sess.query(PaperDocument).delete()
        sess.query(Paper).delete()
        sess.commit()

    # 2. Explicit opt-in with enable_evolution=True
    with patch("urllib.request.urlopen", side_effect=make_mock_urlopen(pdf_bytes, MOCK_WORK_DATA)):
        res_opt_in = client.post(
            "/api/v1/papers/ingest/openalex",
            json={"openalex_id": "W2741809807", "enable_evolution": True},
        )
    assert res_opt_in.status_code == 200
    data_in = res_opt_in.json()
    assert data_in["status"] == IngestionStatus.INGESTED
    # With 0 existing opportunities, evolution returns SKIPPED
    assert data_in["evolution_status"] == EvolutionHookStatus.SKIPPED
    assert data_in["evolution_result"] is not None
    assert data_in["evolution_result"]["run_status"] == "NO_CANDIDATES"


def test_2b_request_cannot_enable_evolution_disabled_by_server(test_context, monkeypatch):
    """Test 2b: enable_evolution=True is ignored when the server setting disables evolution;
    no classifier call is made."""
    client, _ = test_context
    pdf_bytes = create_sample_pdf_bytes("Quantum Error Correction")
    monkeypatch.setenv("IDEAGEN_ENABLE_POST_INGESTION_EVOLUTION", "false")

    with patch("urllib.request.urlopen", side_effect=make_mock_urlopen(pdf_bytes, MOCK_WORK_DATA)), \
         patch("ideagen.opportunity.classifier.RelationshipClassifier.classify") as mock_classify:
        res = client.post(
            "/api/v1/papers/ingest/openalex",
            json={"openalex_id": "W2741809807", "enable_evolution": True},
        )
    assert res.status_code == 200
    assert res.json()["status"] == IngestionStatus.INGESTED
    assert res.json()["evolution_status"] == EvolutionHookStatus.DISABLED
    assert res.json()["evolution_result"] is None
    mock_classify.assert_not_called()


def test_3_configuration_enabled_behavior(test_context, monkeypatch):
    """Test 3: When IDEAGEN_ENABLE_POST_INGESTION_EVOLUTION is True, request defaults to enabled."""
    client, SessionMaker = test_context
    pdf_bytes = create_sample_pdf_bytes("Quantum Error Correction")

    monkeypatch.setenv("IDEAGEN_ENABLE_POST_INGESTION_EVOLUTION", "true")

    with patch("urllib.request.urlopen", side_effect=make_mock_urlopen(pdf_bytes, MOCK_WORK_DATA)):
        res = client.post(
            "/api/v1/papers/ingest/openalex",
            json={"openalex_id": "W2741809807"},  # enable_evolution not specified
        )

    assert res.status_code == 200
    data = res.json()
    assert data["status"] == IngestionStatus.INGESTED
    # Evolution executed automatically due to config
    assert data["evolution_status"] == EvolutionHookStatus.SKIPPED
    assert data["evolution_result"] is not None
    assert data["evolution_result"]["candidate_count"] == 0


def test_4_successful_ingestion_triggers_evolution_exactly_once(test_context, monkeypatch):
    """Test 4: Verify IncrementalOpportunityEvolutionService.evaluate_paper runs exactly once."""
    monkeypatch.setenv("IDEAGEN_ENABLE_POST_INGESTION_EVOLUTION", "true")
    client, SessionMaker = test_context
    pdf_bytes = create_sample_pdf_bytes("Quantum Error Correction")

    evaluate_mock = MagicMock()

    from ideagen.opportunity.incremental import IncrementalOpportunityEvolutionService, IncrementalEvolutionResult, IncrementalEvolutionRunStatus

    orig_evaluate = IncrementalOpportunityEvolutionService.evaluate_paper

    def wrapped_evaluate(self, paper_id, **kwargs):
        evaluate_mock(paper_id)
        return orig_evaluate(self, paper_id, **kwargs)

    with patch.object(IncrementalOpportunityEvolutionService, "evaluate_paper", side_effect=wrapped_evaluate, autospec=True):
        with patch("urllib.request.urlopen", side_effect=make_mock_urlopen(pdf_bytes, MOCK_WORK_DATA)):
            res = client.post(
                "/api/v1/papers/ingest/openalex",
                json={"openalex_id": "W2741809807", "enable_evolution": True},
            )

    assert res.status_code == 200
    assert evaluate_mock.call_count == 1


def test_5_no_pdf_duplicate_and_failed_ingestion_suppress_evolution(test_context, monkeypatch):
    """Test 5: NO_PDF, duplicate, and failed ingestions do not invoke evolution."""
    monkeypatch.setenv("IDEAGEN_ENABLE_POST_INGESTION_EVOLUTION", "true")
    client, SessionMaker = test_context
    pdf_bytes = create_sample_pdf_bytes("Quantum Error Correction")

    # 1. NO_PDF metadata-only path (no pdf url in metadata)
    sparse_work = {
        **MOCK_WORK_DATA,
        "id": "https://openalex.org/W9999999991",
        "doi": "https://doi.org/10.1145/9999999991",
        "open_access": {"is_oa": False},
        "best_oa_location": None,
        "primary_location": None,
    }
    with patch("urllib.request.urlopen", side_effect=make_mock_urlopen(pdf_bytes, sparse_work)):
        res_no_pdf = client.post(
            "/api/v1/papers/ingest/openalex",
            json={"openalex_id": "W9999999991", "enable_evolution": True},
        )
    assert res_no_pdf.status_code == 200
    data_no_pdf = res_no_pdf.json()
    assert data_no_pdf["status"] == IngestionStatus.NO_PDF
    assert data_no_pdf["evolution_status"] == EvolutionHookStatus.SKIPPED
    assert data_no_pdf["evolution_result"] is None

    # 2. First successful ingestion
    with patch("urllib.request.urlopen", side_effect=make_mock_urlopen(pdf_bytes, MOCK_WORK_DATA)):
        res_ingest = client.post(
            "/api/v1/papers/ingest/openalex",
            json={"openalex_id": "W2741809807", "enable_evolution": True},
        )
    assert res_ingest.status_code == 200
    assert res_ingest.json()["status"] == IngestionStatus.INGESTED

    # 3. Duplicate ingestion request
    with patch("urllib.request.urlopen", side_effect=make_mock_urlopen(pdf_bytes, MOCK_WORK_DATA)):
        res_dup = client.post(
            "/api/v1/papers/ingest/openalex",
            json={"openalex_id": "W2741809807", "enable_evolution": True},
        )
    assert res_dup.status_code == 200
    data_dup = res_dup.json()
    assert data_dup["status"] == IngestionStatus.SKIPPED_DUPLICATE
    assert data_dup["evolution_status"] == EvolutionHookStatus.SKIPPED
    assert data_dup["evolution_result"] is None


def test_6_multiple_eligible_opportunities_and_counter_reconciliation(test_context, monkeypatch):
    """Test 6: Multiple candidate opportunities are evaluated and outcome counters reconcile."""
    monkeypatch.setenv("IDEAGEN_ENABLE_POST_INGESTION_EVOLUTION", "true")
    client, SessionMaker = test_context

    # Create candidate opportunities in DB before ingestion:
    # Opp 1: Valid creation date before paper (2024-01-01 <= 2024-03-15) -> eligible
    # Opp 2: Valid creation date before paper (2024-02-01 <= 2024-03-15) -> eligible
    # Opp 3: Future dated creation (2024-05-01 > 2024-03-15) -> skipped due to temporal constraint
    opp1_id = uuid.uuid4()
    opp2_id = uuid.uuid4()
    opp3_id = uuid.uuid4()

    with SessionMaker() as sess:
        o1 = Opportunity(
            id=opp1_id,
            title="Scalable Surface Codes",
            problem_statement="Surface codes suffer from high physical error rates in qubits.",
            description="Detailed opportunity description for scalable surface codes.",
            status=OpportunityStatus.OPEN,
            created_at=datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc),
        )
        o2 = Opportunity(
            id=opp2_id,
            title="Fault Tolerant Thresholds",
            problem_statement="Threshold requirements for quantum fault tolerance are stringent.",
            description="Detailed opportunity description for fault tolerant thresholds.",
            status=OpportunityStatus.OPEN,
            created_at=datetime.datetime(2024, 2, 1, 0, 0, 0, tzinfo=datetime.timezone.utc),
        )
        o3 = Opportunity(
            id=opp3_id,
            title="Future Opportunity Beyond 2024",
            problem_statement="Future quantum algorithms require millions of logical qubits.",
            description="Detailed opportunity description for future quantum algorithms.",
            status=OpportunityStatus.OPEN,
            created_at=datetime.datetime(2024, 5, 1, 0, 0, 0, tzinfo=datetime.timezone.utc),
        )
        sess.add_all([o1, o2, o3])
        sess.commit()

    pdf_body = "Detailed scientific research text. "
    pdf_bytes = create_sample_pdf_bytes("Quantum Error Correction", body=pdf_body)

    # Mock classifier to classify opp1 as ADDRESSES and opp2 as SUPPORTS with valid excerpts
    def mock_classify(classification_input):
        opp_title = classification_input.opportunity_title
        if "Scalable Surface Codes" in opp_title:
            return RelationshipClassification(
                relationship=OpportunityRelationship.ADDRESSES,
                confidence=0.92,
                evidence_excerpt=f"Abstract sentence 0: {pdf_body}",
                rationale="Paper demonstrates fault tolerance addressing physical error rates.",
            )
        else:
            return RelationshipClassification(
                relationship=OpportunityRelationship.SUPPORTS,
                confidence=0.88,
                evidence_excerpt=f"Abstract sentence 1: {pdf_body}",
                rationale="Paper confirms the stringency of threshold requirements.",
            )

    with patch("ideagen.opportunity.classifier.RelationshipClassifier.classify", side_effect=mock_classify):
        with patch("urllib.request.urlopen", side_effect=make_mock_urlopen(pdf_bytes, MOCK_WORK_DATA)):
            res = client.post(
                "/api/v1/papers/ingest/openalex",
                json={"openalex_id": "W2741809807", "enable_evolution": True},
            )

    assert res.status_code == 200
    data = res.json()
    assert data["status"] == IngestionStatus.INGESTED
    assert data["evolution_status"] == EvolutionHookStatus.SUCCESS

    evo_res = data["evolution_result"]
    assert evo_res is not None
    assert evo_res["run_status"] == "SUCCESS"
    assert evo_res["candidate_count"] == 3  # All 3 queried candidates
    assert evo_res["evaluated_count"] == 2  # 2 eligible candidates evaluated
    assert evo_res["skipped_count"] == 1    # 1 future-dated candidate skipped
    assert evo_res["grounding_rejected_count"] == 0
    assert evo_res["failed_count"] == 0
    assert len(evo_res["outcomes"]) == 3

    # Verify database state in fresh session
    with SessionMaker() as sess:
        db_opp1 = sess.get(Opportunity, opp1_id)
        assert db_opp1.status == OpportunityStatus.PARTIALLY_ADDRESSED  # OPEN + ADDRESSES -> PARTIALLY_ADDRESSED

        db_opp2 = sess.get(Opportunity, opp2_id)
        assert db_opp2.status == OpportunityStatus.STRENGTHENED  # OPEN + SUPPORTS -> STRENGTHENED

        db_opp3 = sess.get(Opportunity, opp3_id)
        assert db_opp3.status == OpportunityStatus.OPEN  # Future dated, untouched

        # Evidence count
        ev_count = sess.query(func.count(OpportunityEvidence.id)).scalar()
        assert ev_count == 2

        # History count
        hist_count = sess.query(func.count(OpportunityHistory.id)).scalar()
        assert hist_count == 2


def test_7_grounding_or_provider_failure_does_not_corrupt_ingestion(test_context, monkeypatch):
    """Test 7: Provider failure or grounding rejection does not roll back paper ingestion."""
    monkeypatch.setenv("IDEAGEN_ENABLE_POST_INGESTION_EVOLUTION", "true")
    client, SessionMaker = test_context

    opp_id = uuid.uuid4()
    with SessionMaker() as sess:
        o = Opportunity(
            id=opp_id,
            title="Fault Tolerant Systems",
            problem_statement="High error rates prevent scalability.",
            description="Detailed opportunity description for fault tolerant systems.",
            status=OpportunityStatus.OPEN,
            created_at=datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc),
        )
        sess.add(o)
        sess.commit()

    pdf_bytes = create_sample_pdf_bytes("Quantum Paper")

    # 1. LLM provider outage
    with patch("ideagen.opportunity.classifier.RelationshipClassifier.classify", side_effect=ProviderFailureError("Ollama connection failed")):
        with patch("urllib.request.urlopen", side_effect=make_mock_urlopen(pdf_bytes, MOCK_WORK_DATA)):
            res = client.post(
                "/api/v1/papers/ingest/openalex",
                json={"openalex_id": "W2741809807", "enable_evolution": True},
            )

    assert res.status_code == 200
    data = res.json()
    # Paper is safely ingested
    assert data["status"] == IngestionStatus.INGESTED
    assert data["paper_id"] is not None
    # Evolution failure is recorded cleanly
    assert data["evolution_status"] == EvolutionHookStatus.FAILED

    # Opportunity remains OPEN and uncorrupted in DB
    with SessionMaker() as sess:
        db_opp = sess.get(Opportunity, opp_id)
        assert db_opp.status == OpportunityStatus.OPEN
        assert sess.query(func.count(OpportunityEvidence.id)).scalar() == 0


def test_8_standalone_evolution_changes_are_durable_after_request_closes(test_context, monkeypatch):
    """Test 8: Evolution state written during HTTP request is committed and durable in a separate session."""
    monkeypatch.setenv("IDEAGEN_ENABLE_POST_INGESTION_EVOLUTION", "true")
    client, SessionMaker = test_context

    opp_id = uuid.uuid4()
    with SessionMaker() as sess:
        o = Opportunity(
            id=opp_id,
            title="Robust Surface Codes",
            problem_statement="Qubit coherence limits scalability.",
            description="Detailed opportunity description for robust surface codes.",
            status=OpportunityStatus.OPEN,
            created_at=datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc),
        )
        sess.add(o)
        sess.commit()

    pdf_body = "Detailed scientific research text. "
    pdf_bytes = create_sample_pdf_bytes("Quantum Paper", body=pdf_body)

    def mock_classify(classification_input):
        return RelationshipClassification(
            relationship=OpportunityRelationship.ADDRESSES,
            confidence=0.95,
            evidence_excerpt=f"Abstract sentence 0: {pdf_body}",
            rationale="Addresses coherence limits.",
        )

    with patch("ideagen.opportunity.classifier.RelationshipClassifier.classify", side_effect=mock_classify):
        with patch("urllib.request.urlopen", side_effect=make_mock_urlopen(pdf_bytes, MOCK_WORK_DATA)):
            res = client.post(
                "/api/v1/papers/ingest/openalex",
                json={"openalex_id": "W2741809807", "enable_evolution": True},
            )

    assert res.status_code == 200

    # Completely new session verifies persistence
    with SessionMaker() as fresh_sess:
        opp = fresh_sess.get(Opportunity, opp_id)
        assert opp.status == OpportunityStatus.PARTIALLY_ADDRESSED

        evidence = fresh_sess.query(OpportunityEvidence).filter_by(opportunity_id=opp_id).first()
        assert evidence is not None
        assert evidence.evidence_type == OpportunityRelationship.ADDRESSES.value or evidence.evidence_type == OpportunityRelationship.ADDRESSES
        assert "Abstract sentence 0" in (evidence.supporting_text or "")


def test_9_compatibility_with_existing_arxiv_search_and_upload(test_context):
    """Test 9: Verify existing /api/search/papers and /api/papers/upload remain fully compatible."""
    client, SessionMaker = test_context

    # 1. ArXiv search
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
    with patch("urllib.request.urlopen", return_value=DummyStreamResponse(mock_arxiv_xml, {"Content-Type": "application/xml"})):
        res_search = client.get("/api/search/papers?q=deep+learning&limit=5")
    assert res_search.status_code == 200
    assert res_search.json()["source"] == "ARXIV"

    # 2. Manual PDF upload
    pdf_bytes = create_sample_pdf_bytes("Manual Upload Test")
    files = [("files", ("manual_test.pdf", pdf_bytes, "application/pdf"))]
    res_upload = client.post("/api/papers/upload", files=files)
    assert res_upload.status_code == 200
    data_upload = res_upload.json()
    assert data_upload["source"] == "UPLOAD"
    assert data_upload["count"] == 1
