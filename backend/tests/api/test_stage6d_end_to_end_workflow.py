"""Stage 6D: Complete End-to-End Workflow Validation Tests.

Validates the full IdeaGen backend pipeline using actual HTTP API routes:
1. OpenAlex Search (GET /api/v1/search/openalex)
2. Paper Selection & Ingestion (POST /api/v1/papers/ingest/openalex)
3. Passage & Document Persistence (GET /api/v1/papers/{id})
4. Paper Understanding & Statement Provenance (POST /api/papers/{id}/understand, GET /api/v1/papers/{id}/statements)
5. Gap Discovery & Evidence Grounding (POST /api/gaps/discover, GET /api/v1/gaps/{id})
6. Opportunity Synthesis & Lifecycle Bridging (POST /api/v1/gaps/{id}/opportunity, GET /api/v1/opportunities)
7. Chronologically Later Paper Ingestion with Post-Ingestion Evolution (POST /api/v1/papers/ingest/openalex with enable_evolution=true)
8. Lifecycle Evolution & State Transition Execution
9. Opportunity Evidence & History Retrieval (GET /api/v1/opportunities/{id}/evidence, /history)
10. Failure Path & Robustness Verification:
    - NO_PDF fallback
    - Duplicate ingestion idempotency
    - Invalid OpenAlex ID / DOI validation (HTTP 400)
    - Upstream timeout / provider errors (HTTP 502)
    - Temporally ineligible / undated papers (SKIPPED)
    - Grounding rejection (retains initial status without state corruption)
    - Evolution failure isolation (committed paper records preserved)
    - Read-after-write durability across fresh database sessions
"""

from __future__ import annotations

import datetime
import io
import json
import re
import urllib.request
import urllib.error
import uuid
from typing import Dict, Any, Optional, List
from unittest.mock import patch, MagicMock

import numpy as np
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
    ResearchKnowledge,
    ResearchStatement,
    ResearchGap,
    GapEvidence,
    ResearchOpportunityRecord,
    Opportunity,
    OpportunityEvidence,
    OpportunityHistory,
)
from ideagen.db import get_db
from ideagen.api.main import app
from ideagen.ingestion.openalex_ingestion import IngestionStatus, EvolutionHookStatus
from ideagen.opportunity.models import OpportunityStatus, OpportunityRelationship
from ideagen.opportunity.classifier import RelationshipClassification
from ideagen.understanding.models import (
    StructuredPaperExtractionOutput,
    RawStatementExtraction,
    StatementType,
)
from ideagen.discovery.models import (
    GapStatus,
    GapValidationResult,
    ValidationVerdict,
)
from ideagen.llm.exceptions import ProviderFailureError


# ── In-Memory Deterministic Fixtures ────────────────────────────────────────

class DeterministicEmbeddingProvider:
    """Deterministic in-memory embedding provider for offline FAISS indexing."""
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
            if any(term in text.lower() for term in ["quantum", "qubit", "surface", "code", "syndrome", "coherence", "error", "drift", "parity", "limitation"]):
                base[:16] += 6.0
            norm = np.linalg.norm(base)
            vecs.append(base / (norm if norm > 0 else 1.0))
        return np.array(vecs, dtype=np.float32)


class DeterministicLLMProvider:
    """Deterministic mock LLM provider responding based on prompt intent."""
    def __init__(self):
        self.call_history: List[str] = []

    @property
    def model_name(self) -> str:
        return "mock-deterministic-llm"

    def generate_text(self, prompt: str, **kwargs) -> str:
        self.call_history.append(prompt)
        return "Mock deterministic analysis response."

    def generate_structured(self, prompt: str, response_schema: Any, **kwargs) -> Any:
        self.call_history.append(prompt)

        # 1. Paper Understanding Extraction Schema
        if response_schema == StructuredPaperExtractionOutput:
            return StructuredPaperExtractionOutput(
                research_problem="Scalability limitations in superconducting quantum processors.",
                research_questions=["Can topological surface codes achieve fault-tolerant thresholds below 1% error rate?"],
                methodology="Rotated surface code parity check architecture.",
                datasets=["Simulated 127-qubit heavy-hex quantum circuit traces"],
                findings=["Demonstrated threshold suppression with parity fidelity of 99.4%."],
                limitations=["Coherence time drift in high-density multi-qubit topologies imposes fundamental error limitations."],
                future_work=["Extend real-time syndrome extraction to dynamic decoder pipelines."],
                statements=[
                    RawStatementExtraction(
                        statement_text="Superconducting quantum processors suffer from high fault-tolerant error rate bottlenecks.",
                        statement_type=StatementType.PROBLEM,
                        passage_index=0,
                        confidence=0.92,
                    ),
                    RawStatementExtraction(
                        statement_text="Coherence time drift in high-density multi-qubit topologies imposes fundamental error limitations.",
                        statement_type=StatementType.LIMITATION,
                        passage_index=0,
                        confidence=0.95,
                    ),
                    RawStatementExtraction(
                        statement_text="Rotated surface code parity check architecture.",
                        statement_type=StatementType.METHOD,
                        passage_index=1,
                        confidence=0.90,
                    ),
                    RawStatementExtraction(
                        statement_text="Extend real-time syndrome extraction to dynamic decoder pipelines.",
                        statement_type=StatementType.FUTURE_WORK,
                        passage_index=0,
                        confidence=0.88,
                    ),
                ],
            )

        # 2. Gap Validation Result Schema
        if response_schema == GapValidationResult:
            cited = re.findall(r"\[ID:\s*([0-9a-fA-F-]+)", prompt)
            return GapValidationResult(
                verdict=ValidationVerdict.VALID,
                reasoning="The candidate gap is rigorously grounded in the reported coherence time limitations.",
                supporting_evidence_statement_ids=cited if cited else [],
                confidence=0.92,
            )

        # 3. Opportunity Generation Schema
        if getattr(response_schema, "__name__", "") == "LLMCandidateOpportunitySchema":
            from ideagen.generation.generator import LLMCandidateOpportunitySchema
            return LLMCandidateOpportunitySchema(
                proposed_research_direction="Dynamic syndrome decoding for high-density superconducting qubit arrays.",
                reasoning="Coherence time drift compromises static syndrome decoding schedules.",
                technical_approach="Real-time reinforcement-learning decoder with adaptive syndrome latency.",
                expected_contribution="Sub-threshold fault tolerance under high-drift coherence conditions.",
                required_data="Simulated quantum error syndrome traces",
                required_compute="4x H100 GPU cluster",
                implementation_complexity="Moderate",
                experimental_setup="Compare against static lookup-table decoders",
                feasibility_score=0.86,
                confidence=0.89,
            )

        # 4. Prior-Art Critique Schema
        if getattr(response_schema, "__name__", "") == "LLMPriorArtValidationSchema":
            from ideagen.generation.prior_art import LLMPriorArtValidationSchema
            return LLMPriorArtValidationSchema(
                assessment="LOW_OVERLAP",
                reasoning="Proposed dynamic RL decoder differs from precomputed static surface code parity tables.",
                differentiation="Real-time adaptive syndrome decoding under high coherence drift conditions.",
                overlapping_statement_indices=[],
                confidence=0.91,
            )

        # 5. Generic fallback for any other schema
        data = {
            "proposed_research_direction": "Dynamic syndrome decoding for high-density superconducting qubit arrays.",
            "reasoning": "Coherence time drift compromises static syndrome decoding schedules.",
            "technical_approach": "Real-time reinforcement-learning decoder with adaptive syndrome latency.",
            "expected_contribution": "Sub-threshold fault tolerance under high-drift coherence conditions.",
            "required_data": "Simulated quantum error syndrome traces",
            "required_compute": "4x H100 GPU cluster",
            "implementation_complexity": "Moderate",
            "experimental_setup": "Compare against static lookup-table decoders",
            "feasibility_score": 0.86,
            "confidence": 0.89,
            "assessment": "High novelty in adaptive quantum error correction architectures",
            "differentiation": "Existing methods rely on static precomputed lookup tables.",
            "overlapping_statement_indices": [],
        }
        if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
            valid_fields = {k: v for k, v in data.items() if k in response_schema.model_fields}
            return response_schema(**valid_fields)
        return data


def create_sample_pdf_bytes(title: str = "Test Title", body: str = "Detailed scientific research text. ") -> bytes:
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


MOCK_INITIAL_WORK: Dict[str, Any] = {
    "id": "https://openalex.org/W1000000001",
    "doi": "https://doi.org/10.1145/1000000001",
    "title": "Foundational Limits of Quantum Surface Codes",
    "publication_year": 2023,
    "publication_date": "2023-06-15",
    "abstract_inverted_index": {
        "Foundational": [0],
        "limits": [1],
        "of": [2],
        "quantum": [3],
        "surface": [4],
        "codes.": [5],
    },
    "authorships": [{"author": {"display_name": "Dr. Alice Quantum"}}],
    "open_access": {"is_oa": True, "oa_url": "https://example.org/initial.pdf"},
    "best_oa_location": {"is_oa": True, "pdf_url": "https://example.org/initial.pdf"},
    "cited_by_count": 105,
}

MOCK_LATER_WORK: Dict[str, Any] = {
    "id": "https://openalex.org/W2000000002",
    "doi": "https://doi.org/10.1145/2000000002",
    "title": "Adaptive Real-Time Syndrome Extraction in Quantum Arrays",
    "publication_year": 2024,
    "publication_date": "2024-03-20",  # Chronologically later (2024-03-20 > 2023-06-15)
    "abstract_inverted_index": {
        "Adaptive": [0],
        "real-time": [1],
        "syndrome": [2],
        "extraction.": [3],
    },
    "authorships": [{"author": {"display_name": "Dr. Bob Qubit"}}],
    "open_access": {"is_oa": True, "oa_url": "https://example.org/later.pdf"},
    "best_oa_location": {"is_oa": True, "pdf_url": "https://example.org/later.pdf"},
    "cited_by_count": 12,
}


@pytest.fixture(name="e2e_env")
def e2e_env_fixture(monkeypatch):
    # These workflows model a deployment with post-ingestion evolution enabled server-side;
    # requests still opt out with enable_evolution=false.
    monkeypatch.setenv("IDEAGEN_ENABLE_POST_INGESTION_EVOLUTION", "true")
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

    embedder = DeterministicEmbeddingProvider(dimension=64)
    llm = DeterministicLLMProvider()

    with patch("ideagen.llm.get_default_llm_provider", return_value=llm), \
         patch("ideagen.understanding.pipeline.get_default_llm_provider", return_value=llm), \
         patch("ideagen.api.main.get_default_llm_provider", return_value=llm), \
         patch("ideagen.discovery.pipeline.get_default_llm_provider", return_value=llm), \
         patch("ideagen.discovery.embeddings.get_default_embedding_provider", return_value=embedder), \
         patch("ideagen.discovery.pipeline.get_default_embedding_provider", return_value=embedder), \
         patch("ideagen.api.main.get_default_embedding_provider", return_value=embedder):
        yield client, TestingSessionLocal, llm, embedder

    app.dependency_overrides.clear()


def make_mock_network_dispatcher(work_lookup: Dict[str, Dict[str, Any]], pdf_map: Dict[str, bytes]):
    def mock_urlopen(req, timeout=30.0):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "api.openalex.org" in url:
            # Check for direct ID / DOI lookup in url
            matched = []
            for k, w in work_lookup.items():
                if k in url:
                    matched.append(w)
            if not matched:
                # Return all works for generic search
                matched = list(work_lookup.values())
            payload = json.dumps({"results": matched, "meta": {"count": len(matched)}}).encode("utf-8")
            return DummyStreamResponse(payload, {"Content-Type": "application/json"})

        for pdf_url, pbytes in pdf_map.items():
            if pdf_url in url:
                return DummyStreamResponse(pbytes, {"Content-Type": "application/pdf"})

        raise ValueError(f"Unmocked network request to: {url}")
    return mock_urlopen


# ── Complete Workflow Validation ───────────────────────────────────────────

def test_complete_e2e_workflow(e2e_env):
    """End-to-End Workflow:

    OpenAlex search -> ingestion -> passages -> understanding -> statements ->
    gap discovery -> opportunity synthesis -> later paper ingestion with evolution ->
    read history and evidence verification.
    """
    client, SessionMaker, mock_llm, mock_embedder = e2e_env

    pdf_initial = create_sample_pdf_bytes(
        title="Foundational Limits of Quantum Surface Codes",
        body="Coherence time drift in high-density multi-qubit topologies imposes fundamental error limitations.",
    )
    pdf_later = create_sample_pdf_bytes(
        title="Adaptive Real-Time Syndrome Extraction",
        body="We directly address coherence time drift using adaptive syndrome decoding in surface codes.",
    )

    work_map = {
        "W1000000001": MOCK_INITIAL_WORK,
        "W2000000002": MOCK_LATER_WORK,
    }
    pdf_map = {
        "initial.pdf": pdf_initial,
        "later.pdf": pdf_later,
    }

    network_mock = make_mock_network_dispatcher(work_map, pdf_map)

    with patch("urllib.request.urlopen", side_effect=network_mock):
        # ── Step 1: OpenAlex Search ─────────────────────────────────────────
        search_res = client.get("/api/v1/search/openalex?query=quantum+surface+codes&limit=5")
        assert search_res.status_code == 200
        search_data = search_res.json()
        assert search_data["source"] == "OPENALEX"
        assert search_data["count"] >= 1
        found_ids = [p["id"] for p in search_data["papers"]]
        assert "W1000000001" in found_ids

        # ── Step 2: Ingest Foundational Paper ───────────────────────────────
        ingest_res = client.post(
            "/api/v1/papers/ingest/openalex",
            json={"openalex_id": "W1000000001", "enable_evolution": False},
        )
        assert ingest_res.status_code == 200
        ingest_data = ingest_res.json()
        assert ingest_data["status"] == IngestionStatus.INGESTED
        assert ingest_data["has_pdf"] is True
        initial_paper_id = uuid.UUID(ingest_data["paper_id"])
        assert ingest_data["passage_count"] > 0
        assert ingest_data["evolution_status"] == EvolutionHookStatus.DISABLED

        # ── Step 3: Verify Persistence via GET /api/v1/papers/{id} ──────────
        paper_res = client.get(f"/api/v1/papers/{initial_paper_id}")
        assert paper_res.status_code == 200
        paper_data = paper_res.json()
        assert paper_data["title"] == "Foundational Limits of Quantum Surface Codes"
        assert paper_data["publication_date"] == "2023-06-15"

        # ── Step 4: Paper Understanding (Statement Extraction) ──────────────
        und_res = client.post(f"/api/papers/{initial_paper_id}/understand")
        assert und_res.status_code == 200
        und_data = und_res.json()
        assert und_data["statement_count"] > 0

        # Verify statements provenance via GET /api/v1/papers/{id}/statements
        stmts_res = client.get(f"/api/v1/papers/{initial_paper_id}/statements")
        assert stmts_res.status_code == 200
        stmts = stmts_res.json()
        assert len(stmts) >= 2
        limitation_stmts = [s for s in stmts if s["statement_type"] == "LIMITATION"]
        assert len(limitation_stmts) > 0
        assert "coherence" in limitation_stmts[0]["statement_text"].lower()

        # ── Step 5: Research Gap Discovery ──────────────────────────────────
        gap_disc_res = client.post("/api/gaps/discover")
        assert gap_disc_res.status_code == 200
        gaps = gap_disc_res.json()
        assert len(gaps) > 0

        # Retrieve specific gap via GET /api/v1/gaps/{gap_id}
        discovered_gap_id = gaps[0]["id"]
        gap_detail_res = client.get(f"/api/v1/gaps/{discovered_gap_id}")
        assert gap_detail_res.status_code == 200
        gap_detail = gap_detail_res.json()
        assert gap_detail["status"] == GapStatus.VALIDATED.value
        assert len(gap_detail["supporting_statement_ids"]) > 0

        # ── Step 6: Opportunity Generation & Lifecycle Bridging ─────────────
        opp_gen_res = client.post(f"/api/v1/gaps/{discovered_gap_id}/opportunity")
        assert opp_gen_res.status_code == 200
        opp_report = opp_gen_res.json()
        assert "opportunity_id" in opp_report
        assert opp_report["proposed_research_direction"] is not None
        lifecycle_opp_id = opp_report.get("lifecycle_opportunity_id")
        assert lifecycle_opp_id is not None

        # Verify opportunity bridged in lifecycle catalog
        opps_list_res = client.get("/api/v1/opportunities")
        assert opps_list_res.status_code == 200
        opps_list = opps_list_res.json()
        assert len(opps_list) > 0
        target_opp = opps_list[0]
        assert target_opp["lifecycle_opportunity_id"] == lifecycle_opp_id
        assert target_opp["lifecycle_status"] in ("OPEN", "VALIDATED")

        # Set opportunity creation date to baseline period (2023-07-01) so later paper (2024-03-20) is chronologically eligible
        with SessionMaker() as sess:
            opp_db = sess.get(Opportunity, uuid.UUID(lifecycle_opp_id))
            opp_db.created_at = datetime.datetime(2023, 7, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
            sess.commit()

        # ── Step 7 & 8: Later Paper Ingestion with Incremental Evolution ────
        # Mock relationship classifier to evaluate ADDRESSES with valid grounded excerpt
        grounded_excerpt = "Abstract sentence 0: We directly address coherence time drift using adaptive syndrome decoding in surface codes."

        def mock_classify(classification_input):
            return RelationshipClassification(
                relationship=OpportunityRelationship.ADDRESSES,
                confidence=0.94,
                evidence_excerpt=grounded_excerpt,
                rationale="Paper introduces adaptive syndrome extraction resolving coherence drift limitation.",
            )

        with patch("ideagen.opportunity.classifier.RelationshipClassifier.classify", side_effect=mock_classify):
            later_ingest_res = client.post(
                "/api/v1/papers/ingest/openalex",
                json={"openalex_id": "W2000000002", "enable_evolution": True},
            )

        assert later_ingest_res.status_code == 200
        later_data = later_ingest_res.json()
        assert later_data["status"] == IngestionStatus.INGESTED
        assert later_data["evolution_status"] == EvolutionHookStatus.SUCCESS
        assert later_data["evolution_result"] is not None
        assert later_data["evolution_result"]["evaluated_count"] >= 1
        assert later_data["evolution_result"]["failed_count"] == 0

        # ── Step 9: Verify Opportunity Evidence and History via APIs ────────
        # 1. Evidence endpoint
        ev_res = client.get(f"/api/v1/opportunities/{lifecycle_opp_id}/evidence")
        assert ev_res.status_code == 200
        evidence_list = ev_res.json()
        assert len(evidence_list) >= 1
        ev_item = evidence_list[0]
        assert ev_item["relationship"] == OpportunityRelationship.ADDRESSES.value
        assert "coherence time drift" in ev_item["supporting_text"].lower()

        # 2. History endpoint
        hist_res = client.get(f"/api/v1/opportunities/{lifecycle_opp_id}/history")
        assert hist_res.status_code == 200
        history_list = hist_res.json()
        assert len(history_list) >= 1
        hist_item = history_list[0]
        assert hist_item["trigger_type"] == OpportunityRelationship.ADDRESSES.value
        assert hist_item["new_status"] == OpportunityStatus.PARTIALLY_ADDRESSED.value

        # ── Step 10: Verify Lifecycle State Progression ──────────────────────
        opps_updated_res = client.get("/api/v1/opportunities")
        assert opps_updated_res.status_code == 200
        updated_opp = next(o for o in opps_updated_res.json() if o["lifecycle_opportunity_id"] == lifecycle_opp_id)
        assert updated_opp["lifecycle_status"] == OpportunityStatus.PARTIALLY_ADDRESSED.value


# ── Failure Paths & Edge Cases ─────────────────────────────────────────────

def test_failure_path_no_accessible_pdf(e2e_env):
    """Test failure path: Paper without accessible PDF falls back to NO_PDF and suppresses evolution."""
    client, SessionMaker, _, _ = e2e_env
    pdf_bytes = create_sample_pdf_bytes("No PDF Work")

    work_no_pdf = {
        **MOCK_INITIAL_WORK,
        "id": "https://openalex.org/W9000000001",
        "doi": "https://doi.org/10.1145/9000000001",
        "open_access": {"is_oa": False},
        "best_oa_location": None,
        "primary_location": None,
    }
    dispatcher = make_mock_network_dispatcher({"W9000000001": work_no_pdf}, {})

    with patch("urllib.request.urlopen", side_effect=dispatcher):
        res = client.post(
            "/api/v1/papers/ingest/openalex",
            json={"openalex_id": "W9000000001", "enable_evolution": True},
        )

    assert res.status_code == 200
    data = res.json()
    assert data["status"] == IngestionStatus.NO_PDF
    assert data["has_pdf"] is False
    assert data["evolution_status"] == EvolutionHookStatus.SKIPPED
    assert data["evolution_result"] is None


def test_failure_path_duplicate_ingestion(e2e_env):
    """Test failure path: Duplicate paper returns SKIPPED_DUPLICATE idempotently without re-triggering evolution."""
    client, SessionMaker, _, _ = e2e_env
    pdf_bytes = create_sample_pdf_bytes("Duplicate Test")
    dispatcher = make_mock_network_dispatcher({"W1000000001": MOCK_INITIAL_WORK}, {"initial.pdf": pdf_bytes})

    with patch("urllib.request.urlopen", side_effect=dispatcher):
        # First ingestion
        res1 = client.post("/api/v1/papers/ingest/openalex", json={"openalex_id": "W1000000001", "enable_evolution": False})
        assert res1.status_code == 200
        assert res1.json()["status"] == IngestionStatus.INGESTED

        # Duplicate ingestion
        res2 = client.post("/api/v1/papers/ingest/openalex", json={"openalex_id": "W1000000001", "enable_evolution": True})
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["status"] == IngestionStatus.SKIPPED_DUPLICATE
        assert data2["evolution_status"] == EvolutionHookStatus.SKIPPED


def test_failure_path_invalid_identifiers(e2e_env):
    """Test validation: Malformed OpenAlex ID or DOI returns HTTP 400."""
    client, _, _, _ = e2e_env

    # 1. Invalid OpenAlex ID
    res1 = client.post("/api/v1/papers/ingest/openalex", json={"openalex_id": "invalid_id_not_w"})
    assert res1.status_code == 400
    assert "Invalid OpenAlex ID format" in res1.json()["detail"]

    # 2. Invalid DOI
    res2 = client.post("/api/v1/papers/ingest/openalex", json={"doi": "not_a_doi"})
    assert res2.status_code == 400
    assert "Invalid DOI format" in res2.json()["detail"]

    # 3. Missing both identifiers
    res3 = client.post("/api/v1/papers/ingest/openalex", json={})
    assert res3.status_code == 400
    assert "must be provided" in res3.json()["detail"]


def test_failure_path_provider_timeout_and_error(e2e_env):
    """Test upstream failure: Network error/timeout maps cleanly to HTTP 502."""
    client, _, _, _ = e2e_env

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection timed out")):
        res = client.post("/api/v1/papers/ingest/openalex", json={"openalex_id": "W1000000001"})

    assert res.status_code == 502
    assert "Upstream OpenAlex service error" in res.json()["detail"]


def test_failure_path_undated_or_temporally_ineligible_papers(e2e_env):
    """Test temporal integrity:
    1. OpenAlex record with missing publication date is rejected cleanly at ingestion boundary.
    2. Temporally ineligible papers (published before an opportunity was created)
       are skipped during incremental evolution, preserving temporal causality.
    """
    client, SessionMaker, _, _ = e2e_env
    pdf_bytes = create_sample_pdf_bytes("Undated Work")

    # 1. OpenAlex record with missing publication date is rejected cleanly
    undated_work = {
        **MOCK_INITIAL_WORK,
        "id": "https://openalex.org/W9000000002",
        "doi": "https://doi.org/10.1145/9000000002",
        "publication_year": None,
        "publication_date": None,
    }
    dispatcher_undated = make_mock_network_dispatcher({"W9000000002": undated_work}, {"initial.pdf": pdf_bytes})

    with patch("urllib.request.urlopen", side_effect=dispatcher_undated):
        res_undated = client.post(
            "/api/v1/papers/ingest/openalex",
            json={"openalex_id": "W9000000002", "enable_evolution": True},
        )
    assert res_undated.status_code == 500
    assert "Missing publication date" in res_undated.json()["detail"]

    # 2. Temporally ineligible paper: Opportunity created in 2024; paper published earlier in 2023.
    # Ingestion succeeds, but incremental evolution skips candidate opportunity due to causality constraint.
    opp_id = uuid.uuid4()
    with SessionMaker() as sess:
        o = Opportunity(
            id=opp_id,
            title="Future Opportunity",
            problem_statement="Quantum error syndrome scaling.",
            description="Opportunity created in 2024.",
            status=OpportunityStatus.OPEN,
            created_at=datetime.datetime(2024, 6, 1, 0, 0, 0, tzinfo=datetime.timezone.utc),
        )
        sess.add(o)
        sess.commit()

    ineligible_work = {
        **MOCK_INITIAL_WORK,
        "id": "https://openalex.org/W9000000021",
        "doi": "https://doi.org/10.1145/9000000021",
        "publication_year": 2023,
        "publication_date": "2023-01-15",
    }
    dispatcher_ineligible = make_mock_network_dispatcher({"W9000000021": ineligible_work}, {"initial.pdf": pdf_bytes})

    with patch("urllib.request.urlopen", side_effect=dispatcher_ineligible):
        res = client.post(
            "/api/v1/papers/ingest/openalex",
            json={"openalex_id": "W9000000021", "enable_evolution": True},
        )

    assert res.status_code == 200
    data = res.json()
    assert data["status"] == IngestionStatus.INGESTED
    # Evolution is skipped because the opportunity was created AFTER the paper was published
    assert data["evolution_status"] == EvolutionHookStatus.SKIPPED
    evo_res = data["evolution_result"]
    assert evo_res["candidate_count"] == 1
    assert evo_res["evaluated_count"] == 0
    assert evo_res["skipped_count"] == 1
    assert "after paper publication date" in evo_res["outcomes"][0]["skip_reason"]

    # Verify opportunity state and evidence are completely untouched
    with SessionMaker() as sess:
        opp = sess.get(Opportunity, opp_id)
        assert opp.status == OpportunityStatus.OPEN
        ev_count = sess.query(func.count(OpportunityEvidence.id)).filter(OpportunityEvidence.opportunity_id == opp_id).scalar()
        assert ev_count == 0


def test_failure_path_grounding_rejection_leaves_state_intact(e2e_env):
    """Test evidence integrity: Fabricated / ungrounded excerpt is rejected and leaves opportunity state uncorrupted."""
    client, SessionMaker, _, _ = e2e_env

    # Setup opportunity
    opp_id = uuid.uuid4()
    with SessionMaker() as sess:
        o = Opportunity(
            id=opp_id,
            title="Qubit Stability",
            problem_statement="Decoherence limits scaling.",
            description="Detailed opportunity description.",
            status=OpportunityStatus.OPEN,
            created_at=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc),
        )
        sess.add(o)
        sess.commit()

    pdf_bytes = create_sample_pdf_bytes("Stability Work")
    work_dict = {
        **MOCK_INITIAL_WORK,
        "id": "https://openalex.org/W9000000003",
        "doi": "https://doi.org/10.1145/9000000003",
        "publication_date": "2023-08-01",
    }
    dispatcher = make_mock_network_dispatcher({"W9000000003": work_dict}, {"initial.pdf": pdf_bytes})

    # Classifier hallucinates excerpt not present in paper
    def mock_classify_fabricated(classification_input):
        return RelationshipClassification(
            relationship=OpportunityRelationship.ADDRESSES,
            confidence=0.95,
            evidence_excerpt="This exact sentence is completely fabricated and nowhere in the paper text.",
            rationale="Paper supposedly addresses decoherence.",
        )

    with patch("ideagen.opportunity.classifier.RelationshipClassifier.classify", side_effect=mock_classify_fabricated):
        with patch("urllib.request.urlopen", side_effect=dispatcher):
            res = client.post(
                "/api/v1/papers/ingest/openalex",
                json={"openalex_id": "W9000000003", "enable_evolution": True},
            )

    assert res.status_code == 200
    data = res.json()
    assert data["status"] == IngestionStatus.INGESTED
    evo_res = data["evolution_result"]
    assert evo_res["grounding_rejected_count"] == 1
    assert evo_res["evaluated_count"] == 0

    # Verify DB state: Opportunity remains OPEN with 0 evidence rows
    with SessionMaker() as sess:
        opp = sess.get(Opportunity, opp_id)
        assert opp.status == OpportunityStatus.OPEN
        ev_count = sess.query(func.count(OpportunityEvidence.id)).filter(OpportunityEvidence.opportunity_id == opp_id).scalar()
        assert ev_count == 0


def test_failure_path_evolution_failure_leaves_ingested_paper_intact(e2e_env):
    """Test fault tolerance: Classifier outage during evolution does not roll back or delete ingested paper."""
    client, SessionMaker, _, _ = e2e_env

    # Setup opportunity
    opp_id = uuid.uuid4()
    with SessionMaker() as sess:
        o = Opportunity(
            id=opp_id,
            title="Qubit Stability",
            problem_statement="Decoherence limits scaling.",
            description="Detailed opportunity description.",
            status=OpportunityStatus.OPEN,
            created_at=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc),
        )
        sess.add(o)
        sess.commit()

    pdf_bytes = create_sample_pdf_bytes("Crash Work")
    work_dict = {
        **MOCK_INITIAL_WORK,
        "id": "https://openalex.org/W9000000004",
        "doi": "https://doi.org/10.1145/9000000004",
        "publication_date": "2023-08-01",
    }
    dispatcher = make_mock_network_dispatcher({"W9000000004": work_dict}, {"initial.pdf": pdf_bytes})

    # LLM raises ProviderFailureError
    with patch("ideagen.opportunity.classifier.RelationshipClassifier.classify", side_effect=ProviderFailureError("Ollama service down")):
        with patch("urllib.request.urlopen", side_effect=dispatcher):
            res = client.post(
                "/api/v1/papers/ingest/openalex",
                json={"openalex_id": "W9000000004", "enable_evolution": True},
            )

    assert res.status_code == 200
    data = res.json()
    # Paper is ingested safely
    assert data["status"] == IngestionStatus.INGESTED
    paper_uuid = uuid.UUID(data["paper_id"])
    assert data["evolution_status"] == EvolutionHookStatus.FAILED

    # Check paper and document persist in fresh session
    with SessionMaker() as sess:
        db_paper = sess.get(Paper, paper_uuid)
        assert db_paper is not None
        assert db_paper.source_id == "W9000000004"
        doc_count = sess.query(func.count(PaperDocument.id)).filter(PaperDocument.paper_id == paper_uuid).scalar()
        assert doc_count == 1
        psg_count = sess.query(func.count(PaperPassage.id)).filter(PaperPassage.paper_id == paper_uuid).scalar()
        assert psg_count > 0


def test_read_after_write_durability_across_fresh_sessions(e2e_env):
    """Test durability: Records created through HTTP APIs remain durable when inspected through a brand new session."""
    client, SessionMaker, _, _ = e2e_env

    opp_id = uuid.uuid4()
    with SessionMaker() as sess:
        o = Opportunity(
            id=opp_id,
            title="Durable Scaling Opportunity",
            problem_statement="Decoherence prevents large-scale circuits.",
            description="Detailed opportunity description.",
            status=OpportunityStatus.OPEN,
            created_at=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc),
        )
        sess.add(o)
        sess.commit()

    pdf_body = "Detailed scientific research text. "
    pdf_bytes = create_sample_pdf_bytes("Durable Ingestion", body=pdf_body)
    work_dict = {
        **MOCK_INITIAL_WORK,
        "id": "https://openalex.org/W9000000005",
        "doi": "https://doi.org/10.1145/9000000005",
        "publication_date": "2023-09-01",
    }
    dispatcher = make_mock_network_dispatcher({"W9000000005": work_dict}, {"initial.pdf": pdf_bytes})

    def mock_classify(classification_input):
        return RelationshipClassification(
            relationship=OpportunityRelationship.ADDRESSES,
            confidence=0.96,
            evidence_excerpt=f"Abstract sentence 0: {pdf_body}",
            rationale="Addresses decoherence limits directly.",
        )

    with patch("ideagen.opportunity.classifier.RelationshipClassifier.classify", side_effect=mock_classify):
        with patch("urllib.request.urlopen", side_effect=dispatcher):
            res = client.post(
                "/api/v1/papers/ingest/openalex",
                json={"openalex_id": "W9000000005", "enable_evolution": True},
            )

    assert res.status_code == 200
    paper_uuid = uuid.UUID(res.json()["paper_id"])

    # Read strictly through brand new database session
    with SessionMaker() as fresh_sess:
        # 1. Paper is durable
        paper = fresh_sess.get(Paper, paper_uuid)
        assert paper is not None
        assert paper.source_id == "W9000000005"

        # 2. Opportunity state transition is durable
        opp = fresh_sess.get(Opportunity, opp_id)
        assert opp is not None
        assert opp.status == OpportunityStatus.PARTIALLY_ADDRESSED

        # 3. Evidence is durable
        ev = fresh_sess.query(OpportunityEvidence).filter_by(opportunity_id=opp_id, paper_id=paper_uuid).first()
        assert ev is not None
        assert ev.evidence_type in (OpportunityRelationship.ADDRESSES.value, OpportunityRelationship.ADDRESSES)

        # 4. History is durable
        hist = fresh_sess.query(OpportunityHistory).filter_by(opportunity_id=opp_id).first()
        assert hist is not None
        assert hist.new_status in (OpportunityStatus.PARTIALLY_ADDRESSED.value, OpportunityStatus.PARTIALLY_ADDRESSED)
