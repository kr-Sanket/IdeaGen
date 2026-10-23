"""Tests for Phase 3 End-to-End Research Workflow.

Validates:
1. Server-side paper selections persistence and retrieval.
2. Preflight verification (new papers, corpus matches, PDF availability, planned stages, warnings).
3. Research run creation, stage progress tracking, and per-paper status recording.
4. Fault tolerance: paper-level failure isolation (one paper failure doesn't abort the run).
5. Safe, idempotent retries of failed/interrupted runs.
6. Project data isolation: gaps and opportunities are scoped by project_id.
7. Technical logs retrieval endpoint.
"""

from __future__ import annotations

import datetime
import io
import json
import uuid
from typing import Dict, Any, Optional, List
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
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
    ResearchOpportunityRecord,
    ResearchProjectRecord,
    ProjectPaperSelectionRecord,
    ResearchRunRecord,
    RunPaperRecord,
)
from ideagen.db import get_db
from ideagen.api.main import app
from ideagen.pipeline.workflow_engine import ResearchWorkflowEngine, WorkflowEngine


@pytest.fixture
def workflow_test_env():
    """Sets up an in-memory SQLite database and test client."""
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

    yield client, TestingSessionLocal

    app.dependency_overrides.clear()


def test_project_crud_and_paper_selections(workflow_test_env):
    """Verifies project creation, selection persistence, and deletion."""
    client, SessionLocal = workflow_test_env

    # 1. Create project
    proj_resp = client.post("/api/v1/projects", json={"id": "quantum-ml", "title": "Quantum ML Research"})
    assert proj_resp.status_code == 200
    proj_data = proj_resp.json()
    assert proj_data["id"] == "quantum-ml"
    assert proj_data["title"] == "Quantum ML Research"

    # 2. Add paper selections
    papers_payload = {
        "selections": [
            {
                "openalex_id": "W1001",
                "doi": "10.1000/1001",
                "title": "Quantum Attention Networks",
                "authors": ["Alice A."],
                "publication_year": 2024,
                "pdf_url": "https://example.org/w1001.pdf",
            },
            {
                "openalex_id": "W1002",
                "doi": "10.1000/1002",
                "title": "Error Mitigation in Variational Circuits",
                "authors": ["Bob B."],
                "publication_year": 2023,
                "pdf_url": None,
            },
        ]
    }
    sel_resp = client.post("/api/v1/projects/quantum-ml/selections", json=papers_payload)
    assert sel_resp.status_code == 200
    selections = sel_resp.json()
    assert len(selections) == 2
    ids = {s["openalex_id"] for s in selections}
    assert ids == {"W1001", "W1002"}

    # 3. Retrieve selections (simulating return to page)
    get_sel_resp = client.get("/api/v1/projects/quantum-ml/selections")
    assert get_sel_resp.status_code == 200
    assert len(get_sel_resp.json()) == 2

    # 4. Remove one selection
    del_resp = client.delete("/api/v1/projects/quantum-ml/selections/W1002")
    assert del_resp.status_code == 200
    rem_resp = client.get("/api/v1/projects/quantum-ml/selections")
    assert len(rem_resp.json()) == 1
    assert rem_resp.json()[0]["openalex_id"] == "W1001"


def test_preflight_check(workflow_test_env):
    """Verifies preflight duplicate identification and PDF availability."""
    client, SessionLocal = workflow_test_env

    # Pre-seed an existing paper in corpus
    db = SessionLocal()
    existing_paper = Paper(
        id=uuid.uuid4(),
        source_type="OPENALEX",
        source_id="W1001",
        doi="10.1000/1001",
        title="Quantum Attention Networks",
        abstract="Quantum Attention Networks Abstract",
        authors=["Alice A."],
        referenced_works=[],
        raw_metadata={},
        batch_id="batch-seed-01",
        created_at=datetime.datetime.utcnow(),
    )
    db.add(existing_paper)
    db.commit()
    db.close()

    # Add selections to project: one existing in corpus, one new with PDF, one new without PDF
    payload = {
        "selections": [
            {
                "openalex_id": "W1001",
                "doi": "10.1000/1001",
                "title": "Quantum Attention Networks",
                "pdf_url": "https://example.org/w1001.pdf",
            },
            {
                "openalex_id": "W2001",
                "doi": "10.1000/2001",
                "title": "Fault Tolerant Thresholds",
                "pdf_url": "https://example.org/w2001.pdf",
            },
            {
                "openalex_id": "W3001",
                "doi": "10.1000/3001",
                "title": "Metadata Only Survey",
                "pdf_url": None,
            },
        ]
    }
    assert client.post("/api/v1/projects", json={"id": "proj-preflight", "title": "Preflight"}).status_code == 200
    assert client.post("/api/v1/projects/proj-preflight/selections", json=payload).status_code == 200

    # Call Preflight
    pf_resp = client.get("/api/v1/projects/proj-preflight/preflight")
    assert pf_resp.status_code == 200
    pf = pf_resp.json()
    assert pf["total_selected"] == 3
    assert pf["corpus_matches"] == 1
    assert pf["new_papers"] == 2
    assert pf["pdf_available"] == 2
    assert pf["metadata_only"] == 1
    # Opportunity generation is never a run stage; it requires explicit researcher action.
    assert [st["stage_id"] for st in pf["planned_stages"]] == ["INGESTION", "UNDERSTANDING", "GAP_DISCOVERY"]

    # Check status per paper
    p_map = {p["openalex_id"]: p for p in pf["papers"]}
    assert p_map["W1001"]["status"] == "CORPUS_MATCH"
    assert p_map["W2001"]["status"] == "NEW"
    assert p_map["W3001"]["status"] == "METADATA_ONLY"


def test_research_run_creation_and_polling(workflow_test_env):
    """Verifies that creating a run persists run and paper records and returns status."""
    client, SessionLocal = workflow_test_env

    assert client.post("/api/v1/projects", json={"id": "test-run-proj", "title": "Run Project"}).status_code == 200
    # Add paper selection
    client.post("/api/v1/projects/test-run-proj/selections", json={
        "selections": [
            {
                "openalex_id": "W5001",
                "doi": "10.1000/5001",
                "title": "Test Paper For Run",
                "pdf_url": "https://example.org/w5001.pdf",
            }
        ]
    })

    # Start run (mocking the background executor so it doesn't do network calls; it must still
    # release the active-run registration that start_run_async takes before spawning the thread)
    with patch(
        "ideagen.pipeline.workflow_engine.ResearchWorkflowEngine._execute_marked_run",
        side_effect=lambda run_id, db_override=None: ResearchWorkflowEngine.mark_run_inactive(run_id),
    ):
        run_resp = client.post("/api/v1/projects/test-run-proj/runs", json={})
        assert run_resp.status_code == 200
        run_data = run_resp.json()
        assert run_data["project_id"] == "test-run-proj"
        assert run_data["status"] in ("PENDING", "RUNNING")
        assert len(run_data["papers"]) == 1
        assert run_data["papers"][0]["openalex_id"] == "W5001"

        run_id = run_data["id"]

        # Poll run status
        poll_resp = client.get(f"/api/v1/runs/{run_id}")
        assert poll_resp.status_code == 200
        assert poll_resp.json()["id"] == run_id

        # Check logs endpoint
        logs_resp = client.get(f"/api/v1/runs/{run_id}/logs")
        assert logs_resp.status_code == 200
        assert logs_resp.json()["run_id"] == run_id


def test_run_fault_tolerance_and_retry(workflow_test_env):
    """Verifies that paper errors do not abort the entire run and retries are safe."""
    client, SessionLocal = workflow_test_env
    db = SessionLocal()

    # Create project and run directly via WorkflowEngine
    engine = ResearchWorkflowEngine(db)
    selections = [
        {"openalex_id": "W-PASS", "title": "Passing Paper", "doi": "10.1/pass"},
        {"openalex_id": "W-FAIL", "title": "Failing Paper", "doi": "10.1/fail"},
    ]
    run = engine.create_run("fault-tolerance-proj", selections=selections)
    run_id = run.id

    # Mock stage ingestion so W-PASS succeeds and W-FAIL throws an error
    def fake_stage_ingestion(session, r, stage_states):
        for p in r.papers:
            if p.openalex_id == "W-FAIL":
                p.status = "ERROR"
                p.error_message = "Simulated upstream download timeout"
                ResearchWorkflowEngine._append_log(r, "ERROR", "INGESTION", f"Paper {p.openalex_id} failed: timeout")
            else:
                p.status = "INGESTED"
                p.passages_count = 5
                ResearchWorkflowEngine._append_log(r, "INFO", "INGESTION", f"Paper {p.openalex_id} ingested")
        stage_states["INGESTION"]["status"] = "PARTIAL_SUCCESS"
        session.commit()

    with patch.object(ResearchWorkflowEngine, "_execute_stage_ingestion", side_effect=fake_stage_ingestion):
        with patch.object(ResearchWorkflowEngine, "_execute_stage_understanding"):
            with patch.object(ResearchWorkflowEngine, "_execute_stage_gap_discovery"):
                ResearchWorkflowEngine.execute_run(run_id, db_override=db)

    # Verify run state in database
    db.refresh(run)
    assert run.status in ("PARTIAL_SUCCESS", "AWAITING_REVIEW")
    assert run.summary is not None
    assert run.summary["successful_papers"] == 1
    assert run.summary["failed_papers"] == 1

    papers_map = {p.openalex_id: p for p in run.papers}
    assert papers_map["W-PASS"].status in ("INGESTED", "ANALYZED", "COMPLETED")
    assert papers_map["W-FAIL"].status == "ERROR"
    assert "timeout" in papers_map["W-FAIL"].error_message

    # Test Retry: Fix W-FAIL and retry
    def fake_retry_ingestion(session, r, stage_states):
        for p in r.papers:
            if p.status == "ERROR" or p.status == "PENDING":
                p.status = "INGESTED"
                p.error_message = None
                p.passages_count = 3
        stage_states["INGESTION"]["status"] = "COMPLETED"
        session.commit()

    with patch.object(ResearchWorkflowEngine, "_execute_stage_ingestion", side_effect=fake_retry_ingestion):
        with patch.object(ResearchWorkflowEngine, "_execute_stage_understanding"):
            with patch.object(ResearchWorkflowEngine, "_execute_stage_gap_discovery"):
                engine.retry_run(run_id, db_override=db)


    db.refresh(run)
    assert run.summary["failed_papers"] == 0
    assert run.summary["successful_papers"] == 2
    assert run.status in ("COMPLETED", "AWAITING_REVIEW")
    db.close()


def test_project_data_isolation(workflow_test_env):
    """Verifies that gaps and opportunities are isolated by project_id."""
    client, SessionLocal = workflow_test_env
    db = SessionLocal()

    # Create two projects
    db.add(ResearchProjectRecord(id="proj-alpha", title="Alpha Project", created_at=datetime.datetime.utcnow(), updated_at=datetime.datetime.utcnow()))
    db.add(ResearchProjectRecord(id="proj-beta", title="Beta Project", created_at=datetime.datetime.utcnow(), updated_at=datetime.datetime.utcnow()))

    # Create gaps for each project
    gap_a = ResearchGap(
        id=uuid.uuid4(),
        project_id="proj-alpha",
        gap_statement="Alpha Gap: Lack of quantum decoherence mitigation",
        research_problem="Quantum Decoherence",
        status="VALIDATED",
        evidence_count=2,
        created_at=datetime.datetime.utcnow(),
    )
    gap_b = ResearchGap(
        id=uuid.uuid4(),
        project_id="proj-beta",
        gap_statement="Beta Gap: Lack of LLM reasoning verifiability",
        research_problem="LLM Reasoning",
        status="VALIDATED",
        evidence_count=3,
        created_at=datetime.datetime.utcnow(),
    )
    db.add_all([gap_a, gap_b])

    opp_a = ResearchOpportunityRecord(
        id=uuid.uuid4(),
        project_id="proj-alpha",
        gap_id=gap_a.id,
        research_problem="Quantum Decoherence",
        identified_gap=gap_a.gap_statement,
        proposed_research_direction="Adaptive surface codes",
        reasoning="Rigorous analysis shows potential for decoherence mitigation.",
        technical_approach="Syndrome extraction loop",
        expected_contribution="Novel decoder",
        feasibility={"feasibility_score": 0.85},
        novelty_assessment={"assessment": "High novelty"},
        generation_model="mock-qwen",
        validation_model="mock-qwen",
        embedding_model="mock-embedder",
        status="VALIDATED",
        created_at=datetime.datetime.utcnow(),
    )
    opp_b = ResearchOpportunityRecord(
        id=uuid.uuid4(),
        project_id="proj-beta",
        gap_id=gap_b.id,
        research_problem="LLM Reasoning",
        identified_gap=gap_b.gap_statement,
        proposed_research_direction="Formal verification step extraction",
        reasoning="Formal verification provides step-level guarantees.",
        technical_approach="Interactive theorem prover harness",
        expected_contribution="Verifiable reasoning pipeline",
        feasibility={"feasibility_score": 0.90},
        novelty_assessment={"assessment": "Provable novelty"},
        generation_model="mock-qwen",
        validation_model="mock-qwen",
        embedding_model="mock-embedder",
        status="VALIDATED",
        created_at=datetime.datetime.utcnow(),
    )
    db.add_all([opp_a, opp_b])
    db.commit()
    db.close()

    # Query gaps filtered by proj-alpha
    resp_gaps_a = client.get("/api/v1/gaps?project_id=proj-alpha")
    assert resp_gaps_a.status_code == 200
    gaps_a = resp_gaps_a.json()
    assert len(gaps_a) == 1
    assert gaps_a[0]["research_problem"] == "Quantum Decoherence"

    # Query gaps filtered by proj-beta
    resp_gaps_b = client.get("/api/v1/gaps?project_id=proj-beta")
    assert resp_gaps_b.status_code == 200
    gaps_b = resp_gaps_b.json()
    assert len(gaps_b) == 1
    assert gaps_b[0]["research_problem"] == "LLM Reasoning"

    # Query opportunities filtered by proj-alpha
    resp_opps_a = client.get("/api/v1/opportunities?project_id=proj-alpha")
    assert resp_opps_a.status_code == 200
    opps_a = resp_opps_a.json()
    assert len(opps_a) == 1
    assert opps_a[0]["research_problem"] == "Quantum Decoherence"

    # Query opportunities filtered by proj-beta
    resp_opps_b = client.get("/api/v1/opportunities?project_id=proj-beta")
    assert resp_opps_b.status_code == 200
    opps_b = resp_opps_b.json()
    assert len(opps_b) == 1
    assert opps_b[0]["research_problem"] == "LLM Reasoning"
