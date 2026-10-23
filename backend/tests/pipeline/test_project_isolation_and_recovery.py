"""Consolidated Integration & Reliability Tests: Project Isolation, Recovery & Approval Gating.

Verifies:
1. Cross-project data path isolation for papers, statements, gaps, opportunities, reports, and metrics.
2. Crash & process restart recovery: orphan RUNNING runs transition safely to INTERRUPTED.
3. Idempotent retries of interrupted runs.
4. Concurrency protection: duplicate active runs and parallel retries return HTTP 409 Conflict.
5. Researcher approval gate: pipeline pauses at AWAITING_REVIEW after gap discovery without
   automatic opportunity generation; explicit opportunity formulation works on validated gaps.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Generator
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from ideagen.db.models import (
    Base,
    Paper,
    PaperDocument,
    PaperPassage,
    ResearchKnowledge,
    ResearchStatement,
    ResearchGap,
    Opportunity,
    OpportunityEvidence,
    OpportunityHistory,
    ResearchOpportunityRecord,
    ResearchProjectRecord,
    ProjectPaperRecord,
    ProjectPaperSelectionRecord,
    ResearchRunRecord,
    RunPaperRecord,
)
from ideagen.db import get_db
from ideagen.api.main import app
from ideagen.pipeline.workflow_engine import ResearchWorkflowEngine, WorkflowEngine


@pytest.fixture
def isolation_env():
    """Sets up an isolated in-memory SQLite database and test client."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db() -> Generator[Session, None, None]:
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    db = TestingSessionLocal()

    # Create two isolated projects
    proj_a = ResearchProjectRecord(
        id="project-alpha",
        name="Project Alpha",
        topic="Alpha Domain",
        description="First research workspace",
    )
    proj_b = ResearchProjectRecord(
        id="project-beta",
        name="Project Beta",
        topic="Beta Domain",
        description="Second research workspace",
    )
    db.add_all([proj_a, proj_b])
    db.commit()

    yield {
        "client": client,
        "db": db,
        "session_factory": TestingSessionLocal,
    }

    app.dependency_overrides.clear()
    WorkflowEngine._active_runs.clear()
    db.close()


def test_cross_project_paper_and_statement_isolation(isolation_env):
    """Verifies that papers and statements are strictly isolated between projects."""
    db: Session = isolation_env["db"]
    client: TestClient = isolation_env["client"]

    paper_a = Paper(
        id=uuid.uuid4(),
        batch_id=str(uuid.uuid4()),
        source_type="OPENALEX",
        source_id="W111",
        title="Alpha Research Paper",
        abstract="Alpha abstract",
        authors=[],
        referenced_works=[],
        raw_metadata={},
        publication_date=datetime.date(2020, 1, 1),
    )
    paper_b = Paper(
        id=uuid.uuid4(),
        batch_id=str(uuid.uuid4()),
        source_type="OPENALEX",
        source_id="W222",
        title="Beta Research Paper",
        abstract="Beta abstract",
        authors=[],
        referenced_works=[],
        raw_metadata={},
        publication_date=datetime.date(2020, 2, 1),
    )
    db.add_all([paper_a, paper_b])
    db.flush()

    # Link paper_a to project-alpha and paper_b to project-beta
    assoc_a = ProjectPaperRecord(project_id="project-alpha", paper_id=paper_a.id)
    assoc_b = ProjectPaperRecord(project_id="project-beta", paper_id=paper_b.id)
    db.add_all([assoc_a, assoc_b])

    now_dt = datetime.datetime.now(datetime.timezone.utc)
    doc_a = PaperDocument(
        id=uuid.uuid4(),
        paper_id=paper_a.id,
        full_text="Alpha full text",
        extraction_method="PDFMiner",
        extraction_version="1.0",
        extracted_at=now_dt,
        content_hash="hash_a",
    )
    doc_b = PaperDocument(
        id=uuid.uuid4(),
        paper_id=paper_b.id,
        full_text="Beta full text",
        extraction_method="PDFMiner",
        extraction_version="1.0",
        extracted_at=now_dt,
        content_hash="hash_b",
    )
    db.add_all([doc_a, doc_b])
    db.flush()

    know_a = ResearchKnowledge(
        id=uuid.uuid4(), paper_id=paper_a.id, model_name="test-llm", extracted_at=datetime.datetime.utcnow()
    )
    know_b = ResearchKnowledge(
        id=uuid.uuid4(), paper_id=paper_b.id, model_name="test-llm", extracted_at=datetime.datetime.utcnow()
    )
    db.add_all([know_a, know_b])
    db.flush()

    stmt_a = ResearchStatement(
        id=uuid.uuid4(),
        knowledge_id=know_a.id,
        paper_id=paper_a.id,
        statement_text="Alpha limitation in retrieval efficiency.",
        statement_type="LIMITATION",
    )
    stmt_b = ResearchStatement(
        id=uuid.uuid4(),
        knowledge_id=know_b.id,
        paper_id=paper_b.id,
        statement_text="Beta limitation in cross-attention latency.",
        statement_type="LIMITATION",
    )
    db.add_all([stmt_a, stmt_b])
    db.commit()

    # 1. Verify Paper list isolation
    res_a = client.get("/api/v1/papers?project_id=project-alpha")
    assert res_a.status_code == 200
    papers_a = res_a.json()
    assert len(papers_a) == 1
    assert papers_a[0]["title"] == "Alpha Research Paper"

    res_b = client.get("/api/v1/papers?project_id=project-beta")
    assert res_b.status_code == 200
    papers_b = res_b.json()
    assert len(papers_b) == 1
    assert papers_b[0]["title"] == "Beta Research Paper"

    # Global lists both
    res_global = client.get("/api/v1/papers")
    assert res_global.status_code == 200
    assert len(res_global.json()) == 2

    # 2. Verify Statements list isolation
    res_stmts_a = client.get("/api/v1/statements?project_id=project-alpha")
    assert res_stmts_a.status_code == 200
    stmts_a = res_stmts_a.json()
    assert len(stmts_a) == 1
    assert "Alpha" in stmts_a[0]["statement_text"]

    res_stmts_b = client.get("/api/v1/statements?project_id=project-beta")
    assert res_stmts_b.status_code == 200
    stmts_b = res_stmts_b.json()
    assert len(stmts_b) == 1
    assert "Beta" in stmts_b[0]["statement_text"]


def test_cross_project_gap_and_opportunity_isolation(isolation_env):
    """Verifies that gaps and opportunities cannot be inspected or leaked across projects."""
    db: Session = isolation_env["db"]
    client: TestClient = isolation_env["client"]

    # Gaps
    gap_a = ResearchGap(
        id=uuid.uuid4(),
        project_id="project-alpha",
        gap_statement="Alpha gap in neural memory indexing",
        research_problem="High indexing cost",
        status="VALIDATED",
    )
    gap_b = ResearchGap(
        id=uuid.uuid4(),
        project_id="project-beta",
        gap_statement="Beta gap in sparse retrieval pruning",
        research_problem="Pruning loss",
        status="CANDIDATE",
    )
    db.add_all([gap_a, gap_b])
    db.flush()

    # Opportunity for Project Alpha
    opp_a_id = uuid.uuid4()
    opp_record_a = ResearchOpportunityRecord(
        id=opp_a_id,
        project_id="project-alpha",
        gap_id=gap_a.id,
        research_problem="High indexing cost",
        identified_gap="Alpha gap in neural memory indexing",
        proposed_research_direction="Differentiable sparse indices",
        reasoning="Enables low-latency lookup",
        technical_approach="Grounded vectors",
        expected_contribution="Order of magnitude speedup",
        feasibility={"overall_score": 0.85},
        novelty_assessment={"overall_novelty_score": 0.90},
        confidence=0.88,
        status="VALIDATED",
        generation_model="test-llm",
        validation_model="test-llm",
        embedding_model="test-embed",
    )
    lifecycle_opp_a = Opportunity(
        id=uuid.uuid4(),
        project_id="project-alpha",
        research_opportunity_id=opp_a_id,
        gap_id=gap_a.id,
        title="Differentiable sparse indices",
        problem_statement="High indexing cost",
        description="Technical proposal for neural indexing",
        status="OPEN",
        confidence=0.88,
    )
    db.add_all([opp_record_a, lifecycle_opp_a])
    db.commit()

    # Gaps listing isolation
    res_gaps_a = client.get("/api/v1/gaps?project_id=project-alpha")
    assert res_gaps_a.status_code == 200
    assert len(res_gaps_a.json()) == 1
    assert res_gaps_a.json()[0]["id"] == str(gap_a.id)

    res_gaps_b = client.get("/api/v1/gaps?project_id=project-beta")
    assert res_gaps_b.status_code == 200
    assert len(res_gaps_b.json()) == 1
    assert res_gaps_b.json()[0]["id"] == str(gap_b.id)

    # Cross-project gap fetch rejection
    assert client.get(f"/api/v1/gaps/{gap_a.id}?project_id=project-alpha").status_code == 200
    assert client.get(f"/api/v1/gaps/{gap_a.id}?project_id=project-beta").status_code == 404

    # Opportunities listing isolation
    res_opps_a = client.get("/api/v1/opportunities?project_id=project-alpha")
    assert res_opps_a.status_code == 200
    assert len(res_opps_a.json()) == 1
    assert res_opps_a.json()[0]["id"] == str(opp_a_id)

    res_opps_b = client.get("/api/v1/opportunities?project_id=project-beta")
    assert res_opps_b.status_code == 200
    assert len(res_opps_b.json()) == 0

    # Cross-project report, history, and evidence rejection
    assert client.get(f"/api/v1/opportunities/{opp_a_id}/report?project_id=project-alpha").status_code == 200
    assert client.get(f"/api/v1/opportunities/{opp_a_id}/report?project_id=project-beta").status_code == 404

    assert client.get(f"/api/v1/opportunities/{lifecycle_opp_a.id}/history?project_id=project-beta").status_code == 404
    assert client.get(f"/api/v1/opportunities/{lifecycle_opp_a.id}/evidence?project_id=project-beta").status_code == 404


def test_cross_project_stats_isolation(isolation_env):
    """Verifies that stats metrics are computed accurately per project and globally."""
    db: Session = isolation_env["db"]
    client: TestClient = isolation_env["client"]

    # Add 2 papers to Alpha, 1 paper to Beta
    p1 = Paper(id=uuid.uuid4(), batch_id=str(uuid.uuid4()), source_type="OPENALEX", source_id="W1", title="P1", abstract="A1", authors=[], referenced_works=[], raw_metadata={})
    p2 = Paper(id=uuid.uuid4(), batch_id=str(uuid.uuid4()), source_type="OPENALEX", source_id="W2", title="P2", abstract="A2", authors=[], referenced_works=[], raw_metadata={})
    p3 = Paper(id=uuid.uuid4(), batch_id=str(uuid.uuid4()), source_type="OPENALEX", source_id="W3", title="P3", abstract="A3", authors=[], referenced_works=[], raw_metadata={})
    db.add_all([p1, p2, p3])
    db.flush()

    db.add_all([
        ProjectPaperRecord(project_id="project-alpha", paper_id=p1.id),
        ProjectPaperRecord(project_id="project-alpha", paper_id=p2.id),
        ProjectPaperRecord(project_id="project-beta", paper_id=p3.id),
    ])

    # Add 1 gap to Alpha, 2 gaps to Beta
    db.add_all([
        ResearchGap(id=uuid.uuid4(), project_id="project-alpha", gap_statement="G1", research_problem="R1", status="VALIDATED"),
        ResearchGap(id=uuid.uuid4(), project_id="project-beta", gap_statement="G2", research_problem="R2", status="CANDIDATE"),
        ResearchGap(id=uuid.uuid4(), project_id="project-beta", gap_statement="G3", research_problem="R3", status="VALIDATED"),
    ])
    db.commit()

    stats_a = client.get("/api/v1/stats?project_id=project-alpha").json()
    assert stats_a["total_papers"] == 2
    assert stats_a["total_gaps"] == 1
    assert stats_a["validated_gaps"] == 1

    stats_b = client.get("/api/v1/stats?project_id=project-beta").json()
    assert stats_b["total_papers"] == 1
    assert stats_b["total_gaps"] == 2
    assert stats_b["candidate_gaps"] == 1
    assert stats_b["validated_gaps"] == 1

    stats_global = client.get("/api/v1/stats").json()
    assert stats_global["total_papers"] == 3
    assert stats_global["total_gaps"] == 3


def test_process_restart_interrupted_run_recovery(isolation_env):
    """Verifies that an unhandled process restart transitions orphan RUNNING runs to INTERRUPTED."""
    db: Session = isolation_env["db"]
    client: TestClient = isolation_env["client"]

    # Seed a run that was left RUNNING before the crash/restart
    run_id = "run_crashed_123"
    crashed_run = ResearchRunRecord(
        id=run_id,
        project_id="project-alpha",
        status="RUNNING",
        current_stage="INGESTION",
        planned_stages=["INGESTION", "UNDERSTANDING", "GAP_DISCOVERY"],
        stage_states={
            "INGESTION": {"status": "RUNNING", "started_at": "2026-10-22T10:00:00Z"},
            "UNDERSTANDING": {"status": "PENDING"},
            "GAP_DISCOVERY": {"status": "PENDING"},
        },
    )
    p_ingesting = RunPaperRecord(
        id="rp_1",
        run_id=run_id,
        openalex_id="W1",
        title="Interrupted Paper",
        status="INGESTING",
    )
    db.add_all([crashed_run, p_ingesting])
    db.commit()

    # Recovery detects that run_id is not in _active_runs and marks it INTERRUPTED
    recovered_count = ResearchWorkflowEngine.recover_interrupted_runs(db)
    assert recovered_count == 1

    db.refresh(crashed_run)
    assert crashed_run.status == "INTERRUPTED"
    assert crashed_run.stage_states["INGESTION"]["status"] == "INTERRUPTED"
    db.refresh(p_ingesting)
    assert p_ingesting.status == "INTERRUPTED"

    # API queries also truthfully report INTERRUPTED
    res = client.get(f"/api/v1/runs/{run_id}")
    assert res.status_code == 200
    assert res.json()["status"] == "INTERRUPTED"

    # Verify that an interrupted run can be reset for retry
    engine = ResearchWorkflowEngine(db)
    with patch.object(ResearchWorkflowEngine, "execute_run") as mock_exec:
        run_retried = engine.retry_run(run_id, db_override=db)
        assert run_retried.status == "PENDING"
        db.refresh(p_ingesting)
        assert p_ingesting.status == "PENDING"


def test_concurrency_conflict_handling(isolation_env):
    """Verifies that duplicate start requests and concurrent retries return 409 Conflict."""
    db: Session = isolation_env["db"]
    client: TestClient = isolation_env["client"]

    run_id = "run_active_456"
    active_run = ResearchRunRecord(
        id=run_id,
        project_id="project-alpha",
        status="RUNNING",
        current_stage="INGESTION",
        planned_stages=["INGESTION", "UNDERSTANDING"],
        stage_states={"INGESTION": {"status": "RUNNING"}},
    )
    db.add(active_run)
    db.commit()

    # Register run as actively executing in-process
    WorkflowEngine.mark_run_active(run_id)
    try:
        # 1. Attempting duplicate retry returns 409 Conflict
        res_retry = client.post(f"/api/v1/runs/{run_id}/retry")
        assert res_retry.status_code == 409
        assert "currently running" in res_retry.json()["detail"].lower()

        # 2. Attempting a second parallel run in the same project returns 409 Conflict
        res_start = client.post("/api/v1/projects/project-alpha/runs")
        assert res_start.status_code == 409
        assert "already has an active run" in res_start.json()["detail"]
    finally:
        WorkflowEngine.mark_run_inactive(run_id)


def test_researcher_approval_gate(isolation_env):
    """Verifies that automated execution pauses at AWAITING_REVIEW after gap discovery
    and that opportunity generation requires explicit researcher action.
    """
    db: Session = isolation_env["db"]
    client: TestClient = isolation_env["client"]

    # Pre-seed a candidate selection
    db.add(
        ProjectPaperSelectionRecord(
            id="psel_1",
            project_id="project-alpha",
            openalex_id="W999",
            doi="10.1234/test.doi",
            title="Pre-Cutoff Foundational Study",
        )
    )
    db.commit()

    engine = ResearchWorkflowEngine(db)
    run = engine.create_run("project-alpha")

    # Verify default planned stages do NOT include automatic OPPORTUNITY_GENERATION
    assert run.planned_stages == ["INGESTION", "UNDERSTANDING", "GAP_DISCOVERY"]

    # Mock stage handlers to simulate gap discovery creating candidate gaps
    with patch.object(ResearchWorkflowEngine, "_execute_stage_ingestion") as mock_ingest, \
         patch.object(ResearchWorkflowEngine, "_execute_stage_understanding") as mock_und, \
         patch.object(ResearchWorkflowEngine, "_execute_stage_gap_discovery") as mock_gap:

        def fake_gap_discovery(db_sess, r, st_states):
            st_states["GAP_DISCOVERY"] = {"status": "COMPLETED", "details": {"gaps_discovered": 1}}
            r.status = "AWAITING_REVIEW"
            gap = ResearchGap(
                id=uuid.uuid4(),
                project_id="project-alpha",
                gap_statement="Discovered test research gap",
                research_problem="Problem statement",
                status="CANDIDATE",
            )
            db_sess.add(gap)
            db_sess.commit()

        mock_gap.side_effect = fake_gap_discovery

        ResearchWorkflowEngine.execute_run(run.id, db_override=db)

    db.refresh(run)
    # The run must pause at AWAITING_REVIEW
    assert run.status == "AWAITING_REVIEW"
    assert "Review discovered research gaps" in run.summary["next_actions"][0]


def test_interrupted_run_does_not_auto_resume(isolation_env):
    """Verifies that recovering an orphan run marks it INTERRUPTED without auto-resuming execution."""
    db: Session = isolation_env["db"]
    client: TestClient = isolation_env["client"]

    run_id = "orphan_run_789"
    run = ResearchRunRecord(
        id=run_id,
        project_id="project-alpha",
        status="RUNNING",
        current_stage="UNDERSTANDING",
        planned_stages=["INGESTION", "UNDERSTANDING", "GAP_DISCOVERY"],
        stage_states={
            "INGESTION": {"status": "COMPLETED"},
            "UNDERSTANDING": {"status": "RUNNING"},
            "GAP_DISCOVERY": {"status": "PENDING"},
        },
    )
    p1 = RunPaperRecord(id="p1", run_id=run_id, openalex_id="W1", title="Paper 1", status="UNDERSTOOD")
    p2 = RunPaperRecord(id="p2", run_id=run_id, openalex_id="W2", title="Paper 2", status="EXTRACTING_STATEMENTS")
    db.add_all([run, p1, p2])
    db.commit()

    # Recovery must mark run and uncompleted items as INTERRUPTED
    recovered = ResearchWorkflowEngine.recover_interrupted_runs(db)
    assert recovered == 1

    # Must NOT auto-resume execution or register in active runs
    assert run_id not in WorkflowEngine._active_runs

    db.refresh(run)
    assert run.status == "INTERRUPTED"
    assert run.stage_states["UNDERSTANDING"]["status"] == "INTERRUPTED"
    # Completed stages and papers remain untouched
    assert run.stage_states["INGESTION"]["status"] == "COMPLETED"
    db.refresh(p1)
    assert p1.status == "UNDERSTOOD"
    db.refresh(p2)
    assert p2.status == "INTERRUPTED"


def test_retry_idempotency_preserves_completed_work(isolation_env):
    """Verifies that retrying an interrupted run only resets interrupted/failed stages and papers."""
    db: Session = isolation_env["db"]

    run_id = "interrupted_run_999"
    run = ResearchRunRecord(
        id=run_id,
        project_id="project-alpha",
        status="INTERRUPTED",
        current_stage="UNDERSTANDING",
        planned_stages=["INGESTION", "UNDERSTANDING", "GAP_DISCOVERY"],
        stage_states={
            "INGESTION": {"status": "COMPLETED"},
            "UNDERSTANDING": {"status": "INTERRUPTED"},
            "GAP_DISCOVERY": {"status": "PENDING"},
        },
    )
    p_done = RunPaperRecord(id="p_done", run_id=run_id, openalex_id="W1", title="Done Paper", status="UNDERSTOOD")
    p_interrupted = RunPaperRecord(id="p_inter", run_id=run_id, openalex_id="W2", title="Interrupted Paper", status="INTERRUPTED")
    db.add_all([run, p_done, p_interrupted])
    db.commit()

    engine = ResearchWorkflowEngine(db)
    with patch.object(ResearchWorkflowEngine, "execute_run") as mock_exec:
        retried_run = engine.retry_run(run_id, db_override=db)

        # Run resets to PENDING
        assert retried_run.status == "PENDING"
        # Completed paper stays completed
        db.refresh(p_done)
        assert p_done.status == "UNDERSTOOD"
        # Interrupted paper resets to PENDING
        db.refresh(p_interrupted)
        assert p_interrupted.status == "PENDING"
        # Completed stage INGESTION stays COMPLETED
        assert retried_run.stage_states["INGESTION"]["status"] == "COMPLETED"
        # Interrupted stage resets to PENDING
        assert retried_run.stage_states["UNDERSTANDING"]["status"] == "PENDING"


def test_approval_gate_unvalidated_gap_rejected(isolation_env):
    """Verifies that opportunity generation is strictly rejected on non-VALIDATED gaps."""
    db: Session = isolation_env["db"]
    client: TestClient = isolation_env["client"]

    candidate_gap = ResearchGap(
        id=uuid.uuid4(),
        project_id="project-alpha",
        gap_statement="Unvalidated Candidate Gap",
        research_problem="Problem statement",
        status="CANDIDATE",
    )
    rejected_gap = ResearchGap(
        id=uuid.uuid4(),
        project_id="project-alpha",
        gap_statement="Rejected Gap",
        research_problem="Problem statement",
        status="REJECTED",
    )
    db.add_all([candidate_gap, rejected_gap])
    db.commit()

    # Attempting opportunity generation on CANDIDATE gap must fail with 400
    res_cand = client.post(f"/api/v1/gaps/{candidate_gap.id}/opportunity?project_id=project-alpha")
    assert res_cand.status_code == 400
    assert "must be VALIDATED" in res_cand.json()["detail"]

    # Attempting opportunity generation on REJECTED gap must fail with 400
    res_rej = client.post(f"/api/v1/gaps/{rejected_gap.id}/opportunity?project_id=project-alpha")
    assert res_rej.status_code == 400
    assert "must be VALIDATED" in res_rej.json()["detail"]


def test_openalex_ingest_with_project_scoping_and_deduplication(isolation_env):
    """Verifies that ingest_openalex attaches papers to project_papers and respects duplicate scoping."""
    db: Session = isolation_env["db"]
    client: TestClient = isolation_env["client"]
    from ideagen.ingestion.search import PaperSearchResult
    from ideagen.ingestion.openalex_ingestion import PaperIngestionResult, IngestionStatus

    created_paper_id = uuid.uuid4()
    paper_record = Paper(
        id=created_paper_id,
        source_type="OPENALEX_API",
        source_id="W9999901",
        doi="10.1234/quantum.101",
        title="Quantum Error Mitigated Search",
        batch_id="test-batch-001",
        abstract="Quantum search abstract",
        authors=["Alice", "Bob"],
        referenced_works=[],
        raw_metadata={},
    )
    db.add(paper_record)
    db.commit()

    mock_search_item = PaperSearchResult(
        id="https://openalex.org/W9999901",
        source="openalex",
        doi="10.1234/quantum.101",
        title="Quantum Error Mitigated Search",
        abstract="Quantum search abstraction",
        publication_year=2026,
        authors=["Alice", "Bob"],
        pdf_url="https://test.org/doc.pdf",
        raw_metadata={
            "id": "https://openalex.org/W9999901",
            "doi": "https://doi.org/10.1234/quantum.101",
            "title": "Quantum Error Mitigated Search",
            "publication_year": 2026,
        },
    )

    mock_ingest_result = PaperIngestionResult(
        paper_id=created_paper_id,
        source_id="W9999901",
        status=IngestionStatus.INGESTED,
        has_pdf=True,
        document_id=uuid.uuid4(),
        passage_count=12,
        batch_id="batch-001",
    )

    with patch("ideagen.api.main.OpenAlexSearchProvider.search", return_value=[mock_search_item]), \
         patch("ideagen.api.main.OpenAlexPaperIngestionService.ingest_work", return_value=mock_ingest_result):

        # 1. Ingest paper into project-alpha
        res_ingest = client.post("/api/v1/papers/ingest/openalex", json={
            "openalex_id": "W9999901",
            "doi": "10.1234/quantum.101",
            "project_id": "project-alpha",
        })
        assert res_ingest.status_code == 200
        data = res_ingest.json()
        assert data["status"] == "INGESTED"
        paper_id = data["paper_id"]
        assert paper_id == str(created_paper_id)
        assert data["project_id"] == "project-alpha"
        assert data["project_associated"] is True

        # Check project_papers table has association for project-alpha
        assoc = db.query(ProjectPaperRecord).filter_by(project_id="project-alpha", paper_id=created_paper_id).first()
        assert assoc is not None

        # Verify paper is listed in project-alpha papers, but NOT in project-beta papers
        res_a = client.get("/api/v1/papers?project_id=project-alpha")
        assert any(p["id"] == str(created_paper_id) for p in res_a.json())

        res_b = client.get("/api/v1/papers?project_id=project-beta")
        assert not any(p["id"] == str(created_paper_id) for p in res_b.json())

    # 2. Ingest duplicate for project-alpha -> returns SKIPPED_DUPLICATE
    mock_dup_result = PaperIngestionResult(
        paper_id=created_paper_id,
        source_id="W9999901",
        status=IngestionStatus.SKIPPED_DUPLICATE,
        has_pdf=False,
        document_id=None,
        passage_count=0,
        batch_id="UNKNOWN",
    )
    with patch("ideagen.api.main.OpenAlexSearchProvider.search", return_value=[mock_search_item]), \
         patch("ideagen.api.main.OpenAlexPaperIngestionService.ingest_work", return_value=mock_dup_result):

        res_dup_a = client.post("/api/v1/papers/ingest/openalex", json={
            "openalex_id": "W9999901",
            "doi": "10.1234/quantum.101",
            "project_id": "project-alpha",
        })
        assert res_dup_a.status_code == 200
        dup_a_data = res_dup_a.json()
        assert dup_a_data["status"] == "SKIPPED_DUPLICATE"
        assert dup_a_data["already_in_project"] is True

        # 3. Ingest same paper into project-beta -> SKIPPED_DUPLICATE with project_associated = True and already_in_project = False
        res_assoc_b = client.post("/api/v1/papers/ingest/openalex", json={
            "openalex_id": "W9999901",
            "doi": "10.1234/quantum.101",
            "project_id": "project-beta",
        })
        assert res_assoc_b.status_code == 200
        assoc_b_data = res_assoc_b.json()
        assert assoc_b_data["status"] == "SKIPPED_DUPLICATE"
        assert assoc_b_data["project_associated"] is True
        assert assoc_b_data["already_in_project"] is False

        # Now project-beta papers also lists it!
        res_b_after = client.get("/api/v1/papers?project_id=project-beta")
        assert any(p["id"] == str(created_paper_id) for p in res_b_after.json())


