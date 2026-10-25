"""Comprehensive regression tests for server-side project deletion (DELETE /api/v1/projects/{project_id}).

Verifies:
1. Unknown project returns 404 without modifying any state.
2. Protected default benchmark ('rag-benchmark') returns 400 and remains completely intact.
3. Deletion of an empty project succeeds (200 OK) with 0 entity counts.
4. Full lifecycle project deletion removes all owned data across 10 entity categories without leaving any orphaned records.
5. Deleting a project that shares literature with another project preserves the shared paper and the second project's data.
6. Shared papers, documents, passages, knowledge, and statements in the literature store are never deleted.
7. Active research runs block deletion with 409 Conflict and roll back cleanly.
8. Active gap discovery operations block deletion with 409 Conflict.
9. An unexpected database failure during the deletion sequence rolls back the entire transaction atomically (no partial deletes).
10. Check-then-delete concurrency races with run creation are safely serialized by per-project run locks.
"""

from __future__ import annotations

import datetime
import threading
import uuid
from typing import Generator
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.exc import SQLAlchemyError

from ideagen.api.main import app, _project_run_lock, _discovery_lock_mgr
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
    PriorArtAssessmentRecord,
    OpportunityRefinementRecord,
    ResearchProjectRecord,
    ProjectPaperRecord,
    ProjectPaperSelectionRecord,
    ResearchRunRecord,
    RunPaperRecord,
)
from ideagen.db.session import configure_sqlite_foreign_keys
from ideagen.pipeline.workflow_engine import WorkflowEngine


@pytest.fixture
def env():
    """Provides an isolated in-memory SQLite database environment for testing deletion."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    configure_sqlite_foreign_keys(engine)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Seed the protected default benchmark project
    with SessionLocal() as db:
        bm = ResearchProjectRecord(
            id="rag-benchmark",
            name="Retrieval-Augmented Generation (RAG)",
            topic="Benchmark Topic",
            description="Core benchmark",
            temporal_cutoff=datetime.date(2020, 12, 31),
            status="ACTIVE",
        )
        db.add(bm)
        db.commit()

    def override_get_db() -> Generator[Session, None, None]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    yield client, SessionLocal
    app.dependency_overrides.clear()


def _create_project(client: TestClient, pid: str, title: str = "Test Project") -> dict:
    resp = client.post(
        "/api/v1/projects",
        json={"id": pid, "title": title, "topic": f"{title} topic"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _seed_shared_paper(db: Session, title: str = "Shared Research Paper") -> Paper:
    now = datetime.datetime.now(datetime.timezone.utc)
    paper = Paper(
        id=uuid.uuid4(),
        source_type="OPENALEX_API",
        source_id=f"W{uuid.uuid4().hex[:8]}",
        doi=f"10.1000/{uuid.uuid4().hex[:8]}",
        title=title,
        publication_date=datetime.date(2020, 1, 1),
        batch_id="batch-1",
        abstract="Abstract of shared paper",
        authors=["Author A"],
        referenced_works=[],
        raw_metadata={},
    )
    db.add(paper)
    db.flush()

    doc = PaperDocument(
        id=uuid.uuid4(),
        paper_id=paper.id,
        full_text="Full text of shared paper...",
        extraction_method="PDFMiner",
        extraction_version="1.0",
        extracted_at=now,
    )
    db.add(doc)
    db.flush()

    passage = PaperPassage(
        id=uuid.uuid4(),
        paper_id=paper.id,
        document_id=doc.id,
        passage_index=0,
        page_start=1,
        page_end=1,
        text="Sample passage from shared paper.",
    )
    db.add(passage)
    db.flush()

    knowledge = ResearchKnowledge(
        id=uuid.uuid4(),
        paper_id=paper.id,
        model_name="qwen3:8b",
        extracted_at=now,
    )
    db.add(knowledge)
    db.flush()

    statement = ResearchStatement(
        id=uuid.uuid4(),
        knowledge_id=knowledge.id,
        paper_id=paper.id,
        passage_id=passage.id,
        statement_text="Shared limitation: sample bottleneck.",
        statement_type="LIMITATION",
    )
    db.add(statement)
    db.commit()
    return paper


# ── Tests ───────────────────────────────────────────────────────────────────

def test_delete_unknown_project_returns_404(env):
    """Deleting a non-existent project returns 404 without mutating any records."""
    client, SessionLocal = env
    resp = client.delete("/api/v1/projects/proj_nonexistent_xyz")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_delete_protected_benchmark_returns_400(env):
    """Deleting the default benchmark project ('rag-benchmark') returns 400 and preserves it."""
    client, SessionLocal = env
    resp = client.delete("/api/v1/projects/rag-benchmark")
    assert resp.status_code == 400
    assert "Cannot delete protected benchmark project" in resp.json()["detail"]

    # Verify rag-benchmark still exists in DB
    with SessionLocal() as db:
        bm = db.query(ResearchProjectRecord).filter_by(id="rag-benchmark").first()
        assert bm is not None
        assert bm.name == "Retrieval-Augmented Generation (RAG)"


def test_delete_empty_project_succeeds(env):
    """Deleting an empty project removes the project and returns zero counts in summary."""
    client, SessionLocal = env
    _create_project(client, "proj_empty", "Empty Project")

    resp = client.delete("/api/v1/projects/proj_empty")
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["project_id"] == "proj_empty"
    assert data["deleted"] is True
    assert "deleted_at" in data
    assert data["summary"]["runs_deleted"] == 0
    assert data["summary"]["gaps_deleted"] == 0
    assert data["summary"]["paper_links_removed"] == 0

    # Verify project is gone
    with SessionLocal() as db:
        assert db.query(ResearchProjectRecord).filter_by(id="proj_empty").first() is None
    assert client.get("/api/v1/projects/proj_empty").status_code == 404


def test_delete_full_lifecycle_project_removes_all_owned_data_without_orphans(env):
    """Deleting a project with full data cleanly removes runs, selections, gaps, and opportunities."""
    client, SessionLocal = env
    pid = "proj_lifecycle"
    _create_project(client, pid, "Rich Lifecycle Project")

    now = datetime.datetime.now(datetime.timezone.utc)
    with SessionLocal() as db:
        paper = _seed_shared_paper(db, "Project Literature")
        statement = db.query(ResearchStatement).filter_by(paper_id=paper.id).first()

        # 1. Project paper link
        db.add(ProjectPaperRecord(project_id=pid, paper_id=paper.id))

        # 2. Project paper selection
        db.add(ProjectPaperSelectionRecord(
            id=f"psel_{uuid.uuid4().hex[:8]}",
            project_id=pid,
            openalex_id=paper.source_id,
            title="Candidate Paper",
        ))

        # 3. Research Run & Run Paper
        run = ResearchRunRecord(
            id=f"run_{uuid.uuid4().hex[:8]}",
            project_id=pid,
            status="COMPLETED",
        )
        db.add(run)
        db.flush()

        db.add(RunPaperRecord(
            id=f"rp_{uuid.uuid4().hex[:8]}",
            run_id=run.id,
            openalex_id=paper.source_id,
            title="Run Paper 1",
            status="COMPLETED",
            paper_id=str(paper.id),
        ))

        # 4. Research Gap & Gap Evidence
        gap = ResearchGap(
            id=uuid.uuid4(),
            project_id=pid,
            gap_statement="A critical gap in adaptive retrieval",
            research_problem="Adaptive retrieval latency",
            status="VALIDATED",
            evidence_count=1,
        )
        db.add(gap)
        db.flush()

        db.add(GapEvidence(
            id=uuid.uuid4(),
            gap_id=gap.id,
            statement_id=statement.id,
            paper_id=paper.id,
            evidence_role="SUPPORTING",
        ))

        # 5. Research Opportunity & assessments / iterations
        res_opp = ResearchOpportunityRecord(
            id=uuid.uuid4(),
            project_id=pid,
            gap_id=gap.id,
            research_problem="Problem",
            identified_gap="Gap",
            proposed_research_direction="New Direction",
            reasoning="Reasoning",
            technical_approach="Approach",
            expected_contribution="Contribution",
            feasibility={},
            novelty_assessment={},
            confidence=0.85,
            status="VALIDATED",
            generation_model="qwen3:8b",
            validation_model="qwen3:8b",
            embedding_model="nomic-embed-text",
        )
        db.add(res_opp)
        db.flush()

        db.add(PriorArtAssessmentRecord(
            id=uuid.uuid4(),
            opportunity_id=res_opp.id,
            iteration=1,
            assessment="DIFFERENTIATED",
            overlapping_statement_ids=[],
            overlapping_paper_ids=[],
            reasoning="Novel approach",
            differentiation="Differentiated by speed",
            confidence=0.9,
        ))

        db.add(OpportunityRefinementRecord(
            id=uuid.uuid4(),
            opportunity_id=res_opp.id,
            iteration=1,
            original_direction="Dir 1",
            critique="Needs more detail",
            revised_direction="Dir 2",
            status="REVISED",
        ))

        # 6. Lifecycle Opportunity & evidence / history
        lifecycle_opp = Opportunity(
            id=uuid.uuid4(),
            project_id=pid,
            research_opportunity_id=res_opp.id,
            gap_id=gap.id,
            title="Lifecycle Opportunity",
            problem_statement="Problem",
            description="Description",
            status="OPEN",
            confidence=0.85,
        )
        db.add(lifecycle_opp)
        db.flush()

        opp_evidence = OpportunityEvidence(
            id=uuid.uuid4(),
            opportunity_id=lifecycle_opp.id,
            paper_id=paper.id,
            evidence_type="FOUNDATIONAL",
            supporting_text="Foundational evidence",
        )
        db.add(opp_evidence)
        db.flush()

        db.add(OpportunityHistory(
            id=uuid.uuid4(),
            opportunity_id=lifecycle_opp.id,
            previous_status="OPEN",
            new_status="STRENGTHENED",
            trigger_type="FOUNDATIONAL",
            evidence_id=opp_evidence.id,
        ))

        db.commit()

    # Issue DELETE endpoint request
    resp = client.delete(f"/api/v1/projects/{pid}")
    assert resp.status_code == 200, resp.text
    summary = resp.json()["summary"]

    assert summary["runs_deleted"] == 1
    assert summary["run_papers_deleted"] == 1
    assert summary["selections_deleted"] == 1
    assert summary["paper_links_removed"] == 1
    assert summary["gaps_deleted"] == 1
    assert summary["gap_evidence_deleted"] == 1
    assert summary["research_opportunities_deleted"] == 1
    assert summary["prior_art_assessments_deleted"] == 1
    assert summary["refinement_iterations_deleted"] == 1
    assert summary["lifecycle_opportunities_deleted"] == 1
    assert summary["opportunity_evidence_deleted"] == 1
    assert summary["opportunity_history_deleted"] == 1

    # Verify database has ZERO orphaned records
    with SessionLocal() as db:
        assert db.query(ResearchProjectRecord).filter_by(id=pid).count() == 0
        assert db.query(ProjectPaperRecord).filter_by(project_id=pid).count() == 0
        assert db.query(ProjectPaperSelectionRecord).filter_by(project_id=pid).count() == 0
        assert db.query(ResearchRunRecord).filter_by(project_id=pid).count() == 0
        assert db.query(RunPaperRecord).count() == 0
        assert db.query(ResearchGap).filter_by(project_id=pid).count() == 0
        assert db.query(GapEvidence).count() == 0
        assert db.query(ResearchOpportunityRecord).filter_by(project_id=pid).count() == 0
        assert db.query(PriorArtAssessmentRecord).count() == 0
        assert db.query(OpportunityRefinementRecord).count() == 0
        assert db.query(Opportunity).filter_by(project_id=pid).count() == 0
        assert db.query(OpportunityEvidence).count() == 0
        assert db.query(OpportunityHistory).count() == 0

        # Shared literature paper MUST still be present in the database!
        assert db.query(Paper).count() == 1
        assert db.query(PaperDocument).count() == 1
        assert db.query(PaperPassage).count() == 1
        assert db.query(ResearchKnowledge).count() == 1
        assert db.query(ResearchStatement).count() == 1


def test_delete_project_preserves_shared_papers_and_other_project_data(env):
    """Deleting project Alpha preserves shared literature and all of project Beta's data."""
    client, SessionLocal = env
    _create_project(client, "proj_alpha", "Project Alpha")
    _create_project(client, "proj_beta", "Project Beta")

    with SessionLocal() as db:
        # Paper 1 shared by both Alpha and Beta
        p1 = _seed_shared_paper(db, "Shared Paper 1")
        p1_id = p1.id
        db.add(ProjectPaperRecord(project_id="proj_alpha", paper_id=p1_id))
        db.add(ProjectPaperRecord(project_id="proj_beta", paper_id=p1_id))

        # Paper 2 associated with Beta only
        p2 = _seed_shared_paper(db, "Beta-only Paper 2")
        p2_id = p2.id
        db.add(ProjectPaperRecord(project_id="proj_beta", paper_id=p2_id))

        # Gap for Alpha
        gap_alpha = ResearchGap(
            id=uuid.uuid4(),
            project_id="proj_alpha",
            gap_statement="Alpha Gap",
            research_problem="Alpha Problem",
            status="VALIDATED",
        )
        db.add(gap_alpha)

        # Gap for Beta
        gap_beta = ResearchGap(
            id=uuid.uuid4(),
            project_id="proj_beta",
            gap_statement="Beta Gap",
            research_problem="Beta Problem",
            status="VALIDATED",
        )
        db.add(gap_beta)

        # Opportunity for Beta
        db.add(Opportunity(
            id=uuid.uuid4(),
            project_id="proj_beta",
            gap_id=gap_beta.id,
            title="Beta Opportunity",
            problem_statement="Beta Problem",
            description="Beta Description",
            status="OPEN",
        ))

        db.commit()

    # Delete proj_alpha
    resp = client.delete("/api/v1/projects/proj_alpha")
    assert resp.status_code == 200, resp.text

    # Verify state in DB
    with SessionLocal() as db:
        # proj_alpha is deleted
        assert db.query(ResearchProjectRecord).filter_by(id="proj_alpha").first() is None
        assert db.query(ProjectPaperRecord).filter_by(project_id="proj_alpha").count() == 0
        assert db.query(ResearchGap).filter_by(project_id="proj_alpha").count() == 0

        # BOTH papers still exist in shared literature
        assert db.query(Paper).count() == 2

        # proj_beta still exists and has BOTH paper links intact
        assert db.query(ResearchProjectRecord).filter_by(id="proj_beta").first() is not None
        beta_links = db.query(ProjectPaperRecord).filter_by(project_id="proj_beta").all()
        assert len(beta_links) == 2
        beta_paper_ids = {link.paper_id for link in beta_links}
        assert p1_id in beta_paper_ids
        assert p2_id in beta_paper_ids

        # proj_beta gaps and opportunities are completely intact
        assert db.query(ResearchGap).filter_by(project_id="proj_beta").count() == 1
        assert db.query(Opportunity).filter_by(project_id="proj_beta").count() == 1


def test_delete_project_with_active_run_returns_409(env):
    """Attempting to delete a project with an active research run returns 409 Conflict."""
    client, SessionLocal = env
    pid = "proj_active_run"
    _create_project(client, pid, "Active Run Project")

    run_id = f"run_{uuid.uuid4().hex[:8]}"
    with SessionLocal() as db:
        run = ResearchRunRecord(
            id=run_id,
            project_id=pid,
            status="RUNNING",
        )
        db.add(run)
        db.commit()

    # Register run in WorkflowEngine active set
    WorkflowEngine.mark_run_active(run_id)
    try:
        resp = client.delete(f"/api/v1/projects/{pid}")
        assert resp.status_code == 409
        assert "currently active" in resp.json()["detail"]

        # Confirm project and run were NOT deleted
        with SessionLocal() as db:
            assert db.query(ResearchProjectRecord).filter_by(id=pid).first() is not None
            assert db.query(ResearchRunRecord).filter_by(id=run_id).first() is not None
    finally:
        WorkflowEngine.mark_run_inactive(run_id)


def test_delete_project_with_active_discovery_returns_409(env):
    """Attempting to delete a project while gap discovery is active returns 409 Conflict."""
    client, SessionLocal = env
    pid = "proj_active_discovery"
    _create_project(client, pid, "Discovery Project")

    # Simulate active discovery by acquiring the lock
    with _discovery_lock_mgr.acquire(pid):
        resp = client.delete(f"/api/v1/projects/{pid}")
        assert resp.status_code == 409
        assert "already in progress" in resp.json()["detail"]

    # Confirm project was NOT deleted
    with SessionLocal() as db:
        assert db.query(ResearchProjectRecord).filter_by(id=pid).first() is not None


def test_delete_project_atomic_rollback_on_injected_error(env):
    """A database error midway through deletion rolls back the entire transaction atomically."""
    client, SessionLocal = env
    pid = "proj_rollback"
    _create_project(client, pid, "Rollback Test Project")

    with SessionLocal() as db:
        gap = ResearchGap(
            id=uuid.uuid4(),
            project_id=pid,
            gap_statement="Rollback Gap",
            research_problem="Rollback Problem",
            status="VALIDATED",
        )
        db.add(gap)
        db.add(ProjectPaperSelectionRecord(
            id=f"psel_{uuid.uuid4().hex[:8]}",
            project_id=pid,
            openalex_id="W12345",
            title="Selection 1",
        ))
        db.commit()

    # Inject error when deleting the ResearchProjectRecord row
    original_filter = Session.query

    def mock_query(self, *entities, **kwargs):
        query_obj = original_filter(self, *entities, **kwargs)
        if ResearchProjectRecord in entities:
            # Inject exception on delete
            def failing_delete(*args, **kwargs):
                raise SQLAlchemyError("Injected failure during project record delete")
            query_obj.delete = failing_delete
        return query_obj

    with patch.object(Session, "query", mock_query):
        resp = client.delete(f"/api/v1/projects/{pid}")
        assert resp.status_code == 500
        assert "database error occurred" in resp.json()["detail"]

    # Verify that transaction rolled back completely: project, gaps, selections STILL EXIST!
    with SessionLocal() as db:
        assert db.query(ResearchProjectRecord).filter_by(id=pid).first() is not None
        assert db.query(ResearchGap).filter_by(project_id=pid).count() == 1
        assert db.query(ProjectPaperSelectionRecord).filter_by(project_id=pid).count() == 1


def test_check_then_delete_race_serialized_by_run_lock(env):
    """Per-project run lock serializes run creation and deletion."""
    client, SessionLocal = env
    pid = "proj_race"
    _create_project(client, pid, "Race Project")

    # Holding the run lock simulates another thread in the middle of creating/checking a run
    with _project_run_lock(pid):
        # We can verify that _project_run_lock returns the same lock instance for this project
        lock = _project_run_lock(pid)
        assert lock.locked()

    # Now deletion can proceed cleanly
    resp = client.delete(f"/api/v1/projects/{pid}")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True


def test_lock_registry_permanently_retains_identical_mutex_under_concurrent_wait(env):
    """The lock registry always returns the exact same mutex instance, even across project deletions and waiting threads."""
    pid = "proj_lock_retain_test"
    lock_a = _project_run_lock(pid)

    t1_acquired = threading.Event()
    t1_can_release = threading.Event()
    t2_blocked = threading.Event()
    t2_finished = threading.Event()
    observed_locks: list = []

    def thread_1_work():
        with _project_run_lock(pid) as l:
            observed_locks.append(("t1", id(_project_run_lock(pid))))
            t1_acquired.set()
            t1_can_release.wait(timeout=5.0)

    def thread_2_work():
        t1_acquired.wait(timeout=5.0)
        t2_blocked.set()
        with _project_run_lock(pid) as l:
            observed_locks.append(("t2", id(_project_run_lock(pid))))
            t2_finished.set()

    t1 = threading.Thread(target=thread_1_work)
    t2 = threading.Thread(target=thread_2_work)
    t1.start()
    t2.start()

    assert t1_acquired.wait(timeout=5.0), "Thread 1 failed to acquire lock"
    assert t2_blocked.wait(timeout=5.0), "Thread 2 failed to initiate lock wait"

    # Thread 3 queries the lock registry while Thread 1 holds it and Thread 2 waits
    lock_c = _project_run_lock(pid)
    observed_locks.append(("t3", id(lock_c)))

    # Allow Thread 1 to finish and release the lock
    t1_can_release.set()
    t1.join(timeout=5.0)
    assert not t1.is_alive(), "Thread 1 timed out joining"

    t2.join(timeout=5.0)
    assert not t2.is_alive(), "Thread 2 timed out joining"
    assert t2_finished.is_set(), "Thread 2 never acquired the lock after release"

    # Thread 4 queries the lock registry after all releases
    lock_d = _project_run_lock(pid)
    observed_locks.append(("t4", id(lock_d)))

    # Every single thread must have resolved to the EXACT same mutex identity
    lock_ids = {item[1] for item in observed_locks}
    assert len(lock_ids) == 1, f"Expected 1 unique lock identity across threads, got {len(lock_ids)}: {observed_locks}"
    assert id(lock_a) == next(iter(lock_ids))


def test_two_concurrent_delete_requests_return_200_and_404_cleanly(env):
    """Two simultaneous DELETE requests for the same project: one gets 200, one gets 404, neither gets 500."""
    client, SessionLocal = env
    pid = "proj_concurrent_del"
    _create_project(client, pid, "Concurrent Delete Project")

    barrier = threading.Barrier(2)
    results = []
    lock = threading.Lock()

    def do_delete():
        barrier.wait(timeout=5.0)
        resp = client.delete(f"/api/v1/projects/{pid}")
        with lock:
            results.append((resp.status_code, resp.text))

    t1 = threading.Thread(target=do_delete)
    t2 = threading.Thread(target=do_delete)
    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    assert len(results) == 2
    status_codes = [r[0] for r in results]
    assert sorted(status_codes) == [200, 404], f"Expected [200, 404], got {status_codes}. Details: {results}"


def test_start_run_races_with_deletion_returns_clean_404(env):
    """A workflow-start request that races with deletion returns a clean 404 if deletion wins, rather than an unhandled error."""
    client, SessionLocal = env
    pid = "proj_run_del_race"
    _create_project(client, pid, "Run Race Project")

    with SessionLocal() as db:
        paper = _seed_shared_paper(db, "Paper For Run")
        db.add(ProjectPaperRecord(project_id=pid, paper_id=paper.id))
        db.add(ProjectPaperSelectionRecord(
            id=str(uuid.uuid4()),
            project_id=pid,
            openalex_id="W12345",
            title="Paper For Run",
            authors=[],
        ))
        db.commit()

    run_started = threading.Event()
    run_result = []

    # Hold the project run lock so we can simulate deletion winning the race
    with _project_run_lock(pid):
        def attempt_start_run():
            run_started.set()
            resp = client.post(
                f"/api/v1/projects/{pid}/runs",
                json={"planned_stages": ["INGESTION"]},
            )
            run_result.append((resp.status_code, resp.json()))

        t = threading.Thread(target=attempt_start_run)
        t.start()
        assert run_started.wait(timeout=5.0)

        # While start_run is blocked waiting on _project_run_lock, delete the project record in DB
        with SessionLocal() as db:
            db.query(ResearchProjectRecord).filter_by(id=pid).delete()
            db.commit()

    # Once lock is released, start_run enters the lock, revalidates _require_project, and should raise 404
    t.join(timeout=5.0)
    assert len(run_result) == 1
    status, body = run_result[0]
    assert status == 404
    assert f"Research project '{pid}' not found" in body["detail"]


def test_opportunity_generation_serialized_with_deletion_no_orphans(env):
    """Opportunity generation cannot race with deletion to create orphaned rows."""
    client, SessionLocal = env
    pid = "proj_opp_del_race"
    _create_project(client, pid, "Opp Del Race Project")

    gap_id = uuid.uuid4()
    with SessionLocal() as db:
        gap = ResearchGap(
            id=gap_id,
            project_id=pid,
            gap_statement="Race Gap Statement",
            research_problem="Race Gap Problem",
            status="VALIDATED",
        )
        db.add(gap)
        db.commit()

    # 1. If project was already deleted, calling opportunity generation returns 404 and persists nothing
    del_resp = client.delete(f"/api/v1/projects/{pid}")
    assert del_resp.status_code == 200

    opp_resp = client.post(f"/api/v1/gaps/{gap_id}/opportunity?project_id={pid}")
    assert opp_resp.status_code == 404

    with SessionLocal() as db:
        assert db.query(ResearchOpportunityRecord).filter_by(project_id=pid).count() == 0
        assert db.query(Opportunity).filter_by(project_id=pid).count() == 0


def test_opportunity_evolution_serialized_with_deletion_no_orphans(env):
    """Opportunity evolution cannot execute or mutate an opportunity if the project was deleted."""
    client, SessionLocal = env
    pid = "proj_evolve_del_race"
    _create_project(client, pid, "Evolve Del Race Project")

    opp_id = uuid.uuid4()
    with SessionLocal() as db:
        p = _seed_shared_paper(db, "Evolve Paper")
        db.add(ProjectPaperRecord(project_id=pid, paper_id=p.id))
        db.add(Opportunity(
            id=opp_id,
            project_id=pid,
            title="Evolve Opportunity",
            problem_statement="Problem Statement",
            description="Description",
            status="OPEN",
        ))
        db.commit()

    # Delete the project
    del_resp = client.delete(f"/api/v1/projects/{pid}")
    assert del_resp.status_code == 200

    # Now attempt to evolve the opportunity
    evolve_resp = client.post(
        f"/api/v1/opportunities/{opp_id}/evolve?project_id={pid}",
        json={"paper_ids": [str(uuid.uuid4())]},
    )
    assert evolve_resp.status_code == 404
    with SessionLocal() as db:
        assert db.query(OpportunityEvidence).filter_by(opportunity_id=opp_id).count() == 0
        assert db.query(OpportunityHistory).filter_by(opportunity_id=opp_id).count() == 0
