"""Regression tests for project scoping and workflow gating (2026-10-23 reliability audit).

Covers BUG-003 / BUG-005 / BUG-006 / BUG-008:
1. Projects are explicit server records: GET never creates one; POST is idempotent per ID.
2. Selections, association, preflight and runs reject unknown projects instead of fabricating them.
3. Adding papers: new paper -> associated; existing global paper -> associated with the second
   project; repeat -> idempotent "already in project"; other projects never see it.
4. Gap discovery during a run only reads the project's own papers, and tags gaps with project/run.
5. Runs never contain OPPORTUNITY_GENERATION; they pause at AWAITING_REVIEW only when gaps exist;
   a failed gap discovery fails the run; retries never repeat a finished gap discovery.
6. Ingestion records the real OpenAlex failure instead of "metadata not found".
7. Strict per-ID scoping: records without a project are outside every project scope.
"""

from __future__ import annotations

import datetime
import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ideagen.api.main import app
from ideagen.db import get_db
from ideagen.db.models import (
    Base,
    Paper,
    ProjectPaperRecord,
    ResearchGap,
    ResearchKnowledge,
    ResearchProjectRecord,
    ResearchRunRecord,
    ResearchStatement,
    RunPaperRecord,
)
from ideagen.db.session import configure_sqlite_foreign_keys
from ideagen.discovery.models import GapStatus, ResearchGapResponseDTO
from ideagen.discovery.pipeline import SemanticGapDiscoveryPipeline
from ideagen.ingestion.search import OpenAlexSearchError, PaperSearchResult
from ideagen.pipeline.workflow_engine import ResearchWorkflowEngine, normalize_planned_stages


def _work(n: int) -> dict:
    """A minimal OpenAlex work without any open-access PDF (ingests as metadata only, no network)."""
    return {
        "id": f"https://openalex.org/W90000{n}",
        "doi": f"https://doi.org/10.5555/audit.{n}",
        "title": f"Audit Paper {n}",
        "publication_year": 2020,
        "publication_date": "2020-03-01",
        "authorships": [{"author": {"display_name": "A. Researcher"}}],
        "abstract_inverted_index": {"Adaptive": [0], "detection": [1]},
        "open_access": {"is_oa": False},
        "best_oa_location": None,
        "primary_location": None,
        "referenced_works": [],
        "cited_by_count": 0,
    }


def _search_result(n: int) -> PaperSearchResult:
    work = _work(n)
    return PaperSearchResult(
        id=f"W90000{n}", source="OPENALEX", title=work["title"], authors=[], abstract="", raw_metadata=work
    )


@pytest.fixture
def env():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    configure_sqlite_foreign_keys(engine)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    yield client, SessionLocal
    app.dependency_overrides.clear()


def _create(client, pid: str, title: str = "Project", cutoff: str | None = None):
    body = {"id": pid, "title": title, "topic": f"{title} topic"}
    if cutoff:
        body["temporal_cutoff"] = cutoff
    resp = client.post("/api/v1/projects", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _ingest(client, n: int, project_id: str):
    with patch("ideagen.api.main.OpenAlexSearchProvider.search", return_value=[_search_result(n)]):
        resp = client.post("/api/v1/papers/ingest/openalex", json={"openalex_id": f"W90000{n}", "project_id": project_id})
    assert resp.status_code == 200, resp.text
    return resp.json()


# ── 1-2. Explicit projects ──────────────────────────────────────────────────

def test_get_unknown_project_is_404_and_creates_nothing(env):
    client, SessionLocal = env
    assert client.get("/api/v1/projects/proj_unknown").status_code == 404
    with SessionLocal() as db:
        assert db.query(ResearchProjectRecord).count() == 0


def test_create_project_is_idempotent_and_keeps_fields(env):
    client, _ = env
    first = _create(client, "proj_cyber", "Adaptive AI for Zero-Day Attack Detection", cutoff="2024-12-31")
    assert first["temporal_cutoff"] == "2024-12-31"
    assert first["topic"] == "Adaptive AI for Zero-Day Attack Detection topic"
    again = client.post("/api/v1/projects", json={"id": "proj_cyber", "title": "Renamed"}).json()
    assert again["title"] == "Adaptive AI for Zero-Day Attack Detection"
    assert [p["id"] for p in client.get("/api/v1/projects").json()] == ["proj_cyber"]


def test_create_project_fills_only_unset_fields(env):
    """A project registered by an older backend (no topic/cutoff) gets them from the client, once."""
    client, SessionLocal = env
    with SessionLocal() as db:
        db.add(ResearchProjectRecord(id="proj_legacy", name="Zero-Day"))
        db.commit()
    body = {"id": "proj_legacy", "title": "Other", "topic": "Adaptive detection", "temporal_cutoff": "2024-12-31"}
    filled = client.post("/api/v1/projects", json=body).json()
    assert filled["title"] == "Zero-Day"
    assert filled["topic"] == "Adaptive detection" and filled["temporal_cutoff"] == "2024-12-31"
    again = client.post("/api/v1/projects", json={**body, "topic": "Changed", "temporal_cutoff": "2020-01-01"}).json()
    assert again["topic"] == "Adaptive detection" and again["temporal_cutoff"] == "2024-12-31"


def test_create_project_rejects_unsafe_id(env):
    client, _ = env
    assert client.post("/api/v1/projects", json={"id": "../etc", "title": "x"}).status_code == 422


@pytest.mark.parametrize(
    "method,path",
    [
        ("post", "/api/v1/projects/ghost/selections"),
        ("get", "/api/v1/projects/ghost/preflight"),
        ("post", "/api/v1/projects/ghost/runs"),
        ("post", f"/api/v1/projects/ghost/papers/{uuid.uuid4()}"),
        ("post", "/api/v1/gaps/discover?project_id=ghost"),
    ],
)
def test_project_scoped_endpoints_reject_unknown_project(env, method, path):
    client, SessionLocal = env
    kwargs = {"json": {"selections": [{"openalex_id": "W1", "title": "t"}]}} if path.endswith("selections") else {}
    resp = getattr(client, method)(path, **kwargs)
    assert resp.status_code == 404, resp.text
    with SessionLocal() as db:
        assert db.query(ResearchProjectRecord).count() == 0


def test_ingest_into_unknown_project_is_rejected_before_persisting(env):
    client, SessionLocal = env
    with patch("ideagen.api.main.OpenAlexSearchProvider.search", return_value=[_search_result(1)]):
        resp = client.post("/api/v1/papers/ingest/openalex", json={"openalex_id": "W900001", "project_id": "ghost"})
    assert resp.status_code == 404
    with SessionLocal() as db:
        assert db.query(Paper).count() == 0


# ── 3. Paper association and isolation ──────────────────────────────────────

def test_add_paper_association_duplicate_and_isolation(env):
    client, SessionLocal = env
    _create(client, "proj_a", "Cyber")
    _create(client, "proj_b", "RAG")

    first = _ingest(client, 1, "proj_a")
    assert first["status"] == "NO_PDF"
    assert first["project_associated"] is True and first["already_in_project"] is False

    again = _ingest(client, 1, "proj_a")
    assert again["status"] == "SKIPPED_DUPLICATE"
    assert again["already_in_project"] is True
    assert again["paper_id"] == first["paper_id"]

    # Project B has not received the paper.
    assert client.get("/api/v1/papers", params={"project_id": "proj_b"}).json() == []
    assert client.get("/api/v1/stats", params={"project_id": "proj_b"}).json()["total_papers"] == 0

    # Adding the existing global paper to project B associates it without a new paper row.
    to_b = _ingest(client, 1, "proj_b")
    assert to_b["status"] == "SKIPPED_DUPLICATE"
    assert to_b["project_associated"] is True and to_b["already_in_project"] is False

    a_papers = client.get("/api/v1/papers", params={"project_id": "proj_a"}).json()
    b_papers = client.get("/api/v1/papers", params={"project_id": "proj_b"}).json()
    assert [p["id"] for p in a_papers] == [first["paper_id"]] == [p["id"] for p in b_papers]
    with SessionLocal() as db:
        assert db.query(Paper).count() == 1
        assert db.query(ProjectPaperRecord).count() == 2

    projects = {p["id"]: p for p in client.get("/api/v1/projects").json()}
    assert projects["proj_a"]["paper_count"] == 1
    assert projects["proj_a"]["selection_count"] == 0


def test_associate_endpoint_is_idempotent(env):
    client, _ = env
    _create(client, "proj_a")
    _create(client, "proj_b")
    paper_id = _ingest(client, 2, "proj_a")["paper_id"]
    first = client.post(f"/api/v1/projects/proj_b/papers/{paper_id}").json()
    second = client.post(f"/api/v1/projects/proj_b/papers/{paper_id}").json()
    assert first["already_associated"] is False
    assert second["already_associated"] is True
    assert len(client.get("/api/v1/papers", params={"project_id": "proj_b"}).json()) == 1


# ── 4. Project-scoped gap discovery ─────────────────────────────────────────

def _seed_statement(db, project_id: str, label: str) -> uuid.UUID:
    paper = Paper(
        id=uuid.uuid4(), source_type="OPENALEX_API", source_id=f"W-{label}", title=f"Paper {label}",
        abstract="", authors=[], referenced_works=[], raw_metadata={}, batch_id="b",
    )
    db.add(paper)
    db.flush()
    db.add(ProjectPaperRecord(project_id=project_id, paper_id=paper.id))
    knowledge = ResearchKnowledge(paper_id=paper.id, model_name="test", extracted_at=datetime.datetime.now(datetime.timezone.utc))
    db.add(knowledge)
    db.flush()
    db.add(ResearchStatement(
        paper_id=paper.id, knowledge_id=knowledge.id, statement_text=f"Limitation of {label}",
        statement_type="LIMITATION", confidence=0.9,
    ))
    db.commit()
    return paper.id


class _FakeEmbedder:
    dimension = 4
    model_name = "fake"

    def embed_texts(self, texts):
        import numpy as np

        return np.ones((len(texts), 4), dtype="float32")


def test_discovery_pipeline_only_reads_scoped_papers(env, tmp_path):
    _, SessionLocal = env
    with SessionLocal() as db:
        db.add_all([ResearchProjectRecord(id="proj_a", name="A"), ResearchProjectRecord(id="proj_b", name="B")])
        db.commit()
        paper_a = _seed_statement(db, "proj_a", "cyber")
        _seed_statement(db, "proj_b", "rag")

        from ideagen.db.persistence import SQLAlchemyPersistenceLayer

        pipeline = SemanticGapDiscoveryPipeline(
            persistence=SQLAlchemyPersistenceLayer(db), embedding_provider=_FakeEmbedder(),
            llm_provider=object(), index_dir=tmp_path,
        )
        seen = {}

        def capture(statements, vectors):
            seen["papers"] = {s.paper_id for s in statements}
            return []

        with patch.object(pipeline.grouper, "group_statements", side_effect=capture):
            assert pipeline.discover_gaps(paper_ids={paper_a}) == []
        assert seen["papers"] == {paper_a}
        # Scoped runs never overwrite the shared whole-corpus index.
        assert not (tmp_path / "index.faiss").exists()

        assert pipeline.discover_gaps(paper_ids=set()) == []


def _run_with_stage_states(db, project_id: str, stage_states: dict, status: str, papers=()):
    run = ResearchRunRecord(
        id=f"run_{uuid.uuid4().hex[:8]}", project_id=project_id, status=status, current_stage="COMPLETED",
        planned_stages=["INGESTION", "UNDERSTANDING", "GAP_DISCOVERY"], stage_states=stage_states, logs=[],
    )
    db.add(run)
    for p in papers:
        db.add(RunPaperRecord(run_id=run.id, **p))
    db.commit()
    return run


def test_run_gap_discovery_is_scoped_and_tags_gaps(env):
    _, SessionLocal = env
    with SessionLocal() as db:
        db.add_all([ResearchProjectRecord(id="proj_a", name="A"), ResearchProjectRecord(id="proj_b", name="B")])
        db.commit()
        paper_a = _seed_statement(db, "proj_a", "cyber")
        _seed_statement(db, "proj_b", "rag")
        run = _run_with_stage_states(db, "proj_a", {}, "RUNNING")
        gap = ResearchGap(id=uuid.uuid4(), gap_statement="g", research_problem="p", status="VALIDATED", evidence_count=1)
        db.add(gap)
        db.commit()
        dto = ResearchGapResponseDTO(
            id=gap.id, gap_statement="g", research_problem="p", status=GapStatus.VALIDATED, evidence_count=1,
            supporting_statement_ids=[], supporting_paper_ids=[], limitations=[], future_work=[],
            validation_reasoning="r", validation_confidence=0.9, created_at=gap.created_at, metadata={},
        )
        with patch("ideagen.pipeline.workflow_engine.SemanticGapDiscoveryPipeline") as Pipeline:
            Pipeline.return_value.discover_gaps.return_value = [dto]
            states: dict = {}
            ResearchWorkflowEngine._execute_stage_gap_discovery(db, run, states)
            kwargs = Pipeline.return_value.discover_gaps.call_args.kwargs
        assert kwargs["paper_ids"] == {paper_a}
        # Gaps are tagged with project and run by the pipeline as each one is saved.
        assert kwargs["project_id"] == "proj_a"
        assert kwargs["gap_metadata"] == {"run_id": run.id}
        assert run.status == "AWAITING_REVIEW"
        assert states["GAP_DISCOVERY"]["status"] == "COMPLETED"


# ── 5. Run gating, statuses and retries ─────────────────────────────────────

def test_planned_stages_validation():
    assert normalize_planned_stages(None) == ["INGESTION", "UNDERSTANDING", "GAP_DISCOVERY"]
    assert normalize_planned_stages(["gap_discovery", "INGESTION"]) == ["INGESTION", "GAP_DISCOVERY"]
    with pytest.raises(ValueError, match="OPPORTUNITY_GENERATION"):
        normalize_planned_stages(["INGESTION", "OPPORTUNITY_GENERATION"])
    with pytest.raises(ValueError, match="Unknown run stage"):
        normalize_planned_stages(["PUBLISH"])


def test_run_api_rejects_opportunity_stage_and_empty_project(env):
    client, _ = env
    _create(client, "proj_a")
    resp = client.post("/api/v1/projects/proj_a/runs", json={})
    assert resp.status_code == 400 and "Nothing to process" in resp.json()["detail"]
    _ingest(client, 3, "proj_a")
    resp = client.post("/api/v1/projects/proj_a/runs", json={"planned_stages": ["INGESTION", "OPPORTUNITY_GENERATION"]})
    assert resp.status_code == 400 and "OPPORTUNITY_GENERATION" in resp.json()["detail"]


def _execute(db, run_id):
    ResearchWorkflowEngine.execute_run(run_id, db_override=db)


def test_zero_gaps_completes_without_review(env):
    _, SessionLocal = env
    with SessionLocal() as db:
        db.add(ResearchProjectRecord(id="proj_a", name="A"))
        db.commit()
        run = _run_with_stage_states(db, "proj_a", {}, "PENDING")
        with patch("ideagen.pipeline.workflow_engine.SemanticGapDiscoveryPipeline") as Pipeline:
            Pipeline.return_value.discover_gaps.return_value = []
            _execute(db, run.id)
        db.refresh(run)
        assert run.status == "COMPLETED"
        assert "OPPORTUNITY_GENERATION" not in run.stage_states


def test_failed_gap_discovery_fails_the_run(env):
    _, SessionLocal = env
    with SessionLocal() as db:
        db.add(ResearchProjectRecord(id="proj_a", name="A"))
        db.commit()
        run = _run_with_stage_states(db, "proj_a", {}, "PENDING")
        with patch("ideagen.pipeline.workflow_engine.SemanticGapDiscoveryPipeline") as Pipeline:
            Pipeline.return_value.discover_gaps.side_effect = RuntimeError("Ollama unreachable")
            _execute(db, run.id)
        db.refresh(run)
        assert run.status == "FAILED"
        assert "Ollama unreachable" in run.error_message
        assert run.stage_states["GAP_DISCOVERY"]["status"] == "FAILED"
        assert run.summary["failed_stages"] == ["GAP_DISCOVERY"]
        # The failure reason is visible in the stage list the UI renders.
        stage = next(s for s in run.stage_progress["stages"] if s["stage_id"] == "GAP_DISCOVERY")
        assert "Ollama unreachable" in stage["message"]


def test_retry_rejects_runs_with_nothing_to_retry(env):
    _, SessionLocal = env
    with SessionLocal() as db:
        db.add(ResearchProjectRecord(id="proj_a", name="A"))
        db.commit()
        done = {s: {"status": "COMPLETED"} for s in ("INGESTION", "UNDERSTANDING", "GAP_DISCOVERY")}
        run = _run_with_stage_states(db, "proj_a", done, "AWAITING_REVIEW")
        with pytest.raises(ValueError, match="nothing to retry"):
            ResearchWorkflowEngine(db).retry_run(run.id, db_override=db)


def test_retry_never_repeats_finished_gap_discovery(env):
    _, SessionLocal = env
    with SessionLocal() as db:
        db.add(ResearchProjectRecord(id="proj_a", name="A"))
        db.commit()
        states = {
            "INGESTION": {"status": "PARTIAL_SUCCESS"},
            "UNDERSTANDING": {"status": "COMPLETED"},
            "GAP_DISCOVERY": {"status": "COMPLETED"},
        }
        run = _run_with_stage_states(
            db, "proj_a", states, "AWAITING_REVIEW",
            papers=[{"openalex_id": "W1", "title": "t", "status": "FAILED"}],
        )
        with patch.object(ResearchWorkflowEngine, "_execute_stage_ingestion") as ingest, \
             patch.object(ResearchWorkflowEngine, "_execute_stage_understanding") as understand, \
             patch.object(ResearchWorkflowEngine, "_execute_stage_gap_discovery") as discover:
            ResearchWorkflowEngine(db).retry_run(run.id, db_override=db)
        assert ingest.called and understand.called
        assert not discover.called


# ── 6. Real ingestion failure reasons ───────────────────────────────────────

def test_ingestion_records_real_openalex_failure(env):
    _, SessionLocal = env
    with SessionLocal() as db:
        db.add(ResearchProjectRecord(id="proj_a", name="A"))
        db.commit()
        run = _run_with_stage_states(db, "proj_a", {}, "RUNNING", papers=[{"openalex_id": "W123", "title": "t"}])
        err = OpenAlexSearchError("Network proxy or policy blocked request to api.openalex.org", error_category="proxy_policy_block")
        with patch("ideagen.pipeline.workflow_engine.OpenAlexSearchProvider.search", side_effect=err):
            states: dict = {}
            ResearchWorkflowEngine._execute_stage_ingestion(db, run, states)
        paper = run.papers[0]
        assert paper.status == "FAILED"
        assert "policy blocked" in paper.error_message
        assert "not found" not in paper.error_message
        assert states["INGESTION"]["status"] == "FAILED"


# ── 7. Strict per-ID scoping ────────────────────────────────────────────────

def test_unscoped_gap_is_outside_every_project(env):
    client, SessionLocal = env
    _create(client, "proj_a")
    with SessionLocal() as db:
        gap = ResearchGap(id=uuid.uuid4(), gap_statement="g", research_problem="p", status="VALIDATED", evidence_count=0)
        db.add(gap)
        db.commit()
        gap_id = gap.id
    assert client.get(f"/api/v1/gaps/{gap_id}").status_code == 200
    assert client.get(f"/api/v1/gaps/{gap_id}", params={"project_id": "proj_a"}).status_code == 404
    resp = client.post(f"/api/v1/gaps/{gap_id}/opportunity", params={"project_id": "proj_a"})
    assert resp.status_code == 404


def test_understanding_covers_project_corpus_papers_only(env):
    """Papers added with Paper Search "Add" are corpus papers, not run papers; the run must still
    extract their statements, and must not touch other projects' papers."""
    from ideagen.db.models import PaperDocument, PaperPassage

    _, SessionLocal = env
    with SessionLocal() as db:
        db.add_all([ResearchProjectRecord(id="proj_a", name="A"), ResearchProjectRecord(id="proj_b", name="B")])
        db.commit()
        ids = {}
        for project_id in ("proj_a", "proj_b"):
            paper = Paper(id=uuid.uuid4(), source_type="OPENALEX_API", source_id=f"W-{project_id}", title=project_id,
                          abstract="", authors=[], referenced_works=[], raw_metadata={}, batch_id="b")
            db.add(paper)
            db.flush()
            db.add(ProjectPaperRecord(project_id=project_id, paper_id=paper.id))
            doc = PaperDocument(paper_id=paper.id, full_text="t", extraction_method="m", extraction_version="v",
                                extracted_at=datetime.datetime.now(datetime.timezone.utc), content_hash=f"h-{project_id}")
            db.add(doc)
            db.flush()
            db.add(PaperPassage(paper_id=paper.id, document_id=doc.id, passage_index=0, page_start=1, page_end=1, text="passage"))
            ids[project_id] = paper.id
        db.commit()
        run = _run_with_stage_states(db, "proj_a", {}, "RUNNING")
        with patch("ideagen.pipeline.workflow_engine.PaperUnderstandingPipeline") as Understanding:
            # A real DTO instance, so reading an attribute it does not have fails like in production.
            from ideagen.understanding.models import ResearchKnowledgeResponseDTO

            Understanding.return_value.understand_paper.return_value = ResearchKnowledgeResponseDTO.model_construct(
                statement_count=2, extraction_status="SUCCESS", error_message=None
            )
            states: dict = {}
            ResearchWorkflowEngine._execute_stage_understanding(db, run, states)
            called_with = [c.args[0] for c in Understanding.return_value.understand_paper.call_args_list]
        assert called_with == [ids["proj_a"]]
        assert states["UNDERSTANDING"]["status"] == "COMPLETED"
        assert states["UNDERSTANDING"]["details"]["statements_extracted"] == 2


def test_failed_upstream_stage_with_gaps_is_partial_success(env):
    """Gaps discovered despite a failed extraction stage: reviewable and retryable, not hidden as FAILED."""
    _, SessionLocal = env
    with SessionLocal() as db:
        db.add(ResearchProjectRecord(id="proj_a", name="A"))
        db.commit()
        run = _run_with_stage_states(db, "proj_a", {}, "RUNNING")
        run.status = "AWAITING_REVIEW"
        states = {
            "INGESTION": {"status": "COMPLETED"},
            "UNDERSTANDING": {"status": "FAILED"},
            "GAP_DISCOVERY": {"status": "COMPLETED", "details": {"gaps_discovered": 2}},
        }
        ResearchWorkflowEngine._finalize_run_summary(db, run, states)
        assert run.status == "PARTIAL_SUCCESS"
        assert run.summary["failed_stages"] == ["UNDERSTANDING"]
        assert any("Review discovered research gaps" in a for a in run.summary["next_actions"])
        assert any("Retry" in a for a in run.summary["next_actions"])


def test_retry_with_preserved_gaps_returns_to_review(env):
    _, SessionLocal = env
    with SessionLocal() as db:
        db.add(ResearchProjectRecord(id="proj_a", name="A"))
        db.commit()
        states = {
            "INGESTION": {"status": "COMPLETED"},
            "UNDERSTANDING": {"status": "FAILED"},
            "GAP_DISCOVERY": {"status": "COMPLETED", "details": {"gaps_discovered": 2}},
        }
        run = _run_with_stage_states(db, "proj_a", states, "PARTIAL_SUCCESS")

        def understanding_ok(session, r, st):
            st["UNDERSTANDING"] = {"status": "COMPLETED", "details": {}}

        with patch.object(ResearchWorkflowEngine, "_execute_stage_understanding", side_effect=understanding_ok), \
             patch.object(ResearchWorkflowEngine, "_execute_stage_gap_discovery") as discover:
            ResearchWorkflowEngine(db).retry_run(run.id, db_override=db)
        db.refresh(run)
        assert not discover.called
        assert run.status == "AWAITING_REVIEW"
        assert not any("No research gaps were discovered" in a for a in run.summary["next_actions"])


def _paper_with_passage(db, project_id: str, label: str):
    from ideagen.db.models import PaperDocument, PaperPassage

    paper = Paper(id=uuid.uuid4(), source_type="OPENALEX_API", source_id=f"W-{label}", title=label,
                  abstract="", authors=[], referenced_works=[], raw_metadata={}, batch_id="b")
    db.add(paper)
    db.flush()
    db.add(ProjectPaperRecord(project_id=project_id, paper_id=paper.id))
    doc = PaperDocument(paper_id=paper.id, full_text="t", extraction_method="m", extraction_version="v",
                        extracted_at=datetime.datetime.now(datetime.timezone.utc), content_hash=f"h-{label}")
    db.add(doc)
    db.flush()
    db.add(PaperPassage(paper_id=paper.id, document_id=doc.id, passage_index=0, page_start=1, page_end=1, text="p"))
    db.commit()
    return paper.id


def test_failed_extraction_is_not_analysis_and_is_retried(env):
    """A recorded extraction failure (e.g. Ollama down) must not count as analyzed and must be retried."""
    client, SessionLocal = env
    _create(client, "proj_a")
    with SessionLocal() as db:
        paper_id = _paper_with_passage(db, "proj_a", "failed-once")
        db.add(ResearchKnowledge(paper_id=paper_id, model_name="m", extraction_status="FAILED",
                                 error_message="Ollama connection error", extracted_at=datetime.datetime.now(datetime.timezone.utc)))
        db.commit()

    paper = client.get("/api/v1/papers", params={"project_id": "proj_a"}).json()[0]
    assert paper["knowledge_status"] == "EXTRACTION_FAILED"
    assert client.get("/api/v1/stats", params={"project_id": "proj_a"}).json()["analyzed_papers"] == 0

    with SessionLocal() as db:
        run = _run_with_stage_states(db, "proj_a", {}, "RUNNING")
        from ideagen.understanding.models import ResearchKnowledgeResponseDTO

        with patch("ideagen.pipeline.workflow_engine.PaperUnderstandingPipeline") as Understanding:
            Understanding.return_value.understand_paper.return_value = ResearchKnowledgeResponseDTO.model_construct(
                statement_count=3, extraction_status="SUCCESS", error_message=None
            )
            states: dict = {}
            ResearchWorkflowEngine._execute_stage_understanding(db, run, states)
            assert [c.args[0] for c in Understanding.return_value.understand_paper.call_args_list] == [paper_id]


def test_understand_endpoint_reports_failed_extraction(env):
    client, SessionLocal = env
    _create(client, "proj_a")
    with SessionLocal() as db:
        paper_id = _paper_with_passage(db, "proj_a", "provider-down")
    from ideagen.understanding.models import ResearchKnowledgeResponseDTO

    failed = ResearchKnowledgeResponseDTO.model_construct(statement_count=0, extraction_status="FAILED",
                                                          error_message="Ollama connection error at http://127.0.0.1:9")
    with patch("ideagen.api.main.PaperUnderstandingPipeline.understand_paper", return_value=failed):
        resp = client.post(f"/api/v1/papers/{paper_id}/understand")
    assert resp.status_code == 502
    assert "Ollama connection error" in resp.json()["detail"]


def test_gaps_are_tagged_at_save_time_even_if_discovery_fails_later(env, tmp_path):
    """Gaps saved before a later failure must not be orphaned without a project."""
    from ideagen.db.persistence import SQLAlchemyPersistenceLayer
    from ideagen.discovery.models import EvidenceRole, ResearchGapCreateDTO, StatementEvidenceDTO

    _, SessionLocal = env
    with SessionLocal() as db:
        db.add(ResearchProjectRecord(id="proj_a", name="A"))
        db.commit()
        paper_a = _seed_statement(db, "proj_a", "cyber")
        stmt = db.query(ResearchStatement).first()
        pipeline = SemanticGapDiscoveryPipeline(
            persistence=SQLAlchemyPersistenceLayer(db), embedding_provider=_FakeEmbedder(),
            llm_provider=object(), index_dir=tmp_path,
        )
        candidate_dto = ResearchGapCreateDTO(
            gap_statement="g", research_problem="p", status=GapStatus.VALIDATED, evidence_count=1,
            supporting_statement_ids=[stmt.id], supporting_paper_ids=[paper_a], limitations=[], future_work=[],
            validation_reasoning="r", validation_confidence=0.9, metadata={},
            evidence_items=[StatementEvidenceDTO(statement_id=stmt.id, paper_id=paper_a, statement_type="LIMITATION", statement_text=stmt.statement_text, evidence_role=EvidenceRole.LIMITATION)],
        )
        calls = {"n": 0}

        def validate(**kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("Ollama went away")
            return candidate_dto.model_copy(deep=True)

        with patch.object(pipeline.grouper, "group_statements", return_value=["c1", "c2"]), \
             patch.object(pipeline.pattern_analyzer, "analyze_clusters", return_value=["cand1", "cand2"]), \
             patch.object(pipeline.validator, "validate_candidate", side_effect=validate):
            with pytest.raises(RuntimeError):
                pipeline.discover_gaps(paper_ids={paper_a}, project_id="proj_a", gap_metadata={"run_id": "run_x"})
        saved = db.query(ResearchGap).all()
        assert len(saved) == 1
        assert saved[0].project_id == "proj_a"
        assert saved[0].metadata_json["run_id"] == "run_x"


def test_recovery_marks_stale_pending_runs_but_not_fresh_ones(env):
    _, SessionLocal = env
    with SessionLocal() as db:
        db.add(ResearchProjectRecord(id="proj_a", name="A"))
        db.commit()
        old = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=5)
        stale = _run_with_stage_states(db, "proj_a", {}, "PENDING")
        stale.updated_at = old
        fresh = _run_with_stage_states(db, "proj_a", {}, "PENDING")
        db.commit()
        assert ResearchWorkflowEngine.recover_interrupted_runs(db) == 1
        db.refresh(stale)
        db.refresh(fresh)
        assert stale.status == "INTERRUPTED"
        assert fresh.status == "PENDING"


def test_evolve_rejects_papers_from_another_project(env):
    from ideagen.db.models import Opportunity

    client, SessionLocal = env
    _create(client, "proj_a")
    _create(client, "proj_b")
    b_paper = _ingest(client, 4, "proj_b")["paper_id"]
    with SessionLocal() as db:
        opp = Opportunity(id=uuid.uuid4(), project_id="proj_a", title="t", problem_statement="p", description="d", status="OPEN")
        db.add(opp)
        db.commit()
        opp_id = opp.id
    resp = client.post(f"/api/v1/opportunities/{opp_id}/evolve", params={"project_id": "proj_a"}, json={"paper_ids": [b_paper]})
    assert resp.status_code == 400
    assert "not in project proj_a" in resp.json()["detail"]


def test_preflight_reports_corpus_only_project_as_runnable(env):
    """Papers added with "Add to Project Corpus" are not queued selections, but a run processes them.
    Preflight must report them, and mark the run runnable, instead of showing an empty queue."""
    client, SessionLocal = env
    _create(client, "proj_a")
    with SessionLocal() as db:
        full_text = _paper_with_passage(db, "proj_a", "full-text")
        db.add(ResearchKnowledge(paper_id=full_text, model_name="m", extraction_status="SUCCESS",
                                 extracted_at=datetime.datetime.now(datetime.timezone.utc)))
        db.commit()
        _paper_with_passage(db, "proj_a", "needs-extraction")
    _ingest(client, 5, "proj_a")  # metadata-only paper (no PDF)

    pf = client.get("/api/v1/projects/proj_a/preflight").json()
    assert pf["total_selected"] == 0 and pf["papers"] == []
    assert pf["corpus"] == {
        "total": 3, "with_passages": 2, "statements_extracted": 1, "pending_extraction": 1, "metadata_only": 1,
    }
    assert pf["runnable"] is True
    assert any("metadata only" in w for w in pf["warnings"])


def test_preflight_empty_project_is_not_runnable(env):
    client, _ = env
    _create(client, "proj_empty")
    pf = client.get("/api/v1/projects/proj_empty/preflight").json()
    assert pf["runnable"] is False
    assert pf["corpus"]["total"] == 0
    assert any("corpus is empty" in w for w in pf["warnings"])


def test_preflight_warns_when_no_corpus_paper_has_full_text(env):
    client, _ = env
    _create(client, "proj_meta")
    _ingest(client, 6, "proj_meta")
    pf = client.get("/api/v1/projects/proj_meta/preflight").json()
    assert pf["runnable"] is True
    assert pf["corpus"]["metadata_only"] == 1
    assert any("None of the 1 corpus paper(s) has full text" in w for w in pf["warnings"])
