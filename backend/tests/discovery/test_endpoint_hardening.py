"""Offline deterministic tests for Phase 5F Stage 1: Direct Endpoint Hardening.

Covers:
- Candidate validation bounding on the production pipeline path
- Default limit of 3 candidates and hard maximum of 5
- Config and query validation rejecting invalid limits (<1 or >5)
- Preservation of candidate ranking order when capping
- Per-project concurrency lock rejecting overlapping requests with HTTP 409
- Independent execution across different projects
- Deterministic lock cleanup on success and on exception
- Preservation of Phase 5E provider failure error handling (HTTP 502)
"""

from __future__ import annotations

import datetime
import threading
import time
import uuid
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ideagen.api.main import app, _discovery_lock_mgr
from ideagen.core.config import Settings, get_settings
from ideagen.db import get_db
from ideagen.db.models import Base, ProjectPaperRecord, ResearchProjectRecord
from ideagen.db.persistence import SQLAlchemyPersistenceLayer
from ideagen.discovery.models import (
    EvidenceRole,
    GapCandidate,
    GapStatus,
    ResearchGapCreateDTO,
    ResearchGapResponseDTO,
    StatementEvidenceDTO,
)
from ideagen.discovery.pipeline import SemanticGapDiscoveryPipeline
from ideagen.llm.exceptions import ProviderFailureError


# ── Fixtures & Test In-Memory Database ────────────────────────────────────────

@pytest.fixture
def in_memory_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    # Seed two sample projects for scoping tests
    proj_a = ResearchProjectRecord(
        id="proj_alpha",
        name="Project Alpha",
        topic="Machine Learning",
    )
    proj_b = ResearchProjectRecord(
        id="proj_beta",
        name="Project Beta",
        topic="Computer Vision",
    )
    session.add_all([proj_a, proj_b])
    session.commit()

    yield session

    session.close()
    Base.metadata.drop_all(engine)


@pytest.fixture
def client(in_memory_db):
    def override_get_db():
        try:
            yield in_memory_db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    test_client = TestClient(app, raise_server_exceptions=False)
    yield test_client
    app.dependency_overrides.clear()


def _make_dummy_candidates(count: int) -> List[GapCandidate]:
    """Creates a deterministic list of dummy candidates with known indices."""
    return [
        GapCandidate(
            candidate_id=f"cand_{i}",
            cluster_id=i,
            gap_statement=f"Candidate gap statement {i}",
            research_problem=f"Problem {i}",
            why_existing_work_insufficient=f"Existing methods fail on condition {i}",
            supporting_statement_ids=[uuid.uuid4()],
            supporting_passage_ids=[],
            limiting_statement_ids=[],
            limiting_passage_ids=[],
            cluster_density=0.85 - (i * 0.05),
            cluster_spread=0.10 + (i * 0.02),
            theme_keywords=[f"keyword_{i}"],
        )
        for i in range(count)
    ]


# ── Configuration & Validation Tests ─────────────────────────────────────────

class TestCandidateLimitConfiguration:
    def test_default_limit_is_three(self):
        settings = Settings()
        assert settings.discovery_max_candidates == 3

    def test_configured_limit_accepted_up_to_five(self, monkeypatch):
        monkeypatch.setenv("IDEAGEN_DISCOVERY_MAX_CANDIDATES", "5")
        settings = Settings()
        assert settings.discovery_max_candidates == 5

    def test_configured_limit_minimum_one(self, monkeypatch):
        monkeypatch.setenv("IDEAGEN_DISCOVERY_MAX_CANDIDATES", "1")
        settings = Settings()
        assert settings.discovery_max_candidates == 1

    def test_configured_limit_exceeding_five_rejected(self, monkeypatch):
        monkeypatch.setenv("IDEAGEN_DISCOVERY_MAX_CANDIDATES", "6")
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        assert "discovery_max_candidates" in str(exc_info.value)

    def test_configured_limit_below_one_rejected(self, monkeypatch):
        monkeypatch.setenv("IDEAGEN_DISCOVERY_MAX_CANDIDATES", "0")
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        assert "discovery_max_candidates" in str(exc_info.value)

    def test_configured_limit_non_integer_rejected(self, monkeypatch):
        monkeypatch.setenv("IDEAGEN_DISCOVERY_MAX_CANDIDATES", "invalid_number")
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        assert "discovery_max_candidates" in str(exc_info.value)

    def test_configured_limit_float_rejected(self, monkeypatch):
        monkeypatch.setenv("IDEAGEN_DISCOVERY_MAX_CANDIDATES", "3.8")
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        assert "discovery_max_candidates" in str(exc_info.value)

    def test_configured_limit_boolean_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            Settings(discovery_max_candidates=True)
        assert "discovery_max_candidates" in str(exc_info.value)


# ── Pipeline Enforcement & Ranking Order Tests ───────────────────────────────

class TestPipelineCandidateBounding:
    def test_pipeline_caps_at_default_three_and_preserves_order(self, in_memory_db):
        persistence = SQLAlchemyPersistenceLayer(in_memory_db)
        pipeline = SemanticGapDiscoveryPipeline(persistence=persistence)
        assert pipeline.max_candidates == 3

        candidates = _make_dummy_candidates(8)
        pipeline.pattern_analyzer.analyze_clusters = MagicMock(return_value=candidates)

        validated_candidates = []
        def mock_validate(candidate, statements_by_id, passages_by_id):
            validated_candidates.append(candidate.research_problem)
            stmt_id = candidate.supporting_statement_ids[0]
            paper_id = uuid.uuid4()
            return ResearchGapCreateDTO(
                gap_statement=f"Validated gap for {candidate.research_problem}",
                research_problem=candidate.research_problem,
                status=GapStatus.VALIDATED,
                supporting_statement_ids=[stmt_id],
                supporting_paper_ids=[paper_id],
                evidence_count=1,
                evidence_items=[
                    StatementEvidenceDTO(
                        statement_id=stmt_id,
                        paper_id=paper_id,
                        statement_type="HYPOTHESIS",
                        statement_text="Sample text",
                        evidence_role=EvidenceRole.SUPPORTING,
                    )
                ],
                metadata={"created_at": datetime.datetime.now(datetime.timezone.utc)},
            )

        pipeline.validator.validate_candidate = MagicMock(side_effect=mock_validate)
        persistence.save_research_gap = MagicMock(side_effect=lambda dto, project_id=None: uuid.uuid4())

        # Mock statements loading so pipeline runs
        dummy_stmt = MagicMock()
        dummy_stmt.id = candidates[0].supporting_statement_ids[0]
        dummy_stmt.statement_text = "Sample research text."
        dummy_stmt.paper_id = uuid.uuid4()
        persistence.get_all_research_statements = MagicMock(return_value=[dummy_stmt])
        pipeline.embedding_provider.embed_texts = MagicMock(return_value=[[0.1] * 768])
        pipeline.vector_index.rebuild = MagicMock()
        pipeline.grouper.group_statements = MagicMock(return_value=[])

        results = pipeline.discover_gaps(persist_index=False)

        # Must be capped to 3
        assert len(results) == 3
        assert len(validated_candidates) == 3
        # Must preserve exact order: Problem 0, Problem 1, Problem 2
        assert validated_candidates == ["Problem 0", "Problem 1", "Problem 2"]

    def test_pipeline_explicit_cap_parameter(self, in_memory_db):
        persistence = SQLAlchemyPersistenceLayer(in_memory_db)
        pipeline = SemanticGapDiscoveryPipeline(persistence=persistence)

        candidates = _make_dummy_candidates(8)
        pipeline.pattern_analyzer.analyze_clusters = MagicMock(return_value=candidates)

        validated_candidates = []
        def mock_validate(candidate, statements_by_id, passages_by_id):
            validated_candidates.append(candidate.research_problem)
            stmt_id = candidate.supporting_statement_ids[0]
            paper_id = uuid.uuid4()
            return ResearchGapCreateDTO(
                gap_statement="Gap",
                research_problem=candidate.research_problem,
                status=GapStatus.VALIDATED,
                supporting_statement_ids=[stmt_id],
                supporting_paper_ids=[paper_id],
                evidence_count=1,
                evidence_items=[
                    StatementEvidenceDTO(
                        statement_id=stmt_id,
                        paper_id=paper_id,
                        statement_type="HYPOTHESIS",
                        statement_text="Sample text",
                        evidence_role=EvidenceRole.SUPPORTING,
                    )
                ],
                metadata={"created_at": datetime.datetime.now(datetime.timezone.utc)},
            )

        pipeline.validator.validate_candidate = MagicMock(side_effect=mock_validate)
        persistence.save_research_gap = MagicMock(side_effect=lambda dto, project_id=None: uuid.uuid4())

        dummy_stmt = MagicMock()
        dummy_stmt.id = candidates[0].supporting_statement_ids[0]
        dummy_stmt.statement_text = "Text"
        dummy_stmt.paper_id = uuid.uuid4()
        persistence.get_all_research_statements = MagicMock(return_value=[dummy_stmt])
        pipeline.embedding_provider.embed_texts = MagicMock(return_value=[[0.1] * 768])
        pipeline.vector_index.rebuild = MagicMock()
        pipeline.grouper.group_statements = MagicMock(return_value=[])

        # Explicitly pass cap of 2
        results = pipeline.discover_gaps(persist_index=False, max_candidates=2)
        assert len(results) == 2
        assert validated_candidates == ["Problem 0", "Problem 1"]

    def test_pipeline_rejects_invalid_cap_values(self, in_memory_db):
        persistence = SQLAlchemyPersistenceLayer(in_memory_db)
        pipeline = SemanticGapDiscoveryPipeline(persistence=persistence)

        with pytest.raises(ValueError, match="max_candidates must be an integer between 1 and 5"):
            pipeline.discover_gaps(max_candidates=0)

        with pytest.raises(ValueError, match="max_candidates must be an integer between 1 and 5"):
            pipeline.discover_gaps(max_candidates=6)

        with pytest.raises(ValueError, match="max_candidates must be an integer between 1 and 5"):
            SemanticGapDiscoveryPipeline(persistence=persistence, max_candidates=10)

        # Reject booleans (True / False)
        with pytest.raises(ValueError, match="got boolean True"):
            pipeline.discover_gaps(max_candidates=True)

        with pytest.raises(ValueError, match="got boolean False"):
            SemanticGapDiscoveryPipeline(persistence=persistence, max_candidates=False)

        # Reject floats
        with pytest.raises(ValueError, match="got float 3.5"):
            pipeline.discover_gaps(max_candidates=3.5)

        with pytest.raises(ValueError, match="got float 2.5"):
            SemanticGapDiscoveryPipeline(persistence=persistence, max_candidates=2.5)

    def test_pipeline_prevents_unbounded_bypass_when_attribute_is_none(self, in_memory_db):
        persistence = SQLAlchemyPersistenceLayer(in_memory_db)
        pipeline = SemanticGapDiscoveryPipeline(persistence=persistence)

        # Artificially set max_candidates attribute to None
        pipeline.max_candidates = None

        candidates = _make_dummy_candidates(8)
        pipeline.pattern_analyzer.analyze_clusters = MagicMock(return_value=candidates)
        dummy_stmt = MagicMock()
        dummy_stmt.id = candidates[0].supporting_statement_ids[0]
        dummy_stmt.statement_text = "Sample text."
        dummy_stmt.paper_id = uuid.uuid4()
        persistence.get_all_research_statements = MagicMock(return_value=[dummy_stmt])
        pipeline.embedding_provider.embed_texts = MagicMock(return_value=[[0.1] * 768])
        pipeline.vector_index.rebuild = MagicMock()
        pipeline.grouper.group_statements = MagicMock(return_value=[])

        validated_candidates = []
        def mock_validate(candidate, statements_by_id, passages_by_id):
            validated_candidates.append(candidate.research_problem)
            stmt_id = candidate.supporting_statement_ids[0]
            paper_id = uuid.uuid4()
            return ResearchGapCreateDTO(
                gap_statement="Gap",
                research_problem=candidate.research_problem,
                status=GapStatus.VALIDATED,
                supporting_statement_ids=[stmt_id],
                supporting_paper_ids=[paper_id],
                evidence_count=1,
                evidence_items=[
                    StatementEvidenceDTO(
                        statement_id=stmt_id,
                        paper_id=paper_id,
                        statement_type="HYPOTHESIS",
                        statement_text="Sample text",
                        evidence_role=EvidenceRole.SUPPORTING,
                    )
                ],
                metadata={"created_at": datetime.datetime.now(datetime.timezone.utc)},
            )

        pipeline.validator.validate_candidate = MagicMock(side_effect=mock_validate)
        persistence.save_research_gap = MagicMock(side_effect=lambda dto, project_id=None: uuid.uuid4())

        # Must fall back to default limit of 3, NOT run unbounded with all 8 candidates
        results = pipeline.discover_gaps(persist_index=False, max_candidates=None)
        assert len(results) == 3
        assert len(validated_candidates) == 3

    def test_pipeline_rejects_invalid_mutated_instance_attribute(self, in_memory_db):
        persistence = SQLAlchemyPersistenceLayer(in_memory_db)
        pipeline = SemanticGapDiscoveryPipeline(persistence=persistence)

        # Artificially set max_candidates to an invalid number
        pipeline.max_candidates = 20

        with pytest.raises(ValueError, match="max_candidates must be an integer between 1 and 5"):
            pipeline.discover_gaps()


# ── Production API Endpoint Tests ────────────────────────────────────────────

class TestDirectEndpointHardening:
    @patch("ideagen.api.main.SemanticGapDiscoveryPipeline")
    def test_endpoint_query_validation(self, mock_pipeline_cls, client):
        # Value > 5 rejected with 422
        resp = client.post("/api/v1/gaps/discover?max_candidates=6")
        assert resp.status_code == 422

        # Value < 1 rejected with 422
        resp = client.post("/api/v1/gaps/discover?max_candidates=0")
        assert resp.status_code == 422

        # Non-integer rejected with 422
        resp = client.post("/api/v1/gaps/discover?max_candidates=notanumber")
        assert resp.status_code == 422

    @patch("ideagen.api.main.SemanticGapDiscoveryPipeline")
    def test_endpoint_passes_configured_cap_to_pipeline(self, mock_pipeline_cls, client):
        mock_pipeline = MagicMock()
        mock_pipeline.discover_gaps.return_value = []
        mock_pipeline_cls.return_value = mock_pipeline

        # When not specified in query, max_candidates is None so pipeline uses its configured default
        resp = client.post("/api/v1/gaps/discover?project_id=proj_alpha")
        assert resp.status_code == 200
        mock_pipeline.discover_gaps.assert_called_once()
        assert mock_pipeline.discover_gaps.call_args.kwargs["max_candidates"] is None

        # When specified in query (e.g. 2), it is forwarded
        mock_pipeline.discover_gaps.reset_mock()
        resp = client.post("/api/v1/gaps/discover?project_id=proj_alpha&max_candidates=2")
        assert resp.status_code == 200
        assert mock_pipeline.discover_gaps.call_args.kwargs["max_candidates"] == 2


# ── Per-Project Concurrency Guard Tests ───────────────────────────────────────

class TestPerProjectConcurrencyGuard:
    def test_same_project_overlapping_request_returns_409(self, client):
        start_event = threading.Event()
        proceed_event = threading.Event()

        def slow_discovery(*args, **kwargs):
            start_event.set()
            proceed_event.wait(timeout=5.0)
            return []

        with patch("ideagen.api.main.SemanticGapDiscoveryPipeline") as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.discover_gaps.side_effect = slow_discovery
            mock_pipeline_cls.return_value = mock_pipeline

            first_resp = {}
            def run_first():
                first_resp["resp"] = client.post("/api/v1/gaps/discover?project_id=proj_alpha")

            t1 = threading.Thread(target=run_first)
            t1.start()

            # Wait until request 1 is actively inside discovery
            assert start_event.wait(timeout=3.0)

            # Send overlapping request 2 for the same project
            second_resp = client.post("/api/v1/gaps/discover?project_id=proj_alpha")

            # Must reject with 409 Conflict immediately
            assert second_resp.status_code == 409
            assert "already in progress for project 'proj_alpha'" in second_resp.json()["detail"]

            # Let request 1 complete
            proceed_event.set()
            t1.join(timeout=3.0)
            assert first_resp["resp"].status_code == 200

    def test_different_projects_can_proceed_concurrently(self, client):
        alpha_start = threading.Event()
        alpha_proceed = threading.Event()

        def slow_alpha(*args, **kwargs):
            if kwargs.get("project_id") == "proj_alpha":
                alpha_start.set()
                alpha_proceed.wait(timeout=5.0)
            return []

        with patch("ideagen.api.main.SemanticGapDiscoveryPipeline") as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.discover_gaps.side_effect = slow_alpha
            mock_pipeline_cls.return_value = mock_pipeline

            alpha_resp = {}
            def run_alpha():
                alpha_resp["resp"] = client.post("/api/v1/gaps/discover?project_id=proj_alpha")

            t_alpha = threading.Thread(target=run_alpha)
            t_alpha.start()

            # Wait until Alpha is inside discovery
            assert alpha_start.wait(timeout=3.0)

            # Beta is a different project; it must NOT be blocked or 409'd
            beta_resp = client.post("/api/v1/gaps/discover?project_id=proj_beta")
            assert beta_resp.status_code == 200

            # Let Alpha finish
            alpha_proceed.set()
            t_alpha.join(timeout=3.0)
            assert alpha_resp["resp"].status_code == 200

    def test_lock_released_on_pipeline_exception(self, client):
        with patch("ideagen.api.main.SemanticGapDiscoveryPipeline") as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            # First call raises an exception
            mock_pipeline.discover_gaps.side_effect = RuntimeError("Simulated failure")
            mock_pipeline_cls.return_value = mock_pipeline

            resp1 = client.post("/api/v1/gaps/discover?project_id=proj_alpha")
            assert resp1.status_code == 500

            # Second call immediately after must acquire lock successfully (not 409)
            mock_pipeline.discover_gaps.side_effect = None
            mock_pipeline.discover_gaps.return_value = []
            resp2 = client.post("/api/v1/gaps/discover?project_id=proj_alpha")
            assert resp2.status_code == 200

    def test_lock_registry_does_not_leak_memory(self, client):
        with patch("ideagen.api.main.SemanticGapDiscoveryPipeline") as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.discover_gaps.return_value = []
            mock_pipeline_cls.return_value = mock_pipeline

            # Perform discovery on project
            resp = client.post("/api/v1/gaps/discover?project_id=proj_alpha")
            assert resp.status_code == 200

            # Once finished, lock manager registry must be clean
            assert "proj_alpha" not in _discovery_lock_mgr._locks
            assert "proj_alpha" not in _discovery_lock_mgr._active_counts

    def test_provider_failure_maps_to_502(self, client):
        with patch("ideagen.api.main.SemanticGapDiscoveryPipeline") as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.discover_gaps.side_effect = ProviderFailureError("Ollama connection timed out")
            mock_pipeline_cls.return_value = mock_pipeline

            resp = client.post("/api/v1/gaps/discover?project_id=proj_alpha")
            assert resp.status_code == 502
            assert "Discovery pipeline failed" in resp.json()["detail"]
            assert "Ollama connection timed out" in resp.json()["detail"]
