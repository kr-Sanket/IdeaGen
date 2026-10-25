"""Regression tests for GAP_DISCOVERY retry eligibility in workflow_engine.py.

Verifies:
1. GAP_DISCOVERY is reset to PENDING on retry when previous gaps_discovered == 0.
2. GAP_DISCOVERY remains COMPLETED when previous gaps_discovered >= 1.
3. Missing or malformed stage details are handled safely without unintended resets.
4. Other stage states (INGESTION, UNDERSTANDING) retain expected behavior.
5. Run ID is preserved across retries without duplicate record creation.
"""

from __future__ import annotations

import datetime
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ideagen.db.models import (
    Base,
    ResearchProjectRecord,
    ResearchRunRecord,
    RunPaperRecord,
)
from ideagen.pipeline.workflow_engine import ResearchWorkflowEngine


@pytest.fixture
def db_session():
    """Sets up an in-memory SQLite database session."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


def test_retry_resets_gap_discovery_when_zero_gaps_discovered(db_session):
    """Verifies that GAP_DISCOVERY is reset to PENDING when it previously found 0 gaps."""
    project = ResearchProjectRecord(id="proj_gap_zero", name="Zero Gaps Project")
    db_session.add(project)
    db_session.commit()

    run = ResearchRunRecord(
        id="run_gap_zero",
        project_id="proj_gap_zero",
        status="PARTIAL_SUCCESS",
        planned_stages=["INGESTION", "UNDERSTANDING", "GAP_DISCOVERY"],
        stage_states={
            "INGESTION": {
                "status": "COMPLETED",
                "started_at": "2026-10-25T07:00:00+00:00",
                "completed_at": "2026-10-25T07:01:00+00:00",
                "details": {"ingested": 4},
            },
            "UNDERSTANDING": {
                "status": "PARTIAL_SUCCESS",
                "started_at": "2026-10-25T07:01:00+00:00",
                "completed_at": "2026-10-25T07:02:00+00:00",
                "details": {},
            },
            "GAP_DISCOVERY": {
                "status": "COMPLETED",
                "started_at": "2026-10-25T07:02:00+00:00",
                "completed_at": "2026-10-25T07:03:00+00:00",
                "details": {"gaps_discovered": 0, "project_papers_considered": 4},
            },
        },
    )
    db_session.add(run)

    p1 = RunPaperRecord(
        id="rp_1",
        run_id="run_gap_zero",
        openalex_id="W1",
        title="Paper 1",
        status="FAILED",
    )
    db_session.add(p1)
    db_session.commit()

    engine = ResearchWorkflowEngine(db_session)

    # Patch execute_run to verify reset state without actually running full pipeline
    with patch.object(ResearchWorkflowEngine, "execute_run") as mock_exec:
        retried_run = engine.retry_run("run_gap_zero", db_override=db_session)
        mock_exec.assert_called_once_with("run_gap_zero", db_override=db_session)

    db_session.refresh(run)

    # 1. Run ID must be preserved
    assert retried_run.id == "run_gap_zero"
    assert retried_run.status == "PENDING"

    # 2. GAP_DISCOVERY must be reset to PENDING because gaps_discovered == 0
    gap_state = retried_run.stage_states["GAP_DISCOVERY"]
    assert gap_state["status"] == "PENDING"
    assert gap_state["started_at"] is None
    assert gap_state["completed_at"] is None

    # 3. UNDERSTANDING was PARTIAL_SUCCESS, so it must also be PENDING
    assert retried_run.stage_states["UNDERSTANDING"]["status"] == "PENDING"

    # 4. INGESTION was COMPLETED, so it must remain COMPLETED
    assert retried_run.stage_states["INGESTION"]["status"] == "COMPLETED"

    # 5. Paper status must be reset to PENDING
    db_session.refresh(p1)
    assert p1.status == "PENDING"


def test_retry_preserves_gap_discovery_when_gaps_exist(db_session):
    """Verifies that GAP_DISCOVERY remains COMPLETED if it previously discovered gaps."""
    project = ResearchProjectRecord(id="proj_gap_exist", name="Existing Gaps Project")
    db_session.add(project)
    db_session.commit()

    run = ResearchRunRecord(
        id="run_gap_exist",
        project_id="proj_gap_exist",
        status="PARTIAL_SUCCESS",
        planned_stages=["INGESTION", "UNDERSTANDING", "GAP_DISCOVERY"],
        stage_states={
            "INGESTION": {
                "status": "COMPLETED",
                "started_at": "2026-10-25T07:00:00+00:00",
                "completed_at": "2026-10-25T07:01:00+00:00",
                "details": {"ingested": 4},
            },
            "UNDERSTANDING": {
                "status": "PARTIAL_SUCCESS",
                "started_at": "2026-10-25T07:01:00+00:00",
                "completed_at": "2026-10-25T07:02:00+00:00",
                "details": {},
            },
            "GAP_DISCOVERY": {
                "status": "COMPLETED",
                "started_at": "2026-10-25T07:02:00+00:00",
                "completed_at": "2026-10-25T07:03:00+00:00",
                "details": {"gaps_discovered": 3, "project_papers_considered": 4},
            },
        },
    )
    db_session.add(run)

    p1 = RunPaperRecord(
        id="rp_2",
        run_id="run_gap_exist",
        openalex_id="W2",
        title="Paper 2",
        status="FAILED",
    )
    db_session.add(p1)
    db_session.commit()

    engine = ResearchWorkflowEngine(db_session)

    with patch.object(ResearchWorkflowEngine, "execute_run"):
        engine.retry_run("run_gap_exist", db_override=db_session)

    db_session.refresh(run)

    # GAP_DISCOVERY had 3 gaps, so it must NOT be reset (remains COMPLETED)
    gap_state = run.stage_states["GAP_DISCOVERY"]
    assert gap_state["status"] == "COMPLETED"
    assert gap_state["started_at"] == "2026-10-25T07:02:00+00:00"
    assert gap_state["completed_at"] == "2026-10-25T07:03:00+00:00"

    # UNDERSTANDING was PARTIAL_SUCCESS, so it resets
    assert run.stage_states["UNDERSTANDING"]["status"] == "PENDING"


@pytest.mark.parametrize(
    "malformed_details",
    [
        None,
        {},
        "invalid-details-string",
        {"gaps_discovered": False},
        {"other_key": 10},
    ],
)
def test_retry_handles_missing_or_malformed_gap_details_safely(db_session, malformed_details):
    """Verifies that missing, None, or non-dict details do not cause unintended resets or crashes."""
    project = ResearchProjectRecord(id="proj_malformed", name="Malformed Project")
    db_session.add(project)
    db_session.commit()

    run = ResearchRunRecord(
        id="run_malformed",
        project_id="proj_malformed",
        status="PARTIAL_SUCCESS",
        planned_stages=["INGESTION", "UNDERSTANDING", "GAP_DISCOVERY"],
        stage_states={
            "INGESTION": {"status": "COMPLETED"},
            "UNDERSTANDING": {"status": "PARTIAL_SUCCESS"},
            "GAP_DISCOVERY": {
                "status": "COMPLETED",
                "started_at": "2026-10-25T07:02:00+00:00",
                "completed_at": "2026-10-25T07:03:00+00:00",
                "details": malformed_details,
            },
        },
    )
    db_session.add(run)

    p1 = RunPaperRecord(
        id="rp_3",
        run_id="run_malformed",
        openalex_id="W3",
        title="Paper 3",
        status="FAILED",
    )
    db_session.add(p1)
    db_session.commit()

    engine = ResearchWorkflowEngine(db_session)

    with patch.object(ResearchWorkflowEngine, "execute_run"):
        engine.retry_run("run_malformed", db_override=db_session)

    db_session.refresh(run)

    # When details are missing or malformed, GAP_DISCOVERY should remain COMPLETED (no crash, safe default)
    assert run.stage_states["GAP_DISCOVERY"]["status"] == "COMPLETED"


def test_retry_resets_gap_discovery_when_status_is_failed(db_session):
    """Verifies that GAP_DISCOVERY resets to PENDING if its status is FAILED regardless of details."""
    project = ResearchProjectRecord(id="proj_gap_fail", name="Failed Gap Project")
    db_session.add(project)
    db_session.commit()

    run = ResearchRunRecord(
        id="run_gap_fail",
        project_id="proj_gap_fail",
        status="FAILED",
        planned_stages=["GAP_DISCOVERY"],
        stage_states={
            "GAP_DISCOVERY": {
                "status": "FAILED",
                "started_at": "2026-10-25T07:02:00+00:00",
                "completed_at": "2026-10-25T07:03:00+00:00",
                "details": {"gaps_discovered": 5, "error": "LLM timeout during validation"},
            },
        },
    )
    db_session.add(run)
    db_session.commit()

    engine = ResearchWorkflowEngine(db_session)

    with patch.object(ResearchWorkflowEngine, "execute_run"):
        engine.retry_run("run_gap_fail", db_override=db_session)

    db_session.refresh(run)

    # Because status was FAILED, it must be reset to PENDING
    assert run.stage_states["GAP_DISCOVERY"]["status"] == "PENDING"
